"""
Node functions for the graph. Each takes the ResearchState and returns a
partial state update (LangGraph merges it in). Keep prompts here so they're
easy to iterate on without touching graph wiring.
"""
import json
import re
from .state import ResearchState, SubQuestion, MAX_RETRIES
from .llm import chat
from .retrieval import retrieve

# ---------------------------------------------------------------------------
# 1. Planner
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """You break a user's research query into 2-3 focused, \
non-overlapping sub-questions that together would let someone answer the \
original query well. Return ONLY a JSON array of strings, no preamble, no \
markdown fences.

Context: the underlying knowledge base is a single individual's personal \
resume, projects, and career history — not a company, product, or \
multi-region business. If a query is ambiguous (e.g. uses words like \
"regions", "markets", "performance"), interpret it in terms of that \
individual's skills, projects, and career — not as a business/enterprise \
question — unless the query explicitly names a company or business \
context."""


def planner_node(state: ResearchState) -> dict:
    raw = chat(PLANNER_SYSTEM, f"Query: {state['query']}")
    try:
        questions = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        # Fallback: treat the whole query as a single sub-question rather
        # than crashing the graph on a malformed LLM response.
        questions = [state["query"]]

    sub_questions: list[SubQuestion] = [
        {
            "id": f"q{i+1}",
            "text": q,
            "retries": 0,
            "status": "pending",
            "evidence": "",
            "critic_note": None,
        }
        for i, q in enumerate(questions[:3])
    ]
    return {"sub_questions": sub_questions, "current_index": 0, "critic_log": []}


# ---------------------------------------------------------------------------
# 2. Researcher
# ---------------------------------------------------------------------------

def researcher_node(state: ResearchState) -> dict:
    idx = state["current_index"]
    sub_questions = list(state["sub_questions"])
    sq = sub_questions[idx]

    evidence = retrieve(sq["text"])

    sub_questions[idx] = {**sq, "evidence": evidence}
    return {"sub_questions": sub_questions}


# ---------------------------------------------------------------------------
# 3. Critic
# ---------------------------------------------------------------------------

CRITIC_SYSTEM = """You judge whether retrieved evidence actually answers a \
research sub-question. Be strict: vague, off-topic, or empty evidence should \
be rejected. Return ONLY JSON: {"verdict": "approve" or "reject", "reason": "..."}"""


def critic_node(state: ResearchState) -> dict:
    idx = state["current_index"]
    sub_questions = list(state["sub_questions"])
    sq = sub_questions[idx]

    user_msg = f"Sub-question: {sq['text']}\n\nEvidence:\n{sq['evidence'] or '(empty)'}"
    # Critic only needs to output a simple approve/reject judgment, not
    # synthesize an answer — use Groq's smaller, faster model here to cut
    # latency. This can add up to 6 calls per query, so the saving compounds.
    raw = chat(CRITIC_SYSTEM, user_msg, model="openai/gpt-oss-20b", temperature=0.0)
    try:
        verdict = json.loads(_strip_fences(raw))
    except json.JSONDecodeError:
        verdict = {"verdict": "approve", "reason": "critic output unparsable, defaulting to approve"}

    log_entry = {
        "sub_question": sq["text"],
        "attempt": sq["retries"] + 1,
        "verdict": verdict.get("verdict"),
        "reason": verdict.get("reason"),
    }

    if verdict.get("verdict") == "approve":
        sub_questions[idx] = {**sq, "status": "approved", "critic_note": verdict.get("reason")}
    else:
        if sq["retries"] + 1 >= MAX_RETRIES:
            sub_questions[idx] = {
                **sq,
                "status": "exhausted",
                "retries": sq["retries"] + 1,
                "critic_note": verdict.get("reason"),
            }
        else:
            sub_questions[idx] = {
                **sq,
                "status": "rejected",
                "retries": sq["retries"] + 1,
                "critic_note": verdict.get("reason"),
            }

    return {"sub_questions": sub_questions, "critic_log": [log_entry]}


def route_after_critic(state: ResearchState) -> str:
    """Conditional edge: retry same sub-question, move to next one, or write."""
    idx = state["current_index"]
    sq = state["sub_questions"][idx]

    if sq["status"] == "rejected":
        return "researcher"  # retry this sub-question

    # approved or exhausted -> advance
    if idx + 1 < len(state["sub_questions"]):
        return "advance"

    return "writer"


def advance_node(state: ResearchState) -> dict:
    return {"current_index": state["current_index"] + 1}


# ---------------------------------------------------------------------------
# 4. Writer
# ---------------------------------------------------------------------------

WRITER_SYSTEM = """You write a clear, well-structured research report from \
sub-question evidence. Use markdown headers per sub-question, cite evidence \
inline in your own words, and flag explicitly (in a "Gaps" section) any \
sub-question whose evidence was marked exhausted/incomplete."""


def writer_node(state: ResearchState) -> dict:
    # Cap evidence per sub-question so the combined writer prompt doesn't
    # blow past Groq's free-tier TPM limit (8000 tokens/min). Rough
    # heuristic: ~4 chars/token, so 3000 chars per sub-question keeps a
    # 3-question report comfortably under budget with room for the system
    # prompt and instructions.
    MAX_EVIDENCE_CHARS = 3000

    sections = []
    gaps = []
    for sq in state["sub_questions"]:
        evidence = sq["evidence"] or "(none found)"
        if len(evidence) > MAX_EVIDENCE_CHARS:
            evidence = evidence[:MAX_EVIDENCE_CHARS] + "\n[...truncated to fit token budget...]"
        sections.append(f"## {sq['text']}\n\nEvidence:\n{evidence}")
        if sq["status"] == "exhausted":
            gaps.append(sq["text"])

    user_msg = (
        f"Original query: {state['query']}\n\n"
        f"Sub-question evidence:\n\n" + "\n\n".join(sections) + "\n\n"
        f"Sub-questions with incomplete evidence after retries: {gaps or 'none'}"
    )
    report = chat(WRITER_SYSTEM, user_msg, temperature=0.4)
    return {"final_report": report}


# ---------------------------------------------------------------------------
def _strip_fences(text: str) -> str:
    """LLMs love wrapping JSON in ```json fences even when told not to."""
    return re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()