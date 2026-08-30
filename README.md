# Multi-Agent Research & Report Assistant

LangGraph orchestration on top of your existing ChromaDB + Sentence-Transformers +
GROQ/LLaMA stack (same retrieval setup as AI Second Brain). Four agents — Planner,
Researcher, Critic, Writer — collaborate via a StateGraph. The Critic can send a
sub-question back to the Researcher for another pass, which is the part that
actually earns the "LangGraph" keyword instead of just chaining prompts.

## Architecture

```
        ┌─────────┐
        │ planner │  splits query into 2-3 sub-questions
        └────┬────┘
             │
     ┌───────▼────────┐
     │   researcher    │◄────────┐  retrieves per sub-question
     └───────┬────────┘          │  (ChromaDB, falls back to web search)
             │                   │
        ┌────▼────┐              │ retry (max 2x)
        │  critic │──────────────┘  "does this answer the sub-question?"
        └────┬────┘
             │ all sub-questions covered
        ┌────▼────┐
        │  writer  │  synthesizes final report
        └─────────┘
```

## Day 1 checklist (linear flow)

1. `pip install -r requirements.txt`
2. Point `app/retrieval.py` at your existing AI Second Brain ChromaDB collection
   (same collection, same `all-MiniLM-L6-v2` embedder — don't rebuild anything).
3. Set `GROQ_API_KEY` in your environment.
4. Run `python app/graph.py "your test query"` and confirm you get a report out,
   planner → researcher → critic (approve first pass) → writer.

## Day 2 checklist (the actual loop)

1. In `app/agents.py`, tighten the critic's rejection criteria so it actually
   triggers a re-search sometimes (test with a vague/narrow sub-question).
2. Confirm `graph.py`'s conditional edge sends rejected sub-questions back to
   `researcher` and caps retries (`MAX_RETRIES` in `state.py`) so it can't loop
   forever.
3. Wrap with FastAPI (`app/main.py`) — `uvicorn app.main:app --reload`.
4. Optional: Streamlit demo (`streamlit_app.py`) for a screenshot/GIF for your
   portfolio and resume bullet.

## What's stubbed vs. real

- `retrieval.py` — has a real ChromaDB query function AND a real (optional)
  web-search fallback stub you fill in with whatever search API you have
  access to. Swap in your Second Brain collection name and you're live.
- `llm.py` — real GROQ chat completion calls, model configurable via env var.
- Everything else (state, nodes, graph wiring, FastAPI) is fully implemented,
  not pseudocode — you should be able to run this today.

## Resume bullet (verified against real test runs)

> Built a multi-agent research assistant using LangGraph with a
> planner-researcher-critic-writer architecture; critic agent evaluates
> retrieved evidence per sub-question and flags unanswerable gaps instead of
> generating unsupported content, verified across test runs including cases
> with no relevant source material.

**Why this framing, not a retry-reduction percentage:** in real test runs
(logged below), retries currently re-query with the exact same sub-question
text, so a rejected sub-question almost always gets rejected again on
retry — the retry doesn't yet change what gets retrieved. What the loop does
reliably do is stop the writer from fabricating an answer when evidence is
genuinely missing (e.g. a query about CI/CD/Kubernetes experience — not
present in the source documents — correctly resulted in 3/3 sub-questions
exhausted and flagged as gaps, with no hallucinated content in the report).
That's the claim this project can actually back up today.

**If you want to earn the stronger "reduces incomplete answers" claim
later:** add query reformulation on retry — an LLM call that rewrites the
sub-question using the critic's rejection reason before the researcher
retries, so attempt 2 is a genuinely different search, not a repeat of
attempt 1. That's the next real improvement, not yet built.

### Real test runs so far

| Query | Sub-questions | Approved (1st pass) | Rejected → exhausted | Notes |
|---|---|---|---|---|
| "What projects has Parag built?" | 3 | 2 | 1 | Correctly flagged missing dev-context info as a gap |
| "Who is Parag... projects... tech" | 3 | 3 | 0 | Clean run after refreshing the knowledge base |
| "What CI/CD or Kubernetes experience does Parag have?" | 3 | 0 | 3 | Correctly reported no hallucinated CI/CD/K8s claims |