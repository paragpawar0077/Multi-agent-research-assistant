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

## Measuring "reduced incomplete answers by X%" for your resume bullet

Log every critic verdict (`state["critic_log"]`) to a JSONL file. After ~20-30
test queries, compute: (# sub-questions approved on first pass) vs. (# that
needed retries) vs. (# that hit MAX_RETRIES and were flagged incomplete in the
final report anyway). Compare against a baseline where you force
`MAX_RETRIES=0` (no loop). That delta is your X%.
