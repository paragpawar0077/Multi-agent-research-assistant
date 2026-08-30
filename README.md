# Multi-Agent Research & Report Assistant

A LangGraph-orchestrated research assistant where specialized agents collaborate — plan, retrieve, critique, and write — to produce grounded, gap-aware reports instead of a single monolithic LLM call.

## Overview

Most "RAG chatbot" projects are a single prompt wrapped around a vector search. This project instead coordinates four distinct agents through an explicit state graph, with a critic agent that can send a sub-question back for another retrieval pass — real conditional control flow, not just a linear chain of prompts.

The retrieval layer reuses an existing production knowledge base (ChromaDB + Sentence-Transformers embeddings, per-user scoped) rather than building a new one, so the project is scoped around orchestration, not re-solving retrieval from scratch.

## Architecture

```
        ┌─────────┐
        │ planner │  splits the query into 2-3 sub-questions
        └────┬────┘
             │
     ┌───────▼────────┐
     │   researcher    │◄────────┐  retrieves evidence per sub-question
     └───────┬────────┘          │  (ChromaDB, with a web-search fallback)
             │                   │
        ┌────▼────┐              │  retry (bounded, max 2 attempts)
        │  critic │──────────────┘  judges whether evidence actually
        └────┬────┘                  answers the sub-question
             │ all sub-questions resolved
        ┌────▼────┐
        │  writer  │  synthesizes the final report, explicitly flagging
        └─────────┘  any sub-question that never got sufficient evidence
```

## Key Design Decision: The Critic Doesn't Just Retry — It Gates

A naive RAG pipeline retrieves once and writes from whatever comes back, whether or not it's actually relevant. Here, the critic agent evaluates evidence per sub-question before it ever reaches the writer. If evidence is insufficient, the sub-question is retried up to a bounded limit; if it's still insufficient after retries, it is explicitly marked as a gap in the final report rather than silently papered over.

This was verified directly, not assumed: in a test run asking about CI/CD and Kubernetes experience (not present in the underlying documents), all three sub-questions were correctly rejected across retries and the final report stated plainly that no relevant evidence existed — instead of generating plausible-sounding but fabricated claims. The same run structure, on a query with strong supporting evidence, produced approvals on the first pass with no unnecessary retries.

| Test query | Sub-questions | Approved (first pass) | Exhausted (flagged as gap) |
|---|---|---|---|
| Project portfolio overview | 3 | 2 | 1 |
| Background, projects, and tech stack | 3 | 3 | 0 |
| CI/CD and Kubernetes experience | 3 | 0 | 3 |

**Current limitation:** retries currently re-query with the same sub-question text, so a retry rarely surfaces new evidence on its own — the loop's proven value so far is gap detection and hallucination avoidance, not query refinement. Adding LLM-driven query reformulation on retry (using the critic's rejection reason to rewrite the search) is the natural next step to make retries substantively different from the first attempt.

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (StateGraph, conditional edges) |
| LLM | Groq API (`openai/gpt-oss-120b`) |
| Vector store | ChromaDB, per-user metadata filtering |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Service layer | FastAPI |
| Demo UI | Streamlit |

## Project Structure

```
app/
├── state.py       # Shared graph state schema
├── llm.py         # Groq chat completion wrapper
├── retrieval.py   # ChromaDB query layer with web-search fallback
├── agents.py      # Planner, researcher, critic, and writer node functions
├── graph.py       # LangGraph StateGraph wiring and conditional routing
└── main.py        # FastAPI service (POST /research)
streamlit_app.py    # Interactive demo UI
```

## Getting Started

### Prerequisites
- Python 3.11+
- A Groq API key
- An existing ChromaDB collection to query against (this project does not build a knowledge base — it queries one)

### Installation

```bash
pip install -r requirements.txt
cp .env.example .env
```

Set the following in `.env`:

```
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-120b
CHROMA_DB_PATH=/path/to/your/chroma_db
SECOND_BRAIN_USER_ID=your_numeric_user_id
```

### Running

```bash
# Run a single query from the command line
python -m app.graph "your research question"

# Run as an API service
uvicorn app.main:app --reload

# Run the interactive demo
streamlit run streamlit_app.py
```

## API

**`POST /research`**

```json
{ "query": "your research question" }
```

Returns the synthesized report and the full critic decision log for every sub-question, including any retries.

## Author

Parag Pawar — [github.com/paragpawar0077](https://github.com/paragpawar0077)