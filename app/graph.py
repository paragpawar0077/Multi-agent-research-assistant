"""
Wires the four agents into a LangGraph StateGraph. The interesting part is
the conditional edge out of `critic`: rejected -> back to researcher (retry),
approved/exhausted -> advance to next sub-question or move to writer once
all sub-questions are done.
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# MUST run before any local imports — agents.py -> retrieval.py reads
# CHROMA_DB_PATH / SECOND_BRAIN_USER_ID from os.environ at import time, so
# .env has to be loaded first or those lookups raise KeyError.
#
# Explicit path instead of load_dotenv()'s auto-detection: with `python -m
# app.graph`, frame-based auto-detection can resolve inconsistently on
# Windows. This always points at the .env file next to this project's root
# (one level up from app/), regardless of cwd or how the script was invoked.
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

from langgraph.graph import StateGraph, END

from .state import ResearchState
from .agents import (
    planner_node,
    researcher_node,
    critic_node,
    route_after_critic,
    advance_node,
    writer_node,
)


def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic", critic_node)
    graph.add_node("advance", advance_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "critic")

    # This is the loop: critic decides whether to retry the SAME sub-question,
    # advance to the NEXT one, or move on to writing the final report.
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "researcher": "researcher",  # retry
            "advance": "advance",        # next sub-question
            "writer": "writer",          # all done
        },
    )
    graph.add_edge("advance", "researcher")
    graph.add_edge("writer", END)

    return graph.compile()


def run(query: str) -> dict:
    app = build_graph()
    initial_state: ResearchState = {
        "query": query,
        "sub_questions": [],
        "current_index": 0,
        "critic_log": [],
        "final_report": None,
    }
    return app.invoke(initial_state)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What are the tradeoffs of RAG vs fine-tuning for domain adaptation?"
    result = run(q)
    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)
    print(result["final_report"])
    print("\n" + "=" * 80)
    print("CRITIC LOG")
    print("=" * 80)
    for entry in result["critic_log"]:
        print(entry)