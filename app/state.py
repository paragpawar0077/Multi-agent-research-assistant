"""
Shared state that flows through every node in the graph.

Keep this flat and JSON-serializable — it's what gets checkpointed and what
you'll dump into logs to compute your "reduced incomplete answers by X%"
metric later.
"""
from typing import TypedDict, List, Dict, Optional
import operator
from typing_extensions import Annotated

MAX_RETRIES = 2  # per sub-question, so the critic loop can't run forever


class SubQuestion(TypedDict):
    id: str
    text: str
    retries: int
    status: str  # "pending" | "approved" | "rejected" | "exhausted"
    evidence: str  # concatenated retrieved chunks for this sub-question
    critic_note: Optional[str]  # why the critic rejected it, if it did


class ResearchState(TypedDict):
    query: str                     # original user query
    sub_questions: List[SubQuestion]
    current_index: int             # which sub-question researcher/critic are on
    critic_log: Annotated[List[Dict], operator.add]  # append-only audit trail
    final_report: Optional[str]
