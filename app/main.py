"""
FastAPI wrapper. Run with: uvicorn app.main:app --reload
"""
from fastapi import FastAPI
from pydantic import BaseModel

from .graph import run

app = FastAPI(title="Multi-Agent Research Assistant")


class ResearchRequest(BaseModel):
    query: str


class ResearchResponse(BaseModel):
    query: str
    report: str
    critic_log: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    result = run(req.query)
    return ResearchResponse(
        query=req.query,
        report=result["final_report"] or "",
        critic_log=result["critic_log"],
    )
