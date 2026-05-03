import os
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from graphrag.chat_pipeline import build_retriever_from_env, run_chat_turn

load_dotenv(Path(__file__).resolve().parent / ".env")


class ChatRequest(BaseModel):
    patient_id: str = Field(..., description="Patient unique identifier")
    message: str
    top_k: int = 12
    graph_hops: int = 1
    rerank_top_n: int = 8
    model: str = "qwen2.5:3b-instruct"


class ChatResponse(BaseModel):
    escalated: bool
    risk: Dict
    answer: str
    retrieval: Optional[Dict] = None


app = FastAPI(title="Bipolar GraphRAG Assistant")


def _build_retriever():
    return build_retriever_from_env()


@app.get("/health")
def health() -> Dict:
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    retriever = _build_retriever()
    try:
        out = run_chat_turn(
            retriever,
            req.message,
            req.patient_id,
            top_k=req.top_k,
            graph_hops=req.graph_hops,
            rerank_top_n=req.rerank_top_n,
            model=req.model,
        )
        return ChatResponse(
            escalated=out["escalated"],
            risk=out["risk"],
            answer=out["answer"],
            retrieval=out.get("retrieval"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        retriever.close()

