"""Shared GraphRAG retrieval + LLM chat turn (used by FastAPI and CLI)."""

import os
from typing import Any, Dict, Optional, Sequence

from graphrag.generation import generate_with_ollama
from graphrag.retrieval import HybridRetriever


def merge_user_note_and_image_caption(user_note: str, caption: str) -> str:
    """Combine optional user text with vision caption for retrieval + generation."""
    note = (user_note or "").strip()
    cap = (caption or "").strip()
    if not cap:
        raise ValueError("image caption is required")
    parts: list[str] = []
    if note:
        parts.append(f"User message:\n{note}")
    parts.append(f"[Image description from the user's photo]\n{cap}")
    return "\n\n".join(parts)


def photo_turn_transcript_for_ui(user_note: str, caption: str, *, max_caption_chars: int = 500) -> str:
    """Short line for transcript panel (not necessarily full merged prompt)."""
    note = (user_note or "").strip()
    cap = (caption or "").strip()
    if len(cap) > max_caption_chars:
        cap = cap[: max_caption_chars - 3].rstrip() + "..."
    if note:
        return f"(Photo) {note} — {cap}"
    return f"(Photo) {cap}"


def _env_flag(name: str, *, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def build_retriever_from_env() -> HybridRetriever:
    return HybridRetriever(
        collection_name=os.getenv("QDRANT_COLLECTION", "bipolar_chunks"),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        verify_qdrant_collection=_env_flag("QDRANT_VERIFY_COLLECTION", default=True),
    )


def run_chat_turn(
    retriever: HybridRetriever,
    message: str,
    patient_id: str,
    *,
    user_profile: Optional[Dict[str, str]] = None,
    conversation_history: Optional[Sequence[Dict[str, str]]] = None,
    top_k: int = 12,
    graph_hops: int = 1,
    rerank_top_n: int = 8,
    model: str = "qwen2.5:3b-instruct",
    rag_debug_prompt: Optional[bool] = None,
) -> Dict[str, Any]:
    retrieval_result = retriever.retrieve(
        query=message,
        top_k=top_k,
        graph_hops=graph_hops,
        rerank_top_n=rerank_top_n,
    )
    lang = retrieval_result.get("lang", "en")
    answer = generate_with_ollama(
        message,
        retrieval_result,
        user_profile=user_profile,
        conversation_history=conversation_history,
        model=model,
        debug_prompt=rag_debug_prompt,
    )
    return {
        "escalated": False,
        "risk": {},
        "answer": answer,
        "lang": lang,
        "retrieval": retrieval_result,
    }
