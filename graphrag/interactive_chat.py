"""Terminal REPL for GraphRAG chatbot (shared by chat.py)."""

import sys
from typing import Optional

from graphrag.chat_pipeline import run_chat_turn
from graphrag.retrieval import HybridRetriever

DISCLAIMER = """
This assistant provides general educational information only. It is not medical advice,
diagnosis, or treatment. For emergencies or if you may harm yourself, contact local
emergency services or a clinician immediately.
---
Commands: /quit /exit to leave, /sources toggles citation chunk list, /debug toggles full LLM prompt log on stderr, /help
"""


def run_interactive_session(
    retriever: HybridRetriever,
    *,
    patient_id: str = "cli_user",
    model: str = "qwen2.5:3b-instruct",
    top_k: int = 12,
    rerank_top_n: int = 8,
    graph_hops: int = 1,
    banner: Optional[str] = None,
    debug_llm_prompt: bool = False,
) -> None:
    """REPL using an existing retriever. Caller must close retriever."""
    if banner:
        print(banner)
    print(DISCLAIMER.strip())
    show_sources = False
    session_debug = debug_llm_prompt
    while True:
        try:
            line = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not line:
            continue
        low = line.lower()
        if low in ("/quit", "/exit", "quit", "exit"):
            print("Bye.")
            break
        if low == "/help":
            print(DISCLAIMER.strip())
            continue
        if low == "/sources":
            show_sources = not show_sources
            print(f"Show source chunks after each reply: {show_sources}")
            continue
        if low == "/debug":
            session_debug = not session_debug
            print(
                f"LLM prompt + chunk dump to stderr: {session_debug} "
                f"(or set RAG_DEBUG_PROMPT=1 in .env)"
            )
            continue
        try:
            dbg = True if session_debug else None
            out = run_chat_turn(
                retriever,
                line,
                patient_id,
                top_k=top_k,
                graph_hops=graph_hops,
                rerank_top_n=rerank_top_n,
                model=model,
                rag_debug_prompt=dbg,
            )
        except Exception as exc:
            print(f"\nAssistant: [error] {exc}", file=sys.stderr)
            continue
        print(f"\nAssistant:\n{out['answer']}")
        if show_sources and out.get("retrieval"):
            chunks = out["retrieval"].get("chunks") or []
            if chunks:
                print("\n[Sources]")
                for c in chunks:
                    cid = c.get("chunk_id", "?")
                    meta = c.get("metadata") or {}
                    sec = meta.get("section_title") or ""
                    preview = (c.get("text") or "")[:120].replace("\n", " ")
                    print(f"  - {cid}" + (f" | {sec}" if sec else "") + f" | {preview}...")
