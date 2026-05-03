"""
Interactive terminal chatbot: GraphRAG retrieval + risk triage + local Ollama.

Requires: Neo4j, Qdrant (ingested), Ollama with your model pulled.
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from graphrag.chat_pipeline import build_retriever_from_env
from graphrag.interactive_chat import run_interactive_session
from graphrag.ollama_env import default_ollama_chat_model


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG bipolar support chatbot (CLI).")
    parser.add_argument("--patient_id", default=os.getenv("CHAT_PATIENT_ID", "cli_user"))
    parser.add_argument("--model", default=default_ollama_chat_model())
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--rerank_top_n", type=int, default=8)
    parser.add_argument("--graph_hops", type=int, default=1)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log full LLM system+user prompt and raw chunks to stderr each turn",
    )
    args = parser.parse_args()

    retriever = build_retriever_from_env()
    try:
        run_interactive_session(
            retriever,
            patient_id=args.patient_id,
            model=args.model,
            top_k=args.top_k,
            rerank_top_n=args.rerank_top_n,
            graph_hops=args.graph_hops,
            debug_llm_prompt=args.debug,
        )
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
