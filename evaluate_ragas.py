"""
RAGAS evaluation for GraphRAG: retrieve chunks, generate answers with Ollama, then score
with RAGAS metrics using the same Ollama host for judge LLM and embeddings.

Requires: Neo4j, Qdrant (ingested), Ollama with chat + embedding models pulled.

Example:
  python evaluate_ragas.py --questions eval_ragas.json --limit 2

Retrieval (higher values often help context_recall): defaults top_k=28, rerank_top_n=18, graph_hops=2
(override with --top-k / --rerank-top-n or env EVAL_TOP_K / EVAL_RERANK_TOP_N / EVAL_GRAPH_HOPS).

Generation for eval: extra CONTEXT grounding is on by default (env EVAL_STRICT_GROUNDING=1);
use --no-strict-grounding to match production prompts. Cooler generation for eval only:
env EVAL_GEN_TEMPERATURE (default 0.08). --per-row writes per-example scores plus inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

from datasets import Dataset
from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

from ragas import evaluate
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import AnswerRelevancy, Faithfulness, LLMContextRecall
from ragas.run_config import RunConfig

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(_PROJECT_ROOT / ".env")

from graphrag.chat_pipeline import build_retriever_from_env
from graphrag.generation import generate_with_ollama
from graphrag.ollama_env import default_ollama_chat_model, ollama_base_url
from graphrag.retrieval import HybridRetriever

# Defaults above chat_pipeline (12/8): wider pool + more reranked passages for LLMContextRecall.
_DEFAULT_EVAL_TOP_K = 28
_DEFAULT_EVAL_RERANK_TOP_N = 18
_DEFAULT_EVAL_GRAPH_HOPS = 2


def load_rows(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Evaluation file must be a JSON array of objects.")
    return data


def generate_answer(
    retriever: HybridRetriever,
    question: str,
    chat_model: str,
    *,
    top_k: int,
    graph_hops: int,
    rerank_top_n: int,
    eval_grounding: bool,
) -> Tuple[str, List[str]]:
    retrieval = retriever.retrieve(
        question,
        top_k=top_k,
        graph_hops=graph_hops,
        rerank_top_n=rerank_top_n,
    )
    chunks = retrieval.get("chunks") or []
    contexts = [((c.get("text") or "").strip()) for c in chunks if (c.get("text") or "").strip()]
    if not contexts:
        contexts = ["(no chunks retrieved)"]
    eval_temp = os.getenv("EVAL_GEN_TEMPERATURE", "0.08").strip()
    try:
        gen_temperature = float(eval_temp)
    except ValueError:
        gen_temperature = 0.08
    answer = generate_with_ollama(
        question,
        retrieval,
        model=chat_model,
        conversation_history=None,
        eval_grounding=eval_grounding,
        temperature=gen_temperature,
    )
    return answer, contexts


def build_hf_dataset(
    rows: List[dict],
    retriever: HybridRetriever,
    chat_model: str,
    limit: Optional[int],
    *,
    top_k: int,
    graph_hops: int,
    rerank_top_n: int,
    eval_grounding: bool,
) -> Dataset:
    user_input: List[str] = []
    response: List[str] = []
    retrieved_contexts: List[List[str]] = []
    reference: List[str] = []

    for i, item in enumerate(rows):
        if limit is not None and i >= limit:
            break
        q = (item.get("question") or "").strip()
        ref = (item.get("ground_truth") or item.get("reference") or "").strip()
        if not q:
            print(f"Skipping row without question: {item!r}", file=sys.stderr)
            continue
        if not ref:
            print(f"Skipping row without ground_truth/reference: {item!r}", file=sys.stderr)
            continue
        ans, ctxs = generate_answer(
            retriever,
            q,
            chat_model,
            top_k=top_k,
            graph_hops=graph_hops,
            rerank_top_n=rerank_top_n,
            eval_grounding=eval_grounding,
        )
        user_input.append(q)
        response.append(ans)
        retrieved_contexts.append(ctxs)
        reference.append(ref)

    if not user_input:
        raise RuntimeError("No evaluable rows (need question + ground_truth).")

    return Dataset.from_dict(
        {
            "user_input": user_input,
            "response": response,
            "retrieved_contexts": retrieved_contexts,
            "reference": reference,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS evaluation for GraphRAG + Ollama.")
    parser.add_argument("--questions", type=Path, default=_PROJECT_ROOT / "eval_ragas.json")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows.")
    parser.add_argument("--output", type=Path, default=None, help="Write aggregate scores as JSON.")
    parser.add_argument(
        "--per-row",
        type=Path,
        default=None,
        help="Write per-example columns + metric scores (.csv or .json from the file suffix).",
    )
    parser.add_argument(
        "--no-strict-grounding",
        action="store_true",
        help="Do not append benchmark grounding instructions to the user prompt (see EVAL_STRICT_GROUNDING).",
    )
    parser.add_argument(
        "--chat-model",
        default=os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("OLLAMA_MODEL") or default_ollama_chat_model(),
        help="Model for generate_with_ollama (GraphRAG answer).",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("RAGAS_JUDGE_MODEL") or default_ollama_chat_model(),
        help="Ollama model for RAGAS metric LLM calls (can match chat model).",
    )
    parser.add_argument(
        "--embed-model",
        default=os.getenv("RAGAS_EMBED_MODEL", "nomic-embed-text"),
        help="Ollama embedding model for answer_relevancy (pull with ollama pull).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"Retriever vector top_k (default: env EVAL_TOP_K or {_DEFAULT_EVAL_TOP_K}).",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=None,
        help=f"Reranker shortlist size (default: env EVAL_RERANK_TOP_N or {_DEFAULT_EVAL_RERANK_TOP_N}).",
    )
    parser.add_argument(
        "--graph-hops",
        type=int,
        default=None,
        help=f"Neo4j graph hops (default: env EVAL_GRAPH_HOPS or {_DEFAULT_EVAL_GRAPH_HOPS}).",
    )
    args = parser.parse_args()

    chat_model = (args.chat_model or default_ollama_chat_model()).strip()
    judge_model = (args.judge_model or chat_model).strip()
    embed_model = (args.embed_model or "nomic-embed-text").strip()
    base = ollama_base_url()
    timeout = int(os.getenv("RAGAS_TIMEOUT_SEC", "240"))
    run_config = RunConfig(timeout=timeout)

    llm = LangchainLLMWrapper(
        ChatOllama(
            model=judge_model,
            base_url=base,
            temperature=0.0,
            num_ctx=int(os.getenv("RAGAS_JUDGE_NUM_CTX", "4096")),
        ),
        run_config=run_config,
    )
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=embed_model, base_url=base),
        run_config=run_config,
    )

    top_k = (
        args.top_k
        if args.top_k is not None
        else int(os.getenv("EVAL_TOP_K", str(_DEFAULT_EVAL_TOP_K)))
    )
    rerank_top_n = (
        args.rerank_top_n
        if args.rerank_top_n is not None
        else int(os.getenv("EVAL_RERANK_TOP_N", str(_DEFAULT_EVAL_RERANK_TOP_N)))
    )
    graph_hops = (
        args.graph_hops
        if args.graph_hops is not None
        else int(os.getenv("EVAL_GRAPH_HOPS", str(_DEFAULT_EVAL_GRAPH_HOPS)))
    )
    top_k = max(1, top_k)
    rerank_top_n = max(1, rerank_top_n)
    graph_hops = max(0, graph_hops)

    env_strict = os.getenv("EVAL_STRICT_GROUNDING", "1").strip().lower() in ("1", "true", "yes")
    eval_grounding = env_strict and not args.no_strict_grounding

    rows = load_rows(args.questions)
    retriever = build_retriever_from_env()
    try:
        print(
            f"Building answers from GraphRAG (retrieve + Ollama) top_k={top_k} "
            f"rerank_top_n={rerank_top_n} graph_hops={graph_hops} eval_grounding={eval_grounding}...",
            file=sys.stderr,
        )
        ds = build_hf_dataset(
            rows,
            retriever,
            chat_model,
            args.limit,
            top_k=top_k,
            graph_hops=graph_hops,
            rerank_top_n=rerank_top_n,
            eval_grounding=eval_grounding,
        )
    finally:
        retriever.close()

    metrics = [Faithfulness(), AnswerRelevancy(), LLMContextRecall()]
    print(
        f"Running RAGAS on {len(ds)} row(s) (judge={judge_model}, embed={embed_model}) — may take several minutes...",
        file=sys.stderr,
    )
    result = evaluate(
        ds,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=False,
    )

    out: Dict[str, Any] = {}
    if hasattr(result, "_repr_dict"):
        out = dict(result._repr_dict)
    elif isinstance(result, dict):
        out = dict(result)
    else:
        out = {"raw": repr(result)}

    print(json.dumps(out, indent=2))
    if args.output:
        args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)

    if args.per_row:
        df = result.to_pandas()
        path = args.per_row
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df.to_csv(path, index=False, encoding="utf-8")
        else:
            if suffix not in (".json", ".jsonl"):
                print(
                    f"Warning: unknown suffix {path.suffix!r}; writing JSON array.",
                    file=sys.stderr,
                )
            path.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote per-row results to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
