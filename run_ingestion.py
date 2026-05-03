import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from graphrag.ingestion import (
    build_chunks,
    embed_chunks,
    extract_edges,
    save_local_artifacts,
    upsert_neo4j,
    upsert_qdrant,
)

load_dotenv(Path(__file__).resolve().parent / ".env")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GraphRAG ingestion pipeline.")
    parser.add_argument("--input", default="rag.txt")
    parser.add_argument("--output_dir", default="artifacts")
    parser.add_argument("--model", default="intfloat/multilingual-e5-large")
    parser.add_argument("--target_words", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--doc_version", default="v1")
    parser.add_argument("--qdrant_collection", default="bipolar_chunks")
    parser.add_argument("--skip_neo4j", action="store_true")
    parser.add_argument("--skip_qdrant", action="store_true")
    args = parser.parse_args()

    chunks = build_chunks(Path(args.input), target_words=args.target_words, lang=args.lang, doc_version=args.doc_version)
    vectors = embed_chunks(chunks, model_name=args.model, batch_size=args.batch_size)
    edges = extract_edges(chunks)
    save_local_artifacts(chunks, vectors, Path(args.output_dir))

    if not args.skip_neo4j:
        upsert_neo4j(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            chunks=chunks,
            edges=edges,
        )
    if not args.skip_qdrant:
        collection = os.getenv("QDRANT_COLLECTION", args.qdrant_collection)
        upsert_qdrant(
            collection_name=collection,
            chunks=chunks,
            vectors=vectors,
        )
        print(f"Qdrant: created/updated collection {collection!r} (same QDRANT_URL/API as .env)")
    print(f"Ingestion complete. chunks={len(chunks)}, edges={len(edges)}")


if __name__ == "__main__":
    main()

