"""Verify Neo4j and Qdrant using .env next to this script."""

from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")

from neo4j import GraphDatabase
import os

from graphrag.qdrant_util import build_qdrant_client, list_qdrant_collection_names, resolve_qdrant_collection_name


def main() -> None:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            row = session.run("RETURN 1 AS ok").single()
            assert row and row["ok"] == 1
        driver.close()
        print(f"Neo4j: OK ({uri})")
    except Exception as exc:
        print(f"Neo4j: FAILED ({uri}) -> {exc}")

    try:
        client = build_qdrant_client()
        names = list_qdrant_collection_names(client)
        want = os.getenv("QDRANT_COLLECTION", "bipolar_chunks")
        url = (os.getenv("QDRANT_URL") or "").strip() or f"{os.getenv('QDRANT_HOST', 'localhost')}:{os.getenv('QDRANT_PORT', '6333')}"
        print(f"Qdrant: OK ({len(names)} collection(s)) at {url}")
        print(f"  collections: {names or '(none)'}")
        if names:
            resolved = resolve_qdrant_collection_name(client, want)
            print(f"  QDRANT_COLLECTION in .env: {want!r} -> retriever will use: {resolved!r}")
        else:
            print("  Run: python run_ingestion.py")
    except Exception as exc:
        print(f"Qdrant: FAILED -> {exc}")


if __name__ == "__main__":
    main()
