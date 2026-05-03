import json
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from graphrag.qdrant_util import build_qdrant_client
from graphrag.schema import build_schema_ddl


SECTION_RE = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$")
BULLET_RE = re.compile(r"^\s*[•\-]\s*(.+?)\s*$")

ENTITY_PATTERNS: Dict[str, Sequence[str]] = {
    "EpisodeType": ["mania", "hypomania", "major depressive episode", "mixed episodes"],
    "Risk": ["suicide", "self-neglect", "substance abuse", "psychosis"],
    "Trigger": ["stress", "sleep deprivation", "life changes", "trauma", "substance use"],
    "Treatment": [
        "lithium",
        "valproate",
        "lamotrigine",
        "olanzapine",
        "quetiapine",
        "risperidone",
        "cbt",
        "electroconvulsive therapy",
        "tms",
    ],
    "Comorbidity": ["anxiety", "adhd", "eating disorders", "substance use disorders"],
}


@dataclass
class Chunk:
    chunk_id: str
    source_path: str
    section_id: Optional[int]
    section_title: Optional[str]
    text: str
    token_estimate: int
    lang: str
    clinical_priority: str
    doc_version: str


@dataclass
class ExtractedEdge:
    source_label: str
    source_name: str
    relation: str
    target_label: str
    target_name: str
    chunk_id: str


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def split_long_text(text: str, target_words: int = 120, overlap_words: int = 20) -> List[str]:
    words = text.split()
    if len(words) <= target_words:
        return [text]
    parts: List[str] = []
    step = max(1, target_words - overlap_words)
    start = 0
    while start < len(words):
        end = min(len(words), start + target_words)
        parts.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return parts


def parse_entries(path: Path) -> List[Tuple[Optional[int], Optional[str], str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    entries: List[Tuple[Optional[int], Optional[str], str]] = []
    current_section_id: Optional[int] = None
    current_section_title: Optional[str] = None
    free_text: List[str] = []

    def flush_free_text() -> None:
        nonlocal free_text
        if free_text:
            merged = _clean(" ".join(free_text))
            if merged:
                entries.append((current_section_id, current_section_title, merged))
            free_text = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        match_section = SECTION_RE.match(stripped)
        if match_section:
            new_id = int(match_section.group(1))
            new_title = _clean(match_section.group(2))
            # Lines before the first "N. Title" (e.g. document title) attach to that section, not (None, None).
            if free_text and current_section_id is None:
                merged = _clean(" ".join(free_text))
                free_text = []
                if merged:
                    entries.append((new_id, new_title, merged))
            else:
                flush_free_text()
            current_section_id = new_id
            current_section_title = new_title
            continue
        match_bullet = BULLET_RE.match(stripped)
        if match_bullet:
            flush_free_text()
            entries.append((current_section_id, current_section_title, _clean(match_bullet.group(1))))
            continue
        free_text.append(stripped)

    flush_free_text()
    return entries


def build_chunks(
    source_path: Path,
    target_words: int = 120,
    lang: str = "en",
    doc_version: str = "v1",
) -> List[Chunk]:
    chunks: List[Chunk] = []
    entries = parse_entries(source_path)
    counter = 1
    for section_id, section_title, text in entries:
        parts = split_long_text(text, target_words=target_words)
        for part in parts:
            priority = "high" if section_id in {5, 6, 7} else "normal"
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{counter:04d}",
                    source_path=str(source_path.name),
                    section_id=section_id,
                    section_title=section_title,
                    text=part,
                    token_estimate=estimate_tokens(part),
                    lang=lang,
                    clinical_priority=priority,
                    doc_version=doc_version,
                )
            )
            counter += 1
    return chunks


def extract_edges(chunks: Iterable[Chunk], condition_name: str = "Bipolar Disorder") -> List[ExtractedEdge]:
    edges: List[ExtractedEdge] = []
    for chunk in chunks:
        text = chunk.text.lower()
        for label, terms in ENTITY_PATTERNS.items():
            for term in terms:
                if term in text:
                    relation = "HAS_EPISODE_TYPE" if label == "EpisodeType" else "SUPPORTED_BY_CHUNK"
                    if label == "Risk":
                        relation = "INCREASES_RISK_OF"
                    elif label == "Trigger":
                        relation = "TRIGGERED_BY"
                    elif label == "Treatment":
                        relation = "TREATED_BY"
                    elif label == "Comorbidity":
                        relation = "CO_OCCURS_WITH"
                    edges.append(
                        ExtractedEdge(
                            source_label="Condition",
                            source_name=condition_name,
                            relation=relation,
                            target_label=label,
                            target_name=term.title(),
                            chunk_id=chunk.chunk_id,
                        )
                    )
    return edges


def embed_chunks(chunks: Sequence[Chunk], model_name: str, batch_size: int = 8) -> np.ndarray:
    model = SentenceTransformer(model_name)
    vectors = model.encode(
        [chunk.text for chunk in chunks],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return vectors.astype(np.float32)


def save_local_artifacts(chunks: Sequence[Chunk], vectors: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    np.save(output_dir / "embeddings.npy", vectors)


def upsert_neo4j(
    uri: str,
    user: str,
    password: str,
    chunks: Sequence[Chunk],
    edges: Sequence[ExtractedEdge],
) -> None:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    ddl = build_schema_ddl()
    with driver.session() as session:
        for q in ddl.constraints + ddl.indexes:
            session.run(q)
        for chunk in chunks:
            section_key = f"{chunk.source_path}:{chunk.section_id}"
            session.run(
                """
                MERGE (s:Section {section_key: $section_key})
                SET s.section_id = $section_id, s.title = $section_title, s.source_path = $source_path
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.text = $text, c.lang = $lang, c.token_estimate = $token_estimate,
                    c.clinical_priority = $clinical_priority, c.doc_version = $doc_version, c.section_id = $section_id
                MERGE (s)-[:HAS_CHUNK]->(c)
                """,
                **asdict(chunk),
                section_key=section_key,
            )
        for edge in edges:
            # Provenance as a relationship property — Neo4j cannot attach a relationship to another relationship.
            query = f"""
            MERGE (src:{edge.source_label} {{name: $source_name}})
            MERGE (dst:{edge.target_label} {{name: $target_name}})
            MERGE (src)-[r:{edge.relation}]->(dst)
            SET r.chunk_id = $chunk_id
            """
            session.run(
                query,
                source_name=edge.source_name,
                target_name=edge.target_name,
                chunk_id=edge.chunk_id,
            )
    driver.close()


def upsert_qdrant(
    collection_name: str,
    chunks: Sequence[Chunk],
    vectors: np.ndarray,
    client: Optional[QdrantClient] = None,
) -> None:
    client = client or build_qdrant_client()
    dim = int(vectors.shape[1])
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    points: List[PointStruct] = []
    for idx, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[idx].tolist(),
                payload=asdict(chunk),
            )
        )
    client.upsert(collection_name=collection_name, points=points)

