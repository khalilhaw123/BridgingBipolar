"""Build QdrantClient for local host/port or Qdrant Cloud (URL + API key)."""

import os
import sys
import warnings
from typing import Optional
from urllib.parse import urlparse

from qdrant_client import QdrantClient


def _qdrant_compat_kwargs() -> dict:
    """Avoid noisy version checks when the server is temporarily unreachable."""
    if os.getenv("QDRANT_CHECK_COMPATIBILITY", "").strip().lower() in ("1", "true", "yes"):
        return {}
    # qdrant-client >= 1.7 (ignored on older clients via try/except in _make_qdrant_client)
    return {"check_compatibility": False}


def _make_qdrant_client(**kwargs) -> QdrantClient:
    extra = _qdrant_compat_kwargs()
    try:
        return QdrantClient(**kwargs, **extra)
    except TypeError:
        return QdrantClient(**kwargs)


def build_qdrant_client(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> QdrantClient:
    """
    Priority:
    1) Explicit url or env QDRANT_URL
    2) QDRANT_HOST looks like https://...
    3) host + port (local)
    """
    url = (url or os.getenv("QDRANT_URL") or "").strip()
    api_key = api_key if api_key is not None else os.getenv("QDRANT_API_KEY")
    if api_key is not None:
        api_key = api_key.strip() or None

    if url:
        url = url.rstrip("/")
        kwargs: dict = {"url": url}
        if api_key:
            kwargs["api_key"] = api_key
        return _make_qdrant_client(**kwargs)

    host_val = (host or os.getenv("QDRANT_HOST") or "localhost").strip()
    port_val = port if port is not None else int(os.getenv("QDRANT_PORT", "6333"))

    if host_val.startswith("http://") or host_val.startswith("https://"):
        parsed = urlparse(host_val)
        base = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        kwargs = {"url": base}
        if api_key:
            kwargs["api_key"] = api_key
        return _make_qdrant_client(**kwargs)

    return _make_qdrant_client(host=host_val, port=port_val)


def list_qdrant_collection_names(client: QdrantClient) -> list[str]:
    resp = client.get_collections()
    return sorted({c.name for c in resp.collections})


def _collection_resolution_note(preferred: str, chosen: str, reason: str) -> None:
    msg = (
        f"[graphrag] Qdrant: collection {preferred!r} not found; using {chosen!r} ({reason}). "
        f"Set QDRANT_COLLECTION={chosen!r} in .env to match."
    )
    print(msg, file=sys.stderr)
    warnings.warn(msg, UserWarning, stacklevel=3)


def resolve_qdrant_collection_name(client: QdrantClient, preferred: str) -> str:
    """
    Pick a collection to use for retrieval.

    1) If ``preferred`` exists → use it.
    2) If QDRANT_COLLECTION_STRICT=1 and preferred missing → raise.
    3) If exactly one collection exists → use it.
    4) If ``bipolar_chunks`` exists → use it.
    5) If any name contains 'bipolar' or 'chunk' (case-insensitive) → use first match.
    6) Else → first name alphabetically (last resort).

    Raises if there are no collections at all.
    """
    names = list_qdrant_collection_names(client)
    if not names:
        raise RuntimeError(
            "Qdrant has no collections. Run: python run_ingestion.py "
            "(same QDRANT_URL / QDRANT_API_KEY as this app)."
        )

    pref = (preferred or "bipolar_chunks").strip()
    strict = os.getenv("QDRANT_COLLECTION_STRICT", "").strip().lower() in ("1", "true", "yes")

    if pref in names:
        return pref

    if strict:
        raise RuntimeError(
            f"Qdrant has no collection named {pref!r} (strict mode). "
            f"Existing: {names}. Set QDRANT_COLLECTION to one of these, or unset QDRANT_COLLECTION_STRICT."
        )

    if len(names) == 1:
        chosen = names[0]
        _collection_resolution_note(pref, chosen, "only collection on this cluster")
        return chosen

    if "bipolar_chunks" in names:
        _collection_resolution_note(pref, "bipolar_chunks", "default ingest name exists")
        return "bipolar_chunks"

    for n in names:
        low = n.lower()
        if "bipolar" in low or "chunk" in low:
            _collection_resolution_note(pref, n, "name matched 'bipolar' or 'chunk'")
            return n

    chosen = sorted(names)[0]
    _collection_resolution_note(pref, chosen, f"fallback among {len(names)} collections")
    return chosen


def require_qdrant_collection(client: QdrantClient, collection_name: str) -> None:
    """Strict check: preferred name must exist (used when auto-resolve is disabled)."""
    names = list_qdrant_collection_names(client)
    if collection_name not in names:
        raise RuntimeError(
            f"Qdrant has no collection named {collection_name!r}. "
            f"Existing collections: {names or ['(none)']}. "
            f"Fix: use the same QDRANT_URL / QDRANT_API_KEY as ingest, set QDRANT_COLLECTION if you used a custom name, "
            f"then run: python run_ingestion.py"
        )
