"""Redis-backed short-term session memory for chat turns."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionMemoryStore:
    """Short-term memory backed by Redis lists."""

    redis_client: Optional[object] = None
    key_prefix: str = "graphrag:session_memory"
    ttl_seconds: int = 24 * 60 * 60
    max_turns: int = 10
    available: bool = False
    last_error: Optional[str] = None

    @classmethod
    def from_env(cls) -> "SessionMemoryStore":
        store = cls(
            key_prefix=(os.getenv("SESSION_MEMORY_KEY_PREFIX") or "graphrag:session_memory").strip(),
            ttl_seconds=max(60, _env_int("SESSION_MEMORY_TTL_SECONDS", 24 * 60 * 60)),
            max_turns=max(1, _env_int("SESSION_MEMORY_MAX_TURNS", 10)),
        )
        redis_url = (os.getenv("REDIS_URL") or "").strip()
        if not redis_url:
            store.last_error = "REDIS_URL not set"
            return store
        try:
            import redis  # type: ignore

            client = redis.Redis.from_url(redis_url, decode_responses=True)
            client.ping()
            store.redis_client = client
            store.available = True
            return store
        except Exception as exc:  # pragma: no cover - environment specific
            store.last_error = str(exc)
            return store

    def _key(self, session_id: str) -> str:
        return f"{self.key_prefix}:{session_id}"

    def get_recent_messages(self, session_id: str) -> List[Dict[str, str]]:
        """Return chronological role/content messages for this session."""
        if not self.available or not self.redis_client:
            return []
        key = self._key(session_id)
        try:
            # list holds one item per turn; each item has {"user":"...", "assistant":"...", ...}
            rows = self.redis_client.lrange(key, 0, self.max_turns - 1)
        except Exception:  # pragma: no cover
            return []
        out: List[Dict[str, str]] = []
        for raw in reversed(rows):
            try:
                turn = json.loads(raw)
            except Exception:
                continue
            user_text = (turn.get("user") or "").strip()
            assistant_text = (turn.get("assistant") or "").strip()
            if user_text:
                out.append({"role": "user", "content": user_text})
            if assistant_text:
                out.append({"role": "assistant", "content": assistant_text})
        return out

    def append_turn(self, session_id: str, *, user: str, assistant: str) -> None:
        if not self.available or not self.redis_client:
            return
        key = self._key(session_id)
        record = {
            "created_at": _utc_iso(),
            "user": user,
            "assistant": assistant,
        }
        try:
            pipe = self.redis_client.pipeline()
            pipe.lpush(key, json.dumps(record, ensure_ascii=False))
            pipe.ltrim(key, 0, self.max_turns - 1)
            pipe.expire(key, self.ttl_seconds)
            pipe.execute()
        except Exception:  # pragma: no cover
            return
