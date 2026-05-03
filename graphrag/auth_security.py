"""Password hashing and JWT session helpers for local auth."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import bcrypt
import jwt

ALGO = "HS256"
COOKIE_NAME = "graphrag_auth"

_runtime_secret: Optional[str] = None


def configure_jwt_secret(secret: str) -> None:
    """Call once at app startup (from env or generated ephemeral secret)."""
    global _runtime_secret
    _runtime_secret = secret.strip()


def _jwt_secret() -> str:
    if _runtime_secret:
        return _runtime_secret
    env = (os.getenv("AUTH_JWT_SECRET") or "").strip()
    if env:
        return env
    raise RuntimeError("JWT secret not configured (call configure_jwt_secret or set AUTH_JWT_SECRET)")


def _session_days() -> int:
    raw = (os.getenv("AUTH_SESSION_DAYS") or "14").strip()
    try:
        return max(1, min(90, int(raw)))
    except ValueError:
        return 14


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(plain: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), password_hash.encode("utf-8"))
    except Exception:
        return False


def create_access_token(*, user_id: int, email: str) -> str:
    now = datetime.now(timezone.utc)
    exp = now + timedelta(days=_session_days())
    payload: Dict[str, Any] = {
        "sub": str(user_id),
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int(exp.timestamp()),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=ALGO)


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        return jwt.decode(token, _jwt_secret(), algorithms=[ALGO])
    except jwt.PyJWTError:
        return None
