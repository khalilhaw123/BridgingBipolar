"""Resolve Ollama base URL and default model names from environment."""

from __future__ import annotations

import os
from urllib.parse import urlparse


def _strip_trailing_slash(url: str) -> str:
    return url.rstrip("/")


def ollama_base_url() -> str:
    """Host + optional port, without /api/* path. Works when OLLAMA_URL is generate or chat URL."""
    raw = (os.getenv("OLLAMA_URL") or "http://localhost:11434/api/generate").strip()
    if not raw:
        return "http://localhost:11434"
    lower = raw.lower()
    for suffix in ("/api/generate", "/api/chat"):
        if lower.endswith(suffix):
            return _strip_trailing_slash(raw[: -len(suffix)]) or "http://localhost:11434"
    parsed = urlparse(raw)
    if parsed.scheme and parsed.netloc:
        return _strip_trailing_slash(f"{parsed.scheme}://{parsed.netloc}")
    return _strip_trailing_slash(raw) or "http://localhost:11434"


def ollama_generate_url() -> str:
    """Full URL for Ollama text /api/generate (matches legacy OLLAMA_URL default)."""
    explicit = (os.getenv("OLLAMA_URL") or "").strip()
    if explicit.lower().endswith("/api/generate"):
        return explicit
    return f"{ollama_base_url()}/api/generate"


def ollama_chat_url() -> str:
    """Full URL for Ollama /api/chat (vision and multimodal)."""
    return f"{ollama_base_url()}/api/chat"


def default_ollama_chat_model() -> str:
    return (os.getenv("OLLAMA_CHAT_MODEL") or os.getenv("OLLAMA_MODEL") or "qwen2.5:3b-instruct").strip()


def default_ollama_vision_model() -> str:
    return (os.getenv("OLLAMA_VISION_MODEL") or "llava:7b").strip()
