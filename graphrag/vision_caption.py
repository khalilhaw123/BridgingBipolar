"""Caption user images via Ollama /api/chat (vision models)."""

from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from typing import Optional

import requests
from PIL import Image, ImageOps

from graphrag.ollama_env import default_ollama_vision_model, ollama_chat_url


def _env_int(name: str, default: int, *, lo: int, hi: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        return default
    return max(lo, min(hi, v))


def _prepare_image_bytes_for_vision(image_bytes: bytes) -> bytes:
    """
    Downscale and re-encode as JPEG so Ollama vision uses less VRAM (large photos
    often crash the runner with 'llama runner process has terminated').
    """
    max_side = _env_int("VISION_IMAGE_MAX_SIDE", 1024, lo=256, hi=4096)
    quality = _env_int("VISION_JPEG_QUALITY", 85, lo=60, hi=95)

    buf_in = BytesIO(image_bytes)
    im = Image.open(buf_in)
    im = ImageOps.exif_transpose(im)

    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")

    w, h = im.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:  # pragma: no cover - older Pillow
            resample = Image.LANCZOS  # type: ignore[attr-defined]
        im = im.resize((nw, nh), resample)

    out = BytesIO()
    im.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()


def _ollama_http_error_detail(resp: requests.Response) -> str:
    """Extract Ollama JSON `error` field or a short text body for debugging."""
    try:
        data = resp.json()
        err = data.get("error")
        if err:
            return str(err).strip()
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    text = (resp.text or "").strip()
    if text:
        return text[:2000]
    return resp.reason or f"HTTP {resp.status_code}"

VISION_SYSTEM = (
    "You describe images for a supportive mental-health companion app. "
    "Be factual about what is visible (people, objects, setting, text in image, mood cues). "
    "Do not diagnose medical or psychiatric conditions from pixels. "
    "If something is unclear, say so. Answer in the same language as any clear text in the image; "
    "otherwise use the same language as the user's note if provided, else English."
)

VISION_USER_TEMPLATE = (
    "{user_hint}"
    "Describe this image in plain language for the assistant that will reply next. "
    "Stay under 200 words. No greeting."
)


def caption_image_bytes(
    image_bytes: bytes,
    *,
    model: Optional[str] = None,
    user_note: str = "",
    ollama_chat_endpoint: Optional[str] = None,
    timeout_sec: int = 120,
) -> str:
    """
    Call Ollama vision model with base64 image; return trimmed caption text.
    Raises RuntimeError on Ollama HTTP errors (message includes server `error` JSON if present).
    Raises ValueError on empty model/response.
    """
    if not image_bytes:
        raise ValueError("Empty image bytes")
    model = (model or default_ollama_vision_model()).strip()
    if not model:
        raise ValueError("OLLAMA_VISION_MODEL / vision model is empty")
    url = ollama_chat_endpoint or ollama_chat_url()
    hint = (user_note or "").strip()
    user_block = VISION_USER_TEMPLATE.format(
        user_hint=(f"User added this note: {hint}\n\n" if hint else ""),
    )
    try:
        prepared = _prepare_image_bytes_for_vision(image_bytes)
    except Exception as exc:
        raise ValueError(f"Could not decode or resize image: {exc}") from exc
    b64 = base64.b64encode(prepared).decode("ascii")
    # Single user message + images matches Ollama vision docs; some builds error on
    # system + user with images in one request.
    combined_prompt = f"{VISION_SYSTEM}\n\n{user_block}"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": combined_prompt, "images": [b64]},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 384,
            "num_ctx": _env_int("OLLAMA_VISION_NUM_CTX", 2048, lo=512, hi=8192),
        },
    }
    resp = requests.post(url, json=payload, timeout=timeout_sec)
    if not resp.ok:
        detail = _ollama_http_error_detail(resp)
        raise RuntimeError(f"Ollama /api/chat {resp.status_code}: {detail}") from None
    data = resp.json()
    msg = data.get("message") or {}
    text = (msg.get("content") or "").strip()
    if not text:
        raise ValueError("Vision model returned empty caption")
    return text
