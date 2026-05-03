"""Speech-to-text (Whisper / faster-whisper) and on-demand TTS (edge-tts)."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Optional

# Max upload size for voice (bytes)
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_UPLOAD_MB", "12")) * 1024 * 1024
MAX_TTS_CHARS = int(os.getenv("MAX_TTS_CHARS", "8000"))


def _suffix_from_filename(name: str) -> str:
    suf = Path(name).suffix.lower()
    if suf in {".wav", ".webm", ".mp3", ".ogg", ".m4a", ".flac"}:
        return suf
    return ".webm"


def transcribe_audio_bytes(
    data: bytes,
    *,
    original_filename: str = "recording.webm",
    language: Optional[str] = None,
) -> str:
    """
    Transcribe raw audio bytes using faster-whisper (Whisper weights).
    ``language`` is an ISO code like ``fr`` or ``en``; omit for auto-detect.
    """
    if len(data) > MAX_AUDIO_BYTES:
        raise ValueError(f"Audio too large (max {MAX_AUDIO_BYTES // (1024 * 1024)} MB)")

    try:
        import faster_whisper  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "faster-whisper is required for voice input. Install: pip install faster-whisper"
        ) from exc

    model_name = os.getenv("WHISPER_MODEL", "small")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    suffix = _suffix_from_filename(original_filename)
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        model = _get_whisper_model(model_name, device=device, compute_type=compute_type)
        kwargs: dict = {"beam_size": int(os.getenv("WHISPER_BEAM_SIZE", "5"))}
        if language:
            kwargs["language"] = language
        segments, _info = model.transcribe(tmp_path, **kwargs)
        parts = [s.text for s in segments]
        return "".join(parts).strip()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


_whisper_model = None
_whisper_model_key: Optional[tuple] = None


def _get_whisper_model(model_name: str, *, device: str, compute_type: str):
    """Load Whisper once per (model_name, device, compute_type)."""
    global _whisper_model, _whisper_model_key
    from faster_whisper import WhisperModel

    key = (model_name, device, compute_type)
    if _whisper_model is None or _whisper_model_key != key:
        _whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)
        _whisper_model_key = key
    return _whisper_model


def _edge_voice_for_lang(lang: Optional[str]) -> str:
    """Pick a default neural voice per language code."""
    if not lang:
        return "en-US-JennyNeural"
    low = lang.lower()
    if low.startswith("fr"):
        return "fr-FR-DeniseNeural"
    if low.startswith("ar"):
        return "ar-SA-ZariyahNeural"
    if low.startswith("en"):
        return "en-US-JennyNeural"
    return "en-US-JennyNeural"


async def synthesize_speech_mp3(text: str, *, lang: Optional[str] = None, voice: Optional[str] = None) -> bytes:
    """Synthesize ``text`` to MP3 bytes using edge-tts (Microsoft Edge voices, no API key)."""
    try:
        import edge_tts
    except ImportError as exc:  # pragma: no cover
        raise ImportError("edge-tts is required for TTS. Install: pip install edge-tts") from exc

    if len(text) > MAX_TTS_CHARS:
        raise ValueError(f"Text too long for TTS (max {MAX_TTS_CHARS} characters)")
    if not text.strip():
        raise ValueError("Empty text")

    chosen_voice = voice or os.getenv("EDGE_TTS_VOICE") or _edge_voice_for_lang(lang)
    communicate = edge_tts.Communicate(text.strip(), chosen_voice)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        out_path = tmp.name
    try:
        await communicate.save(out_path)
        return Path(out_path).read_bytes()
    finally:
        if os.path.exists(out_path):
            os.unlink(out_path)


def synthesize_speech_mp3_sync(text: str, *, lang: Optional[str] = None, voice: Optional[str] = None) -> bytes:
    """Run :func:`synthesize_speech_mp3` from sync code (e.g. tests)."""
    return asyncio.run(synthesize_speech_mp3(text, lang=lang, voice=voice))
