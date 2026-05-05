from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from audio_processing import TARGET_SR_EMOTION, load_audio_fixed
from emotion_model import LABELS


matplotlib.use("Agg")


def _normalize_array(values: np.ndarray) -> np.ndarray:
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    denom = max(max_val - min_val, 1e-9)
    return (values - min_val) / denom


def _smooth_signal(values: np.ndarray, window_size: int = 401) -> np.ndarray:
    if window_size <= 1 or len(values) < window_size:
        return values

    if window_size % 2 == 0:
        window_size += 1

    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.convolve(values, kernel, mode="same")


def _find_important_time_regions(
    saliency_by_time: np.ndarray,
    duration_sec: float,
    threshold_percentile: float = 85.0,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    threshold = float(np.percentile(saliency_by_time, threshold_percentile))
    mask = saliency_by_time >= threshold

    segments: List[Dict[str, float]] = []
    start_idx = None

    for idx, is_important in enumerate(mask):
        if is_important and start_idx is None:
            start_idx = idx
        elif not is_important and start_idx is not None:
            end_idx = idx - 1
            segments.append(
                {
                    "start_sec": round(start_idx / len(mask) * duration_sec, 3),
                    "end_sec": round((end_idx + 1) / len(mask) * duration_sec, 3),
                    "mean_importance": round(
                        float(np.mean(saliency_by_time[start_idx : end_idx + 1])),
                        4,
                    ),
                }
            )
            start_idx = None

    if start_idx is not None:
        segments.append(
            {
                "start_sec": round(start_idx / len(mask) * duration_sec, 3),
                "end_sec": round(duration_sec, 3),
                "mean_importance": round(float(np.mean(saliency_by_time[start_idx:])), 4),
            }
        )

    segments.sort(key=lambda item: item["mean_importance"], reverse=True)
    return mask, segments


def _compute_emotion_saliency(
    audio: np.ndarray,
    processor,
    model,
    device,
    target_index: int | None = None,
) -> Tuple[np.ndarray, int, np.ndarray]:
    inputs = processor(audio, sampling_rate=TARGET_SR_EMOTION, return_tensors="pt")

    input_values = inputs.input_values.to(device)
    input_values.requires_grad_(True)

    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    outputs = model(input_values=input_values, attention_mask=attention_mask)
    logits = outputs.logits

    if target_index is None:
        target_index = int(torch.argmax(logits, dim=-1).item())

    target_score = logits[0, target_index]

    model.zero_grad(set_to_none=True)
    target_score.backward()

    gradient = input_values.grad.detach().abs().squeeze(0).cpu().numpy()
    saliency = _normalize_array(_smooth_signal(gradient, window_size=401))

    probs = torch.nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()[0]

    if len(saliency) != len(audio):
        old_x = np.linspace(0, 1, num=len(saliency), dtype=np.float32)
        new_x = np.linspace(0, 1, num=len(audio), dtype=np.float32)
        saliency = np.interp(new_x, old_x, saliency)

    return saliency.astype(np.float32), target_index, probs.astype(np.float32)


def _plot_waveform_xai(
    audio: np.ndarray,
    saliency: np.ndarray,
    out_path: Path,
    pred_label: str,
    probs: np.ndarray,
) -> None:
    duration = len(audio) / float(TARGET_SR_EMOTION)
    time_axis = np.linspace(0, duration, num=len(audio), dtype=np.float32)

    important_mask, _ = _find_important_time_regions(saliency, duration)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(time_axis, audio, color="#2f5aa8", linewidth=1.0)

    for idx, is_important in enumerate(important_mask):
        if is_important:
            ax.axvspan(
                time_axis[idx],
                time_axis[min(idx + 1, len(time_axis) - 1)],
                color="red",
                alpha=0.18,
                linewidth=0,
            )

    prob_text = " ".join([f"{float(p):.3f}" for p in probs])
    ax.set_title(f"Waveform with important zones - pred: {pred_label} | probs: [{prob_text}]")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_spectrogram_xai(
    audio: np.ndarray,
    saliency: np.ndarray,
    out_path: Path,
) -> None:
    duration = len(audio) / float(TARGET_SR_EMOTION)

    n_fft = 1024
    hop_length = 256

    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    spec_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    frame_times = librosa.frames_to_time(
        np.arange(spec_db.shape[1]),
        sr=TARGET_SR_EMOTION,
        hop_length=hop_length,
    )

    saliency_time = np.linspace(0, duration, num=len(saliency), dtype=np.float32)
    frame_saliency = np.interp(frame_times, saliency_time, saliency)

    important_mask, _ = _find_important_time_regions(frame_saliency, duration)

    fig, ax = plt.subplots(figsize=(12, 4))
    librosa.display.specshow(
        spec_db,
        sr=TARGET_SR_EMOTION,
        hop_length=hop_length,
        x_axis="time",
        y_axis="hz",
        cmap="magma",
        ax=ax,
    )

    for frame_idx, is_important in enumerate(important_mask):
        if is_important:
            t0 = frame_times[frame_idx]
            t1 = frame_times[min(frame_idx + 1, len(frame_times) - 1)]
            ax.axvspan(t0, t1, color="red", alpha=0.24, linewidth=0)

    ax.set_title("Spectrogram with important time regions")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def generate_emotion_xai(
    file_path: str,
    original_name: str,
    processor,
    model,
    device,
) -> Dict[str, object]:
    audio = load_audio_fixed(file_path)

    saliency, pred_idx, probs = _compute_emotion_saliency(audio, processor, model, device)

    duration = len(audio) / float(TARGET_SR_EMOTION)
    _, segments = _find_important_time_regions(saliency, duration)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(original_name).stem

    out_dir = Path("processed") / "xai"
    out_dir.mkdir(parents=True, exist_ok=True)

    waveform_path = out_dir / f"{stem}_{timestamp}_waveform_xai.png"
    spectrogram_path = out_dir / f"{stem}_{timestamp}_spectrogram_xai.png"

    _plot_waveform_xai(
        audio=audio,
        saliency=saliency,
        out_path=waveform_path,
        pred_label=LABELS[pred_idx],
        probs=probs,
    )

    _plot_spectrogram_xai(
        audio=audio,
        saliency=saliency,
        out_path=spectrogram_path,
    )

    return {
        "predicted_emotion": LABELS[pred_idx],
        "probabilities": {LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))},
        "important_segments_top5": segments[:5],
        "plots": {
            "waveform": str(waveform_path).replace("\\", "/"),
            "spectrogram": str(spectrogram_path).replace("\\", "/"),
        },
        "plot_urls": {
            "waveform": "/" + str(waveform_path).replace("\\", "/"),
            "spectrogram": "/" + str(spectrogram_path).replace("\\", "/"),
        },
    }
