import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

from audio_processing import load_audio_fixed, TARGET_SR_EMOTION


LABELS = ["happy", "sad", "angry", "neutral"]


def load_emotion_model(model_path="models/wav2vec2-emotion"):
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return processor, model, device


def predict_emotion(file_path, processor, model, device):
    try:
        audio = load_audio_fixed(file_path)

        inputs = processor(
            audio,
            sampling_rate=TARGET_SR_EMOTION,
            return_tensors="pt"
        )

        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            logits = model(input_values).logits

        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))

        return {
            "emotion": LABELS[pred_idx],
            "probabilities": {
                LABELS[i]: round(float(probs[i]), 4)
                for i in range(len(LABELS))
            },
            "confidence": round(float(probs[pred_idx]), 4)
        }

    except Exception as e:
        return {
            "error": str(e),
            "emotion": "neutral",
            "probabilities": {},
            "confidence": 0.0
        }