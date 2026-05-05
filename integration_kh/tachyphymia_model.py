import numpy as np
from tensorflow import keras

from audio_processing import compute_mel_spectrogram, estimate_speech_rate


def load_tachyphymia_model(model_path="models/tachyphemia_cnn_final.keras"):
    model = keras.models.load_model(model_path)
    return model


def predict_tachyphemia(file_path, model):
    spec = compute_mel_spectrogram(file_path)

    prediction = model.predict(
        np.expand_dims(spec, axis=0),
        verbose=0
    )

    prob = float(prediction[0][0])

    label = "tachyphemia" if prob >= 0.5 else "normal"

    tempo_bpm, speech_rate = estimate_speech_rate(file_path)

    agitation_score = round(
        0.6 * prob + 0.4 * min(speech_rate / 8.0, 1.0),
        4
    )

    return {
        "cnn_label": label,
        "cnn_probability": round(prob, 4),
        "tempo_bpm": round(tempo_bpm, 1),
        "speech_rate_syl_per_sec": round(speech_rate, 2),
        "agitation_score": agitation_score
    }

