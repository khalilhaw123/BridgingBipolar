from pathlib import Path
import numpy as np
import librosa
import cv2


SAMPLE_RATE = 22050
DURATION = 3.0
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
IMG_HEIGHT = 128
IMG_WIDTH = 128

TARGET_SR_EMOTION = 16000


def compute_mel_spectrogram(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        target_length = int(SAMPLE_RATE * DURATION)
        y = np.pad(y, (0, max(0, target_length - len(y))))[:target_length]

        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        min_val = mel_db.min()
        max_val = mel_db.max()

        mel_norm = (mel_db - min_val) / (max_val - min_val + 1e-9)

        mel_resized = cv2.resize(
            mel_norm,
            (IMG_WIDTH, IMG_HEIGHT),
            interpolation=cv2.INTER_LINEAR
        )

        return mel_resized[..., np.newaxis].astype(np.float32)

    except Exception as e:
        raise RuntimeError(f"Error while computing mel spectrogram: {str(e)}")


def load_audio_fixed(file_path, sr=TARGET_SR_EMOTION, duration=3.0):
    y, _ = librosa.load(file_path, sr=sr, duration=duration)

    target_length = int(sr * duration)
    y = np.pad(y, (0, max(0, target_length - len(y))))[:target_length]

    return y.astype(np.float32)


def estimate_speech_rate(file_path, sr=SAMPLE_RATE, duration=DURATION):
    """
    Estimate speech rate using the same logic as tachyphemia_detector.ipynb.

    Method:
    - Load the first 3 seconds of audio
    - Compute onset envelope
    - Detect acoustic onsets as approximate syllable starts
    - speech_rate = number_of_onsets / audio_duration
    - tempo = librosa tempo estimation
    """

    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration)

        if y is None or len(y) == 0:
            return 0.0, 0.0

        # Normalize amplitude
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak

        # Onset envelope
        onset_env = librosa.onset.onset_strength(
            y=y,
            sr=sr,
            hop_length=HOP_LENGTH
        )

        # Detect onsets = approximate syllable starts
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=HOP_LENGTH,
            backtrack=True
        )

        n_onsets = len(onsets)

        # Real analyzed duration
        actual_duration = len(y) / sr

        if actual_duration <= 0:
            return 0.0, 0.0

        # Original notebook logic
        speech_rate = n_onsets / actual_duration

        # Tempo estimation
        try:
            tempo = float(np.mean(
                librosa.feature.tempo(
                    onset_envelope=onset_env,
                    sr=sr,
                    hop_length=HOP_LENGTH
                )
            ))
        except Exception:
            tempo = speech_rate * 60.0

        return round(float(tempo), 2), round(float(speech_rate), 3)

    except Exception as e:
        print(f"Speech rate estimation failed: {e}")
        return 0.0, 0.0