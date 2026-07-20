from pathlib import Path
from typing import Any

import librosa
import numpy as np

from src.config import (
    DURATION,
    HOP_LENGTH,
    N_FFT,
    N_MFCC,
    SAMPLE_RATE,
)


def add_summary_statistics(
    output: dict[str, Any],
    feature_name: str,
    feature_values: np.ndarray,
) -> None:
    """
    Add mean and standard-deviation values for an audio feature.

    Two-dimensional features such as MFCCs and chroma receive
    one mean and standard deviation for every coefficient.
    """
    values = np.asarray(feature_values, dtype=np.float64)

    if values.ndim == 1:
        values = values[np.newaxis, :]

    number_of_rows = values.shape[0]

    for row_index, row in enumerate(values, start=1):
        if number_of_rows == 1:
            column_prefix = feature_name
        else:
            column_prefix = f"{feature_name}_{row_index:02d}"

        output[f"{column_prefix}_mean"] = float(np.mean(row))
        output[f"{column_prefix}_std"] = float(np.std(row))


def prepare_audio(
    audio_path: str | Path,
) -> tuple[np.ndarray, int, float]:
    """
    Load an audio file and make its length consistent.

    Returns:
        signal:
            Mono audio signal.

        sample_rate:
            Audio sampling rate.

        original_duration:
            Duration before padding or trimming.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file was not found: {audio_path}"
        )

    signal, sample_rate = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=DURATION,
    )

    if signal.size == 0:
        raise ValueError(
            f"No audio samples were loaded from: {audio_path}"
        )

    if not np.isfinite(signal).all():
        raise ValueError(
            f"Audio contains invalid numerical values: {audio_path}"
        )

    original_duration = float(
        librosa.get_duration(
            y=signal,
            sr=sample_rate,
        )
    )

    target_length = int(
        SAMPLE_RATE * DURATION
    )

    if len(signal) < target_length:
        signal = np.pad(
            signal,
            pad_width=(0, target_length - len(signal)),
            mode="constant",
        )
    else:
        signal = signal[:target_length]

    return signal, sample_rate, original_duration


def extract_audio_features(
    audio_path: str | Path,
) -> dict[str, float]:
    """
    Extract numerical audio features from one music track.
    """
    signal, sample_rate, original_duration = prepare_audio(
        audio_path
    )

    # Short-time Fourier transform
    stft_complex = librosa.stft(
        y=signal,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    stft_magnitude = np.abs(stft_complex)
    power_spectrogram = stft_magnitude ** 2

    # Mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        S=power_spectrogram,
        sr=sample_rate,
        n_mels=128,
    )

    mel_db = librosa.power_to_db(
        mel_spectrogram,
        ref=np.max,
    )

    # Audio features
    mfcc = librosa.feature.mfcc(
        S=mel_db,
        sr=sample_rate,
        n_mfcc=N_MFCC,
    )

    chroma = librosa.feature.chroma_stft(
        S=power_spectrogram,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    rms = librosa.feature.rms(
        S=stft_magnitude,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )

    zero_crossing_rate = (
        librosa.feature.zero_crossing_rate(
            y=signal,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH,
        )
    )

    spectral_centroid = (
        librosa.feature.spectral_centroid(
            S=stft_magnitude,
            sr=sample_rate,
        )
    )

    spectral_bandwidth = (
        librosa.feature.spectral_bandwidth(
            S=stft_magnitude,
            sr=sample_rate,
        )
    )

    spectral_rolloff = (
        librosa.feature.spectral_rolloff(
            S=stft_magnitude,
            sr=sample_rate,
            roll_percent=0.85,
        )
    )

    spectral_contrast = (
        librosa.feature.spectral_contrast(
            S=stft_magnitude,
            sr=sample_rate,
        )
    )

    onset_envelope = librosa.onset.onset_strength(
        S=mel_db,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
    )

    tempo_values = librosa.feature.tempo(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
    )

    features: dict[str, float] = {
        "loaded_duration_seconds": original_duration,
        "tempo_bpm": (
            float(tempo_values[0])
            if len(tempo_values) > 0
            else 0.0
        ),
    }

    add_summary_statistics(
        features,
        "mfcc",
        mfcc,
    )

    add_summary_statistics(
        features,
        "chroma",
        chroma,
    )

    add_summary_statistics(
        features,
        "rms",
        rms,
    )

    add_summary_statistics(
        features,
        "zero_crossing_rate",
        zero_crossing_rate,
    )

    add_summary_statistics(
        features,
        "spectral_centroid",
        spectral_centroid,
    )

    add_summary_statistics(
        features,
        "spectral_bandwidth",
        spectral_bandwidth,
    )

    add_summary_statistics(
        features,
        "spectral_rolloff",
        spectral_rolloff,
    )

    add_summary_statistics(
        features,
        "spectral_contrast",
        spectral_contrast,
    )

    return features