from __future__ import annotations

import tempfile
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from src.predict import predict_genre


# ---------------------------------------------------------
# Application settings
# ---------------------------------------------------------

SAMPLE_RATE = 22_050
MAX_DURATION_SECONDS = 30
N_FFT = 2_048
HOP_LENGTH = 512
N_MELS = 128

SUPPORTED_GENRES = [
    "Blues",
    "Classical",
    "Country",
    "Disco",
    "Hip-hop",
    "Jazz",
    "Metal",
    "Pop",
    "Reggae",
    "Rock",
]


st.set_page_config(
    page_title="Music Genre Classification",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def save_uploaded_file(
    uploaded_bytes: bytes,
    file_suffix: str,
) -> Path:
    """Save an uploaded audio file temporarily."""

    suffix = file_suffix.lower() or ".wav"

    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        suffix=suffix,
    ) as temporary_file:
        temporary_file.write(uploaded_bytes)
        return Path(temporary_file.name)


def load_audio_for_visualization(
    audio_path: Path,
) -> tuple[np.ndarray, int, float]:
    """Load audio for waveform and Mel-spectrogram display."""

    signal, sample_rate = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=MAX_DURATION_SECONDS,
    )

    if signal.size == 0:
        raise ValueError(
            "The uploaded audio file contains no usable audio."
        )

    if not np.isfinite(signal).all():
        raise ValueError(
            "The uploaded audio contains invalid numerical values."
        )

    duration_seconds = float(
        librosa.get_duration(
            y=signal,
            sr=sample_rate,
        )
    )

    return signal, sample_rate, duration_seconds


def create_waveform_figure(
    signal: np.ndarray,
    sample_rate: int,
) -> Figure:
    """Create an audio waveform."""

    figure, axis = plt.subplots(figsize=(11, 3.5))

    librosa.display.waveshow(
        signal,
        sr=sample_rate,
        ax=axis,
    )

    axis.set_title("Audio Waveform")
    axis.set_xlabel("Time in seconds")
    axis.set_ylabel("Amplitude")
    axis.grid(alpha=0.2)

    figure.tight_layout()

    return figure


def create_mel_spectrogram_figure(
    signal: np.ndarray,
    sample_rate: int,
) -> Figure:
    """Create a Mel-spectrogram."""

    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )

    mel_db = librosa.power_to_db(
        mel_spectrogram,
        ref=np.max,
    )

    figure, axis = plt.subplots(figsize=(11, 4.5))

    spectrogram_image = librosa.display.specshow(
        mel_db,
        sr=sample_rate,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        ax=axis,
    )

    figure.colorbar(
        spectrogram_image,
        ax=axis,
        format="%+2.0f dB",
    )

    axis.set_title("Mel-Spectrogram")
    axis.set_xlabel("Time in seconds")
    axis.set_ylabel("Mel frequency")

    figure.tight_layout()

    return figure


def display_top_predictions(
    predictions: list[dict],
) -> None:
    """Display ranked genre probabilities."""

    st.markdown("#### Top predictions")

    for rank, prediction in enumerate(
        predictions,
        start=1,
    ):
        genre = str(prediction["genre"]).title()
        probability = float(prediction["probability"])
        percentage = probability * 100

        st.markdown(
            f"**{rank}. {genre} — {percentage:.2f}%**"
        )

        st.progress(
            min(max(probability, 0.0), 1.0)
        )


# ---------------------------------------------------------
# Main interface
# ---------------------------------------------------------

st.title("🎵 Music Genre Classification")

st.write(
    "Upload one or multiple WAV music files. The machine-learning "
    "model will analyze every file and predict its music genre."
)

st.info(
    "For best results, use clear music clips of up to 30 seconds."
)

uploaded_files = st.file_uploader(
    "Upload WAV audio files",
    type=["wav"],
    accept_multiple_files=True,
)

show_visualizations = st.checkbox(
    "Generate waveform and Mel-spectrogram for every file",
    value=True,
)

if not uploaded_files:
    st.markdown("### Supported genres")
    st.write(", ".join(SUPPORTED_GENRES))

    st.divider()

    st.caption(
        "Upload one or multiple WAV files to start classification."
    )

else:
    total_size_bytes = sum(
        len(uploaded_file.getvalue())
        for uploaded_file in uploaded_files
    )

    total_size_mb = total_size_bytes / (1024 * 1024)

    file_count_column, size_column = st.columns(2)

    with file_count_column:
        st.metric(
            label="Selected Files",
            value=len(uploaded_files),
        )

    with size_column:
        st.metric(
            label="Combined Size",
            value=f"{total_size_mb:.2f} MB",
        )

    st.markdown("#### Selected files")

    selected_files_df = pd.DataFrame(
        [
            {
                "Filename": uploaded_file.name,
                "Size (MB)": round(
                    len(uploaded_file.getvalue())
                    / (1024 * 1024),
                    2,
                ),
            }
            for uploaded_file in uploaded_files
        ]
    )

    st.dataframe(
        selected_files_df,
        use_container_width=True,
        hide_index=True,
    )

    predict_button = st.button(
        "Predict All Files",
        type="primary",
        use_container_width=True,
    )

    if predict_button:
        progress_bar = st.progress(0)
        progress_text = st.empty()

        successful_results: list[dict] = []
        failed_results: list[dict] = []

        total_files = len(uploaded_files)

        for file_index, uploaded_file in enumerate(
            uploaded_files,
            start=1,
        ):
            temporary_path: Path | None = None
            waveform_figure: Figure | None = None
            mel_figure: Figure | None = None

            progress_text.write(
                f"Processing {file_index}/{total_files}: "
                f"{uploaded_file.name}"
            )

            with st.expander(
                f"{file_index}. {uploaded_file.name}",
                expanded=(file_index == 1),
            ):
                uploaded_bytes = uploaded_file.getvalue()
                uploaded_suffix = Path(
                    uploaded_file.name
                ).suffix

                st.audio(
                    uploaded_bytes,
                    format="audio/wav",
                )

                try:
                    if not uploaded_bytes:
                        raise ValueError(
                            "The uploaded file is empty."
                        )

                    temporary_path = save_uploaded_file(
                        uploaded_bytes=uploaded_bytes,
                        file_suffix=uploaded_suffix,
                    )

                    signal, sample_rate, duration_seconds = (
                        load_audio_for_visualization(
                            temporary_path
                        )
                    )

                    prediction_result = predict_genre(
                        audio_path=temporary_path,
                        top_k=3,
                    )

                    predicted_genre = str(
                        prediction_result[
                            "predicted_genre"
                        ]
                    ).title()

                    confidence = prediction_result.get(
                        "confidence"
                    )

                    confidence_percentage = (
                        confidence * 100
                        if confidence is not None
                        else None
                    )

                    st.success(
                        f"Predicted genre: {predicted_genre}"
                    )

                    genre_column, confidence_column, duration_column = (
                        st.columns(3)
                    )

                    with genre_column:
                        st.metric(
                            label="Predicted Genre",
                            value=predicted_genre,
                        )

                    with confidence_column:
                        st.metric(
                            label="Confidence",
                            value=(
                                f"{confidence_percentage:.2f}%"
                                if confidence_percentage is not None
                                else "Unavailable"
                            ),
                        )

                    with duration_column:
                        st.metric(
                            label="Analyzed Duration",
                            value=f"{duration_seconds:.2f} seconds",
                        )

                    top_predictions = prediction_result.get(
                        "top_predictions",
                        [],
                    )

                    if top_predictions:
                        display_top_predictions(
                            top_predictions
                        )

                    if show_visualizations:
                        waveform_figure = (
                            create_waveform_figure(
                                signal=signal,
                                sample_rate=sample_rate,
                            )
                        )

                        mel_figure = (
                            create_mel_spectrogram_figure(
                                signal=signal,
                                sample_rate=sample_rate,
                            )
                        )

                        waveform_tab, mel_tab = st.tabs(
                            [
                                "Waveform",
                                "Mel-Spectrogram",
                            ]
                        )

                        with waveform_tab:
                            st.pyplot(
                                waveform_figure,
                                use_container_width=True,
                            )

                        with mel_tab:
                            st.pyplot(
                                mel_figure,
                                use_container_width=True,
                            )

                    top_prediction_1 = (
                        top_predictions[0]["genre"].title()
                        if len(top_predictions) > 0
                        else ""
                    )

                    top_prediction_2 = (
                        top_predictions[1]["genre"].title()
                        if len(top_predictions) > 1
                        else ""
                    )

                    top_prediction_3 = (
                        top_predictions[2]["genre"].title()
                        if len(top_predictions) > 2
                        else ""
                    )

                    successful_results.append(
                        {
                            "filename": uploaded_file.name,
                            "predicted_genre": predicted_genre,
                            "confidence_percent": (
                                round(
                                    confidence_percentage,
                                    2,
                                )
                                if confidence_percentage is not None
                                else None
                            ),
                            "duration_seconds": round(
                                duration_seconds,
                                2,
                            ),
                            "top_prediction_1": top_prediction_1,
                            "top_prediction_2": top_prediction_2,
                            "top_prediction_3": top_prediction_3,
                            "model": prediction_result[
                                "model_name"
                            ],
                            "status": "success",
                        }
                    )

                except Exception as error:
                    st.error(
                        f"Prediction failed: {error}"
                    )

                    failed_results.append(
                        {
                            "filename": uploaded_file.name,
                            "error": str(error),
                            "status": "failed",
                        }
                    )

                finally:
                    if waveform_figure is not None:
                        plt.close(waveform_figure)

                    if mel_figure is not None:
                        plt.close(mel_figure)

                    if (
                        temporary_path is not None
                        and temporary_path.exists()
                    ):
                        try:
                            temporary_path.unlink()
                        except OSError:
                            pass

            progress_bar.progress(
                file_index / total_files
            )

        progress_text.empty()

        st.success(
            f"Batch processing completed: "
            f"{len(successful_results)} successful, "
            f"{len(failed_results)} failed."
        )

        if successful_results:
            st.header("Batch Results")

            results_df = pd.DataFrame(
                successful_results
            )

            display_columns = [
                "filename",
                "predicted_genre",
                "confidence_percent",
                "duration_seconds",
                "top_prediction_1",
                "top_prediction_2",
                "top_prediction_3",
            ]

            st.dataframe(
                results_df[display_columns],
                use_container_width=True,
                hide_index=True,
            )

            csv_data = results_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name=(
                    "music_genre_classification_results.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

            genre_summary = (
                results_df["predicted_genre"]
                .value_counts()
                .rename_axis("Genre")
                .reset_index(name="Number of Files")
            )

            st.subheader("Predicted Genre Distribution")

            st.bar_chart(
                genre_summary,
                x="Genre",
                y="Number of Files",
            )

        if failed_results:
            st.subheader("Failed Files")

            failed_df = pd.DataFrame(
                failed_results
            )

            st.dataframe(
                failed_df,
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("Model Information"):
            st.write(
                "**Supported genres:**",
                ", ".join(SUPPORTED_GENRES),
            )

            st.write(
                "**Audio processing:** "
                "22,050 Hz, mono audio and a maximum "
                "of 30 seconds per file."
            )

            st.write(
                "**Features:** MFCC, chroma, RMS energy, "
                "zero-crossing rate, spectral centroid, "
                "spectral bandwidth, spectral roll-off, "
                "spectral contrast and tempo."
            )


st.divider()

st.caption(
    "Educational AI/ML project. Music genres can overlap, "
    "so predictions may not always be correct."
)

