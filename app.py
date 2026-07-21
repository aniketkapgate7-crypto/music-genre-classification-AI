from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from src.config import DURATION, HOP_LENGTH, N_FFT, SAMPLE_RATE
from src.predict import predict_genre


# =========================================================
# Application configuration
# =========================================================

N_MELS = 128
MAX_DURATION_SECONDS = float(DURATION)

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
    page_title="Music Genre Classification AI",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# Temporary-file helpers
# =========================================================

def save_temporary_audio(
    audio_bytes: bytes,
    original_filename: str,
) -> Path:
    """Save uploaded audio bytes to a temporary WAV file."""

    suffix = Path(original_filename).suffix.lower()

    if suffix != ".wav":
        suffix = ".wav"

    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=suffix,
        delete=False,
    ) as temporary_file:
        temporary_file.write(audio_bytes)
        return Path(temporary_file.name)


def delete_temporary_file(
    path: Path | None,
) -> None:
    """Delete a temporary file without crashing the app."""

    if path is None:
        return

    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


# =========================================================
# Upload and prediction helpers
# =========================================================

def get_upload_signature(
    uploaded_files: list[Any],
    top_k: int,
) -> tuple[Any, ...]:
    """
    Create a signature for the current upload selection.

    Old results are cleared automatically when uploaded files
    or the number of top predictions changes.
    """

    signature: list[Any] = [top_k]

    for index, uploaded_file in enumerate(uploaded_files):
        file_bytes = uploaded_file.getvalue()

        digest = hashlib.sha256(
            file_bytes
        ).hexdigest()

        signature.append(
            (
                index,
                uploaded_file.name,
                len(file_bytes),
                digest,
            )
        )

    return tuple(signature)


def get_confidence_percent(
    prediction_result: dict[str, Any],
) -> float | None:
    """Read confidence from the supported prediction formats."""

    confidence_percent = prediction_result.get(
        "confidence_percent"
    )

    if confidence_percent is not None:
        return float(confidence_percent)

    confidence = prediction_result.get(
        "confidence"
    )

    if confidence is None:
        return None

    return float(confidence) * 100


def process_uploaded_file(
    uploaded_file: Any,
    file_index: int,
    top_k: int,
) -> dict[str, Any]:
    """Predict the genre of one uploaded WAV file."""

    audio_bytes = uploaded_file.getvalue()

    if not audio_bytes:
        raise ValueError(
            "The uploaded file is empty."
        )

    temporary_path: Path | None = None

    try:
        temporary_path = save_temporary_audio(
            audio_bytes=audio_bytes,
            original_filename=uploaded_file.name,
        )

        prediction_result = predict_genre(
            audio_path=temporary_path,
            top_k=top_k,
        )

        confidence_percent = get_confidence_percent(
            prediction_result
        )

        return {
            "file_index": file_index,
            "filename": uploaded_file.name,
            "status": "success",
            "predicted_genre": str(
                prediction_result["predicted_genre"]
            ).title(),
            "confidence_percent": (
                round(confidence_percent, 2)
                if confidence_percent is not None
                else None
            ),
            "duration_seconds": prediction_result.get(
                "loaded_duration_seconds"
            ),
            "model_name": str(
                prediction_result.get(
                    "model_name",
                    "Unknown",
                )
            ),
            "top_predictions": prediction_result.get(
                "top_predictions",
                [],
            ),
            "error_type": "",
            "error_message": "",
        }

    finally:
        delete_temporary_file(
            temporary_path
        )


# =========================================================
# Audio visualization helpers
# =========================================================

def create_audio_visualizations(
    audio_bytes: bytes,
    original_filename: str,
) -> tuple[Figure, Figure, float, int]:
    """Create waveform and Mel-spectrogram figures."""

    temporary_path: Path | None = None
    waveform_figure: Figure | None = None
    mel_figure: Figure | None = None

    try:
        temporary_path = save_temporary_audio(
            audio_bytes=audio_bytes,
            original_filename=original_filename,
        )

        signal, sample_rate = librosa.load(
            temporary_path,
            sr=SAMPLE_RATE,
            mono=True,
            duration=MAX_DURATION_SECONDS,
        )

        if signal.size == 0:
            raise ValueError(
                "The selected file contains no usable audio samples."
            )

        if not np.isfinite(signal).all():
            raise ValueError(
                "The selected audio contains invalid values."
            )

        duration_seconds = float(
            librosa.get_duration(
                y=signal,
                sr=sample_rate,
            )
        )

        # Waveform
        waveform_figure, waveform_axis = plt.subplots(
            figsize=(11, 3.5)
        )

        librosa.display.waveshow(
            signal,
            sr=sample_rate,
            ax=waveform_axis,
        )

        waveform_axis.set_title(
            "Audio Waveform"
        )

        waveform_axis.set_xlabel(
            "Time in seconds"
        )

        waveform_axis.set_ylabel(
            "Amplitude"
        )

        waveform_axis.grid(
            alpha=0.2
        )

        waveform_figure.tight_layout()

        # Mel-spectrogram
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

        mel_figure, mel_axis = plt.subplots(
            figsize=(11, 4.5)
        )

        mel_image = librosa.display.specshow(
            mel_db,
            sr=sample_rate,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="mel",
            ax=mel_axis,
        )

        mel_figure.colorbar(
            mel_image,
            ax=mel_axis,
            format="%+2.0f dB",
        )

        mel_axis.set_title(
            "Mel-Spectrogram"
        )

        mel_axis.set_xlabel(
            "Time in seconds"
        )

        mel_axis.set_ylabel(
            "Mel frequency"
        )

        mel_figure.tight_layout()

        return (
            waveform_figure,
            mel_figure,
            duration_seconds,
            sample_rate,
        )

    except Exception:
        if waveform_figure is not None:
            plt.close(
                waveform_figure
            )

        if mel_figure is not None:
            plt.close(
                mel_figure
            )

        raise

    finally:
        delete_temporary_file(
            temporary_path
        )


def display_top_predictions(
    top_predictions: list[dict[str, Any]],
) -> None:
    """Display ranked genre predictions."""

    st.subheader(
        "Top Predictions"
    )

    for rank, item in enumerate(
        top_predictions,
        start=1,
    ):
        genre = str(
            item["genre"]
        ).title()

        probability = float(
            item["probability"]
        )

        st.markdown(
            f"**{rank}. {genre} — "
            f"{probability * 100:.2f}%**"
        )

        st.progress(
            float(
                min(
                    max(probability, 0.0),
                    1.0,
                )
            )
        )


# =========================================================
# CSV helper
# =========================================================

def flatten_result_for_csv(
    result: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    """Convert one prediction into a CSV-friendly row."""

    row: dict[str, Any] = {
        "filename": result["filename"],
        "status": result["status"],
        "predicted_genre": result.get(
            "predicted_genre",
            "",
        ),
        "confidence_percent": result.get(
            "confidence_percent"
        ),
        "duration_seconds": result.get(
            "duration_seconds"
        ),
        "model_name": result.get(
            "model_name",
            "",
        ),
        "error_type": result.get(
            "error_type",
            "",
        ),
        "error_message": result.get(
            "error_message",
            "",
        ),
    }

    top_predictions = result.get(
        "top_predictions",
        [],
    )

    for rank in range(
        1,
        top_k + 1,
    ):
        if rank <= len(top_predictions):
            prediction = top_predictions[
                rank - 1
            ]

            probability = float(
                prediction["probability"]
            )

            row[f"top_{rank}_genre"] = str(
                prediction["genre"]
            ).title()

            row[f"top_{rank}_percent"] = round(
                probability * 100,
                2,
            )

        else:
            row[f"top_{rank}_genre"] = ""
            row[f"top_{rank}_percent"] = None

    return row


# =========================================================
# Session state
# =========================================================

if "batch_results" not in st.session_state:
    st.session_state.batch_results = []

if "upload_signature" not in st.session_state:
    st.session_state.upload_signature = None


# =========================================================
# Page header
# =========================================================

st.title(
    "🎵 Music Genre Classification AI"
)

st.write(
    "Upload one or multiple WAV files. The trained model "
    "will analyze every file and predict its music genre."
)

st.info(
    f"Only the first {MAX_DURATION_SECONDS:.0f} seconds "
    "of each audio file are analyzed."
)


# =========================================================
# Upload settings
# =========================================================

settings_column, genres_column = st.columns(
    [1, 2]
)

with settings_column:
    top_k = st.slider(
        "Number of top predictions",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
    )

with genres_column:
    st.write(
        "**Supported genres**"
    )

    st.write(
        ", ".join(SUPPORTED_GENRES)
    )


uploaded_files = st.file_uploader(
    "Upload WAV audio files",
    type=["wav"],
    accept_multiple_files=True,
    help=(
        "Hold Ctrl while selecting files "
        "to upload several WAV tracks."
    ),
)


if not uploaded_files:
    st.info(
        "Upload one or more WAV files to begin."
    )

    st.divider()

    st.caption(
        "Educational AI/ML project. "
        "Predictions may not always be correct."
    )

    st.stop()


# =========================================================
# Reset old results when uploads change
# =========================================================

current_signature = get_upload_signature(
    uploaded_files=uploaded_files,
    top_k=top_k,
)

if (
    st.session_state.upload_signature
    != current_signature
):
    st.session_state.upload_signature = (
        current_signature
    )

    st.session_state.batch_results = []


# =========================================================
# Uploaded file summary
# =========================================================

total_size_bytes = sum(
    len(uploaded_file.getvalue())
    for uploaded_file in uploaded_files
)

file_count_column, size_column = st.columns(
    2
)

with file_count_column:
    st.metric(
        "Selected Files",
        len(uploaded_files),
    )

with size_column:
    st.metric(
        "Combined Size",
        f"{total_size_bytes / (1024 * 1024):.2f} MB",
    )


selected_files_df = pd.DataFrame(
    [
        {
            "File Number": index + 1,
            "Filename": uploaded_file.name,
            "Size (MB)": round(
                len(uploaded_file.getvalue())
                / (1024 * 1024),
                2,
            ),
        }
        for index, uploaded_file
        in enumerate(uploaded_files)
    ]
)


with st.expander(
    "View selected files"
):
    st.dataframe(
        selected_files_df,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# Prediction controls
# =========================================================

predict_column, clear_column = st.columns(
    [3, 1]
)

with predict_column:
    predict_button = st.button(
        "Predict All Files",
        type="primary",
        use_container_width=True,
    )

with clear_column:
    clear_button = st.button(
        "Clear Results",
        use_container_width=True,
    )


if clear_button:
    st.session_state.batch_results = []


# =========================================================
# Process uploaded files
# =========================================================

if predict_button:
    progress_bar = st.progress(
        0.0
    )

    progress_message = st.empty()

    batch_results: list[
        dict[str, Any]
    ] = []

    total_files = len(
        uploaded_files
    )

    for file_index, uploaded_file in enumerate(
        uploaded_files
    ):
        progress_message.write(
            f"Processing {file_index + 1}/{total_files}: "
            f"{uploaded_file.name}"
        )

        try:
            result = process_uploaded_file(
                uploaded_file=uploaded_file,
                file_index=file_index,
                top_k=top_k,
            )

        except Exception as error:
            result = {
                "file_index": file_index,
                "filename": uploaded_file.name,
                "status": "failed",
                "predicted_genre": "",
                "confidence_percent": None,
                "duration_seconds": None,
                "model_name": "",
                "top_predictions": [],
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

        batch_results.append(
            result
        )

        progress_bar.progress(
            float(
                (file_index + 1)
                / total_files
            )
        )

    st.session_state.batch_results = (
        batch_results
    )

    progress_message.empty()

    successful_count = sum(
        result["status"] == "success"
        for result in batch_results
    )

    failed_count = (
        len(batch_results)
        - successful_count
    )

    if failed_count == 0:
        st.success(
            f"All {successful_count} files were "
            "processed successfully."
        )

    else:
        st.warning(
            f"Processing completed: {successful_count} "
            f"successful and {failed_count} failed."
        )


# =========================================================
# Display stored results
# =========================================================

batch_results = (
    st.session_state.batch_results
)

if batch_results:
    st.header(
        "Batch Results"
    )

    flattened_results = [
        flatten_result_for_csv(
            result=result,
            top_k=top_k,
        )
        for result in batch_results
    ]

    results_df = pd.DataFrame(
        flattened_results
    )

    summary_columns = [
        "filename",
        "status",
        "predicted_genre",
        "confidence_percent",
        "duration_seconds",
    ]

    st.dataframe(
        results_df[summary_columns],
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = results_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Batch Results as CSV",
        data=csv_bytes,
        file_name=(
            "music_genre_classification_results.csv"
        ),
        mime="text/csv",
        use_container_width=True,
    )


    successful_results = [
        result
        for result in batch_results
        if result["status"] == "success"
    ]

    failed_results = [
        result
        for result in batch_results
        if result["status"] == "failed"
    ]


    # -----------------------------------------------------
    # Genre distribution
    # -----------------------------------------------------

    if successful_results:
        genre_distribution = (
            pd.Series(
                [
                    result["predicted_genre"]
                    for result in successful_results
                ],
                name="Number of Files",
            )
            .value_counts()
            .rename_axis("Genre")
            .reset_index()
        )

        st.subheader(
            "Predicted Genre Distribution"
        )

        st.bar_chart(
            genre_distribution,
            x="Genre",
            y="Number of Files",
        )


        # -------------------------------------------------
        # Detailed result selector
        # -------------------------------------------------

        st.header(
            "Detailed File Analysis"
        )

        selected_position = st.selectbox(
            "Choose one successful file to inspect",
            options=range(
                len(successful_results)
            ),
            format_func=lambda position: (
                f"{successful_results[position]['file_index'] + 1}. "
                f"{successful_results[position]['filename']}"
            ),
        )

        selected_result = successful_results[
            selected_position
        ]

        selected_uploaded_file = uploaded_files[
            selected_result["file_index"]
        ]

        selected_audio_bytes = (
            selected_uploaded_file.getvalue()
        )

        st.audio(
            selected_audio_bytes,
            format="audio/wav",
        )


        # -------------------------------------------------
        # Result metrics
        # -------------------------------------------------

        (
            genre_metric,
            confidence_metric,
            duration_metric,
        ) = st.columns(3)

        with genre_metric:
            st.metric(
                "Predicted Genre",
                selected_result[
                    "predicted_genre"
                ],
            )

        with confidence_metric:
            confidence_percent = selected_result[
                "confidence_percent"
            ]

            st.metric(
                "Confidence",
                (
                    f"{confidence_percent:.2f}%"
                    if confidence_percent is not None
                    else "Unavailable"
                ),
            )

        with duration_metric:
            duration_seconds = selected_result[
                "duration_seconds"
            ]

            st.metric(
                "Analyzed Duration",
                (
                    f"{float(duration_seconds):.2f} seconds"
                    if duration_seconds is not None
                    else "Unavailable"
                ),
            )


        top_predictions = selected_result[
            "top_predictions"
        ]

        if top_predictions:
            display_top_predictions(
                top_predictions
            )


        # -------------------------------------------------
        # Optional visualizations
        # -------------------------------------------------

        show_visualizations = st.checkbox(
            "Show waveform and Mel-spectrogram",
            value=False,
        )

        if show_visualizations:
            waveform_figure: Figure | None = None
            mel_figure: Figure | None = None

            try:
                with st.spinner(
                    "Generating audio visualizations..."
                ):
                    (
                        waveform_figure,
                        mel_figure,
                        visualization_duration,
                        visualization_sample_rate,
                    ) = create_audio_visualizations(
                        audio_bytes=selected_audio_bytes,
                        original_filename=(
                            selected_uploaded_file.name
                        ),
                    )

                st.caption(
                    f"Visualization duration: "
                    f"{visualization_duration:.2f} seconds "
                    f"at {visualization_sample_rate:,} Hz."
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

            except Exception as error:
                st.error(
                    f"Visualization failed: {error}"
                )

            finally:
                if waveform_figure is not None:
                    plt.close(
                        waveform_figure
                    )

                if mel_figure is not None:
                    plt.close(
                        mel_figure
                    )


        # -------------------------------------------------
        # Model information
        # -------------------------------------------------

        with st.expander(
            "Model Information"
        ):
            st.write(
                "**Selected model:**",
                selected_result[
                    "model_name"
                ],
            )

            st.write(
                "**Supported genres:**",
                ", ".join(
                    SUPPORTED_GENRES
                ),
            )

            st.write(
                "**Audio preprocessing:** "
                f"{SAMPLE_RATE:,} Hz, mono audio, "
                f"maximum {MAX_DURATION_SECONDS:.0f} seconds."
            )

            st.write(
                "**Features:** MFCC, chroma, RMS energy, "
                "zero-crossing rate, spectral centroid, "
                "spectral bandwidth, spectral roll-off, "
                "spectral contrast and tempo."
            )


    # -----------------------------------------------------
    # Failed files
    # -----------------------------------------------------

    if failed_results:
        st.subheader(
            "Failed Files"
        )

        failed_df = pd.DataFrame(
            [
                {
                    "filename": result[
                        "filename"
                    ],
                    "error_type": result[
                        "error_type"
                    ],
                    "error_message": result[
                        "error_message"
                    ],
                }
                for result in failed_results
            ]
        )

        st.dataframe(
            failed_df,
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# Footer
# =========================================================

st.divider()

st.caption(
    "Educational AI/ML project. Music genres can overlap, "
    "and recording quality can affect predictions."
)



