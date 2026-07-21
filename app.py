from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import predict_genre


st.set_page_config(
    page_title="Music Genre Classification AI",
    page_icon="🎵",
    layout="wide",
)

st.title("🎵 Music Genre Classification AI")

st.write(
    "Upload one or multiple WAV files. "
    "The model will predict the genre of every file."
)

uploaded_files = st.file_uploader(
    "Upload WAV files",
    type=["wav"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more WAV files to begin.")
    st.stop()


st.write(f"Selected files: **{len(uploaded_files)}**")

for uploaded_file in uploaded_files:
    st.write(f"- {uploaded_file.name}")


if st.button(
    "Predict All Files",
    type="primary",
    use_container_width=True,
):
    results: list[dict] = []

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    total_files = len(uploaded_files)

    for index, uploaded_file in enumerate(
        uploaded_files,
        start=1,
    ):
        temporary_path: Path | None = None

        status_text.write(
            f"Processing {index}/{total_files}: "
            f"{uploaded_file.name}"
        )

        try:
            audio_bytes = uploaded_file.getvalue()

            if not audio_bytes:
                raise ValueError("Uploaded file is empty.")

            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".wav",
                delete=False,
            ) as temporary_file:
                temporary_file.write(audio_bytes)
                temporary_path = Path(temporary_file.name)

            prediction = predict_genre(
                audio_path=temporary_path,
                top_k=3,
            )

            confidence = prediction.get("confidence")

            results.append(
                {
                    "filename": uploaded_file.name,
                    "status": "Success",
                    "predicted_genre": str(
                        prediction["predicted_genre"]
                    ).title(),
                    "confidence_percent": (
                        round(float(confidence) * 100, 2)
                        if confidence is not None
                        else None
                    ),
                    "top_predictions": prediction.get(
                        "top_predictions",
                        [],
                    ),
                    "error": "",
                }
            )

        except Exception as error:
            results.append(
                {
                    "filename": uploaded_file.name,
                    "status": "Failed",
                    "predicted_genre": "",
                    "confidence_percent": None,
                    "top_predictions": [],
                    "error": str(error),
                }
            )

        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                try:
                    temporary_path.unlink()
                except OSError:
                    pass

        progress_bar.progress(index / total_files)

    status_text.empty()

    successful_count = sum(
        result["status"] == "Success"
        for result in results
    )

    failed_count = len(results) - successful_count

    st.success(
        f"Completed: {successful_count} successful, "
        f"{failed_count} failed."
    )

    st.header("Results")

    table_rows = []

    for result in results:
        top_predictions = result["top_predictions"]

        table_rows.append(
            {
                "Filename": result["filename"],
                "Status": result["status"],
                "Predicted Genre": result[
                    "predicted_genre"
                ],
                "Confidence (%)": result[
                    "confidence_percent"
                ],
                "Second Prediction": (
                    str(top_predictions[1]["genre"]).title()
                    if len(top_predictions) > 1
                    else ""
                ),
                "Third Prediction": (
                    str(top_predictions[2]["genre"]).title()
                    if len(top_predictions) > 2
                    else ""
                ),
                "Error": result["error"],
            }
        )

    results_df = pd.DataFrame(table_rows)

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = results_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Results as CSV",
        data=csv_data,
        file_name="music_genre_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.header("Individual Results")

    for result in results:
        with st.expander(result["filename"]):
            if result["status"] == "Failed":
                st.error(result["error"])
                continue

            st.metric(
                "Predicted Genre",
                result["predicted_genre"],
            )

            confidence = result["confidence_percent"]

            st.metric(
                "Confidence",
                (
                    f"{confidence:.2f}%"
                    if confidence is not None
                    else "Unavailable"
                ),
            )

            top_predictions = result[
                "top_predictions"
            ]

            if top_predictions:
                st.write("**Top predictions:**")

                for rank, item in enumerate(
                    top_predictions,
                    start=1,
                ):
                    probability = float(
                        item["probability"]
                    )

                    st.write(
                        f"{rank}. "
                        f"{str(item['genre']).title()} — "
                        f"{probability * 100:.2f}%"
                    )

                    st.progress(
                        min(
                            max(probability, 0.0),
                            1.0,
                        )
                    )


st.divider()

st.caption(
    "Educational AI/ML project. "
    "Predictions may not always be correct."
)


