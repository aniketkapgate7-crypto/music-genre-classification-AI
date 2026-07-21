from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

def find_project_root(start_path: Path) -> Path:
    """Find the project directory containing src/config.py."""

    current_path = start_path.resolve()

    for candidate in [current_path, *current_path.parents]:
        config_path = candidate / "src" / "config.py"

        if config_path.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find the project root containing src/config.py."
    )


PROJECT_ROOT = find_project_root(
    Path(__file__).resolve().parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.audio_features import extract_audio_features


MODEL_DIR = PROJECT_ROOT / "models"

FINAL_MODEL_FILE = (
    MODEL_DIR / "music_genre_ml_final.joblib"
)

FINAL_METADATA_FILE = (
    MODEL_DIR / "music_genre_ml_final_metadata.json"
)

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
}


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------

def load_metadata(
    metadata_path: Path,
) -> dict[str, Any]:
    """Load and validate model metadata."""

    if not metadata_path.exists():
        raise FileNotFoundError(
            "Model metadata file was not found:\n"
            f"{metadata_path}\n\n"
            "Run the final evaluation notebook first."
        )

    try:
        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Model metadata contains invalid JSON: {error}"
        ) from error

    required_keys = {
        "model_name",
        "feature_columns",
        "genre_labels",
    }

    missing_keys = (
        required_keys - set(metadata.keys())
    )

    if missing_keys:
        raise ValueError(
            "Model metadata is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    feature_columns = metadata["feature_columns"]
    genre_labels = metadata["genre_labels"]

    if not isinstance(feature_columns, list):
        raise TypeError(
            "metadata['feature_columns'] must be a list."
        )

    if not feature_columns:
        raise ValueError(
            "metadata['feature_columns'] is empty."
        )

    if not isinstance(genre_labels, list):
        raise TypeError(
            "metadata['genre_labels'] must be a list."
        )

    return metadata


@lru_cache(maxsize=1)
def load_model_bundle() -> tuple[Any, dict[str, Any]]:
    """
    Load the model and metadata once.

    The cache prevents the model from being loaded again for
    every uploaded audio file in the Streamlit application.
    """

    if not FINAL_MODEL_FILE.exists():
        raise FileNotFoundError(
            "Final trained model was not found:\n"
            f"{FINAL_MODEL_FILE}\n\n"
            "Run 06_final_evaluation.ipynb first."
        )

    metadata = load_metadata(
        FINAL_METADATA_FILE
    )

    try:
        model = joblib.load(
            FINAL_MODEL_FILE
        )

    except Exception as error:
        raise RuntimeError(
            f"Unable to load the trained model: {error}"
        ) from error

    if not hasattr(model, "predict"):
        raise TypeError(
            "The loaded model does not support prediction."
        )

    return model, metadata


def clear_model_cache() -> None:
    """Clear the cached model and metadata."""

    load_model_bundle.cache_clear()


# ---------------------------------------------------------
# Input validation
# ---------------------------------------------------------

def validate_audio_path(
    audio_path: str | Path,
) -> Path:
    """Validate and return an absolute audio-file path."""

    resolved_path = (
        Path(audio_path)
        .expanduser()
        .resolve()
    )

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Audio file was not found:\n{resolved_path}"
        )

    if not resolved_path.is_file():
        raise ValueError(
            f"The supplied path is not a file:\n{resolved_path}"
        )

    file_extension = resolved_path.suffix.lower()

    if file_extension not in SUPPORTED_AUDIO_EXTENSIONS:
        supported_formats = ", ".join(
            sorted(SUPPORTED_AUDIO_EXTENSIONS)
        )

        raise ValueError(
            f"Unsupported audio format: {file_extension or 'none'}.\n"
            f"Supported formats: {supported_formats}"
        )

    if resolved_path.stat().st_size == 0:
        raise ValueError(
            "The audio file is empty."
        )

    return resolved_path


# ---------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------

def create_model_input(
    extracted_features: dict[str, float],
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Arrange extracted features in the exact order used
    during model training.
    """

    missing_features = [
        feature_name
        for feature_name in feature_columns
        if feature_name not in extracted_features
    ]

    if missing_features:
        preview = missing_features[:20]

        raise ValueError(
            "The feature extractor did not generate every "
            "feature required by the model.\n"
            f"Missing features: {preview}"
        )

    ordered_features = {
        feature_name: extracted_features[feature_name]
        for feature_name in feature_columns
    }

    model_input = pd.DataFrame(
        [ordered_features],
        columns=feature_columns,
    )

    numerical_values = model_input.to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(numerical_values).all():
        raise ValueError(
            "The extracted features contain NaN or "
            "infinite values."
        )

    return model_input


def create_ranked_predictions(
    model: Any,
    model_input: pd.DataFrame,
    top_k: int,
) -> tuple[float | None, list[dict[str, Any]]]:
    """Create confidence and ranked genre probabilities."""

    if not hasattr(model, "predict_proba"):
        return None, []

    probabilities = np.asarray(
        model.predict_proba(model_input)[0],
        dtype=np.float64,
    )

    model_classes = [
        str(class_name)
        for class_name in model.classes_
    ]

    if len(probabilities) != len(model_classes):
        raise ValueError(
            "The number of probabilities does not match "
            "the number of model classes."
        )

    ranked_items = sorted(
        zip(model_classes, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )

    safe_top_k = min(
        max(int(top_k), 1),
        len(ranked_items),
    )

    top_predictions = [
        {
            "genre": genre,
            "probability": float(probability),
            "percentage": float(
                probability * 100
            ),
        }
        for genre, probability
        in ranked_items[:safe_top_k]
    ]

    highest_confidence = (
        float(ranked_items[0][1])
        if ranked_items
        else None
    )

    return highest_confidence, top_predictions


# ---------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------

def predict_genre(
    audio_path: str | Path,
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Predict the genre of one audio file.

    Args:
        audio_path:
            Path to a WAV audio file.

        top_k:
            Number of ranked predictions to return.

    Returns:
        Dictionary containing the predicted genre,
        confidence, model name and ranked probabilities.
    """

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    validated_audio_path = validate_audio_path(
        audio_path
    )

    model, metadata = load_model_bundle()

    feature_columns = [
        str(column)
        for column in metadata["feature_columns"]
    ]

    extracted_features = extract_audio_features(
        validated_audio_path
    )

    model_input = create_model_input(
        extracted_features=extracted_features,
        feature_columns=feature_columns,
    )

    prediction = model.predict(
        model_input
    )

    if len(prediction) == 0:
        raise RuntimeError(
            "The model returned no prediction."
        )

    predicted_genre = str(
        prediction[0]
    )

    confidence, top_predictions = (
        create_ranked_predictions(
            model=model,
            model_input=model_input,
            top_k=top_k,
        )
    )

    # Use the predicted class probability as confidence
    # when it is available.
    if top_predictions:
        probability_lookup = {
            item["genre"]: item["probability"]
            for item in top_predictions
        }

        if predicted_genre in probability_lookup:
            confidence = float(
                probability_lookup[predicted_genre]
            )

    loaded_duration = extracted_features.get(
        "loaded_duration_seconds"
    )

    result: dict[str, Any] = {
        "status": "success",
        "audio_file": str(validated_audio_path),
        "filename": validated_audio_path.name,
        "model_name": str(
            metadata["model_name"]
        ),
        "predicted_genre": predicted_genre,
        "confidence": confidence,
        "confidence_percent": (
            float(confidence * 100)
            if confidence is not None
            else None
        ),
        "top_predictions": top_predictions,
        "loaded_duration_seconds": (
            float(loaded_duration)
            if loaded_duration is not None
            else None
        ),
        "feature_count": len(feature_columns),
    }

    return result


def predict_many(
    audio_paths: Iterable[str | Path],
    top_k: int = 3,
    continue_on_error: bool = True,
) -> list[dict[str, Any]]:
    """
    Predict genres for multiple audio files.

    When continue_on_error is True, one failed file does not
    stop the remaining predictions.
    """

    paths = list(audio_paths)

    if not paths:
        raise ValueError(
            "No audio files were provided."
        )

    results: list[dict[str, Any]] = []

    for audio_path in paths:
        try:
            result = predict_genre(
                audio_path=audio_path,
                top_k=top_k,
            )

            results.append(result)

        except Exception as error:
            if not continue_on_error:
                raise

            failed_path = Path(audio_path)

            results.append(
                {
                    "status": "failed",
                    "audio_file": str(audio_path),
                    "filename": failed_path.name,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )

    return results


# ---------------------------------------------------------
# Terminal output
# ---------------------------------------------------------

def print_prediction(
    result: dict[str, Any],
) -> None:
    """Print one prediction in a readable terminal format."""

    print("\n" + "=" * 58)
    print("MUSIC GENRE CLASSIFICATION RESULT")
    print("=" * 58)

    print(
        f"Audio file      : {result['filename']}"
    )

    print(
        f"Model           : {result['model_name']}"
    )

    print(
        "Predicted genre : "
        f"{result['predicted_genre'].title()}"
    )

    confidence_percent = result.get(
        "confidence_percent"
    )

    if confidence_percent is not None:
        print(
            "Confidence      : "
            f"{confidence_percent:.2f}%"
        )
    else:
        print(
            "Confidence      : Unavailable"
        )

    duration = result.get(
        "loaded_duration_seconds"
    )

    if duration is not None:
        print(
            "Audio duration  : "
            f"{duration:.2f} seconds"
        )

    top_predictions = result.get(
        "top_predictions",
        [],
    )

    if top_predictions:
        print("\nTop predictions:")

        for rank, item in enumerate(
            top_predictions,
            start=1,
        ):
            print(
                f"{rank}. "
                f"{item['genre'].title():<14} "
                f"{item['percentage']:>7.2f}%"
            )

    print("=" * 58)


def print_batch_summary(
    results: list[dict[str, Any]],
) -> None:
    """Print a summary of multiple predictions."""

    successful_results = [
        result
        for result in results
        if result.get("status") == "success"
    ]

    failed_results = [
        result
        for result in results
        if result.get("status") == "failed"
    ]

    for result in successful_results:
        print_prediction(result)

    if failed_results:
        print("\nFAILED FILES")
        print("=" * 58)

        for result in failed_results:
            print(
                f"{result['filename']}: "
                f"{result['error_message']}"
            )

    print("\nBATCH SUMMARY")
    print("=" * 58)
    print(f"Total files : {len(results)}")
    print(
        f"Successful  : {len(successful_results)}"
    )
    print(
        f"Failed      : {len(failed_results)}"
    )
    print("=" * 58)


# ---------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    """Create and parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Predict the music genre of one or multiple "
            "WAV audio files."
        )
    )

    parser.add_argument(
        "--audio",
        required=True,
        nargs="+",
        type=Path,
        help=(
            "Path to one or multiple WAV files. "
            "Separate multiple paths with spaces."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help=(
            "Number of ranked genre predictions to return. "
            "Default: 3."
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results in JSON format.",
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help=(
            "Stop processing when the first file fails."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Run command-line genre prediction."""

    arguments = parse_arguments()

    if arguments.top_k < 1:
        print(
            "Error: --top-k must be at least 1.",
            file=sys.stderr,
        )

        return 1

    try:
        results = predict_many(
            audio_paths=arguments.audio,
            top_k=arguments.top_k,
            continue_on_error=(
                not arguments.stop_on_error
            ),
        )

        if arguments.json:
            output: dict[str, Any] | list[dict[str, Any]]

            if len(results) == 1:
                output = results[0]
            else:
                output = results

            print(
                json.dumps(
                    output,
                    indent=4,
                )
            )

        elif len(results) == 1:
            result = results[0]

            if result.get("status") == "failed":
                print(
                    "Prediction failed: "
                    f"{result['error_message']}",
                    file=sys.stderr,
                )

                return 1

            print_prediction(result)

        else:
            print_batch_summary(results)

        failed_count = sum(
            result.get("status") == "failed"
            for result in results
        )

        return 1 if failed_count else 0

    except Exception as error:
        print(
            f"Prediction failed: {error}",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
