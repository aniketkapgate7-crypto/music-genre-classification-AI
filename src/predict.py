from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def find_project_root(start_path: Path) -> Path:
    """Find the project directory containing src/config.py."""
    current_path = start_path.resolve()

    for candidate in [current_path, *current_path.parents]:
        if (candidate / "src" / "config.py").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the project root containing src/config.py."
    )


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)

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


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    """Load and validate the saved model metadata."""
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model metadata was not found:\n{metadata_path}"
        )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    required_keys = {
        "model_name",
        "feature_columns",
        "genre_labels",
    }

    missing_keys = required_keys - set(metadata)

    if missing_keys:
        raise ValueError(
            "Model metadata is missing these keys: "
            f"{sorted(missing_keys)}"
        )

    return metadata


def create_model_input(
    extracted_features: dict[str, float],
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Arrange extracted features in exactly the same order
    used while training the model.
    """
    missing_features = [
        feature_name
        for feature_name in feature_columns
        if feature_name not in extracted_features
    ]

    if missing_features:
        raise ValueError(
            "The prediction code did not generate all features "
            "required by the model.\n"
            f"Missing features: {missing_features[:20]}"
        )

    ordered_features = {
        feature_name: extracted_features[feature_name]
        for feature_name in feature_columns
    }

    model_input = pd.DataFrame(
        [ordered_features],
        columns=feature_columns,
    )

    feature_array = model_input.to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(feature_array).all():
        raise ValueError(
            "One or more extracted features contain "
            "NaN or infinite values."
        )

    return model_input


def predict_genre(
    audio_path: str | Path,
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Predict the genre of one audio file.

    Returns the predicted genre, confidence score,
    and the highest-ranked predictions.
    """
    audio_path = Path(audio_path).expanduser().resolve()

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file was not found:\n{audio_path}"
        )

    if not audio_path.is_file():
        raise ValueError(
            f"The supplied path is not a file:\n{audio_path}"
        )

    if not FINAL_MODEL_FILE.exists():
        raise FileNotFoundError(
            "The final trained model was not found:\n"
            f"{FINAL_MODEL_FILE}\n\n"
            "Run 06_final_evaluation.ipynb first."
        )

    metadata = load_metadata(
        FINAL_METADATA_FILE
    )

    feature_columns = list(
        metadata["feature_columns"]
    )

    model = joblib.load(
        FINAL_MODEL_FILE
    )

    extracted_features = extract_audio_features(
        audio_path
    )

    model_input = create_model_input(
        extracted_features=extracted_features,
        feature_columns=feature_columns,
    )

    predicted_genre = str(
        model.predict(model_input)[0]
    )

    result: dict[str, Any] = {
        "audio_file": str(audio_path),
        "model_name": metadata["model_name"],
        "predicted_genre": predicted_genre,
        "confidence": None,
        "top_predictions": [],
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(
            model_input
        )[0]

        model_classes = [
            str(class_name)
            for class_name in model.classes_
        ]

        ranked_predictions = sorted(
            zip(model_classes, probabilities),
            key=lambda item: item[1],
            reverse=True,
        )

        safe_top_k = max(
            1,
            min(top_k, len(ranked_predictions)),
        )

        result["confidence"] = float(
            ranked_predictions[0][1]
        )

        result["top_predictions"] = [
            {
                "genre": genre,
                "probability": float(probability),
            }
            for genre, probability
            in ranked_predictions[:safe_top_k]
        ]

    return result


def print_prediction(
    prediction_result: dict[str, Any],
) -> None:
    """Display a prediction result in the terminal."""
    print("\n" + "=" * 55)
    print("MUSIC GENRE CLASSIFICATION RESULT")
    print("=" * 55)

    print(
        f"Audio file      : "
        f"{Path(prediction_result['audio_file']).name}"
    )

    print(
        f"Model           : "
        f"{prediction_result['model_name']}"
    )

    print(
        f"Predicted genre : "
        f"{prediction_result['predicted_genre'].title()}"
    )

    confidence = prediction_result["confidence"]

    if confidence is not None:
        print(
            f"Confidence      : "
            f"{confidence * 100:.2f}%"
        )

    top_predictions = prediction_result[
        "top_predictions"
    ]

    if top_predictions:
        print("\nTop predictions:")

        for rank, item in enumerate(
            top_predictions,
            start=1,
        ):
            print(
                f"{rank}. "
                f"{item['genre'].title():<12} "
                f"{item['probability'] * 100:>7.2f}%"
            )

    print("=" * 55)


def parse_arguments() -> argparse.Namespace:
    """Create and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Predict the music genre of an audio file "
            "using the trained machine-learning model."
        )
    )

    parser.add_argument(
        "--audio",
        required=True,
        type=Path,
        help="Path to the audio file.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help=(
            "Number of highest-ranked predictions to display. "
            "Default: 3."
        ),
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete result in JSON format.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the command-line prediction program."""
    arguments = parse_arguments()

    if arguments.top_k < 1:
        print(
            "Error: --top-k must be at least 1.",
            file=sys.stderr,
        )
        return 1

    try:
        prediction_result = predict_genre(
            audio_path=arguments.audio,
            top_k=arguments.top_k,
        )

        if arguments.json:
            print(
                json.dumps(
                    prediction_result,
                    indent=4,
                )
            )
        else:
            print_prediction(
                prediction_result
            )

        return 0

    except Exception as error:
        print(
            f"\nPrediction failed: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
