from pathlib import Path

# Main project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPECTROGRAM_DIR = DATA_DIR / "spectrograms"

# Output directories
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

# Audio settings
SAMPLE_RATE = 22_050
DURATION = 30
N_MFCC = 20
N_FFT = 2_048
HOP_LENGTH = 512

# Supported GTZAN genres
GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


def create_project_directories() -> None:
    """Create required project directories if they do not exist."""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SPECTROGRAM_DIR,
        MODEL_DIR,
        FIGURE_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_project_directories()
    print("Project directories created successfully.")
    