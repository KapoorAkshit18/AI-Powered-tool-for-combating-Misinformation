"""
Configuration management for the AI-Powered Misinformation Detection Tool.

Loads settings from a .env file and provides centralized access
to API keys, database paths, and application settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Project Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "misinformation_kb.db"

# --- API Keys ---
GOOGLE_FACTCHECK_API_KEY: str | None = os.getenv("GOOGLE_FACTCHECK_API_KEY")
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

# --- App Settings ---
APP_TITLE = "AI-Powered Misinformation Detection Tool"
APP_ICON = "🔍"
MAX_HISTORY_RESULTS = 50
DEFAULT_CREDIBILITY_THRESHOLD = 50  # Scores below this are flagged

# --- Scoring Weights ---
# Weights for each component in the overall credibility score (must sum to 1.0)
SCORING_WEIGHTS = {
    "nlp_analysis": 0.25,
    "fact_check": 0.25,
    "source_credibility": 0.25,
    "linguistic_analysis": 0.25,
}

# --- Ensure data directory exists ---
DATA_DIR.mkdir(parents=True, exist_ok=True)


class Config:
    """Class-based access to configuration values."""
    BASE_DIR = BASE_DIR
    DATA_DIR = DATA_DIR
    DB_PATH = DB_PATH
    GOOGLE_FACTCHECK_API_KEY = GOOGLE_FACTCHECK_API_KEY
    GEMINI_API_KEY = GEMINI_API_KEY
    APP_TITLE = APP_TITLE
    APP_ICON = APP_ICON
    MAX_HISTORY_RESULTS = MAX_HISTORY_RESULTS
    DEFAULT_CREDIBILITY_THRESHOLD = DEFAULT_CREDIBILITY_THRESHOLD
    SCORING_WEIGHTS = SCORING_WEIGHTS


def update_env_key(key_name: str, key_value: str) -> None:
    """Persist an API key to the .env file and update the running environment."""
    env_file = BASE_DIR / ".env"
    key_value = key_value.strip()

    # Update in-memory environment
    os.environ[key_name] = key_value
    if key_name == "GEMINI_API_KEY":
        Config.GEMINI_API_KEY = key_value
    elif key_name == "GOOGLE_FACTCHECK_API_KEY":
        Config.GOOGLE_FACTCHECK_API_KEY = key_value

    lines = []
    if env_file.exists():
        lines = env_file.read_text(encoding="utf-8").splitlines()

    updated = False
    new_lines = []
    for line in lines:
        if line.startswith(f"{key_name}=") or line.startswith(f"#{key_name}="):
            new_lines.append(f"{key_name}={key_value}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        new_lines.append(f"{key_name}={key_value}")

    env_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


Config.update_env_key = update_env_key
__all__ = ["Config", "update_env_key"]
