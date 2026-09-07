"""Configuration for NATOE Radiology Reporting Harness."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
BEST_DIR = ARTIFACTS_DIR / "best"
CANDIDATES_DIR = ARTIFACTS_DIR / "candidates"
CACHE_DIR = PROJECT_ROOT / "cache"

# Ensure directories exist
for d in [DATA_DIR, ARTIFACTS_DIR, BEST_DIR, CANDIDATES_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env.local")
load_dotenv(PROJECT_ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "1024"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))

# Target benchmarks
TARGET_RES = 0.20
LEADERBOARD_BEST_RES = 0.23288
