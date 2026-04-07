import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env without overriding existing environment variables.
DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=DOTENV_PATH, override=False)


def get_env(name: str, default: str = "") -> str:
    """Read environment variables from os.environ first, then fallback to .env values."""
    value = os.environ.get(name)
    if value:
        return value
    return os.getenv(name, default) or default
