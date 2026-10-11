import os
import secrets
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ENV_PATH = ROOT / ".env"


def load_config() -> dict:
    with open(ROOT / "config.toml", "rb") as f:
        return tomllib.load(f)


def load_env() -> dict:
    """Read KEY=VALUE lines from .env into os.environ (real env vars win)."""
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.lstrip().startswith("#"):
                os.environ.setdefault(key.strip(), value.strip())
    return os.environ


def hash_salt() -> bytes:
    """Salt for user-ID hashing. Created once and kept in .env, so hashes stay stable across runs."""
    salt = load_env().get("HASH_SALT")
    if not salt:
        salt = secrets.token_hex(16)
        with open(ENV_PATH, "a") as f:
            f.write(f"\nHASH_SALT={salt}\n")
        os.environ["HASH_SALT"] = salt
    return salt.encode()
