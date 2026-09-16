"""Save and load fitted clustering artifacts for fast reuse."""

from __future__ import annotations

from pathlib import Path

import joblib


def save_artifact(artifact: object, path: Path) -> None:
    """Persist a fitted model or preprocessing bundle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)


def load_artifact(path: Path) -> object:
    """Load a previously fitted artifact."""
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)