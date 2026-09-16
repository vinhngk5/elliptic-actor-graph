"""Load validated experiment settings from a JSON configuration file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_experiment_config(path: Path | str) -> dict[str, Any]:
    """Load the experiment configuration and validate required sections."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"dataset", "random_state", "clustering", "leiden", "labels"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Experiment config is missing sections: {sorted(missing)}")
    return config