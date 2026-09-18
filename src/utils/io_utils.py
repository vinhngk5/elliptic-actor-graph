from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def save_results_csv(df: pd.DataFrame, path: str | Path) -> None:
	destination = Path(path)
	destination.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(destination, index=False)


def save_json(payload: dict, path: str | Path) -> None:
	destination = Path(path)
	destination.parent.mkdir(parents=True, exist_ok=True)
	destination.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
