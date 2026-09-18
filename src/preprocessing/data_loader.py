from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocessing.validation import validate_actor_tables


def load_actor_tables(data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
	"""Load the actor tables without mutating the source files."""
	project_root = Path(__file__).resolve().parents[2]
	actor_dir = (
		Path(data_dir)
		if data_dir is not None
		else project_root / "data" / "Actors_Dataset"
	)
	filenames = {
		"features": "wallets_features.csv",
		"classes": "wallets_classes.csv",
		"addr_addr": "AddrAddr_edgelist.csv",
	}
	tables = {
		name: pd.read_csv(actor_dir / filename)
		for name, filename in filenames.items()
	}
	validate_actor_tables(
		tables["features"], tables["classes"], tables["addr_addr"]
	)
	return tables
