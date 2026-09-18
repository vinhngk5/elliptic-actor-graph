from __future__ import annotations

import numpy as np
import pandas as pd


IDENTIFIER_COLUMNS = {"address", "Time step", "time_step", "class"}


def build_actor_table(
	features: pd.DataFrame,
	classes: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
	"""Create one address-level feature row from the temporal actor table."""
	clean_features = features.drop_duplicates().copy()
	numeric_columns = [
		column
		for column in clean_features.select_dtypes(include=np.number).columns
		if column not in IDENTIFIER_COLUMNS
	]
	if not numeric_columns:
		raise ValueError("No numeric actor features are available")

	actor_features = clean_features.groupby("address", sort=True)[numeric_columns].mean()
	active_time_steps = clean_features.groupby("address")["Time step"].nunique()
	actor_features["observed_time_steps"] = active_time_steps

	actor_table = actor_features.reset_index().rename(columns={"address": "actor_id"})
	actor_table = actor_table.merge(classes, left_on="actor_id", right_on="address")
	actor_table = actor_table.drop(columns="address")
	feature_columns = [column for column in actor_table.columns if column not in {"actor_id", "class"}]
	return actor_table, feature_columns
