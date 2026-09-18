from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.preprocessing.actor_builder import build_actor_table
from src.preprocessing.data_loader import load_actor_tables
from src.preprocessing.graph_builder import build_actor_graph


def prepare_common_actor_data(
	data_dir: str | Path | None = None,
	max_nodes: int | None = None,
) -> dict[str, object]:
	"""Prepare the shared address-level representation used by clustering notebooks."""
	tables = load_actor_tables(data_dir)
	actor_table, feature_columns = build_actor_table(
		tables["features"], tables["classes"]
	)
	all_nodes = set(actor_table["actor_id"])
	full_graph = build_actor_graph(tables["addr_addr"], nodes=all_nodes)

	if max_nodes is not None:
		if max_nodes < 1:
			raise ValueError("max_nodes must be positive or None")
		selected_nodes = set(
			node
			for node, _degree in sorted(
				full_graph.degree,
				key=lambda item: (-item[1], str(item[0])),
			)[:max_nodes]
		)
	else:
		selected_nodes = all_nodes

	selected_table = actor_table[actor_table["actor_id"].isin(selected_nodes)].copy()
	selected_table = selected_table.sort_values("actor_id").reset_index(drop=True)
	node_order = selected_table["actor_id"].tolist()
	graph = build_actor_graph(tables["addr_addr"], nodes=set(node_order))

	values = selected_table[feature_columns].replace([np.inf, -np.inf], np.nan)
	values = SimpleImputer(strategy="median").fit_transform(values)
	feature_matrix = StandardScaler().fit_transform(values)

	return {
		"graph": graph,
		"actor_df": selected_table,
		"feature_matrix": feature_matrix,
		"feature_columns": feature_columns,
		"node_order": node_order,
		"validation": tables,
	}
