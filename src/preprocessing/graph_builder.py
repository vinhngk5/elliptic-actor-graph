from __future__ import annotations

import networkx as nx
import pandas as pd


def build_actor_graph(
	edges: pd.DataFrame,
	nodes: set[str] | None = None,
) -> nx.Graph:
	"""Build a weighted undirected graph for community detection.

	The AddrAddr file has no transaction identifier or timestamp. Duplicate
	directed records are therefore treated as interaction multiplicity after
	projecting both directions onto an undirected community graph.
	"""
	required = {"input_address", "output_address"}
	missing = required.difference(edges.columns)
	if missing:
		raise ValueError(f"AddrAddr edges are missing columns: {sorted(missing)}")

	prepared = edges.dropna(subset=list(required)).copy()
	if nodes is not None:
		prepared = prepared[
			prepared["input_address"].isin(nodes)
			& prepared["output_address"].isin(nodes)
		]

	graph = nx.Graph()
	if nodes is not None:
		graph.add_nodes_from(nodes)

	weighted_edges = (
		prepared.groupby(["input_address", "output_address"], sort=False)
		.size()
		.rename("weight")
		.reset_index()
	)
	for row in weighted_edges.itertuples(index=False):
		if row.input_address == row.output_address:
			graph.add_edge(row.input_address, row.output_address, weight=int(row.weight))
			continue
		current = graph.get_edge_data(row.input_address, row.output_address, {}).get("weight", 0)
		graph.add_edge(
			row.input_address,
			row.output_address,
			weight=int(current) + int(row.weight),
		)
	return graph
