from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def save_cluster_figures(
	assignments: pd.DataFrame,
	output_dir: str | Path,
	embedding: np.ndarray | None = None,
	loss_history: list[float] | None = None,
	title: str = "Actor clustering",
) -> list[Path]:
	output = Path(output_dir)
	output.mkdir(parents=True, exist_ok=True)
	paths: list[Path] = []

	counts = assignments["cluster_id"].value_counts().sort_index()
	fig, ax = plt.subplots(figsize=(9, 4))
	ax.bar(counts.index.astype(str), counts.values, color="#176b87")
	ax.set(title=f"{title}: cluster sizes", xlabel="Cluster", ylabel="Actors")
	ax.tick_params(axis="x", labelrotation=90)
	fig.tight_layout()
	path = output / "cluster_sizes.png"
	fig.savefig(path, dpi=160)
	plt.close(fig)
	paths.append(path)

	if embedding is not None and len(embedding) >= 3:
		coordinates = PCA(n_components=2, random_state=42).fit_transform(embedding)
		fig, ax = plt.subplots(figsize=(7, 6))
		scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=assignments["cluster_id"], s=8, alpha=0.65, cmap="tab20")
		ax.set(title=f"{title}: embedding projection", xlabel="PC1", ylabel="PC2")
		fig.colorbar(scatter, ax=ax, label="Cluster")
		fig.tight_layout()
		path = output / "embedding_pca.png"
		fig.savefig(path, dpi=160)
		plt.close(fig)
		paths.append(path)

	if loss_history:
		fig, ax = plt.subplots(figsize=(7, 4))
		ax.plot(range(1, len(loss_history) + 1), loss_history, color="#c94c4c")
		ax.set(title=f"{title}: training loss", xlabel="Epoch", ylabel="Binary cross-entropy")
		fig.tight_layout()
		path = output / "training_loss.png"
		fig.savefig(path, dpi=160)
		plt.close(fig)
		paths.append(path)
	return paths


def save_graph_partition_figure(graph: nx.Graph, assignments: pd.DataFrame, path: str | Path, title: str) -> Path:
	"""Save a readable spring-layout view for the bounded experimental graph."""
	destination = Path(path)
	destination.parent.mkdir(parents=True, exist_ok=True)
	positions = nx.spring_layout(graph, seed=42, iterations=50)
	colors = assignments.set_index("actor_id")["cluster_id"].reindex(graph.nodes()).to_numpy()
	fig, ax = plt.subplots(figsize=(9, 7))
	nx.draw_networkx_edges(graph, positions, ax=ax, alpha=0.15, width=0.4)
	nx.draw_networkx_nodes(graph, positions, ax=ax, node_size=14, node_color=colors, cmap="tab20")
	ax.set_title(title)
	ax.axis("off")
	fig.tight_layout()
	fig.savefig(destination, dpi=160)
	plt.close(fig)
	return destination
