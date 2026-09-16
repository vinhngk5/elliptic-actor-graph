# elliptic-actor-graph

Actor-level clustering and graph analysis for identifying illicit groups in
cryptocurrency transaction networks using the Elliptic++ dataset.

## Reproducible workflow

The current pipeline is designed for a machine with 8 GB RAM and two physical
CPU cores. It uses chunked CSV reads and avoids constructing the full graph in
NetworkX.

```bash
python -m src.data_audit --data-dir data
python -m src.run_experiments --data-dir data --output-dir results --sample-size 20000
python -m src.run_community --data-dir data --output-dir results --resolution 1.0
```

The experiment command writes:

- `results/clustering_metrics.csv`: geometry and known-label agreement metrics;
- `results/cluster_enrichment.csv`: illicit rate and enrichment by cluster;
- `results/clustering_benchmark.png`: comparison figure.

The Leiden command writes:

- `results/leiden_memberships.csv`: one community ID per actor;
- `results/leiden_enrichment.csv`: illicit enrichment by community;
- `results/leiden_summary.txt`: edge, community, modularity, and resolution summary.

The EDA notebook is [notebooks/01_actor_graph_eda.ipynb](notebooks/01_actor_graph_eda.ipynb).
It generates figures in `figures/`, including a density image using every actor
edge without requiring a full force-directed layout.

## Project layout

```text
config/experiment.json       Reproducible settings and resource limits
notebooks/01_actor_graph_eda.ipynb
notebooks/02_graph_communities.ipynb
notebooks/03_cc_mvad.ipynb
notebooks/04_paper_results.ipynb
src/data/                    Data audit and actor feature API
src/graph/                   Graph API and community API
src/models/                  Clustering and multi-view API
reports/                     Research report and paper notes
results/                     Tables, memberships, and experiment summaries
figures/                     Generated visualizations
models/                      Saved preprocessing and clustering artifacts
```

The original flat modules in `src/` remain as the implementation and backward-
compatible CLI surface. The new package directories provide organized imports
for notebooks and future experiments without invalidating existing results.

## Experimental interpretation

Class `3` is treated as unknown and is excluded from known-label ARI/NMI and
illicit-rate calculations. Labels are evaluation-only and are not included in
the clustering matrix.

The first feature-clustering benchmark is intentionally a baseline, not a claimed detector:
KMeans obtains a strong geometric silhouette but near-zero agreement with the
known illicit/licit labels. HDBSCAN identifies noise and smaller communities,
but its label agreement also remains weak. Weighted Leiden on the full actor
graph is a stronger graph-specific experiment: at resolution 1.0 it produces
887 communities with modularity 0.9212, including a large community with
1,142 illicit actors among 1,730 known-label actors (12.28x enrichment). This
must still be tested across resolutions and against size-controlled baselines;
small communities can create misleadingly large enrichment ratios.

Temporal stability is not currently reported. The actor feature file contains
many one-timestep actors and its repeated rows are not sufficient to establish
independent historical snapshots without further validation of the dataset
semantics.

The current paper-ready robustness outputs are `results/leiden_permutation_summary.csv`
and the seed summaries under `results/leiden_seed7/` and `results/leiden_seed21/`.

## Reusing fitted models

Run [notebooks/05_model_visualization.ipynb](notebooks/05_model_visualization.ipynb)
once to create the model cache:

- `models/preprocessing_bundle.joblib`;
- `models/kmeans_4.joblib`;
- `models/kmeans_8.joblib`;
- `models/kmeans_12.joblib`;
- `models/hdbscan.joblib`.

The notebook also writes `results/model_cluster_assignments.csv` and figures
`figures/model_cluster_pca_grid.png` and
`figures/leiden_cluster_visualization.png`. A saved model can be loaded without
refitting:

```python
from pathlib import Path
from src.artifacts import load_artifact

bundle = load_artifact(Path("models/kmeans_8.joblib"))
labels = bundle["model"].predict(embedding)
```
