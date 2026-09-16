# Actor-Level Clustering on Elliptic++

## Abstract

This study investigates whether actor-level behavioural and graph representations
can reveal illicit groups in the Elliptic++ cryptocurrency dataset. We aggregate
actor-time records into one representation per address, augment behavioural
features with directed graph statistics, and compare KMeans with HDBSCAN. The
first resource-bounded benchmark uses a fixed random sample of 20,000 actors.
Although KMeans produces compact geometric clusters, agreement with known
licit/illicit labels is close to zero. The strongest observed illicit enrichment
is approximately 1.21x over the known-label baseline for the KMeans-8 solution.
These results show that unsupervised feature clustering alone is insufficient as
an illicit-group detector and motivate graph-community and richer temporal
representations.

## Research Questions

1. Do actor behavioural features form coherent unsupervised groups?
2. Do actor graph statistics improve those groups?
3. Do the resulting clusters enrich for known illicit actors?
4. Are the clusters a useful proxy for illicit activity, or only geometric
   structure in the feature space?

## Data and preprocessing

The repository contains 822,942 labelled actors, 1,268,260 actor-time feature
records, and 2,868,964 actor graph edge rows. Actor classes are treated as
`1=illicit`, `2=licit`, and `3=unknown`, following the Elliptic++ convention;
class 3 is excluded from label-based evaluation. Labels are never included in
the clustering feature matrix.

Temporal actor records are aggregated by address using mean and maximum values
for numeric features. Graph features include weighted in/out degree, unique in/
out degree, and self-loop count. Missing values are median-imputed, transformed
with a signed log transform, robust-scaled, and reduced with PCA.

## Experimental setup

The benchmark uses 20,000 actors sampled with random seed 42. It compares
KMeans with 4, 8, and 12 clusters and HDBSCAN with `min_cluster_size=50` and
`min_samples=10`. All experiments use two CPU threads and chunked edge-list
processing where applicable.

The complete actor graph is also rendered as a degree-ordered adjacency-density
image. This uses all edges without constructing a force-directed graph in RAM.

## Results: feature clustering

| Method | Clusters | Noise | Silhouette | ARI known | NMI known |
|---|---:|---:|---:|---:|---:|
| KMeans-4 | 4 | 0.000 | 0.7725 | -0.0346 | 0.0073 |
| KMeans-8 | 8 | 0.000 | 0.6940 | -0.0188 | 0.0054 |
| KMeans-12 | 12 | 0.000 | 0.6753 | -0.0182 | 0.0061 |
| HDBSCAN | 61 | 0.609 | 0.4056 | -0.0057 | 0.0492 |

KMeans-4 has the strongest geometric silhouette, but none of the methods has
meaningful agreement with known illicit/licit labels. KMeans-8 has the highest
large-cluster illicit enrichment observed in the first run: approximately 1.21x
with 256 illicit actors among 3,893 known-label actors. This is weak evidence,
not a reliable detector.

## Results: full-graph Leiden

The actor graph was converted to an integer-indexed undirected graph. Self-loops
were excluded, duplicate relations became edge weights, and Leiden was run at
resolution 1.0. The run used 2,822,983 non-self-loop edge rows and 2,772,474
unique undirected edges. It produced 887 communities with modularity 0.9212.

The largest high-enrichment result with a non-trivial community size contained
1,142 illicit actors among 1,730 known-label actors, giving an illicit rate of
0.660 and enrichment of 12.28x over the global known-label illicit rate. Several
small communities reached 18.6x enrichment, but their sizes were only 2--15
actors and must not be used as headline evidence.

This is the strongest signal so far: actor graph structure contains information
that is not recovered by the current aggregated behavioural features. It is an
association, not a causal or predictive claim.

## Robustness checks

The maximum observed enrichment of 12.28x remains above the shuffled-label null
at every tested minimum community size. With 100 class-count-preserving
permutations, the null 95th-percentile maximum enrichment was 3.10x for size
30, 2.02x for size 100, and 1.52x for size 500. The empirical permutation
p-value was 0.0099 in each case. These tests control for the multiple-community
search by comparing the observed maximum with the maximum from every null run.

Leiden seed sensitivity was also checked at resolution 1.0. Seeds 42, 7, and
21 produced 887, 839, and 751 communities, with modularity 0.9212, 0.9237, and
0.9236 respectively. The exact partition is stochastic, but the high-level
graph modularity is stable.

## Limitations

The Leiden result still needs a formal resolution-sweep comparison of
enrichment distributions and confidence intervals for individual communities.
The study does not yet include Node2Vec, GraphSAGE, GAT, or GIN embeddings. The actor feature
file contains many actors observed at only one timestep, and repeated feature
rows do not by themselves establish independent historical snapshots. Therefore
no temporal stability claim is made. Further work must validate the dataset's
temporal semantics before reporting temporal generalization.

The random sample and unsupervised representations also limit statistical power.
Future experiments should use repeated seeds, confidence intervals, graph
community baselines, and a carefully specified temporal protocol.

## Reproducibility

Run:

```bash
python -m src.data_audit --data-dir data
python -m src.run_experiments --data-dir data --output-dir results --sample-size 20000
python -m src.run_community --data-dir data --output-dir results --resolution 1.0
```

Main outputs are stored in `results/` and figures in `figures/`. The notebook
`notebooks/01_actor_graph_eda.ipynb` contains the executable EDA and benchmark
workflow.
