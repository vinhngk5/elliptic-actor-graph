# Research and Codebase Audit

## Audit scope and evidence

This audit treats the repository as an untrusted research implementation. The
claims below are based on the Python modules, notebook code cells, configuration,
CSV schemas, generated artifacts, and the checked-in research report. No claim
of novelty or state of the art is made. The local files are compatible with the
Elliptic++ actor/transaction naming convention, but the repository does not
contain a dataset version, checksum, download manifest, or official provenance
record. The dataset release must therefore be re-verified before publication.

## A. Current project structure and pipeline

The project studies whether actor-level behavioural features and an actor graph
can reveal groups enriched for known illicit Bitcoin actors. The unit of
analysis is one actor/address after aggregating actor-time rows. The repository
also contains transaction-level files, but the current executable pipeline does
not use them.

The current path is:

1. `src.actor_features` loads actor-time features and aggregates every numeric
   feature by address using mean and maximum.
2. `src.graph_features` scans `AddrAddr_edgelist.csv` and produces weighted
   directed degree, unique in/out neighbour counts, and self-loop counts.
3. `src.run_experiments` joins the two views, fills missing values with zero,
   samples 20,000 actors, imputes, signed-log transforms, robust-scales, and
   PCA-reduces the matrix.
4. KMeans with 4, 8, and 12 clusters and HDBSCAN are fitted without labels.
5. `src.evaluation` reports geometric cluster metrics, ARI/NMI on known labels,
   and illicit enrichment.
6. `src.community` removes self-loops, symmetrizes duplicate directed relations
   into weighted undirected edges, and fits Leiden on all actor addresses.
7. `src.robustness` compares the maximum community enrichment with a
   class-count-preserving permutation null.
8. `src.multiview` computes a behavioural distance from each actor to its
   Leiden-community centroid. The notebooks call this CC-MVAD, but no actor
   detection evaluation is implemented.

The package directories `src/data`, `src/graph`, `src/models`, and
`src/evaluation` are currently namespace placeholders; the flat modules remain
the actual implementation. Notebooks execute substantial analysis and write
canonical-looking result files directly, which creates multiple sources of
truth.

## B. Research problem and estimand

The defensible current research question is descriptive:

> Do unsupervised actor behaviour and graph communities contain structure that
> is associated with the known illicit/licit labels in this fixed dataset?

The current code does not establish an inductive detector, early-warning
system, causal mechanism, or temporal generalization. It performs transductive
representation and community analysis over the full observed actor graph.

Class `1` is treated as illicit, `2` as licit, and `3` as unknown. The label is
excluded from feature construction and used after fitting for evaluation, which
is appropriate for an unsupervised benchmark. Unknown labels are not missing at
random by assumption, so excluding them changes the estimand to performance on
the known-label subset rather than performance over all actors.

## C. Dataset validation

Observed local facts from the supplied CSV files:

| Asset | Observed fact |
|---|---:|
| `wallets_classes.csv` | 822,942 rows and unique addresses |
| Actor labels | 14,266 illicit; 251,088 licit; 557,588 unknown |
| `wallets_features_classes_combined.csv` | 1,268,260 actor-time rows, timesteps 1--49 |
| Actor feature rows per address | median 1, mean about 1.54, maximum 1,471 |
| `AddrAddr_edgelist.csv` | 2,868,964 rows; 2,784,344 unique directed rows |
| Actor edge anomalies | 45,981 self-loops and 84,620 duplicate directed rows |
| Transaction files | present but not used by the current actor pipeline |

All observed actor-edge endpoints are represented in the actor class table in
the local audit. This does not validate the external dataset version or the
semantics of the time step. The code assumes that repeated actor rows are
historical snapshots, but does not verify whether they represent disjoint
windows, cumulative statistics, or another release-specific construction.

### Data risks

- No official source URL, release version, license metadata, checksums, or raw
  download script is recorded.
- No explicit duplicate/missing-value report is persisted by `data_audit`.
- The actor and transaction datasets are not joined or reconciled.
- Unknown class `3` is excluded from label metrics, but the missingness process
  is not analysed.
- There is no train/validation/test split in the current benchmark.
- There is no actor-disjoint or time-disjoint evaluation protocol.
- `pd.read_csv` loads the full actor feature file in several paths; the stated
  8 GB resource target is not enforced by the code.

## D. Graph construction audit

The graph is an address-to-address relation graph derived from `AddrAddr`.
`src.community` treats it as undirected for Leiden and uses duplicate counts as
edge weights. This is a reasonable descriptive projection, but it is a strong
modelling choice rather than an identity supplied by the data. Direction,
transaction identity, transaction value, and time are discarded in the Leiden
graph. Self-loops are omitted for Leiden but retained as a separate feature in
the vector baseline.

The implementation does not construct dynamic graph snapshots, does not model
multi-edge attributes beyond a count, and does not report connected components,
isolates, reciprocity, edge-weight distribution, or component-level label
composition as part of the authoritative pipeline. Degree features are computed
from the complete edge list before sampling, so they are full-history features.

This is valid only for a transductive descriptive question. It is temporal
leakage for any claim that a model uses information available before a cutoff.
The graph also includes unlabeled actors, which is useful for unsupervised graph
structure but must be stated in every evaluation protocol.

## E. Feature engineering audit

The behavioural view uses mean and max over all available actor-time rows. The
graph view uses weighted in/out degree, unique in/out degree, and self-loop
count. Missing joined graph values are filled with zero. The matrix then uses
median imputation, signed `log1p`, robust scaling, and PCA.

Strengths:

- Labels are not passed into the clustering matrix.
- Heavy-tailed degree features receive an explicit transformation.
- The preprocessing is implemented in code rather than hidden in a notebook.

Weaknesses:

- Mean/max discard event order, duration, recency, volatility, and exposure.
- Full-history aggregation and preprocessing are incompatible with temporal
  deployment claims.
- PCA retains 99.9955% variance in the recorded run, so the chosen 30-component
  reduction is not a scientifically justified bottleneck.
- No feature provenance table links variables to domain meaning or literature.
- No ablation isolates behaviour, graph structure, direction, weights, or
  temporal summaries.
- CC-MVAD uses Euclidean residuals in PCA space without a null calibration,
  uncertainty estimate, or choice rule fixed before label evaluation.

## F. Algorithm audit

The current baselines are KMeans, HDBSCAN, and Leiden. KMeans is a geometric
baseline; HDBSCAN is a density baseline; Leiden is a graph-community baseline.
This is a sensible small baseline set for an exploratory study. It is not yet a
complete comparison because there is no documented simple graph baseline such
as connected components or Louvain, no embedding-plus-clustering baseline, and
no supervised/semi-supervised benchmark if the intended task becomes detection.

The repository mentions possible Node2Vec, GAE/VGAE, GNNs, and other methods only
as future work. They should not be added as a model zoo. A representation model
is justified only if it tests a stated hypothesis about direction, time,
multi-view structure, or inductive generalization.

## G. Evaluation audit

Reported metrics include silhouette, Davies--Bouldin, Calinski--Harabasz,
ARI/NMI on known labels, modularity, and illicit enrichment. Geometry metrics
measure compactness in the chosen representation, not illicit discovery.
ARI/NMI are appropriate external clustering measures when labels are evaluation
only, but they ignore unknown labels and do not measure ranking utility.
Modularity measures graph partition structure and is not evidence of illicit
enrichment.

The enrichment implementation has a naming problem: `cluster_size` is the
number of known labels after filtering, not the total community size. Results
must report `total_count`, `known_count`, and `unknown_count` separately.

The recorded Leiden result at resolution 1.0 has modularity about 0.9212 and a
community with 1,142 illicit actors among 1,730 known actors, about 12.28 times
the known-label baseline. This is an association discovered after scanning
communities, resolutions, and seeds; it is not an unbiased held-out estimate.
The most extreme ratios are driven by tiny communities (sometimes 2--15 known
actors).

The permutation test is directionally appropriate because it preserves class
counts and compares the maximum across communities. However, 100 permutations
impose a minimum attainable p-value of 0.00990099, and the null does not account
for selecting the headline resolution, seed, feature view, or downstream
residual score. No confidence intervals for individual enrichment estimates are
reported.

## H. Experimental methodology audit

Critical missing components:

- pre-registered primary estimand and primary metric;
- time-respecting train/validation/test protocol;
- repeated seeds with mean and standard deviation;
- selection-aware model and resolution tuning;
- baseline for label prevalence and graph-size confounding;
- ablations for time, direction, weights, behaviour, and graph structure;
- partition similarity across Leiden runs, not only modularity;
- actor-level ranking evaluation for CC-MVAD;
- calibrated uncertainty and multiple-testing treatment;
- runtime, memory, and scalability measurements;
- immutable run manifest with input hashes, dependency versions, git revision,
  configuration, and outputs;
- automated tests for graph aggregation, label exclusion, alignment, and null
  generation.

`config/experiment.json` declares settings but `run_experiments.py` hard-codes
the KMeans/HDBSCAN settings and writes seed 42 in the manifest. Notebook cells
and CLIs can overwrite shared results. Requirements are unpinned and there is
no packaging or test configuration. These are reproducibility blockers.

## I. Literature positioning and verified references

The following references were checked through scholarly metadata or stable DOI
records and are suitable starting points, not a completed systematic review:

| Paper | Year / venue | Dataset or problem | Method / metric relevance | Limitation for this project |
|---|---|---|---|---|
| Weber et al., *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics* | 2019, KDD-related work; DOI `10.48550/arXiv.1908.02591` | Elliptic Bitcoin transaction graph | Graph-based illicit classification | Primarily supervised node classification; not an unsupervised community estimand |
| Traag, Waltman & van Eck, *From Louvain to Leiden: guaranteeing well-connected communities* | 2019, Scientific Reports; DOI `10.1038/s41598-019-41695-z` | General community detection | Leiden connectivity and modularity | Does not establish illicit discovery or temporal validity |
| Blondel et al., *Fast unfolding of communities in large networks* | 2008, Journal of Statistical Mechanics; DOI `10.1088/1742-5468/2008/10/P10008` | General community detection | Louvain baseline | Older static method; still needed as a fair baseline |
| Grover & Leskovec, *node2vec: Scalable Feature Learning for Networks* | 2016, KDD; DOI `10.1145/2939672.2939754` | General graph representation | Embedding plus downstream clustering baseline | Static topology and no explicit transaction semantics |
| Rossi et al., *Temporal Graph Networks for Deep Learning on Dynamic Graphs* | 2020, ICML workshop / arXiv record | Dynamic graphs | Event-time representation and temporal evaluation concepts | Requires event semantics and a different task protocol |
| Zerveas et al., *A Transformer-based Framework for Multivariate Time Series Representation Learning* | 2021, KDD | Multivariate temporal representation | Useful temporal-feature baseline | Not graph-specific and does not solve community evaluation |
| McInnes, Healy & Astels, *hdbscan: Hierarchical density based clustering* | 2017, JOSS; DOI `10.21105/joss.00205` | General clustering | Density baseline and noise handling | Geometry metrics alone do not establish illicit utility |

OpenAlex/Crossref searches did not reliably return a unique peer-reviewed
record for the exact “Elliptic++” title from the local files. The official
dataset release, paper, version, and license must be identified from the actual
download source before citing it as the data paper. The current report should
not claim that the local files are an exact official release until that check is
complete.

## J. Research-worthiness decision

### Strengths

- Large graph and a useful mix of actor-time, actor-edge, and transaction files.
- Clear separation of labels from unsupervised fitting in the main baseline.
- Memory-conscious integer graph conversion and explicit duplicate-edge policy.
- A potentially meaningful problem: structure-aware discovery under incomplete
  labels.

### Critical blockers

- No verified dataset provenance or temporal semantics.
- No held-out protocol, so current enrichment is exploratory and selection-biased.
- CC-MVAD is named as a contribution without actor-level evidence that it adds
  value over simpler scores.
- Current temporal output reports ARI 1.0 but is not generated by the documented
  workflow and independently fits preprocessing in each window; it must not be
  used as evidence.
- Configuration, dependencies, artifacts, and results are not fully tracked.

### Components to remove from claims

- “detector”, “early warning”, “temporal stability”, and “paper-ready” claims;
- headline use of the maximum 12.28x enrichment without selection control;
- any novelty claim for CC-MVAD before comparison with simple residual and
  community baselines.

### Recommended direction

Reframe the project as **time-respecting, community-aware actor discovery under
partial labels**. The central contribution should be a validated evaluation
protocol and a modest multi-view representation that combines graph context
with behaviour only when it improves held-out actor ranking or community
generalization. If the dataset cannot support valid time semantics, fall back to
transductive community enrichment and present it explicitly as exploratory.
