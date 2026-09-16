# Research Plan

## Proposed title

Candidate titles:

1. **Time-Respecting Community-Aware Discovery of Illicit Actors in Bitcoin
   Transaction Graphs**
2. **Community Context and Behavioural Deviation for Illicit-Actor Discovery
   under Partial Labels**
3. **Evaluating Multi-View Actor Representations for Illicit Group Discovery in
   Dynamic Cryptocurrency Graphs**
4. **From Static Enrichment to Held-Out Actor Discovery in Bitcoin Graphs**

These are working titles. They do not imply that the proposed method is novel
until the literature review and experiments establish an improvement.

## Research questions

**RQ1.** Does a graph representation identify actor groups that are enriched for
known illicit labels beyond class-prevalence and community-size baselines?

**RQ2.** Does adding behaviour to graph context improve held-out illicit-actor
ranking or group discovery, compared with graph-only and behaviour-only views?

**RQ3.** Do the findings persist across time cutoffs, graph construction choices,
random seeds, and plausible missing/noisy-edge perturbations?

**RQ4.** Which structural, behavioural, and temporal characteristics explain a
stable high-risk group without using labels to form the representation?

## Hypotheses

- **H1:** A graph-only community score has higher held-out illicit enrichment
  than a prevalence-matched random or degree-only baseline.
- **H2:** A community-conditioned behavioural deviation score improves held-out
  ranking metrics over graph-only and behaviour-only scores.
- **H3:** The improvement in H2 disappears or materially shrinks when future
  edges/features are allowed into preprocessing, demonstrating why temporal
  separation is necessary.
- **H4:** Direction and edge multiplicity provide measurable information beyond
  an unweighted undirected projection.
- **H5:** A useful discovery signal is stable across seeds and perturbations,
  measured by both ranking agreement and partition similarity, not only
  modularity.

## Problem formulation

For each cutoff $t$, construct a graph $G_{\leq t}$ and actor feature matrix
$X_{\leq t}$ using only information observed at or before $t$. Produce an
unsupervised score $s_i(t)$ or a community assignment $c_i(t)$ for actors
available at $t$. Labels observed later or held out at $t$ are used only for
evaluation. The primary estimand is held-out illicit enrichment among actors
with known evaluation labels, with precision-recall area under the curve as a
secondary actor-ranking metric when the label protocol permits it.

The first implementation should support two explicitly named regimes:

- **Transductive descriptive:** all observed graph data may be used; no future
  deployment claim is allowed.
- **Temporal inductive evaluation:** training/tuning uses an earlier period and
  evaluation uses a later period; preprocessing and graph construction are
  fit/constructed inside each allowed history.

## Proposed methodology

1. **Provenance and audit.** Record the verified official dataset release,
   checksums, schema, label semantics, timestep semantics, missingness, and
   duplicate policy. Fail fast if required columns or address alignment differ.
2. **Temporal protocol.** Use at least three chronological cutoffs after
   verifying that timestep is meaningful. Keep an untouched final period for
   primary reporting. If the data cannot support this, use only the transductive
   regime and label the limitation.
3. **Graph views.** Compare directed weighted, directed unweighted, and
   undirected weighted actor graphs. Keep self-loops as a documented sensitivity
   condition rather than silently mixing policies. Report components, isolates,
   reciprocity, degree, weight, and temporal edge statistics.
4. **Behaviour views.** Build causal-at-cutoff summaries: counts, recency,
   active-duration, mean, variability, max, and missingness indicators. Use
   feature names and provenance metadata. Fit imputation/scaling/PCA only on the
   training period where applicable.
5. **Representation.** Start with simple baselines: degree/strength score,
   Louvain/Leiden communities, behaviour-only clustering, and Node2Vec plus
   clustering. The proposed multi-view score combines community context and
   within-community behavioural deviation, with a training-only calibration
   choice. Add temporal embeddings only if the data protocol supports them.
6. **Clustering/discovery.** Treat communities as discovery units and actor
   scores as ranking outputs. Never select resolution, threshold, or headline
   community using the final evaluation labels.
7. **Evaluation.** Report known-count and unknown-count separately. Use PR-AUC,
   recall at fixed review budget, precision at k, enrichment at k, and bootstrap
   intervals for actor ranking. For communities report size-weighted enrichment,
   coverage, conductance, internal density, and partition similarity. ARI/NMI
   remain secondary external clustering metrics.
8. **Interpretation.** Characterize stable groups with feature distributions,
   edge-direction patterns, degree/weight ranges, temporal activity, and
   confidence intervals. Feature importance must be computed without using
   held-out labels for representation selection.

## Baselines

Minimum baseline set:

- prevalence-matched random ranking;
- degree/strength-only ranking;
- behaviour-only clustering or anomaly score;
- Leiden and Louvain on each graph view;
- Node2Vec plus KMeans or HDBSCAN;
- community-conditioned residual score without any learned weighting.

Optional supervised baselines should be clearly separated from unsupervised
discovery and used only if the research question includes detection. A small
logistic regression and a graph model with strict time splits are enough; do not
add GAE, VGAE, GraphSAGE, GAT, and GIN without a hypothesis and compute budget.

## Ablation matrix

| Variant | Removed component | Purpose |
|---|---|---|
| Full | none | Primary proposed method |
| Behaviour-only | graph communities/context | Test graph contribution |
| Graph-only | behavioural residual | Test behaviour contribution |
| No time | temporal summaries/cutoff | Test temporal information and leakage sensitivity |
| Undirected/unweighted | direction or multiplicity | Test graph construction assumptions |
| Global centroid | community-conditioned centroid | Test whether community context matters |
| No PCA | representation bottleneck | Test dependence on dimensionality reduction |
| Fixed resolution sweep | alternative resolution policy | Test selection sensitivity |

Every variant must use the same cutoffs, seeds, evaluation actors, and output
schema. Improvements must be reported as distributions, not a single best run.

## Robustness and statistics

- Run at least 10 documented seeds for stochastic methods and report mean +/-
  standard deviation plus seed-level values.
- Use a final untouched temporal evaluation set; use validation only for
  resolution, thresholds, and hyperparameters.
- Use a permutation or bootstrap procedure that reproduces the full selection
  procedure when testing the maximum enrichment.
- Increase permutation count enough for the planned alpha, or report exact
  finite-sample limitations.
- Perturb edges by deletion, addition, weight noise, and missing actor features
  at realistic rates; report degradation curves.
- Compare Leiden partitions with adjusted Rand index, variation of information,
  or normalized mutual information after optimal label alignment.
- Measure runtime, peak memory, graph size, and scaling by actor/edge count.

## Expected artifacts

Each run writes an immutable run directory containing:

- `manifest.json`: dataset version and hashes, git revision, environment,
  command, seed, cutoff, graph/features/model configuration, and runtime;
- structured raw/processed features with schema metadata;
- memberships, actor scores, and predictions with actor IDs and split labels;
- metrics with denominators, confidence intervals, and selection regime;
- ablation, robustness, runtime, and scalability tables;
- figures generated from structured tables rather than notebook state.

Required paper tables/figures include dataset statistics, graph statistics,
baseline comparison, main held-out results, ablations, robustness, runtime,
degree/weight distributions, partition stability, and interpretable profiles of
stable communities. No artifact may overwrite an existing run silently.

## Paper structure

1. Abstract
2. Introduction and research questions
3. Related work and verified dataset provenance
4. Problem formulation and leakage model
5. Methodology and graph/feature views
6. Experimental protocol and baselines
7. Main held-out results
8. Ablation and robustness analysis
9. Interpretability and case-level community profiles
10. Discussion and practical implications
11. Limitations, dataset uncertainty, and ethical considerations
12. Reproducibility statement and conclusion

## Venue positioning

Potential venues must be selected after the contribution is known. A graph/data
mining venue is appropriate only if the work has a defensible methodological
contribution and rigorous experiments; an applied cybersecurity/financial
crime venue is appropriate if the emphasis is validated discovery and domain
interpretation. Candidate scopes to investigate include *Knowledge and
Information Systems*, *Data Mining and Knowledge Discovery*, *IEEE Transactions
on Network Science and Engineering*, *Applied Network Science*, and
*Computers & Security*. This is a scope shortlist, not an acceptance prediction.
Each submission would require verified citations, a clear contribution beyond
repackaging Leiden, held-out evidence, transparent limitations, and complete
reproducibility materials.
