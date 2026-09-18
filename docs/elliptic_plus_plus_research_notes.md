# Elliptic++ Research Notes

## Source

The dataset documentation and original study are:

- Elmougy and Liu, "Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network for Financial Forensics," KDD 2023.
- Official repository: https://github.com/git-disl/EllipticPlusPlus
- Paper: https://doi.org/10.1145/3580305.3599803

## Dataset interpretation

Elliptic++ exposes two related datasets: transactions and actors. The actor
dataset treats Bitcoin wallet addresses as actor nodes and provides temporal
address interactions, address features, labels, and several graph views.

The source paper distinguishes four graph types:

1. Transaction-to-transaction money-flow graph.
2. Address-to-address interaction graph.
3. Address-to-transaction graph, representing address/transaction flow.
4. User-entity graph, representing clusters of addresses associated with a user.

The fourth graph must not be inferred automatically from a connected component
of the address-to-address graph. In this project, wallet address is therefore
the primary clustering unit. Connected components remain an exploratory proxy
unless an independent entity-resolution rule is introduced and validated.

## Project decisions

- `AddrAddr_edgelist.csv` is the primary graph baseline.
- `AddrTx_edgelist.csv` and `TxAddr_edgelist.csv` are reserved for validating or
  reconstructing a transaction projection.
- `wallets_classes.csv` is address-level ground truth for external evaluation,
  not a clustering input.
- `Time step` describes temporal actor observations. The current AddrAddr file
  has no timestamp, so an induced subgraph filtered by active addresses must not
  be described as a true time-specific interaction graph.
- The source project's feature tables contain repeated exact rows for some
  address/time observations. The common pipeline removes exact duplicates before
  address-level aggregation and records the validation counts.

These decisions are based on the official dataset documentation and are
implemented independently of the illicit/licit labels.