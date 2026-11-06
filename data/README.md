# Datasets

Four datasets, all binary graph classification:

| dataset | source | graphs (train/val/test) | node types | edge classes | max nodes | balance 0/1 |
|---|---|---|---|---|---|---|
| Mutagenicity | TUDataset | 2377 / 792 / 792 (3961) | 13 | 5 | 50 | 57.6 / 42.4 |
| Benzene | Agarwal et al. | 7178 / 2393 / 2393 (11964) | 8 | 5 | 25 | 49.9 / 50.1 |
| PROTEINS | TUDataset | 523 / 174 / 174 (871) | 3 | 2 | 50 | 52.9 / 47.1 |
| TWITTER | TUDataset (TWITTER-Real-Graph-Partial) | 39019 / 13006 / 13006 (65031) | 238 | 2 | 50 | 52.2 / 47.8 |

Edge classes 5 means `{no bond, single, double, triple, aromatic}`; 2 means a
plain binary adjacency.

## Build them

```bash
reproduce/reproduce_gdce_i_results.sh data
# = python src/build_dataset.py --name <dataset> --datadir data/<dataset>/
```

Mutagenicity, PROTEINS and TWITTER are downloaded from the TUDataset
collection; Benzene from the archive of Agarwal et al. Nothing is redistributed
here. The preprocessing is deterministic: the atom-frequency filter of Huang et
al. (node types with a global frequency below 50 are dropped, graphs containing
one of them are removed), the node cap at 50, and the relabelling of the
guidance target with the shared classifier in `judges/`. The totals in the table
above are what you should get.

## What is in git, and why only this

```
data/<dataset>/raw/node_type_map.npy        original TU node label -> contiguous index
data/<dataset>/raw/node_type_marginals.npy  the marginal the noise model starts from
data/<dataset>/raw/edge_type_marginals.npy  entry 0 is P(no edge) and dominates
data/<dataset>/raw/n_nodes_dist.npy         P(graph has n nodes)
data/<dataset>/raw/atom_decoder.json        index -> element symbol (molecular only)
data/<dataset>/raw/atom_encoder.json
```

A few hundred bytes each, and they are the values the paper's models were
trained with. `build_dataset.py` recomputes them from your own build and
overwrites them, so they are a reference, not an input you have to trust.

Watch the startup log for `WARNING: marginals not found on disk` — the run
continues with a uniform marginal and every number afterwards is wrong.

Everything else — `raw/{train,val,test}.csv`, `raw/TU/`, `processed/` — is built
by the command above and is not distributed.

## The split

`SpectreGraphDataset.process()` draws the 60/20/20 split with
`np.random.permutation` and **no seed**. Two builds therefore give two different
splits, and neither is the paper's.

How much that matters is measurable, and the answer is: less than it sounds.
The split sizes, the class balance, the node-type count and the preprocessing
are deterministic and unaffected. The counterfactual metrics are reported as a
mean over five samples of 100 test graphs, and the standard deviations in the
paper already are the sampling noise of such a draw (FR ±0.016, NAFR ±0.029).
A different split is another random fifth of the same graph population, so it
contributes variance of the same order — not a different result.
`docs/training.md` section 3 gives the per-metric bands and names the two
quantities that need more room, τ̃ and the PROTEINS row.

What genuinely does not reproduce is the accuracy of the shipped classifiers
measured on *your* test split: they were trained on ours, so some of your test
graphs were in their training set. That is a statement about a generalization
number, not about the counterfactuals — a counterfactual explains the model it
attacks, and whether a graph was in that model's training set does not change
what flipping it means. Train your own with `reproduce/... judges` if you need
an uncontaminated accuracy.

If you want a build you can repeat, seed the permutation in your own branch —
`TUTORIAL.md` says where.
