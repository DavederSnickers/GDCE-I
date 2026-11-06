# The shared classifiers

One classifier per (dataset, architecture) that every counterfactual method in
the comparison targets, plus a control copy of each that differs **only in the
training seed**. The copy is what makes NAFR meaningful: a counterfactual that
flips the attacked classifier but not an equally good independent one is an
adversarial example rather than an explanation.

Each classifier is two files, `judge_<dataset>_<conv>[_seed1].npz` (weights) and
`.json` (the spec `load_judge` rebuilds the module from). Never a `torch.save`:
the five baselines run in five mutually incompatible environments (Python 3.9 to
3.11, torch 1.11 to 2.5, PyG 2.3 to 2.6), and message passing in
`shared_judge.py` is written with `index_add_` so that nothing depends on PyG
internals that changed in between.

Accuracies below were measured on the paper's split. The repository does not
distribute that split, so a rebuilt dataset will not reproduce these numbers
exactly; `data/README.md` explains why and how far off to expect.

| file | conv | features | edge dim | selected epoch | val acc | test acc |
|---|---|---|---|---|---|---|
| `judge_mutagenicity_mol_gcn`        | gcn  | 13 | 0 | 225 | 0.842 | **0.806** |
| `judge_mutagenicity_mol_gcn_seed1`  | gcn  | 13 | 0 | 195 | 0.840 | 0.806 |
| `judge_mutagenicity_mol_gine`       | gine | 13 | 5 | 130 | 0.924 | **0.813** |
| `judge_mutagenicity_mol_gine_seed1` | gine | 13 | 5 | 120 | 0.920 | 0.818 |
| `judge_benzene_gcn`                 | gcn  |  8 | 0 | 285 | 0.938 | **0.919** |
| `judge_benzene_gcn_seed1`           | gcn  |  8 | 0 | 290 | 0.937 | 0.919 |
| `judge_benzene_gine`                | gine |  8 | 5 |  20 | 1.000 | **1.000** |
| `judge_benzene_gine_seed1`          | gine |  8 | 5 |  15 | 1.000 | 1.000 |
| `judge_proteins_gcn`                | gcn  |  3 | 0 | best val | — | **0.632** |
| `judge_proteins_gcn_seed1`          | gcn  |  3 | 0 | best val | — | 0.638 |
| `judge_twitter_gcn`                 | gcn  | 238 | 0 | best val | — | **0.648** |
| `judge_twitter_gcn_seed1`           | gcn  | 238 | 0 | best val | — | 0.655 |

Twelve files in total, six classifiers and their six control copies. The
preprocessing and some baseline adapters still mention NCI1, which was dropped
from the paper; no NCI1 classifier, split or result is shipped.

Benzene/GINE reaching 1.000 is not a bug: the label is the presence of a benzene
ring, and an edge-type-aware network can read an aromatic ring off the bond
types directly. That is also why the Benzene column of Table 2 is the one where
the bond-type-blind baselines lose the most.

## Recipe

Molecular datasets: Adam, lr 1e-3 with cosine decay, weight decay 1e-4, batch
64, dropout 0.3, BatchNorm, hidden 128, 3 layers, mean pooling, 300 epochs. The
epoch count is selected on **val**, then the model is refit from scratch on
train+val with that count (`--refit_train_val`), and test is touched once. The
`val_acc` column above is therefore the accuracy of the *selection* run, not of
the released weights — for a refit model val is inside the training set.

PROTEINS: hidden 64, 2 layers, weight decay 1e-2, 300 epochs, batch 64.
TWITTER: 60 epochs, weight decay 1e-3, batch 128. Both keep the best validation
epoch without a refit, and both are the weaker classifiers of the five — 0.632
and 0.648. They are reported as such in Section S2; a counterfactual explains
the model it attacks, so a classifier being mediocre does not invalidate the
explanation, but it does mean the two datasets carry less signal about the
method than Mutagenicity and Benzene do.

```bash
reproduce/reproduce_gdce_i_results.sh judges     # retrains all of them, ~1 GPU h
```

`judges_original/` holds the baselines' *own* classifiers, each trained to its
paper's recipe, for the control experiment in Section S3. One of them is
missing: `configs/experiment/*_orig_xplgcn_s3.yaml` references
`judge_mutagenicity_mol_xplgcn`, which is not part of the released set, so that
one configuration cannot be run as shipped.
