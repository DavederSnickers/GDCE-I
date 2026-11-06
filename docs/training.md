# Training the models yourself

The six diffusion models are not distributed with this repository. Everything
that determines them is:

* the classifier each model is conditioned on (`judges/`, in git, 2.9 MB),
* every hyperparameter (`configs/experiment/*.yaml`, and Section S4 of the
  supplement),
* the container the runs happened in (`vm/counter.def`).

GDCE-I trains with `label_source: 'classifier'`, so the condition the diffusion
model learns is **the label the shared classifier predicts**, not the
ground-truth label. Because that classifier is published as weights rather than
as a recipe, your generator is conditioned on the labels of exactly the model
ours was conditioned on, and the same model is what the counterfactual search
attacks and what the metrics are measured against. The decision boundary is
therefore identical to the paper's.

The graphs it sees are not. `SpectreGraphDataset.process()` draws the 60/20/20
split with an unseeded permutation, so your split differs from ours and from any
other build. Section 1 says what that costs.

---

## 1. What is pinned and what is not

Pinned, and identical for you and for us:

| | where |
|---|---|
| guidance labels | the shipped `judges/judge_<ds>_<conv>` |
| node and edge marginals, graph-size distribution | `data/<ds>/raw/*.npy`, measured on the paper's split |
| architecture, T, schedule, loss weights, optimizer, epochs, batch size, `p_uncond` | `configs/experiment/<ds>_<conv>.yaml` |
| evaluation samples | run seeds 42–46, stratified by predicted class, 5 × 100 |
| Python, torch, CUDA, PyG, Lightning, RDKit versions | `vm/counter.def` |

Not pinned: the train/val/test split (unseeded, see above) and the training run
itself. `src/main.py` calls
`seed_everything(cfg.train.seed)` with `seed: 0`, but the Lightning `Trainer` is
constructed without `deterministic=True`, so cuDNN kernel selection and the DDP
reduction order make two runs on different hardware differ. We did not measure
the spread of a fresh training run, and we will not pretend otherwise: the
repository has one training run per model, and the two ablation generations in
our own history differ in the classifier as well, so they do not isolate it.

What this means in practice: treat the standard deviations in the paper as the
spread over the five evaluation samples of **one** model, and expect your
retrained model to land near, not on, the published cells. The checks in
section 3 tell you whether a deviation is your pipeline or your training run.

## 2. The commands

```bash
. reproduce/common.sh                   # exports GDCE_DATA, GDCE_JUDGES, GDCE_CKPT, GDCE_OUT
reproduce/reproduce_gdce_i_results.sh data      # download, filter and split the four datasets
reproduce/reproduce_gdce_i_results.sh train     # the six models
```

`train` is a sequence of six commands of this shape, plus a copy of the
resulting checkpoint to the flat name the inversion configs expect:

```bash
python src/main.py +experiment=mutag_mol_gcn dataset=mutagenicity_mol \
    hydra.run.dir=$GDCE_OUT/train/mutag_mol_gcn
cp $GDCE_OUT/train/mutag_mol_gcn/checkpoints/mutag-mol-gcn/last.ckpt \
   $GDCE_CKPT/mutag_mol_gcn.ckpt
```

| model | experiment | dataset | epochs | batch | `p_uncond` | checkpoint used |
|---|---|---|---|---|---|---|
| `mutag_mol_gcn`  | `mutag_mol_gcn`  | `mutagenicity_mol` | 600 | 128 | 0.1  | last |
| `mutag_mol_gine` | `mutag_mol_gine` | `mutagenicity_mol` | 600 | 128 | 0.1  | last |
| `benzene_gcn`    | `benzene_gcn`    | `benzene`          | 600 | 256 | 0.1  | last |
| `benzene_gine`   | `benzene_gine`   | `benzene`          | 600 | 256 | 0.1  | last |
| `proteins_gcn`   | `proteins_v1`    | `proteins`         | 600 | 128 | 0.05 | last |
| `twitter_gcn`    | `twitter_v1`     | `twitter`          | 300 | 128 | 0.05 | last |

Shared by all six, and not tuned per dataset: the DiGress graph transformer with
8 layers, `dx 256 / de 64 / dy 128`, 8 heads, feed-forward widths
`256/128/256`, no extra structural features; marginal transition matrices,
T = 500, cosine noise schedule; cross-entropy over nodes and edges with the
**edge** term weighted 5 (`model.lambda_train: [5, 0]`); AdamW, lr 2e-4, weight
decay 1e-4, gradient clipping 1.0, no EMA, seed 0.

Cost: about 2–4 GPU days per molecular model on one A100, less for PROTEINS. No
checkpoint is selected on a counterfactual criterion — the last epoch is used,
so nothing has to be held out for model selection.

Before spending that, run the pipeline on two graphs to check the wiring:

```bash
SMOKE=1 reproduce/reproduce_gdce_i_results.sh table2
```

## 3. Is my retrained model comparable?

Four checks, in order. The first three must match **exactly** — they do not
depend on the training run at all, so a mismatch there is a pipeline problem,
not variance.

**(a) The data.** `src/build_dataset.py` prints the split sizes and the number
of node types. Against Table S1:

| dataset | train / val / test | node types | edge classes |
|---|---|---|---|
| Mutagenicity | 2377 / 792 / 792 | 13 | 5 |
| Benzene | 7178 / 2393 / 2393 | 8 | 5 |
| PROTEINS | 523 / 174 / 174 | 3 | 2 |
| TWITTER | 39019 / 13006 / 13006 | 238 | 2 |

Watch the log for `WARNING: marginals not found on disk` — the run continues
with a uniform marginal and every number afterwards is wrong.

**(b) The classifier.** The shipped weights are the ones we attacked. Their
accuracy on *your* test split is not expected to match the numbers below, which
were measured on ours — but it should land close, and a result far off, or at
chance, means the one-hot encoding or the node-type mapping is wrong rather than
the split:

| classifier | test accuracy |
|---|---|
| `judge_mutagenicity_mol_gcn` | 0.8056 |
| `judge_mutagenicity_mol_gine` | 0.8131 |
| `judge_benzene_gcn` | 0.9185 |
| `judge_benzene_gine` | 1.0000 |

`src/judge_agreement.py` additionally prints the agreement of the attacked
classifier and its control copy on unmodified test graphs. That number is the
floor for NA, so measure it on your split and compare your NA against your own
floor, not against the paper's.

**(c) Exact reconstruction.** A property of the inversion, not of the training:
with `guidance.inversion_method: 'posterior'` and the replay under the *source*
condition, the output must equal the input bit for bit. Turn the assertion on
with `++guidance.recon_check=true`, or read it off the placebo column of
`src/review_paired_tau.py`. If it fails on some GPUs because bf16 ties break the
argmax, set `++guidance.inversion_margin=1e-3`. If it fails systematically, the
inversion is misconfigured and nothing downstream is meaningful.

**(d) The counterfactual numbers.** Now your split and your training run both
enter, so this is where exact agreement stops being the criterion. For
Mutagenicity against the shared GCN at s = 3, the published cells are

| | value |
|---|---|
| FR | 0.970 ± 0.016 |
| NAFR | 0.886 ± 0.029 |
| NA | 0.913 |
| NAFR & SMILES | 0.744 ± 0.017 |
| Edge sparsity | 0.657 ± 0.011 |
| Node sparsity | 0.892 ± 0.009 |
| Edge-type sparsity | 0.993 ± 0.000 |
| median flip budget τ̃ | 51 |

**How close is close enough?** The standard deviations above are the spread over
the five samples of 100 test graphs, drawn from one split by one model — in
other words, they already are the sampling noise of an n = 100 draw. A different
split is another random fifth of the same graph population, so a draw from it
has to first order the same sampling distribution, and a mean that lands inside
roughly those bands is agreement, not a discrepancy. The measured per-run ranges
behind the table are FR 0.950–0.990, NAFR 0.860–0.930, NAFR & SMILES
0.730–0.770, edge sparsity 0.643–0.672, node sparsity 0.878–0.901.

Two quantities need more room than that:

* **τ̃**, the median flip budget, ranges 41–66.5 over the same five samples —
  about ±20 %. Compare the *ratio* between noise treatments instead, which is
  what Section 5.3 argues from: 51 against 11 is a factor of 4.6 and nowhere
  near that band.
* **PROTEINS** has 871 graphs in total and the paper evaluates its complete test
  split of 174 in a single run, so there is no sampling average to fall back on.
  Expect that row to move with the split more than the others.

The qualitative findings are what should survive a retrain, not the second
decimal: posterior inversion reaches roughly
twice the edge sparsity of no inversion at a matched budget, returns every input
exactly under the unchanged source condition while no inversion flips 17–26 % of
graphs there, and needs a larger median budget (τ̃ = 51 against 11) because its
edit grows with τ instead of being redrawn at every budget.

One convention to know before you compare anything to a CSV: the per-sample
CSVs store `edge_sparsity`, `node_sparsity` and `edgetype_sparsity` as
**modification rates**, and the tables report 1 − that. The 0.657 above is
1 − 0.343.

## 4. If you want a cheaper comparison

Mutagenicity against the shared GCN is the only configuration every ablation in
the paper uses (Tables 3, 4 and 5 and Fig. 3). Training that one model gets you
every ablation and the first row block of Table 2; the other five models only
add further datasets and the GINE columns.
