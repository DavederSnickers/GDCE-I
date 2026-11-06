# Using GDCE-I on your own data or your own classifier

GDCE-I needs exactly two things from you:

1. **A dataset** of graphs with categorical node types, categorical edge classes
   and a binary graph label, where every graph fits a node cap (50 in the paper).
2. **A classifier** that can be called as `f(x, edge_index, batch, edge_attr)`
   and returns raw logits.

Nothing in the method changes. The diffusion model is trained on your graphs,
conditioned on the label *your* classifier predicts, and the counterfactual
search then asks that same classifier when it has flipped.

This document is the short path through both. Section 4 is the part people
usually get wrong.

---

## 0. Before you start: is your dataset learnable for the wrong reason?

Worth ten minutes, because it invalidates everything downstream. If a trivial
feature already predicts the label, a counterfactual explainer will find that
feature and you will read it as an explanation of the model.

```bash
python src/screen_tudataset.py --name <TU-NAME>        # accuracy from size and density alone
python src/bow_ceiling.py --dataset <name>             # node-type bag-of-words ceiling
```

The protocol used for this paper: a size/density accuracy of 0.60–0.65 is normal
and acceptable, 0.995 is not — that is why AIDS was rejected. If the
bag-of-words ceiling matches your GNN's accuracy, your classifier is not using
the structure, and neither will the explanation (TWITTER is the borderline case
in the paper, and Section S2 says so).

---

## 1. The data contract

Everything downstream reads three CSVs per dataset plus a few small `.npy`
files. If you produce these, you have done the hard part:

```
data/<name>/raw/train.csv
data/<name>/raw/val.csv
data/<name>/raw/test.csv
data/<name>/raw/node_type_map.npy          int array: index -> your original node label
data/<name>/raw/node_type_marginals.npy    float array, sums to 1, length = #node types
data/<name>/raw/edge_type_marginals.npy    float array, sums to 1, length = #edge classes
data/<name>/raw/n_nodes_dist.npy           float array, index n = P(graph has n nodes)
data/<name>/raw/atom_decoder.json          molecules only: ["C", "N", "O", ...]
data/<name>/raw/atom_encoder.json          molecules only: the inverse mapping
```

CSV columns, one row per graph:

| column | content |
|---|---|
| `graph_id` | any unique id |
| `y` | `0` or `1`, the label |
| `node_types` | a python list literal, e.g. `[0, 3, 3, 1]` — one index per node, into `node_type_map` |
| `adjacency_matrix` | the flattened n×n adjacency as `[1.0, 0.0, ...]`, row-major |
| `edge_type_matrix` | the flattened n×n matrix of edge classes; only needed if you have more than a binary adjacency |

Two conventions that are easy to get wrong:

* **The diagonal.** Mutagenicity was built with `np.fill_diagonal(adj, 1.0)`, so
  its `edge_index` contains a self-loop on every node; Benzene was built without
  it. Both work, because the classifier strips self-loops on input and re-adds
  them per its own convention — but your classifier and your CSV have to agree
  with each other. Pick no self-loops; it is the simpler half of the code.
* **Edge class 0 means "no edge"**, so with bond types you have 5 classes
  `{none, single, double, triple, aromatic}` and with a plain adjacency you have
  2. `edge_type_marginals.npy` must have exactly that length, and its first
  entry — the probability of *no* edge — dominates: it is what makes the noise
  model produce sparse graphs.

The marginals are the distribution the forward process destroys the graph into.
Measure them on your training split, not on all of it.

The three CSVs are also the only thing `src/shared_judge/shared_judge.py` reads,
so a classifier trained from them and the diffusion model see literally the same
graphs.

---

## 2. Wiring a new dataset in

`src/datasets/spectre_dataset.py` already has two generic paths, and your
dataset almost certainly fits one of them:

* **binary adjacency** — the `('proteins', 'imdb_binary', 'twitter')` path
* **categorical edge classes** — the `benzene` path (5 bond classes)

So do not write a new branch. Add your name to the existing one, in three
places:

```python
# 1. process(), the node-type count    (search for "_tu_num_node_types")
if self.dataset_name in ('proteins', 'imdb_binary', 'twitter', 'mydata'):
    ...

# 2. process(), the one-hot construction  (the elif with the same tuple)
elif self.dataset_name in ('proteins', 'imdb_binary', 'twitter', 'mydata'):
    ...

# 3. SpectreDatasetInfos.__init__, the marginals
elif dataset_name in ('proteins', 'imdb_binary', 'twitter', 'mydata'):
    ...
```

For a dataset with edge classes, add it to the `benzene` branch in the same two
places in `process()` and to the `benzene` branch of `SpectreDatasetInfos`.

Then one line in the classifier code:

```python
# src/shared_judge/shared_judge.py
NUM_EDGE_TYPES = {..., "mydata": 0}        # 0 for a binary adjacency, 5 with bond types
```

and the `--dataset` choices of `src/shared_judge/train_judge.py` (it validates
the name on purpose, so that a typo does not silently train on the wrong data).

There is **no fallback branch**: `process()` ends in
`raise ValueError('Unkown dataset')`, which is the error you get if you forgot
place 2.

### The configs

```yaml
# configs/dataset/mydata.yaml
name: mydata
datadir: 'data/mydata/'
pin_memory: False
```

Copy `configs/experiment/proteins_v1.yaml` (binary adjacency) or
`benzene_gcn.yaml` (edge classes) to `configs/experiment/mydata_v1.yaml` and
change three things: `general.name`, `guidance.shared_judge_path` (or
`graph_classifier_path`), and the epoch / batch size. Everything else — 8
layers, `dx 256 / de 64 / dy 128`, T = 500, cosine schedule, marginal
transitions, `lambda_train: [5, 0]`, AdamW 2e-4 — was never tuned per dataset
and is a reasonable starting point for yours.

Then copy `inversion_proteins.yaml` / `inversion_benzene_gcn.yaml` to
`inversion_mydata.yaml`, point `general.test_only` at the checkpoint you are
about to produce, and set `guidance.s: 3`.

### Does it load?

```bash
export GDCE_DATA=$PWD/data GDCE_JUDGES=$PWD/judges GDCE_CKPT=$PWD/checkpoints
python src/build_dataset.py --name mydata --datadir data/mydata/
```

This runs `process()` and prints the tensor shapes that `SpectreDatasetInfos`
will expect. If the node-type count, the edge-class count and the marginal
lengths agree, the rest of the pipeline will run.

### One caveat if you let the code build your split

`SpectreGraphDataset.process()` draws its 60/20/20 split with
`np.random.permutation` and **no seed**, so two builds give two different
splits. If you want a reproducible split, either write the three CSVs yourself
(recommended — then `download()` is skipped entirely, because `raw_file_names`
is exactly those three files) or add

```python
np.random.seed(0)          # right before the perm = np.random.permutation(...)
```

in your `download()` branch.

---

## 3. Train and run

```bash
. reproduce/common.sh                      # sets GDCE_* and PYTHONPATH

# a) the classifier you want to explain, and its control copy
python src/shared_judge/train_judge.py --dataset mydata --conv gcn \
    --data_root "$GDCE_DATA" --out_dir "$GDCE_JUDGES" \
    --batchnorm --cosine --dropout 0.3 --refit_train_val --epochs 300
python src/shared_judge/train_judge.py --dataset mydata --conv gcn \
    --data_root "$GDCE_DATA" --out_dir "$GDCE_JUDGES" \
    --batchnorm --cosine --dropout 0.3 --refit_train_val --epochs 300 \
    --seed 1 --tag _seed1

# b) how often do the two already disagree on untouched graphs?  This is the
#    floor for NA -- without it a transfer rate means nothing.
python src/judge_agreement.py --name mydata \
    --judge-a "$GDCE_JUDGES/judge_mydata_gcn" \
    --judge-b "$GDCE_JUDGES/judge_mydata_gcn_seed1"

# c) the conditional diffusion model  (the long part: 1-4 GPU days)
python src/main.py +experiment=mydata_v1 dataset=mydata \
    hydra.run.dir="$GDCE_OUT/train/mydata_gcn"
cp "$GDCE_OUT/train/mydata_gcn/checkpoints/<general.name>/last.ckpt" \
   "$GDCE_CKPT/mydata_gcn.ckpt"

# d) the counterfactual search
python src/inversion_nci1_nofreq.py +experiment=inversion_mydata dataset=mydata \
    ++guidance.n_runs=5 ++guidance.n_samples_per_run=100 \
    ++guidance.t_step=10 ++guidance.t_max=240 ++guidance.dump_variant=s3 \
    "++guidance.transfer_classifier_path=$GDCE_JUDGES/judge_mydata_gcn_seed1" \
    hydra.run.dir="$GDCE_OUT/mydata/gcn_s3"

# e) FR, NAFR, NA and the sparsity columns
python src/shared_judge/adversarial_flip.py \
    --dump "$GDCE_OUT/mydata/gcn_s3/inversion_mydata_results/cf_dump_mydata_s3.jsonl" \
    --dataset mydata --data_root "$GDCE_DATA" \
    --judge_base "$GDCE_JUDGES/judge_mydata_gcn" \
    --judge_ctrl "$GDCE_JUDGES/judge_mydata_gcn_seed1" \
    --label mine/mydata --summary_csv "$GDCE_OUT/results/mydata.csv"
```

Use `src/inversion_mutag_mol.py` instead of `inversion_nci1_nofreq.py` if your
dataset has categorical edge classes. The name is historical: the first is the
binary-adjacency runner, the second the 5-edge-class molecular one.

The diffusion model is conditioned on the label the classifier predicts
(`label_source: 'classifier'`), not on the ground-truth label. That is
deliberate: a counterfactual explains the *model*, so the generator has to model
the model's decision boundary, not the data's. One consequence you cannot avoid:
**a new classifier means a new diffusion model.** This is the expensive part of
applying GDCE-I to your setting.

---

## 4. Your own classifier

### 4.1 If it is a message-passing graph classifier

Train it with `train_judge.py` and set its size on the command line
(`--conv gcn|gine --hidden --num_layers --dropout --batchnorm`). This is by far
the least work and gives you the portable export for free.

### 4.2 If it is your own architecture

The pipeline loads a classifier from `<prefix>.npz` (weights) plus
`<prefix>.json` (spec), never from `torch.save` — the five baselines in the
paper live in five incompatible torch versions, and a pickle does not travel
between them. To plug your own module in:

```python
# in src/shared_judge/shared_judge.py
class MyNet(nn.Module):
    conv = "mynet"                       # the tag that goes into the spec

    def __init__(self, num_features, num_classes, hidden, num_layers,
                 edge_dim=0, dropout=0.0):
        ...

    def forward(self, x, edge_index, batch, edge_attr=None):
        # x          (N, num_features)  one-hot node types
        # edge_index (2, E)             both directions, self-loops already stripped
        # batch      (N,)              graph index per node, or None for one graph
        # edge_attr  (E, edge_dim)     one-hot edge classes, or None
        return logits                    # (B, num_classes), RAW -- no softmax

ORIGINAL_JUDGES = {cls.conv: cls for cls in (..., MyNet)}
```

Then save it once:

```python
from shared_judge import save_judge
save_judge(my_model, "judges/judge_mydata_mynet",
           extra_spec={"dataset": "mydata", "test_acc": 0.81})
```

`load_judge` dispatches on `spec["conv"]`, so every caller — the relabelling of
the training data, the counterfactual search, the NAFR scorer and every baseline
adapter — picks your class up without another change.

Four conventions the rest of the code assumes, and silently relies on:

* **Raw logits.** Not `log_softmax`. The guidance term adds
  `s · (log p(y'|G) − log p(∅|G))` to the reverse logits, and a second
  normalisation there changes the scale of the guidance.
* **Mean pooling, not sum.** With sum pooling a method can inflate a logit by
  adding nodes, and GDCE-I does change node types and edges freely.
* **Self-loops are normalised inside the model.** Any self-loop in the input is
  stripped first; the GCN variant then adds exactly one per node, the GINE
  variant adds none. Otherwise two datasets with different diagonal conventions
  give two different classifiers for the same graph.
* **Class indices are your CSV's `y` field.** The classifier is never trained on
  another classifier's predictions.

### 4.3 The control copy is not optional

NAFR, the metric the paper argues for, needs a second classifier that is
identical in architecture, recipe, data and splits, and differs **only in the
training seed**. A counterfactual that flips the attacked classifier but not
this one is an adversarial example, not an explanation. Any other difference
between the two — capacity, convolution type, training data — confounds that
reading, which is why `--seed 1 --tag _seed1` changes nothing else.

Report the agreement of the two on unmodified test graphs next to your NA
(step b above). Two classifiers that agree on only 80 % of untouched graphs make
an 80 % transfer rate meaningless.

---

## 5. Knobs worth turning, in this order

| knob | where | what it does |
|---|---|---|
| `guidance.s` | config or `++guidance.s=` | guidance scale. 3 everywhere in the paper, chosen once on Mutagenicity (Table 5) and transferred unchanged. `s = 1` barely flips, `s = 5` flips more and edits more. |
| the budget grid | `++guidance.t_step` / `t_max`, or `++guidance.t_list=[...]` | the τ values the search tries, in ascending order. Start coarse (`t_step=10`), then refine where your flips happen. TWITTER needed `1..10` singly because nearly every flip is below τ = 30. |
| `guidance.inversion_method` | config | `posterior` is GDCE-I. `fresh` reuses the forward noise naively, `noinv` draws new noise at every reverse step. Running all three is the ablation of Table 3 — and a good sanity check that the inversion is doing something on *your* data. |
| `guidance.n_runs` / `n_samples_per_run` | `++` override | 5 × 100 in the paper. Use `1 × 2` while you are still debugging. |
| `model.lambda_train` | config | `[5, 0]` weights the **edge** cross-entropy by 5 against the node term. Raise it if your graphs are sparse and the model under-predicts edges. |

Check that the inversion is exact before you trust anything else: with
`inversion_method: 'posterior'` and the *source* condition, the replay must
return your input graph bit for bit. `src/review_paired_tau.py` reports this as
the placebo column, and `guidance.recon_check: true` asserts it during the run.
If it fails, set `guidance.inversion_margin: 1e-3` — on some GPUs bf16 ties
break the argmax.

---

## 6. When it goes wrong

| symptom | cause |
|---|---|
| `ValueError: Unkown dataset` | the `process()` one-hot branch — place 2 in section 2 |
| `[MYDATA] WARNING: marginals not found on disk` | `node_type_marginals.npy` / `edge_type_marginals.npy` / `n_nodes_dist.npy` missing in `data/<name>/raw/` |
| `Environment variable 'GDCE_JUDGES' not found` | `. reproduce/common.sh` first, or export the variable yourself |
| `ModuleNotFoundError: No module named 'src'` inside a container | `PYTHONPATH` does not cross into Apptainer; `APPTAINERENV_PYTHONPATH` does. `common.sh` sets both. |
| `no kernel image is available` | the container's torch supports CUDA capabilities up to sm_86 — an A100 or a 3090, not a newer card |
| flip rate 1.000 at tiny budgets | run the `noinv` variant and the placebo. A search over many budgets is a lottery: without the inversion, every budget draws an independent graph, and one of them flips. Section 5.3 of the paper is about exactly this. |
| RDKit rejects your valid molecule | an aromatic ring truncated at the node cap cannot be sanitised. Benzene's SMILES ceiling is 89 %, not 100 %. |
| `Graph tool not found` at startup | expected; only the planar-graph metrics need it |
