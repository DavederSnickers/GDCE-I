# GDCE-I — Graph Diffusion Counterfactual Explanation via Inversion

Reference implementation of

> **Faithful, Sufficient and Understandable: Rethinking Graph Counterfactual
> Explanations via Discrete Diffusion Inversion**
> David Bechtoldt, Sidney Bender — Machine Learning Group, TU Berlin, and BIFOLD

GDCE-I explains a graph classifier by **editing** the input instead of masking
it. A conditional discrete diffusion model is run forward on the input graph for
a budget of τ steps, and the Gumbel noise that generated that trajectory is
recorded; replaying the reverse pass with the *same* noise under the original
condition returns the input exactly, and replaying it under the target class
returns a counterfactual that differs only where the change of condition moved
an argmax. A per-graph search over τ then returns the smallest budget that flips
the classifier.

Two consequences, and they are what the evaluation here is built around: the
edit space is the full discrete space — node types, edges and categorical edge
types (single / double / triple / aromatic bonds) — and the budget τ is a knob
that controls how much of the graph is regenerated.

```
src/diffusion/discrete_inversion.py    the inversion (naive and posterior Gumbel)
src/inversion_mutag_mol.py             the per-graph budget search
src/shared_judge/                      the shared-classifier protocol and the metrics
```

---

## 1. Install

Everything in the paper ran inside the Apptainer image built from `vm/counter.def`
(Python 3.9, PyTorch 2.0.1 + CUDA 11.8, PyG 2.3.1, PyTorch Lightning 2.0.4,
RDKit 2023.03.2), on a single GPU.

```bash
apptainer build vm/gdce.sif vm/counter.def
export GDCE_SIF=$PWD/vm/gdce.sif          # every reproduce/ command then runs inside it
```

Conda works too:

```bash
conda create -c conda-forge -n gdce python=3.9 rdkit=2023.03.2
conda activate gdce
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric==2.3.1 -r requirements.txt
pip install -e .                           # optional; the scripts set PYTHONPATH themselves
```

`graph_tool` is imported in a `try` block and only needed for the planar-graph
metrics, which this paper does not report — a `Graph tool not found` line at
startup is expected and harmless.

## 2. What this repository does and does not contain

It contains **code**, and the classifiers the method is evaluated against.
It contains no datasets, no trained diffusion models and no result files: every
number in the paper is recomputed from the raw data by the commands in
`reproduce/`.

In git:

| | |
|---|---|
| `src/`, `configs/`, `analysis/`, `reproduce/` | the method, every experiment's configuration, and the scripts that turn runs into the published tables |
| `judges/` | the twelve shared classifiers, 2.9 MB — the models the counterfactuals attack and are measured against |
| `judges_original/` | the baselines' own classifiers, for the control experiment of Section S3 |
| `data/<ds>/raw/*.npy`, `atom_*.json` | the node and edge marginals the noise model uses and the element decoder, a few hundred bytes each |
| `baselines/` | per baseline: the upstream commit, our patch against it, and our adapter |

Not here, and produced by running the pipeline:

| | produced by | cost |
|---|---|---|
| the four datasets, preprocessed and split | `reproduce/... data` | minutes, CPU |
| the six conditional diffusion models | `reproduce/... train`, see **`docs/training.md`** | 2–4 GPU days each |
| every per-run CSV behind Tables 2–5 and Fig. 3 | `reproduce/... table2 table3 table4 table5 fig3` | see the table below |

One consequence to know before you start: `SpectreGraphDataset.process()` draws
the 60/20/20 split with an **unseeded** permutation, so your split is not the
one the paper used. Section 7 says what that does and does not change.

## 3. Reproduce the paper

One script, read top to bottom, holds every command:

```bash
reproduce/reproduce_gdce_i_results.sh data        # build the four datasets, minutes, CPU
SMOKE=1 reproduce/reproduce_gdce_i_results.sh table2   # two graphs: checks the wiring
reproduce/reproduce_gdce_i_results.sh             # everything, from the raw data up
GDCE_DRYRUN=1 reproduce/reproduce_gdce_i_results.sh    # print the commands, run nothing
```

| paper | stage | what it runs | cost |
|---|---|---|---|
| the six diffusion models | `train` | `main.py` once per dataset and classifier — **not shipped, see `docs/training.md`** | 2–4 GPU days each |
| Table 2 — comparison on four datasets | `table2` | the budget search per dataset and classifier, then `adversarial_flip.py` on each dump | 4–12 GPU h per run |
| Table 2 — the four baselines | `baselines/setup_baselines.sh` | see `baselines/README.md` | — |
| Table 3 — noise treatment at the end of the search | `table3` | the same run with `guidance.inversion_method` = `noinv` / `fresh` / `posterior` | ~8 GPU h |
| Table 4 — paired replay at fixed τ ∈ {10,50,100,200} | `table4` | `review_paired_tau.py`, one job per run id, plus the placebo replay under the unchanged source condition | 5 × ~3 GPU h |
| Table 5 — guidance scale s ∈ {1,3,5} | `table5` | s = 1 and s = 5; s = 3 is the Table 2 run | ~8 GPU h |
| Fig. 3 — edit distance over the full budget grid | `fig3` | `review_paired_tau.py` with `+review.grid=paper` | 2 × ~5 GPU h |
| Table S1 — dataset statistics | `data` | printed by `src/build_dataset.py` | minutes, CPU |
| Section S2 — the shared classifiers | `judges` | `train_judge.py` + its seed-1 control copy | ~1 GPU h total |
| Section S3 — the baselines on their own classifiers | `configs/experiment/*_orig_*` | `judges_original/`, same runners | — |
| Section S4 — every hyperparameter | — | the configs are the source of truth | — |

Order: `data`, then `train` (see `docs/training.md`), then any of the table
stages, then `tables`. `judges` is the one stage you can skip — those weights
ship with the repository.

`tables` turns the CSVs your runs wrote into the published numbers. It takes the
directory to read, so it works on whatever you have:

```bash
python analysis/table3_search_smiles.py runs/table3
```


```
analysis/table3_search_smiles.py    Table 3: FR, NAFR, NA, NAFR & SMILES, median τ
analysis/fixedtau_table3_tex.py     Table 4: the LaTeX body
analysis/analyze_paired.py          Fig. 3 and the (i)-(iii) findings of Section 5.3
analysis/origclf_table*.py          Section S3
analysis/audit_tables.py            recomputes every table cell from the CSVs and
                                    diffs it against the numbers in the manuscript
```

## 4. Metrics

All of them are fractions of **all** inputs, not of the successful ones, and all
are oriented so that higher is better.

| metric | definition |
|---|---|
| FR | flip rate: the attacked classifier returns the target class on the discretised counterfactual |
| NAFR | flip **and** the control classifier flips with it. The control copy differs from the attacked one only in its training seed and was never optimised against, so a flip that does not transfer is an adversarial example rather than an explanation |
| NA | mean NAFR / mean FR — of the flips, which fraction transferred |
| NAFR & SMILES | additionally yields an RDKit-parsable molecule |
| Edge / Node / EdgeType sparsity | 1 − the corresponding modification rate, so 1 means that part of the graph was untouched |

The floor for NA is the agreement of the two classifiers on *unmodified* test
graphs — `src/judge_agreement.py` prints it, and the `judges` stage runs it.

## 5. Layout

```
src/                 the method, the datasets, the metrics, the shared-classifier protocol
configs/             Hydra configs: one experiment file per run in the paper
reproduce/           every command of the paper, in order
analysis/            run CSVs -> the numbers in the tables
judges/              the 12 shared classifiers (.npz + spec .json)
judges_original/     the baselines' own classifiers, for Section S3
baselines/           upstream commit + patch + adapters, no upstream code
data/                only the marginals and the element decoder; the datasets are built here
checkpoints/         empty: where the models you train land, see docs/training.md
runs/                empty: where Hydra writes every run and every result CSV
docs/                training.md, and the record of which files are and are not published
vm/                  the container recipe
```

Paths are environment variables with defaults inside the repository, so a fresh
clone runs unedited:

| variable | default | |
|---|---|---|
| `GDCE_ROOT` | the repository | |
| `GDCE_DATA` | `$GDCE_ROOT/data` | |
| `GDCE_JUDGES` | `$GDCE_ROOT/judges` | read by the configs via `${oc.env:...}` |
| `GDCE_JUDGES_ORIG` | `$GDCE_ROOT/judges_original` | |
| `GDCE_CKPT` | `$GDCE_ROOT/checkpoints` | |
| `GDCE_OUT` | `$GDCE_ROOT/runs` | Hydra's run directory |
| `GDCE_SIF` | unset | if set, every command runs inside that image |
| `GDCE_LAUNCHER` | `python` | your own prefix, e.g. `srun --gres=gpu:1 python` |

Weights & Biases is `disabled` in every config; set `general.wandb=online` to
turn it on.

## 6. Your own data or your own classifier

`TUTORIAL.md` walks through both: a new dataset in `src/datasets/spectre_dataset.py`
plus a config, and a new classifier exported to the portable `.npz` format. The
short version is that GDCE-I needs two things from you — a dataset whose graphs
fit the node cap, and a classifier that can be called on a dense
`(X, E)` pair — and nothing else about the method changes.

## 7. Licence and attribution

MIT, see `LICENSE`. This is a derivative of **DiGress** (Vignac et al., ICLR
2023, MIT) and follows **FreeGress** for classifier-free guidance on discrete
graphs; `NOTICE` says which files come from where and what is new here.

```bibtex
@article{bechtoldt2026gdcei,
  title   = {Faithful, Sufficient and Understandable: Rethinking Graph
             Counterfactual Explanations via Discrete Diffusion Inversion},
  author  = {Bechtoldt, David and Bender, Sidney},
  year    = {2026}
}
```
