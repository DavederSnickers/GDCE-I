# Baselines

The paper compares GDCE-I against four published explainers in Table 2 and
against a fifth in Section S3.  None of their code is copied into this
repository.  What is here, per baseline:

```
baselines/<name>/upstream.txt     repository URL, the commit the paper used, licence
baselines/<name>/changes.patch    our diff against that commit
baselines/<name>/adapter/         our own files: the evaluation runner, the dataset
                                  loader for our CSVs, a copy of the portable
                                  classifier, and the config of each run
```

`setup_baselines.sh` clones the upstream, checks out the commit, applies the
patch and copies the adapter over it.

The `adapter/submit_*.sh` and `adapter/run_*_job.sh` scripts are the SLURM /
Apptainer wrappers the runs were actually submitted with. They are site-specific
and are included as provenance, not as something to execute as is: paths that
pointed into our file system now read `${GDCE_JUDGES:?set GDCE_JUDGES}` and the
like, so they fail with the variable's name if you run one unconfigured. The
`run_*_evaluation.py` next to them is the part that matters.

| baseline | upstream | commit | licence | used for |
|---|---|---|---|---|
| C2Explainer | `kano5266/C2Explainer` | `86f3e8c` | MIT | Table 2, S3 |
| CF² | `chrisjtan/gnn_cff` | `9020e74` | **none** | Table 2, S3 |
| D4Explainer | `Graph-and-Geometric-Learning/D4Explainer` | `997b4c7` | MIT | Table 2, S3 |
| UCExplainer | `flaat/COMBINEX` | `707aa65` | MIT | Table 2, S3 |
| XPlore | on GRETEL, `MarioTheOne/GRETEL` | `fa60bd2` | GRETEL MIT, XPlore **none** | S3 |

Two upstreams carry no licence file, so redistributing them is not permitted.
Shipping a patch for all five instead of copying three and patching two also
makes the comparison auditable: the diff is precisely what had to change for the
method to run under the shared protocol, and nothing else.

XPlore additionally reached us as a plain directory rather than a git checkout,
so `baselines/XPlore/` carries the adapter and the GRETEL commit it runs on, but
no patch against XPlore itself. Because it arrived outside git, its algorithm
looked like one of our own untracked files — it is not, and it carries no
licence, so `adapter/src/wrappers/xplore/` ships only our `framework.py` and
`__init__.py` plus **`FETCH.md`**, which is the exact recipe for the four files
that are missing: where to get the authors' `XPlore.py`, the import header to
replace in it, the one-token change of the sign-fix variant, and which two
classes to take from GRETEL v2.

Where an adapter file is a *derivative* of upstream code rather than ours —
currently only `D4Explainer/adapter/main.py`, which extends upstream's MIT
`main.py` — the baseline directory carries an `adapter-provenance.txt` naming
it. The patch removes the upstream original in that case and the adapter
supplies the extended version, so `setup_baselines.sh` produces the right tree
either way.

## What the shared protocol asks of a baseline

Three things, and the adapter is the code that provides them:

1. **The same classifier.**  Every method is optimised against, and evaluated
   by, one classifier per (dataset, architecture) -- `judges/judge_<dataset>_<conv>`.
   It travels as a `.npz` plus a spec `.json`, with message passing implemented
   through `index_add_` and no PyG import, because the five baselines live in
   five mutually incompatible environments (python 3.9 to 3.11, torch 1.11 to
   2.5, PyG 2.3 to 2.6) and a pickled `torch.save` does not load across them.
   `src/shared_judge/shared_judge.py` is that file; each adapter carries a copy.

2. **The same inputs.**  The same five samples of 100 test graphs, drawn with
   seeds 42-46, stratified by the predicted class, without replacement.  The
   adapters reimplement that draw; `adapter/run_*_evaluation.py` is where.

3. **The same metrics.**  A method writes a `cf_dump.jsonl` -- the exact graph
   it produced, no SMILES round trip -- and this repository scores it.  FR, NAFR,
   NA and the sparsity columns therefore come from one implementation for all
   methods.  `src/shared_judge/cf_dump.py` is the format,
   `src/shared_judge/adversarial_flip.py` the scorer.

Section S3 of the supplement reports the same methods on *their own* classifiers
instead, with their own tuning, as a control for the shared protocol; the
classifiers for it are in `judges_original/`.
