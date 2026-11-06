"""Second-seed control judges for the AdversarialFlipRate metric.

Same architecture, same recipe, same data, same splits as the base judges in
train_final_judges.sh -- only `--seed 1` differs. That is the point: if a
counterfactual flips the base judge AND this one, the flip cannot be explained by
the particular weights the base judge happened to land on. Any other difference
(capacity, conv type, training data) would confound that reading.

Naming follows the proteins/twitter convention from train_shared_judges.py:
    judge_<dataset>_<conv>.npz        seed 0, the judge every method targets
    judge_<dataset>_<conv>_seed1.npz  seed 1, never optimised against

A .py driver rather than a shell script because COMBINEX' container runscript is
`exec python "$@"` -- `qsub2d bash <script>.sh` would try to run a file named
"bash" as Python. Submit from ~/COMBINEX:

    qsub2d shared_judge/train_seed1_judges.py
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("GDCE_DATA", "data")
OUT_DIR = os.environ.get("GDCE_JUDGES", "judges")

FLAGS = ["--batchnorm", "--cosine", "--dropout", "0.3", "--refit_train_val",
         "--epochs", "300", "--seed", "1", "--tag", "_seed1"]

RUNS = [
    ("mutagenicity_mol", "gcn"),
    ("mutagenicity_mol", "gine"),
    ("benzene", "gcn"),
    ("benzene", "gine"),
    ("nci1", "gcn"),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    failures = []
    for dataset, conv in RUNS:
        print(f"\n########## {dataset} / {conv} / seed 1", flush=True)
        cmd = [sys.executable, "-u", os.path.join(HERE, "train_judge.py"),
               "--dataset", dataset, "--conv", conv,
               "--data_root", DATA_ROOT, "--out_dir", OUT_DIR] + FLAGS
        rc = subprocess.call(cmd)
        print(f"########## rc={rc} {dataset}/{conv}", flush=True)
        if rc != 0:
            failures.append(f"{dataset}/{conv} (rc={rc})")

    print("\n########## SEED1 JUDGES DONE", flush=True)
    for f in sorted(os.listdir(OUT_DIR)):
        if f.endswith("_seed1.json"):
            print(f"  {f}")
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
