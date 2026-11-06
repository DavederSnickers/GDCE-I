#!/bin/bash
# Control run: C2Explainer against its OWN PaperGCN oracle, with the CURRENT code.
#
# Purpose: separate two possible causes for the shift vs. the paper numbers
# (validity 0.984 -> 0.688, EdgeSparsity 0.084 -> 1.390).
#   * If this reproduces ~0.984 / ~0.084, the integration is sound and the whole
#     difference comes from the shared judge.
#   * If it does not, something in the current code changed the method itself.
#
# Everything except the classifier is identical to the shared-judge run.
set -u
# shellcheck disable=SC1090
source ~/.bashrc
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export APPTAINERENV_PYTHONPATH="$PWD"

mkdir -p results/control_ownoracle/mutag
echo "=== C2 control: own PaperGCN oracle, mutagenicity"
qsub2d python run_c2_mutag_evaluation.py --config configs/mutag.cluster.yaml \
    out_dir=results/control_ownoracle/mutag \
    n_runs=5 n_samples_per_run=100

/opt/slurm/bin/squeue -u davidb -o '%.10i %.24j %.9T %.8M %R'
