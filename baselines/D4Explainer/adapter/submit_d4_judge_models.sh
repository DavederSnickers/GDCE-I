#!/bin/bash
# Submit the three D4Explainer models, one per dataset, all against the shared
# GCN judges.
#
# D4 is a binary-adjacency method — its generator works on the adjacency and only
# inherits/defaults bond types — so it belongs in the GCN column only. That gives
# three jobs, not five: mutagenicity, benzene and nci1, each against that
# dataset's shared GCN judge.
#
# Run from the D4Explainer repo root ON THE CLUSTER (qsub2d needs $PWD/vm/*.sif):
#     bash submit_d4_judge_models.sh

set -u

# shellcheck disable=SC1090
source ~/.bashrc

JUDGE_DIR=${GDCE_JUDGES:?set GDCE_JUDGES}

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export APPTAINERENV_PYTHONPATH="$PWD"

# dataset            judge basename
RUNS=(
    "gdce_mutag      judge_mutagenicity_mol_gcn"
    "gdce_benzene    judge_benzene_gcn"
    "gdce_nci1       judge_nci1_gcn"
)

for entry in "${RUNS[@]}"; do
    # shellcheck disable=SC2086
    set -- $entry
    ds="$1"; judge="$2"
    echo "=== submitting ${ds} against ${judge}"
    # --root keeps each run's results apart; hyperparameters come from
    # PAPER_HPARAMS in main.py and are left untouched.
    # NOTE: no leading `python`. This container's %runscript is `exec python "$@"`,
    # so qsub2d's `apptainer run <sif> "$@"` already invokes python — passing it
    # again yields `python python main.py`. (GDCE's container differs: it execs the
    # arguments directly and therefore does need the explicit `python`.)
    qsub2d main.py \
        --dataset "$ds" \
        --gnn_path "${JUDGE_DIR}/${judge}" \
        --root "results/shared_judge/${ds}/"
done

echo "=== all three submitted"
/opt/slurm/bin/squeue -u davidb -o '%.10i %.24j %.9T %.8M %R'
