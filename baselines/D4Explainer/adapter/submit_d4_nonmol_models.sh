#!/bin/bash
# Train the two missing D4Explainer generators: PROTEINS and TWITTER.
#
# Both datasets are binary-adjacency graph classification with no bond types, so
# D4 belongs in the GCN column here exactly as it does for NCI1 -- one model per
# dataset, trained against that dataset's shared judge (seed 0).
#
# The CSVs must be in place first (raw/{train,val,test}.csv per dataset):
#     bash prepare_d4_nonmol_data.sh
#
# Run from the D4Explainer repo root ON THE CLUSTER:
#     bash submit_d4_nonmol_models.sh

set -u

# shellcheck disable=SC1090
source ~/.bashrc

JUDGE_DIR=${GDCE_JUDGES:?set GDCE_JUDGES}

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export APPTAINERENV_PYTHONPATH="$PWD"
# Without this, python buffers stdout inside the container and the job log sits
# still for hours -- training progress is then only readable from the mtimes of
# the checkpoints, which carry no epoch number.
export APPTAINERENV_PYTHONUNBUFFERED=1

# dataset            judge basename
RUNS=(
    "gdce_proteins   judge_proteins_gcn"
    "gdce_twitter    judge_twitter_gcn"
)

for entry in "${RUNS[@]}"; do
    # shellcheck disable=SC2086
    set -- $entry
    ds="$1"; judge="$2"
    echo "=== submitting ${ds} against ${judge}"
    # Hyperparameters come from PAPER_HPARAMS in main.py (NCI1 row; TWITTER on a
    # shorter schedule because of its 39k training graphs).
    # NOTE: no leading `python` -- this container's %runscript is `exec python "$@"`.
    qsub2d main.py \
        --dataset "$ds" \
        --gnn_path "${JUDGE_DIR}/${judge}" \
        --root "results/shared_judge/${ds}/"
done

echo "=== both submitted"
/opt/slurm/bin/squeue -u davidb -o '%.10i %.24j %.9T %.8M %R'
