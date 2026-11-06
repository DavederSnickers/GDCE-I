#!/bin/bash
# UCExplainer evaluation against the shared judges — five runs.
#
# UCExplainer predicts bond types, so it earns the GINE column alongside GDCE, and
# it also runs against the GCN judge so it appears in the one column every method
# shares. nci1 has no bond types, hence GCN only there.
#
# Run from the COMBINEX repo root ON THE CLUSTER:  bash submit_ucx_judge_eval.sh
#
# Container convention: combinex.sif has `%runscript exec python "$@"`, so the
# script name is passed WITHOUT a leading `python`.

set -u
# shellcheck disable=SC1090
source ~/.bashrc

J=${GDCE_JUDGES:?set GDCE_JUDGES}
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export APPTAINERENV_PYTHONPATH="$PWD"
export APPTAINERENV_WANDB_MODE=disabled
export APPTAINERENV_COMBINEX_GDCE_DATA_ROOT=${GDCE_DATA:?set GDCE_DATA}

# dataset      model  judge                          out
RUNS=(
    "MUTAG-G    gine   judge_mutagenicity_mol_gine    mutag_gine"
    "MUTAG-G    gcn    judge_mutagenicity_mol_gcn     mutag_gcn"
    "BENZENE-G  gine   judge_benzene_gine             benzene_gine"
    "BENZENE-G  gcn    judge_benzene_gcn              benzene_gcn"
    "NCI1-G     gcn    judge_nci1_gcn                 nci1_gcn"
)

for entry in "${RUNS[@]}"; do
    # shellcheck disable=SC2086
    set -- $entry
    ds="$1"; model="$2"; judge="$3"; out="$4"
    mkdir -p "results/shared_judge/${out}"
    echo "=== submitting ${ds} / ${judge}"
    qsub2d run_combinex_mutag_evaluation.py \
        --dataset_name "$ds" \
        --model "$model" \
        --oracle "${J}/${judge}" \
        --out_dir "results/shared_judge/${out}" \
        --num_runs 5 --num_samples 100 \
        --print_cf_features 5
done

echo "=== all five submitted"
/opt/slurm/bin/squeue -u davidb -o '%.10i %.24j %.9T %.8M %R'
