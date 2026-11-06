#!/bin/bash
# UCExplainer/COMBINEX on the non-molecular shared-classifier datasets.
#
# proteins and twitter carry no bond types (num_edge_types == 0), so there is
# only the GCN column here. The flip target is the base classifier (seed 0); the
# transfer rate is computed afterwards from the cf_dump against the control copy
# (seed 1), see adversarial_flip.py.
#
# From the COMBINEX repository root:  bash submit_ucx_nonmol_eval.sh
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

# Sample size -- deliberately different per dataset, and identical for every
# method:
#   PROTEINS: 1 run x 174 = the COMPLETE test split. With only 174 test graphs,
#             5 draws without replacement are five times almost the same sample;
#             the full split says more. No run-to-run spread (std = 0).
#   TWITTER:  5 runs x 100 out of 13006 test graphs, with spread.
# For COMBINEX the fidelity gate matters on top: of 20 raw flips only 3 survived
# on proteins and 9 on twitter after projection, because the flip usually does
# not survive the argmax. Without a larger sample, NAFR is not interpretable
# there.
# Its own directory, so that the earlier 1x20 results are preserved.
OUT_ROOT=${OUT_ROOT:-results/shared_judge_nonmol_full}

# dataset        judge                   out           runs samples
RUNS=(
    "PROTEINS-GDCE  judge_proteins_gcn      proteins_gcn  1  174"
    "TWITTER-G      judge_twitter_gcn       twitter_gcn   5  100"
)

for entry in "${RUNS[@]}"; do
    # shellcheck disable=SC2086
    set -- $entry
    ds="$1"; judge="$2"; out="$3"; n_runs="$4"; n_samples="$5"
    mkdir -p "${OUT_ROOT}/${out}"
    echo "=== submitting ${ds} / ${judge}  (${n_runs} x ${n_samples})"
    qsub2d run_combinex_nci1_evaluation.py \
        --dataset_name "$ds" \
        --model gcn \
        --oracle "${J}/${judge}" \
        --out_dir "${OUT_ROOT}/${out}" \
        --num_runs "$n_runs" --num_samples "$n_samples"
done

echo "=== beide abgeschickt"
/opt/slurm/bin/squeue -u davidb -o '%.10i %.24j %.9T %.8M %R'
