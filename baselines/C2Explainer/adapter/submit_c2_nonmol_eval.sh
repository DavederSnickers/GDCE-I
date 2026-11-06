#!/bin/bash
# C2Explainer on the non-molecular shared-classifier datasets (proteins, twitter).
#
# C2 changes only the edge set, so it belongs in the GCN column -- one run per
# dataset.
#
# Sample size -- deliberately different per dataset, and identical for every
# method:
#   PROTEINS: 1 run x 174 = the COMPLETE test split. With only 174 test graphs,
#             5 draws without replacement are five times almost the same sample;
#             the full split says more. No run-to-run spread (std = 0).
#   TWITTER:  5 runs x 100 out of 13006 test graphs, with spread.
# The first runs used 1 x 20; that left 2-8 counterfactuals and NAFR had no
# spread left at all (transfer_std NaN).
#
# From the C2Explainer repository root:  bash submit_c2_nonmol_eval.sh
#
# Container convention: this image execs its arguments, so the call keeps the
# leading `python` (unlike COMBINEX/CF2/D4).

set -u
# shellcheck disable=SC1090
source ~/.bashrc

J=${GDCE_JUDGES:?set GDCE_JUDGES}
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export APPTAINERENV_PYTHONPATH="$PWD"

# Its own directory, so that the earlier 1x20 results are preserved.
OUT_ROOT=${OUT_ROOT:-results/shared_judge_nonmol_full}

# config                        judge                 out           runs samples
RUNS=(
    "configs/proteins.cluster.yaml  judge_proteins_gcn  proteins_gcn  1  174"
    "configs/twitter.cluster.yaml   judge_twitter_gcn   twitter_gcn   5  100"
)

for entry in "${RUNS[@]}"; do
    # shellcheck disable=SC2086
    set -- $entry
    conf="$1"; judge="$2"; out="$3"; n_runs="$4"; n_samples="$5"
    mkdir -p "${OUT_ROOT}/${out}"
    echo "=== submitting ${conf} / ${judge}  (${n_runs} x ${n_samples})"
    qsub2d python run_c2_nci1_evaluation.py --config "$conf" \
        classifier=shared_judge \
        "shared_judge_path=${J}/${judge}" \
        "out_dir=${OUT_ROOT}/${out}" \
        n_runs="$n_runs" n_samples_per_run="$n_samples"
done

echo "=== beide abgeschickt"
