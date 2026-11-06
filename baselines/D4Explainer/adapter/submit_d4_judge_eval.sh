#!/bin/bash
# D4Explainer evaluation against the shared GCN judges.
#
# Training saves two checkpoints under different criteria:
#   best_model_modrate_* (= best_model.pth, the script's default) -> lowest modification rate
#   best_model_flip_*                                             -> highest flip accuracy
# Both are evaluated rather than picking one silently: validity is the headline
# metric, but modrate is what the eval script defaults to.
set -u
# shellcheck disable=SC1090
source ~/.bashrc
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export APPTAINERENV_PYTHONPATH="$PWD"
export APPTAINERENV_PYTHONUNBUFFERED=1     # so progress is visible while running

J=${GDCE_JUDGES:?set GDCE_JUDGES}
R=${D4EXPLAINER_ROOT:?set D4EXPLAINER_ROOT}/results/shared_judge

# script                       dataset       judge                 ckpt-file
RUNS=(
  "run_d4_nci1_evaluation.py   gdce_nci1     judge_nci1_gcn        best_model_modrate_alpha_001.pt  modrate"
  "run_d4_nci1_evaluation.py   gdce_nci1     judge_nci1_gcn        best_model_flip_alpha_001.pt     flip"
  "run_d4_mutag_evaluation.py  gdce_benzene  judge_benzene_gcn     best_model_modrate_alpha_001.pt  modrate"
  "run_d4_mutag_evaluation.py  gdce_benzene  judge_benzene_gcn     best_model_flip_alpha_001.pt     flip"
)

for entry in "${RUNS[@]}"; do
  # shellcheck disable=SC2086
  set -- $entry
  script="$1"; ds="$2"; judge="$3"; ckpt="$4"; tag="$5"
  out="results/judge_eval/${ds}_${tag}"
  mkdir -p "$out"
  echo "=== ${ds} / ${tag} (${ckpt})"
  # no leading `python`: this container's %runscript is `exec python "$@"`
  qsub2d "$script" \
      --gnn_path "${J}/${judge}" \
      --d4_model "${R}/${ds}/${ds}/${ckpt}" \
      --dataset "$ds" \
      --output_dir "$out" \
      --n_runs 5 --n_samples_per_run 100
done
echo "=== all four submitted"
