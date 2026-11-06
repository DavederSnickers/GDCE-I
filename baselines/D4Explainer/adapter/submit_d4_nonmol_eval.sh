#!/bin/bash
# D4Explainer evaluation on PROTEINS and TWITTER, with the graph-level cf_dump.
#
# Same protocol as submit_d4_cfdump_eval.sh: both checkpoint-selection criteria
# are evaluated, because they trade off against each other and the paper reports
# both rows.
#   best_model_modrate_* -> lowest modification rate (the eval script's default)
#   best_model_flip_*    -> highest flip accuracy
# alpha_cf is 0.01 for both datasets (NCI1 row of PAPER_HPARAMS) -> suffix alpha_001.
#
# Sample budget, identical for every method in the comparison:
#   PROTEINS: 1 run x 174 = the FULL test split. With only 174 test graphs, five
#             runs without replacement are five near-identical samples; the full
#             split says more. No run-to-run spread (std = 0).
#   TWITTER:  5 runs x 100 out of 13006 test graphs, with spread.
#
# Run from the D4Explainer repo root ON THE CLUSTER:  bash submit_d4_nonmol_eval.sh
set -u
# shellcheck disable=SC1090
source ~/.bashrc
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export APPTAINERENV_PYTHONPATH="$PWD"
export APPTAINERENV_PYTHONUNBUFFERED=1     # so progress is visible while running

J=${GDCE_JUDGES:?set GDCE_JUDGES}
R=${D4EXPLAINER_ROOT:?set D4EXPLAINER_ROOT}/results/shared_judge

# dataset         judge                 ckpt-file                        tag      data_path           runs samples
RUNS=(
  "gdce_proteins  judge_proteins_gcn    best_model_modrate_alpha_001.pt  modrate  data/GDCE_PROTEINS  1  174"
  "gdce_proteins  judge_proteins_gcn    best_model_flip_alpha_001.pt     flip     data/GDCE_PROTEINS  1  174"
  "gdce_twitter   judge_twitter_gcn     best_model_modrate_alpha_001.pt  modrate  data/GDCE_TWITTER   5  100"
  "gdce_twitter   judge_twitter_gcn     best_model_flip_alpha_001.pt     flip     data/GDCE_TWITTER   5  100"
)

# The two generators do not finish at the same time, hence the filter:
#   ONLY=gdce_proteins bash submit_d4_nonmol_eval.sh
ONLY=${ONLY:-}

for entry in "${RUNS[@]}"; do
  # shellcheck disable=SC2086
  set -- $entry
  ds="$1"; judge="$2"; ckpt="$3"; tag="$4"; dpath="$5"; n_runs="$6"; n_samples="$7"
  if [ -n "$ONLY" ] && [ "$ds" != "$ONLY" ]; then
    continue
  fi
  if [ ! -f "${R}/${ds}/${ds}/${ckpt}" ]; then
    echo "=== SKIP ${ds} / ${tag}: kein Checkpoint unter ${R}/${ds}/${ds}/${ckpt}"
    continue
  fi
  out="results/judge_eval_v2/${ds}_${tag}"
  mkdir -p "$out"
  echo "=== ${ds} / ${tag} (${ckpt})  (${n_runs} x ${n_samples})"
  # no leading `python`: this container's %runscript is `exec python "$@"`
  qsub2d run_d4_nci1_evaluation.py \
      --gnn_path "${J}/${judge}" \
      --d4_model "${R}/${ds}/${ds}/${ckpt}" \
      --dataset "$ds" \
      --data_path "$dpath" \
      --output_dir "$out" \
      --n_runs "$n_runs" --n_samples_per_run "$n_samples"
done
echo "=== all four submitted"
/opt/slurm/bin/squeue -u davidb -o '%.10i %.24j %.9T %.8M %R'
