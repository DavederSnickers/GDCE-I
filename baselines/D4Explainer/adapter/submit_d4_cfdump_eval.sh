#!/bin/bash
# D4Explainer re-evaluation WITH the graph-level counterfactual dump.
#
# Same protocol as submit_d4_judge_eval.sh, extended by the two mutagenicity runs
# (they were run manually last time) and writing to judge_eval_v2/ so the existing
# results stay untouched.
#
# Both checkpoint-selection criteria are evaluated, as before:
#   best_model_modrate_* -> lowest modification rate (the eval script's default)
#   best_model_flip_*    -> highest flip accuracy
# Note the alpha suffix differs per dataset: mutagenicity trained with alpha_0001,
# benzene and nci1 with alpha_001.
set -u
# shellcheck disable=SC1090
source ~/.bashrc
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export APPTAINERENV_PYTHONPATH="$PWD"
export APPTAINERENV_PYTHONUNBUFFERED=1     # so progress is visible while running

J=${GDCE_JUDGES:?set GDCE_JUDGES}
R=${D4EXPLAINER_ROOT:?set D4EXPLAINER_ROOT}/results/shared_judge

# --data_path is passed explicitly for every run: the mutagenicity script defaults
# to data/MUTAG, which is the wrong corpus for the gdce_* variants.
#
# script                       dataset       judge                        ckpt-file                         tag      data_path
RUNS=(
  "run_d4_mutag_evaluation.py  gdce_mutag    judge_mutagenicity_mol_gcn   best_model_modrate_alpha_0001.pt  modrate  data/GDCE_MUTAG"
  "run_d4_mutag_evaluation.py  gdce_mutag    judge_mutagenicity_mol_gcn   best_model_flip_alpha_0001.pt     flip     data/GDCE_MUTAG"
  "run_d4_mutag_evaluation.py  gdce_benzene  judge_benzene_gcn            best_model_modrate_alpha_001.pt   modrate  data/GDCE_BENZENE"
  "run_d4_mutag_evaluation.py  gdce_benzene  judge_benzene_gcn            best_model_flip_alpha_001.pt      flip     data/GDCE_BENZENE"
  "run_d4_nci1_evaluation.py   gdce_nci1     judge_nci1_gcn               best_model_modrate_alpha_001.pt   modrate  data/GDCE_NCI1"
  "run_d4_nci1_evaluation.py   gdce_nci1     judge_nci1_gcn               best_model_flip_alpha_001.pt      flip     data/GDCE_NCI1"
)

for entry in "${RUNS[@]}"; do
  # shellcheck disable=SC2086
  set -- $entry
  script="$1"; ds="$2"; judge="$3"; ckpt="$4"; tag="$5"; dpath="$6"
  out="results/judge_eval_v2/${ds}_${tag}"
  mkdir -p "$out"
  echo "=== ${ds} / ${tag} (${ckpt})"
  # no leading `python`: this container's %runscript is `exec python "$@"`
  qsub2d "$script" \
      --gnn_path "${J}/${judge}" \
      --d4_model "${R}/${ds}/${ds}/${ckpt}" \
      --dataset "$ds" \
      --data_path "$dpath" \
      --output_dir "$out" \
      --n_runs 5 --n_samples_per_run 100
done
echo "=== all six submitted"
/opt/slurm/bin/squeue -u davidb -o '%.10i %.24j %.9T %.8M %R'
