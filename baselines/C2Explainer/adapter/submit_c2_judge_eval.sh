#!/bin/bash
# C2Explainer evaluation against the shared GCN judges — three runs.
#
# C2 modifies the edge SET only and inherits/defaults bond types, so it belongs in
# the GCN column: one run per dataset.
#
# Run from the C2Explainer repo root ON THE CLUSTER:  bash submit_c2_judge_eval.sh
#
# Container convention: this image execs its arguments, so the call keeps the
# leading `python` (unlike COMBINEX/CF2/D4, whose runscripts already invoke it).

set -u
# shellcheck disable=SC1090
source ~/.bashrc

J=${GDCE_JUDGES:?set GDCE_JUDGES}
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export APPTAINERENV_PYTHONPATH="$PWD"

# script                        config                            judge                        out
RUNS=(
    "run_c2_mutag_evaluation.py configs/mutag.cluster.yaml        judge_mutagenicity_mol_gcn   mutag_gcn"
    "run_c2_mutag_evaluation.py configs/benzene_mol.cluster.yaml  judge_benzene_gcn            benzene_gcn"
    "run_c2_nci1_evaluation.py  configs/nci1_freq.cluster.yaml    judge_nci1_gcn               nci1_gcn"
)

for entry in "${RUNS[@]}"; do
    # shellcheck disable=SC2086
    set -- $entry
    script="$1"; conf="$2"; judge="$3"; out="$4"
    mkdir -p "results/shared_judge/${out}"
    echo "=== submitting ${script} ${conf} / ${judge}"
    qsub2d python "$script" --config "$conf" \
        classifier=shared_judge \
        "shared_judge_path=${J}/${judge}" \
        "out_dir=results/shared_judge/${out}" \
        n_runs=5 n_samples_per_run=100
done

echo "=== all three submitted"
/opt/slurm/bin/squeue -u davidb -o '%.10i %.24j %.9T %.8M %R'
