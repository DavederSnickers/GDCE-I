#!/bin/bash
# C2Explainer with its node-feature perturbation matrix ENABLED (FPM=True).
#
# C2's __init__ defaults FPM to False, and the C2 authors disable it in their own
# experiments (paper Sec. 5.2) so every baseline shares an edge-only action space.
# That makes C2's Node Sparsity 0.000 by configuration rather than by capability.
# These runs give C2 the full action space its paper describes (Sec. 4.2.2), so the
# comparison against GDCE-I is on the method rather than on our settings.
#
# Everything else -- classifier (shared judge), split, seeds, epochs -- is unchanged.
set -u
# shellcheck disable=SC1090
source ~/.bashrc
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export APPTAINERENV_PYTHONPATH="$PWD"

J=${GDCE_JUDGES:?set GDCE_JUDGES}

# script                        config                            judge                       out
RUNS=(
    "run_c2_mutag_evaluation.py configs/mutag.cluster.yaml        judge_mutagenicity_mol_gcn  mutag_gcn"
    "run_c2_mutag_evaluation.py configs/benzene_mol.cluster.yaml  judge_benzene_gcn           benzene_gcn"
    "run_c2_nci1_evaluation.py  configs/nci1_freq.cluster.yaml    judge_nci1_gcn              nci1_gcn"
)

for entry in "${RUNS[@]}"; do
    # shellcheck disable=SC2086
    set -- $entry
    script="$1"; conf="$2"; judge="$3"; out="$4"
    mkdir -p "results/shared_judge_fpm/${out}"
    echo "=== ${script} ${conf} / ${judge} (FPM=True)"
    # leading `python`: this container execs its arguments
    qsub2d python "$script" --config "$conf" \
        classifier=shared_judge \
        "shared_judge_path=${J}/${judge}" \
        "out_dir=results/shared_judge_fpm/${out}" \
        fpm=true \
        n_runs=5 n_samples_per_run=100
done
echo "=== all three submitted"
