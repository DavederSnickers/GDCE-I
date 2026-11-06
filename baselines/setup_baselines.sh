#!/usr/bin/env bash
# Rebuild a working checkout of every baseline from its upstream repository.
#
#   baselines/setup_baselines.sh                 # all five
#   baselines/setup_baselines.sh UCExplainer     # just one
#   GDCE_THIRD_PARTY=/scratch/tp baselines/setup_baselines.sh
#
# For each baseline this clones the upstream repository, checks out the commit
# the paper used, applies baselines/<name>/changes.patch and copies
# baselines/<name>/adapter/ over the result.  No upstream code is stored in this
# repository: two of the five upstreams carry no licence, so they must not be
# redistributed, and doing it the same way for all five keeps the diff -- which
# is the honest statement of what the shared protocol changed -- visible.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
TP="${GDCE_THIRD_PARTY:-$ROOT/third_party}"

declare -A URL=(
    [C2Explainer]=https://github.com/kano5266/C2Explainer
    [CF2]=https://github.com/chrisjtan/gnn_cff
    [D4Explainer]=https://github.com/Graph-and-Geometric-Learning/D4Explainer
    [UCExplainer]=https://github.com/flaat/COMBINEX
    [XPlore]=https://github.com/MarioTheOne/GRETEL
)
declare -A COMMIT=(
    [C2Explainer]=86f3e8c
    [CF2]=9020e74
    [D4Explainer]=997b4c7
    [UCExplainer]=707aa65
    [XPlore]=fa60bd2
)

WANT=("$@")
[ ${#WANT[@]} -eq 0 ] && WANT=(C2Explainer CF2 D4Explainer UCExplainer XPlore)

mkdir -p "$TP"
for name in "${WANT[@]}"; do
    url="${URL[$name]:-}"
    [ -n "$url" ] || { echo "unknown baseline '$name'" >&2; exit 1; }
    dir="$TP/$name"
    echo
    echo "########## $name  ($url @ ${COMMIT[$name]})"
    cat "$HERE/$name/upstream.txt" | sed 's/^/    /'

    if [ ! -d "$dir/.git" ]; then
        git clone --quiet "$url" "$dir"
    fi
    git -C "$dir" fetch --quiet --all
    git -C "$dir" checkout --quiet "${COMMIT[$name]}"
    git -C "$dir" reset --quiet --hard "${COMMIT[$name]}"
    git -C "$dir" clean -qfd

    if [ -f "$HERE/$name/changes.patch" ]; then
        git -C "$dir" apply --verbose "$HERE/$name/changes.patch"
        echo "    patch applied"
    else
        echo "    no patch for this baseline"
    fi

    if [ -d "$HERE/$name/adapter" ]; then
        ( cd "$HERE/$name/adapter" && find . -type f -print0 ) \
          | ( cd "$HERE/$name/adapter" && xargs -0 -I{} sh -c 'mkdir -p "'"$dir"'/$(dirname {})" && cp -p "{}" "'"$dir"'/{}"' )
        echo "    $(find "$HERE/$name/adapter" -type f | wc -l) adapter files copied"
    fi

    echo "    -> $dir"
done

cat <<EOF

Next, per baseline:
  1. build its own environment.  Each upstream pins a different torch and PyG
     (CF2 and UCExplainer need torch 2.5 / PyG 2.6, C2Explainer torch 2.0 /
     PyG 2.3, D4Explainer its own).  That is exactly why the shared classifier
     travels as a .npz and not as a pickled torch module.
  2. point it at this repository's data and classifiers:
       export GDCE_DATA=$ROOT/data
       export GDCE_JUDGES=$ROOT/judges
  3. run its evaluation script (adapter/run_*_evaluation.py).  Each writes a
     cf_dump.jsonl.
  4. score that dump with THIS repository, so every method is measured by one
     implementation of the metrics:
       python src/shared_judge/adversarial_flip.py --dump <dump> \\
              --dataset mutagenicity_mol --data_root \$GDCE_DATA \\
              --judge_base \$GDCE_JUDGES/judge_mutagenicity_mol_gcn \\
              --judge_ctrl \$GDCE_JUDGES/judge_mutagenicity_mol_gcn_seed1 \\
              --label <method>/mutag_gcn --summary_csv runs/results/table2_advflip.csv
EOF
