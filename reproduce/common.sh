#!/usr/bin/env bash
# Shared setup for the reproduction scripts.  Sourced, not executed.
#
# Every path the code needs is an environment variable with a default inside the
# repository, so a fresh clone runs without editing a single file:
#
#   GDCE_ROOT         the repository              (derived from this file)
#   GDCE_DATA         datasets                    $GDCE_ROOT/data
#   GDCE_JUDGES       shared classifiers          $GDCE_ROOT/judges
#   GDCE_JUDGES_ORIG  the baselines' classifiers  $GDCE_ROOT/judges_original
#   GDCE_CKPT         diffusion checkpoints       $GDCE_ROOT/checkpoints
#   GDCE_OUT          run outputs                 $GDCE_ROOT/runs
#
# How python is started is one more variable:
#
#   GDCE_SIF=vm/gdce.sif      run everything inside that Apptainer image
#   GDCE_LAUNCHER="srun ..."  put your own prefix in front of every command
#   GDCE_DRYRUN=1             print the commands instead of running them
#   SMOKE=1                   two graphs and one run per command, for a wiring test
#
# The configs read GDCE_JUDGES / GDCE_CKPT / GDCE_DATA through OmegaConf's
# ${oc.env:...}, so exporting them here is what makes `+experiment=...` work.

set -euo pipefail

GDCE_ROOT="${GDCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export GDCE_ROOT
export GDCE_DATA="${GDCE_DATA:-$GDCE_ROOT/data}"
export GDCE_JUDGES="${GDCE_JUDGES:-$GDCE_ROOT/judges}"
export GDCE_JUDGES_ORIG="${GDCE_JUDGES_ORIG:-$GDCE_ROOT/judges_original}"
export GDCE_CKPT="${GDCE_CKPT:-$GDCE_ROOT/checkpoints}"
export GDCE_OUT="${GDCE_OUT:-$GDCE_ROOT/runs}"

# src/ imports both `src.x` and bare `x`, so both directories go on the path.
export PYTHONPATH="$GDCE_ROOT:$GDCE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Apptainer does not inherit PYTHONPATH; this is the variable that gets it in.
export APPTAINERENV_PYTHONPATH="$GDCE_ROOT:$GDCE_ROOT/src"
export APPTAINERENV_GDCE_DATA="$GDCE_DATA"
export APPTAINERENV_GDCE_JUDGES="$GDCE_JUDGES"
export APPTAINERENV_GDCE_JUDGES_ORIG="$GDCE_JUDGES_ORIG"
export APPTAINERENV_GDCE_CKPT="$GDCE_CKPT"
export APPTAINERENV_GDCE_OUT="$GDCE_OUT"
# RDKit and the SMILES round trip are not thread-scaled; oversubscription only
# costs wall time here.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

PYBIN="${PYBIN:-python}"
if [ -n "${GDCE_SIF:-}" ]; then
    GDCE_LAUNCHER="${GDCE_LAUNCHER:-apptainer run --nv $GDCE_SIF $PYBIN}"
else
    GDCE_LAUNCHER="${GDCE_LAUNCHER:-$PYBIN}"
fi

# Smoke mode: the same commands on two graphs, to check the wiring before
# spending GPU hours.  Numbers produced in this mode are meaningless.
if [ "${SMOKE:-0}" = "1" ]; then
    SMOKE_SEARCH="guidance.n_runs=1 guidance.n_samples_per_run=2"
    SMOKE_PAIRED="+review.n_runs=1 +review.limit=2"
    echo "### SMOKE=1 -- two graphs per command, results are NOT the paper's numbers"
else
    SMOKE_SEARCH=""
    SMOKE_PAIRED=""
fi

_t0=$(date +%s)

# run <command...>   -- echo it, then run it (or only echo, with GDCE_DRYRUN=1)
run() {
    printf '\n\033[1m$ %s\033[0m\n' "$*"
    if [ "${GDCE_DRYRUN:-0}" = "1" ]; then return 0; fi
    local s=$(date +%s)
    "$@"
    printf '  [%s s]\n' "$(( $(date +%s) - s ))"
}

# pyrun <script> <args...>  -- run a python entry point through the launcher
pyrun() {
    # shellcheck disable=SC2086
    run $GDCE_LAUNCHER "$@"
}

stage() { printf '\n\n\033[1;4m%s\033[0m\n' "=== $* ==="; }

need_file() {
    if [ ! -e "$1" ]; then
        echo "MISSING: $1" >&2
        echo "  $2" >&2
        return 1
    fi
}

summary() { printf '\n\033[1mtotal wall time: %s s\033[0m\n' "$(( $(date +%s) - _t0 ))"; }
