"""Greedy search over the learning rate for C2Explainer on NCI1 (no-freq).

Runs `run_c2_nci1_evaluation.py` once per learning rate on a small sample
(default n_runs=1, n_samples_per_run=20), collects each run's run_summary.csv,
and ranks the learning rates.

Only the learning rate is swept (NCI1 has no `beta` knob, unlike Mutag).

Ranking rule ("validity first, then minimal"):
  1. keep lrs whose Validity (accuracy) is within --validity-tol of the best
  2. among those, pick the one with the lowest GED (tie-break: EdgeSparsity)
This mirrors the paper's stance that Validity is the most critical measure,
and only then do #perturbs / similarity matter.

Add-mode is kept (subgraph_mode stays false via the configs). Everything is
measured GDCE-style (undirected) by the underlying eval script.

Usage (inside the container, so the conda env / GDCE paths resolve):
    apprun python grid_search_nci1.py
    apprun python grid_search_nci1.py --lrs 0.003 0.01 0.03 0.1 0.3
    apprun python grid_search_nci1.py --n-samples 20 --epochs 1000
    apprun python grid_search_nci1.py --dry-run        # just print the commands

The eval subprocesses are launched with the SAME interpreter running this
script (sys.executable), so run this script itself via `apprun python ...`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
EVAL_SCRIPT = REPO_ROOT / 'run_c2_nci1_evaluation.py'

# run_summary.csv columns we care about (written by run_c2_nci1_evaluation.py)
METRIC_COLS = [
    'accuracy',                  # = Validity
    'avg_ged',
    'avg_mod_rate',
    'avg_n_perturbs',
    'avg_prop_perturbs',
    'avg_node_sparsity',
    'avg_edge_sparsity',
    'avg_time_s',
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lrs', type=float, nargs='+',
                   default=[0.001, 0.002, 0.003, 0.004, 0.005],
                   help='learning rates to try')
    p.add_argument('--config', default='configs/nci1.cluster.yaml',
                   help='base override config passed to the eval script '
                        '(use configs/nci1.yaml for host paths)')
    p.add_argument('--n-runs', type=int, default=1)
    p.add_argument('--n-samples', type=int, default=20)
    p.add_argument('--epochs', type=int, default=None,
                   help='override epochs (default: whatever the config says)')
    p.add_argument('--out-root', default='sweep/nci1_lr',
                   help='directory under which each lr gets its own subdir')
    p.add_argument('--validity-tol', type=float, default=0.05,
                   help='keep lrs within this Validity gap of the best, '
                        'then pick the most minimal among them')
    p.add_argument('--python', default=sys.executable,
                   help='interpreter used to launch the eval subprocesses')
    p.add_argument('--dry-run', action='store_true',
                   help='print the commands without running them')
    return p.parse_args()


def build_cmd(args: argparse.Namespace, lr: float, out_dir: Path) -> list[str]:
    cmd = [args.python, str(EVAL_SCRIPT),
           '--config', args.config,
           f'n_runs={args.n_runs}',
           f'n_samples_per_run={args.n_samples}',
           f'lr={lr}',
           f'out_dir={out_dir}']
    if args.epochs is not None:
        cmd.append(f'epochs={args.epochs}')
    return cmd


def read_summary(out_dir: Path) -> dict | None:
    csv = out_dir / 'run_summary.csv'
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    if df.empty:
        return None
    # mean across runs (n_runs is small; usually 1)
    return {c: float(df[c].mean()) for c in METRIC_COLS if c in df.columns}


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    log_dir = out_root / 'logs'
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Greedy LR search: {len(args.lrs)} lrs "
          f"(n_samples={args.n_samples}, n_runs={args.n_runs}, config={args.config})")
    print(f"Output root: {out_root.resolve()}\n")

    rows = []
    for i, lr in enumerate(args.lrs, 1):
        tag = f"lr{lr:g}"
        out_dir = out_root / tag
        cmd = build_cmd(args, lr, out_dir)

        if args.dry_run:
            print(' '.join(cmd))
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{tag}.log"
        print(f"[{i}/{len(args.lrs)}] {tag} ... ", end='', flush=True)
        t0 = time.time()
        with open(log_path, 'w') as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                  cwd=str(REPO_ROOT))
        dt = time.time() - t0

        if proc.returncode != 0:
            print(f"FAILED (rc={proc.returncode}, see {log_path}) [{dt:.0f}s]")
            rows.append({'tag': tag, 'lr': lr, 'ok': False})
            continue

        summary = read_summary(out_dir)
        if summary is None:
            print(f"no run_summary.csv (see {log_path}) [{dt:.0f}s]")
            rows.append({'tag': tag, 'lr': lr, 'ok': False})
            continue

        row = {'tag': tag, 'lr': lr, 'ok': True, **summary}
        rows.append(row)
        print(f"Validity={summary.get('accuracy', float('nan')):.3f} "
              f"GED={summary.get('avg_ged', float('nan')):.2f} "
              f"EdgeSp={summary.get('avg_edge_sparsity', float('nan')):.3f} "
              f"[{dt:.0f}s]")

    if args.dry_run:
        return

    df = pd.DataFrame(rows)
    grid_csv = out_root / 'grid_results.csv'
    df.to_csv(grid_csv, index=False)
    print(f"\nFull results written to: {grid_csv.resolve()}")

    ok = df[df.get('ok', False) == True].copy()  # noqa: E712
    if ok.empty:
        print("No successful runs to rank.")
        return

    # ---- ranked table (Validity desc, then GED asc) ----
    ranked = ok.sort_values(['accuracy', 'avg_ged'], ascending=[False, True])
    show = ['tag', 'lr', 'accuracy', 'avg_ged', 'avg_mod_rate',
            'avg_n_perturbs', 'avg_edge_sparsity', 'avg_time_s']
    show = [c for c in show if c in ranked.columns]
    print("\n" + "=" * 78)
    print("All learning rates (sorted by Validity desc, then GED asc):")
    print("=" * 78)
    with pd.option_context('display.width', 200,
                           'display.max_columns', None,
                           'display.float_format', lambda v: f"{v:.3f}"):
        print(ranked[show].to_string(index=False))

    # ---- recommended: validity within tol of best, then most minimal ----
    best_val = ok['accuracy'].max()
    floor = best_val - args.validity_tol
    cand = ok[ok['accuracy'] >= floor]
    rec = cand.sort_values(['avg_ged', 'avg_edge_sparsity']).iloc[0]

    print("\n" + "=" * 78)
    print(f"RECOMMENDED (Validity >= {floor:.3f} = best {best_val:.3f} - tol "
          f"{args.validity_tol}, then min GED):")
    print("=" * 78)
    print(f"  lr   = {rec['lr']:g}")
    print(f"  Validity      = {rec['accuracy']:.3f}")
    print(f"  GED           = {rec['avg_ged']:.2f}")
    print(f"  ModRate       = {rec['avg_mod_rate']:.3f}")
    print(f"  EdgeSparsity  = {rec['avg_edge_sparsity']:.3f}")
    print("\nReproduce this config with:")
    print(f"  apprun python {EVAL_SCRIPT.name} --config {args.config} "
          f"lr={rec['lr']:g} n_runs=5 n_samples_per_run=100 out_dir=c2_nci1_best")


if __name__ == '__main__':
    main()
