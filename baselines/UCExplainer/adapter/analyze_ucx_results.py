"""Summarise a UCExplainer evaluation run — with the one-hot violation question first.

Usage:
    python analyze_ucx_results.py results/ucx/mutag [results/ucx/nci1 ...]

Reads inversion_results.csv from each directory and prints, per dataset:
  1. the one-hot violation rates (nodes and bonds) — is the bug still there?
  2. raw flip rate vs. flip rate after argmax projection back onto valid one-hot
  3. structural metrics (GED, sparsity) and, for molecules, SMILES validity
"""

from __future__ import annotations

import os
import sys

import pandas as pd


def _pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{100 * x:6.2f}%"


def _num(x: float, nd: int = 3) -> str:
    return "n/a" if pd.isna(x) else f"{x:.{nd}f}"


def summarise(path: str) -> None:
    csv = os.path.join(path, "inversion_results.csv")
    if not os.path.exists(csv):
        print(f"!! {csv} not found — skipping\n")
        return

    df = pd.read_csv(csv)
    name = os.path.basename(path.rstrip("/"))
    n = len(df)
    succ = df[df["success"]] if "success" in df.columns else df.iloc[0:0]

    print("=" * 78)
    print(f"{name}  —  {n} samples, {df['run'].nunique()} runs")
    print("=" * 78)

    print("\n[1] ONE-HOT VIOLATIONS  (the bug: several 1s, or none, in one vector)")
    node_v = df.get("onehot_violation_rate")
    edge_v = df.get("edge_onehot_violation_rate")
    if node_v is not None:
        share_any = (node_v.fillna(0) > 0).mean()
        print(f"    nodes: mean violated rows/graph  {_pct(node_v.mean())}"
              f"   graphs with >=1 violation  {_pct(share_any)}")
    if edge_v is not None and edge_v.notna().any():
        share_any_e = (edge_v.fillna(0) > 0).mean()
        print(f"    bonds: mean violated rows/graph  {_pct(edge_v.mean())}"
              f"   graphs with >=1 violation  {_pct(share_any_e)}")
    else:
        print("    bonds: no edge attributes in this dataset")
    if "structurally_valid" in df.columns:
        print(f"    counterfactuals that are fully valid one-hot: "
              f"{_pct(df['structurally_valid'].mean())}")

    print("\n[2] FLIP RATE — raw vs. projected back onto the one-hot manifold")
    if "success" in df.columns:
        print(f"    raw success (oracle flipped):        {_pct(df['success'].mean())}")
    if "projected_success" in df.columns:
        print(f"    projected success (after argmax):    {_pct(df['projected_success'].mean())}")
        gap = df["success"].mean() - df["projected_success"].mean()
        print(f"    gap (flips that live off-manifold):  {_pct(gap)}")

    print("\n[3] STRUCTURE  (successful counterfactuals only)")
    if len(succ):
        geds = succ[succ["ged"] >= 0]["ged"]
        print(f"    avg GED             {_num(geds.mean(), 2)}   "
              f"(GED==0 in {_pct((geds == 0).mean())} of them)")
        for col, label in (("edge_sparsity", "edge sparsity"),
                           ("node_sparsity", "node sparsity"),
                           ("edgetype_sparsity", "bond-type sparsity")):
            if col in succ.columns and succ[col].notna().any():
                print(f"    avg {label:<18} {_num(succ[col].mean())}")
        if "smiles_valid_strict" in succ.columns and succ["smiles_valid_strict"].notna().any():
            print(f"    SMILES valid (strict)  {_pct(pd.to_numeric(succ['smiles_valid_strict'], errors='coerce').mean())}")
    else:
        print("    no successful counterfactuals")

    print()


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        return 1
    for p in paths:
        summarise(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
