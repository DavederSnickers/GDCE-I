"""Phase-1 gate for D4Explainer: does D4 see the graphs the way the judges expect?

Runs each shared GCN judge over D4's own test split for the matching dataset and
compares to the accuracy recorded when the judge was trained. Catches vocabulary
mismatches (nci1 37 vs 10 atom types) and flipped class polarity before any
training starts — both would produce plausible but wrong numbers silently.

    python check_judge_compat.py --judge_dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnns.shared_judge_gnn import load_shared_judge_gnn  # noqa: E402

# D4 dataset -> shared judge basename. D4 is a binary-adjacency method, so it is
# only ever evaluated against the GCN judges.
PAIRS = {
    "gdce_mutag": "judge_mutagenicity_mol_gcn",
    "gdce_nci1": "judge_nci1_gcn",
    "gdce_benzene": "judge_benzene_gcn",
    "gdce_proteins": "judge_proteins_gcn",
    "gdce_twitter": "judge_twitter_gcn",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--judge_dir", required=True)
    p.add_argument("--batch_size", type=int, default=128)
    args = p.parse_args()

    from torch_geometric.loader import DataLoader
    from utils.dataset import get_datasets

    print(f"[d4-compat] torch {torch.__version__}, python {sys.version.split()[0]}",
          flush=True)
    failures = []

    for ds, judge_name in PAIRS.items():
        prefix = os.path.join(args.judge_dir, judge_name)
        with open(prefix + ".json") as f:
            spec = json.load(f)
        model = load_shared_judge_gnn(prefix, torch.device("cpu"))

        try:
            _, _, test_dataset = get_datasets(name=ds)
        except Exception as exc:
            print(f"  [FAIL] {ds}: cannot load dataset — {exc}", flush=True)
            failures.append(ds)
            continue

        loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        correct = total = 0
        x_dim = None
        with torch.no_grad():
            for batch in loader:
                x_dim = batch.x.shape[1]
                probs, _ = model.get_pred(batch.x, batch.edge_index, batch.batch)
                y = batch.y.view(-1)
                correct += int((probs.argmax(dim=-1) == y).sum().item())
                total += int(y.shape[0])

        acc = correct / max(total, 1)
        expected = spec["test_acc"]
        dim_ok = x_dim == spec["num_features"]
        ok = abs(acc - expected) < 0.03 and dim_ok
        print(f"  [{'OK  ' if ok else 'FAIL'}] {ds:<14} -> {judge_name:<28} "
              f"n={total:>5} acc {acc:.4f} (judge {expected:.4f}) "
              f"x_dim={x_dim}/{spec['num_features']}", flush=True)
        if not ok:
            failures.append(ds)
            if not dim_ok:
                print("         ^ vocabulary mismatch", flush=True)
            elif acc < 0.55:
                print("         ^ near chance — class polarity likely flipped",
                      flush=True)

    print()
    if failures:
        print(f"[d4-compat] FAILED: {failures}", flush=True)
        return 1
    print("[d4-compat] D4's data is compatible with the shared GCN judges.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
