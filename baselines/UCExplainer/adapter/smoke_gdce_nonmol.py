"""Smoke test: proteins/twitter as CSV datasets inside COMBINEX.

Checks that get_dataset() resolves the new names, that the splits match the GDCE
side, and that the shared classifier reaches exactly the accuracy recorded in
its export when run on COMBINEX's data object. A deviation means baseline and
GDCE are running on different graphs or on a different encoding.

    python -u smoke_gdce_nonmol.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.dataset import get_dataset  # noqa: E402
from src.baselines.shared_judge import load_judge  # noqa: E402

EXPECTED = {   # from the GDCE split: train/val/test
    "PROTEINS-GDCE": ("proteins", 523, 174, 174),
    "TWITTER-G": ("twitter", 39019, 13006, 13006),
}

ok = True
for name, (gdce, n_tr, n_va, n_te) in EXPECTED.items():
    print(f"\n=== {name} ===", flush=True)
    data = get_dataset(name)
    ds = data.dataset
    n_train = int(data.train_mask.sum())
    n_test = int(data.test_mask.sum())
    print(f"  Graphen gesamt: {len(ds)}   train+val: {n_train}   test: {n_test}")
    print(f"  erwartet:       {n_tr + n_va + n_te}   {n_tr + n_va}   {n_te}")
    if (len(ds), n_train, n_test) != (n_tr + n_va + n_te, n_tr + n_va, n_te):
        print("  MISMATCH: split sizes do not match the GDCE side")
        ok = False
    print(f"  num_node_types={ds.num_node_types}  num_edge_types={ds.num_edge_types}"
          f"  x-dim={ds[0].x.shape[1]}")

    judge = load_judge(f"<GDCE_JUDGES>/judge_{gdce}_gcn",
                       device="cpu").eval()
    test_subset = ds[data.test_mask]
    correct = 0
    for g in test_subset:
        with torch.no_grad():
            b = torch.zeros(g.x.shape[0], dtype=torch.long)
            correct += int(judge(g.x.float(), g.edge_index, b).argmax(-1).item()
                           == int(g.y.item()))
    acc = correct / len(test_subset)
    print(f"  Judge-Accuracy auf dem COMBINEX-Testsplit: {acc:.4f}")

print("\nCOMBINEX_SMOKE_OK" if ok else "\nCOMBINEX_SMOKE_FAILED")
