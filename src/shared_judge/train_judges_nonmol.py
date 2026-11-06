"""Train and export the shared classifiers for PROTEINS and TWITTER.

Trains two GCN classifiers per dataset in the SharedJudge architecture, which
every baseline can consume (GCN, hidden 128, 3 layers, BatchNorm, mean pooling).
Both read the same CSVs as GDCE-I itself -- that is what makes the comparison a
shared-classifier comparison rather than a comparison of method-oracle pairs.

  seed 0 -> judge_<ds>_gcn         the classifier every method is to flip
  seed 1 -> judge_<ds>_gcn_seed1   the control copy, for NAFR

Usage:
    python src/shared_judge/train_judges_nonmol.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_judge import (  # noqa: E402
    NUM_EDGE_TYPES, SharedJudge, collate, load_split, save_judge,
)

GDCE_ROOT = os.environ.get("GDCE_DATA", "data")
OUT_DIR = os.environ.get("GDCE_JUDGES", "judges")

# Binaere Adjazenz ohne Kantentypen -- wie nci1.
NUM_EDGE_TYPES.setdefault("proteins", 0)
NUM_EDGE_TYPES.setdefault("twitter", 0)

# For reference: published GNN results on PROTEINS typically sit around 0.75.
SPECS = [
    dict(ds="proteins", epochs=300, lr=1e-3, wd=1e-2, dropout=0.3, batch=64,
         hidden=64, num_layers=2),
    dict(ds="twitter",  epochs=60,  lr=1e-3, wd=1e-3, dropout=0.3, batch=128),
]


def evaluate(model, graphs, device, batch=256):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(graphs), batch):
            x, ei, bt, _, y = collate(graphs[i:i + batch], device=device)
            logits = model(x, ei, bt)
            correct += int((logits.argmax(-1) == y).sum())
            total += len(y)
    return correct / max(total, 1)


def train_one(ds, seed, epochs, lr, wd, dropout, batch, device,
              hidden=128, num_layers=3):
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr, num_features, _ = load_split(GDCE_ROOT, ds, "train")
    va, _, _ = load_split(GDCE_ROOT, ds, "val")
    te, _, _ = load_split(GDCE_ROOT, ds, "test")

    model = SharedJudge(num_features=num_features, num_classes=2, hidden=hidden,
                        num_layers=num_layers, conv="gcn", dropout=dropout,
                        batchnorm=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_va, best_state, best_ep = -1.0, None, -1
    order = np.arange(len(tr))
    for ep in range(epochs):
        model.train()
        np.random.shuffle(order)
        for i in range(0, len(order), batch):
            x, ei, bt, _, y = collate([tr[j] for j in order[i:i + batch]], device=device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x, ei, bt), y)
            loss.backward()
            opt.step()
        acc_va = evaluate(model, va, device)
        if acc_va > best_va:
            best_va, best_ep = acc_va, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)          # best validation epoch, not the last
    acc_tr = evaluate(model, tr, device)
    acc_te = evaluate(model, te, device)
    print(f"  {ds} seed={seed} h={hidden} L={num_layers} wd={wd} do={dropout}: "
          f"train {acc_tr:.4f}  val {best_va:.4f}  test {acc_te:.4f}  (ep {best_ep})")
    return model, dict(train_acc=acc_tr, val_acc=best_va, test_acc=acc_te,
                       best_epoch=best_ep, seed=seed, dataset=ds)


SWEEP = [
    dict(hidden=128, num_layers=3, wd=1e-3, dropout=0.3),
    dict(hidden=64,  num_layers=3, wd=1e-3, dropout=0.3),
    dict(hidden=64,  num_layers=2, wd=1e-2, dropout=0.3),
    dict(hidden=32,  num_layers=2, wd=1e-2, dropout=0.5),
    dict(hidden=32,  num_layers=3, wd=1e-3, dropout=0.5),
]


def sweep_proteins(device):
    print("=== PROTEINS Kapazitaets-Sweep (seed 0)")
    for cfg in SWEEP:
        train_one("proteins", seed=0, epochs=300, lr=1e-3, batch=64,
                  device=device, **cfg)
    print("PROTEINS_SWEEP_DONE")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if os.environ.get("SWEEP_ONLY"):
        sweep_proteins(device)
        return
    only = os.environ.get("ONLY_DS")
    print(f"device={device}  ->  {OUT_DIR}")
    for spec in SPECS:
        ds = spec.pop("ds")
        if only and ds != only:
            continue
        print(f"=== {ds}")
        for seed in (0, 1):
            model, meta = train_one(ds, seed=seed, device=device, **spec)
            suffix = "" if seed == 0 else "_seed1"
            prefix = os.path.join(OUT_DIR, f"judge_{ds}_gcn{suffix}")
            save_judge(model, prefix, extra_spec={**meta, "dataset": ds})
            print(f"    -> {prefix}.npz / .json")
    print("SHARED_JUDGES_DONE")


if __name__ == "__main__":
    main()
