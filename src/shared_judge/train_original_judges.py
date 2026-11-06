"""Train the ORIGINAL classifiers of the five baselines on GDCE's Mutagenicity split.

Original-classifier ablation (reviewer point R1.4): every baseline is evaluated against
the classifier its own paper/code trains, re-implemented in shared_judge.py, and trained
with that paper's/code's recipe on our data (CSV `y` labels, our train/val/test split).
Seed 0 gives the classifier the methods attack; seed 1 the control copy for NA.

Recipes (as shipped by each baseline; none of them selects a checkpoint, all keep the
last epoch):
    c2gcn    C2Explainer paper, Tab. 8 (MUTAG): 3 x GCN(20), Adam lr 1e-3, weight decay
             1e-3, dropout 0, 5000 epochs. Batch size not reported -> 64.
    cf2gcn   gnn_cff/scripts/train_graph_classification.py: Adam lr 0.002, weight decay
             1e-5 (hard-coded; the --weight_decay 0.005 argument is never read), batch
             128, BCE on the sigmoid output, 1000 epochs.
    d4gcn    D4Explainer/gnns/mutag_gnn.py: Adam lr 1e-3, batch 128, 300 epochs,
             ReduceLROnPlateau(min, factor 0.8, patience 10, min_lr 1e-5) on the
             validation loss, cross-entropy.
    ucxgine  COMBINEX GraphTrainer + config/trainer/naive_trainer.yaml: Adam lr 1e-3,
             gradient-norm clip 2, batch 256, cross-entropy, 500 epochs.
    xplgcn   XPlore_config/GRAPH/MUTAG/MUTAG_GCN_XPlore.jsonc run by GRETEL's
             src/core/torch_base.py: RMSprop lr 0.01 with LinearLR 1.0 -> 0.5 over 300 epochs,
             batch 32 (drop_last), cross-entropy. 5% of the training graphs (at least one batch)
             are held out; training stops once the held-out loss has changed by less than 1e-4
             in four epochs (counted, never reset). The last epoch is kept.

Usage:
    python train_original_judges.py --arch c2gcn --seed 0 --data_root <root> --out_dir <dir>
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_judge import ORIGINAL_JUDGES, collate, load_judge, load_split, save_judge  # noqa: E402

RECIPES = {
    "c2gcn": dict(hidden=20, num_layers=3, dropout=0.0, lr=1e-3, weight_decay=1e-3,
                  epochs=5000, batch_size=64, loss="ce", clip=None, plateau=False),
    "cf2gcn": dict(hidden=128, num_layers=3, dropout=0.0, lr=0.002, weight_decay=1e-5,
                   epochs=1000, batch_size=128, loss="bce", clip=None, plateau=False),
    "d4gcn": dict(hidden=128, num_layers=2, dropout=0.5, lr=1e-3, weight_decay=0.0,
                  epochs=300, batch_size=128, loss="ce", clip=None, plateau=True),
    "ucxgine": dict(hidden=128, num_layers=3, dropout=0.5, lr=1e-3, weight_decay=0.0,
                    epochs=500, batch_size=256, loss="ce", clip=2.0, plateau=False),
    "xplgcn": dict(hidden=None, num_layers=2, dropout=0.0, lr=0.01, weight_decay=0.0,
                   epochs=300, batch_size=32, loss="ce", clip=None, plateau=False,
                   optimizer="rmsprop", linear_lr_end=0.5, drop_last=True,
                   holdout=0.05, early_stop=1e-4, early_stop_count=4),
}


def _loss(logits, y, kind):
    if kind == "bce":
        # logits = [0, z]: BCE on sigmoid(z), exactly what CF2's BCELoss sees
        return F.binary_cross_entropy_with_logits(logits[:, 1], y.float())
    return F.cross_entropy(logits, y)


def _evaluate(model, graphs, device, kind, batch_size=256):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for i in range(0, len(graphs), batch_size):
            x, ei, batch, ea, y = collate(graphs[i:i + batch_size], device)
            logits = model(x, ei, batch, edge_attr=ea)
            loss_sum += float(_loss(logits, y, kind)) * int(y.shape[0])
            correct += int((logits.argmax(dim=-1) == y).sum().item())
            total += int(y.shape[0])
    return correct / max(total, 1), loss_sum / max(total, 1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=sorted(RECIPES))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="mutagenicity_mol")
    p.add_argument("--data_root", required=True,
                   help="dir containing <dataset>/raw/{train,val,test}.csv")
    p.add_argument("--out_dir", default="judges_original")
    p.add_argument("--max_nodes", type=int, default=50)
    p.add_argument("--epochs", type=int, default=None, help="override the recipe (smoke tests)")
    args = p.parse_args()
    r = dict(RECIPES[args.arch])
    if args.epochs is not None:
        r["epochs"] = args.epochs

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[orig_judge] arch={args.arch} seed={args.seed} device={device} "
          f"torch={torch.__version__} recipe={r}", flush=True)

    splits = {}
    for split in ("train", "val", "test"):
        splits[split], n_node_types, n_edge_types = load_split(
            args.data_root, args.dataset, split, max_nodes=args.max_nodes)
        print(f"[orig_judge] {split}: {len(splits[split])} graphs", flush=True)
    label_counts = np.bincount([g["y"] for g in splits["train"]], minlength=2)

    model = ORIGINAL_JUDGES[args.arch](
        n_node_types, 2, hidden=r["hidden"], num_layers=r["num_layers"],
        edge_dim=n_edge_types, dropout=r["dropout"]).to(device)
    if r.get("optimizer") == "rmsprop":
        opt = torch.optim.RMSprop(model.parameters(), lr=r["lr"], weight_decay=r["weight_decay"])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=r["lr"], weight_decay=r["weight_decay"])
    sched = None
    if r["plateau"]:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.8, patience=10, min_lr=1e-5)
    elif r.get("linear_lr_end"):
        sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1.0, end_factor=r["linear_lr_end"], total_iters=r["epochs"])

    train = splits["train"]
    holdout = []
    if r.get("holdout"):
        # GRETEL's TorchBase holds out 5% of the training graphs (at least one batch) for early stopping
        idx = list(range(len(train)))
        random.shuffle(idx)
        n_hold = max(int(r["holdout"] * len(idx)), r["batch_size"])
        holdout = [train[i] for i in idx[len(idx) - n_hold:]]
        train = [train[i] for i in idx[:len(idx) - n_hold]]
        print(f"[orig_judge] early-stopping holdout: {len(holdout)} of the training graphs", flush=True)
    last_losses, small_changes = [0.0, 0.0], 0
    t0 = time.time()
    for epoch in range(r["epochs"]):
        model.train()
        order = np.random.permutation(len(train))
        loss_sum = 0.0
        stop = len(order) - len(order) % r["batch_size"] if r.get("drop_last") else len(order)
        for i in range(0, stop, r["batch_size"]):
            x, ei, batch, ea, y = collate([train[j] for j in order[i:i + r["batch_size"]]], device)
            opt.zero_grad()
            loss = _loss(model(x, ei, batch, edge_attr=ea), y, r["loss"])
            loss.backward()
            if r["clip"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), r["clip"])
            opt.step()
            loss_sum += float(loss.item()) * int(y.shape[0])
        if r["plateau"]:
            sched.step(_evaluate(model, splits["val"], device, r["loss"])[1])
        elif sched is not None:
            sched.step()
        stopped = False
        if holdout:
            # mean over the full held-out batches in a fresh order (GRETEL: shuffle, drop_last)
            model.eval()
            h_order = np.random.permutation(len(holdout))
            h_losses = []
            with torch.no_grad():
                for i in range(0, len(h_order) - len(h_order) % r["batch_size"], r["batch_size"]):
                    x, ei, batch, ea, y = collate([holdout[j] for j in h_order[i:i + r["batch_size"]]], device)
                    h_losses.append(float(_loss(model(x, ei, batch, edge_attr=ea), y, r["loss"])))
            last_losses = [last_losses[1], float(np.mean(h_losses))]
            if abs(last_losses[0] - last_losses[1]) < r["early_stop"]:
                small_changes += 1
                stopped = small_changes == r["early_stop_count"]
        if epoch % max(r["epochs"] // 50, 1) == 0 or epoch == r["epochs"] - 1 or stopped:
            val_acc, val_loss = _evaluate(model, splits["val"], device, r["loss"])
            print(f"  epoch {epoch:>5}  train loss {loss_sum / max(stop, 1):.4f}  "
                  f"val loss {val_loss:.4f}  val acc {val_acc:.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  {time.time() - t0:.0f}s", flush=True)
        if stopped:
            print(f"[orig_judge] early stop after epoch {epoch}", flush=True)
            break

    model.eval()
    acc = {s: _evaluate(model, splits[s], device, r["loss"])[0] for s in ("train", "val", "test")}
    print(f"\n[orig_judge] {args.arch} seed {args.seed}: train {acc['train']:.4f}  "
          f"val {acc['val']:.4f}  TEST {acc['test']:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    suffix = "" if args.seed == 0 else f"_seed{args.seed}"
    prefix = os.path.join(args.out_dir, f"judge_{args.dataset}_{args.arch}{suffix}")
    npz, js = save_judge(model, prefix, extra_spec={
        "dataset": args.dataset,
        "seed": args.seed,
        "max_nodes": args.max_nodes,
        "recipe": r,
        "selected_epoch": epoch,
        "n_holdout": len(holdout),
        "train_acc": acc["train"],
        "val_acc": acc["val"],
        "test_acc": acc["test"],
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "train_label_counts": label_counts.tolist(),
        "origin": "original-classifier ablation (R1.4): baseline architecture + recipe",
    })
    print(f"[orig_judge] wrote {npz}\n[orig_judge] wrote {js}", flush=True)

    # The exported weights must reproduce the test accuracy, otherwise the portable
    # format is broken and every downstream repo would use a different classifier.
    rt_acc = _evaluate(load_judge(prefix, device=device), splits["test"], device, r["loss"])[0]
    ok = abs(rt_acc - acc["test"]) < 1e-9
    print(f"[orig_judge] round-trip test acc {rt_acc:.6f} ({'OK' if ok else 'MISMATCH'})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
