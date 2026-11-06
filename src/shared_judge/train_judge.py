"""Train the shared judge classifiers that every counterfactual method will target.

Five judges in total:
    GCN  : mutagenicity_mol, benzene, nci1
    GINE : mutagenicity_mol, benzene          (nci1 has no bond types)

Unlike the per-method oracles, this trains on the TRAIN split only, selects the
checkpoint on VAL, and touches TEST exactly once for the reported number. The old
oracles merged val into train and reported the final-epoch test accuracy.

Usage:
    python train_judge.py --dataset mutagenicity_mol --conv gcn  --data_root <root>
    python train_judge.py --dataset mutagenicity_mol --conv gine --data_root <root>
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
from shared_judge import SharedJudge, collate, load_split, save_judge  # noqa: E402


def _accuracy(model, graphs, device, batch_size=128):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(graphs), batch_size):
            x, ei, batch, ea, y = collate(graphs[i:i + batch_size], device)
            logits = model(x, ei, batch, edge_attr=ea)
            correct += int((logits.argmax(dim=-1) == y).sum().item())
            total += int(y.shape[0])
    return correct / max(total, 1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=("mutagenicity_mol", "benzene", "nci1"))
    p.add_argument("--conv", required=True, choices=("gcn", "gine"))
    p.add_argument("--data_root", required=True,
                   help="dir containing <dataset>/raw/{train,val,test}.csv")
    p.add_argument("--out_dir", default="judges")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_nodes", type=int, default=50)
    p.add_argument("--batchnorm", action="store_true",
                   help="BatchNorm1d after each message-passing layer.")
    p.add_argument("--cosine", action="store_true",
                   help="Cosine LR decay instead of a constant learning rate.")
    p.add_argument("--refit_train_val", action="store_true",
                   help="After selecting the epoch on val, retrain from scratch on "
                        "train+val for that many epochs. Recovers the ~25%% of data "
                        "held out for selection. Test is still never used.")
    p.add_argument("--tag", default="", help="Suffix for the output filename.")
    args = p.parse_args()

    if args.conv == "gine" and args.dataset == "nci1":
        print("[train_judge] nci1 has no bond types — GINE judge is not applicable.")
        return 2

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_judge] device={device} torch={torch.__version__}", flush=True)

    splits = {}
    for split in ("train", "val", "test"):
        graphs, n_node_types, n_edge_types = load_split(
            args.data_root, args.dataset, split, max_nodes=args.max_nodes
        )
        splits[split] = graphs
        print(f"[train_judge] {split}: {len(graphs)} graphs", flush=True)
    print(f"[train_judge] node types={n_node_types} edge types={n_edge_types}", flush=True)

    label_counts = np.bincount([g["y"] for g in splits["train"]], minlength=2)
    print(f"[train_judge] train label balance: {label_counts.tolist()}", flush=True)

    def _new_model():
        return SharedJudge(
            num_features=n_node_types,
            num_classes=2,
            hidden=args.hidden,
            num_layers=args.num_layers,
            edge_dim=n_edge_types if args.conv == "gine" else 0,
            conv=args.conv,
            dropout=args.dropout,
            batchnorm=args.batchnorm,
        ).to(device)

    def _fit(graphs, n_epochs, eval_on=None):
        """Train a fresh model for n_epochs. If eval_on is given, track the best
        epoch by accuracy on it and return that checkpoint."""
        m = _new_model()
        opt = torch.optim.Adam(m.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
                 if args.cosine else None)
        best, best_st, best_ep = -1.0, None, n_epochs - 1
        for epoch in range(n_epochs):
            m.train()
            order = np.random.permutation(len(graphs))
            total_loss = 0.0
            for i in range(0, len(order), args.batch_size):
                idx = order[i:i + args.batch_size]
                # BatchNorm needs >1 sample; drop a trailing singleton batch.
                if args.batchnorm and len(idx) == 1:
                    continue
                x, ei, batch, ea, y = collate([graphs[j] for j in idx], device)
                opt.zero_grad()
                loss = F.cross_entropy(m(x, ei, batch, edge_attr=ea), y)
                loss.backward()
                opt.step()
                total_loss += float(loss.item()) * len(idx)
            if sched is not None:
                sched.step()

            if eval_on is not None and (epoch % 5 == 0 or epoch == n_epochs - 1):
                acc = _accuracy(m, eval_on, device)
                if acc > best:
                    best, best_ep = acc, epoch
                    best_st = {k: v.detach().cpu().clone()
                               for k, v in m.state_dict().items()}
                print(f"  epoch {epoch:>4}  loss {total_loss / len(graphs):.4f}  "
                      f"val {acc:.4f}  best {best:.4f} @{best_ep}", flush=True)
        if eval_on is not None and best_st is not None:
            m.load_state_dict(best_st)
        return m.to(device).eval(), best_ep

    t0 = time.time()

    # Stage 1: train on TRAIN, select the epoch on VAL.
    model, best_epoch = _fit(splits["train"], args.epochs, eval_on=splits["val"])

    # Stage 2 (optional): refit from scratch on TRAIN+VAL for the selected number of
    # epochs. Uses the ~25% of data held out for selection without touching TEST.
    if args.refit_train_val:
        print(f"\n[train_judge] refitting on train+val for {best_epoch + 1} epochs "
              f"({len(splits['train']) + len(splits['val'])} graphs)", flush=True)
        model, _ = _fit(splits["train"] + splits["val"], best_epoch + 1, eval_on=None)

    train_acc = _accuracy(model, splits["train"], device)
    val_acc = _accuracy(model, splits["val"], device)
    test_acc = _accuracy(model, splits["test"], device)
    elapsed = time.time() - t0

    print(f"\n[train_judge] {args.dataset}/{args.conv}  "
          f"train {train_acc:.4f}  val {val_acc:.4f}  TEST {test_acc:.4f}  "
          f"(best epoch {best_epoch}, {elapsed:.0f}s)", flush=True)

    prefix = os.path.join(
        args.out_dir, f"judge_{args.dataset}_{args.conv}{args.tag}"
    )
    npz, js = save_judge(model, prefix, extra_spec={
        "dataset": args.dataset,
        "seed": args.seed,
        "max_nodes": args.max_nodes,
        "selected_epoch": best_epoch,
        "batchnorm": args.batchnorm,
        "cosine_lr": args.cosine,
        "refit_train_val": args.refit_train_val,
        "lr": args.lr,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "train_label_counts": label_counts.tolist(),
    })
    print(f"[train_judge] wrote {npz}\n[train_judge] wrote {js}", flush=True)

    # Round-trip check: the exported weights must reproduce the test accuracy
    # bit-for-bit, otherwise the portable format is broken and every downstream
    # repo would silently use a different classifier.
    from shared_judge import load_judge
    reloaded = load_judge(prefix, device=device)
    rt_acc = _accuracy(reloaded, splits["test"], device)
    ok = abs(rt_acc - test_acc) < 1e-9
    print(f"[train_judge] round-trip test acc {rt_acc:.6f} "
          f"({'OK' if ok else 'MISMATCH'})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
