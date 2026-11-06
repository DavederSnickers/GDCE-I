"""Train the GDCE GINE classifier recipe on the ORIGINAL (small) MUTAG dataset.

Purpose: enable the clean "same dataset, different classifier" comparison
(is the paper's very good C2 result due to the classifier or the dataset?).
This trains the EXACT GDCE `MutagGINELightning` architecture/recipe (3x GINEConv,
global-sum-pool, BN/ReLU/Dropout head, Adam + ReduceLROnPlateau) — only with
MUTAG's input dims (in_dim=7 atom types, edge_dim=4 bond types, 2 classes) — on
`data/MUTAG.pickle`.

It uses the SAME split as run_c2_mutag_paper_evaluation.py (torch seed 42 shuffle,
last 20% = test) so the trained classifier can be CF-evaluated on the IDENTICAL
test graphs without train/test leakage. The saved Lightning checkpoint is loadable
by `wrappers.mutag_gine_wrapper.MutagGINEWrapperC2`, so run B just points its
`ckpt` at it.

Usage (inside the container, so the conda env + GDCE import resolve):
    apprun python train_gine_mutag.py
    apprun python train_gine_mutag.py --epochs 200 --hidden_dim 128 --lr 1e-4
    apprun python train_gine_mutag.py --gdce_path <GDCE_ROOT>   # local
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / 'src'))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset_path', default='data/MUTAG.pickle',
                   help='Path to the MUTAG TUDataset pickle (relative to repo root).')
    p.add_argument('--gdce_path', default='<GDCE_ROOT>',
                   help='GDCE repo root (for `from src.models.Mutag_GINE import '
                        'MutagGINELightning`). Local: <GDCE_ROOT>')
    p.add_argument('--out_ckpt_dir', default='ckpt/gine_mutag',
                   help='Directory to save the best Lightning checkpoint.')
    # GINE recipe (defaults mirror GDCE Mutag_GINE; dims adapted to MUTAG)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--num_layers', type=int, default=3)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=32)
    # split / reproducibility
    p.add_argument('--split_seed', type=int, default=42,
                   help='MUST match run_c2_mutag_paper_evaluation.py so the test '
                        'split (last 20%) is identical and leakage-free.')
    p.add_argument('--val_frac', type=float, default=0.15,
                   help='Fraction of the 80%% train split held out for validation/'
                        'model selection (test split stays untouched).')
    return p.parse_args()


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


def main() -> None:
    args = parse_args()

    # GDCE's `src` package must be importable for the GINE class + its hparams.
    if args.gdce_path not in sys.path:
        sys.path.insert(0, args.gdce_path)

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch_geometric.loader import DataLoader
    from src.models.Mutag_GINE import MutagGINELightning  # GDCE module

    # ---- dataset ----
    ds_path = _resolve(args.dataset_path)
    print(f"Loading dataset: {ds_path}")
    with open(ds_path, 'rb') as f:
        dataset = pickle.load(f)
    dataset.transform = None  # pickle ships a ToDevice('cuda') transform

    in_dim = int(dataset[0].x.size(1))
    edge_dim = int(dataset[0].edge_attr.size(1)) if dataset[0].edge_attr is not None else 0
    print(f"Detected dims: in_dim={in_dim}, edge_dim={edge_dim}, n_graphs={len(dataset)}")

    # ---- split: IDENTICAL to run_c2_mutag_paper_evaluation.py ----
    # torch seed 42 shuffle -> last 20% = test (CF-eval set, never trained on).
    torch.manual_seed(args.split_seed)
    dataset = dataset.shuffle()
    n = len(dataset)
    n_train_full = int(0.8 * n)
    test_set = dataset[n_train_full:]
    train_full = dataset[:n_train_full]
    n_val = max(1, int(args.val_frac * len(train_full)))
    val_set = train_full[:n_val]
    train_set = train_full[n_val:]
    print(f"Split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)} "
          f"(test is the same 20% used by the paper runner)")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              drop_last=True)  # drop_last avoids 1-sample BN batches
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    # ---- model: GDCE GINE recipe, MUTAG dims ----
    model = MutagGINELightning(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=2,
        num_layers=args.num_layers,
        edge_dim=edge_dim,
        dropout=args.dropout,
        lr=args.lr,
        n_epochs=args.epochs,
    )

    out_dir = _resolve(args.out_ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir), filename='gine_mutag-{epoch}-{val/acc:.3f}',
        monitor='val/acc', mode='max', save_top_k=1, auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        devices=1,
        log_every_n_steps=10,
        callbacks=[ckpt_cb],
        enable_progress_bar=True,
    )

    print(f"\nTraining GINE (recipe=GDCE Mutag_GINE) on MUTAG for {args.epochs} epochs...")
    trainer.fit(model, train_loader, val_loader)

    print("\n--- TEST (best val/acc checkpoint) ---")
    test_metrics = trainer.test(model, test_loader, ckpt_path='best')

    best = ckpt_cb.best_model_path
    print("\n" + "=" * 70)
    print("DONE.")
    print(f"  best checkpoint : {best}")
    print(f"  best val/acc    : {ckpt_cb.best_model_score}")
    print(f"  test metrics    : {test_metrics}")
    print("=" * 70)
    print("\nNext (run B — same dataset, your GINE classifier):")
    print(f"  point the GINE-branch of run_c2_mutag_paper_evaluation.py at:\n    {best}")
    print("  (then compare its CF metrics against run A = paper GCN on the SAME test split)")


if __name__ == '__main__':
    main()
