"""Train the paper-style GCN (C2 exp1_3 architecture) on a GDCE dataset.

Default: GDCE Mutagenicity (`mutagenicity_mol`). This gives the "same classifier
architecture, different dataset" leg of the analysis: the paper's GCN recipe
(topology-only, edge-blind) trained on the harder Mutagenicity set, so C2 can be
evaluated on it (run D) and compared against:
  - run A : GCN @ MUTAG       (dataset effect, classifier fixed = GCN)
  - run C : GINE @ Mutagenicity (classifier effect, dataset fixed = Mutagenicity)

Labels: GDCE *_mol Data objects keep the label in `data.guidance` (one-hot),
with `data.y` empty — handled below.

Note: mutagenicity_mol edge_index already carries the full self-loop diagonal
(adj diagonal = 1.0), so the GCN trains WITH the diagonal; at C2-eval time use
PaperGCNWrapperC2(self_loops='full') to reconstruct it (see paper_gcn.py).

Usage (inside the container):
    apprun python train_paper_gcn.py
    apprun python train_paper_gcn.py --epochs 300 --nhid 64       # bump capacity if underfitting
    apprun python train_paper_gcn.py --gdce_path <GDCE_ROOT>   # local
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from wrappers.paper_gcn import PaperGCN
from wrappers.dataset_loader import build_minimal_cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gdce_path', default='<GDCE_ROOT>',
                   help='GDCE repo root (for the SpectreGraphDataModule). '
                        'Local: <GDCE_ROOT>')
    p.add_argument('--dataset_name', default='mutagenicity_mol')
    p.add_argument('--datadir', default='data/mutagenicity_mol/',
                   help='dataset dir relative to gdce_path')
    p.add_argument('--out_ckpt', default='ckpt/paper_gcn_mutagenicity/gcn.pt',
                   help='where to save the trained state_dict (relative to repo root)')
    # paper-style GCN architecture
    p.add_argument('--nhid', type=int, default=20)
    p.add_argument('--nout', type=int, default=20)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--batchnorm', action='store_true',
                   help='Add BatchNorm after each conv (opt-in; off = exact paper arch).')
    # training
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def _labels(batch) -> torch.Tensor:
    """GDCE *_mol keep the graph label in `guidance` (one-hot); `y` is empty."""
    if getattr(batch, 'y', None) is not None and batch.y.numel() > 0:
        return batch.y.long().view(-1)
    return batch.guidance.argmax(dim=-1).view(-1)


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_splits(gdce_path: str, dataset_name: str, datadir: str):
    if gdce_path not in sys.path:
        sys.path.insert(0, gdce_path)
    from src.datasets.spectre_dataset import SpectreGraphDataModule
    cfg = build_minimal_cfg(dataset_name, datadir)
    dm = SpectreGraphDataModule(cfg)
    return {k: dm.dataloaders[k].dataset for k in ('train', 'val', 'test')}


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x.float(), batch.edge_index, batch.batch)
        pred = logits.argmax(dim=-1)
        y = _labels(batch).to(device)
        correct += int((pred == y).sum()); total += int(y.numel())
    return correct / total if total else float('nan')


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    from torch_geometric.loader import DataLoader
    splits = load_splits(args.gdce_path, args.dataset_name, args.datadir)
    in_dim = int(splits['train'][0].x.size(1))
    print(f"Dataset {args.dataset_name}: in_dim={in_dim}, "
          f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    train_loader = DataLoader(splits['train'], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(splits['val'], batch_size=args.batch_size)
    test_loader = DataLoader(splits['test'], batch_size=args.batch_size)

    model = PaperGCN(in_dim, nhid=args.nhid, nout=args.nout,
                     num_classes=2, dropout=args.dropout,
                     batchnorm=args.batchnorm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    out_ckpt = _resolve(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    best_val = -1.0
    print(f"\nTraining paper-style GCN (in_dim={in_dim}, nhid={args.nhid}, nout={args.nout}) "
          f"for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x.float(), batch.edge_index, batch.batch)
            loss = criterion(logits, _labels(batch).to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * batch.num_graphs
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            val_acc = evaluate(model, val_loader, device)
            train_acc = evaluate(model, train_loader, device)
            print(f"  epoch {epoch:4d}  loss={total_loss/len(splits['train']):.4f}  "
                  f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")
            if val_acc > best_val:
                best_val = val_acc
                torch.save(model.state_dict(), out_ckpt)

    # if we never improved (tiny --epochs), still save the final model
    if best_val < 0:
        torch.save(model.state_dict(), out_ckpt)
        best_val = evaluate(model, val_loader, device)

    # reload best, report test
    model.load_state_dict(torch.load(out_ckpt, map_location=device, weights_only=True))
    test_acc = evaluate(model, test_loader, device)

    print("\n" + "=" * 70)
    print("DONE.")
    print(f"  best val acc : {best_val:.4f}")
    print(f"  test acc     : {test_acc:.4f}")
    print(f"  saved to     : {out_ckpt}")
    print("=" * 70)
    print("\nNext (run D — same dataset as C, paper-style GCN classifier):")
    print(f"  apprun python run_c2_mutag_evaluation.py classifier=gcn \\")
    print(f"      gcn_ckpt={out_ckpt} gcn_in_dim={in_dim}")
    print("  (compare run D's CF metrics vs run C = GINE@Mutagenicity and run A = GCN@MUTAG)")


if __name__ == '__main__':
    main()
