"""
Train a NCI1Filtered classifier with BatchNorm ENABLED (as in the original
D4Explainer repo). Apples-to-apples comparison with the existing
nci1_filtered_gcn.pt classifier (which has BatchNorm commented out).

Key fix vs the broken version:
  Old: self.batch_norms.extend([BatchNorm(128)] * conv_unit)
       -> SAME instance referenced twice, shared params/stats -> training stalls
  New: self.batch_norms.extend([BatchNorm(128) for _ in range(conv_unit)])
       -> independent BatchNorm modules per layer

Saves to param/gnns/nci1_filtered_original_gcn.pt (does NOT touch existing checkpoint).
Runs empty-graph sanity check after training.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.nn import ModuleList, ReLU, Softmax
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, LEConv, global_mean_pool
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.nci1_filtered_dataset import NCI1Filtered
from utils.train_utils import Gtrain, Gtest

EPS = 1


class NCI1GCN_BN(torch.nn.Module):
    """LEConv + BatchNorm (enabled) classifier, with the duplicate-BN bug fixed."""

    def __init__(self, conv_unit=2):
        super().__init__()
        self.convs = ModuleList()
        self.convs.append(LEConv(in_channels=37, out_channels=128))
        for _ in range(conv_unit - 2):
            self.convs.append(LEConv(in_channels=128, out_channels=128))
        self.convs.append(LEConv(in_channels=128, out_channels=128))

        # FIX: independent BatchNorm/ReLU instances per layer
        self.batch_norms = ModuleList([BatchNorm(128) for _ in range(conv_unit)])
        self.relus       = ModuleList([ReLU()         for _ in range(conv_unit)])

        self.ffn = nn.Sequential(
            nn.Linear(128, 128), ReLU(), nn.Dropout(), nn.Linear(128, 2)
        )
        self.softmax = Softmax(dim=1)

    def _conv_stack(self, x, edge_index, edge_weight):
        for conv, bn, relu in zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_weight)
            x = relu(bn(x))
        return x

    def forward(self, x, edge_index, batch):
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        x = self._conv_stack(x, edge_index, edge_weight)
        return self.ffn(global_mean_pool(x, batch))

    def get_pred(self, x, edge_index, batch):
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        x = self._conv_stack(x, edge_index, edge_weight)
        pred = self.ffn(global_mean_pool(x, batch))
        return self.softmax(pred), pred

    def get_pred_explain(self, x, edge_index, edge_mask, batch):
        edge_mask = (edge_mask * EPS).sigmoid()
        x = self._conv_stack(x, edge_index, edge_mask)
        pred = self.ffn(global_mean_pool(x, batch))
        return self.softmax(pred), pred


def empty_graph_check(model, dataset, n_check, device):
    print(f"\n{'='*60}\nEMPTY-GRAPH SANITY CHECK\n{'='*60}")
    model.eval()
    n_check = min(n_check, len(dataset))
    counts = {0: 0, 1: 0}
    confs = []
    with torch.no_grad():
        for i in range(n_check):
            g = dataset[i].clone().to(device)
            empty_edges = torch.zeros((2, 0), dtype=torch.long, device=device)
            batch = torch.zeros(g.x.size(0), dtype=torch.long, device=device)
            prob, _ = model.get_pred(x=g.x, edge_index=empty_edges, batch=batch)
            counts[prob.argmax(-1).item()] += 1
            confs.append(prob.max().item())
    p0 = 100 * counts[0] / n_check
    p1 = 100 * counts[1] / n_check
    print(f"Empty graph -> class 0: {counts[0]}/{n_check} ({p0:.1f}%)")
    print(f"Empty graph -> class 1: {counts[1]}/{n_check} ({p1:.1f}%)")
    print(f"Average confidence:     {sum(confs)/len(confs):.4f}")
    dom = max(p0, p1)
    if dom >= 90:
        print(f"!! BIAS DETECTED ({dom:.1f}%) — D4 will mode-collapse with this classifier")
    elif dom >= 75:
        print(f"   Moderate bias ({dom:.1f}%) — still problematic")
    else:
        print(f"   No strong bias ({dom:.1f}%) — classifier suitable for D4")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",  default="data/NCI1")
    p.add_argument("--out_path",   default="param/gnns/nci1_filtered_original_gcn.pt")
    p.add_argument("--epoch",      type=int, default=300)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_unit",   type=int, default=2)
    p.add_argument("--cuda",       type=int, default=0)
    p.add_argument("--verbose",    type=int, default=10)
    p.add_argument("--n_sanity",   type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Architecture: NCI1GCN_BN (BatchNorm ON, fixed duplicate-instance bug)")
    print(f"Dataset:      NCI1Filtered (graphs <=50 nodes)")
    print(f"Output:       {args.out_path}")
    print(f"Device:       {device}\n")

    train = NCI1Filtered(args.data_path, mode="training")
    val   = NCI1Filtered(args.data_path, mode="evaluation")
    test  = NCI1Filtered(args.data_path, mode="testing")
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test,  batch_size=args.batch_size, shuffle=False)

    model = NCI1GCN_BN(conv_unit=args.num_unit).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=10, min_lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, args.epoch + 1):
        loss = Gtrain(train_loader, model, optimizer, device=device, criterion=criterion)
        val_error, val_acc = Gtest(val_loader, model, device=device, criterion=criterion)
        scheduler.step(val_error)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        if epoch % args.verbose == 0:
            _, test_acc = Gtest(test_loader, model, device=device, criterion=criterion)
            print(f"Epoch {epoch:4d}  loss={loss:.4f}  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")

    _, final_acc = Gtest(test_loader, model, device=device, criterion=criterion)
    print(f"\nFinal test accuracy: {final_acc:.4f}  (best val: {best_val_acc:.4f})")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    torch.save(model.cpu(), args.out_path)
    print(f"Saved: {args.out_path}")

    model.to(device)
    empty_graph_check(model, test, args.n_sanity, device)


if __name__ == "__main__":
    main()
