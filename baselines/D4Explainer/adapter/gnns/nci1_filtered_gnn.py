"""
Train NCI1GCN on NCI1Filtered (graphs ≤50 nodes) and save to
param/gnns/nci1_filtered_gcn.pt so that main.py --dataset nci1_filtered
finds the classifier.

Usage:
  python gnns/nci1_filtered_gnn.py --data_path data/NCI1 --epoch 300
"""

import argparse
import os
import os.path as osp
import sys
import time

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets.nci1_filtered_dataset import NCI1Filtered
from gnns.nci1_gnn import NCI1GCN
from utils.train_utils import Gtrain, Gtest
from utils.helper import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="data/NCI1")
    p.add_argument("--out_dir",   default="param/gnns")
    p.add_argument("--epoch",     type=int, default=300)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--batch_size",type=int, default=128)
    p.add_argument("--num_unit",  type=int, default=3)
    p.add_argument("--cuda",      type=int, default=0)
    p.add_argument("--verbose",   type=int, default=10)
    p.add_argument("--seed",      type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    train_dataset = NCI1Filtered(args.data_path, mode="training")
    val_dataset   = NCI1Filtered(args.data_path, mode="evaluation")
    test_dataset  = NCI1Filtered(args.data_path, mode="testing")
    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    model = NCI1GCN(conv_unit=args.num_unit).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=10, min_lr=1e-5)

    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(1, args.epoch + 1):
        t0 = time.time()
        loss = Gtrain(train_loader, model, optimizer, device=device, criterion=criterion)
        val_error, val_acc = Gtest(val_loader, model, device=device, criterion=criterion)
        scheduler.step(val_error)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if epoch % args.verbose == 0:
            test_error, test_acc = Gtest(test_loader, model, device=device, criterion=criterion)
            print(f"Epoch {epoch:4d} [{time.time()-t0:.1f}s]  "
                  f"loss={loss:.4f}  val_acc={val_acc:.4f}  test_acc={test_acc:.4f}")

    _, test_acc = Gtest(test_loader, model, device=device, criterion=criterion)
    print(f"\nFinal test accuracy: {test_acc:.4f}  (best val: {best_val_acc:.4f})")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = osp.join(args.out_dir, "nci1_filtered_gcn.pt")
    torch.save(model.cpu(), out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
