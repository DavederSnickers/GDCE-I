"""
Diagnostic: check actual NCI1Filtered label values and classifier behavior.
Does NOT train anything — just inspects data and (optionally) loads checkpoint.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.nci1_filtered_dataset import NCI1Filtered


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/NCI1"
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Loading NCI1Filtered from: {data_path}")
    train = NCI1Filtered(data_path, mode="training")
    val   = NCI1Filtered(data_path, mode="evaluation")
    test  = NCI1Filtered(data_path, mode="testing")

    # 1. Inspect label values
    print(f"\n{'='*60}\nLABEL INSPECTION\n{'='*60}")
    for name, ds in [("train", train), ("val", val), ("test", test)]:
        ys = torch.cat([ds[i].y.flatten() for i in range(len(ds))])
        unique, counts = torch.unique(ys, return_counts=True)
        print(f"{name:5s} ({len(ds)} graphs)  "
              f"unique y values: {unique.tolist()}  "
              f"counts: {counts.tolist()}  "
              f"dtype: {ys.dtype}  "
              f"shape per graph: {ds[0].y.shape}")

    # 2. Test CrossEntropyLoss with actual labels (sanity test)
    print(f"\n{'='*60}\nCROSSENTROPYLOSS COMPATIBILITY TEST\n{'='*60}")
    fake_logits = torch.randn(10, 2)  # 2-class output
    sample_y = torch.cat([train[i].y.flatten() for i in range(10)])
    print(f"Sample target values (first 10): {sample_y.tolist()}")
    print(f"Sample target dtype: {sample_y.dtype}")
    try:
        loss = torch.nn.CrossEntropyLoss()(fake_logits, sample_y)
        print(f"CrossEntropyLoss WORKS without error: loss={loss.item():.4f}")
        print("=> Labels are compatible with 2-class output (range [0, C-1])")
    except (IndexError, RuntimeError) as e:
        print(f"CrossEntropyLoss CRASHES: {e}")
        print("=> Labels are OUT OF BOUNDS — Sonnet's hypothesis is correct")

    # 3. If a checkpoint is provided, test what it predicts
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"\n{'='*60}\nCHECKPOINT BEHAVIOR TEST\n{'='*60}")
        print(f"Loading: {ckpt_path}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.load(ckpt_path, map_location=device)
        model.eval()

        preds = []
        labels = []
        from torch_geometric.loader import DataLoader
        loader = DataLoader(test, batch_size=64, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                preds.append(out.argmax(dim=1).cpu())
                labels.append(batch.y.flatten().cpu())
        preds = torch.cat(preds)
        labels = torch.cat(labels)
        print(f"Predictions: unique={torch.unique(preds).tolist()}  "
              f"counts per class: {torch.bincount(preds).tolist()}")
        print(f"Labels:      unique={torch.unique(labels).tolist()}  "
              f"counts per class: {torch.bincount(labels).tolist()}")
        acc = (preds == labels).float().mean().item()
        print(f"Test accuracy (recomputed): {acc:.4f}")
        if len(torch.unique(preds)) == 1:
            print("WARNING: Classifier predicts ONLY one class -> degenerate model.")


if __name__ == "__main__":
    main()
