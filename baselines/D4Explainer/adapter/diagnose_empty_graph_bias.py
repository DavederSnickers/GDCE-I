"""
Empty-graph sanity check for the existing NCI1Filtered classifier.

If the classifier predicts the same class for >90% of empty graphs,
D4Explainer can mode-collapse trivially by removing all edges.
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.nci1_filtered_dataset import NCI1Filtered


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/NCI1"
    ckpt_path = sys.argv[2] if len(sys.argv) > 2 else "param/gnns/nci1_filtered_gcn.pt"
    n_check = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Classifier: {ckpt_path}")
    print(f"Dataset:    {data_path}")
    print(f"Device:     {device}")

    model = torch.load(ckpt_path, map_location=device)
    model.eval()

    test = NCI1Filtered(data_path, mode="testing")
    n_check = min(n_check, len(test))
    print(f"\nClassifying {n_check} test graphs with all edges removed...\n")

    counts = {0: 0, 1: 0}
    confidences = []
    with torch.no_grad():
        for i in range(n_check):
            g = test[i].clone().to(device)
            empty_edges = torch.zeros((2, 0), dtype=torch.long, device=device)
            batch = torch.zeros(g.x.size(0), dtype=torch.long, device=device)
            prob, _ = model.get_pred(x=g.x, edge_index=empty_edges, batch=batch)
            cls = prob.argmax(-1).item()
            counts[cls] += 1
            confidences.append(prob.max().item())

    pct0 = 100 * counts[0] / n_check
    pct1 = 100 * counts[1] / n_check
    avg_conf = sum(confidences) / len(confidences)

    print(f"Empty graph -> class 0: {counts[0]}/{n_check} ({pct0:.1f}%)")
    print(f"Empty graph -> class 1: {counts[1]}/{n_check} ({pct1:.1f}%)")
    print(f"Average confidence:     {avg_conf:.4f}")

    dom_pct = max(pct0, pct1)
    dom_cls = 0 if pct0 > pct1 else 1
    print()
    if dom_pct >= 90:
        print(f"!! BIAS DETECTED: {dom_pct:.1f}% of empty graphs collapse to class {dom_cls}")
        print(f"   D4 will exploit this — flips class {1-dom_cls}->{dom_cls} trivially,")
        print(f"   but flips class {dom_cls}->{1-dom_cls} are essentially impossible.")
        print(f"   This is a likely cause of mode collapse.")
    elif dom_pct >= 75:
        print(f"   Moderate bias ({dom_pct:.1f}%) — D4 has an easy direction but not extreme.")
        print(f"   Probably contributes to mode collapse but isn't the sole cause.")
    else:
        print(f"   No significant empty-graph bias ({dom_pct:.1f}%).")
        print(f"   Mode collapse cause must lie elsewhere (alpha_cf, saving criterion).")


if __name__ == "__main__":
    main()
