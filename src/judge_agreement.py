"""Agreement of two classifiers on the UNMODIFIED test graphs.

A transfer rate on its own is not interpretable: if two classifiers already
agree on only 80 % of the untouched originals, then a transfer rate of 80 % is
not evidence of robustness but the normal state of affairs. This script supplies
that reference point.

Aufruf:
    apprun python -u src/judge_agreement.py --name proteins \\
        --judge-a <ckpt> --judge-b <ckpt>
"""
import argparse
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--judge-a', required=True)
    ap.add_argument('--judge-b', required=True)
    ap.add_argument('--datadir', default=None)
    ap.add_argument('--split', default='test')
    args = ap.parse_args()

    from datasets.spectre_dataset import SpectreGraphDataset
    from src.models.NCI1_GCN import NCI1GCNLightning
    from torch_geometric.loader import DataLoader

    root = os.path.abspath(args.datadir or f'data/{args.name}/')
    ds = SpectreGraphDataset(dataset_name=args.name, split=args.split, root=root)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    preds = {}
    for tag, ckpt in (('A', args.judge_a), ('B', args.judge_b)):
        if not str(ckpt).endswith('.ckpt'):
            # Shared-Judge-Export (`<prefix>.npz` + `<prefix>.json`)
            from src.models.shared_judge_adapter import load_shared_judge
            clf = load_shared_judge(str(ckpt)).to(device).eval()
        else:
            clf = NCI1GCNLightning.load_from_checkpoint(ckpt, map_location='cpu').to(device).eval()
        out = []
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=128):
                batch = batch.to(device)
                logits = clf(batch)   # NCI1GCNLightning.forward nimmt das Data-Objekt
                out.append(logits.argmax(dim=-1).cpu().numpy())
        preds[tag] = np.concatenate(out)
        del clf

    labels = np.array([int(g.guidance.argmax(dim=-1).item()) for g in ds])
    a, b = preds['A'], preds['B']

    print('=' * 74)
    print(f'JUDGE-UEBEREINSTIMMUNG: {args.name} ({args.split}, {len(ds)} Graphen)')
    print('=' * 74)
    print(f'  classifier A accuracy (against the guidance label): {(a == labels).mean():.4f}')
    print(f'  classifier B accuracy (against the guidance label): {(b == labels).mean():.4f}')
    print(f'  agreement A vs. B on the untouched originals:       {(a == b).mean():.4f}')
    print()
    print('  This is the reference point: a transfer rate below this value means')
    print('  the counterfactuals are more model-specific than the models are to')
    print('  each other; above it, that they transfer more robustly.')
    print('=' * 74)


if __name__ == '__main__':
    main()
