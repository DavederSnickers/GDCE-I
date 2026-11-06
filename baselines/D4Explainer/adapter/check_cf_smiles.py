"""
Local SMILES validity check for counterfactuals saved by run_d4_mutag_evaluation.py.

Usage:
  python check_cf_smiles.py results/mutag_evaluation/cf_graphs.pt
"""

import sys
import torch
from rdkit import Chem
from explainers.visual import graph_to_mol


def graph_data_to_smiles(x, edge_index, edge_attr):
    """Convert raw tensors to a SMILES string. Returns (smiles, valid)."""
    X = x.tolist()
    ei = edge_index.T.tolist()
    ea = edge_attr.tolist() if edge_attr is not None else []

    # Skip edges with all-zero attr (unknown bond type added by D4Explainer)
    if ea:
        pairs = [(e, a) for e, a in zip(ei, ea) if any(v > 0 for v in a)]
        ei = [p[0] for p in pairs]
        ea = [p[1] for p in pairs]

    try:
        mol = graph_to_mol(X, ei, ea)
        Chem.SanitizeMol(mol)
        smi = Chem.MolToSmiles(mol)
        return (smi, True) if smi else (None, False)
    except Exception:
        return None, False


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/mutag_evaluation/cf_graphs.pt"
    records = torch.load(path, map_location="cpu")
    print(f"Loaded {len(records)} CF records from {path}\n")

    thresholds = sorted({r['threshold'] for r in records})

    for thresh in thresholds:
        subset = [r for r in records if r['threshold'] == thresh]
        total = len(subset)
        n_valid = 0
        n_valid_succ = 0
        n_succ = sum(1 for r in subset if r['success'])

        for r in subset:
            _, valid = graph_data_to_smiles(r['cf_x'], r['cf_edge_index'], r['cf_edge_attr'])
            r['cf_smiles_valid'] = valid
            if valid:
                n_valid += 1
                if r['success']:
                    n_valid_succ += 1

        pct_all  = 100 * n_valid / total if total > 0 else float('nan')
        pct_succ = 100 * n_valid_succ / n_succ if n_succ > 0 else float('nan')

        print(f"Threshold = {thresh}")
        print(f"  CF Valid SMILES (all):  {n_valid}/{total} ({pct_all:.1f}%)")
        print(f"  CF Valid SMILES (succ): {n_valid_succ}/{n_succ} ({pct_succ:.1f}%)")
        print()


if __name__ == "__main__":
    main()
