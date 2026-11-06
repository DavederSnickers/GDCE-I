"""Compare D4's MutagFiltered against GDCE's pre-processed Mutagenicity.

User claim: both filter only by num_nodes <= 50, so the molecule set should
be identical.  This script checks that empirically.

GDCE layout (observed on cluster):
    <gdce_root>/processed/{train,val,test}.pt   <- GDCE (data, slices)
    <gdce_root>/raw/atom_decoder.json           <- GDCE atom-type list
    <gdce_root>/raw/node_type_map.npy           <- TU-index -> GDCE-index map

Run on the cluster:
    apprun python compare_mutag_datasets.py \
        --d4_path data/MUTAG \
        --gdce_root <GDCE_DATA>/mutagenicity_mol
"""
import argparse
import json
import os
import pickle
import types

import numpy as np
import torch

from datasets.mutag_filtered_dataset import MutagFiltered


# --- robust loader for GDCE .pt files that pickle 'src.*' classes ----------
def _load_gdce_pt(path):
    from torch_geometric.data import Data

    class _SrcStub(Data):
        pass

    class _GDCEUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "src" or module.startswith("src."):
                return type(name, (_SrcStub,), {})
            return super().find_class(module, name)

    _pm = types.ModuleType("_gdce_pickle")
    _pm.Unpickler = _GDCEUnpickler
    _pm.dump, _pm.dumps, _pm.load, _pm.loads = (
        pickle.dump, pickle.dumps, pickle.load, pickle.loads)
    return torch.load(path, map_location="cpu", pickle_module=_pm)


def _make_list(data, slices):
    from torch_geometric.data import Data
    keys = list(slices.keys())
    n = len(slices[keys[0]]) - 1
    out = []
    for i in range(n):
        d = Data()
        for key in keys:
            start, end = int(slices[key][i]), int(slices[key][i + 1])
            val = data[key]
            if val is None:
                d[key] = None
            elif isinstance(val, torch.Tensor):
                if val.dim() == 0:
                    d[key] = val
                else:
                    cat_dim = data.__cat_dim__(key, val) if hasattr(data, "__cat_dim__") else 0
                    d[key] = val.narrow(cat_dim, start, end - start)
            else:
                d[key] = val[start:end]
        out.append(d)
    return out


def load_gdce(gdce_root):
    out = []
    for fn in ["train.pt", "val.pt", "test.pt"]:
        path = os.path.join(gdce_root, "processed", fn)
        data, slices = _load_gdce_pt(path)
        out += _make_list(data, slices)
    return out


def stats(name, dataset):
    n_nodes = [g.x.size(0) for g in dataset]
    feat_dim = dataset[0].x.size(1)
    ea = getattr(dataset[0], "edge_attr", None)
    bond_dim = ea.size(1) if ea is not None and ea.numel() > 0 else 0
    used_cols = set()
    for g in dataset:
        used_cols.update(g.x.argmax(dim=-1).tolist())
    print(f"\n=== {name} ===")
    print(f"  graphs:        {len(dataset)}")
    print(f"  atom feat dim: {feat_dim}")
    print(f"  bond feat dim: {bond_dim}")
    print(f"  node count:    min={min(n_nodes)} max={max(n_nodes)} mean={sum(n_nodes)/len(n_nodes):.1f}")
    print(f"  atom cols used:{sorted(used_cols)} (n={len(used_cols)})")
    # D4 encoding: H lives in column 3 (x_map_mutag). Only meaningful at 14-dim.
    if feat_dim == 14:
        has_H = any(3 in g.x.argmax(dim=-1).tolist() for g in dataset)
        print(f"  contains explicit H (col 3): {has_H}")
    return len(dataset), max(n_nodes), feat_dim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d4_path", default="data/MUTAG")
    p.add_argument("--gdce_root", required=True)
    args = p.parse_args()

    # GDCE atom decoder (ground truth for the 13 vs 14 question)
    dec_path = os.path.join(args.gdce_root, "raw", "atom_decoder.json")
    if os.path.exists(dec_path):
        with open(dec_path) as f:
            print(f"GDCE atom_decoder.json: {json.load(f)}")
    ntm_path = os.path.join(args.gdce_root, "raw", "node_type_map.npy")
    if os.path.exists(ntm_path):
        print(f"GDCE node_type_map.npy: {np.load(ntm_path, allow_pickle=True)}")

    print("\nLoading datasets (all splits concatenated)...")
    d4 = []
    for mode in ["training", "evaluation", "testing"]:
        d4 += list(MutagFiltered(args.d4_path, mode=mode))
    gdce = load_gdce(args.gdce_root)

    n_d4, max_d4, fd_d4 = stats("D4 MutagFiltered", d4)
    n_g, max_g, fd_g = stats("GDCE Mutagenicity", gdce)

    print(f"\n{'='*50}\nVERDICT\n{'='*50}")
    print(f"  same #graphs?   {n_d4 == n_g}  ({n_d4} vs {n_g})")
    print(f"  same max nodes? {max_d4 == max_g}  ({max_d4} vs {max_g})")
    print(f"  same atom dim?  {fd_d4 == fd_g}  ({fd_d4} vs {fd_g})")
    if fd_d4 != fd_g or n_d4 != n_g:
        print("  -> NOT the same dataset.")
        if max_d4 != max_g:
            print(f"     node-count range differs => '<=50' applied to different "
                  f"atom counts (likely H removed in GDCE).")


if __name__ == "__main__":
    main()
