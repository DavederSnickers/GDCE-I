"""
Pinpoint WHY SMILES reconstruction fails for gdce_mutag.

Runs on the cluster (needs data/GDCE_MUTAG + rdkit):
    apprun python diagnose_smiles_failure.py --data_path data/GDCE_MUTAG --n 50

For each original test molecule it rebuilds an RWMol from (x, edge_index,
edge_attr) using the SAME 13-type decoder / 5-dim bond map as the evaluation,
then tries several sanitization strategies WITHOUT swallowing the RDKit error:

  strict          : Chem.SanitizeMol (what the eval does now)
  no_kekulize     : SanitizeMol minus SANITIZE_KEKULIZE
  no_valence      : SanitizeMol minus SANITIZE_PROPERTIES (valence/charge checks)
  set_aromatic    : flag atoms in aromatic bonds as aromatic, then strict
  no_sanitize     : skip sanitization entirely (MolToSmiles only)

The strategy that recovers the most molecules tells us the cause:
  -> no_kekulize / set_aromatic wins  => aromatic bonds (index 4) are the problem
  -> no_valence wins                  => valence/charge problem (need more formal charges)
  -> nothing wins                     => decoder/connectivity is wrong

It also prints the bond-type and atom histograms, and separately checks that
the single-bond projection (all-zero edge_attr row -> SINGLE) behaves correctly.
"""

import argparse
from collections import defaultdict

import torch

from datasets.gdce_mutag_dataset import GDCEMutagDataset
from run_d4_mutag_evaluation import GDCE_ATOM_DECODER, GDCE_SINGLE_BOND_IDX

from rdkit import Chem
from rdkit.Chem import RWMol

_BOND = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.AROMATIC,
}
_BOND_NAME = {0: 'no-bond', 1: 'single', 2: 'double', 3: 'triple', 4: 'aromatic'}


def build_mol(data, project_single=False):
    """Build an (unsanitized) RWMol + nitro N+/O- charges, exactly like the eval.

    project_single: if True, force EVERY bond to SINGLE (simulates D4 projecting
    a binary adjacency with no bond labels). Used to check the projection path.
    """
    atom_idx = data.x.argmax(dim=-1).cpu().tolist()
    node_syms = [GDCE_ATOM_DECODER.get(i, 'C') for i in atom_idx]
    mol = RWMol()
    for sym in node_syms:
        mol.AddAtom(Chem.Atom(sym))

    ei = data.edge_index.cpu().numpy()
    ea = getattr(data, 'edge_attr', None)
    if ea is not None and ea.numel() > 0:
        ea = ea.cpu()
        bt_arr = ea.argmax(dim=-1).numpy()
        has_bond = (ea.sum(dim=-1) > 0).numpy()
    else:
        bt_arr = has_bond = None

    bond_types_map = {}
    added = set()
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        if i >= j or (i, j) in added:
            continue
        if project_single:
            bt = GDCE_SINGLE_BOND_IDX
        elif bt_arr is not None and has_bond[k]:
            bt = int(bt_arr[k])
        else:
            bt = GDCE_SINGLE_BOND_IDX
        if bt in _BOND:
            mol.AddBond(i, j, _BOND[bt])
            bond_types_map[(i, j)] = bt
            added.add((i, j))

    # nitro N+/O- (same as eval)
    nbrs = defaultdict(list)
    for (i, j), bt in bond_types_map.items():
        nbrs[i].append((j, bt))
        nbrs[j].append((i, bt))
    for a, sym in enumerate(node_syms):
        if sym == 'N':
            o_nbrs = [(nb, bt) for nb, bt in nbrs[a] if node_syms[nb] == 'O']
            if len(o_nbrs) >= 2:
                mol.GetAtomWithIdx(a).SetFormalCharge(1)
                for nb, bt in o_nbrs:
                    if bt == 1:
                        mol.GetAtomWithIdx(nb).SetFormalCharge(-1)
                        break
    return mol


def try_strategy(mol, strategy):
    """Return (smiles_or_None, error_str_or_None) for a sanitization strategy."""
    m = RWMol(mol)
    try:
        if strategy == 'strict':
            Chem.SanitizeMol(m)
        elif strategy == 'no_kekulize':
            Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL
                             ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        elif strategy == 'no_valence':
            Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL
                             ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        elif strategy == 'set_aromatic':
            for b in m.GetBonds():
                if b.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                    b.GetBeginAtom().SetIsAromatic(True)
                    b.GetEndAtom().SetIsAromatic(True)
            Chem.SanitizeMol(m)
        elif strategy == 'no_sanitize':
            pass
        smi = Chem.MolToSmiles(m)
        return (smi, None) if smi else (None, "empty_smiles")
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="data/GDCE_MUTAG")
    p.add_argument("--n", type=int, default=50, help="molecules to inspect")
    args = p.parse_args()

    ds = GDCEMutagDataset(args.data_path, mode="testing")
    n = min(args.n, len(ds))
    print(f"Test set: {len(ds)} graphs; inspecting first {n}\n")

    # --- Bond / atom histograms over the whole test set ---
    bond_hist, atom_hist = defaultdict(int), defaultdict(int)
    for data in ds:
        ea = getattr(data, 'edge_attr', None)
        if ea is not None and ea.numel() > 0:
            for b in ea.argmax(dim=-1).tolist():
                bond_hist[int(b)] += 1
        for a in data.x.argmax(dim=-1).tolist():
            atom_hist[GDCE_ATOM_DECODER.get(int(a), '?')] += 1
    tb = sum(bond_hist.values())
    print("Bond-type distribution (whole test set):")
    for b in sorted(bond_hist):
        print(f"  {b} ({_BOND_NAME.get(b, b)}): {bond_hist[b]} ({100*bond_hist[b]/tb:.1f}%)")
    print("Atom distribution:")
    for a, c in sorted(atom_hist.items(), key=lambda kv: -kv[1]):
        print(f"  {a}: {c}")
    print()

    # --- Strategy comparison on originals ---
    strategies = ['strict', 'no_kekulize', 'no_valence', 'set_aromatic', 'no_sanitize']
    wins = {s: 0 for s in strategies}
    strict_errors = defaultdict(int)
    for idx in range(n):
        mol = build_mol(ds[idx])
        for s in strategies:
            smi, err = try_strategy(mol, s)
            if smi is not None:
                wins[s] += 1
            if s == 'strict' and err is not None:
                strict_errors[err.split(':')[0]] += 1

    print(f"Valid SMILES per strategy (out of {n} originals):")
    for s in strategies:
        print(f"  {s:14s}: {wins[s]}/{n}  ({100*wins[s]/n:.1f}%)")
    print("\nStrict-mode error types:")
    for err, c in sorted(strict_errors.items(), key=lambda kv: -kv[1]):
        print(f"  {c:4d}  {err}")

    # --- A few concrete strict failures with full message ---
    print("\nExample strict failures (full RDKit message):")
    shown = 0
    for idx in range(n):
        smi, err = try_strategy(build_mol(ds[idx]), 'strict')
        if err is not None:
            print(f"  graph[{idx}]: {err}")
            shown += 1
            if shown >= 8:
                break

    # --- Single-bond projection sanity ---
    print("\nSingle-bond projection check (force every bond SINGLE):")
    proj_valid = 0
    for idx in range(n):
        smi, _ = try_strategy(build_mol(ds[idx], project_single=True), 'strict')
        if smi is not None:
            proj_valid += 1
    print(f"  valid after forcing SINGLE: {proj_valid}/{n} ({100*proj_valid/n:.1f}%)")
    print("  (low here is expected: forcing single breaks aromatic/double valences;"
          " this is only the worst-case where D4 rewires everything)")


if __name__ == "__main__":
    main()
