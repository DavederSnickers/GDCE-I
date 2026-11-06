"""SMILES validity ceiling for Benzene.

Reconstructs SMILES from the CSV representation of REAL molecules. The result is
the upper bound that generated graphs have to be measured against -- for
mutagenicity_mol it came out at 97.6% (strict) / 99.7% (relaxed).

The check matters most on Benzene: that dataset uses AROMATIC bonds (3.63% of all
matrix entries). RDKit sanitises an aromatic bond only if it is part of a valid
aromatic system -- unlike mutagenicity_mol, where every bond came in Kekule form
and class 4 never occurred. The ceiling can therefore sit well below 100% here.

Usage:
    PYTHONPATH=<repo> python src/check_benzene_smiles_ceiling.py
"""
import argparse
import ast
import csv
import json
import os
import sys

import numpy as np


def build_smiles(node_types, et_matrix, atom_decoder, strict=True):
    """Build an RDKit molecule from node types plus the bond-type matrix."""
    from rdkit import Chem
    from rdkit.Chem import RWMol

    bond_map = {
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
        4: Chem.rdchem.BondType.AROMATIC,
    }

    mol = RWMol()
    for t in node_types:
        mol.AddAtom(Chem.Atom(atom_decoder[str(int(t))]))

    n = len(node_types)
    for i in range(n):
        for j in range(i + 1, n):
            bt = int(et_matrix[i, j])
            if bt > 0:
                mol.AddBond(i, j, bond_map[bt])
                if bt == 4:
                    # An aromatic bond requires both atoms to be marked
                    # aromatic, or sanitisation fails.
                    mol.GetAtomWithIdx(i).SetIsAromatic(True)
                    mol.GetAtomWithIdx(j).SetIsAromatic(True)

    try:
        m = mol.GetMol()
        if strict:
            Chem.SanitizeMol(m)
        else:
            Chem.SanitizeMol(
                m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datadir', default='data/benzene/raw')
    ap.add_argument('--split', default='test.csv')
    ap.add_argument('--limit', type=int, default=2000)
    args = ap.parse_args()

    csv.field_size_limit(sys.maxsize)

    with open(os.path.join(args.datadir, 'atom_decoder.json')) as f:
        atom_decoder = json.load(f)
    print(f'atom_decoder: {atom_decoder}')

    n_ok_strict = n_ok_relaxed = n_total = 0
    n_aromatic_graphs = 0
    fail_examples = []

    with open(os.path.join(args.datadir, args.split), newline='') as f:
        for row in csv.DictReader(f):
            if n_total >= args.limit:
                break
            nt = ast.literal_eval(row['node_types'])
            et_flat = np.fromstring(row['edge_type_matrix'].strip('[]'), sep=',').astype(int)
            n = int(round(np.sqrt(et_flat.size)))
            et = et_flat.reshape(n, n)

            has_arom = bool((et == 4).any())
            n_aromatic_graphs += int(has_arom)

            s_strict = build_smiles(nt, et, atom_decoder, strict=True)
            s_relax = build_smiles(nt, et, atom_decoder, strict=False)
            n_ok_strict += int(s_strict is not None)
            n_ok_relaxed += int(s_relax is not None)
            if s_strict is None and len(fail_examples) < 5:
                fail_examples.append((row['graph_id'], row['y'], has_arom, n))
            n_total += 1

    print('=' * 66)
    print(f'SMILES CEILING on {args.split}  ({n_total} real molecules)')
    print('=' * 66)
    print(f'  Graphen mit aromatischen Bindungen: {n_aromatic_graphs}'
          f'  ({n_aromatic_graphs / n_total:.1%})')
    print(f'  strict  (volle Sanitisierung):      {n_ok_strict / n_total:.4f}'
          f'   ({n_ok_strict}/{n_total})')
    print(f'  relaxed (ohne SANITIZE_PROPERTIES): {n_ok_relaxed / n_total:.4f}'
          f'   ({n_ok_relaxed}/{n_total})')
    if fail_examples:
        print('\n  Fehlerbeispiele (graph_id, y, hat_aromatisch, n_nodes):')
        for ex in fail_examples:
            print(f'    {ex}')
    print('=' * 66)


if __name__ == '__main__':
    main()
