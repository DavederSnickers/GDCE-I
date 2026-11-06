"""Why a method's counterfactuals fail strict SMILES sanitisation.

`graph_to_smiles(strict=True)` runs the full `Chem.SanitizeMol`; `strict=False`
runs the same thing minus `SANITIZE_PROPERTIES`, and SANITIZE_PROPERTIES is the
valence check. So the strict/relaxed gap is, by construction, exactly the set of
counterfactuals with an impossible valence -- nothing about kekulisation,
aromaticity or connectivity is in that number. This script says WHICH atoms, by
how much, and whether the edits responsible were additions or deletions.

It reports three things per dump:

  1. the RDKit error for every strict failure, bucketed;
  2. the over-valent atoms -- element, degree, permitted valence -- and how many
     of them are hydrogens, which can carry exactly one bond;
  3. a control: the ORIGINAL graphs pushed through the identical reconstruction
     path. Every method's counterfactual is written with all bonds single (no
     method in the comparison predicts bond types), and that convention has to
     be shown harmless before any failure can be blamed on the method. If the
     originals come back ~100% valid, the strict column is measuring edits.

Usage (inside the container, from the GRETEL root):
    python -m src.wrappers.smiles_violations \\
        --dump xplore_mutagenicity_mol_results/cf_dump.jsonl \\
        --config configs/mutagenicity_mol.xplore.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

# RDKit's default maximum valences, for the elements our molecular datasets use.
# A nitro nitrogen is charged to N+ by graph_to_smiles and may then carry 4.
MAX_VALENCE = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 1,
               'S': 6, 'P': 5, 'Na': 1, 'K': 1, 'Ca': 2}


def _bucket(msg: str) -> str:
    m = msg.lower()
    if 'valence' in m:
        return 'valence'
    if 'kekulize' in m:
        return 'kekulize'
    if 'aromatic' in m:
        return 'aromaticity'
    return 'other'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--limit', type=int, default=0, help='0 = all records')
    args = ap.parse_args()

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    from src.wrappers.gdce_bootstrap import ensure_gdce_on_src_path
    ensure_gdce_on_src_path(cfg.gdce_path)
    _sj = os.path.join(cfg.gdce_path, 'src', 'shared_judge')
    if _sj not in sys.path:
        sys.path.insert(0, _sj)

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    from src.wrappers import metrics
    from src.wrappers.dataset_loader import load_gdce_dataset

    atom_decoder = json.loads(
        (Path(cfg.gdce_path) / cfg.datadir / 'raw' / 'atom_decoder.json').read_text())

    ds = load_gdce_dataset(cfg.gdce_path, cfg.dataset_name, cfg.datadir,
                           name='viol', max_nodes=int(cfg.n_nodes))
    by_id = {int(i.id): i for i in ds.instances}

    records = [json.loads(l) for l in open(args.dump) if l.strip()]
    if args.limit:
        records = records[:args.limit]
    print(f"{len(records)} counterfactuals from {args.dump}")

    def orig_edges(inst, n_real):
        ei = inst.gdce_edge_index
        mask = (ei[0] != ei[1]) & (ei[0] < n_real) & (ei[1] < n_real)
        ei = ei[:, mask].numpy()
        return {(min(int(a), int(b)), max(int(a), int(b)))
                for a, b in zip(ei[0], ei[1])}

    def build(node_types, edges):
        """Reconstruct exactly as graph_to_smiles does: every bond single."""
        nt = torch.tensor(node_types, dtype=torch.long)
        if edges:
            e = np.array([[i, j] for i, j, _ in edges])
            ei = torch.tensor(np.concatenate([e, e[:, ::-1]]).T.copy(), dtype=torch.long)
        else:
            ei = torch.empty((2, 0), dtype=torch.long)
        ea = torch.zeros((ei.size(1), 5))
        ea[:, 1] = 1.0
        return nt, ei, ea

    def strict_error(nt, ei, ea):
        """Rerun the strict path and keep the exception text."""
        try:
            mol = Chem.RWMol()
            syms = [atom_decoder.get(str(int(t)), 'C') for t in nt.tolist()]
            for s in syms:
                mol.AddAtom(Chem.Atom(s))
            seen = set()
            nbrs = defaultdict(list)
            for a, b in zip(ei[0].tolist(), ei[1].tolist()):
                i, j = min(a, b), max(a, b)
                if i == j or (i, j) in seen:
                    continue
                seen.add((i, j))
                mol.AddBond(i, j, Chem.BondType.SINGLE)
                nbrs[i].append(j)
                nbrs[j].append(i)
            for idx, s in enumerate(syms):
                if s == 'N' and sum(1 for nb in nbrs[idx] if syms[nb] == 'O') >= 2:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
                    for nb in nbrs[idx]:
                        if syms[nb] == 'O':
                            mol.GetAtomWithIdx(nb).SetFormalCharge(-1)
                            break
            Chem.SanitizeMol(mol)
            return None, syms, nbrs
        except Exception as e:                      # noqa: BLE001
            return str(e), syms, nbrs

    buckets = Counter()
    over_by_elem = Counter()
    over_atoms_per_cf, added_per_cf, removed_per_cf = [], [], []
    added_touching_h, added_total = 0, 0
    n_strict, n_relaxed = 0, 0
    del_only_valid = 0
    orig_strict, orig_n = 0, 0
    seen_orig = set()

    for r in records:
        nt, ei, ea = build(r['node_types'], r['edges'])
        s = metrics.graph_to_smiles(nt, ei, ea, atom_decoder, strict=True)
        rx = metrics.graph_to_smiles(nt, ei, ea, atom_decoder, strict=False)
        n_strict += s is not None
        n_relaxed += rx is not None

        err, syms, nbrs = strict_error(nt, ei, ea)
        if err:
            buckets[_bucket(err)] += 1

        # Over-valent atoms. Every bond is single, so valence == degree; N+ from
        # the nitro rule is allowed one more.
        over = 0
        for idx, sym in enumerate(syms):
            cap = MAX_VALENCE.get(sym, 4)
            if sym == 'N' and sum(1 for nb in nbrs[idx] if syms[nb] == 'O') >= 2:
                cap = 4
            if len(nbrs[idx]) > cap:
                over += 1
                over_by_elem[f"{sym} (deg {len(nbrs[idx])} > {cap})"] += 1
        over_atoms_per_cf.append(over)

        inst = by_id.get(int(r['data_idx']))
        if inst is None:
            continue
        n_real = int(inst.n_real_nodes)
        o = orig_edges(inst, n_real)
        c = {(i, j) for i, j, _ in r['edges']}
        added, removed = c - o, o - c
        added_per_cf.append(len(added))
        removed_per_cf.append(len(removed))
        added_total += len(added)
        added_touching_h += sum(1 for i, j in added
                                if syms[i] == 'H' or syms[j] == 'H')

        # Would the counterfactual be valid if only the deletions were kept?
        keep = sorted(o - removed)
        nt2, ei2, ea2 = build(r['node_types'], [[i, j, 1] for i, j in keep])
        del_only_valid += metrics.graph_to_smiles(
            nt2, ei2, ea2, atom_decoder, strict=True) is not None

        # Control: the untouched original through the identical path.
        if int(r['data_idx']) not in seen_orig:
            seen_orig.add(int(r['data_idx']))
            nt3, ei3, ea3 = build(
                inst.gdce_x[:n_real].argmax(dim=-1).tolist(),
                [[i, j, 1] for i, j in sorted(o)])
            orig_strict += metrics.graph_to_smiles(
                nt3, ei3, ea3, atom_decoder, strict=True) is not None
            orig_n += 1

    n = len(records)
    print("=" * 74)
    print(f"strict valid   {n_strict}/{n}  ({n_strict/n:.1%})")
    print(f"relaxed valid  {n_relaxed}/{n}  ({n_relaxed/n:.1%})")
    print(f"  the gap is SANITIZE_PROPERTIES, i.e. the valence check, and nothing else")
    print("-" * 74)
    print("strict failures by RDKit error:")
    for k, v in buckets.most_common():
        print(f"    {k:<14} {v:5d}  ({v/n:.1%})")
    print("-" * 74)
    print(f"over-valent atoms per counterfactual: "
          f"mean {np.mean(over_atoms_per_cf):.2f}, max {max(over_atoms_per_cf)}, "
          f"zero in {sum(1 for x in over_atoms_per_cf if x == 0)}/{n}")
    print("most common violations:")
    for k, v in over_by_elem.most_common(12):
        print(f"    {k:<26} {v:5d}")
    print("-" * 74)
    if added_per_cf:
        print(f"edits: +{np.mean(added_per_cf):.2f} added / "
              f"-{np.mean(removed_per_cf):.2f} removed per counterfactual")
        if added_total:
            print(f"added edges incident to a hydrogen: {added_touching_h}/{added_total} "
                  f"({added_touching_h/added_total:.1%})   [H accepts exactly one bond]")
        print(f"valid if only the DELETIONS were kept: {del_only_valid}/{len(added_per_cf)} "
              f"({del_only_valid/len(added_per_cf):.1%})")
    print("-" * 74)
    if orig_n:
        print(f"CONTROL -- untouched originals through the same all-single-bond path: "
              f"{orig_strict}/{orig_n} ({orig_strict/orig_n:.1%}) strict valid")
        print("  a high number here means the reconstruction convention is not what fails")
    print("=" * 74)


if __name__ == '__main__':
    main()
