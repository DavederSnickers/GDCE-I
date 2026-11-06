"""AdversarialFlipRate: does a counterfactual only fool the classifier it targeted?

A method optimises against ONE judge (seed 0). This script re-classifies its
counterfactuals with a second judge that is identical in every respect except the
training seed (seed 1) and was never optimised against. Two numbers come out:

    transfer rate  P(ctrl judge flips | base judge flips)   -- higher is better
    AdversarialFlipRate = 1 - transfer rate                 -- lower is better

Neither is interpretable on its own. Two judges disagree on *unmodified* test
graphs too, and that disagreement is the floor: a transfer rate at or below the
judges' baseline agreement says the counterfactuals are no more model-specific
than the models are to each other. The script therefore always reports the
agreement of both judges on the untouched test split alongside.

Two input paths:

  --dump   a cf_dump.jsonl written by cf_dump.CFDumpWriter. Exact: the graph the
           method produced, no serialisation loss. Use this whenever it exists.
  --csv    the legacy per-sample CSV, from which the counterfactual is rebuilt
           through its SMILES column. Lossy -- RDKit refuses graphs whose
           aromatic bonds form no perceivable ring, which silently drops ~70% of
           benzene's flips, and NCI1 has no SMILES column at all.

Either way the script first runs a fidelity gate: the BASE judge must reproduce
`target_class` on the reconstructed graphs it originally flipped. A gate below
1.0, or a recovery rate below 1.0, means the dump -- not the method -- is what is
being measured.

Usage:
    python adversarial_flip.py \\
        --dump results/shared_judge/mutag_gcn/cf_dump.jsonl \\
        --dataset mutagenicity_mol --data_root data \\
        --judge_base judges_final/judge_mutagenicity_mol_gcn \\
        --judge_ctrl judges_final/judge_mutagenicity_mol_gcn_seed1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cf_dump import read_dump  # noqa: E402
from shared_judge import NUM_EDGE_TYPES, collate, load_judge, load_split  # noqa: E402

N_EDGE_TYPES = 5

# Datasets whose CSVs store bonds in Kekule form (no aromatic bond type). RDKit
# aromatises on sanitisation, so those have to be kekulised back on read-in;
# datasets that genuinely use bond type 4 must keep the aromatic flags.
KEKULIZE = {"mutagenicity_mol": True, "benzene": False}


def _bond_map():
    from rdkit import Chem
    return {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
            Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}


def make_converter(data_root, dataset):
    """SMILES -> graph dict in the judge's input convention."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    with open(os.path.join(data_root, dataset, "raw", "atom_decoder.json")) as f:
        decoder = json.load(f)
    sym2idx = {v: int(k) for k, v in decoder.items()}
    n_types = len(decoder)
    bond2idx = _bond_map()
    kekulize = KEKULIZE.get(dataset, True)

    def convert(smiles):
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        try:
            # Same relaxed sanitisation the dump used when writing the SMILES:
            # valence checks would reject nitro groups, which carry no formal
            # charge in these datasets.
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL
                             ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception:
            pass
        if kekulize:
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Exception:
                pass

        types = []
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym not in sym2idx:
                return None
            types.append(sym2idx[sym])

        src, dst, bonds = [], [], []
        for bond in mol.GetBonds():
            t = bond2idx.get(bond.GetBondType())
            if t is None:
                return None
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [i, j]
            dst += [j, i]
            bonds += [t, t]

        ei = (np.array([src, dst], dtype=np.int64) if src
              else np.zeros((2, 0), dtype=np.int64))
        return {
            "x": np.eye(n_types, dtype=np.float32)[np.array(types, dtype=np.int64)],
            "edge_index": ei,
            "y": 0,
            "n": len(types),
            "edge_attr": (np.eye(N_EDGE_TYPES, dtype=np.float32)[np.array(bonds, dtype=np.int64)]
                          if bonds else np.zeros((0, N_EDGE_TYPES), dtype=np.float32)),
        }

    return convert


def graph_from_record(rec, n_node_types, n_edge_types):
    """A cf_dump record -> the same graph dict `collate` expects."""
    types = np.array(rec["node_types"], dtype=np.int64)
    edges = rec.get("edges") or []
    src, dst, bonds = [], [], []
    for i, j, b in edges:
        src += [i, j]
        dst += [j, i]
        bonds += [b, b]
    ei = (np.array([src, dst], dtype=np.int64) if src
          else np.zeros((2, 0), dtype=np.int64))
    g = {"x": np.eye(n_node_types, dtype=np.float32)[types],
         "edge_index": ei, "y": 0, "n": int(types.shape[0])}
    if n_edge_types > 0:
        g["edge_attr"] = (np.eye(n_edge_types, dtype=np.float32)[np.array(bonds, dtype=np.int64)]
                          if bonds else np.zeros((0, n_edge_types), dtype=np.float32))
    return g


def load_from_dump(path, data_root, dataset):
    """Read a cf_dump.jsonl. Nothing can be lost here -- that is the point."""
    raw = os.path.join(data_root, dataset, "raw")
    n_node_types = int(len(np.load(os.path.join(raw, "node_type_map.npy"))))
    n_edge_types = NUM_EDGE_TYPES.get(dataset, 0)

    records = read_dump(path)
    graphs, meta = [], []
    for rec in records:
        if max(rec["node_types"], default=0) >= n_node_types:
            raise SystemExit(
                f"[adv_flip] {path}: node type {max(rec['node_types'])} is outside "
                f"{dataset}'s vocabulary of {n_node_types}. The dumping repo is on a "
                f"different atom ordering -- the numbers would be meaningless.")
        graphs.append(graph_from_record(rec, n_node_types, n_edge_types))
        meta.append(rec)
    return graphs, meta


def predict(judge, graphs, device, batch=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(graphs), batch):
            x, ei, b, ea, _ = collate(graphs[i:i + batch], device)
            out.append(judge(x, ei, b, edge_attr=ea).argmax(dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def agreement_on_originals(data_root, dataset, judge_base, judge_ctrl, device):
    """Reference floor: how often do the two judges agree without any edit?"""
    graphs, _, _ = load_split(data_root, dataset, "test")
    y = np.array([g["y"] for g in graphs])
    a = predict(judge_base, graphs, device)
    b = predict(judge_ctrl, graphs, device)
    return dict(n=len(graphs), acc_base=float((a == y).mean()),
                acc_ctrl=float((b == y).mean()), agreement=float((a == b).mean()))


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--dump", help="cf_dump.jsonl (exact; preferred)")
    src.add_argument("--csv", help="legacy per-sample CSV, rebuilt via SMILES")
    p.add_argument("--dataset", required=True)
    p.add_argument("--data_root", default="data")
    p.add_argument("--judge_base", required=True, help="prefix without .npz")
    p.add_argument("--judge_ctrl", required=True, help="prefix without .npz")
    p.add_argument("--label", default=None, help="name for the summary row")
    p.add_argument("--smiles_col", default="flip_smiles")
    p.add_argument("--success_col", default="success")
    p.add_argument("--out_csv", default=None, help="write per-sample verdicts here")
    p.add_argument("--summary_csv", default=None,
                   help="append one summary row here (builds the paper table)")
    args = p.parse_args()

    import pandas as pd

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = load_judge(args.judge_base, device=device).eval()
    ctrl = load_judge(args.judge_ctrl, device=device).eval()

    n_empty = n_unparsable = 0
    if args.dump:
        graphs, records = load_from_dump(args.dump, args.data_root, args.dataset)
        kept = pd.DataFrame(records)
        n_succ = len(records)
    else:
        df = pd.read_csv(args.csv)
        if args.smiles_col not in df.columns:
            print(f"[adv_flip] {args.csv} has no column '{args.smiles_col}' — the run "
                  f"stored no counterfactual graph and cannot be re-judged. Re-run it "
                  f"with a graph-level dump.")
            return 2

        succ = df[df[args.success_col].astype(str).str.lower().isin(("true", "1"))].copy()
        convert = make_converter(args.data_root, args.dataset)

        graphs, keep = [], []
        for idx, row in succ.iterrows():
            smi = row[args.smiles_col]
            if not isinstance(smi, str) or smi in ("", "nan", "None"):
                n_empty += 1
                continue
            g = convert(smi)
            if g is None:
                n_unparsable += 1
                continue
            graphs.append(g)
            keep.append(idx)
        kept = succ.loc[keep].copy()
        n_succ = len(succ)

    target = kept["target_class"].to_numpy()
    pred_base = predict(base, graphs, device)
    pred_ctrl = predict(ctrl, graphs, device)

    gate = float((pred_base == target).mean()) if len(target) else float("nan")
    recovery = len(graphs) / max(n_succ, 1)

    kept["adv_pred_base"] = pred_base
    kept["adv_pred_ctrl"] = pred_ctrl
    kept["adv_flip_base"] = (pred_base == target).astype(int)
    kept["adv_flip_ctrl"] = (pred_ctrl == target).astype(int)

    # Only counterfactuals the base judge still flips after the round trip can say
    # anything about transfer; the rest are round-trip losses, not method failures.
    confirmed = kept[kept["adv_flip_base"] == 1]
    transfer = float(confirmed["adv_flip_ctrl"].mean()) if len(confirmed) else float("nan")

    # Per-run mean then mean over runs — the convention every other number in the
    # paper uses; a flat mean would weight runs by how many samples succeeded.
    if "run" in confirmed.columns and len(confirmed):
        per_run = confirmed.groupby("run")["adv_flip_ctrl"].mean()
        transfer_mean, transfer_std = float(per_run.mean()), float(per_run.std())
    else:
        transfer_mean, transfer_std = transfer, float("nan")

    ref = agreement_on_originals(args.data_root, args.dataset, base, ctrl, device)

    source = args.dump or args.csv
    name = args.label or os.path.basename(source)
    exact = bool(args.dump)
    print("=" * 78)
    print(f"AdversarialFlipRate — {name}  ({args.dataset})")
    print("=" * 78)
    print(f"  input fidelity   [{'graph dump, exact' if exact else 'SMILES round trip'}]")
    print(f"    counterfactuals in input        {n_succ}")
    print(f"    reconstructed                   {len(graphs)}  ({recovery:.1%})"
          + ("" if exact else f"   [empty {n_empty}, unparsable {n_unparsable}]"))
    print(f"    base judge reproduces target    {gate:.4f}   <- fidelity gate")
    if recovery < 0.95 or gate < 0.98:
        print(f"    WARNING: the surviving subset is not the run. Numbers below are")
        print(f"             conditioned on 'the CF happened to serialise cleanly'.")
    print(f"  transfer to the control judge (seed 1)")
    print(f"    confirmed flips analysed        {len(confirmed)}")
    print(f"    transfer rate  P(ctrl|base)     {transfer_mean:.4f} +/- {transfer_std:.4f}")
    print(f"    AdversarialFlipRate             {1 - transfer_mean:.4f}")
    print(f"  reference floor — untouched test graphs")
    print(f"    n={ref['n']}  acc base {ref['acc_base']:.4f}  acc ctrl {ref['acc_ctrl']:.4f}")
    print(f"    judge agreement                 {ref['agreement']:.4f}")
    print(f"    -> a transfer rate below {ref['agreement']:.3f} means the counterfactuals")
    print(f"       are MORE model-specific than the two models are to each other.")
    print("=" * 78)

    if args.out_csv:
        kept.to_csv(args.out_csv, index=False)
        print(f"[adv_flip] per-sample verdicts -> {args.out_csv}")

    if args.summary_csv:
        row = dict(label=name, dataset=args.dataset, source=source, exact=exact,
                   judge_base=os.path.basename(args.judge_base),
                   judge_ctrl=os.path.basename(args.judge_ctrl),
                   n_input=n_succ, n_reconstructed=len(graphs), recovery=recovery,
                   fidelity_gate=gate, n_confirmed=len(confirmed),
                   transfer_rate=transfer_mean, transfer_std=transfer_std,
                   adversarial_flip_rate=1 - transfer_mean,
                   judge_agreement_originals=ref["agreement"],
                   acc_base=ref["acc_base"], acc_ctrl=ref["acc_ctrl"])
        header = not os.path.exists(args.summary_csv)
        pd.DataFrame([row]).to_csv(args.summary_csv, mode="a",
                                   header=header, index=False)
        print(f"[adv_flip] summary row -> {args.summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
