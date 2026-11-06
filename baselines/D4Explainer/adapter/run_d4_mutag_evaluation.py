"""
Evaluate D4Explainer counterfactual explanations on Mutagenicity.

Mirrors GDCE/inversion_mutag_mol multi-run structure:
  - N_RUNS independent runs, each with N_SAMPLES_PER_RUN stratified molecules
  - Final report: mean ± std across runs for Flip Accuracy, Edge Sparsity, Mod Rate

Usage:
  python run_d4_mutag_evaluation.py \
    --gnn_path param/gnns/mutag_gine.pt \
    --d4_model results/mutag/best_model.pth
"""

import argparse
import os
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected

from cf_dump import CFDumpWriter, edges_from_edge_index, node_types_from_x
from datasets.gdce_mutag_dataset import GDCEMutagDataset
from datasets.gdce_benzene_dataset import GDCEBenzeneDataset
from datasets.mutag_dataset import Mutagenicity
from datasets.mutag_filtered_dataset import MutagFiltered
from gnns.mutag_gnn import Mutag_GCN
from explainers.diff_explainer import _gnn_get_pred
from explainers.diffusion.pgnn import Powerful
from explainers.diffusion.graph_utils import (
    graph2tensor,
    tensor2graph,
    gen_list_of_data_single,
    generate_mask,
    lookup_edge_attr,
)


NUM_CLASSES = 2
CLASS_NAMES = {0: 'Non-mutagenic', 1: 'Mutagenic'}
# Benzene's label is ring presence, not mutagenicity -- only affects printed
# reports, but a mislabelled log is worse than none.
CLASS_NAMES_BY_DATASET = {
    "gdce_benzene": {0: 'No benzene ring', 1: 'Benzene ring'},
}

# GDCE Mutagenicity_mol atom decoder: 13 remapped types (no Li). This is the
# contiguous remapping GDCE builds over the size-filtered subset from the
# authoritative TU node-label legend; it matches GDCE's atom_decoder.json and
# the 13-dim one-hot stored in the .pt files D4 reuses (GDCEMutagDataset).
GDCE_ATOM_DECODER = {
    0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br',
    7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Ca',
}
# Benzene uses the same 5-dim bond encoding but a different, 8-type atom
# vocabulary (P is dropped by the frequency filter). Both match the
# atom_decoder.json shipped with the corresponding GDCE dataset.
GDCE_BENZENE_ATOM_DECODER = {
    0: 'C', 1: 'N', 2: 'O', 3: 'S', 4: 'F', 5: 'Cl', 6: 'Br', 7: 'I',
}
# Which datasets use the GDCE encoding (remapped atom types + 5-dim bonds).
GDCE_DECODERS = {
    "gdce_mutag": GDCE_ATOM_DECODER,
    "gdce_benzene": GDCE_BENZENE_ATOM_DECODER,
}
# GDCE 5-dim bond one-hot: index 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic.
GDCE_SINGLE_BOND_IDX = 1


def gdce_cf_to_smiles(data, strict: bool = True, decoder: dict = None):
    """SMILES for a D4 counterfactual on the GDCE Mutagenicity dataset.

    Mirrors GDCE inversion_mutag_mol.graph_to_smiles: 13-type remapped atom
    decoder, 5-dim bond one-hot (1=single..4=aromatic), and nitro formal-charge
    handling (N with >=2 O neighbours -> N+/O-), which the Mutagenicity dataset
    needs to pass strict valence sanitization.

    Kept edges keep their original bond order; D4's newly added edges carry an
    all-zero edge_attr (lookup_edge_attr) and are projected onto a SINGLE bond.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import RWMol
    except ImportError:
        return None, False, "rdkit_not_installed"
    try:
        _BOND = {
            1: Chem.rdchem.BondType.SINGLE,
            2: Chem.rdchem.BondType.DOUBLE,
            3: Chem.rdchem.BondType.TRIPLE,
            4: Chem.rdchem.BondType.AROMATIC,
        }
        atom_idx = data.x.argmax(dim=-1).cpu().tolist()
        _dec = decoder if decoder is not None else GDCE_ATOM_DECODER
        node_syms = [_dec.get(i, 'C') for i in atom_idx]
        mol = RWMol()
        for sym in node_syms:
            mol.AddAtom(Chem.Atom(sym))

        ei = data.edge_index.cpu().numpy()
        ea_raw = getattr(data, 'edge_attr', None)
        if ea_raw is not None and ea_raw.numel() > 0:
            ea = ea_raw.cpu()
            bt_arr = ea.argmax(dim=-1).numpy()        # 0..4
            has_bond = (ea.sum(dim=-1) > 0).numpy()   # False for new (all-zero) edges
        else:
            bt_arr = has_bond = None

        bond_types_map = {}
        added = set()
        for k in range(ei.shape[1]):
            i, j = int(ei[0, k]), int(ei[1, k])
            if i >= j or (i, j) in added:
                continue
            if bt_arr is not None and has_bond[k]:
                bt = int(bt_arr[k])           # kept edge: original bond order
            else:
                bt = GDCE_SINGLE_BOND_IDX     # new D4 edge: project to single bond
            if bt in _BOND:
                mol.AddBond(i, j, _BOND[bt])
                bond_types_map[(i, j)] = bt
                added.add((i, j))

        # Nitro groups: N with >=2 O neighbours -> N+, O- (mirror GDCE).
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
                        if bt == 1:  # single bond -> O-
                            mol.GetAtomWithIdx(nb).SetFormalCharge(-1)
                            break

        if strict:
            Chem.SanitizeMol(mol)
        else:
            Chem.SanitizeMol(
                mol,
                Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        smi = Chem.MolToSmiles(mol)
        return (smi, True, None) if smi else (None, False, "empty_smiles")
    except Exception as e:
        return None, False, f"{type(e).__name__}: {e}"


def _mutag_cf_to_smiles(data, strict: bool = True):
    """SMILES for a counterfactual on the raw-TU mutag / mutag_filtered encoding.

    Uses the SAME sanitization as GDCE's gdce_cf_to_smiles (nitro N+/O- formal
    charges + Chem.SanitizeMol) so that *sanitizability is measured identically*;
    only the decoder differs (14-type x_map_mutag atoms, e_map_mutag bonds).
    Kept edges keep their original bond order; D4's newly added edges (all-zero
    edge_attr) are projected onto a SINGLE bond. Valence violations caused by
    that projection still fail sanitization (by design) -> counted as invalid.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import RWMol
        from explainers.visual import x_map_mutag, e_map_mutag
    except ImportError:
        return None, False, "rdkit_not_installed"
    try:
        atom_idx = data.x.argmax(dim=-1).cpu().tolist()
        node_syms = [x_map_mutag(i).name for i in atom_idx]
        mol = RWMol()
        for sym in node_syms:
            mol.AddAtom(Chem.Atom(sym))

        ei = data.edge_index.cpu().numpy()
        ea_raw = getattr(data, 'edge_attr', None)
        if ea_raw is not None and ea_raw.numel() > 0:
            ea = ea_raw.cpu()
            bt_idx = ea.argmax(dim=-1).numpy()
            has_bond = (ea.sum(dim=-1) > 0).numpy()   # False for new (all-zero) edges
        else:
            bt_idx = has_bond = None

        single_flags = {}
        added = set()
        for k in range(ei.shape[1]):
            i, j = int(ei[0, k]), int(ei[1, k])
            if i >= j or (i, j) in added:
                continue
            if bt_idx is not None and has_bond[k]:
                bond_type = e_map_mutag(int(bt_idx[k]), reverse=True)  # kept: original order
            else:
                bond_type = Chem.BondType.SINGLE                       # new D4 edge -> single
            mol.AddBond(i, j, bond_type)
            single_flags[(i, j)] = (bond_type == Chem.BondType.SINGLE)
            added.add((i, j))

        # Nitro groups: N with >=2 O neighbours -> N+, O- (mirror GDCE).
        nbrs = defaultdict(list)
        for (i, j), sgl in single_flags.items():
            nbrs[i].append((j, sgl))
            nbrs[j].append((i, sgl))
        for a, sym in enumerate(node_syms):
            if sym == 'N':
                o_nbrs = [(nb, sgl) for nb, sgl in nbrs[a] if node_syms[nb] == 'O']
                if len(o_nbrs) >= 2:
                    mol.GetAtomWithIdx(a).SetFormalCharge(1)
                    for nb, sgl in o_nbrs:
                        if sgl:  # single bond -> O-
                            mol.GetAtomWithIdx(nb).SetFormalCharge(-1)
                            break

        if strict:
            Chem.SanitizeMol(mol)
        else:
            Chem.SanitizeMol(
                mol,
                Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
            )
        smi = Chem.MolToSmiles(mol)
        return (smi, True, None) if smi else (None, False, "empty_smiles")
    except Exception as e:
        return None, False, f"{type(e).__name__}: {e}"


def cf_to_smiles_single_bond(data, dataset_name: str):
    """Build a SMILES string for a D4 counterfactual graph.

    Bonds that already existed in the original molecule keep their original bond
    order: tensor2graph carries the original edge_attr over to the kept edges
    (via lookup_edge_attr). D4Explainer only predicts graph *structure*, so the
    edges it newly adds carry no bond-type label (all-zero edge_attr) -> those
    are the only ones projected onto a SINGLE bond.

    Dataset-specific encoding, but the SAME sanitization in all cases (nitro
    N+/O- formal charges + Chem.SanitizeMol), so validity is comparable to GDCE:
      - gdce_mutag: 13-type remapped atom decoder + 5-dim bonds (-> gdce_cf_to_smiles).
      - mutag / mutag_filtered: raw TU 14-type one-hot (x_map_mutag, 0=C..12=Li,
        13=Ca) + e_map_mutag bonds (-> _mutag_cf_to_smiles).
    """
    if dataset_name in GDCE_DECODERS:
        return gdce_cf_to_smiles(data, strict=True, decoder=GDCE_DECODERS[dataset_name])
    return _mutag_cf_to_smiles(data, strict=True)


def measure_smiles_validity(dataset, dataset_name: str):
    try:
        from rdkit import Chem
    except ImportError:
        print("SMILES validity check skipped (rdkit not installed).")
        return
    total = len(dataset)
    n_valid = 0
    n_invalid = 0
    failure_reasons = {}
    bond_hist = defaultdict(int)   # argmax(edge_attr) -> count, over all edges
    for data in dataset:
        # Bond-type histogram over the real (un-projected) original edge_attr.
        # If index 4 (aromatic) dominates, manual AddBond(AROMATIC) without
        # flagging atoms aromatic will make SanitizeMol fail to kekulize.
        ea = getattr(data, 'edge_attr', None)
        if ea is not None and ea.numel() > 0:
            for b in ea.argmax(dim=-1).cpu().tolist():
                bond_hist[int(b)] += 1
        # Originals have real bond labels (no projection), but use the SAME
        # charge-aware reconstruction as the counterfactual metric so this
        # baseline check is on the same footing as GDCE. If originals are not
        # ~100% valid here, the reconstruction (not D4) is the problem.
        if dataset_name in GDCE_DECODERS:
            smi, valid, reason = gdce_cf_to_smiles(
                data, strict=True, decoder=GDCE_DECODERS[dataset_name])
        else:
            smi, valid, reason = _mutag_cf_to_smiles(data, strict=True)
        if valid:
            n_valid += 1
        else:
            n_invalid += 1
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    print(f"\n{'='*60}")
    print(f"SMILES VALIDITY ({dataset_name}, n={total})")
    print(f"  Valid SMILES: {n_valid}/{total}  ({100*n_valid/total:.1f}%)")
    print(f"  Invalid:      {n_invalid}/{total}  ({100*n_invalid/total:.1f}%)")
    _BOND_NAME = {0: 'no-bond', 1: 'single', 2: 'double', 3: 'triple', 4: 'aromatic'}
    if bond_hist:
        tot_b = sum(bond_hist.values())
        print("  Bond-type distribution (original edge_attr argmax):")
        for b in sorted(bond_hist):
            name = _BOND_NAME.get(b, f'idx{b}')
            print(f"    {b} ({name}): {bond_hist[b]}  ({100*bond_hist[b]/tot_b:.1f}%)")
    if failure_reasons:
        print("  Failure reasons:")
        for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
            print(f"    {count:5d}  {reason}")
    print(f"{'='*60}\n")


def parse_args():
    p = argparse.ArgumentParser(description="D4Explainer evaluation on Mutagenicity")
    p.add_argument("--device", type=str,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gnn_path", type=str, default="param/gnns/mutag_filtered_gcn.pt")
    p.add_argument("--d4_model", type=str, default="results/mutag_filtered/best_model.pth")
    p.add_argument("--dataset", type=str, default="mutag_filtered",
                   choices=["gdce_mutag", "gdce_benzene", "mutag", "mutag_filtered"])
    p.add_argument("--data_path", type=str, default="data/MUTAG")
    p.add_argument("--output_dir", type=str, default="results/mutag_evaluation")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.5])
    p.add_argument("--n_runs", type=int, default=5,
                   help="Number of independent evaluation runs (default: 5)")
    p.add_argument("--n_samples_per_run", type=int, default=100,
                   help="Molecules per run, stratified by classifier prediction (default: 100)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pyg_to_adj(edge_index, num_nodes):
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    ei = edge_index.cpu().numpy()
    for i in range(ei.shape[1]):
        adj[ei[0, i], ei[1, i]] = 1.0
    return adj


def compute_ged(adj_orig, adj_cf):
    def _to_nx(adj):
        G = nx.Graph()
        G.add_nodes_from(range(adj.shape[0]))
        rows, cols = np.where(adj >= 1)
        for i, j in zip(rows, cols):
            if i < j:
                G.add_edge(i, j)
        return G
    try:
        ged = nx.graph_edit_distance(_to_nx(adj_orig), _to_nx(adj_cf), timeout=5)
        return ged if ged is not None else -1
    except Exception:
        return -1


def compute_edge_sparsity(adj_orig, adj_cf):
    """Fraction of structurally changed edges (added + removed) relative to n_orig_edges."""
    upper_orig = np.triu(adj_orig > 0.5, k=1)
    upper_cf   = np.triu(adj_cf   > 0.5, k=1)
    n_orig = int(upper_orig.sum())
    structural_changed = int(np.abs(upper_orig.astype(float) - upper_cf.astype(float)).sum())
    return structural_changed / n_orig if n_orig > 0 else 0.0


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    N_RUNS = args.n_runs
    N_SAMPLES_PER_RUN = args.n_samples_per_run
    RUN_SEEDS = list(range(42, 42 + N_RUNS))

    print(f"N_RUNS={N_RUNS}, N_SAMPLES_PER_RUN={N_SAMPLES_PER_RUN}, seeds={RUN_SEEDS}")

    # --- Load test set ---
    if args.dataset == "gdce_mutag":
        test_dataset = GDCEMutagDataset(args.data_path, mode="testing")
    elif args.dataset == "gdce_benzene":
        test_dataset = GDCEBenzeneDataset(args.data_path, mode="testing")
    elif args.dataset == "mutag_filtered":
        test_dataset = MutagFiltered(args.data_path, mode="testing")
    else:
        test_dataset = Mutagenicity(args.data_path, mode="testing")
    print(f"Test set: {len(test_dataset)} graphs")

    measure_smiles_validity(test_dataset, args.dataset)

    # --- Load classifier ---
    # Same shared judge the explainer was trained against, so optimisation target
    # and evaluation classifier are one and the same (as for every other method
    # in the comparison). Judges are addressed by prefix and rebuilt from portable
    # weights rather than unpickled.
    from gnns.shared_judge_gnn import is_shared_judge_path, load_shared_judge_gnn
    if is_shared_judge_path(args.gnn_path):
        print(f"[eval] using SHARED JUDGE at {args.gnn_path}")
        gnn_model = load_shared_judge_gnn(args.gnn_path, device)
    else:
        gnn_model = torch.load(args.gnn_path, map_location=device)
    gnn_model = gnn_model.to(device)
    gnn_model.eval()
    print(f"Loaded {gnn_model.__class__.__name__} from {args.gnn_path}")

    # --- Verify classifier accuracy on full test set ---
    correct = total = 0
    class_counts = {0: 0, 1: 0}
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            edge_attr = getattr(batch, 'edge_attr', None)
            prob, _ = _gnn_get_pred(
                gnn_model, x=batch.x, edge_index=batch.edge_index,
                batch=batch.batch, edge_attr=edge_attr,
            )
            preds = prob.argmax(-1)
            correct += (preds == batch.y.view(-1)).sum().item()
            total += batch.y.size(0)
            for c in range(NUM_CLASSES):
                class_counts[c] += (batch.y.view(-1) == c).sum().item()
    print(f"Classifier accuracy on test set: {correct/total:.2%} ({correct}/{total})")
    _names = CLASS_NAMES_BY_DATASET.get(args.dataset, CLASS_NAMES)
    for c in range(NUM_CLASSES):
        print(f"  {_names[c]}: {class_counts[c]} graphs")

    # --- D4Explainer model config ---
    # Single source of truth for the node-feature width
    # (gdce_mutag 13, gdce_benzene 8, raw TU mutag 14).
    from constants import feature_dict
    feature_in = feature_dict.get(args.dataset, 14)

    class D4Args:
        task = "gc"
        sigma_length = 10; prob_low = 0.0; prob_high = 0.4; noise_list = None
        num_layers = 6; layers_per_conv = 1; n_hidden = 64
        cat_output = True; residual = False; noise_mlp = True
        normalization = "instance"; simplified = False; dropout = 0.001
        threshold = 0.5
    d4_args = D4Args()
    d4_args.dataset = args.dataset
    d4_args.feature_in = feature_in
    d4_args.device = device

    ckpt = torch.load(args.d4_model, map_location=device)

    # n_hidden is NOT the same for every dataset: main.py's PAPER_HPARAMS sets 64
    # for gdce_mutag but 128 for gdce_benzene and gdce_nci1. Read it off the
    # checkpoint instead of hard-coding it, so the two can never drift apart
    # again (a mismatch surfaces as a wall of load_state_dict size errors).
    _sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    _probe = _sd.get("final_lin.0.weight")
    if _probe is not None and _probe.shape[1] != d4_args.n_hidden:
        print(f"[eval] n_hidden from checkpoint: {_probe.shape[1]} "
              f"(script default was {d4_args.n_hidden})")
        d4_args.n_hidden = int(_probe.shape[1])
    print(f"Loaded D4Explainer model from {args.d4_model}")

    print("\nCheckpoint training-time metrics:")
    for k in ("train_loss", "eval acc", "eval sparsity"):
        if k in ckpt:
            print(f"  {k:16s}: {ckpt[k]}")

    # --- Empty-graph sanity check ---
    print("\nEmpty-graph sanity check (no edges, only node features):")
    from torch_geometric.data import Batch as _Batch
    empty_preds = {0: 0, 1: 0}
    n_check = min(20, len(test_dataset))
    with torch.no_grad():
        for i in range(n_check):
            g = test_dataset[i]
            empty = g.clone()
            empty.edge_index = torch.zeros((2, 0), dtype=torch.long)
            if hasattr(empty, "edge_attr") and empty.edge_attr is not None:
                empty.edge_attr = empty.edge_attr[:0]
            b = _Batch.from_data_list([empty]).to(device)
            b_edge_attr = getattr(b, 'edge_attr', None)
            prob, _ = _gnn_get_pred(
                gnn_model, x=b.x, edge_index=b.edge_index,
                batch=b.batch, edge_attr=b_edge_attr,
            )
            empty_preds[prob.argmax(-1).item()] += 1
    for c in range(NUM_CLASSES):
        print(f"  empty graph -> {CLASS_NAMES[c]}: {empty_preds[c]}/{n_check}")
    if max(empty_preds.values()) >= 0.9 * n_check:
        dom = max(empty_preds, key=empty_preds.get)
        print(f"  WARNING: empty graphs collapse to {CLASS_NAMES[dom]}. "
              f"D4 may exploit this by removing all edges.")

    # --- Pre-classify all test graphs for stratified sampling ---
    print(f"\nPre-classifying all {len(test_dataset)} test graphs...")
    all_test_samples = []
    single_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for data_idx, graph in enumerate(single_loader):
            graph = graph.to(device)
            if graph.is_directed():
                ea = getattr(graph, 'edge_attr', None)
                if ea is not None and ea.numel() > 0:
                    graph.edge_index, graph.edge_attr = to_undirected(
                        edge_index=graph.edge_index, edge_attr=ea)
                else:
                    graph.edge_index = to_undirected(edge_index=graph.edge_index)
            edge_attr = getattr(graph, 'edge_attr', None)
            prob, _ = _gnn_get_pred(
                gnn_model, x=graph.x, edge_index=graph.edge_index,
                batch=graph.batch, edge_attr=edge_attr,
            )
            orig_pred = prob.argmax(-1).item()
            all_test_samples.append({'data_idx': data_idx, 'orig_pred': orig_pred, 'graph': graph})

    class_totals = {c: sum(1 for s in all_test_samples if s['orig_pred'] == c)
                    for c in range(NUM_CLASSES)}
    print(f"  Class distribution: " +
          ", ".join(f"{CLASS_NAMES[c]}: {class_totals[c]}" for c in range(NUM_CLASSES)))

    # --- Multi-run evaluation ---
    all_rows = []

    # Graph-level dump in the format shared with GDCE and the other baselines.
    # bond_source='inherited': D4 thresholds an edge score, kept bonds keep their
    # original type and any newly added edge defaults to a single bond.
    cf_writer = CFDumpWriter(
        os.path.join(args.output_dir, 'cf_dump.jsonl'),
        method='d4',
        # GDCE dataset name, so dumps from different repos are joinable.
        dataset={'gdce_mutag': 'mutagenicity_mol', 'gdce_benzene': 'benzene',
                 'gdce_nci1': 'nci1'}.get(args.dataset, args.dataset), judge=args.gnn_path,
        bond_source='inherited', variant=os.path.basename(str(args.d4_model)),
    )
    run_summary_rows = []

    for run_idx, run_seed in enumerate(RUN_SEEDS):
        print(f"\n{'#'*70}")
        print(f"  RUN {run_idx + 1}/{N_RUNS}  (seed={run_seed})")
        print(f"{'#'*70}")

        # Stratified sampling: maintain classifier-predicted class ratio
        rng = np.random.default_rng(run_seed)
        by_class = defaultdict(list)
        for s in all_test_samples:
            by_class[s['orig_pred']].append(s)

        selected_meta = []
        for c in range(NUM_CLASSES):
            n_class = round(N_SAMPLES_PER_RUN * class_totals[c] / len(all_test_samples))
            n_class = min(n_class, len(by_class[c]))
            chosen_indices = rng.choice(len(by_class[c]), size=n_class, replace=False)
            selected_meta.extend([by_class[c][i] for i in chosen_indices])
        rng.shuffle(selected_meta)
        selected_meta = selected_meta[:N_SAMPLES_PER_RUN]

        print(f"  Sampled {len(selected_meta)} graphs:")
        for c in range(NUM_CLASSES):
            n = sum(1 for s in selected_meta if s['orig_pred'] == c)
            print(f"    {CLASS_NAMES[c]}: {n}")

        for threshold in args.thresholds:
            print(f"\n{'='*60}")
            print(f"  RUN {run_idx+1}  THRESHOLD = {threshold}")
            print(f"{'='*60}")

            d4_args.threshold = threshold
            model = Powerful(d4_args).to(device)
            model.load_state_dict(ckpt["model"])
            model.eval()

            for sample_idx, sample in enumerate(selected_meta):
                graph = sample['graph']
                orig_pred = sample['orig_pred']
                target_class = 1 - orig_pred
                n_nodes = graph.x.size(0)
                adj_orig = pyg_to_adj(graph.edge_index, n_nodes)
                n_orig_edges = int(adj_orig.sum() / 2)

                inv_success = False
                inv_ged = -1
                inv_mod_rate = -1.0
                inv_edge_sparsity = -1.0
                cf_smiles_valid = False
                cf_smiles = None
                cf_smiles_reason = None

                try:
                    with torch.no_grad():
                        adj_b, x_b = graph2tensor(graph, device=device)
                        x_b = x_b.to(device)
                        node_flag = adj_b.sum(-1).gt(1e-5).float()

                        sigma_list = list(np.random.uniform(
                            d4_args.prob_low, d4_args.prob_high, d4_args.sigma_length))

                        (x_exp, _, flag_exp, noise_adj, _) = gen_list_of_data_single(
                            x_b, adj_b, node_flag, sigma_list, d4_args)

                        noise_chunks = noise_adj.chunk(len(sigma_list), dim=0)
                        x_chunks = x_exp.chunk(len(sigma_list), dim=0)
                        flag_chunks = flag_exp.chunk(len(sigma_list), dim=0)

                        scores = []
                        for i, sigma in enumerate(sigma_list):
                            mask = generate_mask(flag_chunks[i])
                            s = model(
                                A=noise_chunks[i].to(device),
                                node_features=x_chunks[i].to(device),
                                mask=mask.to(device),
                                noiselevel=sigma,
                            ).to(device)
                            scores.append(s)

                        sub = tensor2graph(graph, scores, mask, threshold=threshold)
                        sub_edge_attr = getattr(sub, 'edge_attr', None)
                        sub_prob, _ = _gnn_get_pred(
                            gnn_model, x=sub.x, edge_index=sub.edge_index,
                            batch=sub.batch, edge_attr=sub_edge_attr,
                        )

                    y_cf_pred = sub_prob.argmax(-1).item()
                    inv_success = (y_cf_pred == target_class)

                    adj_cf = pyg_to_adj(sub.edge_index, n_nodes)
                    inv_ged = compute_ged(adj_orig, adj_cf)
                    inv_mod_rate = inv_ged / n_orig_edges if inv_ged >= 0 and n_orig_edges > 0 else -1.0
                    inv_edge_sparsity = compute_edge_sparsity(adj_orig, adj_cf)

                    # Kept bonds keep their original order; new D4 edges -> single bond.
                    cf_smiles, cf_smiles_valid, cf_smiles_reason = cf_to_smiles_single_bond(
                        sub, args.dataset)

                except Exception as e:
                    print(f"  Sample {sample_idx}: Error: {e}")

                # A valid CF molecule may still be shattered into many disconnected
                # fragments (D4 flips by removing bonds; removal is valence-safe, so
                # fragment soup trivially sanitizes). A chemically meaningful CF is a
                # SINGLE connected molecule -> no '.' in its canonical SMILES. Track
                # both the plain validity and the stricter connected-validity, plus
                # the fragment count as a diagnostic.
                cf_n_fragments = (cf_smiles.count('.') + 1) if cf_smiles else 0
                cf_smiles_valid_connected = bool(cf_smiles_valid and cf_n_fragments == 1)

                if inv_success and sub is not None:
                    cf_writer.write(
                        run=run_idx, seed=run_seed, sample_idx=sample_idx,
                        data_idx=sample['data_idx'], source_class=orig_pred,
                        target_class=target_class,
                        node_types=node_types_from_x(sub.x),
                        # zero_as_default: D4 predicts structure only, so the edges
                        # it ADDS carry an all-zero edge_attr row. Reading that as
                        # "no bond" would drop exactly the method's own edits; they
                        # are projected onto a single bond, which is what
                        # cf_to_smiles_single_bond does for the SMILES column.
                        edges=edges_from_edge_index(
                            sub.edge_index, getattr(sub, 'edge_attr', None),
                            zero_as_default=True),
                        own_pred=int(y_cf_pred), projected=True,
                        threshold=float(threshold),
                    )

                all_rows.append({
                    'run': run_idx,
                    'seed': run_seed,
                    'threshold': threshold,
                    'sample_idx': sample_idx,
                    'source_class': orig_pred,
                    'target_class': target_class,
                    'success': inv_success,
                    'ged': inv_ged,
                    'mod_rate': inv_mod_rate,
                    'edge_sparsity': inv_edge_sparsity,
                    'n_orig_edges': n_orig_edges,
                    'cf_smiles_valid': cf_smiles_valid,
                    'cf_smiles_valid_connected': cf_smiles_valid_connected,
                    'cf_n_fragments': cf_n_fragments,
                    'cf_smiles': cf_smiles,
                    'cf_smiles_reason': cf_smiles_reason,
                })

                status = "OK" if inv_success else "FAIL"
                ged_str = f"GED={inv_ged:.0f}" if inv_ged >= 0 else "GED=N/A"
                mr_str  = f"MR={inv_mod_rate:.3f}" if inv_mod_rate >= 0 else "MR=N/A"
                es_str  = f"ES={inv_edge_sparsity:.3f}" if inv_edge_sparsity >= 0 else ""
                print(f"  [{sample_idx+1}/{len(selected_meta)}] "
                      f"{CLASS_NAMES[orig_pred]}->{CLASS_NAMES[target_class]}"
                      f"  {status}  {ged_str}  {mr_str}  {es_str}")

            # Per-run summary for this (run, threshold)
            df_rt = pd.DataFrame([r for r in all_rows
                                  if r['run'] == run_idx and r['threshold'] == threshold])
            n_total = len(df_rt)
            n_succ = int(df_rt['success'].sum())
            df_succ = df_rt[df_rt['success']]

            def _mean(col):
                vals = df_succ[df_succ[col] >= 0][col] if n_succ > 0 else pd.Series([], dtype=float)
                return vals.mean() if len(vals) > 0 else float('nan')

            # Val.SMILES: fraction of valid counterfactual molecules (single-bond
            # projection). Primary metric is over SUCCESSFUL flips only, matching
            # GDCE's flip_smiles_valid_rate and C2's _rate(succ) -> comparable
            # across methods. The over-all variant is kept as a secondary signal.
            smiles_validity = df_succ['cf_smiles_valid'].mean() if n_succ > 0 else float('nan')
            smiles_validity_all = df_rt['cf_smiles_valid'].mean() if n_total > 0 else float('nan')
            # Stricter: valid AND a single connected molecule (no fragment soup).
            smiles_validity_connected = (
                df_succ['cf_smiles_valid_connected'].mean() if n_succ > 0 else float('nan'))
            smiles_validity_connected_all = (
                df_rt['cf_smiles_valid_connected'].mean() if n_total > 0 else float('nan'))
            # Avg fragment count among valid-flip CFs (diagnostic for shattering).
            df_valid_flips = df_succ[df_succ['cf_smiles_valid']]
            avg_n_fragments = (
                df_valid_flips['cf_n_fragments'].mean() if len(df_valid_flips) > 0 else float('nan'))

            run_summary_rows.append({
                'run': run_idx,
                'seed': run_seed,
                'threshold': threshold,
                'accuracy': n_succ / n_total if n_total > 0 else float('nan'),
                'avg_mod_rate': _mean('mod_rate'),
                'avg_edge_sparsity': _mean('edge_sparsity'),
                'smiles_validity': smiles_validity,
                'smiles_validity_all': smiles_validity_all,
                'smiles_validity_connected': smiles_validity_connected,
                'smiles_validity_connected_all': smiles_validity_connected_all,
                'avg_n_fragments': avg_n_fragments,
                'n_total': n_total,
                'n_success': n_succ,
            })
            print(f"\n  [RUN {run_idx+1}, thresh={threshold}] "
                  f"Accuracy={n_succ}/{n_total} "
                  f"({100*n_succ/n_total if n_total > 0 else 0:.1f}%)  "
                  f"AvgMR={_mean('mod_rate'):.4f}  "
                  f"AvgES={_mean('edge_sparsity'):.4f}  "
                  f"Val.SMILES(flips)={smiles_validity:.4f}  "
                  f"Val.SMILES(all)={smiles_validity_all:.4f}  "
                  f"Val.SMILES.conn(flips)={smiles_validity_connected:.4f}  "
                  f"AvgFrags={avg_n_fragments:.1f}")

    # --- Save CSVs ---
    cf_writer.close()

    df_inv = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.output_dir, "inversion_results.csv")
    df_inv.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # --- Why do counterfactual SMILES fail? (aggregate over all runs) ---
    if 'cf_smiles_reason' in df_inv.columns:
        n_cf = len(df_inv)
        n_cf_valid = int(df_inv['cf_smiles_valid'].sum())
        print(f"\n{'='*60}")
        print(f"COUNTERFACTUAL SMILES FAILURES (n={n_cf}, "
              f"valid={n_cf_valid} = {100*n_cf_valid/n_cf if n_cf else 0:.1f}%)")
        bad = df_inv[~df_inv['cf_smiles_valid']]['cf_smiles_reason'].fillna('unknown')
        # Collapse RDKit messages to their exception type for a compact histogram.
        bad_type = bad.str.split(':').str[0]
        for reason, count in bad_type.value_counts().items():
            print(f"    {count:5d}  {reason}")
        print(f"{'='*60}")

    df_run_summary = pd.DataFrame(run_summary_rows)
    run_csv = os.path.join(args.output_dir, "run_summary.csv")
    df_run_summary.to_csv(run_csv, index=False)
    print(f"Run summary saved to {run_csv}")

    # --- Final mean ± std across runs ---
    print(f"\n{'#'*90}")
    print(f"FINAL SUMMARY: MEAN ± STD ACROSS {N_RUNS} RUNS  "
          f"(n={N_SAMPLES_PER_RUN} per run, stratified)")
    print(f"{'#'*90}")

    METRICS = [
        ('accuracy',                     'Flip Accuracy'),
        ('avg_edge_sparsity',            'Avg Edge Sparsity'),
        ('avg_mod_rate',                 'Avg Mod Rate'),
        ('smiles_validity',              'Val.SMILES (flips)'),
        ('smiles_validity_all',          'Val.SMILES (all)'),
        ('smiles_validity_connected',    'Val.SMILES connected (flips)'),
        ('smiles_validity_connected_all','Val.SMILES connected (all)'),
        ('avg_n_fragments',              'Avg #fragments (valid flips)'),
    ]

    for thresh in args.thresholds:
        df_s = df_run_summary[df_run_summary['threshold'] == thresh]
        print(f"\n--- threshold = {thresh}  ({len(df_s)} runs) ---")
        for col, label in METRICS:
            vals = df_s[col].dropna()
            if len(vals) == 0:
                print(f"  {label:<30}: N/A")
            else:
                mean_v = vals.mean()
                std_v  = vals.std(ddof=1) if len(vals) > 1 else 0.0
                print(f"  {label:<30}: {mean_v:.4f} ± {std_v:.4f}")

    print(f"\nAll results saved in: {os.path.abspath(args.output_dir)}")
    print("DONE!")


if __name__ == "__main__":
    main()
