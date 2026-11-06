"""C2Explainer evaluation on Mutagenicity (mol) using GDCE's GINE classifier.

Mirrors GDCE/src/inversion_mutag_mol.py — same dataset, same oracle, same
per-sample metrics + SMILES-validity, so the resulting CSVs are directly
comparable.

All knobs live in configs/mutag.yaml. Override via CLI:
    apprun python run_c2_mutag_evaluation.py n_runs=1 n_samples_per_run=5
    apprun python run_c2_mutag_evaluation.py --config configs/mutag.cluster.yaml
    apprun python run_c2_mutag_evaluation.py --help

Notes:
  - C2Explainer modifies the edge SET only; bond types live on original
    edges and added edges get a default single-bond label (class index 1).
  - Default is add-mode (subgraph_mode=False), matching the C2 paper's
    exp1_3 notebook for MUTAG.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from cf_dump import CFDumpWriter, edges_from_edge_index, node_types_from_x
from cf_explainer import C2Explainer
from cf_explainer.utils import seed_everything

from wrappers.dataset_loader import load_test_dataset
from wrappers.mutag_gine_wrapper import MutagGINEWrapperC2
from wrappers.paper_gcn import PaperGCNWrapperC2
from wrappers.shared_judge_wrapper import SharedJudgeWrapperC2
from wrappers.cf_io import make_record, save_counterfactuals
from wrappers.metrics import (
    compute_ged, compute_sparsity, graph_to_smiles, smiles_n_fragments,
    diagnose_invalidity, INVALID_REASONS,
)
from wrappers.config import load_config

from torch_geometric.explain import Explainer


NUM_CLASSES = 2
CLASS_NAMES = {0: 'Mutagenic', 1: 'Non-mutagenic'}


# ----------------------------------------------------------------- helpers

def evaluate_one(model, data, explainer_alg, device):
    """Run PyG Explainer on one graph. Returns explanation fields."""
    pyg_explainer = Explainer(
        model=model,
        algorithm=explainer_alg,
        explanation_type='model',
        node_mask_type=None,
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
    )
    explanation = pyg_explainer(data.x.to(device), data.edge_index.to(device), batch=None)
    if not hasattr(explanation, 'perturbs') or explanation.perturbs is None:
        return None, None, None, False, None
    return (
        explanation.cf,
        float(explanation.perturbs),
        float(explanation.prop_perturbs),
        True,
        int(explanation.label),
    )


def cf_edge_attr_from_orig(orig_edge_index, orig_edge_attr, cf_edge_index, edge_dim, default_bond=1):
    """Compute edge_attr for the CF edge set: original bond types for kept
    edges, default single-bond for newly added edges."""
    n = int(max(orig_edge_index.max().item(), cf_edge_index.max().item()) + 1) \
        if (orig_edge_index.numel() > 0 or cf_edge_index.numel() > 0) else 0
    lookup = torch.zeros((n, n), dtype=torch.long, device=cf_edge_index.device)
    if orig_edge_index.numel() > 0:
        ea = orig_edge_attr.to(cf_edge_index.device)
        bonds = ea.argmax(dim=-1) if ea.dim() > 1 else ea.long()
        lookup[orig_edge_index[0].to(cf_edge_index.device),
               orig_edge_index[1].to(cf_edge_index.device)] = bonds
    if cf_edge_index.numel() == 0:
        return torch.zeros((0, edge_dim), device=cf_edge_index.device)
    bonds_cf = lookup[cf_edge_index[0], cf_edge_index[1]]
    bonds_cf = torch.where(bonds_cf == 0, torch.full_like(bonds_cf, default_bond), bonds_cf)
    return torch.nn.functional.one_hot(bonds_cf, num_classes=edge_dim).float()


# ----------------------------------------------------------------- main

def main():
    cfg = load_config(default_name='mutag', script_dir=REPO_ROOT)
    print("=" * 60)
    print("Config:")
    from omegaconf import OmegaConf
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(cfg.cuda))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # GDCE's `src` package must be importable before we build the wrapper
    # (it does `from src.models...`). dataset_loader adds gdce_path later, but
    # the model is constructed first.
    if cfg.gdce_path not in sys.path:
        sys.path.insert(0, cfg.gdce_path)

    # ---- dataset (loaded first so the GCN can size its input layer) ----
    print(f"Loading test dataset from {cfg.gdce_path}/{cfg.datadir}")
    # Dataset variant from config: 'mutagenicity_mol' or 'benzene'. Both share the
    # bond-typed CSV schema and ship atom_decoder.json, so the SMILES path is identical.
    _ds = str(cfg.get('gdce_dataset_name', 'mutagenicity_mol'))
    print(f"  GDCE dataset variant: {_ds}")
    test_dataset = load_test_dataset(cfg.gdce_path, _ds, cfg.datadir)
    print(f"Test set size: {len(test_dataset)}")
    in_dim = int(test_dataset[0].x.size(1))
    data_edge_dim = int(test_dataset[0].edge_attr.size(1)) \
        if test_dataset[0].edge_attr is not None else 5

    # ---- atom_decoder for SMILES ----
    atom_decoder = None
    decoder_path = Path(cfg.gdce_path) / cfg.datadir / 'raw' / 'atom_decoder.json'
    if decoder_path.exists():
        with open(decoder_path) as f:
            atom_decoder = json.load(f)
        print(f"Loaded atom_decoder: {len(atom_decoder)} atom types")
    else:
        print(f"WARNING: atom_decoder.json not found at {decoder_path} — SMILES disabled")

    # ---- model (selectable: your GINE [default], or paper-style GCN) ----
    classifier = str(cfg.get('classifier', 'gine'))
    if classifier == 'gine':
        print(f"Loading GINE classifier: {cfg.ckpt}")
        model = MutagGINEWrapperC2(cfg.ckpt).to(device).eval()
    elif classifier == 'shared_judge':
        # The one classifier every method in the comparison targets. Addressed by
        # prefix (<prefix>.npz + <prefix>.json), rebuilt from portable weights.
        jp = cfg.get('shared_judge_path', None)
        if not jp:
            raise SystemExit("classifier=shared_judge requires shared_judge_path=<prefix>")
        print(f"Loading SHARED JUDGE: {jp}")
        model = SharedJudgeWrapperC2(str(jp)).to(device).eval()
    elif classifier == 'gcn':
        if not cfg.get('gcn_ckpt', None):
            raise SystemExit("classifier=gcn requires gcn_ckpt=<path from train_paper_gcn.py>")
        gcn_in_dim = int(cfg.gcn_in_dim) if cfg.get('gcn_in_dim', None) else in_dim
        print(f"Loading paper-style GCN classifier: {cfg.gcn_ckpt} (in_dim={gcn_in_dim})")
        # Diagonal canonicalization. NOTE: measured 2026-07-22, neither
        # mutagenicity_mol nor benzene actually carries a self-loop diagonal in
        # edge_index (both 0 self-loops), contradicting the wrapper docstring.
        # Default stays 'full' so the published Mutagenicity numbers are unchanged;
        # benzene sets self_loops='none' in its config, matching how its oracle was
        # trained (via GDCE's SpectreGraphDataModule, i.e. without self-loops).
        _sl = str(cfg.get('self_loops', 'full'))
        _bn = bool(cfg.get('batchnorm', False))
        print(f"  PaperGCN: self_loops={_sl}, batchnorm={_bn}")
        model = PaperGCNWrapperC2(str(cfg.gcn_ckpt), in_dim=gcn_in_dim,
                                  nhid=int(cfg.get('nhid', 20)), nout=int(cfg.get('nout', 20)),
                                  self_loops=_sl,
                                  batchnorm=_bn).to(device).eval()
    else:
        raise SystemExit(f"Unknown classifier={classifier!r} (use 'gine' or 'gcn')")

    needs_orig_graph = hasattr(model, 'set_orig_graph')
    edge_dim = getattr(model, 'edge_dim', None) or data_edge_dim

    # ---- pre-classify (stratified sampling base) ----
    print("Pre-classifying test set...")
    all_samples = []
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset))):
            d = test_dataset[idx]
            n_nodes = d.x.size(0)
            if needs_orig_graph:
                model.set_orig_graph(d.edge_index.to(device),
                                     d.edge_attr.to(device) if d.edge_attr is not None
                                     else torch.zeros((d.edge_index.size(1), edge_dim), device=device),
                                     n_nodes)
            logits = model(d.x.to(device), d.edge_index.to(device))
            pred = int(logits.argmax(dim=-1).item())
            all_samples.append({'data_idx': idx, 'orig_pred': pred})
    class_totals = {c: sum(1 for s in all_samples if s['orig_pred'] == c)
                    for c in range(NUM_CLASSES)}
    print(f"  Class distribution: " +
          ", ".join(f"{CLASS_NAMES[c]}: {class_totals[c]}" for c in range(NUM_CLASSES)))

    # ---- multi-run ----
    out_dir = Path(cfg.out_dir) / classifier   # separate gine/ vs gcn/ results
    out_dir.mkdir(parents=True, exist_ok=True)
    run_seeds = list(range(42, 42 + cfg.n_runs))
    all_rows = []
    run_summary_rows = []
    cf_records = []   # paired (original, counterfactual) graphs

    # Graph-level dump in the format shared with GDCE and the other baselines, so
    # a second judge can score these counterfactuals without re-running C2.
    # bond_source='inherited': C2 edits the edge SET only and copies bond types
    # from the original, so its edge-type column says nothing about the method.
    cf_writer = CFDumpWriter(
        out_dir / 'cf_dump.jsonl', method='c2', dataset=_ds,
        judge=str(cfg.get('shared_judge_path', cfg.get('ckpt', 'unknown'))),
        bond_source='inherited', variant=classifier,
    )

    for run_idx, seed in enumerate(run_seeds):
        print(f"\n{'#'*70}\n  RUN {run_idx+1}/{cfg.n_runs}  (seed={seed})\n{'#'*70}")

        rng = np.random.default_rng(seed)
        by_class = defaultdict(list)
        for s in all_samples:
            by_class[s['orig_pred']].append(s)
        selected = []
        for c in range(NUM_CLASSES):
            n_c = round(cfg.n_samples_per_run * class_totals[c] / len(all_samples))
            n_c = min(n_c, len(by_class[c]))
            idxs = rng.choice(len(by_class[c]), size=n_c, replace=False)
            selected.extend([by_class[c][i] for i in idxs])
        rng.shuffle(selected)
        selected = selected[:cfg.n_samples_per_run]
        print(f"  Sampled {len(selected)} graphs:" +
              ', '.join(f" {CLASS_NAMES[c]}={sum(1 for s in selected if s['orig_pred']==c)}"
                        for c in range(NUM_CLASSES)))

        for sample_idx, meta in enumerate(selected):
            data = test_dataset[meta['data_idx']]
            n_nodes = data.x.size(0)
            n_orig_edges = data.edge_index.size(1) // 2
            orig_pred = meta['orig_pred']
            target_class = 1 - orig_pred

            orig_ei = data.edge_index.to(device)
            orig_ea = data.edge_attr.to(device) if data.edge_attr is not None \
                else torch.zeros((orig_ei.size(1), edge_dim), device=device)
            if needs_orig_graph:
                model.set_orig_graph(orig_ei, orig_ea, n_nodes)

            seed_everything(seed + sample_idx, deterministic=False)

            algo = C2Explainer(
                epochs=cfg.epochs,
                lr=cfg.lr,
                undirected=True,
                subgraph_mode=cfg.subgraph_mode,
                silent_mode=True,
                ENT_loss=cfg.ent_loss,
                desired_y=torch.tensor(target_class, device=device),
                beta=cfg.beta,      # -> coeffs['beta']  (perturb-loss weight = paper's lambda)
                gamma=cfg.gamma,    # -> coeffs['gamma'] (entropy-loss weight; only used if ent_loss)
                # C2's node-feature perturbation matrix (paper Sec. 4.2.2). Its own
                # __init__ defaults to False, and the C2 authors also disable it in
                # their experiments (Sec. 5.2) so that all baselines share an
                # edge-only action space. Exposed here so the method can be run with
                # the full action space its paper describes.
                FPM=bool(cfg.get('fpm', False)),
            )

            t0 = time.time()
            try:
                cf_edge_index, n_perturbs, prop_perturbs, success, cf_pred = \
                    evaluate_one(model, data, algo, device)
            except Exception as e:
                print(f"  [{sample_idx+1}/{len(selected)}] idx={meta['data_idx']} FAILED: {e}")
                success = False
                cf_edge_index = None; n_perturbs = None
                prop_perturbs = None; cf_pred = None
            elapsed = time.time() - t0

            row = {
                'run': run_idx, 'seed': seed,
                'sample_idx': sample_idx, 'data_idx': meta['data_idx'],
                'source_class': orig_pred, 'target_class': target_class,
                'orig_pred': orig_pred,
                'success': bool(success),
                'cf_pred': cf_pred if cf_pred is not None else -1,
                'n_perturbs': n_perturbs if n_perturbs is not None else -1,
                'prop_perturbs': prop_perturbs if prop_perturbs is not None else -1,
                'n_orig_edges': n_orig_edges,
                'time_s': elapsed,
                'ged': -1, 'mod_rate': -1.0,
                'node_sparsity': -1.0, 'edge_sparsity': -1.0, 'edgetype_sparsity': -1.0,
                'smiles_valid': False,
                'smiles_connected': False, 'n_fragments': -1,
                'orig_smiles': None, 'cf_smiles': None,
                # invalidity diagnostic (only meaningful for flipped-but-invalid CFs)
                'invalid_reason': '', 'n_added_edges': -1, 'added_edges_to_H': -1,
            }

            cf_edge_attr = None
            orig_smiles = cf_smiles = None
            if success and cf_edge_index is not None:
                ged = compute_ged(orig_ei, cf_edge_index, n_nodes)
                mod_rate = ged / n_orig_edges if (ged >= 0 and n_orig_edges > 0) else -1.0

                cf_edge_attr = cf_edge_attr_from_orig(orig_ei, orig_ea, cf_edge_index,
                                                      edge_dim=edge_dim)
                node_types = data.x.argmax(dim=-1)
                node_sp, edge_sp, edgetype_sp = compute_sparsity(
                    orig_ei, cf_edge_index,
                    orig_node_types=node_types, cf_node_types=node_types,
                    orig_edge_attr=orig_ea, cf_edge_attr=cf_edge_attr,
                )

                smiles_strict = None
                if atom_decoder is not None:
                    smiles_strict = graph_to_smiles(node_types, cf_edge_index.cpu(),
                                                    cf_edge_attr.cpu(), atom_decoder, strict=True)
                    orig_smiles = graph_to_smiles(node_types, orig_ei.cpu(),
                                                  orig_ea.cpu(), atom_decoder, strict=True)
                    cf_smiles = smiles_strict

                cf_writer.write(
                    run=run_idx, seed=seed, sample_idx=sample_idx,
                    data_idx=meta['data_idx'], source_class=orig_pred,
                    target_class=target_class,
                    node_types=node_types_from_x(data.x),
                    edges=edges_from_edge_index(cf_edge_index, cf_edge_attr),
                    own_pred=int(cf_pred), projected=True,
                )

                n_frag = smiles_n_fragments(smiles_strict)
                row.update({
                    'ged': ged, 'mod_rate': mod_rate,
                    'node_sparsity': node_sp, 'edge_sparsity': edge_sp,
                    'edgetype_sparsity': edgetype_sp,
                    'smiles_valid': smiles_strict is not None,
                    # graph_counterfactual.md: a flip's strict-valid SMILES is a
                    # single connected molecule iff it has exactly one fragment.
                    'smiles_connected': (smiles_strict is not None and n_frag == 1),
                    'n_fragments': n_frag,
                    'orig_smiles': orig_smiles, 'cf_smiles': cf_smiles,
                })

                # ---- WHY does an invalid CF break validity? ----
                # C2 edits only the edge set (no valence/aromaticity constraint,
                # no projection), so a flipped-but-invalid CF is explained by the
                # bond it added. Diagnose only the invalid ones.
                if atom_decoder is not None and smiles_strict is None:
                    diag = diagnose_invalidity(
                        node_types, cf_edge_index.cpu(), cf_edge_attr.cpu(),
                        atom_decoder, orig_edge_index=orig_ei.cpu())
                    row.update({
                        'invalid_reason':   diag['reason'],
                        'n_added_edges':    diag['n_added_edges'],
                        'added_edges_to_H': diag['added_edges_to_H'],
                    })
                print(f"  [{sample_idx+1}/{len(selected)}] idx={meta['data_idx']} "
                      f"{CLASS_NAMES[orig_pred]}->{CLASS_NAMES[cf_pred]} "
                      f"perturbs={n_perturbs:.0f} ged={ged} smi={'ok' if smiles_strict else 'no'} "
                      f"t={elapsed:.1f}s")
            else:
                print(f"  [{sample_idx+1}/{len(selected)}] idx={meta['data_idx']} "
                      f"{CLASS_NAMES[orig_pred]} FAIL t={elapsed:.1f}s")

            cf_records.append(make_record(
                data, row, cf_edge_index=cf_edge_index, cf_edge_attr=cf_edge_attr,
                orig_smiles=orig_smiles, cf_smiles=cf_smiles))
            all_rows.append(row)

        # ---- per-run aggregate ----
        df_r = pd.DataFrame([r for r in all_rows if r['run'] == run_idx])
        succ = df_r[df_r['success']]

        def _mean(col):
            v = succ[succ[col] >= 0][col] if len(succ) > 0 else pd.Series([], dtype=float)
            return v.mean() if len(v) > 0 else float('nan')

        def _rate(col):
            return succ[col].mean() if len(succ) > 0 else float('nan')

        n_total = len(df_r); n_succ = int(df_r['success'].sum())
        run_summary_rows.append({
            'run': run_idx, 'seed': seed,
            'accuracy':              n_succ / n_total if n_total > 0 else float('nan'),
            'avg_ged':               _mean('ged'),
            'avg_mod_rate':          _mean('mod_rate'),
            'avg_n_perturbs':        _mean('n_perturbs'),
            'avg_prop_perturbs':     _mean('prop_perturbs'),
            'avg_node_sparsity':     _mean('node_sparsity'),
            'avg_edge_sparsity':     _mean('edge_sparsity'),
            'avg_edgetype_sparsity': _mean('edgetype_sparsity'),
            'smiles_valid_rate':         _rate('smiles_valid'),
            'smiles_connected_rate':     _rate('smiles_connected'),  # valid_connected_over_flip
            'avg_n_fragments':           _mean('n_fragments'),       # valid_avg_fragments (over valid CFs)
            'avg_time_s':            _mean('time_s'),
            'n_total':               n_total, 'n_success': n_succ,
        })

    # ---- write CSVs ----
    cf_writer.close()
    pd.DataFrame(all_rows).to_csv(out_dir / 'c2_results.csv', index=False)
    pd.DataFrame(run_summary_rows).to_csv(out_dir / 'run_summary.csv', index=False)
    save_counterfactuals(cf_records, out_dir)
    print(f"\nResults saved to: {out_dir.resolve()}")

    # ---- final summary ----
    print(f"\n{'#'*70}\nFINAL SUMMARY: mean ± std across {cfg.n_runs} runs"
          f"  (n={cfg.n_samples_per_run} per run, stratified)\n{'#'*70}")
    df_sum = pd.DataFrame(run_summary_rows)
    metrics = [
        ('accuracy',                  'Accuracy'),
        ('smiles_valid_rate',         'SMILES valid (strict)'),
        ('smiles_connected_rate',     'Val SMILES connected'),
        ('avg_n_fragments',           'Avg #fragments'),
        ('avg_n_perturbs',            'Avg #perturbs'),
        ('avg_prop_perturbs',         'Avg prop_perturbs'),
        ('avg_ged',                   'Avg GED'),
        ('avg_mod_rate',              'Avg ModRate'),
        ('avg_node_sparsity',         'Avg NodeSparsity'),
        ('avg_edge_sparsity',         'Avg EdgeSparsity'),
        ('avg_edgetype_sparsity',     'Avg EdgeTypeSparsity'),
        ('avg_time_s',                'Avg time / sample (s)'),
    ]
    for col, label in metrics:
        v = df_sum[col].dropna()
        if len(v) == 0:
            print(f"  {label:<40s}: N/A")
        else:
            std = v.std(ddof=1) if len(v) > 1 else 0.0
            print(f"  {label:<40s}: {v.mean():.4f} ± {std:.4f}")

    # ---- WHY do flipped CFs break SMILES validity? (pooled over all runs) ----
    # C2 (add-mode) edits only the edge set with no valence/aromaticity
    # constraint and no projection onto chemical space, so every invalid flip is
    # explained by the bond it added. This breaks down the *invalid flips*.
    df_all = pd.DataFrame(all_rows)
    invalid = df_all[(df_all['success']) & (~df_all['smiles_valid'])]
    n_inv = len(invalid)
    print(f"\n{'#'*70}\nINVALIDITY BREAKDOWN  (over {n_inv} flipped-but-invalid CFs, pooled)\n{'#'*70}")
    if n_inv == 0:
        print("  No invalid flips — nothing to explain.")
    else:
        for reason in INVALID_REASONS + ('rdkit_missing',):
            frac = (invalid['invalid_reason'] == reason).mean()
            if frac > 0:
                print(f"  reason = {reason:<14s}: {frac:.3f}")
        ae = invalid[invalid['n_added_edges'] >= 0]
        if len(ae) > 0:
            with_h = (ae['added_edges_to_H'] > 0).mean()
            print(f"  {'avg #added edges':<22s}: {ae['n_added_edges'].mean():.2f}")
            print(f"  {'avg added edges to H':<22s}: {ae['added_edges_to_H'].mean():.2f}")
            print(f"  {'invalid w/ H-incident add':<22s}: {with_h:.3f}  "
                  f"(explicit H has valence 1 -> any added bond over-valences it)")

    print("\nDONE")


if __name__ == '__main__':
    main()
