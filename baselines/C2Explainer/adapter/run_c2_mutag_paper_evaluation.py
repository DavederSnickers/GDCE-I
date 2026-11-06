"""C2Explainer evaluation on the ORIGINAL (small) MUTAG dataset from the paper.

Uses the paper's own assets — `data/MUTAG.pickle` (TUDataset, 188 graphs,
7-dim atom features, 4-dim bond edge_attr) and the paper's GCN classifier
`models/GCN_MUTAG_sd.pt` — and runs the SAME evaluation protocol / metrics as
run_c2_nci1_evaluation.py (stratified multi-run, per-sample GED / sparsity /
#perturbs, run_summary.csv + c2_results.csv). This is the faithful reproduction
of the C2 paper's exp1_3 MUTAG experiment, wrapped in the comparable harness.

No GDCE dependency: the classifier uses C2's own edge_weight-aware GCNConv.
No SMILES metrics (the small MUTAG pickle is not the molecular `*_mol` build).

Classifier is selectable via `classifier=gcn|gine`:
  - gcn  (default) = the paper's own GCN (models/GCN_MUTAG_sd.pt)            -> run A
  - gine          = your GDCE GINE recipe trained on MUTAG (train_gine_mutag.py)
                    via gine_ckpt=...  -> run B, on the IDENTICAL test split.
Results are written under <out_dir>/<classifier>/ so A and B don't collide.

All knobs live in configs/mutag_paper.yaml. Override via CLI:
    apprun python run_c2_mutag_paper_evaluation.py n_runs=1 n_samples_per_run=5
    apprun python run_c2_mutag_paper_evaluation.py eval_split=all
    apprun python run_c2_mutag_paper_evaluation.py classifier=gine gine_ckpt=ckpt/gine_mutag/<best>.ckpt
    apprun python run_c2_mutag_paper_evaluation.py --help
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / 'src'))

from cf_explainer import C2Explainer
from cf_explainer.gcn_conv import GCNConv
from cf_explainer.utils import seed_everything

from wrappers.metrics import compute_ged, compute_sparsity
from wrappers.cf_io import make_record, save_counterfactuals
from wrappers.config import load_config

from torch_geometric.explain import Explainer
from torch_geometric.nn import global_max_pool


NUM_CLASSES = 2
CLASS_NAMES = {0: 'class0', 1: 'class1'}  # MUTAG TUDataset labels (0/1)


# ----------------------------------------------------- paper classifier

class PaperMUTAGGCN(nn.Module):
    """Exact GCN from the C2 paper's exp1_3 MUTAG notebook.

    3 x (C2 edge_weight-aware) GCNConv -> global_max_pool -> Linear(2).
    Architecture must match models/GCN_MUTAG_sd.pt (in_dim=7, 2 classes).
    """

    def __init__(self, nhid: int = 20, nout: int = 20, dropout: float = 0.0):
        super().__init__()
        self.conv1 = GCNConv(7, nhid, normalize=False)
        self.conv2 = GCNConv(nhid, nhid, normalize=False)
        self.conv3 = GCNConv(nhid, nout, normalize=False)
        self.lin = Linear(nout, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.conv3(x, edge_index, edge_weight)
        x = global_max_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


class PaperMUTAGGCNWrapperC2(nn.Module):
    """C2Explainer/PyG-Explainer-compatible wrapper around the paper GCN.

    forward(x, edge_index, batch=None, edge_weight=None); batch defaults to a
    single-graph batch so the readout works under the Explainer's per-graph call.
    """

    def __init__(self, ckpt_path: str, nhid: int = 20, nout: int = 20):
        super().__init__()
        self.clf = PaperMUTAGGCN(nhid=nhid, nout=nout, dropout=0.0)
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        self.clf.load_state_dict(sd)
        self.clf.eval()
        for p in self.clf.parameters():
            p.requires_grad_(False)

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.clf(x.float(), edge_index, batch, edge_weight=edge_weight)


# ----------------------------------------------------------------- helpers

def evaluate_one(model, data, explainer_alg, device, orig_pred):
    """Run the PyG Explainer on one graph; return (cf_edge_index, n_perturbs,
    prop_perturbs, success_flag, cf_pred).

    Success is NOT trusted from the explainer: degenerate 0-edit CFs are rejected
    and the flip is re-verified deterministically by re-classifying the discrete
    CF graph with the eval-mode model (same guard as run_c2_nci1_evaluation.py)."""
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
    cf_edge_index = explanation.cf
    n_perturbs = float(explanation.perturbs)
    if cf_edge_index is None or n_perturbs <= 0:
        return None, None, None, False, None
    with torch.no_grad():
        cf_logits = model(data.x.to(device), cf_edge_index.to(device))
    cf_pred = int(cf_logits.argmax(dim=-1).item())
    success = (cf_pred != orig_pred)
    return cf_edge_index, n_perturbs, float(explanation.prop_perturbs), success, cf_pred


def _resolve(p: str) -> Path:
    """Resolve a config path relative to the repo root unless it is absolute."""
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _set_orig_graph(model, data, edge_dim, device):
    """For the GINE classifier: cache the per-graph original bond-type lookup
    (no-op-able; only called when the model exposes set_orig_graph)."""
    ei = data.edge_index.to(device)
    ea = data.edge_attr.to(device) if data.edge_attr is not None \
        else torch.zeros((ei.size(1), edge_dim or 0), device=device)
    model.set_orig_graph(ei, ea, data.x.size(0))


# ----------------------------------------------------------------- main

def main():
    cfg = load_config(default_name='mutag_paper', script_dir=REPO_ROOT)
    print("=" * 60)
    print("Config:")
    from omegaconf import OmegaConf
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(cfg.cuda))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- model (selectable: paper GCN, or your GDCE GINE trained on MUTAG) ----
    classifier = str(cfg.get('classifier', 'gcn'))
    if classifier == 'gcn':
        ckpt = _resolve(cfg.ckpt)
        print(f"Loading PAPER GCN classifier: {ckpt}")
        model = PaperMUTAGGCNWrapperC2(str(ckpt), nhid=cfg.nhid, nout=cfg.nout).to(device).eval()
    elif classifier == 'gine':
        if not cfg.get('gine_ckpt', None):
            raise SystemExit("classifier=gine requires gine_ckpt=<path to the GINE "
                             "checkpoint from train_gine_mutag.py>")
        # The GINE class lives in GDCE; make it importable before the wrapper.
        if cfg.gdce_path not in sys.path:
            sys.path.insert(0, cfg.gdce_path)
        from wrappers.mutag_gine_wrapper import MutagGINEWrapperC2
        gine_ckpt = _resolve(str(cfg.gine_ckpt))
        print(f"Loading GINE classifier (your GDCE recipe, trained on MUTAG): {gine_ckpt}")
        model = MutagGINEWrapperC2(str(gine_ckpt)).to(device).eval()
    else:
        raise SystemExit(f"Unknown classifier={classifier!r} (use 'gcn' or 'gine')")

    needs_orig_graph = hasattr(model, 'set_orig_graph')
    edge_dim = getattr(model, 'edge_dim', None)

    # ---- dataset (original MUTAG.pickle) ----
    ds_path = _resolve(cfg.dataset_path)
    print(f"Loading dataset: {ds_path}")
    with open(ds_path, 'rb') as f:
        dataset = pickle.load(f)
    dataset.transform = None  # pickle ships a ToDevice('cuda') transform; we move tensors ourselves

    # Paper split: shuffle (seed 42, as in exp1_3) then take the last 20% as test.
    if str(cfg.eval_split) == 'all':
        pool = dataset
    else:
        torch.manual_seed(42)
        dataset = dataset.shuffle()
        n = len(dataset)
        pool = dataset[int(0.8 * n):]
    print(f"Eval pool ({cfg.eval_split}): {len(pool)} graphs (of {len(dataset)})")

    # ---- pre-classify (stratified sampling base) ----
    print("Pre-classifying eval pool...")
    all_samples = []
    with torch.no_grad():
        for idx in tqdm(range(len(pool))):
            d = pool[idx]
            if needs_orig_graph:
                _set_orig_graph(model, d, edge_dim, device)
            logits = model(d.x.to(device), d.edge_index.to(device))
            pred = int(logits.argmax(dim=-1).item())
            all_samples.append({'data_idx': idx, 'orig_pred': pred})
    class_totals = {c: sum(1 for s in all_samples if s['orig_pred'] == c)
                    for c in range(NUM_CLASSES)}
    print("  Class distribution: " +
          ", ".join(f"{CLASS_NAMES[c]}: {class_totals[c]}" for c in range(NUM_CLASSES)))

    # ---- multi-run ----
    out_dir = Path(cfg.out_dir) / classifier   # separate gcn/ vs gine/ results
    out_dir.mkdir(parents=True, exist_ok=True)
    run_seeds = list(range(42, 42 + cfg.n_runs))
    all_rows = []
    run_summary_rows = []
    cf_records = []   # paired (original, counterfactual) graphs

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
            data = pool[meta['data_idx']]
            n_nodes = data.x.size(0)
            n_orig_edges = data.edge_index.size(1) // 2
            orig_pred = meta['orig_pred']
            target_class = 1 - orig_pred

            if needs_orig_graph:
                _set_orig_graph(model, data, edge_dim, device)

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
            )

            t0 = time.time()
            try:
                cf_edge_index, n_perturbs, prop_perturbs, success, cf_pred = \
                    evaluate_one(model, data, algo, device, orig_pred)
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
                'node_sparsity': -1.0, 'edge_sparsity': -1.0,
            }

            if success and cf_edge_index is not None:
                orig_ei = data.edge_index.to(device)
                ged = compute_ged(orig_ei, cf_edge_index, n_nodes)
                mod_rate = ged / n_orig_edges if (ged >= 0 and n_orig_edges > 0) else -1.0
                node_sp, edge_sp, _ = compute_sparsity(orig_ei, cf_edge_index)
                row.update({
                    'ged': ged, 'mod_rate': mod_rate,
                    'node_sparsity': node_sp, 'edge_sparsity': edge_sp,
                })
                print(f"  [{sample_idx+1}/{len(selected)}] idx={meta['data_idx']} "
                      f"{CLASS_NAMES[orig_pred]}->{CLASS_NAMES[cf_pred]} "
                      f"perturbs={n_perturbs:.0f} ged={ged} t={elapsed:.1f}s")
            else:
                print(f"  [{sample_idx+1}/{len(selected)}] idx={meta['data_idx']} "
                      f"{CLASS_NAMES[orig_pred]} FAIL t={elapsed:.1f}s")

            cf_records.append(make_record(data, row, cf_edge_index=cf_edge_index))
            all_rows.append(row)

        # ---- per-run aggregate ----
        df_r = pd.DataFrame([r for r in all_rows if r['run'] == run_idx])
        succ = df_r[df_r['success']]

        def _mean(col):
            v = succ[succ[col] >= 0][col] if len(succ) > 0 else pd.Series([], dtype=float)
            return v.mean() if len(v) > 0 else float('nan')

        n_total = len(df_r); n_succ = int(df_r['success'].sum())
        run_summary_rows.append({
            'run': run_idx, 'seed': seed,
            'accuracy':          n_succ / n_total if n_total > 0 else float('nan'),
            'avg_ged':           _mean('ged'),
            'avg_mod_rate':      _mean('mod_rate'),
            'avg_n_perturbs':    _mean('n_perturbs'),
            'avg_prop_perturbs': _mean('prop_perturbs'),
            'avg_node_sparsity': _mean('node_sparsity'),
            'avg_edge_sparsity': _mean('edge_sparsity'),
            'avg_time_s':        _mean('time_s'),
            'n_total':           n_total, 'n_success': n_succ,
        })

    # ---- write CSVs ----
    pd.DataFrame(all_rows).to_csv(out_dir / 'c2_results.csv', index=False)
    pd.DataFrame(run_summary_rows).to_csv(out_dir / 'run_summary.csv', index=False)
    save_counterfactuals(cf_records, out_dir)
    print(f"\nResults saved to: {out_dir.resolve()}")

    # ---- final summary ----
    print(f"\n{'#'*70}\nFINAL SUMMARY: mean ± std across {cfg.n_runs} runs"
          f"  (n={cfg.n_samples_per_run} per run, stratified)\n{'#'*70}")
    df_sum = pd.DataFrame(run_summary_rows)
    metrics = [
        ('accuracy',          'Accuracy (validity)'),
        ('avg_n_perturbs',    'Avg #perturbs'),
        ('avg_prop_perturbs', 'Avg prop_perturbs (~ 1-similarity)'),
        ('avg_ged',           'Avg GED'),
        ('avg_mod_rate',      'Avg ModRate'),
        ('avg_node_sparsity', 'Avg NodeSparsity'),
        ('avg_edge_sparsity', 'Avg EdgeSparsity'),
        ('avg_time_s',        'Avg time / sample (s)'),
    ]
    for col, label in metrics:
        v = df_sum[col].dropna()
        if len(v) == 0:
            print(f"  {label:<40s}: N/A")
        else:
            std = v.std(ddof=1) if len(v) > 1 else 0.0
            print(f"  {label:<40s}: {v.mean():.4f} ± {std:.4f}")

    print("\nDONE")


if __name__ == '__main__':
    main()
