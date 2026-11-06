"""C2Explainer evaluation on NCI1 (no-freq filter) using GDCE's classifier.

Mirrors GDCE/src/inversion_nci1_nofreq.py — same dataset, same oracle,
same per-sample metrics and run aggregation — so the resulting CSVs are
directly comparable.

All knobs live in configs/nci1.yaml. Override via CLI:
    apprun python run_c2_nci1_evaluation.py n_runs=1 n_samples_per_run=5
    apprun python run_c2_nci1_evaluation.py --config configs/nci1.cluster.yaml
    apprun python run_c2_nci1_evaluation.py --help

Hyperparameters default to the C2Explainer paper notebooks
(epochs=1000, lr=0.1, undirected=True, subgraph_mode=false). NCI1 is not
in the original paper; we use the same defaults as MUTAG (graph
classification) since the dataset is structurally analogous.
"""

from __future__ import annotations

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
from wrappers.nci1_gin_wrapper import NCI1GINWrapperC2
from wrappers.paper_gcn import PaperGCNWrapperC2
from wrappers.shared_judge_wrapper import SharedJudgeWrapperC2
from wrappers.metrics import (
    compute_ged, compute_sparsity,
)
from wrappers.cf_io import make_record, save_counterfactuals
from wrappers.config import load_config

from torch_geometric.explain import Explainer


NUM_CLASSES = 2
CLASS_NAMES = {0: 'Class0', 1: 'Class1'}


# --------------------------------------------------------------- evaluation

def evaluate_one(model, data, explainer_alg, device, orig_pred):
    """Run the PyG Explainer on one graph; return (cf_edge_index, n_perturbs,
    prop_perturbs, success_flag, cf_pred).

    Success is NOT trusted from the explainer: we reject degenerate 0-edit
    "counterfactuals" and re-verify the flip deterministically by re-classifying
    the discrete CF graph with the (eval-mode) model. This guards against the
    self-loop/representation mismatch that produced phantom flips at ged=0."""
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
    # Reject degenerate CFs: a real counterfactual must change at least one edge.
    if cf_edge_index is None or n_perturbs <= 0:
        return None, None, None, False, None
    # Re-verify the flip on the discrete CF graph with the eval-mode model.
    with torch.no_grad():
        cf_logits = model(data.x.to(device), cf_edge_index.to(device))
    cf_pred = int(cf_logits.argmax(dim=-1).item())
    success = (cf_pred != orig_pred)
    return cf_edge_index, n_perturbs, float(explanation.prop_perturbs), success, cf_pred


# ----------------------------------------------------------------- main

def main():
    cfg = load_config(default_name='nci1', script_dir=REPO_ROOT)
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
    # gdce_dataset_name selects the GDCE preprocessing variant: 'nci1_nofreq'
    # (37 atom types) or 'nci1' (freq, Huang atom-freq>50 filter -> 10 types).
    gdce_dataset_name = str(cfg.get('gdce_dataset_name', 'nci1_nofreq'))
    print(f"Loading test dataset ({gdce_dataset_name}) from {cfg.gdce_path}/{cfg.datadir}")
    test_dataset = load_test_dataset(cfg.gdce_path, gdce_dataset_name, cfg.datadir)
    print(f"Test set size: {len(test_dataset)}")
    in_dim = int(test_dataset[0].x.size(1))

    # ---- model (selectable: your GIN [default], or paper-style GCN) ----
    classifier = str(cfg.get('classifier', 'gin'))
    if classifier == 'gin':
        print(f"Loading GIN classifier: {cfg.ckpt}")
        model = NCI1GINWrapperC2(cfg.ckpt).to(device).eval()
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
        # nci1_nofreq carries a full self-loop diagonal the GCN trained on -> 'full'
        model = PaperGCNWrapperC2(str(cfg.gcn_ckpt), in_dim=gcn_in_dim,
                                  nhid=int(cfg.get('nhid', 20)), nout=int(cfg.get('nout', 20)),
                                  self_loops='full',
                                  batchnorm=bool(cfg.get('batchnorm', False))).to(device).eval()
    else:
        raise SystemExit(f"Unknown classifier={classifier!r} (use 'gin' or 'gcn')")

    # ---- pre-classify (stratified sampling base) ----
    print("Pre-classifying test set...")
    all_samples = []
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset))):
            d = test_dataset[idx]
            logits = model(d.x.to(device), d.edge_index.to(device))
            pred = int(logits.argmax(dim=-1).item())
            all_samples.append({'data_idx': idx, 'orig_pred': pred})
    class_totals = {c: sum(1 for s in all_samples if s['orig_pred'] == c)
                    for c in range(NUM_CLASSES)}
    print(f"  Class distribution: " +
          ", ".join(f"{CLASS_NAMES[c]}: {class_totals[c]}" for c in range(NUM_CLASSES)))

    # ---- multi-run ----
    out_dir = Path(cfg.out_dir) / classifier   # separate gin/ vs gcn/ results
    out_dir.mkdir(parents=True, exist_ok=True)
    run_seeds = list(range(42, 42 + cfg.n_runs))
    all_rows = []
    run_summary_rows = []
    cf_records = []   # paired (original, counterfactual) graphs

    # Graph-level dump in the format shared with GDCE and the other baselines.
    # NCI1 has no bond types at all, so every edge is written as a single bond
    # and bond_source records that, rather than letting a bond-aware judge read
    # the constant as a prediction.
    cf_writer = CFDumpWriter(
        out_dir / 'cf_dump.jsonl', method='c2', dataset=gdce_dataset_name,
        judge=str(cfg.get('shared_judge_path', cfg.get('ckpt', 'unknown'))),
        bond_source='binary', variant=classifier,
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

            # one-time seed per sample to match C2's reproducibility expectations
            seed_everything(seed + sample_idx, deterministic=False)

            algo = C2Explainer(
                epochs=cfg.epochs,
                lr=cfg.lr,
                undirected=True,
                subgraph_mode=cfg.subgraph_mode,
                silent_mode=True,
                desired_y=torch.tensor(target_class, device=device),
                # C2's node-feature perturbation matrix (paper Sec. 4.2.2), off by
                # default in its own __init__ and in the C2 authors' experiments.
                FPM=bool(cfg.get('fpm', False)),
            )

            t0 = time.time()
            try:
                cf_edge_index, n_perturbs, prop_perturbs, success, cf_pred = \
                    evaluate_one(model, data, algo, device, orig_pred)
            except Exception as e:
                print(f"  [{sample_idx}/{len(selected)}] idx={meta['data_idx']} FAILED: {e}")
                success = False
                cf_edge_index = None
                n_perturbs = None
                prop_perturbs = None
                cf_pred = None
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
                cf_writer.write(
                    run=run_idx, seed=seed, sample_idx=sample_idx,
                    data_idx=meta['data_idx'], source_class=orig_pred,
                    target_class=target_class,
                    node_types=node_types_from_x(data.x),
                    edges=edges_from_edge_index(cf_edge_index),
                    own_pred=int(cf_pred), projected=True,
                )
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
            'accuracy':         n_succ / n_total if n_total > 0 else float('nan'),
            'avg_ged':          _mean('ged'),
            'avg_mod_rate':     _mean('mod_rate'),
            'avg_n_perturbs':   _mean('n_perturbs'),
            'avg_prop_perturbs': _mean('prop_perturbs'),
            'avg_node_sparsity': _mean('node_sparsity'),
            'avg_edge_sparsity': _mean('edge_sparsity'),
            'avg_time_s':       _mean('time_s'),
            'n_total':          n_total, 'n_success': n_succ,
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
        ('accuracy',          'Accuracy'),
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
