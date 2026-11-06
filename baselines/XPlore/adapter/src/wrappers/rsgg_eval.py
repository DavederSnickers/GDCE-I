"""Shared 5x100 evaluation driver for RSGG-CE on GDCE's Mutag / NCI1.

Mirrors the protocol of C2Explainer/run_c2_*_evaluation.py so the output CSVs
are directly comparable:
  * 5 runs, seeds 42..46
  * 100 graphs per run, stratified by the oracle-predicted class
  * per sample: counterfactual target = 1 - orig_pred, flip re-verified
    deterministically with the real GDCE classifier
  * GED can be -1 (timeout) -> filtered out when averaging

RSGG-CE (GraphCounteRGANExplainer) trains its per-class GAN once (cached to
disk), then explains each sampled test instance. The generator/discriminator
are sized to a fixed node count, so every graph is padded to `n_nodes`; metrics
and SMILES use only the molecule's real nodes.
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

NUM_CLASSES = 2
CLASS_NAMES = {0: 'Class0', 1: 'Class1'}
DEFAULT_BOND_CLASS = 1   # single bond (Mutag); RSGG-CE does not model bond types


# --------------------------------------------------------------- edge helpers

def molecular_edges(instance, n_real: int) -> torch.Tensor:
    """(2, E) undirected molecular edge_index over the real nodes (no self-loops)."""
    adj = instance.to_numpy_array()[:n_real, :n_real]
    src, dst = np.nonzero(np.triu(adj, k=1))
    if len(src) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    ei = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    return torch.tensor(ei, dtype=torch.long)


def _edge_set(edge_index: torch.Tensor) -> set:
    s = set()
    if edge_index.numel():
        ei = edge_index.cpu().numpy()
        for a, b in zip(ei[0], ei[1]):
            i, j = int(a), int(b)
            if i < j:
                s.add((i, j))
    return s


def n_perturbations(orig_ei: torch.Tensor, cf_ei: torch.Tensor) -> int:
    return len(_edge_set(orig_ei).symmetric_difference(_edge_set(cf_ei)))


# ------------------------------------------------------------- SMILES (Mutag)

def _orig_molecular_bonds(inst, n_real):
    """Original molecular edge_index + 5-class one-hot bond edge_attr (no self-loops)."""
    ei = inst.gdce_edge_index
    ea = inst.gdce_edge_attr
    mask = (ei[0] != ei[1]) & (ei[0] < n_real) & (ei[1] < n_real)
    return ei[:, mask], (ea[mask] if ea is not None else None)


def _smiles_records(inst, cf_ei, n_real, atom_decoder, metrics):
    """Return (orig_smiles, cf_strict, cf_relaxed). cf bonds default to single."""
    node_types = inst.gdce_x[:n_real].argmax(dim=-1)
    o_ei, o_ea = _orig_molecular_bonds(inst, n_real)
    orig_smiles = metrics.graph_to_smiles(node_types, o_ei, o_ea, atom_decoder, strict=False) \
        if o_ea is not None else None

    e = cf_ei.size(1)
    cf_ea = torch.zeros((e, 5))
    cf_ea[:, DEFAULT_BOND_CLASS] = 1.0
    cf_strict = metrics.graph_to_smiles(node_types, cf_ei, cf_ea, atom_decoder, strict=True)
    cf_relaxed = metrics.graph_to_smiles(node_types, cf_ei, cf_ea, atom_decoder, strict=False)
    return orig_smiles, cf_strict, cf_relaxed


# ------------------------------------------------------------------- main

def evaluate(cfg, kind: str, repo_root: Path):
    """kind in {'mutag','nci1'}."""
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(cfg.cuda))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    # Merge GDCE's src.* into GRETEL's src namespace (see gdce_bootstrap).
    from src.wrappers.gdce_bootstrap import ensure_gdce_on_src_path
    ensure_gdce_on_src_path(cfg.gdce_path)
    # GDCE's shared_judge/ carries no __init__.py and its modules import each
    # other by bare name, so the directory itself has to be importable -- the
    # same thing C2Explainer does by vendoring these two files.
    _sj = os.path.join(cfg.gdce_path, 'src', 'shared_judge')
    if _sj not in sys.path:
        sys.path.insert(0, _sj)

    from src.wrappers.dataset_loader import load_gdce_dataset
    from src.wrappers import metrics
    from src.wrappers.cf_io import make_record, save_counterfactuals
    from src.dataset.converters.weights_converter import DefaultFeatureAndWeightConverter
    from src.utils.samplers.partial_order_samplers import PositiveAndNegativeEdgeSampler
    from src.explainer.explainer_rsgg_ce import GraphCounteRGANExplainer

    # Whether SMILES can be decoded is a property of the dataset, not of its
    # name: benzene is molecular too, proteins and twitter are not. `kind` stays
    # for the two legacy drivers; `cfg.molecular` overrides it when set.
    is_mutag = bool(cfg.get('molecular', kind == 'mutag'))
    print("=" * 60)
    from omegaconf import OmegaConf
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    print(f"Device: {device}")

    # ---- dataset (GDCE train + test split, padded to n_nodes) ----
    ds = load_gdce_dataset(cfg.gdce_path, cfg.dataset_name, cfg.datadir,
                           name=f'rsgg_{cfg.dataset_name}', max_nodes=int(cfg.n_nodes))
    test_ids = ds.splits[0]['test']
    node_dim = int(ds.instances[test_ids[0]].gdce_x.size(1))
    print(f"Instances: {len(ds.instances)} (train={len(ds.splits[0]['train'])}, "
          f"test={len(test_ids)}); node_dim={node_dim}")

    # ---- oracle ----
    # 'gcn_tf' = GRETEL's paper classifier (TF GCN, trained here on GDCE's train
    #            split with gdce_x atom features).
    # 'gdce'   = GDCE's frozen GINE/GIN checkpoint (same oracle as GDCE/C2/D4).
    oracle_kind = str(cfg.get('oracle', 'gcn_tf'))
    print(f"Oracle: {oracle_kind}")

    atom_decoder = None
    if is_mutag:
        ad_path = Path(cfg.gdce_path) / cfg.datadir / 'raw' / 'atom_decoder.json'
        if ad_path.exists():
            atom_decoder = json.loads(ad_path.read_text())
        else:
            print(f"[warn] atom_decoder.json not found at {ad_path}; SMILES disabled")

    if oracle_kind == 'gcn_tf':
        from src.wrappers.gcn_tf_oracle import GCNTFOracle
        # ENCODER/DECODER INVARIANT: classifier node features (gdce_x columns),
        # SMILES node_types and atom_decoder.json must share one atom index space.
        if is_mutag and atom_decoder is not None:
            assert len(atom_decoder) == node_dim, (
                f"atom_decoder has {len(atom_decoder)} entries but gdce_x has "
                f"{node_dim} columns; classifier encoding and SMILES decoding would "
                f"disagree.")
        oracle = GCNTFOracle(id=0, n_node_types=node_dim, max_n_nodes=int(cfg.n_nodes),
                             oracle_store_path=cfg.explainer_store_path,
                             epochs=int(cfg.get('oracle_epochs', 30)))
    elif oracle_kind == 'gdce':
        from src.wrappers.gdce_oracle import GDCEMutagOracle, GDCENci1Oracle
        OracleCls = GDCEMutagOracle if is_mutag else GDCENci1Oracle
        oracle = OracleCls(id=0, ckpt_path=cfg.ckpt, node_dim=node_dim)
    elif oracle_kind == 'shared_judge':
        # The one classifier every method in the comparison targets. Addressed by
        # prefix (<prefix>.npz + <prefix>.json); binary adjacency, no bond types.
        jp = cfg.get('shared_judge_path', None)
        if not jp:
            raise SystemExit("oracle=shared_judge requires shared_judge_path=<prefix>")
        from src.wrappers.shared_judge_oracle import SharedJudgeOracle
        print(f"Loading SHARED JUDGE: {jp}")
        oracle = SharedJudgeOracle(id=0, judge_prefix=str(jp), node_dim=node_dim)
    else:
        raise ValueError(
            f"unknown oracle '{oracle_kind}' "
            f"(expected 'gcn_tf', 'gdce' or 'shared_judge')")

    # ---- explainer ----
    store = Path(cfg.explainer_store_path); store.mkdir(parents=True, exist_ok=True)
    explainer = GraphCounteRGANExplainer(
        id=0, explainer_store_path=str(store),
        converter=DefaultFeatureAndWeightConverter(feature_dim=int(cfg.feature_dim), weight_dim=1),
        n_nodes=int(cfg.n_nodes), n_labels=NUM_CLASSES,
        training_iterations=int(cfg.training_iterations),
        epochs=(int(cfg.epochs) if cfg.get('epochs', None) is not None else None),
        n_features=int(cfg.feature_dim), fold_id=int(cfg.fold_id),
        sampling_iterations=int(cfg.sampling_iterations),
        lr_generator=float(cfg.lr_generator), lr_discriminator=float(cfg.lr_discriminator),
        sampler=PositiveAndNegativeEdgeSampler(int(cfg.sampling_iterations)))

    # ---- graph-level dump in the schema shared with GDCE and the other
    #      baselines, so a second judge can score these counterfactuals later
    #      without re-running RSGG-CE. Flushed per record, so a job killed by the
    #      walltime keeps everything produced so far. ----
    from cf_dump import CFDumpWriter, edges_from_edge_index, node_types_from_x
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cf_writer = CFDumpWriter(
        out_dir / 'cf_dump.jsonl', method='rsgg-ce', dataset=str(cfg.dataset_name),
        judge=str(cfg.get('shared_judge_path', cfg.get('ckpt', 'unknown'))),
        bond_source='inherited', variant=f"oracle_{oracle_kind}")

    # ---- train / load the oracle (no-op for the pretrained GDCE checkpoint;
    #      trains the TF GCN on GDCE's train split for gcn_tf) ----
    oracle.fit(ds, int(cfg.fold_id))

    # ---- train the GAN once (feature-agnostic: no per-sample context) ----
    print("Training / loading RSGG-CE generator (feature-agnostic oracle context)...")
    oracle.clear_current_graph()
    explainer.fit(oracle, ds, int(cfg.fold_id))

    # ---- pre-classify the test split (stratification base) ----
    print("Pre-classifying test set...")
    all_samples = []
    for iid in tqdm(test_ids):
        inst = ds.instances[iid]
        oracle.set_current_graph(inst)
        all_samples.append({'iid': iid, 'orig_pred': oracle.predict(inst)})
    oracle.clear_current_graph()
    class_totals = {c: sum(1 for s in all_samples if s['orig_pred'] == c) for c in range(NUM_CLASSES)}
    print("  Class distribution: " +
          ", ".join(f"{CLASS_NAMES[c]}: {class_totals[c]}" for c in range(NUM_CLASSES)))

    # ---- multi-run ----
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    run_seeds = list(range(42, 42 + int(cfg.n_runs)))
    all_rows, run_summary_rows, cf_records = [], [], []

    for run_idx, seed in enumerate(run_seeds):
        print(f"\n{'#'*70}\n  RUN {run_idx+1}/{cfg.n_runs}  (seed={seed})\n{'#'*70}")
        rng = np.random.default_rng(seed)
        by_class = defaultdict(list)
        for s in all_samples:
            by_class[s['orig_pred']].append(s)
        selected = []
        for c in range(NUM_CLASSES):
            n_c = round(int(cfg.n_samples_per_run) * class_totals[c] / len(all_samples))
            n_c = min(n_c, len(by_class[c]))
            idxs = rng.choice(len(by_class[c]), size=n_c, replace=False)
            selected.extend([by_class[c][i] for i in idxs])
        rng.shuffle(selected)
        selected = selected[:int(cfg.n_samples_per_run)]
        print(f"  Sampled {len(selected)} graphs:" +
              ', '.join(f" {CLASS_NAMES[c]}={sum(1 for s in selected if s['orig_pred']==c)}"
                        for c in range(NUM_CLASSES)))

        for sample_idx, meta in enumerate(selected):
            inst = ds.instances[meta['iid']]
            n_real = int(inst.n_real_nodes)
            orig_pred = meta['orig_pred']
            target_class = 1 - orig_pred
            orig_ei = molecular_edges(inst, n_real)
            n_orig_edges = len(_edge_set(orig_ei))

            torch.manual_seed(seed + sample_idx)
            np.random.seed(seed + sample_idx)

            oracle.set_current_graph(inst)
            t0 = time.time()
            try:
                cf_instance = explainer.explain(inst, oracle, ds)
                cf_ei = molecular_edges(cf_instance, n_real)
                cf_pred = oracle.predict(cf_instance)
            except Exception as e:
                print(f"  [{sample_idx+1}/{len(selected)}] iid={meta['iid']} FAILED: {e}")
                cf_instance, cf_ei, cf_pred = None, None, None
            elapsed = time.time() - t0

            n_perturbs = n_perturbations(orig_ei, cf_ei) if cf_ei is not None else 0

            # PADDING AUDIT. RSGG-CE's sampler builds both its edge pool and its
            # "missing edge" fallback pool over the PADDED node count, so most
            # candidate additions join nodes that are not part of the molecule.
            # The judge never sees them (it slices to n_real) and neither do the
            # metrics, so they are silently discarded -- which would make the
            # method look like a pure deleter. This counts what is lost.
            pad_edges = -1
            if cf_instance is not None:
                _full = cf_instance.to_numpy_array()
                _all = int(np.count_nonzero(np.triu(_full, k=1)))
                pad_edges = _all - len(_edge_set(cf_ei))
            success = (cf_pred is not None) and (cf_pred != orig_pred) and (n_perturbs > 0)

            row = {
                'run': run_idx, 'seed': seed, 'sample_idx': sample_idx, 'data_idx': meta['iid'],
                'source_class': orig_pred, 'target_class': target_class, 'orig_pred': orig_pred,
                'success': bool(success), 'cf_pred': cf_pred if cf_pred is not None else -1,
                'n_perturbs': n_perturbs,
                'prop_perturbs': (n_perturbs / n_orig_edges) if n_orig_edges > 0 else -1.0,
                'n_orig_edges': n_orig_edges, 'time_s': elapsed, 'pad_edges': pad_edges,
                'ged': -1, 'mod_rate': -1.0, 'node_sparsity': -1.0,
                'edge_sparsity': -1.0, 'edgetype_sparsity': -1.0,
                'smiles_valid_strict': -1, 'smiles_valid_relaxed': -1,
            }

            cf_smiles = None
            if success:
                ged = metrics.compute_ged(orig_ei, cf_ei, n_real)
                node_sp, edge_sp, et_sp = metrics.compute_sparsity(orig_ei, cf_ei)
                row.update({
                    'ged': ged,
                    'mod_rate': (ged / n_orig_edges) if (ged >= 0 and n_orig_edges > 0) else -1.0,
                    'node_sparsity': node_sp, 'edge_sparsity': edge_sp, 'edgetype_sparsity': et_sp,
                })
                if is_mutag and atom_decoder is not None:
                    _, cf_strict, cf_relaxed = _smiles_records(inst, cf_ei, n_real, atom_decoder, metrics)
                    cf_smiles = cf_relaxed
                    row['smiles_valid_strict'] = int(cf_strict is not None)
                    row['smiles_valid_relaxed'] = int(cf_relaxed is not None)
                print(f"  [{sample_idx+1}/{len(selected)}] iid={meta['iid']} "
                      f"{CLASS_NAMES[orig_pred]}->{CLASS_NAMES[cf_pred]} "
                      f"perturbs={n_perturbs} ged={row['ged']} t={elapsed:.1f}s")
            else:
                print(f"  [{sample_idx+1}/{len(selected)}] iid={meta['iid']} "
                      f"{CLASS_NAMES[orig_pred]} FAIL t={elapsed:.1f}s")

            oracle.clear_current_graph()
            if success:
                # RSGG-CE edits only the edge set, so node types are the input's
                # and every new edge inherits the default bond class.
                _x = inst.gdce_x[:n_real]
                cf_writer.write(
                    run=run_idx, seed=seed, sample_idx=sample_idx,
                    data_idx=int(meta['iid']), source_class=int(orig_pred),
                    target_class=int(target_class),
                    node_types=node_types_from_x(_x),
                    edges=edges_from_edge_index(
                        cf_ei, edge_attr=None, default_bond=DEFAULT_BOND_CLASS),
                    own_pred=int(cf_pred), projected=True)
            cf_records.append(make_record(_orig_data_view(inst, n_real), row,
                                          cf_edge_index=cf_ei, cf_smiles=cf_smiles))
            all_rows.append(row)

        # ---- per-run aggregate ----
        df_r = pd.DataFrame([r for r in all_rows if r['run'] == run_idx])
        succ = df_r[df_r['success']]

        def _mean(col):
            v = succ[succ[col] >= 0][col] if len(succ) else pd.Series([], dtype=float)
            return v.mean() if len(v) else float('nan')

        n_total = len(df_r); n_succ = int(df_r['success'].sum())
        summary = {
            'run': run_idx, 'seed': seed,
            'accuracy': n_succ / n_total if n_total else float('nan'),
            'avg_ged': _mean('ged'), 'avg_mod_rate': _mean('mod_rate'),
            'avg_n_perturbs': _mean('n_perturbs'), 'avg_prop_perturbs': _mean('prop_perturbs'),
            'avg_node_sparsity': _mean('node_sparsity'), 'avg_edge_sparsity': _mean('edge_sparsity'),
            'avg_time_s': _mean('time_s'), 'n_total': n_total, 'n_success': n_succ,
        }
        if is_mutag:
            summary['smiles_valid_strict_rate'] = _mean('smiles_valid_strict')
            summary['smiles_valid_relaxed_rate'] = _mean('smiles_valid_relaxed')
        run_summary_rows.append(summary)

    # ---- write CSVs ----
    pd.DataFrame(all_rows).to_csv(out_dir / 'rsgg_results.csv', index=False)
    pd.DataFrame(run_summary_rows).to_csv(out_dir / 'run_summary.csv', index=False)
    save_counterfactuals(cf_records, out_dir)
    print(f"\nResults saved to: {out_dir.resolve()}")

    # ---- final summary ----
    print(f"\n{'#'*70}\nFINAL SUMMARY: mean +/- std across {cfg.n_runs} runs"
          f"  (n={cfg.n_samples_per_run} per run, stratified)\n{'#'*70}")
    df_sum = pd.DataFrame(run_summary_rows)
    labels = [
        ('accuracy', 'Flip-Accuracy'), ('avg_n_perturbs', 'Avg #perturbs'),
        ('avg_prop_perturbs', 'Avg prop_perturbs'), ('avg_ged', 'Avg GED'),
        ('avg_mod_rate', 'Avg ModRate'), ('avg_node_sparsity', 'Avg NodeSparsity'),
        ('avg_edge_sparsity', 'Avg EdgeSparsity'), ('avg_time_s', 'Avg time/sample (s)'),
    ]
    if is_mutag:
        labels += [('smiles_valid_strict_rate', 'SMILES valid (strict)'),
                   ('smiles_valid_relaxed_rate', 'SMILES valid (relaxed)')]
    for col, label in labels:
        v = df_sum[col].dropna() if col in df_sum else pd.Series([], dtype=float)
        if len(v) == 0:
            print(f"  {label:<34s}: N/A")
        else:
            std = v.std(ddof=1) if len(v) > 1 else 0.0
            print(f"  {label:<34s}: {v.mean():.4f} +/- {std:.4f}")
    print("\nNOTE: NodeSparsity=0 and (NCI1) EdgeTypeSparsity=0 by construction "
          "(RSGG-CE edits only edges). MMD not reported (single CF per sample).")
    print("DONE")


class _OrigDataView:
    """Minimal stand-in so cf_io.make_record can read x / edge_index / edge_attr."""
    def __init__(self, x, edge_index, edge_attr):
        self.x, self.edge_index, self.edge_attr = x, edge_index, edge_attr


def _orig_data_view(inst, n_real):
    o_ei, o_ea = _orig_molecular_bonds(inst, n_real)
    return _OrigDataView(inst.gdce_x[:n_real], o_ei, o_ea)
