"""Shared 5x100 evaluation driver for XPlore (Beyond Edge Deletion, KDD'26).

Same protocol as rsgg_eval.py / the C2Explainer and D4 drivers, so the numbers
land in the same table:
  * 5 runs, seeds 42..46
  * 100 graphs per run, stratified by the judge-predicted class
  * counterfactual target = 1 - orig_pred, flip re-verified with the judge
  * graph-level cf_dump.jsonl per record, so NA / NAFR can be computed later
    with the seed-1 control judge without re-running anything

XPlore differs from the other baselines in one way that shapes this file: it has
nothing to train. It is a per-instance gradient search over a soft edge mask,
run against the classifier directly, so there is no fit phase and no explainer
store -- each sample is K optimisation steps and stops at the first valid
counterfactual (the authors' own early exit, see the `if self.opt_flag: break`).

VARIANTS. The paper reports three:

  xplore    edges only  (alpha 1.0, K 500)  -- the "beyond edge deletion" claim
  xplore+   + node-feature gating           (alpha 0.1, K 50)
  xplore++  + free node-feature change      (alpha 0.1, K 50)

Only `xplore` yields a graph that can be written to the shared dump as-is. Our
node features are one-hot atom/node types, and both feature variants form
`N_v_bar = P_node_hat * x` -- an ELEMENTWISE product with a one-hot row. That can
only rescale the single active entry; it can never move mass to a different type.
So they cannot change an atom type, and what they do produce (a one-hot row
scaled by a real number, or zeroed out entirely under gating) is not a graph any
judge was trained on. Running them is still informative -- it measures how much
of their flip rate comes from that unprojectable magnitude channel -- so the
variant is a config key, and `project_node_features` records what the projection
did rather than hiding it.
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
DEFAULT_BOND_CLASS = 1   # single bond; XPlore models no bond types

# The authors' own settings, read off XPlore_config/GRAPH/{MUTAG,BZR,AIDS}/*.
VARIANTS = {
    'xplore':   dict(alpha=1e0, K=500, update_node_feat=False, change_node_feat=False),
    'xplore+':  dict(alpha=0.1, K=50,  update_node_feat=True,  change_node_feat=False),
    'xplore++': dict(alpha=0.1, K=50,  update_node_feat=True,  change_node_feat=True),
}


# --------------------------------------------------------------- edge helpers

def _edge_set(edge_index: torch.Tensor) -> set:
    s = set()
    if edge_index is not None and edge_index.numel():
        ei = edge_index.cpu().numpy()
        for a, b in zip(ei[0], ei[1]):
            i, j = int(a), int(b)
            if i < j:
                s.add((i, j))
    return s


def adj_to_edge_index(adj: np.ndarray) -> torch.Tensor:
    """Dense adjacency -> (2, E) undirected edge_index, diagonal dropped.

    XPlore optimises over the full n x n matrix including the diagonal, so a
    self-loop can be switched on. The judge strips self-loops anyway, and none
    of our datasets has them, so they are not counted as edits either.
    """
    a = np.asarray(adj)
    a = a - np.diag(np.diag(a))
    src, dst = np.nonzero(np.triu(a, k=1))
    if len(src) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    ei = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    return torch.tensor(ei, dtype=torch.long)


def n_perturbations(orig_ei: torch.Tensor, cf_ei: torch.Tensor) -> int:
    return len(_edge_set(orig_ei).symmetric_difference(_edge_set(cf_ei)))


# ------------------------------------------------------------ instance bridge

def to_v2_instance(inst, n_real: int, GraphInstance):
    """GRETEL v1 padded DataInstance -> GRETEL v2 GraphInstance over real nodes.

    XPlore allocates several n x n matrices per iteration, so the padding rows
    the RSGG-CE loader adds would be paid for on every one of K steps -- and
    they would also let XPlore "explain" by wiring up isolated padding nodes.
    The instance therefore carries the molecule's real nodes only.
    """
    adj = inst.to_numpy_array()[:n_real, :n_real].astype(np.float64)
    adj = adj - np.diag(np.diag(adj))
    x = inst.gdce_x[:n_real].detach().cpu().numpy().astype(np.float64)
    return GraphInstance(
        id=int(inst.id), label=int(inst.graph_label), data=adj,
        node_features=x, n_atom_types=x.shape[1])


# ------------------------------------------------------------- SMILES helpers

def _orig_molecular_bonds(inst, n_real):
    ei = inst.gdce_edge_index
    ea = inst.gdce_edge_attr
    mask = (ei[0] != ei[1]) & (ei[0] < n_real) & (ei[1] < n_real)
    return ei[:, mask], (ea[mask] if ea is not None else None)


def _smiles_records(inst, cf_ei, n_real, atom_decoder, metrics):
    """(orig_smiles, cf_strict, cf_relaxed); CF bonds default to single."""
    node_types = inst.gdce_x[:n_real].argmax(dim=-1)
    o_ei, o_ea = _orig_molecular_bonds(inst, n_real)
    orig_smiles = metrics.graph_to_smiles(node_types, o_ei, o_ea, atom_decoder,
                                          strict=False) if o_ea is not None else None
    e = cf_ei.size(1)
    cf_ea = torch.zeros((e, 5))
    cf_ea[:, DEFAULT_BOND_CLASS] = 1.0
    return (orig_smiles,
            metrics.graph_to_smiles(node_types, cf_ei, cf_ea, atom_decoder, strict=True),
            metrics.graph_to_smiles(node_types, cf_ei, cf_ea, atom_decoder, strict=False))


# --------------------------------------------------------- node-type recovery

def project_node_features(cf_x: np.ndarray, orig_x: np.ndarray):
    """One-hot node types for the dump, plus how far the CF features had drifted.

    Returns (node_types, n_changed, max_abs_drift). For the `xplore` variant the
    features are untouched and this is the identity with drift 0; for the two
    feature variants it is a projection, and the drift is what the dump cannot
    represent -- reported, never silently discarded.
    """
    types = cf_x.argmax(axis=-1).astype(int)
    orig_types = orig_x.argmax(axis=-1).astype(int)
    return (types.tolist(), int((types != orig_types).sum()),
            float(np.abs(cf_x - orig_x).max()) if cf_x.size else 0.0)


# ------------------------------------------------------------------- main

def evaluate(cfg, repo_root: Path):
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(cfg.cuda))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.wrappers.gdce_bootstrap import ensure_gdce_on_src_path
    ensure_gdce_on_src_path(cfg.gdce_path)
    _sj = os.path.join(cfg.gdce_path, 'src', 'shared_judge')
    if _sj not in sys.path:
        sys.path.insert(0, _sj)

    from src.wrappers.dataset_loader import load_gdce_dataset
    from src.wrappers import metrics
    from src.wrappers.xplore_oracle import XPloreJudgeOracle
    from src.wrappers.xplore.graph_instance import GraphInstance
    if bool(cfg.get('signfix', False)):
        # Diagnostic only (see XPlore_signfix.py): the authors' file with the
        # node-feature update's sign flipped, to test whether that descent is
        # what suppresses XPlore++'s flips. Never a reported baseline number.
        from src.wrappers.xplore.XPlore_signfix import XPlore
        print('[diagnostic] using XPlore_signfix: P_node_hat ASCENDS the loss')
    else:
        from src.wrappers.xplore.XPlore import XPlore
    from cf_dump import CFDumpWriter

    from omegaconf import OmegaConf
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    variant = str(cfg.get('variant', 'xplore'))
    if variant not in VARIANTS:
        raise SystemExit(f"variant must be one of {sorted(VARIANTS)}, got {variant!r}")
    is_molecular = bool(cfg.get('molecular', False))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if variant != 'xplore' and device == 'cuda':
        # The feature variants allocate their perturbation matrix as
        # `torch.ones(instance.node_features.shape, requires_grad=True)` with no
        # device argument, so it lands on the CPU while everything it multiplies
        # is on the GPU. Rather than edit the authors' line, run them on the CPU
        # -- the graphs are small and this path is a diagnostic, not the headline
        # number.
        print("[note] variant %s allocates P_node_hat on the CPU; running on CPU."
              % variant)
        device = 'cpu'
    print(f"Device: {device} | variant: {variant}")

    # ---- dataset ----
    ds = load_gdce_dataset(cfg.gdce_path, cfg.dataset_name, cfg.datadir,
                           name=f'xplore_{cfg.dataset_name}',
                           max_nodes=int(cfg.n_nodes))
    test_ids = ds.splits[0]['test']
    node_dim = int(ds.instances[test_ids[0]].gdce_x.size(1))
    print(f"Instances: {len(ds.instances)} (test={len(test_ids)}); node_dim={node_dim}")

    atom_decoder = None
    if is_molecular:
        ad_path = Path(cfg.gdce_path) / cfg.datadir / 'raw' / 'atom_decoder.json'
        if ad_path.exists():
            atom_decoder = json.loads(ad_path.read_text())
            assert len(atom_decoder) == node_dim, (
                f"atom_decoder has {len(atom_decoder)} entries but node features have "
                f"{node_dim} columns; SMILES decoding and the judge would disagree.")
        else:
            print(f"[warn] atom_decoder.json not found at {ad_path}; SMILES disabled")

    # ---- oracle: the one classifier every method in the comparison targets ----
    jp = cfg.get('shared_judge_path', None)
    if not jp:
        raise SystemExit("xplore_eval requires shared_judge_path=<prefix>")
    print(f"Loading SHARED JUDGE: {jp}")
    oracle = XPloreJudgeOracle(judge_prefix=str(jp), device=device)

    # ---- explainer ----
    # XPlore reads its hyperparameters through GRETEL's config dict, so it is
    # handed one -- the base-class shim then runs check_configuration() (which
    # fills every remaining default) and init(), in that order.
    params = dict(VARIANTS[variant])
    params.update(dict(beta=float(cfg.beta), gamma_edge=float(cfg.gamma_edge),
                       change_all_feat=False, directed=False,
                       dataset_classes=NUM_CLASSES, node_classification=False,
                       multi_label_classification=False, decay_alpha=False,
                       debugging=False, visualize=False, chem_flag=False))
    for k in ('alpha', 'K'):
        if cfg.get(k, None) is not None:
            params[k] = type(params[k])(cfg.get(k))
    print(f"XPlore parameters: {params}")

    class _DatasetShim:
        """What XPlore reads off `self.dataset`: a name, a class count, and the
        manipulator list. We configure no manipulators -- the judge is trained on
        one-hot node types, not on centrality features."""
        name = str(cfg.dataset_name)
        dataset_name = str(cfg.dataset_name)
        num_classes = NUM_CLASSES
        manipulators: list = []

    explainer = XPlore(context=None, local_config={
        'dataset': _DatasetShim(), 'oracle': oracle, 'parameters': params})
    # When the candidate differs from the input, XPlore rebuilds the node-feature
    # matrix from scratch and np.hstack()es the pieces together. With no
    # manipulators the only piece is the one-hot block, and that branch is
    # reached only for datasets whose name it recognises as chemical -- for any
    # other name the list is empty and np.hstack raises. Registering our dataset
    # makes the branch fire, where (no manipulators) it is exactly the identity.
    explainer.CHEMICAL_DATASETS = {str(cfg.dataset_name)}

    # ---- dump, flushed per record ----
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cf_writer = CFDumpWriter(
        out_dir / 'cf_dump.jsonl', method='xplore', dataset=str(cfg.dataset_name),
        judge=str(jp), bond_source='binary', variant=variant)

    # ---- pre-classify the test split (stratification base) ----
    print("Pre-classifying test set...")
    all_samples = []
    for iid in tqdm(test_ids):
        inst = ds.instances[iid]
        gi = to_v2_instance(inst, int(inst.n_real_nodes), GraphInstance)
        all_samples.append({'iid': iid, 'orig_pred': oracle.predict_int(gi)})
    class_totals = {c: sum(1 for s in all_samples if s['orig_pred'] == c)
                    for c in range(NUM_CLASSES)}
    print("  Class distribution: " +
          ", ".join(f"{CLASS_NAMES[c]}: {class_totals[c]}" for c in range(NUM_CLASSES)))

    run_seeds = list(range(42, 42 + int(cfg.n_runs)))
    all_rows, run_summary_rows = [], []

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
            gi = to_v2_instance(inst, n_real, GraphInstance)
            orig_ei = adj_to_edge_index(gi.data)
            n_orig_edges = len(_edge_set(orig_ei))

            torch.manual_seed(seed + sample_idx)
            np.random.seed(seed + sample_idx)

            t0 = time.time()
            try:
                cf_gi = explainer.explain(gi)
                cf_ei = adj_to_edge_index(cf_gi.data)
                cf_pred = oracle.predict_int(cf_gi)
            except Exception as e:                       # noqa: BLE001
                print(f"  [{sample_idx+1}/{len(selected)}] iid={meta['iid']} FAILED: "
                      f"{type(e).__name__}: {e}")
                cf_gi, cf_ei, cf_pred = None, None, None
            elapsed = time.time() - t0

            n_perturbs = n_perturbations(orig_ei, cf_ei) if cf_ei is not None else 0
            success = (cf_pred is not None) and (cf_pred != orig_pred) and (n_perturbs > 0)

            row = {
                'run': run_idx, 'seed': seed, 'sample_idx': sample_idx,
                'data_idx': meta['iid'], 'source_class': orig_pred,
                'target_class': target_class, 'orig_pred': orig_pred,
                'success': bool(success), 'cf_pred': cf_pred if cf_pred is not None else -1,
                'n_perturbs': n_perturbs, 'n_orig_edges': n_orig_edges,
                'n_nodes': n_real, 'time_s': elapsed,
                'prop_perturbs': (n_perturbs / n_orig_edges) if n_orig_edges > 0 else -1.0,
                'ged': -1, 'mod_rate': -1.0, 'node_sparsity': -1.0,
                'edge_sparsity': -1.0, 'edgetype_sparsity': -1.0,
                'n_node_types_changed': -1, 'node_feature_drift': -1.0,
                'smiles_valid_strict': -1, 'smiles_valid_relaxed': -1,
            }

            if success:
                cf_x = np.asarray(cf_gi.node_features, dtype=np.float64)
                orig_x = np.asarray(gi.node_features, dtype=np.float64)
                node_types, n_changed, drift = project_node_features(cf_x, orig_x)

                ged = metrics.compute_ged(orig_ei, cf_ei, n_real)
                node_sp, edge_sp, et_sp = metrics.compute_sparsity(
                    orig_ei, cf_ei,
                    orig_node_types=torch.as_tensor(orig_x.argmax(axis=-1)),
                    cf_node_types=torch.as_tensor(np.asarray(node_types)))
                row.update({
                    'ged': ged,
                    'mod_rate': (ged / n_orig_edges) if (ged >= 0 and n_orig_edges > 0) else -1.0,
                    'node_sparsity': node_sp, 'edge_sparsity': edge_sp,
                    'edgetype_sparsity': et_sp,
                })
                row['n_node_types_changed'] = n_changed
                row['node_feature_drift'] = drift

                if is_molecular and atom_decoder is not None:
                    _, cf_strict, cf_relaxed = _smiles_records(
                        inst, cf_ei, n_real, atom_decoder, metrics)
                    row['smiles_valid_strict'] = int(cf_strict is not None)
                    row['smiles_valid_relaxed'] = int(cf_relaxed is not None)

                cf_writer.write(
                    run=run_idx, seed=seed, sample_idx=sample_idx,
                    data_idx=int(meta['iid']), source_class=int(orig_pred),
                    target_class=int(target_class), node_types=node_types,
                    edges=[[i, j, DEFAULT_BOND_CLASS] for (i, j) in sorted(_edge_set(cf_ei))],
                    own_pred=int(cf_pred), projected=True)
                print(f"  [{sample_idx+1}/{len(selected)}] iid={meta['iid']} "
                      f"{CLASS_NAMES[orig_pred]}->{CLASS_NAMES[cf_pred]} "
                      f"perturbs={n_perturbs} ged={row['ged']} "
                      f"drift={drift:.3f} t={elapsed:.1f}s")
            else:
                print(f"  [{sample_idx+1}/{len(selected)}] iid={meta['iid']} "
                      f"{CLASS_NAMES[orig_pred]} FAIL t={elapsed:.1f}s")

            all_rows.append(row)

        # ---- per-run aggregate ----
        df_r = pd.DataFrame([r for r in all_rows if r['run'] == run_idx])
        succ = df_r[df_r['success']]

        def _mean(col):
            v = succ[succ[col] >= 0][col] if len(succ) else pd.Series([], dtype=float)
            return v.mean() if len(v) else float('nan')

        n_total, n_succ = len(df_r), int(df_r['success'].sum())
        summary = {
            'run': run_idx, 'seed': seed, 'n_samples': n_total, 'n_success': n_succ,
            'flip_accuracy': n_succ / n_total if n_total else float('nan'),
            'avg_perturbs': _mean('n_perturbs'), 'avg_prop_perturbs': _mean('prop_perturbs'),
            'avg_ged': _mean('ged'), 'avg_mod_rate': _mean('mod_rate'),
            'avg_node_sparsity': _mean('node_sparsity'),
            'avg_edge_sparsity': _mean('edge_sparsity'),
            'avg_edgetype_sparsity': _mean('edgetype_sparsity'),
            'avg_node_types_changed': _mean('n_node_types_changed'),
            'avg_node_feature_drift': _mean('node_feature_drift'),
            'avg_time_s': df_r['time_s'].mean(),
            'smiles_valid_strict': _mean('smiles_valid_strict'),
            'smiles_valid_relaxed': _mean('smiles_valid_relaxed'),
        }
        run_summary_rows.append(summary)
        print(f"  run {run_idx+1}: flip={summary['flip_accuracy']:.4f} "
              f"ged={summary['avg_ged']:.3f} modrate={summary['avg_mod_rate']:.3f}")

        pd.DataFrame(all_rows).to_csv(out_dir / 'xplore_results.csv', index=False)
        pd.DataFrame(run_summary_rows).to_csv(out_dir / 'run_summary.csv', index=False)

    # ---- final ----
    df_s = pd.DataFrame(run_summary_rows)
    print(f"\n{'#'*70}\nFINAL SUMMARY: mean +/- std across {len(df_s)} runs "
          f"(n={cfg.n_samples_per_run} per run, stratified)\n{'#'*70}")
    for label, col in [
        ('Flip-Accuracy', 'flip_accuracy'), ('Avg #perturbs', 'avg_perturbs'),
        ('Avg prop_perturbs', 'avg_prop_perturbs'), ('Avg GED', 'avg_ged'),
        ('Avg ModRate', 'avg_mod_rate'), ('Avg NodeSparsity', 'avg_node_sparsity'),
        ('Avg EdgeSparsity', 'avg_edge_sparsity'),
        ('Avg EdgeTypeSparsity', 'avg_edgetype_sparsity'),
        ('Avg #node types changed', 'avg_node_types_changed'),
        ('Avg node-feature drift', 'avg_node_feature_drift'),
        ('Valid SMILES (strict)', 'smiles_valid_strict'),
        ('Valid SMILES (relaxed)', 'smiles_valid_relaxed'),
        ('Avg time/sample (s)', 'avg_time_s'),
    ]:
        if col in df_s and df_s[col].notna().any():
            print(f"  {label:<33}: {df_s[col].mean():.4f} +/- {df_s[col].std():.4f}")
    print("NOTE: XPlore edits only the edge set unless a feature variant is selected, "
          "so EdgeTypeSparsity is 0 by construction and node types are the input's. "
          "'node-feature drift' is the max |cf_x - x| the dump's one-hot projection "
          "had to discard: 0 for variant 'xplore'.")
    print(f"Results saved to: {out_dir.resolve()}")
    print("DONE")
