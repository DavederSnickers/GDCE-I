"""
Evaluate D4Explainer counterfactual explanations on NCI1.

Mirrors GDCE/inversion_nci1_nofreq multi-run structure:
  - N_RUNS independent runs, each with N_SAMPLES_PER_RUN stratified graphs
  - Final report: mean ± std across runs for Flip Accuracy, Edge Sparsity, Mod Rate

Usage:
  python run_d4_nci1_evaluation.py \
    --gnn_path param/gnns/NCI1_gcn.pt \
    --d4_model results/NCI1/best_model.pth \
    --data_path data/NCI1
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
from datasets.NCI1_dataset import NCI1
from datasets.nci1_filtered_dataset import NCI1Filtered
from datasets.gdce_nci1_dataset import GDCENci1Dataset
from datasets.gdce_benzene_dataset import GDCEBenzeneDataset
from datasets.gdce_nonmol_dataset import GDCENonMolDataset
from gnns.nci1_gnn import NCI1GCN
from explainers.diffusion.pgnn import Powerful
from explainers.diffusion.graph_utils import (
    graph2tensor,
    tensor2graph,
    gen_list_of_data_single,
    generate_mask,
)


NUM_CLASSES = 2
CLASS_NAMES = {0: "Inactive", 1: "Active"}

# Only the printed labels differ per dataset; the numeric convention (0/1) is the
# one GDCE preprocessed the CSVs with and is shared by every method.
_CLASS_NAMES_BY_DATASET = {
    "gdce_proteins": {0: "Non-enzyme", 1: "Enzyme"},
    # 0 = Positive verified against the test-split word distribution (y=0 leads
    # with love/lol/hope/haha, y=1 with miss/feel/sleep/hate), 2026-08-05.
    "gdce_twitter":  {0: "Positive", 1: "Negative"},
}


def parse_args():
    p = argparse.ArgumentParser(description="D4Explainer evaluation on NCI1")
    p.add_argument("--device", type=str,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gnn_path", type=str, default="param/gnns/gdce_nci1_gcn.pt")
    p.add_argument("--d4_model", type=str, default="results/gdce_nci1/best_model.pth")
    p.add_argument("--dataset", type=str, default="gdce_nci1",
                   choices=["nci1", "nci1_filtered", "gdce_nci1", "gdce_benzene",
                            "gdce_proteins", "gdce_twitter"])
    p.add_argument("--data_path", type=str, default="data/GDCE_NCI1")
    p.add_argument("--output_dir", type=str, default="results/nci1_evaluation")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.5])
    p.add_argument("--n_runs", type=int, default=5,
                   help="Number of independent evaluation runs (default: 5)")
    p.add_argument("--n_samples_per_run", type=int, default=100,
                   help="Graphs per run, stratified by classifier prediction (default: 100)")
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

    # --- Feature dim per dataset variant (freq=10, raw/nofreq=37) ---
    # gdce_benzene: 8 atom types, gdce_nci1: 10 (freq), raw/nofreq: 37.
    # The non-molecular sets carry node labels rather than atom types:
    # proteins 3 (helix/sheet/turn), twitter 238 (frequency-filtered vocabulary).
    feature_in = (8 if args.dataset == "gdce_benzene"
                  else 10 if args.dataset == "gdce_nci1"
                  else 3 if args.dataset == "gdce_proteins"
                  else 238 if args.dataset == "gdce_twitter" else 37)

    global CLASS_NAMES
    CLASS_NAMES = _CLASS_NAMES_BY_DATASET.get(args.dataset, CLASS_NAMES)

    # --- Load test set ---
    if args.dataset == "gdce_benzene":
        test_dataset = GDCEBenzeneDataset(args.data_path, mode="testing")
    elif args.dataset in ("gdce_proteins", "gdce_twitter"):
        test_dataset = GDCENonMolDataset(
            args.data_path, name=args.dataset[len("gdce_"):], mode="testing")
    elif args.dataset == "gdce_nci1":
        test_dataset = GDCENci1Dataset(args.data_path, mode="testing")
    elif args.dataset == "nci1_filtered":
        test_dataset = NCI1Filtered(args.data_path, mode="testing")
    else:
        test_dataset = NCI1(args.data_path, mode="testing")
    print(f"Dataset: {args.dataset}  |  Test set: {len(test_dataset)} graphs")

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
    if isinstance(gnn_model, dict):
        model_obj = NCI1GCN(conv_unit=3, in_dim=feature_in).to(device)
        model_obj.load_state_dict(gnn_model)
        gnn_model = model_obj
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
            prob, _ = gnn_model.get_pred(
                x=batch.x, edge_index=batch.edge_index, batch=batch.batch)
            preds = prob.argmax(-1)
            correct += (preds == batch.y.view(-1)).sum().item()
            total += batch.y.size(0)
            for c in range(NUM_CLASSES):
                class_counts[c] += (batch.y.view(-1) == c).sum().item()
    print(f"Classifier accuracy on test set: {correct/total:.2%} ({correct}/{total})")
    for c in range(NUM_CLASSES):
        print(f"  {CLASS_NAMES[c]}: {class_counts[c]} graphs")

    # --- D4Explainer model config ---
    class D4Args:
        dataset = "nci1_filtered"; task = "gc"; feature_in = 37
        sigma_length = 10; prob_low = 0.0; prob_high = 0.4; noise_list = None
        num_layers = 6; layers_per_conv = 1; n_hidden = 128
        cat_output = True; residual = False; noise_mlp = True
        normalization = "instance"; simplified = False; dropout = 0.001
        threshold = 0.5
    d4_args = D4Args()
    d4_args.dataset = args.dataset
    d4_args.feature_in = feature_in  # 10 for gdce_nci1, 37 otherwise
    d4_args.device = device

    ckpt = torch.load(args.d4_model, map_location=device)
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
            prob, _ = gnn_model.get_pred(x=b.x, edge_index=b.edge_index, batch=b.batch)
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
                graph.edge_index = to_undirected(edge_index=graph.edge_index)
            prob, _ = gnn_model.get_pred(
                x=graph.x, edge_index=graph.edge_index, batch=graph.batch)
            orig_pred = prob.argmax(-1).item()
            all_test_samples.append({'data_idx': data_idx, 'orig_pred': orig_pred, 'graph': graph})

    class_totals = {c: sum(1 for s in all_test_samples if s['orig_pred'] == c)
                    for c in range(NUM_CLASSES)}
    print(f"  Class distribution: " +
          ", ".join(f"{CLASS_NAMES[c]}: {class_totals[c]}" for c in range(NUM_CLASSES)))

    # --- Multi-run evaluation ---
    all_rows = []

    # Graph-level dump in the format shared with GDCE and the other baselines.
    # NCI1 has no bond types; every kept edge is written as a single bond.
    cf_writer = CFDumpWriter(
        os.path.join(args.output_dir, 'cf_dump.jsonl'),
        method='d4',
        # GDCE dataset name, so dumps from different repos are joinable.
        dataset={'gdce_nci1': 'nci1', 'gdce_benzene': 'benzene',
                 'gdce_proteins': 'proteins', 'gdce_twitter': 'twitter'}.get(
            args.dataset, args.dataset),
        judge=args.gnn_path, bond_source='binary',
        variant=os.path.basename(str(args.d4_model)),
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
                        sub_prob, _ = gnn_model.get_pred(
                            x=sub.x, edge_index=sub.edge_index, batch=sub.batch)

                    y_cf_pred = sub_prob.argmax(-1).item()
                    inv_success = (y_cf_pred == target_class)

                    adj_cf = pyg_to_adj(sub.edge_index, n_nodes)
                    inv_ged = compute_ged(adj_orig, adj_cf)
                    inv_mod_rate = inv_ged / n_orig_edges if inv_ged >= 0 and n_orig_edges > 0 else -1.0
                    inv_edge_sparsity = compute_edge_sparsity(adj_orig, adj_cf)

                except Exception as e:
                    print(f"  Sample {sample_idx}: Error: {e}")

                if inv_success and sub is not None:
                    cf_writer.write(
                        run=run_idx, seed=run_seed, sample_idx=sample_idx,
                        data_idx=sample['data_idx'], source_class=orig_pred,
                        target_class=target_class,
                        node_types=node_types_from_x(sub.x),
                        edges=edges_from_edge_index(sub.edge_index),
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
                    'n_nodes': n_nodes,
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

            run_summary_rows.append({
                'run': run_idx,
                'seed': run_seed,
                'threshold': threshold,
                'accuracy': n_succ / n_total if n_total > 0 else float('nan'),
                'avg_mod_rate': _mean('mod_rate'),
                'avg_edge_sparsity': _mean('edge_sparsity'),
                'n_total': n_total,
                'n_success': n_succ,
            })
            print(f"\n  [RUN {run_idx+1}, thresh={threshold}] "
                  f"Accuracy={n_succ}/{n_total} "
                  f"({100*n_succ/n_total if n_total > 0 else 0:.1f}%)  "
                  f"AvgMR={_mean('mod_rate'):.4f}  "
                  f"AvgES={_mean('edge_sparsity'):.4f}")

    # --- Save CSVs ---
    cf_writer.close()

    df_inv = pd.DataFrame(all_rows)
    csv_path = os.path.join(args.output_dir, "inversion_results.csv")
    df_inv.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

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
        ('accuracy',          'Flip Accuracy'),
        ('avg_edge_sparsity', 'Avg Edge Sparsity'),
        ('avg_mod_rate',      'Avg Mod Rate'),
    ]

    for thresh in args.thresholds:
        df_s = df_run_summary[df_run_summary['threshold'] == thresh]
        print(f"\n--- threshold = {thresh}  ({len(df_s)} runs) ---")
        for col, label in METRICS:
            vals = df_s[col].dropna()
            if len(vals) == 0:
                print(f"  {label:<25}: N/A")
            else:
                mean_v = vals.mean()
                std_v  = vals.std(ddof=1) if len(vals) > 1 else 0.0
                print(f"  {label:<25}: {mean_v:.4f} ± {std_v:.4f}")

    print(f"\nAll results saved in: {os.path.abspath(args.output_dir)}")
    print("DONE!")


if __name__ == "__main__":
    main()
