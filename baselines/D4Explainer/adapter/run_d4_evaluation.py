"""
Evaluate D4Explainer counterfactual explanations on BA3Motif.

Saves per-graph results (original adj, counterfactual adj, classifier predictions)
to a .pt file that can be loaded by compare_results.py for cross-method comparison.

The classifier used is D4Explainer's BA3MotifNet — the same .pt file must be
copied/shared with the GDCE evaluation script so both use the same evaluator.

Usage:
  python run_d4_evaluation.py [--device cuda:0] [--output results/d4_results.pt]
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected, to_dense_adj

from datasets.ba3motif_dataset import BA3Motif
from explainers.diffusion.pgnn import Powerful
from explainers.diffusion.graph_utils import (
    graph2tensor, tensor2graph, gen_list_of_data_single, generate_mask,
)
from gnns.ba3motif_gnn import BA3MotifNet


def parse_args():
    p = argparse.ArgumentParser(description="D4Explainer evaluation on BA3Motif")
    p.add_argument("--device", type=str,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gnn_path", type=str, default="param/gnns/ba3_gcn.pt",
                   help="Path to trained BA3MotifNet")
    p.add_argument("--d4_model", type=str, default="results/ba3/best_model.pth",
                   help="Path to trained D4Explainer Powerful model")
    p.add_argument("--data_path", type=str, default="data/BA3",
                   help="Path to BA3Motif dataset root")
    p.add_argument("--output", type=str, default="results/d4_ba3_results.pt",
                   help="Where to save the per-graph results")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7],
                   help="Sigmoid thresholds for edge prediction")
    p.add_argument("--max_graphs", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def pyg_to_adj(edge_index, num_nodes):
    """Convert edge_index to dense numpy adjacency matrix."""
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    ei = edge_index.cpu().numpy()
    for i in range(ei.shape[1]):
        adj[ei[0, i], ei[1, i]] = 1.0
    return adj


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # --- Load test set ---
    test_dataset = BA3Motif(args.data_path, mode="testing")
    if args.max_graphs < len(test_dataset):
        test_dataset = test_dataset[:args.max_graphs]
    print(f"Test set: {len(test_dataset)} graphs")

    # --- Load classifier ---
    gnn_model = torch.load(args.gnn_path, map_location=device).to(device)
    gnn_model.eval()
    print(f"Loaded BA3MotifNet from {args.gnn_path}")

    # --- Verify classifier accuracy ---
    correct = total = 0
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prob, _ = gnn_model.get_pred(x=batch.x, edge_index=batch.edge_index,
                                          batch=batch.batch)
            correct += (prob.argmax(-1) == batch.y).sum().item()
            total += batch.y.size(0)
    print(f"Classifier accuracy on test set: {correct/total:.2%} ({correct}/{total})")

    # --- D4Explainer args ---
    class D4Args:
        dataset = "ba3"; task = "gc"; feature_in = 4
        sigma_length = 10; prob_low = 0.0; prob_high = 0.4; noise_list = None
        num_layers = 6; layers_per_conv = 1; n_hidden = 64
        cat_output = True; residual = False; noise_mlp = True
        normalization = "instance"; simplified = False; dropout = 0.001
        threshold = 0.5
    d4_args = D4Args()
    d4_args.device = device

    # --- Load D4Explainer model ---
    ckpt = torch.load(args.d4_model, map_location=device)
    print(f"Loaded D4Explainer model from {args.d4_model}")

    # --- Run evaluation per threshold ---
    all_configs = {}

    for threshold in args.thresholds:
        print(f"\n{'='*60}")
        print(f"Threshold = {threshold}")
        print(f"{'='*60}")

        d4_args.threshold = threshold
        model = Powerful(d4_args).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        records = []
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        for idx, graph in enumerate(test_loader):
            if graph.is_directed():
                graph.edge_index = to_undirected(edge_index=graph.edge_index)
            graph = graph.to(device)

            n_nodes = graph.x.size(0)

            # Original classifier prediction
            with torch.no_grad():
                orig_prob, _ = gnn_model.get_pred(
                    x=graph.x, edge_index=graph.edge_index, batch=graph.batch)
            y_orig_pred = orig_prob.argmax(-1).item()
            y_true = graph.y.item()

            # D4Explainer inference
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

                # Build counterfactual subgraph
                sub = tensor2graph(graph, scores, mask, threshold=threshold)

                # Classify counterfactual
                sub_prob, _ = gnn_model.get_pred(
                    x=sub.x, edge_index=sub.edge_index, batch=sub.batch)
            y_cf_pred = sub_prob.argmax(-1).item()

            # Store adjacency matrices as dense numpy
            adj_orig = pyg_to_adj(graph.edge_index, n_nodes)
            adj_cf = pyg_to_adj(sub.edge_index, n_nodes)

            records.append({
                "idx": idx,
                "n_nodes": n_nodes,
                "y_true": y_true,
                "y_orig_pred": y_orig_pred,
                "y_cf_pred": y_cf_pred,
                "adj_orig": adj_orig,
                "adj_cf": adj_cf,
            })

            if (idx + 1) % 100 == 0:
                flipped = sum(1 for r in records if r["y_cf_pred"] != r["y_orig_pred"])
                print(f"  [{idx+1}/{len(test_dataset)}] flip rate: {flipped/(idx+1):.2%}")

        config_key = f"threshold_{threshold}"
        all_configs[config_key] = records

        flipped = sum(1 for r in records if r["y_cf_pred"] != r["y_orig_pred"])
        print(f"  Final flip rate: {flipped}/{len(records)} = {flipped/len(records):.2%}")

    # --- Save ---
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save({
        "method": "D4Explainer",
        "dataset": "BA3Motif",
        "classifier": args.gnn_path,
        "classifier_accuracy": correct / total,
        "configs": all_configs,
    }, args.output)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
