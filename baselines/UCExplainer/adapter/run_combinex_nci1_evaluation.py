"""COMBINEX `combined` explainer evaluation on NCI1-nofreq (max_nodes ≤ 50).

Usage (from repo root):

    apptainer exec --nv vm/combinex.sif python run_combinex_nci1_evaluation.py \
        --oracle data/models/NCI1-G_GCN_<hidden>_epochs_<N>.pt \
        --out_dir results/combinex/nci1
"""

import argparse

from src.baselines.combinex_eval import EvalConfig, run_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, help="Path to oracle state_dict (.pt)")
    parser.add_argument("--out_dir", default="results/combinex/nci1")
    parser.add_argument(
        "--dataset_name", default="NCI1-G",
        choices=("MUTAG-G", "NCI1-G", "BENZENE-G", "PROTEINS-GDCE", "TWITTER-G"),
        help="COMBINEX dataset name; BENZENE-G maps to GDCE's benzene CSVs.",
    )
    parser.add_argument("--model", default="gcn")
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument(
        "--explainer_num_epochs", type=int, default=None,
        help="Override CombinedExplainer optimizer.num_epochs (default: 500).",
    )
    parser.add_argument(
        "--explainer_timeout", type=int, default=None,
        help="Override per-sample timeout in seconds (default: 500).",
    )
    parser.add_argument(
        "--print_cf_features", type=int, default=0,
        help="If > 0, print V_pert vs orig (changed rows) for the first N "
             "successful samples per run. Diagnose only.",
    )
    args = parser.parse_args()

    ev = EvalConfig(
        dataset_name=args.dataset_name,
        model_name=args.model,
        oracle_checkpoint=args.oracle,
        output_dir=args.out_dir,
        num_runs=args.num_runs,
        num_samples=args.num_samples,
        explainer_num_epochs=args.explainer_num_epochs,
        explainer_timeout_s=args.explainer_timeout,
        print_cf_features=args.print_cf_features,
    )
    run_evaluation(ev)


if __name__ == "__main__":
    main()
