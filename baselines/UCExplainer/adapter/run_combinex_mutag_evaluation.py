"""COMBINEX `combined` explainer evaluation on Mutagenicity-mol (max_nodes ≤ 50).

Usage (from repo root):

    apptainer exec --nv vm/combinex.sif python run_combinex_mutag_evaluation.py \
        --oracle data/models/MUTAG-G_GCN_<hidden>_epochs_<N>.pt \
        --out_dir results/combinex/mutag

Per BASELINE_PLAN.md §3.4: COMBINEX produces binary-adjacency counterfactuals,
so EdgeType Sparsity and SMILES validity are reported as NaN. Accuracy is
measured against the COMBINEX-trained GCN oracle (Variante B).
"""

import argparse

from src.baselines.combinex_eval import EvalConfig, run_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True, help="Path to oracle state_dict (.pt)")
    parser.add_argument("--out_dir", default="results/combinex/mutag")
    parser.add_argument(
        "--dataset_name", default="MUTAG-G",
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
        "--override", action="append", default=[],
        help="Extra Hydra override for the explainer config, repeatable "
             "(e.g. --override optimizer.lr=0.01 --override scheduler.initial_alpha=0.5).",
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
        overrides=tuple(args.override),
    )
    run_evaluation(ev)


if __name__ == "__main__":
    main()
