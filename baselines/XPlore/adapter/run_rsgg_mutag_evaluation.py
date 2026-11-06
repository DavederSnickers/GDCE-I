"""RSGG-CE (GraphCounteRGAN) evaluation on Mutagenicity using GDCE's GINE oracle.

Same dataset / oracle / per-sample metrics / run aggregation as
C2Explainer/run_c2_mutag_evaluation.py, so the CSVs are directly comparable.

    apprun python run_rsgg_mutag_evaluation.py n_runs=1 n_samples_per_run=5
    apprun python run_rsgg_mutag_evaluation.py --config configs/mutag.cluster.yaml
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.wrappers.config import load_config
from src.wrappers.rsgg_eval import evaluate


def main():
    cfg = load_config(default_name='mutag', script_dir=REPO_ROOT)
    evaluate(cfg, kind='mutag', repo_root=REPO_ROOT)


if __name__ == '__main__':
    main()
