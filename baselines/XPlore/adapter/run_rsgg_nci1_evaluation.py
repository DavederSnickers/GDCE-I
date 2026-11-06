"""RSGG-CE (GraphCounteRGAN) evaluation on NCI1 (no-freq) using GDCE's GIN oracle.

Same dataset / oracle / per-sample metrics / run aggregation as
C2Explainer/run_c2_nci1_evaluation.py, so the CSVs are directly comparable.
NCI1 is non-molecular: no SMILES validity, no bond types (EdgeTypeSparsity = 0).

    apprun python run_rsgg_nci1_evaluation.py n_runs=1 n_samples_per_run=5
    apprun python run_rsgg_nci1_evaluation.py --config configs/nci1.cluster.yaml
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.wrappers.config import load_config
from src.wrappers.rsgg_eval import evaluate


def main():
    cfg = load_config(default_name='nci1', script_dir=REPO_ROOT)
    evaluate(cfg, kind='nci1', repo_root=REPO_ROOT)


if __name__ == '__main__':
    main()
