"""RSGG-CE evaluation under the shared-judge protocol of the GDCE-I paper.

One driver for every dataset; the dataset, its judge and whether SMILES can be
decoded all come from the config:

    apprun python run_rsgg_evaluation.py --config configs/mutagenicity_mol.shared.yaml
    apprun python run_rsgg_evaluation.py --config configs/twitter.shared.yaml n_runs=1

Replaces run_rsgg_{mutag,nci1}_evaluation.py, which hard-wired the dataset into
the driver and defaulted to RSGG-CE's own TF oracle.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.wrappers.config import load_config
from src.wrappers.rsgg_eval import evaluate


def main():
    cfg = load_config(default_name='mutagenicity_mol.shared', script_dir=REPO_ROOT)
    kind = 'mutag' if bool(cfg.get('molecular', True)) else 'nci1'
    evaluate(cfg, kind=kind, repo_root=REPO_ROOT)


if __name__ == '__main__':
    main()
