#!/usr/bin/env python
"""XPlore under the shared-judge protocol.

    python run_xplore_evaluation.py --config configs/mutagenicity_mol.xplore.yaml
    python run_xplore_evaluation.py --config configs/benzene.xplore.yaml variant=xplore++

Any `key=value` after --config overrides that key in the YAML.
"""

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args, overrides = ap.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    from src.wrappers.xplore_eval import evaluate
    evaluate(cfg, REPO_ROOT)


if __name__ == '__main__':
    main()
