"""Tiny OmegaConf-based config loader.

Layered merge:
    base default config  (configs/<name>.yaml)
        <- optional --config <path/to/file.yaml>
            <- remaining CLI args parsed as `key=value` (dot-paths)

Usage:
    cfg = load_config(default_name='nci1', script_dir=Path(__file__).parent)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from omegaconf import OmegaConf, DictConfig


def parse_cli(argv: Optional[List[str]] = None) -> tuple[Optional[str], List[str], bool]:
    """Pull --config / -c (and --help / -h) out of argv; return (config_path, overrides, want_help)."""
    if argv is None:
        argv = sys.argv[1:]
    config_path: Optional[str] = None
    overrides: List[str] = []
    want_help = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ('--help', '-h'):
            want_help = True
            i += 1
        elif a in ('--config', '-c'):
            i += 1
            if i >= len(argv):
                raise SystemExit("--config requires a path argument")
            config_path = argv[i]
            i += 1
        elif a.startswith('--config='):
            config_path = a.split('=', 1)[1]
            i += 1
        else:
            overrides.append(a)
            i += 1
    return config_path, overrides, want_help


def load_config(default_name: str, script_dir: Path,
                argv: Optional[List[str]] = None) -> DictConfig:
    """Build a layered OmegaConf for an evaluation script.

    Layer 1: configs/<default_name>.yaml relative to script_dir
    Layer 2: --config <path> (optional)
    Layer 3: remaining CLI args interpreted as `key=value` overrides
    """
    config_path, overrides, want_help = parse_cli(argv)

    if want_help:
        _print_help(default_name, script_dir)
        sys.exit(0)

    default_path = script_dir / 'configs' / f'{default_name}.yaml'
    if not default_path.exists():
        raise FileNotFoundError(f"Default config missing: {default_path}")
    cfg: DictConfig = OmegaConf.load(default_path)  # type: ignore[assignment]

    if config_path is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_path))  # type: ignore[assignment]

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(overrides))  # type: ignore[assignment]

    return cfg


def _print_help(default_name: str, script_dir: Path) -> None:
    print(f"""\
Usage:
    python <script>.py [--config PATH] [key=value ...]

Examples:
    # use defaults from configs/{default_name}.yaml
    python <script>.py

    # smoke test (1 run, 5 samples)
    python <script>.py n_runs=1 n_samples_per_run=5

    # cluster paths via override config
    python <script>.py --config configs/{default_name}.cluster.yaml

    # cluster paths + smoke test
    python <script>.py --config configs/{default_name}.cluster.yaml n_runs=1 n_samples_per_run=5

    # arbitrary single override
    python <script>.py ckpt=/some/other/path.ckpt epochs=500

Default config keys (see configs/{default_name}.yaml):
    gdce_path, ckpt, datadir,
    epochs, lr, subgraph_mode,
    n_runs, n_samples_per_run,
    out_dir, cuda
""")
