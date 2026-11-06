"""Build GDCE's SpectreGraphDataModule with a minimal in-memory config.

GDCE's SpectreGraphDataModule expects a Hydra-style cfg with a number of
fields. To avoid pulling in the full Hydra setup just to load a test
dataset, we build a minimal OmegaConf cfg with only the fields the data
module touches.
"""

import os
from omegaconf import OmegaConf


def build_minimal_cfg(dataset_name: str, datadir: str, batch_size: int = 1,
                      num_workers: int = 0) -> OmegaConf:
    cfg = OmegaConf.create({
        'dataset': {
            'name': dataset_name,
            'datadir': datadir,                    # e.g. 'data/nci1_nofreq/'
            'pin_memory': False,
            'weighted_sampling': False,
        },
        'train': {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'num_interops_workers': 1,
            'seed': 0,
        },
        'general': {
            'name': 'c2_eval',
            'wandb': 'disabled',
            'gpus': 1,
        },
        'guidance': {
            'guidance_target': None,
            'label_source': 'original',           # do NOT re-relabel
            'graph_classifier_path': None,
            'include_split': None,
            'node_inference_method': None,
            'n_test_molecules_to_sample': 0,
            'n_samples_per_test_molecule': 0,
            's': 0,
            'node_model_path': None,
            'build_with_partial_charges': False,
            'experiment_type': None,
            'test_thresholds': None,
        },
        'model': {
            'type': 'discrete',
            'extra_features': None,
        },
    })
    return cfg


def load_test_dataset(gdce_path: str, dataset_name: str, datadir: str):
    """Build the SpectreGraphDataModule and return its test dataset."""
    import sys
    if gdce_path not in sys.path:
        sys.path.insert(0, gdce_path)

    # ensure GDCE's data dir resolution works:
    # SpectreGraphDataModule resolves datadir relative to GDCE root.
    from src.datasets.spectre_dataset import SpectreGraphDataModule

    cfg = build_minimal_cfg(dataset_name, datadir)
    dm = SpectreGraphDataModule(cfg)
    return dm.dataloaders['test'].dataset
