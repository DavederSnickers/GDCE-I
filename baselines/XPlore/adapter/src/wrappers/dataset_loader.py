"""Load GDCE's test/train splits and adapt them to GRETEL's Dataset/DataInstance.

Mirrors C2Explainer/src/wrappers/dataset_loader.py for the part that builds
GDCE's SpectreGraphDataModule from a minimal OmegaConf config, then converts
each PyG `Data` into a GRETEL `DataInstance` so RSGG-CE
(GraphCounteRGANExplainer) can consume it.

Key adaptations for RSGG-CE:
  * RSGG's generator/discriminator are sized to a FIXED node count, so every
    instance's adjacency is padded to `max_nodes` (= n_nodes, default 50).
    Padding nodes are isolated; the oracle trims them back off before
    classifying.
  * GDCE's edge_index carries a self-loop on every node (the dataset sets the
    adjacency diagonal to 1.0). We STRIP self-loops for the RSGG molecular
    topology (so GED/sparsity count molecular edges only); the oracle re-adds
    them in the trained classifier's representation.
  * The graph label lives in `data.guidance` (one-hot), NOT `data.y`
    (which GDCE leaves as an empty tensor).
  * The original GDCE tensors (x, edge_index, edge_attr) are attached to the
    instance as `gdce_*` attributes so the oracle can recover atom types / bond
    types for the molecule's fixed node set.
"""

import numpy as np
import torch
from omegaconf import OmegaConf

from src.dataset.dataset_base import Dataset
from src.dataset.data_instance_base import DataInstance


def build_minimal_cfg(dataset_name: str, datadir: str, batch_size: int = 1,
                      num_workers: int = 0) -> OmegaConf:
    """Minimal Hydra-style cfg with only the fields SpectreGraphDataModule touches.

    `label_source: 'original'` -> keep the dataset's real labels (no relabeling).
    """
    return OmegaConf.create({
        'dataset': {
            'name': dataset_name,
            'datadir': datadir,
            'pin_memory': False,
            'weighted_sampling': False,
        },
        'train': {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'num_interops_workers': 1,
            'seed': 0,
        },
        'general': {'name': 'rsgg_eval', 'wandb': 'disabled', 'gpus': 1},
        'guidance': {
            'guidance_target': None,
            'label_source': 'original',
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
        'model': {'type': 'discrete', 'extra_features': None},
    })


def _graph_label(data) -> int:
    """GDCE stores the label one-hot in `data.guidance`; `data.y` is empty."""
    g = getattr(data, 'guidance', None)
    if g is not None and g.numel() > 0:
        return int(g.reshape(-1, g.size(-1)).argmax(dim=-1).item())
    # Fallback (should not happen for mutagenicity_mol / nci1_nofreq)
    y = getattr(data, 'y', None)
    return int(y.item()) if (y is not None and y.numel() > 0) else 0


def _molecular_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """Drop self-loops; return the (2, E) molecular edge_index."""
    if edge_index.numel() == 0:
        return edge_index
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]


def pyg_to_instance(data, inst_id: int, max_nodes: int) -> DataInstance:
    """Convert one GDCE PyG `Data` into a padded GRETEL DataInstance."""
    n_real = int(data.x.size(0))
    mol_ei = _molecular_edges(data.edge_index).cpu().numpy()

    adj = np.zeros((max_nodes, max_nodes), dtype=np.int32)
    for s, t in zip(mol_ei[0], mol_ei[1]):
        adj[int(s), int(t)] = 1
        adj[int(t), int(s)] = 1

    inst = DataInstance(id=inst_id)
    inst.from_numpy_array(adj)                       # sets inst.graph (nx, max_nodes)
    inst.graph_label = _graph_label(data)
    inst.max_n_nodes = max_nodes

    # Original GDCE tensors for the oracle's per-sample feature/bond recovery.
    inst.gdce_x = data.x.detach().cpu()
    inst.gdce_edge_index = data.edge_index.detach().cpu()   # WITH self-loops (as trained)
    ea = getattr(data, 'edge_attr', None)
    inst.gdce_edge_attr = ea.detach().cpu() if ea is not None else None
    inst.n_real_nodes = n_real
    return inst


def load_gdce_dataset(gdce_path: str, dataset_name: str, datadir: str,
                      name: str, max_nodes: int = 50) -> Dataset:
    """Build a GRETEL Dataset wrapping GDCE's train + test splits.

    `get_split_indices()[0]` returns {'train': [...], 'test': [...]} over the
    instance ids (ids == positions in `dataset.instances`, both sorted). RSGG-CE
    trains its generator on the 'train' ids; the evaluation samples from 'test'.
    """
    from src.wrappers.gdce_bootstrap import ensure_gdce_on_src_path
    ensure_gdce_on_src_path(gdce_path)
    from src.datasets.spectre_dataset import SpectreGraphDataModule

    cfg = build_minimal_cfg(dataset_name, datadir)
    dm = SpectreGraphDataModule(cfg)
    train_ds = dm.dataloaders['train'].dataset
    test_ds = dm.dataloaders['test'].dataset

    ds = Dataset(id=name)
    ds.name = name
    ds.max_n_nodes = max_nodes

    instances = []
    iid = 0
    train_ids, test_ids = [], []
    for d in train_ds:
        instances.append(pyg_to_instance(d, iid, max_nodes))
        train_ids.append(iid); iid += 1
    for d in test_ds:
        instances.append(pyg_to_instance(d, iid, max_nodes))
        test_ids.append(iid); iid += 1

    ds.instances = instances
    ds.splits = [{'train': train_ids, 'test': test_ids}]
    # Node-feature dim of GDCE's x (Mutag 13 atom types, NCI1 36 features).
    # The gcn_tf oracle's input layer is sized to this; for Mutag it must equal
    # len(atom_decoder) so classifier features and SMILES decoding share one index space.
    ds.n_node_types = int(instances[0].gdce_x.size(1)) if instances else 0
    return ds
