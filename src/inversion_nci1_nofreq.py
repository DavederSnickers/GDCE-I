"""
Inversion-based Counterfactual Generation for NCI1 (no frequency filter)

Captures Gumbel noise during forward diffusion, then re-runs reverse with new guidance.
Uses GIN or GCN classifier on binary adjacency (no bond types).

Metrics:
- Accuracy: classifier predicts target class
- GED: Graph Edit Distance (via NetworkX)
- MMD: degree / cluster / spectral (in-distribution evaluation)
"""

try:
    import graph_tool as gt
except ModuleNotFoundError:
    pass

import os
import sys as _sys
import warnings
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import wandb

import torch
import torch.nn.functional as F

torch.cuda.empty_cache()

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch_geometric.data import Data, Batch
from tqdm import tqdm

from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
from diffusion_model import LiftedDenoisingDiffusion
from diffusion_planar_discrete import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
from analysis.spectre_utils import PlanarSamplingMetrics
from analysis.visualization import NonMolecularVisualization

from src.diffusion.discrete_inversion import DiscreteInversion

_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shared_judge'))
from cf_dump import CFDumpWriter, edges_from_adj, node_types_from_x

warnings.filterwarnings("ignore", category=PossibleUserWarning)


# ---------------------------------------------------------------------------
# In-Distribution (MMD) helpers
# ---------------------------------------------------------------------------

def _mmd_process_tensor(x, y):
    """Pad two 1-D arrays to the same length."""
    n = max(len(x), len(y))
    if len(x) < n:
        x = np.hstack((x, np.zeros(n - len(x))))
    if len(y) < n:
        y = np.hstack((y, np.zeros(n - len(y))))
    return x, y


def _gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    """Gaussian kernel with EMD (Earth Mover's Distance) as distance."""
    import pyemd
    from scipy.linalg import toeplitz
    x = x.astype(float)
    y = y.astype(float)
    x, y = _mmd_process_tensor(x, y)
    n = len(x)
    d_mat = toeplitz(range(n)).astype(float) / distance_scaling
    emd_val = pyemd.emd(x, y, d_mat)
    return np.exp(-emd_val * emd_val / (2 * sigma * sigma))


def _disc(samples1, samples2, kernel, **kwargs):
    """Discrepancy D(samples1, samples2) under kernel."""
    d = 0.0
    for s1 in samples1:
        for s2 in samples2:
            d += kernel(s1, s2, **kwargs)
    total = len(samples1) * len(samples2)
    return d / total if total > 0 else 1e6


def _compute_mmd(samples1, samples2, kernel, is_hist=True, **kwargs):
    """MMD^2 between two sample sets."""
    if is_hist:
        samples1 = [s / np.sum(s) for s in samples1 if np.sum(s) > 0]
        samples2 = [s / np.sum(s) for s in samples2 if np.sum(s) > 0]
    if not samples1 or not samples2:
        return float('nan')
    return (_disc(samples1, samples1, kernel, **kwargs)
            + _disc(samples2, samples2, kernel, **kwargs)
            - 2 * _disc(samples1, samples2, kernel, **kwargs))


def _degree_mmd(ref_list, pred_list):
    pred_nonempty = [G for G in pred_list if G.number_of_nodes() > 0]
    if not ref_list or not pred_nonempty:
        return float('nan')
    s_ref  = [np.array(nx.degree_histogram(G)) for G in ref_list]
    s_pred = [np.array(nx.degree_histogram(G)) for G in pred_nonempty]
    return _compute_mmd(s_ref, s_pred, kernel=_gaussian_emd)


def _spectral_mmd(ref_list, pred_list):
    from scipy.linalg import eigvalsh
    pred_nonempty = [G for G in pred_list if G.number_of_nodes() > 0]
    if not ref_list or not pred_nonempty:
        return float('nan')

    def _spec(G):
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
        pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
        return pmf

    s_ref  = [_spec(G) for G in ref_list]
    s_pred = [_spec(G) for G in pred_nonempty]
    return _compute_mmd(s_ref, s_pred, kernel=_gaussian_emd)


def _clustering_mmd(ref_list, pred_list, bins=100):
    pred_nonempty = [G for G in pred_list if G.number_of_nodes() > 0]
    if not ref_list or not pred_nonempty:
        return float('nan')

    def _clust(G):
        hist, _ = np.histogram(
            list(nx.clustering(G).values()),
            bins=bins, range=(0.0, 1.0), density=False,
        )
        return hist

    s_ref  = [_clust(G) for G in ref_list]
    s_pred = [_clust(G) for G in pred_nonempty]
    return _compute_mmd(s_ref, s_pred, kernel=_gaussian_emd,
                        sigma=1.0 / 10, distance_scaling=bins)


def eval_mmd(ref_graphs, cf_graphs):
    """Compute degree / cluster / spectral MMD between two NetworkX graph lists."""
    return {
        'degree':   _degree_mmd(ref_graphs, cf_graphs),
        'cluster':  _clustering_mmd(ref_graphs, cf_graphs),
        'spectral': _spectral_mmd(ref_graphs, cf_graphs),
    }


def _adj_to_nx(adj, n_nodes):
    """Convert binary adjacency matrix (tensor or numpy) to undirected nx.Graph."""
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    if isinstance(adj, torch.Tensor):
        adj = adj.cpu().numpy()
    rows, cols = np.where(adj >= 1)
    for i, j in zip(rows, cols):
        if int(i) < int(j):
            G.add_edge(int(i), int(j))
    return G


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_resume(cfg, model_kwargs):
    """Resumes a run. Loads previous config without allowing to update keys (used for testing)."""
    saved_cfg = cfg.copy()
    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only

    batch_size = cfg.train.batch_size
    n_epochs = cfg.train.n_epochs
    n_test_molecules_to_sample = cfg.guidance.n_test_molecules_to_sample
    n_samples_per_test_molecule = cfg.guidance.n_samples_per_test_molecule
    s = cfg.guidance.s
    node_model_path = cfg.guidance.node_model_path
    build_with_partial_charges = cfg.guidance.build_with_partial_charges
    experiment_type = getattr(cfg.guidance, 'experiment_type', None)
    test_thresholds = getattr(cfg.guidance, 'test_thresholds', None)
    wandb_mode = cfg.general.wandb
    gpus = cfg.general.gpus
    include_split = cfg.guidance.include_split
    node_inference_method = cfg.guidance.node_inference_method
    guidance_target = cfg.guidance.guidance_target
    graph_classifier_path = getattr(cfg.guidance, 'graph_classifier_path', None)
    graph_classifier_type = getattr(cfg.guidance, 'graph_classifier_type', 'gin')
    s_values = getattr(cfg.guidance, 's_values', None)
    # get_resume replaces cfg wholesale with the config stored in the
    # checkpoint. Anything not saved here and written back below is lost, and
    # CLI overrides then silently have no effect.
    transfer_classifier_path = getattr(cfg.guidance, 'transfer_classifier_path', None)
    n_runs = getattr(cfg.guidance, 'n_runs', None)
    n_samples_per_run = getattr(cfg.guidance, 'n_samples_per_run', None)
    t_list = getattr(cfg.guidance, 't_list', None)
    t_step = getattr(cfg.guidance, 't_step', None)
    t_max = getattr(cfg.guidance, 't_max', None)
    dump_variant = getattr(cfg.guidance, 'dump_variant', None)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, strict=False, map_location='cpu', **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume, strict=False, map_location='cpu', **model_kwargs)
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name

    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.train.batch_size = batch_size
        cfg.train.n_epochs = n_epochs
        cfg.guidance.n_test_molecules_to_sample = n_test_molecules_to_sample
        cfg.guidance.n_samples_per_test_molecule = n_samples_per_test_molecule
        if transfer_classifier_path is not None:
            cfg.guidance.transfer_classifier_path = transfer_classifier_path
        if n_runs is not None:
            cfg.guidance.n_runs = n_runs
        if n_samples_per_run is not None:
            cfg.guidance.n_samples_per_run = n_samples_per_run
        if t_list is not None:
            cfg.guidance.t_list = t_list
        if t_step is not None:
            cfg.guidance.t_step = t_step
        if t_max is not None:
            cfg.guidance.t_max = t_max
        if dump_variant is not None:
            cfg.guidance.dump_variant = dump_variant
        cfg.guidance.s = s
        cfg.guidance.node_model_path = node_model_path
        cfg.guidance.build_with_partial_charges = build_with_partial_charges
        if experiment_type is not None:
            cfg.guidance.experiment_type = experiment_type
        if test_thresholds is not None:
            cfg.guidance.test_thresholds = test_thresholds
        cfg.general.wandb = wandb_mode
        cfg.general.gpus = gpus
        cfg.guidance.include_split = include_split
        cfg.guidance.node_inference_method = node_inference_method
        cfg.guidance.guidance_target = guidance_target
        cfg.guidance.graph_classifier_path = graph_classifier_path
        cfg.guidance.graph_classifier_type = graph_classifier_type
        if s_values is not None:
            cfg.guidance.s_values = s_values

    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

NUM_CLASSES = 2
CLASS_NAMES = {0: 'Inactive', 1: 'Active'}

# Further datasets with the same representation (binary adjacency, categorical
# node types) run through this same script; only the class names and the output
# directory depend on the dataset.
_CLASS_NAMES_BY_DATASET = {
    'nci1':         {0: 'Inactive', 1: 'Active'},
    'nci1_nofreq':  {0: 'Inactive', 1: 'Active'},
    'proteins':     {0: 'Non-enzyme', 1: 'Enzyme'},
    'imdb_binary':  {0: 'Action', 1: 'Romance'},
    # 0 and 1 were swapped until 2026-08-05. Two independent pieces of evidence
    # for the current assignment: (a) the word distribution in the test split --
    # y=0 is led by love/lol/hope/haha, y=1 by miss/feel/sleep/hate; (b) the
    # direction of the GDCE-I counterfactuals -- 1->0 inserts love/enjoy/haha,
    # 0->1 inserts miss/suck/hate/cry. The metrics are unaffected (all of them
    # are label-symmetric); the printed class names were not.
    'twitter':      {0: 'Positive', 1: 'Negative'},
}


def _apply_dataset_naming(dataset_name):
    """Set CLASS_NAMES for this dataset and return the output directory."""
    global CLASS_NAMES
    CLASS_NAMES = _CLASS_NAMES_BY_DATASET.get(dataset_name, CLASS_NAMES)
    return f"inversion_{dataset_name}_results"


def classify_graph(classifier, adj, node_features, device):
    """Classify a graph given binary adjacency matrix and node features.

    Args:
        classifier: GIN or GCN graph classifier.
        adj: Binary adjacency matrix (n_nodes x n_nodes).
        node_features: FloatTensor (n_nodes, in_dim) — one-hot atom types.
        device: Torch device.

    Returns: (pred_class, probs_array).
    """
    cf_x = node_features.to(device).float()
    cf_edge_index = adj.nonzero(as_tuple=False).t().contiguous()
    cf_data = Data(x=cf_x, edge_index=cf_edge_index)
    cf_batch = Batch.from_data_list([cf_data])
    with torch.no_grad():
        logits = classifier(cf_batch)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = logits.argmax(dim=-1).item()
    return pred, probs


def adj_from_dense(E_tensor, n_nodes):
    """Convert dense edge tensor to binary adjacency matrix (symmetric, no self-loops).

    Symmetrizes the result so it matches orig_adj (built from undirected PyG edge_index).
    Without symmetrization, compute_sparsity (triu) and classify_graph (full nonzero)
    would see different graphs — causing sparsity=0 but prediction flip.
    """
    if E_tensor.dim() == 3:
        adj = E_tensor.argmax(dim=-1)
        adj = (adj > 0).float()
    else:
        adj = (E_tensor > 0).float()
    adj = adj[:n_nodes, :n_nodes]
    adj = torch.max(adj, adj.t())   # symmetrize
    adj.fill_diagonal_(1)           # add self-loops to match original data preprocessing
    return adj


def compute_ged(orig_adj, cf_adj):
    """Compute Graph Edit Distance between two binary adjacency matrices."""
    G_orig = _adj_to_nx(orig_adj, orig_adj.shape[0])
    G_cf   = _adj_to_nx(cf_adj,   cf_adj.shape[0])
    try:
        ged = nx.graph_edit_distance(G_orig, G_cf, timeout=5)
        if ged is None:
            ged = -1
    except Exception:
        ged = -1
    return ged


def compute_sparsity(orig_adj, cf_adj, orig_x, cf_x):
    """Compute Node Sparsity and Edge Sparsity for NCI1 (binary graph, no edge types).

    Node Sparsity: fraction of nodes whose label (argmax of x) changed.
    Edge Sparsity: symmetric difference of edge sets / n_orig_edges (COMBINEX-style,
                   fixed node correspondence — appropriate for inversion).
    """
    # Node Sparsity
    n_orig = orig_x.size(0)
    n_cf = cf_x.size(0)
    orig_types = orig_x.argmax(dim=-1)
    cf_types = cf_x.argmax(dim=-1)
    n_common = min(n_orig, n_cf)
    changed_nodes = sum(orig_types[i].item() != cf_types[i].item() for i in range(n_common))
    changed_nodes += abs(n_orig - n_cf)
    node_sparsity = changed_nodes / max(n_orig, n_cf) if max(n_orig, n_cf) > 0 else 0.0

    # Edge Sparsity — from binary upper-triangle of adjacency matrices
    n = min(orig_adj.size(0), cf_adj.size(0))
    orig_upper = orig_adj[:n, :n].triu(diagonal=1)
    cf_upper   = cf_adj[:n, :n].triu(diagonal=1)
    orig_edges = set(map(tuple, orig_upper.nonzero().cpu().tolist()))
    cf_edges   = set(map(tuple, cf_upper.nonzero().cpu().tolist()))
    n_orig_edges = len(orig_edges)
    structural_changed = len(orig_edges.symmetric_difference(cf_edges))
    edge_sparsity = structural_changed / n_orig_edges if n_orig_edges > 0 else 0.0

    return node_sparsity, edge_sparsity


# ---------------------------------------------------------------------------
# Relabeling diagnostic
# ---------------------------------------------------------------------------

def print_relabeling_diagnostic(datamodule, classifier, cfg, device):
    """Compare original dataset labels (from CSV) vs. classifier-relabeled guidance labels."""
    import pathlib

    print("[Relabeling diagnostic: starting...]")
    print(f"  cfg.dataset.name  = {cfg.dataset.name}")

    print("\n" + "=" * 60)
    print("RELABELING DIAGNOSTIC")
    print("  Original labels (CSV) vs. classifier-relabeled guidance")
    print("=" * 60)

    base_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
    raw_dir = os.path.join(base_path, cfg.dataset.datadir, 'raw')

    for split_name in ['train', 'val', 'test']:
        csv_path = os.path.join(raw_dir, f'{split_name}.csv')
        if not os.path.exists(csv_path):
            print(f"  [{split_name}] CSV not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        orig_labels = torch.tensor(df['y'].values)

        curr_dataset = datamodule.dataloaders[split_name].dataset
        curr_labels = curr_dataset.data.guidance.argmax(dim=-1).cpu()

        n = len(orig_labels)
        assert len(curr_labels) == n, f"Size mismatch: CSV={n}, dataset={len(curr_labels)}"

        print(f"\n--- {split_name.upper()} ({n} graphs) ---")
        print(f"  Original label distribution (TUDataset):")
        for c in range(NUM_CLASSES):
            cnt = (orig_labels == c).sum().item()
            print(f"    {CLASS_NAMES[c]}: {cnt}/{n} ({100*cnt/n:.1f}%)")

        print(f"  After classifier relabeling (= diffusion model guidance):")
        for c in range(NUM_CLASSES):
            cnt = (curr_labels == c).sum().item()
            print(f"    {CLASS_NAMES[c]}: {cnt}/{n} ({100*cnt/n:.1f}%)")

        changed = (orig_labels != curr_labels).sum().item()
        print(f"  Total changed: {changed}/{n} ({100*changed/n:.1f}%)")
        print(f"  Confusion (original → relabeled):")
        for orig_c in range(NUM_CLASSES):
            for new_c in range(NUM_CLASSES):
                cnt = ((orig_labels == orig_c) & (curr_labels == new_c)).sum().item()
                tag = "same" if orig_c == new_c else "*** FLIPPED ***"
                print(f"    {CLASS_NAMES[orig_c]} → {CLASS_NAMES[new_c]}: {cnt:4d}  [{tag}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    print("=" * 60)
    print("INVERSION — NCI1 (no frequency filter)")
    print("=" * 60)

    dataset_config = cfg["dataset"]
    torch.set_num_threads(cfg.train.num_interops_workers)
    torch.set_num_interop_threads(cfg.train.num_interops_workers)
    seed_everything(cfg.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load dataset ----
    datamodule = SpectreGraphDataModule(cfg)
    sampling_metrics = PlanarSamplingMetrics(datamodule)
    dataset_infos = SpectreDatasetInfos(dataset_config)
    train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
    visualization_tools = NonMolecularVisualization()

    if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
    domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(
        datamodule=datamodule, extra_features=extra_features,
        domain_features=domain_features
    )

    model_kwargs = {
        'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
        'extra_features': extra_features, 'domain_features': domain_features
    }

    # ---- Load model ----
    cfg, model = get_resume(cfg, model_kwargs)

    print(f"\n  gdim (guidance dimension): {model.gdim}")
    print(f"  T (diffusion steps): {model.T}")
    _train_clf_path = getattr(model.cfg.guidance, 'graph_classifier_path', 'NOT SET')
    _train_label_src = getattr(model.cfg.guidance, 'label_source', 'original')
    print(f"  Diffusion model was trained with:")
    print(f"    label_source:          {_train_label_src}")
    print(f"    graph_classifier_path: {_train_clf_path}")
    print(f"  Current inversion uses:")
    print(f"    graph_classifier_path: {getattr(cfg.guidance, 'graph_classifier_path', 'NOT SET')}")

    os.chdir(cfg.general.test_only.split('checkpoints')[0])

    model.to(device)
    model.eval()

    # ---- Load classifier ----
    classifier = None
    classifier_path = getattr(cfg.guidance, 'graph_classifier_path', None)
    if classifier_path:
        classifier_type = getattr(cfg.guidance, 'graph_classifier_type', 'gin')
        if not str(classifier_path).endswith('.ckpt'):
            # Shared-judge export (`<prefix>.npz` + `<prefix>.json`) rather than a
            # Lightning checkpoint: the same classifier the baselines attack.
            from src.models.shared_judge_adapter import load_shared_judge
            classifier = load_shared_judge(str(classifier_path))
        elif classifier_type == 'gin':
            from src.models.Mutag_GIN import MutagGINLightning
            classifier = MutagGINLightning.load_from_checkpoint(classifier_path)
            print(f"Loaded GIN classifier from {classifier_path}")
        else:
            from src.models.NCI1_GCN import NCI1GCNLightning
            classifier = NCI1GCNLightning.load_from_checkpoint(classifier_path)
            print(f"Loaded GCN classifier from {classifier_path}")
        classifier.to(device)
        classifier.eval()

        # Sanity check: classifier in_dim vs dataset node feature dim
        clf_in_dim = classifier.hparams.get('in_dim') if hasattr(classifier, 'hparams') else None
        if clf_in_dim is None:
            _sd = torch.load(classifier_path, map_location='cpu').get('state_dict', {})
            _first_w = next((v for k, v in _sd.items() if 'weight' in k), None)
            clf_in_dim = _first_w.shape[1] if _first_w is not None else None

        sample = datamodule.dataloaders['test'].dataset[0]
        data_in_dim = sample.x.shape[1]

        if clf_in_dim is not None and clf_in_dim != data_in_dim:
            raise RuntimeError(
                f"Classifier/dataset mismatch: classifier in_dim={clf_in_dim}, "
                f"dataset in_dim={data_in_dim}. "
                f"Retrain the classifier or re-preprocess the dataset."
            )
        print(f"  Classifier in_dim={clf_in_dim}, dataset in_dim={data_in_dim} — OK")
    else:
        print("ERROR: No graph_classifier_path set in config. Cannot evaluate.")
        return

    # ---- Relabeling diagnostic ----
    try:
        print_relabeling_diagnostic(datamodule, classifier, cfg, device)
    except Exception as e:
        print(f"[Relabeling diagnostic failed: {e}]")
        import traceback; traceback.print_exc()

    # ---- Create inversion wrapper ----
    inverter = DiscreteInversion(model, device=device)

    # ---- Initialize wandb ----
    if cfg.general.wandb != 'disabled':
        utils.setup_wandb(cfg)

    # ---- Output directory ----
    # Dataset-dependent, or proteins/imdb_binary/twitter would all write into
    # the same directory and overwrite each other.
    # Optional second classifier (same architecture, different seed) for the
    # adversarial check. Without a path, nothing changes.
    transfer_classifier = None
    _tp = getattr(cfg.guidance, 'transfer_classifier_path', None)
    if _tp:
        if not str(_tp).endswith('.ckpt'):
            from src.models.shared_judge_adapter import load_shared_judge
            transfer_classifier = load_shared_judge(str(_tp)).to(device).eval()
        else:
            from src.models.NCI1_GCN import NCI1GCNLightning
            transfer_classifier = NCI1GCNLightning.load_from_checkpoint(
                _tp, map_location='cpu').to(device).eval()
        print(f"Transfer-Judge geladen: {_tp}")

    output_dir = _apply_dataset_naming(cfg.dataset.name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")

    # Graph-level counterfactual dump in the canonical format. Without it, a
    # finished run leaves only its metrics behind -- but NAFR needs the
    # counterfactuals themselves, to put them to the control classifier.
    # `variant` keeps proteins/twitter runs with different budgets apart.
    _variant = str(getattr(cfg.guidance, 'dump_variant', 's{}'.format(
        list(getattr(cfg.guidance, 's_values', [3]))[0])))
    cf_writer = CFDumpWriter(
        os.path.join(output_dir, f"cf_dump_{cfg.dataset.name}_{_variant}.jsonl"),
        method='gdce-i', dataset=cfg.dataset.name,
        judge=(classifier_path or 'unknown'),
        bond_source='binary', variant=_variant,
    )

    # ---- Parameters ----
    T = int(model.T)
    # The budget list is configurable: the default tries 110 budgets per graph
    # (1..100 in steps of 1, then 110..200 in steps of 10), each with a full
    # reverse replay. guidance.t_step/t_max coarsens that grid and cuts the run
    # time proportionally; guidance.t_list sets the list outright.
    _t_list = getattr(cfg.guidance, 't_list', None)
    _t_step = getattr(cfg.guidance, 't_step', None)
    if _t_list:
        t_values = [int(t) for t in _t_list]
    elif _t_step:
        _t_max = int(getattr(cfg.guidance, 't_max', 200))
        t_values = list(range(int(_t_step), _t_max + 1, int(_t_step)))
    else:
        t_values = list(range(1, 101)) + list(range(110, 201, 10))
    t_values = sorted(set(t for t in t_values if 1 <= t <= T))
    s_values = list(getattr(cfg.guidance, 's_values', [3]))
    N_RUNS = getattr(cfg.guidance, 'n_runs', 5)
    N_SAMPLES_PER_RUN = getattr(cfg.guidance, 'n_samples_per_run', 100)
    RUN_SEEDS = list(range(42, 42 + N_RUNS))

    print(f"\nParameters:")
    print(f"  T: {T}")
    print(f"  t_values: 1 to {max(t_values)} ({len(t_values)} values, tried in ascending order)")
    print(f"  s_values: {s_values}")
    print(f"  N_RUNS: {N_RUNS}  (seeds: {RUN_SEEDS})")
    print(f"  N_SAMPLES_PER_RUN: {N_SAMPLES_PER_RUN}  (stratified)")
    print(f"  Inversion strategy: iterative (smallest t first, stop at first flip)")

    # ---- Get datasets ----
    test_dataset = datamodule.dataloaders['test'].dataset
    print(f"Test set size: {len(test_dataset)}")

    # ---- Diagnostic: classifier prediction & guidance distribution on ALL splits ----
    for split_name in ['train', 'val', 'test']:
        split_dataset = datamodule.dataloaders[split_name].dataset
        n_split = len(split_dataset)

        clf_pred_counts = {c: 0 for c in range(NUM_CLASSES)}
        guidance_counts = {c: 0 for c in range(NUM_CLASSES)}
        label_vs_pred = defaultdict(int)

        with torch.no_grad():
            for diag_idx in range(n_split):
                d = split_dataset[diag_idx]
                n = d.x.size(0)
                adj = torch.zeros((n, n), device=device)
                src, dst = d.edge_index
                adj[src.to(device), dst.to(device)] = 1
                pass  # self-loops kept: classifier was trained with them
                pred, _ = classify_graph(classifier, adj, d.x, device)
                guidance_label = d.guidance.argmax(dim=-1).item()
                clf_pred_counts[pred] += 1
                guidance_counts[guidance_label] += 1
                label_vs_pred[(guidance_label, pred)] += 1

        print(f"\n--- {split_name.upper()} set ({n_split} graphs) ---")
        print(f"  Guidance distribution (what the generative model was trained on):")
        for c in range(NUM_CLASSES):
            print(f"    {CLASS_NAMES[c]}: {guidance_counts[c]}/{n_split} ({100*guidance_counts[c]/n_split:.1f}%)")
        print(f"  Classifier prediction distribution:")
        for c in range(NUM_CLASSES):
            print(f"    {CLASS_NAMES[c]}: {clf_pred_counts[c]}/{n_split} ({100*clf_pred_counts[c]/n_split:.1f}%)")
        print(f"  Confusion (guidance_label, clf_pred) -> count:")
        for (lbl, pred), cnt in sorted(label_vs_pred.items()):
            match = "MATCH" if lbl == pred else "MISMATCH"
            print(f"    Guidance={CLASS_NAMES[lbl]}, Pred={CLASS_NAMES[pred]}: {cnt} [{match}]")
    print()

    # ---- Pre-classify all test graphs once (used for stratified sampling) ----
    print(f"\nPre-classifying all {len(test_dataset)} test graphs for stratified sampling...")
    all_test_samples = []
    with torch.no_grad():
        for data_idx in range(len(test_dataset)):
            data = test_dataset[data_idx]
            n_nodes = data.x.size(0)
            orig_adj = torch.zeros((n_nodes, n_nodes), device=device)
            edge_src, edge_dst = data.edge_index
            orig_adj[edge_src.to(device), edge_dst.to(device)] = 1
            pass  # self-loops kept: classifier was trained with them
            orig_pred, orig_probs = classify_graph(classifier, orig_adj, data.x, device)
            all_test_samples.append({
                'data_idx': data_idx,
                'orig_pred': orig_pred,
                'orig_probs': orig_probs,
            })
    class_totals = {c: sum(1 for s in all_test_samples if s['orig_pred'] == c)
                    for c in range(NUM_CLASSES)}
    print(f"  Class distribution: " +
          ", ".join(f"{CLASS_NAMES[c]}: {class_totals[c]}" for c in range(NUM_CLASSES)))

    # ---- Multi-run evaluation ----
    all_inversion_rows = []
    mmd_rows = []
    run_summary_rows = []

    for run_idx, run_seed in enumerate(RUN_SEEDS):
        print(f"\n{'#'*90}")
        print(f"  RUN {run_idx + 1}/{N_RUNS}  (seed={run_seed})")
        print(f"{'#'*90}")

        # ---- Stratified sampling: maintain class ratio from test set ----
        rng = np.random.default_rng(run_seed)
        by_class = defaultdict(list)
        for s in all_test_samples:
            by_class[s['orig_pred']].append(s)

        selected_meta = []
        for c in range(NUM_CLASSES):
            n_class = round(N_SAMPLES_PER_RUN * class_totals[c] / len(all_test_samples))
            n_class = min(n_class, len(by_class[c]))
            chosen_indices = rng.choice(len(by_class[c]), size=n_class, replace=False)
            selected_meta.extend([by_class[c][i] for i in chosen_indices])
        rng.shuffle(selected_meta)
        selected_meta = selected_meta[:N_SAMPLES_PER_RUN]

        print(f"  Sampled {len(selected_meta)} graphs:")
        for c in range(NUM_CLASSES):
            n = sum(1 for s in selected_meta if s['orig_pred'] == c)
            print(f"    {CLASS_NAMES[c]}: {n}")

        # ---- Prepare full sample dicts (dense data, adjacency) ----
        selected_samples = []
        with torch.no_grad():
            for meta in selected_meta:
                data_idx = meta['data_idx']
                data = test_dataset[data_idx]
                data_batch = Batch.from_data_list([data.to(device)])
                dense_data, node_mask = utils.to_dense(
                    data_batch.x, data_batch.edge_index, data_batch.edge_attr, data_batch.batch
                )
                n_nodes = data.x.size(0)
                orig_adj = torch.zeros((n_nodes, n_nodes), device=device)
                edge_src, edge_dst = data.edge_index
                orig_adj[edge_src.to(device), edge_dst.to(device)] = 1
                pass  # self-loops kept: classifier was trained with them
                selected_samples.append({
                    'data_idx': data_idx,
                    'data': data,
                    'data_batch': data_batch,
                    'dense_data': dense_data,
                    'node_mask': node_mask,
                    'orig_adj': orig_adj,
                    'orig_pred': meta['orig_pred'],
                    'orig_probs': meta['orig_probs'],
                    'n_orig_edges': int(orig_adj.sum().item() / 2),
                })

        # ---- Run inversion for each s value ----
        for s_val in s_values:
            print(f"\n{'='*60}")
            print(f"  RUN {run_idx+1}  GUIDANCE SCALE s = {s_val}")
            print(f"{'='*60}")

            ref_graphs_s = []
            cf_graphs_s  = []

            with open_dict(cfg):
                cfg.guidance.s = s_val

            with torch.no_grad():
                for sample_idx, sample in enumerate(selected_samples):
                    data_idx = sample['data_idx']
                    data = sample['data']
                    dense_data = sample['dense_data']
                    node_mask = sample['node_mask']
                    orig_adj = sample['orig_adj']
                    orig_pred = sample['orig_pred']
                    orig_probs = sample['orig_probs']
                    n_orig_edges = sample['n_orig_edges']

                    target_class = 1 - orig_pred
                    source_class = orig_pred

                    orig_gui = torch.zeros(1, NUM_CLASSES, device=device)
                    orig_gui[0, orig_pred] = 1.0
                    target_gui = torch.zeros(1, NUM_CLASSES, device=device)
                    target_gui[0, target_class] = 1.0

                    dataset_label = data.guidance.argmax(dim=-1).item()
                    n_nodes_orig = data.x.size(0)
                    avg_degree = (2 * n_orig_edges / n_nodes_orig) if n_nodes_orig > 0 else 0
                    print(f"\n--- [run={run_idx+1}, s={s_val}] Sample {sample_idx} "
                          f"(idx: {data_idx}, {CLASS_NAMES[orig_pred]}) ---")
                    print(f"  Original: Pred={CLASS_NAMES[orig_pred]} ({orig_probs[orig_pred]:.1%}), "
                          f"DatasetLabel={CLASS_NAMES[dataset_label]}, "
                          f"Nodes={n_nodes_orig}, Edges={n_orig_edges}, AvgDegree={avg_degree:.2f}")
                    if dataset_label != orig_pred:
                        print(f"  *** MISMATCH: DatasetLabel={CLASS_NAMES[dataset_label]} != "
                              f"ClassifierPred={CLASS_NAMES[orig_pred]} ***")

                    # ---- INVERSION (iterative, smallest t first) ----
                    inv_t_used = None
                    inv_ged_final = -1
                    inv_mod_rate_final = -1.0
                    inv_node_sparsity_final = -1.0
                    inv_edge_sparsity_final = -1.0
                    inv_success = False
                    inv_transfer_pred = -1
                    inv_transfer_flip = -1
                    inv_cf_graph = None

                    try:
                        inv_output = inverter.inversion_forward(
                            dense_data.X.to(device),
                            dense_data.E.to(device),
                            node_mask.to(device),
                            guidance=orig_gui,
                            prog_bar=False
                        )

                        for t in sorted(t_values):
                            skip = T - t
                            try:
                                X_edited, E_edited = inverter.inversion_reverse(
                                    X_T=inv_output.X_T,
                                    E_T=inv_output.E_T,
                                    node_mask=node_mask.to(device),
                                    new_guidance=target_gui,
                                    gumbels_X=inv_output.gumbels_X,
                                    gumbels_E=inv_output.gumbels_E,
                                    trajectory_X=inv_output.trajectory_X,
                                    trajectory_E=inv_output.trajectory_E,
                                    skip=skip,
                                    prog_bar=False
                                )
                                cf_X = X_edited[0]
                                cf_E = E_edited[0]
                                n_nodes = data.x.size(0)
                                cf_X_soft = cf_X[:n_nodes]
                                # Discretize to one-hot — model outputs soft probabilities,
                                # classifier was trained on discrete data, and sparsity
                                # must be consistent with what the classifier receives.
                                cf_X_actual = torch.zeros_like(cf_X_soft)
                                cf_X_actual.scatter_(-1, cf_X_soft.argmax(dim=-1, keepdim=True), 1.0)
                                cf_adj = adj_from_dense(cf_E, n_nodes)

                                pred_class, probs_arr = classify_graph(classifier, cf_adj, cf_X_actual, device)
                                success = (pred_class == target_class)

                                conf_str = f"{probs_arr[pred_class]:.1%}"
                                status = "SUCCESS" if success else "fail"
                                print(f"    t={t:3d}: {CLASS_NAMES[pred_class]} ({conf_str}) [{status}]")

                                if success:
                                    ged = compute_ged(orig_adj, cf_adj)
                                    mod_rate = ged / n_orig_edges if ged >= 0 and n_orig_edges > 0 else -1.0
                                    mr_str = f"{mod_rate:.3f}" if mod_rate >= 0 else "N/A"
                                    node_sp, edge_sp = compute_sparsity(
                                        orig_adj, cf_adj, data.x.to(device), cf_X_actual
                                    )
                                    print(f"    -> Flipped at t={t}, GED={ged}, ModRate={mr_str}, "
                                          f"NodeSparsity={node_sp:.3f}, EdgeSparsity={edge_sp:.3f}")
                                    inv_t_used = t
                                    inv_ged_final = ged
                                    inv_mod_rate_final = mod_rate
                                    inv_node_sparsity_final = node_sp
                                    inv_edge_sparsity_final = edge_sp
                                    inv_success = True
                                    # Adversarial check: does the same classifier,
                                    # trained independently (different seed), flip
                                    # too? If not, the counterfactual is an artefact
                                    # of the one model rather than a real class change.
                                    if transfer_classifier is not None:
                                        try:
                                            t_pred, _ = classify_graph(
                                                transfer_classifier, cf_adj, cf_X_actual, device)
                                            inv_transfer_pred = int(t_pred)
                                            inv_transfer_flip = int(t_pred == target_class)
                                            print(f"       Transfer-Judge: "
                                                  f"{CLASS_NAMES[int(t_pred)]} "
                                                  f"[{'FLIP' if inv_transfer_flip else 'kein Flip'}]")
                                        except Exception as e:
                                            print(f"       Transfer-Judge Fehler: {e}")
                                    ref_graphs_s.append(_adj_to_nx(orig_adj, n_nodes))
                                    cf_graphs_s.append(_adj_to_nx(cf_adj, n_nodes))
                                    inv_cf_graph = (node_types_from_x(cf_X_actual),
                                                    edges_from_adj(cf_adj))
                                    break

                            except Exception as e:
                                print(f"    t={t}: Error: {e}")

                    except Exception as e:
                        print(f"  Forward pass failed: {e}")
                        import traceback
                        traceback.print_exc()

                    if not inv_success:
                        print(f"  -> No t succeeded. FAIL.")

                    if inv_cf_graph is not None:
                        cf_writer.write(
                            run=run_idx, seed=run_seed, sample_idx=sample_idx,
                            data_idx=data_idx, source_class=source_class,
                            target_class=target_class,
                            node_types=inv_cf_graph[0], edges=inv_cf_graph[1],
                            own_pred=target_class, projected=True,
                            s=s_val, t=inv_t_used,
                        )

                    all_inversion_rows.append({
                        'run': run_idx,
                        'seed': run_seed,
                        's': s_val,
                        'sample_idx': sample_idx,
                        'data_idx': data_idx,
                        'source_class': source_class,
                        'target_class': target_class,
                        'orig_pred': orig_pred,
                        'success': inv_success,
                        'transfer_pred': inv_transfer_pred,
                        'transfer_flip': inv_transfer_flip,
                        't_used': inv_t_used if inv_t_used is not None else -1,
                        'ged': inv_ged_final,
                        'mod_rate': inv_mod_rate_final,
                        'node_sparsity': inv_node_sparsity_final,
                        'edge_sparsity': inv_edge_sparsity_final,
                        'n_orig_edges': n_orig_edges,
                    })

                    # ---- wandb: per-sample ----
                    if wandb.run:
                        log_dict = {
                            'run': run_idx,
                            'sample_idx': sample_idx,
                            f"sample/s{s_val}/success": int(inv_success),
                            f"sample/s{s_val}/t_used": inv_t_used if inv_t_used is not None else -1,
                        }
                        if inv_success:
                            log_dict.update({
                                f"sample/s{s_val}/ged": inv_ged_final if inv_ged_final >= 0 else float('nan'),
                                f"sample/s{s_val}/mod_rate": inv_mod_rate_final if inv_mod_rate_final >= 0 else float('nan'),
                                f"sample/s{s_val}/node_sparsity": inv_node_sparsity_final,
                                f"sample/s{s_val}/edge_sparsity": inv_edge_sparsity_final,
                            })
                        wandb.log(log_dict)

            # ---- MMD for this (run, s) ----
            if ref_graphs_s and cf_graphs_s:
                print(f"\n  MMD for run={run_idx+1}, s={s_val} "
                      f"({len(cf_graphs_s)} successful CFs):")
                try:
                    mmd = eval_mmd(ref_graphs_s, cf_graphs_s)
                    print(f"    Degree MMD:   {mmd['degree']:.6f}")
                    print(f"    Cluster MMD:  {mmd['cluster']:.6f}")
                    print(f"    Spectral MMD: {mmd['spectral']:.6f}")
                    mmd_rows.append({
                        'run': run_idx, 'seed': run_seed, 's': s_val,
                        'n_graphs': len(ref_graphs_s),
                        'degree_mmd': mmd['degree'],
                        'cluster_mmd': mmd['cluster'],
                        'spectral_mmd': mmd['spectral'],
                    })
                except Exception as e:
                    print(f"    MMD computation failed: {e}")
            else:
                print(f"\n  No successful CFs for run={run_idx+1}, s={s_val} — MMD skipped.")

            # ---- Per-run aggregate metrics for this (run, s) ----
            df_rs = pd.DataFrame([r for r in all_inversion_rows
                                  if r['run'] == run_idx and r['s'] == s_val])
            n_total_rs = len(df_rs)
            n_succ_rs = int(df_rs['success'].sum())
            df_succ_rs = df_rs[df_rs['success']]

            def _mean(col):
                vals = df_succ_rs[df_succ_rs[col] >= 0][col] if n_succ_rs > 0 else pd.Series([], dtype=float)
                return vals.mean() if len(vals) > 0 else float('nan')

            acc_rs = n_succ_rs / n_total_rs if n_total_rs > 0 else float('nan')

            run_summary_rows.append({
                'run': run_idx, 'seed': run_seed, 's': s_val,
                'accuracy': acc_rs,
                'avg_ged': _mean('ged'),
                'avg_mod_rate': _mean('mod_rate'),
                'avg_node_sparsity': _mean('node_sparsity'),
                'avg_edge_sparsity': _mean('edge_sparsity'),
                'avg_t': _mean('t_used'),
                'n_total': n_total_rs,
                'n_success': n_succ_rs,
            })

            # ---- wandb: per-run aggregate ----
            if wandb.run:
                wandb.log({
                    'run': run_idx,
                    f"run/s{s_val}/accuracy": acc_rs,
                    f"run/s{s_val}/avg_ged": _mean('ged'),
                    f"run/s{s_val}/avg_mod_rate": _mean('mod_rate'),
                    f"run/s{s_val}/avg_node_sparsity": _mean('node_sparsity'),
                    f"run/s{s_val}/avg_edge_sparsity": _mean('edge_sparsity'),
                    f"run/s{s_val}/avg_t": _mean('t_used'),
                })

    # ---- Save CSVs ----
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")

    cf_writer.close()
    print(f"CF-Dump: {cf_writer.path}")

    df_inv = pd.DataFrame(all_inversion_rows)
    inv_csv = os.path.join(output_dir, "GDCEfresh.csv")
    df_inv.to_csv(inv_csv, index=False)
    print(f"Inversion results: {inv_csv}")

    df_run_summary = pd.DataFrame(run_summary_rows)
    run_csv = os.path.join(output_dir, "run_summary.csv")
    df_run_summary.to_csv(run_csv, index=False)
    print(f"Run summary:       {run_csv}")

    if mmd_rows:
        df_mmd = pd.DataFrame(mmd_rows)
        mmd_csv = os.path.join(output_dir, "mmd_results.csv")
        df_mmd.to_csv(mmd_csv, index=False)
        print(f"MMD results:       {mmd_csv}")

    # ---- Final summary: mean ± std across runs ----
    print(f"\n{'#'*90}")
    print(f"FINAL SUMMARY: MEAN ± STD ACROSS {N_RUNS} RUNS  (n={N_SAMPLES_PER_RUN} per run, stratified)")
    print(f"{'#'*90}")

    METRICS = [
        ('accuracy',           'Accuracy'),
        ('avg_ged',            'Avg GED'),
        ('avg_mod_rate',       'Avg ModRate'),
        ('avg_node_sparsity',  'Avg NodeSparsity'),
        ('avg_edge_sparsity',  'Avg EdgeSparsity'),
        ('avg_t',              'Avg t used'),
    ]

    for s_val in s_values:
        df_s = df_run_summary[df_run_summary['s'] == s_val]
        print(f"\n--- s = {s_val}  ({len(df_s)} runs) ---")
        for col, label in METRICS:
            vals = df_s[col].dropna()
            if len(vals) == 0:
                print(f"  {label:<28}: N/A")
            else:
                mean_v = vals.mean()
                std_v  = vals.std(ddof=1) if len(vals) > 1 else 0.0
                print(f"  {label:<28}: {mean_v:.4f} ± {std_v:.4f}")

        # ---- wandb: final mean ± std ----
        if wandb.run:
            log_final = {}
            for col, label in METRICS:
                vals = df_s[col].dropna()
                if len(vals) > 0:
                    log_final[f"final/s{s_val}/{col}_mean"] = vals.mean()
                    log_final[f"final/s{s_val}/{col}_std"]  = vals.std(ddof=1) if len(vals) > 1 else 0.0
            wandb.log(log_final)

    if mmd_rows:
        print(f"\n{'#'*90}")
        print("IN-DISTRIBUTION EVALUATION (MMD) — mean ± std across runs")
        print(f"{'#'*90}")
        df_mmd_full = pd.DataFrame(mmd_rows)
        for s_val in s_values:
            df_ms = df_mmd_full[df_mmd_full['s'] == s_val]
            print(f"\n--- s = {s_val} ---")
            for col in ['degree_mmd', 'cluster_mmd', 'spectral_mmd']:
                vals = df_ms[col].dropna()
                if len(vals) == 0:
                    print(f"  {col:<16}: N/A")
                else:
                    print(f"  {col:<16}: {vals.mean():.6f} ± {vals.std(ddof=1) if len(vals) > 1 else 0.0:.6f}")

    print(f"\n{'='*60}")
    print(f"\nAll results saved in: {os.path.abspath(output_dir)}")
    print("DONE!")


if __name__ == '__main__':
    main()
