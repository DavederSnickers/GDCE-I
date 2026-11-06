"""
Inversion-based Counterfactual Generation for Mutagenicity (with bond types)

Captures Gumbel noise during forward diffusion, then re-runs reverse with new guidance.
Uses GINEConv classifier that takes edge_attr (5-class bond types).

Metrics:
- Accuracy: classifier predicts target class
- GED: Graph Edit Distance (via NetworkX)
"""

try:
    import graph_tool as gt
except ModuleNotFoundError:
    pass

import os
import sys as _sys
import warnings
from collections import defaultdict

_sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shared_judge'))
from cf_dump import CFDumpWriter, edges_from_edge_index, node_types_from_x

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

warnings.filterwarnings("ignore", category=PossibleUserWarning)


# ---------------------------------------------------------------------------
# In-Distribution (MMD) helpers  — self-contained, uses GDCE dependencies
# (pyemd, scipy, networkx, numpy — all listed in requirements.txt)
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


def _edge_index_to_nx(edge_index, n_nodes):
    """Convert a PyG edge_index to an undirected nx.Graph."""
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    if edge_index.numel() > 0:
        ei = edge_index.cpu().numpy()
        for i, j in zip(ei[0], ei[1]):
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
    experiment_type = cfg.guidance.experiment_type
    test_thresholds = cfg.guidance.test_thresholds
    wandb_mode = cfg.general.wandb
    gpus = cfg.general.gpus
    include_split = cfg.guidance.include_split
    node_inference_method = cfg.guidance.node_inference_method
    guidance_target = cfg.guidance.guidance_target
    graph_classifier_path = getattr(cfg.guidance, 'graph_classifier_path', None)
    s_values = getattr(cfg.guidance, 's_values', None)
    # These live under cfg.guidance, which update_config_with_new_keys() does NOT
    # merge back — so capture them here and re-inject below, else CLI overrides are
    # silently lost when get_resume() rebuilds cfg from the checkpoint.
    inversion_method = getattr(cfg.guidance, 'inversion_method', 'fresh')
    inversion_margin = getattr(cfg.guidance, 'inversion_margin', 0.0)
    n_runs = getattr(cfg.guidance, 'n_runs', 1)
    n_samples_per_run = getattr(cfg.guidance, 'n_samples_per_run', 100)
    recon_check = getattr(cfg.guidance, 'recon_check', False)
    require_valid = getattr(cfg.guidance, 'require_valid', False)
    compute_mmd = getattr(cfg.guidance, 'compute_mmd', True)
    max_cf_images = getattr(cfg.guidance, 'max_cf_images', 30)
    stop_at_first_flip = getattr(cfg.guidance, 'stop_at_first_flip', False)

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
        cfg.guidance.s = s
        cfg.guidance.node_model_path = node_model_path
        cfg.guidance.build_with_partial_charges = build_with_partial_charges
        cfg.guidance.experiment_type = experiment_type
        cfg.guidance.test_thresholds = test_thresholds
        cfg.general.wandb = wandb_mode
        cfg.general.gpus = gpus
        cfg.guidance.include_split = include_split
        cfg.guidance.node_inference_method = node_inference_method
        cfg.guidance.guidance_target = guidance_target
        cfg.guidance.graph_classifier_path = graph_classifier_path
        if s_values is not None:
            cfg.guidance.s_values = s_values
        cfg.guidance.inversion_method = inversion_method
        cfg.guidance.inversion_margin = inversion_margin
        cfg.guidance.n_runs = n_runs
        cfg.guidance.n_samples_per_run = n_samples_per_run
        cfg.guidance.recon_check = recon_check
        cfg.guidance.require_valid = require_valid
        cfg.guidance.compute_mmd = compute_mmd
        cfg.guidance.max_cf_images = max_cf_images
        cfg.guidance.stop_at_first_flip = stop_at_first_flip

    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

NUM_CLASSES = 2

# Label names per dataset. Benzene shares this script because it uses the same
# representation as mutagenicity_mol (5-class bond types, GINE classifier,
# atom_decoder.json, no self-loops); main() rebinds CLASS_NAMES from cfg.
_CLASS_NAMES_BY_DATASET = {
    'mutagenicity_mol': {0: 'Mutagenic', 1: 'Non-mutagenic'},
    'benzene': {0: 'No ring', 1: 'Benzene ring'},
}
CLASS_NAMES = _CLASS_NAMES_BY_DATASET['mutagenicity_mol']


def edge_types_from_dense(E_tensor, n_nodes):
    """Convert dense 5-class edge tensor to (edge_index, edge_attr).

    Args:
        E_tensor: Dense edge tensor of shape (max_nodes, max_nodes) or
                  (max_nodes, max_nodes, 5) with class probabilities/one-hot.
        n_nodes: Number of actual nodes (crop to n_nodes x n_nodes).

    Returns:
        edge_index: LongTensor of shape (2, E)
        edge_attr:  FloatTensor of shape (E, 5) — one-hot bond types
    """
    if E_tensor.dim() == 3:
        # (max_nodes, max_nodes, 5): take argmax to get class indices
        et_matrix = E_tensor[:n_nodes, :n_nodes].argmax(dim=-1)
    else:
        et_matrix = E_tensor[:n_nodes, :n_nodes].long()

    # No self-loops (molecular convention)
    et_matrix.fill_diagonal_(0)

    # Find non-zero entries (bond exists)
    edge_index = (et_matrix > 0).nonzero(as_tuple=False).t().contiguous()

    if edge_index.numel() > 0:
        et_vals = et_matrix[edge_index[0], edge_index[1]]
        edge_attr = F.one_hot(et_vals, num_classes=5).float()
    else:
        edge_attr = torch.zeros((0, 5), dtype=torch.float32, device=E_tensor.device)

    return edge_index, edge_attr


def classify_graph(classifier, edge_index, edge_attr, node_features, device):
    """Classify a graph given edge_index, edge_attr, and node features.

    Args:
        classifier: The GINEConv graph classifier model.
        edge_index: LongTensor (2, E).
        edge_attr: FloatTensor (E, 5) — one-hot bond types.
        node_features: FloatTensor (n_nodes, in_dim) — one-hot atom types.
        device: Torch device.

    Returns: (pred_class, probs_array).
    """
    cf_x = node_features.to(device).float()
    cf_edge_index = edge_index.to(device)
    cf_edge_attr = edge_attr.to(device)
    cf_data = Data(x=cf_x, edge_index=cf_edge_index, edge_attr=cf_edge_attr)
    cf_batch = Batch.from_data_list([cf_data])
    with torch.no_grad():
        logits = classifier(cf_batch)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = logits.argmax(dim=-1).item()
    return pred, probs


def visualize_counterfactual(
    orig_node_types, orig_edge_index, orig_edge_attr,
    cf_node_types, cf_edge_index, cf_edge_attr,
    atom_decoder, sample_idx, s_val, t_used,
    orig_class_name, cf_class_name, output_dir,
    stats=None,
):
    """Save side-by-side SVG + optional PNG of original and counterfactual.

    Layout of the PNG: original on the left, counterfactual on the right, and a
    statistics panel underneath (sparsities, GED, SMILES validity, ...).

    Highlight colors:
      - Red   bonds : removed in CF
      - Green bonds : added in CF
      - Orange bonds: bond type changed
      - Yellow atoms: atom type changed

    Args:
        stats: optional dict of metric name -> value, rendered below the images.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import RWMol
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        print("    -> RDKit not available — skipping visualization.")
        return

    _BOND_TYPES = {
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
        4: Chem.rdchem.BondType.AROMATIC,
    }

    def _build_mol(node_types, edge_index, edge_attr):
        mol = RWMol()
        node_syms = []
        for idx in node_types.cpu().tolist():
            sym = atom_decoder.get(str(idx), atom_decoder.get(idx, 'C'))
            mol.AddAtom(Chem.Atom(sym))
            node_syms.append(sym)
        bond_types_map = {}  # (i, j) with i<j -> int bond type
        if edge_index.numel() > 0:
            ei = edge_index.cpu().numpy()
            ea = edge_attr.argmax(dim=-1).cpu().numpy()
            seen = set()
            for (src, dst), bt in zip(zip(ei[0], ei[1]), ea):
                i, j = int(src), int(dst)
                if i >= j or (i, j) in seen:
                    continue
                btype = _BOND_TYPES.get(int(bt))
                if btype is not None:
                    mol.AddBond(i, j, btype)
                    bond_types_map[(i, j)] = int(bt)
                    seen.add((i, j))
        # Assign formal charges: N with ≥2 O neighbors → nitro group (N+, O-)
        nbrs = defaultdict(list)
        for (i, j), bt in bond_types_map.items():
            nbrs[i].append((j, bt))
            nbrs[j].append((i, bt))
        for atom_idx, sym in enumerate(node_syms):
            if sym == 'N':
                o_nbrs = [(nb, bt) for nb, bt in nbrs[atom_idx] if node_syms[nb] == 'O']
                if len(o_nbrs) >= 2:
                    mol.GetAtomWithIdx(atom_idx).SetFormalCharge(1)
                    for nb, bt in o_nbrs:
                        if bt == 1:  # single bond → O-
                            mol.GetAtomWithIdx(nb).SetFormalCharge(-1)
                            break
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        return mol.GetMol()

    def _edge_dict(edge_index, edge_attr):
        d = {}
        if edge_index.numel() == 0:
            return d
        ei = edge_index.cpu().numpy()
        ea = edge_attr.argmax(dim=-1).cpu().numpy()
        for (src, dst), bt in zip(zip(ei[0], ei[1]), ea):
            i, j = int(src), int(dst)
            if i < j:
                d[(i, j)] = int(bt)
        return d

    orig_mol = _build_mol(orig_node_types, orig_edge_index, orig_edge_attr)
    cf_mol   = _build_mol(cf_node_types,   cf_edge_index,   cf_edge_attr)
    orig_edges = _edge_dict(orig_edge_index, orig_edge_attr)
    cf_edges   = _edge_dict(cf_edge_index,   cf_edge_attr)

    # Changed atom types
    n_common = min(len(orig_node_types), len(cf_node_types))
    changed_atoms = [i for i in range(n_common) if orig_node_types[i].item() != cf_node_types[i].item()]

    # --- Highlights for original ---
    orig_h_atoms = list(changed_atoms)
    orig_atom_cols = {i: (1.0, 0.85, 0.0) for i in changed_atoms}   # yellow
    orig_h_bonds, orig_bond_cols = [], {}
    for (i, j), bt in orig_edges.items():
        bond = orig_mol.GetBondBetweenAtoms(i, j)
        if bond is None:
            continue
        bid = bond.GetIdx()
        if (i, j) not in cf_edges:
            orig_h_bonds.append(bid)
            orig_bond_cols[bid] = (0.9, 0.2, 0.2)       # red: removed
        elif cf_edges[(i, j)] != bt:
            orig_h_bonds.append(bid)
            orig_bond_cols[bid] = (1.0, 0.55, 0.0)      # orange: type changed

    # --- Highlights for CF ---
    cf_h_atoms = list(changed_atoms)
    cf_atom_cols = {i: (1.0, 0.85, 0.0) for i in changed_atoms}
    cf_h_bonds, cf_bond_cols = [], {}
    for (i, j), bt in cf_edges.items():
        bond = cf_mol.GetBondBetweenAtoms(i, j)
        if bond is None:
            continue
        bid = bond.GetIdx()
        if (i, j) not in orig_edges:
            cf_h_bonds.append(bid)
            cf_bond_cols[bid] = (0.2, 0.78, 0.2)        # green: added
        elif orig_edges.get((i, j)) != bt:
            cf_h_bonds.append(bid)
            cf_bond_cols[bid] = (1.0, 0.55, 0.0)        # orange: type changed

    W, H = 500, 380

    def _svg(mol, h_atoms, atom_cols, h_bonds, bond_cols):
        drawer = rdMolDraw2D.MolDraw2DSVG(W, H)
        drawer.drawOptions().addStereoAnnotation = False
        try:
            rdMolDraw2D.PrepareAndDrawMolecule(
                drawer, mol,
                highlightAtoms=h_atoms if h_atoms else None,
                highlightAtomColors=atom_cols if atom_cols else None,
                highlightBonds=h_bonds if h_bonds else None,
                highlightBondColors=bond_cols if bond_cols else None,
            )
        except Exception:
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    prefix = f"sample{sample_idx:03d}_s{s_val}_t{t_used}"
    orig_svg = _svg(orig_mol, orig_h_atoms, orig_atom_cols, orig_h_bonds, orig_bond_cols)
    cf_svg   = _svg(cf_mol,   cf_h_atoms,   cf_atom_cols,   cf_h_bonds,   cf_bond_cols)

    orig_path = os.path.join(output_dir, f"{prefix}_original_{orig_class_name}.svg")
    cf_path   = os.path.join(output_dir, f"{prefix}_cf_{cf_class_name}.svg")
    with open(orig_path, 'w') as f:
        f.write(orig_svg)
    with open(cf_path, 'w') as f:
        f.write(cf_svg)

    # --- Combined PNG (requires cairosvg + Pillow) ---
    try:
        import io
        from PIL import Image, ImageDraw as PILDraw, ImageFont

        def _png_image(mol, h_atoms, atom_cols, h_bonds, bond_cols, svg_text):
            """PNG via RDKit's own Cairo backend; falls back to cairosvg.

            The container ships RDKit with Cairo support but no cairosvg, so
            rendering straight to PNG avoids an extra dependency (and a .sif
            rebuild) entirely.
            """
            try:
                drawer = rdMolDraw2D.MolDraw2DCairo(W, H)
                drawer.drawOptions().addStereoAnnotation = False
                try:
                    rdMolDraw2D.PrepareAndDrawMolecule(
                        drawer, mol,
                        highlightAtoms=h_atoms if h_atoms else None,
                        highlightAtomColors=atom_cols if atom_cols else None,
                        highlightBonds=h_bonds if h_bonds else None,
                        highlightBondColors=bond_cols if bond_cols else None,
                    )
                except Exception:
                    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
                drawer.FinishDrawing()
                return Image.open(io.BytesIO(drawer.GetDrawingText()))
            except Exception:
                import cairosvg
                return Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg_text.encode())))

        orig_png = _png_image(orig_mol, orig_h_atoms, orig_atom_cols,
                              orig_h_bonds, orig_bond_cols, orig_svg)
        cf_png   = _png_image(cf_mol, cf_h_atoms, cf_atom_cols,
                              cf_h_bonds, cf_bond_cols, cf_svg)

        margin, label_h, legend_h = 12, 28, 24

        # Statistics panel: two columns of "key: value" lines under the molecules.
        stat_items = list(stats.items()) if stats else []
        n_stat_rows = (len(stat_items) + 1) // 2
        stats_h = (n_stat_rows * 20 + 14) if stat_items else 0

        total_w = W * 2 + margin * 3
        total_h = H + label_h + legend_h + stats_h + margin * 2
        combined = Image.new('RGBA', (total_w, total_h), (255, 255, 255, 255))
        combined.paste(orig_png, (margin, label_h + margin))
        combined.paste(cf_png,   (W + margin * 2, label_h + margin))

        draw = PILDraw.Draw(combined)

        def _font(size, bold=False):
            name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
            try:
                return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
            except Exception:
                return ImageFont.load_default()

        font, font_b = _font(14), _font(14, bold=True)

        draw.text((margin, 6),         f"Original  ({orig_class_name})",   fill=(40, 40, 40), font=font_b)
        draw.text((W + margin * 2, 6), f"Counterfactual  ({cf_class_name})  t={t_used}",
                  fill=(40, 40, 40), font=font_b)

        # Legend directly under the molecules
        legend_y = label_h + margin + H + 4
        for i, (color, label) in enumerate([
            ((200, 40, 40), "removed"), ((51, 200, 51), "added"),
            ((255, 140, 0), "type changed"), ((255, 220, 0), "atom changed"),
        ]):
            x_legend = margin + i * 110
            draw.rectangle([x_legend, legend_y, x_legend + 12, legend_y + 12], fill=color)
            draw.text((x_legend + 16, legend_y - 1), label, fill=(60, 60, 60), font=font)

        # Statistics block
        if stat_items:
            sep_y = legend_y + legend_h - 4
            draw.line([(margin, sep_y), (total_w - margin, sep_y)], fill=(200, 200, 200), width=1)
            col_w = (total_w - 2 * margin) // 2
            for i, (k, v) in enumerate(stat_items):
                col, row = i % 2, i // 2
                x = margin + col * col_w
                y = sep_y + 10 + row * 20
                draw.text((x, y), f"{k}:", fill=(90, 90, 90), font=font)
                draw.text((x + 150, y), str(v), fill=(20, 20, 20), font=font_b)

        combined_path = os.path.join(output_dir, f"{prefix}_comparison.png")
        combined.convert('RGB').save(combined_path)
        print(f"    -> Visualization: {combined_path}")

    except ImportError:
        # cairosvg/Pillow not installed — SVGs are sufficient
        print(f"    -> Visualization SVGs saved: {os.path.basename(orig_path)}, {os.path.basename(cf_path)}")
    except Exception as e:
        print(f"    -> SVGs saved (PNG failed: {e}): {os.path.basename(orig_path)}, {os.path.basename(cf_path)}")


def graph_to_smiles(node_types_tensor, edge_index, edge_attr, atom_decoder, strict=True):
    """Convert a generated graph to a SMILES string via RDKit.

    Args:
        node_types_tensor: LongTensor (n_nodes,) — remapped atom type indices.
        edge_index: LongTensor (2, E).
        edge_attr: FloatTensor (E, 5) — one-hot bond types (0=no bond skipped,
                   1=single, 2=double, 3=triple, 4=aromatic).
        atom_decoder: dict mapping str(remapped_idx) → element symbol,
                      loaded from data/mutagenicity_mol/raw/atom_decoder.json.
        strict: if True, full RDKit sanitization (incl. valence check).
                if False, skips valence/properties check — appropriate for the
                Mutagenicity dataset which contains no formal charges.

    Returns:
        smiles (str) or None if conversion fails.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import RWMol
    except ImportError:
        print("RDKit not available — cannot convert to SMILES.")
        return None

    # TUDataset Mutagenicity bond-type encoding: argmax gives 0=single, 1=double, 2=triple, 3=aromatic
    # In download() we add +1 offset → stored as 1=single, 2=double, 3=triple, 4=aromatic(never in Kekulé)
    _BOND_TYPES = {
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
        4: Chem.rdchem.BondType.AROMATIC,
    }

    mol = RWMol()
    atom_indices = node_types_tensor.cpu().tolist()
    node_syms = []
    for idx in atom_indices:
        symbol = atom_decoder.get(str(idx), atom_decoder.get(idx, 'C'))
        mol.AddAtom(Chem.Atom(symbol))
        node_syms.append(symbol)

    # Add bonds (only i < j to avoid duplicates)
    bond_types_map = {}  # (i, j) with i<j -> int bond type
    if edge_index.numel() > 0:
        ei = edge_index.cpu().numpy()
        ea = edge_attr.argmax(dim=-1).cpu().numpy()  # bond type 1-4
        added = set()
        for (src, dst), bt in zip(zip(ei[0], ei[1]), ea):
            i, j = int(src), int(dst)
            if i >= j or (i, j) in added:
                continue
            if bt > 0 and bt in _BOND_TYPES:
                mol.AddBond(i, j, _BOND_TYPES[bt])
                bond_types_map[(i, j)] = int(bt)
                added.add((i, j))

    # Assign formal charges for nitro groups: N with >=2 O neighbors -> N+, O-.
    # The Mutagenicity dataset stores no formal charges, so nitroaromatics (very
    # common among mutagens) would otherwise fail strict valence sanitization and
    # be wrongly counted as invalid. Mirrors visualize_counterfactual._build_mol.
    nbrs = defaultdict(list)
    for (i, j), bt in bond_types_map.items():
        nbrs[i].append((j, bt))
        nbrs[j].append((i, bt))
    for atom_idx, sym in enumerate(node_syms):
        if sym == 'N':
            o_nbrs = [(nb, bt) for nb, bt in nbrs[atom_idx] if node_syms[nb] == 'O']
            if len(o_nbrs) >= 2:
                mol.GetAtomWithIdx(atom_idx).SetFormalCharge(1)
                for nb, bt in o_nbrs:
                    if bt == 1:  # single bond -> O-
                        mol.GetAtomWithIdx(nb).SetFormalCharge(-1)
                        break

    try:
        if strict:
            Chem.SanitizeMol(mol)
        else:
            # Skip valence/properties check — the Mutagenicity dataset has no
            # formal charges, so nitro groups etc. would fail strict valence checks.
            Chem.SanitizeMol(
                mol,
                Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
            )
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def load_atom_decoder(cfg):
    """Load atom_decoder.json from the dataset raw directory."""
    import json, pathlib
    base_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
    raw_dir = os.path.join(base_path, cfg.dataset.datadir, 'raw')
    path = os.path.join(raw_dir, 'atom_decoder.json')
    if not os.path.exists(path):
        print(f"WARNING: atom_decoder.json not found at {path}. "
              f"Run dataset download first.")
        return None
    with open(path) as f:
        return json.load(f)


def compute_ged(orig_edge_index, cf_edge_index, n_nodes):
    """Compute Graph Edit Distance between two graphs given edge_index tensors."""
    def _to_nx(edge_index, n):
        G = nx.Graph()
        G.add_nodes_from(range(n))
        if edge_index.numel() > 0:
            ei = edge_index.cpu().numpy()
            for i, j in zip(ei[0], ei[1]):
                if i < j:
                    G.add_edge(int(i), int(j))
        return G

    G_orig = _to_nx(orig_edge_index, n_nodes)
    G_cf = _to_nx(cf_edge_index, n_nodes)
    try:
        ged = nx.graph_edit_distance(G_orig, G_cf, timeout=5)
        if ged is None:
            ged = -1
    except Exception:
        ged = -1
    return ged


def compute_sparsity(orig_node_types, cf_node_types,
                     orig_edge_index, orig_edge_attr,
                     cf_edge_index, cf_edge_attr):
    """Compute Node Sparsity, Edge Sparsity, and EdgeType Sparsity.

    Node Sparsity:     fraction of nodes whose atom type changed (/ n_orig_nodes).
    Edge Sparsity:     fraction of added/removed edges (/ n_orig_edges), COMBINEX-style.
    EdgeType Sparsity: fraction of edges present in both graphs but with changed bond type
                       (/ n_orig_edges).
    """
    # Node Sparsity
    n_orig = len(orig_node_types)
    n_cf = len(cf_node_types)
    n_common = min(n_orig, n_cf)
    changed_nodes = sum(
        orig_node_types[i].item() != cf_node_types[i].item()
        for i in range(n_common)
    )
    changed_nodes += abs(n_orig - n_cf)  # nodes added/removed
    n_total_nodes = max(n_orig, n_cf)
    node_sparsity = changed_nodes / n_total_nodes if n_total_nodes > 0 else 0.0

    # Build edge dicts: (i, j) with i<j -> bond type index
    def _edge_dict(edge_index, edge_attr):
        d = {}
        if edge_index.numel() == 0:
            return d
        ei = edge_index.cpu()
        ea = edge_attr.cpu().argmax(dim=-1) if edge_attr.dim() > 1 else edge_attr.cpu()
        for k, (src, dst) in enumerate(zip(ei[0].tolist(), ei[1].tolist())):
            if src < dst:
                d[(src, dst)] = int(ea[k])
        return d

    orig_edges = _edge_dict(orig_edge_index, orig_edge_attr)
    cf_edges = _edge_dict(cf_edge_index, cf_edge_attr)
    n_orig_edges = len(orig_edges)

    orig_keys = set(orig_edges)
    cf_keys = set(cf_edges)

    # Edge Sparsity (COMBINEX-style): added + removed edges / n_orig_edges
    structural_changed = len(orig_keys.symmetric_difference(cf_keys))
    edge_sparsity = structural_changed / n_orig_edges if n_orig_edges > 0 else 0.0

    # EdgeType Sparsity: edges in both graphs but bond type changed / n_orig_edges
    common_keys = orig_keys & cf_keys
    type_changed = sum(orig_edges[k] != cf_edges[k] for k in common_keys)
    edgetype_sparsity = type_changed / n_orig_edges if n_orig_edges > 0 else 0.0

    return node_sparsity, edge_sparsity, edgetype_sparsity


# ---------------------------------------------------------------------------
# Relabeling diagnostic
# ---------------------------------------------------------------------------

def print_relabeling_diagnostic(datamodule, cfg):
    """Compare original dataset labels (from CSV) vs. relabeled guidance labels."""
    import pathlib

    print("[Relabeling diagnostic: starting...]")
    print(f"  cfg.dataset.name  = {cfg.dataset.name}")
    print(f"  cfg.dataset.datadir = {cfg.dataset.datadir}")

    print("\n" + "=" * 60)
    print("RELABELING DIAGNOSTIC")
    print("  Original labels (CSV) vs. GINEConv-relabeled guidance")
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

        print(f"  After GINEConv relabeling (= diffusion model guidance):")
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
    global CLASS_NAMES
    _ds_name = cfg["dataset"]["name"]
    if _ds_name not in _CLASS_NAMES_BY_DATASET:
        raise ValueError(
            f"inversion_mutag_mol only supports edge-typed datasets "
            f"{sorted(_CLASS_NAMES_BY_DATASET)}, got '{_ds_name}'.")
    CLASS_NAMES = _CLASS_NAMES_BY_DATASET[_ds_name]

    print("=" * 60)
    print(f"INVERSION — {_ds_name.upper()} (with bond types)")
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

    # ---- Load atom_decoder for SMILES reconstruction ----
    atom_decoder = load_atom_decoder(cfg)
    if not atom_decoder:
        print("WARNING: atom_decoder.json missing — SMILES output disabled. "
              "Run dataset download to generate it.")

    # ---- Load GINEConv classifier ----
    classifier = None
    # Shared judge takes precedence: the single classifier every method in the
    # comparison targets, so flip rates are comparable across methods.
    _judge_path = getattr(cfg.guidance, 'shared_judge_path', None)
    classifier_path = getattr(cfg.guidance, 'graph_classifier_path', None)
    if _judge_path:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)), 'shared_judge'))
        from shared_judge import load_judge, SharedJudgeBatchAdapter
        classifier = SharedJudgeBatchAdapter(load_judge(_judge_path, device='cpu'))
        classifier.to(device)
        classifier.eval()
        print(f"Loaded SHARED JUDGE from {_judge_path}")
    elif classifier_path:
        from src.models.Mutag_GINE import MutagGINELightning
        classifier = MutagGINELightning.load_from_checkpoint(classifier_path)
        classifier.to(device)
        classifier.eval()
        print(f"Loaded GINEConv classifier from {classifier_path}")
    else:
        print("ERROR: No graph_classifier_path set in config. Cannot evaluate.")
        return

    # ---- Relabeling diagnostic ----
    try:
        print_relabeling_diagnostic(datamodule, cfg)
    except Exception as e:
        print(f"[Relabeling diagnostic failed: {e}]")
        import traceback; traceback.print_exc()

    # ---- Create inversion wrapper ----
    inverter = DiscreteInversion(model, device=device)

    # ---- Auto-name the run from method + guidance scale(s) (before wandb init) ----
    _method = getattr(cfg.guidance, 'inversion_method', 'fresh')
    _svals = list(getattr(cfg.guidance, 's_values', [getattr(cfg.guidance, 's', 3)]))
    _stag = "-".join(str(v) for v in _svals)
    with open_dict(cfg):
        cfg.general.name = f"{cfg.general.name}_{_method}_s{_stag}"
    print(f"Run name: {cfg.general.name}")

    # ---- Initialize wandb ----
    if cfg.general.wandb != 'disabled':
        utils.setup_wandb(cfg)

    # ---- Output directory (per inversion method, so parallel jobs don't clobber) ----
    output_dir = f"inversion_{_ds_name}_results_{_method}_s{_stag}"
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, 'counterfactual_images')
    os.makedirs(viz_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")

    # Graph-level counterfactual dump: lets any judge re-score this run later
    # without re-generating it. The CSV's SMILES column cannot serve that purpose
    # -- benzene's aromatic bonds make RDKit drop ~70% of the flips.
    cf_writer = CFDumpWriter(
        os.path.join(output_dir, f"cf_dump_{_ds_name}_{_method}_s{_stag}.jsonl"),
        method='gdce-i', dataset=_ds_name,
        judge=(_judge_path or classifier_path or 'unknown'),
        bond_source='predicted', variant=f"{_method}_s{_stag}",
    )

    # Cap the number of rendered comparison images: a full run produces hundreds of
    # flips and each render costs an RDKit depiction + a PNG composite.
    max_viz = int(getattr(cfg.guidance, 'max_cf_images', 30))
    stop_at_first_flip = bool(getattr(cfg.guidance, 'stop_at_first_flip', False))
    if stop_at_first_flip:
        print('stop_at_first_flip=True -> scan stops at the FIRST flip; the '
              '[FIRST VALID] block then only covers samples whose first flip '
              'happened to be chemically valid.')
    n_viz_saved = [0]   # list so the nested sample loop can mutate it
    print(f"Counterfactual images: up to {max_viz} in {viz_dir}")

    # ---- Parameters ----
    T = int(model.T)
    t_values = list(range(1, 101)) + list(range(110, 201, 10))
    t_values = sorted(set(t for t in t_values if t <= T))
    s_values = list(getattr(cfg.guidance, 's_values', [3]))
    N_RUNS = getattr(cfg.guidance, 'n_runs', 1)
    N_SAMPLES_PER_RUN = getattr(cfg.guidance, 'n_samples_per_run', 20)
    RUN_SEEDS = list(range(42, 42 + N_RUNS))

    # ---- Inversion variant: 'fresh' (original) | 'posterior' (Plan A) ----
    inversion_method = getattr(cfg.guidance, 'inversion_method', 'fresh')
    inversion_margin = float(getattr(cfg.guidance, 'inversion_margin', 0.0))
    recon_check = bool(getattr(cfg.guidance, 'recon_check', False))
    # require_valid: pick the minimal t that flips AND yields a valid SMILES
    # (falls back to the first flip if none is valid). False = original behavior.
    require_valid = bool(getattr(cfg.guidance, 'require_valid', False))
    compute_mmd = bool(getattr(cfg.guidance, 'compute_mmd', True))
    # 'fresh' records the full chain (skip = T - t). 'posterior' records the model
    # posterior only up to the deepest t actually used (skip = rec_steps - t).
    rec_steps = max(t_values) if inversion_method == 'posterior' else T

    print("\n" + "=" * 70)
    _desc = {
        'posterior': 'Plan A: posterior Gumbel — exact reconstruction',
        'fresh': 'naive: fresh Gumbel captured along the forward chain',
        'noinv': 'no inversion: fresh sampling per reverse step (original GDCE)',
    }.get(inversion_method, inversion_method)
    print(f"  INVERSION METHOD: {inversion_method.upper()}   ({_desc})")
    print("=" * 70)

    print(f"\nParameters:")
    print(f"  T: {T}")
    print(f"  inversion_method: {inversion_method}  (rec_steps={rec_steps}, margin={inversion_margin})")
    print(f"  t_values (tried in ascending order): {sorted(t_values)}")
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

                # Build edge_index + edge_attr from stored edge data
                # d.edge_attr shape: (E, 5) — already one-hot bond types
                diag_edge_index = d.edge_index.to(device)
                diag_edge_attr = d.edge_attr.to(device) if d.edge_attr is not None else \
                    torch.zeros((d.edge_index.size(1), 5), device=device)

                pred, _ = classify_graph(classifier, diag_edge_index, diag_edge_attr, d.x, device)
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
            orig_edge_index = data.edge_index.to(device)
            orig_edge_attr = data.edge_attr.to(device) if data.edge_attr is not None else \
                torch.zeros((data.edge_index.size(1), 5), device=device)
            orig_pred, orig_probs = classify_graph(
                classifier, orig_edge_index, orig_edge_attr, data.x, device)
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
    run_summary_rows = []  # one row per (run, s) for mean±std aggregation

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

        # ---- Prepare full sample dicts (dense data, edge tensors) ----
        selected_samples = []
        with torch.no_grad():
            for meta in selected_meta:
                data_idx = meta['data_idx']
                data = test_dataset[data_idx]
                data_batch = Batch.from_data_list([data.to(device)])
                dense_data, node_mask = utils.to_dense(
                    data_batch.x, data_batch.edge_index, data_batch.edge_attr, data_batch.batch
                )
                orig_edge_index = data.edge_index.to(device)
                orig_edge_attr = data.edge_attr.to(device) if data.edge_attr is not None else \
                    torch.zeros((data.edge_index.size(1), 5), device=device)
                selected_samples.append({
                    'data_idx': data_idx,
                    'data': data,
                    'data_batch': data_batch,
                    'dense_data': dense_data,
                    'node_mask': node_mask,
                    'orig_edge_index': orig_edge_index,
                    'orig_edge_attr': orig_edge_attr,
                    'orig_pred': meta['orig_pred'],
                    'orig_probs': meta['orig_probs'],
                    'n_orig_edges': orig_edge_index.size(1) // 2,
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
                        orig_edge_index = sample['orig_edge_index']
                        orig_edge_attr = sample['orig_edge_attr']
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
                        print(f"\n--- [run={run_idx+1}, s={s_val}] Sample {sample_idx} "
                              f"(idx: {data_idx}, {CLASS_NAMES[orig_pred]}) ---")
                        n_nodes_orig = data.x.size(0)
                        avg_degree = (2 * n_orig_edges / n_nodes_orig) if n_nodes_orig > 0 else 0
                        print(f"  Original: Pred={CLASS_NAMES[orig_pred]} ({orig_probs[orig_pred]:.1%}), "
                              f"DatasetLabel={CLASS_NAMES[dataset_label]}, "
                              f"Nodes={n_nodes_orig}, Edges(undirected)={n_orig_edges}, "
                              f"AvgDegree={avg_degree:.2f}")

                        # ---- Original SMILES (for post-hoc analysis in CSV) ----
                        orig_smiles = None
                        if atom_decoder is not None:
                            orig_atom_types = data.x.argmax(dim=-1)
                            orig_smiles = graph_to_smiles(
                                orig_atom_types, orig_edge_index.cpu(),
                                orig_edge_attr.cpu(), atom_decoder, strict=True)
                            if orig_smiles is None:
                                # Fall back to non-strict so the real molecule is
                                # still recorded even if strict valence fails.
                                orig_smiles = graph_to_smiles(
                                    orig_atom_types, orig_edge_index.cpu(),
                                    orig_edge_attr.cpu(), atom_decoder, strict=False)

                        # ---- INVERSION ----
                        # Track two operating points per sample:
                        #   flip  = metrics at the FIRST t that flips the classifier
                        #   valid = metrics at the FIRST t that flips AND is strict-valid
                        flip = None
                        valid = None

                        try:
                            # Two inversion variants (toggle via guidance.inversion_method):
                            #   'fresh'     — original: fresh Gumbel along the forward chain.
                            #   'posterior' — Plan A: edit-friendly posterior Gumbel, records
                            #                 only up to max(t_values) for efficiency.
                            if inversion_method == 'posterior':
                                inv_output = inverter.inversion_forward_posterior(
                                    dense_data.X.to(device),
                                    dense_data.E.to(device),
                                    node_mask.to(device),
                                    guidance=orig_gui,
                                    num_steps=rec_steps,
                                    margin=inversion_margin,
                                    prog_bar=False
                                )
                            else:
                                inv_output = inverter.inversion_forward(
                                    dense_data.X.to(device),
                                    dense_data.E.to(device),
                                    node_mask.to(device),
                                    guidance=orig_gui,
                                    prog_bar=False
                                )

                            # ---- RECONSTRUCTION CHECK (diagnostic) ----
                            # Replay with the SAME (orig) guidance at full recorded
                            # depth. 'posterior' must reconstruct X_0 exactly (GED=0);
                            # 'fresh' will not (forward/reverse are not an inverse pair).
                            if recon_check:
                                X_rec, E_rec = inverter.inversion_reverse(
                                    X_T=inv_output.X_T, E_T=inv_output.E_T,
                                    node_mask=node_mask.to(device),
                                    new_guidance=orig_gui,
                                    gumbels_X=inv_output.gumbels_X,
                                    gumbels_E=inv_output.gumbels_E,
                                    trajectory_X=inv_output.trajectory_X,
                                    trajectory_E=inv_output.trajectory_E,
                                    skip=0, prog_bar=False,
                                )
                                n_nodes = data.x.size(0)
                                orig_node_types = data.x.argmax(dim=-1).cpu()
                                rec_node_types = X_rec[0][:n_nodes].argmax(dim=-1).cpu()
                                node_match = (rec_node_types == orig_node_types).float().mean().item()
                                rec_ei, rec_ea = edge_types_from_dense(E_rec[0], n_nodes)
                                rec_ged = compute_ged(orig_edge_index, rec_ei.to(device), n_nodes)
                                _, e_sp, et_sp = compute_sparsity(
                                    orig_node_types, rec_node_types,
                                    orig_edge_index, orig_edge_attr,
                                    rec_ei.to(device), rec_ea.to(device))
                                exact = (node_match == 1.0 and rec_ged == 0)
                                print(f"  [recon-check | {inversion_method}] "
                                      f"node_match={node_match:.3f}  GED(X0,recon)={rec_ged}  "
                                      f"edge_sparsity={e_sp:.3f}  edgetype_sparsity={et_sp:.3f}  "
                                      f"EXACT={exact}")
                                continue

                            for t in sorted(t_values):
                                skip = rec_steps - t
                                try:
                                    if inversion_method == 'noinv':
                                        # Original GDCE (no inversion): forward-perturb to G_t,
                                        # then guided reverse with FRESH sampling (no re-injection).
                                        X_edited, E_edited = inverter.sample_without_inversion(
                                            X_T=inv_output.trajectory_X[t].to(device),
                                            E_T=inv_output.trajectory_E[t].to(device),
                                            node_mask=node_mask.to(device),
                                            guidance=target_gui,
                                            start_t=t,
                                            prog_bar=False,
                                        )
                                    else:
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
                                    cf_X_actual = cf_X[:n_nodes]

                                    cf_edge_index, cf_edge_attr = edge_types_from_dense(cf_E, n_nodes)
                                    cf_edge_index = cf_edge_index.to(device)
                                    cf_edge_attr = cf_edge_attr.to(device)

                                    pred_class, probs_arr = classify_graph(
                                        classifier, cf_edge_index, cf_edge_attr, cf_X_actual, device
                                    )
                                    success = (pred_class == target_class)

                                    conf_str = f"{probs_arr[pred_class]:.1%}"
                                    status = "SUCCESS" if success else "fail"
                                    print(f"    t={t:3d}: {CLASS_NAMES[pred_class]} ({conf_str}) [{status}]")

                                    if success:
                                        ged = compute_ged(orig_edge_index, cf_edge_index, n_nodes)
                                        mod_rate = ged / n_orig_edges if ged >= 0 and n_orig_edges > 0 else -1
                                        orig_node_types = data.x.argmax(dim=-1)
                                        cf_node_types_arg = cf_X_actual.argmax(dim=-1)
                                        node_sp, edge_sp, edgetype_sp = compute_sparsity(
                                            orig_node_types, cf_node_types_arg,
                                            orig_edge_index, orig_edge_attr,
                                            cf_edge_index, cf_edge_attr,
                                        )
                                        smiles_strict = None
                                        smiles_any = None
                                        if atom_decoder is not None:
                                            cf_atom_types = cf_X_actual.argmax(dim=-1)
                                            smiles_strict = graph_to_smiles(
                                                cf_atom_types, cf_edge_index.cpu(),
                                                cf_edge_attr.cpu(), atom_decoder, strict=True)
                                            # Non-strict fallback so even an invalid first
                                            # flip still records a CF SMILES for inspection.
                                            smiles_any = smiles_strict if smiles_strict is not None \
                                                else graph_to_smiles(
                                                    cf_atom_types, cf_edge_index.cpu(),
                                                    cf_edge_attr.cpu(), atom_decoder, strict=False)
                                        is_strict_valid = smiles_strict is not None
                                        # Connectivity: a valid SMILES containing '.' is several
                                        # disconnected fragments, not a single molecule. A flip
                                        # achieved by shattering the graph is chemically meaningless.
                                        n_fragments = (smiles_strict.count('.') + 1) if is_strict_valid else -1
                                        smiles_connected = bool(is_strict_valid and n_fragments == 1)

                                        m = {
                                            't': t, 'ged': ged, 'mod_rate': mod_rate,
                                            'node_sparsity': node_sp, 'edge_sparsity': edge_sp,
                                            'edgetype_sparsity': edgetype_sp,
                                            'smiles_valid': bool(is_strict_valid),
                                            'smiles_connected': smiles_connected,
                                            'n_fragments': n_fragments,
                                            'smiles': smiles_any,
                                            'cf_nx': _edge_index_to_nx(cf_edge_index, n_nodes),
                                            # Kept so the flip that ends up in the CSV can be
                                            # dumped as a graph once the t-scan has finished.
                                            'cf_node_types': node_types_from_x(cf_X_actual),
                                            'cf_edges': edges_from_edge_index(cf_edge_index, cf_edge_attr),
                                        }

                                        if flip is None:
                                            flip = m
                                            print(f"    -> First FLIP at t={t}, GED={ged}, "
                                                  f"EdgeSparsity={edge_sp:.3f}, strict_valid={m['smiles_valid']}")
                                            if atom_decoder is not None and n_viz_saved[0] < max_viz:
                                                n_viz_saved[0] += 1
                                                visualize_counterfactual(
                                                    orig_node_types, orig_edge_index, orig_edge_attr,
                                                    cf_node_types_arg, cf_edge_index, cf_edge_attr,
                                                    atom_decoder, sample_idx, s_val, t,
                                                    CLASS_NAMES[orig_pred], CLASS_NAMES[target_class],
                                                    viz_dir,
                                                    stats={
                                                        'Edge Sparsity': f"{edge_sp:.3f}",
                                                        'Node Sparsity': f"{node_sp:.3f}",
                                                        'EdgeType Sparsity': f"{edgetype_sp:.3f}",
                                                        'GED': f"{ged}",
                                                        'Modification Rate': f"{mod_rate:.3f}",
                                                        'Nodes / Edges': f"{n_nodes_orig} / {n_orig_edges}",
                                                        'Inversion step t': f"{t}",
                                                        'Guidance scale s': f"{s_val}",
                                                        'CF valid SMILES': 'yes' if m['smiles_valid'] else 'no',
                                                        'CF fragments': f"{n_fragments}" if m['smiles_valid'] else 'n/a',
                                                        'CF confidence': f"{probs_arr[pred_class]:.1%}",
                                                        'Transition': f"{CLASS_NAMES[orig_pred]} -> {CLASS_NAMES[target_class]}",
                                                    },
                                                )
                                        if valid is None and is_strict_valid:
                                            valid = m
                                            print(f"    -> First VALID flip at t={t}, GED={ged}, "
                                                  f"EdgeSparsity={edge_sp:.3f}")
                                            print(f"    -> CF SMILES (strict): {smiles_strict}")
                                            # The first flip is frequently not a chemically
                                            # valid molecule, so render this operating point
                                            # too -- it is the one worth showing in a figure.
                                            if (atom_decoder is not None
                                                    and n_viz_saved[0] < max_viz
                                                    and flip is not m):
                                                n_viz_saved[0] += 1
                                                visualize_counterfactual(
                                                    orig_node_types, orig_edge_index, orig_edge_attr,
                                                    cf_node_types_arg, cf_edge_index, cf_edge_attr,
                                                    atom_decoder, sample_idx, s_val, t,
                                                    CLASS_NAMES[orig_pred], CLASS_NAMES[target_class],
                                                    viz_dir,
                                                    stats={
                                                        'Edge Sparsity': f"{edge_sp:.3f}",
                                                        'Node Sparsity': f"{node_sp:.3f}",
                                                        'EdgeType Sparsity': f"{edgetype_sp:.3f}",
                                                        'GED': f"{ged}",
                                                        'Modification Rate': f"{mod_rate:.3f}",
                                                        'Nodes / Edges': f"{n_nodes_orig} / {n_orig_edges}",
                                                        'Inversion step t': f"{t}",
                                                        'Guidance scale s': f"{s_val}",
                                                        'CF valid SMILES': 'yes (first valid)',
                                                        'CF fragments': f"{n_fragments}",
                                                        'CF confidence': f"{probs_arr[pred_class]:.1%}",
                                                        'Transition': f"{CLASS_NAMES[orig_pred]} -> {CLASS_NAMES[target_class]}",
                                                    },
                                                )

                                        # stop_at_first_flip: report only the first flip and
                                        # stop scanning immediately. Without it the scan keeps
                                        # going until it also finds a chemically VALID flip,
                                        # which costs most of the runtime (valid flips sit at
                                        # much larger t than first flips).
                                        if stop_at_first_flip:
                                            break
                                        # Both operating points found -> stop scanning.
                                        if valid is not None:
                                            break

                                except Exception as e:
                                    print(f"    t={t}: Error: {e}")

                        except Exception as e:
                            print(f"  Forward pass failed: {e}")
                            import traceback
                            traceback.print_exc()

                        inv_success = flip is not None
                        valid_found = valid is not None
                        if not inv_success:
                            print(f"  -> No t succeeded. FAIL.")

                        # MMD (if enabled) uses the first-flip CF as the primary graph.
                        if compute_mmd and flip is not None:
                            n_nodes_cf = data.x.size(0)
                            ref_graphs_s.append(_edge_index_to_nx(orig_edge_index, n_nodes_cf))
                            cf_graphs_s.append(flip['cf_nx'])

                        def _fg(d, k, default=-1):
                            return d[k] if d is not None else default

                        def _fgs(d, k):
                            """String-valued field; None/missing -> '' for clean CSV."""
                            v = d[k] if d is not None else None
                            return v if v is not None else ''

                        # Dump exactly the counterfactual the CSV row reports -- the
                        # first flip -- so both files describe the same object.
                        if flip is not None:
                            cf_writer.write(
                                run=run_idx, seed=run_seed, sample_idx=sample_idx,
                                data_idx=data_idx, source_class=source_class,
                                target_class=target_class,
                                node_types=flip['cf_node_types'], edges=flip['cf_edges'],
                                own_pred=target_class, projected=True,
                                s=s_val, t=flip['t'], smiles_valid=bool(flip['smiles_valid']),
                            )

                        all_inversion_rows.append({
                            'run': run_idx,
                            'seed': run_seed,
                            's': s_val,
                            'inversion_method': inversion_method,
                            'sample_idx': sample_idx,
                            'data_idx': data_idx,
                            'source_class': source_class,
                            'target_class': target_class,
                            'orig_pred': orig_pred,
                            'n_orig_edges': n_orig_edges,
                            'orig_smiles': orig_smiles if orig_smiles is not None else '',
                            # ---- first flip (any validity) ----
                            'success': inv_success,
                            'flip_smiles': _fgs(flip, 'smiles'),
                            't_used': _fg(flip, 't'),
                            'ged': _fg(flip, 'ged'),
                            'mod_rate': _fg(flip, 'mod_rate'),
                            'node_sparsity': _fg(flip, 'node_sparsity', -1.0),
                            'edge_sparsity': _fg(flip, 'edge_sparsity', -1.0),
                            'edgetype_sparsity': _fg(flip, 'edgetype_sparsity', -1.0),
                            'smiles_valid': bool(_fg(flip, 'smiles_valid', False)),
                            # ---- first valid flip (strict SMILES) ----
                            'valid_found': valid_found,
                            'valid_smiles': _fgs(valid, 'smiles'),
                            'valid_t': _fg(valid, 't'),
                            'valid_ged': _fg(valid, 'ged'),
                            'valid_mod_rate': _fg(valid, 'mod_rate'),
                            'valid_node_sparsity': _fg(valid, 'node_sparsity', -1.0),
                            'valid_edge_sparsity': _fg(valid, 'edge_sparsity', -1.0),
                            'valid_edgetype_sparsity': _fg(valid, 'edgetype_sparsity', -1.0),
                            'valid_connected': bool(_fg(valid, 'smiles_connected', False)),
                            'valid_n_fragments': _fg(valid, 'n_fragments', -1),
                        })

                        # ---- wandb: per-sample ----
                        if wandb.run:
                            log_dict = {
                                'run': run_idx,
                                'sample_idx': sample_idx,
                                f"sample/s{s_val}/flip_found": int(inv_success),
                                f"sample/s{s_val}/valid_found": int(valid_found),
                            }
                            if flip is not None:
                                log_dict.update({
                                    f"sample/s{s_val}/flip_t": flip['t'],
                                    f"sample/s{s_val}/flip_ged": flip['ged'] if flip['ged'] >= 0 else float('nan'),
                                    f"sample/s{s_val}/flip_edge_sparsity": flip['edge_sparsity'],
                                    f"sample/s{s_val}/flip_smiles_valid": int(flip['smiles_valid']),
                                })
                            if valid is not None:
                                log_dict.update({
                                    f"sample/s{s_val}/valid_t": valid['t'],
                                    f"sample/s{s_val}/valid_ged": valid['ged'] if valid['ged'] >= 0 else float('nan'),
                                    f"sample/s{s_val}/valid_edge_sparsity": valid['edge_sparsity'],
                                    f"sample/s{s_val}/valid_connected": int(valid['smiles_connected']),
                                    f"sample/s{s_val}/valid_n_fragments": valid['n_fragments'],
                                })
                            wandb.log(log_dict)

                # ---- MMD for this (run, s) ----
                if compute_mmd:
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

                # ---- Per-run aggregate: first-FLIP and first-VALID groups ----
                df_rs = pd.DataFrame([r for r in all_inversion_rows
                                      if r['run'] == run_idx and r['s'] == s_val])
                n_total_rs = len(df_rs)
                df_flip = df_rs[df_rs['success']]
                df_valid = df_rs[df_rs['valid_found']]
                n_flip, n_valid = len(df_flip), len(df_valid)

                def _meancol(df, col):
                    if len(df) == 0 or col not in df:
                        return float('nan')
                    vals = df[df[col] >= 0][col]
                    return vals.mean() if len(vals) > 0 else float('nan')

                flip_rate = n_flip / n_total_rs if n_total_rs > 0 else float('nan')
                valid_rate = n_valid / n_total_rs if n_total_rs > 0 else float('nan')

                summary = {
                    'run': run_idx, 'seed': run_seed, 's': s_val,
                    'inversion_method': inversion_method, 'n_total': n_total_rs,
                    # first flip (any validity)
                    'flip_rate': flip_rate, 'n_flip': n_flip,
                    'flip_avg_ged': _meancol(df_flip, 'ged'),
                    'flip_avg_mod_rate': _meancol(df_flip, 'mod_rate'),
                    'flip_avg_node_sparsity': _meancol(df_flip, 'node_sparsity'),
                    'flip_avg_edge_sparsity': _meancol(df_flip, 'edge_sparsity'),
                    'flip_avg_edgetype_sparsity': _meancol(df_flip, 'edgetype_sparsity'),
                    'flip_smiles_valid_rate': df_flip['smiles_valid'].mean() if n_flip > 0 else float('nan'),
                    'flip_avg_t': _meancol(df_flip, 't_used'),
                    # first valid flip (strict SMILES)
                    'valid_rate': valid_rate, 'n_valid': n_valid,
                    'valid_avg_ged': _meancol(df_valid, 'valid_ged'),
                    'valid_avg_mod_rate': _meancol(df_valid, 'valid_mod_rate'),
                    'valid_avg_node_sparsity': _meancol(df_valid, 'valid_node_sparsity'),
                    'valid_avg_edge_sparsity': _meancol(df_valid, 'valid_edge_sparsity'),
                    'valid_avg_edgetype_sparsity': _meancol(df_valid, 'valid_edgetype_sparsity'),
                    'valid_avg_t': _meancol(df_valid, 'valid_t'),
                    # connected valid SMILES as a fraction of ALL successful flips
                    'valid_connected_over_flip': (df_valid['valid_connected'].sum() / n_flip)
                                                 if n_flip > 0 else float('nan'),
                    'valid_avg_fragments': _meancol(df_valid, 'valid_n_fragments'),
                }
                run_summary_rows.append(summary)

                print(f"\n  [run {run_idx+1}, s={s_val}] first-FLIP : rate={flip_rate:.3f}  "
                      f"edge_sp={summary['flip_avg_edge_sparsity']:.3f}  "
                      f"valid={summary['flip_smiles_valid_rate']:.3f}  t={summary['flip_avg_t']:.1f}")
                print(f"  [run {run_idx+1}, s={s_val}] first-VALID: rate={valid_rate:.3f}  "
                      f"edge_sp={summary['valid_avg_edge_sparsity']:.3f}  t={summary['valid_avg_t']:.1f}")
                print(f"  [run {run_idx+1}, s={s_val}] connected : "
                      f"of_flip={summary['valid_connected_over_flip']:.3f}  "
                      f"avg_frags={summary['valid_avg_fragments']:.2f}")

                # ---- wandb: per-run aggregate ----
                if wandb.run:
                    _no_wandb = {'run', 'seed', 's', 'inversion_method',
                                 'flip_avg_ged', 'flip_avg_mod_rate',
                                 'valid_avg_ged', 'valid_avg_mod_rate'}
                    wlog = {'run': run_idx}
                    for k, v in summary.items():
                        if k in _no_wandb:
                            continue
                        wlog[f"run/s{s_val}/{k}"] = v
                    wandb.log(wlog)

    # ---- Save CSVs ----
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")

    cf_writer.close()

    df_inv = pd.DataFrame(all_inversion_rows)
    inv_csv = os.path.join(output_dir, f"GDCE_{_method}_s{_stag}.csv")
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

    # Two groups: first-FLIP (any validity) and first-VALID (strict SMILES).
    METRICS_FLIP = [
        ('flip_rate',                    'Flip rate'),
        ('flip_smiles_valid_rate',       'SMILES valid (strict)'),
        ('flip_avg_ged',                 'Avg GED'),
        ('flip_avg_mod_rate',            'Avg ModRate'),
        ('flip_avg_node_sparsity',       'Avg NodeSparsity'),
        ('flip_avg_edge_sparsity',       'Avg EdgeSparsity'),
        ('flip_avg_edgetype_sparsity',   'Avg EdgeTypeSparsity'),
        ('flip_avg_t',                   'Avg t used'),
    ]
    METRICS_VALID = [
        ('valid_rate',                   'Valid-flip rate'),
        ('valid_avg_ged',                'Avg GED'),
        ('valid_avg_mod_rate',           'Avg ModRate'),
        ('valid_avg_node_sparsity',      'Avg NodeSparsity'),
        ('valid_avg_edge_sparsity',      'Avg EdgeSparsity'),
        ('valid_avg_edgetype_sparsity',  'Avg EdgeTypeSparsity'),
        ('valid_avg_t',                  'Avg t used'),
        ('valid_connected_over_flip',    'Connected yield (of flips)'),
        ('valid_avg_fragments',          'Avg #fragments'),
    ]

    for s_val in s_values:
        df_s = df_run_summary[df_run_summary['s'] == s_val]
        print(f"\n--- s = {s_val}  ({len(df_s)} runs) ---")
        for group_label, METRICS in [("FIRST FLIP", METRICS_FLIP),
                                     ("FIRST VALID", METRICS_VALID)]:
            print(f"  [{group_label}]")
            for col, label in METRICS:
                vals = df_s[col].dropna() if col in df_s else pd.Series([], dtype=float)
                if len(vals) == 0:
                    print(f"    {label:<28}: N/A")
                else:
                    mean_v = vals.mean()
                    std_v  = vals.std(ddof=1) if len(vals) > 1 else 0.0
                    print(f"    {label:<28}: {mean_v:.4f} ± {std_v:.4f}")

        # ---- wandb: final mean ± std (GED / ModRate excluded — CSV/console only) ----
        if wandb.run:
            _no_wandb = {'flip_avg_ged', 'flip_avg_mod_rate',
                         'valid_avg_ged', 'valid_avg_mod_rate'}
            log_final = {}
            for col, _label in METRICS_FLIP + METRICS_VALID:
                if col in _no_wandb:
                    continue
                vals = df_s[col].dropna() if col in df_s else pd.Series([], dtype=float)
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
                if len(vals) > 0:
                    print(f"  {col:<20}: {vals.mean():.6f} ± {vals.std(ddof=1) if len(vals) > 1 else 0.0:.6f}")

    print(f"\n{'='*60}")
    print(f"\nAll results saved in: {os.path.abspath(output_dir)}")
    print("DONE!")


if __name__ == '__main__':
    main()
