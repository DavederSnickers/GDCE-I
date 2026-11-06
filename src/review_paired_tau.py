"""Paired fixed-budget comparison of the three noise treatments (review response).

The inversion ablation of the paper compares the variants at the budget each one needs
to flip, so they are evaluated at different tau. This script removes that confound:
for every input graph ONE forward reference trajectory G^0, ..., G^tau_max is drawn and
shared by all three variants, which then differ only in the noise of the reverse pass:

    noinv      fresh i.i.d. Gumbel noise at every reverse step (= original GDCE)
    naive      the Gumbel noise that generated the forward trajectory
    posterior  the truncated-Gumbel noise that reproduces the trajectory under y (GDCE-I)

Every variant is replayed from the SAME reference state G^tau for every tau on a fixed
grid, under the target condition y' (the counterfactual) and under the source condition
y (a null control: how far does each variant drift from the input when the target is
NOT changed?). All replays of one graph run as one batch, so a graph costs about
2 * tau_max model evaluations regardless of the grid size.

Optional comparison with DICE (He et al. 2024, arXiv 2410.08207), selected with
+review.variants:

    dice<M|I>_<l1>_<l2>[_f<B>]  DICE replay argmax(log p_y' + l1 * z + l2 * fresh Gumbel) with
                                the residual z = log onehot(x_ref) - log p_y(x_ref | ...)
                                (Eq. 5), on the shared Markov trajectory (M) or on DICE's
                                own trajectory of independent q-samples x_t ~ q(x_t | x_0)
                                (I, Eq. 4); l1, l2 in percent (dot-free for Hydra), log(0)
                                is taken as -B (default log 1e-30); e.g. diceI_20_80 is the
                                paper's VQ-Diffusion setting, diceM_100_0 its reconstruction
    posteriorI                  the posterior noise of GDCE-I on DICE's trajectory

Per replay the script records the shared-judge and control-judge predictions, aligned
edit counts, the corruption/restoration split against G^tau, RDKit validity, connected
components, a Morgan-fingerprint similarity to the source, dataset-specific motif edits
(benzene ground-truth rings, aromatic nitro groups) and, for successful counterfactuals,
a greedy reversion that measures how many of the edits the flip actually needs.

Usage (repo root, inside the container):
    PYTHONPATH=$PWD python src/review_paired_tau.py \
        +experiment=inversion_mutag_mol_gcn_s3 dataset=mutagenicity_mol \
        +review.out_dir=$PWD/results_review/paired_tau/mutag_gcn_s3
Smoke test: add +review.n_runs=1 +review.limit=2
"""

try:
    import graph_tool as gt  # noqa: F401  (import order matters for the container)
except ModuleNotFoundError:
    pass

import csv
import gzip
import json
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.warnings import PossibleUserWarning
from torch_geometric.data import Batch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shared_judge'))
from shared_judge import load_judge, SharedJudgeBatchAdapter  # noqa: E402

from src import utils  # noqa: E402
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics  # noqa: E402
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures  # noqa: E402
from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos  # noqa: E402
from analysis.spectre_utils import PlanarSamplingMetrics  # noqa: E402
from analysis.visualization import NonMolecularVisualization  # noqa: E402
from src.diffusion.discrete_inversion import (  # noqa: E402
    DiscreteInversion, gumbel_posterior_noise, sample_gumbel)
from src.diffusion import diffusion_utils  # noqa: E402
from inversion_mutag_mol import get_resume, load_atom_decoder, classify_graph  # noqa: E402
import review_metrics as rm  # noqa: E402

warnings.filterwarnings("ignore", category=PossibleUserWarning)

NUM_CLASSES = 2
VARIANTS = ('noinv', 'naive', 'posterior')
CONDS = ('tgt', 'src')   # replay under the target class y' / under the source class y
DEFAULT_GRID = [1, 2, 3, 5, 8, 11, 15, 20, 30, 40, 51, 60, 75, 100, 125, 150, 200]
# magnitude of log(0) in DICE's log-one-hot; the VQ-Diffusion code base it builds on
# clamps the one-hot at 1e-30 before taking the log
DICE_FLOOR = -float(np.log(1e-30))


def parse_variant(v):
    """Variant name -> (reference trajectory 'M'/'I', kind, lambda1, lambda2, log(0) floor)."""
    if v in VARIANTS:
        return 'M', v, None, None, None
    if v == 'posteriorI':
        return 'I', 'posterior', None, None, None
    parts = v.split('_')
    if parts[0] in ('diceM', 'diceI') and len(parts) in (3, 4):
        floor = DICE_FLOOR
        if len(parts) == 4:
            assert parts[3].startswith('f'), v
            floor = float(parts[3][1:])
        return parts[0][-1], 'dice', int(parts[1]) / 100, int(parts[2]) / 100, floor
    raise ValueError(f'unknown variant {v}')


# ---------------------------------------------------------------------------
# reverse posterior with a per-element time step
# ---------------------------------------------------------------------------

@torch.no_grad()
def reverse_posterior_batched(model, X_t, E_t, node_mask, guidance, s_vec):
    """Marginalised reverse posterior p(x_s | x_t, guidance) for a batch whose elements may
    sit at different steps s (t = s + 1). Same math as DiscreteInversion._reverse_posterior,
    which handles a single scalar s."""
    device = X_t.device
    bs, n = X_t.shape[0], X_t.shape[1]
    s_col = s_vec.to(device).float().view(bs, 1)
    s_norm = s_col / model.T
    t_norm = (s_col + 1) / model.T

    beta_t = model.noise_schedule(t_normalized=t_norm)
    alpha_s_bar = model.noise_schedule.get_alpha_bar(t_normalized=s_norm)
    alpha_t_bar = model.noise_schedule.get_alpha_bar(t_normalized=t_norm)
    Qtb = model.transition_model.get_Qt_bar(alpha_t_bar, device)
    Qsb = model.transition_model.get_Qt_bar(alpha_s_bar, device)
    Qt = model.transition_model.get_Qt(beta_t, device)

    noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': torch.zeros(bs, 0, device=device),
                  't': t_norm, 'node_mask': node_mask}
    extra_data = model.compute_extra_data(noisy_data)
    pred = model.forward(noisy_data, extra_data, node_mask, guidance=guidance)
    pred_X = F.softmax(pred.X, dim=-1)
    pred_E = F.softmax(pred.E, dim=-1)

    p_X = diffusion_utils.compute_batched_over0_posterior_distribution(
        X_t=X_t, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X)
    p_E = diffusion_utils.compute_batched_over0_posterior_distribution(
        X_t=E_t, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E)

    prob_X = (pred_X.unsqueeze(-1) * p_X).sum(dim=2)
    prob_X[torch.sum(prob_X, dim=-1) == 0] = 1e-5
    prob_X = prob_X / torch.sum(prob_X, dim=-1, keepdim=True)

    pred_E_flat = pred_E.reshape((bs, -1, pred_E.shape[-1]))
    prob_E = (pred_E_flat.unsqueeze(-1) * p_E).sum(dim=-2)
    prob_E[torch.sum(prob_E, dim=-1) == 0] = 1e-5
    prob_E = prob_E / torch.sum(prob_E, dim=-1, keepdim=True)
    return prob_X, prob_E.reshape(bs, n, n, -1)


def _symmetrize_upper(g_E):
    n = g_E.shape[1]
    iu = torch.triu_indices(n, n, offset=1, device=g_E.device)
    mask = torch.zeros((1, n, n, 1), device=g_E.device, dtype=g_E.dtype)
    mask[:, iu[0], iu[1], :] = 1.0
    g_upper = g_E * mask
    return g_upper + g_upper.transpose(1, 2)


@torch.no_grad()
def independent_trajectory(model, X_0, E_0, node_mask, num_steps):
    """DICE's reference states (Eq. 4 of He et al.): every x_t ~ q(x_t | x_0) is drawn
    independently, instead of the Markov chain x_t ~ q(x_t | x_{t-1}) of GDCE-I."""
    device = X_0.device
    bs, n, dx = X_0.shape
    de = E_0.shape[-1]
    diag_mask = torch.eye(n, device=device).unsqueeze(0).expand(bs, -1, -1).bool()
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    traj_X, traj_E = [X_0.clone()], [E_0.clone()]
    for t_int in range(1, num_steps + 1):
        t_float = torch.tensor([[t_int / model.T]], device=device).expand(bs, -1)
        alpha_t_bar = model.noise_schedule.get_alpha_bar(t_normalized=t_float)
        Qtb = model.transition_model.get_Qt_bar(alpha_t_bar, device)
        prob_X = X_0 @ Qtb.X
        prob_E = E_0 @ Qtb.E.unsqueeze(1)
        prob_X[~node_mask] = 1.0 / dx
        prob_E[inverse_edge_mask] = 1.0 / de
        prob_E[diag_mask] = 1.0 / de
        xi = torch.argmax(torch.log(prob_X + 1e-10) + sample_gumbel(prob_X.shape, device), dim=-1)
        ei = torch.argmax(torch.log(prob_E + 1e-10) + sample_gumbel(prob_E.shape, device), dim=-1)
        ei = torch.triu(ei, diagonal=1)
        ei = ei + ei.transpose(1, 2)
        traj_X.append(F.one_hot(xi, num_classes=dx).float())
        traj_E.append(F.one_hot(ei, num_classes=de).float())
    return traj_X, traj_E


@torch.no_grad()
def record_posterior_noise(model, traj_X, traj_E, node_mask, guidance, tau_max, margin, chunk,
                           keep_logp=False, draw_noise=True):
    """Posterior Gumbel noise that forces the guided reverse step s+1 -> s to land on the
    reference state G^s (truncated-Gumbel construction, as inversion_forward_posterior).
    The steps only depend on the fixed reference states, so they are recorded in batches.
    With keep_logp the log reverse posteriors of the reference steps are returned as well:
    they are the log pi_theta(x_t, c, t) of DICE's residual (Eq. 5 of He et al.)."""
    g_X, g_E = [None] * tau_max, [None] * tau_max
    l_X, l_E = [None] * tau_max, [None] * tau_max
    for start in range(0, tau_max, chunk):
        steps = list(range(start, min(tau_max, start + chunk)))
        bs = len(steps)
        X_t = torch.cat([traj_X[s + 1] for s in steps])
        E_t = torch.cat([traj_E[s + 1] for s in steps])
        prob_X, prob_E = reverse_posterior_batched(
            model, X_t, E_t, node_mask.expand(bs, -1), guidance.expand(bs, -1),
            torch.tensor(steps))
        tgt_X = torch.cat([traj_X[s] for s in steps]).argmax(dim=-1)
        tgt_E = torch.cat([traj_E[s] for s in steps]).argmax(dim=-1)
        lpx = torch.log(prob_X + 1e-10)
        lpe = torch.log(prob_E + 1e-10)
        if draw_noise:
            gx = gumbel_posterior_noise(lpx, tgt_X, margin)
            ge = _symmetrize_upper(gumbel_posterior_noise(lpe, tgt_E, margin))
        for k, s in enumerate(steps):
            if draw_noise:
                g_X[s] = gx[k:k + 1]
                g_E[s] = ge[k:k + 1]
            if keep_logp:
                l_X[s] = lpx[k:k + 1]
                l_E[s] = lpe[k:k + 1]
    if keep_logp:
        return g_X, g_E, l_X, l_E
    return g_X, g_E


@torch.no_grad()
def replay_all(model, chains, trajs, node_mask, gui, noise, logp, specs, dx, de, chunk):
    """Run every chain (variant, cond, tau) from its reference state G^tau down to G^0.

    All chains of one graph advance in lockstep through the global step s, so every
    model call is a batch of the chains that are active at s (tau > s). Each step takes
    argmax(log p + a), where a is fresh Gumbel noise (noinv), recorded Gumbel noise (naive,
    posterior) or DICE's lambda1 * residual + lambda2 * fresh Gumbel noise."""
    tau_max = max(c[2] for c in chains)
    n = node_mask.shape[1]
    st_X = [None] * len(chains)
    st_E = [None] * len(chains)
    for s in reversed(range(tau_max)):
        active = [k for k, c in enumerate(chains) if c[2] > s]
        for k in active:
            if chains[k][2] == s + 1:
                traj_X, traj_E = trajs[specs[chains[k][0]][0]]
                st_X[k] = traj_X[s + 1]
                st_E[k] = traj_E[s + 1]
        for p0 in range(0, len(active), chunk):
            part = active[p0:p0 + chunk]
            bs = len(part)
            X_t = torch.cat([st_X[k] for k in part])
            E_t = torch.cat([st_E[k] for k in part])
            g = torch.cat([gui[chains[k][1]] for k in part])
            prob_X, prob_E = reverse_posterior_batched(
                model, X_t, E_t, node_mask.expand(bs, -1), g,
                torch.full((bs,), s, dtype=torch.long))
            gx, ge = [], []
            for k in part:
                v = chains[k][0]
                traj, kind, lam1, lam2, floor = specs[v]
                if kind == 'noinv':
                    gx.append(sample_gumbel((1, n, dx), X_t.device))
                    ge.append(sample_gumbel((1, n, n, de), X_t.device))
                elif kind == 'dice':
                    # z_t = log onehot(x_ref) - log p_y(x_ref | x_ref_t), log(0) := -floor
                    ref_X, ref_E = trajs[traj][0][s], trajs[traj][1][s]
                    ax = lam1 * ((ref_X - 1.0) * floor - logp[traj][0][s])
                    ae = lam1 * ((ref_E - 1.0) * floor - logp[traj][1][s])
                    if lam2 > 0:
                        ax = ax + lam2 * sample_gumbel((1, n, dx), X_t.device)
                        ae = ae + lam2 * sample_gumbel((1, n, n, de), X_t.device)
                    gx.append(ax)
                    ge.append(ae)
                else:
                    gx.append(noise[v][0][s].to(X_t.device))
                    ge.append(noise[v][1][s].to(X_t.device))
            gx = torch.cat(gx)
            ge = torch.cat(ge)
            xi = torch.argmax(torch.log(prob_X + 1e-10) + gx, dim=-1)
            ei = torch.argmax(torch.log(prob_E + 1e-10) + ge, dim=-1)
            ei = torch.triu(ei, diagonal=1)
            ei = ei + ei.transpose(1, 2)
            X_new = F.one_hot(xi, num_classes=dx).float()
            E_new = F.one_hot(ei, num_classes=de).float()
            for j, k in enumerate(part):
                st_X[k] = X_new[j:j + 1]
                st_E[k] = E_new[j:j + 1]
    return st_X, st_E


# ---------------------------------------------------------------------------
# judges on dense graph batches
# ---------------------------------------------------------------------------

@torch.no_grad()
def judge_probs(judge, NT, ET, dx):
    """Class probabilities of a SharedJudge for B graphs of equal size n.
    NT: (B, n) long, ET: (B, n, n) long, symmetric, zero diagonal."""
    B, n = NT.shape
    x = F.one_hot(NT.reshape(-1), dx).float()
    b, i, j = torch.nonzero(ET > 0, as_tuple=True)
    ei = torch.stack([b * n + i, b * n + j])
    ea = F.one_hot(ET[b, i, j], 5).float() if judge.edge_dim > 0 else None
    batch = torch.arange(B, device=NT.device).repeat_interleave(n)
    return F.softmax(judge(x, ei, batch, edge_attr=ea), dim=-1)


@torch.no_grad()
def greedy_necessary(judge, nt_src, et_src, nt_cf, et_cf, target, dx, max_edits):
    """Greedily revert edits of a successful counterfactual while the judge keeps predicting
    the target class; at each round the reversion with the highest target probability is
    taken. Returns (#edits, #edits left = a 1-minimal flip-preserving subset, mask of the
    remaining edits as (node ids, pair list)), or None when there are too many edits."""
    device = nt_src.device
    node_ids = torch.nonzero(nt_src != nt_cf, as_tuple=True)[0]
    pi, pj = torch.nonzero(torch.triu(et_src != et_cf, 1), as_tuple=True)
    edits = [('n', int(a), -1) for a in node_ids.tolist()] + \
            [('e', int(a), int(b)) for a, b in zip(pi.tolist(), pj.tolist())]
    k = len(edits)
    if k == 0 or k > max_edits:
        return None
    cur_nt, cur_et = nt_cf.clone(), et_cf.clone()
    remaining = list(range(k))
    while remaining:
        B = len(remaining)
        NT = cur_nt.unsqueeze(0).repeat(B, 1)
        ET = cur_et.unsqueeze(0).repeat(B, 1, 1)
        rows_n = [r for r, e in enumerate(remaining) if edits[e][0] == 'n']
        rows_e = [r for r, e in enumerate(remaining) if edits[e][0] == 'e']
        if rows_n:
            a = torch.tensor([edits[remaining[r]][1] for r in rows_n], device=device)
            rr = torch.tensor(rows_n, device=device)
            NT[rr, a] = nt_src[a]
        if rows_e:
            a = torch.tensor([edits[remaining[r]][1] for r in rows_e], device=device)
            b = torch.tensor([edits[remaining[r]][2] for r in rows_e], device=device)
            rr = torch.tensor(rows_e, device=device)
            ET[rr, a, b] = et_src[a, b]
            ET[rr, b, a] = et_src[b, a]
        probs = judge_probs(judge, NT, ET, dx)
        keeps = probs.argmax(dim=-1) == target
        if not bool(keeps.any()):
            break
        score = torch.where(keeps, probs[:, target], torch.full_like(probs[:, target], -1.0))
        best = int(score.argmax())
        e = edits[remaining[best]]
        if e[0] == 'n':
            cur_nt[e[1]] = nt_src[e[1]]
        else:
            cur_et[e[1], e[2]] = et_src[e[1], e[2]]
            cur_et[e[2], e[1]] = et_src[e[2], e[1]]
        remaining.pop(best)
    left_nodes = [edits[r][1] for r in remaining if edits[r][0] == 'n']
    left_pairs = [(edits[r][1], edits[r][2]) for r in remaining if edits[r][0] == 'e']
    return k, len(remaining), left_nodes, left_pairs


def _np(t):
    return t.detach().cpu().numpy().astype(np.int64)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    from rdkit import RDLogger, DataStructs
    RDLogger.DisableLog('rdApp.*')

    # Everything under cfg.review and the judge path must be read BEFORE get_resume(),
    # which rebuilds cfg from the checkpoint and drops unknown keys.
    rv = OmegaConf.to_container(cfg.review, resolve=True) if 'review' in cfg else {}
    if rv.get('grid') == 'paper':
        # the budget-search grid of inversion_mutag_mol.py (Table 3 operating point)
        tau_grid = list(range(1, 101)) + list(range(110, 201, 10))
    else:
        tau_grid = sorted({int(t) for t in rv.get('tau_grid', DEFAULT_GRID)})
    tau_max = max(tau_grid)
    n_runs = int(rv.get('n_runs', 5))
    run_ids = [int(r) for r in rv.get('run_ids', range(n_runs))]
    n_per_run = int(rv.get('n_samples_per_run', 100))
    limit = rv.get('limit', None)
    margin = float(rv.get('margin', 1e-3))
    chunk = int(rv.get('chunk', 128))
    greedy_max = int(rv.get('greedy_max_edits', 150))
    # greedy reversion is the expensive part on a fine grid: restrict it to these taus
    greedy_taus = set(int(t) for t in rv.get('greedy_taus', tau_grid))
    variants = tuple(rv.get('variants', VARIANTS))
    specs = {v: parse_variant(v) for v in variants}
    dice_trajs = {sp[0] for sp in specs.values() if sp[1] == 'dice'}
    need_I = any(sp[0] == 'I' for sp in specs.values())
    # replays that reproduce the input under the source class by construction
    exact_variants = [v for v, sp in specs.items()
                      if sp[1] == 'posterior' or (sp[1] == 'dice' and sp[2] == 1.0 and sp[3] == 0.0)]
    judge_path = cfg.guidance.shared_judge_path
    ctrl_path = rv.get('ctrl_judge_path', judge_path + '_seed1')
    s_val = float(cfg.guidance.s)
    ds_name = cfg.dataset.name
    out_dir = rv.get('out_dir', os.path.join(hydra.utils.get_original_cwd(),
                                             'results_review', 'paired_tau', ds_name))
    os.makedirs(out_dir, exist_ok=True)

    print('=' * 70)
    print(f'PAIRED FIXED-TAU NOISE COMPARISON  dataset={ds_name}  s={s_val}')
    print(f'  tau grid: {tau_grid}')
    print(f'  variants: {list(variants)}')
    print(f'  runs={n_runs} x {n_per_run} (limit={limit})  margin={margin}  chunk={chunk}')
    print(f'  judge={judge_path}\n  ctrl ={ctrl_path}\n  out  ={out_dir}')
    print('=' * 70)

    torch.set_num_threads(cfg.train.num_interops_workers)
    seed_everything(cfg.train.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datamodule = SpectreGraphDataModule(cfg)
    sampling_metrics = PlanarSamplingMetrics(datamodule)
    dataset_infos = SpectreDatasetInfos(cfg['dataset'])
    train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
    if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
    domain_features = DummyExtraFeatures()
    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                            domain_features=domain_features)
    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics,
                    'visualization_tools': NonMolecularVisualization(),
                    'extra_features': extra_features, 'domain_features': domain_features}

    cfg, model = get_resume(cfg, model_kwargs)
    with open_dict(cfg):
        cfg.guidance.s = s_val      # model.cfg is cfg: this is the CFG scale of forward()
    model.to(device)
    model.eval()
    print(f'  T={model.T}  Xdim={model.Xdim_output}  Edim={model.Edim_output}  s={model.cfg.guidance.s}')
    assert tau_max <= model.T

    atom_decoder = load_atom_decoder(cfg)
    sym2idx = {v: int(k) for k, v in atom_decoder.items()}
    h_idx = sym2idx.get('H', None)
    dx, de = model.Xdim_output, model.Edim_output

    judge = load_judge(judge_path, device=str(device))
    ctrl = load_judge(ctrl_path, device=str(device))
    assert judge.num_features == dx and ctrl.num_features == dx
    adapter = SharedJudgeBatchAdapter(judge).to(device).eval()
    inverter = DiscreteInversion(model, device=device)

    test_dataset = datamodule.dataloaders['test'].dataset

    # ---- the same stratified 5 x 100 selection as inversion_mutag_mol (seeds 42..) ----
    all_meta = []
    for data_idx in range(len(test_dataset)):
        data = test_dataset[data_idx]
        ei = data.edge_index.to(device)
        ea = data.edge_attr.to(device)
        pred, probs = classify_graph(adapter, ei, ea, data.x, device)
        all_meta.append({'data_idx': data_idx, 'orig_pred': pred, 'orig_probs': probs})
    class_totals = {c: sum(1 for m in all_meta if m['orig_pred'] == c) for c in range(NUM_CLASSES)}
    print(f'  test graphs: {len(all_meta)}  judge-class totals: {class_totals}')

    # fingerprints of the whole test split, for the nearest-other-molecule retrieval check
    test_graphs = []
    for data_idx in range(len(test_dataset)):
        d = test_dataset[data_idx]
        nt = _np(d.x.argmax(-1))
        et = np.zeros((len(nt), len(nt)), dtype=np.int64)
        if d.edge_index.numel() > 0:
            src, dst = _np(d.edge_index)
            et[src, dst] = _np(d.edge_attr.argmax(-1))
        np.fill_diagonal(et, 0)
        test_graphs.append((nt, et))
    test_fps = [rm.graph_fingerprint(nt, et, atom_decoder) for nt, et in test_graphs]
    fp_ok = [i for i, f in enumerate(test_fps) if f is not None]
    fp_list = [test_fps[i] for i in fp_ok]
    print(f'  test fingerprints: {len(fp_list)}/{len(test_fps)}')

    chains = [(v, c, tau) for tau in tau_grid for v in variants for c in CONDS]
    csv_path = os.path.join(out_dir, 'chains.csv')
    dump_path = os.path.join(out_dir, 'graphs.jsonl.gz')
    meta_path = os.path.join(out_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump({'dataset': ds_name, 's': s_val, 'tau_grid': tau_grid, 'n_runs': n_runs, 'run_ids': run_ids,
                   'variants': list(variants), 'dice_floor': DICE_FLOOR,
                   'n_samples_per_run': n_per_run, 'margin': margin, 'judge': judge_path,
                   'ctrl': ctrl_path, 'ckpt': cfg.general.test_only, 'T': int(model.T),
                   'atom_decoder': atom_decoder}, f, indent=1)
    csv_file = open(csv_path, 'w', newline='')
    writer = None
    dump_file = gzip.open(dump_path, 'wt')

    t_start = time.time()
    n_done = 0
    for run_idx in run_ids:
        run_seed = 42 + run_idx
        rng = np.random.default_rng(run_seed)
        by_class = defaultdict(list)
        for m in all_meta:
            by_class[m['orig_pred']].append(m)
        selected = []
        for c in range(NUM_CLASSES):
            n_class = round(n_per_run * class_totals[c] / len(all_meta))
            n_class = min(n_class, len(by_class[c]))
            chosen = rng.choice(len(by_class[c]), size=n_class, replace=False)
            selected.extend([by_class[c][i] for i in chosen])
        rng.shuffle(selected)
        selected = selected[:n_per_run]
        if limit is not None:
            selected = selected[:int(limit)]
        print(f'\n### run {run_idx} (seed {run_seed}): {len(selected)} graphs, first data_idx '
              f'{[m["data_idx"] for m in selected[:5]]}')

        for sample_idx, meta in enumerate(selected):
            t0 = time.time()
            data_idx = meta['data_idx']
            data = test_dataset[data_idx]
            src_class = int(meta['orig_pred'])
            tgt_class = 1 - src_class
            # one seed per (run, graph): the reference trajectory, the posterior record and
            # the fresh noise of noinv are all reproducible and shared across variants
            torch.manual_seed(1_000_003 * run_seed + data_idx)

            batch = Batch.from_data_list([data.to(device)])
            dense, node_mask = utils.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            X0, E0 = dense.X.to(device), dense.E.to(device)
            n = X0.shape[1]
            y_src = F.one_hot(torch.tensor([src_class], device=device), NUM_CLASSES).float()
            y_tgt = F.one_hot(torch.tensor([tgt_class], device=device), NUM_CLASSES).float()

            inv = inverter.inversion_forward(X0, E0, node_mask, num_steps=tau_max, prog_bar=False)
            traj_X, traj_E = inv.trajectory_X, inv.trajectory_E
            rec = record_posterior_noise(model, traj_X, traj_E, node_mask, y_src, tau_max, margin,
                                         chunk, keep_logp='M' in dice_trajs)
            trajs = {'M': (traj_X, traj_E)}
            noise = {'naive': (inv.gumbels_X, inv.gumbels_E), 'posterior': (rec[0], rec[1])}
            logp = {'M': (rec[2], rec[3])} if 'M' in dice_trajs else {}
            if need_I:
                # drawn after the Markov record, so the posterior noise above stays the same
                # as in runs without the DICE variants
                trajs['I'] = independent_trajectory(model, X0, E0, node_mask, tau_max)
                rec_I = record_posterior_noise(model, trajs['I'][0], trajs['I'][1], node_mask, y_src,
                                               tau_max, margin, chunk, keep_logp=True,
                                               draw_noise='posteriorI' in specs)
                noise['posteriorI'] = (rec_I[0], rec_I[1])
                logp['I'] = (rec_I[2], rec_I[3])
            outs_X, outs_E = replay_all(model, chains, trajs, node_mask, {'tgt': y_tgt, 'src': y_src},
                                        noise, logp, specs, dx, de, chunk)

            # ---- decode ----
            nt_src_t = X0[0].argmax(-1)
            et_src_t = E0[0].argmax(-1)
            et_src_t.fill_diagonal_(0)
            nt_src, et_src = _np(nt_src_t), _np(et_src_t)
            NT = torch.cat([x.argmax(-1) for x in outs_X])            # (C, n)
            ET = torch.cat([e.argmax(-1) for e in outs_E])            # (C, n, n)
            p_base = judge_probs(judge, NT, ET, dx).cpu().numpy()
            p_ctrl = judge_probs(ctrl, NT, ET, dx).cpu().numpy()
            p_src = judge_probs(judge, nt_src_t[None], et_src_t[None], dx).cpu().numpy()[0]
            if int(p_src.argmax()) != src_class:
                print(f'      WARNING dense source of idx {data_idx} judged {int(p_src.argmax())}, '
                      f'sparse path said {src_class} (p_src={p_src})')

            heavy = (nt_src != h_idx) if h_idx is not None else np.ones(n, dtype=bool)
            n_src_comp, n_src_heavy_comp = rm.components(et_src, heavy)
            fp_src = rm.graph_fingerprint(nt_src, et_src, atom_decoder)
            ring_atoms, nitro_atoms = set(), set()
            gt_src = None
            if ds_name == 'benzene':
                ring_atoms = {a for r in rm.benzene_rings(nt_src, et_src, sym2idx['C']) for a in r}
                gt_src = int(len(ring_atoms) > 0)
            elif {'N', 'O', 'C'} <= set(sym2idx):
                for g in rm.nitro_groups(nt_src, et_src, sym2idx['N'], sym2idx['O'], sym2idx['C']):
                    nitro_atoms |= g
            ref = {}
            for tr, (tX, tE) in trajs.items():
                for tau in tau_grid:
                    ref[(tr, tau)] = (_np(tX[tau][0].argmax(-1)), _np(tE[tau][0].argmax(-1)))
                    np.fill_diagonal(ref[(tr, tau)][1], 0)

            nt_out, et_out = _np(NT), _np(ET)
            idx_of = {c: k for k, c in enumerate(chains)}
            rows = []
            for k, (v, cond, tau) in enumerate(chains):
                nt_cf, et_cf = nt_out[k], et_out[k]
                base_pred = int(p_base[k].argmax())
                ctrl_pred = int(p_ctrl[k].argmax())
                ec = rm.edit_counts(nt_src, et_src, nt_cf, et_cf, heavy)
                ref_nt, ref_et = ref[(specs[v][0], tau)]
                cs = rm.corruption_split(nt_src, et_src, ref_nt, ref_et, nt_cf, et_cf)
                n_comp, n_heavy_comp = rm.components(et_cf, heavy)
                smi = rm.strict_smiles(nt_cf, et_cf, atom_decoder)
                fp = rm.graph_fingerprint(nt_cf, et_cf, atom_decoder)
                tani_src, tani_other = float('nan'), float('nan')
                if fp is not None and fp_src is not None:
                    tani_src = DataStructs.TanimotoSimilarity(fp, fp_src)
                    sims = DataStructs.BulkTanimotoSimilarity(fp, fp_list)
                    tani_other = max(sim for i, sim in zip(fp_ok, sims) if i != data_idx)
                row = {
                    'run': run_idx, 'seed': run_seed, 'sample_idx': sample_idx,
                    'data_idx': data_idx, 'n_nodes': n, 'n_heavy': int(heavy.sum()),
                    'n_src_edges': int(np.triu(et_src > 0, 1).sum()),
                    'src_class': src_class, 'tgt_class': tgt_class,
                    'src_p_src': float(p_src[src_class]),
                    'variant': v, 'cond': cond, 'tau': tau,
                    'base_pred': base_pred, 'base_p_tgt': float(p_base[k][tgt_class]),
                    'ctrl_pred': ctrl_pred, 'ctrl_p_tgt': float(p_ctrl[k][tgt_class]),
                    'flip': int(base_pred == tgt_class),
                    'ctrl_flip': int(ctrl_pred == tgt_class),
                    **ec, **cs,
                    'exact': int(ec['n_total'] == 0),
                    'smiles_valid': int(smi is not None),
                    'n_comp': n_comp, 'n_heavy_comp': n_heavy_comp,
                    'n_comp_src': n_src_comp, 'n_heavy_comp_src': n_src_heavy_comp,
                    'tani_src': tani_src, 'tani_other_max': tani_other,
                }
                if ds_name == 'benzene':
                    row['gt_src'] = gt_src
                    row['gt_cf'] = int(len(rm.benzene_rings(nt_cf, et_cf, sym2idx['C'])) > 0)
                    row['n_ring_atoms'] = len(ring_atoms)
                    row['ring_touch'] = rm.edits_touching(nt_src, et_src, nt_cf, et_cf, ring_atoms)
                else:
                    row['nitro_src'] = int(len(nitro_atoms) > 0)
                    row['nitro_cf'] = int(len(rm.nitro_groups(
                        nt_cf, et_cf, sym2idx['N'], sym2idx['O'], sym2idx['C'])) > 0)
                    row['nitro_touch'] = rm.edits_touching(nt_src, et_src, nt_cf, et_cf, nitro_atoms)
                if cond == 'tgt':
                    kn = idx_of[(v, 'src', tau)]
                    row['shared_with_src'] = rm.shared_edits(
                        nt_src, et_src, nt_cf, et_cf, nt_out[kn], et_out[kn])
                    # greedy reversion on successful counterfactuals
                    row['greedy_k'] = -1
                    row['greedy_nec'] = -1
                    row['greedy_nec_heavy'] = -1
                    row['greedy_nec_motif'] = -1
                    if row['flip'] and tau in greedy_taus:
                        gr = greedy_necessary(judge, nt_src_t, et_src_t, NT[k], ET[k],
                                              tgt_class, dx, greedy_max)
                        if gr is not None:
                            gk, gnec, gnodes, gpairs = gr
                            row['greedy_k'] = gk
                            row['greedy_nec'] = gnec
                            row['greedy_nec_heavy'] = (sum(1 for a in gnodes if heavy[a]) +
                                                       sum(1 for a, b in gpairs if heavy[a] and heavy[b]))
                            motif = ring_atoms if ds_name == 'benzene' else nitro_atoms
                            row['greedy_nec_motif'] = (sum(1 for a in gnodes if a in motif) +
                                                       sum(1 for a, b in gpairs if a in motif or b in motif))
                rows.append(row)
                ii, jj = np.nonzero(np.triu(et_cf > 0, 1))
                dump_file.write(json.dumps({
                    'run': run_idx, 'sample_idx': sample_idx, 'data_idx': data_idx,
                    'variant': v, 'cond': cond, 'tau': tau, 'src_class': src_class,
                    'tgt_class': tgt_class, 'node_types': nt_cf.tolist(),
                    'edges': [[int(a), int(b), int(et_cf[a, b])] for a, b in zip(ii, jj)],
                }) + '\n')

            if writer is None:
                # union of keys: benzene and mutag rows carry different motif columns
                fieldnames = list(rows[0].keys())
                for r in rows:
                    for key in r:
                        if key not in fieldnames:
                            fieldnames.append(key)
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames, restval='')
                writer.writeheader()
            writer.writerows(rows)
            csv_file.flush()
            dump_file.flush()

            # ---- console summary per graph ----
            n_done += 1
            post_null = [r for r in rows if r['variant'] in exact_variants and r['cond'] == 'src']
            exact_all = all(r['exact'] for r in post_null)
            msg = []
            for v in variants:
                fl = [r['tau'] for r in rows if r['variant'] == v and r['cond'] == 'tgt' and r['flip']]
                msg.append(f'{v}:first={min(fl) if fl else "-"}')
            print(f'  [{run_idx}:{sample_idx:3d}] idx={data_idx:4d} n={n:2d} '
                  f'{src_class}->{tgt_class}  {"  ".join(msg)}  post-null exact={exact_all}  '
                  f'{time.time() - t0:.1f}s  (total {time.time() - t_start:.0f}s, {n_done} graphs)')
            if not exact_all:
                bad = [(r['variant'], r['tau'], r['n_total']) for r in post_null if not r['exact']]
                print(f'      WARNING null replay not exact at (variant, tau, edits) = {bad}')

    csv_file.close()
    dump_file.close()
    print(f'\nDONE  {n_done} graphs in {time.time() - t_start:.0f}s -> {out_dir}')


if __name__ == '__main__':
    main()
