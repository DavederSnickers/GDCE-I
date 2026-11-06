"""Guidance operator check (review response R2.4 / R3.2b).

Eq. (12) of the submission writes classifier-free guidance on clean-state probabilities,
p_s = p_u + s (p_c - p_u), which leaves the simplex for s > 1. The implementation
(DiscreteDenoisingDiffusion.forward) applies the same extrapolation to the denoiser's
logits, l_s = l_u + s (l_c - l_u), and normalises with a softmax, i.e. p_s is proportional
to p_u^(1-s) p_c^s. This script replays GDCE-I counterfactuals (posterior noise recorded
under the source class, replay under the target class from tau) and compares the two
operators at every reverse step and every node / edge position:

    neg      share of positions where p_u + s (p_c - p_u) has a negative entry
    negmass  negative mass sum_k max(0, -p_k) at a position
    tv       total variation between the implemented p_s and the clipped and renormalised
             probability-space operator
    agree    share of positions whose argmax is the same under both operators

Usage (repo root, inside the container):
    PYTHONPATH=$PWD python src/review_guidance_check.py \
        +experiment=inversion_mutag_mol_gcn_s3 dataset=mutagenicity_mol \
        '+review.scales=[3,5]' +review.tau=100 +review.limit=60 \
        +review.out=$PWD/results_review/guidance_check/mutag.csv
"""

try:
    import graph_tool as gt  # noqa: F401  (import order matters for the container)
except ModuleNotFoundError:
    pass

import csv
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import seed_everything
from torch_geometric.data import Batch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shared_judge'))
from shared_judge import load_judge, SharedJudgeBatchAdapter  # noqa: E402

from src import utils  # noqa: E402
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics  # noqa: E402
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures  # noqa: E402
from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos  # noqa: E402
from analysis.spectre_utils import PlanarSamplingMetrics  # noqa: E402
from analysis.visualization import NonMolecularVisualization  # noqa: E402
from src.diffusion.discrete_inversion import DiscreteInversion  # noqa: E402
from inversion_mutag_mol import get_resume, classify_graph  # noqa: E402
from review_paired_tau import record_posterior_noise, reverse_posterior_batched  # noqa: E402

NUM_CLASSES = 2
BUCKETS = [(0, 10), (10, 50), (50, 100), (100, 10 ** 6)]


def operator_stats(lc, lu, s):
    """Per-position comparison of probability-space and logit-space guidance (..., K)."""
    lc, lu = lc.double(), lu.double()
    pc, pu = F.softmax(lc, dim=-1), F.softmax(lu, dim=-1)
    plin = pu + s * (pc - pu)
    plog = F.softmax(lu + s * (lc - lu), dim=-1)
    pclip = plin.clamp(min=0)
    pclip = pclip / pclip.sum(-1, keepdim=True)
    return {'neg': (plin.min(-1).values < -1e-9).double(),
            'negmass': (-plin).clamp(min=0).sum(-1),
            'tv': 0.5 * (pclip - plog).abs().sum(-1),
            'agree': (pclip.argmax(-1) == plog.argmax(-1)).double()}


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    rv = OmegaConf.to_container(cfg.review, resolve=True) if 'review' in cfg else {}
    scales = [float(s) for s in rv.get('scales', [3, 5])]
    tau = int(rv.get('tau', 100))
    limit = int(rv.get('limit', 60))
    margin = float(rv.get('margin', 1e-3))
    chunk = int(rv.get('chunk', 128))
    out_path = rv['out']
    judge_path = cfg.guidance.shared_judge_path
    ds_name = cfg.dataset.name
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f'GUIDANCE OPERATOR CHECK  dataset={ds_name}  scales={scales}  tau={tau}  graphs={limit}')

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
    model.to(device)
    model.eval()
    assert tau <= model.T

    judge = load_judge(judge_path, device=str(device))
    adapter = SharedJudgeBatchAdapter(judge).to(device).eval()
    inverter = DiscreteInversion(model, device=device)
    test_dataset = datamodule.dataloaders['test'].dataset

    # the stratified selection of run 0 (seed 42) of inversion_mutag_mol / review_paired_tau
    all_meta = []
    for data_idx in range(len(test_dataset)):
        data = test_dataset[data_idx]
        pred, _ = classify_graph(adapter, data.edge_index.to(device), data.edge_attr.to(device),
                                 data.x, device)
        all_meta.append({'data_idx': data_idx, 'orig_pred': pred})
    class_totals = {c: sum(1 for m in all_meta if m['orig_pred'] == c) for c in range(NUM_CLASSES)}
    rng = np.random.default_rng(42)
    by_class = defaultdict(list)
    for m in all_meta:
        by_class[m['orig_pred']].append(m)
    selected = []
    for c in range(NUM_CLASSES):
        n_class = min(round(100 * class_totals[c] / len(all_meta)), len(by_class[c]))
        chosen = rng.choice(len(by_class[c]), size=n_class, replace=False)
        selected.extend([by_class[c][i] for i in chosen])
    rng.shuffle(selected)
    selected = selected[:limit]

    # the denoiser is called twice per forward (conditional, then null token); keep both
    captured = []
    transformer_forward = model.model.forward

    def capture(*args, **kwargs):
        out = transformer_forward(*args, **kwargs)
        captured.append((out.X.detach().clone(), out.E.detach().clone()))
        return out

    model.model.forward = capture

    acc = defaultdict(lambda: defaultdict(float))
    t_start = time.time()
    for s_val in scales:
        with open_dict(cfg):
            cfg.guidance.s = s_val      # model.cfg is cfg: this is the CFG scale of forward()
        assert float(model.cfg.guidance.s) == s_val
        for g_i, meta in enumerate(selected):
            data_idx = meta['data_idx']
            src_class = int(meta['orig_pred'])
            torch.manual_seed(1_000_003 * 42 + data_idx)
            batch = Batch.from_data_list([test_dataset[data_idx].to(device)])
            dense, node_mask = utils.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            X0, E0 = dense.X.to(device), dense.E.to(device)
            n = X0.shape[1]
            iu = torch.triu_indices(n, n, offset=1, device=device)
            y_src = F.one_hot(torch.tensor([src_class], device=device), NUM_CLASSES).float()
            y_tgt = F.one_hot(torch.tensor([1 - src_class], device=device), NUM_CLASSES).float()

            inv = inverter.inversion_forward(X0, E0, node_mask, num_steps=tau, prog_bar=False)
            g_X, g_E = record_posterior_noise(model, inv.trajectory_X, inv.trajectory_E, node_mask,
                                              y_src, tau, margin, chunk)
            X_t, E_t = inv.trajectory_X[tau].clone(), inv.trajectory_E[tau].clone()
            for s_int in reversed(range(tau)):
                captured.clear()
                prob_X, prob_E = reverse_posterior_batched(model, X_t, E_t, node_mask, y_tgt,
                                                           torch.tensor([s_int]))
                assert len(captured) == 2, len(captured)
                (cX, cE), (uX, uE) = captured
                bucket = next(b for b in BUCKETS if b[0] <= s_int < b[1])
                for kind, st in (('node', operator_stats(cX[0], uX[0], s_val)),
                                 ('edge', operator_stats(cE[0][iu[0], iu[1]], uE[0][iu[0], iu[1]], s_val))):
                    a = acc[(s_val, f'{bucket[0]}-{bucket[1]}', kind)]
                    m = len(st['neg'])
                    a['positions'] += m
                    a['neg'] += float(st['neg'].sum())
                    a['negmass'] += float(st['negmass'].sum())
                    a['negmass_max'] = max(a['negmass_max'], float(st['negmass'].max()))
                    a['tv'] += float(st['tv'].sum())
                    a['tv>0.05'] += float((st['tv'] > 0.05).sum())
                    a['agree'] += float(st['agree'].sum())
                    a['steps_with_neg'] += float(st['neg'].sum() > 0)
                    a['steps'] += 1
                # continue the GDCE-I replay with the recorded posterior noise
                xi = torch.argmax(torch.log(prob_X + 1e-10) + g_X[s_int].to(device), dim=-1)
                ei = torch.argmax(torch.log(prob_E + 1e-10) + g_E[s_int].to(device), dim=-1)
                ei = torch.triu(ei, diagonal=1)
                ei = ei + ei.transpose(1, 2)
                X_t = F.one_hot(xi, num_classes=model.Xdim_output).float()
                E_t = F.one_hot(ei, num_classes=model.Edim_output).float()
            print(f'  s={s_val:g} [{g_i:3d}] idx={data_idx:4d} n={n:2d} {src_class}->{1 - src_class} '
                  f'(total {time.time() - t_start:.0f}s)')

    rows = []
    for (s_val, bucket, kind), a in sorted(acc.items()):
        P = a['positions']
        rows.append({'dataset': ds_name, 's': s_val, 'steps': bucket, 'kind': kind,
                     'positions': int(P), 'share_neg': a['neg'] / P,
                     'mean_negmass': a['negmass'] / P, 'max_negmass': a['negmass_max'],
                     'mean_tv': a['tv'] / P, 'share_tv>0.05': a['tv>0.05'] / P,
                     'argmax_agree': a['agree'] / P,
                     'share_steps_with_neg': a['steps_with_neg'] / a['steps']})
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print('  ' + '  '.join(f'{k}={v:.4g}' if isinstance(v, float) else f'{k}={v}' for k, v in r.items()))
    print(f'DONE -> {out_path}')


if __name__ == '__main__':
    main()
