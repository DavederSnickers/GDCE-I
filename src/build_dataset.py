"""Build one SpectreGraphDataset split and check its marginals.

Triggers download() (TUDataset -> CSV + the .npy marginals) and process()
(CSV -> PyG) and reports whether the tensor shapes match what
SpectreDatasetInfos will later expect. Catches the errors that would otherwise
surface hours into a training job.

Usage:
    python src/build_dataset.py --name proteins
    python src/build_dataset.py --name mutagenicity_mol
"""
import argparse
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--datadir', default=None,
                    help='Default: data/<name>/')
    args = ap.parse_args()

    datadir = args.datadir or f'data/{args.name}/'
    root = os.path.abspath(datadir)

    from datasets.spectre_dataset import SpectreGraphDataset, SpectreDatasetInfos

    print('=' * 78)
    print(f'SMOKE: {args.name}  (root={root})')
    print('=' * 78)

    sizes = {}
    for split in ('train', 'val', 'test'):
        ds = SpectreGraphDataset(dataset_name=args.name, split=split, root=root)
        sizes[split] = len(ds)
        print(f'  {split:5s}: {len(ds):5d} Graphen')

    ds = SpectreGraphDataset(dataset_name=args.name, split='train', root=root)
    g = ds[0]
    print('\n  Beispielgraph:')
    print(f'    x          {tuple(g.x.shape)}   dtype={g.x.dtype}')
    print(f'    edge_index {tuple(g.edge_index.shape)}')
    print(f'    edge_attr  {tuple(g.edge_attr.shape)}')
    print(f'    y          {tuple(g.y.shape)}')
    guid = getattr(g, 'guidance', None)
    print(f'    guidance   {tuple(guid.shape) if guid is not None else None}  {guid}')
    assert bool((g.x.sum(dim=1) == 1).all()), 'x is not one-hot'

    raw = os.path.join(root, 'raw')
    print('\n  Marginals auf Platte:')
    for f, label in [('n_nodes_dist.npy', 'n_nodes'),
                     ('node_type_marginals.npy', 'node_types'),
                     ('edge_type_marginals.npy', 'edge_types')]:
        p = os.path.join(raw, f)
        if os.path.exists(p):
            a = np.load(p)
            head = np.round(a[:8], 4).tolist()
            print(f'    {label:11s} len={len(a):3d}  sum={a.sum():.4f}  {head}'
                  f'{" ..." if len(a) > 8 else ""}')
        else:
            print(f'    {label:11s} MISSING ({p})')

    # SimpleNamespace rather than a class: inside a class body `datadir =
    # datadir` would resolve against the class namespace, not the local
    # variable, and raise NameError.
    from types import SimpleNamespace
    infos = SpectreDatasetInfos(SimpleNamespace(name=args.name, datadir=datadir))
    print('\n  SpectreDatasetInfos:')
    print(f'    n_nodes    {tuple(infos.n_nodes.shape)}  sum={infos.n_nodes.sum():.4f}')
    print(f'    node_types {tuple(infos.node_types.shape)}  {torch.round(infos.node_types, decimals=4).tolist()}')
    print(f'    edge_types {tuple(infos.edge_types.shape)}  {torch.round(infos.edge_types, decimals=4).tolist()}')
    assert infos.node_types.shape[0] == g.x.shape[1], (
        f'node_types ({infos.node_types.shape[0]}) passt nicht zu x-dim ({g.x.shape[1]})')

    print(f'\n  OK: {sum(sizes.values())} Graphen gesamt, '
          f'{g.x.shape[1]} Knotentyp(en), {infos.edge_types.shape[0]} Kantenklassen')
    print('=' * 78)


if __name__ == '__main__':
    main()
