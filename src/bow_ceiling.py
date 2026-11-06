"""Bag-of-words ceiling for an already built dataset.

Answers the question that comes before any hyperparameter tuning: how much
accuracy is in the node types alone, without any topology at all?

On TWITTER the graphs are nearly complete -- a median of 3 nodes and an edge
marginal of 0.885 -- so a graph is essentially a set of words. If a logistic
regression on the bare presence vector reaches the same level as the trained GNN,
then the model is not too weak; the ceiling is low, and tuning buys nothing.

Measured on the same splits as the GNN training, so that the two numbers are
directly comparable.

Usage:
    python src/bow_ceiling.py --name twitter
"""
import argparse
import os

import numpy as np


def softmax_fit(Xtr, ytr, n_classes, epochs=400, lr=0.5, l2=1e-4):
    X = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Y = np.eye(n_classes)[ytr.astype(int)]
    W = np.zeros((X.shape[1], n_classes))
    for _ in range(epochs):
        z = X @ W
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        W -= lr * ((X.T @ (p - Y)) / len(X) + l2 * W)
    return W


def acc(W, X, y):
    X = np.hstack([X, np.ones((len(X), 1))])
    return float(((X @ W).argmax(axis=1) == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--datadir', default=None)
    ap.add_argument('--raw-tu', default=None,
                    help='TUDataset name: measures the ceiling on the '
                         'UNFILTERED vocabulary, to rule out that the frequency '
                         'filter is dragging the result down.')
    args = ap.parse_args()

    if args.raw_tu:
        from torch_geometric.datasets import TUDataset
        import numpy.random as npr
        ds = TUDataset(root='data/tu_screen', name=args.raw_tu)
        n_types = ds[0].x.shape[1]
        F = np.zeros((len(ds), n_types), dtype=np.float32)
        L = np.zeros(len(ds), dtype=np.int64)
        for i, g in enumerate(ds):
            F[i] = g.x.sum(dim=0).clamp(max=1).cpu().numpy()
            L[i] = int(g.y.item())
        rng = npr.RandomState(0); perm = rng.permutation(len(ds))
        a, b = int(len(ds) * .6), int(len(ds) * .8)
        feats = {'train': F[perm[:a]], 'val': F[perm[a:b]], 'test': F[perm[b:]]}
        labels = {'train': L[perm[:a]], 'val': L[perm[a:b]], 'test': L[perm[b:]]}
        n_classes = int(L.max()) + 1
        maj = np.bincount(labels['train'], minlength=n_classes).max() / len(labels['train'])
        print('=' * 78)
        print(f'BAG-OF-WORDS-DECKE (UNGEFILTERT): {args.raw_tu}')
        print(f'  {len(feats["train"])}/{len(feats["val"])}/{len(feats["test"])} Graphen, '
              f'{n_types} Typen, Mehrheitsklasse {maj:.3f}')
        print('=' * 78)
        for l2 in (1e-5, 1e-4, 1e-3, 1e-2):
            W = softmax_fit(feats['train'], labels['train'], n_classes, l2=l2)
            print(f'  L2={l2:<7g} train {acc(W, feats["train"], labels["train"]):.3f}   '
                  f'val {acc(W, feats["val"], labels["val"]):.3f}   '
                  f'test {acc(W, feats["test"], labels["test"]):.3f}')
        print('=' * 78)
        return

    datadir = args.datadir or f'data/{args.name}/'
    root = os.path.abspath(datadir)

    from datasets.spectre_dataset import SpectreGraphDataset

    feats, labels, sd_feats = {}, {}, {}
    n_types = None
    for split in ('train', 'val', 'test'):
        ds = SpectreGraphDataset(dataset_name=args.name, split=split, root=root)
        n_types = ds[0].x.shape[1]
        # Presence vector: which node types occur. For nearly complete graphs
        # that is everything a GNN can extract from the features.
        F = np.zeros((len(ds), n_types), dtype=np.float32)
        L = np.zeros(len(ds), dtype=np.int64)
        for i, g in enumerate(ds):
            F[i] = g.x.sum(dim=0).clamp(max=1).cpu().numpy()
            L[i] = int(g.guidance.argmax(dim=-1).item())
        feats[split], labels[split] = F, L

        # Size and density on the SAME split -- only then is the shortcut
        # comparable to the classifier's accuracy. The screening number comes
        # from 5-fold CV over all graphs and measures a different protocol.
        S = np.zeros((len(ds), 3), dtype=np.float32)
        for i, g in enumerate(ds):
            n = float(g.num_nodes); m = float(g.edge_index.shape[1]) / 2.0
            S[i] = (n, m, 2.0 * m / max(n * (n - 1), 1.0))
        sd_feats[split] = S

    n_classes = int(max(l.max() for l in labels.values())) + 1
    maj = np.bincount(labels['train'], minlength=n_classes).max() / len(labels['train'])

    print('=' * 78)
    print(f'BAG-OF-WORDS-DECKE: {args.name}')
    print(f'  {len(feats["train"])}/{len(feats["val"])}/{len(feats["test"])} Graphen, '
          f'{n_types} Typen, Mehrheitsklasse {maj:.3f}')
    print('=' * 78)

    for l2 in (1e-5, 1e-4, 1e-3, 1e-2):
        W = softmax_fit(feats['train'], labels['train'], n_classes, l2=l2)
        a_tr = acc(W, feats['train'], labels['train'])
        a_va = acc(W, feats['val'], labels['val'])
        a_te = acc(W, feats['test'], labels['test'])
        print(f'  L2={l2:<7g} train {a_tr:.3f}   val {a_va:.3f}   test {a_te:.3f}'
              f'   (train-val gap {a_tr - a_va:+.3f})')

    print('-' * 78)
    print('  size/density on the same split (the actual shortcut):')
    for l2 in (1e-4, 1e-2):
        mu, sd = sd_feats['train'].mean(0), sd_feats['train'].std(0) + 1e-8
        W = softmax_fit((sd_feats['train'] - mu) / sd, labels['train'], n_classes, l2=l2)
        print(f'    L2={l2:<7g} train {acc(W, (sd_feats["train"] - mu) / sd, labels["train"]):.3f}   '
              f'val {acc(W, (sd_feats["val"] - mu) / sd, labels["val"]):.3f}   '
              f'test {acc(W, (sd_feats["test"] - mu) / sd, labels["test"]):.3f}')
    print('-' * 78)
    print('  If val/test reach GNN level here, the ceiling is low and tuning')
    print('  the GNN will not solve the problem.')
    print('=' * 78)


if __name__ == '__main__':
    main()
