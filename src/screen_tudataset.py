"""Shortcut screening for any TUDataset -- run it before training.

Applies the same accept/reject protocol as `screen_benzene.py`, so that the
result is directly comparable to the existing calibration:

  | dataset      | balance   | size/density acc | outcome            |
  |--------------|-----------|------------------|--------------------|
  | NCI1         | 52.9/47.1 | 0.613            | works              |
  | Mutagenicity | 57.6/42.4 | 0.619            | works              |
  | Benzene      | 50.0/50.0 | 0.637            | accepted           |
  | AIDS         | 15.5/84.5 | 0.995            | rejected           |

What decides it is the RAW accuracy of a logistic regression on
[n_nodes, n_edges, density] -- NOT the lift over the majority class. On AIDS the
lift is only +0.149 and the dataset is still the total failure, because the
majority class already sits at 84.5 %. The working range is about 0.60-0.65.

The script additionally checks what GDCE-I can model at all: categorical node
and edge labels. With neither, only topology editing is left.

Usage (inside the container):
    python src/screen_tudataset.py --name PROTEINS
    python src/screen_tudataset.py --name IMDB-BINARY --max-nodes 50
    python src/screen_tudataset.py --name Fingerprint
"""
import argparse

import numpy as np


def auc(scores, labels):
    """ROC-AUC via rank statistics (Mann-Whitney U), without sklearn."""
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def softmax_cv_acc(feats, labels, n_classes, folds=5, epochs=800, lr=0.5, seed=0):
    """Multinomial logistic regression (numpy, GD) with k-fold CV.

    The binary variant from screen_benzene.py, generalised to K classes, so that
    ENZYMES/IMDB-MULTI/Fingerprint (more than 2 classes) can be screened too.
    For K=2 the result is identical to that implementation.
    """
    rng = np.random.RandomState(seed)
    X = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
    X = np.hstack([X, np.ones((len(X), 1))])
    Y = np.eye(n_classes)[labels.astype(int)]

    idx = rng.permutation(len(X))
    accs = []
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        W = np.zeros((X.shape[1], n_classes))
        for _ in range(epochs):
            z = X[tr] @ W
            z -= z.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            W -= lr * (X[tr].T @ (p - Y[tr])) / len(tr)
        pred = (X[te] @ W).argmax(axis=1)
        accs.append((pred == labels[te]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def describe_attributes(ds, graph0):
    """What categorical information is present -- the feasibility check.

    GDCE-I models categorical node and edge types. TUDatasets deliver node and
    edge *labels* as one-hot in x and edge_attr, but *attributes* as continuous
    values. The difference decides whether the dataset exercises the method at
    all.
    """
    print('\n[0] ATTRIBUTES (this decides feasibility)')

    x = getattr(graph0, 'x', None)
    if x is None:
        print('    nodes: NO features  -> only a single constant node type is possible')
    else:
        vals = np.unique(x.cpu().numpy())
        onehot = set(vals.tolist()) <= {0.0, 1.0} and bool((x.sum(dim=1) == 1).all())
        kind = 'kategorial (One-Hot)' if onehot else 'kontinuierlich (Attribute)'
        print(f'    nodes: dim={x.shape[1]}  {kind}')
        if onehot:
            allt = np.concatenate([g.x.argmax(1).cpu().numpy() for g in ds])
            cnt = np.bincount(allt, minlength=x.shape[1])
            share = cnt / cnt.sum()
            top = np.argsort(-share)[:10]
            print(f'      types: {x.shape[1]}   top-10 marginals: '
                  f'{np.round(share[top], 4).tolist()}')
            # Large vocabularies are the real risk: the transition matrix grows
            # quadratically, and rare types are not statistically learnable.
            # Huang et al. filter a global count of <=50 -- the same rule as in
            # the nci1/mutagenicity branches of spectre_dataset.py.
            print(f'      types with <=50 occurrences globally: '
                  f'{int((cnt <= 50).sum())} of {x.shape[1]}'
                  f'  (leaving {int((cnt > 50).sum())})')

    ea = getattr(graph0, 'edge_attr', None)
    if ea is None:
        print('    edges: NO features  -> binary adjacency, no edge-type sparsity')
    else:
        vals = np.unique(ea.cpu().numpy())
        onehot = set(vals.tolist()) <= {0.0, 1.0} and bool((ea.sum(dim=1) == 1).all())
        kind = 'kategorial (One-Hot)' if onehot else 'kontinuierlich (Attribute)'
        print(f'    edges: dim={ea.shape[1]}  {kind}')
        if not onehot:
            print('      -> would have to be discretised to serve as an edge type')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, help='TUDataset-Name, z.B. PROTEINS')
    ap.add_argument('--root', default='data/tu_screen')
    ap.add_argument('--max-nodes', type=int, default=None,
                    help='optionaler Groessenfilter, wie ihn download() spaeter anwendet')
    args = ap.parse_args()

    from torch_geometric.datasets import TUDataset

    ds = TUDataset(root=args.root, name=args.name)
    graph0 = ds[0]

    print('=' * 78)
    print(f'SCREENING: {args.name}')
    print('=' * 78)

    keep = [g for g in ds
            if args.max_nodes is None or g.num_nodes <= args.max_nodes]
    n_dropped = len(ds) - len(keep)

    labels = np.array([int(g.y.item()) for g in keep])
    n_nodes = np.array([g.num_nodes for g in keep], dtype=float)
    n_edges = np.array([g.num_edges / 2.0 for g in keep], dtype=float)
    density = 2.0 * n_edges / np.maximum(n_nodes * (n_nodes - 1), 1)
    avg_deg = 2.0 * n_edges / np.maximum(n_nodes, 1)

    describe_attributes(ds, graph0)

    # --- 1. class balance -------------------------------------------------
    n_tot = len(keep)
    classes = np.unique(labels)
    n_classes = int(classes.max()) + 1
    counts = np.bincount(labels, minlength=n_classes)
    maj = counts.max() / n_tot
    print(f'\n[1] CLASS BALANCE   graphs: {n_tot}'
          + (f'  ({n_dropped} wegen >{args.max_nodes} Knoten verworfen)' if n_dropped else ''))
    for c, cnt in enumerate(counts):
        print(f'    class {c}: {cnt:6d}  ({cnt / n_tot:.1%})')
    print(f'    majority class (= baseline accuracy): {maj:.1%}')
    print(f'    -> {"OK" if maj <= 0.60 else "WARNING: imbalanced"}')

    # --- 2. size / density shortcut ---------------------------------------
    print('\n[2] SIZE / DENSITY SHORTCUT')
    header = f'    {"":22s}' + ''.join(f'{f"Klasse {c}":>12s}' for c in range(n_classes))
    print(header + (f'{"AUC":>8s}' if n_classes == 2 else ''))
    for name, v in [('Knoten', n_nodes), ('Kanten', n_edges),
                    ('Dichte', density), ('mittl. Grad', avg_deg)]:
        cells = ''.join(f'{v[labels == c].mean():12.2f}' for c in range(n_classes))
        tail = f'{auc(v, labels):8.3f}' if n_classes == 2 else ''
        print(f'    {name:22s}{cells}{tail}')

    feats = np.stack([n_nodes, n_edges, density], axis=1)
    acc, sd = softmax_cv_acc(feats, labels, n_classes)
    print(f'\n    logistic regression on [nodes, edges, density] ONLY:')
    print(f'      5-fold CV accuracy: {acc:.3f} +/- {sd:.3f}   (majority class {maj:.3f})')
    print(f'      lift over the majority class: {acc - maj:+.3f}')
    # Thresholds follow the documented calibration: NCI1 0.613 / Mutagenicity
    # 0.619 / Benzene 0.637 work, AIDS 0.995 fails. The working range is
    # therefore 0.60-0.65 -- anything above is already amber, even when it is
    # still far from AIDS.
    if acc >= 0.90:
        print('      -> RED: solvable from size and density (the AIDS failure mode, acc 0.995)')
    elif acc > 0.65:
        print('      -> AMBER: above the working range 0.60-0.65 (NCI1/Mutagenicity/Benzene)')
    else:
        print('      -> GREEN: inside the working range of NCI1/Mutagenicity/Benzene')

    print(f'\n    nodes min/median/max: {n_nodes.min():.0f} / '
          f'{np.median(n_nodes):.0f} / {n_nodes.max():.0f}')
    iso = int((n_edges == 0).sum())
    if iso:
        print(f'    WARNING: {iso} graphs without edges')
    print('=' * 78)


if __name__ == '__main__':
    main()
