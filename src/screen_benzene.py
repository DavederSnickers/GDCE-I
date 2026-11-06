"""Shortcut screening for the Benzene dataset (Sanchez-Lengeling et al., NeurIPS 2020).

Checks, BEFORE any training, the three properties on which AIDS and BA-2Motifs
failed:

  1. class balance            -- AIDS: 400/1600 -> the model learns the prior
  2. size / density shortcut  -- AIDS: 29.3 vs 10.3 nodes -> the model conditions
                                 on size instead of on chemistry
  3. signal strength          -- BA-2Motifs: 1 edge out of 300 slots (0.33%) -> the
                                 class signal vanishes in the reconstruction loss

Runs without graphxai or sklearn: the .npz is parsed directly (the logic of
extract_google_datasets.load_graphs), and AUC and logistic regression are in numpy.

Usage:
    PYTHONPATH=<repo> python src/screen_benzene.py --npz data/benzene/raw/benzene.npz
"""
import argparse

import numpy as np

ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Na', 'Ca', 'I', 'B', 'H', '*']


def load_npz(path):
    """Parse benzene.npz -> (list of graphs, ground-truth node masks).

    Each graph: a dict with 'nodes' (one-hot atom types) and 'senders'/'receivers'.
    att[i][0]['nodes'] is (n_nodes, n_explanations) -- several rings per molecule.
    """
    data = np.load(path, allow_pickle=True)
    att, X, y = data['attr'], data['X'][0], data['y']
    labels = np.array([y[i][0] for i in range(y.shape[0])], dtype=int)

    graphs = []
    for i in range(len(X)):
        n = X[i]['nodes'].shape[0]
        src = np.asarray(X[i]['senders'], dtype=int)
        dst = np.asarray(X[i]['receivers'], dtype=int)

        # deduplicate undirected edges, drop self-loops
        und = {(min(a, b), max(a, b)) for a, b in zip(src, dst) if a != b}

        node_imp = np.asarray(att[i][0]['nodes'])          # (n, n_exp)
        gt_nodes = int((node_imp.sum(axis=1) > 0).sum())   # nodes in ANY ring

        graphs.append({
            'n': n,
            'm': len(und),
            'atoms': X[i]['nodes'].argmax(axis=1),
            'gt_nodes': gt_nodes,
            'n_rings': node_imp.shape[1] if node_imp.ndim > 1 else 1,
        })
    return graphs, labels


def auc(scores, labels):
    """ROC-AUC via rank statistics (Mann-Whitney U), without sklearn."""
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    # average the ties, or AUC is biased for discrete features (n_nodes!)
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


def logreg_cv_acc(feats, labels, folds=5, epochs=800, lr=0.5, seed=0):
    """Logistic regression (numpy, GD) with k-fold CV -> mean test accuracy.

    This is the actual shortcut test: if a model that sees ONLY size and density
    beats the majority class by a wide margin, the dataset is trivially solvable
    and the diffusion model will take exactly that route.
    """
    rng = np.random.RandomState(seed)
    X = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
    X = np.hstack([X, np.ones((len(X), 1))])
    y = labels.astype(float)

    idx = rng.permutation(len(X))
    accs = []
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        w = np.zeros(X.shape[1])
        for _ in range(epochs):
            p = 1.0 / (1.0 + np.exp(-X[tr] @ w))
            w -= lr * (X[tr].T @ (p - y[tr])) / len(tr)
        pred = (1.0 / (1.0 + np.exp(-X[te] @ w))) > 0.5
        accs.append((pred == y[te]).mean())
    return float(np.mean(accs)), float(np.std(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='data/benzene/raw/benzene.npz')
    args = ap.parse_args()

    graphs, labels = load_npz(args.npz)
    n_nodes = np.array([g['n'] for g in graphs], dtype=float)
    n_edges = np.array([g['m'] for g in graphs], dtype=float)
    density = 2.0 * n_edges / np.maximum(n_nodes * (n_nodes - 1), 1)
    avg_deg = 2.0 * n_edges / np.maximum(n_nodes, 1)

    print('=' * 78)
    print('BENZENE SCREENING')
    print('=' * 78)

    # --- 1. class balance -------------------------------------------------
    n_tot = len(graphs)
    c0, c1 = int((labels == 0).sum()), int((labels == 1).sum())
    maj = max(c0, c1) / n_tot
    print(f'\n[1] CLASS BALANCE   graphs: {n_tot}')
    print(f'    class 0 (no benzene ring): {c0:5d}  ({c0 / n_tot:.1%})')
    print(f'    class 1 (benzene ring):    {c1:5d}  ({c1 / n_tot:.1%})')
    print(f'    majority class (= baseline accuracy): {maj:.1%}')
    print(f'    -> {"OK" if maj <= 0.60 else "WARNING: imbalanced"}')

    # --- 2. shortcut ------------------------------------------------------
    print('\n[2] SIZE / DENSITY SHORTCUT')
    print(f'    {"":22s} {"class 0":>12s} {"class 1":>12s} {"delta":>10s} {"AUC":>7s}')
    for name, v in [('nodes', n_nodes), ('edges', n_edges),
                    ('density', density), ('mean degree', avg_deg)]:
        m0, m1 = v[labels == 0].mean(), v[labels == 1].mean()
        rel = abs(m1 - m0) / (abs(m0) + 1e-9)
        print(f'    {name:22s} {m0:12.2f} {m1:12.2f} {rel:9.1%} {auc(v, labels):7.3f}')

    feats = np.stack([n_nodes, n_edges, density], axis=1)
    acc, sd = logreg_cv_acc(feats, labels)
    lift = acc - maj
    print(f'\n    logistic regression on [nodes, edges, density] ONLY:')
    print(f'      5-fold CV accuracy: {acc:.3f} +/- {sd:.3f}   (majority class {maj:.3f})')
    print(f'      lift over the majority class: {lift:+.3f}')
    if lift > 0.20:
        print('      -> RED: solvable from size and density (the AIDS failure mode)')
    elif lift > 0.10:
        print('      -> AMBER: partly short-circuitable')
    else:
        print('      -> GREEN: no size or density shortcut')

    # --- 3. signal strength -----------------------------------------------
    print('\n[3] SIGNAL STRENGTH (against BA-2Motifs = 0.33%)')
    pos = [g for g, l in zip(graphs, labels) if l == 1]
    gt = np.array([g['gt_nodes'] for g in pos], dtype=float)
    npos = np.array([g['n'] for g in pos], dtype=float)
    # a benzene ring is 6 nodes and 6 edges; the dense adjacency has n(n-1)/2 edge slots
    slots = npos * (npos - 1) / 2.0
    ring_edges = gt  # a ring is a cycle, so #edges == #nodes
    print(f'    ground-truth nodes per positive graph: {gt.mean():.2f} of {npos.mean():.2f}'
          f'  ({(gt / npos).mean():.1%} of the nodes)')
    print(f'    motif edges / edge slots:                {(ring_edges / slots).mean():.2%}')
    print(f'    rings per molecule (max explanations):   {np.mean([g["n_rings"] for g in pos]):.2f}')
    ratio = (ring_edges / slots).mean() / 0.0033
    print(f'    -> {ratio:.0f}x stronger than BA-2Motifs')

    # --- extra: atom types ------------------------------------------------
    all_atoms = np.concatenate([g['atoms'] for g in graphs])
    counts = np.bincount(all_atoms, minlength=len(ATOM_TYPES))
    print('\n[4] ATOM TYPES (global)')
    for i, c in enumerate(counts):
        if c > 0:
            name = ATOM_TYPES[i] if i < len(ATOM_TYPES) else f'?{i}'
            print(f'    {name:3s} {c:8d}  {c / counts.sum():6.2%}'
                  f'{"   <- <=50, would be filtered out" if c <= 50 else ""}')

    print(f'\n    nodes min/median/max: {n_nodes.min():.0f} / '
          f'{np.median(n_nodes):.0f} / {n_nodes.max():.0f}')
    print('=' * 78)


if __name__ == '__main__':
    main()
