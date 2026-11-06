"""Effect of different frequency filters on a TUDataset with a large vocabulary.

Answers two questions quantitatively before a dataset is adopted:

  1. How many node types survive threshold k, and how much data does that cost?
     The nci1/mutagenicity branch of the pipeline follows Huang et al.: types
     with a global frequency <= k are dropped AND every graph containing such a
     type is removed entirely. The second part is the expensive one.

  2. How large does the intermediate tensor in
     compute_batched_over0_posterior_distribution get? Its shape is
     (bs, n, d0, d_t-1), i.e. bs * n * dx^2 float32 -- quadratic in the number
     of node types, and the reason why large vocabularies fail here on memory
     rather than on compute time.

Usage:
    python src/analyze_freq_filter.py --name TWITTER-Real-Graph-Partial
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--root', default='data/tu_screen')
    ap.add_argument('--max-nodes', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=128,
                    help='fuer die Speicherabschaetzung')
    ap.add_argument('--thresholds', default='0,50,100,500,1000,5000,10000')
    args = ap.parse_args()

    from torch_geometric.datasets import TUDataset

    ds = TUDataset(root=args.root, name=args.name)
    graphs = [g for g in ds
              if args.max_nodes is None or g.num_nodes <= args.max_nodes]

    per_graph_types = [g.x.argmax(dim=1).cpu().numpy() for g in graphs]
    labels = np.array([int(g.y.item()) for g in graphs])
    n_types_total = graphs[0].x.shape[1]

    counts = np.bincount(np.concatenate(per_graph_types), minlength=n_types_total)
    n_nodes_total = int(counts.sum())
    max_n = max(g.num_nodes for g in graphs)

    print('=' * 92)
    print(f'FREQUENZFILTER: {args.name}')
    print(f'  {len(graphs)} Graphen, {n_nodes_total} Knoteninstanzen, '
          f'{n_types_total} Typen, max {max_n} Knoten')
    print('=' * 92)
    print(f'{"Schwelle":>9s} {"Typen":>7s} {"Knoten%":>9s} {"Graphen":>9s} '
          f'{"Graphen%":>9s} {"Balance":>13s} {"n_avg":>7s} {"n_med":>6s} '
          f'{"2-Knoten":>9s} {"n_max":>6s} {"bs*n*dx^2":>11s}')
    print('-' * 92)

    for k in [int(x) for x in args.thresholds.split(',')]:
        keep_type = counts > k
        n_keep_types = int(keep_type.sum())
        if n_keep_types == 0:
            print(f'{k:>9d} {0:>7d}   -- alle Typen verworfen --')
            continue

        # Huang et al.: a graph is dropped as soon as it contains one rare type.
        keep_graph = np.array([bool(keep_type[t].all()) for t in per_graph_types])
        n_keep_graphs = int(keep_graph.sum())
        node_share = counts[keep_type].sum() / n_nodes_total
        graph_share = n_keep_graphs / len(graphs)

        if n_keep_graphs:
            lab = labels[keep_graph]
            bal = f'{(lab == 0).mean():.3f}/{(lab == 1).mean():.3f}'
        else:
            bal = '--'

        # Size distribution AFTER the filter: rare types sit preferentially in
        # larger graphs, so the Huang rule systematically removes the big ones --
        # that selection effect matters more than the raw count.
        sizes = np.array([len(t) for t, keep in zip(per_graph_types, keep_graph) if keep]) \
            if n_keep_graphs else np.array([0])
        n_avg, n_med = sizes.mean(), np.median(sizes)
        share2 = (sizes <= 2).mean()
        n_max_kept = int(sizes.max())

        # (bs, n, d0, d_t-1) float32 in the posterior; n = largest remaining graph
        gb = args.batch_size * n_max_kept * n_keep_types ** 2 * 4 / 1024 ** 3
        mem = f'{gb * 1024:.0f} MB' if gb < 1 else f'{gb:.1f} GB'

        print(f'{k:>9d} {n_keep_types:>7d} {node_share:>8.1%} {n_keep_graphs:>9d} '
              f'{graph_share:>8.1%} {bal:>13s} {n_avg:>7.2f} {n_med:>6.0f} '
              f'{share2:>8.1%} {n_max_kept:>6d} {mem:>11s}')

    print('-' * 92)
    print(f'  Speicherspalte: batch={args.batch_size}, n={max_n}, float32, '
          f'ein Zwischentensor pro Reverse-Schritt')
    print('=' * 92)


if __name__ == '__main__':
    main()
