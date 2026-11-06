"""Which node and edge features a TUDataset actually carries.

TUDatasets separate *labels* (categorical, one-hot) from *attributes*
(continuous). PyG loads only the labels by default; use_node_attr=True adds the
attributes. The difference decides whether a published oracle accuracy is
comparable to one's own at all -- a classifier on continuous attributes solves a
different task from one on discrete types.

Usage:
    python src/check_tu_features.py PROTEINS TWITTER-Real-Graph-Partial
"""
import sys


def main():
    from torch_geometric.datasets import TUDataset

    for name in sys.argv[1:]:
        print('=' * 70)
        print(name)
        for use_attr in (False, True):
            root = f'data/tu_feat/{"attr" if use_attr else "lab"}'
            try:
                ds = TUDataset(root=root, name=name, use_node_attr=use_attr,
                               use_edge_attr=use_attr)
                g = ds[0]
                x = getattr(g, 'x', None)
                ea = getattr(g, 'edge_attr', None)
                print(f'  use_node_attr={use_attr!s:5s} '
                      f'x={tuple(x.shape) if x is not None else None}  '
                      f'edge_attr={tuple(ea.shape) if ea is not None else None}  '
                      f'graphs={len(ds)}')
                if x is not None:
                    onehot = bool((x.sum(dim=1) == 1).all()) and \
                        set(x.unique().tolist()) <= {0.0, 1.0}
                    print(f'          x ist {"kategorial (One-Hot)" if onehot else "kontinuierlich"}'
                          f'   min={float(x.min()):.3f} max={float(x.max()):.3f}')
            except Exception as e:
                print(f'  use_node_attr={use_attr!s:5s} FEHLER {type(e).__name__}: {str(e)[:80]}')
    print('=' * 70)


if __name__ == '__main__':
    main()
