from src.diffusion.distributions import DistributionNodes
import src.utils as utils
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader


class AbstractDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataloaders = None
        self.input_dims = None
        self.output_dims = None

    def prepare_data(self, datasets, samplers=None) -> None:
        batch_size = self.cfg.train.batch_size
        num_workers = self.cfg.train.num_workers
        samplers = samplers or {}
        self.dataloaders = {}
        for split, dataset in datasets.items():
            sampler = samplers.get(split, None)
            shuffle = (sampler is None) and ('debug' not in self.cfg.general.name) and (split == 'train')
            self.dataloaders[split] = DataLoader(
                dataset, batch_size=batch_size, num_workers=num_workers,
                shuffle=shuffle, sampler=sampler, pin_memory=self.cfg.dataset.pin_memory
            )

    def train_dataloader(self):
        return self.dataloaders["train"]

    def val_dataloader(self):
        return self.dataloaders["val"]

    def test_dataloader(self):
        return self.dataloaders["test"]

    def __getitem__(self, idx):
        return self.dataloaders['train'][idx]

    def node_counts(self, max_nodes_possible=300):
        all_counts = torch.zeros(max_nodes_possible)
        for split in ['train', 'val', 'test']:
            for i, data in enumerate(self.dataloaders[split]):
                unique, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self):
        num_classes = None
        for data in self.dataloaders['train']:
            num_classes = data.x.shape[1]
            break

        counts = torch.zeros(num_classes)

        for split in ['train', 'val', 'test']:
            for i, data in enumerate(self.dataloaders[split]):
                counts += data.x.sum(dim=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.edge_attr.shape[1]
            break

        d = torch.zeros(num_classes, dtype=torch.float)

        for i, data in enumerate(self.train_dataloader()):
            unique, counts = torch.unique(data.batch, return_counts=True)

            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)

            num_edges = data.edge_index.shape[1]
            num_non_edges = all_pairs - num_edges

            edge_types = data.edge_attr.sum(dim=0)
            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]

        d = d / d.sum()
        return d


class MolecularDataModule(AbstractDataModule):
    def valency_count(self, max_n_nodes):
        valencies = torch.zeros(3 * max_n_nodes - 2)   # Max valency possible if everything is connected

        multiplier = torch.Tensor([0, 1, 2, 3, 1.5])

        for split in ['train', 'val', 'test']:
            for i, data in enumerate(self.dataloaders[split]):
                n = data.x.shape[0]

                for atom in range(n):
                    edges = data.edge_attr[data.edge_index[0] == atom]
                    edges_total = edges.sum(dim=0)
                    valency = (edges_total * multiplier).sum()
                    valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        return valencies


class AbstractDatasetInfos:
    def complete_infos(self, n_nodes, node_types):
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(node_types)
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        cfg = datamodule.cfg

        example_batch = next(iter(datamodule.train_dataloader()))
        ex_dense, node_mask = utils.to_dense(example_batch.x, example_batch.edge_index, example_batch.edge_attr,
                                             example_batch.batch)
        example_data = {'X_t': ex_dense.X, 'E_t': ex_dense.E, 'y_t': example_batch['y'], 'node_mask': node_mask}

        self.input_dims = {'X': example_batch['x'].size(1),
                           'E': example_batch['edge_attr'].size(1),
                           'y': example_batch['y'].size(1) + 1}  # + 1 due to time conditioning
        ex_extra_feat = extra_features(example_data)
        self.input_dims['X'] += ex_extra_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_molecular_feat.y.size(-1)

        if (example_batch.guidance is not None):
            guidance_sz = example_batch.guidance.size(-1)

            if (cfg.guidance.guidance_medium in ['y', 'both']):
                self.input_dims['y'] += guidance_sz
            if (cfg.guidance.guidance_medium in ['XE', 'both']):
                self.input_dims['X'] += guidance_sz
                self.input_dims['E'] += guidance_sz

            self.input_dims['guidance'] = guidance_sz

        self.output_dims = {'X': example_batch['x'].size(1),
                            'E': example_batch['edge_attr'].size(1),
                            'y': 0}

        self.compute_class_conditional_node_dist(datamodule)

    def compute_class_conditional_node_dist(self, datamodule):
        """Build a per-class distribution over graph sizes from the training split.

        `sample_batch` otherwise draws the number of nodes from ONE global prior,
        independently of the guidance label. When the classes differ in size, that
        makes conditional generation of the smaller/rarer class impossible: e.g. on
        AIDS class 0 has ~28.5 nodes and class 1 ~10.2, and the global prior is
        dominated by the 85% majority, so "generate class 0" starts from a ~12-node
        canvas on which no class-0 graph exists.

        Sets `self.nodes_dist_per_class` (list indexed by class) or leaves it unset
        if the dataset has no usable one-hot class labels.
        """
        self.nodes_dist_per_class = None
        try:
            loader = datamodule.train_dataloader()
        except Exception:
            return

        max_n = self.max_n_nodes
        hists = {}
        for batch in loader:
            g = getattr(batch, 'guidance', None)
            if g is None or g.numel() == 0 or g.dim() != 2 or g.size(-1) < 2:
                return  # no categorical class label -> keep the global prior
            counts = torch.bincount(batch.batch, minlength=int(batch.batch.max()) + 1)
            cls = g.argmax(dim=-1)
            if cls.size(0) != counts.size(0):
                return
            for c, n in zip(cls.tolist(), counts.tolist()):
                h = hists.setdefault(int(c), torch.zeros(max_n + 1))
                if 0 <= n <= max_n:
                    h[n] += 1

        if len(hists) < 2 or any(h.sum() == 0 for h in hists.values()):
            return

        dists, summary = [], []
        for c in range(max(hists) + 1):
            h = hists.get(c)
            if h is None or h.sum() == 0:
                dists.append(self.nodes_dist)
                continue
            dists.append(DistributionNodes(h))
            mean_n = float((h * torch.arange(len(h))).sum() / h.sum())
            summary.append(f"class {c}: n={int(h.sum())} mean_nodes={mean_n:.1f}")
        self.nodes_dist_per_class = dists
        print(f"[node_dist] class-conditional graph sizes: {' | '.join(summary)}")