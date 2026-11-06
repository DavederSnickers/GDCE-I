import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import AUROC

from torch_geometric.nn import GINEConv, global_add_pool


def _make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """2-layer MLP with BatchNorm and ReLU, as used in each GINE block."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
    )


class MutagGINELightning(pl.LightningModule):
    """
    GINEConv-based Mutagenicity classifier with edge features.

    Architecture:
      - 3 GINEConv layers, each receives raw edge_attr (edge_dim) directly.
        GINEConv handles the per-layer projection internally via its own
        Linear(edge_dim → node_dim), which avoids the information bottleneck
        of a shared external edge_encoder.
      - Global sum pooling → BN → ReLU → Dropout → Linear classifier

    Input:
      - Node features: one-hot atom types (in_dim)
      - Edge features: one-hot bond types (edge_dim=5: 0=no bond, 1-4=bond type)
    Output: 2 (Non-mutagenic / Mutagenic)
    """

    def __init__(
        self,
        in_dim: int = 14,
        hidden_dim: int = 128,
        out_dim: int = 2,
        num_layers: int = 3,
        edge_dim: int = 5,
        dropout: float = 0.3,
        lr: float = 1e-4,
        n_epochs: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Each GINEConv gets raw edge features (edge_dim=5) and handles its
        # own internal projection Linear(edge_dim → node_dim) per layer.
        self.convs = nn.ModuleList()
        self.convs.append(GINEConv(_make_mlp(in_dim, hidden_dim), train_eps=True, edge_dim=edge_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GINEConv(_make_mlp(hidden_dim, hidden_dim), train_eps=True, edge_dim=edge_dim))

        self.post_pool = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.loss_fn = nn.CrossEntropyLoss()

        self.auroc_train = AUROC(task='binary')
        self.auroc_val   = AUROC(task='binary')
        self.auroc_test  = AUROC(task='binary')

    def forward(self, batch):
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        x = x.float()

        ea = edge_attr.float() if edge_attr is not None and edge_attr.numel() > 0 \
            else torch.zeros(edge_index.size(1), self.hparams.edge_dim, device=x.device)

        for conv in self.convs:
            x = conv(x, edge_index, ea)

        x = global_add_pool(x, batch_idx)
        x = self.post_pool(x)
        return self.classifier(x)

    def _shared_step(self, batch, stage: str):
        logits = self(batch)

        if hasattr(batch, 'guidance') and batch.y.numel() == 0:
            y = batch.guidance.argmax(dim=-1)
        else:
            y = batch.y
        y = y.long().view(-1)

        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()

        probs_pos = F.softmax(logits, dim=-1)[:, 1]
        auroc_metric = getattr(self, f'auroc_{stage}')
        auroc_metric.update(probs_pos, y)

        self.log(f"{stage}/loss",  loss,         prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}/acc",   acc,          prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}/auroc", auroc_metric, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=15, min_lr=1e-5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/acc",
                "interval": "epoch",
                "strict": False,
            },
        }
