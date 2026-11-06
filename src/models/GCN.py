import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import Optional

from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, BatchNorm


class GCNLightning(pl.LightningModule):
    """
    Graph-level GCN using PyG with improved architecture.
    - graph classification (binary or multi-class) OR regression
    - assumes input batch is a PyG Batch with fields:
        x: [num_nodes, in_dim]
        edge_index: [2, num_edges]
        batch: [num_nodes] graph assignment
        y: [num_graphs] or [num_graphs, out_dim]
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        dropout: float = 0.1,
        task: str = "classification",   # "classification" or "regression"
        pos_weight: Optional[float] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dropout = dropout
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # GCN layers with BatchNorm
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))

        # MLP head (takes concatenated mean+max pooling = 2*hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

        # Loss setup
        self.task = task
        if task == "classification":
            # If out_dim == 1 -> binary classification with BCEWithLogits
            if out_dim == 1:
                if pos_weight is not None:
                    self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
                else:
                    self.loss_fn = nn.BCEWithLogitsLoss()
            else:
                self.loss_fn = nn.CrossEntropyLoss()
        elif task == "regression":
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unknown task={task}")

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        # Input projection
        x = self.input_proj(x)

        # GCN layers with residual connections
        for i in range(self.num_layers):
            x_res = x
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res  # Residual connection

        # Combine mean and max pooling for richer graph representation
        g_mean = global_mean_pool(x, batch_idx)
        g_max = global_max_pool(x, batch_idx)
        g = torch.cat([g_mean, g_max], dim=-1)  # [num_graphs, hidden_dim * 2]

        out = self.mlp(g)
        return out

    def _shared_step(self, batch, stage: str):
        logits = self(batch)   # [B, out_dim]
        # Use guidance (one-hot encoded) if y is empty, convert to class indices
        if hasattr(batch, 'guidance') and batch.y.numel() == 0:
            y = batch.guidance.argmax(dim=-1)  # Convert one-hot to class indices
        else:
            y = batch.y

        if self.task == "classification":
            if self.hparams.out_dim == 1:
                # Expect y shape [B] or [B,1] with {0,1}
                y = y.float().view(-1, 1)
                loss = self.loss_fn(logits, y)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                acc = (preds.view(-1) == y.view(-1).long()).float().mean()

                self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
                self.log(f"{stage}/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
            else:
                # Multi-class: y shape [B] with class indices
                y = y.long().view(-1)
                loss = self.loss_fn(logits, y)
                preds = logits.argmax(dim=-1)
                acc = (preds == y).float().mean()

                self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
                self.log(f"{stage}/acc", acc, prog_bar=True, on_step=False, on_epoch=True)

        else:  # regression
            y = y.float()
            if y.dim() == 1:
                y = y.view(-1, 1)
            loss = self.loss_fn(logits, y)
            mae = (logits - y).abs().mean()

            self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{stage}/mae", mae, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        # CosineAnnealingWarmRestarts - doesn't require a monitored metric
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=10, T_mult=2, eta_min=1e-6
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
