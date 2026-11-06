import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import AUROC

from torch.nn import ModuleList, ReLU, Softmax
from torch_geometric.nn import BatchNorm, LEConv, global_mean_pool


class MutagGCNLightning(pl.LightningModule):
    """
    Mutagenicity graph classifier — same architecture as NCI1GCNLightning.

    Architecture:
      - LEConv layers (14 -> 128 -> ... -> 128)
      - ReLU activations
      - global_mean_pool
      - FFN: 128 -> 128 -> ReLU -> Dropout -> 2

    Input dim: determined by frequency filtering (freq > 50 threshold, typically ~10 types)
    Output dim: 2 (Non-mutagenic / Mutagenic)
    """
    def __init__(
        self,
        in_dim: int = 10,  # freq>50 filter on Mutagenicity typically yields ~10 atom types
        hidden_dim: int = 128,
        out_dim: int = 2,
        conv_unit: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.relus = ModuleList()

        self.convs.append(LEConv(in_channels=in_dim, out_channels=hidden_dim))
        for _ in range(conv_unit - 2):
            self.convs.append(LEConv(in_channels=hidden_dim, out_channels=hidden_dim))
        self.convs.append(LEConv(in_channels=hidden_dim, out_channels=hidden_dim))

        self.batch_norms.extend([BatchNorm(hidden_dim)] * conv_unit)
        self.relus.extend([ReLU()] * conv_unit)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        self.softmax = Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss()

        self.auroc_train = AUROC(task='binary')
        self.auroc_val   = AUROC(task='binary')
        self.auroc_test  = AUROC(task='binary')

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)

        for conv, batch_norm, relu in zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_weight)
            x = relu(x)

        graph_x = global_mean_pool(x, batch_idx)
        pred = self.ffn(graph_x)
        self.readout = self.softmax(pred)
        return pred

    def _shared_step(self, batch, stage: str):
        logits = self(batch)

        if hasattr(batch, 'guidance') and batch.y.numel() == 0:
            y = batch.guidance.argmax(dim=-1)
        else:
            y = batch.y

        y = y.long().view(-1)
        loss = self.loss_fn(logits, y)

        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()

        probs_pos = F.softmax(logits, dim=-1)[:, 1]
        auroc_metric = getattr(self, f'auroc_{stage}')
        auroc_metric.update(probs_pos, y)

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}/acc",  acc,  prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}/auroc", auroc_metric, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.8, patience=10, min_lr=1e-5
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "strict": False,
            },
        }
