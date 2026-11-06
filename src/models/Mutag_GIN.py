import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import AUROC

from torch_geometric.nn import GINConv, global_add_pool


def _make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """2-layer MLP with BatchNorm and ReLU, as used in each GIN block."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
    )


class MutagGINLightning(pl.LightningModule):
    """
    GIN-based Mutagenicity classifier following Kazius et al. / paper description:
      - 3 GIN layers, each with a 2-layer MLP (dim -> 128 -> 128)
      - Global sum pooling
      - Linear classifier: 128 -> 2
      - Adam optimizer, lr = 5e-5
      - LR decay: lr / (1 + epoch / n_epochs)

    Input dim: number of one-hot atom type features (typically 14 for raw Mutagenicity).
    Output dim: 2 (Non-mutagenic / Mutagenic).
    """

    def __init__(
        self,
        in_dim: int = 14,
        hidden_dim: int = 128,
        out_dim: int = 2,
        num_layers: int = 3,
        lr: float = 5e-5,
        n_epochs: int = 25,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.convs = nn.ModuleList()
        self.convs.append(GINConv(_make_mlp(in_dim, hidden_dim), train_eps=True))
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(_make_mlp(hidden_dim, hidden_dim), train_eps=True))

        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.loss_fn = nn.CrossEntropyLoss()

        self.auroc_train = AUROC(task='binary')
        self.auroc_val   = AUROC(task='binary')
        self.auroc_test  = AUROC(task='binary')

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        x = x.float()

        for conv in self.convs:
            x = conv(x, edge_index)

        x = global_add_pool(x, batch_idx)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        # LR schedule: γ = lr / (1 + epoch/n_epochs)  — same decay as paper
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1.0 / (1.0 + epoch / self.hparams.n_epochs),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
