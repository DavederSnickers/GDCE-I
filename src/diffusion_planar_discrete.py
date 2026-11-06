import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
import os
import math
import pickle
from typing import List, Tuple
import networkx as nx
import pandas as pd

from torchmetrics import MeanAbsoluteError
from torch.distributions import Categorical
from torch_geometric.data import Data, Batch

from models.transformer_model import GraphTransformer
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete, \
    MarginalUniformTransition
from src.diffusion import diffusion_utils
from metrics.train_metrics import TrainLossDiscrete
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from src import utils
from src.timeout_exception import TimeoutException, time_limit
from src.guidance.node_model import QM9NodeModel
from src.models.NCI1_GCN import NCI1GCNLightning

from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFHasAllMoleculeParams, MMFFOptimizeMolecule
from rdkit import Chem
from rdkit.Chem import Crippen

try:
    import psi4
except ModuleNotFoundError:
    pass

from src.analysis.rdkit_functions import build_molecule, mol2smiles, build_molecule_with_partial_charges
from src.metrics.sascorer import calculateScore
from src.utils import graph2mol, clean_mol
from src.metrics.properties import mw, penalized_logp, qed, drd2



class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist
        # Per-class graph-size distribution (see AbstractDatasetInfos). Falls back to
        # the global prior when the dataset has no categorical class labels.
        self.nodes_dist_per_class = getattr(dataset_infos, 'nodes_dist_per_class', None)

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output
            e_limit = torch.ones(self.Edim_output) / self.Edim_output
            y_limit = torch.ones(self.ydim_output) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        elif cfg.model.transition == 'marginal':

            node_types = self.dataset_info.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = self.dataset_info.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)
            print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)

        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

        ##################################################################################

        if (input_dims['guidance'] != None):
            self.gdim = input_dims['guidance']
        else:
            self.gdim = 0

        # this will be used as the guidance when we need to calculate p(G_{t-1}|G_t) without the guidance
        if (self.cfg.guidance.trainable_cf == True):
            self.cf_null_token = torch.nn.parameter.Parameter(torch.randn(size=(1, self.gdim)))
        else:
            self.cf_null_token = torch.zeros(size=(1, self.gdim))

        # specific properties to generate molecules
        self.cond_val = MeanAbsoluteError()
        self.num_valid_molecules = 0
        self.num_total = 0

        # Per-class tracking for NCI1 classification accuracy
        self.nci1_class_correct = {0: 0, 1: 0}
        self.nci1_class_total = {0: 0, 1: 0}

        # wish I had a more elegant way to do this
        self.node_model = None
        self._graph_classifier_ref = [None]  # Stored in list to avoid nn.Module registration

        # stores the generated smiles on the test step
        self.generated_smiles = []

    @property
    def graph_classifier(self):
        return self._graph_classifier_ref[0]

    @graph_classifier.setter
    def graph_classifier(self, value):
        self._graph_classifier_ref[0] = value

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return

        # ---- CFG Diagnostic: verify guidance labels on first batch of epoch 0 ----
        if i == 0 and self.current_epoch == 0:
            g = data.guidance
            if g is not None and g.numel() > 0:
                if g.size(-1) == 1:
                    labels = (g.squeeze(-1) >= 0.5).long()
                else:
                    labels = g.argmax(dim=-1)
                unique, counts = torch.unique(labels, return_counts=True)
                dist_str = ", ".join([f"class{u.item()}={c.item()}" for u, c in zip(unique, counts)])
                self.print(f"[CFG] Epoch 0 | guidance shape: {list(g.shape)}, "
                           f"distribution: {dist_str} (batch={labels.numel()})")
            else:
                self.print(f"[CFG] WARNING: guidance is None or empty! shape={g.shape if g is not None else 'None'}")

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask, data.guidance, train_step=True)
        loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=X, true_E=E, true_y=data.y,
                               log=i % self.log_every_steps == 0)

        self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
                           log=i % self.log_every_steps == 0)

        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr,
                                      weight_decay=self.cfg.train.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.train.n_epochs, eta_min=1e-6
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("Size of the input features", self.Xdim, self.Edim, self.ydim)
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE'] :.3f}"
                   f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
                   f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
                   f" -- {time.time() - self.start_epoch_time:.1f}s ")
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}")

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.sampling_metrics.reset()

    def validation_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask, data.guidance)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=False,
                                    guidance=data.guidance)
        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        metrics = [self.val_nll.compute(), self.val_X_kl.compute() * self.T, self.val_E_kl.compute() * self.T,
                   self.val_X_logp.compute(), self.val_E_logp.compute()]
        if wandb.run:
            wandb.log({"val/epoch_NLL": metrics[0],
                       "val/X_kl": metrics[1],
                       "val/E_kl": metrics[2],
                       "val/X_logp": metrics[3],
                       "val/E_logp": metrics[4]}, commit=False)

        self.print(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Atom type KL {metrics[1] :.2f} -- ",
                   f"Val Edge type KL: {metrics[2] :.2f}")

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))

        self._eval_nci1_guidance_accuracy()

    def _eval_nci1_guidance_accuracy(self):
        """Periodically generate conditional samples and measure classifier accuracy per class."""
        if self.cfg.dataset.name not in ('nci1', 'mutagenicity', 'mutagenicity_mol',
                                         'ba2motifs', 'benzene', 'spmotif'):
            return
        if getattr(self.cfg.guidance, 'graph_classifier_path', None) is None:
            return
        # Run every 5th validation epoch (i.e. every 50 training epochs with check_val_every_n_epochs=10)
        self.val_counter += 1
        if self.val_counter % 5 != 0:
            return

        dataset_name = self.cfg.dataset.name
        self.print(f"Epoch {self.current_epoch}: Evaluating guidance accuracy ({dataset_name})...")
        n_samples_per_class = 10
        accs = {}

        for class_id in [0, 1]:
            guidance = F.one_hot(torch.tensor([class_id]), num_classes=2).float().to(self.device)

            samples = self.sample_batch(
                batch_id=0,
                batch_size=n_samples_per_class,
                keep_chain=0,
                number_chain_steps=1,
                save_final=0,
                guidance=guidance,
            )

            if dataset_name == 'mutagenicity_mol':
                acc = self.accuracy_test_mutag_mol(samples, guidance)
            elif dataset_name == 'mutagenicity':
                acc = self.accuracy_test_mutag(samples, guidance)
            else:
                acc = self.accuracy_test_nci1(samples, guidance)
            accs[class_id] = acc.item()

        overall_acc = (accs[0] + accs[1]) / 2.0
        self.print(f"Guidance accuracy — class0: {accs[0]:.2%}, class1: {accs[1]:.2%}, overall: {overall_acc:.2%}")

        if wandb.run:
            wandb.log({
                "val/guidance_acc_class0": accs[0],
                "val/guidance_acc_class1": accs[1],
                "val/guidance_acc_overall": overall_acc,
            }, commit=False)

        self.log("val/guidance_acc", overall_acc, sync_dist=True)

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def test_step(self, data, i):
        dataset_name = self.cfg.dataset.name


        if (self.cfg.train.batch_size > 1):
            print("WARNING: batch size > 1. You may not have enough batches to run this test.",
                  "Try relaunching this experiment with train.batch_size=1")
            print("batch size is:", self.cfg.train.batch_size)
            data.guidance = data.guidance[0, :]

        # checks if the current guidance element satisfies the required ranges
        if (self.cfg.guidance.experiment_type == 'accuracy'):
            valid_properties = self.cfg.guidance.guidance_target
            test_thresholds = self.cfg.guidance.test_thresholds

            k = 0
            for valid_property in valid_properties:
                curr_property = data.guidance[0, k].item()
                valid_threshold = test_thresholds[valid_property]

                if (valid_threshold[0] > curr_property or curr_property > valid_threshold[1]):
                    print(f"passing {i}-th element (value = {curr_property}).")
                    return None

                k = k + 1

        target_properties = data.guidance.clone()

        print(f'Select No.{i + 1} test sample — TARGET {target_properties.tolist()}')

        # loads self.node_model (only the first time because of the check self.node_model == None)
        if (self.node_model == None and self.cfg.guidance.node_model_path != None):
            input_size = len(self.cfg.guidance.guidance_target)

            node_model_kwargs = {'cfg': self.cfg, 'dataset_infos': self.dataset_info, 'input_size': input_size}
            self.node_model = QM9NodeModel.load_from_checkpoint(self.cfg.guidance.node_model_path, **node_model_kwargs)
            self.node_model.to(self.cfg.general.gpus[0])
            self.node_model.cfg.general.gpus = self.cfg.general.gpus

        num_nodes = None
        if (self.node_model != None):
            input_guidance = data.guidance.repeat(self.cfg.guidance.n_samples_per_test_molecule, 1).to(
                data.guidance.device)
            num_nodes = torch.nn.functional.softmax(self.node_model.forward_data(input_guidance), dim=-1)
            print("num nodes shape:", num_nodes.size())

            if (self.cfg.guidance.node_inference_method == "sample"):
                num_nodes = torch.tensor([Categorical(num_nodes[i, ...]).sample().item()
                                          for i in range(num_nodes.shape[0])]).to(data.guidance.device)
            else:
                num_nodes = torch.argmax(num_nodes, dim=-1).to(data.guidance.device)
            print("num_nodes", num_nodes)

        data.guidance = torch.zeros(data.guidance.shape[0], 0).type_as(data.guidance)

        start = time.time()

        ident = 0
        samples = self.sample_batch(batch_id=ident,
                                    batch_size=self.cfg.guidance.n_samples_per_test_molecule,
                                    num_nodes=num_nodes,
                                    save_final=10,
                                    keep_chain=1,
                                    number_chain_steps=self.number_chain_steps,
                                    guidance=target_properties)
        print(f'Sampling took {time.time() - start:.2f} seconds\n')

        # save conditional generated samples
        mae = self.cond_sample_metric(samples, target_properties)

        # Accumulate per-class accuracy for NCI1
        if dataset_name == 'nci1':
            target_cls = target_properties.argmax(dim=-1).item()
            n_samples = self.cfg.guidance.n_samples_per_test_molecule
            correct_count = int(round(mae.item() * n_samples))  # mae is accuracy for nci1
            self.nci1_class_correct[target_cls] += correct_count
            self.nci1_class_total[target_cls] += n_samples

        print("==============================================================")
        return {'mae': mae}

    def on_test_epoch_end(self) -> None:
        dataset_name = self.cfg.dataset.name

        if dataset_name == 'nci1':
            total = sum(self.nci1_class_total.values())
            correct = sum(self.nci1_class_correct.values())
            final_accuracy = correct / total if total > 0 else 0.0
            cls_names = {0: 'Class-0', 1: 'Class-1'}

            print(f"\n{'='*60}")
            print(f"NCI1 CONDITIONAL GENERATION RESULTS")
            print(f"{'='*60}")
            for cls in (0, 1):
                tot = self.nci1_class_total[cls]
                cor = self.nci1_class_correct[cls]
                acc = cor / tot if tot > 0 else 0.0
                print(f"  {cls_names[cls]:>12s}: {acc:.2%}  ({cor}/{tot})")
            print(f"  {'Overall':>12s}: {final_accuracy:.2%}  ({correct}/{total})")
            print(f"{'='*60}\n")

            if wandb.run:
                wandb.run.summary['final_accuracy'] = final_accuracy
                wandb.log({'final accuracy': final_accuracy})
            return

        # Conditional-generation diagnostic summary (valid SMILES / connectivity / fragments)
        if getattr(self, '_condgen_stats', None):
            class_names = self._class_names()
            print(f"\n{'='*60}")
            print("CONDITIONAL GENERATION — SMILES / CONNECTIVITY / FRAGMENTS")
            print(f"{'='*60}")
            for cls in sorted(self._condgen_stats.keys()):
                flags = self._condgen_stats[cls]
                n = len(flags)
                n_val = sum(1 for v, _, _ in flags if v)
                n_con = sum(1 for _, c, _ in flags if c)
                fr = [nf for v, _, nf in flags if v]
                avg_frag = (sum(fr) / len(fr)) if fr else float('nan')
                con_over_val = (n_con / n_val) if n_val else float('nan')
                print(f"  [{class_names.get(cls, cls)}]  n={n}")
                print(f"    valid SMILES:       {n_val}/{n} = {n_val/n:.3f}")
                print(f"    connected (1 frag): {n_con}/{n} = {n_con/n:.3f}")
                print(f"    connected | valid:  {n_con}/{max(n_val,1)} = {con_over_val:.3f}")
                print(f"    avg #fragments (over valid): {avg_frag:.2f}")
            print(f"{'='*60}")
            print("CONDGEN_DONE")

        final_mae = self.cond_val.compute()
        final_validity = self.num_valid_molecules / self.num_total if self.num_total > 0 else 0.0
        print("Final MAE", final_mae)
        print("Final validity", final_validity * 100)

        if wandb.run:
            wandb.run.summary['final_MAE'] = final_mae
            wandb.run.summary['final_validity'] = final_validity
            wandb.log({'final mae': final_mae,
                       'final validity': final_validity})

    def _class_names(self):
        """Human-readable label names for logging, per dataset."""
        return {
            'nci1': {0: 'Inactive', 1: 'Active'},
            'nci1_nofreq': {0: 'Inactive', 1: 'Active'},
            'aids': {0: 'Inactive', 1: 'Active'},
            'benzene': {0: 'No ring', 1: 'Benzene ring'},
            'spmotif': {0: 'Cycle', 1: 'Crane'},
        }.get(self.cfg.dataset.name, {0: 'Non-mutagenic', 1: 'Mutagenic'})

    def cond_sample_metric(self, samples, input_properties):
        # Route to appropriate metric based on dataset
        dataset_name = self.cfg.dataset.name
        if dataset_name == 'planar':
            return self.accuracy_test_planar(samples, input_properties)
        elif dataset_name in ('nci1', 'nci1_nofreq', 'spmotif', 'proteins', 'imdb_binary', 'twitter'):
            # Same representation as NCI1 (binary adjacency, categorical node types,
            # LEConv judge), so the non-molecular routine applies unchanged. The
            # default branch below would call build_molecule() and needs an
            # atom_decoder that these datasets do not have.
            return self.accuracy_test_nci1(samples, input_properties)
        elif dataset_name in ('mutagenicity_mol', 'benzene'):
            # Benzene shares the representation (5-class bonds) and the classifier
            # class (MutagGINELightning) with mutagenicity_mol, and the routine reads
            # the atom_decoder from cfg.dataset.datadir, so it is dataset-agnostic.
            return self.accuracy_test_mutag_mol(samples, input_properties)
        else:
            return self.accuracy_test(samples, input_properties)

    def accuracy_test_planar(self, samples, input_properties):
        """
        Evaluate adjacency matrix samples by counting edges, normalizing between 7 and 18,
        and comparing to input_properties using MAE.

        Args:
            samples (List of [node_types, adj_matrix]): Each sample has a tensor for node labels and adjacency matrix.
            input_properties (Tensor): Target property tensor to compare against. Shape: [1, 1] or [bs, 1]

        Returns:
            mae (float): Mean Absolute Error between predicted normalized edge counts and target.
        """
        edge_counts = []
        for sample in samples:
            adj = sample[1]  # Adjacency matrix
            adj = adj.cpu() if adj.is_cuda else adj
            adj = adj.int()  # Make sure it's integer for edge count

            # Remove diagonal (no self loops), count edges, divide by 2 since symmetric
            edge_count = (adj.sum() - torch.diag(adj).sum()).item() // 2
            edge_counts.append(edge_count)

        # Normalize edge count to range [0, 1] using known range 7-18
        edge_counts_tensor = torch.tensor(edge_counts).float().unsqueeze(1)
        normalized_edges = (edge_counts_tensor - 7) / (18 - 7)

        target_tensor = input_properties.repeat(normalized_edges.size(0), 1).cpu()

        mae_metric = MeanAbsoluteError()
        mae = mae_metric(normalized_edges, target_tensor)

        return mae

    def accuracy_test_nci1(self, samples, input_properties):
        """
        Evaluate generated graph samples for NCI1 using a pre-trained NCI1GCNLightning classifier.
        Same logic as accuracy_test_ba3motif but loads NCI1GCNLightning (LEConv-based).
        """
        if self.graph_classifier is None:
            classifier_path = getattr(self.cfg.guidance, 'graph_classifier_path', None)
            print(f"Graph classifier path from config: {classifier_path}")
            if classifier_path is None:
                print("Warning: graph_classifier_path not set. Cannot compute classification accuracy.")
                return torch.tensor(0.0)

            if not str(classifier_path).endswith('.ckpt'):
                # A shared-classifier export (`<prefix>.npz` + `<prefix>.json`)
                # instead of a Lightning checkpoint -- the same classifier the
                # baselines attack. The adapter exposes the same interface
                # (forward(batch), hparams.in_dim/out_dim).
                from src.models.shared_judge_adapter import load_shared_judge
                self._graph_classifier_ref[0] = load_shared_judge(str(classifier_path))
            else:
                try:
                    self._graph_classifier_ref[0] = NCI1GCNLightning.load_from_checkpoint(classifier_path)
                    print(f"Loaded NCI1 classifier from {classifier_path}")
                    print(f"Model hyperparameters: in_dim={self.graph_classifier.hparams.in_dim}, "
                          f"hidden_dim={self.graph_classifier.hparams.hidden_dim}, "
                          f"out_dim={self.graph_classifier.hparams.out_dim}")
                except TypeError as e:
                    print(f"Loading with default parameters due to: {e}")
                    self._graph_classifier_ref[0] = NCI1GCNLightning.load_from_checkpoint(
                        classifier_path,
                        in_dim=10,
                        hidden_dim=128,
                        out_dim=2,
                    )
            self.graph_classifier.to(self.device)
            self.graph_classifier.eval()

        classifier_in_dim = self.graph_classifier.hparams.in_dim

        data_list = []
        for sample in samples:
            node_types = sample[0]
            adj = sample[1]

            node_types = node_types.cpu() if node_types.is_cuda else node_types
            adj = adj.cpu() if adj.is_cuda else adj

            if node_types.dim() == 1:
                if classifier_in_dim == 1:
                    x = node_types.float().unsqueeze(-1)
                else:
                    x = F.one_hot(node_types.long(), num_classes=classifier_in_dim).float()
            else:
                if node_types.size(-1) == classifier_in_dim:
                    x = node_types.float()
                else:
                    class_indices = node_types.argmax(dim=-1)
                    x = F.one_hot(class_indices.long(), num_classes=classifier_in_dim).float()

            edge_index = adj.nonzero(as_tuple=False).t().contiguous()
            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)

        batch = Batch.from_data_list(data_list)
        batch = batch.to(self.device)

        with torch.no_grad():
            logits = self.graph_classifier(batch)
            predictions = logits.argmax(dim=-1)

        target_class = input_properties.argmax(dim=-1).item()
        targets = torch.full((len(samples),), target_class, dtype=torch.long, device=self.device)

        correct = (predictions == targets).float()
        accuracy = correct.mean()

        print(f"Predictions: {predictions.tolist()}")
        print(f"Target class: {target_class}")
        print(f"Accuracy: {accuracy.item():.2%} ({correct.sum().int().item()}/{len(samples)})")

        return accuracy

    def accuracy_test_mutag(self, samples, input_properties):
        """Evaluate generated graph samples for Mutagenicity using a pretrained GIN/GCN classifier."""
        if self.graph_classifier is None:
            classifier_path = getattr(self.cfg.guidance, 'graph_classifier_path', None)
            if classifier_path is None:
                print("Warning: graph_classifier_path not set. Cannot compute Mutagenicity accuracy.")
                return torch.tensor(0.0)
            try:
                _ckpt = torch.load(classifier_path, map_location='cpu')
                _keys = set(_ckpt.get('state_dict', {}).keys())
                if any('.nn.' in k for k in _keys):
                    from src.models.Mutag_GIN import MutagGINLightning
                    self._graph_classifier_ref[0] = MutagGINLightning.load_from_checkpoint(classifier_path)
                else:
                    from src.models.Mutag_GCN import MutagGCNLightning
                    self._graph_classifier_ref[0] = MutagGCNLightning.load_from_checkpoint(classifier_path)
                self.graph_classifier.to(self.device)
                self.graph_classifier.eval()
                print(f"Loaded Mutagenicity classifier from {classifier_path}")
            except Exception as e:
                print(f"Failed to load Mutagenicity classifier: {e}")
                return torch.tensor(0.0)

        classifier_in_dim = self.graph_classifier.hparams.in_dim

        data_list = []
        for sample in samples:
            node_types = sample[0]
            adj = sample[1]
            node_types = node_types.cpu() if node_types.is_cuda else node_types
            adj = adj.cpu() if adj.is_cuda else adj

            if node_types.dim() == 1:
                x = F.one_hot(node_types.long(), num_classes=classifier_in_dim).float()
            else:
                if node_types.size(-1) == classifier_in_dim:
                    x = node_types.float()
                else:
                    x = F.one_hot(node_types.argmax(dim=-1).long(), num_classes=classifier_in_dim).float()

            edge_index = adj.nonzero(as_tuple=False).t().contiguous()
            data_list.append(Data(x=x, edge_index=edge_index))

        batch = Batch.from_data_list(data_list).to(self.device)
        with torch.no_grad():
            predictions = self.graph_classifier(batch).argmax(dim=-1)

        target_class = input_properties.argmax(dim=-1).item()
        targets = torch.full((len(samples),), target_class, dtype=torch.long, device=self.device)
        correct = (predictions == targets).float()
        accuracy = correct.mean()

        class_names = self._class_names()
        print(f"Predictions: {[class_names[p] for p in predictions.tolist()]}")
        print(f"Target: {class_names[target_class]}, Accuracy: {accuracy.item():.2%}")
        return accuracy

    def accuracy_test_mutag_mol(self, samples, input_properties):
        """Evaluate generated graph samples for Mutagenicity (bond types) using GINEConv classifier."""
        if self.graph_classifier is None:
            classifier_path = getattr(self.cfg.guidance, 'graph_classifier_path', None)
            if classifier_path is None:
                print("Warning: graph_classifier_path not set. Cannot compute Mutagenicity_mol accuracy.")
                return torch.tensor(0.0)
            try:
                from src.models.Mutag_GINE import MutagGINELightning
                self._graph_classifier_ref[0] = MutagGINELightning.load_from_checkpoint(classifier_path)
                self.graph_classifier.to(self.device)
                self.graph_classifier.eval()
                print(f"Loaded Mutagenicity_mol GINEConv classifier from {classifier_path}")
            except Exception as e:
                print(f"Failed to load Mutagenicity_mol classifier: {e}")
                return torch.tensor(0.0)

        classifier_in_dim = self.graph_classifier.hparams.in_dim

        # Lazy-load SMILES helpers + per-class accumulator for the conditional-
        # generation diagnostic (valid SMILES / connectivity / #fragments).
        if not hasattr(self, '_condgen_smiles_fn'):
            from inversion_mutag_mol import graph_to_smiles, load_atom_decoder
            self._condgen_smiles_fn = graph_to_smiles
            self._condgen_atom_decoder = load_atom_decoder(self.cfg)
            self._condgen_stats = {}  # target_class -> list of (valid, connected, n_frag)

        data_list = []
        smiles_flags = []  # per-sample (valid, connected, n_frag)
        for sample in samples:
            node_types = sample[0]
            E_dense = sample[1]  # (n_nodes, n_nodes) or (n_nodes, n_nodes, 5)
            node_types = node_types.cpu() if node_types.is_cuda else node_types
            E_dense = E_dense.cpu() if E_dense.is_cuda else E_dense

            n_nodes = node_types.size(0)

            if node_types.dim() == 1:
                x = F.one_hot(node_types.long(), num_classes=classifier_in_dim).float()
                atom_idx = node_types.long()
            else:
                if node_types.size(-1) == classifier_in_dim:
                    x = node_types.float()
                else:
                    x = F.one_hot(node_types.argmax(dim=-1).long(), num_classes=classifier_in_dim).float()
                atom_idx = node_types.argmax(dim=-1).long()

            # Extract edge_index + edge_attr from 5-class dense E tensor
            if E_dense.dim() == 3:
                et_matrix = E_dense[:n_nodes, :n_nodes].argmax(dim=-1)
            else:
                et_matrix = E_dense[:n_nodes, :n_nodes].long()
            et_matrix.fill_diagonal_(0)  # no self-loops

            edge_index = (et_matrix > 0).nonzero(as_tuple=False).t().contiguous()
            if edge_index.numel() > 0:
                et_vals = et_matrix[edge_index[0], edge_index[1]]
                edge_attr = F.one_hot(et_vals, num_classes=5).float()
            else:
                edge_attr = torch.zeros((0, 5), dtype=torch.float32)

            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

            # SMILES / connectivity / fragment count (strict RDKit sanitization)
            if self._condgen_atom_decoder is not None:
                smi = self._condgen_smiles_fn(
                    atom_idx, edge_index, edge_attr, self._condgen_atom_decoder, strict=True)
                if smi is not None:
                    n_frag = smi.count('.') + 1
                    smiles_flags.append((True, n_frag == 1, n_frag))
                else:
                    smiles_flags.append((False, False, -1))
            else:
                smiles_flags.append((False, False, -1))

        batch = Batch.from_data_list(data_list).to(self.device)
        with torch.no_grad():
            predictions = self.graph_classifier(batch).argmax(dim=-1)

        target_class = input_properties.argmax(dim=-1).item()
        targets = torch.full((len(samples),), target_class, dtype=torch.long, device=self.device)
        correct = (predictions == targets).float()
        accuracy = correct.mean()

        self._condgen_stats.setdefault(target_class, []).extend(smiles_flags)

        class_names = self._class_names()
        print(f"Predictions: {[class_names[p] for p in predictions.tolist()]}")
        print(f"Target: {class_names[target_class]}, Accuracy: {accuracy.item():.2%}")
        n_val = sum(1 for v, _, _ in smiles_flags if v)
        n_con = sum(1 for _, c, _ in smiles_flags if c)
        print(f"  valid SMILES: {n_val}/{len(smiles_flags)}  "
              f"connected: {n_con}/{len(smiles_flags)}")
        return accuracy

    def accuracy_test_imdb(self, samples, input_properties):
        """Evaluate generated graphs for IMDB using the underlying degree rule.

        The IMDB label is determined by normalised degree:
            deg  = (adj.sum() - n) / (2 * max_num_nodes)
            label = 1  if deg >= deg_mean  else 0

        This avoids relying on a separate classifier and directly tests whether
        the diffusion model has learned the conditional distribution.
        """
        # Lazily compute the degree threshold from training data
        if self.dataset_info.imdb_deg_mean is None:
            self.dataset_info.compute_imdb_deg_mean(self.trainer.datamodule)

        if input_properties.size(-1) == 1:
            target_class = int(input_properties.item() >= 0.5)
        else:
            target_class = input_properties.argmax(dim=-1).item()
        cls_names = {0: 'Low-Degree', 1: 'High-Degree'}

        predictions = []
        degrees = []
        edge_counts = []

        for sample in samples:
            adj = sample[1].cpu()  # (n_nodes, n_nodes)
            n = adj.size(0)

            # Reconstruct the same degree metric used during dataset creation
            # adj includes self-loops from the diffusion output (diagonal may be edge class 1)
            adj_binary = (adj > 0).float()
            # Ensure self-loops are present (matching dataset creation: adj + eye)
            adj_binary.fill_diagonal_(1.0)

            deg = (adj_binary.sum().item() - n) / (2 * self.dataset_info.imdb_max_num_nodes)
            pred = 1 if deg >= self.dataset_info.imdb_deg_mean else 0

            predictions.append(pred)
            degrees.append(deg)

            # Edge count excluding self-loops
            off_diag = adj_binary.clone()
            off_diag.fill_diagonal_(0)
            edge_counts.append(int(off_diag.sum().item()) // 2)

        predictions_t = torch.tensor(predictions, device=self.device)
        targets = torch.full((len(samples),), target_class, dtype=torch.long,
                             device=self.device)
        correct = (predictions_t == targets).float()
        accuracy = correct.mean()

        # Reuse counters: valid = correct predictions, total = all samples
        n_correct = int(correct.sum().item())
        self.num_valid_molecules += n_correct
        self.num_total += len(samples)
        self.imdb_class_correct[target_class] += n_correct
        self.imdb_class_total[target_class] += len(samples)

        print(f"  Target class: {target_class} ({cls_names.get(target_class, '?')})")
        print(f"  Deg threshold: {self.dataset_info.imdb_deg_mean:.4f}")
        print(f"  Sample degrees: {[f'{d:.4f}' for d in degrees]}")
        print(f"  Predictions:    {predictions}")
        print(f"  Edge counts:    {edge_counts}")
        print(f"  Accuracy: {accuracy.item():.2%} ({n_correct}/{len(samples)})")

        return accuracy

    def accuracy_test(self, samples, input_properties):
        valid_properties = self.cfg.guidance.guidance_target
        test_thresholds = self.cfg.guidance.test_thresholds

        if ('mu' in valid_properties or 'homo' in valid_properties):
            try:
                import psi4
                # Hardware side settings (CPU thread number and memory settings used for calculation)
                psi4.set_num_threads(nthread=4)
                psi4.set_memory("5GB")
                psi4.core.set_output_file('psi4_output.dat', False)
            except ModuleNotFoundError:
                print("PSI4 not found")

        print("valid_properties", valid_properties)

        numerical_results_dict = {}
        binary_results_dict = {}
        for tgt in valid_properties:
            numerical_results_dict[tgt] = []
            binary_results_dict[tgt] = []

        split_molecules = 0
        sample_smiles = []
        idxs = []
        mus = []

        for idx, sample in enumerate(samples):
            if (self.cfg.guidance.build_with_partial_charges == "old_method"):
                raw_mol = build_molecule_with_partial_charges(sample[0], sample[1], self.dataset_info.atom_decoder)
            else:
                raw_mol = build_molecule(sample[0], sample[1], self.dataset_info.atom_decoder)

            mol = Chem.rdchem.RWMol(raw_mol)

            try:
                Chem.SanitizeMol(mol)
            except:
                print('invalid chemistry')
                continue

            ring_info = mol.GetRingInfo()
            ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
            has_3ring = int(3 in ring_sizes)
            has_4ring = int(4 in ring_sizes)
            has_5ring = int(5 in ring_sizes)
            has_6ring = int(6 in ring_sizes)
            print('Has Ring', has_3ring, has_4ring, has_5ring, has_6ring)

            # Coarse 3D structure optimization by generating 3D structure from SMILES
            mol = Chem.AddHs(mol)
            params = ETKDGv3()
            params.randomSeed = 1
            try:
                EmbedMolecule(mol, params)
            except Exception as e:
                print(f"EmbedMolecule failed for sample {idx}: {e}")
                continue

            # Structural optimization with MMFF (Merck Molecular Force Field)
            try:
                s = MMFFOptimizeMolecule(mol)
            except:
                print('Bad conformer ID')
                continue

            try:
                conf = mol.GetConformer()
            except:
                print("GetConformer failed")
                continue

            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
            if (len(mol_frags) > 1):
                split_molecules = split_molecules + 1
                if (self.cfg.guidance.include_split == False):
                    print("Ignoring a split molecule")
                    continue

            ##########################################################
            ##########################################################

            if ('mu' in valid_properties or 'homo' in valid_properties):
                # Convert to a format that can be input to Psi4.
                # Set charge and spin multiplicity (below is charge 0, spin multiplicity 1)

                # Get the formal charge
                fc = 'FormalCharge'
                mol_FormalCharge = int(mol.GetProp(fc)) if mol.HasProp(fc) else Chem.GetFormalCharge(mol)

                sm = 'SpinMultiplicity'
                if mol.HasProp(sm):
                    mol_spin_multiplicity = int(mol.GetProp(sm))
                else:
                    # Calculate spin multiplicity using Hund's rule of maximum multiplicity...
                    NumRadicalElectrons = 0
                    for Atom in mol.GetAtoms():
                        NumRadicalElectrons += Atom.GetNumRadicalElectrons()
                    TotalElectronicSpin = NumRadicalElectrons / 2
                    SpinMultiplicity = 2 * TotalElectronicSpin + 1
                    mol_spin_multiplicity = int(SpinMultiplicity)

                mol_input = "%s %s" % (mol_FormalCharge, mol_spin_multiplicity)
                print(mol_input)
                # mol_input = "0 1"

                # Describe the coordinates of each atom in XYZ format
                for atom in mol.GetAtoms():
                    mol_input += "\n " + atom.GetSymbol() + " " + str(conf.GetAtomPosition(atom.GetIdx()).x) \
                                 + " " + str(conf.GetAtomPosition(atom.GetIdx()).y) \
                                 + " " + str(conf.GetAtomPosition(atom.GetIdx()).z)

                try:
                    molecule = psi4.geometry(mol_input)
                except:
                    print('Can not calculate psi4 geometry')
                    continue

                # Convert to a format that can be input to pyscf
                # Set calculation method (functional) and basis set
                level = "b3lyp/6-31G*"

                # Calculation method (functional), example of basis set
                # theory = ['hf', 'b3lyp']
                # basis_set = ['sto-3g', '3-21G', '6-31G(d)', '6-31+G(d,p)', '6-311++G(2d,p)']

                # Perform structural optimization calculations
                print('Psi4 calculation starts!!!')
                # energy, wave_function = psi4.optimize(level, molecule=molecule, return_wfn=True)
                try:
                    with time_limit(3600):
                        energy, wave_function = psi4.energy(level, molecule=molecule, return_wfn=True)
                except:
                    print("Psi4 did not converge")
                    continue

                print('Chemistry information check!!!')

            ##########################################################
            ##########################################################
            try:
                mol = raw_mol
                mol = clean_mol(mol, (self.cfg.guidance.build_with_partial_charges == "no"))

                smile = Chem.MolToSmiles(mol)
                print("Generated SMILES ", smile)
            except:
                print("clean_mol failed")
                continue

            sample_smiles.append(smile)
            self.generated_smiles.append(smile)

            if 'mu' in valid_properties:
                dip_x, dip_y, dip_z = wave_function.variable('SCF DIPOLE')[0], \
                    wave_function.variable('SCF DIPOLE')[1], \
                    wave_function.variable('SCF DIPOLE')[2]
                dipole_moment = math.sqrt(dip_x ** 2 + dip_y ** 2 + dip_z ** 2) * 2.5417464519
                print("Dipole moment", dipole_moment)

                thresholds = test_thresholds['mu']
                if (thresholds[0] <= dipole_moment and dipole_moment <= thresholds[1]):
                    binary_results_dict['mu'].append(1)
                else:
                    binary_results_dict['mu'].append(0)
                numerical_results_dict['mu'].append(dipole_moment)
                idxs.append(idx)

            if 'homo' in valid_properties:
                # Compute HOMO (Unit: au= Hartree）
                LUMO_idx = wave_function.nalpha()
                HOMO_idx = LUMO_idx - 1
                homo = wave_function.epsilon_a_subset("AO", "ALL").np[HOMO_idx]

                # convert unit from a.u. to ev
                homo = homo * 27.211324570273

                thresholds = test_thresholds['homo']
                if (thresholds[0] <= homo and homo <= thresholds[1]):
                    binary_results_dict['homo'].append(1)
                else:
                    binary_results_dict['homo'].append(0)
                numerical_results_dict['homo'].append(homo)
                idxs.append(idx)

            if 'penalizedlogp' in valid_properties:
                plogp_estimate = penalized_logp(mol)

                thresholds = test_thresholds['penalizedlogp']
                if (thresholds[0] <= plogp_estimate and plogp_estimate <= thresholds[1]):
                    binary_results_dict['penalizedlogp'].append(1)
                else:
                    binary_results_dict['penalizedlogp'].append(0)
                numerical_results_dict['penalizedlogp'].append(plogp_estimate)

            if 'qed' in valid_properties:
                qed_estimate = qed(mol)

                thresholds = test_thresholds['qed']
                if (thresholds[0] <= qed_estimate and qed_estimate <= thresholds[1]):
                    binary_results_dict['qed'].append(1)
                else:
                    binary_results_dict['qed'].append(0)
                numerical_results_dict['qed'].append(qed_estimate)

            if 'mw' in valid_properties:
                mw_estimate = mw(mol) / 100

                thresholds = test_thresholds['mw']
                if (thresholds[0] <= mw_estimate and mw_estimate <= thresholds[1]):
                    binary_results_dict['mw'].append(1)
                else:
                    binary_results_dict['mw'].append(0)
                numerical_results_dict['mw'].append(mw_estimate)
                idxs.append(idx)

            if 'sas' in valid_properties:
                sas_estimate = calculateScore(mol)

                thresholds = test_thresholds['sas']
                if (thresholds[0] <= sas_estimate and sas_estimate <= thresholds[1]):
                    binary_results_dict['sas'].append(1)
                else:
                    binary_results_dict['sas'].append(0)
                numerical_results_dict['sas'].append(sas_estimate)

            if 'logp' in valid_properties:
                logp_estimate = Crippen.MolLogP(mol)

                thresholds = test_thresholds['logp']
                if (thresholds[0] <= logp_estimate and logp_estimate <= thresholds[1]):
                    binary_results_dict['logp'].append(1)
                else:
                    binary_results_dict['logp'].append(0)
                numerical_results_dict['logp'].append(logp_estimate)

            if 'has_3ring' in valid_properties:
                binary_results_dict['has_3ring'].append(1)
                numerical_results_dict['has_3ring'].append(float(has_3ring))

            if 'has_4ring' in valid_properties:
                binary_results_dict['has_4ring'].append(1)
                numerical_results_dict['has_4ring'].append(float(has_4ring))

            if 'has_5ring' in valid_properties:
                binary_results_dict['has_5ring'].append(1)
                numerical_results_dict['has_5ring'].append(float(has_5ring))

            if 'has_6ring' in valid_properties:
                binary_results_dict['has_6ring'].append(1)
                numerical_results_dict['has_6ring'].append(float(has_6ring))
                idxs.append(idx)

        num_valid_molecules = 0
        outputs = None
        binary_outputs = None

        log_dict = {}

        i = 0
        for tgt in ['mu', 'homo', 'penalizedlogp', 'qed', 'mw', 'sas', 'logp', 'has_3ring', 'has_4ring', 'has_5ring',
                    'has_6ring']:

            if tgt in valid_properties:
                num_valid_molecules = max(num_valid_molecules, len(numerical_results_dict[tgt]))

                curr_numerical = torch.FloatTensor(numerical_results_dict[tgt]).unsqueeze(1)
                curr_binary = torch.FloatTensor(binary_results_dict[tgt]).unsqueeze(1)

                if (outputs == None):
                    outputs = curr_numerical
                    binary_outputs = curr_binary
                else:
                    outputs = torch.hstack((outputs, curr_numerical))
                    binary_outputs = torch.hstack((binary_outputs, curr_binary))

                if (len(valid_properties) > 1):
                    curr_tgt_key = "val_epoch/" + tgt + "_mae"
                    curr_tgt_vec = input_properties[..., i].repeat(curr_numerical.size(0), 1).cpu()

                    curr_tgt_mae = MeanAbsoluteError()(curr_numerical, curr_tgt_vec)
                    log_dict.update({curr_tgt_key: curr_tgt_mae})

                    # Clean per-property summary: conditioned target vs. what was generated.
                    tgt_val = float(input_properties[..., i].flatten()[0])
                    gen_mean = float(curr_numerical.mean())
                    print(f"  [{tgt}] target={tgt_val:.2f}  generated_mean={gen_mean:.3f}  "
                          f"(MAE={float(curr_tgt_mae):.3f}, n={curr_numerical.size(0)})")

                i = i + 1

        print("Number of valid samples", num_valid_molecules)
        self.num_valid_molecules += num_valid_molecules
        self.num_total += len(samples)

        # we can recycle tgt for outputs[tgt]

        target_tensor = input_properties.repeat(outputs.size(0), 1).cpu()

        mae = self.cond_val(outputs, target_tensor)
        accuracy = torch.mean(torch.count_nonzero(binary_outputs).type(torch.FloatTensor))

        unique_smiles = set(sample_smiles)
        n_unique_smiles = len(unique_smiles)
        if (len(sample_smiles) != 0):
            n_unique_smiles_percentage = n_unique_smiles / len(sample_smiles)
        else:
            n_unique_smiles_percentage = 0

        print(f"Conditional generation metric: MAE={float(mae):.4f}  "
              f"valid={num_valid_molecules}  unique={n_unique_smiles_percentage:.1%}")

        # log_dict.update({"val_epoch/conditional generation mae": mae,
        #                 'Valid molecules'                     : num_valid_molecules,
        #                 'Valid molecules splitted'            : split_molecules,
        #                 "val_epoch/n_unique_smiles"           : n_unique_smiles,
        #                 "val_epoch/n_unique_smiles_percentage": n_unique_smiles_percentage,
        #                 })

        # if(self.cfg.guidance.experiment_type == 'accuracy'):
        # log_dict.update({"val_epoch/accuracy": accuracy})

        # wandb.log(log_dict)

        return mae, idxs, outputs

    def kl_prior(self, X, E, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # Make sure that masked rows do not contribute to the loss
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)

        probX[probX == .0] = 1e-6
        probE[probE == .0] = 1e-6
        limit_dist_X[limit_dist_X == .0] = 1e-6
        limit_dist_E[limit_dist_E == .0] = 1e-6

        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')
        return diffusion_utils.sum_except_batch(kl_distance_X) + \
            diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        # Note: these are not outputs from the NN => we need softmax regardless
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)

        prob_true.X[prob_true.X == 0.0] = 1e-6
        prob_pred.X[prob_pred.X == 0.0] = 1e-6
        prob_true.E[prob_true.E == 0.0] = 1e-6
        prob_pred.E[prob_pred.E == 0.0] = 1e-6

        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, node_mask, guidance=None):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask, guidance=guidance)

        # Normalize predictions
        if (self.cfg.guidance.loss == 'crossentropy'):
            probX0 = F.softmax(pred0.X, dim=-1)
            probE0 = F.softmax(pred0.E, dim=-1)
            proby0 = F.softmax(pred0.y, dim=-1)
        else:
            probX0 = torch.exp(pred0.X)
            probE0 = torch.exp(pred0.E)
            proby0 = torch.exp(pred0.y)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        lowest_t = 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)  # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar,
                                               device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False, guidance=None):
        """Computes an estimator for the variational lower bound.
           pred: (batch_size, n, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """
        t = noisy_data['t']

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Uniform(1/num_classes). Should be close to zero.
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. Diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. Reconstruction loss
        # Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        prob0 = self.reconstruction_logp(t, X, E, node_mask, guidance)

        loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log())

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll
        nll = (self.test_nll if test else self.val_nll)(nlls)  # Average over the batch

        if wandb.run:
            wandb.log({"kl prior": kl_prior.mean(),
                       "Estimator loss terms": loss_all_t.mean(),
                       "log_pn": log_pN.mean(),
                       "loss_term_0": loss_term_0,
                       'batch_test_nll' if test else 'val_nll': nll}, commit=False)
        return nll

    # effectively returns p(G_0|G_t, y)
    def forward(self, noisy_data, extra_data, node_mask, guidance=None, train_step=False):
        bs = extra_data.X.size(0)

        # replicates the null guidance token for the whole batch size
        cf_null_token = self.cf_null_token.repeat((bs, 1))
        # if we don't pass a guidance vector, we use the null guidance token:
        if (guidance is None):
            guidance = cf_null_token
        elif (
                train_step and self.cfg.guidance.p_uncond > 0):  # it has sense only if the guidance is not ENTIRELY equal to cf_null_token
            # proceeds to randomly mask the guidance
            # TODO: this likely will have to be fixed for text-based where
            # the null token is inplanted in the query itself (it's the "pad" token)
            # and hence all of this is already done implicitly

            # True = substitute with null token
            guidance_mask = torch.rand((bs, 1), device="cuda").to(guidance.device) < self.cfg.guidance.p_uncond

            # now it has the same size as cf_null_token
            guidance_mask = guidance_mask.repeat((1, cf_null_token.size(-1)))

            guidance = torch.where(guidance_mask, cf_null_token, guidance)

        # this is used during test when we sample n_samples_per_test_molecule
        # at once using the same guidance (=> the guidance has shape: (1, guidance_size)).
        # we need to repeat it n_samples_per_test_molecule (=> shape: (n_samples_per_test_molecule, guidance_size))
        # the "not train_step" is not necessary but better have it
        if (bs > guidance.size(0) and not train_step):
            guidance = guidance.repeat((bs, 1))

        # we put the production of X, E, y in a method as we need to
        # recycle the whole code later on if we want to calculate
        # the same with the null token in place of the guidance.
        def produce_XEy(noisy_data, extra_data, guidance=None):
            # X      = transformer output which uses guidance (if any)
            # X_null = transformer output which uses the null token
            X = noisy_data['X_t'].clone().float()
            E = noisy_data['E_t'].clone().float()
            y = noisy_data['y_t'].clone().float()

            if (guidance != None):
                if (self.cfg.guidance.guidance_medium in ['y', 'both']):
                    g_y = guidance

                    # adds the guidance
                    y = torch.hstack((y, g_y)).float()

                if (self.cfg.guidance.guidance_medium in ['XE', 'both']):
                    n = extra_data.X.size(1)  # number of nodes

                    # spreads the guidance on all noses
                    # g_X must have size (bs, n, features)
                    # g_E must have size (bs, n, n, features)
                    g_X = torch.reshape(guidance, shape=(bs, 1, -1)).repeat((1, n, 1))
                    g_E = torch.reshape(guidance, shape=(bs, 1, 1, -1)).repeat((1, n, n, 1))

                    # adds the guidance
                    X = torch.cat((X, g_X), dim=-1)
                    E = torch.cat((E, g_E), dim=-1)

            # we finally add the extra data as requested
            X = torch.cat((X, extra_data.X), dim=2).float()
            E = torch.cat((E, extra_data.E), dim=3).float()
            y = torch.hstack((y, extra_data.y)).float()

            return X, E, y

        X, E, y = produce_XEy(noisy_data, extra_data, guidance)

        # p(x_0|x_t, guidance)
        out = self.model(X, E, y, node_mask)

        if (self.cfg.guidance.loss == 'crossentropy'):
            if (train_step == False and self.cfg.guidance.s > 0):
                # first of all, we need to calculate p(x_0|x_t, None) as well:
                X_null, E_null, y_null = produce_XEy(noisy_data, extra_data, cf_null_token)
                out_null = self.model(X_null, E_null, y_null, node_mask)

                out.X = out_null.X + self.cfg.guidance.s * (out.X - out_null.X)
                out.E = out_null.E + self.cfg.guidance.s * (out.E - out_null.E)

            # if the loss is a crossentropy, we are done here.
            # the raw outputs are the only thing we need.
            return utils.PlaceHolder(X=out.X, E=out.E, y=out.y).mask(node_mask)
        else:
            raise NotImplementedError("ERROR: unimplemented loss")

    @torch.no_grad()
    def _sample_n_nodes(self, batch_size, guidance):
        """Draw graph sizes, conditioned on the guidance class when possible.

        The global size prior is dominated by the majority class. If the classes
        differ in size (AIDS: class 0 ~28.5 nodes vs class 1 ~10.2), sampling sizes
        from that prior makes conditional generation of the larger/rarer class
        impossible -- the size is fixed before denoising starts, so no guidance
        scale can recover it.
        """
        per_class = self.nodes_dist_per_class
        if per_class is None or guidance is None or guidance.numel() == 0:
            return self.node_dist.sample_n(batch_size, self.device)

        g = guidance if guidance.dim() == 2 else guidance.unsqueeze(0)
        cls = g.argmax(dim=-1)

        if cls.numel() == 1:
            c = int(cls.item())
            dist = per_class[c] if 0 <= c < len(per_class) else self.node_dist
            return dist.sample_n(batch_size, self.device)

        out = torch.empty(batch_size, dtype=torch.long, device=self.device)
        for i in range(batch_size):
            c = int(cls[i % cls.size(0)].item())
            dist = per_class[c] if 0 <= c < len(per_class) else self.node_dist
            out[i] = dist.sample_n(1, self.device)[0]
        return out

    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None, guidance=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self._sample_n_nodes(batch_size, guidance)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y

        if (guidance == None):
            guidance = self.cf_null_token.repeat((batch_size, 1))

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask, g_T=guidance)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        # Visualize chains
        # if self.visualization_tools is not None:
        #     # self.print('Visualizing chains...')
        #     current_path = os.getcwd()
        #     num_molecules = chain_X.size(1)  # number of molecules
        #     for i in range(num_molecules):
        #         result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
        #                                                  f'epoch{self.current_epoch}/'
        #                                                  f'chains/molecule_{batch_id + i}')
        #         if not os.path.exists(result_path):
        #             os.makedirs(result_path)
        #             _ = self.visualization_tools.visualize_chain(result_path,
        #                                                          chain_X[:, i, :].numpy(),
        #                                                          chain_E[:, i, :].numpy())
        #         # self.print('\r{}/{} complete'.format(i+1, num_molecules), end=''""", flush=True""")
        #     # self.print('\nVisualizing molecules...')
        #
        #     # Visualize the final molecules
        #     current_path = os.getcwd()
        #     result_path = os.path.join(current_path,
        #                                f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
        #     self.visualization_tools.visualize(result_path, molecule_list, save_final)
            # self.print("Done.")

        return molecule_list

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask, g_T=None):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask, g_T)

        # Normalize predictions
        if (self.cfg.guidance.loss == 'crossentropy'):
            # Normalize predictions
            pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
            pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0
        else:
            pred_X = torch.exp(pred.X)
            pred_E = torch.exp(pred.E)

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X  # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)  # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E  # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    def save_cond_samples(self, samples, target, file_path):
        cond_results = {'smiles': [], 'input_targets': target}
        invalid = 0
        disconnected = 0

        print("\tConverting conditionally generated molecules to SMILES ...")
        for sample in samples:
            if (self.cfg.guidance.build_with_partial_charges == "old_method"):  #
                mol = build_molecule_with_partial_charges(sample[0], sample[1], self.dataset_info.atom_decoder)
            else:
                mol = build_molecule(sample[0], sample[1], self.dataset_info.atom_decoder)

            smile = mol2smiles(mol)
            if smile is not None:
                cond_results['smiles'].append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                if len(mol_frags) > 1:
                    print("Disconnected molecule", mol, mol_frags)
                    disconnected += 1
            else:
                print("Invalid molecule obtained.")
                invalid += 1

        print("Number of invalid molecules", invalid)
        print("Number of disconnected molecules", disconnected)

        # save samples
        with open(file_path, 'wb') as f:
            pickle.dump(cond_results, f)

        return cond_results

    def save_graph_planarity_stats(self, samples, file_path: str = None):
        """
        Takes in a list of [node_labels, adj_matrix] and checks how many are planar,
        and how many are both connected and planar.

        Args:
            samples (List[List[Tensor]]): Each sample is [node_labels, adj_matrix]
            file_path (str): Optional path to save results

        Returns:
            dict: {
                'planar_graphs': List[nx.Graph],
                'non_planar_graphs': List[nx.Graph],
                'num_planar': int,
                'num_non_planar': int,
                'num_connected_and_planar': int
            }
        """
        planar_graphs = []
        non_planar_graphs = []
        num_connected_and_planar = 0

        for sample in samples:
            _, adj = sample  # Ignore node labels
            adj = adj.cpu().numpy()

            # Convert adjacency matrix to nx.Graph
            G = nx.from_numpy_array(adj)

            is_planar, _ = nx.check_planarity(G)
            if is_planar:
                planar_graphs.append(G)
                if nx.is_connected(G):
                    num_connected_and_planar += 1
            else:
                non_planar_graphs.append(G)

        print(f"✅ Planar graphs: {len(planar_graphs)}")
        print(f"❌ Non-planar graphs: {len(non_planar_graphs)}")
        print(f"🔗✅ Connected and planar graphs: {num_connected_and_planar}")

        results = {
            "planar_graphs": planar_graphs,
            "non_planar_graphs": non_planar_graphs,
            "num_planar": len(planar_graphs),
            "num_non_planar": len(non_planar_graphs),
            "num_connected_and_planar": num_connected_and_planar,
        }

        if file_path:
            with open(file_path, "wb") as f:
                pickle.dump(results, f)
            print(f"📦 Saved results to {file_path}")

        return results

    def apply_noise_counter(self, t, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        t_int = torch.ones(size=(X.size(0), 1), device=X.device).float() * t
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)  # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)  # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar,
                                               device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def counterfactual(self, t, X, E, y, node_mask, guidance, batch_size):



        for s_int in reversed(range(0, t)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask, g_T=guidance)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        sampled_s = sampled_s.mask(node_mask, collapse=True)  ##for qm9
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y
        counterfactual = {'X': X, 'E': E, 'y': y}
        return counterfactual

