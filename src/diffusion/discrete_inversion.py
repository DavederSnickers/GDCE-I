"""
Discrete Diffusion Inversion Module

This module implements edit-friendly noise space inversion for discrete diffusion models,
based on the concepts from "An Edit Friendly DDPM Noise Space: Inversion and Manipulations" (CVPR 2024),
adapted for categorical/multinomial discrete diffusion.

Key Concept:
-----------
In continuous DDPM, the noise z ~ N(0,1) can be extracted during forward pass and re-injected
during reverse with different conditioning to enable editing while preserving structure.

For discrete diffusion, we use the Gumbel-Max trick:
    sample = argmax(log(p) + g)  where g ~ Gumbel(0,1)

This decomposes categorical sampling into:
    - Deterministic: log(p) (depends on conditioning)
    - Stochastic: g (Gumbel noise, independent of content)

By capturing Gumbel noise during "inversion" and re-injecting during generation with
new conditioning, we can edit discrete structures (molecules) while preserving their core identity.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

from src.utils import PlaceHolder
from src.diffusion import diffusion_utils


def sample_gumbel(shape, device, eps=1e-10):
    """Sample from Gumbel(0, 1) distribution."""
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def sample_gumbel_symmetric(shape, device, eps=1e-10):
    """Sample symmetric Gumbel noise for edge features."""
    bs, n, _, num_classes = shape
    g = sample_gumbel((bs, n, n, num_classes), device, eps)

    # Make symmetric
    upper_triangular_mask = torch.zeros((bs, n, n, num_classes), device=device)
    indices = torch.triu_indices(row=n, col=n, offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    g_upper = g * upper_triangular_mask
    g_symmetric = g_upper + g_upper.transpose(1, 2)

    # Diagonal
    diag_indices = torch.arange(n, device=device)
    g_symmetric[:, diag_indices, diag_indices, :] = g[:, diag_indices, diag_indices, :]

    return g_symmetric


def gumbel_posterior_noise(log_prob, targets, margin: float = 0.0):
    """Sample Gumbel noise ``g`` conditioned on ``argmax(log_prob + g) == target``.

    Closed-form A*-sampling / truncated-Gumbel construction (Maddison 2014):
      * The value at the argmax, ``M = max_v(log_prob_v + g_v)``, is Gumbel(logZ).
      * The target position gets ``g_target = M - log_prob_target``.
      * Every other position gets a Gumbel(log_prob_v) sample truncated to be < M.

    This is the discrete analog of the edit-friendly DDPM noise space: it records
    the EXACT noise that reproduces ``targets`` under ``log_prob``, so replaying it
    with the same conditioning reconstructs the original perfectly, while a changed
    conditioning only flips positions where the new advantage beats the recorded
    margin.

    Args:
        log_prob: (..., V) log-probabilities (any leading dims). In GDCE these are
                  ``log`` of the marginalized reverse posterior ``prob_X`` / ``prob_E``.
        targets:  (...)    class ids that ``argmax`` must select.
        margin:   add this to the target's perturbed value so it dominates the
                  runner-up by >= margin (guards exact reconstruction against
                  float/bf16 ties; slightly biases ``g``).

    Returns:
        g: (..., V) Gumbel noise with ``argmax(log_prob + g) == targets``.
    """
    L = log_prob.double()
    Z = torch.logsumexp(L, dim=-1)                                   # (...)
    top_value = Z + sample_gumbel(Z.shape, L.device)                # (...) ~ Gumbel(logZ)
    gumbel = L + sample_gumbel(L.shape, L.device)                   # (..., V) Gumbel(L_v)
    # Truncate each to be strictly below top_value (numerically stable):
    #   value = -log(exp(-gumbel) + exp(-M)) = -logaddexp(-gumbel, -M)
    value = -torch.logaddexp(-gumbel, -top_value.unsqueeze(-1))      # (..., V)
    # Force the target positions to exactly the max value (+ optional margin).
    value.scatter_(-1, targets.unsqueeze(-1), (top_value + margin).unsqueeze(-1))
    g = value - L
    return g.to(log_prob.dtype)


@dataclass
class InversionOutput:
    """Container for inversion forward process outputs."""
    X_T: torch.Tensor
    E_T: torch.Tensor
    gumbels_X: List[torch.Tensor]
    gumbels_E: List[torch.Tensor]
    trajectory_X: List[torch.Tensor]
    trajectory_E: List[torch.Tensor]


class DiscreteInversion:
    """
    Discrete Diffusion Inversion for edit-friendly noise space.
    """

    def __init__(self, model, device: Optional[str] = None):
        self.model = model
        self.device = device or model.device
        self.T = model.T
        self.Xdim_output = model.Xdim_output
        self.Edim_output = model.Edim_output

    @torch.no_grad()
    def inversion_forward(
        self,
        X_0: torch.Tensor,
        E_0: torch.Tensor,
        node_mask: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        prog_bar: bool = True
    ) -> InversionOutput:
        """
        Forward process that captures Gumbel noise for later injection.
        Uses the same approach as apply_noise_counter but captures the randomness.
        """
        num_steps = num_steps or self.T
        bs, n, dx_input = X_0.shape
        de_input = E_0.shape[-1]

        # Use model's output dimensions to ensure consistency with reverse process
        dx = self.Xdim_output
        de = self.Edim_output

        gumbels_X = []
        gumbels_E = []
        trajectory_X = [X_0.clone()]
        trajectory_E = [E_0.clone()]

        # Start from clean data — each step transitions from previous state
        X_prev = X_0.clone()
        E_prev = E_0.clone()

        iterator = range(1, num_steps + 1)
        if prog_bar:
            iterator = tqdm(iterator, desc="Forward inversion")

        for t_int in iterator:
            t_float = torch.tensor([[t_int / self.T]], device=self.device).expand(bs, -1)

            # Get single-step transition matrix Qt (NOT cumulative Qt_bar)
            beta_t = self.model.noise_schedule(t_normalized=t_float)
            Qt = self.model.transition_model.get_Qt(beta_t, self.device)

            # Sequential forward: q(x_t | x_{t-1}) = x_{t-1} @ Qt
            prob_X = X_prev @ Qt.X          # (bs, n, dx_out)
            prob_E = E_prev @ Qt.E.unsqueeze(1)  # (bs, n, n, de_out)

            # Generate and store Gumbel noise
            g_X = sample_gumbel((bs, n, dx), self.device)
            g_E = sample_gumbel_symmetric((bs, n, n, de), self.device)
            gumbels_X.append(g_X)
            gumbels_E.append(g_E)

            # Sample using Gumbel-Max trick
            # Handle masked positions
            prob_X_safe = prob_X.clone()
            prob_X_safe[~node_mask] = 1.0 / dx

            inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
            diag_mask = torch.eye(n, device=self.device).unsqueeze(0).expand(bs, -1, -1).bool()
            prob_E_safe = prob_E.clone()
            prob_E_safe[inverse_edge_mask] = 1.0 / de
            prob_E_safe[diag_mask] = 1.0 / de

            log_prob_X = torch.log(prob_X_safe + 1e-10)
            log_prob_E = torch.log(prob_E_safe + 1e-10)

            X_t_indices = torch.argmax(log_prob_X + g_X, dim=-1)
            E_t_indices = torch.argmax(log_prob_E + g_E, dim=-1)

            # Make edges symmetric
            E_t_indices = torch.triu(E_t_indices, diagonal=1)
            E_t_indices = E_t_indices + E_t_indices.transpose(1, 2)

            # Convert to one-hot
            X_t = F.one_hot(X_t_indices, num_classes=self.Xdim_output).float()
            E_t = F.one_hot(E_t_indices, num_classes=self.Edim_output).float()

            # Update previous state for next step (Markov chain)
            X_prev = X_t
            E_prev = E_t

            trajectory_X.append(X_t.clone())
            trajectory_E.append(E_t.clone())

        return InversionOutput(
            X_T=trajectory_X[-1],
            E_T=trajectory_E[-1],
            gumbels_X=gumbels_X,
            gumbels_E=gumbels_E,
            trajectory_X=trajectory_X,
            trajectory_E=trajectory_E
        )

    @torch.no_grad()
    def inversion_reverse(
        self,
        X_T: torch.Tensor,
        E_T: torch.Tensor,
        node_mask: torch.Tensor,
        new_guidance: torch.Tensor,
        gumbels_X: List[torch.Tensor],
        gumbels_E: List[torch.Tensor],
        skip: int = 0,
        prog_bar: bool = True,
        trajectory_X: Optional[List[torch.Tensor]] = None,
        trajectory_E: Optional[List[torch.Tensor]] = None,
        capture_last_noisy: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Reverse process using captured Gumbel noise with new conditioning.
        Closely follows the original sample_p_zs_given_zt method.
        """
        num_steps = len(gumbels_X)
        bs = X_T.shape[0]
        n = X_T.shape[1]

        # Determine starting point
        start_t = num_steps - skip
        if start_t <= 0:
            raise ValueError(f"skip={skip} is too large for num_steps={num_steps}")

        # Use trajectory state if available
        if skip > 0 and trajectory_X is not None:
            X_t = trajectory_X[start_t].clone().to(self.device)
            E_t = trajectory_E[start_t].clone().to(self.device)
        else:
            X_t = X_T.clone()
            E_t = E_T.clone()

        y_t = torch.zeros(bs, 0, device=self.device)

        iterator = reversed(range(0, start_t))
        if prog_bar:
            iterator = tqdm(list(iterator), desc=f"Reverse (skip={skip})")

        X_1_noisy: Optional[torch.Tensor] = None
        E_1_noisy: Optional[torch.Tensor] = None

        for s_int in iterator:
            t_int = s_int + 1

            # Capture state at t=1 (input to the last denoising step)
            if capture_last_noisy and s_int == 0:
                X_1_noisy = X_t.clone()
                E_1_noisy = E_t.clone()

            # Marginalized reverse posterior under NEW guidance (CFG handled inside
            # forward() when train_step=False). Shared with the posterior-inversion
            # recorder so record and replay use byte-identical math.
            prob_X, prob_E = self._reverse_posterior(
                X_t, E_t, node_mask, new_guidance, s_int)

            # Use CAPTURED Gumbel noise for sampling (KEY DIFFERENCE!)
            g_X = gumbels_X[s_int].to(self.device)
            g_E = gumbels_E[s_int].to(self.device)

            # Gumbel-Max sampling with captured noise
            log_prob_X = torch.log(prob_X + 1e-10)
            log_prob_E = torch.log(prob_E + 1e-10)

            X_s_indices = torch.argmax(log_prob_X + g_X, dim=-1)
            E_s_indices = torch.argmax(log_prob_E + g_E, dim=-1)

            # Make edges symmetric
            E_s_indices = torch.triu(E_s_indices, diagonal=1)
            E_s_indices = E_s_indices + E_s_indices.transpose(1, 2)

            # Convert to one-hot
            X_t = F.one_hot(X_s_indices, num_classes=self.Xdim_output).float()
            E_t = F.one_hot(E_s_indices, num_classes=self.Edim_output).float()

        if capture_last_noisy:
            return X_t, E_t, X_1_noisy, E_1_noisy
        return X_t, E_t

    # ======================================================================
    #  Plan A: edit-friendly *posterior* Gumbel inversion (exact reconstruction)
    # ======================================================================

    def _reverse_posterior(self, X_t, E_t, node_mask, guidance, s_int):
        """Marginalized reverse posterior ``p(x_s | x_t, guidance)`` for X and E.

        This is exactly the distribution the reverse process samples from at step
        ``s_int`` (t = s_int + 1). Factored out so that the posterior-Gumbel
        RECORDER (``inversion_forward_posterior``) and the REPLAY
        (``inversion_reverse``) call identical code — a prerequisite for the
        exact-reconstruction property.
        """
        bs, n = X_t.shape[0], X_t.shape[1]
        t_int = s_int + 1
        y_t = torch.zeros(bs, 0, device=self.device)

        s_norm = (s_int * torch.ones((bs, 1), device=self.device)) / self.T
        t_norm = (t_int * torch.ones((bs, 1), device=self.device)) / self.T

        beta_t = self.model.noise_schedule(t_normalized=t_norm)
        alpha_s_bar = self.model.noise_schedule.get_alpha_bar(t_normalized=s_norm)
        alpha_t_bar = self.model.noise_schedule.get_alpha_bar(t_normalized=t_norm)

        Qtb = self.model.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.model.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.model.transition_model.get_Qt(beta_t, self.device)

        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t,
                      't': t_norm, 'node_mask': node_mask}
        extra_data = self.model.compute_extra_data(noisy_data)
        pred = self.model.forward(noisy_data, extra_data, node_mask, guidance=guidance)

        pred_X = F.softmax(pred.X, dim=-1)
        pred_E = F.softmax(pred.E, dim=-1)

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(
            X_t=X_t, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X)
        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(
            X_t=E_t, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E)

        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X
        unnormalized_prob_X = weighted_X.sum(dim=2)
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)

        pred_E_flat = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E_flat.unsqueeze(-1) * p_s_and_t_given_0_E
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, -1)

        return prob_X, prob_E

    def _symmetrize_gumbel(self, g_E):
        """Mirror the upper-triangle Gumbel noise to the lower triangle (diag = 0).

        Matches the reverse process, which argmaxes per (i,j), then keeps only the
        upper triangle and mirrors it. So only the upper-triangle noise must force
        the reference; mirroring keeps ``g_E`` symmetric and consistent.
        """
        bs, n = g_E.shape[0], g_E.shape[1]
        iu = torch.triu_indices(n, n, offset=1, device=g_E.device)
        mask = torch.zeros((1, n, n, 1), device=g_E.device, dtype=g_E.dtype)
        mask[:, iu[0], iu[1], :] = 1.0
        g_upper = g_E * mask
        return g_upper + g_upper.transpose(1, 2)

    @torch.no_grad()
    def _build_reference_trajectory(self, X_0, E_0, node_mask, num_steps):
        """Markov forward reference states x_0*, x_1*, ..., x_{num_steps}*.

        Identical noising to ``inversion_forward`` (single-step q(x_t|x_{t-1})) but
        only the STATES are kept; the per-step Gumbel noise is discarded because the
        posterior recorder re-derives the noise in the reverse direction instead.
        """
        bs, n, _ = X_0.shape
        dx, de = self.Xdim_output, self.Edim_output

        trajectory_X = [X_0.clone()]
        trajectory_E = [E_0.clone()]
        X_prev, E_prev = X_0.clone(), E_0.clone()

        diag_mask = torch.eye(n, device=self.device).unsqueeze(0).expand(bs, -1, -1).bool()
        inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))

        for t_int in range(1, num_steps + 1):
            t_float = torch.tensor([[t_int / self.T]], device=self.device).expand(bs, -1)
            beta_t = self.model.noise_schedule(t_normalized=t_float)
            Qt = self.model.transition_model.get_Qt(beta_t, self.device)

            prob_X = X_prev @ Qt.X
            prob_E = E_prev @ Qt.E.unsqueeze(1)

            g_X = sample_gumbel((bs, n, dx), self.device)
            g_E = sample_gumbel_symmetric((bs, n, n, de), self.device)

            prob_X_safe = prob_X.clone()
            prob_X_safe[~node_mask] = 1.0 / dx
            prob_E_safe = prob_E.clone()
            prob_E_safe[inverse_edge_mask] = 1.0 / de
            prob_E_safe[diag_mask] = 1.0 / de

            X_t_indices = torch.argmax(torch.log(prob_X_safe + 1e-10) + g_X, dim=-1)
            E_t_indices = torch.argmax(torch.log(prob_E_safe + 1e-10) + g_E, dim=-1)
            E_t_indices = torch.triu(E_t_indices, diagonal=1)
            E_t_indices = E_t_indices + E_t_indices.transpose(1, 2)

            X_t = F.one_hot(X_t_indices, num_classes=dx).float()
            E_t = F.one_hot(E_t_indices, num_classes=de).float()
            X_prev, E_prev = X_t, E_t
            trajectory_X.append(X_t.clone())
            trajectory_E.append(E_t.clone())

        return trajectory_X, trajectory_E

    @torch.no_grad()
    def inversion_forward_posterior(
        self,
        X_0: torch.Tensor,
        E_0: torch.Tensor,
        node_mask: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        margin: float = 0.0,
        prog_bar: bool = True,
    ) -> InversionOutput:
        """Edit-friendly POSTERIOR-Gumbel inversion (exact reconstruction).

        Unlike ``inversion_forward`` (which samples fresh Gumbel noise along the
        *forward* chain and hopes the reverse retraces it), this records, in the
        *reverse* direction, the exact Gumbel noise that forces the model's reverse
        posterior to reproduce a reference trajectory under ``guidance``:

            1. Build a Markov reference trajectory x_0*, ..., x_T*.
            2. For s = T-1 .. 0, feed x_{s+1}*, compute the reverse posterior
               p(x_s | x_{s+1}*, guidance), and record the posterior Gumbel that
               makes ``argmax(log p + g) == x_s*``.

        Replaying via ``inversion_reverse`` with the SAME guidance then reconstructs
        x_0 EXACTLY; with a changed guidance only positions whose new advantage
        exceeds the recorded margin flip → minimal, on-manifold edits.

        The returned object is drop-in compatible with ``inversion_reverse``.
        """
        num_steps = num_steps or self.T

        trajectory_X, trajectory_E = self._build_reference_trajectory(
            X_0, E_0, node_mask, num_steps)

        gumbels_X: List[Optional[torch.Tensor]] = [None] * num_steps
        gumbels_E: List[Optional[torch.Tensor]] = [None] * num_steps

        iterator = reversed(range(num_steps))
        if prog_bar:
            iterator = tqdm(list(iterator), desc="Posterior inversion (record)")

        for s_int in iterator:
            t_int = s_int + 1
            X_t = trajectory_X[t_int].to(self.device)
            E_t = trajectory_E[t_int].to(self.device)

            prob_X, prob_E = self._reverse_posterior(
                X_t, E_t, node_mask, guidance, s_int)

            target_X = trajectory_X[s_int].argmax(dim=-1).to(self.device)   # (bs, n)
            target_E = trajectory_E[s_int].argmax(dim=-1).to(self.device)   # (bs, n, n)

            g_X = gumbel_posterior_noise(torch.log(prob_X + 1e-10), target_X, margin)
            g_E = gumbel_posterior_noise(torch.log(prob_E + 1e-10), target_E, margin)
            g_E = self._symmetrize_gumbel(g_E)

            gumbels_X[s_int] = g_X.cpu()
            gumbels_E[s_int] = g_E.cpu()

        return InversionOutput(
            X_T=trajectory_X[-1], E_T=trajectory_E[-1],
            gumbels_X=gumbels_X, gumbels_E=gumbels_E,
            trajectory_X=trajectory_X, trajectory_E=trajectory_E,
        )

    @torch.no_grad()
    def compute_attribution_last_step(
        self,
        X_1: torch.Tensor,
        E_1: torch.Tensor,
        node_mask: torch.Tensor,
        orig_guidance: torch.Tensor,
        cf_guidance: torch.Tensor,
        temperature: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-node and per-edge attribution scores from the last denoising step (s=0, t=1).

        At t=1 the molecule has minimal noise — this is the model's sharpest prediction.
        Computes the full posterior p(x_0 | x_1, y) for both guidances and measures
        the TVD between them per node and edge.

        Sensitivity is increased by two mechanisms:
          1. Temperature scaling (τ < 1) sharpens the predicted distributions before
             computing the posterior, amplifying differences between guidances.
          2. Per-molecule min-max normalization maps the raw TVD scores to [0, 1]
             so the full colormap range is always used regardless of absolute magnitude.

        Args:
            X_1: (bs, n, dx) noisy node features at t=1, from inversion_reverse
                 with capture_last_noisy=True.
            E_1: (bs, n, n, de) noisy edge features at t=1.
            node_mask: boolean mask (bs, n).
            orig_guidance: guidance for original class (bs, num_classes).
            cf_guidance:  guidance for counterfactual class (bs, num_classes).
            temperature: softmax temperature τ ∈ (0, 1]. Lower = sharper distributions
                         = larger TVD differences. Default 0.3.

        Returns:
            score_X: (n,) normalized node attribution scores ∈ [0, 1], cpu tensor.
            score_E: (n, n) normalized edge attribution scores ∈ [0, 1], cpu tensor.
        """
        X_1 = X_1.to(self.device)
        E_1 = E_1.to(self.device)
        bs, n = X_1.shape[:2]
        y_t = torch.zeros(bs, 0, device=self.device)

        # s=0, t=1 noise schedule parameters
        s_norm = torch.zeros((bs, 1), device=self.device)
        t_norm = torch.ones((bs, 1), device=self.device) / self.T

        beta_t      = self.model.noise_schedule(t_normalized=t_norm)
        alpha_s_bar = self.model.noise_schedule.get_alpha_bar(t_normalized=s_norm)
        alpha_t_bar = self.model.noise_schedule.get_alpha_bar(t_normalized=t_norm)

        Qtb = self.model.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.model.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt  = self.model.transition_model.get_Qt(beta_t, self.device)

        noisy_data = {
            'X_t': X_1, 'E_t': E_1, 'y_t': y_t,
            't': t_norm, 'node_mask': node_mask,
        }
        extra_data = self.model.compute_extra_data(noisy_data)

        pred_orig = self.model.forward(noisy_data, extra_data, node_mask, guidance=orig_guidance)
        pred_cf   = self.model.forward(noisy_data, extra_data, node_mask, guidance=cf_guidance)

        def _posterior_X(pred):
            # Temperature scaling sharpens the distribution → larger guidance contrast
            pred_X = F.softmax(pred.X / temperature, dim=-1)
            p_s_t_given_0 = diffusion_utils.compute_batched_over0_posterior_distribution(
                X_t=X_1, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
            )
            unnorm = (pred_X.unsqueeze(-1) * p_s_t_given_0).sum(dim=2)
            unnorm[unnorm.sum(dim=-1) == 0] = 1e-5
            return unnorm / unnorm.sum(dim=-1, keepdim=True)  # (bs, n, dx)

        def _posterior_E(pred):
            pred_E = F.softmax(pred.E / temperature, dim=-1)
            p_s_t_given_0 = diffusion_utils.compute_batched_over0_posterior_distribution(
                X_t=E_1, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E
            )
            pred_E_flat = pred_E.reshape((bs, -1, pred_E.shape[-1]))
            unnorm = (pred_E_flat.unsqueeze(-1) * p_s_t_given_0).sum(dim=-2)
            unnorm[unnorm.sum(dim=-1) == 0] = 1e-5
            prob_E = unnorm / unnorm.sum(dim=-1, keepdim=True)
            return prob_E.reshape(bs, n, n, -1)  # (bs, n, n, de)

        prob_X_orig = _posterior_X(pred_orig)
        prob_X_cf   = _posterior_X(pred_cf)
        prob_E_orig = _posterior_E(pred_orig)
        prob_E_cf   = _posterior_E(pred_cf)

        # TVD = 0.5 * L1 per element
        score_X = 0.5 * (prob_X_cf - prob_X_orig).abs().sum(dim=-1)  # (bs, n)
        score_E = 0.5 * (prob_E_cf - prob_E_orig).abs().sum(dim=-1)  # (bs, n, n)
        score_E = (score_E + score_E.transpose(1, 2)) / 2.0           # symmetrize

        # Per-molecule min-max normalization → full colormap range always used
        score_X = score_X[0].cpu()   # (n,)
        score_E = score_E[0].cpu()   # (n, n)

        all_scores = torch.cat([score_X.flatten(), score_E.flatten()])
        s_min, s_max = all_scores.min(), all_scores.max()
        if s_max > s_min:
            score_X = (score_X - s_min) / (s_max - s_min)
            score_E = (score_E - s_min) / (s_max - s_min)
        else:
            score_X = torch.zeros_like(score_X)
            score_E = torch.zeros_like(score_E)

        return score_X, score_E

    @torch.no_grad()
    def sample_without_inversion(
        self,
        X_T: torch.Tensor,
        E_T: torch.Tensor,
        node_mask: torch.Tensor,
        guidance: torch.Tensor,
        start_t: int,
        prog_bar: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard sampling without Gumbel injection (for comparison).
        Uses fresh random noise at each step.
        """
        bs = X_T.shape[0]
        n = X_T.shape[1]

        X_t = X_T.clone()
        E_t = E_T.clone()
        y_t = torch.zeros(bs, 0, device=self.device)

        iterator = reversed(range(0, start_t))
        if prog_bar:
            iterator = tqdm(list(iterator), desc=f"Standard sampling (t={start_t})")

        for s_int in iterator:
            t_int = s_int + 1

            s_array = s_int * torch.ones((bs, 1), device=self.device)
            t_array = t_int * torch.ones((bs, 1), device=self.device)
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            beta_t = self.model.noise_schedule(t_normalized=t_norm)
            alpha_s_bar = self.model.noise_schedule.get_alpha_bar(t_normalized=s_norm)
            alpha_t_bar = self.model.noise_schedule.get_alpha_bar(t_normalized=t_norm)

            Qtb = self.model.transition_model.get_Qt_bar(alpha_t_bar, self.device)
            Qsb = self.model.transition_model.get_Qt_bar(alpha_s_bar, self.device)
            Qt = self.model.transition_model.get_Qt(beta_t, self.device)

            noisy_data = {
                'X_t': X_t, 'E_t': E_t, 'y_t': y_t,
                't': t_norm, 'node_mask': node_mask
            }
            extra_data = self.model.compute_extra_data(noisy_data)
            pred = self.model.forward(noisy_data, extra_data, node_mask, guidance=guidance)

            pred_X = F.softmax(pred.X, dim=-1)
            pred_E = F.softmax(pred.E, dim=-1)

            p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(
                X_t=X_t, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
            )
            p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(
                X_t=E_t, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E
            )

            weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X
            prob_X = weighted_X.sum(dim=2)
            prob_X[torch.sum(prob_X, dim=-1) == 0] = 1e-5
            prob_X = prob_X / torch.sum(prob_X, dim=-1, keepdim=True)

            pred_E_flat = pred_E.reshape((bs, -1, pred_E.shape[-1]))
            weighted_E = pred_E_flat.unsqueeze(-1) * p_s_and_t_given_0_E
            prob_E = weighted_E.sum(dim=-2)
            prob_E[torch.sum(prob_E, dim=-1) == 0] = 1e-5
            prob_E = prob_E / torch.sum(prob_E, dim=-1, keepdim=True)
            prob_E = prob_E.reshape(bs, n, n, -1)

            # Standard multinomial sampling (fresh randomness)
            sampled = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask)
            X_t = F.one_hot(sampled.X, num_classes=self.Xdim_output).float()
            E_t = F.one_hot(sampled.E, num_classes=self.Edim_output).float()

        return X_t, E_t

    @torch.no_grad()
    def edit_molecule(
        self,
        X_0: torch.Tensor,
        E_0: torch.Tensor,
        node_mask: torch.Tensor,
        original_guidance: Optional[torch.Tensor],
        new_guidance: torch.Tensor,
        skip: int = 100,
        prog_bar: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """High-level API: Edit a molecule by changing its guidance."""
        inv_output = self.inversion_forward(
            X_0, E_0, node_mask,
            guidance=original_guidance,
            prog_bar=prog_bar
        )

        X_edited, E_edited = self.inversion_reverse(
            X_T=inv_output.X_T,
            E_T=inv_output.E_T,
            node_mask=node_mask,
            new_guidance=new_guidance,
            gumbels_X=inv_output.gumbels_X,
            gumbels_E=inv_output.gumbels_E,
            trajectory_X=inv_output.trajectory_X,
            trajectory_E=inv_output.trajectory_E,
            skip=skip,
            prog_bar=prog_bar
        )

        return X_edited, E_edited
