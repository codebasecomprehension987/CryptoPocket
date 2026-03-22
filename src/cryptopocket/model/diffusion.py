"""
SE(3) Latent Diffusion Model
==============================
Implements the diffusion process over per-residue SE(3) frames for
hallucinating the holo (ligand-bound) conformational state from the
apo sequence.

Key design decisions:
  - Latent space: [N_res, 4, 4] SE(3) frames (not Cartesian coordinates)
  - Noise schedule: cosine schedule from Ho et al. 2020
  - Sampler: DDIM (Song et al. 2020) for deterministic 200-step inference
  - Frame storage: Rust arena (GC-bypass) — each denoising step's tensors
    are allocated in the arena and released atomically after the step
  - Conditioning: encoder single features are cross-attended at every step

The diffusion objective is to predict the clean SE(3) frame x_0 from
the noised frame x_t, conditioned on the sequence encoder output.
Loss is the Frobenius norm between predicted and true SE(3) matrices.
"""

from __future__ import annotations

import math
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from cryptopocket.arena import FrameArena, create_arena
from cryptopocket.kernels import se3_frame_attention


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

def cosine_beta_schedule(n_steps: int, s: float = 0.008) -> Tensor:
    """
    Cosine noise schedule (Nichol & Dhariwal 2021).

    f(t) = cos((t/T + s) / (1 + s) · π/2)²

    Returns:
        betas [n_steps] — noise level at each step
    """
    steps = torch.arange(n_steps + 1, dtype=torch.float64)
    t = steps / n_steps
    alphas_cumprod = torch.cos(((t + s) / (1.0 + s)) * math.pi / 2.0) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999).float()


class NoiseSchedule(nn.Module):
    """Pre-computes and caches all diffusion coefficients."""

    def __init__(self, n_steps: int = 200):
        super().__init__()
        self.n_steps = n_steps

        betas = cosine_beta_schedule(n_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (moved to device with the module)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.rsqrt(alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        # DDIM eta=0 coefficients
        self.register_buffer("ddim_sigma", torch.zeros(n_steps))

    def q_sample(self, x0: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        """
        Forward diffusion: sample x_t from x_0.
        x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε

        Args:
            x0: Clean frames [B, N, 4, 4]
            t: Timestep indices [B]
            noise: Optional pre-sampled noise (for reproducibility)

        Returns:
            Noised frames [B, N, 4, 4]
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def predict_x0_from_noise(self, xt: Tensor, t: Tensor, noise_pred: Tensor) -> Tensor:
        """Predict x_0 from noisy x_t and predicted noise ε."""
        sqrt_recip = self.sqrt_recip_alphas_cumprod[t][:, None, None, None]
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t][:, None, None, None]
        return sqrt_recip * xt - sqrt_recipm1 * noise_pred

    def ddim_step(
        self,
        xt: Tensor,
        t: int,
        t_prev: int,
        noise_pred: Tensor,
    ) -> Tensor:
        """
        DDIM deterministic update step (η=0).

        x_{t-1} = sqrt(ᾱ_{t-1}) · x̂_0 + sqrt(1 - ᾱ_{t-1}) · ε_θ(x_t, t)
        """
        alpha_bar = self.alphas_cumprod[t]
        alpha_bar_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Predict clean x_0
        x0_pred = self.predict_x0_from_noise(
            xt, torch.tensor([t], device=xt.device), noise_pred
        )
        x0_pred = x0_pred.clamp(-5.0, 5.0)  # stability clamp

        # DDIM update
        dir_xt = torch.sqrt(1.0 - alpha_bar_prev) * noise_pred
        x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt

        return x_prev


# ---------------------------------------------------------------------------
# Denoising network (U-Net over residue sequence)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Residual block with time embedding injection."""

    def __init__(self, dim: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.time_proj = nn.Linear(time_dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        """x: [B, N, D], t_emb: [B, time_dim]"""
        h = self.act(self.linear1(self.norm1(x)))
        h = h + self.time_proj(self.act(t_emb)).unsqueeze(1)
        h = self.linear2(self.norm2(h))
        return x + h


class CrossAttentionBlock(nn.Module):
    """Cross-attention from noisy frames to encoder conditioning."""

    def __init__(self, query_dim: int, context_dim: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        head_dim = query_dim // n_heads
        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.k_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(context_dim, query_dim, bias=False)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        B, N, D = x.shape
        H = self.n_heads
        Dh = D // H

        Q = self.q_proj(x).view(B, N, H, Dh)
        K = self.k_proj(context).view(B, -1, H, Dh)
        V = self.v_proj(context).view(B, -1, H, Dh)

        # Standard scaled dot-product attention for cross-attention
        Q_t = Q.permute(0, 2, 1, 3)
        K_t = K.permute(0, 2, 1, 3)
        V_t = V.permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(Q_t, K_t, V_t)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return self.norm(x + out)


class FrameDenoisingNetwork(nn.Module):
    """
    Predicts the noise ε from (x_t, t, conditioning).
    Input/output: [B, N, 4, 4] flat → [B, N, 16] → ... → [B, N, 16] → [B, N, 4, 4]
    """

    def __init__(
        self,
        n_steps: int = 200,
        hidden_dim: int = 256,
        context_dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        frame_flat_dim = 16  # 4×4

        # Time embedding: sinusoidal → MLP
        time_emb_dim = hidden_dim * 4
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Frame projection: [B, N, 16] → [B, N, hidden_dim]
        self.frame_proj_in = nn.Linear(frame_flat_dim, hidden_dim)

        # Alternating residual + cross-attention blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, time_emb_dim)
            for _ in range(n_layers)
        ])
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, context_dim, n_heads)
            for _ in range(n_layers)
        ])

        # Output head: [B, N, hidden_dim] → [B, N, 16]
        self.frame_proj_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, frame_flat_dim),
        )

    def forward(
        self,
        xt: Tensor,          # [B, N, 4, 4] noisy frames
        t: Tensor,           # [B] timestep indices
        context: Tensor,     # [B, N, context_dim] encoder conditioning
    ) -> Tensor:
        """
        Predict noise ε(x_t, t, context).

        Returns:
            Predicted noise [B, N, 4, 4]
        """
        B, N, _, _ = xt.shape

        # Flatten frames: [B, N, 16]
        x = xt.view(B, N, 16)
        x = self.frame_proj_in(x)  # [B, N, D]

        # Time embedding: [B, time_emb_dim]
        t_emb = self.time_emb(t.float())

        # Denoising layers
        for res_block, cross_attn in zip(self.res_blocks, self.cross_attn_blocks):
            x = res_block(x, t_emb)
            x = cross_attn(x, context)

        # Project back to frame space
        noise_pred_flat = self.frame_proj_out(x)  # [B, N, 16]
        return noise_pred_flat.view(B, N, 4, 4)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion timestep t."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        """t: [B] float → [B, dim]"""
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ---------------------------------------------------------------------------
# Diffusion pipeline
# ---------------------------------------------------------------------------

class SE3DiffusionPipeline(nn.Module):
    """
    Full SE(3) latent diffusion pipeline.

    Combines:
      - NoiseSchedule: cosine schedule, DDIM coefficients
      - FrameDenoisingNetwork: ε-prediction network
      - FrameArena: GC-free tensor lifecycle management

    The denoising loop runs 200 steps. Each step:
      1. Allocate the current frame tensor in the arena (generation = step index)
      2. Run one DDIM update
      3. Release the previous generation's tensors from the arena

    This ensures at most 2 generations of frames are live at any time,
    and Python GC never sees the [N_res, 4, 4] frame tensors.
    """

    def __init__(
        self,
        n_steps: int = 200,
        hidden_dim: int = 256,
        context_dim: int = 256,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.schedule = NoiseSchedule(n_steps)
        self.denoiser = FrameDenoisingNetwork(
            n_steps=n_steps,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
        )

    def training_loss(
        self,
        x0: Tensor,        # [B, N, 4, 4] clean frames
        context: Tensor,   # [B, N, context_dim] from encoder
    ) -> Tensor:
        """
        Compute the simple diffusion training loss:
        L = E[||ε - ε_θ(x_t, t, context)||²_F]

        Args:
            x0: Ground-truth SE(3) frames [B, N, 4, 4]
            context: Encoder conditioning [B, N, D]

        Returns:
            Scalar loss tensor
        """
        B = x0.shape[0]
        device = x0.device

        # Sample random timesteps
        t = torch.randint(0, self.n_steps, (B,), device=device)

        # Forward diffusion
        noise = torch.randn_like(x0)
        xt = self.schedule.q_sample(x0, t, noise)

        # Predict noise
        noise_pred = self.denoiser(xt, t, context)

        # Frobenius norm loss (natural for SE(3) matrices)
        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        return loss

    @torch.no_grad()
    def ddim_sample(
        self,
        shape: tuple[int, int],  # (B, N_res)
        context: Tensor,          # [B, N_res, context_dim]
        progress_callback: Optional[Callable[[int, Tensor], None]] = None,
        use_arena: bool = True,
    ) -> Tensor:
        """
        DDIM sampling: hallucinate holo frames from encoder conditioning.

        This is the inference-time hot path:
          - 200 denoising steps
          - Rust arena for GC-free frame tensor management
          - At most 2 generations live simultaneously

        Args:
            shape: (batch_size, n_residues)
            context: Encoder output conditioning [B, N, D]
            progress_callback: Optional fn(step, frames) for monitoring
            use_arena: Whether to use Rust arena (True by default)

        Returns:
            Final holo-state frames [B, N_res, 4, 4]
        """
        B, N = shape
        device = context.device

        # Initialise from Gaussian noise
        xt = torch.randn(B, N, 4, 4, device=device)

        # Create frame arena for GC-free tensor lifecycle
        arena = create_arena() if use_arena else None

        # DDIM timestep sequence (reversed: T → 0)
        timesteps = list(range(self.n_steps - 1, -1, -1))
        timestep_pairs = list(zip(timesteps[:-1], timesteps[1:]))
        timestep_pairs.append((timesteps[-1], -1))

        for step_idx, (t, t_prev) in enumerate(timestep_pairs):
            # --- Allocate current frame in arena ---
            if arena is not None:
                handle = arena.allocate(
                    generation=step_idx,
                    data=xt.cpu().numpy().astype("float32"),
                )

            # --- Denoising step ---
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = self.denoiser(xt, t_batch, context)
            xt = self.schedule.ddim_step(xt, t, t_prev, noise_pred)

            # --- Progress callback ---
            if progress_callback is not None:
                progress_callback(step_idx, xt)

            # --- Release previous generation (GC bypass) ---
            if arena is not None and step_idx > 0:
                freed = arena.release_generation(step_idx - 1)

        # Final cleanup
        if arena is not None:
            arena.release_all()

        return xt  # [B, N_res, 4, 4]
