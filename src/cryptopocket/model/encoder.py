"""
SE(3)-Equivariant Sequence Encoder
===================================
Maps an amino acid sequence to per-residue SE(3) rigid frames suitable
for the latent diffusion model.

Architecture:
  1. ESM-2 residue embeddings (frozen or fine-tuned)
  2. Pair features: relative positional encodings + contact predictions
  3. SE(3)-equivariant attention stack (uses Triton kernel in hot path)
  4. Frame predictor head → [N_res, 4, 4] homogeneous matrices

The encoder is trained end-to-end with the diffusion model.
At inference the encoder runs once and its output is used to condition
all 200 denoising steps.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

from cryptopocket.kernels import se3_frame_attention
from cryptopocket.utils.geometry import (
    gram_schmidt_se3,
    rotation_6d_to_matrix,
)


# ---------------------------------------------------------------------------
# Residue embedding
# ---------------------------------------------------------------------------

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYX"  # 20 standard + unknown
AA_TO_IDX: dict[str, int] = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def sequence_to_tokens(sequence: str, device: torch.device) -> Tensor:
    """
    Convert an amino acid sequence string to integer token indices.

    Args:
        sequence: Single-letter amino acid sequence
        device: Target device

    Returns:
        Long tensor [N_res]
    """
    tokens = [AA_TO_IDX.get(aa.upper(), AA_TO_IDX["X"]) for aa in sequence]
    return torch.tensor(tokens, dtype=torch.long, device=device)


class ResidueEmbedding(nn.Module):
    """
    Learnable amino acid embedding + sinusoidal positional encoding.
    Optionally concatenates ESM-2 features.
    """

    def __init__(
        self,
        n_vocab: int = len(AMINO_ACIDS),
        embed_dim: int = 256,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_emb = nn.Embedding(n_vocab, embed_dim, padding_idx=0)
        self.pos_enc = self._build_sinusoidal(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    @staticmethod
    def _build_sinusoidal(max_len: int, dim: int) -> Tensor:
        """Pre-compute sinusoidal positional encodings [max_len, dim]."""
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float) * (-math.log(10000.0) / dim)
        )
        enc = torch.zeros(max_len, dim)
        enc[:, 0::2] = torch.sin(pos * div)
        enc[:, 1::2] = torch.cos(pos * div)
        return enc  # not a parameter, registered as buffer in __init__

    def forward(self, tokens: Tensor) -> Tensor:
        """
        Args:
            tokens: Long [B, N_res]

        Returns:
            Float [B, N_res, embed_dim]
        """
        B, N = tokens.shape
        x = self.token_emb(tokens)  # [B, N, D]
        pos = self.pos_enc[:N].to(tokens.device)
        x = x + pos.unsqueeze(0)
        return self.dropout(self.layer_norm(x))


# ---------------------------------------------------------------------------
# SE(3) attention layer
# ---------------------------------------------------------------------------

class SE3AttentionLayer(nn.Module):
    """
    Single SE(3)-equivariant multi-head self-attention layer.
    Projects to Q/K/V, calls the Triton kernel, projects output.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,           # [B, N, D]
        frames: Tensor,      # [B, N, 4, 4]
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, N, D = x.shape
        H = self.n_heads
        Dh = self.head_dim

        # Project to multi-head Q/K/V: [B, N, H, Dh]
        Q = self.q_proj(x).view(B, N, H, Dh)
        K = self.k_proj(x).view(B, N, H, Dh)
        V = self.v_proj(x).view(B, N, H, Dh)

        # SE(3)-equivariant attention (Triton kernel or PyTorch fallback)
        attn_out = se3_frame_attention(Q, K, V, frames)  # [B, N, H, Dh]

        # Merge heads and project
        attn_out = attn_out.contiguous().view(B, N, D)
        attn_out = self.out_proj(attn_out)
        attn_out = self.dropout(attn_out)

        # Pre-norm residual
        return self.layer_norm(x + attn_out)


# ---------------------------------------------------------------------------
# Pair feature module
# ---------------------------------------------------------------------------

class PairFeatureModule(nn.Module):
    """
    Computes pairwise features for residue pairs (i, j):
      - Relative position encoding
      - Outer product of single-residue features
    Outputs [B, N, N, pair_dim] which is used to bias attention.
    """

    def __init__(self, single_dim: int = 256, pair_dim: int = 64):
        super().__init__()
        self.pair_dim = pair_dim
        self.outer_product = nn.Sequential(
            nn.Linear(single_dim, pair_dim),
            nn.ReLU(),
        )
        self.rel_pos_emb = nn.Embedding(129, pair_dim)  # -64..64 + clip
        self.layer_norm = nn.LayerNorm(pair_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Single features [B, N, D]

        Returns:
            Pair features [B, N, N, pair_dim]
        """
        B, N, D = x.shape

        # Outer product features
        proj = self.outer_product(x)  # [B, N, pair_dim]
        pair = proj.unsqueeze(2) + proj.unsqueeze(1)  # [B, N, N, pair_dim]

        # Relative positional encoding
        pos = torch.arange(N, device=x.device)
        rel_pos = (pos.unsqueeze(0) - pos.unsqueeze(1)).clamp(-64, 64) + 64  # [N, N]
        rel_emb = self.rel_pos_emb(rel_pos)  # [N, N, pair_dim]
        pair = pair + rel_emb.unsqueeze(0)

        return self.layer_norm(pair)


# ---------------------------------------------------------------------------
# Frame predictor head
# ---------------------------------------------------------------------------

class FramePredictorHead(nn.Module):
    """
    Predicts per-residue SE(3) rigid frames from single + pair features.
    Outputs [B, N, 4, 4] homogeneous matrices.

    Uses 6D rotation representation (continuous, avoids gimbal lock)
    followed by Gram-Schmidt orthonormalisation.
    """

    def __init__(self, single_dim: int = 256, pair_dim: int = 64):
        super().__init__()
        # 6D rotation + 3D translation = 9 outputs per residue
        self.frame_head = nn.Sequential(
            nn.Linear(single_dim + pair_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 9),
        )

    def forward(self, x: Tensor, pair: Tensor) -> Tensor:
        """
        Args:
            x:    Single features [B, N, D_single]
            pair: Pair features   [B, N, N, D_pair]  (mean-pooled over j dim)

        Returns:
            SE(3) frames [B, N, 4, 4]
        """
        # Pool pair features: [B, N, D_pair]
        pair_pooled = pair.mean(dim=2)

        # Concatenate and predict 9 raw frame parameters
        feat = torch.cat([x, pair_pooled], dim=-1)
        raw = self.frame_head(feat)  # [B, N, 9]

        rot_6d = raw[..., :6]           # [B, N, 6]
        trans = raw[..., 6:]            # [B, N, 3]

        # Convert 6D → 3×3 rotation via Gram-Schmidt
        R = rotation_6d_to_matrix(rot_6d)  # [B, N, 3, 3]

        # Assemble 4×4 homogeneous matrix
        B, N = x.shape[:2]
        T = torch.zeros(B, N, 4, 4, device=x.device, dtype=x.dtype)
        T[..., :3, :3] = R
        T[..., :3, 3] = trans
        T[..., 3, 3] = 1.0

        return T


# ---------------------------------------------------------------------------
# Full encoder
# ---------------------------------------------------------------------------

class SE3SequenceEncoder(nn.Module):
    """
    Maps an amino acid sequence to SE(3) rigid frames for each residue.

    This is the encoder half of the CryptoPocket diffusion model.
    Its output conditions the 200-step DDIM denoising loop.

    Args:
        embed_dim: Width of single-residue feature vectors (default 256)
        pair_dim: Width of pairwise feature vectors (default 64)
        n_layers: Number of SE(3) attention layers (default 8)
        n_heads: Attention heads per layer (default 8)
        dropout: Dropout rate (default 0.1)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        pair_dim: int = 64,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.residue_emb = ResidueEmbedding(embed_dim=embed_dim, dropout=dropout)

        # Initialize attention layers with identity frames initially
        self.attn_layers = nn.ModuleList([
            SE3AttentionLayer(embed_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Feed-forward layers interleaved with attention
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])

        self.pair_module = PairFeatureModule(embed_dim, pair_dim)
        self.frame_head = FramePredictorHead(embed_dim, pair_dim)

    def forward(
        self,
        tokens: Tensor,                     # [B, N_res]
        attention_mask: Optional[Tensor] = None,  # [B, N_res] bool
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            tokens: Integer token indices [B, N_res]
            attention_mask: True for valid positions [B, N_res]

        Returns:
            (frames, single_features)
            frames: SE(3) frames [B, N_res, 4, 4]
            single_features: Encoder output [B, N_res, embed_dim]
        """
        B, N = tokens.shape

        # Embed residues
        x = self.residue_emb(tokens)  # [B, N, D]

        # Initialise frames as identity transforms
        frames = torch.eye(4, device=tokens.device).unsqueeze(0).unsqueeze(0)
        frames = frames.expand(B, N, -1, -1).clone()

        # SE(3) attention stack — frames are updated iteratively
        for attn, ff in zip(self.attn_layers, self.ff_layers):
            x = attn(x, frames, attn_mask=attention_mask)
            x = x + ff(x)

            # Iterative frame update from current features
            pair = self.pair_module(x)
            frames = self.frame_head(x, pair)

        return frames, x

    @torch.no_grad()
    def encode_sequence(self, sequence: str, device: torch.device) -> tuple[Tensor, Tensor]:
        """
        Convenience method: encode a raw FASTA sequence string.

        Args:
            sequence: Single-letter amino acid sequence
            device: Target device

        Returns:
            (frames [1, N, 4, 4], features [1, N, embed_dim])
        """
        tokens = sequence_to_tokens(sequence, device).unsqueeze(0)
        self.eval()
        return self(tokens)
