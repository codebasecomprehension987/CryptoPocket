"""
SE(3) geometry utilities.
Provides rotation conversions, Gram-Schmidt orthonormalisation,
and SE(3) frame manipulation routines used throughout CryptoPocket.
"""

from __future__ import annotations

import torch
import numpy as np
from torch import Tensor


def rotation_6d_to_matrix(rot_6d: Tensor) -> Tensor:
    """
    Convert continuous 6D rotation representation to 3×3 rotation matrix.
    Uses Gram-Schmidt orthonormalisation (Zhou et al. 2019).

    Args:
        rot_6d: [..., 6] tensor

    Returns:
        Rotation matrix [..., 3, 3]
    """
    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:]

    # Normalise first vector
    b1 = torch.nn.functional.normalize(a1, dim=-1)

    # Gram-Schmidt: remove b1 component from a2
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = torch.nn.functional.normalize(a2 - dot * b1, dim=-1)

    # Third column: cross product
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)  # [..., 3, 3]


def gram_schmidt_se3(frames: Tensor) -> Tensor:
    """
    Re-orthonormalise SE(3) frames.

    Ensures the rotation part of each 4×4 frame matrix is a valid SO(3)
    element. Useful for numerical stability after many denoising steps.

    Args:
        frames: [..., 4, 4]

    Returns:
        Orthonormalised frames [..., 4, 4]
    """
    R = frames[..., :3, :3]  # [..., 3, 3]

    # Extract columns
    c1 = R[..., 0]  # [..., 3]
    c2 = R[..., 1]
    # c3 will be recomputed as cross product

    # Normalise first column
    c1 = torch.nn.functional.normalize(c1, dim=-1)

    # Gram-Schmidt second column
    c2 = c2 - (c2 * c1).sum(-1, keepdim=True) * c1
    c2 = torch.nn.functional.normalize(c2, dim=-1)

    # Third column: cross product
    c3 = torch.cross(c1, c2, dim=-1)

    # Reconstruct
    R_new = torch.stack([c1, c2, c3], dim=-1)
    frames_new = frames.clone()
    frames_new[..., :3, :3] = R_new
    return frames_new


def se3_inverse(T: Tensor) -> Tensor:
    """
    Invert SE(3) frames: T^{-1} = [R^T | -R^T t]

    Args:
        T: [..., 4, 4]

    Returns:
        T^{-1}: [..., 4, 4]
    """
    R = T[..., :3, :3]   # [..., 3, 3]
    t = T[..., :3, 3]    # [..., 3]

    R_inv = R.transpose(-2, -1)
    t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)

    T_inv = torch.zeros_like(T)
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, 3] = t_inv
    T_inv[..., 3, 3] = 1.0
    return T_inv


def relative_frames(T_i: Tensor, T_j: Tensor) -> Tensor:
    """
    Compute relative frames T_i^{-1} T_j.

    Args:
        T_i: [..., 4, 4]
        T_j: [..., 4, 4]

    Returns:
        Relative frames [..., 4, 4]
    """
    return se3_inverse(T_i) @ T_j


def frames_to_coords(frames: np.ndarray) -> np.ndarray:
    """
    Extract translation (position) from SE(3) frames.

    Args:
        frames: [N, 4, 4]

    Returns:
        Coordinates [N, 3]
    """
    return frames[:, :3, 3]
