#!/usr/bin/env python3
"""Typed RoPE core module — creates and applies type-aware rotary embeddings.

Provides two key functions:
1. create_type_rotation() — generates cos/sin for type encoding in target subspaces
2. apply_typed_rope() — blends position RoPE with type rotation

Type encoding replaces position rotation in selected "target" subspaces with
a fixed rotation angle determined by the token's type_id. Non-target subspaces
retain standard position-based RoPE. This creates source-aware attention
patterns that persist across all layers.

Mathematical formulation:
    Standard RoPE: rotation_angle = position * frequency
    Typed RoPE (target subspaces): rotation_angle = type_id * rotation_angle
    Typed RoPE (other subspaces): rotation_angle = position * frequency (unchanged)
"""

import math

import torch


def create_type_rotation(head_dim, type_id, target_subspaces, rotation_angle=math.pi / 4):
    """Create cos/sin vectors encoding token type for specified subspaces.

    For target subspaces, replaces position-based rotation with type-based rotation.
    For non-target subspaces, returns identity (cos=1, sin=0).

    When type_id=0: angle=0, cos=1, sin=0 → matches standard RoPE exactly.

    Uses the half-half convention matching HuggingFace Llama RoPE:
    subspace i corresponds to the 2D rotation plane (dim i, dim i + head_dim//2).
    Both dimensions in each plane must receive the same rotation angle for
    the cos(theta_A - theta_B) modulation property to hold exactly.

    Args:
        head_dim: dimension of each attention head (e.g. 128)
        type_id: 0=system/trusted, 1=user/untrusted, 2=external, ...
        target_subspaces: list of subspace indices to encode type in
        rotation_angle: base rotation angle per type increment

    Returns:
        cos_type: tensor of shape (head_dim,) — cos values for type rotation
        sin_type: tensor of shape (head_dim,) — sin values for type rotation
    """
    cos_vals = torch.ones(head_dim)
    sin_vals = torch.zeros(head_dim)

    angle = type_id * rotation_angle
    half = head_dim // 2

    for sub_idx in target_subspaces:
        # Half-half convention: subspace i → dims (i, i + head_dim//2)
        cos_vals[sub_idx] = math.cos(angle)
        cos_vals[sub_idx + half] = math.cos(angle)
        sin_vals[sub_idx] = math.sin(angle)
        sin_vals[sub_idx + half] = math.sin(angle)

    return cos_vals, sin_vals


def apply_typed_rope(q, k, cos_pos, sin_pos, type_ids, target_subspaces,
                     rotation_angle=math.pi / 4):
    """Apply typed RoPE: position encoding + type encoding.

    For target subspaces: replace position rotation with type rotation.
    For other subspaces: keep standard position rotation unchanged.

    Args:
        q: query tensor, shape (batch, num_heads, seq_len, head_dim)
        k: key tensor, shape (batch, num_heads, seq_len, head_dim)
        cos_pos: position cos from RoPE module, shape (batch, seq_len, head_dim)
                 or (1, seq_len, head_dim)
        sin_pos: position sin from RoPE module, shape matching cos_pos
        type_ids: token type IDs, shape (batch, seq_len) — 0=system, 1=user, etc.
        target_subspaces: list of subspace indices for type encoding
        rotation_angle: base rotation angle per type increment

    Returns:
        q_rotated: rotated query tensor
        k_rotated: rotated key tensor
    """
    cos_out = cos_pos.clone()
    sin_out = sin_pos.clone()

    # Override target subspaces with type rotation
    # Half-half convention: subspace i → dims (i, i + head_dim//2)
    head_dim = cos_pos.shape[-1]
    half = head_dim // 2
    unique_types = type_ids.unique()
    for type_val in unique_types:
        angle = type_val.float().item() * rotation_angle
        mask = (type_ids == type_val)  # (batch, seq_len)
        mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)

        for sub_idx in target_subspaces:
            for dim_idx in [sub_idx, sub_idx + half]:
                cos_out[..., dim_idx:dim_idx+1] = torch.where(
                    mask_expanded.expand_as(cos_out[..., dim_idx:dim_idx+1]),
                    torch.full_like(cos_out[..., dim_idx:dim_idx+1], math.cos(angle)),
                    cos_out[..., dim_idx:dim_idx+1]
                )
                sin_out[..., dim_idx:dim_idx+1] = torch.where(
                    mask_expanded.expand_as(sin_out[..., dim_idx:dim_idx+1]),
                    torch.full_like(sin_out[..., dim_idx:dim_idx+1], math.sin(angle)),
                    sin_out[..., dim_idx:dim_idx+1]
                )

    # Apply combined rotation to Q and K
    # RoPE rotation: x_rotated = x * cos + rotate_half(x) * sin
    q_rotated = _apply_rotary_emb(q, cos_out, sin_out)
    k_rotated = _apply_rotary_emb(k, cos_out, sin_out)

    return q_rotated, k_rotated


def _rotate_half(x):
    """Rotate half the hidden dims of the input (standard RoPE helper)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings to input tensor.

    Args:
        x: tensor of shape (..., head_dim)
        cos: tensor broadcastable to x shape
        sin: tensor broadcastable to x shape

    Returns:
        rotated tensor of same shape as x
    """
    # Expand cos/sin to match x dimensions
    # x shape: (batch, num_heads, seq_len, head_dim)
    # cos/sin shape: (batch, seq_len, head_dim) or (1, seq_len, head_dim)
    if cos.dim() == 3 and x.dim() == 4:
        cos = cos.unsqueeze(1)  # (batch, 1, seq_len, head_dim)
        sin = sin.unsqueeze(1)
    return x * cos + _rotate_half(x) * sin
