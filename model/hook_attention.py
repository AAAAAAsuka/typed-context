#!/usr/bin/env python3
"""Hook-based attention patching for typed RoPE at inference time.

Installs forward hooks on all attention layers to replace standard RoPE
with typed RoPE in target subspaces, allowing token type information to
be encoded in the attention mechanism.

Usage:
    from model.hook_attention import install_type_rotation_hooks

    hooks = install_type_rotation_hooks(model, target_subspaces, rotation_angle)
    # Set type_ids before inference
    hooks.set_type_ids(type_ids)
    # Run inference
    output = model(input_ids)
    # Clean up
    hooks.remove()
"""

import math
from typing import List, Optional

import torch

from model.typed_rope import create_type_rotation


class TypedRoPEHooks:
    """Manages typed RoPE hooks across all attention layers.

    Patches rotary embedding modules to inject type rotation into target
    subspaces while preserving position rotation in all other subspaces.
    """

    def __init__(self, target_subspaces, rotation_angle=math.pi / 4):
        self.target_subspaces = target_subspaces
        self.rotation_angle = rotation_angle
        self._type_ids = None  # (batch, seq_len) tensor
        self._original_forwards = {}  # layer_idx -> original forward method
        self._installed = False

    def set_type_ids(self, type_ids):
        """Set type IDs for the current batch.

        Args:
            type_ids: tensor of shape (batch, seq_len) or (seq_len,)
                      0=system, 1=user, 2=external, ...
        """
        if isinstance(type_ids, list):
            type_ids = torch.tensor(type_ids)
        if type_ids.dim() == 1:
            type_ids = type_ids.unsqueeze(0)
        self._type_ids = type_ids

    def _find_rotary_modules(self, model):
        """Find rotary embedding modules in the model.

        Supports two architectures:
        - Per-layer: rotary_emb on each self_attn (older Llama-style)
        - Model-level: single rotary_emb on model.model (Qwen3, newer Llama)

        Also handles PEFT-wrapped models where the inner model is at
        model.model.model instead of model.model.

        Returns:
            list of (key, rotary_module) tuples
        """
        # Resolve the inner transformer model (handles PEFT wrapping)
        inner = model.model
        if not hasattr(inner, 'layers') and hasattr(inner, 'model'):
            inner = inner.model

        modules = []

        # Check per-layer first
        for layer_idx, layer in enumerate(inner.layers):
            attn = layer.self_attn
            if hasattr(attn, 'rotary_emb'):
                modules.append((f"layer_{layer_idx}", attn.rotary_emb))

        # If no per-layer rotary_emb found, check model-level
        if not modules and hasattr(inner, 'rotary_emb'):
            modules.append(("model_level", inner.rotary_emb))

        return modules

    def install(self, model):
        """Install typed RoPE hooks on rotary embedding modules.

        Supports both per-layer and model-level rotary embedding architectures
        (Llama, Qwen3, etc.).
        """
        target_subs = self.target_subspaces
        rotation_angle = self.rotation_angle
        hooks_ref = self  # reference for accessing type_ids

        rotary_modules = self._find_rotary_modules(model)
        if not rotary_modules:
            raise RuntimeError(
                "No rotary embedding modules found. "
                "Checked model.model.layers[*].self_attn.rotary_emb "
                "and model.model.rotary_emb"
            )

        for key, rotary in rotary_modules:
            original_forward = rotary.forward
            self._original_forwards[key] = (rotary, original_forward)

            def make_hooked_forward(orig_fwd, subs, angle, hooks):
                def hooked_forward(x, position_ids):
                    # Get standard RoPE cos/sin
                    cos_pos, sin_pos = orig_fwd(x, position_ids)

                    # If no type_ids set, return standard RoPE
                    if hooks._type_ids is None:
                        return cos_pos, sin_pos

                    type_ids = hooks._type_ids
                    device = cos_pos.device
                    if type_ids.device != device:
                        type_ids = type_ids.to(device)

                    # Handle KV-cache generation: cos_pos may have fewer
                    # positions than type_ids (e.g. seq_len=1 during
                    # autoregressive decoding). Use position_ids to select
                    # the matching slice of type_ids.
                    cos_seq_len = cos_pos.shape[-2]
                    type_seq_len = type_ids.shape[-1]
                    if cos_seq_len < type_seq_len:
                        # position_ids: (batch, cur_seq_len) — indices into
                        # the full sequence. Gather the corresponding types.
                        pos = position_ids
                        if pos.dim() == 1:
                            pos = pos.unsqueeze(0)
                        # Clamp to valid range for safety
                        pos = pos.clamp(max=type_seq_len - 1)
                        type_ids = type_ids.gather(1, pos)  # (batch, cur_seq_len)

                    # Override target subspaces with type rotation
                    cos_out = cos_pos.clone()
                    sin_out = sin_pos.clone()

                    # Half-half convention: subspace i → dims (i, i + head_dim//2)
                    head_dim = cos_out.shape[-1]
                    half_dim = head_dim // 2

                    for type_val in type_ids.unique():
                        type_angle = type_val.float().item() * angle
                        mask = (type_ids == type_val)  # (batch, seq_len)
                        mask_exp = mask.unsqueeze(-1)  # (batch, seq_len, 1)

                        for sub_idx in subs:
                            for dim_idx in [sub_idx, sub_idx + half_dim]:
                                cos_out[..., dim_idx:dim_idx+1] = torch.where(
                                    mask_exp.expand_as(
                                        cos_out[..., dim_idx:dim_idx+1]),
                                    torch.full_like(
                                        cos_out[..., dim_idx:dim_idx+1],
                                        math.cos(type_angle)),
                                    cos_out[..., dim_idx:dim_idx+1]
                                )
                                sin_out[..., dim_idx:dim_idx+1] = torch.where(
                                    mask_exp.expand_as(
                                        sin_out[..., dim_idx:dim_idx+1]),
                                    torch.full_like(
                                        sin_out[..., dim_idx:dim_idx+1],
                                        math.sin(type_angle)),
                                    sin_out[..., dim_idx:dim_idx+1]
                                )

                    return cos_out, sin_out

                return hooked_forward

            rotary.forward = make_hooked_forward(
                original_forward, target_subs, rotation_angle, hooks_ref
            )

        self._installed = True
        return self

    def remove(self):
        """Remove all hooks and restore original rotary embedding forwards."""
        for layer_idx, (rotary, original_forward) in self._original_forwards.items():
            rotary.forward = original_forward
        self._original_forwards.clear()
        self._type_ids = None
        self._installed = False

    @property
    def installed(self):
        return self._installed


def install_type_rotation_hooks(model, target_subspaces, rotation_angle=math.pi / 4):
    """Convenience function to install typed RoPE hooks.

    Args:
        model: HuggingFace causal LM (Llama-style)
        target_subspaces: list of subspace indices for type encoding
        rotation_angle: base rotation angle per type increment

    Returns:
        TypedRoPEHooks instance (call .set_type_ids() before inference,
        .remove() when done)
    """
    hooks = TypedRoPEHooks(target_subspaces, rotation_angle)
    hooks.install(model)
    return hooks
