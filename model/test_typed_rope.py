#!/usr/bin/env python3
"""Unit tests for typed RoPE module.

Tests:
1. type_id=0 produces identity rotation (matches standard RoPE)
2. Different type_ids produce different rotations in target subspaces only
3. Non-target subspaces remain unchanged
4. apply_typed_rope output shape correctness
5. cos(theta_A - theta_B) modulation property

Uses half-half dimension convention matching HuggingFace Llama RoPE:
  subspace i → dim pair (i, i + head_dim//2)

Usage:
    python -m pytest model/test_typed_rope.py -v
"""

import math
import sys
import os

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.typed_rope import create_type_rotation, apply_typed_rope, _apply_rotary_emb, _rotate_half


class TestCreateTypeRotation:
    """Tests for create_type_rotation()."""

    def test_type0_identity(self):
        """type_id=0 should produce identity rotation (cos=1, sin=0)."""
        head_dim = 128
        target_subspaces = [60, 61, 62, 63]

        cos_vals, sin_vals = create_type_rotation(
            head_dim=head_dim,
            type_id=0,
            target_subspaces=target_subspaces,
            rotation_angle=math.pi / 4
        )

        assert torch.allclose(cos_vals, torch.ones(head_dim), atol=1e-7), \
            "type_id=0 cos should be all 1s"
        assert torch.allclose(sin_vals, torch.zeros(head_dim), atol=1e-7), \
            "type_id=0 sin should be all 0s"

    def test_type1_rotation(self):
        """type_id=1 should produce rotation_angle rotation in target subspaces."""
        head_dim = 128
        half = head_dim // 2
        target_subspaces = [62, 63]
        rotation_angle = math.pi / 4

        cos_vals, sin_vals = create_type_rotation(
            head_dim=head_dim,
            type_id=1,
            target_subspaces=target_subspaces,
            rotation_angle=rotation_angle
        )

        expected_cos = math.cos(rotation_angle)
        expected_sin = math.sin(rotation_angle)

        # Target subspaces: both dim i and dim i+half should have rotation
        for sub_idx in target_subspaces:
            assert abs(cos_vals[sub_idx].item() - expected_cos) < 1e-6
            assert abs(cos_vals[sub_idx + half].item() - expected_cos) < 1e-6
            assert abs(sin_vals[sub_idx].item() - expected_sin) < 1e-6
            assert abs(sin_vals[sub_idx + half].item() - expected_sin) < 1e-6

        # Non-target subspaces should be identity
        for i in range(head_dim):
            # Skip target dims
            is_target = False
            for s in target_subspaces:
                if i == s or i == s + half:
                    is_target = True
                    break
            if not is_target:
                assert abs(cos_vals[i].item() - 1.0) < 1e-7
                assert abs(sin_vals[i].item() - 0.0) < 1e-7

    def test_different_types_differ(self):
        """Different type_ids should produce different rotations."""
        head_dim = 128
        target_subspaces = [60, 61]

        cos0, sin0 = create_type_rotation(head_dim, 0, target_subspaces)
        cos1, sin1 = create_type_rotation(head_dim, 1, target_subspaces)
        cos2, sin2 = create_type_rotation(head_dim, 2, target_subspaces)

        s = target_subspaces[0]
        assert not torch.allclose(cos0[s:s+1], cos1[s:s+1]), \
            "type 0 and type 1 should differ in target subspaces"
        assert not torch.allclose(cos1[s:s+1], cos2[s:s+1]), \
            "type 1 and type 2 should differ in target subspaces"

    def test_nontarget_unchanged(self):
        """Non-target subspaces should be identical across all type_ids."""
        head_dim = 128
        half = head_dim // 2
        target_subspaces = [62, 63]

        cos0, sin0 = create_type_rotation(head_dim, 0, target_subspaces)
        cos1, sin1 = create_type_rotation(head_dim, 1, target_subspaces)
        cos2, sin2 = create_type_rotation(head_dim, 2, target_subspaces)

        target_dims = set()
        for s in target_subspaces:
            target_dims.add(s)
            target_dims.add(s + half)

        for i in range(head_dim):
            if i in target_dims:
                continue
            assert cos0[i].item() == cos1[i].item() == cos2[i].item() == 1.0
            assert sin0[i].item() == sin1[i].item() == sin2[i].item() == 0.0

    def test_shape(self):
        """Output should have shape (head_dim,)."""
        head_dim = 128
        cos_vals, sin_vals = create_type_rotation(head_dim, 1, [0, 1])
        assert cos_vals.shape == (head_dim,)
        assert sin_vals.shape == (head_dim,)


class TestApplyTypedRoPE:
    """Tests for apply_typed_rope()."""

    def setup_method(self):
        """Set up common test fixtures."""
        self.batch = 2
        self.num_heads = 4
        self.seq_len = 16
        self.head_dim = 32  # smaller for testing
        self.half = self.head_dim // 2
        self.target_subspaces = [14, 15]  # last two subspaces of head_dim=32

        torch.manual_seed(42)
        self.q = torch.randn(self.batch, self.num_heads, self.seq_len, self.head_dim)
        self.k = torch.randn(self.batch, self.num_heads, self.seq_len, self.head_dim)

        # Simulate standard RoPE cos/sin
        self.cos_pos = torch.randn(self.batch, self.seq_len, self.head_dim) * 0.1 + 1.0
        self.sin_pos = torch.randn(self.batch, self.seq_len, self.head_dim) * 0.1

    def test_type0_matches_standard(self):
        """With all type_ids=0, output should match standard RoPE exactly."""
        type_ids = torch.zeros(self.batch, self.seq_len, dtype=torch.long)

        q_typed, k_typed = apply_typed_rope(
            self.q, self.k, self.cos_pos, self.sin_pos,
            type_ids, self.target_subspaces
        )

        # With type_id=0: type rotation is identity (cos=1, sin=0) in target dims
        cos_expected = self.cos_pos.clone()
        sin_expected = self.sin_pos.clone()
        for sub_idx in self.target_subspaces:
            cos_expected[..., sub_idx] = 1.0
            cos_expected[..., sub_idx + self.half] = 1.0
            sin_expected[..., sub_idx] = 0.0
            sin_expected[..., sub_idx + self.half] = 0.0

        q_expected = _apply_rotary_emb(self.q, cos_expected, sin_expected)
        k_expected = _apply_rotary_emb(self.k, cos_expected, sin_expected)

        assert torch.allclose(q_typed, q_expected, atol=1e-6), \
            "type_id=0 should match standard RoPE with identity in target subspaces"
        assert torch.allclose(k_typed, k_expected, atol=1e-6)

    def test_different_types_produce_different_output(self):
        """Different type_ids should produce different Q/K in target subspaces."""
        type_ids_0 = torch.zeros(self.batch, self.seq_len, dtype=torch.long)
        q0, k0 = apply_typed_rope(
            self.q, self.k, self.cos_pos, self.sin_pos,
            type_ids_0, self.target_subspaces
        )

        type_ids_1 = torch.ones(self.batch, self.seq_len, dtype=torch.long)
        q1, k1 = apply_typed_rope(
            self.q, self.k, self.cos_pos, self.sin_pos,
            type_ids_1, self.target_subspaces
        )

        assert not torch.allclose(q0, q1, atol=1e-3)
        assert not torch.allclose(k0, k1, atol=1e-3)

    def test_nontarget_dims_unchanged(self):
        """Non-target subspace dimensions should be same regardless of type_id."""
        type_ids_0 = torch.zeros(self.batch, self.seq_len, dtype=torch.long)
        type_ids_1 = torch.ones(self.batch, self.seq_len, dtype=torch.long)

        q0, _ = apply_typed_rope(
            self.q, self.k, self.cos_pos, self.sin_pos,
            type_ids_0, self.target_subspaces
        )
        q1, _ = apply_typed_rope(
            self.q, self.k, self.cos_pos, self.sin_pos,
            type_ids_1, self.target_subspaces
        )

        target_dims = set()
        for s in self.target_subspaces:
            target_dims.add(s)
            target_dims.add(s + self.half)

        for d in range(self.head_dim):
            if d in target_dims:
                continue
            assert torch.allclose(q0[:, :, :, d], q1[:, :, :, d], atol=1e-6), \
                f"Non-target dim {d} should be unchanged"

    def test_mixed_types(self):
        """Mixed type_ids within a sequence should apply different rotations."""
        type_ids = torch.zeros(self.batch, self.seq_len, dtype=torch.long)
        type_ids[:, self.seq_len // 2:] = 1

        q_mixed, _ = apply_typed_rope(
            self.q, self.k, self.cos_pos, self.sin_pos,
            type_ids, self.target_subspaces
        )

        s = self.target_subspaces[0]
        first_half = q_mixed[:, :, :self.seq_len // 2, s]
        second_half = q_mixed[:, :, self.seq_len // 2:, s]
        assert not torch.allclose(first_half, second_half, atol=1e-3)

    def test_output_shape(self):
        """Output tensors should have same shape as input."""
        type_ids = torch.zeros(self.batch, self.seq_len, dtype=torch.long)
        q_out, k_out = apply_typed_rope(
            self.q, self.k, self.cos_pos, self.sin_pos,
            type_ids, self.target_subspaces
        )
        assert q_out.shape == self.q.shape
        assert k_out.shape == self.k.shape


class TestCosModulationProperty:
    """Test the core theoretical property: cos(theta_A - theta_B) modulation."""

    def test_exact_cos_modulation(self):
        """Dot product in target subspaces should be modulated by cos(θ_A - θ_B).

        For matched Q=K vectors, the cross term (ad-bc) is zero, so the modulation
        should be exactly cos(θ_A - θ_B).
        """
        head_dim = 128
        half = head_dim // 2
        target_subspaces = [59, 60, 61, 62, 63]
        rotation_angle = math.pi / 4

        torch.manual_seed(42)

        # Target dim indices in the half-half convention
        target_dims = []
        for s in target_subspaces:
            target_dims.extend([s, s + half])

        for type_a in range(3):
            for type_b in range(3):
                # Generate random Q=K vectors
                q = torch.randn(1, 1, 100, head_dim)
                k = q.clone()  # K=Q → cross term = 0

                cos_a, sin_a = create_type_rotation(head_dim, type_a, target_subspaces, rotation_angle)
                cos_b, sin_b = create_type_rotation(head_dim, type_b, target_subspaces, rotation_angle)

                q_rot = q * cos_a.view(1, 1, 1, -1) + _rotate_half(q) * sin_a.view(1, 1, 1, -1)
                k_rot = k * cos_b.view(1, 1, 1, -1) + _rotate_half(k) * sin_b.view(1, 1, 1, -1)

                # Dot product in target dims
                dot_ab = (q_rot[..., target_dims] * k_rot[..., target_dims]).sum(dim=-1)
                dot_aa = (q_rot[..., target_dims] * q_rot[..., target_dims]).sum(dim=-1)

                ratio = (dot_ab / dot_aa).mean().item()
                expected = math.cos((type_a - type_b) * rotation_angle)

                assert abs(ratio - expected) < 1e-5, \
                    f"type ({type_a},{type_b}): ratio={ratio:.6f}, expected={expected:.6f}"


class TestHelpers:
    """Tests for helper functions."""

    def test_rotate_half(self):
        """_rotate_half should swap and negate halves."""
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = _rotate_half(x)
        expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
        assert torch.allclose(result, expected)

    def test_apply_rotary_identity(self):
        """Identity rotation (cos=1, sin=0) should preserve input."""
        x = torch.randn(1, 1, 4, 8)
        cos = torch.ones(1, 4, 8)
        sin = torch.zeros(1, 4, 8)
        result = _apply_rotary_emb(x, cos, sin)
        assert torch.allclose(result, x, atol=1e-6), \
            "Identity rotation should preserve input"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
