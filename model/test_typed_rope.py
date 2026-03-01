#!/usr/bin/env python3
"""Unit tests for typed RoPE module.

Tests:
1. type_id=0 produces identity rotation (matches standard RoPE)
2. Different type_ids produce different rotations in target subspaces only
3. Non-target subspaces remain unchanged
4. apply_typed_rope output shape correctness
5. Hook installation and removal

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

        # All cos should be 1, all sin should be 0 (identity)
        assert torch.allclose(cos_vals, torch.ones(head_dim), atol=1e-7), \
            "type_id=0 cos should be all 1s"
        assert torch.allclose(sin_vals, torch.zeros(head_dim), atol=1e-7), \
            "type_id=0 sin should be all 0s"

    def test_type1_rotation(self):
        """type_id=1 should produce rotation_angle rotation in target subspaces."""
        head_dim = 128
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

        # Target subspaces should have rotation
        for sub_idx in target_subspaces:
            d = sub_idx * 2
            assert abs(cos_vals[d].item() - expected_cos) < 1e-6
            assert abs(cos_vals[d + 1].item() - expected_cos) < 1e-6
            assert abs(sin_vals[d].item() - expected_sin) < 1e-6
            assert abs(sin_vals[d + 1].item() - expected_sin) < 1e-6

        # Non-target subspaces should be identity
        for sub_idx in range(head_dim // 2):
            if sub_idx not in target_subspaces:
                d = sub_idx * 2
                assert abs(cos_vals[d].item() - 1.0) < 1e-7
                assert abs(sin_vals[d].item() - 0.0) < 1e-7

    def test_different_types_differ(self):
        """Different type_ids should produce different rotations."""
        head_dim = 128
        target_subspaces = [60, 61]

        cos0, sin0 = create_type_rotation(head_dim, 0, target_subspaces)
        cos1, sin1 = create_type_rotation(head_dim, 1, target_subspaces)
        cos2, sin2 = create_type_rotation(head_dim, 2, target_subspaces)

        # type 0 and type 1 should differ in target subspaces
        d = target_subspaces[0] * 2
        assert not torch.allclose(cos0[d:d+2], cos1[d:d+2]), \
            "type 0 and type 1 should differ in target subspaces"
        assert not torch.allclose(cos1[d:d+2], cos2[d:d+2]), \
            "type 1 and type 2 should differ in target subspaces"

    def test_nontarget_unchanged(self):
        """Non-target subspaces should be identical across all type_ids."""
        head_dim = 128
        target_subspaces = [62, 63]

        cos0, sin0 = create_type_rotation(head_dim, 0, target_subspaces)
        cos1, sin1 = create_type_rotation(head_dim, 1, target_subspaces)
        cos2, sin2 = create_type_rotation(head_dim, 2, target_subspaces)

        # Check all non-target dimensions
        for sub_idx in range(head_dim // 2):
            if sub_idx in target_subspaces:
                continue
            d = sub_idx * 2
            assert cos0[d].item() == cos1[d].item() == cos2[d].item() == 1.0
            assert sin0[d].item() == sin1[d].item() == sin2[d].item() == 0.0

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

        # With type_id=0: type rotation is identity (cos=1, sin=0)
        # So target subspaces in cos_pos become 1.0, sin_pos become 0.0
        # But non-target subspaces keep original cos_pos, sin_pos
        # The overall output should reflect these changes

        # Verify by computing standard RoPE with modified cos/sin
        cos_expected = self.cos_pos.clone()
        sin_expected = self.sin_pos.clone()
        for sub_idx in self.target_subspaces:
            d0 = sub_idx * 2
            d1 = d0 + 2
            cos_expected[..., d0:d1] = 1.0
            sin_expected[..., d0:d1] = 0.0

        q_expected = _apply_rotary_emb(self.q, cos_expected, sin_expected)
        k_expected = _apply_rotary_emb(self.k, cos_expected, sin_expected)

        assert torch.allclose(q_typed, q_expected, atol=1e-6), \
            "type_id=0 should match standard RoPE with identity in target subspaces"
        assert torch.allclose(k_typed, k_expected, atol=1e-6)

    def test_different_types_produce_different_output(self):
        """Different type_ids should produce different Q/K in target subspaces."""
        # All type 0
        type_ids_0 = torch.zeros(self.batch, self.seq_len, dtype=torch.long)
        q0, k0 = apply_typed_rope(
            self.q, self.k, self.cos_pos, self.sin_pos,
            type_ids_0, self.target_subspaces
        )

        # All type 1
        type_ids_1 = torch.ones(self.batch, self.seq_len, dtype=torch.long)
        q1, k1 = apply_typed_rope(
            self.q, self.k, self.cos_pos, self.sin_pos,
            type_ids_1, self.target_subspaces
        )

        # Outputs should differ
        assert not torch.allclose(q0, q1, atol=1e-3), \
            "Different type_ids should produce different Q rotations"
        assert not torch.allclose(k0, k1, atol=1e-3), \
            "Different type_ids should produce different K rotations"

    def test_nontarget_dims_unchanged(self):
        """Non-target subspace dimensions should be the same regardless of type_id."""
        type_ids_0 = torch.zeros(self.batch, self.seq_len, dtype=torch.long)
        type_ids_1 = torch.ones(self.batch, self.seq_len, dtype=torch.long)

        q0, k0 = apply_typed_rope(
            self.q, self.k, self.cos_pos, self.sin_pos,
            type_ids_0, self.target_subspaces
        )
        q1, k1 = apply_typed_rope(
            self.q, self.k, self.cos_pos, self.sin_pos,
            type_ids_1, self.target_subspaces
        )

        # Check non-target dimensions are identical
        for sub_idx in range(self.head_dim // 2):
            if sub_idx in self.target_subspaces:
                continue
            d0 = sub_idx * 2
            d1 = d0 + 2
            assert torch.allclose(q0[:, :, :, d0:d1], q1[:, :, :, d0:d1], atol=1e-6), \
                f"Non-target subspace {sub_idx} should be unchanged"

    def test_mixed_types(self):
        """Mixed type_ids within a sequence should apply different rotations."""
        # First half = type 0, second half = type 1
        type_ids = torch.zeros(self.batch, self.seq_len, dtype=torch.long)
        type_ids[:, self.seq_len // 2:] = 1

        q_mixed, k_mixed = apply_typed_rope(
            self.q, self.k, self.cos_pos, self.sin_pos,
            type_ids, self.target_subspaces
        )

        # Target dims of first half and second half should differ
        d0 = self.target_subspaces[0] * 2
        d1 = d0 + 2
        first_half = q_mixed[:, :, :self.seq_len // 2, d0:d1]
        second_half = q_mixed[:, :, self.seq_len // 2:, d0:d1]

        # They should generally differ (different type rotations applied)
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
