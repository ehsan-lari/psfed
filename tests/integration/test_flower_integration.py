"""Integration tests for Flower integration."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from psfed import FlattenedModel, Mask, RandomMaskSelector
from psfed.flower.numpy_utils import (
    flat_array_to_ndarrays,
    mask_from_config,
    mask_to_config,
    ndarrays_to_flat_array,
)


class TestMaskSerialization:
    """Tests for mask serialization in Flower configs."""

    def test_mask_to_config(self):
        """Test mask serialization to config dict."""
        mask = Mask.from_indices([0, 5, 10, 15], size=100)
        config = mask_to_config(mask)

        assert config["mask_size"] == 100
        assert config["mask_indices"] == "0,5,10,15"

    def test_mask_from_config(self):
        """Test mask deserialization from config dict."""
        config = {"mask_indices": "1,3,5,7", "mask_size": 20}
        mask = mask_from_config(config)

        assert mask.size == 20
        assert mask.count == 4
        assert np.array_equal(mask.indices, [1, 3, 5, 7])

    def test_mask_roundtrip(self):
        """Test mask serialization roundtrip."""
        original = Mask.random(size=1000, fraction=0.3, seed=42)
        config = mask_to_config(original)
        restored = mask_from_config(config)

        assert original == restored

    def test_empty_mask_serialization(self):
        """Test serialization of empty mask."""
        mask = Mask.all_false(100)
        config = mask_to_config(mask)
        restored = mask_from_config(config)

        assert restored.count == 0
        assert restored.size == 100


class TestArrayConversion:
    """Tests for array conversion utilities."""

    def test_ndarrays_to_flat(self):
        """Test flattening multiple arrays."""
        arrays = [
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([5, 6, 7], dtype=np.float32),
        ]

        flat, shapes = ndarrays_to_flat_array(arrays)

        assert flat.shape == (7,)
        assert np.allclose(flat, [1, 2, 3, 4, 5, 6, 7])
        assert shapes == [(2, 2), (3,)]

    def test_flat_to_ndarrays(self):
        """Test reshaping flat array to multiple arrays."""
        flat = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
        shapes = [(2, 2), (3,)]

        arrays = flat_array_to_ndarrays(flat, shapes)

        assert len(arrays) == 2
        assert arrays[0].shape == (2, 2)
        assert arrays[1].shape == (3,)

    def test_conversion_roundtrip(self):
        """Test array conversion roundtrip."""
        original = [
            np.random.randn(3, 4).astype(np.float32),
            np.random.randn(5).astype(np.float32),
            np.random.randn(2, 2, 2).astype(np.float32),
        ]

        flat, shapes = ndarrays_to_flat_array(original)
        restored = flat_array_to_ndarrays(flat, shapes)

        for orig, rest in zip(original, restored):
            assert np.allclose(orig, rest)


class TestEndToEndPartialSharing:
    """End-to-end tests simulating federated learning rounds."""

    def test_single_round_simulation(self, simple_model):
        """Simulate a single round of partial sharing."""
        # Server setup
        server_model = simple_model
        server_flat = FlattenedModel(server_model)
        selector = RandomMaskSelector(fraction=0.5, seed=42)

        # Generate mask for round 1
        mask = selector.select(server_flat.num_parameters, round_num=1)

        # Server extracts partial parameters
        server_params = server_flat.extract(mask)

        # Simulate sending to client
        client_model = type(simple_model)()  # Fresh model
        client_flat = FlattenedModel(client_model)

        # Client applies partial parameters
        client_flat.apply(server_params, mask)

        # Verify client's masked positions match server
        client_params = client_flat.extract(mask)
        assert np.allclose(client_params, server_params)

        # Verify non-masked positions are different (client's original)
        inv_mask = ~mask
        server_unmasked = server_flat.extract(inv_mask)
        client_unmasked = client_flat.extract(inv_mask)
        # Should NOT be equal (client kept its own values)
        assert not np.allclose(server_unmasked, client_unmasked)

    def test_multi_round_coverage(self, simple_model):
        """Test that multiple rounds cover all parameters."""
        flat = FlattenedModel(simple_model)
        selector = RandomMaskSelector(fraction=0.3, seed=42)

        covered = np.zeros(flat.num_parameters, dtype=bool)

        for round_num in range(1, 15):
            mask = selector.select(flat.num_parameters, round_num)
            covered |= mask.data

        # Should have high coverage after 15 rounds with 30% selection
        coverage = np.mean(covered)
        assert coverage > 0.95, f"Coverage only {coverage:.1%} after 15 rounds"

    def test_aggregation_simulation(self, simple_model):
        """Simulate FedAvg aggregation with partial parameters."""
        # Setup
        selector = RandomMaskSelector(fraction=0.5, seed=42)
        num_clients = 3

        # Server model
        server_flat = FlattenedModel(simple_model)
        mask = selector.select(server_flat.num_parameters, round_num=1)

        # Simulate client updates (each adds different offset)
        client_updates = []
        weights = [100, 200, 300]  # Different data sizes

        for i in range(num_clients):
            # Client would train and return updated params
            # Here we just simulate with offsets
            update = server_flat.extract(mask) + (i + 1) * 0.1
            client_updates.append(update)

        # FedAvg aggregation
        total_weight = sum(weights)
        aggregated = np.zeros(mask.count, dtype=np.float32)
        for update, weight in zip(client_updates, weights):
            aggregated += (weight / total_weight) * update

        # Expected: weighted average of offsets
        expected_offset = sum(w * (i + 1) * 0.1 for i, w in enumerate(weights)) / total_weight
        original = server_flat.extract(mask)

        assert np.allclose(aggregated, original + expected_offset, rtol=1e-5)


class TestDynamicMaskProperties:
    """Tests for dynamic mask behavior."""

    def test_masks_are_different_each_round(self):
        """Verify that different rounds produce different masks."""
        selector = RandomMaskSelector(fraction=0.5, seed=42)
        num_params = 1000

        masks = [selector.select(num_params, round_num=r) for r in range(1, 6)]

        # All masks should be different
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                assert masks[i] != masks[j], f"Round {i+1} and {j+1} have same mask"

    def test_mask_count_is_consistent(self):
        """Verify mask count is consistent across rounds."""
        selector = RandomMaskSelector(fraction=0.3, seed=42)
        num_params = 1000

        for round_num in range(1, 20):
            mask = selector.select(num_params, round_num)
            assert mask.count == 300, f"Round {round_num}: expected 300, got {mask.count}"

    def test_reproducibility_across_runs(self):
        """Verify same seed produces same sequence of masks."""
        num_params = 1000

        selector1 = RandomMaskSelector(fraction=0.5, seed=42)
        selector2 = RandomMaskSelector(fraction=0.5, seed=42)

        for round_num in range(1, 10):
            mask1 = selector1.select(num_params, round_num)
            mask2 = selector2.select(num_params, round_num)
            assert mask1 == mask2, f"Mismatch at round {round_num}"
