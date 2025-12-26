"""Unit tests for mask selector classes."""

import numpy as np
import pytest

from psfed.core.flattener import FlattenedModel
from psfed.core.mask import Mask
from psfed.core.selectors import (
    ClientSpecificMaskSelector,
    CompositeMaskSelector,
    FixedMaskSelector,
    GradientBasedSelector,
    MaskSelector,
    RandomMaskSelector,
    StructuredMaskSelector,
    TopKMagnitudeSelector,
)


class TestMaskSelectorBase:
    """Tests for MaskSelector base class."""

    def test_fraction_validation(self):
        """Test fraction validation in base class."""
        with pytest.raises(ValueError, match="Fraction must be"):
            RandomMaskSelector(fraction=1.5)

        with pytest.raises(ValueError, match="Fraction must be"):
            RandomMaskSelector(fraction=-0.1)

    def test_get_count(self):
        """Test count calculation."""
        selector = RandomMaskSelector(fraction=0.3)
        assert selector.get_count(100) == 30
        assert selector.get_count(10) == 3


class TestRandomMaskSelector:
    """Tests for RandomMaskSelector."""

    def test_basic_selection(self):
        """Test basic random selection."""
        selector = RandomMaskSelector(fraction=0.5, seed=42)
        mask = selector.select(num_parameters=100, round_num=1)

        assert mask.size == 100
        assert mask.count == 50

    def test_reproducibility(self):
        """Test that same seed produces same mask."""
        selector = RandomMaskSelector(fraction=0.5, seed=42)

        mask1 = selector.select(100, round_num=1)
        mask2 = selector.select(100, round_num=1)

        assert mask1 == mask2

    def test_different_rounds(self):
        """Test that different rounds produce different masks."""
        selector = RandomMaskSelector(fraction=0.5, seed=42)

        mask1 = selector.select(100, round_num=1)
        mask2 = selector.select(100, round_num=2)

        assert mask1 != mask2

    def test_coverage_over_rounds(self):
        """Test that multiple rounds eventually cover all parameters."""
        selector = RandomMaskSelector(fraction=0.3, seed=42)
        combined = Mask.all_false(100)

        # Run many rounds
        for round_num in range(1, 20):
            mask = selector.select(100, round_num)
            combined = combined | mask

        # Should have high coverage
        assert combined.fraction > 0.95

    def test_no_seed_different_each_time(self):
        """Test that no seed produces different masks."""
        selector = RandomMaskSelector(fraction=0.5, seed=None)

        # Without seed, each call should be different (with high probability)
        masks = [selector.select(100, round_num=i) for i in range(5)]

        # At least some should be different
        unique_masks = len(set(hash(m) for m in masks))
        assert unique_masks > 1


class TestTopKMagnitudeSelector:
    """Tests for TopKMagnitudeSelector."""

    def test_basic_selection(self, flat_simple_model, simple_model):
        """Test top-k by magnitude selection."""
        # Set some known values
        with torch.no_grad():
            simple_model.fc1.weight.fill_(1.0)
            simple_model.fc1.bias.fill_(10.0)  # Larger magnitude

        selector = TopKMagnitudeSelector(fraction=0.1)
        mask = selector.select(
            num_parameters=flat_simple_model.num_parameters,
            round_num=1,
            flat_model=flat_simple_model,
        )

        # Should select ~7 parameters (10% of 67)
        assert mask.count == 7

        # The 5 bias parameters (value 10) should definitely be selected
        # Check that indices 50-54 (fc1.bias) are in the mask
        bias_indices = set(range(50, 55))
        selected_indices = set(mask.indices.tolist())
        assert bias_indices.issubset(selected_indices)

    def test_missing_flat_model_raises(self):
        """Test that missing flat_model raises error."""
        selector = TopKMagnitudeSelector(fraction=0.5)

        with pytest.raises(ValueError, match="requires 'flat_model'"):
            selector.select(100, round_num=1)


class TestGradientBasedSelector:
    """Tests for GradientBasedSelector."""

    def test_basic_selection(self):
        """Test gradient-based selection."""
        selector = GradientBasedSelector(fraction=0.3)

        # Create gradient array with known values
        gradients = np.zeros(100)
        gradients[10:20] = 10.0  # High gradients

        mask = selector.select(100, round_num=1, gradients=gradients)

        # Should select 30 parameters
        assert mask.count == 30

        # The high-gradient positions should be selected
        high_grad_indices = set(range(10, 20))
        selected_indices = set(mask.indices.tolist())
        assert high_grad_indices.issubset(selected_indices)

    def test_missing_gradients_raises(self):
        """Test that missing gradients raises error."""
        selector = GradientBasedSelector(fraction=0.5)

        with pytest.raises(ValueError, match="requires 'gradients'"):
            selector.select(100, round_num=1)


class TestStructuredMaskSelector:
    """Tests for StructuredMaskSelector."""

    def test_specific_layers(self, flat_simple_model):
        """Test selecting specific layers."""
        selector = StructuredMaskSelector(
            fraction=0.5,
            layer_names=["fc1"],
        )

        mask = selector.select(
            num_parameters=flat_simple_model.num_parameters,
            round_num=1,
            flat_model=flat_simple_model,
        )

        # Should select fc1 parameters (55 out of 67)
        assert mask.count == 55

    def test_rotating_layers(self, flat_simple_model):
        """Test layer rotation across rounds."""
        selector = StructuredMaskSelector(fraction=0.5, rotate_layers=True)

        mask1 = selector.select(
            num_parameters=flat_simple_model.num_parameters,
            round_num=1,
            flat_model=flat_simple_model,
        )
        mask2 = selector.select(
            num_parameters=flat_simple_model.num_parameters,
            round_num=2,
            flat_model=flat_simple_model,
        )

        # Different rounds should select different layers
        assert mask1 != mask2

    def test_missing_flat_model_raises(self):
        """Test that missing flat_model raises error."""
        selector = StructuredMaskSelector(fraction=0.5)

        with pytest.raises(ValueError, match="requires 'flat_model'"):
            selector.select(100, round_num=1)


class TestFixedMaskSelector:
    """Tests for FixedMaskSelector."""

    def test_basic_selection(self):
        """Test fixed index selection."""
        indices = [0, 5, 10, 15]
        selector = FixedMaskSelector(indices=indices)

        mask = selector.select(num_parameters=100, round_num=1)

        assert mask.count == 4
        assert np.array_equal(mask.indices, sorted(indices))

    def test_same_every_round(self):
        """Test that same mask is returned every round."""
        selector = FixedMaskSelector(indices=[0, 1, 2])

        mask1 = selector.select(100, round_num=1)
        mask2 = selector.select(100, round_num=5)
        mask3 = selector.select(100, round_num=100)

        assert mask1 == mask2 == mask3


class TestClientSpecificMaskSelector:
    """Tests for ClientSpecificMaskSelector."""

    def test_different_clients_different_masks(self):
        """Test that different clients get different masks."""
        selector = ClientSpecificMaskSelector(fraction=0.5, seed=42)

        mask1 = selector.select(100, round_num=1, client_id="client_1")
        mask2 = selector.select(100, round_num=1, client_id="client_2")

        assert mask1 != mask2

    def test_same_client_same_round_reproducible(self):
        """Test reproducibility for same client and round."""
        selector = ClientSpecificMaskSelector(fraction=0.5, seed=42)

        mask1 = selector.select(100, round_num=1, client_id="client_1")
        mask2 = selector.select(100, round_num=1, client_id="client_1")

        assert mask1 == mask2

    def test_same_client_different_rounds(self):
        """Test that same client gets different masks in different rounds."""
        selector = ClientSpecificMaskSelector(fraction=0.5, seed=42)

        mask1 = selector.select(100, round_num=1, client_id="client_1")
        mask2 = selector.select(100, round_num=2, client_id="client_1")

        assert mask1 != mask2

    def test_no_client_id(self):
        """Test behavior when no client_id is provided."""
        selector = ClientSpecificMaskSelector(fraction=0.5, seed=42)

        # Should still work, using default client hash
        mask = selector.select(100, round_num=1)
        assert mask.count == 50


class TestCompositeMaskSelector:
    """Tests for CompositeMaskSelector."""

    def test_union_mode(self):
        """Test union (OR) of multiple selectors."""
        fixed1 = FixedMaskSelector(indices=[0, 1, 2])
        fixed2 = FixedMaskSelector(indices=[3, 4, 5])

        composite = CompositeMaskSelector([fixed1, fixed2], mode="union")
        mask = composite.select(100, round_num=1)

        assert mask.count == 6
        assert np.array_equal(mask.indices, [0, 1, 2, 3, 4, 5])

    def test_intersection_mode(self):
        """Test intersection (AND) of multiple selectors."""
        fixed1 = FixedMaskSelector(indices=[0, 1, 2, 3])
        fixed2 = FixedMaskSelector(indices=[2, 3, 4, 5])

        composite = CompositeMaskSelector([fixed1, fixed2], mode="intersection")
        mask = composite.select(100, round_num=1)

        assert mask.count == 2
        assert np.array_equal(mask.indices, [2, 3])

    def test_empty_selectors(self):
        """Test with no selectors."""
        composite = CompositeMaskSelector([], mode="union")
        mask = composite.select(100, round_num=1)

        assert mask.count == 0

    def test_invalid_mode_raises(self):
        """Test that invalid mode raises error."""
        with pytest.raises(ValueError, match="Mode must be"):
            CompositeMaskSelector([], mode="invalid")


# Import torch for fixtures
import torch
