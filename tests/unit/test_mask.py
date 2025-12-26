"""Unit tests for the Mask class."""

import numpy as np
import pytest

from psfed.core.mask import Mask


class TestMaskCreation:
    """Tests for Mask creation methods."""

    def test_from_bool_array(self):
        """Test creating mask from boolean array."""
        data = np.array([True, False, True, False, True])
        mask = Mask(data=data)

        assert mask.size == 5
        assert mask.count == 3
        assert np.array_equal(mask.indices, [0, 2, 4])

    def test_from_int_array_converts_to_bool(self):
        """Test that integer arrays are converted to bool."""
        data = np.array([1, 0, 1, 0])
        mask = Mask(data=data)

        assert mask.data.dtype == bool
        assert mask.count == 2

    def test_from_list(self):
        """Test creating mask from Python list."""
        data = [True, False, True]
        mask = Mask(data=data)

        assert mask.size == 3
        assert mask.count == 2

    def test_from_indices(self):
        """Test creating mask from index list."""
        mask = Mask.from_indices([0, 5, 10], size=20)

        assert mask.size == 20
        assert mask.count == 3
        assert np.array_equal(mask.indices, [0, 5, 10])

    def test_from_indices_empty(self):
        """Test creating mask with no indices."""
        mask = Mask.from_indices([], size=10)

        assert mask.size == 10
        assert mask.count == 0

    def test_from_indices_out_of_bounds_raises(self):
        """Test that out-of-bounds indices raise error."""
        with pytest.raises(ValueError, match="Indices must be in range"):
            Mask.from_indices([0, 100], size=50)

    def test_from_indices_negative_raises(self):
        """Test that negative indices raise error."""
        with pytest.raises(ValueError, match="Indices must be in range"):
            Mask.from_indices([-1, 0], size=10)

    def test_random_mask(self):
        """Test random mask generation."""
        mask = Mask.random(size=100, fraction=0.3, seed=42)

        assert mask.size == 100
        assert mask.count == 30  # Exact with seed

    def test_random_mask_reproducible(self):
        """Test that random masks are reproducible with seed."""
        mask1 = Mask.random(size=100, fraction=0.5, seed=42)
        mask2 = Mask.random(size=100, fraction=0.5, seed=42)

        assert mask1 == mask2

    def test_random_mask_different_rounds(self):
        """Test that different rounds produce different masks."""
        mask1 = Mask.random(size=100, fraction=0.5, seed=42, round_num=1)
        mask2 = Mask.random(size=100, fraction=0.5, seed=42, round_num=2)

        assert mask1 != mask2

    def test_random_mask_fraction_bounds(self):
        """Test fraction validation."""
        with pytest.raises(ValueError, match="Fraction must be"):
            Mask.random(size=100, fraction=1.5)

        with pytest.raises(ValueError, match="Fraction must be"):
            Mask.random(size=100, fraction=-0.1)

    def test_random_mask_size_bounds(self):
        """Test size validation."""
        with pytest.raises(ValueError, match="Size must be positive"):
            Mask.random(size=0, fraction=0.5)

    def test_all_true(self):
        """Test creating all-True mask."""
        mask = Mask.all_true(10)

        assert mask.size == 10
        assert mask.count == 10
        assert mask.fraction == 1.0

    def test_all_false(self):
        """Test creating all-False mask."""
        mask = Mask.all_false(10)

        assert mask.size == 10
        assert mask.count == 0
        assert mask.fraction == 0.0


class TestMaskProperties:
    """Tests for Mask properties."""

    def test_fraction(self):
        """Test fraction property."""
        mask = Mask.from_indices([0, 1, 2], size=10)
        assert mask.fraction == 0.3

    def test_fraction_empty(self):
        """Test fraction with empty mask."""
        mask = Mask.all_false(10)
        assert mask.fraction == 0.0

    def test_indices_sorted(self):
        """Test that indices are returned sorted."""
        mask = Mask.from_indices([5, 2, 8, 1], size=10)
        indices = mask.indices

        assert np.array_equal(indices, [1, 2, 5, 8])


class TestMaskOperations:
    """Tests for Mask logical operations."""

    def test_and(self):
        """Test AND operation."""
        mask1 = Mask.from_indices([0, 1, 2], size=5)
        mask2 = Mask.from_indices([1, 2, 3], size=5)

        result = mask1 & mask2

        assert result.count == 2
        assert np.array_equal(result.indices, [1, 2])

    def test_or(self):
        """Test OR operation."""
        mask1 = Mask.from_indices([0, 1], size=5)
        mask2 = Mask.from_indices([3, 4], size=5)

        result = mask1 | mask2

        assert result.count == 4
        assert np.array_equal(result.indices, [0, 1, 3, 4])

    def test_invert(self):
        """Test NOT operation."""
        mask = Mask.from_indices([0, 1], size=5)
        inverted = ~mask

        assert inverted.count == 3
        assert np.array_equal(inverted.indices, [2, 3, 4])

    def test_and_size_mismatch_raises(self):
        """Test that AND with mismatched sizes raises error."""
        mask1 = Mask.all_true(5)
        mask2 = Mask.all_true(10)

        with pytest.raises(ValueError, match="sizes must match"):
            _ = mask1 & mask2

    def test_or_size_mismatch_raises(self):
        """Test that OR with mismatched sizes raises error."""
        mask1 = Mask.all_true(5)
        mask2 = Mask.all_true(10)

        with pytest.raises(ValueError, match="sizes must match"):
            _ = mask1 | mask2


class TestMaskSerialization:
    """Tests for Mask serialization."""

    def test_to_dict(self):
        """Test serialization to dict."""
        mask = Mask.from_indices([0, 5, 10], size=20)
        data = mask.to_dict()

        assert data["size"] == 20
        assert data["indices"] == [0, 5, 10]

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {"indices": [1, 3, 5], "size": 10}
        mask = Mask.from_dict(data)

        assert mask.size == 10
        assert mask.count == 3
        assert np.array_equal(mask.indices, [1, 3, 5])

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = Mask.random(size=100, fraction=0.4, seed=42)
        data = original.to_dict()
        restored = Mask.from_dict(data)

        assert original == restored


class TestMaskEquality:
    """Tests for Mask equality and hashing."""

    def test_equality(self):
        """Test mask equality."""
        mask1 = Mask.from_indices([0, 1, 2], size=10)
        mask2 = Mask.from_indices([0, 1, 2], size=10)

        assert mask1 == mask2

    def test_inequality_different_indices(self):
        """Test masks with different indices are not equal."""
        mask1 = Mask.from_indices([0, 1, 2], size=10)
        mask2 = Mask.from_indices([0, 1, 3], size=10)

        assert mask1 != mask2

    def test_inequality_different_size(self):
        """Test masks with different sizes are not equal."""
        mask1 = Mask.from_indices([0, 1], size=10)
        mask2 = Mask.from_indices([0, 1], size=20)

        assert mask1 != mask2

    def test_hash_equal_masks(self):
        """Test that equal masks have equal hashes."""
        mask1 = Mask.from_indices([0, 1, 2], size=10)
        mask2 = Mask.from_indices([0, 1, 2], size=10)

        assert hash(mask1) == hash(mask2)

    def test_usable_in_set(self):
        """Test that masks can be used in sets."""
        mask1 = Mask.from_indices([0, 1], size=10)
        mask2 = Mask.from_indices([0, 1], size=10)
        mask3 = Mask.from_indices([0, 2], size=10)

        mask_set = {mask1, mask2, mask3}
        assert len(mask_set) == 2


class TestMaskRepr:
    """Tests for Mask string representation."""

    def test_repr(self):
        """Test string representation."""
        mask = Mask.from_indices([0, 1, 2], size=10)
        repr_str = repr(mask)

        assert "count=3" in repr_str
        assert "size=10" in repr_str
        assert "30.00%" in repr_str
