"""Unit tests for the FlattenedModel class."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from psfed.core.flattener import FlattenedModel
from psfed.core.mask import Mask


class TestFlattenedModelCreation:
    """Tests for FlattenedModel initialization."""

    def test_num_parameters(self, simple_model):
        """Test parameter counting."""
        flat = FlattenedModel(simple_model)

        # fc1: 10*5 + 5 = 55, fc2: 5*2 + 2 = 12, total = 67
        assert flat.num_parameters == 67

    def test_param_shapes(self, simple_model):
        """Test parameter shape tracking."""
        flat = FlattenedModel(simple_model)

        shapes = flat.param_shapes
        assert len(shapes) == 4  # fc1.weight, fc1.bias, fc2.weight, fc2.bias

        # Check shapes
        assert shapes[0] == ("fc1.weight", torch.Size([5, 10]))
        assert shapes[1] == ("fc1.bias", torch.Size([5]))
        assert shapes[2] == ("fc2.weight", torch.Size([2, 5]))
        assert shapes[3] == ("fc2.bias", torch.Size([2]))

    def test_param_numels(self, simple_model):
        """Test parameter element counts."""
        flat = FlattenedModel(simple_model)

        numels = flat.param_numels
        assert numels == [50, 5, 10, 2]

    def test_conv_model(self, conv_model):
        """Test with convolutional model."""
        flat = FlattenedModel(conv_model)

        assert flat.num_parameters > 0
        assert len(flat.param_shapes) > 0


class TestFlattenUnflatten:
    """Tests for flatten and unflatten operations."""

    def test_flatten_shape(self, flat_simple_model):
        """Test that flatten returns correct shape."""
        flat_params = flat_simple_model.flatten()

        assert flat_params.shape == (67,)
        assert flat_params.dtype == np.float32 or flat_params.dtype == np.float64

    def test_flatten_values(self, simple_model):
        """Test that flatten returns correct values."""
        # Set known values
        with torch.no_grad():
            simple_model.fc1.weight.fill_(1.0)
            simple_model.fc1.bias.fill_(2.0)

        flat = FlattenedModel(simple_model)
        flat_params = flat.flatten()

        # First 50 should be 1.0 (fc1.weight)
        assert np.allclose(flat_params[:50], 1.0)
        # Next 5 should be 2.0 (fc1.bias)
        assert np.allclose(flat_params[50:55], 2.0)

    def test_unflatten_updates_model(self, flat_simple_model, simple_model):
        """Test that unflatten updates model parameters."""
        # Create new parameter values
        new_params = np.ones(67, dtype=np.float32) * 3.0

        flat_simple_model.unflatten(new_params)

        # Check model was updated
        assert torch.allclose(simple_model.fc1.weight, torch.full((5, 10), 3.0))
        assert torch.allclose(simple_model.fc1.bias, torch.full((5,), 3.0))

    def test_flatten_unflatten_roundtrip(self, flat_simple_model):
        """Test that flatten/unflatten is lossless."""
        original = flat_simple_model.flatten().copy()

        # Modify and restore
        flat_simple_model.unflatten(original)
        restored = flat_simple_model.flatten()

        assert np.allclose(original, restored)

    def test_unflatten_wrong_size_raises(self, flat_simple_model):
        """Test that wrong-sized array raises error."""
        wrong_size = np.zeros(100)

        with pytest.raises(ValueError, match="Expected 67"):
            flat_simple_model.unflatten(wrong_size)


class TestExtractApply:
    """Tests for partial parameter extraction and application."""

    def test_extract_all(self, flat_simple_model):
        """Test extracting all parameters."""
        mask = Mask.all_true(67)
        extracted = flat_simple_model.extract(mask)

        assert len(extracted) == 67
        assert np.allclose(extracted, flat_simple_model.flatten())

    def test_extract_none(self, flat_simple_model):
        """Test extracting no parameters."""
        mask = Mask.all_false(67)
        extracted = flat_simple_model.extract(mask)

        assert len(extracted) == 0

    def test_extract_partial(self, flat_simple_model):
        """Test extracting partial parameters."""
        mask = Mask.from_indices([0, 1, 2, 10, 20], size=67)
        extracted = flat_simple_model.extract(mask)

        assert len(extracted) == 5

    def test_extract_mask_size_mismatch_raises(self, flat_simple_model):
        """Test that mismatched mask size raises error."""
        wrong_mask = Mask.all_true(100)

        with pytest.raises(ValueError, match="doesn't match"):
            flat_simple_model.extract(wrong_mask)

    def test_apply_updates_selected(self, flat_simple_model):
        """Test that apply only updates masked positions."""
        # Get original values
        original = flat_simple_model.flatten().copy()

        # Create mask for first 10 parameters
        mask = Mask.from_indices(list(range(10)), size=67)

        # Apply new values (all zeros) to masked positions
        new_values = np.zeros(10, dtype=np.float32)
        flat_simple_model.apply(new_values, mask)

        # Get updated parameters
        updated = flat_simple_model.flatten()

        # First 10 should be zeros
        assert np.allclose(updated[:10], 0.0)
        # Rest should be unchanged
        assert np.allclose(updated[10:], original[10:])

    def test_apply_wrong_count_raises(self, flat_simple_model):
        """Test that wrong value count raises error."""
        mask = Mask.from_indices([0, 1, 2], size=67)
        wrong_count = np.zeros(10)  # Should be 3

        with pytest.raises(ValueError, match="Expected 3"):
            flat_simple_model.apply(wrong_count, mask)

    def test_extract_apply_roundtrip(self, flat_simple_model):
        """Test that extract/apply is lossless for selected params."""
        mask = Mask.random(67, fraction=0.5, seed=42)

        # Extract, modify, apply
        extracted = flat_simple_model.extract(mask)
        modified = extracted * 2  # Double the values
        flat_simple_model.apply(modified, mask)

        # Extract again
        re_extracted = flat_simple_model.extract(mask)

        assert np.allclose(re_extracted, modified)


class TestParameterInfo:
    """Tests for parameter information methods."""

    def test_get_param_index_range(self, flat_simple_model):
        """Test getting parameter index range."""
        start, end = flat_simple_model.get_param_index_range("fc1.weight")

        assert start == 0
        assert end == 50  # 5 * 10

    def test_get_param_index_range_not_found(self, flat_simple_model):
        """Test that unknown parameter raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            flat_simple_model.get_param_index_range("nonexistent")

    def test_get_layer_mask(self, flat_simple_model):
        """Test creating layer-based mask."""
        # Select only fc1 parameters
        mask_data = flat_simple_model.get_layer_mask(["fc1"])

        assert mask_data.shape == (67,)
        assert np.sum(mask_data) == 55  # fc1.weight (50) + fc1.bias (5)
        assert mask_data[:55].all()
        assert not mask_data[55:].any()

    def test_get_layer_mask_multiple(self, flat_simple_model):
        """Test selecting multiple layers."""
        mask_data = flat_simple_model.get_layer_mask(["fc1", "fc2"])

        assert np.sum(mask_data) == 67  # All parameters

    def test_parameter_info(self, flat_simple_model):
        """Test parameter info dictionary."""
        info = flat_simple_model.parameter_info()

        assert len(info) == 4
        assert info[0]["name"] == "fc1.weight"
        assert info[0]["shape"] == (5, 10)
        assert info[0]["numel"] == 50
        assert info[0]["start"] == 0
        assert info[0]["end"] == 50


class TestRepr:
    """Tests for string representation."""

    def test_repr(self, flat_simple_model):
        """Test string representation."""
        repr_str = repr(flat_simple_model)

        assert "FlattenedModel" in repr_str
        assert "num_parameters=67" in repr_str
        assert "num_tensors=4" in repr_str
