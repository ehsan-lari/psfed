"""Pytest configuration and fixtures for PSFed tests."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from psfed.core.flattener import FlattenedModel
from psfed.core.mask import Mask


# ---------------------------------------------------------------------------
# Model Fixtures
# ---------------------------------------------------------------------------


class SimpleNet(nn.Module):
    """Simple network for testing."""

    def __init__(self, input_size: int = 10, hidden_size: int = 5, output_size: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ConvNet(nn.Module):
    """Convolutional network for testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@pytest.fixture
def simple_model() -> nn.Module:
    """Create a simple feed-forward network."""
    torch.manual_seed(42)
    return SimpleNet()


@pytest.fixture
def conv_model() -> nn.Module:
    """Create a convolutional network."""
    torch.manual_seed(42)
    return ConvNet()


@pytest.fixture
def flat_simple_model(simple_model) -> FlattenedModel:
    """Create a FlattenedModel from simple network."""
    return FlattenedModel(simple_model)


@pytest.fixture
def flat_conv_model(conv_model) -> FlattenedModel:
    """Create a FlattenedModel from conv network."""
    return FlattenedModel(conv_model)


# ---------------------------------------------------------------------------
# Mask Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_mask() -> Mask:
    """Create a sample mask."""
    return Mask.from_indices([0, 2, 5, 7], size=10)


@pytest.fixture
def random_mask_50() -> Mask:
    """Create a 50% random mask."""
    return Mask.random(size=100, fraction=0.5, seed=42)


# ---------------------------------------------------------------------------
# Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_params() -> np.ndarray:
    """Sample parameter array."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random number generator."""
    return np.random.default_rng(42)
