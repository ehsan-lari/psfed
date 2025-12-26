"""Validation utilities for PSFed.

This module provides validation functions to catch configuration
errors early and provide helpful error messages.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from torch import nn

    from psfed.core.flattener import FlattenedModel
    from psfed.core.mask import Mask


class PSFedValidationError(ValueError):
    """Base exception for PSFed validation errors."""

    pass


class MaskSizeMismatchError(PSFedValidationError):
    """Raised when mask size doesn't match model parameters."""

    pass


class ParameterCountMismatchError(PSFedValidationError):
    """Raised when parameter count doesn't match expected."""

    pass


def validate_fraction(fraction: float, name: str = "fraction") -> None:
    """Validate that a fraction is in [0.0, 1.0].
    
    Args:
        fraction: Value to validate.
        name: Name for error messages.
    
    Raises:
        PSFedValidationError: If fraction is out of range.
    """
    if not isinstance(fraction, (int, float)):
        raise PSFedValidationError(
            f"{name} must be a number, got {type(fraction).__name__}"
        )
    if not 0.0 <= fraction <= 1.0:
        raise PSFedValidationError(
            f"{name} must be in [0.0, 1.0], got {fraction}"
        )


def validate_mask_compatibility(
    mask: Mask,
    num_parameters: int,
    context: str = "",
) -> None:
    """Validate that a mask is compatible with a model.
    
    Args:
        mask: The mask to validate.
        num_parameters: Expected number of parameters.
        context: Additional context for error messages.
    
    Raises:
        MaskSizeMismatchError: If mask size doesn't match.
    """
    if mask.size != num_parameters:
        ctx = f" ({context})" if context else ""
        raise MaskSizeMismatchError(
            f"Mask size {mask.size} doesn't match model parameters "
            f"{num_parameters}{ctx}"
        )


def validate_model_parameters(
    model: nn.Module,
    min_params: int = 1,
) -> int:
    """Validate that a model has trainable parameters.
    
    Args:
        model: PyTorch model to validate.
        min_params: Minimum required parameters.
    
    Returns:
        Total number of parameters.
    
    Raises:
        PSFedValidationError: If model has insufficient parameters.
    """
    num_params = sum(p.numel() for p in model.parameters())
    
    if num_params < min_params:
        raise PSFedValidationError(
            f"Model has {num_params} parameters, minimum required is {min_params}"
        )
    
    return num_params


def validate_partial_params(
    partial_params: np.ndarray,
    mask: Mask,
    context: str = "",
) -> None:
    """Validate that partial parameters match mask count.
    
    Args:
        partial_params: Array of partial parameters.
        mask: The mask indicating which parameters.
        context: Additional context for error messages.
    
    Raises:
        ParameterCountMismatchError: If counts don't match.
    """
    if len(partial_params) != mask.count:
        ctx = f" ({context})" if context else ""
        raise ParameterCountMismatchError(
            f"Expected {mask.count} parameters for mask, "
            f"got {len(partial_params)}{ctx}"
        )


def validate_indices(
    indices: np.ndarray | list[int],
    size: int,
    name: str = "indices",
) -> None:
    """Validate that indices are within bounds.
    
    Args:
        indices: Array or list of indices.
        size: Maximum valid index + 1.
        name: Name for error messages.
    
    Raises:
        PSFedValidationError: If any index is out of bounds.
    """
    indices_arr = np.asarray(indices)
    
    if len(indices_arr) == 0:
        return
    
    min_idx = indices_arr.min()
    max_idx = indices_arr.max()
    
    if min_idx < 0:
        raise PSFedValidationError(
            f"{name} contains negative index: {min_idx}"
        )
    
    if max_idx >= size:
        raise PSFedValidationError(
            f"{name} contains out-of-bounds index {max_idx} (size={size})"
        )
