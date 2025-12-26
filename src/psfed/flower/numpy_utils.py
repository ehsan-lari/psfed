"""Utility functions for Flower parameter conversion.

This module provides utilities for converting between Flower's
parameter format and numpy arrays used by PSFed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from flwr.common import NDArrays, Parameters, ndarrays_to_parameters, parameters_to_ndarrays

from psfed.core.mask import Mask


def parameters_to_numpy(parameters: Parameters) -> NDArrays:
    """Convert Flower Parameters to list of numpy arrays.
    
    Args:
        parameters: Flower Parameters object.
    
    Returns:
        List of numpy arrays.
    """
    return parameters_to_ndarrays(parameters)


def numpy_to_parameters(arrays: NDArrays) -> Parameters:
    """Convert list of numpy arrays to Flower Parameters.
    
    Args:
        arrays: List of numpy arrays.
    
    Returns:
        Flower Parameters object.
    """
    return ndarrays_to_parameters(arrays)


def mask_to_config(mask: Mask) -> dict[str, Any]:
    """Serialize a Mask to a Flower config dictionary.
    
    Flower configs only support scalar types and strings, so we
    serialize the mask indices as a comma-separated string.
    
    Args:
        mask: The mask to serialize.
    
    Returns:
        Dictionary suitable for Flower config.
    
    Example:
        >>> mask = Mask.from_indices([0, 5, 10], size=100)
        >>> config = mask_to_config(mask)
        >>> config
        {'mask_indices': '0,5,10', 'mask_size': 100}
    """
    indices_str = ",".join(map(str, mask.indices.tolist()))
    return {
        "mask_indices": indices_str,
        "mask_size": mask.size,
    }


def mask_from_config(config: dict[str, Any]) -> Mask:
    """Deserialize a Mask from a Flower config dictionary.
    
    Args:
        config: Dictionary containing 'mask_indices' and 'mask_size'.
    
    Returns:
        Reconstructed Mask.
    
    Raises:
        KeyError: If required keys are missing.
        ValueError: If format is invalid.
    
    Example:
        >>> config = {'mask_indices': '0,5,10', 'mask_size': 100}
        >>> mask = mask_from_config(config)
        >>> mask.count
        3
    """
    indices_str = config["mask_indices"]
    size = int(config["mask_size"])
    
    if not indices_str:
        # Empty string means no indices selected
        return Mask.all_false(size)
    
    indices = [int(x) for x in indices_str.split(",")]
    return Mask.from_indices(indices, size)


def flat_array_to_ndarrays(
    flat_params: np.ndarray,
    shapes: list[tuple[int, ...]],
) -> NDArrays:
    """Convert flat parameter array to list of shaped arrays.
    
    This is useful when you need to convert PSFed's flat representation
    back to Flower's per-tensor format for compatibility.
    
    Args:
        flat_params: 1D array of all parameters.
        shapes: List of shapes for each tensor.
    
    Returns:
        List of numpy arrays with specified shapes.
    """
    arrays = []
    offset = 0
    for shape in shapes:
        numel = int(np.prod(shape))
        arr = flat_params[offset : offset + numel].reshape(shape)
        arrays.append(arr)
        offset += numel
    return arrays


def ndarrays_to_flat_array(arrays: NDArrays) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Convert list of arrays to single flat array.
    
    Args:
        arrays: List of numpy arrays.
    
    Returns:
        Tuple of (flat_params, shapes) for reconstruction.
    """
    shapes = [arr.shape for arr in arrays]
    flat = np.concatenate([arr.ravel() for arr in arrays])
    return flat, shapes
