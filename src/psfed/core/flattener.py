"""Flattened model representation for partial parameter sharing.

This module provides the FlattenedModel class that wraps a PyTorch model
and enables efficient flattening/unflattening of parameters for partial sharing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from psfed.core.mask import Mask


class FlattenedModel:
    """Wrapper for PyTorch models enabling partial parameter sharing.
    
    This class provides functionality to:
    - Flatten all model parameters into a single 1D vector
    - Extract a subset of parameters based on a mask
    - Apply partial parameter updates from a mask
    - Reconstruct the original parameter shapes
    
    The flattening order is deterministic based on `model.parameters()`,
    ensuring consistent mapping between server and clients.
    
    Attributes:
        model: The wrapped PyTorch model.
        num_parameters: Total number of scalar parameters.
        param_shapes: List of (name, shape) tuples for each parameter.
        param_numels: Number of elements in each parameter tensor.
    
    Example:
        Basic usage::
        
            >>> model = nn.Linear(10, 5)
            >>> flat = FlattenedModel(model)
            >>> flat.num_parameters
            55  # 10*5 weights + 5 biases
        
        Extract and apply partial parameters::
        
            >>> mask = Mask.random(flat.num_parameters, fraction=0.5)
            >>> partial = flat.extract(mask)  # Get selected params
            >>> flat.apply(partial, mask)     # Apply to model
    """
    
    def __init__(self, model: nn.Module) -> None:
        """Initialize with a PyTorch model.
        
        Args:
            model: Any PyTorch nn.Module.
        """
        self.model = model
        self._build_param_info()
    
    def _build_param_info(self) -> None:
        """Build parameter metadata for flatten/unflatten operations."""
        self.param_shapes: list[tuple[str, torch.Size]] = []
        self.param_numels: list[int] = []
        
        for name, param in self.model.named_parameters():
            self.param_shapes.append((name, param.shape))
            self.param_numels.append(param.numel())
        
        self._num_parameters = sum(self.param_numels)
        
        # Precompute cumulative indices for fast slicing
        self._param_offsets = np.zeros(len(self.param_numels) + 1, dtype=np.int64)
        np.cumsum(self.param_numels, out=self._param_offsets[1:])
    
    @property
    def num_parameters(self) -> int:
        """Total number of scalar parameters in the model."""
        return self._num_parameters
    
    def flatten(self) -> np.ndarray:
        """Flatten all model parameters into a 1D numpy array.
        
        Returns:
            1D numpy array of all parameters concatenated.
        
        Example:
            >>> flat_params = flat_model.flatten()
            >>> flat_params.shape
            (num_parameters,)
        """
        params = []
        for _, param in self.model.named_parameters():
            params.append(param.detach().cpu().numpy().ravel())
        return np.concatenate(params)
    
    def unflatten(self, flat_params: np.ndarray) -> None:
        """Update all model parameters from a 1D array.
        
        Args:
            flat_params: 1D array with exactly `num_parameters` elements.
        
        Raises:
            ValueError: If array size doesn't match model parameters.
        """
        if len(flat_params) != self._num_parameters:
            raise ValueError(
                f"Expected {self._num_parameters} parameters, "
                f"got {len(flat_params)}"
            )
        
        with torch.no_grad():
            offset = 0
            for (name, shape), numel in zip(
                self.param_shapes, self.param_numels, strict=True
            ):
                # Get the parameter tensor
                param = self._get_param_by_name(name)
                
                # Extract and reshape the values
                values = flat_params[offset : offset + numel]
                tensor = torch.from_numpy(values.reshape(shape)).to(
                    dtype=param.dtype, device=param.device
                )
                
                # Update in-place
                param.copy_(tensor)
                offset += numel
    
    def extract(self, mask: Mask) -> np.ndarray:
        """Extract parameters at masked positions.
        
        Args:
            mask: Binary mask indicating which parameters to extract.
        
        Returns:
            1D array containing only the selected parameters.
        
        Raises:
            ValueError: If mask size doesn't match model parameters.
        
        Example:
            >>> mask = Mask.from_indices([0, 1, 2], size=flat.num_parameters)
            >>> partial = flat.extract(mask)
            >>> partial.shape
            (3,)
        """
        if mask.size != self._num_parameters:
            raise ValueError(
                f"Mask size {mask.size} doesn't match "
                f"model parameters {self._num_parameters}"
            )
        
        flat_params = self.flatten()
        return flat_params[mask.data]
    
    def apply(self, partial_params: np.ndarray, mask: Mask) -> None:
        """Apply partial parameter updates at masked positions.
        
        Only the parameters at positions where mask is True are updated.
        Other parameters remain unchanged.
        
        Args:
            partial_params: Values for the selected parameters.
            mask: Binary mask indicating which parameters to update.
        
        Raises:
            ValueError: If array size doesn't match mask count.
        
        Example:
            >>> mask = Mask.random(flat.num_parameters, fraction=0.5)
            >>> partial = np.zeros(mask.count)  # New values
            >>> flat.apply(partial, mask)       # Update selected params
        """
        if len(partial_params) != mask.count:
            raise ValueError(
                f"Expected {mask.count} values for mask, "
                f"got {len(partial_params)}"
            )
        
        if mask.size != self._num_parameters:
            raise ValueError(
                f"Mask size {mask.size} doesn't match "
                f"model parameters {self._num_parameters}"
            )
        
        # Get current parameters, update masked positions, unflatten
        flat_params = self.flatten()
        flat_params[mask.data] = partial_params
        self.unflatten(flat_params)
    
    def _get_param_by_name(self, name: str) -> torch.nn.Parameter:
        """Get a parameter tensor by its dot-separated name.
        
        Args:
            name: Parameter name like 'layer1.weight' or 'classifier.bias'.
        
        Returns:
            The parameter tensor.
        """
        parts = name.split(".")
        module: nn.Module = self.model
        
        for part in parts[:-1]:
            if part.isdigit():
                module = module[int(part)]  # type: ignore[index]
            else:
                module = getattr(module, part)
        
        return getattr(module, parts[-1])
    
    def get_param_index_range(self, param_name: str) -> tuple[int, int]:
        """Get the flattened index range for a named parameter.
        
        Args:
            param_name: Name of the parameter.
        
        Returns:
            Tuple of (start_index, end_index) in the flattened array.
        
        Raises:
            KeyError: If parameter name not found.
        """
        for i, (name, _) in enumerate(self.param_shapes):
            if name == param_name:
                return int(self._param_offsets[i]), int(self._param_offsets[i + 1])
        raise KeyError(f"Parameter '{param_name}' not found in model")
    
    def get_layer_mask(self, layer_names: list[str]) -> np.ndarray:
        """Create a boolean mask selecting specific layers/parameters.
        
        Args:
            layer_names: List of parameter name prefixes to include.
        
        Returns:
            Boolean array of size `num_parameters`.
        
        Example:
            >>> mask_data = flat.get_layer_mask(['classifier'])
            >>> mask = Mask(data=mask_data)
        """
        mask = np.zeros(self._num_parameters, dtype=bool)
        
        for i, (name, _) in enumerate(self.param_shapes):
            for layer_name in layer_names:
                if name.startswith(layer_name):
                    start, end = int(self._param_offsets[i]), int(self._param_offsets[i + 1])
                    mask[start:end] = True
                    break
        
        return mask
    
    def parameter_info(self) -> list[dict]:
        """Get detailed information about each parameter.
        
        Returns:
            List of dicts with name, shape, numel, start, end for each param.
        """
        info = []
        for i, ((name, shape), numel) in enumerate(
            zip(self.param_shapes, self.param_numels, strict=True)
        ):
            info.append({
                "name": name,
                "shape": tuple(shape),
                "numel": numel,
                "start": int(self._param_offsets[i]),
                "end": int(self._param_offsets[i + 1]),
            })
        return info
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FlattenedModel(num_parameters={self._num_parameters}, "
            f"num_tensors={len(self.param_shapes)})"
        )
