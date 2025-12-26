"""Mask selection strategies for partial model sharing.

This module provides the MaskSelector abstract base class and various
implementations for selecting which parameters to share each round.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from psfed.core.mask import Mask

if TYPE_CHECKING:
    from psfed.core.flattener import FlattenedModel


class MaskSelector(ABC):
    """Abstract base class for parameter selection strategies.
    
    Subclasses implement different strategies for selecting which
    parameters to share in each federated learning round.
    
    The selector is called once per round (on the server) to generate
    a mask that is sent to all clients along with the selected parameters.
    
    Attributes:
        fraction: Target fraction of parameters to select (0.0 to 1.0).
    
    Example:
        Implementing a custom selector::
        
            class MySelector(MaskSelector):
                def select(
                    self, 
                    num_parameters: int, 
                    round_num: int,
                    **kwargs
                ) -> Mask:
                    # Your selection logic
                    indices = ...
                    return Mask.from_indices(indices, num_parameters)
    """
    
    def __init__(self, fraction: float = 0.5) -> None:
        """Initialize the selector.
        
        Args:
            fraction: Target fraction of parameters to select.
        
        Raises:
            ValueError: If fraction is not in [0.0, 1.0].
        """
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"Fraction must be in [0.0, 1.0], got {fraction}")
        self.fraction = fraction
    
    @abstractmethod
    def select(
        self,
        num_parameters: int,
        round_num: int,
        **kwargs: Any,
    ) -> Mask:
        """Select parameters to share for a given round.
        
        Args:
            num_parameters: Total number of parameters in the model.
            round_num: Current federation round (1-indexed).
            **kwargs: Additional context (e.g., model, gradients, client_id).
        
        Returns:
            A Mask indicating which parameters to share.
        """
        ...
    
    def get_count(self, num_parameters: int) -> int:
        """Calculate the number of parameters to select.
        
        Args:
            num_parameters: Total number of parameters.
        
        Returns:
            Number of parameters to select based on fraction.
        """
        return int(round(self.fraction * num_parameters))


class RandomMaskSelector(MaskSelector):
    """Per-round random parameter selection.
    
    Selects a random subset of parameters each round. Different rounds
    get different masks, ensuring eventual coverage of all parameters.
    
    This is the default strategy as it provides:
    - Simple implementation
    - Unbiased coverage over time
    - Reproducibility with seed
    
    Attributes:
        fraction: Fraction of parameters to select.
        seed: Base random seed for reproducibility.
    
    Example:
        >>> selector = RandomMaskSelector(fraction=0.5, seed=42)
        >>> mask1 = selector.select(100, round_num=1)
        >>> mask2 = selector.select(100, round_num=2)
        >>> mask1 != mask2  # Different masks each round
        True
    """
    
    def __init__(self, fraction: float = 0.5, seed: int | None = None) -> None:
        """Initialize random selector.
        
        Args:
            fraction: Fraction of parameters to select.
            seed: Random seed for reproducibility. If None, non-deterministic.
        """
        super().__init__(fraction=fraction)
        self.seed = seed
    
    def select(
        self,
        num_parameters: int,
        round_num: int,
        **kwargs: Any,
    ) -> Mask:
        """Select random parameters for this round.
        
        Args:
            num_parameters: Total number of parameters.
            round_num: Current round number (1-indexed).
        
        Returns:
            Mask with randomly selected parameters.
        """
        return Mask.random(
            size=num_parameters,
            fraction=self.fraction,
            seed=self.seed,
            round_num=round_num,
        )


class TopKMagnitudeSelector(MaskSelector):
    """Select parameters with largest absolute values.
    
    This strategy focuses communication on the "most important" parameters,
    assuming larger magnitude weights have more impact on model behavior.
    
    Requires access to current model parameters.
    
    Example:
        >>> selector = TopKMagnitudeSelector(fraction=0.3)
        >>> mask = selector.select(
        ...     num_parameters=1000, 
        ...     round_num=1,
        ...     flat_model=flat_model
        ... )
    """
    
    def select(
        self,
        num_parameters: int,
        round_num: int,
        **kwargs: Any,
    ) -> Mask:
        """Select top-k parameters by magnitude.
        
        Args:
            num_parameters: Total number of parameters.
            round_num: Current round number.
            **kwargs: Must include 'flat_model' (FlattenedModel).
        
        Returns:
            Mask selecting the largest magnitude parameters.
        
        Raises:
            ValueError: If flat_model not provided in kwargs.
        """
        flat_model: FlattenedModel | None = kwargs.get("flat_model")
        if flat_model is None:
            raise ValueError(
                "TopKMagnitudeSelector requires 'flat_model' in kwargs"
            )
        
        params = flat_model.flatten()
        k = self.get_count(num_parameters)
        
        # Get indices of top-k by absolute value
        abs_params = np.abs(params)
        indices = np.argpartition(abs_params, -k)[-k:]
        
        return Mask.from_indices(indices, num_parameters)


class GradientBasedSelector(MaskSelector):
    """Select parameters with largest gradient magnitudes.
    
    This strategy focuses on parameters that are actively changing,
    which may be more important for convergence.
    
    Requires access to gradient information from recent training.
    
    Example:
        >>> selector = GradientBasedSelector(fraction=0.3)
        >>> mask = selector.select(
        ...     num_parameters=1000,
        ...     round_num=1, 
        ...     gradients=grad_array
        ... )
    """
    
    def select(
        self,
        num_parameters: int,
        round_num: int,
        **kwargs: Any,
    ) -> Mask:
        """Select parameters with largest gradients.
        
        Args:
            num_parameters: Total number of parameters.
            round_num: Current round number.
            **kwargs: Must include 'gradients' (np.ndarray).
        
        Returns:
            Mask selecting parameters with largest gradient magnitude.
        
        Raises:
            ValueError: If gradients not provided in kwargs.
        """
        gradients: np.ndarray | None = kwargs.get("gradients")
        if gradients is None:
            raise ValueError(
                "GradientBasedSelector requires 'gradients' in kwargs"
            )
        
        k = self.get_count(num_parameters)
        
        # Get indices of top-k by gradient magnitude
        abs_grads = np.abs(gradients)
        indices = np.argpartition(abs_grads, -k)[-k:]
        
        return Mask.from_indices(indices, num_parameters)


class StructuredMaskSelector(MaskSelector):
    """Layer-aware structured parameter selection.
    
    Instead of random scalar selection, this selects entire layers
    or structured subsets, which may preserve model coherence better.
    
    Example:
        >>> selector = StructuredMaskSelector(
        ...     fraction=0.5,
        ...     layer_names=['encoder', 'classifier']
        ... )
    """
    
    def __init__(
        self, 
        fraction: float = 0.5, 
        layer_names: list[str] | None = None,
        rotate_layers: bool = True,
    ) -> None:
        """Initialize structured selector.
        
        Args:
            fraction: Approximate fraction of parameters to select.
            layer_names: Specific layers to include. If None, rotates all.
            rotate_layers: Whether to rotate through layers across rounds.
        """
        super().__init__(fraction=fraction)
        self.layer_names = layer_names
        self.rotate_layers = rotate_layers
    
    def select(
        self,
        num_parameters: int,
        round_num: int,
        **kwargs: Any,
    ) -> Mask:
        """Select structured parameter groups.
        
        Args:
            num_parameters: Total number of parameters.
            round_num: Current round number.
            **kwargs: Must include 'flat_model' (FlattenedModel).
        
        Returns:
            Mask selecting structured parameter groups.
        """
        flat_model: FlattenedModel | None = kwargs.get("flat_model")
        if flat_model is None:
            raise ValueError(
                "StructuredMaskSelector requires 'flat_model' in kwargs"
            )
        
        if self.layer_names is not None:
            # Select specified layers
            mask_data = flat_model.get_layer_mask(self.layer_names)
            return Mask(data=mask_data)
        
        # Rotate through layers based on round
        param_info = flat_model.parameter_info()
        num_layers = len(param_info)
        
        if num_layers == 0:
            return Mask.all_false(num_parameters)
        
        # Calculate how many layers to select
        target_fraction = self.fraction
        
        if self.rotate_layers:
            # Select different layers each round in rotation
            layers_to_select = max(1, int(num_layers * target_fraction))
            start_idx = (round_num - 1) % num_layers
            selected_indices = [
                (start_idx + i) % num_layers 
                for i in range(layers_to_select)
            ]
        else:
            # Always select first N layers
            layers_to_select = max(1, int(num_layers * target_fraction))
            selected_indices = list(range(layers_to_select))
        
        # Build mask from selected layer indices
        mask_data = np.zeros(num_parameters, dtype=bool)
        for idx in selected_indices:
            info = param_info[idx]
            mask_data[info["start"]:info["end"]] = True
        
        return Mask(data=mask_data)


class FixedMaskSelector(MaskSelector):
    """User-defined fixed parameter selection.
    
    Allows explicit control over which parameters are shared.
    The same mask is used every round.
    
    Example:
        >>> selector = FixedMaskSelector(indices=[0, 1, 2, 100, 101])
        >>> mask = selector.select(num_parameters=1000, round_num=1)
        >>> mask.count
        5
    """
    
    def __init__(self, indices: list[int] | np.ndarray) -> None:
        """Initialize with fixed indices.
        
        Args:
            indices: List of parameter indices to always select.
        """
        self._indices = np.asarray(indices)
        # Fraction will be computed when we know total parameters
        super().__init__(fraction=0.0)  # Placeholder
    
    def select(
        self,
        num_parameters: int,
        round_num: int,
        **kwargs: Any,
    ) -> Mask:
        """Return the fixed mask.
        
        Args:
            num_parameters: Total number of parameters.
            round_num: Ignored for fixed selection.
        
        Returns:
            Mask with the predefined indices selected.
        """
        return Mask.from_indices(self._indices, num_parameters)


class ClientSpecificMaskSelector(MaskSelector):
    """Generate different masks for different clients.
    
    This enables heterogeneous partial sharing where each client
    may send/receive different parameter subsets.
    
    Uses client_id to generate deterministic per-client masks.
    
    Example:
        >>> selector = ClientSpecificMaskSelector(fraction=0.5, seed=42)
        >>> mask_client1 = selector.select(100, round_num=1, client_id="client_1")
        >>> mask_client2 = selector.select(100, round_num=1, client_id="client_2")
        >>> mask_client1 != mask_client2
        True
    """
    
    def __init__(
        self, 
        fraction: float = 0.5, 
        seed: int | None = None,
        overlap_fraction: float = 0.5,
    ) -> None:
        """Initialize client-specific selector.
        
        Args:
            fraction: Fraction of parameters each client shares.
            seed: Base random seed.
            overlap_fraction: Target overlap between client masks (0-1).
                             Higher values mean more parameters are shared
                             by multiple clients.
        """
        super().__init__(fraction=fraction)
        self.seed = seed
        self.overlap_fraction = overlap_fraction
    
    def select(
        self,
        num_parameters: int,
        round_num: int,
        **kwargs: Any,
    ) -> Mask:
        """Generate client-specific mask.
        
        Args:
            num_parameters: Total number of parameters.
            round_num: Current round number.
            **kwargs: Should include 'client_id' for per-client selection.
        
        Returns:
            Mask specific to the client and round.
        """
        client_id: str | None = kwargs.get("client_id")
        
        # Hash client_id to get a numeric seed offset
        if client_id is not None:
            client_hash = hash(client_id) % (2**31)
        else:
            client_hash = 0
        
        # Combine base seed, round, and client for unique mask
        if self.seed is not None:
            effective_seed = self.seed + round_num * 1000003 + client_hash
        else:
            effective_seed = round_num * 1000003 + client_hash
        
        rng = np.random.default_rng(effective_seed)
        
        count = self.get_count(num_parameters)
        indices = rng.choice(num_parameters, size=count, replace=False)
        
        return Mask.from_indices(indices, num_parameters)


class CompositeMaskSelector(MaskSelector):
    """Combine multiple selection strategies.
    
    Useful for hybrid approaches, e.g., always share certain layers
    plus a random subset of others.
    
    Example:
        >>> always_share = FixedMaskSelector(indices=[0, 1, 2])  # First 3
        >>> random_extra = RandomMaskSelector(fraction=0.3)
        >>> selector = CompositeMaskSelector([always_share, random_extra], mode='union')
    """
    
    def __init__(
        self, 
        selectors: list[MaskSelector],
        mode: str = "union",
    ) -> None:
        """Initialize composite selector.
        
        Args:
            selectors: List of selectors to combine.
            mode: How to combine masks - 'union' (OR) or 'intersection' (AND).
        """
        super().__init__(fraction=0.0)  # Computed from combined result
        self.selectors = selectors
        self.mode = mode
        
        if mode not in ("union", "intersection"):
            raise ValueError(f"Mode must be 'union' or 'intersection', got {mode}")
    
    def select(
        self,
        num_parameters: int,
        round_num: int,
        **kwargs: Any,
    ) -> Mask:
        """Combine masks from all selectors.
        
        Args:
            num_parameters: Total number of parameters.
            round_num: Current round number.
            **kwargs: Passed to all child selectors.
        
        Returns:
            Combined mask based on mode.
        """
        if not self.selectors:
            return Mask.all_false(num_parameters)
        
        masks = [
            s.select(num_parameters, round_num, **kwargs) 
            for s in self.selectors
        ]
        
        result = masks[0]
        for mask in masks[1:]:
            if self.mode == "union":
                result = result | mask
            else:
                result = result & mask
        
        return result
