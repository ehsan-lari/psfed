"""Binary mask representation for parameter selection.

This module provides the Mask class for efficient representation and
manipulation of binary masks used to select model parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class Mask:
    """Binary mask for selecting model parameters.
    
    A Mask represents which parameters should be shared in a given round.
    It can be created from boolean arrays, index lists, or generated
    by a MaskSelector.
    
    The mask is stored as a numpy boolean array for memory efficiency
    and fast operations. The class is immutable (frozen) to prevent
    accidental modifications during federated learning rounds.
    
    Attributes:
        data: Boolean numpy array where True indicates selected parameters.
        size: Total number of parameters (length of the mask).
        count: Number of selected parameters (True values).
        indices: Indices of selected parameters (computed on demand).
    
    Example:
        Create from boolean array::
        
            >>> mask = Mask(data=np.array([True, False, True, False]))
            >>> mask.count
            2
            >>> mask.indices
            array([0, 2])
        
        Create from indices::
        
            >>> mask = Mask.from_indices([0, 2], size=4)
            >>> mask.data
            array([ True, False,  True, False])
        
        Create random mask::
        
            >>> mask = Mask.random(size=100, fraction=0.5, seed=42)
            >>> mask.count
            50
    """
    
    data: np.ndarray = field(repr=False)
    
    def __post_init__(self) -> None:
        """Validate mask data after initialization."""
        if not isinstance(self.data, np.ndarray):
            object.__setattr__(self, "data", np.asarray(self.data, dtype=bool))
        elif self.data.dtype != bool:
            object.__setattr__(self, "data", self.data.astype(bool))
        
        if self.data.ndim != 1:
            raise ValueError(
                f"Mask data must be 1-dimensional, got shape {self.data.shape}"
            )
    
    @property
    def size(self) -> int:
        """Total number of parameters (length of the mask)."""
        return len(self.data)
    
    @property
    def count(self) -> int:
        """Number of selected parameters (True values)."""
        return int(np.sum(self.data))
    
    @property
    def fraction(self) -> float:
        """Fraction of parameters selected."""
        return self.count / self.size if self.size > 0 else 0.0
    
    @property
    def indices(self) -> np.ndarray:
        """Indices of selected parameters.
        
        Returns:
            1D array of indices where mask is True.
        """
        return np.nonzero(self.data)[0]
    
    @classmethod
    def from_indices(cls, indices: Sequence[int] | np.ndarray, size: int) -> Mask:
        """Create a mask from a list of selected indices.
        
        Args:
            indices: Indices of parameters to select.
            size: Total number of parameters.
        
        Returns:
            A new Mask with True at the specified indices.
        
        Raises:
            ValueError: If any index is out of bounds.
        
        Example:
            >>> mask = Mask.from_indices([0, 5, 10], size=100)
            >>> mask.count
            3
        """
        indices_arr = np.asarray(indices, dtype=np.intp)
        
        if len(indices_arr) > 0:
            if np.any(indices_arr < 0) or np.any(indices_arr >= size):
                raise ValueError(
                    f"Indices must be in range [0, {size}), "
                    f"got min={indices_arr.min()}, max={indices_arr.max()}"
                )
        
        data = np.zeros(size, dtype=bool)
        data[indices_arr] = True
        return cls(data=data)
    
    @classmethod
    def random(
        cls,
        size: int,
        fraction: float = 0.5,
        *,
        seed: int | None = None,
        round_num: int | None = None,
    ) -> Mask:
        """Create a random mask selecting a fraction of parameters.
        
        Args:
            size: Total number of parameters.
            fraction: Fraction of parameters to select (0.0 to 1.0).
            seed: Base random seed for reproducibility.
            round_num: If provided, combined with seed for per-round randomness.
        
        Returns:
            A new Mask with approximately `fraction * size` True values.
        
        Example:
            >>> mask = Mask.random(size=100, fraction=0.3, seed=42)
            >>> 25 <= mask.count <= 35  # Approximately 30
            True
        """
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"Fraction must be in [0.0, 1.0], got {fraction}")
        
        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")
        
        # Combine seed and round_num for reproducible per-round randomness
        if seed is not None and round_num is not None:
            effective_seed = seed + round_num * 1000003  # Large prime for mixing
        elif seed is not None:
            effective_seed = seed
        else:
            effective_seed = None
        
        rng = np.random.default_rng(effective_seed)
        
        # Exact count selection for precise control
        count = int(round(fraction * size))
        count = max(0, min(count, size))  # Clamp to valid range
        
        # Random selection without replacement
        indices = rng.choice(size, size=count, replace=False)
        
        return cls.from_indices(indices, size)
    
    @classmethod
    def all_true(cls, size: int) -> Mask:
        """Create a mask selecting all parameters."""
        return cls(data=np.ones(size, dtype=bool))
    
    @classmethod
    def all_false(cls, size: int) -> Mask:
        """Create a mask selecting no parameters."""
        return cls(data=np.zeros(size, dtype=bool))
    
    def __and__(self, other: Mask) -> Mask:
        """Logical AND of two masks."""
        if self.size != other.size:
            raise ValueError(f"Mask sizes must match: {self.size} != {other.size}")
        return Mask(data=self.data & other.data)
    
    def __or__(self, other: Mask) -> Mask:
        """Logical OR of two masks."""
        if self.size != other.size:
            raise ValueError(f"Mask sizes must match: {self.size} != {other.size}")
        return Mask(data=self.data | other.data)
    
    def __invert__(self) -> Mask:
        """Logical NOT of the mask."""
        return Mask(data=~self.data)
    
    def __eq__(self, other: object) -> bool:
        """Check equality with another mask."""
        if not isinstance(other, Mask):
            return NotImplemented
        return np.array_equal(self.data, other.data)
    
    def __hash__(self) -> int:
        """Hash based on mask data."""
        return hash(self.data.tobytes())
    
    def to_dict(self) -> dict:
        """Serialize mask to a dictionary for efficient storage."""
        return {"indices": self.indices.tolist(), "size": self.size}
    
    @classmethod
    def from_dict(cls, data: dict) -> Mask:
        """Deserialize mask from a dictionary."""
        return cls.from_indices(data["indices"], data["size"])
    
    def __repr__(self) -> str:
        """String representation showing count and size."""
        return f"Mask(count={self.count}, size={self.size}, fraction={self.fraction:.2%})"
