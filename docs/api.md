# API Reference

## Core Module

### `psfed.FlattenedModel`

```python
class FlattenedModel:
    """Wrapper for PyTorch models enabling partial parameter sharing."""
    
    def __init__(self, model: nn.Module) -> None:
        """Initialize with a PyTorch model."""
    
    @property
    def num_parameters(self) -> int:
        """Total number of scalar parameters."""
    
    def flatten(self) -> np.ndarray:
        """Flatten all parameters into 1D array."""
    
    def unflatten(self, flat_params: np.ndarray) -> None:
        """Update model from 1D array."""
    
    def extract(self, mask: Mask) -> np.ndarray:
        """Extract parameters at masked positions."""
    
    def apply(self, partial_params: np.ndarray, mask: Mask) -> None:
        """Apply partial updates at masked positions."""
    
    def get_param_index_range(self, param_name: str) -> tuple[int, int]:
        """Get flattened index range for a named parameter."""
    
    def get_layer_mask(self, layer_names: list[str]) -> np.ndarray:
        """Create boolean mask for specific layers."""
    
    def parameter_info(self) -> list[dict]:
        """Get detailed info about each parameter tensor."""
```

### `psfed.Mask`

```python
@dataclass(frozen=True)
class Mask:
    """Binary mask for parameter selection."""
    
    data: np.ndarray  # Boolean array
    
    @property
    def size(self) -> int:
        """Total number of parameters."""
    
    @property
    def count(self) -> int:
        """Number of selected parameters."""
    
    @property
    def fraction(self) -> float:
        """Fraction of parameters selected."""
    
    @property
    def indices(self) -> np.ndarray:
        """Indices of selected parameters."""
    
    @classmethod
    def from_indices(cls, indices: Sequence[int], size: int) -> Mask:
        """Create mask from index list."""
    
    @classmethod
    def random(cls, size: int, fraction: float, seed: int = None, round_num: int = None) -> Mask:
        """Create random mask."""
    
    @classmethod
    def all_true(cls, size: int) -> Mask:
        """Create mask selecting all parameters."""
    
    @classmethod
    def all_false(cls, size: int) -> Mask:
        """Create mask selecting no parameters."""
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
    
    @classmethod
    def from_dict(cls, data: dict) -> Mask:
        """Deserialize from dictionary."""
```

### `psfed.MaskSelector`

```python
class MaskSelector(ABC):
    """Abstract base class for parameter selection strategies."""
    
    def __init__(self, fraction: float = 0.5) -> None:
        """Initialize with target fraction."""
    
    @abstractmethod
    def select(self, num_parameters: int, round_num: int, **kwargs) -> Mask:
        """Select parameters for a given round."""
    
    def get_count(self, num_parameters: int) -> int:
        """Calculate number of parameters to select."""
```

## Selector Implementations

### `psfed.RandomMaskSelector`

```python
class RandomMaskSelector(MaskSelector):
    """Per-round random parameter selection."""
    
    def __init__(self, fraction: float = 0.5, seed: int = None) -> None:
        """
        Args:
            fraction: Fraction of parameters to select.
            seed: Random seed for reproducibility.
        """
```

### `psfed.TopKMagnitudeSelector`

```python
class TopKMagnitudeSelector(MaskSelector):
    """Select parameters with largest absolute values.
    
    Requires `flat_model` in kwargs.
    """
```

### `psfed.GradientBasedSelector`

```python
class GradientBasedSelector(MaskSelector):
    """Select parameters with largest gradients.
    
    Requires `gradients` array in kwargs.
    """
```

### `psfed.StructuredMaskSelector`

```python
class StructuredMaskSelector(MaskSelector):
    """Layer-aware structured selection.
    
    Args:
        fraction: Approximate fraction to select.
        layer_names: Specific layers to include (optional).
        rotate_layers: Whether to rotate through layers.
    
    Requires `flat_model` in kwargs.
    """
```

### `psfed.FixedMaskSelector`

```python
class FixedMaskSelector(MaskSelector):
    """User-defined fixed parameter selection.
    
    Args:
        indices: List of parameter indices to always select.
    """
```

### `psfed.ClientSpecificMaskSelector`

```python
class ClientSpecificMaskSelector(MaskSelector):
    """Generate different masks for different clients.
    
    Args:
        fraction: Fraction each client shares.
        seed: Base random seed.
        overlap_fraction: Target overlap between clients.
    
    Uses `client_id` from kwargs for per-client selection.
    """
```

## Flower Integration

### `psfed.flower.PSFedAvg`

```python
class PSFedAvg(FedAvg):
    """FedAvg strategy with partial model sharing.
    
    Args:
        mask_fraction: Fraction of parameters to share (0.0-1.0).
        mask_selector: Custom MaskSelector (optional).
        mask_seed: Random seed for default selector.
        ... (standard FedAvg parameters)
    """
```

### `psfed.flower.PSFedClient`

```python
class PSFedClient(NumPyClient):
    """Flower client with partial sharing support.
    
    Subclasses must implement:
        train_local(config: dict) -> int
    
    Optional override:
        evaluate_local(config: dict) -> tuple[float, int, dict]
    """
```

## Utility Functions

### `psfed.flower.numpy_utils`

```python
def mask_to_config(mask: Mask) -> dict:
    """Serialize mask for Flower config."""

def mask_from_config(config: dict) -> Mask:
    """Deserialize mask from Flower config."""

def parameters_to_numpy(parameters: Parameters) -> NDArrays:
    """Convert Flower Parameters to numpy arrays."""

def numpy_to_parameters(arrays: NDArrays) -> Parameters:
    """Convert numpy arrays to Flower Parameters."""
```
