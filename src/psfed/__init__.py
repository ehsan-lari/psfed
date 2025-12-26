"""PSFed: Partial Model Sharing for Federated Learning.

A research-grade Python package that implements partial model sharing
in federated learning, enabling selective parameter synchronization
based on configurable masking strategies.

Example:
    Basic usage with PyTorch::
    
        from psfed import FlattenedModel, RandomMaskSelector
        
        model = MyNet()
        flat_model = FlattenedModel(model)
        selector = RandomMaskSelector(fraction=0.5)
        
        mask = selector.select(flat_model.num_parameters, round_num=1)
        partial_params = flat_model.extract(mask)
    
    With Flower::
    
        from psfed.flower import PSFedAvg, PSFedClient
        
        strategy = PSFedAvg(mask_fraction=0.5)
"""

from psfed.core.flattener import FlattenedModel
from psfed.core.mask import Mask
from psfed.core.selectors import (
    ClientSpecificMaskSelector,
    FixedMaskSelector,
    GradientBasedSelector,
    MaskSelector,
    RandomMaskSelector,
    StructuredMaskSelector,
    TopKMagnitudeSelector,
)

__version__ = "0.1.0"
__author__ = "Ehsan"

__all__ = [
    # Core classes
    "FlattenedModel",
    "Mask",
    # Selectors
    "MaskSelector",
    "RandomMaskSelector",
    "TopKMagnitudeSelector",
    "GradientBasedSelector",
    "StructuredMaskSelector",
    "FixedMaskSelector",
    "ClientSpecificMaskSelector",
    # Metadata
    "__version__",
    "__author__",
]
