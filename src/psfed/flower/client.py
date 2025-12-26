"""PSFed Flower Client implementation.

This module provides the PSFedClient class that handles partial
parameter receiving and sending in federated learning.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from flwr.client import NumPyClient
from flwr.common import NDArrays

from psfed.core.flattener import FlattenedModel
from psfed.core.mask import Mask
from psfed.flower.numpy_utils import mask_from_config

if TYPE_CHECKING:
    from torch import nn


class PSFedClient(NumPyClient):
    """Flower client with partial model sharing support.
    
    This client handles:
    - Receiving partial parameters from the server
    - Applying them to the local model at masked positions
    - Training locally on all parameters
    - Sending back only the masked parameters
    
    Subclasses must implement `train_local()` with their training logic.
    
    Attributes:
        model: The PyTorch model being trained.
        flat_model: FlattenedModel wrapper for partial operations.
    
    Example:
        Implementing a PSFed client::
        
            class MyClient(PSFedClient):
                def __init__(self, model, trainloader):
                    super().__init__(model)
                    self.trainloader = trainloader
                
                def train_local(self, config):
                    optimizer = torch.optim.SGD(
                        self.model.parameters(), 
                        lr=config.get("lr", 0.01)
                    )
                    criterion = nn.CrossEntropyLoss()
                    
                    self.model.train()
                    for images, labels in self.trainloader:
                        optimizer.zero_grad()
                        loss = criterion(self.model(images), labels)
                        loss.backward()
                        optimizer.step()
                    
                    return len(self.trainloader.dataset)
    """
    
    def __init__(self, model: nn.Module) -> None:
        """Initialize the PSFed client.
        
        Args:
            model: PyTorch model to train.
        """
        super().__init__()
        self.model = model
        self.flat_model = FlattenedModel(model)
        self._current_mask: Mask | None = None
    
    def get_parameters(self, config: dict) -> NDArrays:
        """Get model parameters to send to server.
        
        If a mask is configured, only masked parameters are returned.
        Otherwise, all parameters are returned.
        
        Args:
            config: Configuration from server.
        
        Returns:
            List containing a single 1D array of (partial) parameters.
        """
        if self._current_mask is not None:
            # Return only masked parameters
            partial_params = self.flat_model.extract(self._current_mask)
            return [partial_params]
        else:
            # Return all parameters
            return [self.flat_model.flatten()]
    
    def set_parameters(self, parameters: NDArrays, config: dict) -> None:
        """Apply parameters received from server.
        
        If mask information is in config, applies partial update.
        Otherwise, applies full parameter update.
        
        Args:
            parameters: List containing parameter array from server.
            config: Configuration including mask information.
        """
        params = parameters[0]  # Single flat array
        
        # Check if this is a partial update
        if "mask_indices" in config and "mask_size" in config:
            mask = mask_from_config(config)
            self._current_mask = mask
            
            # Validate sizes
            if len(params) != mask.count:
                raise ValueError(
                    f"Received {len(params)} parameters but mask expects {mask.count}"
                )
            
            # Apply partial update
            self.flat_model.apply(params, mask)
        else:
            # Full parameter update
            self._current_mask = None
            self.flat_model.unflatten(params)
    
    def fit(
        self, 
        parameters: NDArrays, 
        config: dict
    ) -> tuple[NDArrays, int, dict]:
        """Train the model with partial parameter sharing.
        
        1. Apply received (partial) parameters
        2. Train locally on all parameters
        3. Return (partial) updated parameters
        
        Args:
            parameters: Parameters from server.
            config: Training configuration.
        
        Returns:
            Tuple of (updated_parameters, num_samples, metrics).
        """
        # Apply received parameters (full or partial)
        self.set_parameters(parameters, config)
        
        # Train locally - implemented by subclass
        num_samples = self.train_local(config)
        
        # Return updated parameters (respecting current mask)
        updated_params = self.get_parameters(config)
        
        return updated_params, num_samples, {}
    
    def evaluate(
        self, 
        parameters: NDArrays, 
        config: dict
    ) -> tuple[float, int, dict]:
        """Evaluate the model.
        
        Args:
            parameters: Parameters from server.
            config: Evaluation configuration.
        
        Returns:
            Tuple of (loss, num_samples, metrics).
        """
        # Apply received parameters
        self.set_parameters(parameters, config)
        
        # Evaluate - implemented by subclass
        loss, num_samples, metrics = self.evaluate_local(config)
        
        return loss, num_samples, metrics
    
    @abstractmethod
    def train_local(self, config: dict) -> int:
        """Perform local training on the model.
        
        This method should train `self.model` using local data.
        All parameters (both masked and unmasked) should be trained.
        
        Args:
            config: Training configuration from server.
        
        Returns:
            Number of training samples used.
        """
        ...
    
    def evaluate_local(self, config: dict) -> tuple[float, int, dict]:
        """Evaluate the model on local data.
        
        Override this method to implement evaluation logic.
        Default implementation returns dummy values.
        
        Args:
            config: Evaluation configuration.
        
        Returns:
            Tuple of (loss, num_samples, metrics_dict).
        """
        return 0.0, 0, {}
    
    @property
    def num_parameters(self) -> int:
        """Total number of model parameters."""
        return self.flat_model.num_parameters
    
    @property 
    def current_mask(self) -> Mask | None:
        """The current parameter mask, if any."""
        return self._current_mask
