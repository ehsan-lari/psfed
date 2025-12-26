"""PSFed Flower Strategy implementation.

This module provides PSFedAvg, a FedAvg-based strategy that implements
partial model sharing between server and clients.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
from flwr.common import (
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from psfed.core.flattener import FlattenedModel
from psfed.core.mask import Mask
from psfed.core.selectors import MaskSelector, RandomMaskSelector
from psfed.flower.numpy_utils import mask_to_config

logger = logging.getLogger(__name__)


class PSFedAvg(FedAvg):
    """FedAvg strategy with partial model sharing.
    
    This strategy extends FedAvg to support partial parameter communication:
    - Server selects a subset of parameters using a MaskSelector
    - Only selected parameters are sent to clients
    - Clients return updates only for selected parameters
    - Server aggregates partial updates using weighted averaging
    
    Attributes:
        mask_fraction: Fraction of parameters to share (0.0 to 1.0).
        mask_selector: Strategy for selecting parameters.
        mask_seed: Random seed for reproducibility.
    
    Example:
        Basic usage::
        
            strategy = PSFedAvg(
                fraction_fit=0.1,
                min_fit_clients=2,
                min_available_clients=2,
                mask_fraction=0.5,  # Share 50% of parameters
                mask_seed=42,
            )
            
            fl.server.start_server(
                server_address="0.0.0.0:8080",
                config=fl.server.ServerConfig(num_rounds=10),
                strategy=strategy,
            )
        
        With custom selector::
        
            from psfed import TopKMagnitudeSelector
            
            strategy = PSFedAvg(
                mask_selector=TopKMagnitudeSelector(fraction=0.3),
                ...
            )
    """
    
    def __init__(
        self,
        *,
        # PSFed-specific parameters
        mask_fraction: float = 0.5,
        mask_selector: MaskSelector | None = None,
        mask_seed: int | None = None,
        # Standard FedAvg parameters
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None] | None = None,
        on_fit_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        on_evaluate_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        accept_failures: bool = True,
        initial_parameters: Parameters | None = None,
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
    ) -> None:
        """Initialize PSFedAvg strategy.
        
        Args:
            mask_fraction: Fraction of parameters to share each round.
            mask_selector: Custom mask selection strategy. If None, uses
                          RandomMaskSelector with mask_fraction and mask_seed.
            mask_seed: Random seed for default RandomMaskSelector.
            fraction_fit: Fraction of clients for training.
            fraction_evaluate: Fraction of clients for evaluation.
            min_fit_clients: Minimum clients required for training.
            min_evaluate_clients: Minimum clients for evaluation.
            min_available_clients: Minimum available clients.
            evaluate_fn: Server-side evaluation function.
            on_fit_config_fn: Function to generate fit config.
            on_evaluate_config_fn: Function to generate evaluate config.
            accept_failures: Whether to accept client failures.
            initial_parameters: Initial model parameters.
            fit_metrics_aggregation_fn: Function to aggregate fit metrics.
            evaluate_metrics_aggregation_fn: Function to aggregate eval metrics.
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        
        self.mask_fraction = mask_fraction
        self.mask_seed = mask_seed
        
        # Use provided selector or create default
        if mask_selector is not None:
            self.mask_selector = mask_selector
        else:
            self.mask_selector = RandomMaskSelector(
                fraction=mask_fraction,
                seed=mask_seed,
            )
        
        self._current_mask: Mask | None = None
        self._current_round: int = 0
        self._num_parameters: int | None = None
        self._full_parameters: np.ndarray | None = None
    
    def initialize_parameters(
        self, 
        client_manager: ClientManager
    ) -> Parameters | None:
        """Initialize global model parameters.
        
        Args:
            client_manager: Flower client manager.
        
        Returns:
            Initial parameters or None.
        """
        initial = super().initialize_parameters(client_manager)
        
        if initial is not None:
            # Store full parameters and compute total count
            arrays = parameters_to_ndarrays(initial)
            self._full_parameters = np.concatenate([a.ravel() for a in arrays])
            self._num_parameters = len(self._full_parameters)
            self._param_shapes = [a.shape for a in arrays]
            
            logger.info(
                f"PSFedAvg initialized with {self._num_parameters} parameters, "
                f"sharing {self.mask_fraction:.1%} per round"
            )
        
        return initial
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure clients for training with partial parameters.
        
        Args:
            server_round: Current round number.
            parameters: Current global parameters.
            client_manager: Flower client manager.
        
        Returns:
            List of (client, fit_instructions) tuples.
        """
        self._current_round = server_round
        
        # Get full parameters
        arrays = parameters_to_ndarrays(parameters)
        self._full_parameters = np.concatenate([a.ravel() for a in arrays])
        self._param_shapes = [a.shape for a in arrays]
        
        if self._num_parameters is None:
            self._num_parameters = len(self._full_parameters)
        
        # Generate mask for this round
        self._current_mask = self.mask_selector.select(
            num_parameters=self._num_parameters,
            round_num=server_round,
        )
        
        logger.info(
            f"Round {server_round}: Selected {self._current_mask.count}/{self._num_parameters} "
            f"parameters ({self._current_mask.fraction:.1%})"
        )
        
        # Extract partial parameters
        partial_params = self._full_parameters[self._current_mask.data]
        
        # Convert to Flower Parameters (single flat array)
        partial_parameters = ndarrays_to_parameters([partial_params])
        
        # Get base config
        config: dict[str, Scalar] = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)
        
        # Add mask information to config
        mask_config = mask_to_config(self._current_mask)
        config.update(mask_config)
        
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
        )
        
        # Create fit instructions for each client
        fit_ins = FitIns(partial_parameters, config)
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate partial parameter updates from clients.
        
        Uses weighted averaging (FedAvg) on the partial parameters,
        then merges back into the full parameter vector.
        
        Args:
            server_round: Current round number.
            results: Successful client results.
            failures: Failed client results.
        
        Returns:
            Tuple of (aggregated_parameters, metrics).
        """
        if not results:
            return None, {}
        
        if self._current_mask is None or self._full_parameters is None:
            logger.error("No mask or parameters available for aggregation")
            return None, {}
        
        # Collect partial updates and weights
        partial_updates: list[np.ndarray] = []
        weights: list[int] = []
        
        for client, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            if len(arrays) > 0:
                partial_updates.append(arrays[0])
                weights.append(fit_res.num_examples)
        
        if not partial_updates:
            return None, {}
        
        # Weighted average of partial updates
        total_weight = sum(weights)
        aggregated_partial = np.zeros(self._current_mask.count, dtype=np.float32)
        
        for update, weight in zip(partial_updates, weights):
            if len(update) != self._current_mask.count:
                logger.warning(
                    f"Client returned {len(update)} params, expected {self._current_mask.count}"
                )
                continue
            aggregated_partial += (weight / total_weight) * update
        
        # Merge partial update into full parameters
        self._full_parameters[self._current_mask.data] = aggregated_partial
        
        # Convert back to shaped arrays for Flower
        shaped_arrays = self._reshape_to_arrays(self._full_parameters)
        aggregated_parameters = ndarrays_to_parameters(shaped_arrays)
        
        # Aggregate metrics
        metrics: dict[str, Scalar] = {
            "partial_fraction": self._current_mask.fraction,
            "partial_count": self._current_mask.count,
        }
        
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
            metrics.update(aggregated_metrics)
        
        return aggregated_parameters, metrics
    
    def _reshape_to_arrays(self, flat_params: np.ndarray) -> NDArrays:
        """Reshape flat parameters back to per-tensor arrays.
        
        Args:
            flat_params: 1D array of all parameters.
        
        Returns:
            List of shaped numpy arrays.
        """
        arrays = []
        offset = 0
        for shape in self._param_shapes:
            numel = int(np.prod(shape))
            arr = flat_params[offset : offset + numel].reshape(shape)
            arrays.append(arr.astype(np.float32))
            offset += numel
        return arrays
    
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, Any]]:
        """Configure clients for evaluation.
        
        For evaluation, we send full parameters (not partial) to get
        accurate model performance metrics.
        
        Args:
            server_round: Current round number.
            parameters: Current global parameters.
            client_manager: Flower client manager.
        
        Returns:
            List of (client, evaluate_instructions) tuples.
        """
        # Use parent implementation which sends full parameters
        return super().configure_evaluate(
            server_round, parameters, client_manager
        )
    
    @property
    def current_mask(self) -> Mask | None:
        """The mask used in the current/most recent round."""
        return self._current_mask
    
    @property
    def num_parameters(self) -> int | None:
        """Total number of parameters in the model."""
        return self._num_parameters
