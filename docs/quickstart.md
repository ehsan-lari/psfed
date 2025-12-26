# Getting Started with PSFed

This guide will help you get started with PSFed in under 5 minutes.

## Installation

```bash
pip install psfed
```

## Basic Usage (Without Flower)

If you just want to understand the core concepts:

```python
import torch
import torch.nn as nn
from psfed import FlattenedModel, RandomMaskSelector, Mask

# 1. Create your PyTorch model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# 2. Wrap it for partial sharing
flat_model = FlattenedModel(model)
print(f"Total parameters: {flat_model.num_parameters:,}")

# 3. Create a mask selector
selector = RandomMaskSelector(
    fraction=0.5,  # Share 50% of parameters
    seed=42        # For reproducibility
)

# 4. Generate mask for a specific round
mask = selector.select(
    num_parameters=flat_model.num_parameters,
    round_num=1
)
print(f"Selected: {mask.count:,} / {mask.size:,} ({mask.fraction:.1%})")

# 5. Extract selected parameters (what server sends to client)
partial_params = flat_model.extract(mask)
print(f"Partial params shape: {partial_params.shape}")

# 6. Apply partial update (what client does when receiving)
# This updates only the masked positions, keeping others unchanged
flat_model.apply(partial_params, mask)
```

## With Flower (Federated Learning)

### Server Setup

```python
import flwr as fl
from flwr.common import ndarrays_to_parameters
from psfed import FlattenedModel
from psfed.flower import PSFedAvg

# Create model and get initial parameters
model = YourModel()
flat_model = FlattenedModel(model)
initial_params = ndarrays_to_parameters([flat_model.flatten()])

# Create strategy with partial sharing
strategy = PSFedAvg(
    fraction_fit=0.1,
    min_fit_clients=2,
    min_available_clients=2,
    initial_parameters=initial_params,
    # PSFed parameters
    mask_fraction=0.5,  # Share 50% each round
    mask_seed=42,       # Reproducible
)

# Start server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)
```

### Client Setup

```python
import flwr as fl
import torch
from psfed.flower import PSFedClient

class MyClient(PSFedClient):
    def __init__(self, model, train_data):
        super().__init__(model)
        self.train_data = train_data
    
    def train_local(self, config: dict) -> int:
        """Train on local data. Return number of samples."""
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        self.model.train()
        total = 0
        for x, y in self.train_data:
            optimizer.zero_grad()
            loss = criterion(self.model(x), y)
            loss.backward()
            optimizer.step()
            total += len(y)
        
        return total
    
    def evaluate_local(self, config: dict):
        """Optional: evaluate on local data."""
        return 0.0, 0, {}

# Start client
client = MyClient(model, train_loader)
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=client.to_client(),
)
```

## Custom Mask Selectors

PSFed provides several built-in selectors:

```python
from psfed import (
    RandomMaskSelector,      # Per-round random
    TopKMagnitudeSelector,   # Largest values
    GradientBasedSelector,   # Largest gradients
    StructuredMaskSelector,  # Layer-based
    FixedMaskSelector,       # User-defined
    ClientSpecificMaskSelector,  # Per-client different
)

# Random (default)
selector = RandomMaskSelector(fraction=0.5, seed=42)

# Top-k by magnitude
selector = TopKMagnitudeSelector(fraction=0.3)
mask = selector.select(1000, round_num=1, flat_model=flat_model)

# Fixed indices
selector = FixedMaskSelector(indices=[0, 1, 2, 100, 101])
```

## Creating Custom Selectors

```python
from psfed import MaskSelector, Mask

class MySelector(MaskSelector):
    def select(self, num_parameters: int, round_num: int, **kwargs) -> Mask:
        # Your selection logic here
        # Example: select even indices
        indices = list(range(0, num_parameters, 2))
        return Mask.from_indices(indices[:self.get_count(num_parameters)], num_parameters)
```

## Next Steps

- Check out the [examples](../examples/) directory
- Read the [API documentation](api.md)
- Learn about the [research background](research.md)
