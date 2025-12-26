<p align="center">
  <img src="https://raw.githubusercontent.com/ehsan-lari/psfed/blob/main/images/PSFed_icon.png" alt="PSFed Logo" width="400">
</p>

# PSFed: Partial Model Sharing for Federated Learning

[![PyPI version](https://img.shields.io/pypi/v/psfed.svg)](https://pypi.org/project/psfed/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.5+-green.svg)](https://flower.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PSFed** is a research-grade Python package that implements **partial model sharing** in federated learning. Instead of communicating the entire model between server and clients, PSFed enables selective parameter synchronization based on configurable masking strategies.

## Key Features

- 🎯 **Parameter-level granularity**: Share any subset of model parameters
- 🔀 **Dynamic masking**: Masks change per round, ensuring eventual full synchronization
- 📊 **Multiple strategies**: Random, top-k magnitude, gradient-based, and custom selectors
- 🌸 **Flower integration**: Drop-in strategy and client implementations
- 🔥 **PyTorch native**: Works with any `nn.Module`
- ⚡ **Communication efficient**: Reduce bandwidth by 50-90%

## Installation

```bash
pip install psfed
```

Or install from source:

```bash
git clone https://github.com/ehsan-lari/psfed.git
cd psfed
pip install -e ".[dev]"
```

## Quick Start

### Basic Usage (Without Flower)

```python
import torch
import torch.nn as nn
from psfed import FlattenedModel, RandomMaskSelector

# Your PyTorch model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Wrap for partial sharing
flat_model = FlattenedModel(model)
print(f"Total parameters: {flat_model.num_parameters}")

# Create a mask selector (50% of parameters, changes each round)
selector = RandomMaskSelector(fraction=0.5, seed=42)

# Generate mask for round 1
mask = selector.select(flat_model.num_parameters, round_num=1)
print(f"Selected {mask.count} / {mask.size} parameters")

# Extract selected parameters (for sending to clients)
partial_params = flat_model.extract(mask)

# ... transmit partial_params to client ...

# Apply received parameters (on client side)
flat_model.apply(partial_params, mask)
```

### Federated Learning with Flower

**Server:**

```python
import flwr as fl
from psfed.flower import PSFedAvg

# Define strategy with partial sharing
strategy = PSFedAvg(
    fraction_fit=0.1,
    fraction_evaluate=0.1,
    min_fit_clients=2,
    min_available_clients=2,
    # PSFed-specific parameters
    mask_fraction=0.5,           # Share 50% of parameters
    mask_strategy="random",      # Per-round random selection
    mask_seed=42,                # Reproducibility
)

# Start server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)
```

**Client:**

```python
import flwr as fl
from psfed.flower import PSFedClient

class MyClient(PSFedClient):
    def __init__(self, model, trainloader):
        super().__init__(model)
        self.trainloader = trainloader
    
    def train_local(self, epochs: int = 1):
        # Your training logic here
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for _ in range(epochs):
            for images, labels in self.trainloader:
                optimizer.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()

# Start client
client = MyClient(model, trainloader)
fl.client.start_client(server_address="127.0.0.1:8080", client=client)
```

## Mask Selection Strategies

PSFed provides several built-in mask selection strategies:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `RandomMaskSelector` | Per-round random selection | Default, ensures coverage over time |
| `TopKMagnitudeSelector` | Select largest parameters by absolute value | Focus on important weights |
| `GradientBasedSelector` | Select by gradient magnitude | Active/changing parameters |
| `StructuredMaskSelector` | Layer-aware selection | Preserve structure |
| `FixedMaskSelector` | User-defined indices | Full control |

### Custom Selector

```python
from psfed import MaskSelector, Mask

class MyCustomSelector(MaskSelector):
    def select(
        self, 
        num_parameters: int, 
        round_num: int,
        **kwargs
    ) -> Mask:
        # Your selection logic
        indices = your_custom_logic(num_parameters, round_num)
        return Mask.from_indices(indices, size=num_parameters)
```

## API Reference

### Core Classes

- **`FlattenedModel`**: Wraps a PyTorch model for flatten/unflatten operations
- **`Mask`**: Binary mask with efficient storage and operations
- **`MaskSelector`**: Abstract base class for selection strategies

### Flower Integration

- **`PSFedAvg`**: FedAvg strategy with partial model sharing
- **`PSFedClient`**: Base client class handling partial parameters

## Research Background

This package implements the concept of **partial model sharing** in federated learning, where communication efficiency is achieved by transmitting only a subset of model parameters each round.

Key properties:
- **Communication reduction**: Proportional to `(1 - mask_fraction)`
- **Convergence**: Dynamic masking ensures all parameters are eventually synchronized
- **Privacy**: Non-shared parameters remain local

For theoretical analysis, see [docs/research.md](docs/research.md).

## Examples

- [MNIST Basic](examples/mnist_basic/) - Simple partial sharing example
- [CIFAR-10 Advanced](examples/cifar10_advanced/) - Custom selectors and analysis

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Setup development environment
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest

# Type checking
mypy src/psfed

# Linting
ruff check src/psfed
```

## Citation

If you use PSFed in your research, please cite:

```bibtex
@article{lari2025resilience,
  title={Resilience in Online Federated Learning: Mitigating Model-Poisoning Attacks via Partial Sharing},
  author={Lari, Ehsan and Arablouei, Reza and Gogineni, Vinay Chakravarthi and Werner, Stefan},
  journal={IEEE Transactions on Signal and Information Processing over Networks},
  year={2025},
  publisher={IEEE}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.


