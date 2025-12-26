# PSFed Documentation

Welcome to the PSFed documentation!

## Overview

PSFed (Partial Sharing Federated Learning) is a Python package that implements 
partial model sharing in federated learning. Instead of communicating the entire 
model between server and clients, PSFed enables selective parameter synchronization 
based on configurable masking strategies.

## Key Concepts

### Partial Model Sharing

In standard federated learning:
1. Server sends full model θ to clients
2. Clients train locally, compute Δθ
3. Clients send full Δθ back to server
4. Server aggregates all updates

With partial sharing:
1. Server selects a mask M ⊂ {1, ..., d} where d is total parameters
2. Server sends only θ[M] to clients
3. Clients update their local model at masked positions
4. Clients train locally on ALL parameters
5. Clients send back only θ[M] after training
6. Server aggregates partial updates

### Why Partial Sharing?

- **Communication efficiency**: Reduce bandwidth by 50-90%
- **Privacy**: Some parameters never leave clients
- **Personalization**: Non-shared parameters can adapt to local data
- **Convergence**: Dynamic masking ensures all parameters eventually synchronize

## Installation

```bash
pip install psfed
```

## Quick Start

See the [Getting Started](quickstart.md) guide.

## API Reference

See the [API Documentation](api.md).

## Research Background

See [Research Notes](research.md) for theoretical background.
