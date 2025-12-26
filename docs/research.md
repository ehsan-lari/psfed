# Research Background

This document provides the theoretical foundation for partial model sharing in federated learning.

## Problem Setting

Consider a federated learning system with:
- One central server
- $K$ clients, each with local dataset $D_k$
- A shared model parameterized by $\theta \in \mathbb{R}^d$

The objective is to minimize:
$$\min_\theta F(\theta) = \sum_{k=1}^{K} \frac{|D_k|}{|D|} F_k(\theta)$$

where $F_k(\theta) = \mathbb{E}_{(x,y) \sim D_k}[\ell(\theta; x, y)]$

## Standard FedAvg

In FedAvg (McMahan et al., 2017):

1. **Broadcast**: Server sends $\theta^{(t)}$ to selected clients
2. **Local update**: Each client $k$ performs $E$ local SGD steps:
   $$\theta_k^{(t+1)} = \theta^{(t)} - \eta \nabla F_k(\theta^{(t)})$$
3. **Aggregate**: Server computes weighted average:
   $$\theta^{(t+1)} = \sum_k \frac{|D_k|}{|D|} \theta_k^{(t+1)}$$

**Communication cost**: $O(d)$ parameters per client per round.

## Partial Model Sharing

We modify FedAvg by introducing a selection mask $M^{(t)} \subseteq \{1, ..., d\}$:

1. **Partial broadcast**: Server sends only $\theta^{(t)}[M^{(t)}]$
2. **Local integration**: Client $k$ updates their local model:
   $$\theta_k^{(t)}[M^{(t)}] \leftarrow \theta^{(t)}[M^{(t)}]$$
   (non-masked positions retain client's local values)
3. **Full local training**: Client trains on ALL parameters
4. **Partial return**: Client sends back $\theta_k^{(t+1)}[M^{(t)}]$
5. **Partial aggregation**: Server aggregates only masked positions:
   $$\theta^{(t+1)}[M^{(t)}] = \sum_k \frac{|D_k|}{|D|} \theta_k^{(t+1)}[M^{(t)}]$$

**Communication cost**: $O(|M^{(t)}|) \ll O(d)$

## Mask Selection Strategies

### Random Selection (Default)

$$M^{(t)} = \text{RandomSample}(\{1,...,d\}, |M|)$$

With different random seed per round, ensuring:
$$\Pr[\text{parameter } i \text{ selected in any of } T \text{ rounds}] = 1 - (1-p)^T$$

where $p = |M|/d$ is the selection fraction.

For $p = 0.5$ and $T = 10$ rounds: $\Pr \approx 0.999$

### Magnitude-Based Selection

$$M^{(t)} = \text{TopK}(|\theta^{(t)}|, k)$$

Intuition: Large-magnitude parameters may have more impact on model behavior.

### Gradient-Based Selection

$$M^{(t)} = \text{TopK}(|\nabla F(\theta^{(t)})|, k)$$

Intuition: Parameters with large gradients are actively changing.

## Convergence Considerations

### Key Insight

With dynamic masking, every parameter is eventually synchronized:
- Each round synchronizes $p \cdot d$ parameters
- After $T$ rounds, expected coverage is $1 - (1-p)^T$
- For reasonable $p$ and $T$, convergence is maintained

### Potential Issues

1. **Stale parameters**: Non-masked parameters may diverge across clients
2. **Optimizer state**: Momentum/Adam states may become inconsistent
3. **Initialization**: First few rounds have incomplete synchronization

### Mitigations

1. **Full sync periodically**: Every $N$ rounds, do full synchronization
2. **Stateless optimizers**: Use SGD without momentum for simplicity
3. **Warm-up**: Start with higher $p$, decrease over time

## Communication Analysis

| Method | Params/Round | Compression |
|--------|--------------|-------------|
| FedAvg | $d$ | 1x |
| PSFed (50%) | $0.5d$ | 2x |
| PSFed (10%) | $0.1d$ | 10x |

For a ResNet-50 with $d \approx 25M$ parameters:
- FedAvg: 100 MB per round (float32)
- PSFed (50%): 50 MB per round
- PSFed (10%): 10 MB per round

## Related Work

- **FedProx** (Li et al., 2020): Adds proximal term, but full communication
- **Gradient compression** (Alistarh et al., 2017): Compress gradients, not model
- **Model pruning** (Jiang et al., 2022): Permanently remove parameters
- **Partial participation**: Select clients, not parameters

PSFed is complementary to these approaches and can be combined.

## References

1. McMahan, B., et al. (2017). Communication-efficient learning of deep networks from decentralized data. AISTATS.
2. Li, T., et al. (2020). Federated optimization in heterogeneous networks. MLSys.
3. Alistarh, D., et al. (2017). QSGD: Communication-efficient SGD via gradient quantization and encoding. NeurIPS.

## Citation

If you use PSFed in your research, please cite:

```bibtex
@software{psfed2024,
  title = {PSFed: Partial Model Sharing for Federated Learning},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/psfed}
}
```
