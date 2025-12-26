# MNIST Basic Example

This example demonstrates PSFed with partial model sharing on the MNIST dataset.

## Setup

```bash
pip install psfed torchvision
```

## Running

1. Start the server:
   ```bash
   python server.py
   ```

2. In separate terminals, start clients:
   ```bash
   python client.py --client-id 0
   python client.py --client-id 1
   ```

## What This Example Shows

- How to use `PSFedAvg` strategy with 50% parameter sharing
- How to implement a `PSFedClient` subclass
- How partial parameters are communicated each round

## Expected Output

After 10 rounds with 50% sharing, you should see:
- ~50% communication savings compared to full model sharing
- Model accuracy improving over rounds
- Different parameters shared each round (dynamic masking)
