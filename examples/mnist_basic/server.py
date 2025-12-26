"""PSFed server for MNIST example.

This server uses PSFedAvg strategy to orchestrate partial model sharing
with connected clients.

Usage:
    python server.py [--rounds N] [--fraction F] [--seed S]

Example:
    python server.py --rounds 10 --fraction 0.5 --seed 42
"""

import argparse
import logging

import flwr as fl
from flwr.common import ndarrays_to_parameters

from model import create_model
from psfed import FlattenedModel
from psfed.flower import PSFedAvg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the PSFed server."""
    parser = argparse.ArgumentParser(description="PSFed MNIST Server")
    parser.add_argument(
        "--rounds", type=int, default=10, help="Number of federation rounds"
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.5,
        help="Fraction of parameters to share (0.0-1.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--min-clients", type=int, default=2, help="Minimum clients for training"
    )
    parser.add_argument(
        "--address", type=str, default="0.0.0.0:8080", help="Server address"
    )
    args = parser.parse_args()

    # Create initial model
    model = create_model()
    flat_model = FlattenedModel(model)
    
    logger.info(f"Model has {flat_model.num_parameters:,} parameters")
    logger.info(f"Sharing {args.fraction:.0%} = {int(flat_model.num_parameters * args.fraction):,} parameters per round")

    # Get initial parameters as flat array
    initial_params = flat_model.flatten()
    initial_parameters = ndarrays_to_parameters([initial_params])

    # Create PSFedAvg strategy
    strategy = PSFedAvg(
        fraction_fit=1.0,  # Use all available clients
        fraction_evaluate=1.0,
        min_fit_clients=args.min_clients,
        min_evaluate_clients=args.min_clients,
        min_available_clients=args.min_clients,
        initial_parameters=initial_parameters,
        # PSFed-specific parameters
        mask_fraction=args.fraction,
        mask_seed=args.seed,
    )

    # Start server
    logger.info(f"Starting server at {args.address}")
    logger.info(f"Waiting for {args.min_clients} clients...")
    
    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
