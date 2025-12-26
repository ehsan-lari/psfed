"""PSFed client for MNIST example.

This client trains on a subset of MNIST data and communicates with the
server using partial model sharing.

Usage:
    python client.py --client-id ID [--epochs N]

Example:
    python client.py --client-id 0 --epochs 1
"""

import argparse
import logging
from pathlib import Path

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import create_model
from psfed.flower import PSFedClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MNISTClient(PSFedClient):
    """MNIST client with partial model sharing support."""

    def __init__(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        testloader: DataLoader,
        epochs: int = 1,
        lr: float = 0.01,
    ) -> None:
        """Initialize the MNIST client.

        Args:
            model: PyTorch model to train.
            trainloader: Training data loader.
            testloader: Test data loader.
            epochs: Local training epochs per round.
            lr: Learning rate.
        """
        super().__init__(model)
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs = epochs
        self.lr = lr
        self.model.to(DEVICE)

    def train_local(self, config: dict) -> int:
        """Train the model locally.

        Args:
            config: Configuration from server.

        Returns:
            Number of training samples used.
        """
        epochs = config.get("epochs", self.epochs)
        lr = config.get("lr", self.lr)

        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        total_samples = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for images, labels in self.trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                total_samples += len(labels)

            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(self.trainloader):.4f}")

        return total_samples

    def evaluate_local(self, config: dict) -> tuple[float, int, dict]:
        """Evaluate the model on local test data.

        Args:
            config: Configuration from server.

        Returns:
            Tuple of (loss, num_samples, metrics).
        """
        criterion = nn.CrossEntropyLoss()

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(images)

                loss = criterion(outputs, labels)
                total_loss += loss.item() * len(labels)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += len(labels)

        accuracy = correct / total
        avg_loss = total_loss / total

        logger.info(f"Evaluation: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

        return avg_loss, total, {"accuracy": accuracy}


def load_data(client_id: int, num_clients: int = 10) -> tuple[DataLoader, DataLoader]:
    """Load MNIST data for a specific client.

    Each client gets a disjoint subset of the training data.

    Args:
        client_id: Client identifier (0-indexed).
        num_clients: Total number of clients.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # Download/load MNIST
    data_dir = Path("./data")
    trainset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    testset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    # Partition training data among clients
    total_size = len(trainset)
    partition_size = total_size // num_clients
    start_idx = client_id * partition_size
    end_idx = start_idx + partition_size

    indices = list(range(start_idx, end_idx))
    client_trainset = Subset(trainset, indices)

    logger.info(f"Client {client_id}: {len(indices)} training samples")

    trainloader = DataLoader(client_trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    return trainloader, testloader


def main() -> None:
    """Run the PSFed client."""
    parser = argparse.ArgumentParser(description="PSFed MNIST Client")
    parser.add_argument(
        "--client-id", type=int, required=True, help="Client ID (0-indexed)"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Local training epochs per round"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "--server", type=str, default="127.0.0.1:8080", help="Server address"
    )
    args = parser.parse_args()

    logger.info(f"Starting client {args.client_id}")
    logger.info(f"Device: {DEVICE}")

    # Load data
    trainloader, testloader = load_data(args.client_id)

    # Create model
    model = create_model()

    # Create client
    client = MNISTClient(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        epochs=args.epochs,
        lr=args.lr,
    )

    logger.info(f"Model has {client.num_parameters:,} parameters")
    logger.info(f"Connecting to server at {args.server}...")

    # Start client
    fl.client.start_client(
        server_address=args.server,
        client=client.to_client(),
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
