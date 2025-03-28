import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import auxil
from cnn import CNN


def run_training(loss: str = "cross_entropy", epochs: int = 5) -> None:
    """Load MNIST data, and run the training.

    Args:
        loss (str): Loss function to use, either "cross_entropy" or "mse"
        epochs (int): Number of epochs to train

    """
    # Transforms data to PyTorch tensors and normalizes it
    # Normalized with (image - mean) / std, where mean and std are MNIST dataset values
    # For a general dataset mean and std should be calculated, but for MNIST they are known
    transform = transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ],
    )

    # Download and load the predifined 60,000 training data points
    # train_loader automatically batches the data and shuffles in the beginning of each epoch
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Same with 10,000 test data points
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    if loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif loss == "mse":
        criterion = nn.MSELoss()

    train(epochs, train_loader, test_loader, criterion)


def train(epochs: int, train_loader: DataLoader, test_loader: DataLoader, criterion: nn.Module) -> None:
    """Train CNN for a number of epochs.

    Args:
        epochs (int): Numboer of epochs
        train_loader (DataLoader): Object that loads training data in batches
        test_loader (DataLoader): Object that load test data in batches
        criterion (nn.Module): Loss function

    """
    # Setup model and Adam optimizer
    device = auxil.get_device()
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Epoch = complete pass through training data
    for epoch in range(epochs):
        print(f"Training epoch: {epoch + 1}")
        # Set model to training mode to apply dropout
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Run through 60,000/64 = 938 batches
        # Each itteration of a DataLoader returns a batch of data and targets
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move data to device (GPU or CPU)
            data, targets = data.to(device), targets.to(device)

            # Accumulates by default, so need to zero
            optimizer.zero_grad()

            # Forward pass, calculate loss
            outputs = model(data)
            loss = criterion(outputs, targets)

            # Backward pass (calculate gradients), update weights
            loss.backward()
            optimizer.step()

            # Add loss for current batch
            running_loss += loss.item()

            # Get index of max value along dimension 1 (10 output nodes), i.e. get the predicted digit
            # PyTorch dimensions are (batch, features), so dimension 1 is the features
            _, predicted = outputs.max(1)

            # Get batch size (dimension 0)
            total += targets.size(0)

            # Create a bool tensor with True where predicted == target
            result_bool = predicted.eq(targets)

            # Sum the True values and change from tensor to int with .item()
            correct += result_bool.sum().item()

            # Print stat every 100 batches
            # Avg loss for a batch is a weird meassurement here, since the model changes weights through a batch,
            # but correlates well with accuracy
            if batch_idx % 100 == 0 and batch_idx != 0:
                avg_loss = running_loss / (batch_idx + 1)
                print(
                    f"Batch: {batch_idx}, "
                    f"Batch loss: {loss.item():.4f}, Avg loss in current epoch: {avg_loss:.4f}, "
                    f"Avg accuracy in current epoch: {100 * correct / total:.2f}%",
                )

        # Set model to evaluation mode to turn off dropout
        model.eval()
        evaluate(model, device, test_loader)

    save_model(model)


def evaluate(model: CNN, device: torch.device, test_loader: DataLoader) -> None:
    """Evaluate model on test data.

    Args:
        model (CNN): Instance of CNN model to predict with
        device (torch.device): Device to run evaluation on (e.g. GPU or CPU)
        test_loader (DataLoader): Object that load test data in batches

    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


def save_model(model: CNN) -> None:
    """Save parameters of the CNN.

    Args:
        model (CNN): Instance of CNN model to save

    """
    torch.save(model.state_dict(), "mnist_cnn_state_dict.pth")
    print("Model state dict saved as 'mnist_cnn_state_dict.pth'")
