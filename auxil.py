import numpy as np
import torch

from cnn import CNN


def get_device() -> torch.device:
    """Return the device to use for training and prediction."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_for_prediction() -> CNN:
    """Load saved model for prediction."""
    model = CNN()
    model.load_state_dict(torch.load("mnist_cnn_state_dict.pth", weights_only=True))
    model.eval()

    return model


def predict_digit(model: CNN, device: torch.device, image: np.ndarray) -> int:
    """Predict a digit from an image.

    Args:
        model (CNN): Model used for prediction
        device (torch.device): Device to run prediction on
        image (np.ndarray): 28x28 image of a digit

    Returns:
        int: Predicted number

    """
    image_tensor = torch.from_numpy(image).float()
    # Dimension required is [batch size, channels, height, width], so we add batch size and channel dimension to image
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        _, predicted = output.max(1)
        return predicted.item()
