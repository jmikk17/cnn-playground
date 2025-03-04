import torch

from cnn import CNN


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_for_prediction():
    model = CNN()
    model.load_state_dict(torch.load("mnist_cnn_state_dict.pth"))
    model.eval()

    return model


def predict_digit(model, device, image):
    image_tensor = torch.from_numpy(image).float()
    # Dimension = [batch size, channels, height, width]
    # Add batch size and channel dimension to image
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        _, predicted = output.max(1)
        return predicted.item()
