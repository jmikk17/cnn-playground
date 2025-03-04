# CNN Playground

This project is a small implementation of a Convolutional Neural Network (CNN) for recognizing handwritten digits. It's trained on the MNIST dataset, and can be used to predict digits in a supplied image file (with varying success).

## Usage

Install the required dependencies using:
```bash
pip install -r requirements.txt
```

### Predicting Digits

To predict digits from an input image using `mnist_cnn_state_dict.pth` as the model, run the following command:
```bash
python main.py --predict <path_to_image>
```

### Training the Model

To re-train the model and regenerate `mnist_cnn_state_dict.pth`, run the following command (this will download the MNIST dataset):
```bash
python main.py --train
```

## Notes

- The model is trained on the MNIST dataset, which consists of 60,000 28x28 pixel grayscale images of handwritten digits (+ 10,000 for testing).
- The trained model is saved as `mnist_cnn_state_dict.pth` and can be used for predictions.
- The model usually achieves 95-99% when testing on MNIST data, but struggles with real life images for various reasons (work in progress).
