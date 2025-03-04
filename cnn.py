import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        """Predicts numbers from images sized 28x28 pixels.

        NN architecture stolen from https://colab.research.google.com/github/trekhleb/machine-learning-experiments/blob/master/experiments/digits_recognition_cnn/digits_recognition_cnn.ipynb#scrollTo=xC1stUA99j0V
        """
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input):
        # Dimensions = (barth size, channels, height, width)
        # Here assuming grey scale images, so input channels = 1
        # l1 shape: (N, 1, 28, 28) -> (N, 8, 24, 24)
        # d_out = (d_in - kernel_size) / stride + 1 when dilation 1 and padding 0
        l1 = F.relu(self.conv1(input))
        # l2 shape: (N, 8, 24, 24) -> (N, 8, 12, 12)
        l2 = F.max_pool2d(l1, (2, 2))
        # l3 shape: (N, 8, 12, 12) -> (N, 16, 8, 8)
        l3 = F.relu(self.conv2(l2))
        # l4 shape: (N, 16, 8, 8) -> (N, 16, 4, 4)
        l4 = F.max_pool2d(l3, (2, 2))
        # l5 shape: (N, 4, 4, 16) -> (N, 256)
        l5 = torch.flatten(l4, 1)
        # l6 shape: (N, 256) -> (N, 128)
        l6 = F.relu(self.fc1(l5))
        # l7 shape: (N, 128) -> (N, 128)
        l7 = self.dropout(l6)
        # l8 shape: (N, 128) -> (N, 10)
        return self.fc2(l7)


if __name__ == "__main__":
    net = CNN()
    print(net)
