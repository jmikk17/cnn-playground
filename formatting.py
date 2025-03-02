import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def extract_digits(image_path: str, show_steps: bool = True, adaptive_threshold: bool = False):
    if not Path(image_path).exists():
        sys.exit(f"Error: Image file not found: {image_path}")

    original_image = cv2.imread(image_path)
    if original_image is None:
        sys.exit("Error: Reading of image file")

    if len(original_image.shape) == 3:
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = original_image.copy()

    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    if adaptive_threshold:
        # get threshold from mean or weighted sum of area minus C, with area size decided by blockSize
        # this will probalby be helpfull for webcam implementation
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2,
        )
    else:
        # set global threshold to 127
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if False:
        # contour debugging
        test = cv2.drawContours(original_image, contours, -1, (0, 255, 0), 3)
        cv2.imshow("Output", test)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    digits = []
    pos = []

    if show_steps:
        result_image = original_image.copy()

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        rect_range = binary[y : y + h, x : x + w]

        size = max(w, h) + 10
        square_digit = np.zeros((size, size), dtype=np.uint8)

        offset_x = (size - w) // 2
        offset_y = (size - h) // 2
        square_digit[offset_y : offset_y + h, offset_x : offset_x + w] = rect_range

        resized_digit = cv2.resize(square_digit, (28, 28), interpolation=cv2.INTER_AREA)

        # mnist does this
        normalized_digit = resized_digit.astype("float32") / 255.0

        digits.append(normalized_digit)
        pos.append((x, y))

        if show_steps:
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if show_steps:
        # TODO plotting wrong
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].set_title("Detected Digits")
        ax[0].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        ax[0].axis("off")

        ax[1].set_title("MNIST format example")
        ax[1].imshow(digits[0], cmap="gray")
        ax[1].axis("off")

        plt.show()

    return normalized_digit
