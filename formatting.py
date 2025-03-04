import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def extract_digits(image_path: str, *, show_steps: bool = False, ada_thresh: bool = False) -> tuple:
    """Extract digits in MNIST type format from an image.

    Loads of empirical values are used in this function (e.g. for blurring and threshold),
    so it might not work for all images.

    Args:
        image_path (str): Path to image file
        show_steps (bool, optional): Show rectangles on original image and extracted digits in MNIST format.
            Defaults to False.
        ada_thresh (bool, optional): Use adaptive thresholding instead of global for finding digits on image.
            Defaults to False.

    Returns:
        tuple: Tuple containing list of digits in MNIST format, list of positions of digits and image with rectangles

    """
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

    if ada_thresh:
        # Get threshold from mean or weighted sum of area minus C, with area decided by blockSize
        # This will probalby be helpfull for camera implementation
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2,
        )
    else:
        # Set a global threshold to 127
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        sys.exit("Error: No digits found in image")

    digits = []
    pos = []

    result_image = original_image.copy()

    for contour in contours:
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        rect_range = binary[y : y + h, x : x + w]

        size = max(w, h) + 10
        square_digit = np.zeros((size, size), dtype=np.uint8)

        offset_x = (size - w) // 2
        offset_y = (size - h) // 2
        square_digit[offset_y : offset_y + h, offset_x : offset_x + w] = rect_range

        # MNIST values used are 28x28 and normalized to 0-1
        resized_digit = cv2.resize(square_digit, (28, 28), interpolation=cv2.INTER_AREA)
        normalized_digit = resized_digit.astype("float32") / 255.0

        digits.append(normalized_digit)
        pos.append((x, y))

        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if show_steps:
        # Plot original image with contours and example of first digit found transformed to MNIST format
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].set_title("Detected Digits")
        ax[0].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        ax[0].axis("off")

        ax[1].set_title("MNIST format example")
        ax[1].imshow(digits[0], cmap="gray")
        ax[1].axis("off")

        plt.show()

    if show_steps:
        # Plot all found digits in MNIST format
        fig, ax = plt.subplots(1, len(digits), figsize=(12, 6))
        fig.suptitle("Extracted Digits")
        for i, digit in enumerate(digits):
            ax[i].imshow(digit, cmap="gray")
            ax[i].axis("off")

        plt.show()

    return digits, pos, result_image
