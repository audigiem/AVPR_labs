import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from edgeDetection import load_and_preprocess_image


def harris_corner_detection(
    image: np.ndarray,
    block_size: int = 2,
    ksize: int = 3,
    k: float = 0.04,
    threshold: float = 0.01,
) -> np.ndarray:
    """
    Harris Corner Detection
    Args:
        image_path: np.ndarray: Input image in which corners are to be detected
        block_size: int: It is the size of neighbourhood considered for corner detection
        ksize: int: Aperture parameter of the Sobel derivative used.
        k: float: Harris detector free parameter in the equation.
        threshold: float: Threshold for detecting corners based on the response value.

    Returns: np.ndarray: Image with detected corners marked in red
    """
    # Harris corner detection
    dst = cv2.cornerHarris(image, block_size, ksize, k)
    dst = cv2.dilate(dst, None)
    # Create a copy of the original image to draw corners
    corner_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Thresholding to get the corners
    corner_image[dst > threshold * dst.max()] = [0, 0, 255]
    return corner_image



def test_harris_with_various_parameters(image_path: str, target_size=(512, 512), figure_size=(16, 12)):
    """
    Test Harris Corner Detection with various parameters and display results
    """
    possible_params = [
        (2, 3, 0.04, 0.01),
        (2, 5, 0.04, 0.01),
        (4, 3, 0.04, 0.01),
        (2, 3, 0.06, 0.01),
        (2, 3, 0.04, 0.02)
    ]
    corners_with_params = {}

    original_image = load_and_preprocess_image(image_path, target_size)
    for params in possible_params:
        block_size, ksize, k, threshold = params
        corners_img = harris_corner_detection(
            original_image, block_size, ksize, k, threshold
        )
        corners_with_params[params] = corners_img
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=figure_size)

    # set a consistent aspect ratio for all subplots
    for ax in axes.flatten():
        ax.set_aspect('equal', adjustable='box')

    axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    for ax, (params, corner_img) in zip(axes.flatten()[1:], corners_with_params.items()):
        block_size, ksize, k, threshold = params
        ax.imshow(cv2.cvtColor(corner_img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"b:{block_size}, ksize:{ksize}, k:{k}, th:{threshold}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    fig.savefig("../outputs/cornerDetection/harris_parameter_variation.png")



if __name__ == "__main__":
    IMAGES_PATH = [
        # "../assets/100.pgm",
        "../assets/couchersoleil.pgm",
        "../assets/einstein.pgm",
        "../assets/monroe.pgm",
        "../assets/image1.jpg",
        "../assets/image2.jpg",
        "../assets/left_image.png",
    ]
    # Configuration
    TARGET_SIZE = (512, 512)
    FIGURE_SIZE = (16, 8)

    for img_path in IMAGES_PATH:
        test_harris_with_various_parameters(img_path, TARGET_SIZE, FIGURE_SIZE)
