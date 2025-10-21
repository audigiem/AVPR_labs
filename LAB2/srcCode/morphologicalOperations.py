import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from edgeDetection import load_and_preprocess_image


def apply_morphological_operation(
    image: np.ndarray,
    operation: str = "dilation",
    kernel_size: int = 3,
    iterations: int = 1,
) -> np.ndarray:
    """
    Apply morphological operation on the image
    Args:
        image: np.ndarray: Input binary image
        operation: str: Type of morphological operation ('dilation', 'erosion', 'opening', 'closing')
        kernel_size: int: Size of the structuring element
        iterations: int: Number of times the operation is applied

    Returns: np.ndarray: Image after applying the morphological operation
    """
    if operation not in ["dilation", "erosion", "opening", "closing"]:
        raise ValueError(
            "Unsupported morphological operation, choose from 'dilation', 'erosion', 'opening', 'closing'"
        )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    if operation == "dilation":
        return cv2.dilate(image, kernel, iterations=iterations)
    elif operation == "erosion":
        return cv2.erode(image, kernel, iterations=iterations)
    elif operation == "opening":
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == "closing":
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        raise ValueError("Unsupported morphological operation")


def test_morphological_operations_with_various_parameters(
    image_path: str, target_size=(512, 512), figure_size=(16, 12)
):
    """
    Test Morphological Operations with various parameters and display results
    """
    possible_params = [
        ("dilation", 3, 1),
        ("erosion", 3, 1),
        ("opening", 5, 1),
        ("closing", 5, 1),
        ("dilation", 7, 2),
    ]
    morphed_images_with_params = {}

    original_image = load_and_preprocess_image(image_path, target_size)
    # Convert to binary image for morphological operations
    _, binary_image = cv2.threshold(original_image, 127, 255, cv2.THRESH_BINARY)

    for params in possible_params:
        operation, kernel_size, iterations = params
        morphed_img = apply_morphological_operation(
            binary_image, operation, kernel_size, iterations
        )
        morphed_images_with_params[params] = morphed_img
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=figure_size)
    axes = axes.ravel()
    axes[0].imshow(binary_image, cmap="gray")
    axes[0].set_title("Original Binary Image")
    axes[0].axis("off")

    for i, (params, img) in enumerate(morphed_images_with_params.items(), start=1):
        operation, kernel_size, iterations = params
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"{operation}, ksize={kernel_size}, iter={iterations}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
    fig.savefig(
        "../outputs/morphologicalOperations/morphological_parameter_variation.png"
    )


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
        test_morphological_operations_with_various_parameters(
            img_path, TARGET_SIZE, FIGURE_SIZE
        )
