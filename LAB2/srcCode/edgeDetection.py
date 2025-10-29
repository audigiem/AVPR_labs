"""
Task 1: Edge Detection
This script performs edge detection using multiple filters (Sobel, Prewitt, and Scharr)
and compares their results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os

def load_and_preprocess_image(image_path=None, target_size=(512, 512)):
    """Load and preprocess image with consistent sizing"""
    if image_path is None:
        # Create a synthetic test image with clear edges
        image = np.zeros(target_size, dtype=np.uint8)
        # Add rectangles (scaled to target size)
        scale_x, scale_y = target_size[1] / 300, target_size[0] / 300
        cv2.rectangle(
            image,
            (int(50 * scale_x), int(50 * scale_y)),
            (int(150 * scale_x), int(150 * scale_y)),
            255,
            -1,
        )
        cv2.rectangle(
            image,
            (int(180 * scale_x), int(80 * scale_y)),
            (int(250 * scale_x), int(200 * scale_y)),
            128,
            -1,
        )
        # Add circle
        cv2.circle(
            image, (int(200 * scale_x), int(220 * scale_y)), int(40 * scale_x), 200, -1
        )
        # Add diagonal line
        cv2.line(
            image,
            (int(20 * scale_x), int(200 * scale_y)),
            (int(120 * scale_x), int(280 * scale_y)),
            180,
            3,
        )
    else:
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            print(f"Current directory: {os.getcwd()}")
            print(
                f"Files in assets/: {os.listdir('assets') if os.path.exists('assets') else 'Assets folder not found'}"
            )
            raise FileNotFoundError(f"Could not find image at {image_path}")

        # Load image from file
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Resize to target size
        print(f"Original image size: {image.shape}")
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        print(f"Resized to: {image.shape}")

    return image

def apply_sobel_filter(image):
    """Apply Sobel filter for edge detection."""
    # Sobel in X direction
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    # Sobel in Y direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # Combine both directions
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    # Normalize to 0-255 range
    sobel_combined = np.uint8(sobel_combined / sobel_combined.max() * 255)
    return sobel_combined, sobel_x, sobel_y


def apply_prewitt_filter(image):
    """Apply Prewitt filter for edge detection."""
    # Prewitt kernels
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    # Apply convolution
    prewitt_x = ndimage.convolve(image.astype(float), kernel_x)
    prewitt_y = ndimage.convolve(image.astype(float), kernel_y)

    # Combine both directions
    prewitt_combined = np.sqrt(prewitt_x**2 + prewitt_y**2)
    # Normalize to 0-255 range
    prewitt_combined = np.uint8(prewitt_combined / prewitt_combined.max() * 255)
    return prewitt_combined, prewitt_x, prewitt_y


def apply_scharr_filter(image):
    """Apply Scharr filter for edge detection."""
    # Scharr in X direction
    scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    # Scharr in Y direction
    scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
    # Combine both directions
    scharr_combined = np.sqrt(scharr_x**2 + scharr_y**2)
    # Normalize to 0-255 range
    scharr_combined = np.uint8(scharr_combined / scharr_combined.max() * 255)
    return scharr_combined, scharr_x, scharr_y


def display_edge_detection_results(original, sobel, prewitt, scharr):
    """Display all edge detection results for comparison."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Edge Detection Comparison", fontsize=16, fontweight="bold")

    # Original image
    axes[0, 0].imshow(original, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Sobel results
    axes[0, 1].imshow(sobel, cmap="gray")
    axes[0, 1].set_title("Sobel Filter")
    axes[0, 1].axis("off")

    # Prewitt results
    axes[0, 2].imshow(prewitt, cmap="gray")
    axes[0, 2].set_title("Prewitt Filter")
    axes[0, 2].axis("off")

    # Scharr results
    axes[0, 3].imshow(scharr, cmap="gray")
    axes[0, 3].set_title("Scharr Filter")
    axes[0, 3].axis("off")

    # Side by side comparisons
    axes[1, 0].imshow(original, cmap="gray")
    axes[1, 0].set_title("Original")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(sobel, cmap="gray")
    axes[1, 1].set_title("Sobel Edges")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(prewitt, cmap="gray")
    axes[1, 2].set_title("Prewitt Edges")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(scharr, cmap="gray")
    axes[1, 3].set_title("Scharr Edges")
    axes[1, 3].axis("off")

    plt.tight_layout()
    plt.savefig(
        "../outputs/edgeDetection/edge_detection_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


def display_directional_edges(
    original, sobel_x, sobel_y, prewitt_x, prewitt_y, scharr_x, scharr_y
):
    """Display directional edge detection results."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(
        "Directional Edge Detection Comparison", fontsize=16, fontweight="bold"
    )

    # Original
    axes[0, 0].imshow(original, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Sobel directional
    axes[0, 1].imshow(np.abs(sobel_x), cmap="gray")
    axes[0, 1].set_title("Sobel X (Vertical Edges)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(np.abs(sobel_y), cmap="gray")
    axes[0, 2].set_title("Sobel Y (Horizontal Edges)")
    axes[0, 2].axis("off")

    # Prewitt directional
    axes[1, 0].imshow(original, cmap="gray")
    axes[1, 0].set_title("Original Image")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(np.abs(prewitt_x), cmap="gray")
    axes[1, 1].set_title("Prewitt X (Vertical Edges)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(np.abs(prewitt_y), cmap="gray")
    axes[1, 2].set_title("Prewitt Y (Horizontal Edges)")
    axes[1, 2].axis("off")

    # Scharr directional
    axes[2, 0].imshow(original, cmap="gray")
    axes[2, 0].set_title("Original Image")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(np.abs(scharr_x), cmap="gray")
    axes[2, 1].set_title("Scharr X (Vertical Edges)")
    axes[2, 1].axis("off")

    axes[2, 2].imshow(np.abs(scharr_y), cmap="gray")
    axes[2, 2].set_title("Scharr Y (Horizontal Edges)")
    axes[2, 2].axis("off")

    plt.tight_layout()
    plt.savefig(
        "../outputs/edgeDetection/directional_edges_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


def compare_filters_analysis(sobel, prewitt, scharr):
    """Print analysis comparing the three filters."""
    print("\n" + "=" * 70)
    print("EDGE DETECTION FILTER COMPARISON ANALYSIS")
    print("=" * 70)

    print("\n1. SOBEL FILTER:")
    print("   - Uses 3x3 kernels with weights that approximate derivatives")
    print("   - Good balance between noise suppression and edge detection")
    print("   - Most commonly used filter")
    print(f"   - Detected edge intensity range: {sobel.min()} - {sobel.max()}")
    print(f"   - Mean edge intensity: {sobel.mean():.2f}")

    print("\n2. PREWITT FILTER:")
    print("   - Similar to Sobel but with simpler kernels (uniform weights)")
    print("   - Slightly more sensitive to noise than Sobel")
    print("   - Faster computation due to simpler convolution")
    print(f"   - Detected edge intensity range: {prewitt.min()} - {prewitt.max()}")
    print(f"   - Mean edge intensity: {prewitt.mean():.2f}")

    print("\n3. SCHARR FILTER:")
    print("   - Uses optimized 3x3 kernels for better rotational symmetry")
    print("   - More accurate for edge orientation detection")
    print("   - Better at detecting small-scale edges")
    print(f"   - Detected edge intensity range: {scharr.min()} - {scharr.max()}")
    print(f"   - Mean edge intensity: {scharr.mean():.2f}")

    print("\n4. KEY DIFFERENCES:")
    print("   - Sensitivity: Scharr > Sobel > Prewitt")
    print("   - Noise resistance: Sobel > Prewitt > Scharr")
    print("   - Computational cost: Prewitt < Sobel ≈ Scharr")
    print("   - Edge localization: Scharr > Sobel > Prewitt")

    print("\n" + "=" * 70 + "\n")


def main():
    """Main function to execute edge detection tasks."""
    # Load image
    image_path = "../assets/image1.jpg"
    print(f"Loading image from: {image_path}")
    original_image = load_and_preprocess_image(image_path)
    print(f"Image loaded successfully. Shape: {original_image.shape}")

    # Apply Sobel filter
    print("\nApplying Sobel filter...")
    sobel_edges, sobel_x, sobel_y = apply_sobel_filter(original_image)
    print("Sobel filter applied successfully.")

    # Apply Prewitt filter
    print("\nApplying Prewitt filter...")
    prewitt_edges, prewitt_x, prewitt_y = apply_prewitt_filter(original_image)
    print("Prewitt filter applied successfully.")

    # Apply Scharr filter
    print("\nApplying Scharr filter...")
    scharr_edges, scharr_x, scharr_y = apply_scharr_filter(original_image)
    print("Scharr filter applied successfully.")

    # Display results
    print("\nDisplaying edge detection results...")
    display_edge_detection_results(
        original_image, sobel_edges, prewitt_edges, scharr_edges
    )

    print("\nDisplaying directional edge detection results...")
    display_directional_edges(
        original_image, sobel_x, sobel_y, prewitt_x, prewitt_y, scharr_x, scharr_y
    )

    # Print comparison analysis
    compare_filters_analysis(sobel_edges, prewitt_edges, scharr_edges)

    print("Task 1 completed successfully!")
    print("Output images saved:")
    print("  - edge_detection_comparison.png")
    print("  - directional_edges_comparison.png")


if __name__ == "__main__":
    main()
