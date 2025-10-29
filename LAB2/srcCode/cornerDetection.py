"""
Task 2: Corner Detection
This script performs corner detection using the Harris Corner Detection method
and experiments with different parameters.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_and_preprocess_image(image_path=None, target_size=(512, 512)):
    """Load and preprocess image with consistent sizing"""
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        print(f"Current directory: {os.getcwd()}")
        print(
            f"Files in assets/: {os.listdir('assets') if os.path.exists('assets') else 'Assets folder not found'}"
        )
        raise FileNotFoundError(f"Could not find image at {image_path}")

    # Load image from file
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Resize to target size
    print(f"Original image size: {image.shape}")
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    color_image = cv2.resize(color_image, target_size, interpolation=cv2.INTER_AREA)
    print(f"Resized to: {image.shape}")

    return image, color_image

def harris_corner_detection(image, block_size=2, ksize=3, k=0.04, threshold=0.01):
    """
    Apply Harris Corner Detection with specified parameters.

    Parameters:
    - image: Input grayscale image
    - block_size: Size of neighborhood considered for corner detection
    - ksize: Aperture parameter for Sobel derivative
    - k: Harris detector free parameter
    - threshold: Threshold for corner detection (relative to max response)

    Returns:
    - corners: Binary mask of detected corners
    - harris_response: Harris corner response map
    - corner_count: Number of detected corners
    """
    # Convert to float32
    gray_float = np.float32(image)

    # Apply Harris Corner Detection
    harris_response = cv2.cornerHarris(gray_float, block_size, ksize, k)

    # Dilate to mark corners
    harris_response = cv2.dilate(harris_response, None)

    # Threshold for corner detection
    threshold_value = threshold * harris_response.max()
    corners = harris_response > threshold_value

    # Count corners
    corner_count = np.sum(corners)

    return corners, harris_response, corner_count


def mark_corners_on_image(image, corners, color=(0, 255, 0), marker_size=3):
    """Mark detected corners on the image."""
    result = image.copy()
    result[corners] = color

    # Optionally make markers larger for visibility
    if marker_size > 1:
        kernel = np.ones((marker_size, marker_size), np.uint8)
        corners_dilated = cv2.dilate(corners.astype(np.uint8), kernel)
        result[corners_dilated > 0] = color

    return result


def visualize_harris_response(
    gray_image,
    corners,
    harris_response,
    marked_image,
    block_size,
    ksize,
    k,
    corner_count,
):
    """Visualize Harris corner detection results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"Harris Corner Detection\n"
        f"(block_size={block_size}, ksize={ksize}, k={k:.3f}, "
        f"corners={corner_count})",
        fontsize=14,
        fontweight="bold",
    )

    # Original image
    axes[0, 0].imshow(gray_image, cmap="gray")
    axes[0, 0].set_title("Original Grayscale Image")
    axes[0, 0].axis("off")

    # Harris response map
    im1 = axes[0, 1].imshow(harris_response, cmap="jet")
    axes[0, 1].set_title("Harris Corner Response Map")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1])

    # Corner locations
    axes[1, 0].imshow(corners, cmap="gray")
    axes[1, 0].set_title(f"Detected Corners (Total: {corner_count})")
    axes[1, 0].axis("off")

    # Marked image
    axes[1, 1].imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Corners Marked on Original Image")
    axes[1, 1].axis("off")

    plt.tight_layout()
    return fig


def experiment_with_parameters(gray_image, color_image):
    """Experiment with different Harris Corner Detection parameters."""
    print("\n" + "=" * 70)
    print("EXPERIMENTING WITH HARRIS CORNER DETECTION PARAMETERS")
    print("=" * 70)

    # Parameter variations
    experiments = [
        # (block_size, ksize, k, threshold, description)
        (2, 3, 0.04, 0.01, "Default parameters"),
        (3, 3, 0.04, 0.01, "Larger block size"),
        (5, 3, 0.04, 0.01, "Even larger block size"),
        (2, 5, 0.04, 0.01, "Larger aperture size"),
        (2, 7, 0.04, 0.01, "Even larger aperture"),
        (2, 3, 0.02, 0.01, "Smaller k (more corners)"),
        (2, 3, 0.06, 0.01, "Larger k (fewer corners)"),
        (2, 3, 0.04, 0.005, "Lower threshold (more corners)"),
        (2, 3, 0.04, 0.02, "Higher threshold (fewer corners)"),
    ]

    results = []

    for i, (block_size, ksize, k, threshold, description) in enumerate(experiments):
        print(f"\nExperiment {i + 1}: {description}")
        print(
            f"  Parameters: block_size={block_size}, ksize={ksize}, k={k:.3f}, threshold={threshold}"
        )

        corners, harris_response, corner_count = harris_corner_detection(
            gray_image, block_size, ksize, k, threshold
        )

        marked_image = mark_corners_on_image(color_image, corners, marker_size=3)

        print(f"  Detected corners: {corner_count}")
        print(
            f"  Harris response range: [{harris_response.min():.2e}, {harris_response.max():.2e}]"
        )

        results.append(
            {
                "corners": corners,
                "harris_response": harris_response,
                "marked_image": marked_image,
                "corner_count": corner_count,
                "params": (block_size, ksize, k, threshold),
                "description": description,
            }
        )

    return results


def display_parameter_comparison(gray_image, results):
    """Display comparison of different parameter settings."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.suptitle(
        "Harris Corner Detection: Parameter Comparison", fontsize=16, fontweight="bold"
    )

    for i, result in enumerate(results):
        row = i // 3
        col = i % 3

        block_size, ksize, k, threshold = result["params"]
        marked_image = result["marked_image"]
        corner_count = result["corner_count"]
        description = result["description"]

        axes[row, col].imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(
            f"{description}\n"
            f"BS={block_size}, KS={ksize}, k={k:.2f}, T={threshold}\n"
            f"Corners: {corner_count}",
            fontsize=10,
        )
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(
        "../outputs/cornerDetection/harris_parameter_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


def analyze_parameter_effects(results):
    """Analyze and print the effects of different parameters."""
    print("\n" + "=" * 70)
    print("PARAMETER EFFECTS ANALYSIS")
    print("=" * 70)

    print("\n1. BLOCK SIZE (neighborhood size for corner detection):")
    print("   - Smaller values (2): Detects finer corners, more sensitive")
    print("   - Larger values (5): Detects more prominent corners, less noise")
    block_results = [(r["params"][0], r["corner_count"]) for r in results[:3]]
    for bs, count in block_results:
        print(f"   - Block size {bs}: {count} corners detected")

    print("\n2. APERTURE SIZE (ksize - Sobel kernel size):")
    print("   - Smaller values (3): Faster, less smoothing")
    print("   - Larger values (7): More smoothing, better noise handling")
    ksize_results = [
        (r["params"][1], r["corner_count"])
        for r in [results[0], results[3], results[4]]
    ]
    for ks, count in ksize_results:
        print(f"   - Aperture size {ks}: {count} corners detected")

    print("\n3. K PARAMETER (Harris detector constant):")
    print("   - Smaller k (0.02): More sensitive, detects more corners")
    print("   - Larger k (0.06): Less sensitive, detects stronger corners only")
    print("   - Typical range: 0.04 to 0.06")
    k_results = [
        (r["params"][2], r["corner_count"])
        for r in [results[5], results[0], results[6]]
    ]
    for k_val, count in k_results:
        print(f"   - k = {k_val:.2f}: {count} corners detected")

    print("\n4. THRESHOLD (corner response threshold):")
    print("   - Lower threshold (0.005): More corners, including weaker ones")
    print("   - Higher threshold (0.02): Fewer, stronger corners only")
    thresh_results = [
        (r["params"][3], r["corner_count"])
        for r in [results[7], results[0], results[8]]
    ]
    for t, count in thresh_results:
        print(f"   - Threshold {t}: {count} corners detected")

    print("\n5. GENERAL RECOMMENDATIONS:")
    print("   - For noisy images: Increase block_size and ksize")
    print("   - For fine details: Use smaller block_size")
    print("   - For fewer false positives: Increase threshold and k")
    print("   - Standard starting point: block_size=2, ksize=3, k=0.04, threshold=0.01")
    print("\n" + "=" * 70 + "\n")


def main():
    """Main function to execute corner detection tasks."""
    # Load image
    image_path = "../assets/left_image.png"
    print(f"Loading image from: {image_path}")
    gray_image, color_image = load_and_preprocess_image(image_path)
    print(f"Image loaded successfully. Shape: {gray_image.shape}")

    # Apply Harris Corner Detection with default parameters
    print("\nApplying Harris Corner Detection with default parameters...")
    block_size = 2
    ksize = 3
    k = 0.04
    threshold = 0.01

    corners, harris_response, corner_count = harris_corner_detection(
        gray_image, block_size, ksize, k, threshold
    )

    print(f"Detected {corner_count} corners")

    # Mark corners on image
    marked_image = mark_corners_on_image(color_image, corners, marker_size=3)

    # Visualize results
    print("\nVisualizing Harris corner detection results...")
    fig = visualize_harris_response(
        gray_image,
        corners,
        harris_response,
        marked_image,
        block_size,
        ksize,
        k,
        corner_count,
    )
    plt.savefig(
        "../outputs/cornerDetection/harris_corner_detection_default.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

    # Experiment with different parameters
    print("\nExperimenting with different parameters...")
    results = experiment_with_parameters(gray_image, color_image)

    # Display parameter comparison
    print("\nDisplaying parameter comparison...")
    display_parameter_comparison(gray_image, results)

    # Analyze parameter effects
    analyze_parameter_effects(results)

    print("\nTask 2 completed successfully!")
    print("Output images saved:")
    print("  - harris_corner_detection_default.png")
    print("  - harris_parameter_comparison.png")


if __name__ == "__main__":
    main()
