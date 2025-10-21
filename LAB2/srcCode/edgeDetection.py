import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from networkx.algorithms.swap import directed_edge_swap


def apply_sobel_edge_detection(image):
    """Apply Sobel edge detection operator"""
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    # Apply convolution
    grad_x = cv2.filter2D(image, cv2.CV_64F, sobel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, sobel_y)

    # Compute magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))

    return magnitude, grad_x, grad_y


def apply_prewitt_edge_detection(image):
    """Apply Prewitt edge detection operator"""
    # Prewitt kernels
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    # Apply convolution
    grad_x = cv2.filter2D(image, cv2.CV_64F, prewitt_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, prewitt_y)

    # Compute magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))

    return magnitude, grad_x, grad_y


def apply_scharr_edge_detection(image):
    """Apply Scharr edge detection operator"""
    # Scharr operator
    scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32)
    scharr_y = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=np.float32)

    # Apply convolution
    scharr_x = cv2.filter2D(image, cv2.CV_64F, scharr_x)
    scharr_y = cv2.filter2D(image, cv2.CV_64F, scharr_y)

    # Compute magnitude
    magnitude = np.sqrt(scharr_x**2 + scharr_y**2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))

    return magnitude, scharr_x, scharr_y


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


def compare_edge_detectors(
    image_path=None, target_size=(512, 512), figure_size=(16, 8)
):
    """Main function to compare edge detection operators with consistent sizing"""

    # Load image with consistent size
    original_image = load_and_preprocess_image(image_path, target_size)

    # Apply different edge detection operators
    sobel_mag, sobel_x, sobel_y = apply_sobel_edge_detection(original_image)
    prewitt_mag, prewitt_x, prewitt_y = apply_prewitt_edge_detection(original_image)
    scharr_mag, scharr_x, scharr_y = apply_scharr_edge_detection(original_image)

    # Create visualization with fixed figure size
    fig, axes = plt.subplots(2, 4, figsize=figure_size)

    # Set consistent aspect ratio for all subplots
    for ax in axes[0]:
        ax.set_aspect("equal")

    # First row: Original and magnitude results
    axes[0, 0].imshow(original_image, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(sobel_mag, cmap="gray")
    axes[0, 1].set_title("Sobel Edge Detection")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(prewitt_mag, cmap="gray")
    axes[0, 2].set_title("Prewitt Edge Detection")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(scharr_mag, cmap="gray")
    axes[0, 3].set_title("Scharr Edge Detection")
    axes[0, 3].axis("off")

    # Second row: Comparison metrics
    # Calculate edge strength statistics
    sobel_mean = np.mean(sobel_mag)
    prewitt_mean = np.mean(prewitt_mag)
    scharr_mean = np.mean(scharr_mag)

    sobel_std = np.std(sobel_mag)
    prewitt_std = np.std(prewitt_mag)
    scharr_std = np.std(scharr_mag)

    # Plot histograms
    axes[1, 0].hist(original_image.flatten(), bins=50, alpha=0.7, color="gray")
    axes[1, 0].set_title("Original Histogram")
    axes[1, 0].set_xlabel("Pixel Intensity")
    axes[1, 0].set_ylabel("Frequency")

    axes[1, 1].hist(sobel_mag.flatten(), bins=50, alpha=0.7, color="blue")
    axes[1, 1].set_title(f"Sobel\nMean: {sobel_mean:.1f}, Std: {sobel_std:.1f}")
    axes[1, 1].set_xlabel("Edge Magnitude")
    axes[1, 1].set_ylabel("Frequency")

    axes[1, 2].hist(prewitt_mag.flatten(), bins=50, alpha=0.7, color="green")
    axes[1, 2].set_title(f"Prewitt\nMean: {prewitt_mean:.1f}, Std: {prewitt_std:.1f}")
    axes[1, 2].set_xlabel("Edge Magnitude")
    axes[1, 2].set_ylabel("Frequency")

    axes[1, 3].hist(scharr_mag.flatten(), bins=50, alpha=0.7, color="red")
    axes[1, 3].set_title(f"Scharr\nMean: {scharr_mean:.1f}, Std: {scharr_std:.1f}")
    axes[1, 3].set_xlabel("Edge Magnitude")
    axes[1, 3].set_ylabel("Frequency")

    plt.tight_layout()
    # plt.show()

    # Save with consistent naming
    if image_path is not None:
        base_name = os.path.basename(image_path)
        name, _ = os.path.splitext(base_name)
        # move to outputs/edgeDetection/
        dir_path = "../outputs/edgeDetection/"
        fig.savefig(
            f"{dir_path}edge_detection_comparison_{name}.png",
            dpi=150,
            bbox_inches="tight",
        )
    else:
        fig.savefig("edge_detection_comparison.png", dpi=150, bbox_inches="tight")

    # Print comparison analysis
    print("Edge Detection Comparison Analysis:")
    print("=" * 50)
    print(f"Image size: {original_image.shape}")
    print(
        f"Sobel operator  - Mean edge strength: {sobel_mean:.2f}, Std: {sobel_std:.2f}"
    )
    print(
        f"Prewitt operator - Mean edge strength: {prewitt_mean:.2f}, Std: {prewitt_std:.2f}"
    )
    print(
        f"Scharr operator  - Mean edge strength: {scharr_mean:.2f}, Std: {scharr_std:.2f}"
    )

    return {
        "original": original_image,
        "sobel": sobel_mag,
        "prewitt": prewitt_mag,
        "scharr": scharr_mag,
        "statistics": {
            "sobel": {"mean": sobel_mean, "std": sobel_std},
            "prewitt": {"mean": prewitt_mean, "std": prewitt_std},
            "scharr": {"mean": scharr_mean, "std": scharr_std},
        },
    }


def detailed_comparison(image_path=None, target_size=(512, 512), figure_size=(16, 12)):
    """Detailed comparison showing horizontal and vertical components with consistent sizing"""
    original_image = load_and_preprocess_image(image_path, target_size)

    # Apply edge detection
    sobel_mag, sobel_x, sobel_y = apply_sobel_edge_detection(original_image)
    prewitt_mag, prewitt_x, prewitt_y = apply_prewitt_edge_detection(original_image)
    scharr_mag, scharr_x, scharr_y = apply_scharr_edge_detection(original_image)

    # Create detailed visualization with fixed figure size
    fig, axes = plt.subplots(3, 4, figsize=figure_size)

    # Set consistent aspect ratio for all subplots
    for ax in axes.flat:
        ax.set_aspect("equal")

    # Sobel row
    axes[0, 0].imshow(original_image, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(np.abs(sobel_x), cmap="gray")
    axes[0, 1].set_title("Sobel X (Vertical Edges)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(np.abs(sobel_y), cmap="gray")
    axes[0, 2].set_title("Sobel Y (Horizontal Edges)")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(sobel_mag, cmap="gray")
    axes[0, 3].set_title("Sobel Magnitude")
    axes[0, 3].axis("off")

    # Prewitt row
    axes[1, 0].imshow(original_image, cmap="gray")
    axes[1, 0].set_title("Original Image")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(np.abs(prewitt_x), cmap="gray")
    axes[1, 1].set_title("Prewitt X (Vertical Edges)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(np.abs(prewitt_y), cmap="gray")
    axes[1, 2].set_title("Prewitt Y (Horizontal Edges)")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(prewitt_mag, cmap="gray")
    axes[1, 3].set_title("Prewitt Magnitude")
    axes[1, 3].axis("off")

    # Scharr row
    axes[2, 0].imshow(original_image, cmap="gray")
    axes[2, 0].set_title("Original Image")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(np.abs(scharr_x), cmap="gray")
    axes[2, 1].set_title("Scharr X (Vertical Edges)")
    axes[2, 1].axis("off")

    axes[2, 2].imshow(np.abs(scharr_y), cmap="gray")
    axes[2, 2].set_title("Scharr Y (Horizontal Edges)")
    axes[2, 2].axis("off")

    axes[2, 3].imshow(scharr_mag, cmap="gray")
    axes[2, 3].set_title("Scharr Magnitude")
    axes[2, 3].axis("off")

    plt.tight_layout()
    # plt.show()

    # Save with consistent naming
    if image_path is not None:
        base_name = os.path.basename(image_path)
        name, _ = os.path.splitext(base_name)
        # move to outputs/edgeDetection/
        dir_path = "../outputs/edgeDetection/"

        fig.savefig(
            f"{dir_path}detailed_edge_detection_comparison_{name}.png",
            dpi=150,
            bbox_inches="tight",
        )
    else:
        fig.savefig(
            "detailed_edge_detection_comparison.png", dpi=150, bbox_inches="tight"
        )


if __name__ == "__main__":
    # Configuration
    TARGET_SIZE = (512, 512)
    FIGURE_SIZE_BASIC = (16, 8)
    FIGURE_SIZE_DETAILED = (16, 12)

    # Run basic comparison
    IMAGES_PATH = [
        # "../assets/100.pgm",
        "../assets/couchersoleil.pgm",
        "../assets/einstein.pgm",
        "../assets/monroe.pgm",
        "../assets/image1.jpg",
        "../assets/image2.jpg",
        "../assets/left_image.png",
    ]

    for image_path in IMAGES_PATH:
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")
        print(f"{'='*60}")

        try:
            print(f"Running basic edge detection comparison...")
            results = compare_edge_detectors(image_path, TARGET_SIZE, FIGURE_SIZE_BASIC)

            print(f"\nRunning detailed comparison with directional components...")
            detailed_comparison(image_path, TARGET_SIZE, FIGURE_SIZE_DETAILED)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
