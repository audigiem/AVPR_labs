import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def load_and_preprocess_image(image_path=None, target_size=(512, 512)):
    """Load and preprocess image with consistent sizing"""
    if image_path is None:
        # Create a synthetic test image with various features
        image = np.zeros(target_size, dtype=np.uint8)
        # Add rectangles with different intensities
        cv2.rectangle(image, (50, 50), (200, 200), 180, -1)
        cv2.rectangle(image, (300, 100), (450, 250), 120, -1)
        # Add circles
        cv2.circle(image, (150, 350), 60, 220, -1)
        cv2.circle(image, (400, 400), 40, 80, -1)
        # Add some lines
        cv2.line(image, (0, 300), (512, 300), 160, 5)
        cv2.line(image, (256, 0), (256, 512), 200, 3)
        # Add text for texture
        cv2.putText(image, "TEST", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
    else:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Could not find image at {image_path}")

        # Load image (handle both grayscale and color)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize to target size
        print(f"Original image size: {image.shape}")
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        print(f"Resized to: {image.shape}")

    return image


def create_custom_kernels():
    """Create various custom kernels for different effects"""

    kernels = {}

    # 1. Sharpening kernel
    kernels["sharpening"] = np.array(
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32
    )

    # 2. Edge enhancement kernel
    kernels["edge_enhancement"] = np.array(
        [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32
    )

    # 3. Emboss kernel
    kernels["emboss"] = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)

    # 4. Custom blur kernel (Gaussian-like)
    kernels["custom_blur"] = (
        np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16
    )

    # 5. Additional kernels for experimentation

    # Strong sharpening
    kernels["strong_sharpening"] = np.array(
        [[0, -1, 0], [-1, 6, -1], [0, -1, 0]], dtype=np.float32
    )

    # Horizontal edge detection
    kernels["horizontal_edges"] = np.array(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32
    )

    # Vertical edge detection
    kernels["vertical_edges"] = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32
    )

    # Motion blur (horizontal)
    kernels["motion_blur"] = (
        np.array(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        / 5
    )

    return kernels


def apply_kernel(image, kernel):
    """Apply a kernel to an image using cv2.filter2D"""
    filtered = cv2.filter2D(image, cv2.CV_64F, kernel)

    # Handle different kernel effects
    if "emboss" in str(kernel).lower() or "edge" in str(kernel).lower():
        # For emboss and edge effects, add 128 to center around gray
        filtered = filtered + 128

    # Clip values to valid range
    filtered = np.clip(filtered, 0, 255)
    return filtered.astype(np.uint8)


def display_kernel_effects(image, kernels, figure_size=(20, 15)):
    """Display original image and all kernel effects"""

    # Calculate grid size
    num_kernels = len(kernels)
    num_cols = 4
    num_rows = (num_kernels + num_cols) // num_cols  # +1 for original image

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figure_size)
    axes = axes.flatten() if num_rows > 1 else [axes] if num_cols == 1 else axes

    # Display original image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Apply and display each kernel effect
    idx = 1
    for kernel_name, kernel in kernels.items():
        if idx >= len(axes):
            break

        filtered_image = apply_kernel(image, kernel)

        axes[idx].imshow(filtered_image, cmap="gray")
        axes[idx].set_title(
            f'{kernel_name.replace("_", " ").title()}', fontsize=12, fontweight="bold"
        )
        axes[idx].axis("off")

        # Add kernel visualization as text
        kernel_str = np.array2string(kernel, precision=2, separator=", ")
        axes[idx].text(
            0.02,
            0.98,
            f"Kernel:\n{kernel_str}",
            transform=axes[idx].transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        idx += 1

    # Hide unused subplots
    for i in range(idx, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

    return fig


def compare_kernel_variations(image, base_kernel, variations, kernel_name):
    """Compare variations of a single kernel type"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Original image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Base kernel
    filtered = apply_kernel(image, base_kernel)
    axes[1].imshow(filtered, cmap="gray")
    axes[1].set_title(f"Base {kernel_name}")
    axes[1].axis("off")

    # Variations
    for i, (var_name, var_kernel) in enumerate(variations.items()):
        if i + 2 >= len(axes):
            break
        filtered = apply_kernel(image, var_kernel)
        axes[i + 2].imshow(filtered, cmap="gray")
        axes[i + 2].set_title(f"{var_name} {kernel_name}")
        axes[i + 2].axis("off")

    # Hide unused subplots
    for i in range(len(variations) + 2, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def experiment_with_kernels(image):
    """Experiment with different kernel modifications"""

    print("Experimenting with kernel variations...")

    # Experiment 1: Different sharpening intensities
    base_sharpening = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

    sharpening_variations = {
        "Mild": np.array(
            [[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]], dtype=np.float32
        ),
        "Strong": np.array([[0, -1, 0], [-1, 6, -1], [0, -1, 0]], dtype=np.float32),
        "Extreme": np.array([[0, -2, 0], [-2, 9, -2], [0, -2, 0]], dtype=np.float32),
    }

    compare_kernel_variations(
        image, base_sharpening, sharpening_variations, "Sharpening"
    )

    # Experiment 2: Different blur kernels
    base_blur = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32) / 9

    blur_variations = {
        "Gaussian": np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16,
        "Box 5x5": np.ones((5, 5), dtype=np.float32) / 25,
        "Weighted": np.array(
            [
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1],
            ],
            dtype=np.float32,
        )
        / 256,
    }

    compare_kernel_variations(image, base_blur, blur_variations, "Blur")


def analyze_kernel_effects(image, kernels):
    """Analyze and document the effects of different kernels"""

    print("\n" + "=" * 80)
    print("KERNEL EFFECTS ANALYSIS")
    print("=" * 80)

    original_stats = {
        "mean": np.mean(image),
        "std": np.std(image),
        "min": np.min(image),
        "max": np.max(image),
    }

    print(f"Original Image Statistics:")
    print(f"  Mean: {original_stats['mean']:.2f}")
    print(f"  Std:  {original_stats['std']:.2f}")
    print(f"  Range: [{original_stats['min']}, {original_stats['max']}]")
    print("-" * 40)

    for kernel_name, kernel in kernels.items():
        filtered = apply_kernel(image, kernel)

        stats = {
            "mean": np.mean(filtered),
            "std": np.std(filtered),
            "min": np.min(filtered),
            "max": np.max(filtered),
        }

        print(f"{kernel_name.replace('_', ' ').title()}:")
        print(f"  Kernel sum: {np.sum(kernel):.3f}")
        print(
            f"  Mean: {stats['mean']:.2f} (Δ: {stats['mean'] - original_stats['mean']:+.2f})"
        )
        print(
            f"  Std:  {stats['std']:.2f} (Δ: {stats['std'] - original_stats['std']:+.2f})"
        )
        print(f"  Range: [{stats['min']}, {stats['max']}]")
        print("-" * 40)


def main():
    """Main function to demonstrate custom kernel effects"""

    # Configuration
    TARGET_SIZE = (512, 512)
    FIGURE_SIZE = (20, 15)

    # List of test images
    IMAGES_PATH = [
        "assets/100.pgm",
        "assets/couchersoleil.pgm",
        "assets/einstein.pgm",
        "assets/monroe.pgm",
        "assets/mandrill256.pgm",
    ]

    # Create custom kernels
    kernels = create_custom_kernels()

    # Process each image or create synthetic if no images available
    test_images = []
    for image_path in IMAGES_PATH:
        if os.path.exists(image_path):
            test_images.append(image_path)

    if not test_images:
        print("No test images found, using synthetic image...")
        test_images = [None]  # Will create synthetic image

    for image_path in test_images:
        print(f"\n{'='*80}")
        if image_path:
            print(f"Processing: {image_path}")
        else:
            print("Processing: Synthetic Test Image")
        print(f"{'='*80}")

        try:
            # Load and preprocess image
            image = load_and_preprocess_image(image_path, TARGET_SIZE)

            # Display all kernel effects
            print("Displaying all kernel effects...")
            fig = display_kernel_effects(image, kernels, FIGURE_SIZE)

            # Save the comparison
            if image_path:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                fig.savefig(
                    f"custom_kernels_{base_name}.png", dpi=150, bbox_inches="tight"
                )
            else:
                fig.savefig(
                    "custom_kernels_synthetic.png", dpi=150, bbox_inches="tight"
                )

            # Analyze kernel effects
            analyze_kernel_effects(image, kernels)

            # Experiment with kernel variations
            experiment_with_kernels(image)

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue


if __name__ == "__main__":
    main()
