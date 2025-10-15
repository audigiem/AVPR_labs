import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle


def load_and_preprocess_image(image_path="assets/peppers.png", target_size=None):
    """Load and preprocess image"""
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        # Create a synthetic color image with distinct patterns
        if target_size is None:
            target_size = (512, 512)

        # Create a colorful synthetic image
        image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

        # Add colorful rectangles
        cv2.rectangle(image, (50, 50), (200, 200), (255, 100, 100), -1)  # Red
        cv2.rectangle(image, (300, 100), (450, 250), (100, 255, 100), -1)  # Green
        cv2.rectangle(image, (150, 300), (350, 450), (100, 100, 255), -1)  # Blue

        # Add circles
        cv2.circle(image, (150, 150), 60, (255, 255, 100), -1)  # Yellow
        cv2.circle(image, (400, 400), 50, (255, 100, 255), -1)  # Magenta

        # Add some lines and patterns
        for i in range(0, target_size[0], 20):
            cv2.line(image, (i, 0), (i, 100), (200, 200, 200), 2)

        # Add text
        cv2.putText(
            image,
            "MOTION",
            (200, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3,
        )
        cv2.putText(
            image, "BLUR", (250, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3
        )

        print("Using synthetic color image")
    else:
        # Load image from file
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Convert from BGR to RGB for matplotlib
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if target_size is not None:
            print(f"Original image size: {image.shape}")
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            print(f"Resized to: {image.shape}")

        print(f"Loaded image from {image_path}")

    return image


def create_motion_blur_kernel(size, angle, intensity=1.0):
    """Create a motion blur kernel at specified angle

    Args:
        size: Kernel size (should be odd)
        angle: Angle in degrees (0° = horizontal, 90° = vertical)
        intensity: Blur intensity (1.0 = standard)

    Returns:
        Motion blur kernel
    """
    # Ensure odd kernel size
    if size % 2 == 0:
        size += 1

    # Create empty kernel
    kernel = np.zeros((size, size), dtype=np.float32)

    # Calculate center
    center = size // 2

    # Convert angle to radians
    angle_rad = np.radians(angle)

    # Calculate direction vector
    dx = np.cos(angle_rad)
    dy = np.sin(angle_rad)

    # Create motion blur line
    length = size // 2
    for i in range(-length, length + 1):
        x = int(center + i * dx * intensity)
        y = int(center + i * dy * intensity)

        # Check bounds
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0

    # Normalize kernel
    if np.sum(kernel) > 0:
        kernel = kernel / np.sum(kernel)

    return kernel


def create_diagonal_motion_kernel(size, horizontal_weight=0.5, vertical_weight=0.5):
    """Create diagonal motion blur by combining horizontal and vertical kernels

    Args:
        size: Kernel size
        horizontal_weight: Weight for horizontal component
        vertical_weight: Weight for vertical component

    Returns:
        Combined diagonal motion kernel
    """
    # Create horizontal and vertical kernels
    h_kernel = create_motion_blur_kernel(size, 0)  # 0° = horizontal
    v_kernel = create_motion_blur_kernel(size, 90)  # 90° = vertical

    # Combine with weights
    combined_kernel = horizontal_weight * h_kernel + vertical_weight * v_kernel

    # Normalize
    if np.sum(combined_kernel) > 0:
        combined_kernel = combined_kernel / np.sum(combined_kernel)

    return combined_kernel


def apply_motion_blur(image, kernel):
    """Apply motion blur to image using cv2.filter2D"""
    # Handle color images by processing each channel
    if len(image.shape) == 3:
        blurred = np.zeros_like(image)
        for channel in range(image.shape[2]):
            blurred[:, :, channel] = cv2.filter2D(image[:, :, channel], -1, kernel)
        return blurred
    else:
        return cv2.filter2D(image, -1, kernel)


def visualize_kernel(kernel, title="Kernel", figsize=(4, 4)):
    """Visualize a kernel as an image"""
    plt.figure(figsize=figsize)
    plt.imshow(kernel, cmap="hot", interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.axis("off")
    return plt.gcf()


def display_motion_blur_comparison(
    image, angles=[0, 45, 90, 135], kernel_size=21, figure_size=(20, 12)
):
    """Display comprehensive motion blur comparison"""

    # Create motion blur kernels and apply them
    results = {}
    kernels = {}

    for angle in angles:
        kernel = create_motion_blur_kernel(kernel_size, angle)
        blurred = apply_motion_blur(image, kernel)
        results[angle] = blurred
        kernels[angle] = kernel

    # Create visualization
    fig = plt.figure(figsize=figure_size)

    # Create a grid layout: 2 rows, len(angles)+1 columns
    gs = fig.add_gridspec(
        3, len(angles) + 1, height_ratios=[3, 3, 1], hspace=0.3, wspace=0.2
    )

    # Original image (top-left)
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(image)
    ax_orig.set_title("Original Image", fontsize=14, fontweight="bold")
    ax_orig.axis("off")

    # Motion blurred images (top row)
    for i, angle in enumerate(angles):
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(results[angle])
        ax.set_title(
            f"Motion Blur {angle}°\nKernel Size: {kernel_size}x{kernel_size}",
            fontsize=12,
        )
        ax.axis("off")

    # Kernels visualization (bottom row)
    ax_kernel_label = fig.add_subplot(gs[1, 0])
    ax_kernel_label.text(
        0.5,
        0.5,
        "Motion Blur\nKernels",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        transform=ax_kernel_label.transAxes,
    )
    ax_kernel_label.axis("off")

    for i, angle in enumerate(angles):
        ax = fig.add_subplot(gs[1, i + 1])
        im = ax.imshow(kernels[angle], cmap="hot", interpolation="nearest")
        ax.set_title(f"{angle}° Kernel", fontsize=11)
        ax.axis("off")
        # Add small colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    # Add kernel size comparison in bottom row
    ax_info = fig.add_subplot(gs[2, :])
    ax_info.text(
        0.5,
        0.5,
        f"All kernels are {kernel_size}x{kernel_size} pixels",
        ha="center",
        va="center",
        fontsize=12,
        transform=ax_info.transAxes,
        style="italic",
    )
    ax_info.axis("off")

    plt.suptitle(
        "Motion Blur Analysis at Different Angles", fontsize=16, fontweight="bold"
    )

    return fig, results, kernels


def compare_kernel_sizes(image, angle=45, sizes=[15, 21, 31, 51], figure_size=(20, 10)):
    """Compare motion blur with different kernel sizes"""

    fig, axes = plt.subplots(2, len(sizes) + 1, figsize=figure_size)

    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[1, 0].text(
        0.5,
        0.5,
        "Kernel Size\nComparison",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        transform=axes[1, 0].transAxes,
    )
    axes[1, 0].axis("off")

    for i, size in enumerate(sizes):
        # Create kernel and apply blur
        kernel = create_motion_blur_kernel(size, angle)
        blurred = apply_motion_blur(image, kernel)

        # Display blurred image
        axes[0, i + 1].imshow(blurred)
        axes[0, i + 1].set_title(
            f"Kernel Size: {size}x{size}\nAngle: {angle}°", fontsize=11
        )
        axes[0, i + 1].axis("off")

        # Display kernel
        im = axes[1, i + 1].imshow(kernel, cmap="hot", interpolation="nearest")
        axes[1, i + 1].set_title(f"{size}x{size} Kernel", fontsize=10)
        axes[1, i + 1].axis("off")

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1, i + 1], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    plt.suptitle(
        f"Motion Blur Intensity Comparison ({angle}° angle)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    return fig


def demonstrate_diagonal_blur(image, figure_size=(16, 12)):
    """Demonstrate diagonal motion blur creation"""

    # Create different diagonal combinations
    combinations = [
        (1.0, 0.0, "Horizontal Only (0°)"),
        (0.0, 1.0, "Vertical Only (90°)"),
        (0.5, 0.5, "Diagonal (Equal Mix)"),
        (0.7, 0.3, "Mostly Horizontal"),
        (0.3, 0.7, "Mostly Vertical"),
    ]

    kernel_size = 31

    fig, axes = plt.subplots(2, 3, figsize=figure_size)
    axes = axes.flatten()

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    for i, (h_weight, v_weight, title) in enumerate(combinations):
        if i + 1 >= len(axes):
            break

        # Create diagonal kernel
        kernel = create_diagonal_motion_kernel(kernel_size, h_weight, v_weight)
        blurred = apply_motion_blur(image, kernel)

        axes[i + 1].imshow(blurred)
        axes[i + 1].set_title(
            f"{title}\nH:{h_weight:.1f}, V:{v_weight:.1f}", fontsize=11
        )
        axes[i + 1].axis("off")

    plt.suptitle("Diagonal Motion Blur Combinations", fontsize=16, fontweight="bold")
    plt.tight_layout()

    return fig


def analyze_motion_blur_effects(image, results, angles):
    """Analyze the effects of different motion blur angles"""

    print("\n" + "=" * 80)
    print("MOTION BLUR ANALYSIS")
    print("=" * 80)

    # Convert to grayscale for analysis
    if len(image.shape) == 3:
        gray_original = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_original = image

    print("Image Statistics Analysis:")
    print("-" * 40)
    print(
        f"Original - Mean: {np.mean(gray_original):.2f}, Std: {np.std(gray_original):.2f}"
    )

    for angle in angles:
        if len(results[angle].shape) == 3:
            gray_blurred = cv2.cvtColor(results[angle], cv2.COLOR_RGB2GRAY)
        else:
            gray_blurred = results[angle]

        mean_val = np.mean(gray_blurred)
        std_val = np.std(gray_blurred)

        print(f"{angle:3d}° Blur - Mean: {mean_val:.2f}, Std: {std_val:.2f}")


def main():
    """Main function to demonstrate motion blur analysis"""

    print("Motion Blur Simulation and Analysis")
    print("=" * 50)

    # Load image
    image_path = "assets/peppers.png"
    target_size = (512, 512)

    try:
        image = load_and_preprocess_image(image_path, target_size)

        # 1. Basic motion blur comparison at different angles
        print("\n1. Comparing motion blur at different angles...")
        angles = [0, 45, 90, 135]
        kernel_size = 21

        fig1, results, kernels = display_motion_blur_comparison(
            image, angles, kernel_size
        )

        # Save the comparison
        base_name = "peppers" if os.path.exists(image_path) else "synthetic"
        fig1.savefig(
            f"motion_blur_angles_{base_name}.png", dpi=150, bbox_inches="tight"
        )

        # 2. Compare different kernel sizes
        print("\n2. Comparing different kernel sizes...")
        fig2 = compare_kernel_sizes(image, angle=45, sizes=[15, 21, 31, 51])
        fig2.savefig(f"motion_blur_sizes_{base_name}.png", dpi=150, bbox_inches="tight")

        # 3. Demonstrate diagonal blur combinations
        print("\n3. Demonstrating diagonal motion blur combinations...")
        fig3 = demonstrate_diagonal_blur(image)
        fig3.savefig(
            f"motion_blur_diagonal_{base_name}.png", dpi=150, bbox_inches="tight"
        )

        # 4. Analyze motion blur effects
        analyze_motion_blur_effects(image, results, angles)

        # 5. Additional experiments
        print(f"\n4. Additional experiments...")

        # Experiment with different intensities
        print("   Creating motion blur with different intensities...")
        fig4, axes = plt.subplots(1, 4, figsize=(16, 4))
        intensities = [0.5, 1.0, 1.5, 2.0]

        for i, intensity in enumerate(intensities):
            kernel = create_motion_blur_kernel(31, 45, intensity)
            blurred = apply_motion_blur(image, kernel)

            axes[i].imshow(blurred)
            axes[i].set_title(f"Intensity: {intensity}", fontsize=12)
            axes[i].axis("off")

        plt.suptitle(
            "Motion Blur Intensity Variations (45° angle)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        fig4.savefig(
            f"motion_blur_intensity_{base_name}.png", dpi=150, bbox_inches="tight"
        )
        plt.show()

        print(f"\nAll visualizations saved with prefix: motion_blur_*_{base_name}.png")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
