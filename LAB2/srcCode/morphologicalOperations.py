"""
Task 3: Morphological Operations for Image Enhancement
This script performs image enhancement using various morphological operations
and experiments with different structuring elements and parameters.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from edgeDetection import load_and_preprocess_image

def create_structuring_elements():
    """Create various structuring elements for morphological operations."""
    elements = {
        "rect_3x3": cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        "rect_5x5": cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        "rect_7x7": cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
        "ellipse_3x3": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        "ellipse_5x5": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        "ellipse_7x7": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        "cross_3x3": cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
        "cross_5x5": cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),
        "cross_7x7": cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7)),
    }
    return elements


def apply_erosion(image, kernel, iterations=1):
    """Apply erosion operation."""
    return cv2.erode(image, kernel, iterations=iterations)


def apply_dilation(image, kernel, iterations=1):
    """Apply dilation operation."""
    return cv2.dilate(image, kernel, iterations=iterations)


def apply_opening(image, kernel, iterations=1):
    """Apply opening operation (erosion followed by dilation)."""
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)


def apply_closing(image, kernel, iterations=1):
    """Apply closing operation (dilation followed by erosion)."""
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def apply_gradient(image, kernel):
    """Apply morphological gradient (dilation - erosion)."""
    return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)


def apply_tophat(image, kernel):
    """Apply top hat transform (original - opening)."""
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)


def apply_blackhat(image, kernel):
    """Apply black hat transform (closing - original)."""
    return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)


def visualize_basic_operations(original, erosion, dilation, opening, closing):
    """Visualize basic morphological operations."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Basic Morphological Operations", fontsize=16, fontweight="bold")

    # Original
    axes[0, 0].imshow(original, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Erosion
    axes[0, 1].imshow(erosion, cmap="gray")
    axes[0, 1].set_title("Erosion\n(Removes small white regions)")
    axes[0, 1].axis("off")

    # Dilation
    axes[0, 2].imshow(dilation, cmap="gray")
    axes[0, 2].set_title("Dilation\n(Expands white regions)")
    axes[0, 2].axis("off")

    # Opening
    axes[1, 0].imshow(opening, cmap="gray")
    axes[1, 0].set_title("Opening\n(Removes small objects)")
    axes[1, 0].axis("off")

    # Closing
    axes[1, 1].imshow(closing, cmap="gray")
    axes[1, 1].set_title("Closing\n(Fills small holes)")
    axes[1, 1].axis("off")

    # Side by side comparison
    axes[1, 2].imshow(np.hstack([original, opening]), cmap="gray")
    axes[1, 2].set_title("Original | Opening")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(
        "../outputs/morphologicalOperations/basic_morphological_operations.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


def visualize_advanced_operations(original, gradient, tophat, blackhat):
    """Visualize advanced morphological operations."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("Advanced Morphological Operations", fontsize=16, fontweight="bold")

    # Original
    axes[0, 0].imshow(original, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Gradient
    axes[0, 1].imshow(gradient, cmap="gray")
    axes[0, 1].set_title("Morphological Gradient\n(Edge Detection)")
    axes[0, 1].axis("off")

    # Top Hat
    axes[1, 0].imshow(tophat, cmap="gray")
    axes[1, 0].set_title("Top Hat\n(Bright features on dark background)")
    axes[1, 0].axis("off")

    # Black Hat
    axes[1, 1].imshow(blackhat, cmap="gray")
    axes[1, 1].set_title("Black Hat\n(Dark features on bright background)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(
        "../outputs/morphologicalOperations/advanced_morphological_operations.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


def experiment_with_structuring_elements(image):
    """Experiment with different structuring elements."""
    print("\n" + "=" * 70)
    print("EXPERIMENTING WITH DIFFERENT STRUCTURING ELEMENTS")
    print("=" * 70)

    elements = create_structuring_elements()

    # Create figure for comparison
    fig, axes = plt.subplots(3, 6, figsize=(20, 12))
    fig.suptitle(
        "Effect of Different Structuring Elements on Erosion and Dilation",
        fontsize=16,
        fontweight="bold",
    )

    shapes = ["rect", "ellipse", "cross"]
    sizes = ["3x3", "5x5", "7x7"]

    for i, shape in enumerate(shapes):
        for j, size in enumerate(sizes):
            key = f"{shape}_{size}"
            kernel = elements[key]

            # Apply erosion
            erosion = apply_erosion(image, kernel)
            axes[i, j * 2].imshow(erosion, cmap="gray")
            axes[i, j * 2].set_title(f"{shape.capitalize()} {size}\nErosion")
            axes[i, j * 2].axis("off")

            # Apply dilation
            dilation = apply_dilation(image, kernel)
            axes[i, j * 2 + 1].imshow(dilation, cmap="gray")
            axes[i, j * 2 + 1].set_title(f"{shape.capitalize()} {size}\nDilation")
            axes[i, j * 2 + 1].axis("off")

            print(f"\n{shape.upper()} {size}:")
            print(f"  Kernel shape: {kernel.shape}")
            print(f"  Erosion - Mean intensity: {erosion.mean():.2f}")
            print(f"  Dilation - Mean intensity: {dilation.mean():.2f}")

    plt.tight_layout()
    plt.savefig(
        "../outputs/morphologicalOperations/structuring_elements_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


def experiment_with_iterations(image, kernel):
    """Experiment with different numbers of iterations."""
    print("\n" + "=" * 70)
    print("EXPERIMENTING WITH DIFFERENT ITERATION COUNTS")
    print("=" * 70)

    iterations_list = [1, 2, 3, 5, 7, 10]

    fig, axes = plt.subplots(2, 6, figsize=(20, 8))
    fig.suptitle(
        "Effect of Iteration Count on Morphological Operations",
        fontsize=16,
        fontweight="bold",
    )

    for i, iters in enumerate(iterations_list):
        # Erosion
        erosion = apply_erosion(image, kernel, iterations=iters)
        axes[0, i].imshow(erosion, cmap="gray")
        axes[0, i].set_title(f"Erosion\n{iters} iteration(s)")
        axes[0, i].axis("off")

        # Dilation
        dilation = apply_dilation(image, kernel, iterations=iters)
        axes[1, i].imshow(dilation, cmap="gray")
        axes[1, i].set_title(f"Dilation\n{iters} iteration(s)")
        axes[1, i].axis("off")

        print(f"\nIterations: {iters}")
        print(f"  Erosion - Non-zero pixels: {np.count_nonzero(erosion)}")
        print(f"  Dilation - Non-zero pixels: {np.count_nonzero(dilation)}")

    plt.tight_layout()
    plt.savefig(
        "../outputs/morphologicalOperations/iterations_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


def demonstrate_image_enhancement(image):
    """Demonstrate practical image enhancement scenarios."""
    print("\n" + "=" * 70)
    print("PRACTICAL IMAGE ENHANCEMENT SCENARIOS")
    print("=" * 70)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Scenario 1: Noise removal
    print("\n1. NOISE REMOVAL (Opening):")
    opening = apply_opening(image, kernel, iterations=2)
    print("   Opening removes small bright spots (salt noise)")

    # Scenario 2: Gap filling
    print("\n2. GAP FILLING (Closing):")
    closing = apply_closing(image, kernel, iterations=2)
    print("   Closing fills small dark gaps in bright regions")

    # Scenario 3: Feature extraction
    print("\n3. FEATURE EXTRACTION:")
    gradient = apply_gradient(image, kernel)
    print("   Gradient extracts boundaries and edges")

    tophat = apply_tophat(image, kernel)
    print("   Top Hat extracts small bright features")

    blackhat = apply_blackhat(image, kernel)
    print("   Black Hat extracts small dark features")

    # Scenario 4: Combined enhancement
    print("\n4. COMBINED ENHANCEMENT:")
    # Remove noise then enhance edges
    denoised = apply_opening(image, kernel)
    enhanced = apply_gradient(denoised, kernel)
    print("   Opening for noise removal + Gradient for edge enhancement")

    # Visualize enhancement scenarios
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        "Practical Image Enhancement Scenarios", fontsize=16, fontweight="bold"
    )

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(opening, cmap="gray")
    axes[0, 1].set_title("Noise Removal\n(Opening)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(closing, cmap="gray")
    axes[0, 2].set_title("Gap Filling\n(Closing)")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(gradient, cmap="gray")
    axes[0, 3].set_title("Edge Detection\n(Gradient)")
    axes[0, 3].axis("off")

    axes[1, 0].imshow(tophat, cmap="gray")
    axes[1, 0].set_title("Bright Features\n(Top Hat)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(blackhat, cmap="gray")
    axes[1, 1].set_title("Dark Features\n(Black Hat)")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(enhanced, cmap="gray")
    axes[1, 2].set_title("Combined Enhancement\n(Open + Gradient)")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(np.hstack([image, enhanced]), cmap="gray")
    axes[1, 3].set_title("Original | Enhanced")
    axes[1, 3].axis("off")

    plt.tight_layout()
    plt.savefig(
        "../outputs/morphologicalOperations/image_enhancement_scenarios.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()


def analyze_morphological_operations():
    """Print analysis of morphological operations."""
    print("\n" + "=" * 70)
    print("MORPHOLOGICAL OPERATIONS ANALYSIS")
    print("=" * 70)

    print("\n1. EROSION:")
    print("   - Shrinks bright regions")
    print("   - Removes small bright spots and thin connections")
    print("   - Useful for: Noise removal, separating touching objects")

    print("\n2. DILATION:")
    print("   - Expands bright regions")
    print("   - Fills small holes and gaps")
    print("   - Useful for: Connecting broken lines, filling gaps")

    print("\n3. OPENING (Erosion → Dilation):")
    print("   - Removes small bright regions")
    print("   - Smooths contours")
    print("   - Useful for: Salt noise removal, eliminating small objects")

    print("\n4. CLOSING (Dilation → Erosion):")
    print("   - Fills small dark regions")
    print("   - Connects nearby objects")
    print("   - Useful for: Pepper noise removal, gap filling")

    print("\n5. MORPHOLOGICAL GRADIENT (Dilation - Erosion):")
    print("   - Highlights boundaries")
    print("   - Edge detection")
    print("   - Useful for: Contour extraction, edge detection")

    print("\n6. TOP HAT (Original - Opening):")
    print("   - Extracts small bright features")
    print("   - Removes large-scale variations")
    print("   - Useful for: Feature extraction, background removal")

    print("\n7. BLACK HAT (Closing - Original):")
    print("   - Extracts small dark features")
    print("   - Useful for: Dark spot detection")

    print("\n8. STRUCTURING ELEMENT EFFECTS:")
    print("   - Rectangle: Directional, good for linear features")
    print("   - Ellipse: Isotropic, good for circular features")
    print("   - Cross: Minimal dilation/erosion, preserves corners")
    print("   - Larger size: Stronger effect, removes larger features")

    print("\n9. ITERATION EFFECTS:")
    print("   - More iterations: Stronger effect")
    print("   - Erosion iterations: Progressively shrink objects")
    print("   - Dilation iterations: Progressively expand objects")
    print("   - Trade-off: Effect strength vs. computational cost")

    print("\n" + "=" * 70 + "\n")


def main():
    """Main function to execute morphological operations tasks."""
    # Load image
    image_path = "../assets/opening.png"
    print(f"Loading image from: {image_path}")
    original_image = load_and_preprocess_image(image_path)
    print(f"Image loaded successfully. Shape: {original_image.shape}")

    # Create default structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Apply basic morphological operations
    print("\nApplying basic morphological operations...")
    erosion = apply_erosion(original_image, kernel)
    dilation = apply_dilation(original_image, kernel)
    opening = apply_opening(original_image, kernel)
    closing = apply_closing(original_image, kernel)

    print("Basic operations completed.")

    # Visualize basic operations
    print("\nVisualizing basic morphological operations...")
    visualize_basic_operations(original_image, erosion, dilation, opening, closing)

    # Apply advanced morphological operations
    print("\nApplying advanced morphological operations...")
    gradient = apply_gradient(original_image, kernel)
    tophat = apply_tophat(original_image, kernel)
    blackhat = apply_blackhat(original_image, kernel)

    print("Advanced operations completed.")

    # Visualize advanced operations
    print("\nVisualizing advanced morphological operations...")
    visualize_advanced_operations(original_image, gradient, tophat, blackhat)

    # Experiment with structuring elements
    print("\nExperimenting with different structuring elements...")
    experiment_with_structuring_elements(original_image)

    # Experiment with iterations
    print("\nExperimenting with different iteration counts...")
    experiment_with_iterations(original_image, kernel)

    # Demonstrate practical enhancements
    print("\nDemonstrating practical image enhancement scenarios...")
    demonstrate_image_enhancement(original_image)

    # Print analysis
    analyze_morphological_operations()

    print("\nTask 3 completed successfully!")
    print("Output images saved:")
    print("  - basic_morphological_operations.png")
    print("  - advanced_morphological_operations.png")
    print("  - structuring_elements_comparison.png")
    print("  - iterations_comparison.png")
    print("  - image_enhancement_scenarios.png")


if __name__ == "__main__":
    main()
