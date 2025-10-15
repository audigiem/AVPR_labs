import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import restoration
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio
import warnings

warnings.filterwarnings("ignore")


def load_and_preprocess_image(image_path=None, target_size=(512, 512)):
    """Load and preprocess image with consistent sizing"""
    if image_path is None:
        # Create a synthetic test image with various features for noise analysis
        image = np.zeros(target_size, dtype=np.uint8)
        # Add different structures that react differently to noise

        # Large smooth areas
        cv2.rectangle(image, (50, 50), (200, 200), 180, -1)
        cv2.rectangle(image, (300, 100), (450, 250), 120, -1)

        # Fine details
        for i in range(10, 100, 5):
            cv2.line(image, (i, 300), (i, 400), 200, 1)

        # Circular patterns
        cv2.circle(image, (150, 350), 60, 220, -1)
        cv2.circle(image, (400, 400), 40, 80, -1)

        # Text with fine details
        cv2.putText(image, "NOISE", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 2)

        # Add some gradients
        for i in range(target_size[0]):
            for j in range(100):
                image[i, j] = int(255 * (j / 100))
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


def add_salt_pepper_noise(image, noise_level):
    """Add salt and pepper noise to an image

    Args:
        image: Input image (0-255)
        noise_level: Noise level as percentage (0.01 for 1%, 0.05 for 5%, etc.)

    Returns:
        Noisy image
    """
    # Convert to float [0,1] for skimage
    image_float = image.astype(np.float32) / 255.0

    # Add salt and pepper noise
    noisy = random_noise(image_float, mode="s&p", amount=noise_level)

    # Convert back to uint8 [0,255]
    noisy_image = (noisy * 255).astype(np.uint8)

    return noisy_image


def apply_median_filter(image, kernel_size=5):
    """Apply median filter"""
    return cv2.medianBlur(image, kernel_size)


def apply_gaussian_filter(image, kernel_size=5, sigma=1.0):
    """Apply Gaussian filter"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def apply_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    """Apply bilateral filter"""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_nlm_denoising(image, h=10, template_window_size=7, search_window_size=21):
    """Apply Non-Local Means denoising"""
    return cv2.fastNlMeansDenoising(
        image, None, h, template_window_size, search_window_size
    )


def calculate_psnr(original, filtered):
    """Calculate PSNR between original and filtered image"""
    # Convert to float for calculation
    original_float = original.astype(np.float64)
    filtered_float = filtered.astype(np.float64)

    return peak_signal_noise_ratio(original_float, filtered_float, data_range=255)


def process_noise_levels(image):
    """Process image with different noise levels and filters"""

    noise_levels = [0.01, 0.05, 0.10]  # 1%, 5%, 10%
    noise_level_names = ["1%", "5%", "10%"]

    results = {}

    for i, noise_level in enumerate(noise_levels):
        level_name = noise_level_names[i]
        print(f"Processing noise level: {level_name}")

        # Add noise
        noisy_image = add_salt_pepper_noise(image, noise_level)

        # Apply different filters
        median_filtered = apply_median_filter(noisy_image, 5)
        gaussian_filtered = apply_gaussian_filter(noisy_image, 5, 1.0)
        bilateral_filtered = apply_bilateral_filter(noisy_image)
        nlm_filtered = apply_nlm_denoising(noisy_image)

        # Calculate PSNRs
        psnr_noisy = calculate_psnr(image, noisy_image)
        psnr_median = calculate_psnr(image, median_filtered)
        psnr_gaussian = calculate_psnr(image, gaussian_filtered)
        psnr_bilateral = calculate_psnr(image, bilateral_filtered)
        psnr_nlm = calculate_psnr(image, nlm_filtered)

        results[level_name] = {
            "noisy": noisy_image,
            "median": median_filtered,
            "gaussian": gaussian_filtered,
            "bilateral": bilateral_filtered,
            "nlm": nlm_filtered,
            "psnr_noisy": psnr_noisy,
            "psnr_median": psnr_median,
            "psnr_gaussian": psnr_gaussian,
            "psnr_bilateral": psnr_bilateral,
            "psnr_nlm": psnr_nlm,
        }

        print(f"  Noisy PSNR: {psnr_noisy:.2f} dB")
        print(f"  Median PSNR: {psnr_median:.2f} dB")
        print(f"  Gaussian PSNR: {psnr_gaussian:.2f} dB")
        print(f"  Bilateral PSNR: {psnr_bilateral:.2f} dB")
        print(f"  NLM PSNR: {psnr_nlm:.2f} dB")

    return results


def display_comprehensive_comparison(original_image, results, figure_size=(20, 15)):
    """Display comprehensive comparison of all results"""

    noise_levels = ["1%", "5%", "10%"]
    filters = ["noisy", "median", "gaussian", "bilateral", "nlm"]
    filter_names = [
        "Noisy",
        "Median Filter",
        "Gaussian Filter",
        "Bilateral Filter",
        "NLM Denoising",
    ]

    # Create figure
    fig, axes = plt.subplots(4, 4, figsize=figure_size)

    # Original image (top-left)
    axes[0, 0].imshow(original_image, cmap="gray")
    axes[0, 0].set_title("Original Clean Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    # Hide unused subplot in first row
    for i in range(1, 4):
        axes[0, i].axis("off")

    # Display results for each noise level
    for noise_idx, noise_level in enumerate(noise_levels):
        row = noise_idx + 1

        for filter_idx, (filter_key, filter_name) in enumerate(
            zip(filters, filter_names)
        ):
            col = filter_idx
            if col >= 4:  # Skip if too many filters
                break

            image_data = results[noise_level][filter_key]
            psnr_key = f"psnr_{filter_key}"
            psnr_value = results[noise_level][psnr_key]

            axes[row, col].imshow(image_data, cmap="gray")
            axes[row, col].set_title(
                f"{filter_name}\n{noise_level} Noise\nPSNR: {psnr_value:.2f} dB",
                fontsize=10,
            )
            axes[row, col].axis("off")

    # Hide the last column if we have only 4 filters but 5 columns
    if len(filters) < 4:
        for row in range(1, 4):
            axes[row, 3].axis("off")

    plt.tight_layout()
    plt.show()

    return fig


def display_filter_comparison_by_noise(original_image, results, figure_size=(16, 12)):
    """Display filter comparison organized by noise level"""

    noise_levels = ["1%", "5%", "10%"]

    fig, axes = plt.subplots(3, 6, figsize=figure_size)

    for noise_idx, noise_level in enumerate(noise_levels):
        row = noise_idx

        # Original (for reference)
        axes[row, 0].imshow(original_image, cmap="gray")
        axes[row, 0].set_title(f"Original\n{noise_level} Noise Level", fontsize=10)
        axes[row, 0].axis("off")

        # Noisy
        axes[row, 1].imshow(results[noise_level]["noisy"], cmap="gray")
        axes[row, 1].set_title(
            f'Noisy\nPSNR: {results[noise_level]["psnr_noisy"]:.1f} dB', fontsize=10
        )
        axes[row, 1].axis("off")

        # Median
        axes[row, 2].imshow(results[noise_level]["median"], cmap="gray")
        axes[row, 2].set_title(
            f'Median\nPSNR: {results[noise_level]["psnr_median"]:.1f} dB', fontsize=10
        )
        axes[row, 2].axis("off")

        # Gaussian
        axes[row, 3].imshow(results[noise_level]["gaussian"], cmap="gray")
        axes[row, 3].set_title(
            f'Gaussian\nPSNR: {results[noise_level]["psnr_gaussian"]:.1f} dB',
            fontsize=10,
        )
        axes[row, 3].axis("off")

        # Bilateral
        axes[row, 4].imshow(results[noise_level]["bilateral"], cmap="gray")
        axes[row, 4].set_title(
            f'Bilateral\nPSNR: {results[noise_level]["psnr_bilateral"]:.1f} dB',
            fontsize=10,
        )
        axes[row, 4].axis("off")

        # NLM
        axes[row, 5].imshow(results[noise_level]["nlm"], cmap="gray")
        axes[row, 5].set_title(
            f'NLM\nPSNR: {results[noise_level]["psnr_nlm"]:.1f} dB', fontsize=10
        )
        axes[row, 5].axis("off")

    plt.tight_layout()
    plt.show()

    return fig


def analyze_filter_performance(results):
    """Analyze and compare filter performance"""

    print("\n" + "=" * 80)
    print("NOISE REDUCTION FILTER PERFORMANCE ANALYSIS")
    print("=" * 80)

    noise_levels = ["1%", "5%", "10%"]
    filters = ["median", "gaussian", "bilateral", "nlm"]
    filter_names = [
        "Median Filter",
        "Gaussian Filter",
        "Bilateral Filter",
        "NLM Denoising",
    ]

    # Create performance summary
    performance_summary = {}

    for filter_key, filter_name in zip(filters, filter_names):
        performance_summary[filter_name] = {}
        for noise_level in noise_levels:
            psnr_key = f"psnr_{filter_key}"
            performance_summary[filter_name][noise_level] = results[noise_level][
                psnr_key
            ]

    # Print detailed analysis
    for noise_level in noise_levels:
        print(f"\nNoise Level: {noise_level}")
        print("-" * 30)

        # Get PSNRs for this noise level
        psnrs = []
        for filter_key in filters:
            psnr_key = f"psnr_{filter_key}"
            psnrs.append(
                (
                    filter_names[filters.index(filter_key)],
                    results[noise_level][psnr_key],
                )
            )

        # Sort by PSNR (best first)
        psnrs.sort(key=lambda x: x[1], reverse=True)

        for rank, (filter_name, psnr) in enumerate(psnrs, 1):
            print(f"  {rank}. {filter_name}: {psnr:.2f} dB")

        print(f"  Noisy Image PSNR: {results[noise_level]['psnr_noisy']:.2f} dB")

    # Overall performance analysis
    print(f"\nOVERALL PERFORMANCE ANALYSIS:")
    print("=" * 40)

    # Calculate average performance
    avg_performance = {}
    for filter_name in filter_names:
        avg_psnr = np.mean(
            [performance_summary[filter_name][level] for level in noise_levels]
        )
        avg_performance[filter_name] = avg_psnr

    # Sort by average performance
    sorted_filters = sorted(avg_performance.items(), key=lambda x: x[1], reverse=True)

    print("Average PSNR across all noise levels:")
    for rank, (filter_name, avg_psnr) in enumerate(sorted_filters, 1):
        print(f"  {rank}. {filter_name}: {avg_psnr:.2f} dB")

    # Analysis by noise level
    print(f"\nFILTER CHARACTERISTICS:")
    print("-" * 40)
    print("* Median Filter: Excellent for salt-and-pepper noise, preserves edges")
    print("* Gaussian Filter: Good for Gaussian noise, smooths all details")
    print("* Bilateral Filter: Preserves edges while smoothing, good balance")
    print("* NLM Denoising: Advanced method, good for texture preservation")

    return performance_summary


def plot_psnr_comparison(results):
    """Plot PSNR comparison across noise levels"""

    noise_levels = ["1%", "5%", "10%"]
    filters = ["median", "gaussian", "bilateral", "nlm"]
    filter_names = ["Median", "Gaussian", "Bilateral", "NLM"]
    colors = ["blue", "green", "red", "orange"]

    plt.figure(figsize=(12, 8))

    # Plot PSNR for each filter
    for i, (filter_key, filter_name, color) in enumerate(
        zip(filters, filter_names, colors)
    ):
        psnr_values = []
        for noise_level in noise_levels:
            psnr_key = f"psnr_{filter_key}"
            psnr_values.append(results[noise_level][psnr_key])

        plt.plot(
            noise_levels,
            psnr_values,
            marker="o",
            linewidth=2,
            label=filter_name,
            color=color,
            markersize=8,
        )

    # Plot noisy image PSNR for reference
    noisy_psnr_values = [results[level]["psnr_noisy"] for level in noise_levels]
    plt.plot(
        noise_levels,
        noisy_psnr_values,
        marker="s",
        linewidth=2,
        label="Noisy (No Filter)",
        color="black",
        linestyle="--",
        markersize=8,
    )

    plt.xlabel("Noise Level", fontsize=12)
    plt.ylabel("PSNR (dB)", fontsize=12)
    plt.title(
        "Filter Performance Comparison\nHigher PSNR = Better Quality",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Add annotations for best performance
    for i, noise_level in enumerate(noise_levels):
        max_psnr = max([results[noise_level][f"psnr_{f}"] for f in filters])
        plt.annotate(
            f"{max_psnr:.1f} dB",
            xy=(i, max_psnr),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

    plt.tight_layout()
    plt.show()


def main():
    """Main function to demonstrate noise reduction analysis"""

    # Configuration
    TARGET_SIZE = (512, 512)

    # List of test images
    IMAGES_PATH = [
        "assets/100.pgm",
        "assets/couchersoleil.pgm",
        "assets/einstein.pgm",
        "assets/monroe.pgm",
        "assets/mandrill256.pgm",
    ]

    # Process each image or create synthetic if no images available
    test_images = []
    for image_path in IMAGES_PATH:
        if os.path.exists(image_path):
            test_images.append(image_path)

    if not test_images:
        print("No test images found, using synthetic image...")
        test_images = [None]  # Will create synthetic image

    # Process first available image (or synthetic)
    image_path = test_images[0]

    print(f"\n{'='*80}")
    if image_path:
        print(f"Processing: {image_path}")
    else:
        print("Processing: Synthetic Test Image")
    print(f"{'='*80}")

    try:
        # Load and preprocess image
        original_image = load_and_preprocess_image(image_path, TARGET_SIZE)

        # Process different noise levels and apply filters
        print("Processing noise levels and applying filters...")
        results = process_noise_levels(original_image)

        # Display comprehensive comparison
        print("\nDisplaying comprehensive comparison...")
        fig1 = display_filter_comparison_by_noise(original_image, results)

        # Save the comparison
        if image_path:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            fig1.savefig(
                f"noise_analysis_{base_name}.png", dpi=150, bbox_inches="tight"
            )
        else:
            fig1.savefig("noise_analysis_synthetic.png", dpi=150, bbox_inches="tight")

        # Analyze filter performance
        performance_summary = analyze_filter_performance(results)

        # Plot PSNR comparison
        print("\nDisplaying PSNR comparison plot...")
        plot_psnr_comparison(results)
    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    main()
