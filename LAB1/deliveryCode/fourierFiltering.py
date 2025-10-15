import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def load_and_preprocess_image(image_path=None, target_size=(512, 512)):
    """Load and preprocess image with consistent sizing"""
    if image_path is None:
        # Create a synthetic test image with various frequency components
        image = np.zeros(target_size, dtype=np.uint8)

        # Add low frequency components (large structures)
        cv2.rectangle(image, (100, 100), (400, 400), 180, -1)
        cv2.circle(image, (256, 256), 150, 120, -1)

        # Add medium frequency components
        for i in range(50, 450, 40):
            cv2.rectangle(image, (i, 50), (i + 20, 100), 200, -1)

        # Add high frequency components (fine details)
        for i in range(0, target_size[0], 8):
            cv2.line(image, (i, 450), (i, 500), 255, 1)

        # Add some text (high frequency)
        cv2.putText(image, "FREQUENCY", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

        # Add checkerboard pattern (high frequency)
        for i in range(350, 450, 20):
            for j in range(350, 450, 20):
                if (i // 20 + j // 20) % 2:
                    cv2.rectangle(image, (j, i), (j + 20, i + 20), 255, -1)

        print("Using synthetic test image with multiple frequency components")
    else:
        if not os.path.exists(image_path):
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


def compute_2d_fft(image):
    """Compute 2D FFT and shift zero frequency to center"""
    # Perform 2D FFT
    f_transform = np.fft.fft2(image)

    # Shift zero frequency to center
    f_shift = np.fft.fftshift(f_transform)

    return f_transform, f_shift


def compute_magnitude_spectrum(f_shift):
    """Compute magnitude spectrum for visualization"""
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    return magnitude_spectrum


def create_circular_mask(shape, center, radius, mask_type="low_pass"):
    """Create circular mask for frequency filtering

    Args:
        shape: Image shape (height, width)
        center: Center of the circle (cy, cx)
        radius: Radius of the circle
        mask_type: 'low_pass' or 'high_pass'

    Returns:
        Binary mask
    """
    h, w = shape
    cy, cx = center

    # Create coordinate matrices
    y, x = np.ogrid[:h, :w]

    # Calculate distance from center
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    if mask_type == "low_pass":
        # Keep frequencies within radius (low frequencies)
        mask = distance <= radius
    else:  # high_pass
        # Remove frequencies within radius (remove low frequencies)
        mask = distance > radius

    return mask.astype(np.float32)


def apply_frequency_filter(f_shift, mask):
    """Apply frequency domain filter"""
    # Apply mask to shifted FFT
    filtered_f_shift = f_shift * mask

    # Shift back and compute inverse FFT
    filtered_f = np.fft.ifftshift(filtered_f_shift)
    filtered_image = np.fft.ifft2(filtered_f)

    # Take real part and convert to proper range
    filtered_image = np.abs(filtered_image)
    filtered_image = np.uint8(np.clip(filtered_image, 0, 255))

    return filtered_image, filtered_f_shift


def display_frequency_analysis(image, figure_size=(20, 16)):
    """Display comprehensive frequency domain analysis"""

    # Compute FFT
    f_transform, f_shift = compute_2d_fft(image)
    magnitude_spectrum = compute_magnitude_spectrum(f_shift)

    # Image center for masks
    center = (image.shape[0] // 2, image.shape[1] // 2)

    # Test different cutoff radii
    cutoff_radii = [30, 60, 100]

    # Create figure with subplots
    fig = plt.figure(figsize=figure_size)
    gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)

    # Original image and spectrum
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image, cmap="gray")
    ax1.set_title("Original Image", fontsize=12, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(magnitude_spectrum, cmap="gray")
    ax2.set_title("Magnitude Spectrum\n(Original)", fontsize=12)
    ax2.axis("off")

    # Add circles to show cutoff radii on original spectrum
    for radius in cutoff_radii:
        circle = plt.Circle(center[::-1], radius, fill=False, color="red", linewidth=1)
        ax2.add_patch(circle)

    # Low-pass filters at different cutoffs
    for i, radius in enumerate(cutoff_radii):
        # Create low-pass mask
        lp_mask = create_circular_mask(image.shape, center, radius, "low_pass")
        lp_filtered, lp_f_shift = apply_frequency_filter(f_shift, lp_mask)
        lp_spectrum = compute_magnitude_spectrum(lp_f_shift)

        # Display filtered image
        ax_img = fig.add_subplot(gs[1, i])
        ax_img.imshow(lp_filtered, cmap="gray")
        ax_img.set_title(f"Low-pass Filter\nRadius: {radius}px", fontsize=11)
        ax_img.axis("off")

        # Display filtered spectrum
        ax_spec = fig.add_subplot(gs[2, i])
        ax_spec.imshow(lp_spectrum, cmap="gray")
        ax_spec.set_title(f"LP Spectrum\nR={radius}", fontsize=10)
        ax_spec.axis("off")

    # High-pass filter
    hp_radius = 60  # Use middle radius for high-pass
    hp_mask = create_circular_mask(image.shape, center, hp_radius, "high_pass")
    hp_filtered, hp_f_shift = apply_frequency_filter(f_shift, hp_mask)
    hp_spectrum = compute_magnitude_spectrum(hp_f_shift)

    # Display high-pass results
    ax_hp_img = fig.add_subplot(gs[1, 3])
    ax_hp_img.imshow(hp_filtered, cmap="gray")
    ax_hp_img.set_title(f"High-pass Filter\nRadius: {hp_radius}px", fontsize=11)
    ax_hp_img.axis("off")

    ax_hp_spec = fig.add_subplot(gs[2, 3])
    ax_hp_spec.imshow(hp_spectrum, cmap="gray")
    ax_hp_spec.set_title(f"HP Spectrum\nR={hp_radius}", fontsize=10)
    ax_hp_spec.axis("off")

    # Show masks
    ax_lp_mask = fig.add_subplot(gs[3, 0])
    ax_lp_mask.imshow(
        create_circular_mask(image.shape, center, 60, "low_pass"), cmap="gray"
    )
    ax_lp_mask.set_title("Low-pass Mask\n(R=60)", fontsize=10)
    ax_lp_mask.axis("off")

    ax_hp_mask = fig.add_subplot(gs[3, 1])
    ax_hp_mask.imshow(
        create_circular_mask(image.shape, center, 60, "high_pass"), cmap="gray"
    )
    ax_hp_mask.set_title("High-pass Mask\n(R=60)", fontsize=10)
    ax_hp_mask.axis("off")

    # Add frequency information
    ax_info = fig.add_subplot(gs[3, 2:])
    ax_info.text(
        0.1,
        0.5,
        transform=ax_info.transAxes,
        fontsize=11,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )
    ax_info.axis("off")

    plt.suptitle("Frequency Domain Filtering Analysis", fontsize=16, fontweight="bold")

    return fig, {
        "original": image,
        "magnitude_spectrum": magnitude_spectrum,
        "low_pass_results": [
            (r, create_circular_mask(image.shape, center, r, "low_pass"))
            for r in cutoff_radii
        ],
        "high_pass_result": (hp_radius, hp_mask),
    }


def compare_cutoff_effects(
    image, cutoff_radii=[20, 40, 60, 80, 120], figure_size=(20, 12)
):
    """Compare effects of different cutoff frequencies"""

    # Compute FFT
    f_transform, f_shift = compute_2d_fft(image)
    center = (image.shape[0] // 2, image.shape[1] // 2)

    fig, axes = plt.subplots(3, len(cutoff_radii) + 1, figsize=figure_size)

    # Original image
    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[1, 0].text(
        0.5,
        0.5,
        "Low-pass\nFiltered",
        ha="center",
        va="center",
        transform=axes[1, 0].transAxes,
        fontsize=12,
        fontweight="bold",
    )
    axes[1, 0].axis("off")

    axes[2, 0].text(
        0.5,
        0.5,
        "High-pass\nFiltered",
        ha="center",
        va="center",
        transform=axes[2, 0].transAxes,
        fontsize=12,
        fontweight="bold",
    )
    axes[2, 0].axis("off")

    # Process different cutoff radii
    for i, radius in enumerate(cutoff_radii):
        col = i + 1

        # Low-pass filter
        lp_mask = create_circular_mask(image.shape, center, radius, "low_pass")
        lp_filtered, _ = apply_frequency_filter(f_shift, lp_mask)

        # High-pass filter
        hp_mask = create_circular_mask(image.shape, center, radius, "high_pass")
        hp_filtered, _ = apply_frequency_filter(f_shift, hp_mask)

        # Display mask
        axes[0, col].imshow(lp_mask, cmap="gray")
        axes[0, col].set_title(f"Mask\nRadius: {radius}px", fontsize=11)
        axes[0, col].axis("off")

        # Display low-pass result
        axes[1, col].imshow(lp_filtered, cmap="gray")
        axes[1, col].set_title(f"LP R={radius}", fontsize=10)
        axes[1, col].axis("off")

        # Display high-pass result
        axes[2, col].imshow(hp_filtered, cmap="gray")
        axes[2, col].set_title(f"HP R={radius}", fontsize=10)
        axes[2, col].axis("off")

    plt.suptitle("Cutoff Frequency Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()

    return fig


def analyze_frequency_effects(image, cutoff_radii=[30, 60, 100]):
    """Analyze the effects of frequency filtering"""

    print("\n" + "=" * 80)
    print("FREQUENCY DOMAIN FILTERING ANALYSIS")
    print("=" * 80)

    # Compute FFT
    f_transform, f_shift = compute_2d_fft(image)
    center = (image.shape[0] // 2, image.shape[1] // 2)

    # Original image statistics
    original_mean = np.mean(image)
    original_std = np.std(image)

    print(f"Original Image Statistics:")
    print(f"  Mean: {original_mean:.2f}")
    print(f"  Std:  {original_std:.2f}")
    print(f"  Size: {image.shape}")
    print("-" * 50)

    print("Low-pass Filtering Results:")
    print("-" * 30)

    for radius in cutoff_radii:
        # Low-pass filter
        lp_mask = create_circular_mask(image.shape, center, radius, "low_pass")
        lp_filtered, _ = apply_frequency_filter(f_shift, lp_mask)

        # Calculate statistics
        lp_mean = np.mean(lp_filtered)
        lp_std = np.std(lp_filtered)

        # Calculate percentage of frequencies kept
        freq_kept = np.sum(lp_mask) / (image.shape[0] * image.shape[1]) * 100

        print(
            f"Radius {radius:3d}px: Mean={lp_mean:6.2f}, Std={lp_std:6.2f}, "
            f"Frequencies kept: {freq_kept:5.1f}%"
        )

    print(f"\nHigh-pass Filtering Results:")
    print("-" * 30)

    for radius in cutoff_radii:
        # High-pass filter
        hp_mask = create_circular_mask(image.shape, center, radius, "high_pass")
        hp_filtered, _ = apply_frequency_filter(f_shift, hp_mask)

        # Calculate statistics
        hp_mean = np.mean(hp_filtered)
        hp_std = np.std(hp_filtered)

        # Calculate percentage of frequencies kept
        freq_kept = np.sum(hp_mask) / (image.shape[0] * image.shape[1]) * 100

        print(
            f"Radius {radius:3d}px: Mean={hp_mean:6.2f}, Std={hp_std:6.2f}, "
            f"Frequencies kept: {freq_kept:5.1f}%"
        )

    print(f"\nKey Observations:")
    print("-" * 30)
    print("* Low-pass filtering:")
    print("  - Smaller radius → More smoothing, less detail")
    print("  - Reduces noise but blurs edges")
    print("  - Preserves overall image structure")
    print("* High-pass filtering:")
    print("  - Smaller radius → More edge enhancement")
    print("  - Emphasizes fine details and edges")
    print("  - Reduces smooth regions to near zero")
    print("* Applications:")
    print("  - Low-pass: Noise reduction, image smoothing")
    print("  - High-pass: Edge detection, detail enhancement")


def demonstrate_bandpass_filtering(image, figure_size=(16, 8)):
    """Demonstrate bandpass filtering by combining low and high pass"""

    # Compute FFT
    f_transform, f_shift = compute_2d_fft(image)
    center = (image.shape[0] // 2, image.shape[1] // 2)

    # Create bandpass filters
    bandpass_configs = [
        (20, 60, "Low-Mid Frequencies"),
        (60, 100, "Mid-High Frequencies"),
        (40, 80, "Mid Frequencies Only"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=figure_size)

    # Original
    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(compute_magnitude_spectrum(f_shift), cmap="gray")
    axes[1, 0].set_title("Original Spectrum", fontsize=12)
    axes[1, 0].axis("off")

    for i, (inner_radius, outer_radius, title) in enumerate(bandpass_configs):
        col = i + 1

        # Create bandpass mask (donut shape)
        outer_mask = create_circular_mask(image.shape, center, outer_radius, "low_pass")
        inner_mask = create_circular_mask(
            image.shape, center, inner_radius, "high_pass"
        )
        bandpass_mask = outer_mask * inner_mask

        # Apply bandpass filter
        bp_filtered, bp_f_shift = apply_frequency_filter(f_shift, bandpass_mask)
        bp_spectrum = compute_magnitude_spectrum(bp_f_shift)

        # Display results
        axes[0, col].imshow(bp_filtered, cmap="gray")
        axes[0, col].set_title(
            f"{title}\n({inner_radius}-{outer_radius}px)", fontsize=11
        )
        axes[0, col].axis("off")

        axes[1, col].imshow(bp_spectrum, cmap="gray")
        axes[1, col].set_title(
            f"Spectrum\n({inner_radius}-{outer_radius})", fontsize=10
        )
        axes[1, col].axis("off")

    plt.suptitle("Bandpass Filtering (Frequency Bands)", fontsize=16, fontweight="bold")
    plt.tight_layout()

    return fig


def main():
    """Main function to demonstrate frequency domain filtering"""

    print("Frequency Domain Image Processing")
    print("=" * 50)

    # Test images
    IMAGES_PATH = [
        "assets/100.pgm",
        "assets/couchersoleil.pgm",
        "assets/einstein.pgm",
        "assets/monroe.pgm",
        "assets/mandrill256.pgm",
    ]

    # Find available image or use synthetic
    image_path = None
    for path in IMAGES_PATH:
        if os.path.exists(path):
            image_path = path
            break

    try:
        # Load image
        image = load_and_preprocess_image(image_path, (512, 512))

        # 1. Comprehensive frequency analysis
        print("\n1. Performing comprehensive frequency domain analysis...")
        fig1, results = display_frequency_analysis(image)

        # Save results
        base_name = (
            "synthetic"
            if image_path is None
            else os.path.splitext(os.path.basename(image_path))[0]
        )
        fig1.savefig(
            f"frequency_analysis_{base_name}.png", dpi=150, bbox_inches="tight"
        )

        # 2. Compare different cutoff effects
        print("\n2. Comparing different cutoff frequencies...")
        fig2 = compare_cutoff_effects(image)
        fig2.savefig(
            f"frequency_cutoff_comparison_{base_name}.png", dpi=150, bbox_inches="tight"
        )

        # 3. Analyze effects quantitatively
        print("\n3. Analyzing frequency filtering effects...")
        analyze_frequency_effects(image)

        # 4. Demonstrate bandpass filtering
        print("\n4. Demonstrating bandpass filtering...")
        fig3 = demonstrate_bandpass_filtering(image)
        fig3.savefig(
            f"frequency_bandpass_{base_name}.png", dpi=150, bbox_inches="tight"
        )

        print(f"\nAll visualizations saved with prefix: frequency_*_{base_name}.png")

        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
