"""
Plotting script for LAB6 Object Detection results
Generates comprehensive visualizations from task results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Create output directory for plots
OUTPUT_DIR = Path("outputs/plots")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def load_json(filepath):
    """Load JSON file"""
    with open(filepath, "r") as f:
        return json.load(f)


def plot_task1_results(data):
    """Plot Task 1: Hyperparameter Exploration results"""
    print("Plotting Task 1 results...")

    # Extract data
    configs = list(data.keys())
    final_losses = [data[c]["final_loss"] for c in configs]
    training_times = [data[c]["training_time"] for c in configs]

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Final Loss Comparison
    ax1 = plt.subplot(2, 3, 1)
    bars = ax1.bar(
        range(len(configs)), final_losses, color=sns.color_palette("husl", len(configs))
    )
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(
        [c.replace("config", "C").replace("_", "\n") for c in configs],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax1.set_ylabel("Final Loss", fontsize=11)
    ax1.set_title(
        "Final Training Loss by Configuration", fontsize=12, fontweight="bold"
    )
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, final_losses)):
        if val > 100:  # Special handling for outliers
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="red",
                fontweight="bold",
            )
        else:
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # 2. Training Time Comparison
    ax2 = plt.subplot(2, 3, 2)
    bars = ax2.bar(
        range(len(configs)),
        training_times,
        color=sns.color_palette("husl", len(configs)),
    )
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(
        [c.replace("config", "C").replace("_", "\n") for c in configs],
        rotation=45,
        ha="right",
        fontsize=9,
    )
    ax2.set_ylabel("Training Time (seconds)", fontsize=11)
    ax2.set_title("Training Time by Configuration", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, training_times):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.0f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # 3. Loss Evolution for each config
    ax3 = plt.subplot(2, 3, 3)
    for i, config in enumerate(configs):
        epoch_losses = data[config]["epoch_losses"]
        ax3.plot(
            range(1, len(epoch_losses) + 1),
            epoch_losses,
            marker="o",
            label=config.replace("config", "C").replace("_", " "),
            linewidth=2,
            markersize=4,
        )

    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("Loss", fontsize=11)
    ax3.set_title("Training Loss Evolution", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(alpha=0.3)

    # 4. Learning Rate Analysis
    ax4 = plt.subplot(2, 3, 4)
    lr_configs = {c: data[c] for c in configs if "lr" in c or "baseline" in c}
    lr_values = [0.0001, 0.001, 0.00001]  # baseline, high_lr, low_lr
    lr_losses = [
        data["config1_baseline"]["final_loss"],
        data["config2_high_lr"]["final_loss"],
        data["config3_low_lr"]["final_loss"],
    ]

    bars = ax4.bar(
        ["1e-4\n(baseline)", "1e-3\n(high)", "1e-5\n(low)"],
        lr_losses,
        color=["#2ecc71", "#e74c3c", "#3498db"],
    )
    ax4.set_ylabel("Final Loss", fontsize=11)
    ax4.set_title("Learning Rate Impact", fontsize=12, fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, lr_losses):
        if val > 100:
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="red",
                fontweight="bold",
            )
        else:
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # 5. Batch Size Analysis
    ax5 = plt.subplot(2, 3, 5)
    batch_configs = ["config1_baseline", "config4_batch2", "config5_batch4"]
    batch_sizes = [1, 2, 4]
    batch_losses = [data[c]["final_loss"] for c in batch_configs]
    batch_times = [data[c]["training_time"] for c in batch_configs]

    ax5_twin = ax5.twinx()

    bars1 = ax5.bar(
        np.arange(len(batch_sizes)) - 0.2,
        batch_losses,
        0.4,
        label="Final Loss",
        color="#3498db",
        alpha=0.8,
    )
    bars2 = ax5_twin.bar(
        np.arange(len(batch_sizes)) + 0.2,
        batch_times,
        0.4,
        label="Training Time",
        color="#e67e22",
        alpha=0.8,
    )

    ax5.set_xticks(range(len(batch_sizes)))
    ax5.set_xticklabels([f"Batch {bs}" for bs in batch_sizes])
    ax5.set_ylabel("Final Loss", fontsize=11, color="#3498db")
    ax5_twin.set_ylabel("Training Time (s)", fontsize=11, color="#e67e22")
    ax5.set_title("Batch Size Impact", fontsize=12, fontweight="bold")
    ax5.tick_params(axis="y", labelcolor="#3498db")
    ax5_twin.tick_params(axis="y", labelcolor="#e67e22")
    ax5.grid(axis="y", alpha=0.3)

    # Add legend
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    # 6. Epochs Analysis
    ax6 = plt.subplot(2, 3, 6)
    epoch_configs = ["config1_baseline", "config6_more_epochs"]
    epoch_counts = [10, 15]

    for config, epoch_count in zip(epoch_configs, epoch_counts):
        losses = data[config]["epoch_losses"]
        ax6.plot(
            range(1, len(losses) + 1),
            losses,
            marker="o",
            label=f"{epoch_count} epochs",
            linewidth=2,
            markersize=5,
        )

    ax6.set_xlabel("Epoch", fontsize=11)
    ax6.set_ylabel("Loss", fontsize=11)
    ax6.set_title("Effect of Training Duration", fontsize=12, fontweight="bold")
    ax6.legend(fontsize=10)
    ax6.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "task1_hyperparameter_exploration.png",
        dpi=300,
        bbox_inches="tight",
    )
    print(f"  Saved: {OUTPUT_DIR / 'task1_hyperparameter_exploration.png'}")
    plt.close()


def plot_task2_results(data):
    """Plot Task 2: Transfer Learning results"""
    print("Plotting Task 2 results...")

    configs = list(data.keys())
    final_losses = [data[c]["final_loss"] for c in configs]
    training_times = [data[c]["training_time"] for c in configs]
    trainable_params = [
        data[c]["trainable_params"] / 1e6 for c in configs
    ]  # in millions

    fig = plt.figure(figsize=(16, 10))

    # 1. Final Loss Comparison
    ax1 = plt.subplot(2, 3, 1)
    bars = ax1.bar(
        range(len(configs)), final_losses, color=sns.color_palette("husl", len(configs))
    )
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(
        [c.replace("_", "\n") for c in configs], rotation=45, ha="right", fontsize=9
    )
    ax1.set_ylabel("Final Loss", fontsize=11)
    ax1.set_title(
        "Final Loss by Transfer Learning Strategy", fontsize=12, fontweight="bold"
    )
    ax1.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, final_losses):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # 2. Trainable Parameters
    ax2 = plt.subplot(2, 3, 2)
    colors = [
        (
            "#e74c3c"
            if "mobilenet" in c
            else (
                "#3498db"
                if "no_freeze" in c
                else "#2ecc71" if "freeze" in c else "#f39c12"
            )
        )
        for c in configs
    ]
    bars = ax2.barh(range(len(configs)), trainable_params, color=colors)
    ax2.set_yticks(range(len(configs)))
    ax2.set_yticklabels([c.replace("_", " ").title() for c in configs], fontsize=9)
    ax2.set_xlabel("Trainable Parameters (Millions)", fontsize=11)
    ax2.set_title("Model Complexity", fontsize=12, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, trainable_params):
        ax2.text(
            val,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}M",
            ha="left",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    # 3. Training Loss Evolution
    ax3 = plt.subplot(2, 3, 3)
    for config in configs:
        losses = data[config]["epoch_losses"]
        ax3.plot(
            range(1, len(losses) + 1),
            losses,
            marker="o",
            label=config.replace("_", " ").title(),
            linewidth=2,
            markersize=4,
        )

    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("Loss", fontsize=11)
    ax3.set_title("Training Loss Evolution", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(alpha=0.3)
    ax3.set_ylim(bottom=0)

    # 4. Efficiency Analysis (Loss vs Training Time)
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(
        training_times,
        final_losses,
        s=200,
        c=range(len(configs)),
        cmap="viridis",
        alpha=0.7,
        edgecolors="black",
        linewidth=2,
    )

    for i, config in enumerate(configs):
        ax4.annotate(
            config.replace("_", "\n"),
            (training_times[i], final_losses[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
        )

    ax4.set_xlabel("Training Time (seconds)", fontsize=11)
    ax4.set_ylabel("Final Loss", fontsize=11)
    ax4.set_title("Efficiency: Loss vs Training Time", fontsize=12, fontweight="bold")
    ax4.grid(alpha=0.3)

    # 5. Freezing Strategy Comparison
    ax5 = plt.subplot(2, 3, 5)
    freeze_configs = ["no_freeze", "freeze_backbone", "gradual_unfreeze"]
    freeze_losses = [data[c]["final_loss"] for c in freeze_configs]
    freeze_params = [data[c]["trainable_params"] / 1e6 for c in freeze_configs]

    x = np.arange(len(freeze_configs))
    width = 0.35

    ax5_twin = ax5.twinx()
    bars1 = ax5.bar(
        x - width / 2,
        freeze_losses,
        width,
        label="Final Loss",
        color="#3498db",
        alpha=0.8,
    )
    bars2 = ax5_twin.bar(
        x + width / 2,
        freeze_params,
        width,
        label="Trainable Params (M)",
        color="#e74c3c",
        alpha=0.8,
    )

    ax5.set_xticks(x)
    ax5.set_xticklabels(
        ["No\nFreeze", "Freeze\nBackbone", "Gradual\nUnfreeze"], fontsize=9
    )
    ax5.set_ylabel("Final Loss", fontsize=11, color="#3498db")
    ax5_twin.set_ylabel("Trainable Params (M)", fontsize=11, color="#e74c3c")
    ax5.set_title("Freezing Strategy Impact", fontsize=12, fontweight="bold")
    ax5.tick_params(axis="y", labelcolor="#3498db")
    ax5_twin.tick_params(axis="y", labelcolor="#e74c3c")
    ax5.grid(axis="y", alpha=0.3)

    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    # 6. Backbone Comparison (ResNet50 vs MobileNet)
    ax6 = plt.subplot(2, 3, 6)
    backbones = ["ResNet-50\n(no freeze)", "MobileNetV3"]
    backbone_configs = ["no_freeze", "mobilenet"]
    backbone_data = {
        "Final Loss": [data[c]["final_loss"] for c in backbone_configs],
        "Training Time": [data[c]["training_time"] for c in backbone_configs],
        "Params (M)": [data[c]["trainable_params"] / 1e6 for c in backbone_configs],
    }

    x = np.arange(len(backbones))
    width = 0.25

    # Normalize values for comparison
    norm_loss = [
        v / max(backbone_data["Final Loss"]) for v in backbone_data["Final Loss"]
    ]
    norm_time = [
        v / max(backbone_data["Training Time"]) for v in backbone_data["Training Time"]
    ]
    norm_params = [
        v / max(backbone_data["Params (M)"]) for v in backbone_data["Params (M)"]
    ]

    bars1 = ax6.bar(
        x - width,
        norm_loss,
        width,
        label="Final Loss (norm)",
        color="#3498db",
        alpha=0.8,
    )
    bars2 = ax6.bar(
        x, norm_time, width, label="Training Time (norm)", color="#e74c3c", alpha=0.8
    )
    bars3 = ax6.bar(
        x + width,
        norm_params,
        width,
        label="Parameters (norm)",
        color="#2ecc71",
        alpha=0.8,
    )

    ax6.set_xticks(x)
    ax6.set_xticklabels(backbones, fontsize=10)
    ax6.set_ylabel("Normalized Value", fontsize=11)
    ax6.set_title(
        "Backbone Architecture Comparison\n(Normalized Metrics)",
        fontsize=12,
        fontweight="bold",
    )
    ax6.legend(fontsize=8)
    ax6.grid(axis="y", alpha=0.3)
    ax6.set_ylim([0, 1.2])

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "task2_transfer_learning.png", dpi=300, bbox_inches="tight"
    )
    print(f"  Saved: {OUTPUT_DIR / 'task2_transfer_learning.png'}")
    plt.close()


def plot_task3_results(data):
    """Plot Task 3: Data Augmentation results"""
    print("Plotting Task 3 results...")

    configs = list(data.keys())
    final_losses = [data[c]["final_loss"] for c in configs]
    training_times = [data[c]["training_time"] for c in configs]

    fig = plt.figure(figsize=(16, 10))

    # 1. Final Loss Comparison
    ax1 = plt.subplot(2, 3, 1)
    bars = ax1.bar(
        range(len(configs)), final_losses, color=sns.color_palette("husl", len(configs))
    )
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(
        [c.replace("_", "\n") for c in configs], rotation=45, ha="right", fontsize=8
    )
    ax1.set_ylabel("Final Loss", fontsize=11)
    ax1.set_title("Final Loss by Augmentation Strategy", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, final_losses):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # 2. Training Time Comparison
    ax2 = plt.subplot(2, 3, 2)
    bars = ax2.bar(
        range(len(configs)),
        training_times,
        color=sns.color_palette("husl", len(configs)),
    )
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(
        [c.replace("_", "\n") for c in configs], rotation=45, ha="right", fontsize=8
    )
    ax2.set_ylabel("Training Time (seconds)", fontsize=11)
    ax2.set_title(
        "Training Time by Augmentation Strategy", fontsize=12, fontweight="bold"
    )
    ax2.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, training_times):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.0f}s",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # 3. Training Loss Evolution
    ax3 = plt.subplot(2, 3, 3)
    for config in configs:
        losses = data[config]["epoch_losses"]
        ax3.plot(
            range(1, len(losses) + 1),
            losses,
            marker="o",
            label=config.replace("_", " ").title(),
            linewidth=2,
            markersize=4,
        )

    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("Loss", fontsize=11)
    ax3.set_title("Training Loss Evolution", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=7, loc="upper right")
    ax3.grid(alpha=0.3)

    # 4. Augmentation vs Baseline Comparison
    ax4 = plt.subplot(2, 3, 4)
    baseline_loss = data["basic_transform"]["final_loss"]
    improvements = [
        (baseline_loss - data[c]["final_loss"]) / baseline_loss * 100 for c in configs
    ]

    colors = ["#2ecc71" if imp > 0 else "#e74c3c" for imp in improvements]
    bars = ax4.barh(range(len(configs)), improvements, color=colors, alpha=0.7)
    ax4.set_yticks(range(len(configs)))
    ax4.set_yticklabels([c.replace("_", "\n") for c in configs], fontsize=8)
    ax4.set_xlabel("Improvement over Baseline (%)", fontsize=11)
    ax4.set_title(
        "Relative Performance vs Basic Transform", fontsize=12, fontweight="bold"
    )
    ax4.axvline(x=0, color="black", linestyle="--", linewidth=1)
    ax4.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, improvements):
        x_pos = val + (2 if val > 0 else -2)
        ax4.text(
            x_pos,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}%",
            ha="left" if val > 0 else "right",
            va="center",
            fontsize=8,
            fontweight="bold",
        )

    # 5. Loss Stability Analysis
    ax5 = plt.subplot(2, 3, 5)
    stds = [np.std(data[c]["epoch_losses"]) for c in configs]
    means = [np.mean(data[c]["epoch_losses"]) for c in configs]

    scatter = ax5.scatter(
        stds,
        means,
        s=200,
        c=range(len(configs)),
        cmap="viridis",
        alpha=0.7,
        edgecolors="black",
        linewidth=2,
    )

    for i, config in enumerate(configs):
        ax5.annotate(
            config.replace("_", "\n"),
            (stds[i], means[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=6,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
        )

    ax5.set_xlabel("Loss Std Dev (Stability)", fontsize=11)
    ax5.set_ylabel("Mean Loss", fontsize=11)
    ax5.set_title("Training Stability Analysis", fontsize=12, fontweight="bold")
    ax5.grid(alpha=0.3)

    # 6. Summary Comparison
    ax6 = plt.subplot(2, 3, 6)

    # Normalize metrics for comparison
    norm_loss = [data[c]["final_loss"] / max(final_losses) for c in configs]
    norm_time = [data[c]["training_time"] / max(training_times) for c in configs]

    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax6.bar(
        x - width / 2,
        norm_loss,
        width,
        label="Final Loss (norm)",
        color="#3498db",
        alpha=0.8,
    )
    bars2 = ax6.bar(
        x + width / 2,
        norm_time,
        width,
        label="Training Time (norm)",
        color="#e74c3c",
        alpha=0.8,
    )

    ax6.set_xticks(x)
    ax6.set_xticklabels(
        [c.replace("_", "\n") for c in configs], rotation=45, ha="right", fontsize=7
    )
    ax6.set_ylabel("Normalized Value", fontsize=11)
    ax6.set_title(
        "Overall Performance Comparison\n(Normalized)", fontsize=12, fontweight="bold"
    )
    ax6.legend(fontsize=9)
    ax6.grid(axis="y", alpha=0.3)
    ax6.set_ylim([0, 1.2])

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "task3_data_augmentation.png", dpi=300, bbox_inches="tight"
    )
    print(f"  Saved: {OUTPUT_DIR / 'task3_data_augmentation.png'}")
    plt.close()


def plot_combined_comparison(task1_data, task2_data, task3_data):
    """Create a combined comparison across all tasks"""
    print("Plotting combined comparison...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Task 1: Best configurations
    ax1 = axes[0, 0]
    task1_configs = list(task1_data.keys())
    task1_losses = [task1_data[c]["final_loss"] for c in task1_configs]
    # Exclude outliers for better visualization
    task1_losses_filtered = [l if l < 10 else 10 for l in task1_losses]

    bars = ax1.barh(
        range(len(task1_configs)),
        task1_losses_filtered,
        color=sns.color_palette("husl", len(task1_configs)),
    )
    ax1.set_yticks(range(len(task1_configs)))
    ax1.set_yticklabels(
        [c.replace("config", "C").replace("_", " ") for c in task1_configs], fontsize=8
    )
    ax1.set_xlabel("Final Loss", fontsize=10)
    ax1.set_title("Task 1: Hyperparameter Exploration", fontsize=11, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    # Task 2: Transfer learning strategies
    ax2 = axes[0, 1]
    task2_configs = list(task2_data.keys())
    task2_losses = [task2_data[c]["final_loss"] for c in task2_configs]

    bars = ax2.barh(
        range(len(task2_configs)),
        task2_losses,
        color=sns.color_palette("Set2", len(task2_configs)),
    )
    ax2.set_yticks(range(len(task2_configs)))
    ax2.set_yticklabels(
        [c.replace("_", " ").title() for c in task2_configs], fontsize=8
    )
    ax2.set_xlabel("Final Loss", fontsize=10)
    ax2.set_title("Task 2: Transfer Learning", fontsize=11, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    # Task 3: Augmentation strategies
    ax3 = axes[1, 0]
    task3_configs = list(task3_data.keys())
    task3_losses = [task3_data[c]["final_loss"] for c in task3_configs]

    bars = ax3.barh(
        range(len(task3_configs)),
        task3_losses,
        color=sns.color_palette("Set3", len(task3_configs)),
    )
    ax3.set_yticks(range(len(task3_configs)))
    ax3.set_yticklabels(
        [c.replace("_", " ").title() for c in task3_configs], fontsize=8
    )
    ax3.set_xlabel("Final Loss", fontsize=10)
    ax3.set_title("Task 3: Data Augmentation", fontsize=11, fontweight="bold")
    ax3.grid(axis="x", alpha=0.3)

    # Overall best comparison
    ax4 = axes[1, 1]
    best_configs = {
        "Task 1\nBaseline": task1_data["config1_baseline"]["final_loss"],
        "Task 1\nMore Epochs": task1_data["config6_more_epochs"]["final_loss"],
        "Task 2\nNo Freeze": task2_data["no_freeze"]["final_loss"],
        "Task 2\nGradual": task2_data["gradual_unfreeze"]["final_loss"],
        "Task 3\nNormalized": task3_data["normalized"]["final_loss"],
        "Task 3\nBasic": task3_data["basic_transform"]["final_loss"],
    }

    colors_best = ["#3498db", "#2980b9", "#e74c3c", "#c0392b", "#2ecc71", "#27ae60"]
    bars = ax4.bar(
        range(len(best_configs)),
        list(best_configs.values()),
        color=colors_best,
        alpha=0.8,
    )
    ax4.set_xticks(range(len(best_configs)))
    ax4.set_xticklabels(list(best_configs.keys()), rotation=45, ha="right", fontsize=8)
    ax4.set_ylabel("Final Loss", fontsize=10)
    ax4.set_title("Best Configurations Comparison", fontsize=11, fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, best_configs.values()):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "combined_comparison.png", dpi=300, bbox_inches="tight")
    print(f"  Saved: {OUTPUT_DIR / 'combined_comparison.png'}")
    plt.close()


def main():
    """Main plotting function"""
    print("\n" + "=" * 60)
    print("LAB6 Object Detection - Results Visualization")
    print("=" * 60 + "\n")

    # Load data
    print("Loading results...")
    task1_data = load_json("outputs/task1_hyperparameters/task1_results.json")
    task2_data = load_json("outputs/task2_transfer_learning/task2_results.json")
    task3_data = load_json("outputs/task3_augmentation/task3_results.json")
    print("  ✓ All data loaded\n")

    # Generate plots
    plot_task1_results(task1_data)
    plot_task2_results(task2_data)
    plot_task3_results(task3_data)
    plot_combined_comparison(task1_data, task2_data, task3_data)

    print("\n" + "=" * 60)
    print(f"All plots saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
