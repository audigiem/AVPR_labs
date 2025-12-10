"""
Task 1: Hyperparameter Exploration
Objective: Explore the impact of hyperparameters on the customized ResNet model's performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets, models
from tqdm import tqdm
import matplotlib.pyplot as plt
import time


# Define a custom ResNet for one channel input
class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # Change the first convolutional layer to accept one channel
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Change the input size in the fully connected layer
        self.resnet.fc = nn.Linear(512, 10)  # Assuming you want 10 classes for MNIST

    def forward(self, x):
        return self.resnet(x)


def train_model(
    model, train_loader, test_loader, criterion, optimizer, num_epochs, device
):
    """Train the model and return training history"""
    model.to(device)
    train_losses = []
    test_accuracies = []
    epoch_times = []

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Test accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        test_accuracies.append(accuracy)

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, "
            f"Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s"
        )

    return train_losses, test_accuracies, epoch_times


def experiment_learning_rates():
    """Experiment 1: Learning Rate Adjustment"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: LEARNING RATE ADJUSTMENT")
    print("=" * 70)

    learning_rates = [0.0001, 0.001, 0.01]
    results = {}

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Prepare data
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

    for lr in learning_rates:
        print(f"\nTesting Learning Rate: {lr}")
        print("-" * 50)

        model = CustomResNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_losses, test_accuracies, epoch_times = train_model(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            num_epochs=5,
            device=device,
        )

        results[lr] = {
            "losses": train_losses,
            "accuracies": test_accuracies,
            "times": epoch_times,
            "final_accuracy": test_accuracies[-1],
        }

    # Plot results
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    for lr in learning_rates:
        plt.plot(results[lr]["losses"], label=f"LR={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Learning Rate")
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 3, 2)
    for lr in learning_rates:
        plt.plot(results[lr]["accuracies"], label=f"LR={lr}")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy vs Learning Rate")
    plt.legend()
    plt.grid(True)

    # Final comparison
    plt.subplot(1, 3, 3)
    lrs_str = [str(lr) for lr in learning_rates]
    final_accs = [results[lr]["final_accuracy"] for lr in learning_rates]
    plt.bar(lrs_str, final_accs)
    plt.xlabel("Learning Rate")
    plt.ylabel("Final Test Accuracy (%)")
    plt.title("Final Accuracy Comparison")
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig("task1_learning_rates.png")
    print("\nPlot saved as 'task1_learning_rates.png'")

    return results


def experiment_batch_sizes():
    """Experiment 2: Batch Size Variance"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: BATCH SIZE VARIANCE")
    print("=" * 70)

    batch_sizes = [32, 64, 128]
    results = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    for batch_size in batch_sizes:
        print(f"\nTesting Batch Size: {batch_size}")
        print("-" * 50)

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=False
        )

        model = CustomResNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses, test_accuracies, epoch_times = train_model(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            num_epochs=5,
            device=device,
        )

        results[batch_size] = {
            "losses": train_losses,
            "accuracies": test_accuracies,
            "times": epoch_times,
            "avg_time": sum(epoch_times) / len(epoch_times),
            "final_accuracy": test_accuracies[-1],
        }

    # Plot results
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    for bs in batch_sizes:
        plt.plot(results[bs]["losses"], label=f"BS={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Batch Size")
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 3, 2)
    for bs in batch_sizes:
        plt.plot(results[bs]["accuracies"], label=f"BS={bs}")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy vs Batch Size")
    plt.legend()
    plt.grid(True)

    # Training time comparison
    plt.subplot(1, 3, 3)
    bs_str = [str(bs) for bs in batch_sizes]
    avg_times = [results[bs]["avg_time"] for bs in batch_sizes]
    plt.bar(bs_str, avg_times)
    plt.xlabel("Batch Size")
    plt.ylabel("Average Epoch Time (s)")
    plt.title("Training Speed Comparison")
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig("task1_batch_sizes.png")
    print("\nPlot saved as 'task1_batch_sizes.png'")

    return results


def experiment_epochs():
    """Experiment 3: Epoch Sensitivity"""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: EPOCH SENSITIVITY")
    print("=" * 70)

    epoch_configs = [5, 10, 15]
    results = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

    for num_epochs in epoch_configs:
        print(f"\nTesting Number of Epochs: {num_epochs}")
        print("-" * 50)

        model = CustomResNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses, test_accuracies, epoch_times = train_model(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            num_epochs=num_epochs,
            device=device,
        )

        results[num_epochs] = {
            "losses": train_losses,
            "accuracies": test_accuracies,
            "times": epoch_times,
            "final_accuracy": test_accuracies[-1],
        }

    # Plot results
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    for epochs in epoch_configs:
        plt.plot(
            range(1, epochs + 1),
            results[epochs]["losses"],
            label=f"Epochs={epochs}",
            marker="o",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Convergence")
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 3, 2)
    for epochs in epoch_configs:
        plt.plot(
            range(1, epochs + 1),
            results[epochs]["accuracies"],
            label=f"Epochs={epochs}",
            marker="o",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy Progression")
    plt.legend()
    plt.grid(True)

    # Final accuracy comparison
    plt.subplot(1, 3, 3)
    epochs_str = [str(e) for e in epoch_configs]
    final_accs = [results[e]["final_accuracy"] for e in epoch_configs]
    plt.bar(epochs_str, final_accs)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Final Test Accuracy (%)")
    plt.title("Final Accuracy Comparison")
    plt.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig("task1_epochs.png")
    print("\nPlot saved as 'task1_epochs.png'")

    return results


def main():
    print("\n" + "=" * 70)
    print("TASK 1: HYPERPARAMETER EXPLORATION")
    print("=" * 70)

    # Run all experiments
    lr_results = experiment_learning_rates()
    bs_results = experiment_batch_sizes()
    epoch_results = experiment_epochs()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF RESULTS")
    print("=" * 70)

    print("\n1. Learning Rate Results:")
    for lr, result in lr_results.items():
        print(f"   LR={lr}: Final Accuracy = {result['final_accuracy']:.2f}%")

    print("\n2. Batch Size Results:")
    for bs, result in bs_results.items():
        print(
            f"   BS={bs}: Final Accuracy = {result['final_accuracy']:.2f}%, "
            f"Avg Time = {result['avg_time']:.2f}s"
        )

    print("\n3. Epoch Results:")
    for epochs, result in epoch_results.items():
        print(f"   Epochs={epochs}: Final Accuracy = {result['final_accuracy']:.2f}%")

    print("\n" + "=" * 70)
    print(
        "All experiments completed! Check the generated PNG files for visualizations."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
