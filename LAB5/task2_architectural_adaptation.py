"""
Task 2: Architectural Adaptation
Objective: Modify the ResNet architecture for improved performance on the MNIST dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import time

# Define a custom ResNet with Dropout layers
class CustomResNetWithDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CustomResNetWithDropout, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # Change the first convolutional layer to accept one channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Add Dropout before the final layer
        self.dropout = nn.Dropout(p=dropout_rate)

        # Change the input size in the fully connected layer
        self.resnet.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # Apply dropout
        x = self.resnet.fc(x)

        return x


# Define a custom ResNet with different kernel sizes
class CustomResNetKernelSize(nn.Module):
    def __init__(self, kernel_size=7):
        super(CustomResNetKernelSize, self).__init__()
        self.resnet = models.resnet18(pretrained=True)

        # Change the first convolutional layer with custom kernel size
        padding = kernel_size // 2
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=kernel_size,
                                     stride=2, padding=padding, bias=False)

        # Change the input size in the fully connected layer
        self.resnet.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.resnet(x)


# Define a custom ResNet with different filter depths
class CustomResNetFilterDepth(nn.Module):
    def __init__(self, base_filters=64):
        super(CustomResNetFilterDepth, self).__init__()
        self.resnet = models.resnet18(pretrained=True)

        # Change the first convolutional layer
        self.resnet.conv1 = nn.Conv2d(1, base_filters, kernel_size=7,
                                     stride=2, padding=3, bias=False)

        # Modify batch norm to match new filter count
        self.resnet.bn1 = nn.BatchNorm2d(base_filters)

        # Change the input size in the fully connected layer
        self.resnet.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.resnet(x)


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
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

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, '
              f'Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s')

    return train_losses, test_accuracies, epoch_times


def experiment_dropout_layers():
    """Experiment 1: Layer Modification (Adding Dropout)"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: DROPOUT LAYER MODIFICATION")
    print("="*70)

    dropout_rates = [0.0, 0.3, 0.5, 0.7]
    results = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Prepare data
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

    for dropout_rate in dropout_rates:
        print(f"\nTesting Dropout Rate: {dropout_rate}")
        print("-" * 50)

        model = CustomResNetWithDropout(dropout_rate=dropout_rate)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses, test_accuracies, epoch_times = train_model(
            model, train_loader, test_loader, criterion, optimizer,
            num_epochs=5, device=device
        )

        results[dropout_rate] = {
            'losses': train_losses,
            'accuracies': test_accuracies,
            'times': epoch_times,
            'final_accuracy': test_accuracies[-1]
        }

    # Plot results
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    for dr in dropout_rates:
        plt.plot(results[dr]['losses'], label=f'Dropout={dr}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Dropout Rate')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 3, 2)
    for dr in dropout_rates:
        plt.plot(results[dr]['accuracies'], label=f'Dropout={dr}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs Dropout Rate')
    plt.legend()
    plt.grid(True)

    # Final comparison
    plt.subplot(1, 3, 3)
    dr_str = [str(dr) for dr in dropout_rates]
    final_accs = [results[dr]['final_accuracy'] for dr in dropout_rates]
    plt.bar(dr_str, final_accs)
    plt.xlabel('Dropout Rate')
    plt.ylabel('Final Test Accuracy (%)')
    plt.title('Final Accuracy Comparison')
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('task2_dropout.png')
    print("\nPlot saved as 'task2_dropout.png'")

    return results


def experiment_kernel_sizes():
    """Experiment 2: Filter Size Adjustments"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: KERNEL/FILTER SIZE ADJUSTMENTS")
    print("="*70)

    kernel_sizes = [3, 5, 7]
    results = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

    for kernel_size in kernel_sizes:
        print(f"\nTesting Kernel Size: {kernel_size}x{kernel_size}")
        print("-" * 50)

        model = CustomResNetKernelSize(kernel_size=kernel_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses, test_accuracies, epoch_times = train_model(
            model, train_loader, test_loader, criterion, optimizer,
            num_epochs=5, device=device
        )

        results[kernel_size] = {
            'losses': train_losses,
            'accuracies': test_accuracies,
            'times': epoch_times,
            'final_accuracy': test_accuracies[-1]
        }

    # Plot results
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    for ks in kernel_sizes:
        plt.plot(results[ks]['losses'], label=f'Kernel={ks}x{ks}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Kernel Size')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 3, 2)
    for ks in kernel_sizes:
        plt.plot(results[ks]['accuracies'], label=f'Kernel={ks}x{ks}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs Kernel Size')
    plt.legend()
    plt.grid(True)

    # Final comparison
    plt.subplot(1, 3, 3)
    ks_str = [f'{ks}x{ks}' for ks in kernel_sizes]
    final_accs = [results[ks]['final_accuracy'] for ks in kernel_sizes]
    plt.bar(ks_str, final_accs)
    plt.xlabel('Kernel Size')
    plt.ylabel('Final Test Accuracy (%)')
    plt.title('Final Accuracy Comparison')
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('task2_kernel_sizes.png')
    print("\nPlot saved as 'task2_kernel_sizes.png'")

    return results


def experiment_filter_depth():
    """Experiment 3: Feature Map Depth Variation"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: FEATURE MAP DEPTH VARIATION")
    print("="*70)

    filter_depths = [32, 64, 128]
    results = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

    for depth in filter_depths:
        print(f"\nTesting Filter Depth: {depth}")
        print("-" * 50)

        model = CustomResNetFilterDepth(base_filters=depth)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses, test_accuracies, epoch_times = train_model(
            model, train_loader, test_loader, criterion, optimizer,
            num_epochs=5, device=device
        )

        results[depth] = {
            'losses': train_losses,
            'accuracies': test_accuracies,
            'times': epoch_times,
            'final_accuracy': test_accuracies[-1],
            'avg_time': sum(epoch_times) / len(epoch_times)
        }

    # Plot results
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    for depth in filter_depths:
        plt.plot(results[depth]['losses'], label=f'Filters={depth}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Filter Depth')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 3, 2)
    for depth in filter_depths:
        plt.plot(results[depth]['accuracies'], label=f'Filters={depth}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs Filter Depth')
    plt.legend()
    plt.grid(True)

    # Training time comparison
    plt.subplot(1, 3, 3)
    depth_str = [str(d) for d in filter_depths]
    avg_times = [results[d]['avg_time'] for d in filter_depths]
    plt.bar(depth_str, avg_times)
    plt.xlabel('Number of Filters')
    plt.ylabel('Average Epoch Time (s)')
    plt.title('Training Speed Comparison')
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('task2_filter_depth.png')
    print("\nPlot saved as 'task2_filter_depth.png'")

    return results


def main():
    print("\n" + "="*70)
    print("TASK 2: ARCHITECTURAL ADAPTATION")
    print("="*70)

    # Run all experiments
    dropout_results = experiment_dropout_layers()
    kernel_results = experiment_kernel_sizes()
    depth_results = experiment_filter_depth()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)

    print("\n1. Dropout Layer Results:")
    for dr, result in dropout_results.items():
        print(f"   Dropout={dr}: Final Accuracy = {result['final_accuracy']:.2f}%")

    print("\n2. Kernel Size Results:")
    for ks, result in kernel_results.items():
        print(f"   Kernel={ks}x{ks}: Final Accuracy = {result['final_accuracy']:.2f}%")

    print("\n3. Filter Depth Results:")
    for depth, result in depth_results.items():
        print(f"   Filters={depth}: Final Accuracy = {result['final_accuracy']:.2f}%, "
              f"Avg Time = {result['avg_time']:.2f}s")

    print("\n" + "="*70)
    print("All experiments completed! Check the generated PNG files for visualizations.")
    print("="*70)


if __name__ == "__main__":
    main()

