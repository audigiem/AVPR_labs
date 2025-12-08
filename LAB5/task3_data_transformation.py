"""
Task 3: Data Transformation Techniques
Objective: Evaluate the influence of data transformations on model performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import time
import numpy as np

# Define a custom ResNet for one channel input
class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # Change the first convolutional layer to accept one channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
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


def experiment_transformation_sequences():
    """Experiment 1: Transformation Sequence Experimentation"""
    print("\n" + "="*70)
    print("EXPERIMENT 1: TRANSFORMATION SEQUENCE EXPERIMENTATION")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Define different transformation sequences
    transformations = {
        'basic': transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        'with_rotation': transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        'with_brightness': transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ColorJitter(brightness=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        'combined': transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    }

    # Test transform without augmentation
    test_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    results = {}

    for name, transform in transformations.items():
        print(f"\nTesting Transformation: {name}")
        print("-" * 50)

        train_data = datasets.MNIST(root='./data', train=True, download=True,
                                   transform=transform)
        test_data = datasets.MNIST(root='./data', train=False, download=True,
                                  transform=test_transform)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

        model = CustomResNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses, test_accuracies, epoch_times = train_model(
            model, train_loader, test_loader, criterion, optimizer,
            num_epochs=5, device=device
        )

        results[name] = {
            'losses': train_losses,
            'accuracies': test_accuracies,
            'times': epoch_times,
            'final_accuracy': test_accuracies[-1]
        }

    # Plot results
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    for name in transformations.keys():
        plt.plot(results[name]['losses'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Transformation')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 3, 2)
    for name in transformations.keys():
        plt.plot(results[name]['accuracies'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs Transformation')
    plt.legend()
    plt.grid(True)

    # Final comparison
    plt.subplot(1, 3, 3)
    names = list(transformations.keys())
    final_accs = [results[name]['final_accuracy'] for name in names]
    plt.bar(names, final_accs)
    plt.xlabel('Transformation Type')
    plt.ylabel('Final Test Accuracy (%)')
    plt.title('Final Accuracy Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('task3_transformations.png')
    print("\nPlot saved as 'task3_transformations.png'")

    return results


def experiment_data_augmentation():
    """Experiment 2: Data Augmentation Trials"""
    print("\n" + "="*70)
    print("EXPERIMENT 2: DATA AUGMENTATION TRIALS")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Define different augmentation strategies
    augmentations = {
        'no_augmentation': transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        'rotation': transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        'affine': transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        'heavy_augmentation': transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    }

    # Test transform without augmentation
    test_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    results = {}

    for name, augmentation in augmentations.items():
        print(f"\nTesting Augmentation: {name}")
        print("-" * 50)

        train_data = datasets.MNIST(root='./data', train=True, download=True,
                                   transform=augmentation)
        test_data = datasets.MNIST(root='./data', train=False, download=True,
                                  transform=test_transform)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

        model = CustomResNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses, test_accuracies, epoch_times = train_model(
            model, train_loader, test_loader, criterion, optimizer,
            num_epochs=5, device=device
        )

        results[name] = {
            'losses': train_losses,
            'accuracies': test_accuracies,
            'times': epoch_times,
            'final_accuracy': test_accuracies[-1]
        }

    # Plot results
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    for name in augmentations.keys():
        plt.plot(results[name]['losses'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Augmentation')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 3, 2)
    for name in augmentations.keys():
        plt.plot(results[name]['accuracies'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs Augmentation')
    plt.legend()
    plt.grid(True)

    # Final comparison
    plt.subplot(1, 3, 3)
    names = list(augmentations.keys())
    final_accs = [results[name]['final_accuracy'] for name in names]
    plt.bar(names, final_accs)
    plt.xlabel('Augmentation Type')
    plt.ylabel('Final Test Accuracy (%)')
    plt.title('Final Accuracy Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('task3_augmentation.png')
    print("\nPlot saved as 'task3_augmentation.png'")

    return results


def experiment_normalization_methods():
    """Experiment 3: Normalization and Standardization"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: NORMALIZATION AND STANDARDIZATION")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Calculate MNIST dataset statistics for proper normalization
    # Mean and std for MNIST: mean=0.1307, std=0.3081

    # Define different normalization methods
    normalizations = {
        'simple_0.5': transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]),
        'mnist_stats': transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]),
        'no_normalization': transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ]),
        'custom_norm': transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1,), (0.25,))
        ])
    }

    results = {}

    for name, normalization in normalizations.items():
        print(f"\nTesting Normalization: {name}")
        print("-" * 50)

        train_data = datasets.MNIST(root='./data', train=True, download=True,
                                   transform=normalization)
        test_data = datasets.MNIST(root='./data', train=False, download=True,
                                  transform=normalization)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

        model = CustomResNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_losses, test_accuracies, epoch_times = train_model(
            model, train_loader, test_loader, criterion, optimizer,
            num_epochs=5, device=device
        )

        results[name] = {
            'losses': train_losses,
            'accuracies': test_accuracies,
            'times': epoch_times,
            'final_accuracy': test_accuracies[-1]
        }

    # Plot results
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    for name in normalizations.keys():
        plt.plot(results[name]['losses'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss vs Normalization')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 3, 2)
    for name in normalizations.keys():
        plt.plot(results[name]['accuracies'], label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy vs Normalization')
    plt.legend()
    plt.grid(True)

    # Final comparison
    plt.subplot(1, 3, 3)
    names = list(normalizations.keys())
    final_accs = [results[name]['final_accuracy'] for name in names]
    plt.bar(names, final_accs)
    plt.xlabel('Normalization Method')
    plt.ylabel('Final Test Accuracy (%)')
    plt.title('Final Accuracy Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('task3_normalization.png')
    print("\nPlot saved as 'task3_normalization.png'")

    return results


def visualize_augmented_samples():
    """Visualize some augmented samples"""
    print("\n" + "="*70)
    print("VISUALIZING AUGMENTED SAMPLES")
    print("="*70)

    # Create augmented transformation
    augmented_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor()
    ])

    # Load dataset
    dataset = datasets.MNIST(root='./data', train=True, download=True,
                           transform=augmented_transform)

    # Show 10 examples of the same digit with different augmentations
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Examples of Data Augmentation on MNIST', fontsize=16)

    # Get one image
    original_dataset = datasets.MNIST(root='./data', train=True, download=True)
    img, label = original_dataset[0]

    for i, ax in enumerate(axes.flat):
        # Apply augmentation
        augmented_img = augmented_transform(img)
        ax.imshow(augmented_img.squeeze(), cmap='gray')
        ax.axis('off')
        ax.set_title(f'Augmented {i+1}')

    plt.tight_layout()
    plt.savefig('task3_augmentation_examples.png')
    print("Sample visualizations saved as 'task3_augmentation_examples.png'")


def main():
    print("\n" + "="*70)
    print("TASK 3: DATA TRANSFORMATION TECHNIQUES")
    print("="*70)

    # Visualize augmented samples first
    visualize_augmented_samples()

    # Run all experiments
    transform_results = experiment_transformation_sequences()
    augment_results = experiment_data_augmentation()
    norm_results = experiment_normalization_methods()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)

    print("\n1. Transformation Sequence Results:")
    for name, result in transform_results.items():
        print(f"   {name}: Final Accuracy = {result['final_accuracy']:.2f}%")

    print("\n2. Data Augmentation Results:")
    for name, result in augment_results.items():
        print(f"   {name}: Final Accuracy = {result['final_accuracy']:.2f}%")

    print("\n3. Normalization Method Results:")
    for name, result in norm_results.items():
        print(f"   {name}: Final Accuracy = {result['final_accuracy']:.2f}%")

    print("\n" + "="*70)
    print("All experiments completed! Check the generated PNG files for visualizations.")
    print("="*70)


if __name__ == "__main__":
    main()

