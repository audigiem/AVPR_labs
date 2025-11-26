"""
Improved BoW with better augmentation strategies and evaluation
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
from collections import Counter

# Create output directories
OUTPUT_DIR = "results"
IMPROVED_DIR = os.path.join(OUTPUT_DIR, "improved_models")
os.makedirs(IMPROVED_DIR, exist_ok=True)


# ==================== IMPROVED AUGMENTATION ====================

def advanced_augmentation(img):
    """
    Apply multiple augmentation techniques that preserve semantic content
    Returns a list of augmented images
    """
    augmented_images = []

    # 1. Rotation (small angles to preserve orientation)
    for angle in [15, -15]:
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        augmented_images.append(rotated)

    # 2. Horizontal flip (safe for most categories)
    augmented_images.append(cv2.flip(img, 1))

    # 3. Brightness variations
    augmented_images.append(cv2.convertScaleAbs(img, alpha=0.8, beta=-10))
    augmented_images.append(cv2.convertScaleAbs(img, alpha=1.2, beta=10))

    # 4. Gaussian blur (simulate different focus)
    augmented_images.append(cv2.GaussianBlur(img, (3, 3), 0))

    # 5. Add slight noise
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    augmented_images.append(noisy)

    # 6. Scale variations
    h, w = img.shape[:2]
    # Zoom in
    scaled_up = cv2.resize(img, (int(w * 1.1), int(h * 1.1)))
    start_h = (scaled_up.shape[0] - h) // 2
    start_w = (scaled_up.shape[1] - w) // 2
    augmented_images.append(scaled_up[start_h:start_h + h, start_w:start_w + w])

    return augmented_images


def selective_augmentation(img, category):
    """
    Apply category-specific augmentation
    Some categories shouldn't be flipped (e.g., text-like objects)
    """
    augmented_images = []

    # Categories where horizontal flip makes sense
    flip_safe = ['city', 'green', 'sea', 'house_building']

    # Rotation safe for most categories
    for angle in [10, -10]:
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        augmented_images.append(rotated)

    if any(cat in category for cat in flip_safe):
        augmented_images.append(cv2.flip(img, 1))

    # Brightness variations
    augmented_images.append(cv2.convertScaleAbs(img, alpha=1.15, beta=5))

    # Gaussian noise
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    augmented_images.append(noisy)

    return augmented_images


# ==================== IMPROVED DATA LOADING ====================

def load_and_augment_dataset(path, augment=False, selective=False):
    """
    Load dataset with improved augmentation strategy
    """
    name_to_class = {
        "city": 0, "face": 1, "green": 2, "house_building": 3,
        "house_indoor": 4, "office": 5, "sea": 6
    }

    images_data = []
    labels = []

    for folder in sorted(os.listdir(path)):
        folder_path = os.path.join(path, folder)
        if not os.path.isdir(folder_path):
            continue

        class_idx = name_to_class.get(folder, 6)

        for file in os.listdir(folder_path):
            if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue

            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, 0)

            if img is None:
                continue

            img = cv2.resize(img, (150, 150))
            img = cv2.equalizeHist(img)

            # Add original image
            images_data.append(img)
            labels.append(class_idx)

            # Add augmented versions
            if augment:
                if selective:
                    aug_imgs = selective_augmentation(img, folder)
                else:
                    aug_imgs = advanced_augmentation(img)

                for aug_img in aug_imgs:
                    images_data.append(aug_img)
                    labels.append(class_idx)

    return images_data, np.array(labels)


def extract_sift_features(images, nOctaveLayers=3, contrastThreshold=0.04):
    """
    Extract SIFT descriptors from images
    """
    sift = cv2.SIFT_create(nOctaveLayers=nOctaveLayers, contrastThreshold=contrastThreshold)

    descriptor_list = []
    valid_indices = []

    for idx, img in enumerate(images):
        kp, des = sift.detectAndCompute(img, None)
        if des is not None and len(des) > 0:
            descriptor_list.append(des)
            valid_indices.append(idx)

    return descriptor_list, valid_indices


def build_vocabulary(descriptor_list, n_clusters=100, use_minibatch=True):
    """
    Build visual vocabulary using KMeans clustering
    """
    # Stack all descriptors
    all_descriptors = np.vstack(descriptor_list)

    print(f"Total descriptors: {len(all_descriptors)}")

    # Use MiniBatchKMeans for large datasets (faster)
    if use_minibatch and len(all_descriptors) > 50000:
        print("Using MiniBatchKMeans for efficiency...")
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42,
                                 batch_size=1000, max_iter=100)
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    kmeans.fit(all_descriptors)

    return kmeans


def compute_bow_features(descriptor_list, kmeans, n_clusters):
    """
    Compute BoW histogram features
    """
    features = np.zeros((len(descriptor_list), n_clusters))

    for i, descriptors in enumerate(descriptor_list):
        if len(descriptors) > 0:
            # Predict cluster for each descriptor
            predictions = kmeans.predict(descriptors)
            # Create histogram
            hist = np.bincount(predictions, minlength=n_clusters)
            # Normalize histogram
            features[i] = hist[:n_clusters] / len(descriptors)

    return features


def train_improved_model(train_path, n_clusters=100, augment=False, selective=False):
    """
    Train improved BoW model
    """
    print(f"\n{'='*60}")
    print(f"Training with {n_clusters} clusters, augment={augment}, selective={selective}")
    print(f"{'='*60}")

    # Load data
    print("Loading and preprocessing images...")
    images, labels = load_and_augment_dataset(train_path, augment, selective)
    print(f"Total images: {len(images)}")
    print(f"Class distribution: {Counter(labels)}")

    # Extract SIFT features
    print("Extracting SIFT descriptors...")
    descriptor_list, valid_indices = extract_sift_features(images)
    labels = labels[valid_indices]
    print(f"Valid images with features: {len(descriptor_list)}")

    # Build vocabulary
    print("Building visual vocabulary...")
    kmeans = build_vocabulary(descriptor_list, n_clusters, use_minibatch=True)

    # Compute BoW features
    print("Computing BoW features...")
    features = compute_bow_features(descriptor_list, kmeans, n_clusters)

    # Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    # Train SVM with better parameters
    print("Training SVM with grid search...")
    param_grid = {
        'C': [0.1, 1.0, 10.0, 100.0],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf']
    }

    svm = GridSearchCV(
        SVC(class_weight='balanced'),
        param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=1
    )

    svm.fit(features_normalized, labels)

    print(f"Best parameters: {svm.best_params_}")
    print(f"Best CV score: {svm.best_score_:.3f}")

    return kmeans, scaler, svm


def test_improved_model(test_path, kmeans, scaler, svm, n_clusters, config_name="model"):
    """
    Test improved model
    """
    print("\nTesting model...")

    # Load test data (no augmentation)
    images, labels = load_and_augment_dataset(test_path, augment=False)
    print(f"Test images: {len(images)}")

    # Extract features
    descriptor_list, valid_indices = extract_sift_features(images)
    labels = labels[valid_indices]

    # Compute BoW features
    features = compute_bow_features(descriptor_list, kmeans, n_clusters)
    features_normalized = scaler.transform(features)

    # Predict
    predictions = svm.predict(features_normalized)

    # Evaluate
    accuracy = accuracy_score(labels, predictions)
    print(f"\nTest Accuracy: {accuracy:.3f}")

    # Classification report
    class_names = ["city", "face", "green", "house_building", "house_indoor", "office", "sea"]
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Normalized Confusion Matrix - {config_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    save_path = os.path.join(IMPROVED_DIR, f'confusion_matrix_{config_name.replace(" ", "_").replace("(", "").replace(")", "")}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  → Saved confusion matrix to {save_path}")
    plt.close()

    return accuracy


def compare_configurations(train_path, test_path):
    """
    Compare different configurations
    """
    configs = [
        {"name": "Baseline (50 clusters)", "n_clusters": 50, "augment": False, "selective": False},
        {"name": "More clusters (150)", "n_clusters": 150, "augment": False, "selective": False},
        {"name": "With augmentation (50 clusters)", "n_clusters": 50, "augment": True, "selective": False},
        {"name": "Selective augmentation (50 clusters)", "n_clusters": 50, "augment": True, "selective": True},
        {"name": "Best (150 clusters + selective aug)", "n_clusters": 150, "augment": True, "selective": True},
    ]

    results = []

    for config in configs:
        print(f"\n{'#'*60}")
        print(f"Testing: {config['name']}")
        print(f"{'#'*60}")

        try:
            kmeans, scaler, svm = train_improved_model(
                train_path,
                n_clusters=config['n_clusters'],
                augment=config['augment'],
                selective=config['selective']
            )

            accuracy = test_improved_model(
                test_path,
                kmeans,
                scaler,
                svm,
                n_clusters=config['n_clusters'],
                config_name=config['name']
            )

            results.append({
                'name': config['name'],
                'accuracy': accuracy
            })
        except Exception as e:
            print(f"Error in configuration {config['name']}: {e}")
            results.append({
                'name': config['name'],
                'accuracy': 0.0
            })

    # Plot comparison
    plt.figure(figsize=(12, 6))
    names = [r['name'] for r in results]
    accuracies = [r['accuracy'] for r in results]

    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    bars = plt.bar(range(len(names)), accuracies, color=colors)
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title('Model Configuration Comparison')
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    save_path = os.path.join(IMPROVED_DIR, 'model_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n→ Saved model comparison to {save_path}")
    plt.close()

    return results


if __name__ == '__main__':
    train_path = 'dataset/train'
    test_path = 'dataset/test'

    # Run comparison
    results = compare_configurations(train_path, test_path)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for r in results:
        print(f"{r['name']}: {r['accuracy']:.3f}")

