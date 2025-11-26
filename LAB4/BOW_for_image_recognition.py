import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns


# ==================== HELPER FUNCTIONS ====================

def getFiles(train, path):
    """Collect image file paths from directories"""
    images = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                images.append(os.path.join(path, folder, file))
    if train:
        np.random.shuffle(images)
    return images


def readImage(img_path):
    """Read and resize image"""
    img = cv2.imread(img_path, 0)
    return cv2.resize(img, (150, 150))


def getDescriptors(sift, img):
    """Detect keypoints and compute descriptors using SIFT"""
    kp, des = sift.detectAndCompute(img, None)
    return des


def vstackDescriptors(descriptor_list):
    """Vertically stack descriptors"""
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor))
    return descriptors


def clusterDescriptors(descriptors, no_clusters):
    """Cluster descriptors using KMeans"""
    kmeans = KMeans(n_clusters=no_clusters, random_state=42).fit(descriptors)
    return kmeans


def extractFeatures(kmeans, descriptor_list, image_count, no_clusters):
    """Extract features based on clustering model"""
    im_features = np.array([np.zeros(no_clusters) for i in range(image_count)])
    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, 128)
            idx = kmeans.predict(feature)
            im_features[i][idx] += 1
    return im_features


def normalizeFeatures(scale, features):
    """Normalize features using StandardScaler"""
    return scale.transform(features)


def plotHistogram(im_features, no_clusters, title="Complete Vocabulary Generated"):
    """Plot histogram of visual words"""
    x_scalar = np.arange(no_clusters)
    y_scalar = np.array([abs(np.sum(im_features[:, h], dtype=np.int32)) for h in range(no_clusters)])
    plt.figure(figsize=(10, 6))
    plt.bar(x_scalar, y_scalar)
    plt.xlabel("Visual Word Index")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()


def svcParamSelection(X, y, kernel, nfolds):
    """Grid search for optimal SVM parameters"""
    Cs = [0.5, 0.1, 0.15, 0.2, 0.3]
    gammas = [0.1, 0.11, 0.095, 0.105]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(SVC(kernel=kernel), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    return grid_search.best_params_


def findSVM(im_features, train_labels, kernel, class_weight=None):
    """Train SVM classifier"""
    features = im_features
    if kernel == "precomputed":
        features = np.dot(im_features, im_features.T)

    params = svcParamSelection(features, train_labels, kernel, 5)
    C_param, gamma_param = params.get("C"), params.get("gamma")
    print(f"Best C: {C_param}, Best gamma: {gamma_param}")

    if class_weight is None:
        class_weight = 'balanced'

    svm = SVC(kernel=kernel, C=C_param, gamma=gamma_param, class_weight=class_weight)
    svm.fit(features, train_labels)
    return svm


def plotConfusionMatrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """Generate confusion matrix"""
    if not title:
        title = 'Normalized confusion matrix' if normalize else 'Confusion matrix'

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def findAccuracy(true, predictions):
    """Calculate accuracy"""
    acc = accuracy_score(true, predictions)
    print(f'Accuracy score: {acc:.3f}')
    return acc


# ==================== TASK 1: PARAMETER EXPLORATION ====================

def task1_explore_parameters(train_path, test_path):
    """
    Task 1: Explore impact of different parameters on BoW model
    """
    print("\n" + "=" * 60)
    print("TASK 1: PARAMETER EXPLORATION AND VISUALIZATION")
    print("=" * 60)

    # Different parameter configurations to test
    configs = [
        {"name": "Baseline", "clusters": 50, "nOctaveLayers": 3, "contrastThreshold": 0.04},
        {"name": "More Clusters", "clusters": 150, "nOctaveLayers": 3, "contrastThreshold": 0.04},
        {"name": "Fewer Clusters", "clusters": 25, "nOctaveLayers": 3, "contrastThreshold": 0.04},
        {"name": "More Octave Layers", "clusters": 50, "nOctaveLayers": 5, "contrastThreshold": 0.04},
        {"name": "Lower Contrast Threshold", "clusters": 50, "nOctaveLayers": 3, "contrastThreshold": 0.02},
    ]

    results = []

    for config in configs:
        print(f"\n--- Testing Configuration: {config['name']} ---")
        print(f"Clusters: {config['clusters']}, Octave Layers: {config['nOctaveLayers']}, "
              f"Contrast Threshold: {config['contrastThreshold']}")

        # Train with current configuration
        kmeans_model, scaler, svm_model, train_features = trainModelWithParams(
            train_path,
            config['clusters'],
            'linear',
            nOctaveLayers=config['nOctaveLayers'],
            contrastThreshold=config['contrastThreshold']
        )

        # Test with current configuration
        accuracy = testModelWithParams(
            test_path,
            kmeans_model,
            scaler,
            svm_model,
            train_features,
            config['clusters'],
            'linear',
            nOctaveLayers=config['nOctaveLayers'],
            contrastThreshold=config['contrastThreshold'],
            show_confusion=False
        )

        results.append({
            'config': config['name'],
            'accuracy': accuracy,
            'clusters': config['clusters']
        })

    # Visualize results
    visualize_parameter_results(results)

    return results


def trainModelWithParams(path, no_clusters, kernel, nOctaveLayers=3, contrastThreshold=0.04, custom_weights=None):
    """Train model with custom SIFT parameters"""
    images = getFiles(True, path)
    print(f"Found {len(images)} training images")

    # Create SIFT with custom parameters
    sift = cv2.SIFT_create(nOctaveLayers=nOctaveLayers, contrastThreshold=contrastThreshold)

    descriptor_list = []
    train_labels = np.array([])

    name_to_class = {
        "city": 0, "face": 1, "green": 2, "house_building": 3,
        "house_indoor": 4, "office": 5, "sea": 6
    }

    for img_path in images:
        # Determine class
        class_index = next((v for k, v in name_to_class.items() if k in img_path), 6)
        train_labels = np.append(train_labels, class_index)

        img = readImage(img_path)
        des = getDescriptors(sift, img)
        if des is not None:
            descriptor_list.append(des)

    print("Computing descriptors...")
    descriptors = vstackDescriptors(descriptor_list)

    print("Clustering descriptors...")
    kmeans = clusterDescriptors(descriptors, no_clusters)

    print("Extracting features...")
    im_features = extractFeatures(kmeans, descriptor_list, len(images), no_clusters)

    print("Normalizing features...")
    scale = StandardScaler().fit(im_features)
    im_features = scale.transform(im_features)

    print("Training SVM...")
    svm = findSVM(im_features, train_labels, kernel, class_weight=custom_weights)

    return kmeans, scale, svm, im_features


def testModelWithParams(path, kmeans, scale, svm, im_features, no_clusters, kernel,
                        nOctaveLayers=3, contrastThreshold=0.04, show_confusion=True):
    """Test model with custom parameters"""
    test_images = getFiles(False, path)
    print(f"Found {len(test_images)} test images")

    sift = cv2.SIFT_create(nOctaveLayers=nOctaveLayers, contrastThreshold=contrastThreshold)

    descriptor_list = []
    true_labels = []

    name_dict = {0: "city", 1: "face", 2: "green", 3: "house_building",
                 4: "house_indoor", 5: "office", 6: "sea"}

    for img_path in test_images:
        img = readImage(img_path)
        des = getDescriptors(sift, img)

        if des is not None:
            descriptor_list.append(des)
            # Determine true label
            true_label = next((k for k, v in name_dict.items() if v in img_path), "sea")
            true_labels.append(name_dict.get(true_label, "sea"))

    descriptors = vstackDescriptors(descriptor_list)
    test_features = extractFeatures(kmeans, descriptor_list, len(descriptor_list), no_clusters)
    test_features = scale.transform(test_features)

    kernel_test = test_features
    if kernel == "precomputed":
        kernel_test = np.dot(test_features, im_features.T)

    predictions = [name_dict[int(i)] for i in svm.predict(kernel_test)]

    if show_confusion:
        class_names = ["city", "face", "green", "house_building", "house_indoor", "office", "sea"]
        plotConfusionMatrix(true_labels, predictions, classes=class_names, normalize=True,
                            title='Normalized Confusion Matrix')

    accuracy = findAccuracy(true_labels, predictions)
    return accuracy


def visualize_parameter_results(results):
    """Visualize parameter exploration results"""
    configs = [r['config'] for r in results]
    accuracies = [r['accuracy'] for r in results]

    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(configs)))
    bars = plt.bar(configs, accuracies, color=colors)
    plt.xlabel('Configuration', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Parameter Exploration Results', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


# ==================== TASK 2: DATA AUGMENTATION ====================

def augment_image(img, augmentation_type):
    """Apply various augmentation techniques"""
    if augmentation_type == 'rotate_90':
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif augmentation_type == 'rotate_270':
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif augmentation_type == 'flip_horizontal':
        return cv2.flip(img, 1)
    elif augmentation_type == 'flip_vertical':
        return cv2.flip(img, 0)
    elif augmentation_type == 'scale_up':
        h, w = img.shape[:2]
        scaled = cv2.resize(img, (int(w * 1.2), int(h * 1.2)))
        start_h = (scaled.shape[0] - h) // 2
        start_w = (scaled.shape[1] - w) // 2
        return scaled[start_h:start_h + h, start_w:start_w + w]
    elif augmentation_type == 'scale_down':
        h, w = img.shape[:2]
        scaled = cv2.resize(img, (int(w * 0.8), int(h * 0.8)))
        padded = np.zeros((h, w), dtype=img.dtype)
        start_h = (h - scaled.shape[0]) // 2
        start_w = (w - scaled.shape[1]) // 2
        padded[start_h:start_h + scaled.shape[0], start_w:start_w + scaled.shape[1]] = scaled
        return padded
    elif augmentation_type == 'brightness':
        return cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    elif augmentation_type == 'noise':
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        return cv2.add(img, noise)
    else:
        return img


def task2_augmentation_robustness(train_path, test_path, no_clusters=50):
    """
    Task 2: Enhance model robustness with data augmentation
    """
    print("\n" + "=" * 60)
    print("TASK 2: MODEL ROBUSTNESS WITH AUGMENTATION")
    print("=" * 60)

    augmentation_types = ['rotate_90', 'flip_horizontal', 'scale_up', 'brightness']

    print("\n--- Training Model WITHOUT Augmentation ---")
    kmeans_base, scale_base, svm_base, features_base = trainModelBasic(
        train_path, no_clusters, 'linear', augment=False
    )

    print("\n--- Testing Model WITHOUT Augmentation ---")
    acc_base = testModelBasic(test_path, kmeans_base, scale_base, svm_base,
                              features_base, no_clusters, 'linear', show_confusion=False)

    print("\n--- Training Model WITH Augmentation ---")
    kmeans_aug, scale_aug, svm_aug, features_aug = trainModelBasic(
        train_path, no_clusters, 'linear', augment=True, aug_types=augmentation_types
    )

    print("\n--- Testing Model WITH Augmentation ---")
    acc_aug = testModelBasic(test_path, kmeans_aug, scale_aug, svm_aug,
                             features_aug, no_clusters, 'linear', show_confusion=True)

    # Compare results
    print("\n" + "=" * 60)
    print("AUGMENTATION IMPACT SUMMARY")
    print("=" * 60)
    print(f"Accuracy WITHOUT augmentation: {acc_base:.3f}")
    print(f"Accuracy WITH augmentation: {acc_aug:.3f}")
    print(f"Improvement: {(acc_aug - acc_base):.3f} ({((acc_aug - acc_base) / acc_base) * 100:.1f}%)")

    # Visualize comparison
    plt.figure(figsize=(10, 6))
    models = ['Without Augmentation', 'With Augmentation']
    accuracies = [acc_base, acc_aug]
    colors = ['#ff6b6b', '#4ecdc4']
    bars = plt.bar(models, accuracies, color=colors, width=0.6)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Model Performance Comparison: Augmentation Impact', fontsize=14, fontweight='bold')
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.show()

    return acc_base, acc_aug


def trainModelBasic(path, no_clusters, kernel, augment=False, aug_types=None):
    """Train model with optional augmentation"""
    images = getFiles(True, path)
    print(f"Found {len(images)} original training images")

    sift = cv2.SIFT_create()
    descriptor_list = []
    train_labels = np.array([])

    name_to_class = {
        "city": 0, "face": 1, "green": 2, "house_building": 3,
        "house_indoor": 4, "office": 5, "sea": 6
    }

    # Process original images
    for img_path in images:
        class_index = next((v for k, v in name_to_class.items() if k in img_path), 6)

        img = readImage(img_path)
        des = getDescriptors(sift, img)
        if des is not None:
            descriptor_list.append(des)
            train_labels = np.append(train_labels, class_index)

        # Add augmented versions
        if augment and aug_types:
            for aug_type in aug_types:
                aug_img = augment_image(img, aug_type)
                aug_des = getDescriptors(sift, aug_img)
                if aug_des is not None:
                    descriptor_list.append(aug_des)
                    train_labels = np.append(train_labels, class_index)

    print(f"Total training samples (with augmentation): {len(descriptor_list)}")

    descriptors = vstackDescriptors(descriptor_list)
    kmeans = clusterDescriptors(descriptors, no_clusters)
    im_features = extractFeatures(kmeans, descriptor_list, len(descriptor_list), no_clusters)

    scale = StandardScaler().fit(im_features)
    im_features = scale.transform(im_features)

    svm = findSVM(im_features, train_labels, kernel, class_weight='balanced')

    return kmeans, scale, svm, im_features


def testModelBasic(path, kmeans, scale, svm, im_features, no_clusters, kernel, show_confusion=True):
    """Test model"""
    test_images = getFiles(False, path)

    sift = cv2.SIFT_create()
    descriptor_list = []
    true_labels = []

    name_dict = {0: "city", 1: "face", 2: "green", 3: "house_building",
                 4: "house_indoor", 5: "office", 6: "sea"}

    for img_path in test_images:
        img = readImage(img_path)
        des = getDescriptors(sift, img)

        if des is not None:
            descriptor_list.append(des)
            true_label = next((name_dict[k] for k in name_dict if name_dict[k] in img_path), "sea")
            true_labels.append(true_label)

    if len(descriptor_list) == 0:
        print("No valid test images found!")
        return 0.0

    descriptors = vstackDescriptors(descriptor_list)
    test_features = extractFeatures(kmeans, descriptor_list, len(descriptor_list), no_clusters)
    test_features = scale.transform(test_features)

    predictions = [name_dict[int(i)] for i in svm.predict(test_features)]

    if show_confusion:
        class_names = ["city", "face", "green", "house_building", "house_indoor", "office", "sea"]
        plotConfusionMatrix(true_labels, predictions, classes=class_names, normalize=True)

    accuracy = findAccuracy(true_labels, predictions)
    return accuracy


# ==================== MAIN EXECUTION ====================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BoW Tasks: Parameter Exploration & Augmentation')
    parser.add_argument('--train_path', default='dataset/train', help='Path to training dataset')
    parser.add_argument('--test_path', default='dataset/test', help='Path to test dataset')
    parser.add_argument('--task', choices=['task1', 'task2', 'both'], default='both',
                        help='Which task to run')
    parser.add_argument('--clusters', type=int, default=50, help='Number of clusters for Task 2')

    args = parser.parse_args()

    if args.task in ['task1', 'both']:
        task1_results = task1_explore_parameters(args.train_path, args.test_path)

    if args.task in ['task2', 'both']:
        task2_results = task2_augmentation_robustness(args.train_path, args.test_path, args.clusters)

    print("\n" + "=" * 60)
    print("ALL TASKS COMPLETED!")
    print("=" * 60)