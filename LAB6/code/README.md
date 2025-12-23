# LAB6: Object Detection Using Deep Learning and Transfer Learning

## Description Complète du Projet

Ce projet implémente un système complet de détection d'objets en utilisant PyTorch et des techniques de Transfer Learning avec Faster R-CNN.

## Structure du Projet

```
code/
├── LAB6_Complete.py          # Code principal complet (TOUS LES TASKS)
├── Example1.py                # Exemple 1: Détection avec RetinaNet
├── Example2.py                # Exemple 2: Faster R-CNN avec NMS
├── Example3.py                # Exemple 3: Training de base
├── Example4.py                # Exemple 4: Freezing des couches
├── README.md                  # Ce fichier
├── family-and-dog.jpg         # Image de test
└── data/                      # Dossier pour vos données
    ├── images/                # Images d'entraînement (.jpg)
    └── labels/                # Labels YOLO format (.txt)
```

## Prérequis

```bash
pip install torch torchvision pillow tqdm
```

## Format des Données

### Structure des Dossiers
```
data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/
    ├── image1.txt
    ├── image2.txt
    └── ...
```

### Format des Labels (YOLO)
Chaque fichier `.txt` contient une ligne par objet :
```
<class_id> <x_center> <y_center> <width> <height>
```

Exemple :
```
0 0.5 0.5 0.3 0.4
```

Où toutes les valeurs sont normalisées entre 0 et 1.

## Utilisation du Code Complet (LAB6_Complete.py)

### Exécution Interactive

```bash
python LAB6_Complete.py
```

Le programme vous demandera de choisir :
1. **Task 1** : Exploration des Hyperparamètres
2. **Task 2** : Adaptation Architecturale et Transfer Learning
3. **Task 3** : Transformation et Augmentation de Données
4. **Task 4** : Évaluation et Inférence
5. **Run All Tasks** : Exécuter tous les tasks

### Task 1: Exploration des Hyperparamètres

**Objectif** : Tester différents hyperparamètres et observer leur impact

**Ce qui est testé :**
- **Learning Rates** : 0.00001, 0.0001, 0.001
- **Batch Sizes** : 1, 2, 4
- **Epochs** : 2, 3, 5

**Sortie :**
- `checkpoint_config1_baseline.pth` - Configuration de base
- `checkpoint_config2_high_lr.pth` - Taux d'apprentissage élevé
- `checkpoint_config3_low_lr.pth` - Taux d'apprentissage faible
- `checkpoint_config4_batch2.pth` - Batch size 2
- `checkpoint_config5_more_epochs.pth` - Plus d'époques
- `task1_results.json` - Résumé des résultats

**Résultats analysés :**
- Convergence de la loss
- Temps d'entraînement
- Stabilité de l'entraînement

### Task 2: Adaptation Architecturale et Transfer Learning

**Objectif** : Comparer différentes stratégies de transfer learning

**Configurations testées :**

1. **No Freezing (Baseline)** - Entraîne toutes les couches
2. **Freeze Backbone** - Gèle le ResNet-50, entraîne seulement ROI heads
3. **Gradual Unfreezing** - Gèle le backbone mais dégèle layer4
4. **MobileNetV3 Backbone** - Utilise MobileNetV3 au lieu de ResNet-50

**Sortie :**
- `model_no_freeze.pth`
- `model_freeze_backbone.pth`
- `model_gradual_unfreeze.pth`
- `model_mobilenet.pth`
- `task2_results.json`

**Métriques comparées :**
- Nombre de paramètres entraînables
- Temps d'entraînement
- Loss finale
- Vitesse de convergence

### Task 3: Transformation et Augmentation de Données

**Objectif** : Évaluer l'impact des augmentations sur la robustesse

**Configurations testées :**

1. **Basic Transform** - Seulement ToTensor()
2. **Horizontal Flip** - Avec retournement horizontal aléatoire
3. **Color Jitter** - Variations de luminosité, contraste, saturation
4. **Normalized** - Avec normalisation ImageNet
5. **Combined Augmentation** - Flip + Color Jitter

**Sortie :**
- `model_basic_transform.pth`
- `model_horizontal_flip.pth`
- `model_color_jitter.pth`
- `model_normalized.pth`
- `model_combined_augmentation.pth`
- `task3_results.json`

**Impact analysé :**
- Généralisation du modèle
- Robustesse aux variations
- Temps d'entraînement

### Task 4: Évaluation et Inférence

**Objectif** : Tester les prédictions avec différentes configurations

**Ce qui est testé :**

1. **Confidence Thresholds** : 0.3, 0.5, 0.7, 0.9
2. **NMS IoU Thresholds** : 0.3, 0.5, 0.7
3. **Comparaison avec/sans NMS**

**Sortie :**
- `model_task4.pth` - Modèle entraîné
- `task4_output_conf0.5_iou0.5.jpg` - Visualisation avec conf=0.5, iou=0.5
- `task4_no_nms.jpg` - Prédictions sans NMS
- `task4_with_nms.jpg` - Prédictions avec NMS

**Résultats visualisés :**
- Bounding boxes avec labels
- Scores de confiance
- Effet du NMS sur les détections

## Exemples d'Utilisation Avancée

### 1. Utiliser un Modèle Entraîné pour l'Inférence

```python
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image

# Charger le modèle
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Charger les poids entraînés
model.load_state_dict(torch.load('model_freeze_backbone.pth'))
model.eval()

# Prédire sur une nouvelle image
image = Image.open('test_image.jpg').convert('RGB')
image_tensor = transforms.ToTensor()(image).unsqueeze(0)

with torch.no_grad():
    predictions = model(image_tensor)[0]

print(f"Détections : {len(predictions['boxes'])}")
print(f"Scores : {predictions['scores']}")
```

### 2. Personnaliser l'Entraînement

```python
from LAB6_Complete import CustomDataset, train_with_config

# Entraîner avec vos propres paramètres
model, results = train_with_config(
    images_dir="data/images",
    labels_dir="data/labels",
    freeze_backbone=True,
    unfreeze_layer4=True,
    backbone_type="resnet50",
    lr=0.0002,
    epochs=5,
    config_name="custom_config"
)
```

### 3. Évaluer sur un Ensemble de Test

```python
from LAB6_Complete import task4_evaluation_inference

# Évaluer avec votre propre modèle
task4_evaluation_inference(
    images_dir="data/test_images",
    labels_dir="data/test_labels",
    model_path="model_freeze_backbone.pth",
    test_image_path="specific_test_image.jpg"
)
```

## Résultats Attendus

### Task 1 - Observations
- **Learning Rate élevé (0.001)** : Convergence rapide mais possiblement instable
- **Learning Rate faible (0.00001)** : Convergence lente mais stable
- **Batch Size plus grand** : Utilisation mémoire accrue, convergence plus stable
- **Plus d'époques** : Meilleure performance mais risque d'overfitting

### Task 2 - Observations
- **Freezing backbone** : Entraînement plus rapide, moins de paramètres
- **Gradual unfreezing** : Meilleur équilibre performance/temps
- **MobileNetV3** : Plus léger, plus rapide, légèrement moins précis

### Task 3 - Observations
- **Augmentation** : Améliore la généralisation
- **Color Jitter** : Robustesse aux variations d'éclairage
- **Normalisation** : Convergence plus stable

### Task 4 - Observations
- **Confidence threshold** : Trade-off précision/rappel
- **NMS** : Réduit les détections redondantes
- **IoU threshold** : Contrôle l'agressivité du NMS

## Fichiers de Sortie

### Checkpoints (.pth)
Contiennent les poids du modèle entraîné

### Résultats JSON
Contiennent les métriques détaillées :
- Losses par époque
- Temps d'entraînement
- Paramètres utilisés

### Images de Visualisation
Montrent les détections avec bounding boxes et labels

## Troubleshooting

### Problème : "No images found"
**Solution** : Vérifiez que vos images sont au format .jpg/.jpeg/.png dans `data/images/`

### Problème : "CUDA out of memory"
**Solution** : Réduisez le batch_size à 1 ou utilisez un backbone plus léger (MobileNetV3)

### Problème : "No module named 'torchvision'"
**Solution** : 
```bash
pip install torch torchvision
```

### Problème : Loss ne diminue pas
**Solution** : 
- Vérifiez le format de vos labels
- Essayez un learning rate différent
- Vérifiez que vos bounding boxes sont valides

## Notes Importantes

1. **Format des Labels** : Assurez-vous que vos labels sont au format YOLO (normalisés 0-1)
2. **GPU** : Le code détecte automatiquement CUDA, mais fonctionne aussi sur CPU
3. **Mémoire** : Pour des datasets volumineux, ajustez le batch_size
4. **Classes** : Le code est configuré pour 1 classe + background. Modifiez `num_classes` si nécessaire

## Ressources Supplémentaires

- **PyTorch Documentation** : https://pytorch.org/docs/
- **Torchvision Models** : https://pytorch.org/vision/stable/models.html
- **Faster R-CNN Paper** : https://arxiv.org/abs/1506.01497

## Auteur

LAB6 - Object Detection Using Deep Learning and Transfer Learning
Cours AVPR - FIB

## License

Ce code est fourni à des fins éducatives.

