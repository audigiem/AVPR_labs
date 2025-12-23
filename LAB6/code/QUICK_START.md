# LAB6 - Guide de Démarrage Rapide

## 📋 Vue d'Ensemble

Ce projet contient une implémentation complète du LAB6 sur la détection d'objets avec Deep Learning et Transfer Learning.

## 🎯 Fichiers Créés

### 1. **LAB6_Complete.py** - Code Principal ⭐
Le fichier le plus important qui implémente **TOUS les 4 tasks** du lab :

#### **Task 1: Exploration des Hyperparamètres**
- Teste différents learning rates (0.00001, 0.0001, 0.001)
- Teste différents batch sizes (1, 2, 4)
- Teste différents nombres d'époques (2, 3, 5)
- Génère des checkpoints et un fichier JSON de résultats

#### **Task 2: Adaptation Architecturale et Transfer Learning**
- Configuration sans freezing (baseline)
- Configuration avec backbone gelé
- Configuration avec dégel graduel (layer4)
- Configuration avec MobileNetV3
- Compare les performances et temps d'entraînement

#### **Task 3: Transformation et Augmentation de Données**
- Transform basique (ToTensor seulement)
- Random Horizontal Flip
- Color Jitter
- Normalisation ImageNet
- Augmentation combinée
- Évalue l'impact sur la robustesse

#### **Task 4: Évaluation et Inférence**
- Teste différents seuils de confiance (0.3, 0.5, 0.7, 0.9)
- Teste différents seuils IoU pour NMS (0.3, 0.5, 0.7)
- Compare avec/sans NMS
- Génère des visualisations

### 2. **quick_demo.py** - Démos Rapides 🚀
Scripts de démonstration sans entraînement pour tester rapidement :

- **Demo 1** : Détection avec modèle pré-entraîné
- **Demo 2** : Comparaison Faster R-CNN vs RetinaNet
- **Demo 3** : Impact du NMS avec différents seuils
- **Demo 4** : Pipeline d'inférence personnalisé (classe ObjectDetector)

### 3. **README.md** - Documentation Complète 📖
Documentation détaillée avec :
- Description de chaque task
- Format des données
- Exemples d'utilisation
- Troubleshooting
- Résultats attendus

## 🚀 Comment Démarrer

### Étape 1: Installer les Dépendances

```bash
pip install torch torchvision pillow tqdm
```

### Étape 2: Préparer les Données

Créer la structure suivante :
```
code/
└── data/
    ├── images/      # Vos images .jpg
    └── labels/      # Vos labels .txt (format YOLO)
```

**Format des labels (YOLO)** - Chaque ligne dans le .txt :
```
<class_id> <x_center> <y_center> <width> <height>
```
(Toutes les valeurs normalisées entre 0 et 1)

### Étape 3: Tester Rapidement (Sans Entraînement)

```bash
python quick_demo.py
```

Choisissez une démo pour tester la détection d'objets avec des modèles pré-entraînés.

### Étape 4: Exécuter le Lab Complet

```bash
python LAB6_Complete.py
```

Menu interactif :
1. Task 1 seulement
2. Task 2 seulement
3. Task 3 seulement
4. Task 4 seulement
5. **TOUS les tasks** (recommandé pour le lab complet)

## 📊 Fichiers de Sortie

### Après Task 1 :
```
checkpoint_config1_baseline.pth
checkpoint_config2_high_lr.pth
checkpoint_config3_low_lr.pth
checkpoint_config4_batch2.pth
checkpoint_config5_more_epochs.pth
task1_results.json
```

### Après Task 2 :
```
model_no_freeze.pth
model_freeze_backbone.pth
model_gradual_unfreeze.pth
model_mobilenet.pth
task2_results.json
```

### Après Task 3 :
```
model_basic_transform.pth
model_horizontal_flip.pth
model_color_jitter.pth
model_normalized.pth
model_combined_augmentation.pth
task3_results.json
```

### Après Task 4 :
```
model_task4.pth
task4_output_conf0.5_iou0.5.jpg
task4_no_nms.jpg
task4_with_nms.jpg
```

## 💡 Exemples d'Utilisation Avancée

### Utiliser un Modèle Entraîné

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image

# Charger le modèle
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Charger les poids entraînés
model.load_state_dict(torch.load('model_freeze_backbone.pth'))
model.eval()

# Inférence
image = Image.open('test.jpg').convert('RGB')
img_tensor = transforms.ToTensor()(image).unsqueeze(0)

with torch.no_grad():
    predictions = model(img_tensor)[0]

print(f"Détections: {len(predictions['boxes'])}")
```

### Entraînement Personnalisé

```python
from LAB6_Complete import train_with_config

model, results = train_with_config(
    images_dir="data/images",
    labels_dir="data/labels",
    freeze_backbone=True,
    unfreeze_layer4=True,
    backbone_type="resnet50",
    lr=0.0002,
    epochs=5,
    config_name="my_custom_config"
)

print(f"Loss finale: {results['final_loss']}")
print(f"Temps: {results['training_time']}s")
```

## 🔍 Points Clés pour le Lab

### Task 1 - Ce qu'il faut analyser :
- Quel learning rate converge le mieux ?
- Impact du batch size sur la stabilité
- Trade-off entre nombre d'époques et overfitting

### Task 2 - Ce qu'il faut comparer :
- Temps d'entraînement avec/sans freezing
- Nombre de paramètres entraînables
- Performance finale (loss)
- ResNet-50 vs MobileNetV3 (précision vs vitesse)

### Task 3 - Ce qu'il faut observer :
- Impact de chaque augmentation sur la généralisation
- Robustesse aux variations (flip, color jitter)
- Effet de la normalisation sur la convergence

### Task 4 - Ce qu'il faut évaluer :
- Trade-off confiance vs nombre de détections
- Importance du NMS pour éliminer les duplicatas
- Choix du seuil IoU optimal

## 📈 Métriques à Analyser

### Dans les fichiers JSON :
- `epoch_losses` : Évolution de la loss par époque
- `final_loss` : Loss finale (plus bas = meilleur)
- `training_time` : Temps d'entraînement en secondes
- `trainable_params` : Nombre de paramètres modifiables

### Dans les visualisations :
- Nombre de détections
- Scores de confiance
- Qualité des bounding boxes
- Faux positifs / faux négatifs

## ⚠️ Troubleshooting Courant

### "No images found"
- Vérifiez que les images sont dans `data/images/`
- Vérifiez l'extension (.jpg, .jpeg, .png)

### "CUDA out of memory"
- Réduisez batch_size à 1
- Utilisez MobileNetV3 au lieu de ResNet-50
- Ou ajoutez au début du script :
  ```python
  device = torch.device('cpu')  # Forcer CPU
  ```

### Loss ne diminue pas
- Vérifiez le format des labels (YOLO normalisé)
- Essayez un learning rate plus petit (0.00001)
- Vérifiez que les bounding boxes sont valides (x1 < x2, y1 < y2)

### Labels vides
Le code gère automatiquement les images sans annotations (boxes vides).

## 🎓 Structure du Rapport Lab

Pour votre rapport, incluez :

1. **Introduction**
   - Contexte de la détection d'objets
   - Modèles utilisés (Faster R-CNN, etc.)

2. **Task 1 - Hyperparamètres**
   - Tableaux comparatifs
   - Graphiques de loss
   - Analyse de la convergence

3. **Task 2 - Transfer Learning**
   - Comparaison des stratégies
   - Impact du freezing
   - Choix du backbone

4. **Task 3 - Augmentation**
   - Impact de chaque technique
   - Robustesse améliorée
   - Recommandations

5. **Task 4 - Évaluation**
   - Visualisations
   - Analyse NMS
   - Choix des seuils

6. **Conclusion**
   - Meilleure configuration trouvée
   - Leçons apprises

## 📚 Ressources Additionnelles

- **PyTorch Tutorial**: https://pytorch.org/tutorials/
- **Faster R-CNN Paper**: https://arxiv.org/abs/1506.01497
- **COCO Dataset**: https://cocodataset.org/
- **Transfer Learning**: https://cs231n.github.io/transfer-learning/

## ✅ Checklist pour Compléter le Lab

- [ ] Installer les dépendances
- [ ] Préparer le dataset (images + labels)
- [ ] Tester quick_demo.py pour vérifier l'installation
- [ ] Exécuter Task 1 et analyser les résultats
- [ ] Exécuter Task 2 et comparer les modèles
- [ ] Exécuter Task 3 et évaluer les augmentations
- [ ] Exécuter Task 4 et générer les visualisations
- [ ] Analyser tous les fichiers JSON
- [ ] Sauvegarder les meilleures visualisations
- [ ] Rédiger le rapport avec les conclusions

## 🎯 Résumé des Commandes

```bash
# Test rapide sans entraînement
python quick_demo.py

# Lab complet interactif
python LAB6_Complete.py

# Exécuter un exemple spécifique
python Example1.py  # RetinaNet
python Example2.py  # Faster R-CNN avec NMS
python Example3.py  # Training basique
python Example4.py  # Avec freezing
```

## 📞 Support

Si vous avez des questions sur le code :
1. Consultez d'abord le README.md détaillé
2. Vérifiez les erreurs communes dans le troubleshooting
3. Regardez les exemples dans quick_demo.py

---

**Bon courage pour votre LAB6 ! 🚀**

Le code est complet, testé et prêt à l'emploi. Tous les tasks du lab sont implémentés dans `LAB6_Complete.py`.

