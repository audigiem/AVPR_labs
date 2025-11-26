# 🎯 Bag of Words (BoW) pour la Reconnaissance d'Images - LAB4

## 📋 Vue d'Ensemble

Ce projet implémente un système de reconnaissance d'images basé sur le modèle **Bag of Words (BoW)** avec des descripteurs SIFT et un classificateur SVM. Le code a été optimisé pour corriger les problèmes de performance et d'overfitting.

## ✅ Améliorations Apportées

### 1. **Performance (10-50x plus rapide)**
- ✅ Vectorisation de `extractFeatures()` 
- ✅ Opérations batch au lieu de boucles imbriquées
- ✅ Utilisation de `np.bincount()` pour les histogrammes

### 2. **Sauvegarde Automatique**
- ✅ Toutes les figures sont sauvegardées (pas affichées)
- ✅ Organisation en dossiers par tâche
- ✅ Noms de fichiers descriptifs

### 3. **Détection d'Overfitting**
- ✅ Comparaison Train CV vs Test
- ✅ Rapport détaillé par classe
- ✅ Matrices de confusion pour chaque configuration

### 4. **Gestion d'Erreurs**
- ✅ Vérification des images None
- ✅ Filtrage des extensions de fichiers
- ✅ Seed fixe pour la reproductibilité

## 🚀 Utilisation

### Installation
```bash
# Installer les dépendances
pip install -r requirements.txt
```

### Exécution

#### Option 1: Code Principal (Recommandé)
```bash
# Exécuter les deux tâches
python BOW_for_image_recognition.py --task both

# Ou juste Task 1 (exploration des paramètres)
python BOW_for_image_recognition.py --task task1

# Ou juste Task 2 (impact de l'augmentation)
python BOW_for_image_recognition.py --task task2
```

#### Option 2: Version Avancée
```bash
# Tester plusieurs configurations avec RBF kernel
python BOW_improved.py
```

## 📊 Résultats Attendus

### Task 1: Exploration des Paramètres

| Configuration | Précision | Commentaire |
|--------------|-----------|-------------|
| **Baseline (50 clusters)** | **62.4%** | ✅ **OPTIMAL** |
| More Clusters (150) | 61.0% | Légère baisse |
| Fewer Clusters (25) | 47.6% | ❌ Insuffisant |
| More Octave Layers (5) | 60.5% | Légère baisse |
| Lower Contrast (0.02) | 58.6% | Baisse modérée |

**Conclusion:** 50 clusters avec paramètres par défaut est optimal pour ce dataset.

### Task 2: Impact de l'Augmentation

| Configuration | Précision | Amélioration |
|--------------|-----------|--------------|
| Sans augmentation | 61.0% | - |
| Avec augmentation | 61.4% | +0.4% seulement |

**Conclusion:** L'augmentation n'apporte presque rien sur ce dataset (+0.4% seulement).

### BOW_improved.py: Détection d'Overfitting

| Configuration | Train CV | Test | Écart |
|--------------|----------|------|-------|
| Baseline (50 clusters) | 61.7% | 61.0% | -0.7% ✅ |
| More clusters (150) | 68.4% | 60.0% | -8.4% ⚠️ |
| With augmentation (50) | **86.9%** | 55.7% | **-31.2%** ❌ |

**Conclusion:** L'augmentation avec RBF kernel cause un **overfitting sévère**.

## 📂 Structure des Résultats

Après exécution, les résultats sont organisés ainsi:

```
results/
├── ANALYSIS_REPORT.md                    # Rapport détaillé
│
├── task1_parameter_exploration/
│   ├── parameter_comparison.png          # Comparaison graphique
│   └── confusion_matrix_50clusters.png   # Matrices pour chaque config
│
├── task2_augmentation/
│   ├── augmentation_comparison.png       # Avec vs sans
│   └── confusion_matrix_with_augmentation.png
│
└── improved_models/
    ├── model_comparison.png              # Toutes les configs
    └── confusion_matrix_*.png            # Une par configuration
```

## 🎯 Configuration Optimale

```python
# MEILLEURS PARAMÈTRES
n_clusters = 50
kernel = 'linear'
nOctaveLayers = 3
contrastThreshold = 0.04
augment = False  # PAS d'augmentation!
C = 0.1
class_weight = 'balanced'
```

**Précision: 62.4%**

## 📈 Problèmes par Classe

| Classe | F1-Score | Diagnostic |
|--------|----------|------------|
| face | 0.79 | ✅ Très bien |
| sea | 0.78 | ✅ Très bien |
| house_building | 0.64 | ✅ OK |
| city | 0.62 | ✅ OK |
| office | 0.58 | ⚠️ Beaucoup de faux positifs |
| green | 0.52 | ⚠️ Features peu distinctives |
| **house_indoor** | **0.27** | ❌ **TRÈS MAUVAIS** (42 images seulement) |

## 🔍 Analyse des Problèmes

### Pourquoi l'augmentation ne fonctionne pas ?

1. **Dataset trop petit** (807 images train)
   - L'augmentation crée 4000-7000 images artificielles
   - Le modèle "mémorise" au lieu de généraliser

2. **SVM RBF trop flexible**
   - Avec C=100 et gamma='scale', le modèle surfit
   - Train accuracy: 86-91% vs Test: 55-60%

3. **Augmentations non-réalistes**
   - Rotations de 90° changent l'orientation
   - Le modèle apprend des patterns artificiels

### Classes Problématiques

1. **house_indoor (F1: 0.27)**
   - Seulement 42 images d'entraînement
   - Confondu avec "office" (features similaires)
   - **Solution:** Collecter plus de données

2. **green (F1: 0.52)**
   - Features SIFT peu distinctives (textures naturelles)
   - **Solution:** Ajouter features de couleur

3. **office (precision: 0.48)**
   - Prédit trop souvent (recall: 0.73)
   - **Solution:** Ajuster les poids de classe

## 💡 Recommandations

### ✅ À Faire

1. **Utiliser la configuration baseline** (50 clusters, linear)
   - Meilleure précision: 62.4%
   - Pas d'overfitting
   - Rapide à entraîner

2. **Collecter plus de données** pour house_indoor
   - Minimum 100 images par classe
   - Données réelles > Augmentation artificielle

3. **Essayer Spatial Pyramid Matching**
   - Diviser l'image en grilles
   - Capturer l'information spatiale

### ❌ À Éviter

1. **Ne PAS utiliser l'augmentation** sur ce dataset
   - Cause de l'overfitting sévère
   - N'améliore pas la généralisation

2. **Ne PAS augmenter trop les clusters**
   - 150 clusters → overfitting léger
   - 50 est optimal pour ce dataset

3. **Ne PAS utiliser RBF kernel avec augmentation**
   - Trop de flexibilité = overfitting
   - Linear kernel plus robuste

## 🔧 Détails Techniques

### Vectorisation de extractFeatures()

**Avant (lent):**
```python
for i in range(image_count):
    for j in range(len(descriptor_list[i])):
        feature = descriptor_list[i][j].reshape(1, 128)
        idx = kmeans.predict(feature)  # Un à la fois!
        im_features[i][idx] += 1
```

**Après (rapide):**
```python
for i in range(image_count):
    if len(descriptor_list[i]) > 0:
        idx = kmeans.predict(descriptor_list[i])  # Tous en batch!
        hist = np.bincount(idx, minlength=no_clusters)
        im_features[i] = hist[:no_clusters]
```

**Gain: 10-50x plus rapide**

### Meilleure Recherche d'Hyperparamètres

**Avant:**
```python
Cs = [0.5, 0.1, 0.15, 0.2, 0.3]
gammas = [0.1, 0.11, 0.095, 0.105]
# → 20 combinaisons, plage étroite
```

**Après:**
```python
Cs = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
gammas = ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]  # Si RBF
# → 36 combinaisons, plage large
```

## 📚 Fichiers du Projet

- `BOW_for_image_recognition.py` - Code principal optimisé
- `BOW_improved.py` - Version avancée avec détection d'overfitting
- `results/ANALYSIS_REPORT.md` - Rapport détaillé d'analyse
- `requirements.txt` - Dépendances Python

## 🎓 Leçons Apprises

1. **Plus de données ≠ Toujours mieux**
   - L'augmentation artificielle peut nuire
   - Surveiller l'overfitting (Train vs Test)

2. **Simplicité parfois meilleure**
   - 50 clusters > 150 clusters
   - Linear kernel robuste
   - Pas d'augmentation nécessaire

3. **Importance des données équilibrées**
   - house_indoor: 42 images vs sea: 142 images
   - Explique les mauvaises performances

4. **Vectorisation cruciale**
   - 10-50x de gain de performance
   - Toujours privilégier les opérations batch

## 📞 Support

Pour plus de détails, voir:
- `results/ANALYSIS_REPORT.md` - Analyse complète des résultats
- `IMPROVEMENTS.md` - Documentation technique des améliorations

---

**Auteur:** Optimisé pour LAB4 AVPR
**Date:** 2025-11-26
**Meilleure Configuration:** Baseline (50 clusters, linear, no augmentation)
**Meilleure Précision:** 62.4%

