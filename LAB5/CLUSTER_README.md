# 🚀 Quick Start - Exécution sur le cluster ensicompute

## 📦 Configuration initiale (une seule fois)

```bash
# 1. Connectez-vous à nash
ssh votre_login@nash.ensimag.fr

# 2. Naviguez vers votre projet
cd ~/Bureau/FIB/cours/AVPR/Labs/LAB5

# 3. Configurez l'environnement
./setup_cluster.sh
```

## 🎯 Lancer vos entraînements

```bash
# Exécuter toutes les tâches
./run_cluster.sh

# Exécuter une tâche spécifique
./run_cluster.sh --task=1
./run_cluster.sh --task=2
./run_cluster.sh --task=3

# Avec plus de ressources
./run_cluster.sh --task=all --mem=16GB --cpus=12 --time=8:00:00
```

## 📊 Surveiller vos jobs

```bash
# Voir le statut
./check_status.sh

# Suivre les logs en temps réel
tail -f cluster_logs/output/*.out

# Annuler un job
scancel <JOB_ID>
```

## 📚 Documentation complète

Voir **CLUSTER_GUIDE.md** pour des instructions détaillées.

## 🔧 Scripts disponibles

| Script | Description |
|--------|-------------|
| `setup_cluster.sh` | Configuration initiale de l'environnement |
| `run_cluster.sh` | Soumettre des jobs au cluster |
| `check_status.sh` | Vérifier l'état des jobs |

## 💡 Exemples d'utilisation

### Exemple 1 : Test rapide d'une tâche
```bash
./run_cluster.sh --task=1 --time=1:00:00 --mem=4GB
```

### Exemple 2 : Entraînement complet avec GPU puissant
```bash
./run_cluster.sh --task=all --partition=a40 --mem=16GB --time=12:00:00
```

### Exemple 3 : Vérification et suivi
```bash
# Soumettre le job
./run_cluster.sh --task=2

# Noter le Job ID (ex: 12345)

# Vérifier le statut
./check_status.sh 12345

# Suivre la progression
tail -f cluster_logs/output/lab5_training_task2_*.out
```

## 🖥️ Partitions GPU disponibles

- **rtx6000** (défaut) : 33 GPU Quadro RTX 6000, 24GB VRAM
- **v100** : 1 GPU Tesla V100, 32GB VRAM  
- **a40** : 3 GPU NVIDIA A40, 46GB VRAM

## ⚠️ Important

- Les jobs continuent même après déconnexion
- Pensez à récupérer vos résultats dans `cluster_logs/`
- Les modèles sauvegardés restent dans votre répertoire de travail

## 🆘 Besoin d'aide ?

Consultez **CLUSTER_GUIDE.md** pour :
- Guide détaillé d'utilisation
- Résolution de problèmes
- Exemples de workflows complets
- Optimisation des ressources

