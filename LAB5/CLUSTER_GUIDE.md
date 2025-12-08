# Guide d'utilisation du cluster ensicompute pour LAB5

## 📋 Prérequis

1. **Compte Ensimag** : Vous devez avoir un compte informatique Ensimag
2. **Connexion VPN** : Connectez-vous au VPN Ensimag ou grenet.fr (ou utilisez une salle TP)
3. **Environnement préparé** : Assurez-vous que votre environnement virtuel est configuré

## 🚀 Démarrage rapide

### 1. Connexion au cluster

```bash
# Depuis votre machine locale (avec VPN actif)
ssh votre_login@nash.ensimag.fr

# OU depuis une machine de salle TP avec forwarding de clés
ssh -K votre_login@nash.ensimag.fr
```

### 2. Navigation vers votre projet

```bash
cd ~/Bureau/FIB/cours/AVPR/Labs/LAB5
```

### 3. Configuration de l'environnement (première fois uniquement)

```bash
# Créer et activer l'environnement virtuel
python3 -m venv lab5_env
source lab5_env/bin/activate

# Installer PyTorch avec support CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Installer les autres dépendances
pip install -r requirements.txt
```

### 4. Lancer un job

```bash
# Exécuter toutes les tâches (recommandé)
./run_cluster.sh

# Exécuter une tâche spécifique
./run_cluster.sh --task=1
./run_cluster.sh --task=2
./run_cluster.sh --task=3

# Avec configuration personnalisée
./run_cluster.sh --task=all --mem=16GB --cpus=12 --time=8:00:00

# Utiliser une partition spécifique
./run_cluster.sh --task=1 --partition=a40    # GPU A40 (plus puissant)
./run_cluster.sh --task=2 --partition=v100   # GPU Tesla V100
```

## 📊 Options disponibles

| Option | Description | Valeur par défaut |
|--------|-------------|-------------------|
| `--task=N` | Tâche à exécuter (1, 2, 3, ou all) | `all` |
| `--mem=SIZE` | Mémoire RAM allouée | `8GB` |
| `--cpus=N` | Nombre de CPUs | `8` |
| `--time=TIME` | Limite de temps (HH:MM:SS) | `4:00:00` |
| `--partition=P` | Partition GPU (rtx6000, v100, a40) | `rtx6000` |

## 🔍 Surveillance des jobs

### Vérifier l'état de vos jobs

```bash
# Voir tous vos jobs
squeue -u $USER

# Voir un job spécifique
squeue -j <JOB_ID>

# Voir tous les jobs du cluster
squeue
```

### Suivre les logs en temps réel

```bash
# Suivre la sortie standard
tail -f cluster_logs/output/lab5_training_task*_*.out

# Suivre les erreurs
tail -f cluster_logs/errors/lab5_training_task*_*.err
```

### Annuler un job

```bash
# Annuler un job spécifique
scancel <JOB_ID>

# Annuler tous vos jobs
scancel -u $USER
```

## 📂 Structure des logs

Après l'exécution, les logs sont organisés comme suit :

```
cluster_logs/
├── output/          # Sorties standard (.out)
├── errors/          # Erreurs (.err)
├── checkpoints/     # Checkpoints de modèles (si applicable)
└── slurm_job_*.sh   # Scripts SLURM générés
```

## 🖥️ Informations sur les GPU disponibles

### RTX 6000 (Quadro) - Partition par défaut
- **Nœuds** : turing-1 à turing-11 (33 GPUs au total)
- **VRAM** : 24GB par GPU
- **Bon pour** : Entraînement standard, charge modérée

### Tesla V100
- **Nœuds** : tesla (1 GPU)
- **VRAM** : 32GB
- **Bon pour** : Modèles nécessitant plus de mémoire

### NVIDIA A40
- **Nœuds** : ampere (3 GPUs)
- **VRAM** : 46GB par GPU
- **Bon pour** : Modèles très larges, batch size élevé

## 💡 Conseils d'utilisation

### 1. Ressources appropriées

Pour LAB5 (MNIST), les valeurs par défaut sont suffisantes :
```bash
./run_cluster.sh --mem=8GB --cpus=8 --time=4:00:00
```

Si vous avez des timeouts ou des erreurs de mémoire :
```bash
./run_cluster.sh --mem=16GB --cpus=12 --time=8:00:00
```

### 2. Tester d'abord localement

Avant de lancer sur le cluster, testez rapidement en local :
```bash
python3 lab5_runner.py --task 1  # Test rapide d'une tâche
```

### 3. Exécution détachée

Le script utilise `sbatch`, donc votre job continue même si vous vous déconnectez. Vous pouvez :
- Fermer votre terminal
- Vous déconnecter du VPN
- Revenir plus tard pour vérifier les résultats

### 4. Optimisation des ressources

```bash
# Pour des expérimentations rapides
./run_cluster.sh --task=1 --time=1:00:00 --mem=4GB --cpus=4

# Pour des entraînements longs avec beaucoup de variations
./run_cluster.sh --task=all --time=12:00:00 --mem=16GB --cpus=12
```

## 🐛 Résolution de problèmes

### Job en attente (PD - Pending)
```bash
# Vérifier pourquoi
squeue -j <JOB_ID> -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"

# Raisons courantes :
# - Resources : Pas assez de ressources disponibles, attendez
# - Priority : D'autres jobs ont la priorité
```

### Job échoue immédiatement
```bash
# Vérifier les logs d'erreur
cat cluster_logs/errors/lab5_training_task*_*.err

# Problèmes courants :
# - Environnement virtuel non activé
# - CUDA non disponible
# - Fichiers de données manquants
```

### Erreurs CUDA
```bash
# Vérifier que PyTorch détecte CUDA
python3 -c "import torch; print(torch.cuda.is_available())"

# Si False, réinstaller PyTorch avec CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Mémoire insuffisante
```bash
# Augmenter la mémoire allouée
./run_cluster.sh --mem=16GB

# Ou réduire le batch size dans votre code
```

## 📧 Support

En cas de problème avec le cluster :
- Email : support.info@ensimag.fr
- Documentation : https://ensicompute.ensimag.fr (si disponible)

## 📝 Exemple de workflow complet

```bash
# 1. Connexion
ssh votre_login@nash.ensimag.fr

# 2. Navigation
cd ~/Bureau/FIB/cours/AVPR/Labs/LAB5

# 3. Vérification de l'environnement (première fois)
source lab5_env/bin/activate
python3 -c "import torch; print(torch.cuda.is_available())"

# 4. Lancement du job
./run_cluster.sh --task=all

# 5. Note du Job ID affiché
# Job ID: 12345

# 6. Surveillance
squeue -j 12345
tail -f cluster_logs/output/lab5_training_task*_*.out

# 7. Déconnexion (le job continue)
exit

# 8. Reconnexion plus tard
ssh votre_login@nash.ensimag.fr
cd ~/Bureau/FIB/cours/AVPR/Labs/LAB5

# 9. Vérification des résultats
ls -lh *.png *.pth  # Modèles et graphiques générés
cat cluster_logs/output/lab5_training_task*_*.out | grep -i "accuracy\|loss"
```

## ✅ Checklist avant soumission

- [ ] Environnement virtuel créé et dépendances installées
- [ ] Code testé localement (au moins un petit test)
- [ ] Script run_cluster.sh exécutable (`chmod +x run_cluster.sh`)
- [ ] Connecté à nash.ensimag.fr
- [ ] Répertoire de travail correct
- [ ] Logs et checkpoints précédents sauvegardés si nécessaire

Bon entraînement ! 🚀

