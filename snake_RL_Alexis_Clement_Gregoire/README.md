# 🐍 Snake RL - Reinforcement Learning Comparison

Projet de comparaison d'algorithmes de Reinforcement Learning (PPO, DQN, SAC) appliqués au jeu Snake.

## 📋 Description

Ce projet implémente et compare trois algorithmes de Deep Reinforcement Learning pour apprendre à un agent à jouer au jeu Snake :

- **PPO (Proximal Policy Optimization)** - Algorithme on-policy stable et robuste
- **DQN (Deep Q-Network)** - Algorithme off-policy classique avec replay buffer
- **SAC (Soft Actor-Critic)** - Algorithme off-policy moderne avec optimisation d'entropie

## 🎯 Objectifs

- Implémenter un environnement Snake compatible avec Gymnasium
- Entraîner et comparer les performances de PPO, DQN et SAC
- Analyser les forces et faiblesses de chaque algorithme
- Visualiser les agents entraînés


### Setup

```bash
# Cloner le repository
git clone https://github.com/Unity1202/Snake_RL.git
cd snake_RL_Alexis_Clement_Gregoire

# Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## 🎮 Utilisation

### 1. Entraîner les agents

```bash
# Entraîner PPO (recommandé en premier)
cd training
python train_ppo.py

# Entraîner DQN
python train_dqn.py

# Entraîner SAC
python train_sac.py
```

**Durée d'entraînement :** ~1-2 heures par algorithme (3M steps)

### 2. Suivre l'entraînement avec TensorBoard

```bash
# Dans un nouveau terminal
tensorboard --logdir=logs
```

Ouvrir http://localhost:6006 dans votre navigateur

### 3. Visualiser les agents entraînés

```bash
# Mode interactif
python play_agent.py

# Ou avec arguments
python play_agent.py --algo ppo --episodes 5 --speed 2 --grid-size 6
python play_agent.py --algo ppo --checkpoint 100000 --grid-size 6

# Évaluation rapide
python play_agent.py --algo ppo --episodes 20 --eval-only

# Visualiser la progression
python play_agent.py --algo ppo --progression --progression-start 10000 --progression-end 100000 --progression-interval 10000
```

#### Options disponibles

| Option | Description | Exemple |
|--------|-------------|---------|
| `--algo` | Algorithme (ppo/dqn/sac) | `--algo ppo` |
| `--episodes` | Nombre d'épisodes | `--episodes 10` |
| `--grid-size` | Taille de la grille (défaut: 10) | `--grid-size 6` |
| `--speed` | Vitesse (1=normal, 0=max) | `--speed 5` |
| `--model` | Chemin vers le modèle | `--model training/models/ppo/best_model.zip` |
| `--checkpoint` | Nombre de steps du checkpoint à charger | `--checkpoint 100000` |
| `--list-checkpoints` | Lister tous les checkpoints disponibles | `--list-checkpoints` |
| `--no-render` | Pas de rendu visuel | `--no-render` |
| `--eval-only` | Stats uniquement | `--eval-only` |
| `--progression` | Visualiser la progression en testant les checkpoints | `--progression` |
| `--progression-start` | Checkpoint de départ pour --progression | `--progression-start 10000` |
| `--progression-end` | Checkpoint de fin pour --progression | `--progression-end 100000` |
| `--progression-interval` | Intervalle entre checkpoints | `--progression-interval 10000` |

## 🏗️ Architecture

### Environnement Snake

**Observation Space:**
- Grille 3D : `(grid_size, grid_size, 3)` (défaut: 6x6 pour entraînement, 10x10 pour visualisation)
  - Channel 0 : Position du serpent (1=corps, 2=tête)
  - Channel 1 : Position de la nourriture
  - Channel 2 : Murs/limites

**Action Space:**
- Discrete(4) : Haut, Bas, Gauche, Droite

**Système de récompenses:**
- +10 + (longueur × 0.5) : Manger la nourriture
- +0.3 × (amélioration distance) : Se rapprocher de la nourriture
- -0.4 × (dégradation distance) : S'éloigner de la nourriture
- -10 : Collision (mur ou auto-collision)
- -0.01 : Pénalité par step
- -0.3 × (nombre de boucles) : Détection de boucles répétitives
- +0.05 / -0.2 : Bonus/malus selon l'espace libre disponible
- +100 : Victoire (grille remplie)
- Pénalité de faim croissante (quadratique avec le temps)

### Hyperparamètres

#### PPO
```python
learning_rate = 3e-4
n_steps = 2048
batch_size = 64
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5
```

#### DQN
```python
learning_rate = 1e-4
buffer_size = 100,000
learning_starts = 10,000
batch_size = 32
tau = 1.0
gamma = 0.99
train_freq = 4
gradient_steps = 1
target_update_interval = 1000
exploration_fraction = 0.3
exploration_initial_eps = 1.0
exploration_final_eps = 0.05
```

#### SAC
```python
learning_rate = 3e-4
buffer_size = 50,000
learning_starts = 5,000
batch_size = 128
tau = 0.005
gamma = 0.99
train_freq = 4
gradient_steps = 1
ent_coef = "auto"
target_update_interval = 1
target_entropy = "auto"
```

**Note :** SAC est moins adapté aux actions discrètes comme Snake

## 🛠️ Technologies Utilisées

- **Gymnasium** - Framework d'environnements RL
- **Stable-Baselines3** - Implémentations des algorithmes RL
- **PyTorch** - Backend pour les réseaux de neurones
- **Pygame** - Rendu visuel du jeu
- **TensorBoard** - Visualisation de l'entraînement
- **NumPy** - Calculs numériques
- **tqdm** - Barres de progression
- **rich** - Affichage formaté dans le terminal

## 📈 Métriques de Comparaison

- **Score moyen** : Longueur du serpent atteinte
- **Temps d'entraînement** : Durée pour 500k steps
- **Stabilité** : Variance des performances
- **Sample efficiency** : Nombre d'expériences nécessaires
- **Comportement** : Qualité des stratégies apprises

## 🐛 Problèmes Connus

- **Boucles locales** : L'agent peut parfois tourner autour de la nourriture (pénalité de boucles implémentée)
- **SAC moins adapté** : Conçu pour actions continues, nécessite un wrapper pour actions discrètes → Optimisé avec `train_freq=4` et `batch_size=128`

## 🔬 Améliorations Futures

- [ ] Ajouter A2C et PPO avec LSTM
- [ ] Implémenter curriculum learning
- [ ] Tester sur grilles plus grandes (15x15, 20x20)
- [ ] Ajouter des obstacles
- [ ] Multi-agent Snake

## 📚 Références

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [DQN Paper](https://arxiv.org/abs/1312.5602)
- [SAC Paper](https://arxiv.org/abs/1801.01290)

