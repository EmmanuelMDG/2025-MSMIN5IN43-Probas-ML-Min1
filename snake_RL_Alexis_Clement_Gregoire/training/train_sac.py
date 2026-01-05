"""
SCRIPT D'ENTRAÎNEMENT - SAC (Soft Actor-Critic)
================================================

SAC est un algorithme Actor-Critic "soft" qui :
- Combine Policy Gradient (Actor) et Q-Learning (Critic)
- Maximise à la fois la récompense ET l'entropie (encourage exploration)
- Utilise deux réseaux Q pour réduire le biais de surestimation

Avantages de SAC :
✓ Très efficace en termes d'échantillons (sample-efficient)
✓ Robuste aux hyperparamètres
✓ Excellente exploration grâce à l'optimisation d'entropie
✓ État de l'art sur beaucoup de benchmarks

Comment ça marche :
1. Actor apprend une politique stochastique
2. Deux Critics (Q1 et Q2) estiment les valeurs Q
3. Optimise : récompense + entropie (diversité des actions)
4. Utilise le minimum des deux Q pour éviter la surestimation

Limitations :
✗ Conçu pour actions CONTINUES (on va l'adapter pour Snake)
✗ Plus complexe que PPO/DQN
✗ Peut être plus lent à converger sur actions discrètes

NOTE IMPORTANTE pour Snake :
SAC est normalement pour actions continues, mais Stable-Baselines3
n'a pas de version discrète officielle. On va quand même l'essayer
pour la comparaison, mais PPO et DQN sont mieux adaptés à Snake.
"""

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
from snake_env import SnakeEnv
import numpy as np


# Wrapper pour convertir l'action discrète en continue
class DiscreteToBoxWrapper(gym.Wrapper):
    """
    Convertit un espace d'actions discret en continu pour SAC.
    Snake a 4 actions discrètes, on les mappe sur [-1, 1].
    """
    def __init__(self, env):
        super().__init__(env)
        # Transformer Discrete(4) en Box(1,) continu entre -1 et 1
        self.action_space = gym.spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float32
        )
        self._discrete_action_space = env.action_space
    
    def step(self, action):
        # Convertir action continue [-1, 1] en action discrète [0, 3]
        # Gérer le cas où action est un array ou un float
        if isinstance(action, np.ndarray):
            action_value = float(action[0])
        else:
            action_value = float(action)
        
        # Map [-1, 1] → [0, 4)
        discrete_action = int((action_value + 1) * 2)
        # Assurer que c'est dans [0, 3]
        discrete_action = np.clip(discrete_action, 0, 3)
        
        return self.env.step(discrete_action)


def make_env():
    """Crée et wrap l'environnement Snake pour SAC."""
    env = SnakeEnv(grid_size=6, render_mode=None)
    env = DiscreteToBoxWrapper(env)  # Convertir pour SAC
    env = Monitor(env)
    return env


def train_sac():
    """Entraîne un agent SAC sur l'environnement Snake."""
    
    print("="*60)
    print("ENTRAÎNEMENT SAC - SNAKE RL")
    print("="*60)
    print("\n⚠️  NOTE: SAC est conçu pour actions continues.")
    print("   On l'adapte pour Snake, mais PPO/DQN sont plus adaptés.")
    
    # Créer les dossiers
    os.makedirs("models/sac", exist_ok=True)
    os.makedirs("logs/sac", exist_ok=True)
    
    # Créer les environnements
    print("\n1. Création de l'environnement...")
    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])
    
    print("✓ Environnement créé (avec wrapper Discrete→Continuous)")
    
    # Hyperparamètres SAC
    print("\n2. Configuration de l'agent SAC...")
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,           # Taux d'apprentissage
        buffer_size=50_000,           # Réduit de 100k → 50k (plus rapide)
        learning_starts=5_000,        # Réduit de 10k → 5k (commence plus tôt)
        batch_size=128,               # Réduit de 256 → 128 (2x plus rapide)
        tau=0.005,                    # Soft update coefficient (lissage du target)
        gamma=0.99,                   # Facteur de discount
        train_freq=4,                 # Changé de 1 → 4 (4x plus rapide !)
        gradient_steps=1,             # Gradient steps par update
        ent_coef="auto",              # Coefficient d'entropie (auto-ajusté)
        target_update_interval=1,     # Update le target à chaque step
        target_entropy="auto",        # Entropie cible auto-ajustée
        verbose=1,
        tensorboard_log="logs/sac"
    )
    
    print("✓ Agent SAC configuré")
    print("\nHyperparamètres:")
    print(f"  - Learning rate: {model.learning_rate}")
    print(f"  - Buffer size: {model.buffer_size}")
    print(f"  - Batch size: {model.batch_size}")
    print(f"  - Gamma (discount): {model.gamma}")
    print(f"  - Tau (soft update): {model.tau}")
    print(f"  - Entropy coefficient: {model.ent_coef}")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/sac/checkpoints",
        name_prefix="snake_sac"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/sac",
        log_path="logs/sac",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    # Entraînement
    print("\n3. Début de l'entraînement...")
    print("   (Cela peut prendre 15-30 minutes selon votre machine)")
    print("   Suivez la progression dans TensorBoard: tensorboard --logdir=logs/sac")
    print("\n" + "-"*60)
    
    TOTAL_TIMESTEPS = 3_000_000
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
        
        # Sauvegarder le modèle final
        model.save("models/sac/snake_sac_final")
        print("\n" + "="*60)
        print("✓ ENTRAÎNEMENT TERMINÉ!")
        print("="*60)
        print(f"\nModèle sauvegardé dans: models/sac/")
        print(f"Meilleur modèle: models/sac/best_model.zip")
        print(f"Modèle final: models/sac/snake_sac_final.zip")
        print(f"\nPour visualiser l'entraînement:")
        print("  tensorboard --logdir=logs/sac")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Entraînement interrompu par l'utilisateur")
        model.save("models/sac/snake_sac_interrupted")
        print("Modèle partiel sauvegardé: models/sac/snake_sac_interrupted.zip")
    
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    train_sac()