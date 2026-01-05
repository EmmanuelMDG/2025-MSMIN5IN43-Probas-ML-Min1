"""
SCRIPT D'ENTRAÎNEMENT - PPO (Proximal Policy Optimization)
============================================================

PPO est un algorithme de Policy Gradient qui :
- Apprend une politique (policy) qui mappe directement état → action
- Utilise une fonction de valeur pour estimer les récompenses futures
- Optimise la politique de manière "proximale" (petites mises à jour stables)

Avantages de PPO :
✓ Très stable et robuste
✓ Bon équilibre exploration/exploitation
✓ Fonctionne bien sur des espaces d'actions discrets et continus
✓ Généralement le meilleur choix par défaut

Comment ça marche :
1. Collecte des expériences en jouant avec la politique actuelle
2. Calcule les avantages (advantage) pour chaque action
3. Met à jour la politique en maximisant les avantages
4. Limite les mises à jour (clipping) pour rester "proche" de l'ancienne politique
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
from snake_env import SnakeEnv
import numpy as np


def make_env():
    """Crée et wrap l'environnement Snake."""
    env = SnakeEnv(grid_size=6, render_mode=None)
    env = Monitor(env)  # Pour logger les statistiques
    return env


def train_ppo():
    """Entraîne un agent PPO sur l'environnement Snake."""
    
    print("="*60)
    print("ENTRAÎNEMENT PPO - SNAKE RL")
    print("="*60)
    
    # Créer les dossiers pour sauvegarder
    os.makedirs("models/ppo", exist_ok=True)
    os.makedirs("logs/ppo", exist_ok=True)
    
    # Créer l'environnement d'entraînement
    print("\n1. Création de l'environnement...")
    env = DummyVecEnv([make_env])  # Vectoriser pour de meilleures performances
    
    # Créer un environnement d'évaluation séparé
    eval_env = DummyVecEnv([make_env])
    
    print("✓ Environnement créé")
    
    # Hyperparamètres PPO optimisés pour Snake
    print("\n2. Configuration de l'agent PPO...")
    model = PPO(
        policy="MlpPolicy",           # Réseau de neurones Multi-Layer Perceptron
        env=env,
        learning_rate=3e-4,           # Taux d'apprentissage (vitesse d'apprentissage)
        n_steps=2048,                 # Nombre de steps avant mise à jour
        batch_size=64,                # Taille des mini-batches pour l'optimisation
        n_epochs=10,                  # Nombre de passes sur les données collectées
        gamma=0.99,                   # Facteur de discount (importance du futur)
        gae_lambda=0.95,              # Lambda pour Generalized Advantage Estimation
        clip_range=0.2,               # Limite du clipping PPO (stabilité)
        ent_coef=0.01,                # Coefficient d'entropie (encourage exploration)
        vf_coef=0.5,                  # Coefficient de la value function
        max_grad_norm=0.5,            # Gradient clipping (stabilité)
        verbose=1,                    # Affichage des infos d'entraînement
        tensorboard_log="logs/ppo"    # Logs pour TensorBoard
    )
    
    print("✓ Agent PPO configuré")
    print("\nHyperparamètres:")
    print(f"  - Learning rate: {model.learning_rate}")
    print(f"  - N steps: {model.n_steps}")
    print(f"  - Batch size: {model.batch_size}")
    print(f"  - Gamma (discount): {model.gamma}")
    print(f"  - Clip range: {model.clip_range}")
    
    # Callbacks pour sauvegarder et évaluer
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,              # Sauvegarder tous les 10k steps
        save_path="models/ppo/checkpoints",
        name_prefix="snake_ppo"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/ppo",
        log_path="logs/ppo",
        eval_freq=5000,               # Évaluer tous les 5k steps
        deterministic=True,
        render=False,
        n_eval_episodes=10            # Nombre d'épisodes pour l'évaluation
    )
    
    # Entraînement
    print("\n3. Début de l'entraînement...")
    print("   (Cela peut prendre 15-30 minutes selon votre machine)")
    print("   Suivez la progression dans TensorBoard: tensorboard --logdir=logs/ppo")
    print("\n" + "-"*60)
    
    TOTAL_TIMESTEPS = 3_000_000  # Nombre total de steps d'entraînement
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
        
        # Sauvegarder le modèle final
        model.save("models/ppo/snake_ppo_final")
        print("\n" + "="*60)
        print("✓ ENTRAÎNEMENT TERMINÉ!")
        print("="*60)
        print(f"\nModèle sauvegardé dans: models/ppo/")
        print(f"Meilleur modèle: models/ppo/best_model.zip")
        print(f"Modèle final: models/ppo/snake_ppo_final.zip")
        print(f"\nPour visualiser l'entraînement:")
        print("  tensorboard --logdir=logs/ppo")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Entraînement interrompu par l'utilisateur")
        model.save("models/ppo/snake_ppo_interrupted")
        print("Modèle partiel sauvegardé: models/ppo/snake_ppo_interrupted.zip")
    
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    train_ppo()