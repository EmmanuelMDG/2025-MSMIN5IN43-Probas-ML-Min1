"""
SCRIPT D'ENTRAÎNEMENT - DQN (Deep Q-Network)
=============================================

DQN est un algorithme de Q-Learning qui :
- Apprend une fonction Q(état, action) qui estime la récompense future
- Utilise un réseau de neurones profond pour approximer Q
- Choisit l'action avec la plus haute valeur Q (greedy)

Avantages de DQN :
✓ Pionnier du Deep RL (a battu des humains sur Atari)
✓ Simple conceptuellement
✓ Efficace en mémoire (replay buffer)
✓ Fonctionne bien sur espaces d'actions discrets

Comment ça marche :
1. Collecte des transitions (état, action, récompense, état suivant)
2. Stocke dans un replay buffer
3. Échantillonne des mini-batches du buffer
4. Met à jour Q pour minimiser l'erreur TD (Temporal Difference)
5. Utilise un réseau cible (target network) pour la stabilité

Limitations :
✗ Ne fonctionne QUE sur actions discrètes (parfait pour Snake!)
✗ Peut être instable sur certains environnements
"""

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
from snake_env import SnakeEnv


def make_env():
    """Crée et wrap l'environnement Snake."""
    env = SnakeEnv(grid_size=6, render_mode=None)
    env = Monitor(env)
    return env


def train_dqn():
    """Entraîne un agent DQN sur l'environnement Snake."""
    
    print("="*60)
    print("ENTRAÎNEMENT DQN - SNAKE RL")
    print("="*60)
    
    # Créer les dossiers
    os.makedirs("models/dqn", exist_ok=True)
    os.makedirs("logs/dqn", exist_ok=True)
    
    # Créer les environnements
    print("\n1. Création de l'environnement...")
    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])
    
    print("✓ Environnement créé")
    
    # Hyperparamètres DQN optimisés pour Snake
    print("\n2. Configuration de l'agent DQN...")
    model = DQN(
        policy="MlpPolicy",           # Réseau de neurones
        env=env,
        learning_rate=1e-4,           # Taux d'apprentissage (plus petit que PPO)
        buffer_size=100_000,          # Taille du replay buffer
        learning_starts=10_000,       # Steps avant de commencer l'apprentissage
        batch_size=32,                # Taille des mini-batches
        tau=1.0,                      # Soft update du target network (1.0 = hard update)
        gamma=0.99,                   # Facteur de discount
        train_freq=4,                 # Fréquence d'entraînement (tous les 4 steps)
        gradient_steps=1,             # Nombre de gradient steps par update
        target_update_interval=1000,  # Fréquence de mise à jour du target network
        exploration_fraction=0.3,     # Fraction du training pour exploration
        exploration_initial_eps=1.0,  # Epsilon initial (100% exploration au début)
        exploration_final_eps=0.05,   # Epsilon final (5% exploration à la fin)
        verbose=1,
        tensorboard_log="logs/dqn"
    )
    
    print("✓ Agent DQN configuré")
    print("\nHyperparamètres:")
    print(f"  - Learning rate: {model.learning_rate}")
    print(f"  - Buffer size: {model.buffer_size}")
    print(f"  - Batch size: {model.batch_size}")
    print(f"  - Gamma (discount): {model.gamma}")
    print(f"  - Exploration: {model.exploration_initial_eps} → {model.exploration_final_eps}")
    print(f"  - Target update interval: {model.target_update_interval}")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="models/dqn/checkpoints",
        name_prefix="snake_dqn"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/dqn",
        log_path="logs/dqn",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    
    # Entraînement
    print("\n3. Début de l'entraînement...")
    print("   (Cela peut prendre 15-30 minutes selon votre machine)")
    print("   Suivez la progression dans TensorBoard: tensorboard --logdir=logs/dqn")
    print("\n" + "-"*60)
    
    TOTAL_TIMESTEPS = 3_000_000
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
        
        # Sauvegarder le modèle final
        model.save("models/dqn/snake_dqn_final")
        print("\n" + "="*60)
        print("✓ ENTRAÎNEMENT TERMINÉ!")
        print("="*60)
        print(f"\nModèle sauvegardé dans: models/dqn/")
        print(f"Meilleur modèle: models/dqn/best_model.zip")
        print(f"Modèle final: models/dqn/snake_dqn_final.zip")
        print(f"\nPour visualiser l'entraînement:")
        print("  tensorboard --logdir=logs/dqn")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Entraînement interrompu par l'utilisateur")
        model.save("models/dqn/snake_dqn_interrupted")
        print("Modèle partiel sauvegardé: models/dqn/snake_dqn_interrupted.zip")
    
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    train_dqn()