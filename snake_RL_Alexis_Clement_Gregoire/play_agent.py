"""
SCRIPT DE VISUALISATION - Agents RL Snake
==========================================

Ce script permet de :
1. Charger un agent entraîné (PPO, DQN ou SAC)
2. Le regarder jouer à Snake en temps réel
3. Afficher les statistiques de performance
"""

import gymnasium as gym
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import time
import os
import sys
sys.path.append('training')
from snake_env import SnakeEnv
import argparse


# Wrapper pour SAC (actions continues → discrètes)
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
        if isinstance(action, np.ndarray):
            action_value = float(action[0])
        else:
            action_value = float(action)
        
        discrete_action = int((action_value + 1) * 2)
        discrete_action = np.clip(discrete_action, 0, 3)
        
        return self.env.step(discrete_action)


def list_checkpoints(algo: str):
    """
    Liste tous les checkpoints disponibles pour un algorithme.
    
    Args:
        algo: Type d'algorithme ('ppo', 'dqn', 'sac')
    
    Returns:
        Liste des nombres de steps disponibles
    """
    checkpoint_dir = f"training/models/{algo}/checkpoints"
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    prefix = f"snake_{algo}_"
    suffix = "_steps.zip"
    
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith(prefix) and filename.endswith(suffix):
            try:
                steps_str = filename[len(prefix):-len(suffix)]
                steps = int(steps_str)
                checkpoints.append(steps)
            except ValueError:
                continue
    
    return sorted(checkpoints)


def load_agent(algo: str, model_path: str):
    """
    Charge un agent entraîné.
    
    Args:
        algo: Type d'algorithme ('ppo', 'dqn', 'sac')
        model_path: Chemin vers le modèle .zip
    
    Returns:
        Le modèle chargé
    """
    print(f"\n📦 Chargement du modèle {algo.upper()}...")
    print(f"   Chemin: {model_path}")
    
    if algo.lower() == 'ppo':
        model = PPO.load(model_path)
    elif algo.lower() == 'dqn':
        model = DQN.load(model_path)
    elif algo.lower() == 'sac':
        model = SAC.load(model_path)
    else:
        raise ValueError(f"Algorithme inconnu: {algo}. Utilisez 'ppo', 'dqn' ou 'sac'")
    
    print(f"✓ Modèle {algo.upper()} chargé avec succès!")
    return model


def play_episode(model, env, render=True, deterministic=True, speed=1.0):
    """
    Joue un épisode complet avec l'agent.
    
    Args:
        model: L'agent entraîné
        env: L'environnement
        render: Afficher le jeu visuellement
        deterministic: Utiliser la politique déterministe (sans exploration)
        speed: Vitesse de la simulation (1.0 = normal, 2.0 = 2x plus rapide, 0 = max)
    
    Returns:
        total_reward: Récompense totale de l'épisode
        score: Score final (longueur du serpent - 3)
        steps: Nombre de steps
    """
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    # Calculer le délai entre chaque frame
    if speed > 0:
        frame_delay = 0.1 / speed  # Plus speed est grand, plus c'est rapide
    else:
        frame_delay = 0  # Vitesse maximale
    
    while not done:
        if render:
            env.render()
            if frame_delay > 0:
                time.sleep(frame_delay)
        
        # Prédire l'action avec l'agent
        action, _states = model.predict(obs, deterministic=deterministic)
        
        # Exécuter l'action
        obs, reward, done, info = env.step(action)
        # Extraire la valeur scalaire si reward est un array
        if isinstance(reward, np.ndarray):
            reward = float(reward.item() if reward.size == 1 else reward[0])
        total_reward += reward
        steps += 1
    
    # Récupérer le score final
    score = info[0]['score'] if isinstance(info, list) else info['score']
    
    # Convertir total_reward en float si c'est un array numpy
    if isinstance(total_reward, np.ndarray):
        total_reward = float(total_reward.item() if total_reward.size == 1 else total_reward[0])
    
    return total_reward, score, steps


def evaluate_agent(model, env, n_episodes=10, speed=0):
    """
    Évalue un agent sur plusieurs épisodes.
    
    Args:
        model: L'agent entraîné
        env: L'environnement
        n_episodes: Nombre d'épisodes pour l'évaluation
        speed: Vitesse de simulation (0 = max speed pour évaluation)
    
    Returns:
        stats: Dictionnaire avec les statistiques
    """
    print(f"\n📊 Évaluation sur {n_episodes} épisodes...")
    
    rewards = []
    scores = []
    steps_list = []
    
    for episode in range(n_episodes):
        total_reward, score, steps = play_episode(
            model, env, render=False, deterministic=True, speed=speed
        )
        
        # S'assurer que total_reward est un float
        if isinstance(total_reward, np.ndarray):
            total_reward = float(total_reward.item() if total_reward.size == 1 else total_reward[0])
        
        rewards.append(total_reward)
        scores.append(score)
        steps_list.append(steps)
        
        print(f"  Épisode {episode+1}/{n_episodes}: Score={score}, Reward={total_reward:.2f}, Steps={steps}")
    
    stats = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'mean_steps': np.mean(steps_list),
        'max_score': np.max(scores),
        'min_score': np.min(scores)
    }
    
    return stats


def print_stats(stats, algo_name):
    """Affiche les statistiques de manière formatée."""
    print("\n" + "="*60)
    print(f"📈 STATISTIQUES - {algo_name.upper()}")
    print("="*60)
    print(f"Score moyen:        {stats['mean_score']:.2f} ± {stats['std_score']:.2f}")
    print(f"Score max:          {stats['max_score']}")
    print(f"Score min:          {stats['min_score']}")
    print(f"Récompense moyenne: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Steps moyen:        {stats['mean_steps']:.1f}")
    print("="*60)


def visualize_progression(algo: str, grid_size: int = 10, start: int = 10000, end: int = 100000, 
                         step_interval: int = 10000, render: bool = True, speed: float = 1.0):
    """
    Visualise la progression de l'entraînement en testant différents checkpoints.
    
    Args:
        algo: Type d'algorithme ('ppo', 'dqn', 'sac')
        grid_size: Taille de la grille
        start: Checkpoint de départ (défaut: 10000)
        end: Checkpoint de fin (défaut: 100000)
        step_interval: Intervalle entre les checkpoints (défaut: 10000)
        render: Afficher le rendu visuel
        speed: Vitesse de simulation
    """
    print("="*60)
    print(f"📊 VISUALISATION DE LA PROGRESSION - {algo.upper()}")
    print("="*60)
    
    # Récupérer tous les checkpoints disponibles
    all_checkpoints = list_checkpoints(algo)
    if not all_checkpoints:
        print(f"\n❌ Aucun checkpoint trouvé pour {algo}")
        return
    
    # Filtrer les checkpoints dans la plage demandée
    target_checkpoints = []
    for checkpoint in all_checkpoints:
        if start <= checkpoint <= end and checkpoint % step_interval == 0:
            target_checkpoints.append(checkpoint)
    
    if not target_checkpoints:
        print(f"\n❌ Aucun checkpoint trouvé entre {start} et {end} avec un intervalle de {step_interval}")
        print(f"   Checkpoints disponibles: {all_checkpoints[:10]}...")
        return
    
    print(f"\n🎯 Checkpoints à tester: {target_checkpoints}")
    print(f"   Total: {len(target_checkpoints)} checkpoints\n")
    
    # Créer l'environnement
    render_mode = 'human' if render else None
    env = SnakeEnv(grid_size=grid_size, render_mode=render_mode)
    
    # Pour SAC, appliquer le wrapper
    if algo.lower() == 'sac':
        env = DiscreteToBoxWrapper(env)
    
    env = DummyVecEnv([lambda: env])
    
    results = []
    
    try:
        for i, checkpoint in enumerate(target_checkpoints):
            checkpoint_path = f"training/models/{algo}/checkpoints/snake_{algo}_{checkpoint}_steps.zip"
            
            if not os.path.exists(checkpoint_path):
                print(f"⚠️  Checkpoint {checkpoint:,} steps non trouvé, ignoré")
                continue
            
            print(f"\n{'='*60}")
            print(f"📦 Checkpoint {i+1}/{len(target_checkpoints)}: {checkpoint:,} steps")
            print(f"{'='*60}")
            
            # Charger le modèle
            try:
                if algo.lower() == 'ppo':
                    model = PPO.load(checkpoint_path)
                elif algo.lower() == 'dqn':
                    model = DQN.load(checkpoint_path)
                elif algo.lower() == 'sac':
                    model = SAC.load(checkpoint_path)
                else:
                    print(f"❌ Algorithme inconnu: {algo}")
                    continue
            except Exception as e:
                print(f"❌ Erreur lors du chargement: {e}")
                continue
            
            # Jouer un épisode
            try:
                total_reward, score, steps = play_episode(
                    model, env, render=render, deterministic=True, speed=speed
                )
                
                results.append({
                    'checkpoint': checkpoint,
                    'score': score,
                    'reward': total_reward,
                    'steps': steps
                })
                
                print(f"\n✓ Résultat:")
                print(f"  Score: {score}")
                print(f"  Récompense: {total_reward:.2f}")
                print(f"  Steps: {steps}")
                
            except Exception as e:
                print(f"❌ Erreur lors de la partie: {e}")
                continue
            
            # Pause entre les checkpoints (sauf le dernier)
            if i < len(target_checkpoints) - 1 and render:
                time.sleep(1)
        
        # Afficher le résumé
        print("\n" + "="*60)
        print("📊 RÉSUMÉ DE LA PROGRESSION")
        print("="*60)
        print(f"{'Checkpoint':<12} {'Score':<8} {'Récompense':<12} {'Steps':<8}")
        print("-"*60)
        for result in results:
            print(f"{result['checkpoint']:>10,}  {result['score']:>6}  {result['reward']:>10.2f}  {result['steps']:>6}")
        
        if len(results) > 1:
            print("\n📈 Évolution:")
            first_score = results[0]['score']
            last_score = results[-1]['score']
            improvement = last_score - first_score
            print(f"  Score initial ({results[0]['checkpoint']:,} steps): {first_score}")
            print(f"  Score final ({results[-1]['checkpoint']:,} steps): {last_score}")
            print(f"  Amélioration: {improvement:+d} ({improvement/first_score*100:+.1f}%)" if first_score > 0 else f"  Amélioration: {improvement:+d}")
        
    finally:
        env.close()


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description='Visualiser un agent RL jouant à Snake')
    parser.add_argument('--algo', type=str, default='ppo', 
                        choices=['ppo', 'dqn', 'sac'],
                        help='Algorithme à visualiser (ppo, dqn, sac)')
    parser.add_argument('--model', type=str, default=None,
                        help='Chemin vers le modèle (par défaut: training/models/{algo}/best_model.zip)')
    parser.add_argument('--checkpoint', type=int, default=None,
                        help='Nombre de steps du checkpoint à charger (ex: 100000 pour snake_ppo_100000_steps.zip)')
    parser.add_argument('--list-checkpoints', action='store_true',
                        help='Lister tous les checkpoints disponibles')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Nombre d\'épisodes à jouer')
    parser.add_argument('--no-render', action='store_true',
                        help='Ne pas afficher le rendu visuel')
    parser.add_argument('--eval-only', action='store_true',
                        help='Évaluation uniquement (pas de rendu)')
    parser.add_argument('--grid-size', type=int, default=10,
                        help='Taille de la grille (défaut: 10)')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Vitesse de simulation (1.0=normal, 2.0=2x rapide, 5.0=5x rapide, 0=max)')
    parser.add_argument('--progression', action='store_true',
                        help='Visualiser la progression en testant les checkpoints de 10k en 10k')
    parser.add_argument('--progression-start', type=int, default=10000,
                        help='Checkpoint de départ pour --progression (défaut: 10000)')
    parser.add_argument('--progression-end', type=int, default=100000,
                        help='Checkpoint de fin pour --progression (défaut: 100000)')
    parser.add_argument('--progression-interval', type=int, default=10000,
                        help='Intervalle entre les checkpoints pour --progression (défaut: 10000)')
    
    args = parser.parse_args()
    
    # Mode progression
    if args.progression:
        visualize_progression(
            algo=args.algo,
            grid_size=args.grid_size,
            start=args.progression_start,
            end=args.progression_end,
            step_interval=args.progression_interval,
            render=not args.no_render,
            speed=args.speed
        )
        return
    
    print("="*60)
    print("🐍 VISUALISATION AGENT RL - SNAKE")
    print("="*60)
    
    # Lister les checkpoints si demandé
    if args.list_checkpoints:
        print(f"\n📋 Checkpoints disponibles pour {args.algo.upper()}:")
        checkpoints = list_checkpoints(args.algo)
        if checkpoints:
            print(f"\n  Total: {len(checkpoints)} checkpoints")
            print(f"  Premiers: {checkpoints[:10]}")
            if len(checkpoints) > 10:
                print(f"  ...")
            print(f"  Derniers: {checkpoints[-10:]}")
            print(f"\n  Utilisez --checkpoint <nombre> pour charger un checkpoint")
            print(f"  Exemple: --checkpoint {checkpoints[0]}")
        else:
            print(f"  Aucun checkpoint trouvé dans training/models/{args.algo}/checkpoints/")
        return
    
    # Déterminer le chemin du modèle
    if args.checkpoint is not None:
        # Charger un checkpoint spécifique
        checkpoint_path = f"training/models/{args.algo}/checkpoints/snake_{args.algo}_{args.checkpoint}_steps.zip"
        if not os.path.exists(checkpoint_path):
            print(f"\n❌ ERREUR: Le checkpoint n'existe pas: {checkpoint_path}")
            checkpoints = list_checkpoints(args.algo)
            if checkpoints:
                print(f"\nCheckpoints disponibles pour {args.algo}:")
                print(f"  {checkpoints[:20]}")  # Afficher les 20 premiers
                if len(checkpoints) > 20:
                    print(f"  ... et {len(checkpoints) - 20} autres")
            return
        model_path = checkpoint_path
    elif args.model is None:
        # Utiliser le meilleur modèle par défaut
        model_path = f"training/models/{args.algo}/best_model.zip"
    else:
        model_path = args.model
    
    # Vérifier que le modèle existe
    if not os.path.exists(model_path):
        print(f"\n❌ ERREUR: Le modèle n'existe pas: {model_path}")
        print(f"\nModèles disponibles:")
        for algo in ['ppo', 'dqn', 'sac']:
            path = f"training/models/{algo}/best_model.zip"
            if os.path.exists(path):
                print(f"  ✓ {path}")
            else:
                print(f"  ✗ {path}")
        print(f"\nUtilisez --list-checkpoints pour voir les checkpoints disponibles")
        return
    
    # Charger le modèle
    model = load_agent(args.algo, model_path)
    
    # Afficher le nombre de steps si c'est un checkpoint
    if args.checkpoint is not None:
        print(f"   Checkpoint: {args.checkpoint:,} steps d'entraînement")
    
    # Créer l'environnement
    render_mode = None if (args.no_render or args.eval_only) else 'human'
    env = SnakeEnv(grid_size=args.grid_size, render_mode=render_mode)
    
    # Pour SAC, appliquer le wrapper DiscreteToBox
    if args.algo.lower() == 'sac':
        env = DiscreteToBoxWrapper(env)
    
    env = DummyVecEnv([lambda: env])
    
    if args.eval_only:
        # Mode évaluation uniquement
        stats = evaluate_agent(model, env, n_episodes=args.episodes, speed=0)
        print_stats(stats, args.algo)
    else:
        # Mode visualisation + statistiques
        speed_desc = f"{args.speed}x" if args.speed > 0 else "MAX"
        print(f"\n🎮 Lancement de {args.episodes} épisode(s) avec rendu visuel (vitesse: {speed_desc})...")
        print("   (Appuyez sur Ctrl+C pour arrêter)\n")
        
        all_rewards = []
        all_scores = []
        all_steps = []
        
        try:
            for episode in range(args.episodes):
                print(f"\n--- Épisode {episode + 1}/{args.episodes} ---")
                
                total_reward, score, steps = play_episode(
                    model, env, render=not args.no_render, deterministic=True, speed=args.speed
                )
                
                all_rewards.append(total_reward)
                all_scores.append(score)
                all_steps.append(steps)
                
                print(f"✓ Épisode terminé!")
                print(f"  Score: {score}")
                print(f"  Récompense totale: {total_reward:.2f}")
                print(f"  Nombre de steps: {steps}")
                
                if episode < args.episodes - 1:
                    time.sleep(1)  # Pause entre les épisodes
            
            # Afficher les statistiques finales
            stats = {
                'mean_reward': np.mean(all_rewards),
                'std_reward': np.std(all_rewards),
                'mean_score': np.mean(all_scores),
                'std_score': np.std(all_scores),
                'mean_steps': np.mean(all_steps),
                'max_score': np.max(all_scores),
                'min_score': np.min(all_scores)
            }
            print_stats(stats, args.algo)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Visualisation interrompue par l'utilisateur")
    
    env.close()
    print("\n✓ Terminé!")


if __name__ == "__main__":
    # Si lancé sans arguments, mode interactif
    import sys
    if len(sys.argv) == 1:
        print("="*60)
        print("🐍 VISUALISATION AGENT RL - SNAKE")
        print("="*60)
        print("\nQuel agent voulez-vous visualiser?")
        print("1. PPO (Proximal Policy Optimization)")
        print("2. DQN (Deep Q-Network)")
        print("3. SAC (Soft Actor-Critic)")
        
        choice = input("\nChoisissez (1/2/3): ").strip()
        
        algo_map = {'1': 'ppo', '2': 'dqn', '3': 'sac'}
        algo = algo_map.get(choice, 'ppo')
        
        episodes = input("Nombre d'épisodes à jouer (défaut: 5): ").strip()
        episodes = int(episodes) if episodes else 5
        
        speed_input = input("Vitesse de simulation (1=normal, 2=2x rapide, 5=5x rapide, 0=max, défaut: 1): ").strip()
        speed = float(speed_input) if speed_input else 1.0
        
        # Lancer avec les paramètres choisis
        sys.argv = ['play_agent.py', '--algo', algo, '--episodes', str(episodes), '--speed', str(speed)]
    
    main()