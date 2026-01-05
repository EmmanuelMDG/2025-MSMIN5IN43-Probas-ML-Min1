import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional, Tuple


class SnakeEnv(gym.Env):
    """
    Environnement Snake personnalisé compatible Gymnasium.
    
    Observation Space:
        - Box de shape (grid_size, grid_size, 3) représentant:
          Channel 0: Position du serpent (1 pour le corps, 2 pour la tête)
          Channel 1: Position de la nourriture (1 si nourriture)
          Channel 2: Murs/limites (1 si mur)
    
    Action Space:
        - Discrete(4): 0=Haut, 1=Bas, 2=Gauche, 3=Droite
    
    Reward:
        - +10 + (longueur × 0.5) pour manger la nourriture
        - -10 pour collision (mur ou soi-même)
        - -0.01 pour chaque step (encourage l'efficacité)
        - Pénalité de faim croissante (quadratique avec le temps)
        - +0.3 × (amélioration distance) pour se rapprocher de la nourriture
        - -0.4 × (dégradation distance) pour s'éloigner de la nourriture
        - -0.3 × (nombre de boucles) pour détecter les boucles répétitives
        - +0.05 si beaucoup d'espace libre (évite de se coincer)
        - -0.2 si peu d'espace libre (pénalité pour se coincer)
        - +100 pour victoire (grille remplie)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}
    
    def __init__(self, grid_size: int = 15, render_mode: Optional[str] = None):
        super().__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        # Espaces d'observation et d'action
        self.observation_space = spaces.Box(
            low=0, high=2, 
            shape=(grid_size, grid_size, 3), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        
        # Directions: Haut, Bas, Gauche, Droite
        self.directions = [
            np.array([-1, 0]),  # Haut
            np.array([1, 0]),   # Bas
            np.array([0, -1]),  # Gauche
            np.array([0, 1])    # Droite
        ]
        
        # Variables de jeu
        self.snake = None
        self.snake_direction = None
        self.food = None
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0  # Compteur de steps depuis la dernière nourriture
        self.max_steps = grid_size * grid_size * 10  # Limite pour forcer l'efficacité
        self.recent_positions = []  # Pour détecter les boucles
        self.max_recent_positions = 12  # Augmenté pour mieux détecter les boucles longues
        
        # Pour le rendu
        self.window = None
        self.clock = None
        self.cell_size = 30
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Réinitialise l'environnement."""
        super().reset(seed=seed)
        
        # Initialiser le serpent au centre
        center = self.grid_size // 2
        self.snake = [
            np.array([center, center]),
            np.array([center, center + 1]),
            np.array([center, center + 2])
        ]
        self.snake_direction = 2  # Commence en allant à gauche
        
        # Placer la nourriture
        self._place_food()
        
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.recent_positions = []
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Exécute une action dans l'environnement."""
        self.steps += 1
        
        # Empêcher le demi-tour (ne peut pas aller dans la direction opposée)
        if self._is_opposite_direction(action, self.snake_direction):
            action = self.snake_direction
        
        self.snake_direction = action
        
        # Calculer la nouvelle position de la tête
        new_head = self.snake[0] + self.directions[action]
        
        # Calculer la distance à la nourriture avant le mouvement (distance de Manhattan)
        old_distance = np.sum(np.abs(self.snake[0] - self.food))
        
        # Vérifier les collisions
        terminated = False
        reward = -0.01  # Petite pénalité par step
        
        # Pénalité de faim qui augmente avec le temps sans manger (plus agressive)
        self.steps_since_food += 1
        # Pénalité qui augmente de façon quadratique, plus agressive
        hunger_penalty = -0.02 * (self.steps_since_food / 20) ** 1.8
        reward += hunger_penalty
        
        # Collision avec les murs
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            reward = -10
            terminated = True
        
        # Collision avec soi-même
        elif any(np.array_equal(new_head, segment) for segment in self.snake):
            reward = -10
            terminated = True
        
        # Manger la nourriture
        elif np.array_equal(new_head, self.food):
            reward = 10 + len(self.snake) * 0.5  # Bonus progressif
            self.score += 1
            self.steps_since_food = 0  # Réinitialiser le compteur de faim
            self.recent_positions = []  # Réinitialiser les positions récentes
            self.snake.insert(0, new_head)
            self._place_food()
            
            # Victoire si le serpent remplit toute la grille
            if len(self.snake) == self.grid_size * self.grid_size:
                reward = 100
                terminated = True
        
        # Déplacement normal
        else:
            self.snake.insert(0, new_head)
            self.snake.pop()
            
            # Détecter les boucles : vérifier si on revient à une position récente
            head_tuple = tuple(new_head)
            loop_penalty = 0
            if head_tuple in self.recent_positions:
                # Pénalité progressive plus forte selon le nombre de fois qu'on revient
                loop_count = self.recent_positions.count(head_tuple)
                loop_penalty = -0.5 * (loop_count + 1) ** 1.2  # Pénalité plus agressive et croissante
                reward += loop_penalty
                
                # Pénalité supplémentaire si on fait une boucle longue (plus de 4 positions)
                if len(self.recent_positions) >= 4:
                    # Vérifier si on fait un cycle (retour à une position récente)
                    recent_cycle = self.recent_positions[-4:]
                    if head_tuple in recent_cycle[:-1]:  # Pas la dernière (c'est la position actuelle)
                        reward -= 0.3  # Pénalité supplémentaire pour cycle court
            else:
                # Ajouter la position actuelle à l'historique
                self.recent_positions.append(head_tuple)
                if len(self.recent_positions) > self.max_recent_positions:
                    self.recent_positions.pop(0)
            
            # Récompenser si on se rapproche de la nourriture (distance de Manhattan)
            new_distance = np.sum(np.abs(new_head - self.food))
            distance_reward = 0
            distance_diff = old_distance - new_distance
            
            if distance_diff > 0:
                # Se rapprocher : récompense proportionnelle à l'amélioration
                distance_reward = 0.3 * (distance_diff / max(old_distance, 1))
            elif distance_diff < 0:
                # S'éloigner : pénalité proportionnelle
                distance_reward = -0.4 * (abs(distance_diff) / max(old_distance, 1))
            
            reward += distance_reward
            
            # Bonus pour éviter de se coincer (vérifier les cases adjacentes libres)
            free_adjacent = 0
            for direction in self.directions:
                adjacent_pos = new_head + direction
                # Vérifier si la case adjacente est libre (pas de mur, pas de corps)
                if (0 <= adjacent_pos[0] < self.grid_size and 
                    0 <= adjacent_pos[1] < self.grid_size and
                    not any(np.array_equal(adjacent_pos, segment) for segment in self.snake)):
                    free_adjacent += 1
            
            # Bonus si on a plusieurs cases libres (évite de se coincer)
            if free_adjacent >= 3:
                reward += 0.05  # Petit bonus pour avoir de l'espace
            elif free_adjacent <= 1:
                reward -= 0.2  # Pénalité si on se coince
        
        # Limite de steps (éviter boucles infinies)
        truncated = self.steps >= self.max_steps
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _is_opposite_direction(self, action: int, current_direction: int) -> bool:
        """Vérifie si l'action est opposée à la direction actuelle."""
        opposites = {0: 1, 1: 0, 2: 3, 3: 2}
        return action == opposites.get(current_direction, -1)
    
    def _place_food(self):
        """Place la nourriture à une position aléatoire libre."""
        while True:
            food = np.array([
                self.np_random.integers(0, self.grid_size),
                self.np_random.integers(0, self.grid_size)
            ])
            # Vérifier que la nourriture n'est pas sur le serpent
            if not any(np.array_equal(food, segment) for segment in self.snake):
                self.food = food
                break
    
    def _get_observation(self) -> np.ndarray:
        """Génère l'observation (état du jeu)."""
        obs = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
        
        # Channel 0: Serpent (1 pour corps, 2 pour tête)
        for i, segment in enumerate(self.snake):
            if 0 <= segment[0] < self.grid_size and 0 <= segment[1] < self.grid_size:
                obs[segment[0], segment[1], 0] = 2 if i == 0 else 1
        
        # Channel 1: Nourriture
        if 0 <= self.food[0] < self.grid_size and 0 <= self.food[1] < self.grid_size:
            obs[self.food[0], self.food[1], 1] = 1
        
        # Channel 2: Murs (bordures)
        obs[0, :, 2] = 1  # Haut
        obs[-1, :, 2] = 1  # Bas
        obs[:, 0, 2] = 1  # Gauche
        obs[:, -1, 2] = 1  # Droite
        
        return obs
    
    def _get_info(self) -> dict:
        """Retourne des informations supplémentaires."""
        return {
            'score': self.score,
            'snake_length': len(self.snake),
            'steps': self.steps,
            'food_position': self.food.tolist(),
            'head_position': self.snake[0].tolist()
        }
    
    def render(self):
        """Affiche le jeu."""
        if self.render_mode == 'rgb_array':
            return self._render_frame()
        elif self.render_mode == 'human':
            return self._render_frame()
    
    def _render_frame(self):
        """Dessine le frame actuel."""
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.grid_size * self.cell_size, self.grid_size * self.cell_size)
            )
            pygame.display.set_caption('Snake RL Environment')
        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
        canvas.fill((0, 0, 0))  # Fond noir
        
        # Dessiner la grille
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                rect = pygame.Rect(
                    j * self.cell_size,
                    i * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(canvas, (40, 40, 40), rect, 1)
        
        # Dessiner le serpent
        for i, segment in enumerate(self.snake):
            rect = pygame.Rect(
                segment[1] * self.cell_size,
                segment[0] * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            color = (0, 255, 0) if i == 0 else (0, 200, 0)  # Tête plus claire
            pygame.draw.rect(canvas, color, rect)
            pygame.draw.rect(canvas, (0, 150, 0), rect, 2)
        
        # Dessiner la nourriture
        food_rect = pygame.Rect(
            self.food[1] * self.cell_size,
            self.food[0] * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (self.food[1] * self.cell_size + self.cell_size // 2,
             self.food[0] * self.cell_size + self.cell_size // 2),
            self.cell_size // 3
        )
        
        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata['render_fps'])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Ferme l'environnement."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# Test de l'environnement
if __name__ == "__main__":
    # Créer l'environnement
    env = SnakeEnv(grid_size=10, render_mode='human')
    
    # Tester avec des actions aléatoires
    obs, info = env.reset()
    
    print("Environnement Snake initialisé!")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Info initiale: {info}")
    
    done = False
    total_reward = 0
    
    for step in range(500):
        env.render()
        
        # Action aléatoire
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        done = terminated or truncated
        
        if done:
            print(f"\nÉpisode terminé!")
            print(f"Score final: {info['score']}")
            print(f"Longueur du serpent: {info['snake_length']}")
            print(f"Récompense totale: {total_reward:.2f}")
            print(f"Steps: {info['steps']}")
            
            # Attendre un peu avant de fermer
            pygame.time.wait(2000)
            break
    
    env.close()