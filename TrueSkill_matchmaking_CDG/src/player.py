"""
Classe Player : Représente un joueur avec sa vraie compétence et son rating TrueSkill
"""
import random
from trueskill import Rating


class Player:
    """
    Représente un joueur dans le système TrueSkill
    
    Attributes:
        name (str): Nom du joueur
        true_skill (float): Compétence réelle (cachée au système)
        rating (Rating): Rating TrueSkill actuel (mu, sigma)
        history_mu (list): Historique des valeurs de mu
        history_sigma (list): Historique des valeurs de sigma
        matches_played (int): Nombre de matchs joués
        wins (int): Nombre de victoires
        losses (int): Nombre de défaites
    """
    
    def __init__(self, name, true_skill, initial_mu=25.0, initial_sigma=8.333):
        """
        Initialise un joueur
        
        Args:
            name (str): Nom du joueur
            true_skill (float): Compétence réelle du joueur
            initial_mu (float): Mu initial (défaut:  25.0)
            initial_sigma (float): Sigma initial (défaut: 8.333)
        """
        self. name = name
        self.true_skill = true_skill
        self.rating = Rating(mu=initial_mu, sigma=initial_sigma)
        
        # Historique pour les visualisations
        self.history_mu = [self.rating.mu]
        self.history_sigma = [self.rating.sigma]
        
        # Statistiques
        self.matches_played = 0
        self.wins = 0
        self.losses = 0
    
    def play_match(self, beta=25/6):
        """
        Simule la performance du joueur dans un match
        
        La performance est tirée d'une distribution gaussienne centrée sur
        la vraie compétence avec une variance beta² (représente la chance/forme du jour)
        
        Args:
            beta (float): Écart-type de la performance (défaut: 25/6 ≈ 4.17)
        
        Returns:
            float: Performance du joueur pour ce match
        """
        return random.gauss(self.true_skill, beta)
    
    def update_rating(self, new_rating):
        """
        Met à jour le rating du joueur après un match
        
        Args: 
            new_rating (Rating): Nouveau rating TrueSkill
        """
        self.rating = new_rating
        self.history_mu. append(new_rating.mu)
        self.history_sigma.append(new_rating.sigma)
        self.matches_played += 1
    
    def record_win(self):
        """Enregistre une victoire"""
        self.wins += 1
    
    def record_loss(self):
        """Enregistre une défaite"""
        self.losses += 1
    
    @property
    def conservative_rating(self):
        """
        Rating conservateur (mu - 3*sigma)
        
        C'est souvent utilisé pour le classement car il pénalise l'incertitude. 
        On est sûr à 99. 7% que la vraie compétence est au-dessus de cette valeur.
        
        Returns:
            float: Rating conservateur
        """
        return self.rating.mu - 3 * self.rating.sigma
    
    @property
    def win_rate(self):
        """
        Taux de victoire
        
        Returns:
            float: Pourcentage de victoires (0-100)
        """
        if self.matches_played == 0:
            return 0.0
        return (self.wins / self.matches_played) * 100
    
    def __repr__(self):
        """Représentation string du joueur"""
        return (f"Player(name={self.name}, "
                f"μ={self.rating.mu:. 2f}, "
                f"σ={self.rating.sigma:. 2f}, "
                f"true_skill={self.true_skill})")
    
    def __str__(self):
        """Affichage formaté du joueur"""
        return (f"{self.name:12} | "
                f"μ={self.rating.mu:5.1f} ± {self.rating.sigma:4.2f} | "
                f"Conserv. ={self.conservative_rating:5.1f} | "
                f"Vrai={self.true_skill:4.1f} | "
                f"W/L={self.wins}/{self. losses} ({self.win_rate:.1f}%)")
