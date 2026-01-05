"""
Implémentation du système de classement ELO
Utilisé pour la comparaison avec TrueSkill
"""


class EloPlayer:
    """
    Représente un joueur avec le système ELO classique
    
    Attributes:
        name (str): Nom du joueur
        true_skill (float): Compétence réelle (cachée)
        rating (float): Rating ELO actuel
        history (list): Historique des ratings
        matches_played (int): Nombre de matchs joués
        wins (int): Nombre de victoires
        losses (int): Nombre de défaites
        k_factor (float): Facteur K (sensibilité aux changements)
    """
    
    def __init__(self, name, true_skill, initial_rating=1500, k_factor=32):
        """
        Initialise un joueur ELO
        
        Args:
            name (str): Nom du joueur
            true_skill (float): Compétence réelle
            initial_rating (float): Rating ELO initial (défaut:  1500)
            k_factor (float): Facteur K (défaut: 32 pour joueurs normaux)
        """
        self.name = name
        self.true_skill = true_skill
        self.rating = initial_rating
        self.history = [initial_rating]
        
        # Statistiques
        self.matches_played = 0
        self.wins = 0
        self.losses = 0
        
        # Paramètre ELO
        self.k_factor = k_factor
    
    def expected_score(self, opponent):
        """
        Calcule le score attendu contre un adversaire (formule ELO)
        
        Args:
            opponent (EloPlayer): Adversaire
        
        Returns:
            float:  Probabilité de victoire (0-1)
        """
        return 1 / (1 + 10**((opponent.rating - self.rating) / 400))
    
    def update_rating(self, opponent, won):
        """
        Met à jour le rating après un match
        
        Args: 
            opponent (EloPlayer): Adversaire
            won (bool): True si victoire, False si défaite
        """
        expected = self.expected_score(opponent)
        actual = 1.0 if won else 0.0
        
        # Formule ELO :  nouveau_rating = ancien + K * (résultat - attendu)
        self.rating += self.k_factor * (actual - expected)
        self.history.append(self.rating)
        self.matches_played += 1
        
        if won:
            self.wins += 1
        else:
            self.losses += 1
    
    @property
    def win_rate(self):
        """Taux de victoire en pourcentage"""
        if self.matches_played == 0:
            return 0.0
        return (self.wins / self.matches_played) * 100
    
    def __repr__(self):
        return f"EloPlayer(name={self.name}, rating={self. rating:.0f}, true_skill={self.true_skill})"
    
    def __str__(self):
        return (f"{self.name:12} | "
                f"ELO={self.rating:7.1f} | "
                f"Vrai={self.true_skill:4.1f} | "
                f"W/L={self.wins}/{self. losses} ({self.win_rate:.1f}%)")


class EloSimulator:
    """
    Simulateur de matchs pour le système ELO
    """
    
    def __init__(self, players):
        """
        Initialise le simulateur
        
        Args:
            players (list[EloPlayer]): Liste des joueurs ELO
        """
        self.players = players
        self. match_history = []
    
    def simulate_match(self, player1, player2, actual_winner):
        """
        Simule un match et met à jour les ratings
        
        Args: 
            player1 (EloPlayer): Premier joueur
            player2 (EloPlayer): Deuxième joueur
            actual_winner (EloPlayer): Le vrai gagnant (déterminé par performance)
        """
        # Mettre à jour les ratings
        if actual_winner == player1:
            player1.update_rating(player2, won=True)
            player2.update_rating(player1, won=False)
        else:
            player2.update_rating(player1, won=True)
            player1.update_rating(player2, won=False)
        
        # Enregistrer l'historique
        self.match_history.append({
            'player1': player1.name,
            'player2': player2.name,
            'winner': actual_winner. name
        })
    
    def get_leaderboard(self):
        """
        Retourne le classement trié par rating
        
        Returns:
            list[EloPlayer]:  Joueurs triés par rating décroissant
        """
        return sorted(self.players, key=lambda p: p.rating, reverse=True)
    
    def print_leaderboard(self):
        """Affiche le classement"""
        sorted_players = self.get_leaderboard()
        
        print(f"\n{'='*80}")
        print(f"{'Rang':<6} | {'Joueur': <12} | {'ELO':<10} | {'Vrai':<6} | {'Matchs (W/L)'}")
        print(f"{'='*80}")
        
        for rank, player in enumerate(sorted_players, 1):
            print(f"{rank:<6} | {player}")
        
        print(f"{'='*80}\n")
