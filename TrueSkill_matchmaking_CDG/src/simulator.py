"""
Simulateur de matchs TrueSkill
"""
import random
from trueskill import rate_1vs1, quality_1vs1


class MatchSimulator:
    """
    Gère la simulation de matchs entre joueurs
    """
    
    def __init__(self, players):
        """
        Initialise le simulateur
        
        Args:
            players (list[Player]): Liste des joueurs
        """
        self.players = players
        self.match_history = []
    
    def simulate_1v1(self, player1, player2, verbose=False):
        """
        Simule un match 1v1 entre deux joueurs
        
        Le gagnant est déterminé par la performance simulée (vraie compétence + aléa).
        Les ratings TrueSkill sont ensuite mis à jour.
        
        Args:
            player1 (Player): Premier joueur
            player2 (Player): Deuxième joueur
            verbose (bool): Afficher les détails du match
        
        Returns:  
            tuple: (gagnant, perdant)
        """
        # Simuler les performances
        perf1 = player1.play_match()
        perf2 = player2.play_match()
        
        # Déterminer le gagnant
        if perf1 > perf2:
            winner, loser = player1, player2
        else:
            winner, loser = player2, player1
        
        # Qualité du match avant (0=déséquilibré, 1=équilibré)
        match_quality = quality_1vs1(player1.rating, player2.rating)
        
        # Sauvegarder les anciens ratings
        old_rating_winner = winner.rating.mu
        old_rating_loser = loser.rating.mu
        
        # Mettre à jour les ratings TrueSkill
        if winner == player1:
            new_r1, new_r2 = rate_1vs1(player1.rating, player2.rating)
        else:
            new_r2, new_r1 = rate_1vs1(player2.rating, player1.rating)
        
        player1.update_rating(new_r1)
        player2.update_rating(new_r2)
        
        # Mettre à jour les stats
        winner.record_win()
        loser.record_loss()
        
        # Enregistrer l'historique
        match_record = {
            'player1': player1.name,
            'player2': player2.name,
            'winner': winner.name,
            'quality': match_quality,
            'perf1': perf1,
            'perf2': perf2
        }
        self.match_history.append(match_record)
        
        if verbose:  
            print(f"\n{'='*60}")
            print(f"⚔️  {player1.name} vs {player2.name}")
            print(f"{'='*60}")
            print(f"Performances:   {player1.name}={perf1:.1f} | {player2.name}={perf2:.1f}")
            print(f"🏆 Gagnant: {winner.name}")
            print(f"📊 Qualité du match: {match_quality:.1%}")
            print(f"\n{player1.name}:  μ {old_rating_winner:.1f} → {player1.rating.mu:.1f} "
                f"({player1.rating.mu - old_rating_winner:+.1f})")
            print(f"{player2.name}: μ {old_rating_loser:.1f} → {player2.rating.mu:.1f} "
                f"({player2.rating.mu - old_rating_loser:+.1f})")
        
        return winner, loser
    
    def simulate_random_matches(self, num_matches, verbose=False):
        """
        Simule un nombre donné de matchs aléatoires
        
        Args:
            num_matches (int): Nombre de matchs à simuler
            verbose (bool): Afficher les détails
        """
        print(f"\n🎮 Simulation de {num_matches} matchs aléatoires...")
        print("="*60)
        
        for i in range(num_matches):
            # Choisir 2 joueurs aléatoirement
            player1, player2 = random.sample(self.players, 2)
            
            # Simuler le match
            self.simulate_1v1(player1, player2, verbose=verbose)
            
            # Afficher un résumé tous les 20 matchs
            if (i + 1) % 20 == 0 and not verbose:
                print(f"\n--- Après {i + 1} matchs ---")
                self.print_leaderboard()
    
    def simulate_round_robin(self, rounds=1, verbose=False):
        """
        Simule un tournoi round-robin (chacun contre tous)
        
        Args:
            rounds (int): Nombre de fois que chaque paire se rencontre
            verbose (bool): Afficher les détails
        """
        n = len(self.players)
        total_matches = (n * (n - 1) // 2) * rounds
        
        print(f"\n🏆 Tournoi Round-Robin ({rounds} round{'s' if rounds > 1 else ''})")
        print(f"   {n} joueurs × {total_matches} matchs")
        print("="*60)
        
        for round_num in range(rounds):
            if rounds > 1:
                print(f"\n--- Round {round_num + 1}/{rounds} ---")
            
            # Chaque joueur affronte chaque autre joueur
            for i in range(len(self.players)):
                for j in range(i + 1, len(self.players)):
                    self.simulate_1v1(self.players[i], self.players[j], verbose=verbose)
        
        print(f"\n✅ Tournoi terminé !")
        self.print_leaderboard()
    
    def print_leaderboard(self):
        """Affiche le classement actuel"""
        # Trier par rating conservateur (mu - 3*sigma)
        sorted_players = sorted(self.players, 
                               key=lambda p:  p.conservative_rating, 
                               reverse=True)
        
        print(f"\n{'='*90}")
        print(f"{'Classement':<12} | {'μ (skill)':<15} | {'Conserv.':<10} | {'Vrai': <6} | {'Matchs (W/L)'}")
        print(f"{'='*90}")
        
        for rank, player in enumerate(sorted_players, 1):
            print(f"{rank}. {player}")
        
        print(f"{'='*90}\n")
    
    def get_leaderboard(self):
        """
        Retourne le classement sous forme de liste
        
        Returns:
            list[Player]:  Joueurs triés par rating conservateur
        """
        return sorted(self.players, 
                     key=lambda p: p.conservative_rating, 
                     reverse=True)
