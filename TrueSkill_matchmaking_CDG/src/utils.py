"""
Fonctions utilitaires
"""
import random
from src.player import Player


def create_random_players(num_players, min_skill=10, max_skill=40):
    """
    Crée des joueurs avec des compétences aléatoires
    
    Args:
        num_players (int): Nombre de joueurs à créer
        min_skill (float): Compétence minimale
        max_skill (float): Compétence maximale
    
    Returns:
        list[Player]: Liste des joueurs créés
    """
    names = [
        "Alice", "Bob", "Charlie", "David", "Eve", "Frank", 
        "Grace", "Heidi", "Ivan", "Judy", "Kevin", "Laura",
        "Mallory", "Niaj", "Olivia", "Peggy", "Quinn", "Romeo",
        "Sybil", "Trent", "Ursula", "Victor", "Walter", "Xander",
        "Yvonne", "Zelda"
    ]
    
    players = []
    for i in range(num_players):
        name = names[i] if i < len(names) else f"Player{i+1}"
        true_skill = random.uniform(min_skill, max_skill)
        players.append(Player(name, true_skill))
    
    return players


def create_tiered_players():
    """
    Crée des joueurs de différents niveaux (tiers)
    
    Returns:
        list[Player]: Liste de joueurs avec compétences prédéfinies
    """
    players = [
        # Tier S - Pro players
        Player("ProGamer", true_skill=35),
        Player("Champion", true_skill=33),
        
        # Tier A - Very good
        Player("Veteran", true_skill=28),
        Player("Expert", true_skill=27),
        
        # Tier B - Good
        Player("Skilled", true_skill=23),
        Player("Solid", true_skill=22),
        
        # Tier C - Average
        Player("Casual", true_skill=18),
        Player("Regular", true_skill=17),
        
        # Tier D - Beginners
        Player("Newbie", true_skill=12),
        Player("Rookie", true_skill=10),
    ]
    
    return players


def print_player_stats(players):
    """
    Affiche les statistiques détaillées de tous les joueurs
    
    Args:
        players (list[Player]): Liste des joueurs
    """
    print(f"\n{'='*100}")
    print(f"{'Joueur':<12} | {'Vraie Compét.':<13} | {'TrueSkill (μ±σ)':<20} | "
          f"{'Conserv.':<10} | {'Matchs':<7} | {'Victoires'}")
    print(f"{'='*100}")
    
    for player in sorted(players, key=lambda p:  p.true_skill, reverse=True):
        trueskill_str = f"{player.rating.mu:.2f} ± {player.rating.sigma:.2f}"
        print(f"{player.name:<12} | "
              f"{player.true_skill: >6.1f}        | "
              f"{trueskill_str:<20} | "
              f"{player.conservative_rating:>7.1f}    | "
              f"{player. matches_played:>4}    | "
              f"{player.wins}/{player.losses} ({player.win_rate:.0f}%)")
    
    print(f"{'='*100}\n")
