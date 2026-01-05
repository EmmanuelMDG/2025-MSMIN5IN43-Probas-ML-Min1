"""
Script principal pour tester le simulateur TrueSkill
"""
from src.utils import create_tiered_players, create_random_players, print_player_stats
from src.simulator import MatchSimulator


def demo_basic():
    """Démonstration basique avec quelques joueurs"""
    print("\n" + "="*60)
    print("🎮 DÉMONSTRATION TRUESKILL - MODE BASIQUE")
    print("="*60)
    
    # Créer des joueurs de différents niveaux
    players = create_tiered_players()
    
    print("\n📋 Joueurs créés :")
    print_player_stats(players)
    
    # Créer le simulateur
    simulator = MatchSimulator(players)
    
    # Afficher le classement initial
    print("\n📊 Classement initial (avant tout match) :")
    simulator.print_leaderboard()
    
    # Simuler quelques matchs avec détails
    print("\n" + "="*60)
    print("🎯 Simulation de 3 matchs détaillés")
    print("="*60)
    simulator.simulate_random_matches(3, verbose=True)
    
    # Simuler beaucoup de matchs
    simulator.simulate_random_matches(97, verbose=False)
    
    # Afficher les stats finales
    print("\n📈 STATISTIQUES FINALES :")
    print_player_stats(players)


def demo_round_robin():
    """Démonstration avec un tournoi round-robin"""
    print("\n" + "="*60)
    print("🏆 DÉMONSTRATION TRUESKILL - TOURNOI ROUND-ROBIN")
    print("="*60)
    
    # Créer 6 joueurs avec compétences variées
    players = create_tiered_players()[:6]  # Prendre seulement 6 joueurs
    
    print("\n📋 Participants :")
    for p in players:
        print(f"  {p.name:12} - Vraie compétence: {p.true_skill:.1f}")
    
    # Créer le simulateur
    simulator = MatchSimulator(players)
    
    # Lancer un tournoi (3 rounds = chaque paire se rencontre 3 fois)
    simulator.simulate_round_robin(rounds=3, verbose=False)
    
    # Stats finales
    print("\n📈 RÉSULTATS DU TOURNOI :")
    print_player_stats(players)


def demo_convergence():
    """Démonstration de la convergence (beaucoup de matchs)"""
    print("\n" + "="*60)
    print("📉 DÉMONSTRATION - CONVERGENCE DE TRUESKILL")
    print("="*60)
    
    # Créer des joueurs aléatoires
    players = create_random_players(8, min_skill=15, max_skill=35)
    
    print("\n📋 8 joueurs avec compétences aléatoires créés")
    print("🎯 Objectif : Observer la convergence après 200 matchs\n")
    
    # Créer le simulateur
    simulator = MatchSimulator(players)
    
    # Classement initial
    print("📊 AVANT (tous à μ=25, σ=8. 33) :")
    simulator.print_leaderboard()
    
    # Simuler beaucoup de matchs
    simulator.simulate_random_matches(200, verbose=False)
    
    # Afficher la convergence
    print("\n📊 APRÈS 200 MATCHS :")
    print("\nObservations :")
    print("  • μ (mu) converge vers la vraie compétence")
    print("  • σ (sigma) diminue (l'incertitude baisse)")
    print("  • Le classement reflète mieux les vraies compétences\n")
    
    print_player_stats(players)
    
    # Analyser la précision
    sorted_by_trueskill = sorted(players, key=lambda p: p.rating. mu, reverse=True)
    sorted_by_true = sorted(players, key=lambda p: p.true_skill, reverse=True)
    
    print("\n🎯 PRÉCISION DU CLASSEMENT :")
    print(f"{'Rang':<6} | {'Par TrueSkill':<15} | {'Par Vraie Compét.':<15} | {'Match? '}")
    print("="*60)
    for i in range(len(players)):
        match = "✅" if sorted_by_trueskill[i].name == sorted_by_true[i].name else "❌"
        print(f"{i+1: <6} | {sorted_by_trueskill[i].name:<15} | "
              f"{sorted_by_true[i].name:<15} | {match}")


if __name__ == "__main__":
    # Décommenter la démo que vous voulez tester
    
    # 1. Démonstration basique
    demo_basic()
    
    # 2. Tournoi round-robin
    # demo_round_robin()
    
    # 3. Convergence sur beaucoup de matchs
    # demo_convergence()
    
    print("\n✅ Simulation terminée !\n")
