"""
Script de démonstration des visualisations
"""
import random
from src.utils import create_tiered_players, create_random_players
from src.simulator import MatchSimulator
from src.visualizer import create_all_visualizations


def demo_full_visualization():
    """Démonstration complète avec toutes les visualisations"""
    print("\n" + "="*60)
    print("🎮 DÉMONSTRATION COMPLÈTE - VISUALISATIONS TRUESKILL")
    print("="*60)
    
    # Créer des joueurs
    print("\n📋 Création de 8 joueurs avec compétences variées...")
    players = create_tiered_players()[: 8]
    
    print("\n🎯 Joueurs créés :")
    for p in players:
        print(f"  • {p.name:12} - Vraie compétence: {p. true_skill:.1f}")
    
    # Créer le simulateur
    simulator = MatchSimulator(players)
    
    # Simuler beaucoup de matchs pour avoir une belle convergence
    print(f"\n⚔️  Simulation de 150 matchs aléatoires...")
    print("   (Cela peut prendre quelques secondes... )")
    
    # Désactiver le verbose pour la vitesse
    simulator.simulate_random_matches(150, verbose=False)
    
    print("\n✅ Simulation terminée !")
    print(f"   • {sum(p.matches_played for p in players) // 2} matchs simulés")
    print(f"   • Incertitude moyenne :  {sum(p.rating.sigma for p in players) / len(players):.2f}")
    
    # Afficher le classement final
    print("\n🏆 Classement Final :")
    simulator.print_leaderboard()
    
    # Générer TOUTES les visualisations
    create_all_visualizations(players)
    
    print("\n🎉 Démonstration terminée !")
    print("💡 Ouvrez les fichiers PNG dans le dossier 'results/' pour voir les graphiques")


def demo_quick_visualization():
    """Démonstration rapide avec moins de matchs"""
    print("\n" + "="*60)
    print("⚡ DÉMONSTRATION RAPIDE - VISUALISATIONS TRUESKILL")
    print("="*60)
    
    # Créer 6 joueurs aléatoires
    players = create_random_players(6, min_skill=15, max_skill=35)
    
    print("\n🎯 6 joueurs créés avec compétences aléatoires")
    
    # Simuler 100 matchs
    simulator = MatchSimulator(players)
    simulator.simulate_random_matches(100, verbose=False)
    
    print("\n✅ 100 matchs simulés !")
    
    # Générer les visualisations essentielles
    from src.visualizer import (plot_skill_convergence, 
                                plot_uncertainty_decrease,
                                plot_before_after,
                                plot_matchmaking_heatmap)
    
    import os
    os.makedirs('results', exist_ok=True)
    
    print("\n🎨 Génération des visualisations principales...")
    plot_skill_convergence(players)
    plot_uncertainty_decrease(players)
    plot_before_after(players)
    plot_matchmaking_heatmap(players)
    
    print("\n✅ Visualisations prêtes dans 'results/'!")


if __name__ == "__main__":
    # Choisir la démo
    print("\n" + "="*60)
    print("Quelle démonstration voulez-vous lancer ?")
    print("  1. Complète (150 matchs + 7 graphiques) - ~30 secondes")
    print("  2. Rapide (100 matchs + 4 graphiques) - ~15 secondes")
    print("="*60)
    
    choice = input("\nVotre choix (1 ou 2) : ").strip()
    
    if choice == "1":
        demo_full_visualization()
    elif choice == "2":
        demo_quick_visualization()
    else:
        print("❌ Choix invalide.  Lancement de la démo complète par défaut...")
        demo_full_visualization()
