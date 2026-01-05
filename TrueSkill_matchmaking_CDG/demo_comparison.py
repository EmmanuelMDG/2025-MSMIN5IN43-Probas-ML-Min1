"""
Script de démonstration de la comparaison TrueSkill vs ELO
"""
from comparison import (
    create_parallel_players,
    run_parallel_simulation,
    calculate_ranking_accuracy,
    print_comparison_results
)
from src.visualizer import plot_trueskill_vs_elo_convergence, plot_comparison_metrics
import os


def demo_comparison_full():
    """Comparaison complète avec visualisations"""
    print("\n" + "="*80)
    print("⚔️  COMPARAISON TRUESKILL vs ELO")
    print("="*80)
    
    # Créer le dossier results
    os.makedirs('results', exist_ok=True)
    
    # Paramètres
    num_players = 8
    num_matches = 200
    seed = 42
    
    print(f"\n Configuration :")
    print(f"  • Nombre de joueurs : {num_players}")
    print(f"  • Nombre de matchs : {num_matches}")
    print(f"  • Seed : {seed} (reproductible)")
    
    # Créer les joueurs
    print(f"\n Création de {num_players} joueurs identiques pour les deux systèmes...")
    ts_players, elo_players = create_parallel_players(num_players, seed=seed)
    
    print("\n Joueurs créés :")
    for ts_p, elo_p in zip(ts_players, elo_players):
        print(f"  • {ts_p.name:12} - Vraie compétence:  {ts_p.true_skill:.1f}")
    
    # Lancer la simulation
    ts_sim, elo_sim = run_parallel_simulation(
        ts_players, elo_players, num_matches, seed=seed, verbose=True
    )
    
    # Afficher les classements
    print("\n CLASSEMENT FINAL TRUESKILL :")
    ts_sim. print_leaderboard()
    
    print("\n CLASSEMENT FINAL ELO :")
    elo_sim.print_leaderboard()
    
    # Calculer les métriques
    print("\n Calcul des métriques de comparaison...")
    metrics = calculate_ranking_accuracy(ts_players, elo_players)
    
    # Afficher les résultats
    print_comparison_results(metrics)
    
    # Générer les visualisations
    print("\n Génération des visualisations...")
    plot_trueskill_vs_elo_convergence(ts_players, elo_players)
    plot_comparison_metrics(metrics)
    
    print("\n✅ Comparaison terminée !")
    print("📁 Les graphiques sont sauvegardés dans 'results/'")


def demo_comparison_quick():
    """Comparaison rapide sans visualisations détaillées"""
    print("\n" + "="*80)
    print("⚡ COMPARAISON RAPIDE TRUESKILL vs ELO")
    print("="*80)
    
    # Paramètres
    num_players = 6
    num_matches = 100
    
    # Créer et simuler
    ts_players, elo_players = create_parallel_players(num_players, seed=42)
    ts_sim, elo_sim = run_parallel_simulation(
        ts_players, elo_players, num_matches, seed=42, verbose=False
    )
    
    print(f"\n✅ {num_matches} matchs simulés")
    
    # Métriques
    metrics = calculate_ranking_accuracy(ts_players, elo_players)
    print_comparison_results(metrics)


if __name__ == "__main__":
    # Choisir la démo
    print("\n" + "="*80)
    print("Quelle comparaison voulez-vous lancer ?")
    print("  1. Complète (8 joueurs, 200 matchs + graphiques) - ~30 secondes")
    print("  2. Rapide (6 joueurs, 100 matchs) - ~10 secondes")
    print("="*80)
    
    choice = input("\nVotre choix (1 ou 2) : ").strip()
    
    if choice == "1":
        demo_comparison_full()
    elif choice == "2":
        demo_comparison_quick()
    else:
        print("❌ Choix invalide.  Lancement de la démo complète par défaut...")
        demo_comparison_full()
