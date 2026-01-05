"""
Module de visualisation pour TrueSkill
Contient toutes les fonctions pour créer des graphiques impressionnants
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import numpy as np
from scipy. stats import norm
from trueskill import quality_1vs1

# Configuration du style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_skill_convergence(players, save_path='results/convergence_mu. png'):
    """
    Graphique de convergence de μ (mu) vers la vraie compétence
    
    Args:
        players (list[Player]): Liste des joueurs
        save_path (str): Chemin de sauvegarde
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for player in players:
        matches = range(len(player.history_mu))
        line, = ax.plot(matches, player.history_mu, 
                       label=f"{player.name} (vrai={player.true_skill:.0f})", 
                       linewidth=2.5, marker='o', markersize=3, alpha=0.8)
        
        # Ligne pointillée pour la vraie compétence
        ax.axhline(y=player.true_skill, linestyle='--', alpha=0.4, 
                  color=line.get_color(), linewidth=1.5)
    
    ax.set_xlabel('Nombre de matchs', fontsize=14, fontweight='bold')
    ax.set_ylabel('Compétence estimée (μ)', fontsize=14, fontweight='bold')
    ax.set_title('Convergence de TrueSkill vers la Vraie Compétence', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Ajouter une annotation
    ax.text(0.02, 0.98, 
           '💡 Les lignes pleines convergent vers les lignes pointillées',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé : {save_path}")
    plt.show()


def plot_uncertainty_decrease(players, save_path='results/convergence_sigma.png'):
    """
    Graphique de diminution de σ (sigma) - l'incertitude
    
    Args:
        players (list[Player]): Liste des joueurs
        save_path (str): Chemin de sauvegarde
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for player in players: 
        matches = range(len(player.history_sigma))
        ax.plot(matches, player.history_sigma, 
               label=player.name, linewidth=2.5, marker='o', 
               markersize=3, alpha=0.8)
    
    # Ligne de référence pour σ initial
    ax.axhline(y=8.333, linestyle=':', color='red', alpha=0.5, 
              linewidth=2, label='σ initial (8.33)')
    
    ax.set_xlabel('Nombre de matchs', fontsize=14, fontweight='bold')
    ax.set_ylabel('Incertitude (σ)', fontsize=14, fontweight='bold')
    ax.set_title('Diminution de l\'Incertitude au fil des Matchs', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Zone d'incertitude faible
    ax.axhspan(0, 2, alpha=0.1, color='green', label='Zone de confiance élevée')
    
    # Annotation
    ax.text(0.02, 0.98, 
           'Plus σ est bas, plus le système est sûr de la compétence',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé : {save_path}")
    plt.show()


def plot_before_after(players, save_path='results/before_after.png'):
    """
    Comparaison visuelle Avant/Après la simulation
    
    Args:
        players (list[Player]): Liste des joueurs
        save_path (str): Chemin de sauvegarde
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Trier par nom pour cohérence
    players_sorted = sorted(players, key=lambda p: p.name)
    names = [p.name for p in players_sorted]
    true_skills = [p.true_skill for p in players_sorted]
    estimated_skills = [p.rating. mu for p in players_sorted]
    uncertainties = [p.rating.sigma * 3 for p in players_sorted]
    
    x = np.arange(len(players_sorted))
    width = 0.6
    
    # AVANT (tout le monde à 25)
    bars1 = ax1.bar(x, [25]*len(players_sorted), width, 
                    label='Estimation initiale (μ=25)', 
                    color='gray', alpha=0.6, edgecolor='black', linewidth=1.5)
    
    scatter1 = ax1.scatter(x, true_skills, color='red', s=300, marker='*', 
                          label='Vraie compétence', zorder=5, edgecolors='darkred', linewidth=2)
    
    # Zone d'incertitude initiale
    ax1.errorbar(x, [25]*len(players_sorted), yerr=[8.333*3]*len(players_sorted), 
                fmt='none', ecolor='black', capsize=8, alpha=0.3, linewidth=2,
                label='Incertitude (±3σ)')
    
    ax1.set_ylabel('Compétence', fontsize=14, fontweight='bold')
    ax1.set_title('AVANT : Système Aveugle', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=11)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax1.grid(alpha=0.3, axis='y', linestyle='--')
    ax1.set_ylim(0, 45)
    ax1.set_facecolor('#f8f9fa')
    
    # APRÈS (convergé)
    bars2 = ax2.bar(x, estimated_skills, width, 
                    label='TrueSkill (μ)', 
                    color='steelblue', alpha=0.8, edgecolor='darkblue', linewidth=1.5)
    
    ax2.errorbar(x, estimated_skills, yerr=uncertainties, fmt='none', 
                ecolor='black', capsize=8, alpha=0.5, linewidth=2,
                label='Incertitude (±3σ)')
    
    scatter2 = ax2.scatter(x, true_skills, color='red', s=300, marker='*', 
                          label='Vraie compétence', zorder=5, edgecolors='darkred', linewidth=2)
    
    ax2.set_ylabel('Compétence', fontsize=14, fontweight='bold')
    ax2.set_title(f'APRÈS : {players_sorted[0].matches_played} matchs joués', 
                 fontsize=16, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=11)
    ax2.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax2.grid(alpha=0.3, axis='y', linestyle='--')
    ax2.set_ylim(0, 45)
    ax2.set_facecolor('#f0f8ff')
    
    plt.suptitle('Évolution du Système TrueSkill', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé :  {save_path}")
    plt.show()


def plot_matchmaking_heatmap(players, save_path='results/heatmap_matchmaking.png'):
    """
    Heatmap des probabilités de victoire et qualité des matchs
    
    Args: 
        players (list[Player]): Liste des joueurs
        save_path (str): Chemin de sauvegarde
    """
    n = len(players)
    win_probs = np.zeros((n, n))
    match_quality = np.zeros((n, n))
    
    # Calculer les matrices
    for i in range(n):
        for j in range(n):
            if i == j: 
                win_probs[i][j] = np.nan  # Diagonale = pas de match contre soi-même
                match_quality[i][j] = np.nan
            else:
                # Probabilité de victoire (formule TrueSkill)
                delta_mu = players[i].rating.mu - players[j].rating.mu
                sum_sigma = players[i].rating.sigma**2 + players[j].rating.sigma**2
                beta = 25/6  # Paramètre TrueSkill standard
                win_probs[i][j] = norm.cdf(delta_mu / np.sqrt(2 * beta**2 + sum_sigma))
                
                # Qualité du match
                match_quality[i][j] = quality_1vs1(players[i].rating, players[j].rating)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Heatmap probabilités de victoire
    mask1 = np.eye(n, dtype=bool)  # Masquer la diagonale
    sns.heatmap(win_probs, annot=True, fmt='.0%', cmap='RdYlGn', 
                xticklabels=[p.name for p in players],
                yticklabels=[p.name for p in players],
                cbar_kws={'label': 'Probabilité de victoire', 'shrink': 0.8},
                ax=ax1, vmin=0, vmax=1, linewidths=0.5, 
                linecolor='gray', mask=mask1, annot_kws={'size': 10})
    ax1.set_title('Probabilités de Victoire\n(Joueur Ligne vs Joueur Colonne)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel('Adversaire', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Joueur', fontsize=12, fontweight='bold')
    
    # Heatmap qualité des matchs
    mask2 = np.eye(n, dtype=bool)
    sns.heatmap(match_quality, annot=True, fmt='.0%', cmap='Blues',
                xticklabels=[p.name for p in players],
                yticklabels=[p.name for p in players],
                cbar_kws={'label': 'Qualité du match', 'shrink': 0.8},
                ax=ax2, vmin=0, vmax=1, linewidths=0.5,
                linecolor='gray', mask=mask2, annot_kws={'size': 10})
    ax2.set_title('⚖️ Qualité des Matchs\n(100% = parfaitement équilibré)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel('Adversaire', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Joueur', fontsize=12, fontweight='bold')
    
    plt.suptitle('Matrice de Matchmaking Optimal', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé : {save_path}")
    plt.show()


def plot_ranking_comparison(players, save_path='results/ranking_comparison.png'):
    """
    Compare le classement TrueSkill vs la vraie compétence
    
    Args:
        players (list[Player]): Liste des joueurs
        save_path (str): Chemin de sauvegarde
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Trier les joueurs
    sorted_by_trueskill = sorted(players, key=lambda p: p.rating. mu, reverse=True)
    sorted_by_true = sorted(players, key=lambda p: p.true_skill, reverse=True)
    
    names = [p.name for p in sorted_by_trueskill]
    trueskill_mus = [p.rating.mu for p in sorted_by_trueskill]
    trueskill_sigmas = [p.rating.sigma * 3 for p in sorted_by_trueskill]
    true_skills = [p.true_skill for p in sorted_by_trueskill]
    
    x = np.arange(len(players))
    width = 0.35
    
    # Barres TrueSkill
    bars1 = ax.bar(x - width/2, trueskill_mus, width, 
                   label='TrueSkill (μ)', color='steelblue', 
                   alpha=0.8, edgecolor='darkblue', linewidth=1.5)
    
    # Barres vraie compétence
    bars2 = ax.bar(x + width/2, true_skills, width, 
                   label='Vraie Compétence', color='coral', 
                   alpha=0.8, edgecolor='darkred', linewidth=1.5)
    
    # Barres d'erreur pour l'incertitude
    ax.errorbar(x - width/2, trueskill_mus, yerr=trueskill_sigmas, 
               fmt='none', ecolor='black', capsize=5, alpha=0.5, linewidth=2)
    
    ax.set_xlabel('Joueur (classé par TrueSkill)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Compétence', fontsize=14, fontweight='bold')
    ax.set_title('🏆 Classement Final :  TrueSkill vs Vraie Compétence', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(alpha=0.3, axis='y', linestyle='--')
    
    # Calculer la précision
    accuracy_count = sum(1 for i in range(len(players)) 
                        if sorted_by_trueskill[i].name == sorted_by_true[i].name)
    accuracy = accuracy_count / len(players)
    
    # Annotation de précision
    ax.text(0.98, 0.98, 
           f'Précision du classement : {accuracy:.0%}\n'
           f' {accuracy_count}/{len(players)} positions correctes',
           transform=ax. transAxes, fontsize=12, verticalalignment='top',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé : {save_path}")
    plt.show()


def plot_confidence_intervals(players, save_path='results/confidence_intervals.png'):
    """
    Visualise les intervalles de confiance pour chaque joueur
    
    Args:
        players (list[Player]): Liste des joueurs
        save_path (str): Chemin de sauvegarde
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Trier par rating conservateur
    sorted_players = sorted(players, key=lambda p: p.conservative_rating, reverse=True)
    
    names = [p. name for p in sorted_players]
    mus = [p.rating.mu for p in sorted_players]
    sigmas_1 = [p.rating.sigma for p in sorted_players]
    sigmas_3 = [p.rating.sigma * 3 for p in sorted_players]
    true_skills = [p.true_skill for p in sorted_players]
    
    y = np.arange(len(sorted_players))
    
    # Intervalles de confiance (±3σ = 99.7%)
    for i, (mu, sigma_3, true_skill) in enumerate(zip(mus, sigmas_3, true_skills)):
        # Barre d'intervalle
        ax.barh(i, sigma_3*2, left=mu-sigma_3, height=0.6, 
               color='steelblue', alpha=0.3, edgecolor='darkblue', linewidth=1.5)
        # Point central (μ)
        ax.plot(mu, i, 'o', color='darkblue', markersize=10, zorder=3)
        # Vraie compétence
        ax. plot(true_skill, i, '*', color='red', markersize=15, zorder=4)
    
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('Compétence', fontsize=14, fontweight='bold')
    ax.set_title('📏 Intervalles de Confiance (±3σ = 99.7%)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(alpha=0.3, axis='x', linestyle='--')
    ax.invert_yaxis()  # Meilleur joueur en haut
    
    # Légende
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.3, edgecolor='darkblue', label='Intervalle ±3σ'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', 
               markersize=10, label='Compétence estimée (μ)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
               markersize=15, label='Vraie compétence')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.9)
    
    # Annotation
    ax.text(0.02, 0.98, 
           'Plus l\'intervalle est étroit, plus le système est confiant',
           transform=ax. transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé : {save_path}")
    plt.show()


def plot_all_stats(players, save_path='results/all_stats.png'):
    """
    Dashboard complet avec toutes les stats
    
    Args:
        players (list[Player]): Liste des joueurs
        save_path (str): Chemin de sauvegarde
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Convergence μ
    ax1 = fig.add_subplot(gs[0, :2])
    for player in players:
        ax1.plot(player.history_mu, label=player.name, linewidth=2, alpha=0.7)
        ax1.axhline(y=player.true_skill, linestyle='--', alpha=0.3)
    ax1.set_xlabel('Matchs')
    ax1.set_ylabel('μ')
    ax1.set_title('Convergence de μ', fontweight='bold')
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)
    
    # 2. Diminution σ
    ax2 = fig.add_subplot(gs[0, 2])
    for player in players: 
        ax2.plot(player. history_sigma, linewidth=2, alpha=0.7)
    ax2.set_xlabel('Matchs')
    ax2.set_ylabel('σ')
    ax2.set_title('Diminution de σ', fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. Classement final
    ax3 = fig. add_subplot(gs[1, :])
    sorted_players = sorted(players, key=lambda p: p.rating.mu, reverse=True)
    names = [p.name for p in sorted_players]
    mus = [p.rating.mu for p in sorted_players]
    true_skills = [p.true_skill for p in sorted_players]
    x = np.arange(len(players))
    width = 0.35
    ax3.bar(x - width/2, mus, width, label='TrueSkill', color='steelblue', alpha=0.8)
    ax3.bar(x + width/2, true_skills, width, label='Vrai', color='coral', alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.set_ylabel('Compétence')
    ax3.set_title('Classement Final', fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3, axis='y')
    
    # 4. Win rates
    ax4 = fig. add_subplot(gs[2, 0])
    win_rates = [p.win_rate for p in sorted_players]
    ax4.barh(names, win_rates, color='green', alpha=0.7)
    ax4.set_xlabel('Taux de victoire (%)')
    ax4.set_title('Taux de Victoire', fontweight='bold')
    ax4.grid(alpha=0.3, axis='x')
    
    # 5. Nombre de matchs
    ax5 = fig.add_subplot(gs[2, 1])
    matches = [p.matches_played for p in sorted_players]
    ax5.barh(names, matches, color='purple', alpha=0.7)
    ax5.set_xlabel('Matchs joués')
    ax5.set_title('Matchs Joués', fontweight='bold')
    ax5.grid(alpha=0.3, axis='x')
    
    # 6. Incertitude finale
    ax6 = fig. add_subplot(gs[2, 2])
    sigmas = [p.rating.sigma for p in sorted_players]
    ax6.barh(names, sigmas, color='orange', alpha=0.7)
    ax6.set_xlabel('σ final')
    ax6.set_title('Incertitude Finale', fontweight='bold')
    ax6.grid(alpha=0.3, axis='x')
    
    plt.suptitle('TrueSkill - Dashboard Complet', 
                fontsize=20, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé : {save_path}")
    plt.show()


def create_all_visualizations(players):
    """
    Crée toutes les visualisations d'un coup
    
    Args:
        players (list[Player]): Liste des joueurs
    """
    print("\n" + "="*60)
    print("GÉNÉRATION DE TOUTES LES VISUALISATIONS")
    print("="*60 + "\n")
    
    import os
    os.makedirs('results', exist_ok=True)
    
    print("1️  Convergence de μ...")
    plot_skill_convergence(players, save_path='results/convergence_mu.png')
    
    print("\n2️  Diminution de σ...")
    plot_uncertainty_decrease(players, save_path='results/convergence_sigma.png')
    
    print("\n3️  Avant/Après...")
    plot_before_after(players, save_path='results/before_after.png')
    
    print("\n4️  Heatmap Matchmaking...")
    plot_matchmaking_heatmap(players, save_path='results/heatmap_matchmaking.png')
    
    print("\n5️  Comparaison classement...")
    plot_ranking_comparison(players, save_path='results/ranking_comparison.png')
    
    print("\n6️  Intervalles de confiance...")
    plot_confidence_intervals(players, save_path='results/confidence_intervals. png')
    
    print("\n7️  Dashboard complet...")
    plot_all_stats(players, save_path='results/all_stats.png')
    
    print("\n" + "="*60)
    print("TOUTES LES VISUALISATIONS SONT PRÊTES !")
    print("Fichiers sauvegardés dans le dossier 'results/'")
    print("="*60 + "\n")

def plot_trueskill_vs_elo_convergence(ts_players, elo_players, save_path='results/ts_vs_elo.png'):
    """
    Compare la vitesse de convergence TrueSkill vs ELO
    
    Args:
        ts_players (list[Player]): Joueurs TrueSkill
        elo_players (list[EloPlayer]): Joueurs ELO
        save_path (str): Chemin de sauvegarde
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # TrueSkill - Normaliser pour comparaison
    for player in ts_players:
        # Convertir μ (échelle ~0-50) vers échelle ELO (~1000-2000)
        normalized_history = [(mu - 25) * 60 + 1500 for mu in player.history_mu]
        true_skill_elo = (player.true_skill - 25) * 60 + 1500
        
        ax1.plot(normalized_history, label=player.name, linewidth=2.5, alpha=0.8)
        ax1.axhline(y=true_skill_elo, linestyle='--', alpha=0.3)
    
    ax1.set_xlabel('Nombre de matchs', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Rating (échelle ELO)', fontsize=14, fontweight='bold')
    ax1.set_title('TrueSkill - Convergence', fontsize=16, fontweight='bold', pad=15)
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(alpha=0.3)
    ax1.set_facecolor('#f0f8ff')
    
    # ELO
    for player in elo_players:
        true_skill_elo = (player.true_skill - 25) * 60 + 1500
        
        ax2.plot(player.history, label=player.name, linewidth=2.5, alpha=0.8)
        ax2.axhline(y=true_skill_elo, linestyle='--', alpha=0.3)
    
    ax2.set_xlabel('Nombre de matchs', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Rating ELO', fontsize=14, fontweight='bold')
    ax2.set_title(' ELO - Convergence', fontsize=16, fontweight='bold', pad=15)
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(alpha=0.3)
    ax2.set_facecolor('#fff8f0')
    
    plt.suptitle('TrueSkill vs ELO :  Vitesse de Convergence', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé : {save_path}")
    plt.show()


def plot_comparison_metrics(metrics, save_path='results/comparison_metrics.png'):
    """
    Visualise les métriques de comparaison
    
    Args: 
        metrics (dict): Métriques calculées
        save_path (str): Chemin de sauvegarde
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Précision exacte
    systems = ['TrueSkill', 'ELO']
    accuracies = [metrics['trueskill_exact_accuracy'] * 100, 
                  metrics['elo_exact_accuracy'] * 100]
    colors = ['steelblue', 'coral']
    
    bars1 = ax1.bar(systems, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Précision (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Précision du Classement\n(% positions exactes)', 
                 fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.grid(alpha=0.3, axis='y')
    
    # Annoter les barres
    for bar, acc in zip(bars1, accuracies):
        height = bar. get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # 2. Corrélation de Spearman
    correlations = [metrics['trueskill_spearman'], metrics['elo_spearman']]
    
    bars2 = ax2.bar(systems, correlations, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Corrélation', fontsize=12, fontweight='bold')
    ax2.set_title('Corrélation de Spearman\n(ordre du classement)', 
                 fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.axhline(y=1.0, linestyle='--', color='green', alpha=0.5, label='Parfait')
    ax2.grid(alpha=0.3, axis='y')
    ax2.legend()
    
    for bar, corr in zip(bars2, correlations):
        height = bar.get_height()
        ax2.text(bar. get_x() + bar.get_width()/2., height,
                f'{corr:.3f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # 3. Erreur Moyenne Absolue (inverse :  plus bas = mieux)
    maes = [metrics['trueskill_mae'], metrics['elo_mae']]
    
    bars3 = ax3.bar(systems, maes, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Erreur (points)', fontsize=12, fontweight='bold')
    ax3.set_title('📏 Erreur Moyenne Absolue\n(plus bas = mieux)', 
                 fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    
    for bar, mae in zip(bars3, maes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{mae:.1f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # 4. Tableau récapitulatif
    ax4.axis('off')
    
    table_data = [
        ['Métrique', 'TrueSkill', 'ELO', 'Gagnant'],
        ['Précision', f"{accuracies[0]:.1f}%", f"{accuracies[1]:.1f}%", 
         '🏆 TS' if accuracies[0] > accuracies[1] else '🏆 ELO'],
        ['Corrélation', f"{correlations[0]:.3f}", f"{correlations[1]:.3f}",
         '🏆 TS' if correlations[0] > correlations[1] else '🏆 ELO'],
        ['MAE', f"{maes[0]:.1f}", f"{maes[1]:.1f}",
         '🏆 TS' if maes[0] < maes[1] else '🏆 ELO'],
        ['Incertitude (σ)', f"{metrics['avg_sigma']:.2f}", 'N/A', 'TS only']
    ]
    
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 3)
    
    # Styliser l'en-tête
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alterner les couleurs des lignes
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax4.set_title('📊 Récapitulatif des Métriques', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('⚔️ TrueSkill vs ELO : Comparaison Complète', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé : {save_path}")
    plt.show()