"""
Comparaison entre TrueSkill et ELO
"""
import random
import numpy as np
from src. player import Player
from src.simulator import MatchSimulator
from src.elo import EloPlayer, EloSimulator
from trueskill import rate_1vs1


def create_parallel_players(num_players, min_skill=15, max_skill=35, seed=None):
    """
    Crée deux ensembles identiques de joueurs (TrueSkill et ELO)
    
    Args:
        num_players (int): Nombre de joueurs
        min_skill (float): Compétence minimale
        max_skill (float): Compétence maximale
        seed (int): Seed pour reproductibilité
    
    Returns:
        tuple: (trueskill_players, elo_players)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    names = [
        "Alice", "Bob", "Charlie", "David", "Eve", "Frank",
        "Grace", "Heidi", "Ivan", "Judy", "Kevin", "Laura",
        "Mallory", "Niaj", "Olivia", "Peggy"
    ]
    
    # Générer les compétences
    true_skills = [random.uniform(min_skill, max_skill) for _ in range(num_players)]
    
    # Créer les joueurs TrueSkill
    trueskill_players = []
    for i in range(num_players):
        name = names[i] if i < len(names) else f"Player{i+1}"
        trueskill_players.append(Player(name, true_skills[i]))
    
    # Créer les joueurs ELO (mêmes noms et compétences)
    elo_players = []
    for i in range(num_players):
        name = names[i] if i < len(names) else f"Player{i+1}"
        elo_players.append(EloPlayer(name, true_skills[i]))
    
    return trueskill_players, elo_players


def run_parallel_simulation(trueskill_players, elo_players, num_matches, seed=None, verbose=False):
    """
    Lance la même séquence de matchs pour TrueSkill et ELO
    
    Args:
        trueskill_players (list[Player]): Joueurs TrueSkill
        elo_players (list[EloPlayer]): Joueurs ELO
        num_matches (int): Nombre de matchs
        seed (int): Seed pour reproductibilité
        verbose (bool): Afficher les détails
    
    Returns:
        tuple: (ts_simulator, elo_simulator)
    """
    if seed is not None:
        random.seed(seed)
    
    ts_simulator = MatchSimulator(trueskill_players)
    elo_simulator = EloSimulator(elo_players)
    
    if verbose:
        print(f"\n🎮 Simulation de {num_matches} matchs identiques pour TrueSkill et ELO...")
        print("="*80)
    
    for i in range(num_matches):
        # Choisir les mêmes paires pour les deux systèmes
        idx1, idx2 = random.sample(range(len(trueskill_players)), 2)
        
        ts_p1, ts_p2 = trueskill_players[idx1], trueskill_players[idx2]
        elo_p1, elo_p2 = elo_players[idx1], elo_players[idx2]
        
        # Simuler la performance (basée sur la vraie compétence)
        beta = 25 / 6
        perf1 = random.gauss(ts_p1.true_skill, beta)
        perf2 = random.gauss(ts_p2.true_skill, beta)
        
        # Déterminer le gagnant (même pour les deux systèmes)
        ts_winner = ts_p1 if perf1 > perf2 else ts_p2
        ts_loser = ts_p2 if ts_winner == ts_p1 else ts_p1
        
        elo_winner = elo_p1 if perf1 > perf2 else elo_p2
        elo_loser = elo_p2 if elo_winner == elo_p1 else elo_p1
        
        # Mettre à jour TrueSkill
        if ts_winner == ts_p1:
            new_r1, new_r2 = rate_1vs1(ts_p1. rating, ts_p2.rating)
        else:
            new_r2, new_r1 = rate_1vs1(ts_p2.rating, ts_p1.rating)
        
        ts_p1.update_rating(new_r1)
        ts_p2.update_rating(new_r2)
        ts_winner.record_win()
        ts_loser.record_loss()
        
        # Mettre à jour ELO
        elo_simulator.simulate_match(elo_p1, elo_p2, elo_winner)
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  Match {i + 1}/{num_matches} simulé...")
    
    if verbose:
        print(f"✅ Simulation terminée !\n")
    
    return ts_simulator, elo_simulator


def calculate_ranking_accuracy(players_trueskill, players_elo):
    """
    Compare la précision du classement par rapport aux vraies compétences
    
    Args:
        players_trueskill (list[Player]): Joueurs TrueSkill
        players_elo (list[EloPlayer]): Joueurs ELO
    
    Returns:
        dict: Métriques de comparaison
    """
    # Classement par vraie compétence (référence)
    true_ranking = sorted(players_trueskill, key=lambda p: p. true_skill, reverse=True)
    true_names = [p.name for p in true_ranking]
    
    # Classement TrueSkill
    ts_ranking = sorted(players_trueskill, key=lambda p: p.rating. mu, reverse=True)
    ts_names = [p.name for p in ts_ranking]
    
    # Classement ELO
    elo_ranking = sorted(players_elo, key=lambda p: p.rating, reverse=True)
    elo_names = [p.name for p in elo_ranking]
    
    # Précision exacte (même position)
    ts_exact = sum(1 for i in range(len(true_names)) if ts_names[i] == true_names[i])
    elo_exact = sum(1 for i in range(len(true_names)) if elo_names[i] == true_names[i])
    
    # Corrélation de Spearman (ordre)
    from scipy.stats import spearmanr
    
    ts_indices = [ts_names.index(name) for name in true_names]
    elo_indices = [elo_names. index(name) for name in true_names]
    
    ts_corr, _ = spearmanr(range(len(true_names)), ts_indices)
    elo_corr, _ = spearmanr(range(len(true_names)), elo_indices)
    
    # Erreur moyenne absolue (MAE) sur les ratings normalisés
    # Normaliser TrueSkill (μ - 25) * 60 + 1500 pour être comparable à ELO
    ts_normalized = [(p.rating. mu - 25) * 60 + 1500 for p in players_trueskill]
    elo_ratings = [p.rating for p in players_elo]
    true_normalized = [(p.true_skill - 25) * 60 + 1500 for p in players_trueskill]
    
    ts_mae = np.mean([abs(ts_normalized[i] - true_normalized[i]) for i in range(len(players_trueskill))])
    elo_mae = np.mean([abs(elo_ratings[i] - true_normalized[i]) for i in range(len(players_elo))])
    
    # Incertitude moyenne (TrueSkill uniquement)
    avg_sigma = np.mean([p.rating.sigma for p in players_trueskill])
    
    return {
        'trueskill_exact_accuracy': ts_exact / len(true_names),
        'elo_exact_accuracy': elo_exact / len(true_names),
        'trueskill_spearman': ts_corr,
        'elo_spearman': elo_corr,
        'trueskill_mae': ts_mae,
        'elo_mae': elo_mae,
        'avg_sigma': avg_sigma
    }


def print_comparison_results(metrics):
    """
    Affiche les résultats de comparaison
    
    Args: 
        metrics (dict): Métriques calculées
    """
    print("\n" + "="*80)
    print("📊 RÉSULTATS DE LA COMPARAISON TRUESKILL vs ELO")
    print("="*80)
    
    print("\n🎯 PRÉCISION DU CLASSEMENT (% positions exactes) :")
    print(f"  • TrueSkill : {metrics['trueskill_exact_accuracy']:.1%} ✅")
    print(f"  • ELO       : {metrics['elo_exact_accuracy']:.1%}")
    
    winner = "TrueSkill" if metrics['trueskill_exact_accuracy'] > metrics['elo_exact_accuracy'] else "ELO"
    diff = abs(metrics['trueskill_exact_accuracy'] - metrics['elo_exact_accuracy'])
    print(f"  → Gagnant : {winner} (+{diff:.1%})")
    
    print("\n📈 CORRÉLATION DE SPEARMAN (ordre du classement) :")
    print(f"  • TrueSkill : {metrics['trueskill_spearman']:.3f} ✅")
    print(f"  • ELO       : {metrics['elo_spearman']:.3f}")
    
    print("\n📏 ERREUR MOYENNE ABSOLUE (MAE) sur les ratings :")
    print(f"  • TrueSkill : {metrics['trueskill_mae']:.1f}")
    print(f"  • ELO       : {metrics['elo_mae']:.1f} ✅")
    
    print(f"\n💡 Incertitude moyenne TrueSkill (σ) : {metrics['avg_sigma']:.2f}")
    print("   (Plus c'est bas, plus le système est confiant)")
    
    print("\n" + "="*80)
    
    # Verdict final
    score_ts = 0
    score_elo = 0
    
    if metrics['trueskill_exact_accuracy'] > metrics['elo_exact_accuracy']:
        score_ts += 1
    else:
        score_elo += 1
    
    if metrics['trueskill_spearman'] > metrics['elo_spearman']:
        score_ts += 1
    else: 
        score_elo += 1
    
    if metrics['trueskill_mae'] < metrics['elo_mae']:
        score_ts += 1
    else:
        score_elo += 1
    
    print(f"\n🏆 VERDICT FINAL : TrueSkill {score_ts} - {score_elo} ELO")
    
    if score_ts > score_elo:
        print("   ✅ TrueSkill est MEILLEUR !")
    elif score_elo > score_ts:
        print("   ⚠️  ELO est meilleur sur cette simulation")
    else:
        print("   ⚖️  Match nul")
    
    print("="*80 + "\n")
