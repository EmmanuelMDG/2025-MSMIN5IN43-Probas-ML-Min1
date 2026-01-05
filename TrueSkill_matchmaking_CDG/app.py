"""
Interface Web Interactive pour TrueSkill Matchmaking Simulator
Lancer avec : streamlit run app. py
"""
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
from src.player import Player
from src. simulator import MatchSimulator
from src.utils import create_tiered_players, create_random_players
from src.visualizer import (
    plot_skill_convergence,
    plot_uncertainty_decrease,
    plot_matchmaking_heatmap,
    plot_before_after
)

# Configuration de la page
st.set_page_config(
    page_title="TrueSkill Simulator",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown("""
    <h1 style='text-align: center; color:  #2c3e50;'>
        🎮 TrueSkill Matchmaking Simulator
    </h1>
    <p style='text-align:  center; font-size: 18px; color: #7f8c8d;'>
        Visualisez comment TrueSkill apprend la vraie compétence des joueurs ! 
    </p>
    <hr style='margin-bottom: 30px;'>
""", unsafe_allow_html=True)

# Sidebar - Configuration
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("---")

# Mode de création des joueurs
mode = st.sidebar.radio(
    "Mode de création",
    ["🎯 Joueurs prédéfinis (10 joueurs)", "🎲 Joueurs aléatoires"],
    help="Choisissez comment créer les joueurs"
)

# Paramètres selon le mode
if mode == "🎲 Joueurs aléatoires":
    num_players = st.sidebar.slider(
        "Nombre de joueurs",
        min_value=4,
        max_value=15,
        value=8,
        step=1,
        help="Plus il y a de joueurs, plus la simulation est intéressante"
    )
    
    col1, col2 = st. sidebar.columns(2)
    with col1:
        min_skill = st.slider("Compétence min", 10, 25, 15)
    with col2:
        max_skill = st.slider("Compétence max", 25, 40, 35)
else: 
    num_players = 10
    min_skill = 10
    max_skill = 35
    st.sidebar.info("📊 10 joueurs avec compétences prédéfinies (du Pro au Débutant)")

st.sidebar.markdown("---")

# Nombre de matchs
num_matches = st.sidebar.slider(
    "Nombre de matchs à simuler",
    min_value=20,
    max_value=500,
    value=150,
    step=10,
    help="Plus de matchs = meilleure convergence (mais plus lent)"
)

# Options avancées
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Options avancées")

show_uncertainty = st.sidebar.checkbox("Afficher les intervalles de confiance", value=True)
show_heatmap = st.sidebar.checkbox("Afficher la heatmap de matchmaking", value=True)
show_stats = st.sidebar.checkbox("Afficher les statistiques détaillées", value=True)

# Seed pour la reproductibilité
use_seed = st.sidebar.checkbox("Utiliser un seed (résultats reproductibles)", value=False)
if use_seed:
    seed_value = st.sidebar.number_input("Seed", min_value=0, max_value=9999, value=42)
else:
    seed_value = None

st.sidebar.markdown("---")

# Bouton principal de simulation
if st.sidebar.button("🚀 LANCER LA SIMULATION", type="primary"):
    # Initialiser le seed si nécessaire
    if seed_value is not None:
        random.seed(seed_value)
    
    # Créer les joueurs
    with st.spinner("🎲 Création des joueurs... "):
        if mode == "🎲 Joueurs aléatoires":
            players = create_random_players(num_players, min_skill, max_skill)
        else:
            players = create_tiered_players()
        
        time.sleep(0.5)  # Pour l'effet visuel
    
    st.success(f"✅ {len(players)} joueurs créés !")
    
    # Afficher les joueurs créés
    with st.expander("👥 Voir les joueurs créés (avec leurs vraies compétences cachées)", expanded=False):
        player_data = []
        for p in sorted(players, key=lambda x: x.true_skill, reverse=True):
            player_data.append({
                "Nom": p.name,
                "Vraie Compétence (cachée)": f"{p.true_skill:.1f}",
                "TrueSkill Initial (μ)": f"{p.rating.mu:.1f}",
                "Incertitude Initiale (σ)": f"{p.rating.sigma:.2f}"
            })
        st.dataframe(pd.DataFrame(player_data), use_container_width=True, hide_index=True)
    
    # Simuler les matchs
    simulator = MatchSimulator(players)
    
    st.markdown("---")
    st.subheader("⚔️ Simulation en cours...")
    
    # Barre de progression
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simuler par batches pour mettre à jour la progression
    batch_size = 10
    for i in range(0, num_matches, batch_size):
        batch_end = min(i + batch_size, num_matches)
        
        # Simuler le batch
        for j in range(batch_end - i):
            p1, p2 = random.sample(players, 2)
            simulator.simulate_1v1(p1, p2)
        
        # Mettre à jour la progression
        progress = batch_end / num_matches
        progress_bar.progress(progress)
        status_text.text(f"Match {batch_end}/{num_matches} simulé...")
    
    progress_bar.progress(1.0)
    status_text.text(f"✅ {num_matches} matchs simulés avec succès !")
    time.sleep(0.5)
    
    st.success("🎉 Simulation terminée !")
    
    # Sauvegarder dans session_state pour persistance
    st.session_state['players'] = players
    st.session_state['simulator'] = simulator
    st.session_state['simulation_done'] = True

# Afficher les résultats si une simulation a été lancée
if 'simulation_done' in st.session_state and st.session_state['simulation_done']:
    players = st.session_state['players']
    simulator = st.session_state['simulator']
    
    st.markdown("---")
    st.markdown("## 📊 Résultats de la Simulation")
    
    # Métriques clés
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_sigma = sum(p.rating.sigma for p in players) / len(players)
        st.metric(
            label="📉 Incertitude Moyenne",
            value=f"{avg_sigma:.2f}",
            delta=f"{8.33 - avg_sigma:.2f}",
            delta_color="inverse",
            help="σ moyen (plus c'est bas, mieux c'est)"
        )
    
    with col2:
        total_matches = sum(p.matches_played for p in players) // 2
        st.metric(
            label="⚔️ Total de Matchs",
            value=total_matches,
            help="Nombre total de matchs joués"
        )
    
    with col3:
        sorted_by_trueskill = sorted(players, key=lambda p: p.rating.mu, reverse=True)
        sorted_by_true = sorted(players, key=lambda p: p.true_skill, reverse=True)
        accuracy = sum(1 for i in range(len(players)) 
                      if sorted_by_trueskill[i].name == sorted_by_true[i].name) / len(players)
        st.metric(
            label="🎯 Précision Classement",
            value=f"{accuracy:.0%}",
            help="% de positions correctes dans le classement"
        )
    
    with col4:
        avg_matches_per_player = sum(p.matches_played for p in players) / len(players)
        st.metric(
            label="🎮 Matchs/Joueur",
            value=f"{avg_matches_per_player:.0f}",
            help="Moyenne de matchs par joueur"
        )
    
    st.markdown("---")
    
    # Graphiques principaux
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Convergence", "🏆 Classement", "🔥 Heatmap", "📊 Stats"])
    
    with tab1:
        st.subheader("Convergence de TrueSkill")
        
        col1, col2 = st. columns(2)
        
        with col1:
            st.markdown("#### Convergence de μ (compétence)")
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            for player in players:
                ax1.plot(player.history_mu, label=f"{player.name} (vrai={player.true_skill:.0f})", linewidth=2, alpha=0.7)
                ax1.axhline(y=player.true_skill, linestyle='--', alpha=0.3)
            ax1.set_xlabel("Nombre de matchs", fontsize=12)
            ax1.set_ylabel("Compétence estimée (μ)", fontsize=12)
            ax1.legend(fontsize=8, loc='best')
            ax1.grid(alpha=0.3)
            st.pyplot(fig1)
            plt.close()
            
            st.info("💡 Les courbes convergent vers les lignes pointillées (vraie compétence)")
        
        with col2:
            st. markdown("#### Diminution de σ (incertitude)")
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            for player in players:
                ax2.plot(player.history_sigma, label=player.name, linewidth=2, alpha=0.7)
            ax2.axhline(y=8.333, linestyle=':', color='red', alpha=0.5, linewidth=2, label='σ initial')
            ax2.set_xlabel("Nombre de matchs", fontsize=12)
            ax2.set_ylabel("Incertitude (σ)", fontsize=12)
            ax2.legend(fontsize=8, loc='best')
            ax2.grid(alpha=0.3)
            st.pyplot(fig2)
            plt.close()
            
            st.info("💡 Plus σ diminue, plus le système est confiant")
    
    with tab2:
        st.subheader("🏆 Classement Final")
        
        # Tableau de classement
        leaderboard = sorted(players, key=lambda p:  p.conservative_rating, reverse=True)
        
        ranking_data = []
        for rank, player in enumerate(leaderboard, 1):
            # Emoji selon le rang
            if rank == 1:
                emoji = "🥇"
            elif rank == 2:
                emoji = "🥈"
            elif rank == 3:
                emoji = "🥉"
            else:
                emoji = f"{rank}."
            
            ranking_data. append({
                "Rang":  emoji,
                "Joueur": player.name,
                "TrueSkill (μ)": f"{player.rating.mu:.1f}",
                "Incertitude (σ)": f"{player.rating.sigma:.2f}",
                "Rating Conserv.": f"{player.conservative_rating:.1f}",
                "Vraie Compét.": f"{player.true_skill:.1f}",
                "W/L": f"{player.wins}/{player.losses}",
                "Taux Victoire": f"{player.win_rate:.0f}%"
            })
        
        st.dataframe(
            pd.DataFrame(ranking_data),
            use_container_width=True,
            hide_index=True
        )
        
        # Graphique comparatif
        st.markdown("#### Comparaison :   TrueSkill vs Vraie Compétence")
        
        fig3, ax3 = plt. subplots(figsize=(12, 6))
        
        sorted_by_ts = sorted(players, key=lambda p: p.rating.mu, reverse=True)
        names = [p.name for p in sorted_by_ts]
        mus = [p.rating.mu for p in sorted_by_ts]
        true_skills = [p.true_skill for p in sorted_by_ts]
        
        x = range(len(players))
        width = 0.35
        
        ax3.bar([i - width/2 for i in x], mus, width, label='TrueSkill (μ)', color='steelblue', alpha=0.8)
        ax3.bar([i + width/2 for i in x], true_skills, width, label='Vraie Compétence', color='coral', alpha=0.8)
        
        if show_uncertainty:
            sigmas = [p.rating.sigma * 3 for p in sorted_by_ts]
            ax3.errorbar([i - width/2 for i in x], mus, yerr=sigmas, 
                        fmt='none', ecolor='black', capsize=5, alpha=0.5)
        
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=45, ha='right')
        ax3.set_ylabel('Compétence', fontsize=12)
        ax3.set_title('Classement TrueSkill vs Vraie Compétence', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(alpha=0.3, axis='y')
        
        st.pyplot(fig3)
        plt.close()
    
    with tab3:
        if show_heatmap:
            st.subheader("🔥 Heatmap de Matchmaking")
            st.info("💡 Cette heatmap montre les probabilités de victoire et la qualité des matchs potentiels")
            
            # Générer et afficher la heatmap
            import numpy as np
            from scipy.stats import norm
            from trueskill import quality_1vs1
            import seaborn as sns
            
            n = len(players)
            win_probs = np.zeros((n, n))
            match_quality = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        win_probs[i][j] = np.nan
                        match_quality[i][j] = np.nan
                    else:
                        delta_mu = players[i].rating.mu - players[j].rating.mu
                        sum_sigma = players[i].rating.sigma**2 + players[j].rating.sigma**2
                        beta = 25/6
                        win_probs[i][j] = norm.cdf(delta_mu / np.sqrt(2 * beta**2 + sum_sigma))
                        match_quality[i][j] = quality_1vs1(players[i].rating, players[j].rating)
            
            col1, col2 = st. columns(2)
            
            with col1:
                st. markdown("#### Probabilités de Victoire")
                fig4, ax4 = plt.subplots(figsize=(8, 7))
                mask = np.eye(n, dtype=bool)
                sns.heatmap(win_probs, annot=True, fmt='.0%', cmap='RdYlGn',
                           xticklabels=[p.name for p in players],
                           yticklabels=[p.name for p in players],
                           cbar_kws={'label': 'P(victoire)'},
                           ax=ax4, vmin=0, vmax=1, mask=mask, annot_kws={'size': 8})
                ax4.set_title('Ligne vs Colonne', fontsize=12, fontweight='bold')
                plt. xticks(rotation=45, ha='right', fontsize=9)
                plt.yticks(rotation=0, fontsize=9)
                st.pyplot(fig4)
                plt.close()
            
            with col2:
                st. markdown("#### Qualité des Matchs")
                fig5, ax5 = plt.subplots(figsize=(8, 7))
                sns.heatmap(match_quality, annot=True, fmt='.0%', cmap='Blues',
                           xticklabels=[p.name for p in players],
                           yticklabels=[p.name for p in players],
                           cbar_kws={'label': 'Qualité'},
                           ax=ax5, vmin=0, vmax=1, mask=mask, annot_kws={'size': 8})
                ax5.set_title('100% = parfaitement équilibré', fontsize=12, fontweight='bold')
                plt.xticks(rotation=45, ha='right', fontsize=9)
                plt.yticks(rotation=0, fontsize=9)
                st.pyplot(fig5)
                plt.close()
        else:
            st.info("✋ Heatmap désactivée.  Activez-la dans les options avancées.")
    
    with tab4:
        if show_stats:
            st. subheader("📊 Statistiques Détaillées")
            
            col1, col2 = st. columns(2)
            
            with col1:
                st. markdown("#### Distribution des Compétences")
                fig6, ax6 = plt.subplots(figsize=(8, 6))
                
                mus = [p.rating.mu for p in players]
                true_skills = [p.true_skill for p in players]
                
                ax6.hist(true_skills, bins=10, alpha=0.5, label='Vraie Compétence', color='coral', edgecolor='black')
                ax6.hist(mus, bins=10, alpha=0.5, label='TrueSkill (μ)', color='steelblue', edgecolor='black')
                ax6.set_xlabel('Compétence', fontsize=12)
                ax6.set_ylabel('Nombre de joueurs', fontsize=12)
                ax6.set_title('Distribution', fontsize=14, fontweight='bold')
                ax6.legend()
                ax6.grid(alpha=0.3, axis='y')
                st.pyplot(fig6)
                plt.close()
            
            with col2:
                st. markdown("#### Évolution de l'Incertitude Moyenne")
                fig7, ax7 = plt.subplots(figsize=(8, 6))
                
                # Calculer la moyenne de sigma à chaque étape
                max_len = max(len(p.history_sigma) for p in players)
                avg_sigma_history = []
                
                for i in range(max_len):
                    sigmas_at_i = [p.history_sigma[i] for p in players if i < len(p.history_sigma)]
                    avg_sigma_history.append(sum(sigmas_at_i) / len(sigmas_at_i))
                
                ax7.plot(avg_sigma_history, linewidth=3, color='purple')
                ax7.axhline(y=8.333, linestyle='--', color='red', alpha=0.5, label='σ initial')
                ax7.fill_between(range(len(avg_sigma_history)), avg_sigma_history, alpha=0.3, color='purple')
                ax7.set_xlabel('Nombre de matchs', fontsize=12)
                ax7.set_ylabel('σ moyen', fontsize=12)
                ax7.set_title('Convergence de l\'Incertitude Globale', fontsize=14, fontweight='bold')
                ax7.legend()
                ax7.grid(alpha=0.3)
                st.pyplot(fig7)
                plt.close()
            
            # Stats additionnelles
            st.markdown("#### 🔢 Statistiques Numériques")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Compétence (μ)**")
                st. write(f"• Moyenne: {sum(p.rating.mu for p in players) / len(players):.2f}")
                st.write(f"• Min: {min(p.rating. mu for p in players):.2f}")
                st.write(f"• Max: {max(p.rating.mu for p in players):.2f}")
            
            with col2:
                st.markdown("**Incertitude (σ)**")
                st.write(f"• Moyenne: {avg_sigma:.2f}")
                st.write(f"• Min: {min(p.rating.sigma for p in players):.2f}")
                st.write(f"• Max: {max(p.rating.sigma for p in players):.2f}")
            
            with col3:
                st.markdown("**Matchs**")
                st.write(f"• Total: {total_matches}")
                st.write(f"• Par joueur (moy): {avg_matches_per_player:.0f}")
                st.write(f"• Max par joueur: {max(p. matches_played for p in players)}")
        else:
            st.info("✋ Statistiques désactivées. Activez-les dans les options avancées.")
    
    # Bouton de réinitialisation
    st.markdown("---")
    if st.button("🔄 Nouvelle Simulation"):
        del st.session_state['simulation_done']
        del st.session_state['players']
        del st.session_state['simulator']
        st.rerun()

else:
    # Affichage initial (avant simulation)
    st.info("👈 Configurez les paramètres dans la barre latérale et cliquez sur **LANCER LA SIMULATION**")
    
    # Section explicative
    st.markdown("---")
    st.markdown("## 📚 Qu'est-ce que TrueSkill ?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Le Concept
        
        TrueSkill est un système de classement probabiliste développé par **Microsoft** pour Xbox Live. 
        
        Contrairement à ELO (échecs), TrueSkill gère l'**incertitude**. 
        """)
    
    with col2:
        st. markdown("""
        ### 📊 Les Paramètres
        
        Chaque joueur a :
        - **μ (mu)** : compétence estimée
        - **σ (sigma)** : incertitude
        
        Au fil des matchs, μ converge et σ diminue.
        """)
    
    with col3:
        st.markdown("""
        ### ✅ Avantages
        
        - Gère les équipes (2v2, 5v5)
        - Matchmaking équilibré
        - Confiance statistique
        - Convergence rapide
        """)
    
    st.markdown("---")
    
    # Exemple visuel
    st.markdown("## 🎬 Aperçu du Résultat")
    st.image("https://via.placeholder.com/1200x400/4CAF50/FFFFFF?text=Exemple+de+Graphiques+TrueSkill", 
             caption="Les graphiques de convergence, classement et heatmap s'afficheront ici après la simulation")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>🎓 Projet réalisé dans le cadre du cours <strong>MSMIN5IN43 - EPF 2025</strong></p>
        <p>📖 Basé sur le système <a href='https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/' target='_blank'>TrueSkill de Microsoft Research</a></p>
    </div>
""", unsafe_allow_html=True)
