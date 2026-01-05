# 6. Interface Web Interactive (Streamlit)

## 🌐 Vue d'Ensemble

L'interface web permet une **démonstration interactive** du système TrueSkill en temps réel, idéale pour : 
- ✅ Présentation en live devant un public
- ✅ Expérimentation avec différents paramètres
- ✅ Visualisation instantanée des résultats
- ✅ Compréhension intuitive du système

**Technologie** : Streamlit (framework Python pour applications data science)

---

## 🚀 Lancement de l'Application

```bash
# Installation
pip install streamlit

# Lancement
streamlit run app.py

# L'app s'ouvre automatiquement sur http://localhost:8501
```

---

## 🎨 Architecture de l'Interface

### Structure Principale

```
┌─────────────────────────────────────────────┐
│           TITRE & DESCRIPTION               │
├──────────────┬──────────────────────────────┤
│   SIDEBAR    │      CONTENU PRINCIPAL       │
│              │                              │
│ Paramètres:   │  Tabs:                        │
│ • Mode       │  📈 Convergence              │
│ • Joueurs    │  🏆 Classement               │
│ • Matchs     │  🔥 Heatmap                  │
│ • Options    │  📊 Statistiques             │
│              │                              │
│ [LANCER]     │  Graphiques + Tableaux       │
└──────────────┴──────────────────────────────┘
```

---

## 🎛️ Panneau de Configuration (Sidebar)

### 1. Mode de Création des Joueurs

```python
mode = st.sidebar.radio(
    "Mode de création",
    ["🎯 Joueurs prédéfinis (10 joueurs)", "🎲 Joueurs aléatoires"]
)
```

**Joueurs prédéfinis** :  10 joueurs avec compétences fixes
- ProGamer (35), Champion (33), Veteran (28), etc.
- Permet des résultats **reproductibles**

**Joueurs aléatoires** : Compétences tirées aléatoirement
- Permet de **tester différents scénarios**

### 2. Paramètres Dynamiques

```python
if mode == "🎲 Joueurs aléatoires":
    num_players = st.sidebar.slider("Nombre de joueurs", 4, 15, 8)
    
    col1, col2 = st. sidebar.columns(2)
    with col1:
        min_skill = st.slider("Compétence min", 10, 25, 15)
    with col2:
        max_skill = st.slider("Compétence max", 25, 40, 35)
```

**Sliders interactifs** : 
- Nombre de joueurs :  4 à 15
- Compétence min/max :  Définit la dispersion

### 3. Nombre de Matchs

```python
num_matches = st.sidebar.slider(
    "Nombre de matchs à simuler",
    min_value=20,
    max_value=500,
    value=150,
    step=10,
    help="Plus de matchs = meilleure convergence (mais plus lent)"
)
```

**Impact** : 
- 20 matchs :  Convergence partielle (~30%)
- 100 matchs : Bonne convergence (~60%)
- 500 matchs : Convergence maximale (~70%)

### 4. Options Avancées

```python
show_uncertainty = st.sidebar.checkbox("Afficher les intervalles de confiance", value=True)
show_heatmap = st.sidebar.checkbox("Afficher la heatmap de matchmaking", value=True)
show_stats = st.sidebar.checkbox("Afficher les statistiques détaillées", value=True)

use_seed = st.sidebar.checkbox("Utiliser un seed (résultats reproductibles)", value=False)
if use_seed:
    seed_value = st.sidebar.number_input("Seed", min_value=0, max_value=9999, value=42)
```

**Fonctionnalités** :
- ✅ Toggle pour activer/désactiver certaines visualisations
- ✅ Seed pour reproductibilité (important pour démos)

---

## 🎬 Processus de Simulation

### 1. Bouton de Lancement

```python
if st.sidebar.button("🚀 LANCER LA SIMULATION", type="primary"):
    # Initialiser le seed si nécessaire
    if seed_value is not None:
        random.seed(seed_value)
    
    # Créer les joueurs
    with st.spinner("🎲 Création des joueurs..."):
        players = create_random_players(num_players, min_skill, max_skill)
        time.sleep(0.5)  # Effet visuel
    
    st.success(f"✅ {len(players)} joueurs créés !")
```

### 2. Affichage des Joueurs Créés

```python
with st.expander("👥 Voir les joueurs créés", expanded=False):
    player_data = []
    for p in sorted(players, key=lambda x: x.true_skill, reverse=True):
        player_data.append({
            "Nom": p.name,
            "Vraie Compétence (cachée)": f"{p.true_skill:.1f}",
            "TrueSkill Initial (μ)": f"{p.rating.mu:.1f}",
            "Incertitude Initiale (σ)": f"{p.rating.sigma:.2f}"
        })
    st.dataframe(pd.DataFrame(player_data), use_container_width=True)
```

**Expander** : Section repliable pour ne pas encombrer l'interface

### 3. Barre de Progression

```python
st.subheader("⚔️ Simulation en cours...")
progress_bar = st.progress(0)
status_text = st.empty()

# Simuler par batches
batch_size = 10
for i in range(0, num_matches, batch_size):
    batch_end = min(i + batch_size, num_matches)
    
    # Simuler le batch
    for j in range(batch_end - i):
        p1, p2 = random.sample(players, 2)
        simulator. simulate_1v1(p1, p2)
    
    # Mettre à jour la progression
    progress = batch_end / num_matches
    progress_bar.progress(progress)
    status_text.text(f"Match {batch_end}/{num_matches} simulé...")

status_text.text(f"✅ {num_matches} matchs simulés avec succès !")
```

**UX** :
- Barre visuelle (0% → 100%)
- Texte dynamique ("Match 50/150...")
- Feedback immédiat

### 4. Sauvegarde dans Session State

```python
# Sauvegarder pour persistance (éviter de recalculer)
st.session_state['players'] = players
st.session_state['simulator'] = simulator
st.session_state['simulation_done'] = True
```

**Session State** : Les données persistent entre interactions (changements d'onglets, etc.)

---

## 📊 Affichage des Résultats

### Métriques Clés (Cartes)

```python
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_sigma = sum(p.rating.sigma for p in players) / len(players)
    st.metric(
        label="📉 Incertitude Moyenne",
        value=f"{avg_sigma:.2f}",
        delta=f"{8.33 - avg_sigma:.2f}",
        delta_color="inverse",  # Rouge si augmente, vert si diminue
        help="σ moyen (plus c'est bas, mieux c'est)"
    )

with col2:
    total_matches = sum(p.matches_played for p in players) // 2
    st.metric(label="⚔️ Total de Matchs", value=total_matches)

with col3:
    # Calculer la précision
    sorted_by_ts = sorted(players, key=lambda p: p.rating.mu, reverse=True)
    sorted_by_true = sorted(players, key=lambda p: p.true_skill, reverse=True)
    accuracy = sum(1 for i in range(len(players)) 
                  if sorted_by_ts[i].name == sorted_by_true[i].name) / len(players)
    st.metric(label="🎯 Précision Classement", value=f"{accuracy:.0%}")

with col4:
    avg_matches = sum(p.matches_played for p in players) / len(players)
    st.metric(label="🎮 Matchs/Joueur", value=f"{avg_matches:.0f}")
```

**Métriques Streamlit** :
- Valeur principale (grande)
- Delta (changement, avec couleur)
- Tooltip d'aide

---

## 📑 Système d'Onglets

### Tab 1 :  Convergence

```python
tab1, tab2, tab3, tab4 = st.tabs(["📈 Convergence", "🏆 Classement", "🔥 Heatmap", "📊 Stats"])

with tab1:
    st.subheader("Convergence de TrueSkill")
    
    col1, col2 = st. columns(2)
    
    with col1:
        st.markdown("#### Convergence de μ (compétence)")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        for player in players:
            ax1.plot(player.history_mu, label=f"{player.name}", linewidth=2)
            ax1.axhline(y=player.true_skill, linestyle='--', alpha=0.3)
        ax1.set_xlabel("Nombre de matchs")
        ax1.set_ylabel("Compétence estimée (μ)")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)
        plt.close()
        
        st.info("💡 Les courbes convergent vers les lignes pointillées")
    
    with col2:
        st.markdown("#### Diminution de σ (incertitude)")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for player in players: 
            ax2.plot(player. history_sigma, label=player.name, linewidth=2)
        ax2.axhline(y=8.333, linestyle=':', color='red', alpha=0.5)
        ax2.set_xlabel("Nombre de matchs")
        ax2.set_ylabel("Incertitude (σ)")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)
        plt.close()
        
        st.info("💡 Plus σ diminue, plus le système est confiant")
```

**Layout** :  2 colonnes pour afficher μ et σ côte à côte

### Tab 2 : Classement

```python
with tab2:
    st.subheader("🏆 Classement Final")
    
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
    
    st.dataframe(pd.DataFrame(ranking_data), use_container_width=True, hide_index=True)
```

**Tableau interactif** : 
- Emojis 🥇🥈🥉 pour le podium
- Toutes les statistiques
- Triable par colonne

### Tab 3 :  Heatmap

```python
with tab3:
    if show_heatmap:
        st.subheader("🔥 Heatmap de Matchmaking")
        st.info("💡 Probabilités de victoire et qualité des matchs")
        
        # Calculer les matrices
        n = len(players)
        win_probs = np.zeros((n, n))
        match_quality = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Probabilité
                    delta_mu = players[i].rating. mu - players[j].rating. mu
                    sum_sigma = players[i].rating.sigma**2 + players[j].rating.sigma**2
                    beta = 25/6
                    win_probs[i][j] = norm.cdf(delta_mu / np.sqrt(2*beta**2 + sum_sigma))
                    
                    # Qualité
                    match_quality[i][j] = quality_1vs1(players[i].rating, players[j].rating)
        
        col1, col2 = st. columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 7))
            sns.heatmap(win_probs, annot=True, fmt='.0%', cmap='RdYlGn',
                       xticklabels=[p.name for p in players],
                       yticklabels=[p.name for p in players],
                       ax=ax, vmin=0, vmax=1)
            ax.set_title('Probabilités de Victoire')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 7))
            sns.heatmap(match_quality, annot=True, fmt='.0%', cmap='Blues',
                       xticklabels=[p.name for p in players],
                       yticklabels=[p.name for p in players],
                       ax=ax, vmin=0, vmax=1)
            ax.set_title('Qualité des Matchs')
            st.pyplot(fig)
```

### Tab 4 : Statistiques

```python
with tab4:
    if show_stats:
        st. subheader("📊 Statistiques Détaillées")
        
        col1, col2 = st. columns(2)
        
        with col1:
            st. markdown("#### Distribution des Compétences")
            fig, ax = plt.subplots(figsize=(8, 6))
            
            mus = [p.rating.mu for p in players]
            true_skills = [p.true_skill for p in players]
            
            ax.hist(true_skills, bins=10, alpha=0.5, label='Vraie', color='coral')
            ax.hist(mus, bins=10, alpha=0.5, label='TrueSkill', color='steelblue')
            ax.set_xlabel('Compétence')
            ax.set_ylabel('Nombre de joueurs')
            ax.legend()
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### Évolution σ Moyen")
            # ...  (graphique de σ moyen au fil du temps)
        
        # Stats numériques
        st.markdown("#### 🔢 Statistiques Numériques")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Compétence (μ)**")
            st.write(f"• Moyenne:  {np.mean(mus):.2f}")
            st.write(f"• Min: {np.min(mus):.2f}")
            st.write(f"• Max: {np.max(mus):.2f}")
```

---

## 🎨 Personnalisation CSS

### Style Personnalisé

```python
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    . stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color:  white;
        font-size: 18px;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)
```

**Personnalisations** :
- Fond gris clair
- Bouton vert large
- Effets hover

---

## 🔄 Bouton de Réinitialisation

```python
if st.button("🔄 Nouvelle Simulation"):
    # Effacer le session state
    del st.session_state['simulation_done']
    del st. session_state['players']
    del st.session_state['simulator']
    
    # Recharger l'app
    st.rerun()
```

---

## 📱 Responsive Design

Streamlit est **automatiquement responsive** :
- Desktop :  Layout large (sidebar + contenu)
- Mobile : Layout vertical (sidebar se replie)

---

## 🚀 Déploiement (Optionnel)

### Streamlit Cloud (Gratuit)

```bash
# 1. Push sur GitHub
git push origin main

# 2. Aller sur streamlit.io/cloud
# 3. Connecter le repo
# 4. L'app est déployée !  
# → URL publique : https://username-app.streamlit.app
```

**Avantages** :
- ✅ Gratuit
- ✅ HTTPS automatique
- ✅ Accessible de n'importe où

---

## 🎯 Scénarios d'Utilisation

### 1. Présentation en Classe

```
Vous :  "Je vais maintenant vous montrer en LIVE."
      *Ouvre l'app*
      *Choisit 8 joueurs, 150 matchs*
      *Clique sur LANCER*
      
      → Barre de progression apparaît (effet dramatique)
      → Graphiques se génèrent en 10 secondes
      
Vous : "Regardez, après 150 matchs, TrueSkill a retrouvé
       le classement avec 62% de précision !"
      *Montre l'onglet Classement*
      
Prof : "Et si on doublait le nombre de matchs ?"

Vous : *Change le slider à 300, relance*
      → Nouvelle simulation en 15 secondes
      "Précision monte à 75% !"
      
Prof : "Impressionnant ! 🤯"
```

### 2. Expérimentation

```
Étudiant 1 : "Que se passe-t-il avec seulement 4 joueurs ?"
            *Change slider à 4*
            *Lance*
            
Étudiant 2 : "Et si les compétences sont très rapprochées ?"
            *Min skill = 20, Max skill = 25*
            *Lance*
            
→ Permet d'explorer différents scénarios facilement
```

---

## 📊 Performance

### Temps de Simulation

| Joueurs | Matchs | Temps |
|---------|--------|-------|
| 4       | 50     | ~2s   |
| 8       | 150    | ~10s  |
| 12      | 300    | ~30s  |
| 15      | 500    | ~60s  |

**Optimisations possibles** :
- Parallélisation (multiprocessing)
- Cache des calculs (@st.cache_data)
- Réduction de la fréquence de mise à jour de la barre de progression

---

## 🐛 Gestion d'Erreurs

```python
try:
    # Simulation
    simulator. simulate_random_matches(num_matches)
    st.success("✅ Simulation terminée !")
except Exception as e: 
    st.error(f"❌ Erreur lors de la simulation : {str(e)}")
    st.exception(e)  # Stack trace pour debug
```

---

## 📚 Ressources Streamlit

- [Documentation officielle](https://docs.streamlit.io/)
- [Galerie d'exemples](https://streamlit.io/gallery)
- [Cheat Sheet](https://docs.streamlit.io/library/cheatsheet)

---

**→ Prochaine section : [Résultats](07-RESULTS.md)**
