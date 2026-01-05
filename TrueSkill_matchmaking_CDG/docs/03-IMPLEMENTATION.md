# 3. Implémentation Technique

## 🏗️ Architecture du Projet

```
trueskill-matchmaking/
├── src/
│   ├── player.py          # Classe Player (TrueSkill)
│   ├── simulator.py       # Simulation de matchs
│   ├── elo. py            # Implémentation ELO
│   ├── utils.py          # Fonctions utilitaires
│   └── visualizer.py     # Génération de graphiques
├── main.py               # Script de démonstration
├── demo_visualizations.py
├── demo_comparison.py
├── comparison. py         # Comparaison TrueSkill vs ELO
├── app.py               # Interface Streamlit
├── requirements.txt
└── results/             # Graphiques générés
```

---

## 🎮 Classe Player (TrueSkill)

### Structure de Base

```python
from trueskill import Rating

class Player:
    def __init__(self, name, true_skill):
        self.name = name
        self.true_skill = true_skill  # Compétence cachée (référence)
        self.rating = Rating()         # μ=25, σ=8.33
        
        # Historique pour visualisations
        self.history_mu = [self.rating.mu]
        self.history_sigma = [self. rating.sigma]
        
        # Statistiques
        self.matches_played = 0
        self.wins = 0
        self.losses = 0
```

### Méthodes Clés

#### 1. Simuler une Performance

```python
def play_match(self, beta=25/6):
    """
    Simule la performance du joueur
    Performance = Vraie Compétence + Aléa
    """
    return random.gauss(self.true_skill, beta)
```

**Explication** :
- La performance est tirée d'une gaussienne
- Centrée sur `true_skill`
- Avec un écart-type `beta` (représente la chance)

#### 2. Mettre à Jour le Rating

```python
def update_rating(self, new_rating):
    """
    Met à jour μ et σ après un match
    """
    self.rating = new_rating
    self.history_mu.append(new_rating.mu)
    self.history_sigma.append(new_rating.sigma)
    self.matches_played += 1
```

#### 3. Rating Conservateur

```python
@property
def conservative_rating(self):
    """
    μ - 3σ :  utilisé pour le classement
    """
    return self.rating.mu - 3 * self.rating.sigma
```

---

## ⚔️ Classe MatchSimulator

### Simulation d'un Match 1v1

```python
from trueskill import rate_1vs1

def simulate_1v1(self, player1, player2):
    # 1. Simuler les performances
    perf1 = player1.play_match()
    perf2 = player2.play_match()
    
    # 2. Déterminer le gagnant
    if perf1 > perf2:
        winner, loser = player1, player2
    else:
        winner, loser = player2, player1
    
    # 3. Mettre à jour les ratings TrueSkill
    if winner == player1:
        new_r1, new_r2 = rate_1vs1(player1.rating, player2.rating)
    else:
        new_r2, new_r1 = rate_1vs1(player2.rating, player1.rating)
    
    player1.update_rating(new_r1)
    player2.update_rating(new_r2)
    
    # 4. Mettre à jour les stats
    winner.record_win()
    loser.record_loss()
    
    return winner, loser
```

### Simulation de Plusieurs Matchs

```python
def simulate_random_matches(self, num_matches):
    """
    Simule des matchs aléatoires
    """
    for i in range(num_matches):
        # Choisir 2 joueurs aléatoirement
        player1, player2 = random.sample(self.players, 2)
        
        # Simuler le match
        self.simulate_1v1(player1, player2)
```

---

## 📊 Implémentation ELO (Comparaison)

### Classe EloPlayer

```python
class EloPlayer:
    def __init__(self, name, true_skill, initial_rating=1500, k_factor=32):
        self.name = name
        self.true_skill = true_skill
        self.rating = initial_rating  # Un seul nombre (pas de σ)
        self.history = [initial_rating]
        self.k_factor = k_factor  # Sensibilité (32 = standard)
```

### Formule ELO

```python
def expected_score(self, opponent):
    """
    Probabilité de victoire
    """
    return 1 / (1 + 10**((opponent.rating - self.rating) / 400))

def update_rating(self, opponent, won):
    """
    Mise à jour après match
    """
    expected = self.expected_score(opponent)
    actual = 1.0 if won else 0.0
    
    # Formule ELO
    self.rating += self.k_factor * (actual - expected)
    self.history.append(self.rating)
```

---

## 🎨 Visualisations

### 1. Convergence de μ

```python
import matplotlib.pyplot as plt

def plot_skill_convergence(players):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for player in players:
        # Courbe de l'estimation
        ax.plot(player.history_mu, label=player.name, linewidth=2.5)
        
        # Ligne de la vraie compétence
        ax.axhline(y=player.true_skill, linestyle='--', alpha=0.4)
    
    ax.set_xlabel('Nombre de matchs')
    ax.set_ylabel('Compétence estimée (μ)')
    ax.set_title('Convergence de TrueSkill')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.savefig('results/convergence_mu.png', dpi=300)
    plt.show()
```

### 2. Diminution de σ

```python
def plot_uncertainty_decrease(players):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for player in players: 
        ax.plot(player.history_sigma, label=player.name, linewidth=2.5)
    
    ax.axhline(y=8.333, linestyle=':', color='red', label='σ initial')
    ax.set_xlabel('Nombre de matchs')
    ax.set_ylabel('Incertitude (σ)')
    ax.set_title('Diminution de l\'Incertitude')
    ax.legend()
    
    plt.savefig('results/convergence_sigma.png', dpi=300)
    plt.show()
```

### 3. Heatmap de Matchmaking

```python
import seaborn as sns
from scipy.stats import norm
from trueskill import quality_1vs1

def plot_matchmaking_heatmap(players):
    n = len(players)
    win_probs = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Probabilité de victoire
                delta_mu = players[i].rating. mu - players[j].rating. mu
                sum_sigma = players[i].rating.sigma**2 + players[j].rating.sigma**2
                beta = 25/6
                win_probs[i][j] = norm.cdf(delta_mu / np.sqrt(2*beta**2 + sum_sigma))
    
    sns.heatmap(win_probs, annot=True, fmt='.0%', cmap='RdYlGn',
                xticklabels=[p.name for p in players],
                yticklabels=[p.name for p in players])
    
    plt.title('Probabilités de Victoire')
    plt.savefig('results/heatmap. png', dpi=300)
    plt.show()
```

---

## 🔄 Comparaison TrueSkill vs ELO

### Simulation Parallèle

```python
def run_parallel_simulation(ts_players, elo_players, num_matches, seed=42):
    """
    Lance les MÊMES matchs pour les deux systèmes
    """
    random.seed(seed)
    
    for i in range(num_matches):
        # Choisir les mêmes paires
        idx1, idx2 = random. sample(range(len(ts_players)), 2)
        
        # Même performance
        beta = 25/6
        perf1 = random.gauss(ts_players[idx1].true_skill, beta)
        perf2 = random.gauss(ts_players[idx2].true_skill, beta)
        
        # Même gagnant
        winner_idx = idx1 if perf1 > perf2 else idx2
        loser_idx = idx2 if winner_idx == idx1 else idx1
        
        # Mettre à jour TrueSkill
        ts_winner = ts_players[winner_idx]
        ts_loser = ts_players[loser_idx]
        # ... (mise à jour TrueSkill)
        
        # Mettre à jour ELO
        elo_winner = elo_players[winner_idx]
        elo_loser = elo_players[loser_idx]
        elo_winner.update_rating(elo_loser, won=True)
        elo_loser.update_rating(elo_winner, won=False)
```

### Métriques de Comparaison

```python
def calculate_ranking_accuracy(ts_players, elo_players):
    # Classement par vraie compétence (référence)
    true_ranking = sorted(ts_players, key=lambda p:  p.true_skill, reverse=True)
    
    # Classement TrueSkill
    ts_ranking = sorted(ts_players, key=lambda p: p.rating.mu, reverse=True)
    
    # Classement ELO
    elo_ranking = sorted(elo_players, key=lambda p: p.rating, reverse=True)
    
    # Précision (% positions exactes)
    ts_accuracy = sum(1 for i in range(len(true_ranking)) 
                     if ts_ranking[i].name == true_ranking[i].name) / len(true_ranking)
    
    elo_accuracy = sum(1 for i in range(len(true_ranking)) 
                      if elo_ranking[i].name == true_ranking[i].name) / len(true_ranking)
    
    return ts_accuracy, elo_accuracy
```

---

## 🌐 Interface Streamlit

### Structure de l'App

```python
import streamlit as st

# Configuration
st.set_page_config(page_title="TrueSkill Simulator", layout="wide")

# Sidebar - Paramètres
with st.sidebar:
    num_players = st.slider("Nombre de joueurs", 4, 15, 8)
    num_matches = st.slider("Nombre de matchs", 20, 500, 150)
    
    if st.button("🚀 LANCER"):
        # Créer les joueurs
        players = create_random_players(num_players)
        
        # Simuler avec barre de progression
        progress_bar = st.progress(0)
        simulator = MatchSimulator(players)
        
        for i in range(num_matches):
            p1, p2 = random.sample(players, 2)
            simulator.simulate_1v1(p1, p2)
            progress_bar.progress((i+1) / num_matches)
        
        # Sauvegarder dans session
        st.session_state['players'] = players

# Affichage des résultats
if 'players' in st.session_state:
    players = st.session_state['players']
    
    # Onglets
    tab1, tab2, tab3 = st. tabs(["Convergence", "Classement", "Heatmap"])
    
    with tab1:
        # Graphique de convergence
        fig, ax = plt.subplots()
        for p in players:
            ax.plot(p.history_mu, label=p.name)
        st.pyplot(fig)
    
    with tab2:
        # Tableau de classement
        leaderboard = sorted(players, key=lambda p:  p.rating.mu, reverse=True)
        st.dataframe(leaderboard)
```

---

## ⚙️ Technologies Utilisées

| Technologie | Version | Utilisation |
|-------------|---------|-------------|
| **Python** | 3.12 | Langage principal |
| **trueskill** | 0.4.5 | Implémentation TrueSkill |
| **NumPy** | ≥1.26.0 | Calculs numériques |
| **Matplotlib** | ≥3.7.1 | Visualisations |
| **Seaborn** | ≥0.12.2 | Heatmaps |
| **SciPy** | ≥1.10.1 | Statistiques |
| **Pandas** | ≥2.0.2 | Manipulation de données |
| **Streamlit** | Latest | Interface web |

---

## 🚀 Guide d'Installation

```bash
# 1. Cloner le projet
git clone <repo>
cd trueskill-matchmaking

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Tester l'installation
python -c "import trueskill; print('✅ OK')"
```

---

## 📝 Exemple d'Utilisation

```python
# Créer des joueurs
from src.utils import create_tiered_players
from src.simulator import MatchSimulator

players = create_tiered_players()

# Simuler 100 matchs
simulator = MatchSimulator(players)
simulator.simulate_random_matches(100)

# Afficher le classement
simulator.print_leaderboard()

# Générer les visualisations
from src.visualizer import create_all_visualizations
create_all_visualizations(players)
```

---

**→ Prochaine section : [Visualisations](04-VISUALIZATIONS.md)**
