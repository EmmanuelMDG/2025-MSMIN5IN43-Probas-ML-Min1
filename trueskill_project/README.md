# TrueSkill & Matchmaking — Mini projet (implémentation simple)

Ce mini-projet fournit une **implémentation pédagogique** de TrueSkill (mise à jour de `mu` et `sigma`) pour :
- matchs **1v1** (par défaut),
- matchs **équipes** (liste de joueurs par équipe),
- **nuls** via un `draw_margin`,
- **dynamique temporelle** (skill drift) via `tau`,
- visualisation de la **convergence de l’incertitude** (`sigma`) au fil des matchs.

> Remarque : TrueSkill “complet” s’appuie sur des graphes de facteurs et Expectation Propagation.
> Ici, on code une version “formules fermées” utilisée couramment pour 1v1/équipes (pédagogique et suffisante pour ce mini-projet).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows PowerShell

pip install -r requirements.txt
```

## Lancer une démo

Simulation 1v1 (les “vraies” compétences sont cachées, on observe uniquement les résultats) :

```bash
python examples/simulate_1v1.py
```

Cela génère un graphique `examples/output_simulation.png` :
- une courbe par joueur (mu) + son intervalle d’incertitude (± 3*sigma)
- un graphe séparé de la convergence de `sigma` (plus il est bas, plus l’algorithme est confiant)

Vous verrez également un petit résumé de la dernière mise à jour **EP** (c, v, w, variations de mu/sigma).

## API rapide

```python
from trueskill_simple.core import TrueSkillEnv

env = TrueSkillEnv()
alice = env.make_player("Alice")
bob = env.make_player("Bob")

# Alice bat Bob
report = env.rate(teams=[[alice], [bob]], ranks=[0, 1], return_report=True)[0]
print(report.v, report.w, report.delta_mu)

print(alice.mu, alice.sigma)
print(bob.mu, bob.sigma)
```

## Paramètres importants

- `mu0`, `sigma0` : initialisation des joueurs
- `beta` : bruit de performance (plus grand => matchs plus “chanceux”)
- `tau` : drift temporel (sigma augmente légèrement avant chaque match)
- `draw_probability` / `draw_margin` : gestion du nul
- `trust` : inspiration TrueSkill 2, ajoute une variance de performance proportionnelle à l'incertitude (sigma²) des joueurs

## Notes sur l'inférence (EP)

La mise à jour des scores s'appuie sur **Expectation Propagation** sur le graphe de facteurs TrueSkill (facteurs de performance, somme d'équipe, différence, troncature pour la victoire ou le nul). L'appel `return_report=True` renvoie un `EPUpdateReport` (c, t, v, w, epsilon, delta_mu, nouvelles sigmas) pour instrumenter ou comparer différentes variantes (équipes hétérogènes, draw margin, drift temporel via `tau`).

## Ressources

- TrueSkill (Herbrich, Minka, Graepel 2006) : https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/
- TrueSkill 2 (Minka et al.) : https://www.microsoft.com/en-us/research/publication/trueskill-2-improved-bayesian-skill-rating-system/
- Chapitre TrueSkill dans MBML : https://mbmlbook.com/TrueSkill.html

## Interface Streamlit (démo interactive)

1) Installe les dépendances (Streamlit/Pandas ajoutés dans `requirements.txt`) :
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2) Lance l’interface :
   ```bash
   PYTHONPATH=. streamlit run examples/app_streamlit.py
   ```
3) Dans le navigateur :
   - ajoute des joueurs ;
   - simule des matchs (win/lose/nul), avec une ou plusieurs personnes par équipe ;
   - vois les courbes μ/σ et le dernier rapport EP (c, t, v, w, delta_mu, sigma).
