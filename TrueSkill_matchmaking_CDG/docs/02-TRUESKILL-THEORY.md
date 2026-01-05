# 2. Théorie TrueSkill - Fondements Mathématiques

## 📐 Modèle Mathématique

### Les Deux Paramètres

Chaque joueur *i* est représenté par : 

1. **μᵢ (mu)** : Compétence moyenne estimée
   - Valeur initiale : 25.0
   - Plage typique : 0 à 50

2. **σᵢ (sigma)** : Incertitude (écart-type)
   - Valeur initiale : 8.333
   - Diminue avec le nombre de matchs
   - Plage :  8.333 → ~2.0

### Distribution de Compétence

La compétence d'un joueur suit une **loi normale** : 

```
Compétence_i ~ N(μᵢ, σᵢ²)
```

**Interprétation** :
- On est sûr à **68%** que la vraie compétence est dans [μ - σ, μ + σ]
- On est sûr à **99. 7%** qu'elle est dans [μ - 3σ, μ + 3σ]

---

## 🎮 Modélisation d'un Match

### Performance d'un Joueur

Dans un match, la performance *pᵢ* est : 

```
pᵢ ~ N(μᵢ, σᵢ² + β²)
```

Où **β** (beta) représente la **variance de performance** (chance, forme du jour).
- Valeur par défaut : β = 25/6 ≈ 4.17

### Résultat du Match

Le joueur avec la plus haute performance gagne : 

```
Joueur 1 gagne si :  p₁ > p₂
```

---

## 🔄 Mise à Jour des Ratings

### Principe Bayésien

Après un match, on met à jour les croyances : 

```
P(compétence | résultat) = P(résultat | compétence) × P(compétence) / P(résultat)
```

**En pratique**, TrueSkill utilise l'algorithme **Expectation Propagation** qui calcule de nouvelles valeurs de μ et σ. 

### Formules Simplifiées (1v1)

Pour un match où le joueur 1 bat le joueur 2 :

```python
# Différence de compétence
c = √(2β² + σ₁² + σ₂²)
delta = (μ₁ - μ₂) / c

# Facteur de mise à jour
v = φ(delta) / Φ(delta)  # φ = PDF, Φ = CDF de la loi normale

# Nouveau μ
μ₁_nouveau = μ₁ + (σ₁² / c) × v
μ₂_nouveau = μ₂ - (σ₂² / c) × v

# Nouveau σ (diminue)
σ₁_nouveau = σ₁ × √(1 - σ₁²/c² × w)
σ₂_nouveau = σ₂ × √(1 - σ₂²/c² × w)

avec w = v × (v + delta)
```

**Intuition** : 
- Si un fort joueur bat un faible → petite mise à jour
- Si un faible joueur bat un fort → grosse mise à jour (surprise !)

---

## 📊 Rating Conservateur

Pour le classement, on utilise le **rating conservateur** :

```
Rating_conservateur = μ - 3σ
```

**Pourquoi ?**
- Un nouveau joueur (σ élevé) ne doit pas être classé trop haut
- On pénalise l'incertitude
- Garantie à 99.7% que la vraie compétence est au-dessus

---

## 🎯 Qualité d'un Match

TrueSkill peut prédire si un match sera équilibré : 

```python
def quality_1vs1(rating1, rating2):
    """
    Retourne un score entre 0 (déséquilibré) et 1 (parfait)
    """
    delta_mu = rating1.mu - rating2.mu
    sum_sigma = rating1.sigma² + rating2.sigma² + 2β²
    
    return √(2β² / sum_sigma) × exp(-delta_mu² / (2 × sum_sigma))
```

**Utilisation** :  Matchmaking optimal (chercher qualité ≈ 1)

---

## 🏆 Extension aux Équipes

TrueSkill supporte nativement les matchs en équipe (2v2, 3v3, etc.)

### Compétence d'une Équipe

Pour une équipe composée des joueurs *i* : 

```
μ_équipe = Σ μᵢ
σ²_équipe = Σ σᵢ²
```

### Mise à Jour

Après le match, **tous** les joueurs de chaque équipe voient leur μ et σ mis à jour proportionnellement à leur contribution estimée.

---

## ⚙️ Paramètres du Système

| Paramètre | Symbole | Valeur par défaut | Description |
|-----------|---------|-------------------|-------------|
| Compétence initiale | μ₀ | 25.0 | Point de départ |
| Incertitude initiale | σ₀ | 8.333 | Grande incertitude |
| Variance de performance | β | 4.167 | Chance/aléa |
| Dynamique | τ (tau) | 0.0833 | Les joueurs progressent |
| Probabilité de nul | ε (epsilon) | 0.0 | Marge pour les nuls |

### Dynamique (τ)

Permet de modéliser que les joueurs **progressent ou régressent** :

```
σᵢ_nouveau = √(σᵢ² + τ²)
```

Appliqué avant chaque match (optionnel).

---

## 📈 Convergence Théorique

### Théorème

Sous certaines conditions (matchs variés, pas de biais), TrueSkill **converge** vers la vraie compétence :

```
lim (n→∞) μᵢ = vraie_compétence_i
lim (n→∞) σᵢ = σ_min (≈ 2.0)
```

### Vitesse de Convergence

- **Après 10 matchs** : σ ≈ 6.0 (réduction de 28%)
- **Après 50 matchs** : σ ≈ 3.5 (réduction de 58%)
- **Après 200 matchs** : σ ≈ 2.5 (réduction de 70%)

---

## 🔬 Avantages Théoriques sur ELO

| Aspect | ELO | TrueSkill |
|--------|-----|-----------|
| **Modèle** | Déterministe | Probabiliste bayésien |
| **Incertitude** | ❌ Non géré | ✅ σ explicite |
| **Convergence** | O(n) linéaire | O(log n) logarithmique |
| **Équipes** | ❌ Extension ad-hoc | ✅ Natif |
| **Matchmaking** | Basique (diff. rating) | Optimal (qualité de match) |
| **Intervalle de confiance** | ❌ Aucun | ✅ [μ - 3σ, μ + 3σ] |

---

## 📚 Références Académiques

1. **Herbrich, R., Minka, T., & Graepel, T. (2006)**  
   *"TrueSkill™:  A Bayesian Skill Rating System"*  
   Advances in Neural Information Processing Systems 19  
   [Lien PDF](https://papers.nips.cc/paper/2006/file/f44ee263952e65b3610b8ba51229d1f9-Paper.pdf)

2. **Minka, T.  (2013)**  
   *"TrueSkill 2: An improved Bayesian skill rating system"*  
   Microsoft Research Technical Report  
   [Lien](https://www.microsoft.com/en-us/research/publication/trueskill-2-improved-bayesian-skill-rating-system/)

3. **Winn, J.  & Bishop, C. M. (2005)**  
   *"Variational Message Passing"*  
   Journal of Machine Learning Research

---

## 🧮 Exemple Numérique

### Situation Initiale

- **Alice** : μ = 25, σ = 8.33
- **Bob** : μ = 25, σ = 8.33

### Match :  Alice gagne

**Calculs** :
```
c = √(2×4.17² + 8.33² + 8.33²) ≈ 14.1
delta = (25 - 25) / 14.1 = 0
v ≈ 0.798

Alice :  μ → 25 + (8.33²/14.1) × 0.798 ≈ 28.9
Bob   : μ → 25 - (8.33²/14.1) × 0.798 ≈ 21.1

σ (les deux) :  8.33 → 7.91 (diminue)
```

---

**→ Prochaine section :  [Implémentation](docs/03-IMPLEMENTATION.md)**
