from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import math

# -----------------------------
# Normal pdf/cdf helpers
# -----------------------------
_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)

def _phi(x: float) -> float:
    '''Densité normale standard (PDF).'''
    return math.exp(-0.5 * x * x) / _SQRT_2PI

def _Phi(x: float) -> float:
    '''CDF de la normale standard (via erf).'''
    return 0.5 * (1.0 + math.erf(x / _SQRT2))

def _v_truncate(t: float) -> float:
    '''v(t) pour victoire/défaite (Gaussienne tronquée).'''
    denom = _Phi(t)
    if denom < 1e-12:
        return -t  # garde-fou numérique
    return _phi(t) / denom

def _w_truncate(t: float) -> float:
    '''w(t) pour victoire/défaite (Gaussienne tronquée).'''
    v = _v_truncate(t)
    return v * (v + t)

def _v_draw(t: float, eps: float) -> float:
    '''v(t, eps) pour le cas nul.'''
    a = -eps - t
    b = eps - t
    denom = _Phi(b) - _Phi(a)
    if denom < 1e-12:
        return 0.0
    return (_phi(a) - _phi(b)) / denom

def _w_draw(t: float, eps: float) -> float:
    '''w(t, eps) pour le cas nul.'''
    a = -eps - t
    b = eps - t
    denom = _Phi(b) - _Phi(a)
    if denom < 1e-12:
        return 0.0
    v = _v_draw(t, eps)
    return v * v + (b * _phi(b) - a * _phi(a)) / denom

def _draw_margin(beta: float, draw_probability: float) -> float:
    '''
    Convertit une probabilité de nul en marge epsilon dans l’espace de différence de performance.

    Cas 2 joueurs : diff ~ N(0, 2*beta^2) si compétences égales.
    On cherche epsilon tel que P(|diff| <= eps) = draw_probability.
    '''
    target = (draw_probability + 1.0) / 2.0

    # Inverse CDF via binary search
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _Phi(mid) < target:
            lo = mid
        else:
            hi = mid
    z = (lo + hi) / 2.0
    return z * math.sqrt(2.0) * beta


@dataclass
class Player:
    name: str
    mu: float = 25.0
    sigma: float = 25.0 / 3.0

    def __repr__(self) -> str:
        return f"Player(name={self.name!r}, mu={self.mu:.3f}, sigma={self.sigma:.3f})"


@dataclass
class EPUpdateReport:
    '''Stocke une mise à jour EP pour inspection/visualisation.'''
    is_draw: bool
    eps: float
    c: float
    t: float
    v: float
    w: float
    team_winner: List[str]
    team_loser: List[str]
    delta_mu: Dict[str, float]
    new_sigma: Dict[str, float]


class TrueSkillEnv:
    '''
    Environnement TrueSkill pédagogique (EP) :
    - 1v1 et équipes (facteur somme d'équipe)
    - nuls via marge de troncature (draw margin)
    - drift temporel (tau) comme dans MBML/TrueSkill
    - option "trust" inspirée de TrueSkill 2 : variance de performance augmente avec l'incertitude
    '''

    def __init__(
        self,
        mu0: float = 25.0,
        sigma0: float = 25.0 / 3.0,
        beta: float = 25.0 / 6.0,
        tau: float = 25.0 / 300.0,
        draw_probability: float = 0.10,
        trust: float = 0.0,
    ) -> None:
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.beta = beta
        self.tau = tau
        self.draw_probability = draw_probability
        # trust>0 applique une variance de perf additionnelle proportionnelle à sigma^2 (cf. TrueSkill 2)
        self.trust = trust
        self.eps = _draw_margin(beta=self.beta, draw_probability=self.draw_probability)

    def make_player(self, name: str) -> Player:
        return Player(name=name, mu=self.mu0, sigma=self.sigma0)

    def _apply_drift(self, player: Player) -> None:
        # MBML / TrueSkill : facteur dynamique (tau) augmente l'incertitude avec le temps.
        player.sigma = math.sqrt(player.sigma * player.sigma + self.tau * self.tau)

    def rate(
        self,
        teams: Sequence[Sequence[Player]],
        ranks: Sequence[int],
        *,
        return_report: bool = False,
    ) -> Optional[List[EPUpdateReport]]:
        '''
        Met à jour les joueurs en place (EP sur facteur graph).

        teams : liste d'équipes, chaque équipe est une liste de Player
        ranks : plus petit = meilleur. Exemples :
            - victoire 1v1 pour team0 : ranks=[0,1]
            - nul : ranks=[0,0]
        return_report : si True, renvoie une liste de EPUpdateReport décrivant l'étape d'inférence.
        '''
        if len(teams) != len(ranks):
            raise ValueError("teams and ranks must have same length")
        if len(teams) < 2:
            raise ValueError("Need at least 2 teams")

        # Applique le drift temporel à tous les joueurs.
        for team in teams:
            for p in team:
                self._apply_drift(p)

        reports: List[EPUpdateReport] = []

        # Free-for-all multi-équipes : décomposer en paires adjacentes par rang
        if len(teams) > 2:
            order = sorted(range(len(teams)), key=lambda i: ranks[i])
            for i in range(len(order) - 1):
                a, b = order[i], order[i + 1]
                ra, rb = ranks[a], ranks[b]
                rep = self._rate_two_teams(teams[a], teams[b], is_draw=(ra == rb), collect=return_report)
                if rep:
                    reports.append(rep)
            return reports if return_report else None

        # Cas à deux équipes exactement
        is_draw = (ranks[0] == ranks[1])
        if is_draw:
            rep = self._rate_two_teams(teams[0], teams[1], is_draw=True, collect=return_report)
            return [rep] if return_report and rep else None

        # S'assurer que la première équipe passée à la mise à jour est la gagnante
        winner_first = ranks[0] < ranks[1]
        if winner_first:
            rep = self._rate_two_teams(teams[0], teams[1], is_draw=False, collect=return_report)
        else:
            rep = self._rate_two_teams(teams[1], teams[0], is_draw=False, collect=return_report)

        if return_report and rep:
            return [rep]
        return None

    def _rate_two_teams(
        self,
        team_winner: Sequence[Player],
        team_loser: Sequence[Player],
        is_draw: bool,
        collect: bool = False,
    ) -> Optional[EPUpdateReport]:
        # Facteur somme d'équipe : somme des mus/sigmas (TrueSkill 2006, facteur "summation").
        mu_w = sum(p.mu for p in team_winner)
        mu_l = sum(p.mu for p in team_loser)

        sig2_w = sum(p.sigma * p.sigma for p in team_winner)
        sig2_l = sum(p.sigma * p.sigma for p in team_loser)

        # Variance de performance : beta^2 par joueur + option "trust" (TrueSkill 2) ~ sigma^2 dépendant.
        perf_noise = 0.0
        for p in team_winner:
            perf_noise += self.beta * self.beta + self.trust * (p.sigma * p.sigma)
        for p in team_loser:
            perf_noise += self.beta * self.beta + self.trust * (p.sigma * p.sigma)

        c = math.sqrt(sig2_w + sig2_l + perf_noise)

        # normalized mean difference
        t = (mu_w - mu_l) / c

        if is_draw:
            # Facteur troncature pour le nul (annexe TrueSkill, MBML chapitre TrueSkill)
            v = _v_draw(t, self.eps / c)
            w = _w_draw(t, self.eps / c)
        else:
            # Facteur troncature victoire/défaite (Herbrich et al. 2006)
            v = _v_truncate(t)
            w = _w_truncate(t)

        # Messages EP vers chaque joueur : gagnant +, perdant - (voir MBML pour les deltas fermés)
        delta_mu: Dict[str, float] = {}
        new_sigma: Dict[str, float] = {}
        for p in team_winner:
            sigma2 = p.sigma * p.sigma
            old_mu = p.mu
            p.mu += (sigma2 / c) * v
            p.sigma = math.sqrt(max(1e-9, sigma2 * (1.0 - (sigma2 / (c * c)) * w)))
            delta_mu[p.name] = p.mu - old_mu
            new_sigma[p.name] = p.sigma

        for p in team_loser:
            sigma2 = p.sigma * p.sigma
            old_mu = p.mu
            p.mu -= (sigma2 / c) * v
            p.sigma = math.sqrt(max(1e-9, sigma2 * (1.0 - (sigma2 / (c * c)) * w)))
            delta_mu[p.name] = p.mu - old_mu
            new_sigma[p.name] = p.sigma

        if not collect:
            return None
        return EPUpdateReport(
            is_draw=is_draw,
            eps=self.eps,
            c=c,
            t=t,
            v=v,
            w=w,
            team_winner=[p.name for p in team_winner],
            team_loser=[p.name for p in team_loser],
            delta_mu=delta_mu,
            new_sigma=new_sigma,
        )
