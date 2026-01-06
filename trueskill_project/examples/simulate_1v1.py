import random
from pathlib import Path

import matplotlib.pyplot as plt

from trueskill_simple.core import TrueSkillEnv


def play_match(true_skill_a: float, true_skill_b: float, beta: float) -> int:
    """
    Return: 0 if A wins, 1 if B wins, 2 if draw
    performance = skill + noise, noise ~ N(0, beta^2)
    """
    pa = random.gauss(true_skill_a, beta)
    pb = random.gauss(true_skill_b, beta)
    diff = pa - pb
    eps = 0.1 * beta  # simple draw threshold for the simulator
    if abs(diff) <= eps:
        return 2
    return 0 if diff > 0 else 1


def main(seed: int = 7, n_matches: int = 90):
    random.seed(seed)

    env = TrueSkillEnv(draw_probability=0.10)
    alice = env.make_player("Alice")
    bob = env.make_player("Bob")
    clara = env.make_player("Clara")

    # Hidden true skills (unknown to the model)
    true = {"Alice": 30.0, "Bob": 25.0, "Clara": 20.0}
    players = {p.name: p for p in [alice, bob, clara]}

    history = {name: {"mu": [], "sigma": []} for name in players}
    ep_reports = []

    pairs = [("Alice", "Bob"), ("Alice", "Clara"), ("Bob", "Clara")]

    for _ in range(n_matches):
        a_name, b_name = random.choice(pairs)
        A, B = players[a_name], players[b_name]

        outcome = play_match(true[a_name], true[b_name], beta=env.beta)
        if outcome == 0:
            report = env.rate([[A], [B]], ranks=[0, 1], return_report=True)[0]
        elif outcome == 1:
            report = env.rate([[A], [B]], ranks=[1, 0], return_report=True)[0]
        else:
            report = env.rate([[A], [B]], ranks=[0, 0], return_report=True)[0]
        ep_reports.append(report)

        for name, p in players.items():
            history[name]["mu"].append(p.mu)
            history[name]["sigma"].append(p.sigma)

    # Plot
    fig, (ax_mu, ax_sigma) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    xs = list(range(1, n_matches + 1))
    for name in players:
        mu = history[name]["mu"]
        sig = history[name]["sigma"]
        ax_mu.plot(xs, mu, label=f"{name} (mu)")
        lower = [m - 3*s for m, s in zip(mu, sig)]
        upper = [m + 3*s for m, s in zip(mu, sig)]
        ax_mu.fill_between(xs, lower, upper, alpha=0.15)
        ax_sigma.plot(xs, sig, label=f"{name} (sigma)")

    # true skill lines
    for name, val in true.items():
        ax_mu.axhline(val, linestyle="--")

    ax_mu.set_title("TrueSkill — EP update, mu ± 3σ")
    ax_mu.set_ylabel("Skill estimate")
    ax_mu.legend()
    ax_sigma.set_title("Convergence de l'incertitude (sigma)")
    ax_sigma.set_xlabel("Match #")
    ax_sigma.set_ylabel("Sigma")
    ax_sigma.legend()
    fig.tight_layout()

    out = Path(__file__).resolve().parent / "output_simulation.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved: {out}")

    # Show a couple of EP details for inspection
    last = ep_reports[-1]
    print("Dernière mise à jour EP :")
    print(f"  draw={last.is_draw}, eps={last.eps:.3f}, c={last.c:.3f}, t={last.t:.3f}, v={last.v:.3f}, w={last.w:.3f}")
    for name in sorted(last.delta_mu):
        print(f"  {name}: Δmu={last.delta_mu[name]:+.3f}, sigma={last.new_sigma[name]:.3f}")


if __name__ == "__main__":
    main()
