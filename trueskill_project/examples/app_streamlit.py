"""
Interface Streamlit pour piloter TrueSkill en interactif.

Commandes :
    PYTHONPATH=. streamlit run examples/app_streamlit.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from trueskill_simple.core import TrueSkillEnv


def _init_state() -> None:
    """Initialise ou réinitialise l'état Streamlit (environnement + historique)."""
    params = st.session_state.get("params") or {
        "mu0": 25.0,
        "sigma0": 25.0 / 3.0,
        "beta": 25.0 / 6.0,
        "tau": 25.0 / 300.0,
        "draw_probability": 0.10,
        "trust": 0.0,
    }
    st.session_state.params = params
    st.session_state.env = TrueSkillEnv(**params)
    st.session_state.players = {}
    st.session_state.history = {}  # name -> {"mu": [], "sigma": []}
    st.session_state.match_idx = []  # indices (1..N)
    st.session_state.ep_reports = []
    st.session_state.match_records = []  # historique des matchs (affichage)


def _sidebar_params() -> None:
    st.sidebar.header("Paramètres TrueSkill")
    p = st.session_state.params
    p["mu0"] = st.sidebar.number_input("mu0", value=float(p["mu0"]))
    p["sigma0"] = st.sidebar.number_input("sigma0", value=float(p["sigma0"]))
    p["beta"] = st.sidebar.number_input("beta (bruit perf)", value=float(p["beta"]))
    p["tau"] = st.sidebar.number_input("tau (drift)", value=float(p["tau"]))
    p["draw_probability"] = st.sidebar.number_input(
        "Probabilité de nul", min_value=0.0, max_value=0.5, value=float(p["draw_probability"]), step=0.01
    )
    p["trust"] = st.sidebar.number_input(
        "Trust (variance perf dépend de sigma)", min_value=0.0, max_value=5.0, value=float(p["trust"]), step=0.1
    )

    if st.sidebar.button("Réinitialiser l'environnement"):
        _init_state()
        st.experimental_rerun()


def _add_player_form():
    st.subheader("Ajouter un joueur")
    name = st.text_input("Nom du joueur")
    if st.button("Ajouter", key="add_player_btn"):
        if not name:
            st.warning("Nom vide.")
        elif name in st.session_state.players:
            st.warning("Ce joueur existe déjà.")
        else:
            env = st.session_state.env
            player = env.make_player(name)
            st.session_state.players[name] = player
            st.session_state.history[name] = {"mu": [player.mu], "sigma": [player.sigma]}
            st.success(f"Joueur {name} créé.")


def _run_match():
    st.subheader("Simuler un match")
    players = list(st.session_state.players.keys())
    if len(players) < 2:
        st.info("Ajoute au moins deux joueurs pour lancer un match.")
        return

    st.caption("Sélectionne un ou plusieurs joueurs par équipe (ils jouent ensemble).")
    col1, col2 = st.columns(2)
    with col1:
        team_a_names = st.multiselect("Equipe A", players, key="team_a")
    with col2:
        team_b_names = st.multiselect("Equipe B", players, key="team_b")

    outcome = st.selectbox("Résultat", ["A gagne", "B gagne", "Nul"])

    if st.button("Lancer le match"):
        if not team_a_names or not team_b_names:
            st.warning("Chaque équipe doit avoir au moins un joueur.")
            return
        if set(team_a_names) & set(team_b_names):
            st.warning("Un joueur ne peut pas être dans les deux équipes.")
            return
        env = st.session_state.env
        team_a = [st.session_state.players[n] for n in team_a_names]
        team_b = [st.session_state.players[n] for n in team_b_names]

        ranks = {
            "A gagne": [0, 1],
            "B gagne": [1, 0],
            "Nul": [0, 0],
        }[outcome]

        report = env.rate([team_a, team_b], ranks=ranks, return_report=True)[0]
        st.session_state.ep_reports.append(report)

        # Historique
        st.session_state.match_idx.append(len(st.session_state.match_idx) + 1)
        for name, p in st.session_state.players.items():
            st.session_state.history[name]["mu"].append(p.mu)
            st.session_state.history[name]["sigma"].append(p.sigma)

        st.session_state.match_records.append(
            {
                "Match": len(st.session_state.match_idx),
                "Equipe A": ", ".join(team_a_names),
                "Equipe B": ", ".join(team_b_names),
                "Résultat": outcome,
            }
        )

        st.success("Match appliqué.")


def _tables():
    st.subheader("Joueurs / scores courants")
    st.caption(f"Matchs joués: {len(st.session_state.match_idx)}")
    rows = []
    for name, p in st.session_state.players.items():
        rows.append({"Joueur": name, "mu": round(p.mu, 3), "sigma": round(p.sigma, 3)})
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Joueur"))
    else:
        st.info("Aucun joueur pour l'instant.")

    if st.session_state.ep_reports:
        last = st.session_state.ep_reports[-1]
        st.markdown("**Dernier rapport EP**")
        st.json(
            {
                "draw": last.is_draw,
                "eps": round(last.eps, 4),
                "c": round(last.c, 4),
                "t": round(last.t, 4),
                "v": round(last.v, 4),
                "w": round(last.w, 4),
                "delta_mu": {k: round(v, 4) for k, v in last.delta_mu.items()},
                "sigma": {k: round(v, 4) for k, v in last.new_sigma.items()},
            }
        )

    st.subheader("Historique des matchs")
    if st.session_state.match_records:
        st.dataframe(pd.DataFrame(st.session_state.match_records).set_index("Match"))
    else:
        st.info("Aucun match enregistré.")


def _charts():
    st.subheader("Courbes μ et σ")
    if not st.session_state.match_idx:
        st.info("Lance un match pour voir les courbes.")
        return

    xs = st.session_state.match_idx
    # μ
    mu_df = pd.DataFrame({"Match": xs})
    sigma_df = pd.DataFrame({"Match": xs})
    for name, hist in st.session_state.history.items():
        # hist inclut l'état initial en position 0 ; on aligne sur len(xs)+1
        mu_vals = hist["mu"][1:]
        sig_vals = hist["sigma"][1:]

        # Si un joueur a été ajouté après plusieurs matchs, on propage la dernière valeur
        # pour garder les DataFrames alignés.
        if len(mu_vals) < len(xs):
            mu_vals = mu_vals + [mu_vals[-1]] * (len(xs) - len(mu_vals))
            sig_vals = sig_vals + [sig_vals[-1]] * (len(xs) - len(sig_vals))

        mu_df[name] = mu_vals
        sigma_df[name] = sig_vals

    st.line_chart(mu_df.set_index("Match"))
    st.line_chart(sigma_df.set_index("Match"))


def main():
    st.title("TrueSkill interactif (EP)")
    st.caption("Ajoute des joueurs, simule des matchs, inspecte les rapports EP.")
    if "env" not in st.session_state:
        _init_state()

    _sidebar_params()
    _add_player_form()
    _run_match()
    _tables()
    _charts()


if __name__ == "__main__":
    main()
