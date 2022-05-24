import math
import numbers
import typing
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from framework.game import *
from framework.q_learning import *
from framework.utils import *


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Times"]
})
LARGE_FONTDICT = {"fontsize": 26}


def plot_on_time_logscale(quantities: typing.Dict[str, typing.Tuple[typing.List[numbers.Number]]],
                          title: str, xlabel: str, file: pathlib.Path, include_yticks=False, include_legend=False):
    plt.title(title, fontdict=LARGE_FONTDICT)
    if not include_yticks:
        plt.yticks([0])
    plt.axhline(0.0, linestyle='--', color='black')
    plt.xscale("log")
    plt.xlabel(xlabel, fontdict=LARGE_FONTDICT)
    for (idx, entry) in enumerate(quantities):
        plt.plot(quantities[entry][0], quantities[entry][1], label=entry, color=f"C{idx}")
    if include_legend:
        plt.legend()
    plt.savefig(file)
    plt.close()


def plot_policy_convergence_l1(game: StochasticGame, pi_history: typing.List[JointPolicy], result_dir: pathlib.Path):
    K = len(pi_history)
    I = game.I
    S = game.S
    A = game.A
    l1_distances = dict()
    for i in I:
        l1_distances_i = []
        times_i = []
        pi_i_K = pi_history[-1][i]
        for k in range(K):
            pi_i_k = pi_history[k][i]
            l1_distance = sum(abs(pi_i_k[(s, a_i)] - pi_i_K[(s, a_i)]) for s in S for a_i in A[i])
            times_i.append(k)
            l1_distances_i.append(l1_distance)
        l1_distances[str(i)] = (times_i, l1_distances_i)
    plot_on_time_logscale(
        quantities=l1_distances, title="$\|\pi_{i}^{k} - \pi_{i}^{K}\|_{1}$", xlabel="$k$",
        file=result_dir / "pi_l1.jpg"
    )


def plot_local_Q_convergence_l1(game: StochasticGame, q_tilde_history: typing.List[JointPolicy],
                                result_dir: pathlib.Path):
    K = len(q_tilde_history)
    I = game.I
    S = game.S
    A = game.A
    l1_distances = dict()
    for i in I:
        l1_distances_i = []
        times_i = []
        q_tilde_i_K = q_tilde_history[-1][i]
        for k in range(K):
            q_tilde_i_k = q_tilde_history[k][i]
            l1_distance = sum(abs(q_tilde_i_k[(s, a_i)] - q_tilde_i_K[(s, a_i)]) for s in S for a_i in A[i])
            times_i.append(k)
            l1_distances_i.append(l1_distance)
        l1_distances[str(i)] = (times_i, l1_distances_i)
    plot_on_time_logscale(
        quantities=l1_distances, title="$\|\\tilde{q}_{i}^{k} - \\tilde{q}_{i}^{K}\|_{1}$", xlabel="$k$",
        file=result_dir / "q_tilde_l1.jpg"
    )


def plot_value_iteration_convergence_l1(game: StochasticGame, T:int,
                                        value_iteration_history: typing.Dict[Player, typing.List[np.array]],
                                        result_dir: pathlib.Path):
    I = game.I
    l1_distances = dict()
    for i in I:
        l1_distances_i = []
        times_i = []
        V_i_T = value_iteration_history[i][-1]
        for t in range(T):
            times_i.append(t)
            l1_distances_i.append(np.linalg.norm(V_i_T - value_iteration_history[i][t], ord=1))
        l1_distances[str(i)] = (times_i, l1_distances_i)
    plot_on_time_logscale(quantities=l1_distances, title="Value iteration convergence", xlabel="$t$",
                          file=result_dir / "aux_VI_convergence.jpg")


def plot_policy_convergence_to_nash_l1(game: StochasticGame, pi_history: typing.List[JointPolicy],
                                       result_dir: pathlib.Path):
    K = len(pi_history)
    I = game.I
    S = game.S
    A = game.A
    P = game.P
    R = game.R
    delta = game.delta
    T = int(1e5)
    S_list = list(S)
    vi_histories = dict()
    pi_opt = pi_history[-1]
    for i in I:
        V_opt_i_history = value_iteration(i, pi_opt.minus(i), game.P, game.R, S_list, A, game.delta, T)
        vi_histories[i] = V_opt_i_history
    plot_value_iteration_convergence_l1(game, T, vi_histories, result_dir)
    l1_distances = dict()
    for i in I:
        l1_distances_i = []
        times_i = []
        V_opt_i = vi_histories[i][-1]
        for k in range(K):
            pi_k = pi_history[k]
            P_pi = construct_P_pi(i, pi_k[i], pi_opt.minus(i), P, S_list, A)
            r_pi = construct_r_pi(i, pi_k[i], pi_opt.minus(i), R, S_list, A)
            V_k_i = np.linalg.inv(np.eye(len(S_list)) - delta * P_pi) @ r_pi
            l1_distance = np.linalg.norm(V_k_i - V_opt_i, ord=1)
            times_i.append(k)
            l1_distances_i.append(l1_distance)
        l1_distances[str(i)] = (times_i, l1_distances_i)
    plot_on_time_logscale(quantities=l1_distances,
                          title="$\|V_{i}(\pi_{i}^{k}, \pi_{-i}^{K}) - V_{i}(\pi_{i}^{\star}, \pi_{-i}^{K})\|_{1}$",
                          xlabel="$k$", file=result_dir / "nash_l1.jpg", include_yticks=True)
