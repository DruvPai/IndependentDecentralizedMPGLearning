import math
import numbers
import typing
import pathlib
import statistics

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
PLOT_SPACING = 1000


def plot_on_time_logscale(quantities: typing.Dict[str, typing.Tuple[typing.List[numbers.Number]]],
                          stdevs: typing.Dict[str, typing.Tuple[typing.List[numbers.Number]]],
                          title: str, xlabel: str, file: pathlib.Path,
                          include_yticks=False, include_legend=False):
    plt.title(title, fontdict=LARGE_FONTDICT)
    if not include_yticks:
        plt.yticks([0])
    plt.axhline(0.0, linestyle='--', color='black')
    plt.xscale("log")
    plt.xlabel(xlabel, fontdict=LARGE_FONTDICT)
    for (idx, entry) in enumerate(quantities):
        plt.plot(
            quantities[entry][0],
            quantities[entry][1],
            label=entry, color=f"C{idx}"
        )
        if stdevs:
            plt.fill_between(
                stdevs[entry][0],
                [quantities[entry][1][i] - stdevs[entry][1][i] for i in range(len(quantities[entry][1]))],
                [quantities[entry][1][i] + stdevs[entry][1][i] for i in range(len(quantities[entry][1]))],
                color=f"C{idx}", alpha=0.2
            )
    if include_legend:
        plt.legend()
    plt.savefig(file)
    plt.close()


def plot_policy_convergence_l1(games: typing.List[StochasticGame], pi_histories: typing.List[typing.List[JointPolicy]],
                               result_dir: pathlib.Path):
    N_trials = len(games)
    K = len(pi_histories[0])
    game = games[0]
    I = game.I
    S = game.S
    A = game.A
    l1_distances_mean = dict()
    l1_distances_stdev = dict()
    for i in I:
        l1_distances_mean_i = []
        l1_distances_stdev_i = []
        times_i = []
        for k in range(0, K, K // PLOT_SPACING):
            l1_distances_ik = []
            for j in range(N_trials):
                pi_i_K = pi_histories[j][-1][i]
                pi_i_k = pi_histories[j][k][i]
                l1_distance = sum(abs(pi_i_k[(s, a_i)] - pi_i_K[(s, a_i)]) for s in S for a_i in A[i])
                l1_distances_ik.append(l1_distance)
            times_i.append(k)
            l1_distances_mean_i.append(statistics.mean(l1_distances_ik))
            l1_distances_stdev_i.append(statistics.stdev(l1_distances_ik))

        l1_distances_mean[str(i)] = (times_i, l1_distances_mean_i)
        l1_distances_stdev[str(i)] = (times_i, l1_distances_stdev_i)

    plot_on_time_logscale(
        quantities=l1_distances_mean, stdevs=l1_distances_stdev,
        title="$\|\pi_{i}^{k} - \pi_{i}^{K}\|_{1}$", xlabel="$k$",
        file=result_dir / "pi_l1.jpg",
    )


def plot_local_Q_convergence_l1(games: typing.List[StochasticGame],
                                q_tilde_histories: typing.List[typing.List[JointPolicy]],
                                result_dir: pathlib.Path):
    N_trials = len(games)
    K = len(q_tilde_histories[0])
    game = games[0]
    I = game.I
    S = game.S
    A = game.A
    l1_distances_mean = dict()
    l1_distances_stdev = dict()
    for i in I:
        l1_distances_mean_i = []
        l1_distances_stdev_i = []
        times_i = []
        for k in range(0, K, K // PLOT_SPACING):
            l1_distances_ik = []
            for j in range(N_trials):
                q_tilde_i_K = q_tilde_histories[j][-1][i]
                q_tilde_i_k = q_tilde_histories[j][k][i]
                l1_distance = sum(abs(q_tilde_i_k[(s, a_i)] - q_tilde_i_K[(s, a_i)]) for s in S for a_i in A[i])
                l1_distances_ik.append(l1_distance)
            times_i.append(k)
            l1_distances_mean_i.append(statistics.mean(l1_distances_ik))
            l1_distances_stdev_i.append(statistics.stdev(l1_distances_ik))
        l1_distances_mean[str(i)] = (times_i, l1_distances_mean_i)
        l1_distances_stdev[str(i)] = (times_i, l1_distances_stdev_i)
    plot_on_time_logscale(
        quantities=l1_distances_mean, stdevs=l1_distances_stdev,
        title="$\|\\tilde{q}_{i}^{k} - \\tilde{q}_{i}^{K}\|_{1}$", xlabel="$k$",
        file=result_dir / "q_tilde_l1.jpg",
    )


def plot_value_iteration_convergence_l1(games: typing.List[StochasticGame],
                                        value_iteration_histories: typing.List[typing.Dict[Player, typing.List[np.array]]],
                                        result_dir: pathlib.Path):
    N_trials = len(games)
    T = len(value_iteration_histories[0])
    game = games[0]
    I = game.I
    l1_distances_mean = dict()
    l1_distances_stdev = dict()
    for i in I:
        l1_distances_mean_i = []
        l1_distances_stdev_i = []
        times_i = []
        for t in range(T):
            l1_distances_ik = []
            for j in range(N_trials):
                V_i_T = value_iteration_histories[j][i][-1]
                V_i_t = value_iteration_histories[j][i][t]
                l1_distances_ik.append(np.linalg.norm(V_i_T - V_i_t, ord=1))
            times_i.append(t)
            l1_distances_mean_i.append(statistics.mean(l1_distances_ik))
            l1_distances_stdev_i.append(statistics.stdev(l1_distances_ik))
        l1_distances_mean[str(i)] = (times_i, l1_distances_mean_i)
        l1_distances_stdev[str(i)] = (times_i, l1_distances_stdev_i)
    plot_on_time_logscale(quantities=l1_distances_mean, stdevs=l1_distances_stdev,
                          title="Value iteration convergence", xlabel="$t$",
                          file=result_dir / "aux_VI_convergence.jpg")


def plot_policy_convergence_to_nash_l1(games: typing.List[StochasticGame], pi_histories: typing.List[typing.List[JointPolicy]],
                                       result_dir: pathlib.Path):
    N_trials = len(games)
    K = len(pi_histories[0])
    game = games[0]
    I = game.I
    S = game.S
    A = game.A
    P = game.P
    R = game.R
    delta = game.delta
    T = int(1e5)
    S_list = list(S)
    vi_histories = []
    for j in range(N_trials):
        v_opt_i_histories = dict()
        pi_opt = pi_histories[j][-1]
        for i in I:
            v_opt_i_history = value_iteration(i, pi_opt.minus(i), game.P, game.R, S_list, A, game.delta, T)
            v_opt_i_histories[i] = v_opt_i_history
        vi_histories.append(v_opt_i_histories)
    plot_value_iteration_convergence_l1(games, vi_histories, result_dir)
    l1_distances_mean = dict()
    l1_distances_stdev = dict()
    for i in I:
        l1_distances_mean_i = []
        l1_distances_stdev_i = []
        times_i = []
        for k in range(0, K, K // PLOT_SPACING):
            l1_distances_ik = []
            for j in range(N_trials):
                V_opt_i = vi_histories[j][i][-1]
                pi_opt = pi_histories[j][-1]
                pi_k = pi_histories[j][k]
                P_pi = construct_P_pi(i, pi_k[i], pi_opt.minus(i), P, S_list, A)
                r_pi = construct_r_pi(i, pi_k[i], pi_opt.minus(i), R, S_list, A)
                V_k_i = np.linalg.inv(np.eye(len(S_list)) - delta * P_pi) @ r_pi
                l1_distance = np.linalg.norm(V_k_i - V_opt_i, ord=1)
                l1_distances_ik.append(l1_distance)
            times_i.append(k)
            l1_distances_mean_i.append(statistics.mean(l1_distances_ik))
            l1_distances_stdev_i.append(statistics.stdev(l1_distances_ik))
        l1_distances_mean[str(i)] = (times_i, l1_distances_mean_i)
        l1_distances_stdev[str(i)] = (times_i, l1_distances_stdev_i)
    plot_on_time_logscale(quantities=l1_distances_mean, stdevs=l1_distances_stdev,
                          title="$\|V_{i}(\pi_{i}^{k}, \pi_{-i}^{K}) - V_{i}(\pi_{i}^{\star}, \pi_{-i}^{K})\|_{1}$",
                          xlabel="$k$", file=result_dir / "nash_l1.jpg", include_yticks=True)
