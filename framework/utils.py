import copy
import typing

import numpy as np

from framework.game import *


def indicator(x: bool):
    return 1.0 if x else 0.0


def construct_P_pi(
        i: Player, pi_i: Policy, pi_minus_i: JointPolicy, P: ProbabilityTransitionKernel,
        S_list: typing.List[State], A: ActionProfileSet
):
    P_pi = np.zeros(shape=(len(S_list), len(S_list)))
    for j1 in range(len(S_list)):
        for j2 in range(len(S_list)):
            P_pi[j1, j2] = sum(
                P[(S_list[j1], a, S_list[j2])]
                * pi_i[(S_list[j1], a[i])]
                * pi_minus_i.prob(S_list[j1], a.minus(i))
                for a in A
            )
    return P_pi


def construct_r_pi(
        i: Player, pi_i: Policy, pi_minus_i: JointPolicy, R: RewardFunction,
        S_list: typing.List[State], A: ActionProfileSet
):
    r = np.zeros(len(S_list))
    for j in range(len(S_list)):
        r[j] = sum(
            R.get_reward(i, S_list[j], a)
            * pi_i[(S_list[j], a[i])]
            * pi_minus_i.prob(S_list[j], a.minus(i))
            for a in A
        )
    return r


def value_iteration(i: Player, pi_minus_i: JointPolicy, P: ProbabilityTransitionKernel, R: RewardFunction,
                    S_list: typing.List[State], A: ActionProfileSet, delta: float, T: int = int(1e5)):
    Ai_list = list(A[i])

    P_reduced = np.zeros(shape=(len(S_list), len(Ai_list), len(S_list)))
    for j1 in range(len(S_list)):
        for j2 in range(len(Ai_list)):
            for j3 in range(len(S_list)):
                P_reduced[j1, j2, j3] = sum(
                    P[(S_list[j1], ActionProfile.merge(Ai_list[j2], a_minus_i), S_list[j3])]
                    * pi_minus_i.prob(S_list[j1], a_minus_i)
                    for a_minus_i in A.minus(i)
                )

    R_reduced = np.zeros(shape=(len(S_list), len(Ai_list)))
    for j1 in range(len(S_list)):
        for j2 in range(len(A[i])):
            R_reduced[j1, j2] = sum(
                R.get_reward(i, S_list[j1], ActionProfile.merge(Ai_list[j2], a_minus_i))
                * pi_minus_i.prob(S_list[j1], a_minus_i)
                for a_minus_i in A.minus(i)
            )

    V_opt_i = np.zeros(shape=(len(S_list), ))
    V_opt_i_history = []

    for _ in range(T):
        new_V_opt_i = copy.deepcopy(V_opt_i)
        for j1 in range(len(S_list)):
            new_V_opt_i[j1] = max(
                R_reduced[j1, j2] + delta * sum(
                    P_reduced[j1, j2, j3]
                    * V_opt_i[j3]
                    for j3 in range(len(S_list))
                )
                for j2 in range(len(Ai_list))
            )
        V_opt_i_history.append(V_opt_i)
        V_opt_i = new_V_opt_i
    return V_opt_i_history


__all__ = ["indicator", "construct_P_pi", "construct_r_pi", "value_iteration"]
