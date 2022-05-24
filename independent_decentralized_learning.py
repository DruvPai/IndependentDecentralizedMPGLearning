import copy
import math
import typing

import tqdm

from framework.game import *
from framework.q_learning import *


def xlogx(x):
    if x == 0.0:
        return 0.0
    else:
        return x * math.log(x)


def independent_decentralized_algo(game: StochasticGame, K: int,
                                   alpha: typing.Callable[[int], float] = lambda n: 1 / (n ** 0.5),
                                   beta: typing.Callable[[int], float] = lambda n: 1 / n,
                                   tau: float = 0.000001
                                   ):
    I = game.I  # player set
    S = game.S  # state set
    A = game.A  # action profile set
    mu = game.mu  # initial state distribution
    P = game.P  # probability transition kernel
    R = game.R  # reward function
    delta = game.delta  # discount factor

    N = {s: 0 for s in S}  # number of times visited state
    N_tilde = {i: {(s, a_i): 0 for s in S for a_i in A[i]} for i in I}

    pi = dict()  # policies
    q_tilde = dict()  # local Q functions
    for i in I:
        pi[i] = Policy(S, A[i], lambda s, ai: 1 / len(A[i]))  # initialize uniform policy
        q_tilde[i] = LocalQFunction(S, A[i], lambda s, ai: 0)  # initialize Q function at 0

    pi = JointPolicy(pi)
    q_tilde = JointLocalQFunction(q_tilde)

    s_k = mu.sample_initial_state()

    pi_history = []
    q_tilde_history = []
    s_history = []
    a_history = []

    for k in tqdm.tqdm(range(K)):
        # sample action, update state, collect reward
        a_k = pi.sample_joint_action(s_k)

        N[s_k] += 1
        for i in I:
            N_tilde[i][(s_k, a_k[i])] += 1

        s_k_plus_1 = P.sample_next_state(s_k, a_k)

        # update beliefs, policies, Q functions
        new_pi = copy.deepcopy(pi)
        new_q_tilde = copy.deepcopy(q_tilde)

        for i in I:
            # update Q_i
            nu_i = sum(xlogx(pi[i][(s_k, a_i)]) for a_i in A[i])
            new_q_tilde[i][(s_k, a_k[i])] = q_tilde[i][(s_k, a_k[i])] + alpha(N_tilde[i][(s_k, a_k[i])]) * (
                R.get_reward(i, s_k, a_k) - (tau * nu_i)
                + delta * sum(pi[i][(s_k, a_i)] * q_tilde[i][(s_k, a_i)] for a_i in A[i])
                - q_tilde[i][(s_k, a_k[i])]
            )
            # update pi_i
            max_q_tilde = max(new_q_tilde[i][(s_k, a_i)] for a_i in A[i])
            softmax_denom = sum(math.exp((new_q_tilde[i][(s_k, a_i)] - max_q_tilde) / tau) for a_i in A[i])
            for a_i in A[i]:
                new_pi[i][(s_k, a_i)] = pi[i][(s_k, a_i)] + beta(N[s_k]) * (
                    (math.exp((new_q_tilde[i][(s_k, a_i)] - max_q_tilde) / tau) / softmax_denom)
                    - pi[i][(s_k, a_i)]
                )

        # put sigma, pi, Q, s_k, a_t into history
        pi_history.append(pi)
        q_tilde_history.append(q_tilde)
        s_history.append(s_k)
        a_history.append(a_k)

        # update sigma, pi, Q
        pi = new_pi
        q_tilde = new_q_tilde

        # transition to next state
        s_k = s_k_plus_1

    return pi_history, q_tilde_history, s_history, a_history
