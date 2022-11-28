import itertools
import numbers
import random
import typing

from framework.game import *
from framework.utils import *


def create_routing_game(
        N: int, M: int, U: int, m: typing.List[numbers.Number], b: typing.List[numbers.Number],
        lambda_1: float = 0.8, lambda_2: float = 0.2, delta: float = 0.5,
        common_interest: bool = False, strategy_independent_transitions: bool = False,
        seed: int = 0
):
    if seed:
        random.seed(seed)
    else:
        random.seed(100)

    assert N >= 1
    assert M >= 1
    assert U >= 1

    SAFE_STATUS = 0
    UNSAFE_STATUS = 1
    STATUSES = (SAFE_STATUS, UNSAFE_STATUS)

    players = [Player(idx=i, label=str(i + 1)) for i in range(N)]
    I = PlayerSet(players)
    S = StateSet([State(value=a) for a in itertools.product(STATUSES, repeat=M)])
    A = ActionProfileSet([ActionSet(i, [Action(i, value=j) for j in range(M)]) for i in I])

    def reward_oneplayer(i, s, a):
        route = a[i].value
        status = s.value[route]
        multiplier = 1.0 if status == UNSAFE_STATUS else 2.0
        return b[route] - multiplier * m[route] * sum(indicator(a[j].value == route) for j in I)

    def reward(i, s, a):
        if common_interest:
            return sum(reward_oneplayer(i_prime, s, a) for i_prime in I)
        else:
            return reward_oneplayer(i, s, a)

    def strategy_dependent_transition_kernel(s, a, s_prime):
        pr = 1.0
        counts_a = {route: sum(indicator(a[i].value == route) for i in I) for route in range(M)}
        for route in range(M):
            a_status = UNSAFE_STATUS if counts_a[route] >= U else SAFE_STATUS
            s_prime_status = s_prime.value[route]
            if a_status == SAFE_STATUS and s_prime_status == SAFE_STATUS:
                pr *= lambda_1
            elif a_status == SAFE_STATUS and s_prime_status == UNSAFE_STATUS:
                pr *= 1 - lambda_1
            elif a_status == UNSAFE_STATUS and s_prime_status == SAFE_STATUS:
                pr *= lambda_2
            elif a_status == UNSAFE_STATUS and s_prime_status == UNSAFE_STATUS:
                pr *= 1 - lambda_2
        return pr

    if strategy_independent_transitions:
        transition_matrix = {s: {s_prime: sum(strategy_dependent_transition_kernel(s, a, s_prime) for a in A) for s_prime in S} for s in S}
        for s in S:
            normalization = sum(transition_matrix[s][s_prime] for s_prime in S)
            for s_prime in S:
                transition_matrix[s][s_prime] /= normalization

        def transition_kernel(s, a, s_prime):
            return transition_matrix[s][s_prime]

    else:
        transition_kernel = strategy_dependent_transition_kernel

    mu = InitialStateDistribution(S, lambda s: 1 / len(S))
    P = ProbabilityTransitionKernel(S, A, transition_kernel)
    R = RewardFunction(I, S, A, reward)
    game = StochasticGame(I, S, A, mu, P, R, delta)
    return game
