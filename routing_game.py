import itertools
import numbers
import random
import typing

from framework.game import *
from framework.utils import *


def create_routing_game(
        N: int, M: int, U: int, m: typing.List[numbers.Number] = None, b: typing.List[numbers.Number] = None,
        lambda_1: float = 0.8, lambda_2: float = 0.2, delta: float = 0.5
):
    random.seed(100)

    assert N >= 1
    assert M >= 1
    assert U >= 1

    SAFE_STATUS = 0
    UNSAFE_STATUS = 1
    STATUSES = (SAFE_STATUS, UNSAFE_STATUS)

    L_reward_slope = 1
    U_reward_slope = 5
    L_reward_intercept = 5
    U_reward_intercept = 20

    players = [Player(idx=i, label=str(i + 1)) for i in range(N)]
    I = PlayerSet(players)
    S = StateSet([State(value=a) for a in itertools.product(STATUSES, repeat=M)])
    A = ActionProfileSet([ActionSet(i, [Action(i, value=j) for j in range(M)]) for i in I])
    if m is None:
        m = [random.randint(L_reward_slope, U_reward_slope) for _ in range(M)]
    if b is None:
        b = [random.randint(L_reward_intercept, U_reward_intercept) for _ in range(M)]

    print("m_j:", m)
    print("b_j:", b)

    def reward(i, s, a):
        route = a[i].value
        status = s.value[route]
        multiplier = 1.0 if status == UNSAFE_STATUS else 2.0
        return b[route] - multiplier * m[route] * sum(indicator(a[j].value == route) for j in I)

    def transition_kernel(s, a, s_prime):
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

    mu = InitialStateDistribution(S, lambda s: 1 / len(S))
    P = ProbabilityTransitionKernel(S, A, transition_kernel)
    R = RewardFunction(I, S, A, reward)
    game = StochasticGame(I, S, A, mu, P, R, delta)
    return game, m, b
