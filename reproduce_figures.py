from framework.game import *
from framework.q_learning import *
from framework.utils import *
from framework.plotting import *
from independent_decentralized_learning import *
from routing_game import *
from utils import *


def reproduce_figure_1():
    N = 4
    M = 2
    U = 2
    m = [2, 4]
    b = [9, 16]
    lambda_1 = 0.8
    lambda_2 = 0.2
    delta = 0.5
    common_interest = True
    strategy_independent = False
    game, m, b = create_routing_game(N=N, M=M, U=U, m=m, b=b, lambda_1=lambda_1, lambda_2=lambda_2, delta=delta,
                                     common_interest=common_interest,
                                     strategy_independent_transitions=strategy_independent)
    K = int(1e5)
    tau = 1e-6
    pi_history, q_tilde_history, s_history, a_history = independent_decentralized_algo(
        game=game,
        K=K,
        tau=tau
    )
    result_dir = create_result_folder(N, M, U, lambda_1, lambda_2, m, b, K, tau, common_interest, strategy_independent)
    plot_policy_convergence_l1(game, pi_history, result_dir)
    plot_policy_convergence_to_nash_l1(game, pi_history, result_dir)
    plot_local_Q_convergence_l1(game, q_tilde_history, result_dir)


def reproduce_figure_2():
    N = 4
    M = 2
    U = 2
    m = [2, 4]
    b = [9, 16]
    lambda_1 = 0.8
    lambda_2 = 0.2
    delta = 0.5
    common_interest = True
    strategy_independent = False
    game, m, b = create_routing_game(N=N, M=M, U=U, m=m, b=b, lambda_1=lambda_1, lambda_2=lambda_2, delta=delta,
                                     common_interest=common_interest,
                                     strategy_independent_transitions=strategy_independent)
    K = int(1e5)
    tau = 1e-3
    pi_history, q_tilde_history, s_history, a_history = independent_decentralized_algo(
        game=game,
        K=K,
        tau=tau
    )
    result_dir = create_result_folder(N, M, U, lambda_1, lambda_2, m, b, K, tau, common_interest, strategy_independent)
    plot_policy_convergence_l1(game, pi_history, result_dir)
    plot_policy_convergence_to_nash_l1(game, pi_history, result_dir)
    plot_local_Q_convergence_l1(game, q_tilde_history, result_dir)


def reproduce_figure_3():
    N = 8
    M = 2
    U = 2
    m = [2, 4]
    b = [9, 16]
    lambda_1 = 0.8
    lambda_2 = 0.2
    delta = 0.5
    common_interest = True
    strategy_independent = False
    game, m, b = create_routing_game(N=N, M=M, U=U, m=m, b=b, lambda_1=lambda_1, lambda_2=lambda_2, delta=delta,
                                     common_interest=common_interest,
                                     strategy_independent_transitions=strategy_independent)
    K = int(1e5)
    tau = 1e-6
    pi_history, q_tilde_history, s_history, a_history = independent_decentralized_algo(
        game=game,
        K=K,
        tau=tau
    )
    result_dir = create_result_folder(N, M, U, lambda_1, lambda_2, m, b, K, tau, common_interest, strategy_independent)
    plot_policy_convergence_l1(game, pi_history, result_dir)
    plot_policy_convergence_to_nash_l1(game, pi_history, result_dir)
    plot_local_Q_convergence_l1(game, q_tilde_history, result_dir)


def reproduce_figure_4():
    N = 8
    M = 2
    U = 2
    m = [2, 4]
    b = [9, 16]
    lambda_1 = 0.8
    lambda_2 = 0.2
    delta = 0.5
    common_interest = True
    strategy_independent = False
    game, m, b = create_routing_game(N=N, M=M, U=U, m=m, b=b, lambda_1=lambda_1, lambda_2=lambda_2, delta=delta,
                                     common_interest=common_interest,
                                     strategy_independent_transitions=strategy_independent)
    K = int(1e5)
    tau = 1e-6
    pi_history, q_tilde_history, s_history, a_history = independent_decentralized_algo(
        game=game,
        K=K,
        tau=tau,
        alpha=lambda n:1/n,
        beta=lambda n: 1/(n**0.5)
    )
    result_dir = create_result_folder(N, M, U, lambda_1, lambda_2, m, b, K, tau, common_interest, strategy_independent)
    plot_policy_convergence_l1(game, pi_history, result_dir)
    plot_policy_convergence_to_nash_l1(game, pi_history, result_dir)
    plot_local_Q_convergence_l1(game, q_tilde_history, result_dir)


def reproduce_figure_5():
    N = 8
    M = 2
    U = 2
    m = [2, 4]
    b = [9, 16]
    lambda_1 = 0.8
    lambda_2 = 0.2
    delta = 0.5
    common_interest = False
    strategy_independent = True
    game, m, b = create_routing_game(N=N, M=M, U=U, m=m, b=b, lambda_1=lambda_1, lambda_2=lambda_2, delta=delta,
                                     common_interest=common_interest,
                                     strategy_independent_transitions=strategy_independent)
    K = int(1e5)
    tau = 1e-3
    pi_history, q_tilde_history, s_history, a_history = independent_decentralized_algo(
        game=game,
        K=K,
        tau=tau
    )
    result_dir = create_result_folder(N, M, U, lambda_1, lambda_2, m, b, K, tau, common_interest, strategy_independent)
    plot_policy_convergence_l1(game, pi_history, result_dir)
    plot_policy_convergence_to_nash_l1(game, pi_history, result_dir)
    plot_local_Q_convergence_l1(game, q_tilde_history, result_dir)



#reproduce_figure_1()
#reproduce_figure_2()
#reproduce_figure_3()
reproduce_figure_4()
#reproduce_figure_5()
