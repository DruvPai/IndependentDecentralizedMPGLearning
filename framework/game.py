import functools
import itertools
import math
import numbers
import random
import typing


@functools.total_ordering
class Player:
    def __init__(self, idx: int, label: str = None):
        self.idx: int = idx
        self.label: str = label

    def __hash__(self):
        return self.idx

    def __eq__(self, other):
        return isinstance(other, Player) and other.idx == self.idx

    def __lt__(self, other):
        return self.idx < other.idx

    def __repr__(self):
        if self.label is None:
            return repr(self.idx)
        else:
            return repr(self.label)

    def __str__(self):
        if self.label is None:
            return str(self.idx)
        else:
            return str(self.label)



@functools.total_ordering
class State:
    def __init__(self, value: numbers.Number, label: str = None):
        self.value: numbers.Number = value
        self.label: str = label

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return isinstance(other, State) and other.value == self.value

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        if self.label is None:
            return repr(self.value)
        else:
            return repr(self.label)


@functools.total_ordering
class Action:
    def __init__(self, player: Player, value: numbers.Number, label: str = None):
        self.player: Player = player
        self.value: numbers.Number = value
        self.label: str = label

    def __hash__(self):
        return hash(self.player) + hash(self.value)

    def __eq__(self, other):
        return isinstance(other, Action) and other.player == self.player and other.value == self.value

    def __lt__(self, other):
        return (self.player, self.value) < (other.player, other.value)

    def __repr__(self):
        if self.label is None:
            return repr(self.value)
        else:
            return repr(self.label)


class ActionProfile:
    def __init__(self, actions: typing.List[Action]):
        self.player_action_map: typing.Dict[Player, Action] = {action.player: action for action in actions}

    def __getitem__(self, item):
        return self.player_action_map[item]

    def __eq__(self, other):
        return isinstance(other, ActionProfile) and self.player_action_map == other.player_action_map

    def __hash__(self):
        players = sorted(list(self.player_action_map.keys()))
        return sum(hash(self.player_action_map[players[i]]) for i in range(len(players)))

    def __repr__(self):
        return repr(self.player_action_map)

    def minus(self, player: Player):
        return ActionProfile([self.player_action_map[p] for p in self.player_action_map.keys() if p != player])

    @staticmethod
    def merge(a_i: Action, a_minus_i):
        return ActionProfile([a_i] + list(a_minus_i.player_action_map.values()))


class PlayerSet:
    def __init__(self, players: typing.List[Player]):
        self.players = players

    def __eq__(self, other):
        return isinstance(other, PlayerSet) and self.players == other.players

    def __hash__(self):
        return sum(hash(player) for player in self.players)

    def __len__(self):
        return len(self.players)

    def __iter__(self):
        return iter(self.players)


class StateSet:
    def __init__(self, states: typing.List[State]):
        self.states: typing.List[State] = list(states)

    def __hash__(self):
        return sum(hash(state) for state in self.states)

    def __eq__(self, other):
        return isinstance(other, StateSet) and self.states == other.states

    def __iter__(self):
        return iter(self.states)

    def __len__(self):
        return len(self.states)

    def __repr__(self):
        return repr(self.states)


class ActionSet:
    def __init__(self, player: Player, actions: typing.List[Action]):
        self.player: Player = player
        self.actions: typing.List[Action] = list(actions)

    def __hash__(self):
        return sum(hash(action) for action in self.actions)

    def __eq__(self, other):
        return isinstance(other, ActionSet) and self.player == other.player and self.actions == other.actions

    def __iter__(self):
        return iter(self.actions)

    def __len__(self):
        return len(self.actions)

    def __repr__(self):
        return repr(self.actions)


class ActionProfileSet:
    def __init__(self, action_sets: typing.List[ActionSet]):
        self.action_sets: typing.List[ActionSet] = action_sets
        action_lists: typing.List[typing.List[Action]] = [action_set.actions for action_set in self.action_sets]
        self.joint_actions: typing.List[ActionProfile] = [
            ActionProfile(list(joint_action)) for joint_action in itertools.product(*action_lists)
        ]

    def __hash__(self):
        return sum(hash(joint_action) for joint_action in self.joint_actions)

    def __eq__(self, other):
        return isinstance(other, ActionProfileSet) and self.joint_actions == other.joint_actions

    def __iter__(self):
        return iter(self.joint_actions)

    def __len__(self):
        return len(self.joint_actions)

    def __getitem__(self, item):
        return [self.action_sets[i] for i in range(len(self.action_sets)) if item == self.action_sets[i].player][0]

    def __repr__(self):
        return repr(self.joint_actions)

    def minus(self, player: Player):
        return ActionProfileSet([
            self.action_sets[i] for i in range(len(self.action_sets))
            if player != self.action_sets[i].player
        ])


class ProbabilityTransitionKernel:
    def __init__(self, state_set: StateSet, action_profile_set: ActionProfileSet,
                 kernel: typing.Callable[[State, ActionProfile, State], float]):
        self.state_set = state_set
        self.joint_action_set = action_profile_set

        triples = itertools.product(self.state_set, self.joint_action_set, self.state_set)
        self.kernel = {(s, a, s_prime): kernel(s, a, s_prime) for (s, a, s_prime) in triples}

    def __getitem__(self, item):
        return self.kernel[item]

    def __repr__(self):
        return repr(self.kernel)

    def sample_next_state(self, s: State, a: ActionProfile):
        s_prime_values: typing.List[State] = list(iter(self.state_set))
        s_prime_probabilities: typing.List[float] = [self.kernel[(s, a, s_prime)] for s_prime in s_prime_values]
        s_prime: State = random.choices(population=s_prime_values, weights=s_prime_probabilities)[0]
        return s_prime


class InitialStateDistribution:
    def __init__(self, state_set: StateSet, kernel: typing.Callable[[State], float]):
        self.state_set = state_set
        self.kernel = {s: kernel(s) for s in self.state_set}

    def __getitem__(self, item):
        return self.kernel[item]

    def __repr__(self):
        return repr(self.kernel)

    def sample_initial_state(self):
        list_of_states = list(self.state_set)
        s: State = random.choices(population=list_of_states, weights=[self.kernel[state] for state in list_of_states])[0]
        return s


class RewardFunction:
    def __init__(self, player_set: PlayerSet, states: StateSet, action_profiles:ActionProfileSet,
                 utilities: typing.Callable[[Player, State, ActionProfile], float]):
        self.kernel: typing.Dict[Player, typing.Dict[typing.Tuple[State, ActionProfile], float]] = {
            player: {
                (state, action_profile): utilities(player, state, action_profile)
                for (state, action_profile) in itertools.product(states, action_profiles)
            }
            for player in player_set
        }

    def __getitem__(self, item):
        return self.kernel[item[0]][(item[1], item[2])]

    def __repr__(self):
        return repr(self.kernel)

    def get_reward(self, player: Player, joint_state: State, joint_action: ActionProfile):
        return self.kernel[player][(joint_state, joint_action)]


class StochasticGame:
    def __init__(self, player_set: PlayerSet, states: StateSet, joint_actions: ActionProfileSet,
                 initial_distribution: InitialStateDistribution, probability_transition_kernel: ProbabilityTransitionKernel,
                 reward_function: RewardFunction, delta: float):
        self.I: PlayerSet = player_set
        self.S: StateSet = states
        self.A: ActionProfileSet = joint_actions
        self.mu: InitialStateDistribution = initial_distribution
        self.P: ProbabilityTransitionKernel = probability_transition_kernel
        self.R: RewardFunction = reward_function
        self.delta: float = delta

    def __repr__(self):
        return f"({repr(self.I)}, {repr(self.S)}, {repr(self.A)}, {repr(self.mu)}, {repr(self.P)}, {repr(self.R)}, {repr(self.delta)})"


class Policy:
    def __init__(self, states: StateSet, actions: ActionSet,
                 initialization_policy: typing.Callable[[State, Action], float]):
        self.joint_states: StateSet = states
        self.actions: ActionSet = actions
        self.kernel: typing.Dict[typing.Tuple[State, Action], float] = {
            (s, a): initialization_policy(s, a)
            for (s, a) in itertools.product(states, actions)
        }

    def __getitem__(self, item):
        return self.kernel[item]

    def __setitem__(self, key, value):
        self.kernel[key] = value

    def __repr__(self):
        return repr(self.kernel)

    def sample_action(self, s: State):
        a_values: typing.List[Action] = list(self.actions)
        a_probabilities: typing.List[float] = [self.kernel[(s, a)] for a in a_values]
        a: Action = random.choices(population=a_values, weights=a_probabilities)[0]
        return a


class JointPolicy:
    def __init__(self, player_policy_map: typing.Dict[Player, Policy]):
        self.player_policy_map: typing.Dict[Player, Policy] = player_policy_map

    def __getitem__(self, item):
        return self.player_policy_map[item]

    def __repr__(self):
        return repr(self.player_policy_map)

    def prob(self, s: State, a: ActionProfile):
        return math.prod(self.player_policy_map[i][(s, a[i])] for i in self.player_policy_map.keys())

    def minus(self, player: Player):
        return JointPolicy({p: self.player_policy_map[p] for p in self.player_policy_map.keys() if p != player})

    def sample_joint_action(self, s: State):
        return ActionProfile([policy.sample_action(s) for policy in self.player_policy_map.values()])


__all__ = [
    "Player", "PlayerSet", "State", "StateSet", "Action", "ActionSet", "ActionProfile", "ActionProfileSet",
    "ProbabilityTransitionKernel", "RewardFunction", "InitialStateDistribution",
    "StochasticGame", "Policy", "JointPolicy"
]
