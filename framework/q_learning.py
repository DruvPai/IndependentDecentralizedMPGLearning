import typing
import itertools

from .game import *


class LocalQFunction:
    def __init__(self, states: StateSet, actions: ActionSet,
                 initialization_q_fn: typing.Callable[[State, Action], float]):
        self.states: StateSet = states
        self.actions: ActionSet = actions
        self.kernel: typing.Dict[typing.Tuple[State, Action], float] = {
            (s, a_i): initialization_q_fn(s, a_i)
            for (s, a_i) in itertools.product(self.states, self.actions)
        }

    def __getitem__(self, item):
        return self.kernel[item]

    def __setitem__(self, key, value):
        self.kernel[key] = value

    def __eq__(self, other):
        return isinstance(other, LocalQFunction) and self.kernel == other.kernel

    def __repr__(self):
        return repr(self.kernel)

    def expected_value(self, state: State, action_or_policy: typing.Union[Action, Policy]):
        if isinstance(action_or_policy, Action):
            a_i = action_or_policy
            return self.kernel[(state, a_i)]

        elif isinstance(action_or_policy, Policy):
            pi_i = action_or_policy
            return sum(
                self.kernel[(state, a_i)] * pi_i[(state, a_i)]
                for a_i in pi_i.actions
            )


class JointLocalQFunction:
    def __init__(self, player_q_fn_map: typing.Dict[Player, LocalQFunction]):
        self.player_q_fn_map = player_q_fn_map

    def __getitem__(self, item):
        return self.player_q_fn_map[item]

    def __eq__(self, other):
        return isinstance(other, JointLocalQFunction) and self.player_q_fn_map == other.player_q_fn_map

    def __repr__(self):
        return repr(self.player_q_fn_map)


class QFunction:
    def __init__(self, states: StateSet, action_profiles: ActionProfileSet,
                 initialization_q_fn: typing.Callable[[State, ActionProfile], float]):
        self.states: StateSet = states
        self.action_profiles: ActionProfileSet = action_profiles
        self.kernel: typing.Dict[typing.Tuple[State, ActionProfile], float] = {
            (s, a): initialization_q_fn(s, a)
            for (s, a) in itertools.product(self.states, self.action_profiles)
        }

    def __getitem__(self, item):
        return self.kernel[item]

    def __setitem__(self, key, value):
        self.kernel[key] = value

    def __eq__(self, other):
        return isinstance(other, QFunction) and self.kernel == other.kernel

    def __repr__(self):
        return repr(self.kernel)


class JointQFunction:
    def __init__(self, player_q_fn_map: typing.Dict[Player, QFunction]):
        self.player_q_fn_map = player_q_fn_map

    def __getitem__(self, item):
        return self.player_q_fn_map[item]

    def __eq__(self, other):
        return isinstance(other, JointQFunction) and self.player_q_fn_map == other.player_q_fn_map

    def __repr__(self):
        return repr(self.player_q_fn_map)


class VFunction:
    def __init__(self, states: StateSet,
                 initialization_v_fn: typing.Callable[[State], float]):
        self.states = states
        self.kernel = {s: initialization_v_fn(s) for s in self.states}

    def __getitem__(self, item):
        return self.kernel[item]

    def __setitem__(self, key, value):
        self.kernel[key] = value

    def __eq__(self, other):
        return isinstance(other, VFunction) and self.kernel == other.kernel

    def __repr__(self):
        return repr(self.kernel)


class JointVFunction:
    def __init__(self, player_v_fn_map: typing.Dict[Player, VFunction]):
        self.player_v_fn_map = player_v_fn_map

    def __getitem__(self, item):
        return self.player_v_fn_map[item]

    def __eq__(self, other):
        return isinstance(other, JointVFunction) and self.player_v_fn_map == other.player_v_fn_map

    def __repr__(self):
        return repr(self.player_v_fn_map)


__all__ = ["LocalQFunction", "JointLocalQFunction", "QFunction", "JointQFunction", "VFunction", "JointVFunction"]
