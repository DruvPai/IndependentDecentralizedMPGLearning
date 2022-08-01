import numbers
import pathlib
import typing


EXPERIMENTS_DIR = pathlib.Path("experiments")


def create_result_folder(N: int, M: int, U: int, lambda_1: float, lambda_2: float,
                         m: typing.List[numbers.Number], b: typing.List[numbers.Number],
                         K: int, tau: float,
                         common_interest: bool = False, strategy_independent_transitions: bool = False
                         ):
    remove_spaces_from_str = lambda s: str(s).replace(" ", "")
    path = EXPERIMENTS_DIR / f"N{N}_M{M}_U{U}_lo{lambda_1}_lt{lambda_2}_m{remove_spaces_from_str(m)}_b{remove_spaces_from_str(b)}_K{K}_tau{tau}_ci{common_interest}_si{strategy_independent_transitions}"
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = ["create_result_folder"]
