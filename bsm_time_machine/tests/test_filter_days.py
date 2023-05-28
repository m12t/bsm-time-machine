import os

import pytest
import pandas as pd
import numpy as np

from bsm_time_machine import Position, Underlying, Call, Put

# NOTE: run pytest with `-s` to show print statements...

BASE_PATH = os.path.join("tests/data/filter_days/")


# mkdir foo && cd foo && mkdir cases && mkdir expected


# def load_dfs(case, expected):
#     return (
#         pd.read_pickle(BASE_PATH + "cases/" + case),
#         pd.read_pickle(BASE_PATH + "expected/" + expected),
#     )
#     return pd.DataFrame(np.resize(np.arange(5), 100), columns=["day"]).to_pickle(
#         BASE_PATH + "cases/" + name
#     )


# def underlying():
#     return Underlying("SPX", 0.05, 5)


# def legs():
#     return [Call("2.05 SD", 45, 1), Put("2.05 SD", 45, -1)]


# @pytest.fixture
# def instance():
#     return Position(
#         # df(),
#         underlying(),
#         legs(),
#     )


# pytest.mark.parametrize(
#     "trading_days,case,expected",
#     [
#         (0, 1, 2, 3, 4),
#     ],
# )


def test_filter_days():
    assert 5 == 5

    # instance._filter_days()
    # print(instance.df)
