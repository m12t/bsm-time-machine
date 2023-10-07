import os
from typing import Tuple

import pytest
import pandas as pd

from bsm_time_machine import Position, Underlying, Call, Put


BASE_PATH = os.path.join("tests/data/filter_days/")


@pytest.fixture(scope="function")
def position(case_path: str, trading_days: Tuple[int]):
    yield Position(
        pd.read_pickle(BASE_PATH + "cases/" + case_path),
        Underlying("SPX", 0.05, 5),
        [Call("2.05 SD", 45, 1), Put("2.05 SD", 45, -1)],
        5,
        days_of_the_week=trading_days,
    )


@pytest.fixture
def yield_expected(expected_path: str):
    try:
        yield pd.read_pickle(BASE_PATH + "expected/" + expected_path)
    except FileNotFoundError:
        # * this try block is so that when writing new expected dfs,
        #   the `expected_path` can be used in `to_pickle` and won't
        #   error out at this stage. Not necessary for testing.
        yield None


@pytest.mark.parametrize(
    "trading_days,case_path,expected_path",
    [
        ([0, 1, 2, 3, 4], "100_trading_days.pkl", "100_trading_days.pkl"),
        ([0], "100_trading_days.pkl", "mondays_only.pkl"),
        ([4], "100_trading_days.pkl", "fridays_only.pkl"),
        ([5, 6, 7], "100_trading_days.pkl", "no_trading_days.pkl"),
        ([0, 2, 3], "100_trading_days.pkl", "mo_we_thu.pkl"),
        ([4, 3, 2, 1, 0], "100_trading_days.pkl", "100_trading_days.pkl"),
    ],
)
def test_filter_days(position, yield_expected):
    position._filter_days()
    pd.testing.assert_frame_equal(position.df, yield_expected)
