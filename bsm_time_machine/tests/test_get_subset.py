import os

import pytest
import pandas as pd

from bsm_time_machine import Position, Underlying, Call, Put


BASE_PATH = os.path.join("tests/data/get_subset/")


@pytest.fixture(scope="function")
def position(
    case_path: str, lookback: int, holding_period: int, start_date: str, end_date: str
):
    yield Position(
        pd.read_pickle(BASE_PATH + "cases/" + case_path),
        Underlying("SPX", 0.05, 5),
        [Call("2.05 SD", 45, 1), Put("2.05 SD", 45, -1)],
        holding_period,
        lookback=lookback,
        start_date=start_date,
        end_date=end_date,
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
    "holding_period,lookback,start_date,end_date,case_path,expected_path",
    [
        (
            0,
            0,
            None,
            None,
            "20000101_20230711.pkl",
            "20000101_20230711.pkl",
        ),
        (
            0,
            0,
            None,
            "2019-12-31",
            "20000101_20230711.pkl",
            "20000101_20191231.pkl",
        ),
        (
            0,
            0,
            "2015-07-01",
            "2016-06-30",
            "20000101_20230711.pkl",
            "20150701_20160630.pkl",
        ),
        (
            35,
            2,
            "2015-07-01",
            "2016-06-30",
            "20000101_20230711.pkl",
            "20150701_20160630_hp35_lb2.pkl",
        ),
    ],
)
def test_filter_days(position, yield_expected):
    position._get_subset()
    pd.testing.assert_frame_equal(
        position.df.reset_index(drop=True), yield_expected.reset_index(drop=True)
    )
