import os

import pytest
import pandas as pd
import numpy as np

from bsm_time_machine import Position, Underlying, Call, Put


BASE_PATH = os.path.join("tests/data/calc_fudge_factor/")


@pytest.fixture
def position(case_path: str):
    yield Position(
        pd.read_pickle(BASE_PATH + "cases/" + case_path),
        Underlying(
            symbol="SPX",
            spread_loss=0.1,
            min_strike_gap=1,
        ),
        legs=[
            Call("2.05 SD", 45, -1),
            Put("2.05 SD", 45, -1),
        ],
        holding_period=45,
        stop_loss=-0.56,
        max_deviations=3,
        start_date="2017-01-01",
        scalping=True,
        sequential_positions=True,
        pom_threshold=0.7,
        num_simulations=500,
    )


@pytest.fixture
def yield_expected(expected_path: str):
    yield np.load(BASE_PATH + "expected/" + expected_path)


@pytest.fixture
def tensor(tensor_path: str):
    yield np.load(BASE_PATH + "cases/" + tensor_path)


@pytest.mark.parametrize(
    "tensor_path,case_path,expected_path",
    [
        ("tensor1.npy", "test1.pkl", "test1.npy"),
    ],
)
def test_calc_fudge_factor(position, tensor, yield_expected):
    position._calc_fudge_factor(tensor)
    np.testing.assert_array_equal(tensor, yield_expected)
