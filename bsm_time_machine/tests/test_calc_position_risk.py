import os

import pytest
import pandas as pd
import numpy as np

from bsm_time_machine import Position, Underlying, Call


BASE_PATH = os.path.join("tests/data/calc_position_risk/")


@pytest.fixture(scope="function")
def position():
    yield Position(
        pd.DataFrame(),
        Underlying("SPX", 0.05, 5),
        [Call("2.05 SD", 45, 1)],
        5,
        num_simulations=20_000,
    )


@pytest.fixture
def expected(expected_path: str):
    yield np.load(BASE_PATH + "expected/" + expected_path)


@pytest.fixture
def arr(case_path: str):
    yield np.load(BASE_PATH + "cases/" + case_path)


@pytest.mark.parametrize(
    "case_path,expected_path",
    [("a.npy", "a.npy")],
)
def test_calc_position_risk(position, arr, expected):
    position._calc_position_risk(arr)
    np.testing.assert_allclose(arr, expected, rtol=0.25)
