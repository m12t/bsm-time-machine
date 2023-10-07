import os

import pytest
import pandas as pd
import numpy as np

from bsm_time_machine import Position, Underlying, Call


BASE_PATH = os.path.join("tests/data/calc_returns/")


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
    "i,case_path,expected_path",
    [(i, f"a_{i}.npy", f"a_{i}.npy") for i in range(45)],
)
def test_calc_returns(i, position, arr, expected):
    position._calc_returns(arr, i)
    np.testing.assert_array_equal(arr, expected)
