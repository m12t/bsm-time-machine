import os

import pytest
import pandas as pd

from bsm_time_machine import Position, Underlying, Call


BASE_PATH = os.path.join("tests/data/filter_days/")


@pytest.fixture(scope="function")
def position():
    yield Position(
        pd.DataFrame(), Underlying("SPX", 0.05, 5), [Call("2.05 SD", 45, 1)], 5
    )


@pytest.mark.parametrize(
    "p,b,expected",
    [
        (1, 1, 1),
        (0.99, 1, 0.98),
        (0.124, 1.3131, -0.5431236006397075),
    ],
)
def test_kelly(position, p, b, expected):
    assert position._kelly(p, b) == expected


@pytest.mark.parametrize(
    "b,c,expected",
    [
        (1, 1, 1),
        (1.12321, -0.88, 1.12321),
        (1.12321, -1.01, 1.112089108910891),
    ],
)
def test_scale_b(position, b, c, expected):
    assert position._scale_b(b, c) == expected
