import os

import pytest
import pandas as pd
import numpy as np

from bsm_time_machine import Position, Underlying, Call, Put


BASE_PATH = os.path.join("tests/data/calc_bsm/")


@pytest.fixture(scope="function")
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
def spot(spot_path: str):
    yield np.load(BASE_PATH + "cases/" + spot_path)


@pytest.fixture
def strike(strike_path: str):
    yield np.load(BASE_PATH + "cases/" + strike_path)


@pytest.fixture
def sigma(sigma_path: str):
    yield np.load(BASE_PATH + "cases/" + sigma_path)


@pytest.fixture
def tenor(tenor_path: str):
    yield np.load(BASE_PATH + "cases/" + tenor_path)


@pytest.fixture
def d1(d1_path: str):
    yield np.load(BASE_PATH + "expected/" + d1_path)


@pytest.fixture
def d2(d2_path: str):
    yield np.load(BASE_PATH + "expected/" + d2_path)


@pytest.fixture
def prices(prices_path: str):
    yield np.load(BASE_PATH + "expected/" + prices_path)


@pytest.mark.parametrize(
    "right,case_path,d1_path,d2_path,spot_path,strike_path,sigma_path,tenor_path,prices_path,rfr",
    [
        (
            "call",
            "case_path_1.pkl",
            f"call_d1_{i}.npy",
            f"call_d2_{i}.npy",
            f"call_spot_{i}.npy",
            f"call_strike_{i}.npy",
            f"call_sigma_{i}.npy",
            f"call_tenor_{i}.npy",
            f"call_prices_{i}.npy",
            0.04,
        )
        for i in range(45)
    ],
)
def test_bsm_calcs(position, right, d1, d2, spot, strike, sigma, tenor, prices, rfr):
    case_d1 = position._calc_d1(spot, strike, sigma, tenor, rfr)
    case_d2 = position._calc_d2(case_d1, sigma, tenor)
    opt = None
    if right == "call":
        opt = position._calc_call(spot, strike, sigma, tenor)
    else:
        opt = position._calc_put(spot, strike, sigma, tenor)
    np.testing.assert_array_equal(case_d1, d1)
    np.testing.assert_array_equal(case_d2, d2)
    np.testing.assert_array_equal(opt, prices)
