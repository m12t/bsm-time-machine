import pytest
import pandas as pd

from bsm_time_machine import Underlying, Call, Put, Position


@pytest.mark.parametrize(
    "relative_strike,tenor,quantity",
    [
        # test relative strike
        ("1.0 x", 5, 1),
        ("1.0 SD", 5, 1),
        ("2 sd", 5, 1),
        ("5 Sd", 5, 1),
        ("1.0 X", 5, 1),
        ("0.01 sD", 5, 1),
        ("0.3 x", 5, 1),
        # test tenor
        ("1.0 x", 500, 1),
        ("1.0 x", 0.1, 1),
        ("1.0 x", 100.1, 1),
        ("1.0 x", 1, 1),
        ("1.0 x", 8.88382, 1),
        # test quantity
        ("1.0 x", 5, 1),
        ("1.0 x", 5, -1),
        ("1.0 x", 5, 1099),
        ("1.0 x", 5, -1393),
    ],
)
def test_valid_option_creation(relative_strike, tenor, quantity):
    call = Call(relative_strike, tenor, quantity)
    assert call.relative_strike == relative_strike
    assert call.tenor == tenor
    assert call.quantity == quantity

    put = Put(relative_strike, tenor, quantity)
    assert put.relative_strike == relative_strike
    assert put.tenor == tenor
    assert put.quantity == quantity


@pytest.mark.parametrize(
    "relative_strike,tenor,quantity,excep",
    [
        # test relative strike
        (None, 5, 1, TypeError),
        ("1.0 p", 5, 1, ValueError),
        ("1.0 xx", 5, 1, ValueError),
        ("0 x", 5, 1, ValueError),
        ("0.0 x", 5, 1, ValueError),
        ("1x", 5, 1, ValueError),
        ("1SD", 5, 1, ValueError),
        ("1X", 5, 1, ValueError),
        ("1sd", 5, 1, ValueError),
        # test tenor
        ("1.0 x", "5", 1, TypeError),
        ("1.0 x", None, 1, TypeError),
        ("1.0 x", 0, 1, ValueError),
        ("1.0 x", -5.0, 1, ValueError),
        # test quantity
        ("1.0 x", 5, "1", TypeError),
        ("1.0 x", 5, 0, ValueError),
        ("1.0 x", 5, 0.0, TypeError),
        ("1.0 x", 5, None, TypeError),
    ],
)
def test_invalid_option_creation(relative_strike, tenor, quantity, excep):
    with pytest.raises(excep):
        Call(relative_strike, tenor, quantity)
    with pytest.raises(excep):
        Put(relative_strike, tenor, quantity)


@pytest.mark.parametrize(
    "symbol,spread_loss,min_strike_gap",
    [
        ("SPX", 0.05, 5),
        ("SPX", 0.10, 10),
        ("SPX", 0.05, 5.0),
        ("SPX", 0.3210, 1.34280),
    ],
)
def test_valid_underlying_creation(symbol, spread_loss, min_strike_gap):
    u = Underlying(symbol, spread_loss, min_strike_gap)
    assert u.symbol == symbol
    assert u.spread_loss == spread_loss
    assert u.min_strike_gap == min_strike_gap


@pytest.mark.parametrize(
    "symbol,spread_loss,min_strike_gap,excep",
    [
        # test spread_loss
        ("SPX", 5, 5, TypeError),
        ("SPX", None, 5, TypeError),
        ("SPX", "0.05", 5, TypeError),
        ("SPX", -5.0, 10, ValueError),
        # test min_strike_gap
        ("SPX", 0.05, None, TypeError),
        ("SPX", 0.05, "5", TypeError),
        ("SPX", 0.05, -1, ValueError),
        ("SPX", 0.05, -1.0, ValueError),
    ],
)
def test_invalid_underlying_creation(symbol, spread_loss, min_strike_gap, excep):
    with pytest.raises(excep):
        Underlying(symbol, spread_loss, min_strike_gap)


@pytest.mark.parametrize(
    "df,underlying,legs,holding_period,stop_loss,max_deviations,start_date,scalping,sequential_positions,pom_threshold,num_simulations",
    [
        # success instantiation
        (
            pd.DataFrame(),
            Underlying("SPX", 0.3210, 1.34280),
            [
                Call("2.05 SD", 45, -1),
                Put("2.05 SD", 45, -1),
            ],
            5,
            -0.56,
            3,
            "2017-01-01",
            True,
            True,
            0.7,
            500,
        ),
    ],
)
def test_valid_position_creation(
    df,
    underlying,
    legs,
    holding_period,
    stop_loss,
    max_deviations,
    start_date,
    scalping,
    sequential_positions,
    pom_threshold,
    num_simulations,
):
    p = Position(
        df,
        underlying=underlying,
        legs=legs,
        holding_period=holding_period,
        stop_loss=stop_loss,
        max_deviations=max_deviations,
        start_date=start_date,
        scalping=scalping,
        sequential_positions=sequential_positions,
        pom_threshold=pom_threshold,
        num_simulations=num_simulations,
    )
    assert p.df.equals(df)
    assert p.underlying == underlying
    assert p.legs == legs
    assert p.holding_period == holding_period
    assert p.stop_loss == stop_loss
    assert p.max_deviations == max_deviations
    assert p.start_date == start_date
    assert p.scalping == scalping
    assert p.sequential_positions == sequential_positions
    assert p.pom_threshold == pom_threshold
    assert p.num_simulations == num_simulations
