import pytest

from bsm_time_machine import Underlying, Position, Call, Put


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
