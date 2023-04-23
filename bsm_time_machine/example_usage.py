"""
* This snippet demonstrates how to model
  a simple PMCC (Poor Man's Covered Call)
  using bsm_time_machine.
"""

import pandas as pd

from bsm_time_machine import Underlying, Position, Call, Put


df = pd.read_pickle("../dfs/1day_spx_max_iv_rth.pkl")

# instantiate an underlying with some specifics
u = Underlying(
    symbol="SPX",
    spread_loss=0.1,
    min_strike_gap=1,
)

# * This example use case will show a simple PMCC (Poor Man's Covered Call).
# * Our long call will be deep in-the-money LEAPS
#   and will have a far-out maturity (> 1 year)
# * The relative strike of our long call, `0.34 X`, is
#   a constant multiplier of our strike and is the first
#   of two relative strike schemes supported.
#   roughly equal to 1360 strike when the spot is 4000
# long_call = Call("1 X", 650, 1)

# * Our short call will be out-of-the-money and nearer-term.
# * Here, we've gone with 45 DTE and `0.2 SD` relative strike.
# * This represents the second type of relative strike scheme
#   supported, volatility-based.
# * Here, we're selling a call that is 0.2 standard deviations
#   out-of-the-money.
short_call = Call("0.3 x", 500, 1)
long_call = Call("1.2 x", 45, -1)


p = Position(
    df,
    underlying=u,
    legs=[
        Call("0.2 SD", 5, -1),
        Put("0.2 SD", 5, -1),
    ],  # simply wrap each leg in a list
    holding_period=5,
    stop_loss=-0.56,
    max_deviations=3,
    scalping=True,
    sequential_positions=True,
    pom_threshold=0.63,
    num_simulations=500,
    lrr_only=False,
    vol_threshold=2.0,
    lookback=3,
    vol_greater_than=True,
    iv_min_threshold=0.0,
    iv_max_threshold=0.15,
    iv_greater_than=False,
    iv_less_than=True,
)
p.run()

# * plot the risk return payoff diagram for the portfolio
# * this is also used to validate that the risk analysis looks
#   the way you expect it to.
p.plot_payoff()

# p.plot(show=256)

# p.analyze_results()

# p.to_pickle("output.pkl")  # save the output df if you'd like
