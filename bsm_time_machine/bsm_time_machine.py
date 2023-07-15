import math
import random
from datetime import datetime, date, timedelta
from typing import Optional, List, Tuple

from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", 50, "display.max_columns", 10)


class Option:
    """private class that is wrapped by
    the public classes `Call` and `Put`"""

    def __init__(self, rs, tenor, right, quantity):
        # * rs is the relative strike in one of the two formats below:
        #   '0.5 SD' or '1.05 x' (multiple of strike)

        self.id = None  # placeholder
        self.relative_strike = rs
        self.tenor = tenor
        self.right = right
        self.quantity = quantity

        self._post_init()

        self._generate_id()

    def _generate_id(self):
        self.id = f"{self.relative_strike.replace(' ', '')}_{self.tenor}_{self.right}"

    def _post_init(self):
        """run some basic validation on the inputs"""
        # validate the relative strike
        if not isinstance(self.relative_strike, str):
            raise TypeError(
                "relative strike must be type <str> in format `1.0 SD` or `1.0 X`"
            )
        rsval, rstype = self.relative_strike.split(" ")
        rsval = float(rsval)
        if rstype.casefold() not in {"x", "sd"}:
            raise ValueError(
                "relative strike must use constant multiplier, `X`, or volatility multiplier, `SD`"
            )
        if rstype.casefold() == "x" and rsval <= 0.0:
            raise ValueError("relative price must be nonzero")

        # validate the tenor
        if not isinstance(self.tenor, int) and not isinstance(self.tenor, float):
            raise TypeError("tenor must be numeric")

        if self.tenor <= 0:
            raise ValueError("tenor must be positive")

        # validate the quantity
        if not isinstance(self.quantity, int):
            raise TypeError("quantity must be an integer")

        if self.quantity == 0:
            raise ValueError("quantity must be nonzero")

        # assert the right
        if self.right not in {"call", "put"}:
            # in theory this never happens. But it's a low-overhead test, so do it anyways.
            raise ValueError("unexpected right encountered")


class Call(Option):
    """
    * Create a call option with relative strike, tenor,
    and quantity (>0 for long, <0 for short)
    * relative strike is of the format: `1.5 X`
      (for 1.5x the current spot price)
      or `1.5 SD` (or 1.5 standard deviations OTM)
    """

    def __init__(
        self,
        relative_strike: str,
        tenor: int,
        quantity: int,
    ):
        super().__init__(relative_strike, tenor, "call", quantity)


class Put(Option):
    """
    * Create a put option with relative strike, tenor,
    and quantity (>0 for long, <0 for short)
    * relative strike is of the format: `1.5 X`
      (for 1.5x the current spot price)
      or `1.5 SD` (or 1.5 standard deviations OTM)
    """

    def __init__(
        self,
        relative_strike: str,
        tenor: int,
        quantity: int,
    ):
        super().__init__(relative_strike, tenor, "put", quantity)


class Underlying:
    def __init__(
        self,
        symbol: str = "",
        spread_loss: float = 0.05,
        min_strike_gap: int = 5,
    ):
        self.symbol = symbol
        self.spread_loss = spread_loss
        self.min_strike_gap = min_strike_gap

        self._post_init()

    def _post_init(self):
        if not isinstance(self.spread_loss, float):
            raise TypeError("spread_loss must be of type <float>")
        if self.spread_loss < 0.0:
            raise ValueError("spread_loss must be positive")

        if not (
            isinstance(self.min_strike_gap, int)
            or isinstance(self.min_strike_gap, float)
        ):
            raise TypeError("min_strike_gap must be numeric")
        if self.min_strike_gap < 0:
            raise ValueError("min_strike_gap must be positive")


class Position:
    """
    * The Position class takes arguments on instantiation
      that define the parameters you'd like to backtest.
    * All class attributes are public and can be directly
      modified as needed between tests.
    * The 2 public methods of the class are `run()`, which
      runs the backtest and `plot()` to optionally plot the results.

    """

    def __init__(
        self,
        df: pd.DataFrame,
        underlying: Underlying,
        legs: List[Option],
        holding_period: int,
        stop_loss: Optional[float] = None,
        scalping: bool = False,
        pom_threshold: Optional[float] = None,
        risk_return_threshold: Optional[float] = None,
        sequential_positions: bool = True,
        riskfree_rate: float = 0.04,
        days_of_the_week: List = [0, 1, 2, 3, 4],
        start_date: date = None,
        end_date: date = None,
        vol_threshold: float = 0.0,
        lookback: int = 1,
        vol_greater_than: bool = True,
        vol_type: str = "max",  # max (high-low) or real (open-close)
        max_deviations: float = 2.0,
        iv_min_threshold=0.10,
        iv_max_threshold=0.40,
        iv_greater_than=False,  # iv_lower_limit
        iv_less_than=False,  # iv_upper_limit
        sample=False,
        return_only=True,
        lrr_only=True,
        num_simulations=10_000,
    ):
        self.df = df
        self.underlying = underlying
        self.legs = legs
        self.holding_period = min(holding_period, min(l.tenor for l in legs))
        self.stop_loss = stop_loss
        self.scalping = scalping
        self.pom_threshold = pom_threshold
        self.risk_return_threshold = risk_return_threshold
        self.sequential_positions = sequential_positions
        self.riskfree_rate = riskfree_rate
        self.days_of_the_week = days_of_the_week
        self.start_date = start_date
        self.end_date = end_date
        self.vol_threshold = vol_threshold
        self.lookback = lookback
        self.vol_greater_than = vol_greater_than
        self.vol_type = vol_type
        self.iv_min_threshold = iv_min_threshold
        self.iv_max_threshold = iv_max_threshold
        self.iv_greater_than = iv_greater_than
        self.iv_less_than = iv_less_than
        self.sample = sample
        self.return_only = return_only
        self.lrr_only = lrr_only
        self.max_deviations = max_deviations
        self.num_simulations = num_simulations

        # * The below offsets are for the 3D numpy array that is dyanamic regarding
        #   the number of positions. It does this by storing offsets and step_size.

        # these indices are from the historical data
        self.__SPOT_OPEN_IDX = 0
        self.__SPOT_CLOSE_IDX = 1
        self.__SIGMA_OPEN_IDX = 2
        self.__SIGMA_CLOSE_IDX = 3

        # These indices are used in BSM calculations
        self.__NET_POS_OPEN_IDX = 4  # previously a[:, 10]
        self.__NET_POS_CLOSE_IDX = 5  # previously a[:, 11]
        self.__RISK_RETURN_IDX = 6  # previously a[:, 23]
        self.__POM_RETURN_IDX = 7  # previously a[:, 24]
        self.__MAX_DOWNSWING_IDX = 8  # (previously a[:, 20])
        self.__MAX_UPSWING_IDX = 9  # (previously a[:, 21])
        self.__MAX_RETURN_IDX = 10
        self.__MAX_RISK_IDX = 11  # (previously a[:, 22])

        # * these indices are dynamic columns that
        #   are applied to each leg in the position
        self.__STRIKE_OFFSET = 12
        self.__PRICE_OPEN_OFFSET = 13
        self.__PRICE_CLOSE_OFFSET = 14
        self.__TENOR_OPEN_OFFSET = 15
        self.__TENOR_CLOSE_OFFSET = 16

        # the number of unique fields for each position;
        # eg. id_price_o, id_tenor_o, etc.
        self.__STEP_SIZE = self.__TENOR_CLOSE_OFFSET - self.__STRIKE_OFFSET + 1

        # placeholder vals that get populated downstream for other uses:
        self.tensor: np.ndarray = None  # placeholder
        self.pbc: Tuple = None
        self.lrr: float = 0.0
        self.num_valid_days: int = 0
        self.num_winners: int = 0
        self.expected_return: float = 0.0
        self.wager: float = 0.0
        self._post_init()

    def _post_init(self):
        """run some basic validation on the inputs"""
        assert self.legs
        for leg in self.legs:
            assert leg.quantity != 0
            assert isinstance(leg.quantity, int)

        assert self.vol_type in {"max", "real"}

        assert self.df is not None

    def _filter_days(self):
        """filter for day of the week. Eg. only include Mondays and Fridays.
        This assumes Monday=0, Sunday=6. It is generated by
        df['date'].astype('datetime64[ns]').dt.dayofweek
        """
        # filter the df to meet specific parameters:
        # TODO: need to retain future days equivalent to `holding_period`
        self.df = self.df[self.df["day"].isin(self.days_of_the_week)]

    def run(self):
        """
        run a simulation with the given params
        """
        self._get_subset()  # copy of df, narrowed down for a specific time period

        self._backtest()  # BSM over holding period

        # all shift(s) must be before filtering so they don't skip days that are filtered out and get messed up
        if self.vol_type == "max":
            self.df["rolling_vol"] = (
                self.df["max_vol"].shift(1).rolling(self.lookback).mean()
            )
        else:
            # use realized vol
            self.df["rolling_vol"] = (
                self.df["real_vol"].shift(1).rolling(self.lookback).mean()
            )
        # spot movement over the hp
        self.df["spot_movement"] = (
            self.df["close"].shift(-self.holding_period) - self.df["open"]
        ) / self.df["open"]
        # all shifts have been performed; clear out NaN rows.
        self._clean_rows()
        self._filter_iv()
        self._filter_vol()

        if self.sequential_positions:
            self._filter_sequential_positions()
            self.num_valid_days, self.num_winners = self._get_valid_days()
        else:
            self.num_valid_days, self.num_winners = self._get_valid_days()
            if self.sample:
                # sample is mutually exclusive with sequential_positions.
                # optionally limit the rows to a random sample
                self._sample_df()

        # only calculate LRR on sequential_positions
        p, b, c = self._calc_pbc()
        self.pbc = (p, b, c)

        self.expected_return = p * b - (1 - p) * abs(c)
        if self.expected_return <= 0:
            self.wager = 0
        else:
            self.wager = self._kelly(p, self._scale_b(b, c))
        self.lrr = (1 + self.wager * self.expected_return) ** self.num_winners - 1

        self.df.reset_index(inplace=True, drop=True)

    def _parse_relative_strike(
        self,
        a: np.ndarray,
        leg: Option,
    ) -> np.ndarray:
        coef_idx, stddev_idx = 0, 1
        temp = np.zeros((a.shape[0], 2))
        rsval, rstype = leg.relative_strike.split(" ")
        rsval = float(rsval)
        if rstype.casefold() == "x":
            temp[:, coef_idx] = rsval
        else:
            # it's vol-based (Std Dev)
            temp[:, stddev_idx] = (
                rsval
                * a[:, self.__SIGMA_OPEN_IDX]
                / math.sqrt(252 / self.holding_period)
            )
            if leg.right == "call":
                temp[:, coef_idx] = 1 + temp[:, stddev_idx]
            else:
                temp[:, coef_idx] = 1 - temp[:, stddev_idx]
        return temp[:, coef_idx]

    def _calc_strikes(self, a: np.ndarray) -> None:
        """add the exact strikes based upon the relative strike for each leg"""
        msg = self.underlying.min_strike_gap
        for i, leg in enumerate(self.legs):
            step = i * self.__STEP_SIZE
            # below is pseudocode, port this over
            coef = self._parse_relative_strike(a, leg)
            a[:, self.__STRIKE_OFFSET + step] = (
                np.round(a[:, self.__SPOT_OPEN_IDX] * coef / msg) * msg
            )

    def _backtest(self) -> None:
        """
        * Calculate the Black-Scholes-Merton model price for
          each leg at each point in the holding period.

        * The main data structure is a 3D tensor, `tensor`, with:
          - "height" (rows) == same as rows of the original 2D dataframe,
          - "width" (colums) == (dynamic columns * num_legs) + static columns
          - "depth" (slices) == holding period
        """
        # * First create a 2D array, then add the strikes and duplicate that
        #   first 2D slice precisely `self.holding_period` times to make it 3D.
        height = self.df.shape[0]
        width = self.__STRIKE_OFFSET + self.__STEP_SIZE * len(self.legs)
        depth = self.holding_period
        tensor = np.zeros((height, width, depth))
        # np.save("./tests/data/shift_ohlc/cases/tensor", tensor)

        # run these methods once here to calculate the strikes and max risk
        # `_shift_ohlc()` is idempotent, so safe to do this twice on slice 0.
        self._shift_ohlc(tensor[:, :, 0], 0)
        self._calc_strikes(tensor[:, :, 0])
        self._calc_fudge_factor(tensor[:, :, 0])

        for i in range(self.holding_period):
            # * this loop is where the magic happens:
            #   the price of each leg is calculated at each period in the
            #   holding period so that the net position value can be later
            #   aggregated and downstream code can simulate stop-losses or
            #   profit-taking based on the position at any point in the
            #   holding period.
            self._shift_ohlc(tensor[:, :, i], i)
            self._calc_bsm(tensor, i)
            if i == 0:
                # * is relies on one run-through of `_calc_bsm()` for the
                #   opening position values.
                self._calc_position_risk(tensor[:, :, 0])
            self._calc_returns(tensor[:, :, :], i)

        tensor = np.nan_to_num(tensor)  # clear out NaN values and set them to 0
        # position value open (net opening credit for credits, net debit for debits)
        self.df["net_pos_open"] = tensor[:, self.__NET_POS_OPEN_IDX, 0]
        # position_value_close (net closing debit for credits, net credit for debits)
        self.df["net_pos_close"] = tensor[:, self.__NET_POS_CLOSE_IDX, -1]

        self.df["max_risk"] = tensor[:, self.__MAX_RISK_IDX, 0]

        if self.scalping or self.stop_loss:
            tensor = self._scalp_stoploss(tensor)

        self.df["spot_open"] = tensor[:, self.__SPOT_OPEN_IDX, 0]
        # spot price at the time the position was closed [hp] trading days after open
        self.df["spot_close"] = tensor[:, self.__SPOT_CLOSE_IDX, -1]

        # the highest net position value at a market open over the holding_period
        open_max = np.amax(tensor[:, self.__NET_POS_OPEN_IDX, :], axis=1)
        # the highest net position value at a market close over the holding_period
        close_max = np.amax(tensor[:, self.__NET_POS_CLOSE_IDX, :], axis=1)
        # the lowest net position value at a market open over the holding_period
        open_min = np.amin(tensor[:, self.__NET_POS_OPEN_IDX, :], axis=1)
        # the lowest net position value at a market close over the holding_period
        close_min = np.amin(tensor[:, self.__NET_POS_CLOSE_IDX, :], axis=1)

        # MAX position value (open, close) over the holding_period
        self.df["pos_max"] = np.maximum(open_max, close_max)
        # MIN position value (open, close) over the holding_period
        self.df["pos_min"] = np.minimum(open_min, close_min)
        self.df["risk_return"] = tensor[:, self.__RISK_RETURN_IDX, -1]
        self.df["pom_return"] = tensor[:, self.__POM_RETURN_IDX, -1]
        self.df["winner"] = (
            self.df["risk_return"] >= 0
        )  # (bool), used by later functions. TODO: can this be removed or calculated later?

        # spot price at the time the position was opened ... ?
        self.tensor = tensor

    def _shift_ohlc(self, a: np.ndarray, shift: int) -> None:
        """
        * _shift_ohlc() shifts the ohlc data by `shift` periods and
          decrements the opening and closing tenors by `shift`.
        * these changes are assigned in place to the slice `a`
          that is passed in.
        """
        rth = 0.27  # regular trading hours == 6.5 hours ~= 0.27 days

        a[:, self.__SPOT_OPEN_IDX] = self.df["open"].shift(-shift).to_numpy()
        a[:, self.__SPOT_CLOSE_IDX] = self.df["close"].shift(-shift).to_numpy()
        a[:, self.__SIGMA_OPEN_IDX] = self.df["iv_open"].shift(-shift).to_numpy()
        a[:, self.__SIGMA_CLOSE_IDX] = self.df["iv_close"].shift(-shift).to_numpy()

        for i, leg in enumerate(self.legs):
            step = i * self.__STEP_SIZE
            # * Decrement the tenor of the option by `shift`.
            # * This method is called on each iteration of a loop, which calculates
            #   position value at each unit (typically days) in the holding period.
            a[:, self.__TENOR_OPEN_OFFSET + step] = leg.tenor - shift
            a[:, self.__TENOR_CLOSE_OFFSET + step] = max(0, leg.tenor - shift - rth)

    def _calc_bsm(self, a: np.ndarray, i) -> None:
        """
        calculat the value of each option at each day open/close in the holding period
        """
        spot_open = a[:, self.__SPOT_OPEN_IDX, i]
        spot_close = a[:, self.__SPOT_CLOSE_IDX, i]
        sigma_open = a[:, self.__SIGMA_OPEN_IDX, i]
        sigma_close = a[:, self.__SIGMA_CLOSE_IDX, i]

        for j, leg in enumerate(self.legs):
            step = j * self.__STEP_SIZE
            k = a[:, self.__STRIKE_OFFSET + step, 0]  # strikes only exist at slice 0
            to = a[:, self.__TENOR_OPEN_OFFSET + step, i] / 252
            tc = a[:, self.__TENOR_CLOSE_OFFSET + step, i] / 252

            if leg.right == "call":
                po = self._calc_call(spot_open, k, sigma_open, to)
                pc = self._calc_call(spot_close, k, sigma_close, tc)
            else:
                po = self._calc_put(spot_open, k, sigma_open, to)
                pc = self._calc_put(spot_close, k, sigma_close, tc)

            # * add the position premium to the price offset (this is on a per-leg level)
            #   notice that the quantity is not applied here; the quantity is applied
            #   more precisely downstream in `_assess_position_risk()`.
            a[:, self.__PRICE_OPEN_OFFSET + step, i] = po
            a[:, self.__PRICE_CLOSE_OFFSET + step, i] = pc

            # * the overall position o/c values, unlike the leg-specific prices,
            #   should account for the quantity.
            a[:, self.__NET_POS_OPEN_IDX, i] += po * leg.quantity
            a[:, self.__NET_POS_CLOSE_IDX, i] += pc * leg.quantity

    def _calc_d1(
        self, s: np.ndarray, k: np.ndarray, sigma: np.ndarray, t: np.ndarray, r: float
    ) -> np.ndarray:
        return (np.log(s / k) + (r + sigma**2 / 2) * t) / (sigma * np.sqrt(t))

    def _calc_d2(self, d1: np.ndarray, sigma: np.ndarray, t: np.ndarray) -> np.ndarray:
        return d1 - sigma * np.sqrt(t)

    def _calc_call(
        self, s: np.ndarray, k: np.ndarray, sigma: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        d1 = self._calc_d1(s, k, sigma, t, self.riskfree_rate)
        d2 = self._calc_d2(d1, sigma, t)
        call = np.maximum(
            0, s * norm.cdf(d1) - k * np.exp(-self.riskfree_rate * t) * norm.cdf(d2)
        )
        return np.nan_to_num(call)

    def _calc_put(
        self, s: np.ndarray, k: np.ndarray, sigma: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        d1 = self._calc_d1(s, k, sigma, t, self.riskfree_rate)
        d2 = self._calc_d2(d1, sigma, t)
        put = np.maximum(
            0, k * np.exp(-self.riskfree_rate * t) * norm.cdf(-d2) - s * norm.cdf(-d1)
        )
        return np.nan_to_num(put)

    def _calc_fudge_factor(self, a: np.ndarray) -> None:
        """
        * Estimate the max upside and downside based on
          ±self.max_deviations standard deviation move in the underlying.
        * This is needed to be able to estimate the risk for naked calls,
          and also serves as a more realistic floor for puts, where
          spot of $0 is only marginally useful in the real world for
          calculating the max payout of a put.
        * NOTE: for naked short positions, increasing max_deviations is
                associated with less severe negative risk-returns.
                eg. instead of -4x losses, it might be -1.2x because the
                estimated risk using a higher max_deviations is greater
                so more collateral is reserved for these positions,
                which is good in the long run.
        * to calculate the standard deviation, take the number of desired
          deviations to use, `self.max_deviations`, multiplied by the
          iv_open for the day (this is unique to each row in `a`), divided
          by normalized iv.
          * normalize IV (quoted as annual) for the holding period by
            taking sqrt(trading_days / hp), where 252 is the average
            number of trading days per year.
        """
        spot_open = a[:, self.__SPOT_OPEN_IDX]
        sigma = a[:, self.__SIGMA_OPEN_IDX]
        std_devs: np.ndarray = sigma / math.sqrt(252 / self.holding_period)
        max_movement = self.max_deviations * std_devs

        a[:, self.__MAX_DOWNSWING_IDX] = np.maximum(0, spot_open * (1 - max_movement))
        a[:, self.__MAX_UPSWING_IDX] = spot_open * (1 + max_movement)

    def _calc_position_risk(self, a: np.ndarray, plot: bool = False) -> None:
        """
        a risk-based test to determine overall risk

        * the goal here is to take each opening and calculate
          the max risk in a few eventualities.

        across a range of prices (and IV combos, though IV doesn't really matter in extreme moves), tenor
        """
        # * 3 1D arrays to store generated spot prices, sigma values,
        #   and portfolio outcome values.
        spot, sigma, val = (
            np.zeros((a.shape[0])),
            np.zeros((a.shape[0])),
            np.zeros((a.shape[0])),
        )
        for _ in range(self.num_simulations):
            val[:] = 0  # zero out val array
            t = random.random() * self.holding_period
            sigma = np.random.choice(a[:, self.__SIGMA_OPEN_IDX])
            spot = np.random.uniform(
                a[:, self.__MAX_DOWNSWING_IDX], a[:, self.__MAX_UPSWING_IDX]
            )
            for i, leg in enumerate(self.legs):
                step = i * self.__STEP_SIZE
                k = a[:, self.__STRIKE_OFFSET + step]
                # * subtract no more than `holding_period` from the tenor.
                # * this allows the position to maintain the offsets between
                #   tenors.
                t = (a[:, self.__TENOR_OPEN_OFFSET + step] - t) / 252
                if leg.right == "call":
                    val += leg.quantity * self._calc_call(spot, k, sigma, t)
                else:
                    val += leg.quantity * self._calc_put(spot, k, sigma, t)

            # * subtract the simulated value from the opening value.
            # * the opening value is negative (due to the fact that it's
            #   calculated within `_calc_bsm` and only the actualy
            #   opening needs a negative value and is easily corrected
            #   like below. All other uses for that value assume it's
            #   the _closing_ value and therefore should be negative.)
            val = (-a[:, self.__NET_POS_OPEN_IDX]) + val

            a[:, self.__MAX_RISK_IDX] = np.minimum(a[:, self.__MAX_RISK_IDX], val)
            a[:, self.__MAX_RETURN_IDX] = np.maximum(a[:, self.__MAX_RETURN_IDX], val)

            if plot:
                # * grab the last position since the payoff diagram
                #   should be be the same for all positions.
                # * divide the randomly generated spot by the spot open at index `-1`
                #   and divide the payoff by the net position open to give relative payoff
                plt.scatter(
                    spot[-1] / a[-1, self.__SPOT_OPEN_IDX],
                    val[-1] / -a[-1, self.__MAX_RISK_IDX],
                    s=1,
                )

    def _calc_returns(self, a: np.ndarray, i: int) -> None:
        """calculate the overall position return at the given period, i."""
        # nvc is the net position value change between t=0 and t=i
        nvc = a[:, self.__NET_POS_CLOSE_IDX, i] - a[:, self.__NET_POS_OPEN_IDX, 0]

        a[:, self.__RISK_RETURN_IDX, i] = nvc / a[:, self.__MAX_RISK_IDX, 0]
        a[:, self.__POM_RETURN_IDX, i] = nvc / a[:, self.__MAX_RETURN_IDX, 0]

    def _kelly(self, p: float, b: float):
        """the kelly criterion for wagering
        p: probability of win
        b: net payout for a win"""
        return p + (p - 1) / b

    def _scale_b(self, b: float, c: float):
        """the kelly criterion expects max loss of 1.0
        so scale the winners and losers accordingly."""
        if c >= -1.0:
            # no need to scale
            return b
        return b / abs(c)

    def _scalp_stoploss(self, a):
        """
        _scalp_stoploss() oversees positions and simulated position exits based on:
        * self.stop_loss: a negative float indicating the max risk return loss allowed
        * one of {self.pom_threshold, self.risk_return_threshold}: which indicate
          profit-taking threshold for returns
            * eg. POM 0.5 == 50% of max possible
            * eg RR 0.2 == 20% return on capital at risk
        * based on how the stop_loss and scalping are simulated (by locking in the
          value at the trigger point), it's possible to simulate a stoploss with
          scalping due to the fact that the earliest event will "win" and overwrite
          all future values, regardless of the order in which things were evaluated
          in the code.
        """
        if self.stop_loss:
            # * use np.argmax() on a boolean array to find the first truthy index,
            #   ie. return the first period that triggers a stop loss. This period
            #   is then "locked in" for the rest of the holding period, simulating
            #   the stoploss triggering.
            # axis == 2???
            trigger_points = np.argmax(
                a[:, self.__RISK_RETURN_IDX, :] <= self.stop_loss, axis=1
            )
            for row, tp in enumerate(trigger_points):
                if tp == 0 and a[row, self.__RISK_RETURN_IDX, tp] > self.stop_loss:
                    # * if the trigger point is zero it means that either the stoploss
                    #   was never reached or that it was reached immediately.
                    # * first check for tp == 0, then check that the risk return at
                    #   index 0 is greater than the stoploss. If it is, we know that
                    #   the stoploss never triggers and we need to skip the below
                    #   logic (which would lock in the risk return at index 0).
                    continue
                # freeze the PoM return at the time of scalp
                a[row, self.__POM_RETURN_IDX, tp:] = a[row, self.__POM_RETURN_IDX, tp]
                # freeze the risk return at the time of scalp
                a[row, self.__RISK_RETURN_IDX, tp:] = a[row, self.__RISK_RETURN_IDX, tp]
                # TODO: find out why is POM positive when stoploss triggers sometimes?
        if not self.scalping:
            # prevent the rest from executing
            return a
        if self.pom_threshold:
            indices = np.argmax(
                a[:, self.__POM_RETURN_IDX, :] >= self.pom_threshold, axis=1
            )
            for row, tp in enumerate(indices):
                if tp == 0 and a[row, self.__POM_RETURN_IDX, tp] < self.pom_threshold:
                    # the threshold was never reached
                    continue
                a[row, self.__POM_RETURN_IDX, tp:] = self.pom_threshold
                a[row, self.__RISK_RETURN_IDX, tp:] = a[row, self.__RISK_RETURN_IDX, tp]
        else:
            # scalp on risk return
            indices = np.argmax(
                a[:, self.__RISK_RETURN_IDX, :] >= self.risk_return_threshold, axis=1
            )
            for row, tp in enumerate(indices):
                if tp == 0 and (
                    a[row, self.__RISK_RETURN_IDX, tp] < self.risk_return_threshold
                ):
                    # the threshold was never reached
                    continue
                # freeze the PoM return at the time of scalp
                a[row, self.__POM_RETURN_IDX, tp:] = a[row, self.__POM_RETURN_IDX, tp]
                # freeze the risk return at the time of scalp
                a[row, self.__RISK_RETURN_IDX, tp:] = self.risk_return_threshold
        return a

    def plot(self, show=100, all=False, pom=True, risk=True):
        """this plots all positions, not just sequential positions."""
        # TODO: use attributes instead of passing params around...
        # negate the shifted rows (TODO: pass in hp as a parameter)
        a = self.tensor[: -self.holding_period, :, :]
        if risk:
            # first, plot the percent of max return
            rows = a.shape[0]
            x = np.arange(a.shape[2])
            if all:
                for i in range(rows):
                    y = a[i, self.__RISK_RETURN_IDX, :]
                    plt.plot(x, y)
            else:
                for i in range(show):
                    # sample the df
                    row = int(np.random.random() * rows)
                    y = a[row, self.__RISK_RETURN_IDX, :]
                    plt.plot(x, y, alpha=0.25)  # , color='k')
            mean = np.mean(a[:, self.__RISK_RETURN_IDX, :])
            median = np.median(a[:, self.__RISK_RETURN_IDX, :])
            print(f"rr mean: {100*mean:.2f}% | rr median: {100*median:.2f}%")
            plt.axhline(mean, color="b", linestyle="solid", linewidth=1)
            plt.axhline(median, color="r", linestyle="dashed", linewidth=1)
            plt.title("Risk return")
            plt.show()
        if pom:
            # now plot the risk return
            rows = a.shape[0]
            x = np.arange(a.shape[2])
            if all:
                for i in range(rows):
                    y = a[i, self.__POM_RETURN_IDX, :]
                    plt.plot(x, y)
            else:
                for i in range(show):
                    # sample the df
                    row = int(np.random.random() * rows)
                    y = a[row, self.__POM_RETURN_IDX, :]
                    plt.plot(x, y, alpha=0.25)  # , color='k')
            mean = np.mean(a[:, self.__POM_RETURN_IDX, :])
            median = np.median(a[:, self.__POM_RETURN_IDX, :])
            print(f"PoM mean: {100*mean:.2f}% | PoM median: {100*median:.2f}%")
            plt.axhline(mean, color="b", linestyle="solid", linewidth=1)
            plt.axhline(median, color="r", linestyle="dashed", linewidth=1)
            plt.title("POM return")
            plt.show()

    def plot_payoff(self):
        self._calc_position_risk(self.tensor[:, :, 0], plot=True)
        plt.title("Position Payoff Diagram")
        plt.xlabel("Relative spot price")
        plt.ylabel("Risk Return")
        plt.grid(axis="both", alpha=0.34)
        plt.show()

    def to_pickle(self, path: str) -> None:
        self.df.to_pickle(path)

    def _clean_rows(self):
        if self.holding_period > 0:
            # ignore the first and last few rows that will be NaN due to shift()s above
            # +1 to include the last valid entry (equivalent to [start_date:-(hp-1)])
            self.df = self.df[self.lookback : -self.holding_period + 1]
        else:
            # just ignore the first rows
            self.df = self.df[self.start_date :]

    def _add_spot_returns(self):
        """calculate the spot returns from open"""
        # shift(1) is to prevent look-ahead bias
        self.df["1_day_return"] = (
            self.df["close"].shift(1) - self.df["open"].shift(2)
        ) / self.df["open"].shift(2)
        self.df["20_day_avg_daily_return"] = (
            self.df["1_day_return"].shift(1).rolling(20).mean()
        )
        self.df["5_day_return"] = (
            self.df["close"].shift(1) - self.df["open"].shift(5)
        ) / self.df["open"].shift(5)
        self.df["10_day_return"] = (
            self.df["close"].shift(1) - self.df["open"].shift(10)
        ) / self.df["open"].shift(10)
        self.df["15_day_return"] = (
            self.df["close"].shift(1) - self.df["open"].shift(15)
        ) / self.df["open"].shift(15)
        self.df["20_day_return"] = (
            self.df["close"].shift(1) - self.df["open"].shift(20)
        ) / self.df["open"].shift(20)
        self.df["50_day_return"] = (
            self.df["close"].shift(1) - self.df["open"].shift(50)
        ) / self.df["open"].shift(50)
        self.df["100_day_return"] = (
            self.df["close"].shift(1) - self.df["open"].shift(100)
        ) / self.df["open"].shift(100)

    def analyze_results(self):
        self._add_spot_returns()
        y = self.df["risk_return"]  # the y axis is returns
        subjects = self.df[
            [
                "iv_open",
                "rolling_vol",
                "previous_vol",
                "close_open",
                "1_day_return",
                "20_day_avg_daily_return",
                "5_day_return",
                "10_day_return",
                "20_day_return",
                "50_day_return",
                "100_day_return",
                "day",
            ]
        ]
        for subject in subjects:
            # * calculate some regressions and plot
            #   scatterplots with regression imbedded
            x = self.df[subject]
            try:
                m, b = np.polyfit(x, y, 1)
            except (np.linalg.LinAlgError, TypeError):
                continue
            yp = np.polyval([m, b], x)
            plt.plot(x, yp)
            plt.scatter(x, y)
            plt.title(f"risk_return vs {subject}")
            plt.show()

    def _calc_pbc(self):
        # p: probability of a win
        # b: payout for a win
        # c: capital loss in the event of a loss (essentially `b` for losses)
        if self.df[self.df["risk_return"] >= 0].count()[0] == 0:
            p, b = 0, 0
        else:
            # df is filtered for valid days, so use its entirety
            p = self.df[self.df["risk_return"] >= 0].count()[0] / len(self.df)
            b = self.df[self.df["risk_return"] >= 0]["risk_return"].mean()
        if self.df[self.df["risk_return"] < 0].count()[0] == 0:
            # No losers found, return 0 since mean of empty sequence will return nan
            return p, b, 0
        # mean payout for losers
        c = self.df[self.df["risk_return"] < 0]["risk_return"].mean()
        # median payout for losers
        d = self.df[self.df["risk_return"] < 0]["risk_return"].median()
        return p, b, min(c, d)

    def _filter_iv(self):
        """filter position entry on IV range"""
        if self.iv_greater_than:
            self.df = self.df[self.df["iv_open"] >= self.iv_min_threshold]
        if self.iv_less_than:
            self.df = self.df[self.df["iv_open"] <= self.iv_max_threshold]

    def _filter_vol(self):
        """filter position entry on realized vol range"""
        if self.vol_greater_than:
            self.df = self.df[self.df["rolling_vol"] > self.vol_threshold]
        else:
            self.df = self.df[self.df["rolling_vol"] < self.vol_threshold]

    def _filter_sequential_positions(self):
        # * limit 1 open position at a time, simulating real trading
        #   where trades are sequential and non-concurrent
        self.df["valid"] = False
        self.df["idx"] = self.df.index
        next_valid = -1  # initial value so the below condition runs the first time
        for index in self.df.copy()["idx"]:
            if index > next_valid:
                self.df.at[index, "valid"] = True
                next_valid = index + self.holding_period
        self.df = self.df[self.df["valid"] == True]

    def _get_valid_days(self):
        # must happen after all filtering but before sequential or sample.
        valid_days = len(self.df)
        # TODO: this is just a thought, not final...
        winners = self.df[self.df["risk_return"] > 0].count()[0]
        # win = winners / valid_days
        return valid_days, winners

    def _sample_df(self):
        """
        random sample of the dataframe output.
        NOTE: this is mutually exclusive with sequential positions.
        this uses the variable `self.sample`, and depending on if it's
        a float or an int, we'll filter a fraction (float) or number, n (int).
        """
        if not self.sample:
            return
        if self.sample < 1:
            # sample a random percent of the dataframe
            self.df = self.df.sample(frac=self.sample)
        else:
            # sample >= 1, so sample a given random number of rows
            self.df = self.df.sample(n=self.sample)

    def find_max_losers(df, silently=True, return_tally=False):
        # NOT vectorized.
        maximum = 0
        counter = 0
        for i, winner in enumerate(df["winner"]):
            if winner == False:
                counter += 1
            else:
                if counter == maximum and counter > 0:
                    # find all local maxima
                    if not silently:
                        print(f"{df['date'].iloc[i-1]} ({counter})")
                counter = 0
            if counter > maximum:
                maximum = counter
        if not silently:
            print(f"max consecutive losing days: {maximum}")
        if return_tally:
            return maximum

    def generate_report(self):
        mean_move = (
            self.df["spot_movement"].abs().mean()
        )  # median is more useful than mean.
        median_move = (
            self.df["spot_movement"].abs().median()
        )  # median is more useful than mean.
        print("=====================================")
        print(f"\t\tgenerating report for {self.underlying.symbol}")
        print("-------------------------------------")
        print(
            f"\n{100*self.num_valid_days/(len(self.df) - self.holding_period - self.lookback):.1f}% of days meet criteria ({self.num_valid_days})"
        )
        print(
            f"holding period = {self.holding_period} trading (~{round(7/5*self.holding_period)} calendar) days"
        )
        print("-------------------------------------")
        print(f"median spot movement:\t{100 * median_move:.2f}%")
        print(f"mean spot movement:\t{100 * mean_move:.2f}%")
        print("-------------------------------------")
        print(f"probability of a win:\t{100 * self.pbc[0]:.2f}%")
        print(f"payout for winners:\t{100 * self.pbc[1]:.2f}%")
        print(f"loss for losers:\t{100 * self.pbc[2]:.2f}%")
        print(f"expected_return:\t{100 * self.expected_return:.2f}%")
        print(f"wager:\t\t\t{100 * self.wager:.2f}%")
        print(f"wager expected_return:\t{100 * self.wager * self.expected_return:.2f}%")
        print(f"average expected lrr:\t{100 * self.lrr:,.0f}%")
        # print('-------------------------------------')
        # print(f'true computed lrr:\t{100 * true_lrr:,.0f}%')
        # print(f'computed bsm lrr:\t{100 * bsm_lrr:,.0f}%')
        # print(f'$100,000 at true lrr:\t${100000 * (1 + true_lrr):,.0f}')
        print("-------------------------------------")
        print(f"mean iv open:\t\t{100 * self.df['iv_open'].mean():.2f}")
        print(f"median iv open\t\t{100 * self.df['iv_open'].median():.2f}")
        print("-------------------------------------")
        print(f"cumulative risk return:\t{100 * self.df['risk_return'].sum():.2f}%")
        print(f"highest risk return:\t{100 * self.df['risk_return'].max():.2f}%")
        # !NOTE: mean risk == expected_return!
        print(f"mean risk return:\t{100 * self.df['risk_return'].mean():.2f}%")
        print(f"median risk return:\t{100 * self.df['risk_return'].median():.2f}%")
        print("=====================================")

    def get_consecutive_failures(
        max_consecutive_days, duration, sequential_positions=False
    ):
        if sequential_positions:
            if max_consecutive_days == 0:
                message = "[PASSED]"
            else:
                message = (
                    f"!!!WARNING!!! {max_consecutive_days} consecutive losing trades!!!"
                )
            return message
        else:
            if max_consecutive_days >= duration:
                if duration > 0:
                    losing_trades = math.ceil(max_consecutive_days / duration)
                else:
                    losing_trades = max_consecutive_days
                message = f"!!!WARNING!!! {losing_trades} consecutive losing trades!!! ({max_consecutive_days} days)"
            else:
                message = "[PASSED]"
            return message

    def _get_subset(self):
        # * lookback and hp are here so that this subset filtering can be performed *before* calculating BSM prices
        #   for performance reasons. By adding hp to the end and subtracting lookback from the beginning, shifts
        #   in later functions can be performed and the beginning and end of the subset aren't NaN due to shifting
        #   out of range of the df. Later trimming will remove rows to get back to the intended subset.
        # NOTE: `None` works as a valid index and df[None:None] returns the entire df
        lookback = timedelta(days=self.lookback)
        hp = timedelta(days=self.holding_period)
        if self.start_date is not None:
            self.start_date = (
                datetime.strptime(self.start_date, "%Y-%m-%d").date() - lookback
            )
            self.start_date = self.df[self.df["date"] >= self.start_date].index[0]
        if self.end_date is not None:
            self.end_date = datetime.strptime(self.end_date, "%Y-%m-%d").date() + hp
            # +1 to include the last entry.
            self.end_date = self.df[self.df["date"] <= self.end_date].index[-1] + 1
        self.df = self.df[self.start_date : self.end_date].copy()

    def plot_histograms(df):
        # grab the mean and median for PoM and RR for plotting inside the histograms:
        risk_mean = df["risk_return"].mean()
        pom_mean = df["pom_return"].mean()
        risk_median = df["risk_return"].median()
        pom_median = df["pom_return"].median()

        # plot the risk return
        plt.hist(df["risk_return"], bins=50, histtype="stepfilled")
        plt.axvline(risk_mean, color="k", linestyle="dashed", linewidth=1)
        plt.axvline(risk_median, color="r", linestyle="solid", linewidth=1)
        plt.title("RR histogram")
        plt.show()

        # plot the PoM return
        plt.hist(df["pom_return"], bins=50, histtype="stepfilled")
        plt.axvline(pom_mean, color="k", linestyle="dashed", linewidth=1)
        plt.axvline(pom_median, color="r", linestyle="solid", linewidth=1)
        plt.title("PoM histogram")
        plt.show()

    def get_true_lrr(df, wager, b, sequential_positions=False, silently=True):
        if not sequential_positions:
            # the true lrr can't be calculated if simultaneous trades are counted.
            return float("nan"), float("nan")
        # calculate the true long run return using kelly criterion SEQUENTIALLY.
        df["winner"] = (df["risk_return"] >= 0) * 1  # quantify winners
        df["srr"] = (df["winner"] * wager * b) + (
            (1 + df["winner"] == 1) * -wager
        )  # short-run return
        true_lrr = 1.0
        for i, trade_return in enumerate(df["srr"]):
            true_lrr *= 1 + trade_return
            if not silently:
                print(f"trade {i+1}: long run return after trade: {true_lrr:.2f}")
        bsm_lrr = 1.0
        for i, trade_return in enumerate(df["risk_return"]):
            bsm_lrr *= 1 + (wager * trade_return)
            if not silently:
                print(f"trade {i+1}: long run return after trade: {bsm_lrr:.2f}")
        true_lrr -= 1
        bsm_lrr -= 1
        if not silently:
            print(f"final long run return: {true_lrr:,.2f}%")
            print(f"final long run bsm return: {bsm_lrr:,.2f}%")
        return true_lrr, bsm_lrr
