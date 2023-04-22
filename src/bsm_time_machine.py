import math
import time
import random
from datetime import datetime, date, timedelta
from dataclasses import dataclass
from typing import Optional, List, Literal

from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_market_calendars as pmc

pd.set_option("display.max_rows", 50, "display.max_columns", 10)


class Option:
    """private class that is wrapped by
    the public classes `Call` and `Put`"""

    def __init__(self, rs, tenor, right, quantity):
        # * rs is the relative strike in one of the two formats below:
        #   '0.5 SD' or '1.05 x' (multiple of strike)

        self.id = f"{rs.replace(' ', '')}_{tenor}_{right}"
        self.relative_strike_coef = 0.0
        self.relative_strike_type = 0.0
        self.relative_strike = rs
        self.tenor = tenor
        self.right = right
        self.quantity = quantity


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
        min_tick: float = 0.05,
        spread_loss: float = 0.05,
        min_strike_gap: int = 5,
    ):
        self.symbol = symbol
        self.min_tick = min_tick
        self.spread_loss = spread_loss
        self.min_strike_gap = min_strike_gap


class Position:
    """
    # * The Position class takes arguments on instantiation
    #   that define the parameters you'd like to backtest.
    # * All class attributes are public and can be directly
    #   modified as needed between tests.
    # * The only public method of the class is `run()`, which
    #   runs the backtest and returns a summary and optionally
    #   plots the data.

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
        days_of_the_week: set = (0, 1, 2, 3, 4),
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
        num_simulations=15000,
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
        self.subset_size = 0
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

        self._post_init()

    def _post_init(self):
        """run some basic validation on the inputs"""
        print("legs:", self.legs, type(self.legs))
        assert self.legs
        for leg in self.legs:
            assert leg.quantity != 0
            assert isinstance(leg.quantity, int)

        assert self.df is not None

    def _filter_days(self):
        """filter for day of the week. Eg. only include Mondays and Fridays."""
        # filter the df to meet specific parameters:
        # TODO: need to retain future days equivalent to `holding_period`
        self.df = self.df[self.df["day"].isin(self.days_of_the_week)]

    def run(self):
        """
        what are the steps when the engine is `run`??

        1. get_subset()
            - shift/trim the dates of the df
        2. calculate the option strikes for each date in the df
            - this will need to be revamped for the new positions list
        2.5 (might need to calculate the total risk?)
        3. calculate the BSM prices in a 2D array across the holding period
            - this function will also need an overhaul to be dynamic and use the ID instead of
        4. filter the df for things like trading days, filter IV and realized vol, clean df
        5.

        """
        print("running...")
        # copy of df, narrowed down for a specific time period
        self._trim_df()
        print("just trimmed df")

        a = self._backtest()  # BSM over hp
        # plot_positions(a, show=100, all=False, pom=True, risk=True)  # DAT

        # # all shift(s) must be before filtering so they don't skip days that are filtered out and get messed up
        # if self.vol_type == "max":
        #     self.df["rolling_vol"] = (
        #         self.df["max_vol"].shift(1).rolling(self.lookback).mean()
        #     )
        # elif self.vol_type == "real":
        #     self.df["rolling_vol"] = (
        #         self.df["max_vol"].shift(1).rolling(self.lookback).mean()
        #     )
        # else:
        #     raise ValueError(
        #         "invalid vol_type encountered. must be either `max` or `real`"
        #     )
        # # spot movement over the hp
        # df["spot_movement"] = (df["close"].shift(-hp) - df["open"]) / df["open"]
        # # all shifts have been performed; clear out NaN rows.
        # df = clean_rows(df, lookback, hp)

        # df = filter_iv(
        #     df, iv_greater_than, iv_min_threshold, iv_less_than, iv_max_threshold
        # )
        # df = filter_vol(df, vol_greater_than, vol_threshold)

        # if sequential_positions:
        #     df = filter_sequential_positions(df, hp)
        #     valid_days, num_winners = get_valid_days(df)
        #     # used as denominator for % days valid post-sequential_positions filter
        #     max_potential_days = subset_size // hp
        # else:
        #     valid_days, num_winners = get_valid_days(df)
        #     if self.sample:
        #         # TODO: add an assert to enforce this
        #         # sample is mutually exclusive with sequential_positions.
        #         # optionally limit the rows to a random sample
        #         self.df = sample_df()

        # # only calculate LRR on sequential_positions
        # p, b, c = calc_pbc(df)

        # expected_return = p * b - (1 - p) * abs(c)
        # if expected_return <= 0:
        #     wager = 0
        # else:
        #     wager = kelly(p, scale_b(b, c))
        # #         if not is_hedged:
        # #             wager = kelly(p, scale_b(b, c))
        # #         else:
        # #             wager = kelly(p, b)
        # lrr = (1 + wager * expected_return) ** num_winners - 1
        # if self.lrr_only:
        #     return lrr, num_winners
        # df.reset_index(inplace=True, drop=True)

        # print(df[["date", "spo", "short_put_k", "iv_open"]])
        # print(list(df))

        # if not self.return_only:
        #     #         print_df(df, is_credit)
        #     if plot:
        #         # TODO: change these arguments to be parameters.
        #         # TODO: filter pom and risk based on if they apply, eg. long-only positions have no PoM.
        #         plot_positions(a, show=100, all=False, pom=True, risk=True)
        # return (
        #     lrr,
        #     valid_days,
        #     num_winners,
        #     p,
        #     b,
        #     c,
        #     df,
        #     expected_return,
        #     wager,
        #     subset_size,
        # )

    def _trim_df(self):
        """
        * lookback and hp are here so that this subset filtering can be performed *before* calculating BSM prices
          for performance reasons. By adding hp to the end and subtracting lookback from the beginning, shifts
          in later functions can be performed and the beginning and end of the subset aren't NaN due to shifting
          out of range of the df. Later trimming will remove rows to get back to the intended subset.
        NOTE: `None` works as valid indexes and df[None:None] returns the entire df
        """
        lookback = timedelta(days=self.lookback)
        holding_period = timedelta(days=self.holding_period)
        start, end = None, None
        if self.start_date is not None:
            start = datetime.strptime(self.start_date, "%Y-%m-%d").date() - lookback
            start = self.df[self.df["date"] >= self.start_date].index[0]
        if self.end_date is not None:
            end = datetime.strptime(self.end_date, "%Y-%m-%d").date() + holding_period
            end = (
                self.df[self.df["date"] <= self.end_date].index[-1] + 1
            )  # +1 to include the last entry.
        self.df = self.df[start:end].copy()
        self.subset_size = len(self.df)  # used in calculations later

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
        elif rstype.casefold() == "sd":
            temp[:, stddev_idx] = (
                rsval
                * a[:, self.__SIGMA_OPEN_IDX]
                / math.sqrt(252 / self.holding_period)
            )
            if leg.right == "call":
                temp[:, coef_idx] = 1 + temp[:, stddev_idx]
            else:
                temp[:, coef_idx] = 1 - temp[:, stddev_idx]
        else:
            raise ValueError("Unexpected relative strike type encountered")
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

    def _backtest(self) -> np.ndarray:
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

        # run these methods once here to calculate the strikes and max risk
        # `_shift_ohlc()` is idempotent, so safe to do this twice on slice 0.
        self._shift_ohlc(tensor[:, :, 0], 0)
        print("just shifted")
        self._calc_strikes(tensor[:, :, 0])
        print("just calculated_strikes")
        print(tensor[0, self.__STRIKE_OFFSET, 0])
        self._calc_fudge_factor(tensor[:, :, 0])
        print("just calculated fudge factor")

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

        # position value open (net opening credit for credits, net debit for debits)
        self.df["net_pos_open"] = tensor[:, self.__NET_POS_OPEN_IDX, 0]
        # position_value_close (net closing debit for credits, net credit for debits)
        self.df["net_pos_close"] = tensor[:, self.__NET_POS_CLOSE_IDX, -1]

        self.df["max_risk"] = tensor[:, self.__MAX_RISK_IDX, 0]

        if self.scalping or self.stop_loss:
            tensor = self.scalp_stoploss(tensor)

        # self.df["spot_open"] = tensor[:, self.__SPOT_OPEN_IDX, 0]
        # # spot price at the time the position was closed [hp] trading days after open
        # self.df["spot_close"] = tensor[:, self.__SPOT_CLOSE_IDX, -1]

        # # the highest net position value at a market open over the holding_period
        # open_max = np.amax(tensor[:, self.__NET_POS_OPEN_IDX, :], axis=1)
        # # the highest net position value at a market close over the holding_period
        # close_max = np.amax(tensor[:, self.__NET_POS_CLOSE_IDX, :], axis=1)
        # # the lowest net position value at a market open over the holding_period
        # open_min = np.amin(tensor[:, self.__NET_POS_OPEN_IDX, :], axis=1)
        # # the lowest net position value at a market close over the holding_period
        # close_min = np.amin(tensor[:, self.__NET_POS_CLOSE_IDX, :], axis=1)

        # # MAX position value (open, close) over the holding_period
        # self.df["pos_max"] = np.maximum(open_max, close_max)
        # # MIN position value (open, close) over the holding_period
        # self.df["pos_min"] = np.minimum(open_min, close_min)
        # self.df["risk_return"] = tensor[:, self.__RISK_RETURN_IDX, -1]
        # self.df["pom_return"] = tensor[:, self.__POM_RETURN_IDX, -1]
        # self.df["winner"] = (
        #     self.df["risk_return"] >= 0
        # )  # (bool), used by later functions. TODO: can this be removed or calculated later?

        # # spot price at the time the position was opened ... ?
        return tensor  # return a so it can be optionally plotted and df for further analysis

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

    def _calc_position_risk(self, a: np.ndarray) -> None:
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
            t = random.random() * self.holding_period / 252 + 0.000001
            sigma = np.random.choice(self.__SIGMA_OPEN_IDX)
            spot = np.random.uniform(
                a[:, self.__MAX_DOWNSWING_IDX], a[:, self.__MAX_UPSWING_IDX]
            )
            for i, leg in enumerate(self.legs):
                k = a[:, self.__STRIKE_OFFSET + i * self.__STEP_SIZE]
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

    def _calc_returns(self, a: np.ndarray, i: int) -> None:
        """calculate the overall position return at the given period, i."""
        # nvc is the net position value change between t=0 and t=i
        nvc = a[:, self.__NET_POS_CLOSE_IDX, i] - a[:, self.__NET_POS_OPEN_IDX, 0]

        a[:, self.__RISK_RETURN_IDX, i] = nvc / a[:, self.__MAX_RISK_IDX, 0]
        a[:, self.__POM_RETURN_IDX, i] = nvc / a[:, self.__MAX_RETURN_IDX, 0]

    def kelly(self, p: float, b: float):
        """the kelly criterion for wagering
        p: probability of win
        b: net payout for a win"""
        return p + (p - 1) / b

    def scale_b(self, b: float, c: float):
        """the kelly criterion expects max loss of 1.0
        so scale the winners and losers accordingly."""
        return b / abs(c)

    def scalp_stoploss(self, a):
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


""" <><><><><><><><><><><><><><><><><><><><> end class <><><><><><><><><><><><><><><><><><><><>"""
"""





















































































"""


def analyze_results(df):
    y = df["risk_return"]  # the y axis is returns
    subjects = df[
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
        x = df[subject]
        try:
            m, b = np.polyfit(x, y, 1)
        except np.linalg.LinAlgError:
            continue
        yp = np.polyval([m, b], x)
        plt.plot(x, yp)
        plt.scatter(x, y)
        plt.title(f"risk_return vs {subject}")
        plt.show()


def calc_pbc(df):
    # p: probability of a win
    # b: payout for a win
    # c: capital loss in the event of a loss (essentially `b` for losses)
    if df[df["risk_return"] >= 0].count()[0] == 0:
        p, b = 0, 0
    else:
        # df is filtered for valid days, so use its entirety
        p = df[df["risk_return"] >= 0].count()[0] / len(df)
        b = df[df["risk_return"] >= 0]["risk_return"].mean()
    if df[df["risk_return"] < 0].count()[0] == 0:
        # No losers found, return 0 since mean of empty sequence will return nan
        return p, b, 0
    # mean payout for losers
    c = df[df["risk_return"] < 0]["risk_return"].mean()
    # median payout for losers
    d = df[df["risk_return"] < 0]["risk_return"].median()
    return p, b, min(c, d)


def clean_rows(df, lookback, hp):
    if hp > 0:
        # ignore the first and last few rows that will be NaN due to shift()s above
        # +1 to include the last valid entry (equivalent to [start_date:-(hp-1)])
        df = df[lookback : -hp + 1]
    else:
        # just ignore the first rows
        df = df[start_date:]
    return df


def filter_iv(df, iv_greater_than, iv_min_threshold, iv_less_than, iv_max_threshold):
    if iv_greater_than:
        df = df[df["iv_open"] >= iv_min_threshold]
    if iv_less_than:
        df = df[df["iv_open"] <= iv_max_threshold]
    return df


def filter_vol(df, vol_greater_than, vol_threshold):
    if vol_greater_than:
        df = df[df["rolling_vol"] > vol_threshold]
    else:
        df = df[df["rolling_vol"] < vol_threshold]
    return df


def filter_sequential_positions(df, hp):
    # * limit 1 open position at a time, simulating real trading
    #   where trades are sequential and non-concurrent
    df["valid"] = False
    df["idx"] = df.index
    next_valid = -1  # initial value so the below condition runs the first time
    for index in df.copy()["idx"]:
        if index > next_valid:
            df.at[index, "valid"] = True
            next_valid = index + hp
    return df[df["valid"] == True]


def get_valid_days(df):
    # must happen after all filtering but before sequential or sample.
    valid_days = len(df)
    # TODO: this is just a thought, not final...
    winners = df[df["risk_return"] > 0].count()[0]
    #     win = winners / valid_days
    return valid_days, winners


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


def get_subset():
    # * lookback and hp are here so that this subset filtering can be performed *before* calculating BSM prices
    #   for performance reasons. By adding hp to the end and subtracting lookback from the beginning, shifts
    #   in later functions can be performed and the beginning and end of the subset aren't NaN due to shifting
    #   out of range of the df. Later trimming will remove rows to get back to the intended subset.
    # NOTE: `None` works as valid indexes and df[None:None] returns the entire df
    lookback = timedelta(days=lookback)
    hp = timedelta(days=hp)
    if start_date is not None:
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date() - lookback
        start_date = df[df["date"] >= start_date].index[0]
    if end_date is not None:
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date() + hp
        # +1 to include the last entry.
        end_date = df[df["date"] <= end_date].index[-1] + 1
    return df[start_date:end_date].copy()


def plot_positions(a, show=100, all=False, pom=True, risk=True):
    """this plots all positions, not just sequential positions."""
    print("plotting...")
    # negate the shifted rows (TODO: pass in hp as a parameter)
    a = a[:-hp, :, :]
    if risk:
        # first, plot the percent of max return
        rows = a.shape[0]
        x = np.arange(a.shape[2])
        if all:
            for i in range(rows):
                y = a[i, 23, :]
                plt.plot(x, y)
        else:
            for i in range(show):
                # sample the df
                row = int(np.random.random() * rows)
                y = a[row, 23, :]
                plt.plot(x, y, alpha=0.25)  # , color='k')
        mean = np.mean(a[:, 23, :])
        median = np.median(a[:, 23, :])
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
                y = a[i, 24, :]
                plt.plot(x, y)
        else:
            for i in range(show):
                # sample the df
                row = int(np.random.random() * rows)
                y = a[row, 24, :]
                plt.plot(x, y, alpha=0.25)  # , color='k')
        mean = np.mean(a[:, 24, :])
        median = np.median(a[:, 24, :])
        print(f"PoM mean: {100*mean:.2f}% | PoM median: {100*median:.2f}%")
        plt.axhline(mean, color="b", linestyle="solid", linewidth=1)
        plt.axhline(median, color="r", linestyle="dashed", linewidth=1)
        plt.title("POM return")
        plt.show()


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
