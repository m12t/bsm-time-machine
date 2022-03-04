# bsm-time-machine

This is a rough and ready options backtesting model that estimates historical options pricing by plugging in realized implied volatility to the [Black-Scholes Model](https://en.wikipedia.org/wiki/Black%E2%80%93Scholes_model) to roughly price options. It is for educational purposes and is useful for those of us who don't have $$$ to spend for historical options data.


## How `backtester.ipynb` Works
1. The notebook will read a pickled pandas dataframe containing historical data in a specific format containing OHLC (open, high, low, close) price *and* implied volatility data, as well as a few other calculated columns spit out from `utils/get_data.ipynb` (more on this below)
2. Based on the input parameters, the strikes will be calculated for the position (either a fixed coefficient such as `'1.05 x'` for 5% OTM at open, or `'1.2 SD'` for a position that is 1.2 standard deviations OTM for the given holding period (in days) (based of implied volatility, not realized volatility).
3. Then, the options are priced according to the Black-Scholes model using a 3 dimensional numpy array that is of size `hp` (holding period) in the 3rd dimension. The third dimension is used to calculate the price at the end of every day of the holding period so the position can be scalped for a profit of closed at a stop-loss, and for plotting the positions across the hp
4. Once BSM prices have been calculated, returns are added for each period in the HP within the same 3D array. If the position has a defined risk (ie. is hedged), the calculations are straightforward. The risk-return, RR, is the net gain in value over the max risk. The percent-of-max possible return, PoM, is the position's value bounded by 0 and 100% where 100% is the max return possible. For undefined (ie. naked) positions, a fudge factor is calculated akin to risk-based margining that uses the implied volatility to estimate standard deviation movements over the holding period. a 2 standard deviation movement is used to estimate the max_loss and is bounded by a 2% lower limit of risk
5. After prices and returns have been calculated on the entire df such that shifts are accurate and sequential, filtering the dataframe by the specific parameters can be performed. Filtering can be performed on a rolling mean of `lookback` days on realized volatility and/or implied volatility. For example, you can filter to open a position such that: 30% <= IV < 50% and the mean realized open-close (annualized) volatility for the previous 5 trading days is < 20%
7. At this stage, the dataframe is populated with return data and is filtered for days that meet the specified parameters. Analysis is then performed on the dataframe for expected return, wager (using the kelly criterion), long run return, the probability of a positive trade, the mean gain for a winning trade, and the mean loss for a losing trade. Optionally, columns of the dataframe can be analyzed in a scatterplot between risk_return and another column such as `50_day_return` (returns from t=-50 to t=0 trading days). A regression line is fitted and can be used to tweak the model


### Positions That Can be Formulated With the Backtester
1. long call or put
1. naked short call or put
1. call or put credit spread
1. call or put debit spreads
1. long iron condor
1. short iron condor
1. long straddle or strangle
1. naked short straddle or strangle


## how `get_data.ipynb` works
It's worth noting that any data source can be used to supply the columns required by backtester. However, this repository has a file which is explicitly built to request these data and build dataframes with the specific columns, saving the pickle for use by backtester. It is capable of requesting data from scratch as well as updating dataframes to include the latest data
1. Data are requested from IBKR historical data (this requires an account with IBKR) the notebook also relies on the wonderful library to interact with IBKR's API, called [IB-insync](https://github.com/erdewit/ib_insync)
1. If requesting data from scratch, set the flag `is_update` to `False` and specify the parameters you wish such as the underlying's symbol, exchange, and security type, whether to include IV data, the duration of data to grab (eg. 5 years back from today, or `'MAX'` for all available data), and data granulatiry (1 day bars is the default, any size less than this requires more complex logic to avoid exceeding rate limits). Based on these parameters, a file name will be automatically generated that is decodable for updates such that:
1. To update a pickle, simply enter the filename under the variable `file_name` and set the flag, `is_update` to `True`. From there, simply run all the cells. The program will decode the file name, request data from the last known date in the dataframe until present, append that to the existing df, drop duplicates, recalculate columns (shifts must be recalculated else `nan`s will show up on every update).
1. Lastly, the program will check for gaps in the data > 5 days, optionally trimming the dataframe to include only data after the last gap


## What I have *tried* to do when designing this program
1. Be conscious of biases that impact these types of models. Primarily, look-ahead bias and overfitment. A conscious effors has been made to ensure no `.shift()`s introduce future data into current decisions. To address the latter, there are flags `training_df` and `full_df` for testing and validation, respectively.


## What I Have Learned in the Process
Options markets are efficient. I have yet to find a combination of parameters that provides risk-adjusted returns in excess of a buy-and-hold strategy in the long run.


## Contributions
Please feel free to contribute in any capacity! Some ideas for simple contributions:
1. Adding new filtering parameters
1. Bug fixes


#### Disclosures
1. I am not a data scientist or a quant. There are bound to be fundamental errors in my logic, I find new ones all the time. If you notice anything, please let me know! I would greatly appreciate the feedback!
1. This program is not mean to be used as justification for a trading strategy. Do NOT base an investing strategy around the outputs of this program. Recall the fundamental axion of finance: `Past returns are NOT indicative of future performance`!
