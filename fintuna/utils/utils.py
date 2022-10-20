import pandas as pd
import quantstats as qs
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick


def create_tearsheet(performance, benchmark_performance, file):
    """
    generate sophisticated performance analysis as html file
    """
    returns = performance.resample('1d').sum(min_count=1)
    returns.index = pd.to_datetime(returns.index.date, format="%Y-%m-%d")
    benchmark_returns = benchmark_performance.resample('1d').sum()
    benchmark_returns.index = pd.to_datetime(benchmark_returns.index.date, format="%Y-%m-%d")
    qs.reports.html(returns, benchmark_returns, compounded=False, periods_per_year=365, output=file)

def get_performance_metrics(performance, benchmark_performance, display=False):
    """
    get performance metrics
    """
    returns = performance.resample('1d').sum(min_count=1)
    returns.index = pd.to_datetime(returns.index.date, format="%Y-%m-%d")
    benchmark_returns = benchmark_performance.resample('1d').sum()
    benchmark_returns.index = pd.to_datetime(benchmark_returns.index.date, format="%Y-%m-%d")
    return qs.reports.metrics(returns, benchmark_returns, compounded=False, periods_per_year=365, prepare_returns=False, display=display)


# simple performance plot
def plot_backtest(performance, benchmark_performance):
    """
    plot backtest
    """
    ax = (performance * 100).cumsum().plot(title='Cumulative Returns', label='strategy')
    (benchmark_performance * 100).cumsum().plot(label='buy & hold', style='--')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.legend()
    plt.show()

# get close and volume for multiple assets from binance
def get_crypto_data(client, symbols, since, until, interval):
    """
    get klines from binance for multiple symbols
    """
    expected_ts = pd.date_range(since, until, freq=interval, closed='right')
    asset_feature_columns = pd.MultiIndex.from_product([symbols, ['volume', 'return']])
    data = pd.DataFrame(index=expected_ts, columns=asset_feature_columns)
    for symbol in symbols:
        print(f'loading {symbol}')
        klines = client.get_historical_klines(symbol, interval, str(since), str(until))
        klines = pd.DataFrame(klines)
        klines.index = pd.to_datetime(klines[6], unit='ms', utc=True).round(interval)  # .asfreq(interval)
        data.loc[:, (symbol, 'volume')] = klines[5].astype('float')
        data.loc[:, (symbol, 'close')] = klines[4].astype('float')
    return data

# transform time series data to include lagged data (and percentage change)
def lagged_features(data: pd.DataFrame, period, n_periods) -> pd.DataFrame:
    """
    add lagged features to a DataFrame
    """
    assert n_periods > 10  # reasonable number of past periods is essential
    feature_names = data.fin.feature_names
    period_dt = pd.Timedelta(period)

    # extract changes and shifted values
    change = data.pct_change(freq=period_dt)\
        .rename(columns={name: f'{name}__change' for name in feature_names}, level=1)
    shift_1 = data.shift(freq=period_dt)\
        .rename(columns={name: f'{name}__shift1' for name in feature_names}, level=1)
    shift_2 = data.shift(freq=2 * period_dt)\
        .rename(columns={name: f'{name}__shift2' for name in feature_names}, level=1)

    # two non-overlapping periods for taking the mean
    midterm_periods = n_periods // 3  # one third
    longterm_periods = n_periods // 3 * 2  # two thirds
    rollingmean_midterm = data.rolling(midterm_periods * period_dt).mean().shift(freq=3 * period_dt)\
        .rename(columns={name: f'{name}__rollingmean_midterm' for name in feature_names}, level=1)
    rollingmean_longterm = data.rolling(longterm_periods * period_dt).mean().shift(freq=(midterm_periods + 3) * period_dt)\
        .rename(columns={name: f'{name}__rollingmean_longterm' for name in feature_names}, level=1)

    data_extended = pd.concat([change, shift_1, shift_2, rollingmean_midterm, rollingmean_longterm], axis=1)
    data_extended = data_extended.loc[data.index, :]
    return data_extended

def zscore(x, n_periods, period, min_periods=None):
    """
    rolling zscore
    """
    period_dt = pd.Timedelta(period)
    r = x.rolling(window=n_periods * period_dt, min_periods=min_periods)
    s = r.std(ddof=0).shift(freq=period_dt).reindex(x.index)
    m = r.mean().shift(freq=period_dt).reindex(x.index)
    z = (x-m)/s
    return z