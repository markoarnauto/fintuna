import fintuna as ft
import pandas as pd
from matplotlib import pyplot as plt
from binance.client import Client as BinanceClient

if __name__ == '__main__':
    # we pick some crypto assets with enough historic data on binance,
    # comparable volatility (which excludes stable coins, doge, etc.)
    # and market cap (which excludes bitcoin, ethereum and tiny coins)
    altcoin_pairs = {'XRPUSDT', 'LINKUSDT', 'DASHUSDT', 'ATOMUSDT', 'ZECUSDT', 'BATUSDT', 'VETUSDT', 'UNIUSDT',
                     'AAVEUSDT', 'ALGOUSDT', 'DOTUSDT', 'ETCUSDT', 'OMGUSDT', 'COMPUSDT',
                     'THETAUSDT', 'SOLUSDT', 'KAVAUSDT', 'MATICUSDT', 'HNTUSDT', 'FILUSDT',
                     'CHZUSDT', 'XEMUSDT', 'ZILUSDT'}

    # we want 12h candle data ranging from 2020-01-01 to 2022-09-01
    interval = '12h'
    since = pd.Timestamp('2020-01-01', tz='utc')
    until = pd.Timestamp('2022-09-01', tz='utc')

    # implement the [data generating process](link)
    client = BinanceClient()


    def get_crypto_features(since, until, assets):
        initiation_periods = 90  # needed for feature transformation
        since_hat = since - pd.Timedelta(interval) * initiation_periods
        altcoin_data = ft.utils.get_crypto_data(client, assets, since_hat, until, interval)

        ## data preprocessing
        # use rolling-zscore of volume in order to be comparable across assets
        volumes = altcoin_data.loc[:, (slice(None), 'volume')]
        zscore_initiation_periods = 60
        volumes_zscored = ft.utils.zscore(volumes, zscore_initiation_periods, interval, zscore_initiation_periods)
        # use returns, as prices are not stationary
        prices = altcoin_data.loc[:, (slice(None), 'close')]
        returns = prices.pct_change(freq=interval).rename({'close': 'return'}, axis=1)

        ## features extraction
        # include past observations
        features = volumes_zscored.join(returns)
        lagg_initiation_periods = initiation_periods - zscore_initiation_periods - 1
        features = ft.utils.lagged_features(features, interval, n_periods=lagg_initiation_periods)

        features = features[since + pd.Timedelta(interval):]  # exclude initiation data
        return features


    # get train data
    data = get_crypto_features(since, until, altcoin_pairs)

    ## exploration
    data_specs = {'returns_column': 'return', 'period': interval}
    crypto_study = ft.FinStudy(ft.model.LongOnly, data, data_specs=data_specs)
    results = crypto_study.explore(n_trials=50, ensemble_size=4)

    # analyze features
    top_features = crypto_study.ensemble.feature_importances().sort_values()
    ax = top_features.plot(kind='barh')
    ax.grid(False)
    plt.show()

    # show backtest
    ft.utils.plot_backtest(results['performance'], results['benchmark'])

    # ok good enough, let's deploy it
    crypto_study.finish()
    logsink = ft.sink.LogSink()  # let's simply log the out, usually
    ft.run(get_crypto_features, crypto_study, logsink)
