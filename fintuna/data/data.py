# wrapper module to load various data sets
import os

import pandas as pd
import numpy as np
from fintuna.utils import lagged_features


# High-dimensional features of BTC-pairs including social-sentiment from cortecs.ai
def get_btcpairs_with_social_sentiment() -> tuple:
    specs = {
        "offset": "21m",
        "returns_column": "return",
        "period": "24h",
        "sampling_freq": "3h"
    }
    data = pd.read_hdf(os.path.dirname(__file__) + '/btcpairs_with_social_sentiment.h5')  # type: pd.DataFrame
    ## preprocessing
    # lagging those features makes less sense
    features_to_exclude = ['trading_volume__max', 'spread', 'rsi__6h', 'trading_volume__ema_6h',
     'longterm_return', 'overall_longterm_return', 'longterm_volatility',
     'longterm_skew', 'overall_twitter_activity__ema_6h', 'high', 'low']
    data_transf = data.drop(features_to_exclude, axis=1, level=1)
    data_not_transf = data.loc[:, (slice(None), features_to_exclude)]
    transformed_data = lagged_features(data_transf, specs['period'], 20)
    data = pd.concat([data_not_transf, transformed_data], axis=1)

    ## do some post processing
    # clip initiation data
    data = data[pd.Timestamp('2020-01-01', tz='utc') + pd.Timedelta(specs['period']):]
    # treat inf as nan
    data = data.replace([np.inf, -np.inf], np.nan)
    return data, specs

# simple altcoins features: lagged returns and zscored trading volume
# see end2end_crypto_predictions.py or "End2End Crypto Prediction" in the docs
def get_crypto_features() -> tuple:
    specs = {
        "returns_column": "return",
        "period": "12h"
    }
    data = pd.read_hdf(os.path.dirname(__file__) + '/crypto_features.h5')
    return data, specs
