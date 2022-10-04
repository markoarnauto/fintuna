# wrapper module to load various data sets
import os

import pandas as pd

# High-dimensional features of BTC-pairs including social-sentiment from cortecs.ai
def get_btcpairs_with_social_sentiment() -> tuple:
    specs = {
        "offset": "21m",
        "returns_column": "return",
        "period": "24h",
        "sampling_freq": "3h"
    }
    data = pd.read_hdf(os.path.dirname(__file__) + '/btcpairs_with_social_sentiment.h5')
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
