# wrapper module to load various data sets
import pandas as pd
import json


def get_btcpairs_with_social_sentiment() -> tuple:
    specs = json.load(open('./fintuna/data/btcpairs_with_social_sentiment.json'))
    data = pd.read_hdf('./fintuna/data/btcpairs_with_social_sentiment.h5')
    return data, specs