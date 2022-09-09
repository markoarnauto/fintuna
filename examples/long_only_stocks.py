import fintuna
from fintuna.model.LongOnly import LongOnly
from fintuna.sink.log_sink import LogSink
import pandas as pd


if __name__ == '__main__':

    altcoins = pd.read_hdf('./fintuna/data/BTC_24h_6h.hdf')  # type: pd.DataFrame

    ## exploration
    # first study
    study = fintuna.FinStudy(LongOnly, altcoins, data_specs={'returns_column': 'return', 'period': '24h', 'sampling_freq': '6h'})
    results = study.explore(n_trials=3, ensemble_size=2)
    print(results)
    # visualize results

    # ...
    # after several studies

    study.finetune(n_trials=3)  # persist all parameters, and the DataSet

    def get_recent_altcoins(since, until, asset_ids, features):
        return
    fintuna.run(study, get_recent_altcoins, LogSink())
