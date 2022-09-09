from fintuna.finstudy import FinStudy
from fintuna.run import run
import pandas as pd

@pd.api.extensions.register_dataframe_accessor("fin")
class FinAccessor:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if not obj.columns.is_unique:
            raise AttributeError("Wrong Dataframe format. Assets must be unique.")
        if obj.columns.nlevels != 2:
            raise AttributeError("Wrong Dataframe format. First level must be assets, second level must be features.")

    @property
    def asset_names(self):
        return self._obj.columns.get_level_values(0).unique()

    @property
    def feature_names(self):
        return self._obj.columns.get_level_values(1).unique()

    def stack_asset_data(self):
        data_stacked = pd.DataFrame()
        for asset_id in self._obj.fin.asset_names:
            data_asset = self._obj[asset_id].copy()
            data_asset['asset_id'] = asset_id
            data_stacked = data_stacked.append(data_asset)
        return data_stacked.sort_index()
