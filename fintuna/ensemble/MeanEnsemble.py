import pandas as pd

from fintuna.ensemble.BaseEnsemble import BaseEnsemble


class MeanEnsemble(BaseEnsemble):
    """
    Use mean predictions and mean confidence threshold and pass it to one model.
    Assuming that realized_returns method across model instances behave identical (like a static method).
    """
    def __init__(self, models: list, returns_column, period, min_support=None, **kwargs):
        super().__init__(models, returns_column, period, **kwargs)
        self.min_support = min_support if min_support else len(models) // 2

    def publish(self, data):
        predictions, conf_thrs = self._average(data)
        yield data.index[0], predictions.iloc[0], conf_thrs, self.period

    def _average(self, data):
        index = pd.MultiIndex.from_product([self.asset_ids, list(range(len(self.models)))])
        predictions = pd.DataFrame(columns=index, index=data.index, dtype='float')
        conf_thrs = []
        for i, model in enumerate(self.models):
            predictions_model = model.predict(data)
            for asset_id in predictions_model.columns.get_level_values(0).unique():
                predictions[asset_id, i] = predictions_model[asset_id]
            conf_thrs.append(model.trial.params['conf_thrs'])
        conf_thrs = sum(conf_thrs) / len(conf_thrs)

        enough_support = (~predictions.isna()).sum(axis=1, level=0) >= self.min_support
        predictions = predictions.mean(axis=1, level=0).loc[:, enough_support.all()]
        return predictions, conf_thrs

    def realized_returns(self, data):
        predictions, conf_thrs = self._average(data)
        returns = data.loc[:, (slice(None), self.returns_column)].shift(freq=f'-{self.period}').reindex(predictions.index)
        # assuming realized_returns method to be static
        realized_returns = self.models[0].realized_returns(predictions, conf_thrs, returns, self.period)
        return realized_returns

