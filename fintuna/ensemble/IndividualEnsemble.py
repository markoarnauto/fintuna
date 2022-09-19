import pandas as pd

from fintuna.ensemble.BaseEnsemble import BaseEnsemble


class IndividualEnsemble(BaseEnsemble):
    """

    """
    def __init__(self, models: list, returns_column, period, **kwargs):
        super().__init__(models, returns_column, period, **kwargs)
        self.weight = 1 / len(models)  # equally weighted. todo support weighting

    def publish(self, data):
        for i, model in enumerate(self.models):
            yield i, data.index[0], model.predict(data).iloc[0], model.trial.params['conf_thrs'], self.period, self.weight

    def realized_returns(self, data):
        ensemble_realized_returns = []
        for model in self.models:
            predictions = model.predict(data)
            conf_thrs = model.trial.params['conf_thrs']
            returns = data.loc[:, (slice(None), self.returns_column)].shift(freq=f'-{self.period}').reindex(predictions.index)
            realized_returns = model.realized_returns(predictions, conf_thrs, returns, self.period)
            ensemble_realized_returns.append(realized_returns)
        ensemble_realized_returns = pd.DataFrame(ensemble_realized_returns)
        return (ensemble_realized_returns * self.weight).sum()  # equally weighted

