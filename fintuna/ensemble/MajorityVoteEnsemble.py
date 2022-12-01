import pandas as pd

from fintuna.ensemble.BaseEnsemble import BaseEnsemble
import numpy as np

from fintuna.model.ModelBase import EXECUTION_COSTS


class MajorityVoteEnsemble(BaseEnsemble):
    def __init__(self, models: list, returns_column, period, min_support=None, max_exposure=1., **kwargs):
        super().__init__(models, returns_column, period, **kwargs)
        self.min_support = min_support
        self.max_exposure = max_exposure

    def publish(self, data):

        predictions, conf_thrs = self._vote(data)
        yield data.index[0], predictions.iloc[0], conf_thrs, self.period

    def _vote(self, data):
        index = pd.MultiIndex.from_product([self.asset_ids, list(range(len(self.models)))])
        votes = pd.DataFrame(columns=index, index=data.index, dtype='float')
        for i, model in enumerate(self.models):
            predictions = model.predict(data)
            conf_thrs = model.trial.params['conf_thrs']
            votes_model = (predictions.rank(axis=1, ascending=False, method='first') <= 1) & (predictions >= conf_thrs)
            for asset_id in predictions.columns.get_level_values(0).unique():
                votes[asset_id, i] = votes_model[asset_id]
        return votes.sum(axis=1, level=0)

    def realized_returns(self, data):
        votes = self._vote(data)

        if not self.min_support:
            exp_time_in_market = .25
            self.min_support = votes.max(axis=1).quantile(1 - exp_time_in_market)

        trades = (votes.rank(axis=1, ascending=False, method='first') <= 1) & (votes >= self.min_support)
        positions = (votes / len(self.models)).clip(upper=self.max_exposure)
        returns = data.loc[:, (slice(None), self.returns_column)].shift(freq=f'-{self.period}').reindex(votes.index)

        realized_returns = returns.xs('return', axis=1, level=1)
        realized_returns = realized_returns.mask(~trades, np.nan)
        realized_returns = realized_returns * positions
        realized_returns -= EXECUTION_COSTS * positions
        realized_returns = realized_returns.sum(axis=1, min_count=1)

        trades = trades.any(axis=1)
        for trades_p in trades.rolling(self.period):
            if trades_p.sum() > 1:  # more than one trade within a period is impossible
                trades_p[-1] = False
        realized_returns = realized_returns.mask(~trades, np.nan)
        return realized_returns.resample(self.period).sum(min_count=1)  # keep nan for empty periods