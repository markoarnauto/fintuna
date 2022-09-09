import numpy as np

from fintuna.model.ModelBase import ModelBase, EXECUTION_COSTS


class LongOnly(ModelBase):
    def extract_label(self, data, period):
        returns = data.xs('return', axis=1, level=1)
        next_returns = returns.shift(freq=f'-{period}')
        return next_returns > 0.

    def realized_returns(self, predictions, conf_threshold, returns, period):
        trades = predictions.rank(axis=1, ascending=False, method='first') <= 1
        trades[predictions < conf_threshold] = False

        realized_returns = returns.xs('return', axis=1, level=1)
        realized_returns = realized_returns.mask(~trades, np.nan)
        realized_returns -= EXECUTION_COSTS
        realized_returns = realized_returns.sum(axis=1, min_count=1)

        trades = trades.any(axis=1)
        for trades_p in trades.rolling(period):
            if trades_p.sum() > 1:  # more than one trade within a period is impossible
                trades_p[-1] = False
        realized_returns = realized_returns.mask(~trades, np.nan)
        return realized_returns.resample(period).sum(min_count=1)  # keep nan for empty periods
