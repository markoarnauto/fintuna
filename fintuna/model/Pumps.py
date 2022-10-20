from fintuna.model.LongOnly import LongOnly
import numpy as np

class Pumps(LongOnly):
    """
    A long-only strategy focussing on abnormally high returns
    """
    def _init_classifier(self):
        clf = super()._init_classifier()
        clf.set_params(is_unbalance=True)  # use unbalanced sampling
        return clf

    def extract_label(self, data, period):
        """
        Predict if asset returns will be in the top quantile. The quantile is a hyperparameter ranging from 20% to 2%.
        """
        anomaly_magnitude = self.trial.suggest_float('anomaly_magnitude', .8, .98)
        returns = data.xs('return', axis=1, level=1)

        thrs = returns.stack().quantile(anomaly_magnitude)
        next_returns = returns.shift(freq=f'-{period}')
        labels = next_returns > thrs
        labels[next_returns.isna()] = np.nan
        return labels
