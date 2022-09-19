from fintuna.model.LongOnly import LongOnly


class Pumps(LongOnly):
    """

    """
    def init_classifier(self):
        clf = super().init_classifier()
        clf.set_params(is_unblance=True)
        return clf

    def extract_label(self, data, period):
        anomaly_magnitude = self.trial.suggest_uniform('anomaly_magnitude', .8, .98)
        returns = data.xs('return', axis=1, level=1)

        thrs = returns.stack().quantile(anomaly_magnitude)
        next_returns = returns.shift(freq=f'-{period}')
        return next_returns > thrs
