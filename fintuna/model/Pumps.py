from fintuna.model.LongOnly import LongOnly


class Pumps(LongOnly):
    """
    Buy assets which are predicted to have abnormally high returns.
    """
    def _init_classifier(self):
        clf = super()._init_classifier()
        clf.set_params(is_unblance=True)
        return clf

    def extract_label(self, data, period):
        """
        Predict if assets yield abnormal returns. What is qualified as `abnormal` is tuned (as hyper-parameter).
        """
        anomaly_magnitude = self.trial.suggest_float('anomaly_magnitude', .8, .98)
        returns = data.xs('return', axis=1, level=1)

        thrs = returns.stack().quantile(anomaly_magnitude)
        next_returns = returns.shift(freq=f'-{period}')
        return next_returns > thrs
