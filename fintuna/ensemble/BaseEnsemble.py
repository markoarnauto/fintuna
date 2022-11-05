from abc import ABC, abstractmethod
import pandas as pd

class BaseEnsemble(ABC):
    """
    An interface for combining multiple models.
    """
    def __init__(self, models: list, returns_column, period, **kwargs):
        self.models = models
        self.returns_column = returns_column
        self.period = period
        asset_ids = set()
        features = set()
        for model in models:
            asset_ids.update(model.asset_ids)
            features.update(model.selected_features)
        # assure determinism
        self.asset_ids = list(asset_ids)
        self.asset_ids.sort()
        self.features = list(features)
        self.features.sort()

    @abstractmethod
    def publish(self, data):
        """
        Specify which data is passed to sink

        :param data: Panel data
        """
        pass

    @abstractmethod
    def realized_returns(self, data):
        """
        Specify how models are combined to generate returns

        :param data: Panel data
        """
        pass

    # average feature importance
    def feature_importances(self):
        """
        Combine features importances

        :return: feature importance
        """
        feature_importances = []
        for model in self.models:
            feature_importances.append(model.get_feature_importances())
        return pd.DataFrame(feature_importances).mean()

    # average shap values
    def shap_values(self, data):
        """
        Combine shap values

        :param data: Panel data
        :return: tuple (shap values, expected value, observations)
        """
        shap_values = []
        for model in self.models:  # average shap values
            shap_values_model, expected_value, observations = model.explain(data)
            shap_values.append(shap_values_model)
        shap_values = pd.concat(shap_values, axis=1, keys=range(len(self.models)))
        shap_values = shap_values.mean(axis=1, level=1)
        supported_observations = ~shap_values.isna().all(axis=1)
        return shap_values[supported_observations], expected_value, observations[supported_observations]

    def __len__(self):
        return len(self.models)
