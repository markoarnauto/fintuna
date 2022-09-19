from abc import ABC, abstractmethod
import pandas as pd

class BaseEnsemble(ABC):
    """
    explain the concept
    """
    def __init__(self, models: list, returns_column, period, **kwargs):
        self.models = models
        self.returns_column = returns_column
        self.period = period
        self.asset_ids = set()
        self.features = set()
        for model in models:
            self.asset_ids.update(model.asset_ids)
            self.features.update(model.selected_features)

    # specify which data is passed to sink
    @abstractmethod
    def publish(self, data):
        pass

    @abstractmethod
    def realized_returns(self, data):
        pass

    # average feature importance
    def feature_importances(self):
        feature_importances = []
        for model in self.models:
            feature_importances.append(model.get_feature_importances())
        return pd.DataFrame(feature_importances).mean()

    # average shap values
    def shap_values(self, data):
        shap_values = []
        for model in self.models:
            shap_values.append(model.explain(data))
        shap_values = pd.concat(shap_values, axis=1, keys=range(len(self.models)))
        return shap_values.mean(axis=1, level=1)

    def __len__(self):
        return len(self.models)
