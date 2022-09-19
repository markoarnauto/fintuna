import logging
from abc import ABC, abstractmethod

import lightgbm as lgb
import optuna
import pandas as pd
import sklearn


tune_log = logging.getLogger('tune_log')
live_log = logging.getLogger('live')
pd.options.mode.chained_assignment = None

slippage = .002
fees = .002
EXECUTION_COSTS = fees + slippage
class ModelBase(ABC):
    """
    as example refer to Pumps
    """
    def __init__(self, trial: optuna.Trial, n_jobs=-1, group_by_asset=None):
        self.trial = trial
        self.clf = None
        self.explainer = None
        self.selected_features = None
        self.n_jobs = n_jobs
        if group_by_asset is not None:
            self.group_by_asset = group_by_asset
        else:
            self.group_by_asset = self.trial.suggest_categorical('group_by_asset', [True, False])

    def select_assets(self, data, period) -> list:
        return data.fin.asset_names

    def init_classifier(self) -> sklearn.base.ClassifierMixin:
        clf = lgb.LGBMClassifier(**{'verbosity': -1,
                                    'objective': 'binary',
                                    'metric': 'binary_logloss',
                                    'learning_rate': .05,
                                    'max_bin': 127,
                                    'random_state': 1
                                    })
        clf.set_params(**{
            'n_estimators': self.trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': self.trial.suggest_int('max_depth', 2, 31),
            'subsample': self.trial.suggest_uniform('subsample', .4, 1.),
            'lambda_l1': self.trial.suggest_loguniform('lambda_l1', .01, 10.),
            'lambda_l2': self.trial.suggest_loguniform('lambda_l2', .01, 10.),
            'num_leaves': self.trial.suggest_int('num_leaves', 2, 63),
            'feature_fraction': self.trial.suggest_uniform('feature_fraction', .5, 1.)
        })
        return clf

    def select_features(self, X_train, y_train, X_estimation, y_estimation, period):
        return X_train.columns

    # build an explainer if selected model doesn't already support shap values
    def create_explainer(self, model, X, y):
        pass

    def shap_values(self, X):
        return self.clf.predict_proba(X, pred_contrib=True)[:, :-1]

    def get_feature_importances(self) -> pd.Series:
        return pd.Series(self.clf.booster_.feature_importance(importance_type='gain'), index=self.selected_features)

    def _fit(self, X, y):
        if not self.group_by_asset:
            self.selected_features = self.selected_features.drop('asset_id')

        return self.clf.fit(X[self.selected_features], y)

    def _predict_proba(self, X):
        X = X[self.selected_features]
        return pd.DataFrame(self.clf.predict_proba(X)[:, 1], index=X.index)

    def train(self, data, period, explainer=False):
        # self.explainer = None  # if train is called multiple times -> reset explainer
        # assert self.trial['prediction_dur'] == data.attrs['prediction_dur']

        self.asset_ids = self.select_assets(data, period)
        if not len(self.asset_ids):  # todo make some more checks
            raise ValueError('No assets selected')
        data = data[self.asset_ids]

        labels = self.extract_label(data, period)
        if set(labels.columns) != set(data.fin.asset_names):
            raise ValueError('columns of labels do not match assets')

        ## prepare data for ml model
        # merge labels into data
        for asset_id in data.fin.asset_names:
            data[asset_id, 'label'] = labels[asset_id]
        # stack features
        X = data.fin.stack_asset_data()
        no_label = X['label'].isna()
        X, y = X[~no_label].drop('label', axis=1), X[~no_label]['label'].astype(bool)
        # keep categorical features
        categorical_features = data.select_dtypes('category').fin.feature_names.to_list() + ['asset_id']
        X[categorical_features] = X[categorical_features].astype('category')

        split_at = X.index[int(len(X) * .75)]  # using 25% as evaluation data
        # avoid look ahead bias
        X_train, y_train = X[:split_at - pd.Timedelta(period)], y[:split_at - pd.Timedelta(period)]
        X_test, y_test = X[split_at:], y[split_at:]
        self.selected_features = self.select_features(X_train, y_train, X_test, y_test, period)
        selected_features_type = type(self.selected_features)
        if len(self.selected_features) < 1 or len(self.selected_features) > len(data.fin.feature_names) \
                or set(self.selected_features) - set(data.fin.feature_names):
            ValueError('Selected features are invalid')
        if selected_features_type is not pd.Series and selected_features_type is not pd.Index:
            ValueError('Selected features have invalid type')

        self.clf = self.init_classifier()
        self._fit(X[self.selected_features], y)  # refit

        if explainer:
            self.explainer = self.create_explainer(self.clf, X[self.selected_features], y)
        return self  # function chaining

    def predict(self, data):

        data = data[self.asset_ids]
        X = data.fin.stack_asset_data()
        # keep categorical features
        categorical_features = data.select_dtypes('category').fin.feature_names.to_list() + ['asset_id']
        X[categorical_features] = X[categorical_features].astype('category')

        X_asset_ids = X['asset_id']
        predictions = self._predict_proba(X)

        # unstack predictions
        predictions['asset_id'] = X_asset_ids.astype(int)
        predictions = predictions.pivot(columns=['asset_id']).droplevel(0, axis=1)
        return predictions

    def explain(self, data):
        data = data[self.asset_ids]
        X = data.fin.stack_asset_data()
        categorical_features = data.select_dtypes('category').fin.feature_names.to_list() + ['asset_id']
        X[categorical_features] = X[categorical_features].astype('category')
        X = X[self.selected_features]

        # subsampling big data sets
        max_size = 1000
        if len(X) > max_size:
            X = X.sample(max_size, random_state=0).sort_index()
        return pd.DataFrame(self.shap_values(X), columns=self.selected_features, index=X.index)

    @abstractmethod
    def extract_label(self, data, period) -> pd.DataFrame:
        """

        :param data:
        :param period:
        :return:
        """
        pass

    @abstractmethod
    def realized_returns(self, predictions, conf_threshold, returns, period) -> pd.Series:
        """

        :param predictions:
        :param conf_threshold:
        :param returns:
        :param period:
        :return:
        """
        pass

    def get_performance(self, realized_returns) -> float:
        return realized_returns.sum()

