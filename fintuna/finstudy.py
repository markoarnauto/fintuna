import logging
import multiprocessing
import pickle
from typing import Type, List

import numpy as np
import optuna
import pandas as pd
from dateutil.tz import tzutc
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial

from fintuna.ensemble.IndividualEnsemble import IndividualEnsemble
from fintuna.ensemble.BaseEnsemble import BaseEnsemble
from fintuna.model.ModelBase import ModelBase

n_cores = multiprocessing.cpu_count()
np.random.seed(0)
log = logging.getLogger('train')

class FinStudy:
    """A finstudy corresponds to a financial optimization task.
    A :class:`Model <fintuna.model.ModelBase>` is optimized based on the given `data`.
    It provides interfaces to get the out-of-sample performance and let's you fine-tune a model for production.
    """
    def __init__(self, Model: Type[ModelBase], data: pd.DataFrame, data_specs: dict = {}, split_specs: dict = {}, name: str = 'finstudy'):
        """
        :param Model:
        :param data: Panel data
        :param data_specs:
                Panel data specification.
                    'return_column' str (mandatory): column containing asset returns

                    'period' str (mandatory): aggregation period (e.g. '1h', '2d')

                    'sampling_freq' str (optional): sampling frequency (e.g. '1h', '2d'), defaults to period

                    'offset' str (optional): offset of sampling_freq (e.g. '20m')
        :param split_specs:
                Split specification for train, tune and eval data.
                    'tune_until' date string, int, float: end of tune data, defaults to 80% of the data

                    'train_until' date string, int, float: end of train data, defaults to 60% of tune data
        :param name: Name of study.
        """
        if 'sampling_freq' not in data_specs:
            if data.index.freqstr:
                self.sampling_freq = data.index.freqstr
            elif 'period' in data_specs:
                self.sampling_freq = data_specs['period']
            else:
                raise ValueError('Could not infer period from data. Please specify a period.')
        if 'period' not in data_specs:
            self.period = self.sampling_freq
        if 'period' in data_specs:
            self.period = data_specs['period']
        if 'sampling_freq' in data_specs:
            self.sampling_freq = data_specs['sampling_freq']

        if data.index.freqstr and pd.Timedelta(data.index.freqstr) != pd.Timedelta(self.sampling_freq):
            logging.warning(
                f'data.index.freq {data.index.freq} divers from sampling_freq {self.sampling_freq}. Using sampling_freq')

        if data.index.inferred_type != 'datetime64':
            raise ValueError('data index not supported. Please use datetime64 dataframe.')

        if not data.index.is_monotonic:
            raise ValueError('data index must be monotonic')
        if 'offset' not in data_specs:
            offsets = data.index - data.index.floor(self.sampling_freq)
            if len(offsets.unique()) == 1:
                self.offset = str(offsets[0]) if offsets[0].total_seconds else None
            else:
                raise ValueError('data has irregular offsets')
        else:
            self.offset = data_specs['offset']

        if 'tune_until' in split_specs:
            if type(split_specs['tune_until']) is str:
                self.tune_until = pd.Timestamp(split_specs['tune_until'], tz=tzutc())  # todo support multiple timezones
            if type(split_specs['tune_until']) is int:
                self.tune_until = data.index[split_specs['tune_until']]
            if type(split_specs['tune_until']) is float:
                self.tune_until = data.index[int(len(data) * split_specs['tune_until'])]
        else:
            self.tune_until = data.index[int(len(data) * .8)]

        test_len = len(data[:self.tune_until])
        if 'train_until' in split_specs:
            if type(split_specs['train_until']) is str:
                self.train_until = pd.Timestamp(split_specs['train_until'], tz=tzutc())
            if type(split_specs['train_until']) is int:
                self.train_until = data.index[split_specs['train_until']]
            if type(split_specs['train_until']) is float:
                self.train_until = data.index[int(test_len * split_specs['train_until'])]
        else:
            self.train_until = data.index[int(test_len * .6)]
        if self.train_until > self.tune_until:
            raise ValueError('train_until must be smaller than tune_until')

        self.returns_column = data_specs['returns_column']
        self.name = name
        self.data = data
        self.Model = Model

    def _create_ensemble(self, trials, data):
        models = []
        for trial in trials:
            model = self.Model(trial, **self.model_params)
            model.train(data, self.period, explainer=True)
            models.append(model)
        return self.ensemble_class(models, self.returns_column, self.period, **self.ensemble_params)

    # todo make private
    def _get_best_trials(self, n_trials) -> List[FrozenTrial]:
        trials_df = self.study.trials_dataframe()
        ascending = self.study.direction == StudyDirection.MINIMIZE
        best_trial_ids = trials_df.sort_values('value', ascending=ascending)[:n_trials]['number']
        best_trials = [trial for trial in self.study.get_trials() if trial.number in best_trial_ids]
        return best_trials

    def explore(self, ensemble_class: Type[BaseEnsemble] = IndividualEnsemble, study_params: dict = None, sampling_params: dict = None, model_params: dict = None, ensemble_size=1, ensemble_params: dict = None, conf_thrs_trials=20, n_trials=100) -> dict:
        """
        Train, Tune and explore the performance of a Model. Use Ensemble to combine the best performing models.

        :param ensemble_class: Ensemble to use
        :param study_params: Params of the `optuna study <https://optuna.readthedocs.io/en/v3.0.0/reference/generated/optuna.study.create_study.html#optuna.study.create_study>`_
        :param sampling_params:  Params of `optuna sampler <https://optuna.readthedocs.io/en/v3.0.0/reference/samplers/generated/optuna.samplers.TPESampler.html>`_
        :param model_params: Params of :py:class:`Model <fintuna.model.ModelBase>`
        :param ensemble_size: Number of best models to combine, defaults to 1 (= no ensemble)
        :param ensemble_params: Params of :class:`Model <fintuna.ensemble.BaseEnsemble>`
        :param conf_thrs_trials: Number of trials to simulate different confidence thresholds
        :param n_trials: Number of trials
        :return:
            Tuning results
                'feature_importances': pandas Series of feature importances

                'shap': tuple of (shap values, expected value, observations)

                'performance': out-of-sample returns (nan values indicate no exposure)

                'benchmark_performance': buy-and-hold returns weighted by exposure of 'performance'
        """
        study_params = study_params if study_params else {'direction': 'maximize'}
        if sampling_params:
            if 'n_startup_trials' in sampling_params:
                sampling_params['n_startup_trials'] *= conf_thrs_trials
        else:
            sampling_params = {
                'n_startup_trials': 10 * conf_thrs_trials,
                'seed': 0
                               }
        self.model_params = model_params if model_params else {}
        self.ensemble_params = ensemble_params if ensemble_params else {}
        self.ensemble_class = ensemble_class

        study_params['study_name'] = self.name

        self.ensemble_size = ensemble_size
        self.study = optuna.create_study(**study_params, sampler=optuna.samplers.TPESampler(**sampling_params))

        # train test split
        # exclude one prediction_dur to avoid look ahead bias at feature stacking
        explore_data_train = self.data[:self.train_until - pd.Timedelta(self.period)]
        explore_data_test = self.data[self.train_until:self.tune_until]

        log.info('start tuning')
        for i in range(n_trials):
            log.info(f'iteration {i + 1}/{n_trials}')
            tmp_trial = self.study.ask()
            model = self.Model(tmp_trial, **self.model_params)
            model.train(explore_data_train, self.period)

            predictions = model.predict(explore_data_test)
            returns = explore_data_test.loc[:, (model.asset_ids, [self.returns_column])]
            returns = returns - returns.xs(self.returns_column, axis=1, level=1).stack().mean()  # remove trend
            returns = returns.shift(freq=f'-{self.period}').reindex(predictions.index)
            
            # create multiple trials with varying confidence thresholds
            # note: there might be easier solutions as we are not sampling from conf_thrs (fintuna >= 0.0.4)
            min_prediction, max_prediction = predictions.stack().min(), predictions.stack().max()
            for thrs in np.random.uniform(min_prediction, max_prediction, conf_thrs_trials):
                realized_returns = model.realized_returns(predictions, thrs, returns, self.period)
                # todo make some checks on realized_returns

                performance = model.get_performance(realized_returns, self.period)

                # add trial copy to study
                params_thrs = tmp_trial.params.copy()
                params_thrs['conf_thrs'] = thrs
                dist_thrs = tmp_trial.distributions.copy()
                dist_thrs['conf_thrs'] = optuna.distributions.FloatDistribution(min_prediction, max_prediction)
                trial_thrs = optuna.trial.create_trial(params=params_thrs, distributions=dist_thrs, value=performance)
                self.study.add_trial(trial_thrs)
        # select best trials to form an ensemble
        best_trials = self._get_best_trials(self.ensemble_size)

        # held-out data
        self.real_data_train = self.data[:self.tune_until - pd.Timedelta(self.period)]
        self.real_data_test = self.data[self.tune_until:]
        self.ensemble = self._create_ensemble(best_trials, self.real_data_train)

        out_of_sample_realized_returns = self.ensemble.realized_returns(self.real_data_test)
        out_of_sample_realized_returns.name = 'performance'
        # calculate exposure
        exposed_dur = (~out_of_sample_realized_returns.isna()).sum() * pd.Timedelta(self.period)
        dur = out_of_sample_realized_returns.index[-1] - out_of_sample_realized_returns.index[0]
        exposure = exposed_dur / dur
        # get exposure adjusted benchmark
        benchmark_realized_returns = self.real_data_test.loc[:, (slice(None), self.returns_column)].resample(self.period).last().mean(axis=1) * exposure
        benchmark_realized_returns.name = 'benchmark_performance'

        feature_importances = self.ensemble.feature_importances()
        shap_values = self.ensemble.shap_values(self.real_data_test)
        # todo support asset specific explanations
        return {'feature_importances': feature_importances, 'shap': shap_values, 'performance': out_of_sample_realized_returns, 'benchmark': benchmark_realized_returns}


    def finish(self, n_candidates=None):
        """
        Prepares a model for production.
        Uses best models of tune data as candidates and sub-selects models that also perform well on eval data.
        Refits models on all data.

        :param n_candidates: Number of candidates to consider for selection, defaults to 5 * ensemble_size
        """
        if not n_candidates:
            n_candidates = self.ensemble_size * 5

        # todo store candidate models while exploring
        candidate_trials = self._get_best_trials(n_candidates)
        log.info('selecting best out-of-sample models')
        for trial in candidate_trials:
            model = self.Model(trial, **self.model_params)

            predictions = model.train(self.real_data_train, self.period).predict(self.real_data_test)
            returns_test = self.real_data_test.loc[:, (model.asset_ids, [self.returns_column])]
            returns_test = returns_test.shift(freq=f'-{self.period}').reindex(predictions.index)

            conf_thrs = model.trial.params['conf_thrs']
            real_returns = model.realized_returns(predictions, conf_thrs, returns_test, self.period)
            trial.performance = model.get_performance(real_returns, self.period)
            log.info(f'out-of-sample performance: {trial.performance}')
        final_trials = sorted(candidate_trials, key=lambda x: x.performance, reverse=True)[:self.ensemble_size]
    
        # refit on all data
        self.ensemble = self._create_ensemble(final_trials, self.data)
        self._data_validation = self.data[-10:]  # keep a tiny fraction of data for consistency checks
        self._pub_data_validation = [pd for pd in self.ensemble.publish(self._data_validation[-1:])]  # todo pick periods
        del self.data  # not needed anymore, makes pickeling cheaper
        del self.real_data_train
        del self.real_data_test

