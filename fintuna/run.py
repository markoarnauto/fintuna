import logging

import numpy as np
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler

from fintuna import FinStudy

logging.basicConfig(level = logging.INFO)
log = logging.getLogger('live')

def run(data_func, finstudy: FinStudy, sink):
    """
    This function calls a finstudy according to its sampling_freq periodically. It forwards the results to the sink.
    It also checks whether the data generating process (data_func) is valid (e.g. look-ahead-bias detection).

    :param data_func:
        A function which expects parameters (since, until, assets) and returns the panel data (see *Crypto End2End Prediction*)
    :param finstudy: The finstudy to deploy
    :param sink: A callable object that forwards or processes the output, e.g. a trading-engine (see :py:class:`fintuna.sink.LogSink`)
    """

    # check data_func
    since_val, until_val = finstudy._data_validation.index[0], finstudy._data_validation.index[-1]
    since_val -= pd.Timedelta(finstudy.period)  # data_func is right bound -> first observation at since + period
    data_validation_hat = data_func(since_val, until_val, finstudy._data_validation.fin.asset_names.to_list())
    pd.testing.assert_frame_equal(data_validation_hat, finstudy._data_validation)

    # test finstudy
    for i, pub_data in enumerate(finstudy.ensemble.publish(finstudy._data_validation[-1:])):
        for j, value in enumerate(pub_data):
            if type(value) is pd.Series:
                    np.testing.assert_allclose(value.values, finstudy._pub_data_validation[i][j].values)
            else:
                assert value == finstudy._pub_data_validation[i][j]
    log.info('checks complete!')

    scheduler = BlockingScheduler(timezone='utc')
    trigger_dur_dt = pd.Timedelta(finstudy.sampling_freq)

    now = pd.Timestamp.utcnow().floor(trigger_dur_dt) + pd.Timedelta(finstudy.offset)
    next_interval = now + trigger_dur_dt

    class LookaheadCheck:
        def __init__(self):
            self.prev_data = pd.DataFrame()

        def __call__(self, *args, **kwargs):
            now = pd.Timestamp.utcnow().floor(finstudy.sampling_freq)
            now += pd.Timedelta(finstudy.offset)
            # get data. also fetch the previous data point for lookahead checks
            since = now - pd.Timedelta(finstudy.period) - pd.Timedelta(finstudy.sampling_freq)
            data = data_func(since, now, finstudy.ensemble.asset_ids)
            # todo do some checks on the data

            cnt_data = data[-1:]
            prev_data_hat = data.iloc[0]  # watch for lookahead bias in the data
            try:
                _prev_data = self.prev_data.loc[prev_data_hat.name]
                identical = np.isclose(prev_data_hat, _prev_data, rtol=.01, equal_nan=True)
                if identical.sum() != len(_prev_data):
                    incorrect = pd.Series(~identical, index=_prev_data.index)
                    log.warning(f'Lookahead bias detected:\n {pd.concat([prev_data_hat.loc[incorrect], _prev_data.loc[incorrect]], axis=1)}')

                self.prev_data = self.prev_data.drop(_prev_data.name)
                self.prev_data = self.prev_data.append(data.iloc[-1])
            except KeyError as e:  # previous data is not available at startup
                log.debug('collecting data for lookahead checks..')
                self.prev_data = self.prev_data.append(data.iloc[-1])
                pass

            for args in finstudy.ensemble.publish(cnt_data):
                sink(*args)

    lookahead_check = LookaheadCheck()
    scheduler.add_job(lookahead_check, 'interval', seconds=int(trigger_dur_dt.total_seconds()),
                  start_date=next_interval.to_pydatetime())
    log.info(f'Scheduler started. {type(sink).__name__} will be triggered each {finstudy.sampling_freq}')
    scheduler.start()
