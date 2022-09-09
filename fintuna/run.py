import logging
from logging import handlers

import numpy as np
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler

from fintuna import FinStudy

logging.basicConfig(level=logging.DEBUG, filemode='w')  # this must be lowest log level
logging.getLogger('matplotlib.font_manager').disabled = True
log = logging.getLogger('live')
fh_info = handlers.RotatingFileHandler('./logs/ensemble.info')
fh_info.setLevel(logging.INFO)
fh_info.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(fh_info)


def run(data_func, finstudy: FinStudy, sink):
    # check data_func
    since_val, until_val = finstudy._data_validation.index[0], finstudy._data_validation.index[-1]
    data_validation_hat = data_func(since_val, until_val, finstudy._data_validation.fin.asset_names.to_list())
    np.testing.assert_allclose(data_validation_hat.values, finstudy._data_validation.values)

    # test finstudy
    for i, pub_data in enumerate(finstudy.ensemble.publish(finstudy._data_validation[-1:])):
        for j, value in enumerate(pub_data):
            if type(value) is pd.Series:
                np.testing.assert_allclose(value.values, finstudy._pub_data_validation[i][j].values)
            else:
                assert value == finstudy._pub_data_validation[i][j]


    scheduler = BlockingScheduler(timezone='utc')
    trigger_dur_dt = pd.Timedelta(finstudy.sampling_freq)

    now = pd.Timestamp.utcnow().floor(trigger_dur_dt) + pd.Timedelta(finstudy.offset)
    next_interval = now + trigger_dur_dt

    class LookaheadCheck:
        def __init__(self):
            self.prev_data = pd.DataFrame()

        def __call__(self, *args, **kwargs):
            now = pd.Timestamp.utcnow().floor(finstudy.period)
            if finstudy.offset:
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


    scheduler.add_job(LookaheadCheck(), 'interval', seconds=int(trigger_dur_dt.total_seconds()),
                  start_date=next_interval.to_pydatetime())
    scheduler.start()
