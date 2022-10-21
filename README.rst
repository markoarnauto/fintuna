.. image:: https://github.com/markoarnauto/fintuna/blob/main/docs/source/images/fintuna-logo.png?raw=true
    :alt: Fintuna Logo
    :width: 700

Fintuna: Parameter optimization for finance
============================================

**Fintuna** is a framework that uses machine learning for asset management. It enables fast prototyping for multi-asset applications such as stock-picking.

Features:
 * model training
 * hyper-parameters tuning
 * walk-forward backtesting
 * strategy evaluation

It is a lightweight framework that combines `LightGBM <https://lightgbm.readthedocs.io>`_, `Optuna <https://optuna.readthedocs.io>`_, `Quantstats <https://github.com/ranaroussi/quantstats>`_ and `Shap <https://shap.readthedocs.io>`_ to develop ML-based stock-picking strategies.

..
    Check out the section for further information, including
    how to the project.


Multi Asset
------------

Looking at multiple assets is supposed to reveal more alpha-opportunities than looking at a single one. Also,
the more assets the more data which is beneficial for machine learning tasks.
Therefore, *Fintuna* is designed for multi-asset applications. The data structure
is a `Pandas Multiindex Dataframe <https://pandas.pydata.org/docs/user_guide/advanced.html#multiindex-advanced-indexing>`_ where the index is time, the first column-level is the asset and the second is the feature (= panel or longitudinal data).
Internally features are stacked and a model is trained to learn cross-asset patterns.

===== ========  ========  ========= ========  ========  ========  ========= ========
#       Asset 1               Asset 2             Asset 3             Asset 4
----- ------------------  ------------------  ------------------  ------------------
#     feature1  feature2  feature1  feature2  feature1  feature2  feature1  feature2
===== ========  ========  ========= ========  ========  ========  ========= ========
t0    float     category   float    category  float     category  float     NaN
t1    float     category   float    category  float     category  float     NaN
===== ========  ========  ========= ========  ========  ========  ========= ========

Strategy Agnostic
------------------

Fintuna is not tied to one specific trading strategy. Strategies are implemented as `fintuna.model.ModelBase`.
It defines the classification task (= `extract_label`)
as well as a a classification-to-returns mapping (= `realized_returns`).
A simple example is to predict the directional change and buy the asset with the
most confident prediction (see `fintuna.model.LongOnly`).

Backtesting
------------

Fintuna uses walk-forward backtesting.

* Train data is used to train the classifier.
* Tune data is used for hyper-parameter optimization.
* Eval data is used for backtesting

Executing the `fintuna.Finstudy.explore` method multiple times on same data introduces the risk of overfitting.
**Use feature importance and shap values, rather than merely looking at trading performance.**

.. image:: https://github.com/markoarnauto/fintuna/blob/main/docs/source/images/backtesting.png?raw=true
    :alt: Walk-Forward Backtesting


Calling `fintuna.Finstudy.finish` prepares the model for deployment. It sub-selects models that also perform well on evaluation data.
and refits them on all data.

Data First
------------

A good trading strategy demands good and possibly unique data.
Fintuna does **NOT** help you in finding the right data. But consider the following guidelines:

* Have at least a few hundreds of observations.
* Use multiple assets.
* Use assets with similar characteristics (e.g. cryptos, tech-stocks, etc.).
* Make sure features across assets have similar properties (otherwise use zscore).
* Use lagged features to boost performance.


Usage
=======

Install fintuna via pip.

.. code:: bash

    pip install fintuna

Run the most basic example below. For detailed guidance look at examples at `docs <https://markoarnauto.github.io/fintuna/examples.html>`_ or at `docs/source/examples`.

.. code:: python

    import fintuna as ft

    # get data
    data, specs = ft.data.get_crypto_features()

    # explore
    crypto_study = ft.FinStudy(ft.model.LongOnly, data, data_specs=specs)
    results = crypto_study.explore(n_trials=50, ensemble_size=3)

    # analyze
    ft.utils.plot_backtest(results['performance'], results['benchmark'])

TODO
-----
* Binance Trading Sink
* MajorityVoteEnsemble
* Backtest plots with shap values
