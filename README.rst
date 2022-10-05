.. image:: https://github.com/markoarnauto/fintuna/blob/main/docs/source/images/fintuna-logo.png?raw=true
    :alt: Fintuna Logo
    :width: 700

.. include:: docs/source/intro.rst
.. include:: docs/source/concept.rst

Usage
=======

Install fintuna via pip.

.. code:: bash

    pip install fintuna

Run the most basic example below. For detailed guidance look at the `docs <https://fintuna.readthedocs.io/en/latest/>`_.

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
