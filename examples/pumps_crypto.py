

if __name__ == '__main__':

    study = Study('nth-study')  # load study object
    study.backtest_ensemble(n=4)

    study.ensemble.plot_feature_importance()
    study.ensemble.plot_feature_contribution()

    study.ensemble.plot_performance_distribution(benchmark=None)

