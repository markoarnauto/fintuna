import optuna
from matplotlib import pyplot as plt
import fintuna as ft

if __name__ == '__main__':

    data, data_specs = ft.data.get_btcpairs_with_social_sentiment()
    crypto_pumps_study = ft.FinStudy(ft.model.Pumps, data, data_specs=data_specs)

    results = crypto_pumps_study.explore(n_trials=50, ensemble_size=4)

    # analyze tuning progress
    optuna.visualization.matplotlib.plot_optimization_history(crypto_pumps_study.study)
    plt.tight_layout()
    plt.show()

    # analyze meta params
    optuna.visualization.matplotlib.plot_param_importances(crypto_pumps_study.study)
    plt.tight_layout()
    plt.show()

    # analyze performance
    ft.utils.plot_backtest(results['performance'], results['benchmark'])

