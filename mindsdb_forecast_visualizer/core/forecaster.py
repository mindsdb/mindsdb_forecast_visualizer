from typing import Union
from copy import deepcopy
from itertools import product
from collections import OrderedDict

import pandas as pd
from mindsdb_forecast_visualizer.core.plotter import plot


def forecast(model,
             data: pd.DataFrame,
             subset: Union[list, None] = None,
             rolling: int = 1,
             show_anomaly: bool = False):
    """
    TODO: assert query data has > window data points per each group!
    TODO reintroduce limit?: used to arbitrarily cutoff df
    rolling: amount of rolling predictions to do. None triggers T+N based on nr_predictions specified when training.
    (note: rolling mode generally has worse forecast quality)
    subset: groups that you want to visualize
    """
    # instantiate series according to groups
    group_values = OrderedDict()
    for g in model.problem_definition.timeseries_settings.group_by:
        group_values[g] = list(data[g].unique())
    group_keys = group_values.keys()
    groups = list(product(*[set(x) for x in group_values.values()]))

    # prediction advanced args TODO pass in settings
    # advanced_args = {'anomaly_error_rate': 0.01, 'anomaly_cooldown': 1, 'anomaly_detection': show_anomaly}
    target = model.problem_definition.target
    order = model.problem_definition.timeseries_settings.order_by

    # extract each series, predict for it, then plot
    for g in groups:
        try:
            filtered_data = pd.DataFrame()
            group_dict = {k: v for k, v in zip(group_keys, g)}

            if subset is None or group_dict in subset:
                filtered_data = data
                for k, v in group_dict.items():
                    filtered_data = deepcopy(filtered_data[filtered_data[k] == v])

            filtered_data = filtered_data.drop_duplicates(subset=order)

            if filtered_data.shape[0] > 0:
                preds = model.predict(filtered_data)  # TODO: advanced args?
                observed = preds["truth"]

                # rolling prediction
                if rolling != 1 or model.problem_definition.timeseries_settings.nr_predictions == 1:
                    all_preds = [preds]
                    pred = preds['prediction']

                    # impute predictions as new data, go through inferring procedure again
                    for t, target_t in enumerate(range(1, rolling)):
                        if isinstance(pred[0], list):
                            pred = [p[0] for p in pred]
                        filtered_data[target] = pred
                        new_preds = model.predict(filtered_data)   # TODO advanced_args?
                        pred = new_preds['prediction']
                        new_preds['truth'] = observed
                        all_preds.append(new_preds)

                    for i, p in enumerate(all_preds):
                        titles = {'title': f'MindsDB forecast for group {g} (T+{i + 1})',
                                  'xtitle': 'Date (Unix timestamp)',
                                  'ytitle': target,
                                  'legend_title': 'Legend'
                                  }
                        results = pd.DataFrame.from_dict(p._data)
                        time_target = list(filtered_data[order].values.flatten())
                        if any(results['truth']):
                            real_target = [float(r) for r in results['truth']]
                        else:
                            print("Warning: no true data points to plot!")
                            real_target = None
                        pred_target = [p for p in results['prediction']]
                        if isinstance(pred_target[0], list):
                            pred_target = [p[0] for p in pred_target]
                        anomalies = results['anomaly'] if show_anomaly else None
                        fig = plot(time_target, real_target, pred_target, results['lower'], results['upper'],
                                   fh_idx=len(pred_target), labels=titles, anomalies=anomalies)
                        fig.show()

                # trained T+N
                else:
                    forecasting_window = model.problem_definition.timeseries_settings.nr_predictions
                    idx = len(observed) - forecasting_window

                    results = preds
                    time_target = list(results[f'order_{order[0]}'].values.flatten())
                    pred_target = [None for _ in range(idx)] + [p for p in preds['prediction'][idx]]
                    if any(results['truth']):
                        real_target = [float(r) for r in results['truth']][:idx + forecasting_window]
                    else:
                        print("Warning: no true data points to plot!")
                        real_target = None
                    delta = model.ts_analysis['deltas'][frozenset(g)][order[0]]
                    time_target = time_target.append([time_target[-1] + delta * i for i in range(forecasting_window)])
                    conf_lower = [None for _ in range(idx)] + [p for p in preds['lower'][idx]]
                    conf_upper = [None for _ in range(idx)] + [p for p in preds['upper'][idx]]

                    titles = {'title': f'MindsDB forecast for group {g} (T+{forecasting_window})',
                              'xtitle': 'Date (Unix timestamp)',
                              'ytitle': target,
                              'legend_title': 'Legend'
                              }

                    anomalies = [c for c in results['anomaly']] if show_anomaly else None
                    fig = plot(time_target, real_target, pred_target, conf_lower, conf_upper,
                               fh_idx=idx, labels=titles, anomalies=anomalies)
                    fig.show()
        except Exception as e:
            print(e)
            print(f"error in group {g}")
