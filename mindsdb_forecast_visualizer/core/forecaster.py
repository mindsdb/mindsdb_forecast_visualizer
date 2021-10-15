from typing import Union
from copy import deepcopy
from itertools import product
from collections import OrderedDict

import pandas as pd
from mindsdb_forecast_visualizer.core.plotter import plot


def forecast(model,
             data: pd.DataFrame,
             subset: Union[list, None] = None,  # groups to visualize
             show_anomaly: bool = False,
             renderer='browser'
             ):

    # instantiate series according to groups
    group_values = OrderedDict()
    tss = model.problem_definition.timeseries_settings
    gby = tss.group_by if tss.group_by is not None else []
    for g in gby:
        group_values[g] = list(data[g].unique())
    group_keys = group_values.keys()
    groups = list(product(*[set(x) for x in group_values.values()]))

    # prediction advanced args TODO pass in settings
    # advanced_args = {'anomaly_error_rate': 0.01, 'anomaly_cooldown': 1, 'anomaly_detection': show_anomaly}
    target = model.problem_definition.target
    order = tss.order_by

    # extract each series, predict for it, then plot
    for g in groups:
        try:
            filtered_data = pd.DataFrame() if g != () else data
            group_dict = {k: v for k, v in zip(group_keys, g)}

            if subset is None or group_dict in subset:
                filtered_data = data
                for k, v in group_dict.items():
                    filtered_data = deepcopy(filtered_data[filtered_data[k] == v])

            filtered_data = filtered_data.drop_duplicates(subset=order)

            if filtered_data.shape[0] > 0:
                assert filtered_data.shape[0] > tss.window
                preds = model.predict(filtered_data)  # TODO: advanced args?

                if not isinstance(preds['prediction'].iloc[0], list):
                    for k in ['prediction', 'lower', 'upper'] + [f'order_{i}' for i in tss.order_by]:
                        preds[k] = preds[k].apply(lambda x: [x])  # convert one-step-ahead predictions to unitary lists

                observed = preds["truth"]

                forecasting_window = tss.nr_predictions
                idx = len(observed) - forecasting_window

                group_key = frozenset(g) if g != () else '__default'
                delta = model.ts_analysis['deltas'][group_key][order[0]]
                time_target = [p[0] for p in preds[f'order_{order[0]}']]
                time_target.append([time_target[-1] + delta * i for i in range(1, forecasting_window+1)])

                # add one-step-ahead predictions for all observed data points
                pred_target = []
                conf_lower = []
                conf_upper = []

                for i in range(idx):
                    pred_target += [preds['prediction'][i][0]]
                    conf_lower += [preds['lower'][i][0]]
                    conf_upper += [preds['upper'][i][0]]

                # add forecast based on latest observed data point
                if any(preds['truth']):
                    real_target = [float(r) for r in preds['truth']][:idx + forecasting_window]
                else:
                    print("Warning: no true data points to plot!")
                    real_target = None

                fcst = {
                    'prediction': preds['prediction'][idx],
                    'lower': preds['lower'][idx],
                    'upper': preds['upper'][idx]
                }

                # wrap if needed
                for k, v in fcst.items():
                    if not isinstance(v, list):
                        fcst[k] = [v]

                if forecasting_window == 1:
                    separate = False
                    for k, v in fcst.items():
                        fcst[k] = [v[0]]
                else:
                    separate = True

                pred_target += [p for p in fcst['prediction']]
                conf_lower += [p for p in fcst['lower']]
                conf_upper += [p for p in fcst['upper']]

                titles = {'title': f'MindsDB forecast for group {g} (T+{forecasting_window})',
                          'xtitle': 'Date (Unix timestamp)',
                          'ytitle': target,
                          'legend_title': 'Legend'
                          }

                anomalies = [c for c in preds['anomaly']] if show_anomaly else None

                fig = plot(time_target,
                           real_target,
                           pred_target,
                           conf_lower,
                           conf_upper,
                           fh_idx=idx,
                           renderer=renderer,
                           labels=titles,
                           anomalies=anomalies,
                           separate=separate)
                fig.show()

        except Exception as e:
            print(e)
            print(f"error in group {g}")
