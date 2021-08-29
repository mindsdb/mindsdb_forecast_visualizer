from collections import OrderedDict
from copy import deepcopy
from itertools import product

import pandas as pd
from mindsdb_forecast_visualizer.core.plotter import plot


def forecast(model, data, params, subset=None, rolling=1, show_anomaly=False):
    """
    model: the MindsDB predictor object
    df: query data; make sure it has > window data points per each group!
    params: dict() that includes parameters used to train the model
    TODO reintroduce limit?: used to arbitrarily cutoff df
    rolling: amount of rolling predictions to do. None triggers T+N based on nr_predictions specified when training. Note: rolling mode generally has worse forecast quality.
    subset: groups that you want to visualize
    show_anomaly: self-descriptive
    """
    # instantiate series according to groups
    group_values = OrderedDict()
    for g in params.get('group', []):
        group_values[g] = list(data[g].unique())
    group_keys = group_values.keys()
    groups = list(product(*[set(x) for x in group_values.values()]))

    # prediction advanced args  TODO pass in settings
    advanced_args = {'anomaly_error_rate': 0.01, 'anomaly_cooldown': 1, 'anomaly_detection': show_anomaly}
    target = params['target']
    order = params['order']

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
                preds = model.predict(when_data=filtered_data, advanced_args=advanced_args)
                observed = preds._data[f"__observed_{params['target']}"]

                # rolling prediction
                if rolling != 1 or model.transaction.lmd.get('tss', {}).get('nr_predictions', 1) == 1:
                    all_preds = [preds]
                    pred = preds._data[f"{params['target']}"]

                    # impute predictions as new data, go through inferring procedure again
                    for t, target_t in enumerate(range(1, rolling)):
                        if isinstance(pred[0], list):
                            pred = [p[0] for p in pred]
                        filtered_data[params['target']] = pred
                        new_preds = model.predict(when_data=filtered_data, advanced_args=advanced_args)
                        pred = new_preds._data[f"{params['target']}"]
                        new_preds._data[f"__observed_{params['target']}"] = observed
                        all_preds.append(new_preds)

                    for i, p in enumerate(all_preds):
                        titles = {'title': f'MindsDB forecast for group {g} (T+{i + 1})',
                                  'xtitle': 'Date (Unix timestamp)',
                                  'ytitle': params['target'],
                                  'legend_title': 'Legend'
                                  }
                        results = pd.DataFrame.from_dict(p._data)
                        time_target = list(filtered_data[order].values.flatten())
                        if any(results[f'__observed_{target}']):
                            real_target = [float(r) for r in results[f'__observed_{target}']]
                        else:
                            print("Warning: no true data points to plot!")
                            real_target = None
                        pred_target = [p for p in results[f'{target}']]
                        if isinstance(pred_target[0], list):
                            pred_target = [p[0] for p in pred_target]
                        conf_lower = [c[0] for c in results[f'{target}_confidence_range']]
                        conf_upper = [c[1] for c in results[f'{target}_confidence_range']]
                        anomalies = [c for c in results[f'{target}_anomaly']] if show_anomaly else None
                        fig = plot(time_target, real_target, pred_target, conf_lower, conf_upper, labels=titles,
                                   anomalies=anomalies)
                        fig.show()

                # trained T+N
                else:
                    forecasting_window = params.get('nr_predictions', 1)
                    idx = len(observed) - forecasting_window

                    results = pd.DataFrame.from_dict(preds._data)
                    time_target = list(results[order].values.flatten())
                    pred_target = [None for _ in range(idx)] + [p for p in preds._data[f'{target}'][idx]]
                    if any(results[f'__observed_{target}']):
                        real_target = [float(r) for r in results[f'__observed_{target}']][:idx + forecasting_window]
                    else:
                        print("Warning: no true data points to plot!")
                        real_target = None
                    delta = time_target[-1] - time_target[-2]
                    time_target = time_target.append([time_target[-1] + delta * i for i in range(forecasting_window)])
                    conf_lower = [None for _ in range(idx)] + [p[0] for p in preds._data[f'{target}_confidence_range'][idx]]
                    conf_upper = [None for _ in range(idx)] + [p[1] for p in preds._data[f'{target}_confidence_range'][idx]]

                    titles = {'title': f'MindsDB forecast for group {g} (T+{params["nr_predictions"]})',
                              'xtitle': 'Date (Unix timestamp)',
                              'ytitle': params['target'],
                              'legend_title': 'Legend'
                              }

                    anomalies = [c for c in results[f'{target}_anomaly']] if show_anomaly else None
                    fig = plot(time_target, real_target, pred_target, conf_lower, conf_upper, labels=titles,
                               anomalies=anomalies)
                    fig.show()
        except Exception as e:
            print(e)
            print(f"error in group {g}")
