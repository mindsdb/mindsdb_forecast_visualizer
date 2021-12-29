import traceback
from typing import Union, Optional
from copy import deepcopy
from itertools import product
from collections import OrderedDict

import numpy as np
import pandas as pd
from mindsdb_forecast_visualizer.core.plotter import plot
from lightwood.data.cleaner import _standardize_datetime
from lightwood.mixer import Neural, Unit, LightGBM, LightGBMArray, SkTime, Regression, QClassic


def forecast(model,
             data: pd.DataFrame,
             subset: Union[list, None] = None,  # groups to visualize
             show_anomaly: bool = False,
             renderer: str = 'browser',
             backfill: pd.DataFrame = pd.DataFrame(),
             show_train_fit: bool = True,  # whether to show residuals or not
             predargs: dict = {},  # predictor arguments for inference
             warm_start_offset: Union[bool, int] = 0  # None
             ):
    # TODO: make the sktime viz default, and opt to show residuals only if flag is active
    # this means `data` should be the forecasted from the iloc[0]
    # TODO: make sure show_train_fit etc is compatible with all mixers/ensembles
    # TODO: add warm_start_offset once the rest is correct
    # prediction advanced args TODO pass in settings

    if len(backfill) and show_train_fit:
        raise Exception("Either pass all data at once if you want to `show_train_fit`, or decativate this option and pass training data as backfill.")  # noqa

    # instantiate series according to groups
    group_values = OrderedDict()
    tss = model.problem_definition.timeseries_settings
    gby = tss.group_by if tss.group_by is not None else []
    for g in gby:
        group_values[g] = list(data[g].unique())
        if g not in backfill.columns:
            backfill[g] = []
    group_keys = group_values.keys()
    groups = list(product(*[set(x) for x in group_values.values()]))

    if warm_start_offset is None:
        warm_start_offset = tss.window

    target = model.problem_definition.target
    order = tss.order_by

    # extract each series, predict for it, then plot
    for g in groups:
        try:
            filtered_backfill, filtered_data = get_group(g, subset, data, backfill, group_keys, order)

            if filtered_data.shape[0] > 0:
                print(f'Plotting for group {g}...')
                if not tss.allow_incomplete_history:
                    assert filtered_data.shape[0] > tss.window

                forecasting_window = tss.nr_predictions
                # cutoff between in and out-sample for test data
                idx = 0  # filtered_data.shape[0] - forecasting_window - warm_start_offset

                if idx < 0:
                    raise Exception("Too little data.")

                # arrays to plot
                pred_target = []
                time_target = []
                conf_lower = []
                conf_upper = []
                anomalies = []
                real_target = []

                # add data to backfill, if any
                if len(filtered_backfill) > 0:
                    pred_target += [None for _ in range(len(filtered_backfill))]
                    conf_lower += [None for _ in range(len(filtered_backfill))]
                    conf_upper += [None for _ in range(len(filtered_backfill))]
                    anomalies += [None for _ in range(len(filtered_backfill))]
                    real_target += [float(r) for r in filtered_backfill[target]]

                # forecast
                if show_train_fit:
                    predargs['forecast_offset'] = -idx

                predictions = model.predict(filtered_data, args=predargs)
                real_target += [float(r) for r in filtered_data[target]]

                # convert one-step-ahead predictions to unitary lists
                if not isinstance(predictions['prediction'].iloc[0], list):
                    for k in ['prediction', 'lower', 'upper'] + [f'order_{i}' for i in tss.order_by]:
                        predictions[k] = predictions[k].apply(
                            lambda x: [x])

                # divide into in-sample and out-sample predictions, if required
                if show_train_fit:
                    model_fit = predictions[:idx-warm_start_offset]
                    model_forecast = predictions[idx-warm_start_offset:]
                else:
                    model_fit = None
                    model_forecast = predictions

                if show_train_fit:
                    pred_target += [p[0] for p in model_fit['prediction']]
                    conf_lower += [p[0] for p in model_fit['lower']]
                    conf_upper += [p[0] for p in model_fit['upper']]
                    time_target += [p[0] for p in model_fit[f'order_{order[0]}']]
                    if 'anomaly' in model_fit.columns:
                        anomalies += [p for p in model_fit['anomaly']]
                else:
                    pred_target += [None for _ in range(idx+warm_start_offset)]
                    conf_lower += [None for _ in range(idx+warm_start_offset)]
                    conf_upper += [None for _ in range(idx+warm_start_offset)]
                    time_target += [None for _ in range(idx+warm_start_offset)]
                    if 'anomaly' in model_forecast.columns:
                        anomalies += [None for _ in range(idx+warm_start_offset)]

                # forecast corresponds to predicted arrays for the first out-of-sample query data point
                fcst = {
                    'prediction': model_forecast['prediction'].iloc[warm_start_offset],
                    'lower': model_forecast['lower'].iloc[warm_start_offset],
                    'upper': model_forecast['upper'].iloc[warm_start_offset]
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

                # fix timestamps
                time_target = [_standardize_datetime(p) for p in filtered_data[order[0]]]
                delta = pd.Series(time_target).diff().value_counts().index[0]  # inferred
                for i in range(len(pred_target) - len(time_target)):
                    time_target += [time_target[-1] + delta]

                # round confidences
                conf = round(np.array(model_forecast['confidence'].iloc[0][0]).mean(), 2)

                # set titles and legends
                if g != ():
                    title = f'MindsDB T+{forecasting_window} forecast for group {g} (confidence: {conf})'
                else:
                    title = f'MindsDB T+{forecasting_window} forecast (confidence: {conf})'
                titles = {'title': title,
                          'xtitle': 'Date (Unix timestamp)',
                          'ytitle': target,
                          'legend_title': 'Legend'
                          }

                fig = plot(time_target,
                           real_target,
                           pred_target,
                           conf_lower,
                           conf_upper,
                           fh_idx=len(pred_target)-tss.nr_predictions,
                           renderer=renderer,
                           labels=titles,
                           anomalies=anomalies if show_anomaly else None,
                           separate=separate)
                fig.show()

        except Exception:
            print(f"Error in group {g}:")
            print(traceback.format_exc())


def get_group(g, subset, data, backfill, group_keys, order):
    filtered_data = pd.DataFrame() if g != () else data
    filtered_backfill = pd.DataFrame() if g != () else data
    group_dict = {k: v for k, v in zip(group_keys, g)}

    if subset is None or group_dict in subset:
        filtered_data = data
        filtered_backfill = backfill
        for k, v in group_dict.items():
            filtered_data = deepcopy(filtered_data[filtered_data[k] == v])
            filtered_backfill = deepcopy(filtered_backfill[filtered_backfill[k] == v])

    filtered_data = filtered_data.drop_duplicates(subset=order)
    filtered_backfill = filtered_backfill.drop_duplicates(subset=order)

    return filtered_backfill, filtered_data
