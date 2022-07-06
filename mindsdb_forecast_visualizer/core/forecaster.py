import traceback
from typing import Union
from copy import deepcopy
from itertools import product
from collections import OrderedDict

import numpy as np
import pandas as pd
from mindsdb_forecast_visualizer.core.plotter import plot

from lightwood.mixer.nhits import NHitsMixer
from lightwood.data.cleaner import _standardize_datetime


def forecast(model,
             data: pd.DataFrame,
             subset: Union[list, None] = None,
             show_anomaly: bool = False,
             renderer: str = 'browser',
             backfill: pd.DataFrame = pd.DataFrame(),
             show_insample: bool = True,
             predargs: dict = {},
             warm_start_offset: Union[bool, int] = None
             ):
    """
    :param data: data for which forecasts will be made. Note that if a predictor has cold start, you need to provide a warm_start_offset for best results.
    :param subset: what groups to visualize, provided a predictor that was trained with a group-by set of columns
    :param show_anomaly: whether to show anomaly bars
    :param renderer: plots can be generated using any supported plotly backend
    :param backfill: dataframe that is used as historical context in the visualization, but for which no forecasts are needed. NOTE: MindsDB, Lightwood and this tool all assume there is no gap between the backfill data and the forecasted data.
    :param show_insample: if True, in-sample forecasts will be plotted for any available backfill data.
    :param predargs: arguments passed to the predictor when generating forecasts for the visualization.
    :param warm_start_offset: how many rows from the backfill dataframe should be appended at the start of `data` to serve as historical context for mixers that have a cold start (e.g. neural, gbm).
    """  # noqa

    if show_insample and len(backfill) == 0:
        raise Exception("You must pass a dataframe with the predictor's training data to show in-sample forecasts.")

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

    target = model.problem_definition.target
    forecasting_window = tss.horizon
    order = tss.order_by

    if warm_start_offset is None:
        warm_start_offset = tss.window

    # extract each series, predict for it, then plot
    for g in groups:
        try:
            filtered_backfill, filtered_data = get_group(g, subset, data, backfill, group_keys, order)

            if filtered_data.shape[0] > 0:
                print(f'Plotting for group {g}...')

                # check offset for warm start
                if isinstance(model.mixers[model.ensemble.indexes_by_accuracy[0]], NHitsMixer):
                    filtered_data = pd.concat([filtered_backfill.iloc[-warm_start_offset:], filtered_data.iloc[[0]]])
                else:
                    filtered_data = pd.concat([filtered_backfill.iloc[-warm_start_offset:], filtered_data])


                if not tss.allow_incomplete_history:
                    assert filtered_data.shape[0] > tss.window

                # arrays to plot
                pred_target = []
                time_target = []
                conf_lower = []
                conf_upper = []
                anomalies = []
                real_target = []

                # add data to backfill, if any
                if len(filtered_backfill) > 0:
                    real_target += [float(r) for r in filtered_backfill[target]]

                # forecast & divide into in-sample and out-sample predictions, if required
                if show_insample:
                    offset = predargs.get('forecast_offset', 0)
                    predargs['forecast_offset'] = -len(filtered_backfill)
                    model_fit = model.predict(filtered_backfill, args=predargs)
                    predargs['forecast_offset'] = offset
                else:
                    model_fit = None
                    if len(filtered_backfill) > 0:
                        pred_target += [None for _ in range(len(filtered_backfill))]
                        conf_lower += [None for _ in range(len(filtered_backfill))]
                        conf_upper += [None for _ in range(len(filtered_backfill))]
                        anomalies += [None for _ in range(len(filtered_backfill))]

                predargs['forecast_offset'] = -warm_start_offset
                model_forecast = model.predict(filtered_data, args=predargs).iloc[warm_start_offset:]
                filtered_data = filtered_data.iloc[warm_start_offset:]
                real_target += [float(r) for r in filtered_data[target]][:tss.horizon]

                # convert one-step-ahead predictions to unitary lists
                if not isinstance(model_forecast['prediction'].iloc[0], list):
                    for k in ['prediction', 'lower', 'upper'] + [f'order_{i}' for i in tss.order_by]:
                        model_forecast[k] = model_forecast[k].apply(lambda x: [x])
                        if show_insample:
                            model_fit[k] = model_fit[k].apply(lambda x: [x])

                if show_insample:
                    pred_target += [p[0] for p in model_fit['prediction']]
                    conf_lower += [p[0] for p in model_fit['lower']]
                    conf_upper += [p[0] for p in model_fit['upper']]
                    time_target += [p[0] for p in model_fit[f'order_{order[0]}']]
                    if 'anomaly' in model_fit.columns:
                        anomalies += [p for p in model_fit['anomaly']]

                # forecast corresponds to predicted arrays for the first out-of-sample query data point
                fcst = {
                    'prediction': model_forecast['prediction'].iloc[0],
                    'lower': model_forecast['lower'].iloc[0],
                    'upper': model_forecast['upper'].iloc[0]
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
                delta = model.ts_analysis['deltas'][g]
                for i in range(len(pred_target) - len(time_target)):
                    time_target += [time_target[-1] + delta]

                # round confidences
                conf = model_forecast['confidence'].values.mean()

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
                           fh_idx=len(pred_target)-tss.horizon,
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
