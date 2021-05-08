import os

import pandas as pd
import mindsdb
from mindsdb_native import CONFIG, Predictor

from mindsdb_forecast_visualizer.core.forecaster import forecast


def visualize(predictor_name, df, subset=None, mode='Native', pred_path=None, rolling=1):
    if mode == 'Native':
        # Path configuration
        mdb_path = mindsdb.root_storage_dir  if not pred_path else pred_path
        CONFIG.MINDSDB_STORAGE_PATH = os.path.join(mdb_path, 'predictors')
        print(f'Storage path: {CONFIG.MINDSDB_STORAGE_PATH}')

        mdb_predictor = Predictor(name=predictor_name)

        # TODO: fetch these automatically from training params
        params = {
            'order': ['datetime'],
            'target': 'PM10',
            'group': ['station'],
            'window': 6,
            'nr_predictions': 1,
            'pred_name': predictor_name
        }

        forecast(mdb_predictor,
                 df,
                 params,
                 rolling=rolling,
                 subset=subset)

    dataset = None

    # Native - grouped
    if dataset == 'arrivals':
        # split Arrivals data temporally
        data_path = '/MindsDB/private-benchmarks/benchmarks/datasets/arrivals/data.csv'
        data = pd.read_csv(data_path)

        params = {
            'order': ['T'],
            'target': 'Traffic',
            'group': ['Country'],
            'window': 5,
            'nr_predictions': 5,
            'pred_name': 'arrivals_notebook_tn'
        }

        train = pd.DataFrame()
        test = pd.DataFrame()

        for group in data['Country'].unique():
            subdata = data[data['Country'] == group]
            subtrain = subdata[:-20]
            subtest = subdata[-20 - params['window']:]
            train = train.append(subtrain)
            test = test.append(subtest)

        query_df = test

        # Train directly with native
        predictor = Predictor(name=params['pred_name'])
        predictor.learn(from_data=train, to_predict=params['target'], timeseries_settings={
            'order_by': params['order'],
            'group_by': params['group'],
            'window': params['window'],
            'nr_predictions': params['nr_predictions']},
                        advanced_args={'fixed_confidence': 0.25})

    # Native - ungrouped
    if dataset == 'monthly_sunspots':
        data_path = '/MindsDB/private-benchmarks/benchmarks/datasets/monthly_sunspots/data.csv'
        data = pd.read_csv(data_path)

        params = {
            'order': ['Month'],
            'target': 'Sunspots',
            # 'group': ['Country'],
            'window': 5,
            'nr_predictions': 5,
            'pred_name': 'sunspots_notebook_tn'
        }

        train = data[:-20]
        test = data[-20:]
        query_df = test

        # Train directly with native
        predictor = Predictor(name=params['pred_name'])
        predictor.learn(from_data=train, to_predict=params['target'], timeseries_settings={
            'order_by': params['order'],
            # 'group_by': params['group'],
            'window': params['window'],
            'nr_predictions': params['nr_predictions']
        })

