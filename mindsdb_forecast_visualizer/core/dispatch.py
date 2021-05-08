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

    # TODO mode SDK
