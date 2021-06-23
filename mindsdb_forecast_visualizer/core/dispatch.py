import os
import mindsdb
from mindsdb_native import CONFIG, Predictor, __version__, __file__

from mindsdb_forecast_visualizer.core.forecaster import forecast


def visualize(predictor_name, df, params=None, subset=None, mode='Native', pred_path=None, rolling=1):
    if pred_path is None and mode == 'Native':
            native_path = os.path.dirname(__file__)
            pred_path = os.path.join(native_path, 'mindsdb_storage', __version__.replace('.', '_'))
    elif pred_path is None and mode == 'SDK':
        pred_path = '/storage/predictors'

    if mode == 'Native':
        # Path configuration
        mdb_path = mindsdb.root_storage_dir  if not pred_path else pred_path
        CONFIG.MINDSDB_STORAGE_PATH = mdb_path
        print(f'Storage path: {CONFIG.MINDSDB_STORAGE_PATH}')

        mdb_predictor = Predictor(name=predictor_name)

        # TODO: fetch params automatically from training params
        forecast(mdb_predictor,
                 df,
                 params,
                 rolling=rolling,
                 subset=subset)

    # TODO mode SDK
