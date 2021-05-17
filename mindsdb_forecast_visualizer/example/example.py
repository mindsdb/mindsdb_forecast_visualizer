"""
This example file shows how to use the visualizer with a trained predictor.
To train the predictor, refer to:
    - train_native.py: for training a predictor using MindsDB Native
    - TODO: train_sdk.py: for training a predictor using the MindsDB Python SDK
"""
import pandas as pd
import mindsdb_native
from mindsdb_forecast_visualizer.core.dispatch import visualize


if __name__ == '__main__':
    # Mode: can be SDK or Native
    mode = 'Native'

    # Specify predictor name here, as specified when training it for the first time
    predictor_name = 'arrival_forecast_example'

    # Specify a DataFrame that has your queries (make sure there are enough rows for each group!)
    query_df = pd.read_csv('./arrivals_test.csv')

    # None predicts for all groups
    # Examples: [{'col1': 'val1'}, {'col1': 'val1', 'col2': 'val2', ...}]
    subset = None

    # Set rolling amount of predictions (1 if predictor was trained for t+N with N>1)
    rolling = 1

    pred_path = None  # set if predictor was saved in a non-default location (or e.g. a previous native version)
    if pred_path is None and mode == 'Native':
        pred_path = '/Users/Pato/Work/MindsDB/mindsdb_native/mindsdb_native/mindsdb_storage/' + \
                    mindsdb_native.__version__.replace('.', '_')
    elif pred_path is None and mode == 'SDK':
        pred_path = '/storage/predictors'

    # Plot!
    visualize(predictor_name,
              query_df,
              subset=subset,
              mode=mode,
              rolling=rolling,
              pred_path=pred_path)
