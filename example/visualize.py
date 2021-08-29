"""
This example shows how to use the visualizer with a trained predictor.
To train the predictor, refer to:
    - train.py: for training a predictor using Lightwood
    - TODO: train_sdk.py: for training a predictor using the MindsDB Python SDK
"""
import pandas as pd
from mindsdb_forecast_visualizer.core.dispatch import visualize


if __name__ == '__main__':
    # Mode: can be SDK or Native
    mode = 'Native'

    # Specify predictor name here, as specified when training it for the first time
    predictor_name = 'arrival_forecast_example'

    # Specify a DataFrame that has your queries (make sure there are enough rows for each group!)
    query_df = pd.read_csv('example/arrivals_test.csv')

    # Examples: [{'col1': 'val1'}, {'col1': 'val1', 'col2': 'val2', ...}]
    subset = None  # None predicts for all groups

    # Set rolling amount of predictions (1 if predictor was trained for t+N with N>1)
    rolling = 1

    # Set other predictor parameters
    params = {
        'order': ['T'],
        'target': 'Traffic',
        'group': ['Country'],
        'window': 10,
        'nr_predictions': 1,
        'pred_name': predictor_name
    }

    pred_path = None  # set this if the predictor was saved in a non-default location

    visualize(predictor_name,
              query_df,
              params=params,
              subset=subset,
              mode=mode,
              rolling=rolling,
              pred_path=pred_path)
