"""
This example shows how to use the visualizer with a trained predictor.
To train the predictor, refer to:
    - train.py: for training a predictor using Lightwood
    - TODO: train_sdk.py: for training a predictor using the MindsDB Python SDK
"""
import pandas as pd
from lightwood.api.high_level import predictor_from_state
from mindsdb_forecast_visualizer.core.dispatch import visualize


if __name__ == '__main__':
    # Mode: can be SDK or Native
    mode = 'Native'

    # Load predictor
    predictor_name = 'arrival_forecast_example'
    with open(f'./{predictor_name}.py', 'r') as f:
        code = f.read()
    predictor = predictor_from_state(f'./{predictor_name}.pkl', code)

    # Specify a DataFrame that has your queries (make sure there are enough rows for each group!)
    query_df = pd.read_csv('./arrivals_test.csv')

    # Examples: [{'col1': 'val1'}, {'col1': 'val1', 'col2': 'val2', ...}]
    subset = None  # None predicts for all groups

    # Set rolling amount of predictions (1 if predictor was trained for t+N with N>1)
    rolling = 1

    visualize(predictor,
              query_df,
              subset=subset,
              mode=mode,
              rolling=rolling)
