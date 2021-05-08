"""
This example file shows how to use the visualizer with a predictor trained from Native.
TODO: using the SDK or MindsDB
"""
import pandas as pd
from mindsdb_forecast_visualizer.core.dispatch import visualize


if __name__ == '__main__':
    # Mode: can be SDK or Native
    mode = 'Native'

    # Specify predictor name here, as specified when training it for the first time
    predictor_name = 'BeijingAirLean'

    # Specify a DataFrame that has your queries (make sure there are enough rows for each group!)
    test_path = '/MindsDB/temp/TimeSeries/MariaDBWebinar2021/BeijingAir/beijing/PRSA_Data_20130301-20170228/lean_cleaned_te.csv'
    query_df = pd.read_csv(test_path)

    # None predicts for all groups
    # Examples: [{'col1': 'val1'}, {'col1': 'val1', 'col2': 'val2', ...}]
    subset = None

    # Set rolling amount of predictions (1 if predictor was trained for t+N with N>1)
    rolling = 1

    # Plot!
    visualize(predictor_name,
              query_df,
              subset=subset,
              mode=mode,
              rolling=rolling,
              pred_path='/storage')
