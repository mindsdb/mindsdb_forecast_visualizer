"""
This example file shows how to use the visualizer with a trained predictor.
To train the predictor, refer to:
    - train_native.py: for training a predictor using MindsDB Native
    - TODO: train_sdk.py: for training a predictor using the MindsDB Python SDK
"""
import pandas as pd
import mindsdb_native
from mindsdb_forecast_visualizer.core.dispatch import visualize

# Demo slices:
#   [:170], subset None; explain how weekly dynamics and weekend dynamics are learnt for
#   [5380:5474], subset None, same for DDS, even though series can present changing behavior
#   [8000:8100], subset None, same for VTS
#   [11645:11744], subset None, same for '1'
#   [13400:13485], subset None, same for '2'
idx_pairs = [[0, 170], [5380, 5474], [8000, 8100], [11645, 11744], [13400, 13485]]


if __name__ == '__main__':
    # Mode: can be SDK or Native
    mode = 'Native'
    base_path = './'
    pred_path = base_path

    # Specify a DataFrame that has your queries (make sure there are enough rows for each group!)
    all_data = pd.read_csv(base_path+'webinarTe.csv')
    query_df = all_data

    rolling = 1

    # None predicts for all groups
    subset = [{'vendor_id': 'VTS'}, {'vendor_id': 'CMT'}, {'vendor_id': '1'}, {'vendor_id': '2'}, {'vendor_id': 'DDS'},]

    # Specify predictor name here, as specified when training it for the first time
    # 1) t+1 predictions
    predictor_name = 'None@@@@@Pretrained'
    nr_predictions = 1

    params = {
        'order': ['pickup_hour'],
        'target': 'fares',
        'group': ['vendor_id'],
        'window': 10,
        'nr_predictions': nr_predictions,
        'pred_name': predictor_name
    }

    # Plot!
    visualize(predictor_name,
              query_df,
              params=params,
              subset=subset,
              mode=mode,
              rolling=rolling,
              pred_path=pred_path)


    predictor_name = 'None@@@@@Taxi_24_12'
    nr_predictions = 12
    params['nr_predictions'] = nr_predictions

    for fro, to in idx_pairs:
        query_df = all_data[fro:to]

        # Plot!
        visualize(predictor_name,
                  query_df,
                  params=params,
                  subset=subset,
                  mode=mode,
                  rolling=rolling,
                  pred_path=pred_path)
