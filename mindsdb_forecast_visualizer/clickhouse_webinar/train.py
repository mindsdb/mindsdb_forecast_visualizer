import pandas as pd
import mindsdb_native
from mindsdb_native import CONFIG


if __name__ == '__main__':
    names = ['None@@@@@Pretrained', 'None@@@@@Taxi_24_12']
    nr_preds = [1, 12]

    train_df = pd.read_csv('./webinarTr.csv')
    CONFIG.MINDSDB_STORAGE_PATH = './'

    for nr_preds, name in zip(nr_preds, names):
        predictor = mindsdb_native.Predictor(name=name)
        predictor.learn(from_data=train_df,
                        timeseries_settings={
                            'window': 24,
                            'nr_predictions': nr_preds,
                            'order_by': ['pickup_hour'],
                            'group_by': ['vendor_id']
                        },
                        to_predict='fares')
