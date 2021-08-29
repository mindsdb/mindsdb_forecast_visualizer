import pandas as pd
import mindsdb_native


if __name__ == '__main__':
    p_name = 'arrival_forecast_example'
    train_df = pd.read_csv('./arrivals_train.csv')

    predictor = mindsdb_native.Predictor(name=p_name)
    predictor.learn(from_data=train_df,
                    timeseries_settings={
                        'window': 3,
                        'order_by': ['T'],
                        'group_by': ['Country']
                    },
                    to_predict='Traffic')
