import mindsdb_native
import pandas as pd

if __name__ == '__main__':
    p_name = 'ontime_mariadb'

    train_path = '/MindsDB/mindsdb_customer_projects/MariaDB/time-series-webinar/new_ontime2019_5carrs_tr.csv'
    test_path = '/MindsDB/mindsdb_customer_projects/MariaDB/time-series-webinar/new_ontime2019_5carrs_te.csv'

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    target = ''

    learn = True

    predictor = mindsdb_native.Predictor(name=p_name)

    if learn:
        predictor.learn(from_data=train_df, to_predict=target)