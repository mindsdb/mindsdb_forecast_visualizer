import pandas as pd
from lightwood.api.high_level import predictor_from_problem, ProblemDefinition


if __name__ == '__main__':
    p_name = 'arrival_forecast_example'
    train_df = pd.read_csv('./arrivals_train.csv')

    pdef = ProblemDefinition.from_dict({'target': 'Traffic',
                                        'time_aim': 120,
                                        'nfolds': 10,
                                        'anomaly_detection': True,
                                        'timeseries_settings': {
                                            'use_previous_target': True,
                                            'group_by': ['Country'],
                                            'nr_predictions': 1,
                                            'order_by': ['T'],
                                            'window': 10
                                        }})

    predictor = predictor_from_problem(train_df, problem_definition=pdef)
    predictor.learn(train_df)
