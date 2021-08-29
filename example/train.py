import pandas as pd
from lightwood.api.high_level import ProblemDefinition, predictor_from_code, code_from_problem


if __name__ == '__main__':
    # Load data and define the task
    train_df = pd.read_csv('./arrivals_train.csv')
    pdef = ProblemDefinition.from_dict({'target': 'Traffic',              # column to forecast
                                        'time_aim': 120,                  # time budget to build a predictor
                                        'nfolds': 10,
                                        'anomaly_detection': True,
                                        'timeseries_settings': {
                                            'use_previous_target': True,
                                            'group_by': ['Country'],
                                            'nr_predictions': 1,          # forecast horizon length
                                            'order_by': ['T'],
                                            'window': 10                  # qty of previous data to use when predicting
                                        }})

    # name and generate predictor code
    p_name = 'arrival_forecast_example'
    predictor_class_code = code_from_problem(train_df, problem_definition=pdef)

    # instantiate and train predictor
    predictor = predictor_from_code(predictor_class_code)
    predictor.learn(train_df)

    # save predictor and its code
    predictor.save(f'./{p_name}.pkl')
    with open(f'./{p_name}.py', 'wb') as fp:
        fp.write(predictor_class_code.encode('utf-8'))

