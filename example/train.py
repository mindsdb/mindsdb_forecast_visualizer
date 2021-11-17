import pandas as pd

from lightwood.data.splitter import stratify
from lightwood.api.high_level import ProblemDefinition, predictor_from_code, json_ai_from_problem, code_from_json_ai


if __name__ == '__main__':
    # Load data and define the task
    df = pd.read_csv('./arrivals.csv')
    gby = ['Country']
    train_df, _, _ = stratify(df, pct_train=0.8, pct_dev=0, pct_test=0.2, stratify_on=gby, seed=1, reshuffle=False)

    pdef = ProblemDefinition.from_dict({'target': 'Traffic',              # column to forecast
                                        'time_aim': 120,                  # time budget to build a predictor
                                        'nfolds': 10,
                                        'anomaly_detection': True,
                                        'fit_on_all': True,
                                        'timeseries_settings': {
                                            'use_previous_target': True,
                                            'group_by': gby,
                                            'nr_predictions': 5,          # forecast horizon length
                                            'order_by': ['T'],
                                            'window': 10                  # qty of previous data to use when predicting
                                        }})

    # name and generate predictor code
    p_name = 'arrival_forecast_example'
    json_ai = json_ai_from_problem(train_df, problem_definition=pdef)
    predictor_class_code = code_from_json_ai(json_ai)

    # instantiate and train predictor
    predictor = predictor_from_code(predictor_class_code)
    predictor.learn(train_df)

    # save predictor and its code
    predictor.save(f'./{p_name}.pkl')
    with open(f'./{p_name}.py', 'wb') as fp:
        fp.write(predictor_class_code.encode('utf-8'))

