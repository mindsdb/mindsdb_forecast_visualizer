"""
This example shows how to use the visualizer with a trained predictor.
To train a Lightwood predictor, refer to example/train.py
To visualize from a Jupyter notebook, refer to example/visualize.ipynb
"""
import pandas as pd
from lightwood.data.splitter import stratify
from lightwood.api.high_level import predictor_from_state
from mindsdb_forecast_visualizer.core.dispatcher import forecast


if __name__ == '__main__':

    # Load predictor
    predictor_name = 'arrival_forecast_example'

    with open(f'./{predictor_name}.py', 'r') as f:
        code = f.read()
        predictor = predictor_from_state(f'./{predictor_name}.pkl', code)

    # Load DataFrame with queries
    df = pd.read_csv('./arrivals.csv')
    train_df, _, query_df = stratify(df,
                                     pct_train=0.8,
                                     pct_dev=0,
                                     pct_test=0.2,
                                     stratify_on=['Country'],
                                     seed=1,
                                     reshuffle=False)

    # Specify series and plot
    subset = [{'Country': 'UK'}, {'Country': 'US'}]  # None will plot all available series

    forecast(
        predictor,
        query_df,
        subset=subset,
        backfill=train_df,
        show_insample=False
    )
