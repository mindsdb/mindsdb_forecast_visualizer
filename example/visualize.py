"""
This example shows how to use the visualizer with a trained predictor.
To train a Lightwood predictor, refer to example/train.py
To visualize from a Jupyter notebook, refer to example/visualize.ipynb
"""
import pandas as pd
from lightwood.api.high_level import predictor_from_state
from mindsdb_forecast_visualizer.core.dispatcher import visualize


if __name__ == '__main__':

    # Load predictor
    predictor_name = 'arrival_forecast_example'

    with open(f'./{predictor_name}.py', 'r') as f:
        code = f.read()
        predictor = predictor_from_state(f'./{predictor_name}.pkl', code)

    # Specify a DataFrame that has your queries (ensuring there are enough rows for each group!)
    query_df = pd.read_csv('./arrivals_test.csv')

    # Determine what series to plot
    subset = [{'Country': 'UK'}, {'Country': 'US'}]  # None will plot all available series

    visualize(predictor, query_df, subset=subset)
