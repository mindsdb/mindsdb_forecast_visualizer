from mindsdb_forecast_visualizer.core.forecaster import forecast


def visualize(mdb_predictor, df, subset=None):
    forecast(mdb_predictor, df, subset=subset)
