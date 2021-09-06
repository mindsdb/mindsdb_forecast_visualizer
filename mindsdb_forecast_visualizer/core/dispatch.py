from mindsdb_forecast_visualizer.core.forecaster import forecast


def visualize(mdb_predictor, df, subset=None, mode='Native'):
    if mode == 'Native':
        forecast(mdb_predictor, df, subset=subset)

    if mode == 'SDK':
        print('SDK mode currently not supported.')  # TODO mode SDK
