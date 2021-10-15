from mindsdb_forecast_visualizer.core.forecaster import forecast


def visualize(predictor, df, subset=None, mode='Lightwood'):
    if mode == 'Lightwood':
        forecast(predictor, df, subset=subset)
    # @TODO: adds MindsDB
    else:
        raise Exception(f"Visualizer not supported for mode {mode}")
