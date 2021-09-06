# MindsDB Forecast Visualizer

The purpose of this tool is to aid in obtaining quick visualizations for time series forecasts provided by a Lightwood predictor.

![](./docs/plot.png)

At the moment, the tool supports predictors trained from (or loaded to) `lightwood >= 1.0`, but MindsDB Python SDK support is coming soon~ish.

## Documentation

For now, there is no documentation but most if not all functionality is showcased through examples. Please refer to `example/train.py` to train a forecasting model for airplane arrival data (which includes 4 different time series), and then `example/visualize.py` to call the plotter.

Note: make sure the path to `mindsdb_forecast_visualizer` is added to your python path environment variable before running these scripts from the package root folder.
