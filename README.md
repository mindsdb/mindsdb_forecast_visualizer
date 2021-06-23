# MindsDB Forecast Visualizer

The purpose of this tool is to aid in obtaining quick visualizations for time series forecasts provided by a MindsDB predictor.

![](./docs/plot.png)

At the moment, the tool supports predictors trained from (or loaded to) `mindsdb_native`, but MindsDB Python SDK support is coming soon.

## Documentation

For now, there is no documentation apart from a quick example. Please refer to `example/train_native.py` to train a forecasting model for airplane arrival data, and then `example/example.py` which shows what parameters can be configured to call the plotter.

Note: make sure the path to `mindsdb_forecast_visualizer` is added to your python path environment variable before running these scripts from the package root folder.