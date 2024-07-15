# MindsDB Forecast Visualizer

:warning:  This repository is now archived and won’t be maintained further. We recommend using other libraries such as ![nixtla/utilsforecast](https://nixtlaverse.nixtla.io/utilsforecast/plotting.html) instead

The purpose of this tool is to aid in obtaining quick visualizations for time series forecasts provided by a Lightwood predictor.

![](./docs/plot.png)

The tool supports predictors trained with `lightwood >= 1.0`.

## Documentation

There is no documentation. Most functionality is showcased through examples. Please refer to:
* `example/train.py` to train a Lightwood forecaster for airplane arrival data (which includes 4 different time series)
* `example/visualize.py` to plot predictions from this model in your web browser
* `example/visualize.ipynb` to plot predictions from this model inside a jupyter notebook

Note: if you’ve cloned the repository (as opposed to `pip install`ing), make sure the path to `mindsdb_forecast_visualizer` is added to your python path environment variable before running these scripts from the package root folder.
