CEEMDAN_LSTM
===
GitHub: https://github.com/FateMurphy/CEEMDAN_LSTM  
Thanks for everyone's support and advice. I recently made a major update to the code, and now you can install the module directly using `pip install CEEMDAN_LSTM`. Feel free to email me if you have any questions or error reports.
#### old version has been moved to `__old__`
data: you can find them in `/CEEMDAN_LSTM/dataset` and they have been already built into the module, using `cl.load_dataset(dataset_name='sse_index.csv')` to get them.  
flowchart: only rename.

## Background 
CEEMDAN_LSTM is a Python module for decomposition-integration forecasting models based on EMD methods and LSTM. It aims at helping beginners quickly make a decomposition-integration forecasting by `CEEMDAN`, Complete Ensemble Empirical Mode Decomposition with Adaptive Noise [(Torres et al. 2011)](https://ieeexplore.ieee.org/abstract/document/5947265/), and `LSTM`, Long Short-Term Memory recurrent neural network [(Hochreiter and Schmidhuber, 1997)](https://ieeexplore.ieee.org/abstract/document/6795963). If you use or refer to the content of this module, please cite paper: [(F. Zhou, Z. Huang, C. Zhang,
Carbon price forecasting based on CEEMDAN and LSTM, Applied Energy, 2022, Volume 311, 118601, ISSN 0306-2619.)](https://doi.org/10.1016/j.apenergy.2022.118601.)
### Flowchart
![](https://github.com/FateMurphy/CEEMDAN_LSTM/blob/main/figure/Hybrid%20forecasting%20method.svg)
#### Note, as it decomposes the entire series first, there is some look-ahead bias.

## Install
### (1) PyPi (recommended)
The quickest way to install package is through pip.
```python
pip install CEEMDAN_LSTM
```
### (2) From package
Download the package `CEEMDAN_LSTM-1.2a0.tar.gz` by click `Code` -> `Download ZIP`. After unzipping, move the package where you like.
```python
pip install .(your file path)/CEEMDAN_LSTM-1.2a0.tar.gz
```
### (3) From source
If you want to modify the code, you should download the code and build package yourself. The source is publicaly available and hosted on GitHub: https://github.com/FateMurphy/CEEMDAN_LSTM. To download the code you can either go to the source code page and click `Code` -> `Download ZIP`, or use git command line.  
After modify the code, you can install the modified package by using command line:
```python
python setup.py install
```
Or, you can link to the path for the convenient modification, eg. `sys.path.append(.your file path/)`, and then import.

## Import and quickly predict
```python
import CEEMDAN_LSTM as cl
cl.quick_keras_predict(data=None) # default dataset: sse_index.csv
```
#### Load dataset
```python
data = cl.load_dataset()
# data = pd.read_csv(your_file_path + its_name + '.csv', header=0, index_col=['date'], parse_dates=['date'])
```

## Help and example
You can use the code to call for a help. You can copy the code from the output of `cl.show_keras_example()` to run forecasting and help you learn more about the code.
```python
cl.help()
cl.show_keras_example()
cl.show_keras_example_model()
cl.details_keras_predict(data=None)
```

## Start to Forecast
Take Class: keras_predictor() as an example.
### Brief summary and forecast
```python
data = cl.load_dataset()
series = data['close'] # choose a DataFrame column 
cl.statis_tests(series)
kr = cl.keras_predictor()
df_result = kr.hybrid_keras_predict(data=series, show_data=True, show_model=True, plot_result=True, save_result=True)
```

### 0. Statistical tests (not necessary)
The code will ouput the reuslt of ADF test, Ljung-Box Test, Jarque-Bera Test, and plot ACF and PACF figures to evaluate stationarity, autocorrelation, and normality.
```python
cl.statis_tests()
```

### 1.Declare the parameters
Note, when declare the PATH, folders will be created automatically, inculding the figure and log folders.
```python
kr = cl.keras_predictor(PATH=None, FORECAST_HORIZONS=30, FORECAST_LENGTH=30, KERAS_MODEL='GRU', 
                        DECOM_MODE='CEEMDAN', INTE_LIST=None, REDECOM_LIST={'co-imf0':'vmd'},
                        NEXT_DAY=False, DAY_AHEAD=1, NOR_METHOD='minmax', FIT_METHOD='add', 
                        USE_TPU=False , **kwargs))
```
| HyperParameters | Description | 
| :-----| :----- | 
|PATH               |the saving path of figures and logs, eg. 'D:/CEEMDAN_LSTM/'|
|FORECAST_HORIZONS  |the length of each input row(x_train.shape), which means the number of previous days related to today, also called Timestep, Forecast_horizons, or Sliding_windows_length in some papers|
|FORECAST_LENGTH    |the length of the days to forecast (test set)|
|KERAS_MODEL        |the Keras model, eg. 'GRU', 'LSTM', 'DNN', 'BPNN', or model = Sequential()|
|DECOM_MODE         |the decomposition method, eg.'EMD', 'VMD', 'CEEMDAN'|
|INTE_LIST          |the integration list, eg. pd.Dataframe, (int) 3, (str) '233', (list) [0,0,1,1,1,2,2,2], ...|
|REDECOM_LIST       |the re-decomposition list, eg. '{'co-imf0':'vmd', 'co-imf1':'emd'}', pd.DataFrame|
|NEXT_DAY           |set True to only predict next out-of-sample value|
|DAY_AHEAD          |define to forecast n days' ahead, eg. 0, 1, 2 (default int 1)|
|NOR_METHOD         |the normalizing method, eg. 'minmax'-MinMaxScaler, 'std'-StandardScaler, otherwise without normalization|
|FIT_METHOD         |the fitting method to stablize the forecasting result (not necessarily useful), eg. 'add', 'ensemble'|
|USE_TPU            |change Keras model to TPU model (for google Colab)|

| Keras Parameters | Description (more details refer to https://keras.io) | 
| :-----| :----- | 
|epochs             |training epochs/iterations, eg. 30-1000|
|dropout            |dropout rate of 3 dropout layers, eg. 0.2-0.5|
|units              |the units of network layers, which (3 layers) will set to 4*units, 2*units, units, eg. 4-32|
|activation         |activation function, all layers will be the same, eg. 'tanh', 'relu'|
|batch_size         |training batch_size for parallel computing, eg. 4-128|
|shuffle            |whether randomly disorder the training set during training process, eg. True, False|
|verbose            |report of training process, eg. 0 not displayed, 1 detailed, 2 rough|
|valid_split        |proportion of validation set during training process, eg. 0.1-0.2|
|opt                |network optimizer, eg. 'adam', 'sgd'|
|opt_lr             |optimizer learning rate, eg. 0.001-0.1|
|opt_loss           |optimizer loss, eg. 'mse','mae','mape','hinge', refer to https://keras.io/zh/losses/.|
|opt_patience       |optimizer patience of adaptive learning rate, eg. 10-100|
|stop_patience      |early stop patience, eg. 10-100|

### 2. Forecast
You can try the following forecasting methods. Note, `kr.` is the class defined in step 1, necessary for the code.
```python
df_result = kr.single_keras_predict(data, show_data=True, show_model=True, plot_result=True, save_result=True)
# df_result = kr.ensemble_keras_predict(data, show_data=True, show_model=True, plot_result=True, save_result=True)
# df_result = kr.respective_keras_predict(data, show_data=True, show_model=True, plot_result=True, save_result=True)
# df_result = kr.hybrid_keras_predict(data, show_data=True, show_model=True, plot_result=True, save_result=True)
# df_result = kr.multiple_predict(data, run_times=10, predict_method='single', save_each_result=False)
```
| Forecast Method | Description | 
| :-----| :----- | 
|Single Method      |Use Keras model to directly forecast with vector input|
|Ensemble Method    |Use decomposition-integration Keras model to directly forecast with matrix input|
|Respective Method  |Use decomposition-integration Keras model to respectively forecast each IMFs with vector input|
|Hybrid Method      |Use the ensemble method to forecast high-frequency IMF and the respective method for other IMFs.|
|Multiple Method    |Multiple run of above method|
|Rolling Method     |Rolling run of above method to avoid the look-ahead bias, but take a long long time|

### 3. Validate 
#### (1) Plot heatmap
You need to install `seaborn` first, and the input should be 2D-array.
```python
cl.plot_heatmap(data, corr_method='pearson', fig_path=None)
```
#### (2) Diebold-Mariano-Test (DM test)
Dm test will output the DM test statistics and its p-value. You can refer to https://github.com/johntwk/Diebold-Mariano-Test.
```python
rt = cl.dm_test(actual_lst, pred1_lst, pred2_lst, h=1, crit="MSE", power=2)
```

### 4. Next-day Forecast
Set `NEXT_DAY=True`.
```python
kr = cl.keras_predictor(NEXT_DAY=True)
df_result = kr.hybrid_keras_predict(data, show_data=True, show_model=True, plot_result=True, save_result=True)
# df_result = kr.rolling_keras_predict(data, predict_method='single', save_each_result=False)
```
## Discussion
### 1. Look-ahead bias
As the predictor will decompose the entire series first before splitting the training and test set, there is a look-ahead bias. It is still an issue about how to avoid the look-ahead bias.
### 2. VMD decompose
The vmdpy module can only decompose the even-numbered length time series. When forecasting an odd-numbered length one, this module will delete the oldest data point. It is still an issue how to modify VMD decompose. Moreover, selecting the K parameters is important for the VMD method, and hence, I will add some methods to choose a suitable K, such as OVMD, REI, SampEn, and so on.
### 3. Rolling forecasting 
Rolling forecasting costs a lot of time. Like a 30-forecast-length prediction, it will run 30 times cl.hybrid_keras_predict(), so I am not sure if it is really effective or not.
