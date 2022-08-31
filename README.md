CEEMDAN_LSTM
===
## Background 
CEEMDAN_LSTM is a Python project for decomposition-integration forecasting models based on EMD methods and LSTM. It is a relatively imperfect module but beginners can quickly use it to make a decomposition-integration prediction by `CEEMDAN`, Complete Ensemble Empirical Mode Decomposition with Adaptive Noise [(Torres et al. 2011)](https://ieeexplore.ieee.org/abstract/document/5947265/), and `LSTM`, Long Short-Term Memory recurrent neural network [(Hochreiter and Schmidhuber, 1997)](https://ieeexplore.ieee.org/abstract/document/6795963). If you use or refer to the content of this project, please cite paper: [(F. Zhou, Z. Huang, C. Zhang,
Carbon price forecasting based on CEEMDAN and LSTM, Applied Energy, 2022, Volume 311, 118601, ISSN 0306-2619.)](https://doi.org/10.1016/j.apenergy.2022.118601.)

## This program is an in-sample prediction with some look-ahead bias.
![](https://github.com/FateMurphy/CEEMDAN_LSTM/blob/main/figure/Hybrid%20CEEMDAN-VMD-LSTM%20predictor%20flowchart.svg)
## This 
## Install
First, download `CEEMDAN_LSTM.py` to your computer, and create a new file that can run python. If you can use Jupyter, you can directly download `CEEMDAN LSTM of CarbonPrice.ipynb` at the same time and follow the steps inside to complete the prediction.

```python
import CEEMDAN_LSTM as cl
```
If an error occurs, please check the modules that need to be installed. Note if the error is that `pyemd` does not exist, please install `EMD-signal` rather than pyemd module.

```python
pip install datetime
pip install EMD-signal # pyemd
pip install vmdpy
pip install sampen
pip install tensorflow==2.5.0
```
  
## Preparation
### Help or Example
You can use the code to call for a guideline or help. It is not recommended to use `cl.run_exmaple()` because the integration way by sample entropy is not the same every time.
```python
cl.guideline()
cl.guideline_vars()
cl.example()
```

### Declare the path
Determine folders to store datasets and output results. The folder will be created automatically when the code is run for the first time. Please download `cl_sample_dataset.csv` and put it into the declared folder. If it runs successfully, the program will display a picture of the series.  
The default dataset saving path: D:\\CEEMDAN_LSTM\\  
The default figures saving path: D:\\CEEMDAN_LSTM\\figures\\  
The default logs and output saving path: D:\\CEEMDAN_LSTM\\subset\\  
The default dataset name: cl_sample_dataset (must be csv file)    
```python
series = cl.declare_path()
```
If you want to use anthor folders, you can use `cl.declare_path(path="",figure_path="",log_path="",dataset_name="")` (must be csv file).  
If you want to use your own data, you can use `cl.declare_path(series=pd.Series)` (must be pd.Series).

  
## Start to predict
### 0. Statistical tests (not necessary)
The code will ouput the reuslt of ADF test, Ljung-Box Test, Jarque-Bera Test, and plot ACF and PACF figures.
```python
cl.statistical_tests()
```
### 1. Declare mode and variables
`Importantly!`, you need to declare decomposition mode and global variables.It is recommended to declare variables first every time you predict or decompose and other operations.  
Default value:
`mode='ceemdan'`, Mainly determine the decomposition method  
`form=''`, Integration form only effective after integration  
`data_back=30`, The number of previous days related to today  
`periods=100`, The length of the days to forecast  
`epochs=100`, LSTM epochs  
`patience=10`, Patience of adaptive learning rate and early stop, suggesting epochs/10  
```python
cl.declare_vars()
```

### 2. Decomposition and Integration
Choose a method to decompose the original data from `mode='emd'`, `mode='eemd'`, or `mode='ceemdan'` by `cl.declare_vars(mode='')`.
```python
imfs = cl.emd_decom()
```
Calculate sample entropy
```python
cl.sample_entropy()
```
Integrate subsequences, where form represents the integration way.
```python
form = [[0,1],[2,3,4],[5,6,7]]
cl.integrate(inte_form=form) 
```

### 3. Forecast
Before forecasting, it is recommended to redeclare the variables. Variable `form` you can get from integration.
There are many methods to choose, including `single`, `ensemble`, `respective`, `hybrid`, and `mutiple`.
```python
cl.declare_vars(mode='ceemdan',form='233',epochs=100) # 
cl.Single_LSTM()
```
You can try other methods
```python
cl.Single_LSTM(draw=False,show_model=False)
cl.Ensemble_LSTM()
cl.Hybrid_LSTM()
cl.Multi_pred(run_times=10,ensemble_lstm=True,respective_lstm=True)
```

### 4. Re-decomposition
After the first decomposition above, some sub-sequences are still difficult to predict, so they can be decomposed again.
```python
df_redecom = cl.re_decom(redecom_mode='vmd',redecom_list=[0]) 
df_input['sum'] = series.values
cl.Ensemble_LSTM(df=df_redecom,show_model=False)
```
For hybrid models, you can directly add parameters `redecom='vmd'`.
```python
cl.Hybrid_LSTM(redecom='vmd')
```
  
## An example of BeijingETS.csv
### Hybrid method
```python
df_bjETS = pd.read_csv(PATH+'data\\BeijingETS.csv',header=0,parse_dates=["date"],
                       date_parser=lambda x: datetime.strptime(x, "%Y%m%d"))
series_bj = pd.Series(df_bjETS['close'].values,index = df_bjETS['date']).sort_index().astype(float)
cl.declare_vars(mode='ceemdan')
ceemdan_bj = cl.emd_decom(series=series_bj)
cl.sample_entropy(imfs_df=ceemdan_bj)
inte_bj = cl.integrate(df=ceemdan_bj,inte_form=[[0,1],[2,3],[4,5],[6,7,8]]) # form may not be the same every time
cl.declare_vars(mode='ceemdan_se',form='233',epochs=10)
cl.Hybrid_LSTM(df=inte_bj,redecom='vmd')
```

### Time-saving method
```python
df_bjETS = pd.read_csv(PATH+'data\\BeijingETS.csv',header=0,parse_dates=["date"],
                       date_parser=lambda x: datetime.strptime(x, "%Y%m%d"))
series_bj = pd.Series(df_bjETS['close'].values,index = df_bjETS['date']).sort_index().astype(float)
cl.declare_vars(mode='ceemdan',epochs=1000)
ceemdan_bj = cl.emd_decom(series=series_bj)
df_vmd_bj = cl.re_decom(df=ceemdan_bj,redecom_mode='vmd',redecom_list=0) 
cl.Ensemble_LSTM(df=df_vmd_bj)
```

### Predict the next day
Set `next_pred` for `cl.Hybrid_LSTM(next_pred=True)`.
```python
cl.Hybrid_LSTM(next_pred=True)
```
or you can try time saving method by `cl.run_predict()`
```python
cl.run_predict(series=series,epochs=1000)
```


## Postscript
If you have any questions, please leave your comment or email me.
