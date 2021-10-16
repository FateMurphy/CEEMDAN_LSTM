CEEMDAN_LSTM
===
## Background 
CEEMDAN_LSTM is a Python project for decomposition-integration forecasting models based on EMD methods and LSTM. It is a relatively imperfect module but beginners can quickly use it to make a decomposition-integration prediction by `CEEMDAN`, Complete Ensemble Empirical Mode Decomposition with Adaptive Noise [(Torres et al. 2011)](https://ieeexplore.ieee.org/abstract/document/5947265/), and `LSTM`, Long Short-Term Memory recurrent neural network [(Hochreiter and Schmidhuber, 1997)](https://ieeexplore.ieee.org/abstract/document/6795963). If you use or refer to the content of this project, please cite paper: "Zhou, F. T. (2021). Carbon price forecasting based on CEEMDAN and LSTM. Unpublished now."

## Flowchart
![](https://github.com/FateMurphy/CEEMDAN_LSTM/blob/02d156d9fffd6b4a748337f6ea44c7122051d223/Hybrid%20CEEMDAN-VMD-LSTM%20predictor%20flowchart.svg)

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

## Usage
### 0. Help or Example
You can use the code to call for a guideline or help.

```python
cl.guideline()
cl.example()
```

### 1. Declare the path
Determine the folder to store dataset and the folder to store output results. If it runs successfully, the program will display the picture of the series。  
The default dataset saving path: D:\\CEEMDAN_LSTM\\  
The default figures saving path: D:\\CEEMDAN_LSTM\\figures\\  
The default logs and output saving path: D:\\CEEMDAN_LSTM\\subset\\  
The default dataset name: cl_sample_dataset (must be csv file)  
    
```python
series = cl.declare_path()
```
If you want to use anthor folders, you can use `cl.declare_path(path="",figure_path="",log_path="",dataset_name="")` (must be csv file)
If you want to use your own data, you can use `cl.declare_path(series=pd.Series)` (must be pd.Series)

### 2. Help or Example
  
