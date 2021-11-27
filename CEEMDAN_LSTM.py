#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 
# Created on 2021-10-1 22:37
# Author: FATE ZHOU
# 

from __future__ import division, print_function
__version__ = '1.0.0a'
__module_name__ = 'CEEMDAN_LSTM'
print('Importing...', end = '')
# 0.Introduction
#==============================================================================================
# This is a initialization module for a complete CEEMDAN-LSTM forecasting process.
# This is a module still testing. Some errors may occur during runtime.
# The following modules need to be installed before importing at Anaconda3.
# Otherwise, please install the corresponding modules by yourself according to the warnings.
# pip install EMD-signal
# pip install sampen
# pip install vmdpy
# pip install datetime
# pip install tensorflow-gpu==2.5.0
# pip install scikit-learn


# import CEEMDAN_LSTM as cl


# Contents
#==============================================================================================
# 0.Introduction
# 1.Guideline functions
# 2.Declare default variables
# 3.Decomposition, Sample entropy, Re-decomposition, and Integration
# 4.LSTM Model Functions
# 5.CEEMDAN-LSTM Forecasting Functions
# 6.Hybrid Forecasting Functions
# 7.Statistical Tests
# Appendix if main run example
#==============================================================================================

# Import basic modules
# More modules will be imported before the corresponding function
# import logging # logger = logging.getLogger(__name__)
import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") # Ignore some annoying warnings
from datetime import datetime

# Import module for EMD decomposition
# It is the EMD-signal module with different name to import
from PyEMD import EMD,EEMD,CEEMDAN #For module 'PyEMD', please use 'pip install EMD-signal' instead.

# Import module for sample entropy
from sampen import sampen2

# Import modules for LSTM prediciton
# Sklearn
from sklearn.preprocessing import MinMaxScaler # Normalization
from sklearn.metrics import r2_score # R2
from sklearn.metrics import mean_squared_error # MSE
from sklearn.metrics import mean_absolute_error # MAE
from sklearn.metrics import mean_absolute_percentage_error # MAPE
# Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from tensorflow.keras.layers import GRU,Flatten
#from tcn import TCN # pip install keras-tcn

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model # To use plot_model, you need to install software graphviz
from tensorflow.python.client import device_lib

# Statistical tests
from statsmodels.tsa.stattools import adfuller # adf_test
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test # LB_test
from statsmodels.stats.stattools import jarque_bera as jb_test # JB_test
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # plot_acf_pacf

# 1.Guideline functions
#==============================================================================================
# A guideline for this module
def guideline():
    print(__module_name__ +' is a module still testing. Some errors may occur during runtime.')
    print("For module 'PyEMD', please use 'pip install EMD-signal' instead.")
    print("Without input, all functions will try to run the sample dataset cl_sample_dataset.csv where you can change it by cl.declare_path(dataset_name=)")
    print('##################################')
    print('Available Functions')
    print('##################################')
    print('#1.Guideline functions: ')
    print("-------------------------------")
    print("cl.guideline()")
    print("cl.guideline_vars()")
    print("cl.example()")
    print("cl.run_example()")
    print("cl.show_devices()")
    print()
    print('#2.Declare variables: ')
    print("-------------------------------")
    print("cl.declare_path(path,figure_path,log_path,dataset_name,series)")
    print("cl.declare_vars(mode,form,data_back,periods,epochs,patience)")
    print("cl.declare_LSTM_vars(cells,dropout,optimizer_loss,batch_size,validation_split,verbose,shuffle)")
    print("cl.declare_LSTM_MODEL(model)")
    print("For the details of each variable, use cl.guideline_vars() to call for a help.")
    print()
    print("#3.Decomposition, Sample entropy, Re-decomposition, and Integration")
    print("-------------------------------")
    print("cl.emd_decom(series,trials=10,draw=True)")
    print("cl.vmd_decom(series,alpha=2000,tau=0,K=5,DC=0,init=1,tol=1e-7,draw=True)")
    print("cl.sample_entropy(imfs_df)")
    print("cl.re_decom(df,redecom_mode='ceemdan',redecom_list=[0],draw=True,trials=10,imfs_num=10)")
    print("cl.integrate(df,inte_form=[[0,1],[2,3,4],[5,6,7]])")
    print()
    print("#4.LSTM Model Functions and #5.CEEMDAN-LSTM Forecasting Functions")
    print("-------------------------------")
    print("cl.evl(y_test, y_pred, scale='0 to 1')")
    print("cl.plot_all(lstm_type,pred_ans)")
    print("cl.Single_LSTM(series,draw=True,uni=False,show_model=True)")
    print("cl.Ensemble_LSTM(df,draw=True,uni=False,show_model=True)")
    print("cl.Respective_LSTM(df,draw=True,uni=False,show_model=True)")
    print("cl.Hybrid_LSTM(df,draw=True,enlarge=10,redecom='vmd')")
    print("cl.Multi_pred(df,run_times=10,uni_nor=False,single_lstm=False,ensemble_lstm=False,respective_lstm=False,hybrid_lstm=False,redecom='vmd'):")
    print()
    print("#Other.Statistical Tests for Data:")
    print("-------------------------------")
    print("cl.statistical_tests(series)")

# A guideline for variables
def guideline_vars():
    print("The details of each variable")
    print('##################################')
    print("cl.declare_path() is used to declare following variables:")
    print("These variables are used to control saving path and dataset.")
    print("-------------------------------")
    print("PATH : The default dataset saving path")
    print("FIGURE_PATH : The default figures saving path")
    print("LOG_PATH : The default logs and output saving path")
    print("DATASET_NAME : The default dataset name of a csv file")
    print("SERIES : The default time series dataset. Load from DATASET_NAME or input a pd.Series.")
    print()
    print("cl.declare_vars() is used to declare following variables:")
    print("IMPORTANT!!! These variables are used to control forecast")
    print("-------------------------------")
    print("MODE : Mainly determine the decomposition method such as 'ceemdan' and 'ceemdan_se'")
    print("FORM : Integration form only effective after integration")
    print("DATE_BACK : The number of previous days related to today")
    print("PERIODS : The length of the days to forecast")
    print("EPOCHS : LSTM epochs")
    print("PATIENCE : Patience of adaptive learning rate and early stop, suggest 1-20")
    print()
    print("cl.declare_LSTM_vars is used to declare following variables:")
    print("These variables are used to control LSTM model")
    print("-------------------------------")
    print("CELLS : The units of LSTM layers and 3 LSTM layers will set to 4*CELLS, 2*CELLS, CELLS.")
    print("DROPOUT : Dropout rate of 3 Dropout layers, suggest 0.1-0.5")
    print("OPTIMIZER_LOSS : Adam optimizer loss such as 'mse','mae','mape','hinge' refer to https://keras.io/zh/losses/")
    print("BATCH_SIZE : LSTM training batch_size for parallel computing, suggest 10-100")
    print("VALIDATION_SPLIT : Proportion of validation set to training set, suggest 0-0.2")
    print("VERBOSE : Report of the training process, 0 not displayed, 1 detailed, 2 rough")
    print("SHUFFLE : In the training process, whether to randomly disorder the training set")
    print()
    print("cl.declare_LSTM_vars is used to set up a keras model independently:")
    print("-------------------------------")
    print("LSTM_MODEL : Define the Keras model by model = Sequential() with input shape [DATE_BACK,the number of features]")
    print()
    print("cl.declare_uni_method is used to set up the unified normalization method:")
    print("-------------------------------")
    print("METHOD : Method for unified normalization only 0,1,2,3")

# An example
def example():
    print("Start your first prediction by following steps:")
    print("check your dataset and use cl.run_example() can directly run the following example around 1000 seconds.")
    print("##################################")
    print("(0) Import:")
    print("    import CEEMDAN_LSTM as cl")
    print("(1) Declare a path for saving files:")
    print("    series = cl.declare_path()")
    print("(2) CEEMDAN decompose:")
    print("    cl.declare_vars(mode='ceemdan') # set decomposition method")
    print("    imfs = cl.emd_decom()")
    print("(3) Sample Entropy:")
    print("    cl.sample_entropy()")
    print("(4) Integrating IMFs:")
    print("    cl.integrate(inte_form=[[0,1],[2,3,4],[5,6,7]]) # form 233")
    print("(5) Forecast:")
    print("    cl.declare_vars(mode='ceemdan_se',form='233',epochs=100) # declare variables for forecast")
    print("    cl.Hybrid_LSTM(redecom='vmd') # ceemdan_se233_data.csv")
    print("Also you can try other methods such as:")
    print("    cl.statistical_tests()")
    print("    cl.Single_LSTM()")
    print("    cl.Ensemble_LSTM()")
    print("    cl.Respective_LSTM()")

# Run the example above
def run_example():
    print('An example of cl.example() is running around 1000 seconds.')
    print('##################################')
    print("\n(1) Declare a path for saving files:")
    print("-------------------------------")
    series = declare_path()
    print("\n(2) CEEMDAN decompose:")
    print("-------------------------------")
    declare_vars(mode='ceemdan') # reset to default value
    imfs = emd_decom()
    print("\n(3) Sample Entropy:")
    print("-------------------------------")
    sample_entropy()
    print("\n(4) Integrating IMFs:")
    print("-------------------------------")
    integrate(inte_form=[[0,1],[2,3,4],[5,6,7]]) # form 233
    print("\n(5) Forecast:")
    print("-------------------------------")
    declare_vars(mode='ceemdan_se',form='233') # declare variables for forecast
    Hybrid_LSTM(redecom='vmd') # ceemdan_se233_data.csv

# Run the example above
def run_predict(series,next_pred=True,epochs=1000):
    print('An example of time-saving method is running around 400 seconds.')
    print('##################################')
    declare_vars(mode='ceemdan') # reset to default value
    df_ceemdan = emd_decom(series=series)
    df_vmd = re_decom(df=df_ceemdan,redecom_mode='vmd',redecom_list=0) 
    global EPOCHS,PATIENCE
    tmp_epochs, tmp_patience = EPOCHS,PATIENCE
    EPOCHS,PATIENCE = epochs,int(epochs/10)
    Ensemble_LSTM(df=df_vmd,show_model=False,next_pred=next_pred)
    EPOCHS,PATIENCE = tmp_epochs, tmp_patience

# Show Tensorflow running device
def show_devices():
    import tensorflow as tf
    print(device_lib.list_local_devices())

# 2.Declare default variables
#==============================================================================================

# Files variables
# -------------------------------
# The default dataset saving path: D:\\CEEMDAN_LSTM\\
PATH = 'D:\\CEEMDAN_LSTM\\'
# The default figures saving path: D:\\CEEMDAN_LSTM\\figures\\
FIGURE_PATH = PATH+'figures\\'
# The default logs and output saving path: D:\\CEEMDAN_LSTM\\subset\\
LOG_PATH = PATH+'subset\\'
# The default dataset name of a csv file: cl_sample_dataset.csv (must be csv file)
DATASET_NAME = 'cl_sample_dataset'
# The default time series dataset. Load from DATASET_NAME or input a pd.Series.
SERIES = None

# Files variables declare functions
# -------------------------------
# Declare the path
# You can also enter the time series data directly by declare_path(series)
def declare_path(path=PATH,figure_path=FIGURE_PATH,log_path=LOG_PATH,dataset_name=DATASET_NAME,series=SERIES):
    # Check input
    global PATH,FIGURE_PATH,LOG_PATH,DATASET_NAME,SERIES
    for x in ['path','figure_path','log_path','dataset_name']:
        if type(vars()[x])!=str: raise TypeError(x+' should be strings such as D:\\\\CEEMDAN_LSTM\\\\...\\\\.')
    if path == '' or figure_path == '' or log_path == '':
        raise TypeError('PATH should be strings such as D:\\\\CEEMDAN_LSTM\\\\...\\\\.')
    # declare FIGURE_PATH,LOG_PATH if user only inputs PATH or inputs them at different folders

    # Change path
    ori_figure_path, ori_log_path = FIGURE_PATH, LOG_PATH
    if path != PATH: 
        # Fill path if lack like 'PATH=D:\\CEEMDAN_LSTM' to 'PATH=D:\\CEEMDAN_LSTM\\'
        if path[-1] != '\\': path = path + '\\' 
        PATH = path
        FIGURE_PATH,LOG_PATH = PATH+'figures\\',PATH+'subset\\' 
    if figure_path != ori_figure_path: 
        if figure_path[-1] != '\\': figure_path  = figure_path + '\\'
        FIGURE_PATH = figure_path # Separate figure saving path
    if log_path != ori_log_path: 
        if log_path[-1] != '\\': log_path  = log_path + '\\'
        LOG_PATH = log_path # Separate log saving path
    DATASET_NAME,SERIES = dataset_name,series # update variables

    # Check or create a folder for saving 
    print('Saving path: %s'%PATH)
    for p in [PATH,FIGURE_PATH,LOG_PATH]:
        if not os.path.exists(p): os.makedirs(p)

    # Check whether inputting a series 
    if SERIES is not None:
        if not isinstance(series, pd.Series): raise ValueError('The inputting series must be pd.Series.')
        else: 
            print('Get input series named:',str(series.name))
            SERIES = series.sort_index() # sorting
    
    # Load Data for csv file
    else:
        # Check csv file
        if not (os.path.exists(PATH+DATASET_NAME+'.csv')):
            raise ImportError('Dataset is not exists. Please input dataset_name='+DATASET_NAME+' and check it in: '+PATH
                              +'. You can also input a pd.Series directly.')
        else:
            print('Load sample dataset: '+DATASET_NAME+'.csv')
            # Load sample dataset
            df_ETS = pd.read_csv(PATH+DATASET_NAME+'.csv',header=0,parse_dates=['date'], 
                                  date_parser=lambda x: datetime.strptime(x, '%Y%m%d'))

            # Select close data and convert it to time series data 
            if 'date' not in df_ETS.columns or 'close' not in df_ETS.columns: 
                raise ValueError("Please name the date column and the required price column as 'date' and 'close' respectively.")
            SERIES = pd.Series(df_ETS['close'].values,index = df_ETS['date']) #选择收盘价
            SERIES = SERIES.sort_index() # sorting

    # Save the required data to avoid chaanging the original data
    pd.DataFrame.to_csv(SERIES,PATH+'demo_data.csv')

    # Show data plotting
    fig = plt.figure(figsize=(10,4))
    SERIES.plot(label='Original data', color='#0070C0') #F27F19 orange #0070C0 blue
    plt.title('Original Dataset Figure')
    plt.xlabel('Time')
    plt.legend()
    plt.savefig(FIGURE_PATH+'Original Dataset Figure.svg', bbox_inches='tight')
    plt.show()

    return SERIES # pd.Series

# Model variables
# -------------------------------
# Mainly determine the decomposition method 
MODE = 'ceemdan' 
# Integration form only effective after integration
FORM = '' # such as '233' or 233
# The number of previous days related to today
DATE_BACK = 30 
# The length of the days to forecast
PERIODS = 100 
# LSTM epochs
EPOCHS = 100
# Patience of adaptive learning rate and early stop, suggest 1-20
PATIENCE = 10

# Declare model variables
def declare_vars(mode=MODE,form=FORM,data_back=DATE_BACK,periods=PERIODS,epochs=EPOCHS,patience=None):
    print('##################################')
    print('Global Variables')
    print('##################################')
    
    # Change and Check
    global MODE,FORM,DATE_BACK,PERIODS,EPOCHS,PATIENCE
    FORM = str(form)
    MODE,DATE_BACK,PERIODS,EPOCHS = mode.lower(),data_back,periods,epochs
    if patience is None: PATIENCE = int(EPOCHS/10)
    else: PATIENCE = patience
    check_vars()

    # Show
    print('MODE:'+str.upper(MODE))
    print('FORM:'+str(FORM))
    print('DATE_BACK:'+str(DATE_BACK))
    print('PERIODS:'+str(PERIODS))
    print('EPOCHS:'+str(EPOCHS))
    print('PATIENCE:'+str(PATIENCE))

# Check the type of model variables
def check_vars():
    global FORM
    if MODE not in ['emd','eemd','ceemdan','emd_se','eemd_se','ceemdan_se']:
        raise TypeError('MODE should be emd,eemd,ceemdan,emd_se,eemd_se,or ceemdan_se rather than %s.'%str(MODE))
    if not type(FORM) == str:
        raise TypeError('FORM should be strings in digit such as 233 or "233" rather than %s.'%str(FORM))
    if not (type(DATE_BACK) == int and DATE_BACK>0):
        raise TypeError('DATE_BACK should be a positive integer rather than %s.'%str(DATE_BACK))
    if not (type(PERIODS) == int and PERIODS>=0):
        raise TypeError('PERIODS should be a positive integer rather than %s.'%str(PERIODS))
    if not (type(EPOCHS) == int and EPOCHS>0):
        raise TypeError('EPOCHS should be a positive integer rather than %s.'%str(EPOCHS))
    if not (type(PATIENCE) == int and PATIENCE>0):
        raise TypeError('PATIENCE should be a positive integer rather than %s.'%str(PATIENCE))
    if FORM == '' and (MODE in ['emd_se','eemd_se','ceemdan_se']):
        raise ValueError('FORM is not delcared. Please delcare is as form = 233 or "233".')

# Check dataset input a test one or use the default one
# -------------------------------
def check_dataset(dataset,input_form,no_se=False,use_series=False,uni_nor=False): # uni_nor is using unified normalization method or not
    file_name = ''
    # Change MODE
    global MODE
    if no_se: # change MODE to the MODE without se 
        check_vars()
        if MODE[-3:] == '_se':
            print('MODE is',str.upper(MODE),'now, using %s instead.'%(str.upper(MODE[:-3])))
            MODE = MODE[:-3]
    # Use SERIES as not dataset
    if use_series:
        if SERIES is None: 
            raise ValueError('SERIES is not declared. Please declare it by series=cl.declare_path(path=PATH).')
    # Check user input 
    if dataset is not None:  
        if input_form == 'series' :
            if isinstance(dataset, pd.Series):  
                print('Get input pd.Series named:',str(dataset.name))
                input_dataset = dataset.copy(deep=True)
            else: raise ValueError('The inputting series must be pd.Seriesrather than %s.'%type(dataset))
        elif input_form == 'df':
            if isinstance(dataset, pd.DataFrame): 
                print('Get input pd.DataFrame.')
                tmp_sum = None
                if 'sum' in dataset.columns:
                    tmp_sum = dataset['sum']
                    dataset = dataset.drop('sum', axis=1, inplace=False)
                if 'co-imf0' in dataset.columns: col_name = 'co-imf'
                else: col_name = 'imf'
                dataset.columns = [col_name+str(i) for i in range(len(dataset.columns))] # change column names to imf0,imf1,...
                if tmp_sum is not None:  dataset['sum'] = tmp_sum
                input_dataset = dataset.copy(deep=True)
            else: raise ValueError('The inputting df must be pd.DataFrame rather than %s.'%type(dataset))
        else: raise ValueError('Something wrong happen in module %s.'%__name__)
        file_name = 'test_'
    else: # Check default dataset and load
        if input_form == 'series' : # Check SERIES
            if not isinstance(SERIES, pd.Series): 
                raise ValueError('SERIES is not declared. Please declare it by series=cl.declare_path(path=PATH).')
            else: input_dataset = SERIES.copy(deep=True)
        elif input_form == 'df':
            check_vars()
            data_path = PATH+MODE+FORM+'_data.csv'
            if not os.path.exists(data_path):
                raise ImportError('Dataset %s does not exist in '%(data_path)+PATH)
            else: input_dataset = pd.read_csv(data_path,header=0,index_col=0)

    # other warnings
    if METHOD == 0 and uni_nor: 
        print('Attention!!! METHOD = 0 means no using the unified normalization method. Declare METHOD by declare_uni_method(method=METHOD)')

    return input_dataset,file_name

# Declare LSTM model variables
# -------------------------------
# The units of LSTM layers and 3 LSTM layers will set to 4*CELLS, 2*CELLS, CELLS.
CELLS = 32
# Dropout rate of 3 Dropout layers
DROPOUT = 0.2 
# Adam optimizer loss such as 'mse','mae','mape','hinge' refer to https://keras.io/zh/losses/
OPTIMIZER_LOSS = 'mse'
# LSTM training batch_size for parallel computing, suggest 10-100
BATCH_SIZE = 16
# Proportion of validation set to training set, suggest 0-0.2
VALIDATION_SPLIT = 0.1
# Report of the training process, 0 not displayed, 1 detailed, 2 rough
VERBOSE = 0
# In the training process, whether to randomly disorder the training set
SHUFFLE = True


# Declare LSTM variables
def declare_LSTM_vars(cells=CELLS,dropout=DROPOUT,optimizer_loss=OPTIMIZER_LOSS,batch_size=BATCH_SIZE,validation_split=VALIDATION_SPLIT,verbose=VERBOSE,shuffle=SHUFFLE):
    print('##################################')
    print('LSTM Model Variables')
    print('##################################')
    PATIENCE
    # Changepatience=
    global CELLS,DROPOUT,OPTIMIZER_LOSS,BATCH_SIZE,VALIDATION_SPLIT,VERBOSE,SHUFFLE
    CELLS,DROPOUT,OPTIMIZER_LOSS = cells,dropout,optimizer_loss
    BATCH_SIZE,VALIDATION_SPLIT,VERBOSE,SHUFFLE = batch_size,validation_split,verbose,shuffle

    # Check
    if not (type(CELLS) == int and CELLS>0): raise TypeError('CELLS should a positive integer.')
    if not (type(DROPOUT) == float and DROPOUT>0 and DROPOUT<1): raise TypeError('DROPOUT should a number between 0 and 1.')
    if not (type(BATCH_SIZE) == int and BATCH_SIZE>0):
        raise TypeError('BATCH_SIZE should be a positive integer.')
    if not (type(VALIDATION_SPLIT) == float and VALIDATION_SPLIT>0 and VALIDATION_SPLIT<1):
        raise TypeError('VALIDATION_SPLIT should be a number best between 0.1 and 0.4.')
    if VERBOSE not in [0,1,2]:
        raise TypeError('VERBOSE should be 0, 1, or 2. The detail level of the training message')
    if type(SHUFFLE) != bool:
        raise TypeError('SHUFFLE should be True or False.')
    
    # Show
    print('CELLS:'+str(CELLS))
    print('DROPOUT:'+str(DROPOUT))
    print('OPTIMIZER_LOSS:'+str(OPTIMIZER_LOSS))
    print('BATCH_SIZE:'+str(BATCH_SIZE))
    print('VALIDATION_SPLIT:'+str(VALIDATION_SPLIT))
    print('VERBOSE:'+str(VERBOSE))
    print('SHUFFLE:'+str(SHUFFLE))

# Define the Keras model by model = Sequential() with input shape [DATE_BACK,the number of features]
LSTM_MODEL = None 

# Change Kreas model
def declare_LSTM_MODEL(model=LSTM_MODEL):
    print("LSTM_MODEL has changed to be %s and start your forecast."%model)
    global LSTM_MODEL
    LSTM_MODEL = model
            

# LSTM model example
def LSTM_example():
    print('Please input a Keras model with input_shape = (DATE_BACK, the number of features)')
    print('##################################')
    print("model = Sequential()")
    print("model.add(LSTM(100, input_shape=(30, 1), activation='tanh'))")
    print("model.add(Dropout(0.5))")
    print("model.add(Dense(1,activation='tanh'))")
    print("model.compile(loss='mse', optimizer='adam')")
    print("cl.declare_LSTM_MODEL(model=model)")

# Build LSTM model
def LSTM_model(shape):
    if LSTM_MODEL is None:
        model = Sequential()
        model.add(LSTM(CELLS*4, input_shape=(shape[1], shape[2]), activation='tanh', return_sequences=True))
        model.add(Dropout(DROPOUT))
        model.add(LSTM(CELLS*2,activation='tanh',return_sequences=True))
        model.add(Dropout(DROPOUT))
        model.add(LSTM(CELLS,activation='tanh',return_sequences=False))
        model.add(Dropout(DROPOUT))
        model.add(Dense(1,activation='tanh'))
        model.compile(loss=OPTIMIZER_LOSS, optimizer='adam')
        return model
    elif LSTM_MODEL == 'GRU':
        model = Sequential()
        model.add(GRU(CELLS*4, input_shape=(shape[1], shape[2]), activation='tanh', return_sequences=True))
        model.add(Dropout(DROPOUT))
        model.add(GRU(CELLS*2,activation='tanh',return_sequences=True))
        model.add(Dropout(DROPOUT))
        model.add(GRU(CELLS,activation='tanh',return_sequences=False))
        model.add(Dropout(DROPOUT))
        model.add(Dense(1,activation='tanh'))
        model.compile(loss=OPTIMIZER_LOSS, optimizer='adam')
        return model
    elif LSTM_MODEL == 'DNN':
        model = Sequential()
        model.add(Dense(CELLS*4, input_shape=(shape[1], shape[2]), activation='tanh'))
        model.add(Dropout(DROPOUT))
        model.add(Dense(CELLS*2,activation='tanh'))
        model.add(Dropout(DROPOUT))
        model.add(Flatten())
        model.add(Dense(CELLS,activation='tanh'))
        model.add(Dropout(DROPOUT))
        model.add(Dense(1,activation='tanh'))
        model.compile(loss=OPTIMIZER_LOSS, optimizer='adam')
        return model
    elif LSTM_MODEL == 'BPNN':
        model = Sequential()
        model.add(Dense(CELLS*4, input_shape=(shape[1], shape[2]), activation='tanh'))
        model.add(Dropout(DROPOUT))
        model.add(Flatten())
        model.add(Dense(1,activation='tanh'))
        model.compile(loss=OPTIMIZER_LOSS, optimizer='adam')
        return model
    else: return LSTM_MODEL

"""
# TCN
model = Sequential()
model.add(TCN(CELLS*4, input_shape=(shape[1], shape[2]), activation='tanh'))
model.add(Dropout(DROPOUT))
model.add(Dense(1,activation='tanh'))
model.compile(loss=OPTIMIZER_LOSS, optimizer='adam')
return model
"""
# Other variables
# -------------------------------
# Method for unified normalization only 0,1,2,3
METHOD = 0 

# declare Method for unified normalization
def declare_uni_method(method=None):
    if method not in [0,1,2,3]: raise TypeError('METHOD should be 0,1,2,3.')
    global METHOD
    METHOD = method
    print('Unified normalization method (%d) is start using.'%method)


# 3.Decomposition, Sample entropy, Re-decomposition, and Integration
#==============================================================================================

# EMD decomposition
# -------------------------------
# Decompose adaptively and plot function
# Residue is named the last IMF
# Declare MODE by declare_vars first
def emd_decom(series=None,trials=10,re_decom=False,re_imf=0,draw=True): 
    # Check input
    dataset,file_name = check_dataset(series,input_form='series') # include check_vars()
    series = dataset.values

    # Initialization
    print('%s decomposition is running.'%str.upper(MODE))
    if MODE == 'emd':decom = EMD()
    elif MODE == 'eemd':decom = EEMD()
    elif MODE == 'ceemdan':decom = CEEMDAN()
    else: raise ValueError('MODE must be emd, eemd, ceemdan when EMD decomposing.')

    # Decompose
    decom.trials = trials # Number of the white noise input
    imfs_emd = decom(series)
    imfs_num = np.shape(imfs_emd)[0]

    if draw:
        # Plot original data
        series_index = range(len(series))
        fig = plt.figure(figsize=(16,2*imfs_num))
        plt.subplot(1+imfs_num, 1, 1 )
        plt.plot(series_index, series, color='#0070C0') #F27F19 orange #0070C0 blue
        plt.ylabel('Original data')
    
        # Plot IMFs
        for i in range(imfs_num):
            plt.subplot(1 + imfs_num,1,2 + i)
            plt.plot(series_index, imfs_emd[i, :], color='#F27F19')
            plt.ylabel(str.upper(MODE)+'-IMF'+str(i))
 
        # Save figure
        fig.align_labels()
        plt.tight_layout()
        if file_name == '':
            if (re_decom==False): plt.savefig(FIGURE_PATH+file_name+str.upper(MODE)+' Result.svg', bbox_inches='tight')
            else: plt.savefig(FIGURE_PATH+'IMF'+str(re_imf)+' '+str.upper(MODE)+' Re-decomposition Result.svg', bbox_inches='tight')
        plt.show()
    
    # Save data
    imfs_df = pd.DataFrame(imfs_emd.T)
    imfs_df.columns = ['imf'+str(i) for i in range(imfs_num)]
    if file_name == '':
        if (re_decom==False): 
            pd.DataFrame.to_csv(imfs_df,PATH+file_name+MODE+'_data.csv')
            print(str.upper(MODE)+' finished, check the dataset: ',PATH+file_name+MODE+'_data.csv')

    return imfs_df # pd.DataFrame

# Sample entropy
# -------------------------------
# You can also enter the imfs_df directly 
def sample_entropy(imfs_df=None): # imfs_df is pd.DataFrame
    df_emd,file_name = check_dataset(imfs_df,input_form='df') # include check_vars()
    if file_name == '': file_name = str.upper(MODE+FORM)
    else: file_name = 'a Test'
    print('Sample entropy of %s is running.'%file_name)  
        
    # Calculate sample entropy with m=1,2 and r=0.1,0.2
    imfs = df_emd.T.values
    sampen = []
    for i in imfs:
        for j in (0.1,0.2):
            sample_entropy = sampen2(list(i),mm=2,r=j,normalize=True)
            sampen.append(sample_entropy)
    
    # Output
    entropy_r1m1,entropy_r1m2,entropy_r2m1,entropy_r2m2 = [],[],[],[]
    for i in range(len(sampen)):
        if (i%2)==0: # r=0.1    
            entropy_r1m1.append(sampen[i][1][1])# m=1
            entropy_r1m2.append(sampen[i][2][1])# m=2
        else: # r=0.2
            entropy_r2m1.append(sampen[i][1][1])# m=1
            entropy_r2m2.append(sampen[i][2][1])# m=2
 
    # Plot     
    fig = plt.figure()
    x = list(range(0,len(imfs),1))
    plt.plot(x,entropy_r1m1,'k:H',label='m=1 r=0.1')
    plt.plot(x,entropy_r2m1,'b:D',label='m=1 r=0.2')
    plt.plot(x,entropy_r1m2,'c:s',label='m=2 r=0.1')
    plt.plot(x,entropy_r2m2,'m:h',label='m=2 r=0.2')
    plt.xlabel('IMFs')
    plt.ylabel('Sample Entropy')
    plt.legend()
    if file_name == '': fig.savefig(FIGURE_PATH+'Sample Entropy of %s IMFs.svg'%(file_name), bbox_inches='tight')
    plt.show()

# Integrate IMFs and Residue
# -------------------------------
# Residue is the last IMF in dataset
# inte_form defines the IMFs to be integrated such as [[0,1],[2,3,4],[5,6,7]]
def integrate(df=None,inte_form=[[0,1],[2,3,4],[5,6,7]]):
    # Check inte_form
    if type(inte_form)!=list: raise ValueError('inte_form must be a list like [[0,1],[2,3,4],[5,6,7]].')
    # Change to a one-line list for checking duplicates
    check_list = sum(inte_form,[]) # if error, check your inte_form.
    if len(check_list)!=len(set(check_list)): # Check duplicates by set
        raise ValueError('inte_form has repeated IMFs. Please set it again.')
    
    # Check input df and load dataset
    df_emd,file_name = check_dataset(df,input_form='df',no_se=True,use_series=True) # include check_vars()

    # Check inte_form
    if len(check_list) != len(df_emd.columns):
        raise ValueError('inte_form does not match the total number %d of IMFs'%len(df_emd.columns))

    # Integrating (Create Co-IMFs)
    num = len(inte_form) # num is the number of IMFs after integrating, the Co-IMFs
    form = ''
    for i in range(num):
        str_form = ['imf'+str(i) for i in inte_form[i]]
        locals()['co-imf'+str(i)] = pd.Series(df_emd[str_form].sum(axis=1))
        form = form+str(len(inte_form[i])) # name the file and [[0,1],[2,3,4],[4,5,6]] is 233
    print('The Integrating Form:',form)
    
    # Plot original data by SERIES
    fig = plt.figure(figsize=(16,2*num)) 
    plt.subplot(1+num, 1, 1)
    if file_name == '':
        plt.plot(range(len(SERIES)), SERIES, color='#0070C0') #F27F19 orange #0070C0 blue
    else: # if input plot the sum
        df_sum = df_emd.T.sum()
        plt.plot(range(len(df_sum.index)), df_sum.values, color='#0070C0')
    plt.ylabel('Original data')

    # Name the figure of each Co-IMF
    re = ''
    if form=='44': re='Re1-'
    elif form=='323': re='Re2-'
    elif form=='233': re='Ori-'
    elif form=='224': re='Re3-'
    elif form=='2222': re='Re4-'
    elif form=='2123': re='Re5-'
    elif form=='1133': re='Re6-'
        
    # Plot Co-IMFs
    imfs_name, co_imfs = [], []
    for i in range(num):
        plt.subplot(1+num,1,i+2)
        plt.plot(range(len(df_emd.index)), vars()['co-imf'+str(i)], color='#F27F19') #F27F19
        plt.ylabel(re+'Co-IMF'+str(i))
        imfs_name.append('co-imf'+str(i)) # series name of Co-IMFs
        co_imfs.append(vars()['co-imf'+str(i)])
        
    # Save figure
    fig.align_labels()
    plt.tight_layout()
    plt.savefig(FIGURE_PATH+file_name+str.upper(MODE)+' Integration Figure in form '+form+'.svg', bbox_inches='tight')
    plt.show()
    
    # Save Co-IMFs
    df_co_emd = pd.DataFrame(co_imfs).T
    df_co_emd.columns=imfs_name
    if file_name == '': pd.DataFrame.to_csv(df_co_emd,PATH+file_name+MODE+'_se'+form+'_data.csv')
    print('Integration finished, check the dataset: ',PATH+file_name+MODE+'_se'+form+'_data.csv')
    if file_name != '': return df_co_emd

# Re-decomposition
# -------------------------------
# re_list is the IMF for re-decomposition
def re_decom(df=None,redecom_mode='ceemdan',redecom_list=[0],draw=True,trials=10,imfs_num=10): 
    # Check inputs
    if isinstance(redecom_list, int): redecom_list=[redecom_list] # if redecom_list is int
    if not isinstance(redecom_list, list): 
        raise ValueError('redecom_list must be a list like [0,1] or an integer like 0 or 1.')
    df_emd,file_name = check_dataset(df,input_form='df') # include check_vars()

    # Check redecom_list
    if len(redecom_list) > len(df_emd.columns) or max(redecom_list) > len(df_emd.columns)-1:
        raise  ValueError('redecom_list exceeds the final IMF: '+str(len(df_emd.columns)-1))
    # Check duplicates
    if len(redecom_list)!=len(set(redecom_list)): raise ValueError('redecom_list has repeated IMFs. Please set it again.')
    col_name = df_emd.columns[0][:-1] # get the IMF name, such as co-imf, imf, co-imf-re
    
    # Name the dataset file and change MODE like co-imf0-re0 
    global MODE
    tmp_mode = MODE # for saving 
    redecom_mode = str.lower(redecom_mode)
    if redecom_mode == 'emd': 
        redecom_file_name,MODE = 're','emd'# co-imf0-re0 
    elif redecom_mode == 'eemd': 
        redecom_file_name,MODE = 'ree','eemd'# co-imf0-ree0 
    elif redecom_mode == 'ceemdan': 
        redecom_file_name,MODE = 'rce','ceemdan'# co-imf0-rce0 
    elif redecom_mode == 'vmd':
        redecom_file_name = 'rv' # co-imf0-rv0 
    else: raise ValueError('redecom_mode must be emd, eemd, ceemdan, or vmd.')
    # Re-decompose and create dataset
    redecom_list.sort()
    redecom_imfs_name = '-'+redecom_file_name # new imfs name
    df_redecom = df_emd.copy(deep=True)
    ori_col_names = list(df_emd.columns) # col_names
    df_col_location = 1 # change columns location if re-decompose multiple IMFs
    for i in redecom_list:
        if not isinstance(i, int): raise ValueError('redecom_list must be a list like [0,1] or an integer like 0 or 1.')
        redecom_file_name = redecom_file_name+str(i) # file name
        print('Re-decomposition is running for %s.'%(col_name+str(i)))

        # Re-decompose (figure is saved with name)
        if redecom_mode == 'vmd': df_redecom_ans = vmd_decom(df_emd[col_name+str(i)],re_decom=True,re_imf=i,K=imfs_num,draw=draw)
        else: df_redecom_ans = emd_decom(df_emd[col_name+str(i)],trials=trials,re_decom=True,re_imf=i,draw=draw) # use emd_decom()
        
        df_redecom_ans.columns = [col_name+str(i)+redecom_imfs_name+str(x) for x in range(len(df_redecom_ans.columns))]
        
        # Abandon the original IMF and insert the re-decomposed value
        df_redecom = df_redecom.drop(col_name+str(i), axis=1, inplace=False) # delete original IMF
        df_col_location = i + df_col_location - 1
        ori_col_names.pop(df_col_location) # delete corresponding name
        df_redecom = pd.concat([df_redecom, df_redecom_ans],axis=1)
        
        # Change order for co-imf0-re0 
        ori_col_names[df_col_location:df_col_location] = df_redecom_ans.columns # List of column names in the correct order
        df_col_location = df_col_location + len(df_redecom_ans.columns) - i
        df_redecom = df_redecom.reindex(columns=ori_col_names)

    # Save data and revert MODE
    MODE =  tmp_mode # for saving 
    redecom_file_name = '_'+redecom_file_name # such as _rce0
    if file_name == '':
        print('Re-decomposition finished, check the dataset: ',PATH+file_name+MODE+FORM+redecom_file_name+'_data.csv')
        pd.DataFrame.to_csv(df_redecom,PATH+file_name+MODE+FORM+redecom_file_name+'_data.csv') # ceemdan_se233_rce0_data.csv

    return df_redecom # pd.DataFrame

# VMD # There are some problems in this module
# -------------------------------
def vmd_decom(series=None,alpha=2000,tau=0,K=5,DC=0,init=1,tol=1e-7,re_decom=True,re_imf=0,draw=True):
    # Check input
    dataset,file_name = check_dataset(series,input_form='series') # include check_vars()

    from vmdpy import VMD  
    # VMD parameters
    #alpha = 2000       # moderate bandwidth constraint  
    #tau = 0.           # noise-tolerance (no strict fidelity enforcement)  
    #K = 3              # 3 modes  
    #DC = 0             # no DC part imposed  
    #init = 1           # initialize omegas uniformly  
    #tol = 1e-7         

    # VMD 
    imfs_vmd, imfs_hat, omega = VMD(series, alpha, tau, K, DC, init, tol)  
    imfs_num = np.shape(imfs_vmd)[0]
    
    if draw:
        # Plot original data
        series_index = range(len(series))
        fig = plt.figure(figsize=(16,2*imfs_num))
        plt.subplot(1+imfs_num, 1, 1 )
        plt.plot(series_index, series, color='#0070C0') #F27F19 orange #0070C0 blue
        plt.ylabel('VMD Original data')
    
        # Plot IMFs
        for i in range(imfs_num):
            plt.subplot(1 + imfs_num,1,2 + i)
            plt.plot(series_index, imfs_vmd[i, :], color='#F27F19')
            plt.ylabel('VMD-IMF'+str(i))

        # Save figure
        fig.align_labels()
        plt.tight_layout()
        if (re_decom==False): plt.savefig(FIGURE_PATH+file_name+'VMD Result.svg', bbox_inches='tight')
        else: plt.savefig(FIGURE_PATH+'IMF'+str(re_imf)+' VMD Re-decomposition Result.svg', bbox_inches='tight')
        plt.show()
    
    # Save data
    imfs_df = pd.DataFrame(imfs_vmd.T)
    imfs_df.columns = ['imf'+str(i) for i in range(imfs_num)]
    if file_name == '':
        if (re_decom==False): 
            pd.DataFrame.to_csv(imfs_df,PATH+file_name+'vmd_data.csv')
            print('VMD finished, check the dataset: ',PATH+file_name+'vmd_data.csv')

    return imfs_df # pd.DataFrame


# 4.LSTM Model Functions
#==============================================================================================

# Model evaluation function
# -------------------------------
def evl(y_test, y_pred, scale='0 to 1'): # MSE and MAE are different on different scales
    y_test,y_pred = np.array(y_test).ravel(),np.array(y_pred).ravel()
    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)*100
    print('##################################')
    print('Model Evaluation with scale of',scale)
    print('##################################')
    print('R2:', r2)
    print('RMSE:', rmse)
    print('MAE:', mae)
    print("MAPE:",mape) # MAPE before normalization may error beacause of negative values
    return [r2,rmse,mae,mape]

# DATE_BACK functions for inputting sets
# -------------------------------
# IMPORTANT!!! it may cause some error when the input format is wrong.
# Method here is used to determine the Unified normalization, use declare_uni_method(method=METHOD) to declare.
def create_dateback(df,uni=False,ahead=1):
    # Normalize for DataFrame
    if uni and METHOD != 0 and ahead == 1: # Unified normalization
        # Check input and load dataset
        if SERIES is None: raise ValueError('SERIES is not declared. Please declare it by series=cl.declare_path(path=PATH).')
        if MODE not in ['emd','eemd','ceemdan']: raise ValueError('MODE must be emd, eemd, ceemdan if you want to try unified normalization method.')
        if not (os.path.exists(PATH+MODE+'_data.csv')): raise ImportError('Dataset %s does not exist in '%(PATH+MODE+'_data.csv'),PATH)
      
        # Load data
        df_emd = pd.read_csv(PATH+MODE+'_data.csv',header=0,index_col=0)
        # Method (1)
        print('##################################')
        if METHOD == 1:
            scalar,min0 = SERIES.max()-SERIES.min(),0 
            print('Unified normalization Method (1):')
        # Method (2)
        elif METHOD == 2:
            scalar,min0 = df_emd.max().max()-df_emd.min().min(),df_emd.min().min()
            print('Unified normalization Method (2):')
        # Method (3)
        elif METHOD == 3:
            scalar,min0 = SERIES.max()-df_emd.min().min(),df_emd.min().min()
            print('Unified normalization Method (3):')

        # Normalize
        df = (df-min0)/scalar
        scalarY = {'scalar':scalar,'min':min0}
        print(df)
        if isinstance(df, pd.DataFrame):
            trainX = df.drop('sum', axis=1, inplace=False)
            trainY = np.array(df['sum']).reshape(-1, 1)
        # Normalize for each IMF in Series
        else:
            trainY = np.array(df.values).reshape(-1, 1)
            trainX = trainY
    else:
        # Normalize without unifying
        if isinstance(df, pd.DataFrame):
            trainX = df.drop('sum', axis=1, inplace=False)
            scalarX = MinMaxScaler(feature_range=(0,1))#sklearn normalize
            trainX = scalarX.fit_transform(trainX)
            trainY = np.array(df['sum']).reshape(-1, 1)
            scalarY = MinMaxScaler(feature_range=(0,1))#sklearn normalize
            trainY = scalarY.fit_transform(trainY)
        # Normalize for each IMF in Series
        else:
            trainY = np.array(df.values).reshape(-1, 1)
            scalarY = MinMaxScaler(feature_range=(0,1))#sklearn normalize
            trainY = scalarY.fit_transform(trainY)
            trainX = trainY
    
    # Create dateback
    dataX, dataY = [], []
    ahead = ahead - 1
    for i in range(len(trainY)-DATE_BACK-ahead):
        dataX.append(np.array(trainX[i:(i+DATE_BACK)]))
        dataY.append(np.array(trainY[i+DATE_BACK+ahead]))
    return np.array(dataX),np.array(dataY),scalarY,np.array(trainX[-DATE_BACK:])

# Plot original data and forecasting data
def plot_all(lstm_type,pred_ans):
    # Check and Change
    if not isinstance(SERIES, pd.Series):
        raise ValueError('SERIES is not declared. Please declare it by series=cl.declare_path(path=PATH).')
    pred_ans = pred_ans.ravel()
    series_pred = SERIES.copy(deep=True) # copy original data
    for i in range(PERIODS):
        series_pred[-i-1] = pred_ans[-i-1]

    # Plot
    fig = plt.figure(figsize=(10,4))
    SERIES[-PERIODS*3:].plot(label= 'Original data', color='#0070C0') #F27F19 orange #0070C0 blue
    series_pred[-PERIODS:].plot(label= 'Forecasting data', color='#F27F19')
    plt.xlabel('')
    plt.title(lstm_type+' LSTM forecasting results')
    plt.legend()
    plt.savefig(FIGURE_PATH+lstm_type+' LSTM forecasting results.svg', bbox_inches='tight')
    plt.show()
    return 

# Declare LSTM forecasting function
# Have declared LSTM model variables at Section 0 before
# -------------------------------
def LSTM_pred(data=None,draw=True,uni=False,show_model=True,train_set=None,next_pred=False,ahead=1):
    # Divide the training and test set
    if train_set is None:
        trainX,trainY,scalarY,next_trainX = create_dateback(data,uni=uni,ahead=ahead)
    else: trainX,trainY,scalarY,next_trainX = train_set[0],train_set[1],train_set[2],train_set[3]
    if uni==True and next_pred==True: raise ValueError('Next pred does not support unified normalization.')

    if PERIODS == 0:
        train_X = trainX
        y_train = trainY
    else:
        x_train,x_test = trainX[:-PERIODS],trainX[-PERIODS:]
        y_train,y_test = trainY[:-PERIODS],trainY[-PERIODS:]
        # Convert to tensor 
        train_X = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
        test_X = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    # Build and train the model
    # print('trainX:\n',train_X[-1:])
    print('\nInput Shape: (%d,%d)\n'%(train_X.shape[1],train_X.shape[2]))
    model = LSTM_model(train_X.shape)
    if show_model: model.summary() # The summary of layers and parameters
    EarlyStop = EarlyStopping(monitor='val_loss',patience=5*PATIENCE,verbose=VERBOSE, mode='auto') # realy stop at small learning rate
    Reduce = ReduceLROnPlateau(monitor='val_loss',patience=PATIENCE,verbose=VERBOSE,mode='auto') # Adaptive learning rate
    history = model.fit(train_X, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT,
                        verbose=VERBOSE, shuffle=SHUFFLE, callbacks=[EarlyStop,Reduce])

    # Plot the model structure
    #plot_model(model,to_file=FIGURE_PATH+'model.png',show_shapes=True)

    # Predict
    if PERIODS != 0:
        pred_test = model.predict(test_X)
        # Evaluate model with scale 0 to 1
        evl(y_test, pred_test) 
    else: pred_test = np.array([])

    if next_pred:# predict tomorrow not in test set
        next_ans = model.predict(next_trainX.reshape((1, trainX.shape[1], trainX.shape[2])))
        pred_test = np.append(pred_test,next_ans)
    pred_test = pred_test.ravel().reshape(-1,1)

    # De-normalize 
    # IMPORTANT!!! It may produce some negative data impact evaluating
    if isinstance(scalarY, MinMaxScaler):
        test_pred = scalarY.inverse_transform(pred_test)
        if PERIODS != 0: test_y = scalarY.inverse_transform(y_test)
    else:     
        test_pred = pred_test*scalarY['scalar']+scalarY['min']
        if PERIODS != 0:test_y = y_test*scalarY['scalar']+scalarY['min']
    
    # Plot 
    if draw and PERIODS != 0:
        # determing the output name of figures
        fig_name = ''
        if isinstance(data,pd.Series): 
            if str(data.name) == 'None': fig_name = 'Series'
            else: fig_name = str(data.name)
        else: fig_name = 'DataFrame'

        # Plot the loss figure
        fig = plt.figure(figsize=(5,2))
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='validation loss')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend()
        plt.title(fig_name+' LSTM loss chart')
        plt.savefig(FIGURE_PATH+fig_name+' LSTM loss chart.svg', bbox_inches='tight')
        plt.show()
        
        # Plot observation figures
        fig = plt.figure(figsize=(5,2))
        plt.plot(test_y)
        plt.plot(test_pred)
        plt.title(fig_name+' LSTM forecasting result')
        plt.savefig(FIGURE_PATH+fig_name+' LSTM forecasting result.svg', bbox_inches='tight')
        plt.show()

    return test_pred


# 5.CEEMDAN-LSTM Forecasting Functions
# Please use cl.declare_vars() to determine variables.
#==============================================================================================

# Single LSTM Forecasting without CEEMDAN
# ------------------------------- 
# It uses LSTM directly for prediction wiht input_shape=[DATE_BACK,1]
def Single_LSTM(series=None,draw=True,uni=False,show_model=True,next_pred=False,ahead=1):
    print('==============================================================================================')
    print('This is Single LSTM Forecasting running...')
    print('==============================================================================================')
    # Check input series and load dataset
    input_series,file_name = check_dataset(series,input_form='series',uni_nor=uni) # include check_vars()

    # Show the inputting data
    print('Part of Inputting dataset:')
    print(input_series)
    
    # Forecast and save result
    start = time.time()
    test_pred = LSTM_pred(data=input_series,draw=draw,uni=uni,show_model=show_model,next_pred=next_pred,ahead=ahead)
    end = time.time()
    df_pred = pd.DataFrame(test_pred)
    pd.DataFrame.to_csv(df_pred,LOG_PATH+file_name+'single_pred.csv')

    # Evaluate model 
    if draw and file_name == '': plot_all('Single',test_pred[0:PERIODS])  # plot chart to campare
    df_evl = evl(input_series[-PERIODS:].values,test_pred[0:PERIODS],scale='input series') 
    print('Running time: %.3fs'%(end-start))
    df_evl.append(end-start)
    df_evl = pd.DataFrame(df_evl).T #['R2','RMSE','MAE','MAPE','Time']
    pd.DataFrame.to_csv(df_evl,LOG_PATH+file_name+'single_log.csv',index=False,header=0,mode='a') # log record
    print('Single LSTM Forecasting finished, check the logs',LOG_PATH+file_name+'single_log.csv')
    if next_pred: 
        print('##################################')
        print('Today is',input_series[-1:].values,'but predict as',df_pred[-2:-1].values)
        print('Next day is',df_pred[-1:].values)
    if file_name != '': return df_pred

# Ensemble LSTM Forecasting with 3 Co-IMFs
# ------------------------------- 
# It uses LSTM directly for prediction wiht input_shape=[DATE_BACK,the number of features]
def Ensemble_LSTM(df=None,draw=True,uni=False,show_model=True,next_pred=False,ahead=1):
    print('==============================================================================================')
    print('This is Ensemble LSTM Forecasting running...')
    print('==============================================================================================')
    # Check input dataset and load 
    input_df,file_name = check_dataset(df,input_form='df',use_series=True,uni_nor=uni) # include check_vars()
    
    # Create ans show the inputting data set
    if file_name == '': input_df['sum'] = SERIES.values # add a column for sum data
    # if input a df, the sum of all columns will be used as the sum price to forecast
    elif 'sum' not in input_df.columns: input_df['sum'] = input_df.T.sum().values 
    print('Part of Inputting dataset:')
    print(input_df)

    # Forecast
    start = time.time()
    test_pred = LSTM_pred(data=input_df,draw=draw,uni=uni,show_model=show_model,next_pred=next_pred,ahead=ahead)
    end = time.time()
    df_pred = pd.DataFrame(test_pred)
    pd.DataFrame.to_csv(df_pred,LOG_PATH+file_name+'ensemble_'+MODE+FORM+'_pred.csv')
    
    # Evaluate model 
    if PERIODS != 0:
        if draw and file_name == '': plot_all('Ensemble',test_pred[0:PERIODS])  # plot chart to campare
        df_evl = evl(input_df['sum'][-PERIODS:].values,test_pred[0:PERIODS],scale='input df') 
        print('Running time: %.3fs'%(end-start))
        df_evl.append(end-start)
        df_evl = pd.DataFrame(df_evl).T #['R2','RMSE','MAE','MAPE','Time']
        if next_pred: 
            print('##################################')
            print('Today is',input_df['sum'][-1:].values,'but predict as',df_pred[-2:-1].values)
            print('Next day is',df_pred[-1:].values)
        pd.DataFrame.to_csv(df_evl,LOG_PATH+file_name+'ensemble_'+MODE+FORM+'_log.csv',index=False,header=0,mode='a') # log record
        print('Ensemble LSTM Forecasting finished, check the logs',LOG_PATH+file_name+'ensemble_'+MODE+FORM+'_log.csv')
    return df_pred

# Respective LSTM Forecasting for each Co-IMF
# ------------------------------- 
# It uses LSTM to predict each IMFs respectively input_shape=[DATE_BACK,1]
def Respective_LSTM(df=None,draw=True,uni=False,show_model=True,next_pred=False,ahead=1):
    print('==============================================================================================')
    print('This is Respective LSTM Forecasting running...')
    print('==============================================================================================')
    # Check input dataset and load 
    input_df,file_name = check_dataset(df,input_form='df',use_series=True,uni_nor=uni) # include check_vars()
    data_pred = [] # list for saving results of each Co-IMF
    print('Part of Inputting dataset:')
    print(input_df)
    
    # Forecast
    start = time.time()
    if MODE[-3:]=='_se': col_name = 'co-imf'
    else: col_name = 'imf'
    df_len = len(input_df.columns)
    if 'sum' in input_df.columns: df_len = df_len - 1
    for i in range(df_len):
        print('==============================================================================================')
        print(str.upper(MODE)+'--IMF'+str(i))
        print('==============================================================================================')
        test_pred = LSTM_pred(data=input_df[col_name+str(i)],draw=draw,uni=uni,show_model=show_model,next_pred=next_pred,ahead=ahead)
        data_pred.append(test_pred.ravel())
    end = time.time()

    # Save the forecasting result
    df_pred = pd.DataFrame(data_pred).T
    df_pred.columns = [col_name+str(i) for i in range(len(df_pred.columns))]
    pd.DataFrame.to_csv(df_pred,LOG_PATH+file_name+'respective_'+MODE+FORM+'_pred.csv')

    # Evaluate model 
    if PERIODS != 0:
        res_pred = df_pred.T.sum()
        if draw and file_name == '': plot_all('Respective',res_pred[:PERIODS])  # plot chart to campare
        if file_name == '': input_df['sum'] = SERIES.values # add a column for sum data
        elif 'sum' not in input_df.columns: input_df['sum'] = input_df.T.sum().values 
        df_evl = evl(input_df['sum'][-PERIODS:].values,res_pred[:PERIODS],scale='input df') 
        print('Running time: %.3fs'%(end-start))
        df_evl.append(end-start)
        df_evl = pd.DataFrame(df_evl).T #['R2','RMSE','MAE','MAPE','Time']
        if next_pred: 
            print('##################################')
            print('Today is',input_df['sum'][-1:].values,'but predict as',res_pred[-2:-1].values)
            print('Next day is',res_pred[-1:].values)
        pd.DataFrame.to_csv(df_evl,LOG_PATH+file_name+'respective_'+MODE+FORM+'_log.csv',index=False,header=0,mode='a') # log record
        print('Respective LSTM Forecasting finished, check the logs',LOG_PATH+file_name+'respective_'+MODE+FORM+'_log.csv')
    return df_pred

# Multiple predictions 
# ------------------------------- 
# Each Multi_pred() takes long time to run around 1000s unless setting the EPOCHS and n.
class HiddenPrints: # used to hide the print
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def Multi_pred(df=None,run_times=10,uni_nor=False,single_lstm=False,ensemble_lstm=False,respective_lstm=False,hybrid_lstm=False,redecom=None,ahead=1):
    print('Multiple predictions of '+str.upper(MODE)+FORM+' is running...')
    input_df,file_name = check_dataset(df,input_form='df',use_series=True,uni_nor=uni_nor) # include check_vars()
    if file_name == '': input_series = None
    else: input_series = input_df.T.sum()
    start = time.time()
    with HiddenPrints():
        for i in range(run_times):
            if single_lstm: Single_LSTM(series=input_series,draw=False,uni=uni_nor,ahead=ahead)
            if ensemble_lstm: Ensemble_LSTM(df=df,draw=False,uni=uni_nor,ahead=ahead)
            if respective_lstm: Respective_LSTM(df=df,draw=False,uni=uni_nor,ahead=ahead)
            if hybrid_lstm: Hybrid_LSTM(df=df,draw=False,redecom=redecom,ahead=ahead)
    end = time.time()
    print('Multiple predictions completed, taking %.3fs'%(end-start))
    print('Please check the logs in: '+LOG_PATH)


# 6.Hybrid Forecasting Functions
# Please use cl.declare_vars() to determine variables.
#==============================================================================================

def Hybrid_LSTM(df=None,draw=True,enlarge=10,redecom=None,next_pred=False,ahead=1):
    print('==============================================================================================')
    print('This is Hybrid LSTM Forecasting running...')
    print('==============================================================================================')
    # Check input dataset and load 
    input_df,file_name = check_dataset(df,input_form='df',use_series=True) # include check_vars()
    print('Part of Inputting dataset:')
    print(input_df)

    # Respective method first
    start = time.time()
    with HiddenPrints():
        df_res_pred = Respective_LSTM(df=input_df,draw=False,next_pred=next_pred,show_model=False,ahead=ahead) # include checking

    # initialize some variables
    global EPOCHS,PATIENCE,FORM
    EPOCHS,PATIENCE = EPOCHS*enlarge,PATIENCE*enlarge
    if 'co-imf0' in df_res_pred.columns: col_name = 'co-imf'
    else: col_name = 'imf'
    next_trainX_row = df_res_pred[-1:].copy(deep=True)
    if PERIODS != 0: df_res_pred = df_res_pred[:PERIODS]

    # Load result of the respective method
    if file_name == '':
        df_emd = pd.read_csv(PATH+MODE+FORM+'_data.csv',header=0,index_col=0)
        input_df['sum'] = SERIES.values # add a column for sum data
    elif 'sum' not in input_df.columns: 
        df_emd = input_df.copy(deep=True)
        input_df['sum'] = input_df.T.sum().values 
    else: df_emd = input_df[input_df.columns.difference(['sum'])].copy(deep=True)

    # Show respective method result
    print('\nRespective method result:')
    res_pred = df_res_pred.T.sum().values
    if PERIODS != 0: df_evl = evl(input_df['sum'][-PERIODS:].values,res_pred,scale='input df') 
    else: print('Tomorrow is',res_pred,'of '+FORM)
    print('Hybrid LSTM Forecasting is still running...')

    # VMD-Ensemble LSTM predict for IMF0 or Co-IMF0
    redecom_name = ''
    if redecom is not None: # only re-decompse IMF0 or Co-IMF0
        redecom_name = '_'+redecom
        with HiddenPrints():
            df_redecom = re_decom(df=input_df,redecom_mode=redecom,redecom_list=0,draw=False,imfs_num=10)
            vmd_input = df_redecom[[col_name+'0-rv'+str(i) for i in range(10)]]
            vmd_input['sum'] = df_emd[col_name+'0']
            df_vmd = Ensemble_LSTM(df=vmd_input,draw=False,next_pred=next_pred,show_model=False,ahead=ahead)
        next_trainX_row[col_name+'0'] = df_vmd[-1:].values
        print('\nVMD method result:')
        if PERIODS != 0:
            df_res_pred[col_name+'0'] = df_vmd[0:PERIODS].values
            res_vmd_pred = df_res_pred.T.sum().values
            #df_evl = evl(df_emd['imf0'][-PERIODS:],df_vmd,scale='IMF0') # result of IMF0
            df_evl_vmd = evl(input_df['sum'][-PERIODS:].values,res_vmd_pred,scale='input df') # result of overall
            end_vmd = time.time()
            print('Running time: %.3fs'%(end_vmd-start))
            df_evl_vmd.append(end_vmd-start)
            df_evl_vmd = pd.DataFrame(df_evl_vmd).T #['R2','RMSE','MAE','MAPE','Time']
            pd.DataFrame.to_csv(df_evl_vmd,LOG_PATH+file_name+'respective_'+MODE+FORM+redecom_name+'_log.csv',index=False,header=0,mode='a') # log record
            df_res_add = df_res_pred.copy(deep=True)
            df_res_add.append(next_trainX_row,ignore_index=True)
            df_res_add = df_res_add.sum(axis=1)
            pd.DataFrame.to_csv(df_res_add,LOG_PATH+file_name+'respective_'+MODE+FORM+redecom_name+'_pred.csv') # pred record
        else:  print('Tomorrow is ',next_trainX_row.T.sum().values,' of '+FORM)
        print('Hybrid LSTM Forecasting is still running...')

    # Normalize
    df_res_pred.columns = df_emd.columns
    rate = df_emd.max()-df_emd.min()
    tmp_emd = (df_emd-df_emd.min())/rate 
    tmp_pred = (df_res_pred-df_emd.min())/rate

    # Split and create trainX trainY
    trainY = input_df['sum'].values.reshape(-1,1)
    scalarY = MinMaxScaler(feature_range=(0,1)) #sklearn normalize
    trainY = scalarY.fit_transform(trainY)
    trainX = tmp_emd.copy(deep=True)

    dataX, dataY = [], []
    l = len(trainY)-DATE_BACK-PERIODS
    for i in range(len(trainY)-DATE_BACK):
        x = np.array(trainX[i:(i+DATE_BACK)])
        if i < l:
            a = tmp_emd.values[i+DATE_BACK]
        else: a = tmp_pred.values[i-l]
        x = np.row_stack((x,a))
        dataX.append(x)
        dataY.append(np.array(trainY[i+DATE_BACK]))
    next_trainX_row = (next_trainX_row-df_emd.min())/rate
    next_trainX = np.row_stack((trainX[-DATE_BACK:],next_trainX_row))
    #if next_pred: print('\nnext_trainX:',next_trainX)

    # Ensemble method 
    test_pred = LSTM_pred(data=None,show_model=False,draw=draw,ahead=ahead,train_set=[np.array(dataX),np.array(dataY),scalarY,next_trainX],next_pred=next_pred)
    end = time.time()
    EPOCHS,PATIENCE = int(EPOCHS/enlarge),int(PATIENCE/enlarge)
    df_pred = pd.DataFrame(test_pred)
    if PERIODS == 0: pd.DataFrame.to_csv(df_pred,LOG_PATH+FORM+redecom_name+'_next_pred.csv',mode='a')
    else:
        # Evaluate model 
        pd.DataFrame.to_csv(df_pred,LOG_PATH+file_name+'hybrid_'+MODE+FORM+redecom_name+'_pred.csv')
        if draw and file_name == '': plot_all('Hybrid',test_pred[0:PERIODS])  # plot chart to campare
        df_evl = evl(input_df['sum'][-PERIODS:].values,test_pred[0:PERIODS],scale='input df') 
        print('Running time: %.3fs'%(end-start))
        df_evl.append(end-start)
        df_evl = pd.DataFrame(df_evl).T #['R2','RMSE','MAE','MAPE','Time']
        pd.DataFrame.to_csv(df_evl,LOG_PATH+file_name+'hybrid_'+MODE+FORM+redecom_name+'_log.csv',index=False,header=0,mode='a') # log record
        print('Hybrid LSTM Forecasting finished, check the logs',LOG_PATH+file_name+'hybrid_'+MODE+FORM+redecom_name+'_log.csv')
    if next_pred: 
        print('Running time: %.3fs'%(end-start))
        print('##################################')
        print('Today is',input_df['sum'][-1:].values,'but predict as',df_pred[-2:-1].values)
        print('Next day is',df_pred[-1:].values)
    return df_pred


# 7.Statistical Tests
# Please use cl.declare_vars() to determine variables.
#==============================================================================================
def statistical_tests(series=None): # total version
    input_series,file_name = check_dataset(series,input_form='series',use_series=True) # include check_vars()
    adf_test(input_series)
    print()
    LB_test(input_series)
    print()
    JB_test(input_series)
    print()
    plot_acf_pacf(input_series)

# Augmented Dickey-Fuller test (ADF test) for stationarity
# ------------------------------- 
def adf_test(series=None):
    if series is None: raise ValueError('This is no proper input.')
    adf_ans = adfuller(series) # The outcomes are test value, p-value, lags, degree of freedom.
    print('##################################')
    print('ADF Test')
    print('##################################')
    print('Test value:',adf_ans[0])
    print('P value:',adf_ans[1])
    print('Lags:',adf_ans[2])
    print('1% confidence interval:',adf_ans[4]['1%'])
    print('5% confidence interval:',adf_ans[4]['5%'])
    print('10% confidence interval:',adf_ans[4]['10%'])
    #print(adf_ans) 
    
    # Brief review
    adf_status = ''
    if adf_ans[0]<=adf_ans[4]['1%']: adf_status = 'very strong'
    elif adf_ans[0]<=adf_ans[4]['5%']: adf_status = 'strong'
    elif adf_ans[0]<=adf_ans[4]['10%']: adf_status = 'normal'
    else: adf_status = 'no'
    print('The p-value is '+str(adf_ans[1])+', so the series has '+str(adf_status)+' stationarity.')
    print('The automatic selecting lags is '+str(adf_ans[2])+', advising the past '+str(adf_ans[2])+' days as the features.')

# Ljung-Box Test for autocorrelation
# ------------------------------- 
def LB_test(series=None):
    if series is None: raise ValueError('This is no proper input.')
    lb_ans = lb_test(series,lags=None,boxpierce=False) #The default lags=40 for long series.
    print('##################################')
    print('Ljung-Box Test')
    print('##################################')

    # Plot p-values in a figure
    fig = plt.figure(figsize=(10,3))
    pd.Series(lb_ans[1]).plot(label="Ljung-Box Test p-values")
    plt.xlabel('Lag')
    plt.legend()
    plt.show()
    
    # Brief review
    if np.sum(lb_ans[1])<=0.05: 
        print('The sum of p-value is '+str(np.sum(lb_ans[1]))+'<=0.05, rejecting the null hypothesis that the series has very strong autocorrelation.')
    else: print('Please view with the line chart, the autocorrelation of the series may be not strong.')
    
    # Show the outcome
    # print(pd.DataFrame(lb_ans)) # The outcomes are test value at line 0, and p-value at line 1.

# Jarque-Bera Test for normality
# ------------------------------- 
def JB_test(series=None):
    if series is None: raise ValueError('This is no proper input.')
    jb_ans = jb_test(series) # The outcomes are test value, p-value, skewness and kurtosis.
    print('##################################')
    print('Jarque-Bera Test')
    print('##################################')
    print('Test value:',jb_ans[0])
    print('P value:',jb_ans[1])
    print('Skewness:',jb_ans[2])
    print('Kurtosis:',jb_ans[3])

    # Brief review
    if jb_ans[1]<=0.05: 
        print('p-value is '+str(jb_ans[1])+'<=0.05, rejecting the null hypothesis that the series has no normality.')
    else:
        print('p-value is '+str(jb_ans[1])+'>=0.05, accepting the null hypothesis that the series has certain normality.')

# Plot ACF and PACF figures
# ------------------------------- 
def plot_acf_pacf(series=None):
    print('##################################')
    print('ACF and PACF')
    print('##################################')
    if series is None: raise ValueError('This is no proper input.')
    fig = plt.figure(figsize=(10,5))
    fig1 = fig.add_subplot(211)
    plot_acf(series, lags=40, ax=fig1)
    fig2 = fig.add_subplot(212)
    plot_pacf(series, lags=40, ax=fig2)

    #Save the figure
    plt.savefig(FIGURE_PATH+'ACF and PACF of Series.svg', bbox_inches='tight')
    plt.tight_layout() 
    plt.show()

# 8.Forecasting of SVR LASSO BPNN 
# Please use cl.declare_vars() to determine variables.
#==============================================================================================

import math
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score,GridSearchCV

# SVR
def SVR_pred(data=None,gstimes=5,draw=True,ahead=1):
    # Divide the training and test set
    start = time.time()
    trainX,trainY,scalarY,next_trainX = create_dateback(data,ahead=ahead)
    trainX = trainX.reshape((trainX.shape[0], trainX.shape[1]))
    x_train,x_test = trainX[:-PERIODS],trainX[-PERIODS:]
    y_train,y_test = trainY[:-PERIODS],trainY[-PERIODS:]

    # Grid Search of K-Fold CV
    # logspace(a,b,N)Divide the interval from the a power of 10 to the b power of 10 into N parts
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)
    best_gamma,best_C = 0,0
    for i in range(gstimes):
        param_grid = dict(gamma=gamma_range, C=C_range)
        grid = GridSearchCV(SVR(), param_grid=param_grid, cv=10)
        grid.fit(x_train, y_train)
        print('Iteration',i)
        print('Best parameters:', grid.best_params_)
        if best_gamma == grid.best_params_['gamma'] and best_C == grid.best_params_['C']: break
        best_gamma=grid.best_params_['gamma']
        best_C=grid.best_params_['C']
        gamma_range = np.append(np.linspace(best_gamma/10,best_gamma*0.9,9),np.linspace(best_gamma,best_gamma*10,10)).ravel()
        C_range = np.append(np.linspace(best_C/10,best_C*0.9,9),np.linspace(best_C,best_C*10,10)).ravel()

    # Predict
    clf = SVR(kernel='rbf', gamma=best_gamma ,C=best_C)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    end = time.time()

    # De-normalize and Evaluate
    test_pred = scalarY.inverse_transform(y_pred.reshape(y_pred.shape[0],1))
    test_y = scalarY.inverse_transform(y_test)
    evl(test_pred, test_y)
    print('Running time: %.3fs'%(end-start))

    # Plot observation figures
    if draw:
        fig = plt.figure(figsize=(5,2))
        plt.plot(test_y)
        plt.plot(test_pred)
        plt.title('SVR forecasting result')
        #plt.savefig(FIGURE_PATH+fig_name+' LSTM forecasting result.svg', bbox_inches='tight')
        plt.show()

# DM test # Author: John Tsang
def dm_test(actual_lst, pred1_lst, pred2_lst, h = 1, crit="MSE", power = 2):
    # Routine for checking errors
    def error_check():
        rt = 0
        msg = ""
        # Check if h is an integer
        if (not isinstance(h, int)):
            rt = -1
            msg = "The type of the number of steps ahead (h) is not an integer."
            return (rt,msg)
        # Check the range of h
        if (h < 1):
            rt = -1
            msg = "The number of steps ahead (h) is not large enough."
            return (rt,msg)
        len_act = len(actual_lst)
        len_p1  = len(pred1_lst)
        len_p2  = len(pred2_lst)
        # Check if lengths of actual values and predicted values are equal
        if (len_act != len_p1 or len_p1 != len_p2 or len_act != len_p2):
            rt = -1
            msg = "Lengths of actual_lst, pred1_lst and pred2_lst do not match."
            return (rt,msg)
        # Check range of h
        if (h >= len_act):
            rt = -1
            msg = "The number of steps ahead is too large."
            return (rt,msg)
        # Check if criterion supported
        if (crit != "MSE" and crit != "MAPE" and crit != "MAD" and crit != "poly"):
            rt = -1
            msg = "The criterion is not supported."
            return (rt,msg)  
        # Check if every value of the input lists are numerical values
        from re import compile as re_compile
        comp = re_compile("^\d+?\.\d+?$")  
        def compiled_regex(s):
            """ Returns True is string is a number. """
            if comp.match(s) is None:
                return s.isdigit()
            return True
        for actual, pred1, pred2 in zip(actual_lst, pred1_lst, pred2_lst):
            is_actual_ok = compiled_regex(str(abs(actual)))
            is_pred1_ok = compiled_regex(str(abs(pred1)))
            is_pred2_ok = compiled_regex(str(abs(pred2)))
            if (not (is_actual_ok and is_pred1_ok and is_pred2_ok)):  
                msg = "An element in the actual_lst, pred1_lst or pred2_lst is not numeric."
                rt = -1
                return (rt,msg)
        return (rt,msg)
    
    # Error check
    error_code = error_check()
    # Raise error if cannot pass error check
    if (error_code[0] == -1):
        raise SyntaxError(error_code[1])
        return
    # Import libraries
    from scipy.stats import t
    import collections
    
    # Initialise lists
    e1_lst,e2_lst,d_lst = [],[],[]
    
    # convert every value of the lists into real values
    actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
    pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
    pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()
    
    # Length of lists (as real numbers)
    T = float(len(actual_lst))
    
    # construct d according to crit
    if (crit == "MSE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append((actual - p1)**2)
            e2_lst.append((actual - p2)**2)
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAD"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs(actual - p1))
            e2_lst.append(abs(actual - p2))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAPE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs((actual - p1)/actual))
            e2_lst.append(abs((actual - p2)/actual))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "poly"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(((actual - p1))**(power))
            e2_lst.append(((actual - p2))**(power))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)    
    
    # Mean of d        
    mean_d = pd.Series(d_lst).mean()
    
    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
              autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        return (1/(T))*autoCov
    gamma = []
    for lag in range(0,h):
        gamma.append(autocovariance(d_lst,len(d_lst),lag,mean_d)) # 0, 1, 2
    V_d = (gamma[0] + 2*sum(gamma[1:]))/T
    DM_stat=V_d**(-0.5)*mean_d
    harvey_adj=((T+1-2*h+h*(h-1)/T)/T)**(0.5)
    DM_stat = harvey_adj*DM_stat
    # Find p-value
    p_value = 2*t.cdf(-abs(DM_stat), df = T - 1)
    
    # Construct named tuple for return
    dm_return = collections.namedtuple('dm_return', 'DM p_value')
    rt = dm_return(DM = DM_stat, p_value = p_value)
    return rt

# Appendix if main run example
#==============================================================================================
print('Finished. Use cl.guideline() to call for a guideline or cl.example() for an exmple.')
if __name__ == '__main__':
    run_example()
