﻿#!/usr/bin/env python
# coding: utf-8
#
# Module: core
# Description: Some basic function was compiled here.
#
# Created: 2021-10-1 22:37
# Updated: 2022-9-12 17:28
# Author: Feite Zhou
# Email: jupiterzhou@foxmail.com
# URL: 'http://github.com/FateMurphy/CEEMDAN_LSTM'
# Feel free to email me if you have any questions or error reports.

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


# 0.Guideline
# ------------------------------------------------------
def help():
    print('CEEMDAN_LSTM is a module that helps you make a quick decomposition-integration time series forecasting.')
    print('===============================')
    print('Available Functions')
    print('===============================')
    print('# 0.Quick Forecasting: ')
    print("-------------------------------")
    print("cl.quick_keras_predict(data=None, **kwargs)")
    print("cl.details_keras_predict(data=None, fitting=False, **kwargs)")
    print()
    print('# 1.Guideline functions: ')
    print("-------------------------------")
    print("cl.help()")
    print("cl.show_keras_example()")
    print("cl.show_keras_example_model()")
    print("cl.show_devices()")
    print("cl.load_dataset(dataset_name='sse_index.csv')")
    print("cl.check_dataset(data, show_data=False, decom_mode=None, plot_heatmap=False)")
    print("cl.check_path(PATH)")
    print("cl.plot_save_result(data, name=None, plot=True, save=True, path=None, type=None)")
    print()
    print('# 2.Data preprocessor: ')
    print("-------------------------------")
    print("cl.decom(series=None, decom_mode='ceemdan', **kwargs)")
    print("cl.decom_vmd(series=None, alpha=2000, tau=0, K=10, DC=0, init=1, tol=1e-7, **kwargs)")
    print("cl.inte(df_decom=None, inte_list=None, num_clusters=3)")
    print("cl.inte_sampen(df_decom=None, max_len=1, tol=0.1, nor=True, **kwargs)")
    print("cl.inte_kmeans(df_sampen=None, num_clusters=3, random_state=0, **kwargs)")
    print("cl.redecom(data=None, show_data=False, decom_mode='ceemdan', inte_list=None, redecom_list={'co-imf0':'vmd'}, **kwargs)")
    print("cl.eval_result(y_real, y_pred)")
    print("cl.normalize_dataset(data=None, FORECAST_LENGTH=None, NOR_METHOD='MinMax')")
    print("cl.create_train_test_set(data=None, FORECAST_LENGTH=None, FORECAST_HORIZONS=None, NOR_METHOD='MinMax', DAY_AHEAD=1, fitting_set=None)")
    print()
    print('# 2.X Data preprocessor for Statistical tests: ')
    print("-------------------------------")    
    print("cl.statis_tests(series=None)")
    print("cl.adf_test(series=None)")
    print("cl.lb_test(series=None)")
    print("cl.jb_test(series=None)")
    print("cl.plot_acf_pacf(series=None, fig_path=None)")
    print("cl.plot_heatmap(data, corr_method='pearson', fig_path=None)")
    print("cl.dm_test(actual_lst, pred1_lst, pred2_lst, h=1, crit='MSE', power=2)")
    print()
    print("# 3.Keras predictor: ")
    print("-------------------------------")
    print("IMPORTANT! Please change parameters via kr = cl.keras_predictor().")
    print("This class is still under compilation.")
    print("cl.keras_predict(self, data=None, show_model=False, fitting_set=None, **kwargs)")
    print("cl.single_keras_predict(self, data=None, show=False, plot=False, save=False, **kwargs)")
    print("cl.ensemble_keras_predict(self, data=None, show=False, plot=False, save=False, **kwargs)")
    print("cl.respective_keras_predict(self, data=None, show=False, plot=False, save=False, **kwargs)")
    print("cl.hybrid_keras_predict(self, data=None, show=False, plot=False, save=False, **kwargs)")
    print("cl.multiple_keras_predict(self, data=None, run_times=10, predict_method=None, **kwargs)")
    print("cl.rolling_keras_predict(self, data=None, predict_method=None, **kwargs)")
    print()
    print("# 4.Sklearn predictor: ")
    print("-------------------------------")
    print("This class is still under compilation.")
    print("IMPORTANT! Please change parameters via sr = cl.sklearn_predictor().")
    print("cl.svm_predict(self, data=None, grid_search_times=5, show_data=False, plot_result=False, save_result=False)")


# An keras example
def show_keras_example():
    print("Start your first prediction by following steps:")
    print("Use cl.details_keras_predict(data) to run following steps.")
    print("Or directly copy following code to forecast.")
    print("===============================")
    print("\nprint('\\n0.Initialize')")
    print("print('-------------------------------')")
    print("import CEEMDAN_LSTM as cl")
    print("import matplotlib.pyplot as plt")
    print("kr = cl.keras_predictor(epochs=10)")
    print("\nprint('\\n1.Load raw data')")
    print("print('-------------------------------')")
    print("dataset = cl.load_dataset()")
    print("data = pd.Series(dataset['close'].values, index=dataset.index)")
    print("data.plot(title='Original Data') # plot")
    print("plt.show() # plot")
    print("\nprint('\\n2.CEEMDAN decompose')")
    print("print('-------------------------------')")
    print("df_ceemdan = cl.decom(data)")
    print("df_ceemdan.plot(title='CEEMDAN Decomposition', subplots=True, figsize=(6, 1*(df_ceemdan.columns.size))) # plot")
    print("plt.show() # plot")
    print("\nprint('\\n3.Sample Entropy Calculate')")
    print("print('-------------------------------')")
    print("df_sampen = cl.inte_sampen(df_ceemdan)")
    print("df_sampen.plot(title='Sample Entropy') # plot")
    print("plt.show() # plot")
    print("\nprint('\\n4.K-Means Cluster by Sample Entropy')")
    print("print('-------------------------------')")
    print("df_integrate_form = cl.inte_kmeans(df_sampen)")
    print("print(df_integrate_form) # show")
    print("\nprint('\\n5.Integrate IMFs and Residue to be 3 Co-IMFs')")
    print("print('-------------------------------')")
    print("df_integrate_result = cl.inte(df_ceemdan, df_integrate_form)")
    print("df_integrate_result = df_integrate_result[0]")
    print("df_integrate_result.plot(title='Integrated IMFs (Co-IMFs) of CEEMDAN', subplots=True, figsize=(6,3)) # plot")
    print("plt.show() # plot")
    print("\nprint('\\n6.Secondary Decompose the high-frequency Co-IMF0 by OVMD')")
    print("print('-------------------------------')")
    print("df_vmd_co_imf0 = cl.decom(df_integrate_result['co-imf0'], decom_mode='ovmd')")
    print("df_vmd_co_imf0.plot(title='OVMD Decomposition of Co-IMF0', subplots=True, figsize=(6,11)) # plot")
    print("plt.show() # plot")
    print("\nprint('\\n7.Predict Co-IMF0 by matrix-input GRU (ensemble method)')")
    print("print('-------------------------------')")
    print("co_imf0_predict_raw, co_imf0_gru_evaluation, co_imf0_train_loss = kr.keras_predict(df_vmd_co_imf0)")
    print("print('======Co-IMF0 Predicting Finished======\\n', co_imf0_gru_evaluation) # show")
    print("co_imf0_predict_raw.plot(title='Co-IMF0 Predicting Result') # plot")
    print("co_imf0_train_loss.plot(title='Co-IMF0 Training Loss') # plot")
    print("plt.show() # plot")
    print("\nprint('\\n8.Predict Co-IMF1 and Co-IMF2 by vector-input GRU (respective method)')")
    print("print('-------------------------------')")
    print("co_imf1_predict_raw, co_imf1_gru_evaluation, co_imf1_train_loss = kr.keras_predict(df_integrate_result['co-imf1'])")
    print("print('======Co-IMF1 Predicting Finished======\\n', co_imf1_gru_evaluation) # show")
    print("co_imf1_predict_raw.plot(title='Co-IMF1 Predicting Result') # plot")
    print("co_imf1_train_loss.plot(title='Co-IMF1 Training Loss') # plot")
    print("plt.show() # plot")
    print("co_imf2_predict_raw, co_imf2_gru_evaluation, co_imf2_train_loss = kr.keras_predict(df_integrate_result['co-imf2'])")
    print("print('======Co-IMF2 Predicting Finished======\\n', co_imf2_gru_evaluation) # show")
    print("co_imf2_predict_raw.plot(title='Co-IMF2 Predicting Result') # plot")
    print("co_imf2_train_loss.plot(title='Co-IMF2 Training Loss') # plot")
    print("plt.show() # plot")
    print("\nprint('\\n9. Add 3 result to get the final forecasting result')")
    print("print('-------------------------------')")
    print("forecast_length = len(co_imf0_predict_raw)")
    print("series_add_predict_result = co_imf0_predict_raw['predict']+co_imf1_predict_raw['predict']+co_imf2_predict_raw['predict']")
    print("df_add_predict_raw = pd.DataFrame({'predict': series_add_predict_result.values, 'raw': data[-forecast_length:].values}, index=range(forecast_length))")
    print("df_add_evaluation = cl.eval_result(data[-forecast_length:],series_add_predict_result)")
    print("print('======Hybrid CEEMDAN-VMD-GRU Keras Forecasting Finished======\\n', df_add_evaluation) # show")
    print("df_add_predict_raw.plot(title='Hybrid CEEMDAN-VMD-GRU Keras Forecasting Result') # plot")
    print("plt.show() # plot")
    print("# pd.DataFrame.to_csv(df_add_predict_raw, PATH+'_predict_output.csv')")

# LSTM model example
def show_keras_example_model():
    print('Please input a Keras model with input_shape = (FORECAST_HORIZONS, the number of features)')
    print("===============================")
    print("model = Sequential()")
    print("model.add(LSTM(100, input_shape=(30, 1), activation='tanh'))")
    print("model.add(Dropout(0.5))")
    print("model.add(Dense(1,activation='tanh'))")
    print("model.compile(loss='mse', optimizer='adam')")
    print("kr = cl.keras_predictor(KERAS_MODEL=model)")

# Show Tensorflow running device
def show_devices():
    try:
        try:
            import tensorflow as tf # for tensorflow and keras
            print("Tensorflow version:", tf.__version__)
            print("Available devices:") 
            print(tf.config.list_physical_devices())
        except: 
            import torch # if install Pytorch
            print("Pytorch version:", torch.__version__)
            print("CUDA:", torch.cuda.is_available(), "version:", torch.version.cuda, ) # 查看CUDA的版本号
            print("GPU:", torch.cuda.get_device_name())
    except: raise ImportError('Please install Tensorflow or Pytorch firstly!')

# Load the example data set
def load_dataset(dataset_name='sse_index.csv'):
    """
    Load dataset eg. cl.load_dataset(dataset_name='sse_index.csv')
    Available dataset 'sse_index.csv';
    Input:
    ---------------------
    dataset_name   - the datasets in this module check them in '/datasets/'
    
    Output:
    ---------------------
    df_raw_data    - pd.DataFrame or pd.Series
    """
    dataset_location = os.path.dirname(os.path.realpath(__file__)) + '/datasets/'
    df_raw_data = pd.read_csv(dataset_location+dataset_name, header=0, index_col=['date'], parse_dates=['date'])
    return df_raw_data

# Check dataset
def check_dataset(data, show_data=False, decom_mode=None, redecom_list=None):
    """
    Check dataset and change data to pd.DataFrame or pd.Series
    Example: output = cl.check_dataset(data)

    Input:
    ---------------------
    data       - the original data 
    
    Output:
    ---------------------
    check_data - pd.DataFrame or pd.Series
    """

    # Check Input
    if data is None: raise ValueError('Please input data!')
    try: check_data = pd.DataFrame(data)
    except: raise ValueError('Invalid input of dataset %s!'%type(data))
    if decom_mode is None: decom_mode = ''
    if redecom_list is None: redecom_list = {}
    if pd.isnull(check_data.values).any(): raise ValueError('Please check inputs! There is NaN!')
    if not check_data.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).all(): 
        raise ValueError('Please check inputs! Cannot convert it to number.')

    # As vmdpy has some error and cannot decomose odd-number series, delete one data point here
    for x in redecom_list.values(): 
        if 'vmd' in x.lower() and len(check_data)%2: check_data = check_data.sort_index()[1:] 
    if 'vmd' in decom_mode.lower() and len(check_data)%2: check_data = check_data.sort_index()[1:] 

    # Set target
    if 'target' not in check_data.columns:
        if len(check_data.columns) == 1: check_data.columns=['target']
        elif decom_mode.lower() in ['emd', 'eemd', 'ceemdan']: 
            check_data['target'] = check_data.sum(axis=1).values
            print('Warning! The sum of all column has been set as the target column.') 
        else:
            check_data.columns = check_data.columns[1:].insert(0, 'target') # set first columns as target 
            print('Warning! The first column has been set as the target column.')  

    # Show the inputting data
    if show_data:
        print('Data type is %s.'%type(check_data))
        print('Part of inputting dataset:')
        print(check_data)
        # print('\nData description:')
        # print(check_data.describe())
    return check_data

# Check PATH
def check_path(PATH):
    """
    Check PATH.
    Example: PATH, FIG_PATH, LOG_PATH = cl.check_path(PATH)

    Input:
    ---------------------
    PATH      - the saving path

    Output:
    ---------------------
    PATH      - the saving path
    FIG_PATH  - the saving path for figures
    LOG_PATH  - the saving path for log (forecasting result)
    """

    if PATH is not None: 
        if type(PATH) != str: raise TypeError('PATH should be strings such as D:/CEEMDAN_LSTM/.../.')
        else:
            if PATH[-1] != '/': PATH = PATH + '/'
            FIG_PATH = PATH+'figure/'
            LOG_PATH = PATH+'log/' 
            # print('Saving path of figures and logs: %s'%(PATH))
            for p in [PATH, FIG_PATH, LOG_PATH]: # Create folders for saving 
                if not os.path.exists(p): os.makedirs(p)
    else: PATH, FIG_PATH, LOG_PATH = None, None, None
    return PATH, FIG_PATH, LOG_PATH

# Name the predictor
def name_predictor(now, name, module, model, decom_mode=None, redecom_list=None, next_pred=False):
    """
    Name the predictor for convenient saving.
    """

    redecom_mode = ''
    if redecom_list is not None: # Check redecom_list and get redecom_mode
        try: redecom_list = pd.DataFrame(redecom_list, index=[0]) 
        except: raise ValueError("Invalid input for redecom_list! Please input eg. None, '{'co-imf0':'vmd', 'co-imf1':'emd'}'.")
        for i in range(redecom_list.size): redecom_mode = redecom_mode+redecom_list.values.ravel()[i]+'-'

    from tensorflow.python.keras.models import Sequential
    if type(model) == str and '.h5' not in str(model):
        if 'Single' not in name:
            if decom_mode is not None: 
                name = name + ' ' + decom_mode.upper() + '-' 
                name = name + redecom_mode.upper()
        else: name = name + ' '
        name = name + model.upper()  # forecasting model
    else: name = name+' Custom Model'
    if next_pred: name = name+' Next-day' # Next-day forecasting or not
    name = name + ' ' + module+' Forecasting'
    print('==============================================================================')
    print(str(now.strftime('%Y-%m-%d %H:%M:%S'))+' '+ name +' is running...')
    print('==============================================================================')
    return name

# Change name
def change_name(s): 
    s = s.replace('-','_')
    s = s.replace(' ','_')
    return s

# Plot and save data
def plot_save_result(data, name=None, plot=True, save=True, path=None, type=None): 
    """
    Plot and save data.
    Example: cl.plot_save_result(df)

    Input and Parameters:
    ---------------------
    data         - the original data 
    plot         - plot the figure or only save data
    name         - the saving file start name
    path         - the saving path
    """
    
    PATH, FIG_PATH, LOG_PATH = check_path(path)

    # Check Name
    if name is None: 
        name = datetime.now().strftime('%Y%m%d_%H%M%S_')
    elif isinstance(name, datetime):
        name = name.strftime('%Y%m%d_%H%M%S_')
    else: name = ''
    if PATH is None: save = False

    # Default output function
    def default_output(df): 
        if 'Evaluation' not in df.name:
            if plot:
                df.plot(figsize=(8,4))
                plt.title(df.name, fontsize=16, y=1)
                if save: plt.savefig(FIG_PATH + name + change_name(df.name)+'.jpg', dpi=300, bbox_inches='tight') # Save figure
                plt.show()
        if save: pd.DataFrame.to_csv(df, LOG_PATH + name + change_name(df.name)+'.csv', encoding='utf-8') # Save log

    # Ouput
    if isinstance(data, tuple):
        for df in data: # plot and save forecasting result
            try: df.name # Check df Name
            except: df.name = 'output'
            default_output(df)
    elif isinstance(data, pd.DataFrame):
        try: data.name # Check data Name
        except: data.name = 'output'
        if 'decom' in data.name: # plot and save plot EMD result
            if plot:
                data.plot(figsize=(8,2*data.columns.size), subplots=True)
                plt.gcf().suptitle(data.name, fontsize=16, y=0.9) # Enlarge and Move the title     
                if save: plt.savefig(FIG_PATH + name + change_name(data.name)+'.jpg', dpi=300, bbox_inches='tight') # Save figure
                plt.show()
            if save: pd.DataFrame.to_csv(data, LOG_PATH + name + change_name(data.name)+'.csv', encoding='utf-8') # Save log
        else:
            try: default_output(data)
            except: print('Data is:\n', data)
    else:
        try: series = pd.Series(data)
        except: raise ValueError('Sorry! %s is not supported to plot and save, please input pd.DataFrame, pd.Series, nd.array(<=2D)'%type(data))
        default_output(series)
    if PATH is not None and save: print('The figures and logs of forecasting results have been saved into', PATH)



# Run the qucik forecasting
def quick_keras_predict(data=None, **kwargs):
    """
    Run a qucik forecasting with only input your dataset.
    """
    from CEEMDAN_LSTM.keras_predictor import keras_predictor
    if data is None: 
        print('Load the sample dataset of the SSE index. Check it by cl.load_dataset()')
        dataset = load_dataset()
        data = pd.Series(dataset['close'].values, index=dataset.index)
    try: data = pd.Series(data) # 1.Load raw data
    except: raise ValueError('Sorry! %s is not supported, please input pd.DataFrame, pd.Series, nd.array(<=2D)'%type(data))

    kr = keras_predictor(**kwargs)
    df_result = kr.hybrid_keras_predict(data=data, show=True, plot=True, save=True)
    
    return df_result


# Run the details forecasting
def details_keras_predict(data=None, fitting=False, **kwargs):
    """
    This function aims to help users know the forecasting framework more clearly, running the forecasting step by step.
    You can also refer the CEEMDAN-VMD-GRU repository: https://github.com/FateMurphy/CEEMDAN-VMD-GRU
    """
    # Decompose, integrate 
    import time
    from CEEMDAN_LSTM.data_preprocessor import decom, inte_sampen, inte_kmeans, inte, eval_result
    from CEEMDAN_LSTM.keras_predictor import keras_predictor

    # 0.Initialize
    print("\n0.Initialize")
    print("-------------------------------")
    kr = keras_predictor(**kwargs)

    # 1.Load raw data
    print("\n1.Load raw data")
    print("-------------------------------")
    if data is None: 
        print('Load the sample dataset of the SSE index. Check it by cl.load_dataset()')
        dataset = load_dataset()
        data = pd.Series(dataset['close'].values, index=dataset.index)
    data = check_dataset(data, False, 'vmd')
    data = pd.Series(data.values.ravel(), index=data.index)
    data.plot(title='Original Data')
    plt.show()

    # 2.CEEMDAN decompose
    print("\n2.CEEMDAN decompose")
    print("-------------------------------")
    start = time.time()
    df_ceemdan = decom(data, decom_mode='ceemdan')
    df_ceemdan.plot(title='CEEMDAN Decomposition', subplots=True, figsize=(6, 1*(df_ceemdan.columns.size)))
    plt.show()

    # 3.Sample Entropy Calculate
    print("\n3.Sample Entropy Calculate")
    print("-------------------------------")
    df_sampen = inte_sampen(df_ceemdan) 
    df_sampen.plot(title='Sample Entropy')
    plt.show()

    # 4.K-Means Cluster by Sample Entropy
    print("\n4.K-Means Cluster by Sample Entropy")
    print("-------------------------------")
    df_integrate_form = inte_kmeans(df_sampen)
    print(df_integrate_form)

    # 5.Integrate IMFs and Residue to be 3 Co-IMFs
    print("\n5.Integrate IMFs and Residue to be 3 Co-IMFs")
    print("-------------------------------")
    df_integrate_result = inte(df_ceemdan, df_integrate_form)
    df_integrate_result = df_integrate_result[0]
    df_integrate_result.plot(title='Integrated IMFs (Co-IMFs) of CEEMDAN', subplots=True, figsize=(6, 3))
    plt.show()

    # 6.Secondary Decompose the high-frequency Co-IMF0 by VMD (cl.redecom cna finish step 2 to 6)
    print("\n6.Secondary Decompose the high-frequency Co-IMF0 by OVMD")
    print("-------------------------------")
    df_vmd_co_imf0 = decom(df_integrate_result['co-imf0'], decom_mode='ovmd')
    df_vmd_co_imf0.plot(title='OVMD Decomposition of Co-IMF0', subplots=True, figsize=(6, 11))
    plt.show()

    # 7.Predict Co-IMF0 by matrix-input GRU (ensemble method)
    print("\n7.Predict Co-IMF0 by matrix-input GRU (ensemble method)")
    print("-------------------------------")
    time0 = time.time()
    df_vmd_co_imf0['target'] = df_integrate_result['co-imf0']
    co_imf0_predict_raw, co_imf0_gru_evaluation, co_imf0_train_loss = kr.keras_predict(df_vmd_co_imf0)
    print('======Co-IMF0 Predicting Finished======\n', co_imf0_gru_evaluation)
    time1 = time.time()
    print('Running time: %.3fs'%(time1-time0))
    co_imf0_predict_raw.plot(title='Co-IMF0 Predicting Result')
    co_imf0_train_loss.plot(title='Co-IMF0 Training Loss')
    plt.show()

    # 8.Predict Co-IMF1 and Co-IMF2 by vector-input GRU (respective method)
    print("\n8.Predict Co-IMF1 and Co-IMF2 by vector-input GRU (respective method)")
    print("-------------------------------")
    co_imf1_predict_raw, co_imf1_gru_evaluation, co_imf1_train_loss = kr.keras_predict(df_integrate_result['co-imf1'])
    print('======Co-IMF1 Predicting Finished======\n', co_imf1_gru_evaluation)
    time2 = time.time()
    print('Running time: %.3fs'%(time2-time1))
    co_imf1_predict_raw.plot(title='Co-IMF1 Predicting Result')
    co_imf1_train_loss.plot(title='Co-IMF1 Training Loss')
    plt.show()

    co_imf2_predict_raw, co_imf2_gru_evaluation, co_imf2_train_loss = kr.keras_predict(df_integrate_result['co-imf2'])
    print('======Co-IMF2 Predicting Finished======\n', co_imf2_gru_evaluation)
    time3 = time.time()
    print('Running time: %.3fs'%(time3-time2))
    co_imf2_predict_raw.plot(title='Co-IMF2 Predicting Result')
    co_imf2_train_loss.plot(title='Co-IMF2 Training Loss')
    plt.show()

    # 9. Add 3 result to get the final forecasting result (instead fitting method )
    if not fitting:
        print("\n9. Add 3 result to get the final forecasting result")
        print("-------------------------------")
        forecast_length = 30 # duration
        series_add_predict_result = co_imf0_predict_raw['predict']+co_imf1_predict_raw['predict']+co_imf2_predict_raw['predict']
        df_add_predict_raw = pd.DataFrame({'predict': series_add_predict_result.values, 'raw': data[-forecast_length:].values}, index=range(forecast_length))
        df_add_evaluation = eval_result(data[-forecast_length:],series_add_predict_result)
        print('======Hybrid CEEMDAN-VMD-GRU Keras Forecasting Finished======\n', df_add_evaluation)
        end = time.time()
        print('Total Running time: %.3fs'%(end-start))
        df_add_predict_raw.plot(title='Hybrid CEEMDAN-VMD-GRU Keras Forecasting Result')
        # pd.DataFrame.to_csv(df_add_predict_raw, PATH+CODE+'_predict_output.csv')
        plt.show()

    # 10.Fit 3 result to get the final forecasting result (instead adding method) (kr.hybrid_keras_forecast() can finish all step)
    else:
        print("\n9.Fit 3 result to get the final forecasting result")
        print("-------------------------------")
        df_fitting_set =  pd.DataFrame({'co-imf0': co_imf0_predict_raw['predict'], 'co-imf1': co_imf1_predict_raw['predict'], 'co-imf2': co_imf2_predict_raw['predict']}, index=range(len(co_imf0_predict_raw)))
        df_training_set = df_integrate_result
        df_training_set['target'] = data.values
        df_predict_raw, df_gru_evaluation, df_train_loss = kr.keras_predict(df_training_set, fitting_set=df_fitting_set)
        print('======Hybrid CEEMDAN-VMD-GRU Keras Forecasting Finished======\n', df_gru_evaluation)
        end = time.time()
        print('Running time: %.3fs'%(end-time3))
        print('Total Running time: %.3fs'%(end-start))
        df_predict_raw.plot(title='Hybrid CEEMDAN-VMD-GRU Keras Forecasting with Fitting Result')
        df_train_loss.plot(title='Hybrid CEEMDAN-VMD-GRU Keras Forecasting with Fitting Training Loss')
        # pd.DataFrame.to_csv(df_predict_raw, PATH+CODE+'_predict_output.csv')
        plt.show()