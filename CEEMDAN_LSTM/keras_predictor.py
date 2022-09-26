#!/usr/bin/env python
# coding: utf-8
#
# Module: keras_predictor
# Description: Forecast time series by Keras.
#
# Created: 2021-10-1 22:37
# Updated: 2022-9-12 17:28
# Author: Feite Zhou
# Email: jupiterzhou@foxmail.com
# URL: 'http://github.com/FateMurphy/CEEMDAN_LSTM'
# Feel free to email me if you have any questions or error reports.

"""
References:
K-Means, scikit-learn(sklearn):
    scikit-learn: https://github.com/scikit-learn/scikit-learn
Keras, tensorflow=2.x:
    keras: https://github.com/keras-team/keras
    tensorflow=2.x: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/keras
"""

# Import modules for keras_predictor
import os
import sys
import time
import pandas as pd
import warnings
warnings.filterwarnings("ignore") # Ignore some annoying warnings
# CEEMDAN_LSTM
from CEEMDAN_LSTM.core import check_dataset, check_path, plot_save_result, name_predictor
# Keras
try: from tensorflow.python.keras.models import Sequential
except: raise ImportError('Cannot import tensorflow, install or check your tensorflow verison!')
from tensorflow import constant
# from tcn import TCN # pip install keras-tcn
from tensorflow.python.keras.layers import Dense, Activation, Dropout, LSTM, GRU, Flatten
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.callbacks import ReduceLROnPlateau, EarlyStopping
# from tensorflow.keras.utils import plot_model # To use plot_model, you need to install software graphviz

class keras_predictor:
    # 0.Initialize
    # ------------------------------------------------------
    def __init__(self, PATH=None, FORECAST_HORIZONS=30, FORECAST_LENGTH=30, KERAS_MODEL='GRU', DECOM_MODE='CEEMDAN', INTE_LIST=None, REDECOM_LIST={'co-imf0':'vmd'}, 
                 NEXT_DAY=False, DAY_AHEAD=1, NOR_METHOD='minmax', FIT_METHOD='add', USE_TPU=False , **kwargs):
        """
        Initialize the keras_predictor.
        Configuration can be passed as kwargs (keyword arguments).

        Input and HyperParameters:
        ---------------------
        PATH               - the saving path of figures and logs
        FORECAST_HORIZONS  - also called Timestep or Forecast_horizons or sliding_windows_length in some papers
                           - the length of each input row(x_train.shape), which means the number of previous days related to today
        FORECAST_LENGTH    - the length of the days to forecast (test set)
        KERAS_MODEL        - the Keras model, eg. 'GRU', 'LSTM', 'DNN', 'BPNN', or model = Sequential()
        DECOM_MODE         - the decomposition method, eg.'EMD', 'VMD', 'CEEMDAN'
        INTE_LIST          - the integration list, eg. pd.Dataframe, (int) 3, (str) '233', (list) [0,0,1,1,1,2,2,2], ...
        REDECOM_LIST       - the re-decomposition list eg. '{'co-imf0':'vmd', 'co-imf1':'emd'}', pd.DataFrame
        NEXT_DAY           - set True to only predict next out-of-sample value
        DAY_AHEAD          - define to forecast n days' ahead eg. 0, 1 (default int 1)
        NOR_METHOD         - the normalizing method, eg. 'minmax'-MinMaxScaler, 'std'-StandardScaler, otherwise without normalization
        FIT_METHOD         - the fitting method to stablize the forecasting result (not necessarily useful), eg. 'add', 'ensemble'
        USE_TPU            - change Keras model to TPU model (for google Colab)

        Examples for custom KERAS_MODEL
        --------------------------
        model = Sequential()
        model.add(LSTM(64, input_shape=(trainset_shape[1], trainset_shape[2]), activation='tanh', return_sequences=False))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1, activation='tanh'))
        model.compile(loss='mse', optimizer='adam')
        cl.keras_predictor(KERAS_MODEL=model)

        Keras Parameters:
        https://keras.io
        ---------------------
        epochs             - training epochs/iterations, eg. 30-1000
        dropout            - dropout rate of 3 dropout layers, eg. 0.2-0.5
        units              - the units of network layers, which (3 layers) will set to 4*units, 2*units, units, eg. 4-32
        activation         - activation function, all layers will be the same, eg. 'tanh', 'relu'
        batch_size         - training batch_size for parallel computing, eg. 4-128
        shuffle            - whether randomly disorder the training set during training process, eg. True, False
        verbose            - report of training process, eg. 0 not displayed, 1 detailed, 2 rough
        valid_split        - proportion of validation set during training process, eg. 0.1-0.2
        opt                - network optimizer, eg. 'adam', 'sgd'
        opt_lr             - optimizer learning rate, eg. 0.001-0.1
        opt_loss           - optimizer loss, eg. 'mse','mae','mape','hinge', refer to https://keras.io/zh/losses/.
        opt_patience       - optimizer patience of adaptive learning rate, eg. 10-100
        stop_patience      - early stop patience, eg. 10-100

        Temp global variables (cannot directly change):
        ---------------------
        TARGET             - saving target column from input data['target']
        REDECOM_MODE       - the re-decomposition method from REDECOM_LIST
        FIG_PATH           - saving path of figures from PATH
        LOG_PATH           - saving path of logs from PATH
        """

        # Declare hyperparameters
        self.PATH = PATH
        self.FORECAST_HORIZONS = int(FORECAST_HORIZONS)
        self.FORECAST_LENGTH = int(FORECAST_LENGTH)
        self.KERAS_MODEL = KERAS_MODEL
        self.DECOM_MODE = str(DECOM_MODE)
        self.INTE_LIST = INTE_LIST
        self.REDECOM_LIST = REDECOM_LIST
        self.NEXT_DAY = bool(NEXT_DAY)
        self.DAY_AHEAD = int(DAY_AHEAD)
        self.NOR_METHOD = str(NOR_METHOD)
        self.FIT_METHOD = str(FIT_METHOD)
        self.USE_TPU = bool(USE_TPU)

        self.TARGET = None
        self.REDECOM_MODE = None
        self.FIG_PATH = None
        self.LOG_PATH = None

        # Declare Keras parameters
        self.epochs = int(kwargs.get('epochs', 100))
        self.dropout = float(kwargs.get('dropout', 0.2))
        self.units = int(kwargs.get('units', 32))
        self.activation = str(kwargs.get('activation', 'tanh'))
        self.batch_size = int(kwargs.get('batch_size', 32))
        self.shuffle = bool(kwargs.get('shuffle', True))
        self.verbose = int(kwargs.get('verbose', 0))
        self.valid_split = float(kwargs.get('valid_split', 0.1))
        self.opt = str(kwargs.get('opt', 'adam'))
        self.opt_lr = float(kwargs.get('opt_lr', 0.001))
        self.opt_loss = str(kwargs.get('opt_loss', 'mse'))
        self.opt_patience = int(kwargs.get('opt_patience', 10))
        self.stop_patience = int(kwargs.get('stop_patience', 50))

        # Check parameters
        self.PATH, self.FIG_PATH, self.LOG_PATH = check_path(PATH) # Check PATH

        if self.FORECAST_HORIZONS <= 0: raise ValueError("Invalid input for FORECAST_HORIZONS! Please input a positive integer >0.")
        if self.FORECAST_LENGTH <= 0: raise ValueError("Invalid input for FORECAST_LENGTH! Please input a positive integer >0.")
        if self.DAY_AHEAD < 0: raise ValueError("Invalid input for DAY_AHEAD! Please input a integer >=0.")
        if self.epochs <= 0: raise ValueError("Invalid input for epochs! Please input a positive integer >0.")
        if self.units <= 0: raise ValueError("Invalid input for units! Please input a positive integer >0.")
        if self.verbose not in [0, 1, 2] <= 0: raise ValueError("Invalid input for verbose! Please input 0 - not displayed, 1 - detailed, 2 - rough.")
        if self.opt_patience <= 0: raise ValueError("Invalid input for opt_patience! Please input a positive integer >0.")
        if self.stop_patience <= 0: raise ValueError("Invalid input for stop_patience! Please input a positive integer >0.")
        if self.dropout < 0 or self.dropout > 1: raise ValueError("Invalid input for dropout! Please input a number between 0 and 1.")
        if self.opt_lr < 0 or self.opt_lr > 1: raise ValueError("Invalid input for opt_lr! Please input a number between 0 and 1.")
        
        if self.FORECAST_HORIZONS < 0: raise ValueError("Invalid input for FORECAST_HORIZONS! Please input a integer >=0.")

        if not isinstance(KERAS_MODEL, Sequential): # Check KERAS_MODEL
            if type(KERAS_MODEL) == str: self.KERAS_MODEL = KERAS_MODEL.upper()
            else: raise ValueError("Invalid input for KERAS_MODEL! Please input eg. 'GRU', 'LSTM', or model = Sequential().")

        if type(DECOM_MODE) == str: self.DECOM_MODE = str(DECOM_MODE).upper() # Check DECOM_MODE

        if REDECOM_LIST is not None:# Check REDECOM_LIST and Input REDECOM_MODE
            REDECOM_MODE = ''
            try:
                REDECOM_LIST = pd.DataFrame(REDECOM_LIST, index=range(1)) 
                for i in range(REDECOM_LIST.size): REDECOM_MODE = REDECOM_MODE+REDECOM_LIST.values.ravel()[i]+REDECOM_LIST.columns[i][-1]+'-'
            except: raise ValueError("Invalid input for REDECOM_LIST! Please input eg. None, '{'co-imf0':'vmd', 'co-imf1':'emd'}'.")
            self.REDECOM_MODE = str(REDECOM_MODE).upper()
        else: REDECOM_MODE = None

        if DAY_AHEAD == 0 and FIT_METHOD == 'ensemble':
                raise ValueError('Warning! When DAY_AHEAD = 0, it is not support the fitting method, already fit today.')

        if self.opt_lr != 0.001 and self.opt == 'adam': self.opt = Adam(learning_rate=self.opt_lr)# Check optimizer
        if self.opt_patience > self.epochs: self.opt_patience = self.epochs // 10
        if self.stop_patience > self.epochs: self.stop_patience = self.epochs // 2
        
        


    # 1.Basic functions
    # ------------------------------------------------------
    # 1.1 Build model
    def build_model(self, trainset_shape):
        """
        Build Keras model, eg. 'GRU', 'LSTM', 'DNN', 'BPNN', or model = Sequential().
        """

        if isinstance(self.KERAS_MODEL, Sequential): 
            return self.KERAS_MODEL  
        elif self.KERAS_MODEL == 'LSTM':
            model = Sequential()
            model.add(LSTM(self.units*4, input_shape=(trainset_shape[1], trainset_shape[2]), activation=self.activation, return_sequences=True))
            model.add(Dropout(self.dropout))
            model.add(LSTM(self.units*2, activation=self.activation, return_sequences=True))
            model.add(Dropout(self.dropout))
            model.add(LSTM(self.units, activation=self.activation, return_sequences=False))
            model.add(Dropout(self.dropout))
            model.add(Dense(1, activation=self.activation))
            model.compile(loss=self.opt_loss, optimizer=self.opt)
            return model
        elif self.KERAS_MODEL == 'GRU':
            model = Sequential()
            model.add(GRU(self.units*4, input_shape=(trainset_shape[1], trainset_shape[2]), activation=self.activation, return_sequences=True))
            model.add(Dropout(self.dropout))
            model.add(GRU(self.units*2, activation=self.activation, return_sequences=True))
            model.add(Dropout(self.dropout))
            model.add(GRU(self.units, activation=self.activation, return_sequences=False))
            model.add(Dropout(self.dropout))
            model.add(Dense(1, activation=self.activation))
            model.compile(loss=self.opt_loss, optimizer=self.opt)
            return model
        elif self.KERAS_MODEL == 'DNN':
            model = Sequential()
            model.add(Dense(self.units*4, input_shape=(trainset_shape[1], trainset_shape[2]), activation=self.activation))
            model.add(Dropout(self.dropout))
            model.add(Dense(self.units*2, activation=self.activation))
            model.add(Dropout(self.dropout))
            model.add(Flatten())
            model.add(Dense(self.units, activation=self.activation))
            model.add(Dropout(self.dropout))
            model.add(Dense(1, activation=self.activation))
            model.compile(loss=self.opt_loss, optimizer=self.opt)
            return model
        elif self.KERAS_MODEL == 'BPNN':
            model = Sequential()
            model.add(Dense(self.units*4, input_shape=(trainset_shape[1], trainset_shape[2]), activation=self.activation))
            model.add(Dropout(self.dropout))
            model.add(Flatten())
            model.add(Dense(1, activation=self.activation))
            model.compile(loss=self.opt_loss, optimizer=self.opt)
            return model
        else: raise ValueError("%s is an invalid input for KERAS_MODEL! eg. 'GRU', 'LSTM', or model = Sequential()"%self.KERAS_MODEL)

    # 1.2 Change Keras model to TPU model (for google Colab)
    def tpu_model(self, shape):
        """
        Change Keras model to TPU model (for google Colab)
        """
        import tensorflow as tf
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        with strategy.scope(): # Change Keras model to TPU form
            model = self.build_model(shape) # Build the model # Use model.summary() to show the model structure
        return model

    # 1.Main Forecast by Keras
    def keras_predict(self, data=None, show_model=False, fitting_set=None, **kwargs): # GRU forecasting function
        """
        Forecast by Keras
        
        Input and Parameters:
        ---------------------
        data               - data set (include training set and test set)
        show_data          - show the inputting data set
        show_model         - show Keras model structure
        **kwargs           - any parameters of model.fit()

        Output
        ---------------------
        df_result          - forecasting results and the original real series set
        df_eval            - evaluating results of forecasting 
        df_loss            - training loss log
        df_result(next_y)  - forecasting result of the next day when NEXT_DAY=Ture

        Plot by pandas: 
        ---------------------
        df_result.plot(title='Forecasting Results')
        print(df_eval)
        df_loss.plot(title='Training Loss')
        """

        # Initialize
        from CEEMDAN_LSTM.data_preprocessor import create_train_test_set
        x_train, x_test, y_train, y_test, scalarY, next_x = create_train_test_set(data, self.FORECAST_LENGTH, self.FORECAST_HORIZONS, self.NOR_METHOD, self.DAY_AHEAD, fitting_set) 

        # Convert to tensor = tf.constant()
        today_x = x_train[-1].reshape(1, x_train.shape[1], x_train.shape[2])
        x_train = constant(x_train)
        x_test = constant(x_test)
        # x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2])) 
        # x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2])) 

        # Build the model 
        if self.USE_TPU: model = self.tpu_model(x_train.shape)
        else: model = self.build_model(x_train.shape) 
        if show_model: 
            print('\nInput Shape: (%d,%d)\n'%(x_train.shape[1], x_train.shape[2]))
            model.summary() # The summary of layers and parameters

        # Set callbacks and predict
        Reduce = ReduceLROnPlateau(monitor='val_loss', patience=self.opt_patience, verbose=0, mode='auto') # Adaptive learning rate
        EarlyStop = EarlyStopping(monitor='val_loss', patience=self.stop_patience, verbose=0, mode='auto') # Early stop at small learning rate
        history = model.fit(x_train, y_train, # Train the model 
                            epochs=self.epochs, 
                            batch_size=self.batch_size, 
                            validation_split=self.valid_split, 
                            verbose=self.verbose, 
                            shuffle=self.shuffle, 
                            callbacks=[EarlyStop, Reduce],
                            **kwargs)
        # model.save(imf+'keras_model.h5')

        if not self.NEXT_DAY:
            # Get results and evaluate
            from CEEMDAN_LSTM.data_preprocessor import eval_result
            y_predict = model.predict(x_test) # Predict
            df_eval = eval_result(y_test, y_predict) # Evaluate model
            y_predict = y_predict.ravel().reshape(-1,1) 
            if scalarY is not None: 
                y_test = scalarY.inverse_transform(y_test)
                y_predict = scalarY.inverse_transform(y_predict) # De-normalize 
            if self.TARGET is not None: result_index = self.TARGET.index[-self.FORECAST_LENGTH:] # Forecasting result idnex
            else: result_index = range(len(y_test.ravel()))
            df_result = pd.DataFrame({'real': y_test.ravel(), 'predict': y_predict.ravel()}, index=result_index) # Output
            df_loss = pd.DataFrame({'loss': history.history['loss'], 'val_loss': history.history['val_loss']}, index=range(len(history.history['val_loss'])))
            return df_result, df_eval, df_loss
        else:
            # Get next day's results and evaluate
            today_y = model.predict(today_x)
            next_y = model.predict(next_x.reshape(1, today_x.shape[1], today_x.shape[2]))
            if scalarY is not None: # De-normalize 
                today_real = scalarY.inverse_transform([y_test[-1]])
                today_y = scalarY.inverse_transform(today_y)
                next_y = scalarY.inverse_transform(next_y)
            else: today_real = y_test[-1]
            if self.TARGET is not None: next_index = [self.TARGET.index[-1]] # [self.TARGET.index[-1] + (self.TARGET.index[1] - self.TARGET.index[0])]
            else: next_index = range(1)
            df_result = pd.DataFrame({'today_real': today_real.ravel()[0], 'today_pred': today_y.ravel()[0], 'next_pred': next_y.ravel()[0]}, index=next_index) # Output
            return df_result



    # 2.Advanced forecasting functions
    # ------------------------------------------------------
    # 2.1. Single Method (directly forecast)
    def single_keras_predict(self, data=None, show_data=False, show_model=False, plot_result=False, save_result=False, **kwargs):
        """
        Single Method (directly forecast)
        Use Keras model to directly forecast with vector input
        Example: 
        kr = cl.keras_predictor(epochs=10, verbose=1)
        df_result = kr.single_keras_predict(data, show_data=True, show_model=True, plot_result=True, save_result=True)

        Input and Parameters:
        ---------------------
        Note important hyperarameters of class cl.keras_predict()
        data            - data set (include training set and test set)
        show_data       - show the inputting data set
        show_model      - show Keras model structure or not
        plot_result     - show figure result or not
        save_result     - save forecasting result or not
        **kwargs        - any parameters of self.keras_predict()

        Output
        ---------------------
        df_result = (df_predict_real, df_eval, df_train_loss) or next_y
        df_predict_real - forecasting results and the original real series set
        df_eval         - evaluating results of forecasting 
        df_train_loss   - training loss log
        next_y          - forecasting result of the next day when NEXT_DAY=Ture
        """

        # Name and set target
        now = pd.datetime.now()
        predictor_name = name_predictor(now, 'Single', 'Keras', self.KERAS_MODEL, None, None, self.NEXT_DAY)
        data = check_dataset(data, show_data, self.DECOM_MODE)
        self.TARGET = data['target']

        # Forecast
        start = time.time()
        df_result = self.keras_predict(data=data, show_model=show_model, **kwargs)
        end = time.time()

        # Output
        print('\n=========='+predictor_name+' Finished==========')
        if not self.NEXT_DAY: 
            df_result[1]['Runtime'] = end-start # Output Runtime
            df_result[0].name, df_result[1].name, df_result[2].name = predictor_name+' Result', predictor_name+' Evaluation', predictor_name+' Loss'
            print(df_result[1]) # print df_eval
        else: 
            df_result['Runtime'] = end-start # Output Runtime
            df_result.name = predictor_name+' Result'
            print(df_result)
        plot_save_result(df_result, name=now, plot=plot_result, save=save_result ,path=self.PATH)
        return df_result



    # 2.2. Ensemble Method (decompose then directly forecast)
    def ensemble_keras_predict(self, data=None, show_data=False, show_model=False, plot_result=False, save_result=False, **kwargs):
        """
        Ensemble Method (decompose then directly forecast)
        Use decomposition-integration Keras model to directly forecast with matrix input
        Example: 
        kr = cl.keras_predictor(epochs=10, verbose=1)
        df_result = kr.ensemble_keras_predict(data, show_data=True, show_model=True, plot_result=True, save_result=True)

        Input and Parameters:
        ---------------------
        Note important hyperarameters of class cl.keras_predict()
        data            - data set (include training set and test set)
        show_data       - show the inputting data set
        show_model      - show Keras model structure or not
        plot_result     - show figure result or not
        save_result     - save forecasting result or not
        **kwargs        - any parameters of self.keras_predict()       

        Output
        ---------------------
        df_result = (df_predict_real, df_eval, df_train_loss) or next_y
        df_predict_real - forecasting results and the original real series set
        df_eval         - evaluating results of forecasting 
        df_train_loss   - training loss log
        next_y          - forecasting result of the next day when NEXT_DAY=Ture
        """

        # Name
        now = pd.datetime.now()
        predictor_name = name_predictor(now, 'Ensemble', 'Keras', self.KERAS_MODEL, self.DECOM_MODE, self.REDECOM_MODE, self.NEXT_DAY)

        # Decompose, integrate, re-decompose
        start = time.time()
        from CEEMDAN_LSTM.data_preprocessor import redecom
        df_redecom, df_redecom_list = redecom(data, show_data, self.DECOM_MODE, self.INTE_LIST, self.REDECOM_LIST)
        self.TARGET = df_redecom['target']

        # Forecast
        df_result = self.keras_predict(data=df_redecom, show_model=show_model, **kwargs)
        end = time.time()

        # Output
        print('\n=========='+predictor_name+' Finished==========')
        if not self.NEXT_DAY: 
            df_result[1]['Runtime'] = end-start # Output Runtime
            df_result[0].name, df_result[1].name, df_result[2].name = predictor_name+' Result', predictor_name+' Evaluation', predictor_name+' Loss'
            print(df_result[1]) # print df_eval
        else: 
            df_result['Runtime'] = end-start # Output Runtime
            df_result.name = predictor_name+' Result'
            print(df_result)
        plot_save_result(df_result, name=now, plot=plot_result, save=save_result, path=self.PATH)
        return df_result


    
    # 2.3. Respective Method (decompose then respectively forecast each IMFs)
    def respective_keras_predict(self, data=None, show_data=False, show_model=False, plot_result=False, save_result=False, **kwargs):
        """
        Respective Method (decompose then respectively forecast each IMFs)
        Use decomposition-integration Keras model to respectively forecast each IMFs with vector input
        Example: 
        kr = cl.keras_predictor(epochs=10, verbose=1)
        df_result = kr.respective_keras_predict(data, show_data=True, show_model=True, plot_result=True, save_result=True)

        Input and Parameters:
        ---------------------
        Note important hyperarameters of class cl.keras_predict()
        data            - data set (include training set and test set)
        show_data       - show the inputting data set
        show_model      - show Keras model structure or not
        plot_result     - show figure result or not
        save_result     - save forecasting result or not
        **kwargs        - any parameters of self.keras_predict()

        Output
        ---------------------
        df_result = (df_predict_real, df_eval, df_train_loss) or next_y
        df_predict_real - forecasting results and the original real series set
        df_eval         - evaluating results of forecasting 
        df_train_loss   - training loss log
        next_y          - forecasting result of the next day when NEXT_DAY=Ture
        """

        # Name
        now = pd.datetime.now()
        predictor_name = name_predictor(now, 'Respective', 'Keras', self.KERAS_MODEL, self.DECOM_MODE, self.REDECOM_MODE, self.NEXT_DAY)

        # Decompose, integrate, re-decompose
        start = time.time()
        from CEEMDAN_LSTM.data_preprocessor import redecom
        df_redecom, df_redecom_list = redecom(data, show_data, self.DECOM_MODE, self.INTE_LIST, self.REDECOM_LIST)
        self.TARGET = df_redecom['target']

        # Forecast and ouput each Co-IMF
        df_pred_result = pd.DataFrame(index = self.TARGET.index[-self.FORECAST_LENGTH:0]) # df for storing forecasting result
        df_eval_result = pd.DataFrame(columns=['Scale', 'R2', 'RMSE', 'MAE', 'MAPE', 'Runtime', 'IMF']) # df for storing evaluation result
        df_next_result = pd.DataFrame(columns=['today_real', 'today_pred', 'next_pred', 'Runtime', 'IMF']) # df for storing Next-day forecasting result
        for imf in df_redecom.columns.difference(['target']):
            time1 = time.time()
            df_result = self.keras_predict(data=df_redecom[imf], show_model=show_model, **kwargs)
            time2 = time.time()
            print('----'+predictor_name+' of '+imf+' Finished----')
            if not self.NEXT_DAY: 
                df_result[1]['Runtime'] = time2-time1 # Output Runtime
                df_result[1]['IMF'] = imf
                df_result[0].name, df_result[1].name, df_result[2].name = predictor_name+' Result of '+imf, predictor_name+' Evaluation of '+imf, predictor_name+' Loss of '+imf
                print(df_result[1]) # print df_eval
                df_pred_result[imf+'-real'] = df_result[0]['real'] # add real values
                df_pred_result[imf+'-predict'] = df_result[0]['predict'] # add forecasting result
                df_eval_result = pd.concat((df_eval_result, df_result[1])) # add evaluation result
            else: 
                df_result['Runtime'] = time2-time1 # Output Runtime
                df_result['IMF'] = imf
                df_result.name = predictor_name+' Result of '+imf
                print(df_result)
                df_next_result = pd.concat((df_next_result, df_result)) # add Next-day forecasting result
            plot_save_result(df_result, name=now, plot=plot_result, save=False, path=self.PATH) # do not save Co-IMF results
            print('')
        end = time.time()

        # Final Output
        print('=========='+predictor_name+' Finished==========')
        from CEEMDAN_LSTM.data_preprocessor import eval_result
        if not self.NEXT_DAY: 
            df_pred_result['real'] = self.TARGET[-self.FORECAST_LENGTH:]
            df_pred_result['predict'] = df_pred_result[[x for x in df_pred_result.columns if 'predict' in x]].sum(axis=1)
            final_eval = eval_result(df_pred_result['predict'], df_pred_result['real'])
            final_eval['Runtime'] = end-start # Output Runtime
            final_eval['IMF'] = 'Final'
            print(final_eval) # print df_eval
            final_eval = pd.concat((final_eval, df_eval_result))
            df_plot = df_pred_result[['real', 'predict']]
            df_pred_result.name, final_eval.name, df_plot.name = predictor_name+' Result', predictor_name+' Evaluation', predictor_name+' Result'
            plot_save_result(df_plot, name=now, plot=plot_result, save=False, path=self.PATH)
            df_result = (df_pred_result, final_eval)
        else:
            df_result = pd.DataFrame({'today_real': self.TARGET[-1], 'today_pred': df_next_result['today_pred'].sum(), 
                                      'next_pred': df_next_result['next_pred'].sum()}, index=[df_next_result.index[0]]) # Output
            df_result['Runtime'] = end-start # Output Runtime
            df_result['IMF'] = 'Final'
            df_result = pd.concat((df_result, df_next_result))
            df_result.name = predictor_name+' Result'
            print(df_result)
        plot_save_result(df_result, name=now, plot=False, save=save_result, path=self.PATH)
        return df_result



    # 2.4. Hybrid Method (decompose then forecast high-frequency IMF by ensemble method and forecast other by respective method)
    def hybrid_keras_predict(self, data=None, show_data=False, show_model=False, plot_result=False, save_result=False, **kwargs):
        """
        Hybrid Method (decompose then forecast high-frequency IMF by ensemble method and forecast others by respective method)
        Use the ensemble method to forecast high-frequency IMF and the respective method for other IMFs.
        Example: 
        kr = cl.keras_predictor(epochs=10, verbose=1)
        df_result = kr.hybrid_keras_predict(data, show_data=True, show_model=True, plot_result=True, save_result=True)

        Input and Parameters:
        ---------------------
        Note important hyperarameters of class cl.keras_predict()
        data            - data set (include training set and test set)
        show_data       - show the inputting data set
        show_model      - show Keras model structure or not
        plot_result     - show figure result or not
        save_result     - save forecasting result or not
        **kwargs        - any parameters of self.keras_predict()

        Output
        ---------------------
        df_result = (df_predict_real, df_eval, df_train_loss) or next_y
        df_predict_real - forecasting results and the original real series set
        df_eval         - evaluating results of forecasting 
        df_train_loss   - training loss log
        next_y          - forecasting result of the next day when NEXT_DAY=Ture
        """

        # Name
        now = pd.datetime.now()
        predictor_name = name_predictor(now, 'Hybrid', 'Keras', self.KERAS_MODEL, self.DECOM_MODE, self.REDECOM_MODE, self.NEXT_DAY)
        # try: data = pd.Series(data)
        # except: raise ValueError('Sorry! %s is not supported for the Hybrid Method, please input pd.DataFrame, pd.Series, nd.array(<=2D)'%type(data))

        # Decompose, integrate, re-decompose
        start = time.time()
        from CEEMDAN_LSTM.data_preprocessor import redecom
        df_redecom, df_redecom_list = redecom(data, show_data, self.DECOM_MODE, self.INTE_LIST, self.REDECOM_LIST)
        self.TARGET = df_redecom['target']
        df_rest_columns, df_rest_list = [], []
        for x in df_redecom_list: # create df_rest_list
            df_redecom[x.name] = x['target']
            df_rest_columns.append(x.name)
            df_redecom = df_redecom[df_redecom.columns.difference(x.columns.difference(['target']))]
        for x in df_redecom.columns.difference(df_rest_columns): 
            if x != 'target': df_rest_list.append(df_redecom[x])

        # Forecast
        df_pred_result = pd.DataFrame(index = self.TARGET.index[-self.FORECAST_LENGTH:0]) # df for storing forecasting result
        df_eval_result = pd.DataFrame(columns=['Scale', 'R2', 'RMSE', 'MAE', 'MAPE', 'Runtime', 'IMF']) # df for storing evaluation result
        df_next_result = pd.DataFrame(columns=['today_real', 'today_pred', 'next_pred', 'Runtime', 'IMF']) # df for storing Next-day forecasting result
        for df in df_redecom_list+df_rest_list: # both ensemble (matrix-input) and respective (vector-input) method 
            time1 = time.time()
            df_result = self.keras_predict(data=df, show_model=show_model, **kwargs) # the ensemble method with matrix input 
            time2 = time.time()
            imf = df.name
            print('\n----------'+predictor_name+' of '+imf+' Finished----------')
            if not self.NEXT_DAY: 
                df_result[1]['Runtime'] = time2-time1 # Output Runtime
                df_result[1]['IMF'] = imf
                df_result[0].name, df_result[1].name, df_result[2].name = predictor_name+' Result of '+imf, predictor_name+' Evaluation of '+imf, predictor_name+' Loss of '+imf
                print(df_result[1]) # print df_eval
                df_pred_result[imf+'-real'] = df_result[0]['real'] # add real values
                df_pred_result[imf+'-predict'] = df_result[0]['predict'] # add forecasting result
                df_eval_result = pd.concat((df_eval_result, df_result[1])) # add evaluation result
                plot_save_result(df_result, name=now, plot=plot_result, save=False, path=self.PATH) # do not save Co-IMF results
            else: 
                df_result['Runtime'] = time2-time1 # Output Runtime
                df_result['IMF'] = imf
                df_result.name = predictor_name+' Result of '+imf
                print(df_result)
                df_next_result = pd.concat((df_next_result, df_result)) # add Next-day forecasting result
            print('')

        # Fitting 
        if not self.NEXT_DAY: 
            if self.FIT_METHOD == 'ensemble': # fitting method
                print('Fitting of the ensemble method is running...')        
                fitting_set = df_pred_result[[x for x in df_pred_result.columns if '-predict' in x]]
                time1 = time.time()
                df_result = self.keras_predict(df_redecom, show_model, fitting_set, **kwargs) # the ensemble method with matrix input 
                time2 = time.time()
                print('-------------'+predictor_name+' of Fitting Finished-------------')
                df_result[1]['Runtime'] = time2-time1 # Output Runtime
                df_result[1]['IMF'] = 'fitting'
                df_result[0].name, df_result[1].name, df_result[2].name = predictor_name+' Result of Fitting', predictor_name+' Evaluation of Fitting', predictor_name+' Loss of Fitting'
                print(df_result[1]) # print df_eval
                df_eval_result = pd.concat((df_eval_result, df_result[1])) # add evaluation result
                df_pred_result['real'] = self.TARGET[-self.FORECAST_LENGTH:]
                df_pred_result['add-predict'] = df_pred_result[[x for x in df_pred_result.columns if '-predict' in x]].sum(axis=1) # add all imf predict result
                df_pred_result['predict'] = df_result[0]['predict']
                plot_save_result(df_result, name=now, plot=plot_result, save=False, path=self.PATH) # do not save Co-IMF results
            else: # adding method
                df_pred_result['real'] = self.TARGET[-self.FORECAST_LENGTH:]
                df_pred_result['predict'] = df_pred_result[[x for x in df_pred_result.columns if '-predict' in x]].sum(axis=1) # add all imf predict result
        else: # if self.NEXT_DAY:
            if self.FIT_METHOD == 'ensemble': # fitting method
                print('Fitting of the ensemble method is running...')        
                fitting_set = df_redecom[df_redecom.columns.difference(['target'])][-self.FORECAST_LENGTH+1:]
                fitting_set_next_index = fitting_set.index[-1]+(fitting_set.index[-1]-fitting_set.index[-2])
                fitting_set.loc[fitting_set_next_index] = [x for x in range(fitting_set.columns.size)] # create blank row
                for i in range(df_next_result.index.size): # add next_pred of each imf to fitting_set
                    point_row = df_next_result[i:i+1]
                    fitting_set[point_row['IMF'][0]][fitting_set_next_index] = point_row['next_pred'][0]
                time1 = time.time()
                df_result = self.keras_predict(df_redecom, show_model, fitting_set, **kwargs) # the ensemble method with matrix input 
                time2 = time.time()
                print('---------'+predictor_name+' of Fitting Finished---------')
                df_result['Runtime'] = time2-time1 # Output Runtime
                df_result['IMF'] = 'fitting'
                df_result.name = predictor_name+' Result of Fitting'
                print(df_result)
                df_next_result = pd.concat((df_next_result, df_result)) # add Next-day forecasting result
                today_result = df_result['today_pred']
                next_reuslt = df_result['next_pred']
            else: # adding method
                today_result = df_next_result['today_pred'].sum()
                next_reuslt = df_next_result['next_pred'].sum() # adding method
        end = time.time()

        # Final Output
        print('\n================'+predictor_name+' Finished================')
        from CEEMDAN_LSTM.data_preprocessor import eval_result
        if not self.NEXT_DAY: 
            final_eval = eval_result(df_pred_result['predict'], df_pred_result['real'])
            final_eval['Runtime'] = end-start # Output Runtime
            final_eval['IMF'] = 'Final'
            print(final_eval) # print df_eval
            final_eval = pd.concat((final_eval, df_eval_result))
            df_plot = df_pred_result[['real', 'predict']]
            df_pred_result.name, final_eval.name, df_plot.name = predictor_name+' Result', predictor_name+' Evaluation', predictor_name+' Result'
            plot_save_result(df_plot, name=now, plot=plot_result, save=False, path=self.PATH)
            df_result = (df_pred_result, final_eval)
        else:
            df_result = pd.DataFrame({'today_real': self.TARGET[-1], 'today_pred': today_result, 'next_pred': next_reuslt}, index=[df_next_result.index[0]]) # Output
            df_result['Runtime'] = end-start # Output Runtime
            df_result['IMF'] = 'Final'
            df_result = pd.concat((df_result, df_next_result))
            df_result.name = predictor_name+' Result'
            print(df_result)
        plot_save_result(df_result, name=now, plot=False, save=save_result, path=self.PATH)
        return df_result

    class HiddenPrints: # used to hide the print
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout

     # 2.5. Multiple Run Predictor
    def multiple_keras_predict(self, data=None, run_times=10, predict_method=None, save_each_result=False, **kwargs):
        """
        Multiple Run Predictor, multiple run of above method
        Example: 
        kr = cl.keras_predictor(epochs=10, verbose=1)
        df_result = kr.multiple_predict(data, run_times=10, predict_method='single', save_each_result=False)

        Input and Parameters:
        ---------------------
        Note important hyperarameters of class cl.keras_predict()
        data               - data set (include training set and test set)
        run_times          - running times
        predict_method     - run different method eg. 'single', 'ensemble', 'respective', 'hybrid'
                           - single: self.single_keras_predict()
                           - ensemble: self.ensemble_keras_predict()
                           - respective: self.respective_keras_predict()
                           - hybrid: self.hybrid_keras_predict()
        save_each_result   - save each run's result
        **kwargs           - any parameters of self.keras_predict()

        Output
        ---------------------
        df_eval_result     - evaluating forecasting results of each run
        """

        if data is None: raise ValueError('Please input a data set!')
        if predict_method is None: raise ValueError("Please input a predict method! eg. 'single', 'ensemble', 'respective', 'hybrid'.")
        else: predict_method = predict_method.capitalize()
        now = pd.datetime.now()
        if predict_method == 'Single': predictor_name = name_predictor(now, 'Multiple Single', 'Keras', self.KERAS_MODEL, None, None, self.NEXT_DAY)
        else: predictor_name = name_predictor(now, 'Multiple '+predict_method, 'Keras', self.KERAS_MODEL, self.DECOM_MODE, self.REDECOM_MODE, self.NEXT_DAY)

        start = time.time()
        df_eval_result = pd.DataFrame(columns=['Scale', 'R2', 'RMSE', 'MAE', 'MAPE', 'Runtime', 'IMF']) # df for storing evaluation result
        for i in range(run_times):
            print('Run %d is running...'%i, end='')
            time1 = time.time()
            with self.HiddenPrints():
                if predict_method == 'Single': df_result = self.single_keras_predict(data=data, save_result=save_each_result, **kwargs)
                if predict_method == 'Ensemble': df_result = self.ensemble_keras_predict(data=data, save_result=save_each_result, **kwargs) 
                if predict_method == 'Respective': df_result = self.respective_keras_predict(data=data, save_result=save_each_result, **kwargs)
                if predict_method == 'Hybrid': df_result = self.hybrid_keras_predict(data=data, save_result=save_each_result, **kwargs)
            time2 = time.time()
            print('taking time: %.3fs'%(time2-time1))
            df_result[1]['IMF'] = predict_method
            df_eval_result = pd.concat((df_eval_result, df_result[1])) # add evaluation result
        end = time.time()
        print('\n============='+predictor_name+' Finished=============')
        print('Running time: %.3fs'%(end-start))
        df_eval_result.name = predictor_name+' Result'
        plot_save_result(df_eval_result, name=now, plot=False, save=True, path=self.PATH)
        return df_eval_result



    # 2.6.Rolling Method (forecast each value one by one)
    def rolling_keras_predict(self, data=None, predict_method=None, save_each_result=False, **kwargs):
        """
        Rolling Method (forecast each value one by one to avoid lookahead bias)
        Rolling run of above method to avoid the look-ahead bias, but take a long long time
        Example: 
        kr = cl.keras_predictor(epochs=10, verbose=1)
        df_result = kr.rolling_keras_predict(data, predict_method='single', save_each_result=False)

        Input and Parameters:
        ---------------------
        Note important hyperarameters of class cl.keras_predict()
        data               - data set (include training set and test set)
        predict_method     - run different method eg. 'single', 'ensemble', 'respective', 'hybrid'
                           - single: self.single_keras_predict()
                           - ensemble: self.ensemble_keras_predict()
                           - respective: self.respective_keras_predict()
                           - hybrid: self.hybrid_keras_predict()
        save_each_result   - save each run's result
        **kwargs           - any parameters of self.keras_predict()

        Output
        ---------------------
        df_eval_result     - evaluating forecasting results of each run
        """
        self.NEXT_DAY = True
        if data is None: raise ValueError('Please input a data set!')
        if predict_method is None: raise ValueError("Please input a predict method! eg. 'single', 'ensemble', 'respective', 'hybrid'.")
        else: predict_method = predict_method.capitalize()
        now = pd.datetime.now()
        if predict_method == 'Single': predictor_name = name_predictor(now, 'Rolling Single', 'Keras', self.KERAS_MODEL, None, None, False)
        else: predictor_name = name_predictor(now, 'Rolling '+predict_method, 'Keras', self.KERAS_MODEL, self.DECOM_MODE, self.REDECOM_MODE, False)

        start = time.time()
        df_next_result = pd.DataFrame(columns=['today_real', 'today_pred', 'next_pred', 'Runtime', 'Run']) # df for storing Next-day forecasting result
        for i in range(self.FORECAST_LENGTH):
            print('Run %d is running...'%i, end='')
            time1 = time.time()
            with self.HiddenPrints():
                if predict_method == 'Single': df_result = self.single_keras_predict(data=data[:-self.FORECAST_LENGTH+i], save_result=save_each_result, **kwargs)
                if predict_method == 'Ensemble': df_result = self.ensemble_keras_predict(data=data[:-self.FORECAST_LENGTH+i], save_result=save_each_result, **kwargs) 
                if predict_method == 'Respective': df_result = self.respective_keras_predict(data=data[:-self.FORECAST_LENGT+i], save_result=save_each_result, **kwargs)
                if predict_method == 'Hybrid': df_result = self.hybrid_keras_predict(data=data[:-self.FORECAST_LENGTH+i], save_result=save_each_result, **kwargs)
            time2 = time.time()
            print('taking time: %.3fs'%(time2-time1))
            df_result['Runtime'] = time2-time1 # Output Runtime
            df_result['Run'] = i
            df_result.name = predictor_name+' Result of Run '+str(i)
            print(df_result)
            df_next_result = pd.concat((df_next_result, df_result)) # add evaluation result
        end = time.time()
        print('\n============='+predictor_name+' Finished=============')
        print('Running time: %.3fs'%(end-start))
        
        next_day_set = [] # add next_pred of each run to next_day_set
        from CEEMDAN_LSTM.data_preprocessor import eval_result
        df_pred_result = pd.DataFrame()
        df_pred_result['real'] = self.TARGET[-self.FORECAST_LENGTH:]
        df_pred_result['predict'] = df_next_result['next_pred']
        df_eval_result = eval_result(df_pred_result['predict'], df_pred_result['real'])
        df_eval_result['Runtime'] = end-start # Output Runtime
        df_eval_result['Run'] = 'Final'
        print(df_eval_result)
        df_eval_result = pd.concat((df_eval_result, df_next_result)) # add evaluation result
        df_result = (df_pred_result, df_eval_result)
        df_pred_result.name, df_eval_result.name = predictor_name+' Result', predictor_name+' Evaluation'
        plot_save_result(df_result, name=now, plot=True, save=True, path=self.PATH)
        return df_result