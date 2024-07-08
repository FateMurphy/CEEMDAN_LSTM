#!/usr/bin/env python
# coding: utf-8
#
# Module: keras_predictor
# Description: Forecast time series by Keras.
#
# Created: 2021-10-1 22:37
# Updated: 2022-9-12 17:28
# Updated: 2023-7-14 00:50
# Updated: 2024-7-08 10:55
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
import numpy as np
import warnings
from datetime import datetime
warnings.filterwarnings("ignore") # Ignore some annoying warnings
# CEEMDAN_LSTM
from CEEMDAN_LSTM.core import check_dataset, check_path, plot_save_result, name_predictor, output_result
# Keras
try: from tensorflow import constant 
except: raise ImportError('Cannot import tensorflow, install or check your tensorflow verison!')
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, GRU, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


class keras_predictor:
    # 0.Initialize
    # ------------------------------------------------------
    def __init__(self, PATH=None, FORECAST_HORIZONS=30, FORECAST_LENGTH=30, KERAS_MODEL='GRU', DECOM_MODE='CEEMDAN', INTE_LIST='auto', 
                 REDECOM_LIST={'co-imf0':'ovmd'}, NEXT_DAY=False, DAY_AHEAD=1, NOR_METHOD='minmax', FIT_METHOD='add', USE_TPU=False , VMD_PARAMS=None, **kwargs):
        """
        Initialize the keras_predictor.
        Configuration can be passed as kwargs (keyword arguments).

        Input and HyperParameters:
        ---------------------
        PATH               - the saving path of figures and logs
        FORECAST_HORIZONS  - also called Timestep or Forecast_horizons or sliding_windows_length in some papers
                           - the length of each input row(x_train.shape), which means the number of previous days related to today
        FORECAST_LENGTH    - the length of the days to forecast (test set)
        KERAS_MODEL        - the Keras model, eg. 'GRU', 'LSTM', 'DNN', 'BPNN', model = Sequential(), keras file, dict, pd.DataFrame.
                           - eg. {'co-imf0':'co_imf0_model.keras', 'co-imf1':'co_imf1_model.keras'}
        DECOM_MODE         - the decomposition method, eg.'EMD', 'EEMD', 'CEEMDAN', 'VMD', 'OVDM', 'SMVD'
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
        callbacks_monitor  - monitor of Keras callbacks function, eg. 'loss', 'val_loss'

        Temp global variables (cannot directly change):
        ---------------------
        TARGET             - saving target column from input data['target']
        REDECOM_PARAMS     - fixed parameters of VMD and OVMD at re-decomposition (reduce time and error)
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
        self.VMD_PARAMS = VMD_PARAMS # {'K':10, 'tau':0, 'alpha':2000}

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
        self.callbacks_monitor = str(kwargs.get('callbacks_monitor', 'val_loss'))

        # Check parameters
        self.PATH, self.FIG_PATH, self.LOG_PATH = check_path(PATH) # Check PATH

        if self.FORECAST_HORIZONS <= 0: raise ValueError("Invalid input for FORECAST_HORIZONS! Please input a positive integer >0.")
        if self.FORECAST_LENGTH <= 0: raise ValueError("Invalid input for FORECAST_LENGTH! Please input a positive integer >0.")
        if self.DAY_AHEAD < 0: raise ValueError("Invalid input for DAY_AHEAD! Please input a integer >=0.")
        if self.epochs < 0: raise ValueError("Invalid input for epochs! Please input a positive integer >0.")
        if self.units <= 0: raise ValueError("Invalid input for units! Please input a positive integer >0.")
        if self.verbose not in [0, 1, 2] <= 0: raise ValueError("Invalid input for verbose! Please input 0 - not displayed, 1 - detailed, 2 - rough.")
        if self.opt_patience <= 0: raise ValueError("Invalid input for opt_patience! Please input a positive integer >0.")
        if self.stop_patience <= 0: raise ValueError("Invalid input for stop_patience! Please input a positive integer >0.")
        if self.dropout < 0 or self.dropout > 1: raise ValueError("Invalid input for dropout! Please input a number between 0 and 1.")
        if self.opt_lr < 0 or self.opt_lr > 1: raise ValueError("Invalid input for opt_lr! Please input a number between 0 and 1.")
        
        # Check parameters
        if not isinstance(KERAS_MODEL, Sequential): # Check KERAS_MODEL
            if type(KERAS_MODEL) == str: 
                if '.keras' not in str(self.KERAS_MODEL): self.KERAS_MODEL = KERAS_MODEL.upper()
                else:
                    if self.PATH is None: raise ValueError("Please set a PATH to load keras model in .keras file.")
                    if not os.path.exists(self.PATH+self.KERAS_MODEL): raise ValueError("File does not exist:", self.PATH+self.KERAS_MODEL)
            else:
                if self.PATH is None: raise ValueError("Please set a PATH to load keras model in .keras file.")
                try: KERAS_MODEL = pd.DataFrame(KERAS_MODEL, index=[0])
                except: raise ValueError("Invalid input for KERAS_MODEL! Please input eg. 'GRU', 'LSTM', or model = Sequential(), keras file, dict, pd.DataFrame.")
                for file_name in KERAS_MODEL.values.ravel():
                    if '.keras' not in str(file_name): raise ValueError("Invalid input for KERAS_MODEL values! Please input eg. 'GRU', 'LSTM', or model = Sequential(), keras file, dict, pd.DataFrame.")
                    if not os.path.exists(self.PATH+file_name): raise ValueError("File does not exist:", self.PATH+file_name)

        if type(DECOM_MODE) == str: self.DECOM_MODE = str(DECOM_MODE).upper() # Check DECOM_MODE
        if REDECOM_LIST is not None:# Check REDECOM_LIST
            try: REDECOM_LIST = pd.DataFrame(REDECOM_LIST, index=[0]) 
            except: raise ValueError("Invalid input for REDECOM_LIST! Please input eg. None, '{'co-imf0':'vmd', 'co-imf1':'emd'}'.")
        if DAY_AHEAD == 0 and FIT_METHOD == 'ensemble': # Check DAY_AHEAD
                raise ValueError('Warning! When DAY_AHEAD = 0, it is not support the fitting method, already fit today.')

        if self.opt_lr != 0.001 and self.opt == 'adam': self.opt = Adam(learning_rate=self.opt_lr) # Check optimizer
        if self.opt_patience > self.epochs: self.opt_patience = self.epochs // 10 # adjust opt_patience
        if self.stop_patience > self.epochs and self.stop_patience > 10: self.stop_patience = self.epochs // 2 # adjust stop_patience
        if VMD_PARAMS is not None and type(VMD_PARAMS) != dict: raise ValueError('Invalid input of VMD_PARAMS!') 
        


    # 1.Basic functions
    # ------------------------------------------------------
    # 1.1 Build model
    def build_model(self, trainset_shape, model_name='Keras model', model_file=None):
        """
        Build Keras model, eg. 'GRU', 'LSTM', 'DNN', 'BPNN', model = Sequential(), or load_model.
        """
        if model_file is not None and os.path.exists(str(model_file)):
            print('Load Keras model:', model_file)
            return load_model(model_file) # load user's saving custom model
        elif isinstance(self.KERAS_MODEL, Sequential): # if not load a model
            return self.KERAS_MODEL  
        elif self.KERAS_MODEL == 'LSTM':
            model = Sequential(name=model_name)
            model.add(LSTM(self.units*4, input_shape=(trainset_shape[1], trainset_shape[2]), recurrent_activation='sigmoid', 
                           activation=self.activation, return_sequences=True))
            model.add(Dropout(self.dropout))
            model.add(LSTM(self.units*2, recurrent_activation='sigmoid', activation=self.activation, return_sequences=True))
            model.add(Dropout(self.dropout))
            model.add(LSTM(self.units, recurrent_activation='sigmoid', activation=self.activation, return_sequences=False))
            model.add(Dropout(self.dropout))
            model.add(Dense(1, activation=self.activation))
            model.compile(loss=self.opt_loss, optimizer=self.opt)
            return model
        elif self.KERAS_MODEL == 'GRU':
            model = Sequential(name=model_name)
            model.add(GRU(self.units*4, input_shape=(trainset_shape[1], trainset_shape[2]), recurrent_activation='sigmoid', 
                          reset_after=True, activation=self.activation, return_sequences=True))
            model.add(Dropout(self.dropout))
            model.add(GRU(self.units*2, recurrent_activation='sigmoid', reset_after=True, activation=self.activation, return_sequences=True))
            model.add(Dropout(self.dropout))
            model.add(GRU(self.units, recurrent_activation='sigmoid', reset_after=True, activation=self.activation, return_sequences=False))
            model.add(Dropout(self.dropout))
            model.add(Dense(1, activation=self.activation))
            model.compile(loss=self.opt_loss, optimizer=self.opt)
            return model
        elif self.KERAS_MODEL == 'DNN':
            model = Sequential(name=model_name)
            model.add(Flatten(input_shape=(trainset_shape[1], trainset_shape[2])))
            for _ in range(8):
                model.add(Dense(self.units*4, activation=self.activation)) 
                # model.add(BatchNormalization())
                model.add(Dropout(self.dropout))
            model.add(Dense(self.units*2, activation=self.activation)) 
            model.add(Dropout(self.dropout))
            model.add(Dense(self.units, activation=self.activation)) 
            model.add(Dropout(self.dropout))
            model.add(Dense(1, activation=self.activation))
            model.compile(loss=self.opt_loss, optimizer=self.opt)
            return model
        elif self.KERAS_MODEL == 'BPNN':
            model = Sequential(name=model_name)
            model.add(Dense(self.units*4, input_shape=(trainset_shape[1], trainset_shape[2]), activation=self.activation))
            model.add(Dropout(self.dropout))
            model.add(Flatten())
            model.add(Dense(1, activation=self.activation))
            model.compile(loss=self.opt_loss, optimizer=self.opt)
            return model
        else: raise ValueError("%s is an invalid input for KERAS_MODEL! eg. 'GRU', 'LSTM', or model = Sequential()"%self.KERAS_MODEL)

    # 1.2 Change Keras model to TPU model (for google Colab)
    def tpu_model(self, shape, model_name, model_file):
        """
        Change Keras model to TPU model (for google Colab)
        """
        import tensorflow as tf
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        with strategy.scope(): # Change Keras model to TPU form
            model = self.build_model(shape, model_name, model_file) # Build the model # Use model.summary() to show the model structure
        return model

    # 1.3 Main Forecast by Keras
    def keras_predict(self, data=None, show_model=False, fitting_set=None, **kwargs): # GRU forecasting function
        """
        Forecast by Keras
        
        Input and Parameters:
        ---------------------
        data               - data set (include training set and test set)
        show_model         - show Keras model structure
        fitting_set        - input a fitting set to create new train and test set
                           - generally the pd.DataFrame of IMFs' forecasting results
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
        x_train, x_test, y_train, y_test, scalarY, next_x = create_train_test_set(data, self.FORECAST_LENGTH, self.FORECAST_HORIZONS, self.NEXT_DAY, self.NOR_METHOD, self.DAY_AHEAD, fitting_set) 

        # Convert to tensor = tf.constant()
        today_x = x_train[-1].reshape(1, x_train.shape[1], x_train.shape[2]) # aviod resahpe tf.constant -> tensor
        x_train = constant(x_train) # x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2])) 
        if not self.NEXT_DAY: x_test = constant(x_test) # x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2])) 
        
        # Set callbacks
        Reduce = ReduceLROnPlateau(monitor=self.callbacks_monitor, patience=self.opt_patience, verbose=self.verbose, mode='auto') # Adaptive learning rate
        EarlyStop = EarlyStopping(monitor=self.callbacks_monitor, patience=self.stop_patience, verbose=self.verbose, mode='auto') # Early stop at small learning rate 
        callbacks_list = [Reduce, EarlyStop]

        # Name model
        try: 
            for i in ['\'',' ','\"','-',':',',','.','{','}','[',']','(',')']: data.name = data.name.replace(i,'')
            if '.keras' not in str(data.name): data.name = data.name+'.keras'
        except: data.name = 'Keras_model.keras'
        
        # Load and save model
        model_file = None
        if self.PATH is not None:
            # Load model if get input of self.KERAS_MODEL
            if isinstance(self.KERAS_MODEL, dict): self.KERAS_MODEL = pd.DataFrame(self.KERAS_MODEL, index=[0])
            if isinstance(self.KERAS_MODEL, pd.DataFrame): # KERAS_MODEL eg. {'co-imf0':'co_imf0_model.keras', 'co-imf1':'co_imf1_model.keras'}
                for x in self.KERAS_MODEL.columns:
                    if (x).replace('-','_').replace(' ','_') in data.name: model_file = x # change to be key value
                if model_file is not None: model_file = self.PATH + self.KERAS_MODEL[model_file][0]
                else: raise KeyError("Cannot match an appropriate model file by the column name of pd.DataFrame. Please check KERAS_MODEL.")
            if isinstance(self.KERAS_MODEL, str) and '.keras' in str(self.KERAS_MODEL): model_file = self.PATH + self.KERAS_MODEL
            # Save model by CheckPoint with model name = data.name = df_redecom.name
            CheckPoint = ModelCheckpoint(self.PATH+data.name, monitor=self.callbacks_monitor, save_best_only=True, verbose=self.verbose, mode='auto') # Save the model to self.PATH after each epoch
            callbacks_list.append(CheckPoint) # save Keras model in .keras file eg. predictor_name+'_of_'+imf+'_model.keras'

        # Build or load the model
        if self.USE_TPU: model = self.tpu_model(x_train.shape, data.name, model_file)
        else: model = self.build_model(x_train.shape, data.name, model_file)
        if show_model: 
            print('\nInput Shape: (%d,%d)\n'%(x_train.shape[1], x_train.shape[2]))
            model.summary() # The summary of layers and parameters

        # from tensorflow.keras.utils import plot_model # To use plot_model, you need to install software graphviz
        # plot_model(model, self.PATH+data.name+'.jpg')

        # Forecast
        df_loss = pd.DataFrame({'loss': 0, 'val_loss': 0}, index=[0])
        if self.epochs != 0:
            history = model.fit(x_train, y_train, # Train the model 
                                epochs=self.epochs, 
                                batch_size=self.batch_size, 
                                validation_split=self.valid_split, 
                                verbose=self.verbose, 
                                shuffle=self.shuffle, 
                                callbacks=callbacks_list,
                                **kwargs)
            df_loss = pd.DataFrame({'loss': history.history['loss'], 'val_loss': history.history['val_loss']}, index=range(len(history.history['val_loss'])))
            # load the best one to predict when set PATH
            if self.PATH is not None: model = load_model(self.PATH+data.name) 

        # Get results and evaluate
        from CEEMDAN_LSTM.data_preprocessor import eval_result
        if not self.NEXT_DAY:
            # Get general results and evaluate
            y_predict = model(x_test) # Predict # replace y_predict = model.predict(x_test) to aviod warning
            y_predict = np.array(y_predict).ravel().reshape(-1,1) 
            if scalarY is not None: 
                y_test = scalarY.inverse_transform(y_test)
                y_predict = scalarY.inverse_transform(y_predict) # De-normalize 
            if self.TARGET is not None: result_index = self.TARGET.index[-self.FORECAST_LENGTH:] # Forecasting result idnex
            else: result_index = range(len(y_test.ravel()))
            df_eval = eval_result(y_test, y_predict) # Evaluate model
            df_result = pd.DataFrame({'real': y_test.ravel(), 'predict': y_predict.ravel()}, index=result_index) # Output
            return df_result, df_eval, df_loss
        else:
            # Get next day's results
            today_y = np.array(model(today_x))
            next_y = np.array(model(next_x.reshape(1, today_x.shape[1], today_x.shape[2])))
            if scalarY is not None: # De-normalize 
                today_real = scalarY.inverse_transform([y_train[-1]])
                today_y = scalarY.inverse_transform(today_y)
                next_y = scalarY.inverse_transform(next_y)
            else: today_real = self.TARGET.values[-1]
            if self.TARGET is not None: next_index = [self.TARGET.index[-1]] # [self.TARGET.index[-1] + (self.TARGET.index[1] - self.TARGET.index[0])]
            else: next_index = range(1)
            df_result = pd.DataFrame({'today_real': today_real.ravel()[0], 'today_pred': today_y.ravel()[0], 'next_pred': next_y.ravel()[0]}, index=next_index) # Output
            return df_result



    # 2.Advanced forecasting functions
    # ------------------------------------------------------
    # 2.1 Single Method (directly forecast)
    def single_keras_predict(self, data=None, show=False, plot=False, save=False, **kwargs):
        """
        Single Method (directly forecast)
        Use Keras model to directly forecast with vector input
        Example: 
        kr = cl.keras_predictor(epochs=10, verbose=1)
        df_result = kr.single_keras_predict(data, show=True, plot=True, save=True)

        Input and Parameters:
        ---------------------
        Note important hyperarameters of class cl.keras_predictor()
        data            - data set (include training set and test set)
        show            - show the inputting data set and Keras model structure
        plot            - show figure result or not
        save            - save forecasting result when set PATH
        **kwargs        - any parameters of self.keras_predict()

        Output
        ---------------------
        df_result = (df_predict_real, df_eval, df_train_loss) or next_y
        df_predict_real - forecasting results and the original real series set
        df_eval         - evaluating results of forecasting 
        df_train_loss   - training loss log
        next_y          - forecasting result of the next day when NEXT_DAY=Ture
        """

        # Name and set 
        now = datetime.now()
        predictor_name = name_predictor(now, 'Single', 'Keras', self.KERAS_MODEL, None, None, self.NEXT_DAY)
        data = check_dataset(data, show, self.DECOM_MODE, None)
        data.name = (predictor_name+' model.keras')
        self.TARGET = data['target']

        # Forecast
        start = time.time()
        df_result = self.keras_predict(data=data, show_model=show, **kwargs)
        end = time.time()

        # Output
        df_result = output_result(df_result, predictor_name, end-start, imf='Final', next_day=self.NEXT_DAY) # (df_result, df_eval, df_loss) for keras_predict()
        plot_save_result(df_result, name=now, plot=plot, save=save, path=self.PATH)
        return df_result



    # 2.2 Ensemble Method (decompose then directly forecast)
    def ensemble_keras_predict(self, data=None, show=False, plot=False, save=False, **kwargs):
        """
        Ensemble Method (decompose then directly forecast)
        Use decomposition-integration Keras model to directly forecast with matrix input
        Example: 
        kr = cl.keras_predictor(epochs=10, verbose=1)
        df_result = kr.ensemble_keras_predict(data, show=True, plot=True, save=True)

        Input and Parameters:
        ---------------------
        Note important hyperarameters of class cl.keras_predictor()
        data            - data set (include training set and test set)  
        show            - show the inputting data set and Keras model structure
        plot            - show figure result or not
        save            - save forecasting result when set PATH
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
        now = datetime.now()
        predictor_name = name_predictor(now, 'Ensemble', 'Keras', self.KERAS_MODEL, self.DECOM_MODE, self.REDECOM_LIST, self.NEXT_DAY)

        # Decompose, integrate, re-decompose
        start = time.time()
        from CEEMDAN_LSTM.data_preprocessor import redecom
        df_redecom, df_redecom_list = redecom(data, show, self.DECOM_MODE, self.INTE_LIST, self.REDECOM_LIST, self.VMD_PARAMS, self.FORECAST_LENGTH)
        df_redecom.name = predictor_name+'_model.keras' # model name input to self.keras_predict
        self.TARGET = df_redecom['target']

        # Forecast
        df_result = self.keras_predict(data=df_redecom, show_model=show, **kwargs)
        end = time.time()

        # Output
        df_result = output_result(df_result, predictor_name, end-start, imf='Final', next_day=self.NEXT_DAY) # return (df_result, df_eval, df_loss)
        plot_save_result(df_result, name=now, plot=plot, save=save, path=self.PATH)
        return df_result


    
    # 2.3 Respective Method (decompose then respectively forecast each IMFs)
    def respective_keras_predict(self, data=None, show=False, plot=False, save=False, **kwargs):
        """
        Respective Method (decompose then respectively forecast each IMFs)
        Use decomposition-integration Keras model to respectively forecast each IMFs with vector input
        Example: 
        kr = cl.keras_predictor(epochs=10, verbose=1)
        df_result = kr.respective_keras_predict(data, show=True, plot=True, save=True)

        Input and Parameters:
        ---------------------
        Note important hyperarameters of class cl.keras_predictor()
        data            - data set (include training set and test set)
        show            - show the inputting data set and Keras model structure
        plot            - show figure result or not
        save            - save forecasting result when set PATH
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
        now = datetime.now()
        predictor_name = name_predictor(now, 'Respective', 'Keras', self.KERAS_MODEL, self.DECOM_MODE, self.REDECOM_LIST, self.NEXT_DAY)

        # Decompose, integrate, re-decompose
        start = time.time()
        from CEEMDAN_LSTM.data_preprocessor import redecom
        df_redecom, df_redecom_list = redecom(data, show, self.DECOM_MODE, self.INTE_LIST, self.REDECOM_LIST, self.VMD_PARAMS, self.FORECAST_LENGTH)
        self.TARGET = df_redecom['target']

        # Forecast and ouput each Co-IMF
        df_pred_result = pd.DataFrame(index = self.TARGET.index[-self.FORECAST_LENGTH:0]) # df for storing forecasting result
        df_eval_result = pd.DataFrame(columns=['Scale', 'R2', 'RMSE', 'MAE', 'MAPE', 'Runtime', 'IMF']) # df for storing evaluation result
        df_next_result = pd.DataFrame(columns=['today_real', 'today_pred', 'next_pred', 'Runtime', 'IMF']) # df for storing Next-day forecasting result
        for imf in df_redecom.columns.difference(['target']):
            df_redecom[imf].name = predictor_name+'_of_'+imf+'_model.keras'
            time1 = time.time()
            df_result = self.keras_predict(data=df_redecom[imf], show_model=show, **kwargs)
            time2 = time.time()
            df_result = output_result(df_result, predictor_name, time2-time1, imf, next_day=self.NEXT_DAY) # return (imf_pred, imf_eval, imf_loss)
            if not self.NEXT_DAY:
                df_pred_result = pd.concat((df_pred_result, df_result[0]), axis=1) # add forecasting result
                df_eval_result = pd.concat((df_eval_result, df_result[1])) # add evaluation result
                plot_save_result(df_result, name=now, plot=plot, save=False, path=self.PATH) # do not save Co-IMF results
            else: df_next_result = pd.concat((df_next_result, df_result)) # add Next-day forecasting result
            print('')
        end = time.time()

        # Final Output
        if not self.NEXT_DAY: 
            df_pred_result['real'] = df_redecom['target'][-self.FORECAST_LENGTH:]
            df_pred_result['predict'] = df_pred_result[[x for x in df_pred_result.columns if 'predict' in x]].sum(axis=1)
            df_result = (df_pred_result, df_eval_result)
        else:
            df_result = pd.DataFrame({'today_real': df_redecom['target'][-1:], 'today_pred': df_next_result['today_pred'].sum(), 
                                      'next_pred': df_next_result['next_pred'].sum()}) # Output
            df_result = (df_result, df_next_result)
        df_result = output_result(df_result, predictor_name, end-start, imf='Final', next_day=self.NEXT_DAY)
        plot_save_result(df_result, name=now, plot=plot, save=save, path=self.PATH)
        return df_result 



    # 2.4 Hybrid Method (decompose then forecast high-frequency IMF by ensemble method and forecast other by respective method)
    def hybrid_keras_predict(self, data=None, show=False, plot=False, save=False, method='respective', **kwargs):
        """
        Hybrid Method (decompose then forecast high-frequency IMF by ensemble method and forecast others by respective method)
        Use the ensemble method to forecast high-frequency IMF and the respective method for other IMFs.
        Example: 
        kr = cl.keras_predictor(epochs=10, verbose=1)
        df_result = kr.hybrid_keras_predict(data, show=True, plot=True, save=True)

        Input and Parameters:
        ---------------------
        Note important hyperarameters of class cl.keras_predictor()
        data            - data set (include training set and test set)
        show            - show the inputting data set and Keras model structure
        plot            - show figure result or not
        save            - save forecasting result when set PATH
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
        now = datetime.now()
        predictor_name = name_predictor(now, 'Hybrid', 'Keras', self.KERAS_MODEL, self.DECOM_MODE, self.REDECOM_LIST, self.NEXT_DAY)
        if method != 'ensemble' and method != 'respective': raise ValueError("Please input a vaild predict method! eg. 'ensemble', 'respective'.")

        # Decompose, integrate, re-decompose
        start = time.time()
        from CEEMDAN_LSTM.data_preprocessor import redecom
        df_redecom, df_redecom_list = redecom(data, show, self.DECOM_MODE, self.INTE_LIST, self.REDECOM_LIST, self.VMD_PARAMS, self.FORECAST_LENGTH)
        self.TARGET = df_redecom['target']

        # Forecast
        df_pred_result = pd.DataFrame(index = self.TARGET.index[-self.FORECAST_LENGTH:0]) # df for storing forecasting result
        df_eval_result = pd.DataFrame(columns=['Scale', 'R2', 'RMSE', 'MAE', 'MAPE', 'Runtime', 'IMF']) # df for storing evaluation result
        df_next_result = pd.DataFrame(columns=['today_real', 'today_pred', 'next_pred', 'Runtime', 'IMF']) # df for storing Next-day forecasting result
        for df in df_redecom_list: # both ensemble (matrix-input) and respective (vector-input) method 
            imf = df.name
            if 'redecom' not in imf:
                if method=='respective': df = pd.DataFrame(df[imf]) # if method=='ensemble': df = df
            else: imf = imf[8:]
            df.rename(columns={imf: 'target'}, inplace=True)
            df.name = predictor_name+'_of_'+imf+'_model.keras' # Save model
            time1 = time.time()
            df_result = self.keras_predict(data=df, show_model=show, **kwargs) # the ensemble method with matrix input 
            time2 = time.time()
            df_result = output_result(df_result, predictor_name, time2-time1, imf, next_day=self.NEXT_DAY) # return (imf_pred, imf_eval, imf_loss)
            if not self.NEXT_DAY:
                df_pred_result = pd.concat((df_pred_result, df_result[0]), axis=1) # add forecasting result
                df_eval_result = pd.concat((df_eval_result, df_result[1])) # add evaluation result
                plot_save_result(df_result, name=now, plot=plot, save=False, path=self.PATH) # do not save Co-IMF results
            else: df_next_result = pd.concat((df_next_result, df_result)) # add Next-day forecasting result
            print('')

        # Fitting 
        if not self.NEXT_DAY: 
            df_pred_result['real'] = df_redecom['target'][-self.FORECAST_LENGTH:]
            if self.FIT_METHOD == 'ensemble': # fitting method
                print('Fitting of the ensemble method is running...')        
                fitting_set = df_pred_result[[x for x in df_pred_result.columns if '-predict' in x]]
                df_redecom.name = predictor_name+'_of_Fitting_model.keras' # Save model
                time1 = time.time()
                df_result = self.keras_predict(data=df_redecom, show_model=show, fitting_set=fitting_set, **kwargs) # the ensemble method with matrix input 
                time2 = time.time()
                df_result = output_result(df_result, predictor_name, time2-time1, imf='fitting', next_day=self.NEXT_DAY)
                df_eval_result = pd.concat((df_eval_result, df_result[1])) # add evaluation result of fitting
                df_pred_result['add-predict'] = df_pred_result[[x for x in df_pred_result.columns if '-predict' in x]].sum(axis=1) # add all imf predict result
                df_pred_result['predict'] = df_result[0]['fitting-predict']
                plot_save_result(df_result, name=now, plot=plot, save=False, path=self.PATH) # do not save Co-IMF results
            else: # adding method
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
                df_redecom.name = predictor_name+'_of_Fitting_model.keras' # Save model
                time1 = time.time()
                df_result = self.keras_predict(data=df_redecom, show_model=show, fitting_set=fitting_set, **kwargs) # the ensemble method with matrix input 
                time2 = time.time()
                df_result = output_result(df_result, predictor_name, time2-time1, imf='fitting', next_day=self.NEXT_DAY)
                df_next_result = pd.concat((df_next_result, df_result)) # add Next-day forecasting result
                today_result = df_result['today_pred']
                next_reuslt = df_result['next_pred']
            else: # adding method
                today_result = df_next_result['today_pred'].sum()
                next_reuslt = df_next_result['next_pred'].sum() # adding method
        end = time.time()

        # Final Output
        if not self.NEXT_DAY: df_result = (df_pred_result, df_eval_result)
        else:
            df_result = pd.DataFrame({'today_real': df_redecom['target'][-1:], 'today_pred': today_result, 'next_pred': next_reuslt}) # Output                
            df_result = (df_result, df_next_result)
        df_result = output_result(df_result, predictor_name, end-start, imf='Final', next_day=self.NEXT_DAY)
        plot_save_result(df_result, name=now, plot=plot, save=save, path=self.PATH)
        return df_result 



    # A class used to hide the print
    class HiddenPrints: 
        """
        A class used to hide the print
        """
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout

    # 2.5 Multiple Run Predictor
    def multiple_keras_predict(self, data=None, run_times=10, predict_method=None, **kwargs):
        """
        Multiple Run Predictor, multiple run of above method
        Example: 
        kr = cl.keras_predictor(epochs=10, verbose=1)
        df_result = kr.multiple_keras_predict(data, run_times=10, predict_method='single')

        Input and Parameters:
        ---------------------
        Note important hyperarameters of class cl.keras_predictor()
        data               - data set (include training set and test set)
        run_times          - running times
        predict_method     - run different method eg. 'single', 'ensemble', 'respective', 'hybrid'
                           - single: self.single_keras_predict()
                           - ensemble: self.ensemble_keras_predict()
                           - respective: self.respective_keras_predict()
                           - hybrid: self.hybrid_keras_predict()
        **kwargs           - any parameters of self.keras_predict()

        Output
        ---------------------
        df_eval_result     - evaluating forecasting results of each run
        """

        # Initialize 
        now = datetime.now()
        data = check_dataset(data, False, self.DECOM_MODE, self.REDECOM_LIST)
        self.TARGET = data['target']
        
        if self.PATH is None: print('Do not set a PATH! It is recommended to set a PATH to prevent the loss of running results')
        if predict_method is None: raise ValueError("Please input a predict method! eg. 'single', 'ensemble', 'respective', 'hybrid'.")
        else: predict_method = predict_method.capitalize()
        if predict_method == 'Single': predictor_name = name_predictor(now, 'Multiple Single', 'Keras', self.KERAS_MODEL, None, None, self.NEXT_DAY)
        else: predictor_name = name_predictor(now, 'Multiple '+predict_method, 'Keras', self.KERAS_MODEL, self.DECOM_MODE, self.REDECOM_LIST, self.NEXT_DAY)    

        # Forecast
        start = time.time()
        df_pred_result = pd.DataFrame(index = self.TARGET.index[-self.FORECAST_LENGTH:0]) # df for storing forecasting result
        df_eval_result = pd.DataFrame(columns=['Scale', 'R2', 'RMSE', 'MAE', 'MAPE', 'Runtime', 'IMF']) # df for storing evaluation result
        df_next_result = pd.DataFrame(columns=['today_real', 'today_pred', 'next_pred', 'Runtime', 'IMF']) # df for storing Next-day forecasting result
        for i in range(run_times):
            print('Run %d is running...'%i, end='')
            time1 = time.time()
            with self.HiddenPrints():
                if predict_method == 'Single': df_result = self.single_keras_predict(data=data, **kwargs)
                elif predict_method == 'Ensemble': df_result = self.ensemble_keras_predict(data=data, **kwargs) 
                elif predict_method == 'Respective': df_result = self.respective_keras_predict(data=data, **kwargs)
                elif predict_method == 'Hybrid': df_result = self.hybrid_keras_predict(data=data, **kwargs)
                elif predict_method == 'Hybrid-ensemble': df_result = self.hybrid_keras_predict(data=data, method='ensemble', **kwargs)
                else: raise ValueError("Please input a vaild predict method! eg. 'single', 'ensemble', 'respective', 'hybrid'.")
                time2 = time.time()
                df_result = output_result(df_result, predictor_name, time2-time1, imf='Final', run=i, next_day=self.NEXT_DAY)
                if not self.NEXT_DAY:
                    df_pred_result = pd.concat((df_pred_result, df_result[0]), axis=1) # add forecasting result
                    df_eval_result = pd.concat((df_eval_result, df_result[1])) # add evaluation result
                    df_pred_result.name, df_eval_result.name = predictor_name+' Result', predictor_name+' Evaluation'
                    df_result = (df_pred_result, df_eval_result)
                else: 
                    df_next_result = pd.concat((df_next_result, df_result)) # add Next-day forecasting result
                    df_next_result.name = predictor_name+' Result'
                    df_result = df_next_result
                plot_save_result(df_result, name=now, plot=False, save=True, path=self.PATH) # Temporary storage
            print('taking time: %.3fs'%(time2-time1))
        end = time.time()
        print('\n============='+predictor_name+' Finished=============')
        print('Running time: %.3fs'%(end-start))
        return df_result



    # 2.6 Rolling Method (forecast each value one by one)
    def rolling_keras_predict(self, data=None, predict_method=None, transfer_learning=False, out_of_sample=False, **kwargs):
        """
        Rolling Method (forecast each value one by one to avoid lookahead bias)
        Rolling run of above method to avoid the look-ahead bias, but take a long long time
        Example: 
        kr = cl.keras_predictor(epochs=10, verbose=1)
        df_result = kr.rolling_keras_predict(data, predict_method='single', transfer_learning=True)

        Input and Parameters:
        ---------------------
        Note important hyperarameters of class cl.keras_predictor()
        data               - data set (include training set and test set)
        predict_method     - run different method eg. 'single', 'ensemble', 'respective', 'hybrid'
                           - single: self.single_keras_predict()
                           - ensemble: self.ensemble_keras_predict()
                           - respective: self.respective_keras_predict()
                           - hybrid: self.hybrid_keras_predict()
        transfer_learning  - use transfer learning method, load previous model and continue training; save Keras model in .keras file
        **kwargs           - any parameters of self.keras_predict()

        Output
        ---------------------
        df_eval_result     - evaluating forecasting results of each run
        """

        # Initialize 
        self.NEXT_DAY = True
        now = datetime.now()
        start = time.time()
        data = check_dataset(data, False, self.DECOM_MODE, self.REDECOM_LIST)
        if self.PATH is None: 
            if transfer_learning: raise ValueError('Please set a PATH to start transfer Learning!')
            print('You do not set a PATH! It is recommended to set a PATH to prevent the loss of running results')
        if predict_method is None: raise ValueError("Please input a predict method! eg. 'single', 'ensemble', 'respective', 'hybrid'.")
        else: predict_method = predict_method.capitalize()
        if predict_method == 'Single': predictor_name = name_predictor(now, 'Rolling Single', 'Keras', self.KERAS_MODEL, None, None, False)
        else: predictor_name = name_predictor(now, 'Rolling '+predict_method, 'Keras', self.KERAS_MODEL, self.DECOM_MODE, self.REDECOM_LIST, False)
        
        # Transfer Learning (save and load keras model file)
        model, model_name = None, ''
        if transfer_learning: 
            print('Strat transfer Learning for Keras model.')

            # pre-training
            time1 = time.time()
            print('Pre-training is running...', end='')
            with self.HiddenPrints():
                self.TARGET = data[:-self.FORECAST_LENGTH]
                if predict_method == 'Single': df_result = self.single_keras_predict(data=data[:-self.FORECAST_LENGTH], **kwargs)
                if predict_method == 'Ensemble': df_result = self.ensemble_keras_predict(data=data[:-self.FORECAST_LENGTH], **kwargs) 
                if predict_method == 'Respective': df_result = self.respective_keras_predict(data=data[:-self.FORECAST_LENGTH], **kwargs)
                if predict_method == 'Hybrid': df_result = self.hybrid_keras_predict(data=data[:-self.FORECAST_LENGTH], **kwargs)
            time2 = time.time()
            print('taking time: %.3fs'%(time2-time1))

            # get model name and VMD_PARAMS 
            with self.HiddenPrints():
                model_name = name_predictor(now, predict_method, 'Keras', self.KERAS_MODEL, self.DECOM_MODE, self.REDECOM_LIST, self.NEXT_DAY)
            from CEEMDAN_LSTM.data_preprocessor import redecom
            if predict_method == 'Single': model = (model_name).replace('-','_').replace(' ','_')+'_model.keras'
            else:
                df_redecom = None
                if self.INTE_LIST is None: print('Warning! It is recommended to set INTE_LIST. Otherwise, a different number of decomposed IMFs per run may lead to errors.')
                if self.REDECOM_LIST is not None:
                    if 'vmd' not in str(self.REDECOM_LIST).lower(): print('Warning! It is recommended to use VMD or OVMD in REDECOM_LIST. Otherwise, a different number of decomposed IMFs per run may lead to errors.')
                    model = pd.DataFrame(index=[0])
                    if predict_method == 'Ensemble': 
                        model = (model_name).replace('-','_').replace(' ','_')+'_model.keras'
                        if self.REDECOM_LIST is not None: df_redecom, df_redecom_list = redecom(data, False, self.DECOM_MODE, self.INTE_LIST, self.REDECOM_LIST, self.FORECAST_LENGTH)
                    elif predict_method == 'Respective': 
                        df_redecom, df_redecom_list = redecom(data, False, self.DECOM_MODE, self.INTE_LIST, self.REDECOM_LIST, self.FORECAST_LENGTH)
                        for imf in df_redecom.columns.difference(['target']): model[imf] = (model_name+'_of_'+imf+'_model.keras').replace('-','_').replace(' ','_')
                    elif predict_method == 'Hybrid': 
                        df_redecom, df_redecom_list = redecom(data, False, self.DECOM_MODE, self.INTE_LIST, self.REDECOM_LIST, self.FORECAST_LENGTH)
                        for imf in eval(df_redecom.name.split('_')[0]): model[imf] = (model_name+'_of_'+imf+'_model.keras').replace('-','_').replace(' ','_')
                    self.VMD_PARAMS = eval(df_redecom.name.split('_')[1])
                    if self.VMD_PARAMS is not None: print('Get vmd_params:', self.VMD_PARAMS)

        # Forecast
        df_next_result = pd.DataFrame(columns=['today_real', 'today_pred', 'next_pred', 'Runtime', 'Run']) # df for storing Next-day forecasting result
        series_next_forecast = [] # record the Next-day forecasting series
        for i in range(self.FORECAST_LENGTH):
            print('Run %d is running...'%i, end='')
            time1 = time.time()
            with self.HiddenPrints():
                if i == 0:
                    if model is not None: self.KERAS_MODEL = model
                transfer_data = data[:-self.FORECAST_LENGTH+i]
                self.TARGET = transfer_data.copy(deep=True)
                if out_of_sample and i != 0: transfer_data['target'][-i:] = series_next_forecast
                transfer_data.name = (model_name).replace('-','_').replace(' ','_')
                if predict_method == 'Single': df_result = self.single_keras_predict(data=transfer_data, **kwargs)
                if predict_method == 'Ensemble': df_result = self.ensemble_keras_predict(data=transfer_data, **kwargs) 
                if predict_method == 'Respective': df_result = self.respective_keras_predict(data=transfer_data, **kwargs)
                if predict_method == 'Hybrid': df_result = self.hybrid_keras_predict(data=transfer_data, **kwargs)
            time2 = time.time()
            print('taking time: %.3fs'%(time2-time1))
            df_result['Run'] = i
            df_result['today_real'][0] = self.TARGET[-1:].values
            print(df_result)
            df_next_result = pd.concat((df_next_result, df_result)) # add evaluation result
            df_next_result.name = predictor_name+' Running Log'
            series_next_forecast.append(df_result['next_pred'][0])
            plot_save_result(df_next_result, name=now, plot=False, save=True, path=self.PATH)
        end = time.time()

        print('\n============='+predictor_name+' Finished=============')
        print('Running time: %.3fs'%(end-start))
        from CEEMDAN_LSTM.data_preprocessor import eval_result
        df_pred_result = pd.DataFrame()
        df_pred_result['real'] = data[-self.FORECAST_LENGTH:]
        if predict_method == 'Single' or predict_method == 'Ensemble': df_pred_result['predict'] = df_next_result['next_pred'].values
        else: df_pred_result['predict'] = df_next_result.loc[df_next_result['IMF'] == 'Final']['next_pred'].values
        df_eval_result = eval_result(df_pred_result['real'], df_pred_result['predict'])
        df_eval_result['Runtime'] = end-start # Output Runtime
        df_eval_result['Run'] = 'Final'
        print(df_eval_result)
        df_result = (df_pred_result, df_eval_result, df_next_result)
        df_pred_result.name, df_eval_result.name = predictor_name+' Result', predictor_name+' Evaluation'
        plot_save_result(df_result[:2], name=now, plot=True, save=True, path=self.PATH)
        return df_result