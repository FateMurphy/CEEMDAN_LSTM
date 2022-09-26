#!/usr/bin/env python
# coding: utf-8
#
# Module: sklearn_predictor
# Description: Forecast time series by sklearn.
#
# Created: 2021-10-1 22:37
# Updated: 2022-9-18 17:28
# Author: Feite Zhou
# Email: jupiterzhou@foxmail.com
# URL: 'http://github.com/FateMurphy/CEEMDAN_LSTM'
# Feel free to email me if you have any questions or error reports.

"""
References:
Scikit-learn(sklearn):
    scikit-learn: https://github.com/scikit-learn/scikit-learn
"""

# Import modules for sklearn_predictor
import os
import sys
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore") # Ignore some annoying warnings
# CEEMDAN_LSTM
from CEEMDAN_LSTM.core import check_dataset, check_path, plot_save_result, name_predictor
# Sklearn
import matplotlib as plt
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV

class sklearn_predictor:
    # 0.Initialize
    # ------------------------------------------------------
    def __init__(self, PATH=None, FORECAST_HORIZONS=30, FORECAST_LENGTH=30, SKLEARN_MODEL='SVM', DECOM_MODE='CEEMDAN', INTE_LIST=None, REDECOM_LIST={'co-imf0':'vmd'}, 
                 NEXT_DAY=False, DAY_AHEAD=1, NOR_METHOD='minmax', FIT_METHOD='add', **kwargs):
        """
        Initialize the sklearn_predictor.
        Configuration can be passed as kwargs (keyword arguments).

        Input and HyperParameters:
        ---------------------
        PATH               - the saving path of figures and logs
        FORECAST_HORIZONS  - also called Timestep or Forecast_horizons or sliding_windows_length in some papers
                           - the length of each input row(x_train.shape), which means the number of previous days related to today
        FORECAST_LENGTH    - the length of the days to forecast (test set)
        SKLEARN_MODEL      - the Sklearn model, eg. 'LASSO', 'SVM', 'LightGBM' ...
        DECOM_MODE         - the decomposition method, eg.'EMD', 'VMD', 'CEEMDAN'
        INTE_LIST          - the integration list, eg. pd.Dataframe, (int) 3, (str) '233', (list) [0,0,1,1,1,2,2,2], ...
        REDECOM_LIST       - the re-decomposition list, eg. '{'co-imf0':'vmd', 'co-imf1':'emd'}', pd.DataFrame
        NEXT_DAY           - set True to only predict next out-of-sample value
        DAY_AHEAD          - define to forecast n days' ahead, eg. 0, 1 (default int 1)
        NOR_METHOD         - the normalizing method, eg. 'minmax'-MinMaxScaler, 'std'-StandardScaler, otherwise without normalization
        FIT_METHOD         - the fitting method to stablize the forecasting result (not necessarily useful), eg. 'add', 'ensemble'

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
        self.SKLEARN_MODEL = str(SKLEARN_MODEL)
        self.DECOM_MODE = str(DECOM_MODE)
        self.INTE_LIST = INTE_LIST
        self.REDECOM_LIST = REDECOM_LIST
        self.NEXT_DAY = bool(NEXT_DAY)
        self.DAY_AHEAD = int(DAY_AHEAD)
        self.NOR_METHOD = str(NOR_METHOD)
        self.FIT_METHOD = str(FIT_METHOD)

        self.TARGET = None
        self.REDECOM_MODE = None
        self.FIG_PATH = None
        self.LOG_PATH = None


        # Check parameters
        self.PATH, self.FIG_PATH, self.LOG_PATH = check_path(PATH) # Check PATH

        if self.FORECAST_HORIZONS <= 0: raise ValueError("Invalid input for FORECAST_HORIZONS! Please input a positive integer >0.")
        if self.FORECAST_LENGTH <= 0: raise ValueError("Invalid input for FORECAST_LENGTH! Please input a positive integer >0.")
        if self.DAY_AHEAD < 0: raise ValueError("Invalid input for DAY_AHEAD! Please input a integer >=0.")

        if type(SKLEARN_MODEL) == str: self.SKLEARN_MODEL = SKLEARN_MODEL.upper()
        else: raise ValueError("Invalid input for SKLEARN_MODEL! Please input eg. 'SVM', 'LGB'.")

        if type(DECOM_MODE) == str: self.DECOM_MODE = str(DECOM_MODE).upper() # Check DECOM_MODE

        if REDECOM_LIST is not None:# Check REDECOM_LIST and Input REDECOM_MODE
            REDECOM_MODE = ''
            try:
                REDECOM_LIST = pd.DataFrame(REDECOM_LIST, index=range(1)) 
                for i in range(REDECOM_LIST.size): REDECOM_MODE = REDECOM_MODE+REDECOM_LIST.values.ravel()[i]+REDECOM_LIST.columns[i][-1]+'-'
            except: raise ValueError("Invalid input for redecom_list! Please input eg. None, '{'co-imf0':'vmd', 'co-imf1':'emd'}'.")
            self.REDECOM_MODE = str(REDECOM_MODE).upper()
        else: REDECOM_MODE = None

        if self.NEXT_DAY == True:
            print('Sklearn predictor does not support next day. Maybe update in the future.')
            self.NEXT_DAY = False

    

    # SVM (Support Vector Machine Regression)
    def svm_predict(self, data=None, grid_search_times=5, show_data=False, plot_result=False, save_result=False):
        """
        Single Method (directly forecast)
        Use keras model to directly forecast wiht vector input
        Example: 
        sr = cl.sklearn_predictor()
        df_result = kr.svm_predict(data, show_data=True, plot_result=True)

        Input and Parameters:
        ---------------------
        Note important hyperarameters of class cl.sklearn_predictor()
        data               - data set (include training set and test set)
        grid_search_times  - grid search times
        show_data          - show the inputting data set
        plot_result        - show figure result or not
        save_result        - save forecasting result or not

        Output
        ---------------------
        df_result = (df_predict_real, df_eval) or next_y
        df_predict_real   - forecasting results and the original real series set
        df_eval           - evaluating results of forecasting 
        next_y            - forecasting result of the next day when NEXT_DAY=Ture
        """

        # Name and set target
        self.SKLEARN_MODEL = 'SVM'
        now = pd.datetime.now()
        predictor_name = name_predictor(now, '', 'Sklearn', self.SKLEARN_MODEL, None, None, False)
        data = check_dataset(data, show_data)
        self.TARGET = data['target']

        # Divide the training and test set
        from CEEMDAN_LSTM.data_preprocessor import normalize_dataset, eval_result
        start = time.time()
        trainX, trainY, scalarY = normalize_dataset(data, self.FORECAST_LENGTH)
        trainX = trainX.reshape((trainX.shape[0], trainX.shape[1]))
        x_train, x_test = trainX[:-self.FORECAST_LENGTH], trainX[-self.FORECAST_LENGTH:]
        y_train, y_test = trainY[:-self.FORECAST_LENGTH], trainY[-self.FORECAST_LENGTH:]

        # Grid Search of K-Fold CV
        # logspace(a,b,N)Divide the interval from the a power of 10 to the b power of 10 into N parts
        C_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)
        best_gamma, best_C = 0, 0
        for i in range(grid_search_times):
            param_grid = dict(gamma=gamma_range, C=C_range)
            grid = GridSearchCV(SVR(), param_grid=param_grid, cv=10)
            grid.fit(x_train, y_train)
            print('Iteration',i)
            print('Best parameters:', grid.best_params_)
            if best_gamma == grid.best_params_['gamma'] and best_C == grid.best_params_['C']: break
            best_gamma = grid.best_params_['gamma']
            best_C = grid.best_params_['C']
            gamma_range = np.append(np.linspace(best_gamma/10,best_gamma*0.9,9), np.linspace(best_gamma,best_gamma*10,10)).ravel()
            C_range = np.append(np.linspace(best_C/10,best_C*0.9,9), np.linspace(best_C,best_C*10,10)).ravel()

        # Predict
        clf = SVR(kernel='rbf', gamma=best_gamma, C=best_C)
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        end = time.time()

        # De-normalize and Evaluate
        y_predict = y_predict.ravel().reshape(-1,1) 
        if scalarY is not None: 
            y_test = scalarY.inverse_transform(y_test)
            y_predict = scalarY.inverse_transform(y_predict) # De-normalize           
        if self.TARGET is not None: result_index = self.TARGET.index[-self.FORECAST_LENGTH:] # Forecasting result idnex
        else: result_index = range(len(y_test.ravel()))
        df_eval = eval_result(y_predict, y_test)
        df_eval['Runtime'] = end-start
        df_predict_result = pd.DataFrame({'real': y_test.ravel(), 'predict': y_predict.ravel()}, index=result_index) # Output

        # Plot observation figures
        print('\n=========='+predictor_name+' Finished==========')
        df_predict_result.name, df_eval.name = predictor_name+' Result', predictor_name+' Evaluation'
        print(df_eval) # print df_eval
        df_result = (df_predict_result, df_eval)
        plot_save_result(df_result, name=now, plot=plot_result, save=save_result ,path=self.PATH)
        return df_result


    '''
    def lgb_predictor():
        # Bayesian Optimization LightGBM
        import gc
        import lightgbm as lgb
        from bayes_opt import BayesianOptimization
        from sklearn.model_selection import KFold

        # Refernece: https://github.com/jia-zhuang/xgboost-lightgbm-hyperparameter-tuning
        def lgb_cv(data, target, max_depth, num_leaves, min_data_in_leaf, bagging_freq, bagging_fraction, feature_fraction, lambda_l1, lambda_l2):
            folds = KFold(n_splits=4, shuffle=True, random_state=11)
            oof = np.zeros(data.shape[0])
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(data, target)):
                trn_data = lgb.Dataset(data[trn_idx], label=target[trn_idx])
                val_data = lgb.Dataset(data[val_idx], label=target[val_idx])
                param = {
                    # general parameters
                    'objective': 'regression',
                    'boosting': 'gbdt',
                    'metric': 'rmse',
                    'learning_rate': 0.01,
                    # tuning parameters
                    'num_leaves': int(num_leaves),
                    'min_data_in_leaf': int(min_data_in_leaf),
                    'bagging_freq': 1,
                    'bagging_fraction': bagging_fraction,
                    'feature_fraction': feature_fraction,
                    'lambda_l1': lambda_l1
                }
                clf = lgb.train(param, trn_data, 1000, valid_sets=[trn_data, val_data], verbose_eval=10, early_stopping_rounds=50)
                oof[val_idx] = clf.predict(data[val_idx], num_iteration=clf.best_iteration)
                del clf, trn_idx, val_idx
                gc.collect()
            return -mean_squared_error(target, oof)**0.5

        def optimize_lgb(data, target, feature_name='auto', categorical_feature='auto'):
            def lgb_crossval(max_depth, num_leaves, min_data_in_leaf, bagging_freq, bagging_fraction, feature_fraction, lambda_l1, lambda_l2):
                return lgb_cv(data, target, max_depth, num_leaves, min_data_in_leaf, bagging_freq, bagging_fraction, feature_fraction, lambda_l1, lambda_l2)
    
            optimizer = BayesianOptimization(lgb_crossval, {
                'max_depth': (3, 11),
                'num_leaves': (5, 128),
                'min_data_in_leaf': (1, 50),
                'bagging_freq': (1, 20),
                'bagging_fraction': (0.5, 1.0),
                'feature_fraction': (0.5, 1.0),
                'lambda_l1': (0, 2),
                'lambda_l2': (0, 2)
                # "max_depth":hp.choice("max_depth",range(3,11)),
                # "min_data_in_leaf":hp.choice("min_data_in_leaf",range(1,50)),
                # "min_gain_to_split":hp.uniform("min_gain_to_split",0.0,1.0),
                # "reg_alpha": hp.uniform("reg_alpha", 0, 2),
                # "reg_lambda": hp.uniform("reg_lambda", 0, 2),
                # "feature_fraction":hp.uniform("feature_fraction",0.5,1.0),
                # "bagging_fraction":hp.uniform("bagging_fraction",0.5,1.0),
                # "bagging_freq":hp.choice("bagging_freq",range(1,20))
            }, verbose=2)

            optimizer.maximize(init_points=5, n_iter=100, acq='ucb', kappa=10)
            print("Final result:", optimizer.max)
            return optimizer.max

        def LGB_predict(data=None, sliding_windows_length=30, fitting=None): # GRU forecasting function
            trainX,trainY,scalarY,next_x = create_train_test_set(data, co_imf_predict_for_fitting=fitting) # Get training and test X Y
            trainX = trainX.reshape((trainX.shape[0], trainX.shape[1]*trainX.shape[2]))
            trainY = trainY.ravel()
            x_train,x_test = trainX[:-sliding_windows_length],trainX[-sliding_windows_length:] # Split training and test set
            y_train,y_test = trainY[:-sliding_windows_length],trainY[-sliding_windows_length:]
            lgb_train = lgb.Dataset(x_train, y_train)
            lgb_test = lgb.Dataset(x_test, y_test, reference=lgb_train)

            """
            best_parameters = {
            'target': -0.01,
            'params':{
            # general parameters
            'objective': 'regression',
            'boosting': 'gbdt',
            'metric': 'rmse',
            'learning_rate': 0.01,
            # tuning parameters
            'max_depth': 5.289,
            'num_leaves': 19.88,
            'min_data_in_leaf': 82.93,
            'bagging_freq': 7.054,
            'bagging_fraction': 0.5059,
            'feature_fraction': 0.702,
            'lambda_l1': 0.2507,
            'lambda_l2': 1.344}
            }
            """

            best_parameters = optimize_lgb(x_train, y_train)
            best_parameters['params']['max_depth'] = int(best_parameters['params']['max_depth'])
            best_parameters['params']['num_leaves'] = int(best_parameters['params']['num_leaves'])
            best_parameters['params']['min_data_in_leaf'] = int(best_parameters['params']['min_data_in_leaf'])
            best_parameters['params']['bagging_freq'] = int(best_parameters['params']['bagging_freq'])

            lgb_train = lgb.Dataset(x_train, y_train)
            lgb_test = lgb.Dataset(x_test, y_test, reference=lgb_train)

            lgb_predictor = lgb.train(best_parameters['params'], lgb_train, 1000)
            y_test_predict = lgb_predictor.predict(x_test) # Predict

            df_lgb_evaluation = evaluation_model(y_test, y_test_predict) # Evaluate model
            y_test_predict = y_test_predict.ravel().reshape(-1,1) 
            y_test_predict_result = scalarY.inverse_transform(y_test_predict) # De-normalize 
            y_test_raw = scalarY.inverse_transform(y_test.reshape(-1,1) )    
            df_predict_raw = pd.DataFrame({'raw': y_test_raw.ravel(), 'predict': y_test_predict_result.ravel()}, index=range(len(y_test_raw))) # Output
            return df_predict_raw, df_lgb_evaluation

        # Final result: {'target': -0.010294956335114456, 'params': {'bagging_fraction': 0.5, 
        # 'feature_fraction': 0.5, 'lambda_l1': 0.0, 'min_data_in_leaf': 10.0, 'num_leaves': 104.67659148016335}}
        # 'params': {'bagging_fraction': 0.7239462869377233, 'bagging_freq': 1.0228441079058146, 'colsample_bytree': 0.946124585759883, 'lambda_l1': 0.21648102358688573, 'lambda_l2': 0.6271262272239254, 'max_depth': 24.428874475939505, 'num_leaves': 38.34681839143175}}

        if __name__ == "__main__":
            start = time.time()
            from CEEMDAN_LSTM.core import load_dataset
            df_raw_data = load_dataset()
            df_raw_data = df_raw_data.drop(['date','code','adjustflag','tradestatus','isST'], axis=1)
            df_raw_data.rename(columns={'close':'target'},inplace=1)
            df_predict_raw, df_lgb_evaluation = LGB_predict(df_raw_data)
            df_predict_raw.plot(title=CODE+' Training Loss')
            end = time.time()
            print('Total Running time: %.3fs'%(end-start))
            print(df_lgb_evaluation)
    '''