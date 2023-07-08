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
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") # Ignore some annoying warnings
# CEEMDAN_LSTM
from CEEMDAN_LSTM.core import check_dataset, check_path, plot_save_result, name_predictor
# Sklearn
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

class sklearn_predictor:
    # 0.Initialize
    # ------------------------------------------------------
    def __init__(self, PATH=None, FORECAST_HORIZONS=30, FORECAST_LENGTH=30, SKLEARN_MODEL='SVM', OPTIMIZER='Bayes',
                 DECOM_MODE='VMD', INTE_LIST='auto', REDECOM_LIST=None, 
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
        SKLEARN_MODEL      - the Sklearn model, eg. 'LASSO', 'SVM', 'LGB' ...
        OPTIMIZER          - the Sklearn model optimizer, eg. 'BAYES', 'GS' ...
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
        self.OPTIMIZER = str(OPTIMIZER)
        self.DECOM_MODE = str(DECOM_MODE)
        self.INTE_LIST = INTE_LIST
        self.REDECOM_LIST = REDECOM_LIST
        self.NEXT_DAY = bool(NEXT_DAY)
        self.DAY_AHEAD = int(DAY_AHEAD)
        self.NOR_METHOD = str(NOR_METHOD)
        self.FIT_METHOD = str(FIT_METHOD)
        
        self.TARGET = None
        self.VMD_PARAMS = None

        # Check parameters
        self.PATH, self.FIG_PATH, self.LOG_PATH = check_path(PATH) # Check PATH
        if self.FORECAST_HORIZONS <= 0: raise ValueError("Invalid input for FORECAST_HORIZONS! Please input a positive integer >0.")
        if self.FORECAST_LENGTH <= 0: raise ValueError("Invalid input for FORECAST_LENGTH! Please input a positive integer >0.")
        if self.DAY_AHEAD < 0: raise ValueError("Invalid input for DAY_AHEAD! Please input a integer >=0.")

        if type(SKLEARN_MODEL) == str: 
            self.SKLEARN_MODEL = SKLEARN_MODEL.upper()
            if self.SKLEARN_MODEL not in ['LASSO', 'SVM', 'LGB']: raise ValueError("Invalid input for SKLEARN_MODEL! Please input eg. 'LASSO', 'SVM', 'LGB'.")
        if type(OPTIMIZER) == str: 
            self.OPTIMIZER = OPTIMIZER.upper()
            if self.OPTIMIZER not in ['BAYES', 'GS']: raise ValueError("Invalid input for OPTIMIZER! Please input eg. 'BAYES', 'GS'.")
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



    # Lasso (Least Absolute Shrinkage and Selection Operator regression)
    def lasso_predict(self, data=None, optimizer='GridSearch', trials=100, show=False, plot=False, save=False):
        """
        Single Method (directly forecast)
        Use sklearn Lasso to directly forecast wiht vector input
        Example: 
        sr = cl.sklearn_predictor()
        df_result = sr.lasso_predict(data, show=True, plot=True)

        Input and Parameters:
        ---------------------
        Note important hyperparameters of class cl.sklearn_predictor()
        data               - data set (include training set and test set)
        optimzer           - optimizer of model, eg. GridSearch and Bayes
        trials             - optimizer search times, trials//10 for GridSearch
        show               - show the inputting data set
        plot               - show figure result or not
        save               - save forecasting result or not

        Output
        ---------------------
        df_result = (df_predict_real, df_eval) or next_y
        df_predict_real   - forecasting results and the original real series set
        df_eval           - evaluating results of forecasting 
        next_y            - forecasting result of the next day when NEXT_DAY=Ture
        """

        # Name and set target
        now = datetime.now()
        predictor_name = name_predictor(now, '', 'Sklearn', 'LASSO', None, None, False)
        data = check_dataset(data, show)
        self.TARGET = data['target']

        # Divide the training and test set
        from CEEMDAN_LSTM.data_preprocessor import create_train_test_set, eval_result
        start = time.time()
        x_train, x_test, y_train, y_test, scalarY, next_x = create_train_test_set(data, self.FORECAST_LENGTH, self.FORECAST_HORIZONS, self.NEXT_DAY, self.NOR_METHOD, self.DAY_AHEAD) 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]))

        # Grid Search of K-Fold CV
        # logspace(a,b,N)Divide the interval from the a power of 10 to the b power of 10 into N parts
        if self.OPTIMIZER == 'GS':
            alpha_range = np.logspace(-9, 3, 15)
            best_alpha = 1
            for i in range(trials//10):
                param_grid = dict(alpha=alpha_range)
                grid = GridSearchCV(Lasso(max_iter=10000), param_grid=param_grid, cv=10)
                grid.fit(x_train, y_train)
                print('Iteration',i)
                print('Best parameters:', grid.best_params_)
                if best_alpha == grid.best_params_['alpha']: break
                best_alpha = grid.best_params_['alpha']
                alpha_range = np.append(np.linspace(best_alpha/10,best_alpha*0.9,9), np.linspace(best_alpha,best_alpha*10,10)).ravel()
        
        # Bayes optimization
        if self.OPTIMIZER == 'BAYES':
            try: import optuna
            except: raise ImportError('Cannot import optuna, run: pip install optuna!')
            def objective(trial):
                alpha = trial.suggest_float('alpha', 0, 1) 
                clf = Lasso(alpha=alpha).fit(x_train, y_train)
                score = cross_val_score(clf, x_train, y_train, cv=5, scoring='r2')
                return score.mean() 
            study = optuna.create_study(study_name='LASSO alpha', direction='maximize') # TPESampler is used
            if not show:optuna.logging.set_verbosity(optuna.logging.WARNING) # not to print
            study.optimize(objective, n_trials=trials, n_jobs=-1, gc_after_trial=True)  # number of iterations
            best_alpha = study.best_params['alpha']

        # Predict
        clf = Lasso(alpha=best_alpha)
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
        plot_save_result(df_result, name=now, plot=plot, save=save ,path=self.PATH)
        return df_result




    # SVM (Support Vector Machine Regression)
    def svm_predict(self, data=None, optimizer='GridSearch', trials=100, show=False, plot=False, save=False):
        """
        Single Method (directly forecast)
        Use sklearn SVR to directly forecast wiht vector input
        Example: 
        sr = cl.sklearn_predictor()
        df_result = sr.svm_predict(data, show=True, plot=True)

        Input and Parameters:
        ---------------------
        Note important hyperparameters of class cl.sklearn_predictor()
        data               - data set (include training set and test set)
        optimzer           - optimizer of model, eg. GridSearch and Bayes
        trials             - optimizer search times, trials//10 for GridSearch
        show               - show the inputting data set
        plot               - show figure result or not
        save               - save forecasting result or not

        Output
        ---------------------
        df_result = (df_predict_real, df_eval) or next_y
        df_predict_real   - forecasting results and the original real series set
        df_eval           - evaluating results of forecasting 
        next_y            - forecasting result of the next day when NEXT_DAY=Ture
        """

        # Name and set target
        now = datetime.now()
        predictor_name = name_predictor(now, '', 'Sklearn', 'SVM', None, None, False)
        data = check_dataset(data, show)
        self.TARGET = data['target']

        # Divide the training and test set
        from CEEMDAN_LSTM.data_preprocessor import create_train_test_set, eval_result
        start = time.time()
        x_train, x_test, y_train, y_test, scalarY, next_x = create_train_test_set(data, self.FORECAST_LENGTH, self.FORECAST_HORIZONS, self.NEXT_DAY, self.NOR_METHOD, self.DAY_AHEAD) 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]))

        # Grid Search of K-Fold CV
        # logspace(a,b,N)Divide the interval from the a power of 10 to the b power of 10 into N parts
        if self.OPTIMIZER == 'GS':
            C_range = np.logspace(-2, 10, 13)
            gamma_range = np.logspace(-9, 3, 13)
            best_gamma, best_C = 0, 0
            for i in range(trials//10):
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
        
        # Bayes optimization
        if self.OPTIMIZER == 'BAYES':
            try: import optuna
            except: raise ImportError('Cannot import optuna, run: pip install optuna!')
            def objective(trial):
                # kernel = trial.suggest_categorical('kernel', ['linear','rbf','poly','sigmoid'])
                gamma = trial.suggest_loguniform('gamma',1e-5,1e5)
                C = trial.suggest_loguniform('C',1e-5,1e5)
                epsilon = trial.suggest_loguniform('epsilon',1e-5,1e5)
                clf = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon).fit(x_train, y_train)
                score = cross_val_score(clf, x_train, y_train, cv=5, scoring='r2')
                return score.mean() 
            study = optuna.create_study(study_name='SVR C gamma epsilon', direction='maximize') # TPESampler is used
            if not show: optuna.logging.set_verbosity(optuna.logging.WARNING) # not to print
            study.optimize(objective, n_trials=trials, n_jobs=-1, gc_after_trial=True)  # number of iterations
            # best_kernel = study.best_params['kernel']
            best_gamma = study.best_params['gamma']
            best_C = study.best_params['C']
            best_epsilon = study.best_params['epsilon']

        # Predict
        clf = SVR(kernel='rbf', gamma=best_gamma, C=best_C, epsilon=best_epsilon)
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
        plot_save_result(df_result, name=now, plot=plot, save=save ,path=self.PATH)
        return df_result



    # LGB or LightGBM (Light Gradient Boosting Machine)
    def lgb_predict(self, data=None, optimizer='Bayes', trials=100, show=False, plot=False, save=False):
        """
        Single Method (directly forecast)
        Use LGB to directly forecast wiht vector input
        Example: 
        sr = cl.sklearn_predictor()
        df_result = sr.lgb_predict(data, show=True, plot=True)

        Input and Parameters:
        ---------------------
        Note important hyperparameters of class cl.sklearn_predictor()
        data               - data set (include training set and test set)
        optimzer           - optimizer of model, eg. GridSearch and Bayes
        trials             - optimizer search times, trials//10 for GridSearch
        show               - show the inputting data set
        plot               - show figure result or not
        save               - save forecasting result or not

        Output
        ---------------------
        df_result = (df_predict_real, df_eval) or next_y
        df_predict_real   - forecasting results and the original real series set
        df_eval           - evaluating results of forecasting 
        next_y            - forecasting result of the next day when NEXT_DAY=Ture
        """

        # Name and set target
        try: import lightgbm as lgb
        except: raise ImportError('Cannot import lightgbm, run: pip install lightgbm!')

        self.SKLEARN_MODEL = 'LGB'
        now = datetime.now()
        predictor_name = name_predictor(now, '', 'Sklearn', 'LGB', None, None, False)
        data = check_dataset(data, show)
        self.TARGET = data['target']

        # Divide the training and test set
        from CEEMDAN_LSTM.data_preprocessor import create_train_test_set, eval_result
        start = time.time()
        x_train, x_test, y_train, y_test, scalarY, next_x = create_train_test_set(data, self.FORECAST_LENGTH, self.FORECAST_HORIZONS, self.NEXT_DAY, self.NOR_METHOD, self.DAY_AHEAD) 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]))
        # lgb_train = lgb.Dataset(x_train, y_train)
        # lgb_test = lgb.Dataset(x_test, y_test, reference=lgb_train)

        # Grid Search of K-Fold CV
        # logspace(a,b,N)Divide the interval from the a power of 10 to the b power of 10 into N parts
        if self.OPTIMIZER == 'GS':
            print('Grid search is not recommended. Change to Bayes optimization...')
            self.OPTIMIZER = 'BAYES'
        
        # Bayes optimization
        if self.OPTIMIZER == 'BAYES':
            try: import optuna
            except: raise ImportError('Cannot import optuna, run: pip install optuna!')
            def objective(trial):
                param_grid = {
                    'metric': 'rmse', 
                    'random_state': 48,
                    'n_estimators': 20000,
                    'max_depth': trial.suggest_categorical('max_depth', [3, 4, 5]),
                    'num_leaves': trial.suggest_categorical('num_leaves', [5, 6, 7, 12, 13, 14, 15, 28, 29, 30, 31]),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
                    'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
                    'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.6,0.7,0.8,0.9,1.0]),
                    'subsample': trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0]),
                }
                clf = lgb.LGBMRegressor(**param_grid).fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=100, verbose=False)
                score = cross_val_score(clf, x_train, y_train, cv=5, scoring='r2')
                return score.mean() 
            study = optuna.create_study(study_name='LGB hyperparameters', direction='maximize') # TPESampler is used
            if not show: optuna.logging.set_verbosity(optuna.logging.WARNING) # not to print
            study.optimize(objective, n_trials=100, n_jobs=-1, gc_after_trial=True)  # number of iterations
            best_params = study.best_params

        # Predict
        clf = lgb.LGBMRegressor(**best_params)
        clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=100, verbose=False)
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
        plot_save_result(df_result, name=now, plot=plot, save=save ,path=self.PATH)
        return df_result



    # 2.3. Respective Method (decompose then respectively forecast each IMFs)
    def respective_sklearn_predict(self, data=None, show=False, plot=False, save=False, **kwargs):
        """
        Respective Method (decompose then respectively forecast each IMFs)
        Use decomposition-integration Sklearn model to respectively forecast each IMFs with vector input
        Example: 
        sr = cl.sklearn_predictor()
        df_result = sr.respective_sklearn_predict(data, show=True, plot=True, save=False)

        Input and Parameters:
        ---------------------
        Note important hyperarameters of class cl.sklearn_predictor()
        data            - data set (include training set and test set)
        show            - show the inputting data set and Sklearn model structure
        plot            - show figure result or not
        save            - save forecasting result when set PATH
        **kwargs        - any parameters of self.sklearn_predictor()

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
        predictor_name = name_predictor(now, 'Respective', 'Sklearn', self.SKLEARN_MODEL, self.DECOM_MODE, self.REDECOM_LIST, self.NEXT_DAY)

        # Decompose, integrate, re-decompose
        start = time.time()
        from CEEMDAN_LSTM.data_preprocessor import redecom
        df_redecom, df_redecom_list = redecom(data, show, self.DECOM_MODE, self.INTE_LIST, self.REDECOM_LIST, self.VMD_PARAMS, self.FORECAST_LENGTH)
        
        # set target and model
        try: 
            if 'Sklearn_Forecasting' in data.name: predictor_name = data.name
            else:
                data.name = ''
                self.TARGET = df_redecom['target']
        except: 
            data.name = ''
            self.TARGET = df_redecom['target']

        # Forecast and ouput each Co-IMF
        df_pred_result = pd.DataFrame(index = self.TARGET.index[-self.FORECAST_LENGTH:0]) # df for storing forecasting result
        df_eval_result = pd.DataFrame(columns=['Scale', 'R2', 'RMSE', 'MAE', 'MAPE', 'Runtime', 'IMF']) # df for storing evaluation result
        df_next_result = pd.DataFrame(columns=['today_real', 'today_pred', 'next_pred', 'Runtime', 'IMF']) # df for storing Next-day forecasting result
        for imf in df_redecom.columns.difference(['target']):
            df_redecom[imf].name = (predictor_name+'_of_'+imf+'_model.h5').replace('-','_').replace(' ','_')
            time1 = time.time()
            with self.HiddenPrints():
                if self.SKLEARN_MODEL == 'LASSO': df_result = self.lasso_predict(data=df_redecom[imf], **kwargs)
                if self.SKLEARN_MODEL == 'SVM': df_result = self.svm_predict(data=df_redecom[imf], **kwargs) 
                if self.SKLEARN_MODEL == 'LGB': df_result = self.lgb_predict(data=df_redecom[imf], **kwargs)
            time2 = time.time()
            print('----'+predictor_name+' of '+imf+' Finished----')
            if not self.NEXT_DAY: 
                df_result[1]['Runtime'] = time2-time1 # Output Runtime
                df_result[1]['IMF'] = imf
                df_result[0].name, df_result[1].name = predictor_name+' Result of '+imf, predictor_name+' Evaluation of '+imf
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
            plot_save_result(df_result, name=now, plot=plot, save=False, path=self.PATH) # do not save Co-IMF results
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
            plot_save_result(df_plot, name=now, plot=plot, save=False, path=self.PATH)
            df_result = (df_pred_result, final_eval)
        else:
            df_result = pd.DataFrame({'today_real': self.TARGET.values[-1], 'today_pred': df_next_result['today_pred'].sum(), 
                                      'next_pred': df_next_result['next_pred'].sum()}, index=[df_next_result.index[0]]) # Output
            df_result['Runtime'] = end-start # Output Runtime
            df_result['IMF'] = 'Final'
            df_result = pd.concat((df_result, df_next_result))
            df_result.name = predictor_name+' Result'
            print(df_result)
        plot_save_result(df_result, name=now, plot=False, save=save, path=self.PATH)
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

    # 2.5. Multiple Run Predictor
    def multiple_sklearn_predict(self, data=None, run_times=10, predict_method=None, **kwargs):
        """
        Multiple Run Predictor, multiple run of above method
        Example: 
        kr = cl.sklearn_predictor()
        df_result = kr.multiple_sklearn_predict(data, run_times=10, predict_method='lasso')

        Input and Parameters:
        ---------------------
        Note important hyperarameters of class cl.sklearn_predictor()
        data               - data set (include training set and test set)
        run_times          - running times
        predict_method     - run different method eg. 'lasso', 'svm', 'lgb'
                           - single: for lasso: self.lasso_predict(), svm: self.svm_predict(), lgb: self.lgb_predict()
                           - respective: self.respective_sklearn_predict()
        **kwargs           - any parameters of singe or respective method

        Output
        ---------------------
        df_eval_result     - evaluating forecasting results of each run
        """

        # Initialize 
        now = datetime.now()
        data = check_dataset(data, False, self.DECOM_MODE, self.REDECOM_LIST)
        self.TARGET = data['target']
        
        if self.PATH is None: print('Do not set a PATH! It is recommended to set a PATH to prevent the loss of running results')
        if predict_method is None: raise ValueError("Please input a predict method! eg. 'single', 'ensemble', 'respective'.")
        else: predict_method = predict_method.capitalize()
        if predict_method == 'Single': predictor_name = name_predictor(now, 'Multiple Single', 'Sklearn', self.SKLEARN_MODEL, None, None, self.NEXT_DAY)
        else: predictor_name = name_predictor(now, 'Multiple '+predict_method, 'Sklearn', self.SKLEARN_MODEL, self.DECOM_MODE, self.REDECOM_LIST, self.NEXT_DAY)    

        # Forecast
        start = time.time()
        df_pred_result = pd.DataFrame(index = self.TARGET.index[-self.FORECAST_LENGTH:0]) # df for storing forecasting result
        df_pred_result['real'] = self.TARGET[-self.FORECAST_LENGTH:]
        df_eval_result = pd.DataFrame(columns=['Scale', 'R2', 'RMSE', 'MAE', 'MAPE', 'Runtime', 'IMF']) # df for storing evaluation result
        for i in range(run_times):
            print('Run %d is running...'%i, end='')
            time1 = time.time()
            with self.HiddenPrints():
                if predict_method == 'Single':
                    if self.SKLEARN_MODEL == 'LASSO': df_result = self.lasso_predict(data=data, **kwargs)
                    if self.SKLEARN_MODEL == 'SVM': df_result = self.svm_predict(data=data, **kwargs) 
                    if self.SKLEARN_MODEL == 'LGB': df_result = self.lgb_predict(data=data, **kwargs)
                if predict_method == 'Respective': df_result = self.respective_sklearn_predict(data=data, **kwargs)
                if 'IMF' not in df_result[1].columns: df_result[1]['IMF'] = predict_method + '-Run' + str(i)
                else: df_result[1]['IMF'] = predict_method + '-Run' + str(i) + '-' + df_result[1]['IMF']
                df_eval_result = pd.concat((df_eval_result, df_result[1])) # add evaluation result
                df_pred_result['Run'+str(i)+'-predict'] = df_result[0]['predict']
                df_result = (df_pred_result, df_eval_result)
                df_pred_result.name, df_eval_result.name = predictor_name+' Result', predictor_name+' Evaluation'
                plot_save_result(df_result, name=now, plot=False, save=True, path=self.PATH)
            time2 = time.time()
            print('taking time: %.3fs'%(time2-time1))
        end = time.time()
        print('\n============='+predictor_name+' Finished=============')
        print('Running time: %.3fs'%(end-start))
        plot_save_result(df_result, name=now, plot=False, save=True, path=self.PATH)
        return df_result