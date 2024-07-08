#!/usr/bin/env python
# coding: utf-8
#
# Module: sklearn_predictor
# Description: Forecast time series by sklearn.
#
# Created: 2021-10-1 22:37
# Updated: 2022-9-18 17:28
# Updated: 2023-7-14 00:50
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
import warnings
warnings.filterwarnings("ignore") # Ignore some annoying warnings
# CEEMDAN_LSTM
from CEEMDAN_LSTM.core import check_dataset, check_path, plot_save_result, name_predictor, output_result
# Sklearn
# import joblib # load and save model 
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV

class sklearn_predictor:
    # 0.Initialize
    # ------------------------------------------------------
    def __init__(self, PATH=None, FORECAST_HORIZONS=30, FORECAST_LENGTH=30, SKLEARN_MODEL='LASSO', OPTIMIZER='Bayes',
                 DECOM_MODE='OVMD', INTE_LIST='auto', REDECOM_LIST=None, 
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
        trials             - optimizer search times, trials//10 for GridSearch
        opt_cv             - optimizer cross validation division number, cross_val_score(cv=opt_cv), eg.5-20
        opt_score          - optimizer cross validation scoring method,  cross_val_score(scoring=opt_score), eg.'r2', 'explained_variance', 'neg_mean_absolute_error'

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

        # Declare Sklearn parameters
        self.opt_trials = int(kwargs.get('opt_trials', 100))
        self.opt_cv = int(kwargs.get('opt_cv', 5))
        self.opt_score = str(kwargs.get('opt_score', 'r2')) 
        self.opt_direction = str(kwargs.get('opt_direction', 'maximize')) 

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
        if REDECOM_LIST is not None: # Check REDECOM_LIST and Input REDECOM_MODE
            REDECOM_MODE = ''
            try:
                REDECOM_LIST = pd.DataFrame(REDECOM_LIST, index=range(1)) 
                for i in range(REDECOM_LIST.size): REDECOM_MODE = REDECOM_MODE+REDECOM_LIST.values.ravel()[i]+REDECOM_LIST.columns[i][-1]+'-'
            except: raise ValueError("Invalid input for redecom_list! Please input eg. None, '{'co-imf0':'vmd', 'co-imf1':'emd'}'.")
            self.REDECOM_MODE = str(REDECOM_MODE).upper()
        else: REDECOM_MODE = None

        if self.opt_score == 'r2': self.opt_direction = 'maximize' # Check opt_direction
        else: self.opt_direction = 'minimize'

        if self.NEXT_DAY == True:
            print('Sklearn predictor does not support next day. Maybe update in the future.')
            self.NEXT_DAY = False


    # 1.Basic functions
    # ------------------------------------------------------
    # 1.1 Lasso (Least Absolute Shrinkage and Selection Operator regression)
    def lasso_predict(self, train_test_set=None, show=False):
        """
        Single Method (directly forecast)
        Use sklearn Lasso to directly forecast wiht vector input
        Example: 
        sr = cl.sklearn_predictor()
        df_result = sr.lasso_predict(data, show=True, plot=True)

        Input and Parameters:
        ---------------------
        Note important hyperparameters of class cl.sklearn_predictor()
        train_test_set     - data set (include training set and test set)
        show               - show optimzing process

        Output
        ---------------------
        y_predict         - forecasting result
        """

        # Get the training and test set
        x_train, x_test, y_train, y_test, scalarY, next_x = train_test_set 

        # Grid Search of K-Fold CV
        # logspace(a,b,N)Divide the interval from the a power of 10 to the b power of 10 into N parts
        if self.OPTIMIZER == 'GS':
            alpha_range = np.logspace(-9, 3, 15)
            best_alpha = 1
            for i in range(self.opt_trials//10):
                param_grid = dict(alpha=alpha_range)
                grid = GridSearchCV(Lasso(max_iter=10000), param_grid=param_grid, cv=self.opt_cv, scoring=self.opt_score)
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
                model = Lasso(alpha=alpha).fit(x_train, y_train)
                score = cross_val_score(model, x_train, y_train, cv=self.opt_cv, scoring=self.opt_score)
                return score.mean() 
            study = optuna.create_study(study_name='LASSO alpha', direction=self.opt_direction) # TPESampler is used
            if not show:optuna.logging.set_verbosity(optuna.logging.WARNING) # not to print
            study.optimize(objective, n_trials=self.opt_trials, n_jobs=-1, gc_after_trial=True)  # number of iterations
            best_alpha = study.best_params['alpha']
            # training_loss = study.trials_dataframe()['value']

        # Predict
        model = Lasso(alpha=best_alpha)
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        return y_predict



    # 1.2 SVM (Support Vector Machine Regression)
    def svm_predict(self, train_test_set=None, show=False):
        """
        Single Method (directly forecast)
        Use sklearn SVR to directly forecast wiht vector input
        Example: 
        sr = cl.sklearn_predictor()
        df_result = sr.svm_predict(data, show=True, plot=True)

        Input and Parameters:
        ---------------------
        Note important hyperparameters of class cl.sklearn_predictor()
        train_test_set     - data set (include training set and test set)
        show               - show optimzing process

        Output
        ---------------------
        y_predict         - forecasting result
        """

        # Get the training and test set
        x_train, x_test, y_train, y_test, scalarY, next_x = train_test_set 

        # Grid Search of K-Fold CV
        # logspace(a,b,N)Divide the interval from the a power of 10 to the b power of 10 into N parts
        if self.OPTIMIZER == 'GS':
            C_range = np.logspace(-2, 10, 13)
            gamma_range = np.logspace(-9, 3, 13)
            best_gamma, best_C = 0, 0
            for i in range(self.opt_trials//10):
                param_grid = dict(gamma=gamma_range, C=C_range)
                grid = GridSearchCV(SVR(), param_grid=param_grid, cv=self.opt_cv, scoring=self.opt_score)
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
                model = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon).fit(x_train, y_train)
                score = cross_val_score(model, x_train, y_train, cv=self.opt_cv, scoring=self.opt_score)
                return score.mean() 
            study = optuna.create_study(study_name='SVR C gamma epsilon', direction=self.opt_direction) # TPESampler is used
            if not show: optuna.logging.set_verbosity(optuna.logging.WARNING) # not to print
            study.optimize(objective, n_trials=self.opt_trials, n_jobs=-1, gc_after_trial=True)  # number of iterations
            # best_kernel = study.best_params['kernel']
            best_gamma = study.best_params['gamma']
            best_C = study.best_params['C']
            best_epsilon = study.best_params['epsilon']

        # Predict
        model = SVR(kernel='rbf', gamma=best_gamma, C=best_C, epsilon=best_epsilon)
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        return y_predict



    # 1.3 LGB or LightGBM (Light Gradient Boosting Machine)
    def lgb_predict(self, train_test_set=None, show=False):
        """
        Single Method (directly forecast)
        Use LGB to directly forecast wiht vector input
        Example: 
        sr = cl.sklearn_predictor()
        df_result = sr.lgb_predict(data, show=True, plot=True)

        Input and Parameters:
        ---------------------
        Note important hyperparameters of class cl.sklearn_predictor()
        train_test_set     - data set (include training set and test set)
        show               - show optimzing process

        Output
        ---------------------
        y_predict         - forecasting result
        """

        # Name and set target
        try: import lightgbm as lgb
        except: raise ImportError('Cannot import lightgbm, run: pip install lightgbm!')

        # Get the training and test set
        x_train, x_test, y_train, y_test, scalarY, next_x = train_test_set 

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
                model = lgb.LGBMRegressor(**param_grid).fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=100, verbose=False)
                score = cross_val_score(model, x_train, y_train, cv=self.opt_cv, scoring=self.opt_score)
                return score.mean() 
            study = optuna.create_study(study_name='LGB hyperparameters', direction=self.opt_direction) # TPESampler is used
            if not show: optuna.logging.set_verbosity(optuna.logging.WARNING) # not to print
            study.optimize(objective, n_trials=self.opt_trials, n_jobs=-1, gc_after_trial=True)  # number of iterations
            best_params = study.best_params

        # Predict
        model = lgb.LGBMRegressor(**best_params)
        model.fit(x_train, y_train, eval_set=[(x_test, y_test)], early_stopping_rounds=100, verbose=False)
        y_predict = model.predict(x_test)
        return y_predict



    # 2.Advanced forecasting functions
    # ------------------------------------------------------
    # 2.1. Single Method (directly forecast)
    def single_sklearn_predict(self, data=None, show=False, plot=False, save=False, **kwargs):
        """
        Single Method (directly forecast)
        Use Sklearn models to directly forecast with vector input
        Example: 
        sr = cl.sklearn_predictor()
        df_result = sr.single_sklearn_predict(data, show=True, plot=True, save=False)

        Input and Parameters:
        ---------------------
        Note important hyperarameters of class cl.sklearn_predictor()
        data            - data set (include training set and test set)
        show            - show the inputting data set and Keras model structure
        plot            - show figure result or not
        save            - save forecasting result when set PATH
        **kwargs        - any parameters of self.lasso_predict() or self.svm_predict() or ...

        Output
        ---------------------
        df_result = (df_pred_result, df_eval_result)
        df_pred_result  - forecasting results and the original real series set
        df_eval_result  - evaluating results of forecasting 
        """

        # Name and set 
        now = datetime.now()
        predictor_name = name_predictor(now, 'Single', 'Sklearn', self.SKLEARN_MODEL, None, None, self.NEXT_DAY)
        data = check_dataset(data, show, self.DECOM_MODE, None)
        data.name = predictor_name+' model.pkl'

        # Divide the training and test set
        from CEEMDAN_LSTM.data_preprocessor import create_train_test_set
        start = time.time()
        x_train, x_test, y_train, y_test, scalarY, next_x = create_train_test_set(data, self.FORECAST_LENGTH, self.FORECAST_HORIZONS, self.NEXT_DAY, self.NOR_METHOD, self.DAY_AHEAD) 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]))
        train_test_set = (x_train, x_test, y_train, y_test, scalarY, next_x)

        # Forecast
        if self.SKLEARN_MODEL == 'LASSO': y_predict = self.lasso_predict(train_test_set, show=show, **kwargs)
        if self.SKLEARN_MODEL == 'SVM': y_predict = self.svm_predict(train_test_set, show=show, **kwargs) 
        if self.SKLEARN_MODEL == 'LGB': y_predict = self.lgb_predict(train_test_set, show=show, **kwargs)
        end = time.time()

        # De-normalize
        y_predict = y_predict.ravel().reshape(-1,1) 
        if scalarY is not None: y_predict = scalarY.inverse_transform(y_predict)         

        # Output and Evaluate
        df_predict_result = pd.DataFrame({'real': data['target'][-self.FORECAST_LENGTH:], 'predict': y_predict.ravel()}) # Output
        df_result = output_result(df_predict_result, predictor_name, end-start, imf='Final') # return (final_pred, final_eval)
        plot_save_result(df_result, name=now, plot=plot, save=save, path=self.PATH)
        return df_result



    # 2.2. Respective Method (decompose then respectively forecast each IMFs)
    def respective_sklearn_predict(self, data=None, show=False, plot=False, save=False, **kwargs):
        """
        Respective Method (decompose then respectively forecast each IMFs)
        Use decomposition-integration method and Sklearn models to respectively forecast each IMFs with vector input
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
        df_result = (df_pred_result, df_eval_result)
        df_pred_result  - forecasting results and the original real series set
        df_eval_result  - evaluating results of forecasting 
        """

        # Name
        now = datetime.now()
        predictor_name = name_predictor(now, 'Respective', 'Sklearn', self.SKLEARN_MODEL, self.DECOM_MODE, self.REDECOM_LIST, self.NEXT_DAY)

        # Decompose, integrate, re-decompose
        start = time.time()
        from CEEMDAN_LSTM.data_preprocessor import redecom
        df_redecom, df_redecom_list = redecom(data, show, self.DECOM_MODE, self.INTE_LIST, self.REDECOM_LIST, self.VMD_PARAMS, self.FORECAST_LENGTH)

        # Forecast and ouput each Co-IMF
        df_pred_result = pd.DataFrame(index = df_redecom['target'].index[-self.FORECAST_LENGTH:]) # df for storing forecasting result
        df_eval_result = pd.DataFrame(columns=['Scale', 'R2', 'RMSE', 'MAE', 'MAPE', 'Runtime', 'IMF']) # df for storing evaluation result
        df_next_result = pd.DataFrame(columns=['today_real', 'today_pred', 'next_pred', 'Runtime', 'IMF']) # df for storing Next-day forecasting result
        for imf in df_redecom.columns.difference(['target']):
            df_redecom[imf].name = (predictor_name+'_of_'+imf+'_model.h5').replace('-','_').replace(' ','_')
            time1 = time.time()
            with self.HiddenPrints():
                df_result = self.single_sklearn_predict(data=df_redecom[imf], **kwargs) # return (df_pred, df_eval)
            time2 = time.time()
            df_result = output_result(df_result, predictor_name, time2-time1, imf) # return (imf_pred, imf_eval)
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
            df_result = output_result((df_pred_result, df_eval_result), predictor_name, end-start, imf='Final')
        else:
            df_result = pd.DataFrame({'today_real': df_redecom['target'][-1:], 'today_pred': df_next_result['today_pred'].sum(), 
                                      'next_pred': df_next_result['next_pred'].sum()}) # Output
            df_result = output_result((df_result, df_next_result), predictor_name, end-start, imf='Final')
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

    # 2.3. Multiple Run Predictor
    def multiple_sklearn_predict(self, data=None, run_times=10, predict_method=None, **kwargs):
        """
        Multiple Run Predictor, multiple run of above method
        Example: 
        sr = cl.sklearn_predictor()
        df_result = sr.multiple_sklearn_predict(data, run_times=10, predict_method='respective')

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
        if self.PATH is None: print('Do not set a PATH! It is recommended to set a PATH to prevent the loss of running results')
        if predict_method is None: raise ValueError("Please input a predict method! eg. 'single', 'ensemble', 'respective'.")
        else: predict_method = predict_method.capitalize()
        if predict_method == 'Single': predictor_name = name_predictor(now, 'Multiple Single', 'Sklearn', self.SKLEARN_MODEL, None, None, self.NEXT_DAY)
        else: predictor_name = name_predictor(now, 'Multiple '+predict_method, 'Sklearn', self.SKLEARN_MODEL, self.DECOM_MODE, self.REDECOM_LIST, self.NEXT_DAY)    

        # Forecast
        start = time.time()
        df_pred_result = pd.DataFrame(index = data['target'].index[-self.FORECAST_LENGTH:]) # df for storing forecasting result
        df_eval_result = pd.DataFrame(columns=['Scale', 'R2', 'RMSE', 'MAE', 'MAPE', 'Runtime', 'IMF']) # df for storing evaluation result
        for i in range(run_times):
            print('Run %d is running...'%i, end='')
            time1 = time.time()
            with self.HiddenPrints():
                if predict_method == 'Single': df_result = self.single_sklearn_predict(data=data, **kwargs) # return (df_pred, df_eval)
                elif predict_method == 'Respective': df_result = self.respective_sklearn_predict(data=data, **kwargs) # return (df_all_pred, df_eval)
                else: raise ValueError("Please input a vaild predict method! eg. 'single', 'respective'.")
                time2 = time.time()
                df_result = output_result(df_result, predictor_name, time2-time1, imf='Final', run=i)
                df_pred_result = pd.concat((df_pred_result, df_result[0]), axis=1) # add forecasting result
                df_eval_result = pd.concat((df_eval_result, df_result[1])) # add evaluation result
                df_pred_result.name, df_eval_result.name = predictor_name+' Result', predictor_name+' Evaluation'
                plot_save_result((df_pred_result, df_eval_result), name=now, plot=False, save=True, path=self.PATH) # Temporary storage
            print('taking time: %.3fs'%(time2-time1))
        end = time.time()
        print('\n============='+predictor_name+' Finished=============')
        print('Running time: %.3fs'%(end-start))
        return (df_pred_result, df_eval_result)