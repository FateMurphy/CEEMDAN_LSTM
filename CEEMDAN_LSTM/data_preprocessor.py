#!/usr/bin/env python
# coding: utf-8
#
# Module: data_preprocessor
# Description: Some data preprocessing function was compiled here, eg. decompose, integrate
#
# Created: 2021-10-1 22:37
# Updated: 2022-9-12 17:28
# Updated: 2023-7-14 00:50
# Author: Feite Zhou
# Email: jupiterzhou@foxmail.com
# URL: 'http://github.com/FateMurphy/CEEMDAN_LSTM'
# Feel free to email me if you have any questions or error reports.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") # Ignore some annoying warnings

"""
References:
EMD, Empirical Mode Decomposition: 
    PyEMD: https://github.com/laszukdawid/PyEMD
    Laszuk & Dawid, 2017, https://doi.org/10.5281/zenodo.5459184
    EMD method: Huang et al., 1998, https://doi.org/10.1098/rspa.1998.0193.
    EEMD method: Wu & Huang, 2009, https://doi.org/10.1142/S1793536909000047.
    CEEMDAN method: Torres et al., 2011, https://doi.org/10.1109/ICASSP.2011.5947265.
VMD, Variational mode decomposition: 
    vmdpy: https://github.com/vrcarva/vmdpy, 
    Vinícius et al., 2020, https://doi.org/10.1016/j.bspc.2020.102073.
    VMD method: Dragomiretskiy & Zosso, 2014, https://doi.org/10.1109/TSP.2013.2288675.
SampEn, Sample Entropy:
    sampen: https://github.com/bergantine/sampen
K-Means, scikit-learn(sklearn):
    scikit-learn: https://github.com/scikit-learn/scikit-learn
Optuna, Bayesian Optimization: 
    optuna: https://github.com/optuna/optuna
"""



# 1.Decompose
# ------------------------------------------------------
# 1.Main. Decompose series
def decom(series=None, decom_mode='ceemdan', vmd_params=None, FORECAST_LENGTH=None, **kwargs): 
    """
    Decompose time series adaptively and return results in pd.Dataframe by PyEMD.EMD/EEMD/CEEMDAN or vmdpy.VMD.
    Example: df_decom = cl.decom(series, decom_mode='ceemdan')
    Plot by pandas: df_decom.plot(title='Decomposition Results', subplots=True)

    Input and Parameters:
    ---------------------
    series     - the time series (1D) to be decomposed
    decom_mode - the decomposing methods eg. 'emd', 'eemd', 'ceemdan', 'vmd'
    vmd_params - the best parameters K and tau of OVMD to jump the optimization
    **kwargs   - any parameters of PyEMD.EMD(), PyEMD.EEMD(), PyEMD.CEEMDAN(), vmdpy.VMD()
               - eg. trials for PyEMD.CEEMDAN(), change the number of inputting white noise 
    Output:
    ---------------------
    df_decom   - the decomposing results in pd.Dataframe
    """

    if series is None: raise ValueError('Please input pd.Series, or pd.DataFrame(1D), nd.array(1D).')
    try: series = pd.Series(series)
    except: raise ValueError('Sorry! %s is not supported to decompose, please input pd.Series, or pd.DataFrame(1D), nd.array(1D).'%type(series))
    decom_mode = decom_mode.lower()
    best_params = 'None'
    if decom_mode in ['emd', 'eemd', 'ceemdan']:
        try: import PyEMD
        except ImportError: raise ImportError('Cannot import EMD-signal!, run: pip install EMD-signal!')
        if decom_mode == 'emd': decom = PyEMD.EMD(**kwargs) # EMD
        elif decom_mode == 'eemd': decom = PyEMD.EEMD(**kwargs) # EEMD
        elif decom_mode == 'ceemdan': decom = PyEMD.CEEMDAN(**kwargs) # CEEMDAN
        decom_result = decom(series.values).T
        df_decom = pd.DataFrame(decom_result, columns=['imf'+str(i) for i in range(len(decom_result[0]))])
    elif decom_mode == 'vmd': 
        df_decom, imfs_hat, omega = decom_vmd(series, **kwargs)
    elif decom_mode == 'ovmd': 
        df_decom, best_params = decom_ovmd(series, vmd_params=vmd_params, **kwargs)
    # Some approaches are trying
    elif decom_mode == 'semd': 
        df_decom = decom_semd(series, decom_mode='emd', FORECAST_LENGTH=FORECAST_LENGTH, **kwargs)
    elif decom_mode == 'sceemdan': 
        df_decom = decom_semd(series, decom_mode='ceemdan', FORECAST_LENGTH=FORECAST_LENGTH, **kwargs)
    elif decom_mode == 'nsvmd': 
        df_decom = decom_semd(series, decom_mode='vmd', FORECAST_LENGTH=FORECAST_LENGTH, **kwargs)
    elif decom_mode == 'svmd': 
        df_decom = decom_svmd(series, vmd_params=vmd_params, FORECAST_LENGTH=FORECAST_LENGTH, **kwargs)
    elif decom_mode == None:
         print('Warning! The data does not decomposed.')
         df_decom = series
    else: raise ValueError('%s is not a supported decomposition method!'%(decom_mode))
    if isinstance(series, pd.Series): 
        if 'vmd' in decom_mode and len(series)%2: df_decom.index = series[1:].index # change index
        else: df_decom.index = series.index # change index
    df_decom['target'] = series
    df_decom.name = decom_mode.lower()+'_'+str(best_params)
    return df_decom

# 1.1 VMD
def decom_vmd(series=None, alpha=2000, tau=0, K=10, DC=0, init=1, tol=1e-7, **kwargs): # VMD Decomposition
    """
    Decompose time series by VMD, Variational mode decomposition by vmdpy.VMD.
    Example: df_vmd, imfs_hat, imfs_omega = cl.decom_vmd(series, alpha, tau, K, DC, init, tol)
    Plot by pandas: df_vmd.plot(title='VMD Decomposition Results', subplots=True)

    Input and Parameters:
    ---------------------
    series     - the time series (1D) to be decomposed
    alpha      - the balancing parameter of the data-fidelity constraint
    tau        - time-step of the dual ascent ( pick 0 for noise-slack )
    K          - the number of modes to be recovered
    DC         - true if the first mode is put and kept at DC (0-freq)
    init       - 0 = all omegas start at 0
                 1 = all omegas start uniformly distributed
                 2 = all omegas initialized randomly
    tol        - tolerance of convergence criterion eg. around 1e-6
    **kwargs   - any parameters of vmdpy.VMD()

    Output:
    ---------------------
    df_vmd     - the collection of decomposed modes in pd.Dataframe
    imfs_hat   - spectra of the modes
    imfs_omega - estimated mode center-frequencies    
    """

    try: import vmdpy
    except ImportError: raise ImportError('Cannot import vmdpy, run: pip install vmdpy!')
    if series is None: raise ValueError('Please input pd.Series, or pd.DataFrame(1D), nd.array(1D).')
    if len(series)%2: print('Warning! The vmdpy module will delete the last one data point of series before decomposition')
    imfs_vmd, imfs_hat, imfs_omega = vmdpy.VMD(series, alpha, tau, K, DC, init, tol, **kwargs)  
    df_vmd = pd.DataFrame(imfs_vmd.T, columns=['imf'+str(i) for i in range(K)])
    return df_vmd, imfs_hat, imfs_omega

# 1.2 Optimized VMD (OVMD)
def decom_ovmd(series=None, vmd_params=None, trials=100): # VMD Decomposition
    """
    Decompose time series by VMD and use Bayesian Optimization to find the best K and tau
    Example: df_vmd = cl.decom_ovmd(series)
    Plot by pandas: df_vmd.plot(title='VMD Decomposition Results', subplots=True)

    Input and Parameters:
    ---------------------
    series     - the time series (1D) to be decomposed
    vmd_params - the best parameters K and tau of OVMD to jump the optimization
    trials     - the number of optimization iteration

    Output:
    ---------------------
    df_vmd     - the collection of decomposed modes in pd.Dataframe 
    """
    
    if series is None: raise ValueError('Please input pd.Series, or pd.DataFrame(1D), nd.array(1D).')
    try: series = pd.Series(series)
    except: raise ValueError('Sorry! %s is not supported to decompose, please input pd.Series, or pd.DataFrame(1D), nd.array(1D)'%type(series))
    if len(series)%2: 
        print('Warning! The vmdpy module will delete the last one data point of series before decomposition')
        series = series[1:]
    if vmd_params is None:
        try: import optuna
        except: raise ImportError('Cannot import optuna, run: pip install optuna!')
        def objective(trial):
            K = trial.suggest_int('K', 1, 10) # set hyperparameter range
            alpha = trial.suggest_int('alpha', 1, 10000)
            tau = trial.suggest_float('tau', 0, 1) 
            df_vmd, imfs_hat, imfs_omega = decom_vmd(series, K=K, alpha=alpha, tau=tau)
            return abs((df_vmd.sum(axis=1).values - series.values).sum()) # residual of decomposed and original series 
        study = optuna.create_study(study_name='OVMD Method', direction='minimize') # TPESampler is used
        optuna.logging.set_verbosity(optuna.logging.WARNING) # not to print
        study.optimize(objective, n_trials=trials, n_jobs=-1, gc_after_trial=True)  # number of iterations
        vmd_params = study.best_params
    df_vmd, imfs_hat, imfs_omega = decom_vmd(series, K=vmd_params['K'], alpha=vmd_params['alpha'], tau=vmd_params['tau'])
    return df_vmd, vmd_params

# 1.3 Separated VMD (SVMD)
def decom_svmd(series=None, FORECAST_LENGTH=None, vmd_params=None, trials=100): # VMD Decomposition
    """
    Decompose time series by VMD separatey for traning and test set,
    and use Bayesian Optimization to find the best K and tau.
    Example: df_vmd = cl.decom_svmd(series)
    Plot by pandas: df_vmd.plot(title='VMD Decomposition Results', subplots=True)

    Input and Parameters:
    ---------------------
    series             - the time series (1D) to be decomposed
    vmd_params         - the best parameters K and tau of OVMD to jump the optimization
    trials             - the number of optimization iteration
    FORECAST_LENGTH    - the length of the days to forecast (test set)

    Output:
    ---------------------
    df_vmd             - the collection of decomposed modes in pd.Dataframe 
    """
    
    if series is None: raise ValueError('Please input pd.Series, or pd.DataFrame(1D), nd.array(1D).')
    try: series = pd.Series(series)
    except: raise ValueError('Sorry! %s is not supported to decompose, please input pd.Series, or pd.DataFrame(1D), nd.array(1D)'%type(series))
    if len(series)%2: 
        print('Warning! The vmdpy module will delete the last one data point of series before decomposition')
        series = series[1:]
    if FORECAST_LENGTH is None: raise ValueError('Please input FORECAST_LENGTH.')

    series_train = series[:-FORECAST_LENGTH]
    series_test = series[-FORECAST_LENGTH:]
    if vmd_params is None:
        try: import optuna
        except: raise ImportError('Cannot import optuna, run: pip install optuna!')
        def objective(trial):
            K = trial.suggest_int('K', 1, 10) # set hyperparameter range
            alpha = trial.suggest_int('alpha', 1, 10000)
            tau = trial.suggest_float('tau', 0, 1) 
            df_vmd, imfs_hat, imfs_omega = decom_vmd(series_train, K=K, alpha=alpha, tau=tau)
            return abs((df_vmd.sum(axis=1).values - series_train.values).sum()) # residual of decomposed and original training set series 
        study = optuna.create_study(study_name='SVMD Method for training set', direction='minimize') # TPESampler is used
        optuna.logging.set_verbosity(optuna.logging.WARNING) # not to print
        study.optimize(objective, n_trials=trials, n_jobs=-1, gc_after_trial=True)  # number of iterations
        vmd_params = study.best_params
    df_vmd_train, imfs_hat, imfs_omega = decom_vmd(series_train, K=vmd_params['K'], alpha=vmd_params['alpha'], tau=vmd_params['tau'])
    df_vmd_test, imfs_hat, imfs_omega = decom_vmd(series_test, K=vmd_params['K'], alpha=vmd_params['alpha'], tau=vmd_params['tau'])
    df_vmd = pd.concat((df_vmd_train, df_vmd_test))
    df_vmd.index = series.index
    return df_vmd

# 1.4 Separated EMD (SEMD)
def decom_semd(series=None, decom_mode='emd', FORECAST_LENGTH=None, **kwargs): 
    """
    Decompose time series by EMD separatey for traning and test set after integration,
    and use Bayesian Optimization to find the best K and tau.
    Example: df_vmd = cl.decom_svmd(series)
    Plot by pandas: df_vmd.plot(title='VMD Decomposition Results', subplots=True)

    Input and Parameters:
    ---------------------
    series             - the time series (1D) to be decomposed
    decom_mode         - the decomposing methods eg. 'emd', 'eemd', 'ceemdan', 'vmd'
    FORECAST_LENGTH    - the length of the days to forecast (test set)

    Output:
    ---------------------
    df_decom           - the collection of decomposed modes in pd.Dataframe 
    """

    if series is None: raise ValueError('Please input pd.Series, or pd.DataFrame(1D), nd.array(1D).')
    try: series = pd.Series(series)
    except: raise ValueError('Sorry! %s is not supported to decompose, please input pd.Series, or pd.DataFrame(1D), nd.array(1D)'%type(series))
    if FORECAST_LENGTH is None: raise ValueError('Please input FORECAST_LENGTH.')

    df_decom_train = redecom(series[:-FORECAST_LENGTH], decom_mode=decom_mode, redecom_list=None, **kwargs)
    df_decom_test = redecom(series[-FORECAST_LENGTH:], decom_mode=decom_mode, redecom_list=None, **kwargs)
    df_decom = pd.concat((df_decom_train[0], df_decom_test[0]))
    df_decom.index = series.index
    return df_decom



# 2.Integrate
# ------------------------------------------------------
# 2.Main. Integrate
def inte(df_decom=None, inte_list='auto', num_clusters=3):
    """
    Integrate IMFs to be CO-IMFs by sampen.sampen2 and sklearn.cluster.
    Example: df_inte, df_inte_list = cl.inte(df_decom)
    Plot by pandas: df_inte.plot(title='Integrated IMFs (Co-IMFs) Results', subplots=True)
    Custom integration: please use cl.inte_sampen() and cl.inte_kmeans()

    Input and Parameters:
    ---------------------
    df_decom       - the decomposing results in pd.Dataframe, or a group of series
    inte_list      - the integration list, eg. pd.Dataframe, (int) 3, (str) '233', (list) [0,0,1,1,1,2,2,2], ...
    num_clusters   - number of categories/clusters eg. num_clusters = 3

    Output:
    ---------------------
    df_inte        - the integrating form of each time series
    df_inte_list   - the integrating set of each co-imf
    """
    
    # Check input
    df_inte_list = []
    try: df_decom = pd.DataFrame(df_decom)
    except: raise ValueError('Invalid input of df_decom!')
    if 'target' in df_decom.columns: 
        tmp_target = df_decom['target']
        df_decom = df_decom.drop('target', axis=1, inplace=False)
    else: tmp_target = None

    # Convert inte_list
    if inte_list is not None:
        if str(inte_list).lower() == 'auto': # Without 
            df_sampen = inte_sampen(df_decom)
            inte_list = inte_kmeans(df_sampen, num_clusters=num_clusters)
        elif isinstance(inte_list, pd.DataFrame):
            if len(inte_list) == 1: inte_list = inte_list.T
        elif type(inte_list) == str and len(inte_list) < df_decom.columns.size:
            df_list, n, c, s = {}, 0, 0, 0
            for i in inte_list: s = s + int(i) 
            if s != df_decom.columns.size: raise ValueError('Invalid inte_list! %s (%s columns) does not match the number of dataset columns=%s.'%(inte_list, s, df_decom.columns.size)) 
            for i in inte_list:
                for j in ['imf'+str(x) for x in range(n, n+int(i))]: 
                    df_list[j], n = c, n + 1
                c += 1
            inte_list = pd.DataFrame(df_list, index=['Cluster']).T
        elif type(inte_list) == str and len(inte_list) == df_decom.columns.size:
            inte_list = pd.DataFrame([int(x) for x in inte_list], columns=['Cluster'], index=['imf'+str(x) for x in range(len(inte_list))])
        elif type(inte_list) == int and inte_list < df_decom.columns.size:
            print('Integrate %d columns to be %d columns by K-means.'%(df_decom.columns.size, inte_list))
            df_sampen = inte_sampen(df_decom)
            inte_list = inte_kmeans(df_sampen, num_clusters=inte_list)
        else:
            try: inte_list = pd.DataFrame(inte_list, columns=['Cluster'], index=['imf'+str(x) for x in range(len(inte_list))])
            except: raise ValueError('Invalid inte_list of %s with type %s. Check your input or length.'%(inte_list, type(inte_list)))

    # Integrate, resort, and output
    if inte_list is not None: # Check inte_list after Convert
        # Integrate
        df_inte = pd.DataFrame()
        for i in range(inte_list.values.max()+1):
            df_inte['co-imf'+str(i)] = df_decom[inte_list[(inte_list['Cluster']==i)].index].sum(axis=1)
            df_tmp = pd.concat((df_inte['co-imf'+str(i)], df_decom[inte_list[(inte_list['Cluster']==i)].index]), axis=1)
            df_tmp.name = 'co-imf'+str(i)
            df_inte_list.append(df_tmp)

        # Use Sample Entropy resorting the Co-IMFs
        df_tmp = df_inte.T
        df_tmp['sampen'] = inte_sampen(df_tmp.T).values
        df_tmp.sort_values(by=['sampen'], ascending=False, inplace=True)
        df_tmp['related'] = [x for x in range(len(df_tmp))]
        df_related = pd.DataFrame(df_tmp['related']).T
        for df in df_inte_list: # rename df_inte_list
            imf_name = 'co-imf'+str(df_related[df.name]['related'])
            df.rename(columns={df.name:imf_name}, inplace=True)
            df.name = imf_name
        df_inte_list.sort(key=lambda x: x.name)
        df_tmp.index = ['co-imf'+str(i) for i in range(len(df_tmp))]
        df_inte = df_tmp.drop(['related', 'sampen'], axis=1, inplace=False).T

        # output
        df_inte.name = 'df_inte_list_'+''.join(str(x) for x in inte_list.values.ravel()) # record integrate list
        df_inte.index = df_decom.index
    else: 
        df_decom.columns = ['co-imf'+str(i) for i in range(df_decom.columns.size)] # less than num_clusters, stop inte and output
        df_inte = df_decom
        for col in df_inte.columns: 
            df_tmp = pd.DataFrame(df_inte[col])
            df_tmp.name = col
            df_inte_list.append(df_tmp)
    if tmp_target is not None: df_inte.insert(0, 'target', tmp_target.values) # add tmp target column
    return df_inte, df_inte_list
    
# 2.1 K-Means
def inte_kmeans(df_sampen=None, num_clusters=3, random_state=0, **kwargs):
    """
    Get integrating form by K-Means by sklearn.cluster.
    Example: inte_list = cl.inte_kmeans(df_sampen)
    Print: print(inte_list)

    Input and Parameters:
    ---------------------
    df_sampen    - the Sample Entropy of each time series in pd.Dataframe, or an one-column Dataframe with specific index
    num_clusters - number of categories/clusters eg. num_clusters = 3
    random_state - control the random state to guarantee the same result every time
    **kwargs     - any parameters of sklearn.cluster.KMeans()

    Output:
    ---------------------
    inte_list    - the integrating form of each time series in pd.Dataframe
    """

    # Get integrating form by K-Means
    try: from sklearn.cluster import KMeans
    except ImportError: raise ImportError('Cannot import scikit-learn(sklearn), run: pip install scikit-learn!')
    if df_sampen.index.size > num_clusters:
        try: df_sampen = pd.DataFrame(df_sampen)
        except: raise ValueError('Invalid input of df_sampen!')
        np_inte_list = KMeans(n_clusters=num_clusters, random_state=random_state, **kwargs).fit_predict(df_sampen)
        inte_list = pd.DataFrame(np_inte_list, index=['imf'+str(i) for i in range(df_sampen.index.size)], columns=['Cluster'])
    else: inte_list = None
    return inte_list

# 2.2 Sample Entropy
def inte_sampen(df_decom=None, max_len=1, tol=0.1, nor=True, **kwargs):
    """
    Calculate Sample Entropy for each IMF or series by sampen.sampen2.
    Example: df_sampen = cl.inte_sampen(df_decom)
    Plot by pandas: df_sampen.plot(title='Sample Entropy')

    Input and Parameters:
    ---------------------
    df_decom   - the decomposing results in pd.Dataframe, or a group of series
    max_len    - maximum length of epoch (subseries)
    tol        - tolerance eg. 0.1 or 0.2
    nor        - normalize or not 
    **kwargs   - any parameters of sampen.sampen2()

    Output:
    ---------------------
    df_sampen  - the Sample Entropy of each time series in pd.Dataframe
    """
    try: 
        df_decom = pd.DataFrame(df_decom)
        if 'target' in df_decom.columns: df_decom = df_decom.drop('target', axis=1, inplace=False)
        if 'imf0' not in df_decom.columns or 'co-imf0' not in df_decom.columns: 
            df_decom.columns = ['imf'+str(i) for i in range(df_decom.columns.size)]
    except: raise ValueError('Invalid input of df_decom!') 
    try: import sampen
    except ImportError: raise ImportError('Cannot import sampen, run: pip install sampen!')
    np_sampen = []
    for i in range(df_decom.columns.size):
        sample_entropy = sampen.sampen2(list(df_decom['imf'+str(i)].values), mm=max_len, r=tol, normalize=nor, **kwargs)
        np_sampen.append(sample_entropy[1][1])
    df_sampen = pd.DataFrame(np_sampen, index=['imf'+str(i) for i in range(df_decom.columns.size)])
    return df_sampen



# 3.Other Mains
# ------------------------------------------------------
# 3.Main. Redecompose (inculd decom() and inte())
def redecom(data=None, show=False, decom_mode='ceemdan', inte_list='auto', redecom_list={'co-imf0':'ovmd'}, vmd_params=None, FORECAST_LENGTH=None, **kwargs):
    """
    redecompose data adaptively and return results in pd.Dataframe.
    Example: df_decom = cl.redecom(series, decom_mode='ceemdan', redecom_list={'co-imf0':'vmd'})
    Plot by pandas: df_decom.plot(title='Decomposition Results', subplots=True)

    Input and Parameters:
    ---------------------
    series          - the time series (1D) to be decomposed
    show            - show the inputting data set
    decom_mode      - the decomposing methods eg. 'emd', 'eemd', 'ceemdan', 'vmd'
    inte_list       - the integration list, eg. pd.Dataframe, (int) 3, (str) '233', (list) [0,0,1,1,1,2,2,2], ...
    redecom_list    - the re-decomposition list eg. '{'co-imf0':'vmd', 'co-imf1':'emd'}', pd.DataFrame
    vmd_params      - the best parameters K and tau of OVMD to jump the optimization
    **kwargs        - any parameters of PyEMD.EMD(), PyEMD.EEMD(), PyEMD.CEEMDAN(), vmdpy.VMD()
                    - eg. trials for PyEMD.CEEMDAN(), change the number of inputting white noise 
    Output:
    ---------------------
    df_redecom      - the redecomposing results in pd.Dataframe
    df_redecom_list - each IMF's redecomposing results and itself as target
    """

    from CEEMDAN_LSTM.core import check_dataset
    data = check_dataset(data, show, decom_mode, redecom_list)
    if vmd_params is not None and type(vmd_params) != dict: raise ValueError('Invalid input of vmd_params!') 
    if len(data.columns) == 1: 
        df_decom = decom(data[data.columns[0]], decom_mode=decom_mode, vmd_params=vmd_params, FORECAST_LENGTH=FORECAST_LENGTH)
    else: df_decom = data # pd.DataFrame
    df_inte, df_inte_list = inte(df_decom, inte_list=inte_list)

    # Re-decompose 
    best_params_dict, inte_columns = {}, str(df_inte.columns.to_list())
    if redecom_list is not None:
        try: redecom_list = pd.DataFrame(redecom_list, index=range(1))
        except: raise ValueError("Invalid input for redecom_list! Please input eg. None, '{'co-imf0':'vmd', 'co-imf1':'emd'}'.")
        for i in redecom_list.columns:
            try: df_inte[i]
            except: raise ValueError('Invalid input for redecom_list! Please check your key value of column name.')
            if vmd_params is None: df_imf_decom = decom(df_inte[i], decom_mode=redecom_list[i][0], FORECAST_LENGTH=FORECAST_LENGTH, **kwargs)
            else: 
                print('Get vmd_params:', vmd_params)
                df_imf_decom = decom(df_inte[i], decom_mode=redecom_list[i][0], vmd_params=vmd_params[i], FORECAST_LENGTH=FORECAST_LENGTH, **kwargs)
            best_params_dict[i] = eval(df_imf_decom.name.split('_')[1]) # get best_params_dict
            df_imf_decom = df_imf_decom.drop('target', axis=1, inplace=False)
            df_imf_decom.columns = [i+'-'+redecom_list[i][0].lower()+str(x) for x in range(df_imf_decom.columns.size)] # rename
            df_imf_decom.insert(0, i, df_inte[i].values)
            df_imf_decom.name = 'redecom_'+i
            for j in range(len(df_inte_list)):
                if df_inte_list[j].name == i: df_inte_list[j] = df_imf_decom # record each redecomposition result
        df_redecom = df_inte['target']
        for df in df_inte_list: df_redecom = pd.concat((df_redecom, df.iloc[:,1:]),axis=1)
        df_redecom.name = inte_columns+'_'+str(best_params_dict) # record redecomposition parameters
    else: df_redecom = df_inte
        
    if show:
        print('\nPart of preprocessing dataset (inculde training and test set):')
        print(df_redecom)
    return df_redecom, df_inte_list



# 3.Main. Evaluate
def eval_result(y_real, y_pred): 
    """
    Evaluate forecasting result by sklearn.metrics, eg.'R2', 'RMSE', 'MAE', 'MAPE'.
    Example: df_eval = cl.eval_result(y_real, y_pred)
    Print: print(df_eval)

    Input and Parameters:
    ---------------------
    y_real    - target values in test set
    y_pred    - forecasting result

    Output:
    ---------------------
    df_eval   - the evaluate forecasting result in pd.Dataframe
    """

    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error # R2, MSE, MAE, MAPE
    y_real, y_pred = np.array(y_real).ravel(), np.array(y_pred).ravel()
    scale = np.max(y_real) - np.min(y_real) # scale is important for RMSE and MAE
    r2 = r2_score(y_real, y_pred)
    rmse = mean_squared_error(y_real, y_pred, squared=False) # RMSE and MAE are various on different scales
    mae = mean_absolute_error(y_real, y_pred)
    mape = mean_absolute_percentage_error(y_real, y_pred) # Note that dataset cannot have any 0 value.
    df_eval = pd.DataFrame({'Scale':scale, 'R2':r2, 'RMSE':rmse, 'MAE':mae, 'MAPE':mape}, index=[0])
    return df_eval

# 3.1. Normalize
def normalize_dataset(data=None, FORECAST_LENGTH=None, NOR_METHOD='MinMax'):
    """
    Normalize and split x-feature and y-target

    Input and Parameters:
    ---------------------
    data               - data set
    FORECAST_LENGTH    - the length of the days to forecast (test set)
    fit_method         - set the fitting method to stablize the forecasting result (not necessarily useful) eg. 'add', 'ensemble'
    nor_method         - set normalizing method 'minmax'-MinMaxScaler, 'std'-StandardScaler, otherwise without normalization

    Output
    ---------------------
    x_train, y_train   - training set
    x_test, y_test     - test set
    scalarY            - normalization scalar for y
    next_x             - use for predicting only next day (useless if next_predict=False)
    """
    
    # import 
    try: from sklearn.preprocessing import MinMaxScaler, StandardScaler
    except ImportError: raise ImportError('Cannot import scikit-learn(sklearn), run: pip install scikit-learn!')
    try: data = pd.DataFrame(data)
    except: raise ValueError('Invalid input!')

    # Split
    if len(data.columns) == 1: # Initialize Series
        dataY = data.values.reshape(-1, 1)
        dataX = dataY
    else: # Initialize DataFrame training set and test set
        dataY = data['target'].values.reshape(-1, 1)
        dataX = data.drop('target', axis=1, inplace=False)

    # Setting normalizing method
    if NOR_METHOD is None: NOR_METHOD=''
    if NOR_METHOD.lower() == 'minmax':
        scalarX = MinMaxScaler(feature_range=(0,1)) 
        scalarY = MinMaxScaler(feature_range=(0,1)) 
    elif NOR_METHOD.lower() == 'std':
        scalarX = StandardScaler() 
        scalarY = StandardScaler() 
    else: 
        scalarY = None
        print("Warning! Data is not normalized, please set nor_method = eg.'minmax', 'std'")

    # Normalize by sklearn
    if scalarY is not None:
        if FORECAST_LENGTH is None: dataX = scalarX.fit(dataX) # Normalize X
        else: scalarX.fit(dataX[:-FORECAST_LENGTH]) # Avoid using training set for normalization  
        dataX = scalarX.transform(dataX)
        # if fit_method is not None: fit_method = scalarX.transform(fitting_set)

        if FORECAST_LENGTH is None: dataY = scalarY.fit(dataY)
        else: scalarY.fit(dataY[:-FORECAST_LENGTH]) # Avoid using training set for normalization
        dataY = scalarY.transform(dataY)

    return np.array(dataX), np.array(dataY), scalarY

# 3.Main. Create training set and test set
def create_train_test_set(data=None, FORECAST_LENGTH=None, FORECAST_HORIZONS=None, NEXT_DAY=False, NOR_METHOD='MinMax', DAY_AHEAD=1, fitting_set=None):
    """
    Create training set and test set with normalization

    Input and Parameters:
    ---------------------
    data               - data set
    FORECAST_HORIZONS  - the length of each input row(x_train.shape), which means the number of previous days related to today
    FORECAST_LENGTH    - the length of the days to forecast (test set)
    fit_method         - set the fitting method to stablize the forecasting result (not necessarily useful) eg. 'add', 'ensemble'
    nor_method         - set normalizing method 'minmax'-MinMaxScaler, 'std'-StandardScaler, otherwise without normalization

    Output
    ---------------------
    x_train, y_train   - training set
    x_test, y_test     - test set
    scalarY            - normalization scalar for y
    next_x             - use for predicting only next day (useless if next_predict=False)
    """

    # Normalize
    if FORECAST_LENGTH is None: raise ValueError('Please input a FORECAST_LENGTH!')
    if FORECAST_HORIZONS is None: raise ValueError('Please input a FORECAST_HORIZONS!')
    dataX, dataY, scalarY  = normalize_dataset(data, FORECAST_LENGTH, NOR_METHOD) # data X Y is np.array here

    # Create training set and test set
    trainX, trainY = [], [] 
    next_x = np.array(dataX[-FORECAST_HORIZONS:])
    for i in range(len(dataY)-FORECAST_HORIZONS-DAY_AHEAD+1): # the original version is for DAY_AHEAD=1
        trainX.append(np.array(dataX[i:(i+FORECAST_HORIZONS)])) # each x_train unit is a vector or matrix
        trainY.append(np.array(dataY[i+FORECAST_HORIZONS+DAY_AHEAD-1]))  
            
        if fitting_set is not None: # When fitting, it uses today's forecasting result 
            len_train = len(dataY)-FORECAST_HORIZONS-FORECAST_LENGTH-DAY_AHEAD+1
            # for Next-day forecast, fitting_set has only 1 row
            if i == len_train: next_x = np.insert(next_x, FORECAST_HORIZONS, fitting_set.values[-1], axis=0) 
            if DAY_AHEAD == 0:
                raise ValueError('Warning! When DAY_AHEAD = 0, it is not support the fitting method, already fit today.')
            elif DAY_AHEAD == 1:
                if i < len_train: trainX[i] = np.insert(trainX[i], FORECAST_HORIZONS, dataX[i+FORECAST_HORIZONS], axis=0)
                else: trainX[i] = np.insert(trainX[i], FORECAST_HORIZONS, fitting_set.values[i-len_train], axis=0) # fitting_set is pd.DataFrame
            else: # if DAY_AHEAD > 1:
                k = DAY_AHEAD - i + len_train
                if i < len_train: 
                    for j in range(DAY_AHEAD): trainX[i] = np.insert(trainX[i], FORECAST_HORIZONS+j, dataX[i+FORECAST_HORIZONS+j], axis=0)
                elif i >= len_train and i < len_train+DAY_AHEAD: 
                    for x in range(0, k): trainX[i] = np.insert(trainX[i], FORECAST_HORIZONS+x, dataX[i+FORECAST_HORIZONS+x], axis=0)
                    for y in range(k, DAY_AHEAD): trainX[i] = np.insert(trainX[i], FORECAST_HORIZONS+y, fitting_set.values[y-k], axis=0)
                else: 
                    for j in range(DAY_AHEAD): trainX[i] = np.insert(trainX[i], FORECAST_HORIZONS+j, fitting_set.values[j-k], axis=0) 
    if NEXT_DAY:
        x_train, x_test = np.array(trainX), None
        y_train, y_test = np.array(trainY), None
    else:
        x_train, x_test = np.array(trainX[:-FORECAST_LENGTH]), np.array(trainX[-FORECAST_LENGTH:])
        y_train, y_test = np.array(trainY[:-FORECAST_LENGTH]), np.array(trainY[-FORECAST_LENGTH:])

    return x_train, x_test, y_train, y_test, scalarY, next_x # return np.array



# 4.Analysis
# ------------------------------------------------------
def check_series(series):
    try: series = pd.Series(series)
    except: raise ValueError('Sorry! %s is not supported for the Hybrid Method, please input pd.DataFrame, pd.Series, nd.array(<=2D)'%type(data))
    return series

# 4.Main. Statistical tests
def statis_tests(series=None): 
    """
    Make statistical tests, including ADF test, Ljung-Box Test, Jarque-Bera Test, and plot ACF and PACF, to evaluate stationarity, autocorrelation, and normality.
    Input: series     - the time series (1D)
    """
    try: import statsmodels
    except: raise ImportError('Cannot import statsmodels, run: pip install statsmodels!')
    adf_test(series)
    lb_test(series)
    jb_test(series)
    plot_acf_pacf(series)

# 4.1 ADF test
def adf_test(series=None):
    """
    Make Augmented Dickey-Fuller test (ADF test) to evaluate stationarity.
    Input: series     - the time series (1D)
    """
    from statsmodels.tsa.stattools import adfuller # adf_test
    series = check_series(series)
    adf_ans = adfuller(series) # The outcomes are test value, p-value, lags, degree of freedom.
    print('\n==========ADF Test==========')
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

# 4.2 Ljung-Box Test
def lb_test(series=None):
    """
    Make Ljung-Box Test to evaluate autocorrelation.
    Input: series     - the time series (1D)
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test # LB_test
    series = check_series(series)
    lb_ans = lb_test(series,lags=None,boxpierce=False) # The default lags=40 for long series.
    print('\n==========Ljung-Box Test==========')
    pd.Series(lb_ans['lb_pvalue']).plot(title='Ljung-Box Test p-values') # Plot p-values in a figure
    plt.show()
    print(lb_ans)
    if np.sum(lb_ans['lb_pvalue'])<=0.05: # Brief review
        print('The sum of p-value is '+str(np.sum(lb_ans['lb_pvalue']))+'<=0.05, rejecting the null hypothesis that the series has very strong autocorrelation.')
    else: print('Please view with the line chart, the autocorrelation of the series may be not strong.')
    # print(pd.DataFrame(lb_ans)) # Show outcomes with test value at line 0, and p-value at line 1.

# 4.3 Jarque-Bera Test 
def jb_test(series=None):
    """
    Make Jarque-Bera Test to evaluate normality (whether conforms to a normal distribution).
    Input: series     - the time series (1D)
    """
    from statsmodels.stats.stattools import jarque_bera as jb_test # JB_test
    series = check_series(series)
    jb_ans = jb_test(series) # The outcomes are test value, p-value, skewness and kurtosis.
    print('\n==========Jarque-Bera Test==========')
    print('Test value:',jb_ans[0])
    print('P value:',jb_ans[1])
    print('Skewness:',jb_ans[2])
    print('Kurtosis:',jb_ans[3])
    # Brief review
    if jb_ans[1]<=0.05: 
        print('p-value is '+str(jb_ans[1])+'<=0.05, rejecting the null hypothesis that the series has no normality.')
    else:
        print('p-value is '+str(jb_ans[1])+'>=0.05, accepting the null hypothesis that the series has certain normality.')

# 4.4 Plot ACF and PACF figures
def plot_acf_pacf(series=None, fig_path=None):
    """
    Plot ACF and PACF figures to evaluate autocorrelation and find the lag.
    Input: 
    series     - the time series (1D)
    fig_path   - the figure saving path
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf # plot_acf_pacf
    series = check_series(series)
    print('\n==========ACF and PACF==========')
    fig = plt.figure(figsize=(10,5))
    fig1 = fig.add_subplot(211)
    plot_acf(series, lags=40, ax=fig1)
    fig2 = fig.add_subplot(212)
    plot_pacf(series, lags=40, ax=fig2)
    plt.tight_layout()
    # Save the figure
    if fig_path is not None:
        plt.savefig(fig_path+'Figures_ACF_PACF.jpg', dpi=300, bbox_inches='tight')
    plt.show()

# Plot Heatmap
def plot_heatmap(data, corr_method='pearson', fig_path=None):
    """
    Plot heatmap to check the correlation between variables.
    Input: 
    data         - the 2D array
    corr_method  - the method to calculate the correlation
    fig_path     - the figure saving path
    """
    try: import seaborn as sns
    except: raise ImportError('Cannot import seaborn, run: pip install seaborn!')
    try: data = pd.DataFrame(data)
    except: raise ValueError('Invalid input!')
    f, ax= plt.subplots(figsize = (14, 10))
    sns.heatmap(data.corr(corr_method), cmap='OrRd', linewidths=0.05, ax=ax, annot=True, fmt='.5g') #RdBu
    if fig_path is not None:
        plt.savefig(fig_path+'_Heatmap.jpg', dpi=300, bbox_inches='tight')
        plt.tight_layout() 
        plt.show()
    plt.show()

# 4.5 DM test
def dm_test(actual_lst, pred1_lst, pred2_lst, h=1, crit="MSE", power=2):
    """
    Copyright (c) 2017 John Tsang
    Author: John Tsang https://github.com/johntwk/Diebold-Mariano-Test
    Diebold-Mariano-Test (DM test) is used to compare time series forecasting result performance.

    Input and Parameters:
    ---------------------
    actual_lst    - target values in test set
    pred1_lst     - forecasting result
    pred2_lst     - another forecasting result
    h             - the number of stpes ahead
    crit          - a string specifying the criterion eg. MSE, MAD, MAPE, poly
                        1)  MSE : the mean squared error               
                        2)  MAD : the mean absolute deviation
                        3) MAPE : the mean absolute percentage error
                        4) poly : use power function to weigh the errors
    poly          - the power for crit power (it is only meaningful when crit is "poly")
        
    Output:
    ---------------------
    DM	          - The DM test statistics
    p-value	      - The p-value of DM test statistics
    """

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

    # Import libraries
    from scipy.stats import t
    import collections
    import pandas as pd
    import numpy as np
    
    # Initialise lists
    e1_lst = []
    e2_lst = []
    d_lst  = []
    
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

