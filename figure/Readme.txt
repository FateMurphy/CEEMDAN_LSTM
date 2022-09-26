These are flowcharts of different methods:

Ensemble forecasting method 
---------------------------------------
kr = cl.keras_predictor(REDECOM_LIST=None)
df_result = kr.ensemble_keras_predict(data, show_data=True, show_model=True, plot_result=True, save_result=True)

Ensemble forecasting method with VMD redecomposition
---------------------------------------
kr = cl.keras_predictor()
df_result = kr.ensemble_keras_predict(data, show_data=True, show_model=True, plot_result=True, save_result=True)

Respective forecasting method
---------------------------------------
kr = cl.keras_predictor(REDECOM_LIST=None)
df_result = kr.respective_keras_predict(data, show_data=True, show_model=True, plot_result=True, save_result=True)

Hybrid forecasting method
---------------------------------------
kr = cl.keras_predictor()
df_result = kr.hybrid_keras_predict(data, show_data=True, show_model=True, plot_result=True, save_result=True)