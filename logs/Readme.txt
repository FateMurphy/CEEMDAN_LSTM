These csv files contain the model evaluation results for each run, respectively R2, RMSE, MAE, MAPE and time in order.
They are the results of my paper. If you are interested, you can look my paper and compare the results here.
It took a total of nearly 200,000 seconds to complete the test, about 55 hours. As for the test of 1000 epochs, it is not necessary.


###Column name: R2 RMSE MAE MAPE Time


All works in this paper are completed on a computer with Anaconda3 Individual Edition, Anaconda Navigator 2.0.4, and Jupyter Notebook 6.3.0. 
With the help of CUDA 11.2, it mainly uses Python 3.8.8 and the Keras module of TensorFlow 2.5.0 with TensorFlow-gpu 2.5.0 to build an LSTM recurrent neural network model.
As for CEEMDAN, it uses the module called EMD-signal 1.0.0, which also includes EMD and EEMD methods. 
Besides, the Python module called Statsmodels 0.12.2 is used for statistical tests.
Matplotlib 3.3.4 is used for plotting pictures, and Scikit-learn 0.24.1 is used to evaluate models’ performance.
Table below shows the hardware and compiling environment of the computer used. 
Compared with CPU, using GPU to train neural network models to predict financial time series, such as carbon price, can shorten the calculation time by nearly 30% in subsequent forecasting tests. 


###Hardware and compiling environment

Central Processing Unit (CPU): AMD Ryzen5 3600X 6-Core Processor 3.80GHz
Random Access Memory (RAM): 32GB 2133MHz
Operating System (OS): Windows 10 Professional x64
Graphics Processing Unit (GPU): NVIDIA GeForce GTX 1060 6GB
Programming language: Python 3.8.8
Development Environment: Anaconda3 Individual Edition; Anaconda Navigator 2.0.4; Jupyter Notebook 6.3.0; CUDA 11.2
Main Python Modules: NumPy 1.19.5; Pandas 1.2.4; Matplotlib 3.3.4; Scikit-learn 0.24.1; Statsmodels 0.12.2; EMD-signal 1.0.0; Tensorflow 2.5.0; Tensorflow-gpu 2.5.0

###Patience and epochs

The correspondence between patience and epochs is below and the remaining parameters are unchanged.
100 epochs - 10 patience
300 epochs - 30 patience
1000 epochs - 100 patience
(patience=EPOCHS/10)
(early stop=5*PATIENCE)
