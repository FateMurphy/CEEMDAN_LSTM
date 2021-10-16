These csv files contain the model evaluation results for each run, respectively R2, RMSE, MAE, MAPE and time in order.

Column name: R2 RMSE MAE MAPE Time

It took a total of nearly 200,000 seconds to complete the test, about 55 hours. As for the test of 1000 epochs, it is not necessary.

The correspondence between patience and epochs is below and the remaining parameters are unchanged.
100 epochs - 10 patience
300 epochs - 30 patience
1000 epochs - 100 patience
(patience=EPOCHS/10)
(early stop=5*PATIENCE)
