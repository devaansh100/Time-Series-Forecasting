This code was written as part of the project for the course MATH F432, Applied Statistical Methods at BITS Pilani. 

# Time Series Forecasting
The aim of the project was to analyse the wind speed data for five different locations in Rajasthan in order to predict the weekly and daily wind speed forecasts.

For this purpose, 5 models were trained - the AR, MA, ARMA, ARIMA and SARIMA.

The code reads all the data, transforms it into the necessary format, trains the 5 models and finally plots and saves the output forecasts, along with three evaluation metrics for each model.

The parameters of the model were chosen by a close examination of the autocorrelation and partial autocorrelation plots of the dataset.
