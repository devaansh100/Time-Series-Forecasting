import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from collections import Counter
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf as get_pacf
from statsmodels.tsa.stattools import acf as get_acf
from statsmodels.tsa.arima.model import ARIMA as ARIMA_model
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from math import sqrt
from math import ceil
import os

def plot_correlations(daily_data, weekly_data):
	for idx, region in enumerate(daily_data):
		pacf = get_pacf(region['Wind Speed'])
		x = [i for i in range(len(pacf))]
		plt.figure(f'Plots/PACF_Daily_Rajasthan{idx+1}')
		plt.stem(x, pacf)
		plt.savefig(f'Plots/PACF_Daily_Rajasthan{idx+1}.png')

	for idx, region in enumerate(weekly_data):
		pacf = get_pacf(region['Wind Speed'])
		x = [i for i in range(len(pacf))]
		plt.figure(f'Plots/PACF_Weekly_Rajasthan{idx+1}')
		plt.stem(x, pacf)
		plt.savefig(f'Plots/PACF_Weekly_Rajasthan{idx+1}')

	for idx, region in enumerate(daily_data):
		pacf = get_acf(region['Wind Speed'])
		x = [i for i in range(len(pacf))]
		plt.figure(f'Plots/ACF_Daily_Rajasthan{idx+1}')
		plt.stem(x, pacf)
		plt.savefig(f'Plots/ACF_Daily_Rajasthan{idx+1}')

	for idx, region in enumerate(weekly_data):
		pacf = get_acf(region['Wind Speed'])
		x = [i for i in range(len(pacf))]
		plt.figure(f'Plots/ACF_Weekly_Rajasthan{idx+1}')
		plt.stem(x, pacf)
		plt.savefig(f'Plots/ACF_Weekly_Rajasthan{idx+1}')

	plt.show()

def fit_model(models, test, model_name, daily = False, show_prediction_plots = True):
	for idx, model in enumerate(models):
		print(f'Summary of Model for Region {idx + 1}')
		print(model.summary())
	n_preds = [len(test[i]) for i in range(len(test))]
	if model_name != 'SARIMA':
		preds = [models[idx].predict(start = len(train[idx]), end = len(train[idx]) + n_preds[idx] - 1, dynamic = False) for idx, model in enumerate(models)]
	else:
		preds = [models[idx].forecast(steps=n_preds[idx]) for idx in range(len(train))]
	gt = [test[idx][0:n_preds[idx]] for idx in range(len(test))]
	save_op(preds, daily, model_name)
	if show_prediction_plots:
		plot_op(n_preds, preds, gt, daily, model_name)

def save_op(preds, daily, model):
	time = 'daily' if daily else 'weekly'
	preds = np.array(preds)
	np.save(f"Model Outputs/{model}_{time}_preds", preds)

def plot_op(n_preds, preds, gt, daily, model):
	time = 'daily' if daily else 'weekly'
	try:
		for idx, pred in enumerate(preds):
			x = [i for i in range(n_preds[idx])]
			plt.figure(f"{model}_{time}_Rajasthan{idx + 1}")
			plt.plot(x, gt[idx], color = 'red', label = f'Ground Truth')
			plt.plot(x, pred, color = 'green', label = f'Predictions')
			plt.legend()
			# plt.show()
			plt.savefig(f"Plots/{model}_{time}_Rajasthan{idx + 1}.png")
	except:
		pass

def read_data():
	root = os.getcwd() + '/'
	os.chdir(root + 'Rajasthan1')
	files = os.listdir()
	r1 = pd.concat([pd.read_csv(file, skiprows = 2, usecols = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Wind Speed']) for file in files])

	os.chdir(root + 'Rajasthan2')
	files = os.listdir()
	r2 = pd.concat([pd.read_csv(file, skiprows = 2, usecols = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Wind Speed']) for file in files])

	os.chdir(root + 'Rajasthan3')
	files = os.listdir()
	r3 = pd.concat([pd.read_csv(file, skiprows = 2, usecols = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Wind Speed']) for file in files])

	os.chdir(root + 'Rajasthan4')
	files = os.listdir()
	r4 = pd.concat([pd.read_csv(file, skiprows = 2, usecols = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Wind Speed']) for file in files])

	os.chdir(root + 'Rajasthan5')
	files = os.listdir()
	r5 = pd.concat([pd.read_csv(file, skiprows = 2, usecols = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Wind Speed']) for file in files])

	os.chdir(root)

	return r1, r2, r3, r4, r5


def split_data(data):
	train_data = []
	test_data = []
	# Each element in train_data and test_data is for a different region
	for region in data:
		train_data.append(list(region['Wind Speed'][:-30]))
		test_data.append(list(region['Wind Speed'][-30:]))

	return train_data, test_data

def check_stationarity(data):
	p = [] # Weekly and Daily data has stationarity
	for region in data:
		ws = region['Wind Speed']
		out = adfuller(ws)
		p.append(out[1])
	print(p)

def get_years_in_data(data):
	years = []
	for region in data:
		years.append(sorted(list(Counter(region['Year']).keys())))
	print(years)

def convert_data(data, skips):
	dfs = []
	time = 'Day' if skips == 24 else 'Week'
	mean = lambda x : sum(x)/len(x)
	for region in data:
		rows = []
		for i in range(ceil(len(region)/skips)):
			try:
				rows.append([f'{time} {i}', mean(region['Wind Speed'][skips*i:skips*(i+1)])])
			except:
				rows.append([f'{time} {i}', mean(region['Wind Speed'][skips*i:])])
		dfs.append(pd.DataFrame(rows, columns = [f'{time}s', 'Wind Speed']))

	return dfs[0], dfs[1], dfs[2], dfs[3], dfs[4]

def eval_model(preds, gt, model, daily = False):
	eval_rmse = []
	eval_mae = []
	eval_mape = []
	time = 'daily' if daily else 'weekly'

	for idx in range(len(preds)):
		eval_mape.append(mape(preds[idx], gt[idx]))
		eval_mae.append(mae(preds[idx], gt[idx]))
		eval_rmse.append(sqrt(mse(preds[idx], gt[idx])))

	eval_rmse = np.array(eval_rmse)
	eval_mae = np.array(eval_mae)
	eval_mape = np.array(eval_mape)

	np.save(f'Model Outputs/RMSE_{model}_{time}', eval_rmse)
	np.save(f'Model Outputs/MAE_{model}_{time}', eval_mae)
	np.save(f'Model Outputs/MAPE_{model}_{time}', eval_mape)

def create_eval_table():
	rmse_ar = np.load(f'Model Outputs/RMSE_AR_weekly.npy')
	mae_ar = np.load(f'Model Outputs/MAE_AR_weekly.npy')
	mape_ar = np.load(f'Model Outputs/MAPE_AR_weekly.npy')

	rmse_ma = np.load(f'Model Outputs/RMSE_MA_weekly.npy')
	mae_ma = np.load(f'Model Outputs/MAE_MA_weekly.npy')
	mape_ma = np.load(f'Model Outputs/MAPE_MA_weekly.npy')

	rmse_arma = np.load(f'Model Outputs/RMSE_ARMA_weekly.npy')
	mae_arma = np.load(f'Model Outputs/MAE_ARMA_weekly.npy')
	mape_arma = np.load(f'Model Outputs/MAPE_ARMA_weekly.npy')

	rmse_arima = np.load(f'Model Outputs/RMSE_ARIMA_weekly.npy')
	mae_arima = np.load(f'Model Outputs/MAE_ARIMA_weekly.npy')
	mape_arima = np.load(f'Model Outputs/MAPE_ARIMA_weekly.npy')

	rmse_sarima = np.load(f'Model Outputs/RMSE_SARIMA_weekly.npy')
	mae_sarima = np.load(f'Model Outputs/MAE_SARIMA_weekly.npy')
	mape_sarima = np.load(f'Model Outputs/MAPE_SARIMA_weekly.npy')

	rmse_ar_d = np.load(f'Model Outputs/RMSE_AR_daily.npy')
	mae_ar_d = np.load(f'Model Outputs/MAE_AR_daily.npy')
	mape_ar_d = np.load(f'Model Outputs/MAPE_AR_daily.npy')

	header = ['Model', 'Parameters', 'Frequency', 'Metric', 'Region 1', 'Region 2', 'Region 3', 'Region 4', 'Region 5']
	row1  = ['AR', 		'35', 						'Weekly', 'RMSE', *rmse_ar]
	row2  = ['MA', 		'31', 						'Weekly', 'RMSE', *rmse_ma]
	row3  = ['ARMA', 	'(35, 31)', 				'Weekly', 'RMSE', *rmse_arma]
	row4  = ['ARIMA', 	'(35, 1, 31)', 				'Weekly', 'RMSE', *rmse_arima]
	row5  = ['SARIMA',  '(35, 0, 31) (0, 0, 0, 0)', 'Weekly', 'RMSE', *rmse_sarima]
	row6  = ['AR', 		'35', 						'Weekly', 'MAE',  *mae_ar]
	row7  = ['MA', 		'31', 						'Weekly', 'MAE',  *mae_ma]
	row8  = ['ARMA', 	'(35, 31)', 				'Weekly', 'MAE',  *mae_arma]
	row9  = ['ARIMA', 	'(35, 1, 31)', 				'Weekly', 'MAE',  *mae_arima]
	row10 = ['SARIMA',  '(35, 0, 31) (0, 0, 0, 0)', 'Weekly', 'MAE',  *mae_sarima]
	row11 = ['AR', 		'35', 						'Weekly', 'MAPE', *mape_ar]
	row12 = ['MA', 		'31', 						'Weekly', 'MAPE', *mape_ma]
	row13 = ['ARMA', 	'(35, 31)', 				'Weekly', 'MAPE', *mape_arma]
	row14 = ['ARIMA', 	'(35, 1, 31)', 				'Weekly', 'MAPE', *mape_arima]
	row15 = ['SARIMA',  '(35, 0, 31) (0, 0, 0, 0)', 'Weekly', 'MAPE', *mape_sarima]
	row16 = ['AR',      '245', 						'Daily', 'RMSE', *rmse_ar_d]
	row17 = ['AR', 		'245', 						'Daily', 'MAPE', *mape_ar_d]
	row18 = ['AR', 		'245', 						'Daily', 'MAE',  *mae_ar_d]
	rows = [row1, row2, row3, row4, row5, row6, row7, row8, row9, row10, row11, row12, row13, row14, row15, row16, row17, row18]
	df = pd.DataFrame(rows, columns = header)
	df.to_csv('Model Outputs/eval.csv')
	return df

if __name__ == '__main__':
	warnings.filterwarnings("ignore")
	data = read_data()
	weekly_data = convert_data(data, 168)
	daily_data = convert_data(data, 24)
	plot_correlations(daily_data, weekly_data)
	train, test   = split_data(weekly_data)
	print("#############################################")
	print("Running AR model for weekly data with p = 35")
	ar_models     = [AutoReg(train[idx], 35).fit() for idx, region in enumerate(data)]
	fit_model(ar_models, test, 'AR')
	print("#############################################")
	print("Running MA model for weekly data with q = 31")
	ma_models     = [ARIMA_model(train[idx], order = (0,0,31)).fit() for idx, region in enumerate(data)]
	fit_model(ma_models, test, 'MA')
	print("#############################################")
	print("Running ARMA model for weekly data with (35, 0, 31)")
	arma_models   = [ARIMA_model(train[idx], order = (35,0,31)).fit() for idx, region in enumerate(data)]
	fit_model(arma_models, test, 'ARMA')
	print("#############################################")
	print("Running ARIMA model for weekly data with (35, 1, 31)")
	arima_models  = [ARIMA_model(train[idx], order = (35,1,31)).fit() for idx, region in enumerate(data)]
	fit_model(arima_models, test, 'ARIMA')
	print("#############################################")
	print("Running SARIMA model for weekly data with (35, 0, 31)(0, 0, 0, 0)")
	sarima_models = [SARIMAX(train[idx], order = (35,1,31), seasonal_order = (0, 0, 0, 0)).fit() for idx, region in enumerate(data)]
	fit_model(sarima_models, test, 'SARIMA')

	train, test   = split_data(daily_data)
	print("#############################################")
	print("Running AR model for daily data with p = 245")
	ar_models_d   = [AutoReg(train[idx], 245).fit() for idx, region in enumerate(data)]
	fit_model(ar_models_d, test, 'AR', daily = True)

	preds = np.load('Model Outputs/ARIMA_weekly_preds.npy')
	_, gt = split_data(weekly_data)
	eval_model(preds, gt, 'ARIMA')

	preds = np.load('Model Outputs/AR_weekly_preds.npy')
	eval_model(preds, gt, 'AR')

	preds = np.load('Model Outputs/MA_weekly_preds.npy')
	eval_model(preds, gt, 'MA')

	preds = np.load('Model Outputs/ARMA_weekly_preds.npy')
	eval_model(preds, gt, 'ARMA')

	preds = np.load('Model Outputs/SARIMA_weekly_preds.npy')
	eval_model(preds, gt, 'SARIMA')

	preds = np.load('Model Outputs/AR_daily_preds.npy')
	_, gt = split_data(daily_data)
	eval_model(preds, gt, 'AR', True)
	create_eval_table()
