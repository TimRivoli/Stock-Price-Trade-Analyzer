import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from _classes.Prices import PriceSnapshot, PricingData
from _classes.SeriesPrediction import StateSurprisePredictionNN
from _classes.Utility import *

dataFolder = 'data/prediction/'
CreateFolder(dataFolder)

def train_model(sourceDF, prediction_target_days:int = 5, epochs:int = 300):
	model = StateSurprisePredictionNN(horizon=prediction_target_days)
	model.LoadSource(sourceDF)
	model.LoadTarget()
	model.MakeTrainTest()
	model.BuildModel()
	model.Train(epochs=epochs)
	return model.Predict()

def evaluate_model(rawSourceDF, predictions, horizon):
# predictions: DataFrame from model.Predict() or PredictOnNewData()
# Must contain: baseline, predicted_price, direction_prob, magnitude
# rawSourceDF: original OHLCV DataFrame used for prediction
# horizon: prediction horizon (same as model.horizon)
	future_price = (rawSourceDF['Average'].shift(-horizon).loc[predictions.index])
	df = predictions.copy()
	df['actual_price'] = future_price
	df = df.dropna()
	b = df['baseline'].values
	p = df['predicted_price'].values
	y = df['actual_price'].values
	r = y - b                 # true residual
	r_hat = p - b             # predicted residual
	# ----------------------------
	# 1️ Residual RMSE
	# ----------------------------
	rmse_baseline = np.sqrt(np.mean(r ** 2))
	rmse_model = np.sqrt(np.mean((r - r_hat) ** 2))

	print("\n=== Residual RMSE ===")
	print(f"Baseline (zero residual): {rmse_baseline:.4f}")
	print(f"Model residual RMSE:      {rmse_model:.4f}")
	print(f"Improvement:              {rmse_baseline - rmse_model:.4f}")

	# ----------------------------
	# 2️ Residual correlation
	# ----------------------------
	corr = np.corrcoef(r, r_hat)[0, 1]
	print("\n=== Residual Correlation ===")
	print(f"Corr(true, predicted): {corr:.4f}")

	# ----------------------------
	# 3 Directional accuracy (conditional)
	# ----------------------------
	# Only evaluate where model says magnitude is large
	mag_threshold = np.percentile(np.abs(r_hat), 70)
	mask = np.abs(r_hat) > mag_threshold

	if mask.sum() > 0:
		direction_acc = np.mean(
			np.sign(r[mask]) == np.sign(r_hat[mask])
		)
		print("\n=== Directional Accuracy (Top 30% Magnitude) ===")
		print(f"Accuracy: {direction_acc:.4f}")
		print(f"Samples:  {mask.sum()}")
	else:
		print("\n=== Directional Accuracy ===")
		print("No samples above magnitude threshold.")

	# ----------------------------
	# 4️ Calibration by decile
	# ----------------------------
	calib_df = pd.DataFrame({        'r_hat': r_hat,        'r': r    })
	calib_df['decile'] = pd.qcut(calib_df['r_hat'], 10, duplicates='drop')
	calib = calib_df.groupby('decile').mean()
	print("\n=== Calibration (Mean True Residual by Decile) ===")
	print(calib)

	# Plot calibration
	plt.figure(figsize=(6, 4))
	plt.plot(calib['r_hat'], calib['r'], marker='o')
	plt.axhline(0, color='gray', linestyle='--')
	plt.xlabel("Predicted Residual (Decile Mean)")
	plt.ylabel("Actual Residual (Mean)")
	plt.title("Residual Calibration")
	plt.grid(True)
	plt.show()

	# ----------------------------
	# 5 Simple PnL sanity check (optional)
	# ----------------------------
	position = np.sign(r_hat) #1 = Buy, 2 = Short
	pnl = position * r		  #return from that decision
	sharpe_daily = (pnl.mean() / pnl.std() if pnl.std() > 0 else 0.0) #Returns.mean()/Returns.std()
	sharpe_annualized = sharpe_daily * np.sqrt(252)
	print("\n=== Simple PnL Sanity Check ===")
	print(f"Mean PnL: {pnl.mean():.6f}")
	print(f"Sharpe Daily (naive): {sharpe_daily:.3f}")
	print(f"Sharpe Annualized (naive): {sharpe_annualized:.3f}")

	# ----------------------------
	# Return metrics for programmatic use
	# ----------------------------
	return {
		'rmse_baseline': rmse_baseline,
		'rmse_model': rmse_model,
		'residual_corr': corr,
		'directional_accuracy': direction_acc if mask.sum() > 0 else np.nan,
		'mean_pnl': pnl.mean(),
		'sharpe_daily': sharpe_daily,
		'sharpe_annualized': sharpe_annualized
	}

def plot_results(df, start_date, latest_date):
	df = df.reset_index()
	df = df[(df['Date'] >= start_date) & (df['Date'] <= latest_date)].copy()
	# Create a two-panel subplot
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

	# Panel 1: Price Comparison
	ax1.plot(df['Date'], df['baseline'], label='Baseline Price', marker='o', color='#1f77b4', linewidth=2)
	ax1.plot(df['Date'], df['predicted_price'], label='Predicted Price', marker='x', linestyle='--', color='#ff7f0e', linewidth=2)
	ax1.set_ylabel('Price Value')
	ax1.set_title('Financial Forecast: Price vs. Baseline')
	ax1.legend()
	ax1.grid(True, alpha=0.3)

	# Panel 2: Confidence Metrics (Probability and Magnitude)
	ax2.bar(df['Date'], df['direction_prob'], color='lightgray', label='Direction Prob', width=0.4)
	ax2.set_ylabel('Probability', color='gray')
	ax2.set_ylim(0, 1.1)
	ax2.set_title('Model Confidence & Movement Magnitude')

	# Secondary axis for Magnitude
	ax3 = ax2.twinx()
	ax3.plot(df['Date'], df['magnitude'], color='red', marker='s', label='Magnitude', linewidth=1)
	ax3.set_ylabel('Magnitude', color='red')
	ax3.tick_params(axis='y', labelcolor='red')

	plt.xticks(df['Date'], rotation=45)
	plt.xticks(rotation=45) 
	ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1)) 
	#ax2.xaxis.set_major_locator(mdates.DayLocator(interval=7))
	ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
	fig.autofmt_xdate()
	plt.tight_layout()
	plt.show()

ticker = 'GOOGL'
do_training = True
print('Loading ' + ticker)
prices = PricingData(ticker, useDatabase=False)
startDate = '1/1/2005'
endDate = '1/1/2026'
prediction_target_days = 22

if prices.LoadHistory(requestedStartDate=startDate, requestedEndDate=endDate):
	csvFile = os.path.join(dataFolder, f"{ticker}_{prediction_target_days}_days_predictions.csv")
	sourceDF = prices.historicalPrices
	print(sourceDF)
	if do_training:
		predictions  = train_model(sourceDF, prediction_target_days, 450)
		predictions.reset_index().to_csv(csvFile, index=False)
	else:
		predictions = pd.read_csv(csvFile, index_col=0, parse_dates=['Date'])
	evaluate_model(sourceDF, predictions, prediction_target_days)
	end_date = predictions.index.max()
	start_date = end_date - pd.Timedelta(days=90)
	plot_results(predictions, start_date, end_date)