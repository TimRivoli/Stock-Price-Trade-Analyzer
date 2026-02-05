import json, os
import numpy as np
import pandas as pd
from keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


def compute_volatility_stats(stateDF):
    vol = stateDF['vol_pct']
    return {
        'mean_vol_pct': float(vol.mean()),
        'std_vol_pct': float(vol.std()),
        'pct_high_vol': float((vol > 0.8).mean()),
        'pct_low_vol': float((vol < 0.2).mean())
    }

#-------------------------------------------- Classes -----------------------------------------------
class StateEncoder:
	def transform(self, df):
		# Use typical price as Average if not present
		if 'Average' not in df.columns:
			df = df.copy()
			df['Average'] = (df['High'] + df['Low'] + df['Close']) / 3.0

		returns = df['Average'].pct_change()

		# Volatility regime
		vol_10 = returns.rolling(10).std()
		vol_20 = returns.rolling(20).std()
		vol_40 = returns.rolling(40).std()

		vol_pct = vol_20.rank(pct=True)
		vol_change = vol_10 - vol_40
		vol_of_vol = vol_20.rolling(20).std()

		# Range compression
		true_range = df['High'] - df['Low']
		range_compression = (
			true_range.rolling(5).mean() /
			true_range.rolling(40).mean()
		)

		# Trend / position
		ema_20 = df['Average'].ewm(span=20, adjust=False).mean()
		dist_from_trend = (df['Average'] - ema_20) / vol_20
		momentum_slope = returns.rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0],raw=True)
		drawdown = (df['Average'] - df['Average'].rolling(60).max()	) / df['Average'].rolling(60).max()

		# Participation / pressure
		vol_mean = df['Volume'].rolling(20).mean()
		volume_surprise = df['Volume'] / vol_mean

		dir_vol_imbalance = (
			df['Volume'] * np.sign(returns)
		).rolling(5).sum() / df['Volume'].rolling(5).sum()

		close_pos = (
			(df['Close'] - df['Low']) /
			(df['High'] - df['Low'])
		)

		self.window_config = {
			'vol_short': 10,
			'vol_long': 40,
			'trend_ema': 20,
			'drawdown': 60,
			'range_short': 5,
			'range_long': 40,
			'momentum': 10,
			'volume': 20
		}

		features = pd.DataFrame({
			'vol_pct': vol_pct,
			'vol_change': vol_change,
			'range_compression': range_compression,
			'vol_of_vol': vol_of_vol,
			'dist_from_trend': dist_from_trend,
			'momentum_slope': momentum_slope,
			'drawdown': drawdown,
			'volume_surprise': volume_surprise,
			'dir_vol_imbalance': dir_vol_imbalance,
			'close_position': close_pos
		})

		return features.replace([np.inf, -np.inf], np.nan).dropna()

class BaselineEstimator:
	def predict(self, df: pd.DataFrame, horizon: int) -> pd.Series:
		slope = df['Average'].rolling(10).apply(
			lambda x: np.polyfit(range(len(x)), x, 1)[0],
			raw=True
		)
		return df['Average'] + horizon * slope

class StateSurprisePredictionNN:
	def __init__(self, horizon=5, train_split=0.9):
		self.horizon = horizon
		self.train_split = train_split
		self.encoder = StateEncoder()
		self.baseline = BaselineEstimator()
		self.model = None

	def LoadSource(self, sourceDF):
		self.rawDF = sourceDF.copy()
		self.stateDF = self.encoder.transform(self.rawDF)
		self.input_data = self.stateDF.values
		self.index = self.stateDF.index
		self.num_features = self.input_data.shape[1]

	def LoadTarget(self):
		future_price = (self.rawDF['Average'].shift(-self.horizon))
		baseline_price = self.baseline.predict(self.rawDF, self.horizon)
		residual = future_price - baseline_price
		target_df = pd.DataFrame({'residual': residual})
		target_df = target_df.loc[self.index].dropna()
		self.valid_index = target_df.index
		self.direction = (target_df['residual'] > 0).astype(int).values
		self.magnitude = np.abs(target_df['residual'].values)
		self.magnitude = np.clip(self.magnitude, 0, self.magnitude.mean() * 5)
		self.input_data = self.stateDF.loc[self.valid_index].values
		print("Direction mean:", self.direction.mean())
		print("Magnitude stats:",   np.min(self.magnitude),  np.mean(self.magnitude), np.max(self.magnitude))
		assert not np.isnan(self.magnitude).any()
		assert not np.isinf(self.magnitude).any()
	
	def MakeTrainTest(self):
		n = len(self.input_data)
		cut = int(n * self.train_split)
		self.X_train = self.input_data[:cut]
		self.X_test  = self.input_data[cut:]
		self.y_dir_train = self.direction[:cut]
		self.y_dir_test  = self.direction[cut:]
		self.y_mag_train = self.magnitude[:cut]
		self.y_mag_test  = self.magnitude[cut:]

	def BuildModel(self):
		inp = Input(shape=(self.num_features,))
		x = Dense(64, activation='relu')(inp)
		x = Dense(64, activation='relu')(x)
		direction_out = Dense(1, activation='sigmoid', name='direction')(x)
		magnitude_out = Dense(1, activation='relu', name='magnitude')(x)
		self.model = Model(inp, [direction_out, magnitude_out])
		self.model.compile(
			optimizer=Adam(1e-3),
			loss={
				'direction': 'binary_crossentropy',
				'magnitude': 'huber'
			},
			loss_weights={
				'direction': 1.0,
				'magnitude': 0.5
			}
		)

	def Train(self, epochs=50, batch_size=32):
		self.model.fit(
			self.X_train,
			{
				'direction': self.y_dir_train,
				'magnitude': self.y_mag_train
			},
			validation_data=(
				self.X_test,
				{
					'direction': self.y_dir_test,
					'magnitude': self.y_mag_test
				}
			),
			epochs=epochs,
			batch_size=batch_size
		)
		
	def Predict(self, sourceDF=None, forecast_mode: bool = False):
		"""
		Predict residuals and reconstruct prices.
		forecast_mode=False:
			Evaluation mode. Requires future price to exist.
			Last `horizon` rows are excluded.
		forecast_mode=True:
			Forecast mode. Does NOT require future price.
			Includes last `horizon` rows.
		"""
		if sourceDF is None:
			if not hasattr(self, 'rawDF'):
				raise ValueError("No sourceDF provided and no training data available.")
			sourceDF = self.rawDF
		stateDF = self.encoder.transform(sourceDF)
		baseline = self.baseline.predict(sourceDF, self.horizon)
		if forecast_mode:
			valid_index = (	stateDF.index.intersection(baseline.dropna().index))
		else:
			future_price = sourceDF['Average'].shift(-self.horizon)
			valid_index = (stateDF.index.intersection(baseline.dropna().index).intersection(future_price.dropna().index))
		X = stateDF.loc[valid_index].values
		dir_pred, mag_pred = self.model.predict(X)
		signed_residual = (	mag_pred.flatten() * (2 * dir_pred.flatten() - 1))
		predicted_price = baseline.loc[valid_index].values + signed_residual
		result = pd.DataFrame({
			'baseline': baseline.loc[valid_index].values,
			'predicted_price': predicted_price,
			'direction_prob': dir_pred.flatten(),
			'magnitude': mag_pred.flatten()
		}, index=valid_index)
		if not forecast_mode:
			result['actual_price'] = (
				self.rawDF['Average']
				.shift(-self.horizon)
				.loc[valid_index]
				.values
			)
		return result

	def Save(self, folder='models/', name='state_surprise', metrics=None, notes=None):
		os.makedirs(folder, exist_ok=True)
		self.model.save(os.path.join(folder, name + '.keras'))
		meta = {
			'model_name': name,
			'trained_timestamp_utc': datetime.utcnow().isoformat(),
			'horizon': self.horizon,
			'num_features': self.num_features,
			'feature_names': list(self.stateDF.columns),
			'train_start_date': str(self.valid_index[0]),
			'train_end_date': str(self.valid_index[-1]),
			'num_samples': len(self.valid_index),
			'window_config': getattr(self.encoder, 'window_config', None),
			'git_commit': get_git_commit(),
			'environment': get_env_versions(),
			'volatility_stats': compute_volatility_stats(self.stateDF.loc[self.valid_index]),
		}
		if metrics is not None:
			meta['metrics'] = metrics
		if notes is not None:
			meta['notes'] = notes
		with open(os.path.join(folder, name + '.json'), 'w') as f:
			json.dump(meta, f, indent=2)
		print(f"Model saved to {folder}{name}.keras")

	def Load(self, folder='models/', name='state_surprise'):
		self.model = load_model(os.path.join(folder, name + '.keras'))
		with open(os.path.join(folder, name + '.json'), 'r') as f:
			self.meta = json.load(f)
		self.horizon = self.meta['horizon']
		self.num_features = self.meta['num_features']
		print(f"Model loaded: {name}")
		print(f"Trained on {self.meta['train_start_date']} → {self.meta['train_end_date']}")

	def ForecastFuture(self, sourceDF):
		return self.Predict(sourceDF=sourceDF,	forecast_mode=True)
