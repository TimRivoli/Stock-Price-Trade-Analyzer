import time, random, os, logging
import numpy as np, pandas as pd
import _classes.Constants as CONSTANTS
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay, CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from _classes.DataIO import DataDownload, PTADatabase
from _classes.Graphing import PlotHelper
from _classes.Utility import *

#-------------------------------------------- Global settings -----------------------------------------------
logger = logging.getLogger(__name__)	
def _standardize_datetime_index(df: pd.DataFrame) -> pd.DataFrame | None:
	if df is None or df.empty:
		return None
	df = df.copy()	
	df.index = pd.to_datetime(df.index, errors="coerce") # Force DatetimeIndex 	
	df = df[~df.index.isna()] # Drop bad dates		
	if df.index.tz is not None: # Force timezone naive
		df.index = df.index.tz_convert(None)		
	df.index = df.index.normalize() # Normalize time to midnight		
	df = df.sort_index() # Sort and remove duplicates
	df = df[~df.index.duplicated(keep="last")]
	df.ffill(inplace=True)
	df.bfill(inplace=True)
	if not isinstance(df.index, pd.DatetimeIndex):
		logging.warning(f"{self.ticker} CSV index is not DatetimeIndex")
		return None
	return df

#-------------------------------------------- Classes PriceSnapshot and PricingData -----------------------------------------------
class PriceSnapshot:
	def __init__(self, **kwargs): self.__dict__.update(kwargs)

	@classmethod
	def generate(cls, ticker, columns, valuesDF):
		if valuesDF.size > 0:
			return cls(**dict(zip(columns, valuesDF.iloc[0])))
		else:
			empty_values = [0] * len(columns)
			empty_values[0] = ticker
			return cls(**dict(zip(columns, empty_values)))
	
class PricingData:
	_failed_tickers = set() #Global class variable for tracking lookup failures to prevent spamming download requests
	
	#Historical prices for a given stock, along with statistics, and future estimated prices
	def __init__(self, ticker:str, dataFolderRoot:str='', useDatabase:bool=True):
		self.ticker = ticker
		self.historicalPrices = None	#dataframe with price history indexed on date
		self.pricePredictions = None #dataframe with price predictions indexed on date
		self.historyStartDate = None
		self.historyEndDate = None
		self.pricesLoaded = False
		self.statsLoaded = False
		self.predictionsLoaded = False
		self.predictionDeviation = 0	#Average percentage off from target
		self.pricesNormalized = False
		self.pricesInPercentages = False
		self.database = None
		self._lookbackBuffer = None
		self._dataFolderhistoricalPrices = 'data/historical/'
		self._dataFolderCharts = 'data/charts/'
		self._dataFolderDailyPicks = 'data/dailypicks/'
		if not dataFolderRoot =='':
			if CreateFolder(dataFolderRoot):
				if not dataFolderRoot[-1] =='/': dataFolderRoot += '/'
				self._dataFolderCharts = dataFolderRoot + 'charts/'
				self._dataFolderhistoricalPrices = dataFolderRoot + 'historical/'
				self._dataFolderDailyPicks = dataFolderRoot + 'dailypicks/'
		else: CreateFolder('data')
		CreateFolder(self._dataFolderhistoricalPrices)
		CreateFolder(self._dataFolderCharts)
		CreateFolder(self._dataFolderDailyPicks)
		if useDatabase:
			db = PTADatabase(verbose=False)
			if db.database_configured:
				self.database = db
			else:
				useDatabase = False
		self.useDatabase = useDatabase

	def _assert_lookback_validity(self):
		if self._lookbackBuffer is not None and not self._lookbackBuffer.empty:
			lb_max = self._lookbackBuffer.index.max()
			hp_min = self.historicalPrices.index.min()
			if lb_max >= hp_min:
				print(" LOOKBACK BUFFER ERROR")
				print("  Ticker:", self.ticker)
				print("  Lookback max:", lb_max)
				print("  History min:", hp_min)
				print("  Lookback rows:", len(self._lookbackBuffer))
				print("  History rows:", len(self.historicalPrices))
				raise AssertionError
	
	def _get_full_price_frame(self):
		if self._lookbackBuffer is not None and not self._lookbackBuffer.empty:
			self._assert_lookback_validity()
			return pd.concat([self._lookbackBuffer, self.historicalPrices]).sort_index()
		return self.historicalPrices

#-------------------------------------------- Calculations -----------------------------------------------
	
	def ConvertToPercentages(self):
		if self.pricesInPercentages:
			self.historicalPrices.iloc[0] = self.CTPFactor
			for i in range(1, self.historicalPrices.shape[0]):
				self.historicalPrices.iloc[i] = (1 + self.historicalPrices.iloc[i]) * self.historicalPrices.iloc[i-1]
			if self.predictionsLoaded:
				self.pricePredictions.iloc[0] = self.CTPFactor['Average']
				for i in range(1, self.pricePredictions.shape[0]):
					self.pricePredictions.iloc[i] = (1 + self.pricePredictions.iloc[i]) * self.pricePredictions.iloc[i-1]
			self.pricesInPercentages = False
			print(' ConvertToPercentages: Prices have been converted back from percentages.')
		else:
			self.CTPFactor = self.historicalPrices.iloc[0].copy()
			self.historicalPrices = self.historicalPrices[['Open','Close','High','Low','Average']].pct_change(1)
			self.historicalPrices[:1] = 0
			if self.predictionsLoaded:
				self.pricePredictions = self.pricePredictions.pct_change(1)
			self.statsLoaded = False
			self.pricesInPercentages = True
			self._assert_lookback_validity()
			print(' ConvertToPercentages: Prices have been converted to percentage change from previous day.')
		
	def NormalizePrices(self, verbose:bool=False):
		#(x-min(x))/(max(x)-min(x))
		x = self.historicalPrices
		if not self.pricesNormalized:
			low = x['Low'].min(axis=0) - .000001 #To prevent div zero calculation errors.
			high = x['High'].max(axis=0)
			diff = high-low
			x['Open'] = (x['Open']-low)/diff
			x['Close'] = (x['Close']-low)/diff
			x['High'] = (x['High']-low)/diff
			x['Low'] = (x['Low']-low)/diff
			if self.predictionsLoaded:
				self.pricePredictions[['Predicted_Low']] = (self.pricePredictions[['Predicted_Low']]-low)/diff
				self.pricePredictions[['estAverage']] = (self.pricePredictions[['estAverage']]-low)/diff
				self.pricePredictions[['Predicted_High']] = (self.pricePredictions[['Predicted_High']]-low)/diff
			self.PreNormalizationLow = low
			self.PreNormalizationHigh = high
			self.PreNormalizationDiff = diff
			self.pricesNormalized = True
			if verbose: print(' NormalizePrices: Prices have been normalized.')
		else:
			low = self.PreNormalizationLow
			high = self.PreNormalizationHigh 
			diff = self.PreNormalizationDiff
			x['Open'] = (x['Open'] * diff) + low
			x['Close'] = (x['Close'] * diff) + low
			x['High'] = (x['High'] * diff) + low
			x['Low'] = (x['Low'] * diff) + low
			if self.predictionsLoaded:
				self.pricePredictions[['Predicted_Low']] = (self.pricePredictions[['Predicted_Low']] * diff) + low
				self.pricePredictions[['estAverage']] = (self.pricePredictions[['estAverage']] * diff) + low
				self.pricePredictions[['Predicted_High']] = (self.pricePredictions[['Predicted_High']] * diff) + low
			self.pricesNormalized = False
			if verbose: print(' Prices have been un-normalized.')
		x['Average'] = (x['Open'] + x['Close'] + x['High'] + x['Low'])/4
		#x['Average'] = x.loc[:,CONSTANTS.BASE_FIELD_LIST].mean(axis=1, skipna=True) #Wow, this doesn't work.
		if (x['Average'] < x['Low']).any() or (x['Average'] > x['High']).any(): 
			print(x.loc[x['Average'] < x['Low']])
			print(x.loc[x['Average'] > x['High']])
			print(x.loc[x['Low'] > x['High']])
			print(self.PreNormalizationLow, self.PreNormalizationHigh, self.PreNormalizationDiff, self.PreNormalizationHigh-self.PreNormalizationLow)
			#print(x.loc[:,CONSTANTS.BASE_FIELD_LIST].mean(axis=1))
			print(' NormalizePrices: Stop! averages not computed correctly.')
			assert(False)
		x[['Average']] = x[['Average']].round(2)
		self.historicalPrices = x
		self._assert_lookback_validity()
		if self.statsLoaded: self.CalculateStats()
		if verbose: print(self.historicalPrices[:1])

	def CalculateStats(self, fullStats: bool = True, fancyPantsStats: bool = False):
		if not self.pricesLoaded:
			if not self.LoadHistory(): return False
		lookback_cutoff = None
		if self._lookbackBuffer is not None and not self._lookbackBuffer.empty:
			lookback_cutoff = self._lookbackBuffer.index.max()
		tm = CONSTANTS.TRADING_MONTH
		ty = CONSTANTS.TRADING_YEAR
		df = self._get_full_price_frame()
		avg = df['Average']
		a2 = avg.rolling(2).mean()
		a3 = avg.rolling(3).mean() #.round(2) #TCR: accidental noise filter, moved to PC values
		a5 = avg.rolling(5).mean()
		df['Average_2Day'] = a2
		df['Average_3Day'] = a3
		df['Average_5Day'] = a5
		df["Return"] = df["Close"].pct_change()		
		gain_monthly = ((a3 / a3.shift(tm)) - 1).fillna(0).round(3)
		df['Gain_Monthly'] = gain_monthly
		neg_returns = gain_monthly.where(gain_monthly < 0, 0)
		neg_only = gain_monthly.where(gain_monthly < 0)
		returns = np.log(df['Average'] / df['Average'].shift(1))
		rolling_std = returns.rolling(60, min_periods=20).std()
		rolling_max = returns.abs().rolling(60, min_periods=20).max()
		price = df['Average']
		peak_5y = df['Average'].rolling(ty*5, min_periods=50).max()
		drawdown_5y = (df['Average'] - peak_5y) / (peak_5y + 1e-9)
		peak_index = np.maximum.accumulate(np.where(price >= peak_5y, np.arange(len(price)), -1))
		days_since_peak = np.arange(len(price)) - peak_index
		df['LossStd_1Year']  = neg_returns.rolling(ty, min_periods=1).std().fillna(0)
		df['LossSkew_1Year'] = neg_only.rolling(ty, min_periods=1).skew().fillna(0)
		df['MaxLoss_1Year']  = gain_monthly.rolling(ty, min_periods=1).min()
		df['LogDrawdown'] = np.log(df['Average'] / df['Average'].cummax())
		df['JumpRatio'] = (rolling_max / (rolling_std + 1e-9)).clip(upper=25)
		df['PathologyScore'] = abs(df['LogDrawdown']) * df['JumpRatio']
		df['DamageScore'] = abs(drawdown_5y) * np.sqrt(np.clip(days_since_peak,0,None) / ty)
		df['PC_1Day'] = avg.pct_change(1) * ty
		df['PC_3Day'] = avg.pct_change(3) * ty/3
		df['PC_10Day'] = ((a3 / a3.shift(10) - 1) * 12.5).round(3)
		df['PC_10Day5DayEMA'] = df['PC_10Day'].ewm(span=5, adjust=False).mean()
		df['PC_1Month'] = ((a3 / a3.shift(tm) - 1) * 12.5).round(3)
		df['PC_1Month3WeekEMA'] = df['PC_1Month'].ewm(span=15, adjust=False).mean()
		df['PC_2Month'] = (a3 / a3.shift(41) - 1) * 6.097
		df['PC_3Month'] = (a3 / a3.shift(62) - 1) * 4.03
		df['PC_6Month'] = ((a3 / a3.shift(125) - 1) * 2).round(3)
		df['PC_9Month'] = ((a3 / a3.shift(188) - 1) * 4/3).round(3)
		df['PC_1Year'] = ((a3 / a3.shift(ty)) - 1).round(3)
		df['PC_18Month'] = (a3 / a3.shift(375) - 1) * 0.667
		df['PC_2Year'] = (a3 / a3.shift(500) - 1) / 2
		# alpha = 0.5
		# m1 = np.power(df['PC_1Year'].clip(lower=0), alpha)
		# m9 = np.power(df['PC_9Month'].clip(lower=0), alpha)
		# m6 = np.power(df['PC_6Month'].clip(lower=0), alpha)
		# pv = (9*m1 + 3*m9 + m6)/13
		pv = (10*df['PC_1Year'] + 100*df['LossStd_1Year'] - 6).round()
		pv -= (df['PC_1Month'] < 0.00175) * 3
		pv = pv.where(pv >= 2, 0)
		pv_mask = df.eval("(PC_1Month3WeekEMA > -0.05) & (PC_9Month > 0) & (LogDrawdown > -0.5) & (MaxLoss_1Year > -0.6) & (PathologyScore < 2) & (DamageScore < 0.5)")
		pv *= pv_mask
		df['Point_Value'] = (pv.ewm(span=3, adjust=False).mean().fillna(0)) * 10
		df['Gain_10Day'] = ((a2 / a2.shift(10)) - 1).fillna(0)
		df['Target'] = (a2 * (1 + df['Gain_10Day'] / 9)).round(2)

		if fullStats:
			df["Vol20"] = df["Return"].ewm(span=20, adjust=False).std() * (ty ** 0.5) # Annualized standard deviation
			df["Vol60"] = df["Return"].ewm(span=60, adjust=False).std() * (ty ** 0.5)
			df["MA200"] = df["Close"].rolling(200).mean()
			df["Distance200DMA"] = df["Close"] / df["MA200"] - 1
			df['EMA_12Day'] = avg.ewm(span=12, adjust=False).mean()
			df['EMA_1Month'] = avg.ewm(span=tm).mean()
			df['EMA_3Month'] = avg.ewm(span=tm*3).mean()
			df['EMA_6Month'] = avg.ewm(span=tm*6).mean()
			df['EMA_1Year']  = avg.ewm(span=ty).mean()
			df['MACD_Line'] = df['EMA_12Day'] - df['EMA_1Month']
			df['MACD_Signal'] = df['MACD_Line'].ewm(span=9).mean()
			df['MACD_Histogram'] = df['MACD_Line'] - df['MACD_Signal']
			df['Deviation_1Day'] = (df['High'] - df['Low']) / df['Low']
			df['Deviation_5Day'] = df['Deviation_1Day'].rolling(5).mean()
			df['Deviation_10Day'] = df['Deviation_1Day'].rolling(10).mean()
			df['Deviation_15Day'] = df['Deviation_1Day'].rolling(15).mean()
			df['Channel_High'] = df['EMA_1Year'] + (avg * df['Deviation_15Day'])
			df['Channel_Low']  = df['EMA_1Year'] - (avg * df['Deviation_15Day'])
			df['EMA_3Month_Slope'] = df['EMA_3Month'].pct_change()
			df['EMA_1Year_Slope']  = df['EMA_1Year'].pct_change()
		if fancyPantsStats:
			temp = df.copy()
			vwma_period = 20
			temp['VWMA'] = (temp['Close'] * temp['Volume']).rolling(vwma_period).sum() / temp['Volume'].rolling(vwma_period).sum()
			df['VWMA'] = temp['VWMA']
			ao_short, ao_long = 5, 34
			temp['Midpoint'] = (temp['High'] + temp['Low']) / 2
			temp['AO'] = (temp['Midpoint'].rolling(ao_short).mean()- temp['Midpoint'].rolling(ao_long).mean())
			df['AO'] = temp['AO']
			uo_short, uo_medium, uo_long = 7, 14, 28
			temp['BP'] = temp['Close'] - np.minimum(temp['Low'], temp['Close'].shift(1))
			temp['TR'] = np.maximum(temp['High'], temp['Close'].shift(1)) - np.minimum(temp['Low'], temp['Close'].shift(1))
			temp['UO'] = (
				4 * (temp['BP'].rolling(uo_short).sum() / temp['TR'].rolling(uo_short).sum()) +
				2 * (temp['BP'].rolling(uo_medium).sum() / temp['TR'].rolling(uo_medium).sum()) +
				1 * (temp['BP'].rolling(uo_long).sum() / temp['TR'].rolling(uo_long).sum())
			) / 7
			df['UO'] = temp['UO']
			adx_period = 14
			temp['DM+'] = np.where((temp['High'] - temp['High'].shift(1)) > (temp['Low'].shift(1) - temp['Low']),np.maximum(temp['High'] - temp['High'].shift(1), 0),0)
			temp['DM-'] = np.where((temp['Low'].shift(1) - temp['Low']) > (temp['High'] - temp['High'].shift(1)),np.maximum(temp['Low'].shift(1) - temp['Low'], 0),0)
			temp['TR'] = np.maximum(temp['High'] - temp['Low'],np.maximum(	abs(temp['High'] - temp['Close'].shift(1)),	abs(temp['Low'] - temp['Close'].shift(1))))
			temp['DI+'] = 100 * temp['DM+'].rolling(adx_period).mean() / temp['TR'].rolling(adx_period).mean()
			temp['DI-'] = 100 * temp['DM-'].rolling(adx_period).mean() / temp['TR'].rolling(adx_period).mean()
			temp['DX'] = 100 * abs(temp['DI+'] - temp['DI-']) / (temp['DI+'] + temp['DI-'])
			df['ADX'] = temp['DX'].rolling(adx_period).mean()
		df = df.loc[df.index >= self.historyStartDate].copy() # Trim back to visible window
		self.historicalPrices = df.ffill().bfill()
		self._assert_lookback_validity()
		self.statsLoaded = True
		return True
		
	def PredictPrices(self, method:int=1, daysIntoFuture:int=1, NNTrainingEpochs:int=0):
		#Predict current prices from previous days info
		TRAINING_EPOCHS = 250
		self.predictionsLoaded = False
		self.pricePredictions = pd.DataFrame()	#Clear any previous data
		if not self.statsLoaded: self.CalculateStats(fullStats=True)
		if method >= 2: method = 2
		if method < 3:
			minActionableSlope = 0.001
			if method==0:	#Same as previous day
				self.pricePredictions = pd.DataFrame()
				self.pricePredictions['Predicted_Low'] =  self.historicalPrices['Low'].shift(1)
				self.pricePredictions['Predicted_High'] = self.historicalPrices['High'].shift(1)
			elif method==1 :	#Slope plus momentum with some consideration for trend.
					#++,+-,-+,==
				bucket = self.historicalPrices.copy()
				bucket['Predicted_Low']  = bucket['Average'].shift(1) * (1-bucket['Deviation_15Day']/2) + (abs(bucket['PC_1Day'].shift(1)))
				bucket['Predicted_High'] = bucket['Average'].shift(1) * (1+bucket['Deviation_15Day']/2) + (abs(bucket['PC_1Day'].shift(1)))
				bucket = bucket.query('EMA_1Year_Slope >= -' + str(minActionableSlope) + ' or EMA_3Month_Slope >= -' + str(minActionableSlope)) #must filter after rolling calcuations
				bucket = bucket[['Predicted_Low','Predicted_High']]
				self.pricePredictions = bucket
					#-- downward trend
				bucket = self.historicalPrices.copy()
				bucket['Predicted_Low'] = bucket['Low'].shift(1).rolling(3).min() * .99
				bucket['Predicted_High'] = bucket['High'].shift(1).rolling(3).min() 
				bucket = bucket.query('not (EMA_1Year_Slope >= -' + str(minActionableSlope) + ' or EMA_3Month_Slope >= -' + str(minActionableSlope) +')')
				bucket = bucket[['Predicted_Low','Predicted_High']]
				self.pricePredictions = pd.concat([self.pricePredictions, bucket], ignore_index=False)
				self.pricePredictions.sort_index(inplace=True)	
			elif method==2:	#Slope plus momentum with full consideration for trend.
					#++ Often over bought, strong momentum
				bucket = self.historicalPrices.copy() 
				#bucket['Predicted_Low']  = bucket['Low'].shift(1).rolling(4).max()  * (1 + abs(bucket['EMA_3Month_Slope'].shift(1)))
				#bucket['Predicted_High'] = bucket['High'].shift(1).rolling(4).max()  * (1 + abs(bucket['EMA_3Month_Slope'].shift(1)))
				bucket['Predicted_Low']  = bucket['Low'].shift(1).rolling(4).max()  + (abs(bucket['PC_1Day'].shift(1)))
				bucket['Predicted_High'] = bucket['High'].shift(1).rolling(4).max() + (abs(bucket['PC_1Day'].shift(1)))
				bucket = bucket.query('EMA_1Year_Slope >= ' + str(minActionableSlope) + ' and EMA_3Month_Slope >= ' + str(minActionableSlope)) #must filter after rolling calcuations
				bucket = bucket[['Predicted_Low','Predicted_High']]
				self.pricePredictions = bucket
					#+- correction or early down turn, loss of momentum
				bucket = self.historicalPrices.copy()
				bucket['Predicted_Low']  = bucket['Low'].shift(1).rolling(2).min() 
				bucket['Predicted_High'] = bucket['High'].shift(1).rolling(3).max()  * (1.005 + abs(bucket['EMA_3Month_Slope'].shift(1)))
				bucket = bucket.query('EMA_1Year_Slope >= ' + str(minActionableSlope) + ' and EMA_3Month_Slope < -' + str(minActionableSlope))
				bucket = bucket[['Predicted_Low','Predicted_High']]
				self.pricePredictions = pd.concat([self.pricePredictions, bucket], ignore_index=False)
					 #-+ bounce or early recovery, loss of momentum
				bucket = self.historicalPrices.copy()
				bucket['Predicted_Low']  = bucket['Low'].shift(1)
				bucket['Predicted_High'] = bucket['High'].shift(1).rolling(3).max() * 1.02 
				bucket = bucket.query('EMA_1Year_Slope < -' + str(minActionableSlope) + ' and EMA_3Month_Slope >= ' + str(minActionableSlope))
				bucket = bucket[['Predicted_Low','Predicted_High']]
					#-- Often over sold
				self.pricePredictions = pd.concat([self.pricePredictions, bucket], ignore_index=False)
				bucket = self.historicalPrices.copy() 
				bucket['Predicted_Low'] = bucket['Low'].shift(1).rolling(3).min() * .99
				bucket['Predicted_High'] = bucket['High'].shift(1).rolling(3).min() 
				bucket = bucket.query('EMA_1Year_Slope < -' + str(minActionableSlope) + ' and EMA_3Month_Slope < -' + str(minActionableSlope))
				bucket = bucket[['Predicted_Low','Predicted_High']]
				self.pricePredictions = pd.concat([self.pricePredictions, bucket], ignore_index=False)
					#== no significant slope
				bucket = self.historicalPrices.copy() 
				bucket['Predicted_Low']  = bucket['Low'].shift(1).rolling(4).mean()
				bucket['Predicted_High'] = bucket['High'].shift(1).rolling(4).mean()
				bucket = bucket.query(str(minActionableSlope) + ' > EMA_1Year_Slope >= -' + str(minActionableSlope) + ' or ' + str(minActionableSlope) + ' > EMA_3Month_Slope >= -' + str(minActionableSlope))
				bucket = bucket[['Predicted_Low','Predicted_High']]
				self.pricePredictions = pd.concat([self.pricePredictions, bucket], ignore_index=False)
				self.pricePredictions.sort_index(inplace=True)	
			d = self.historicalPrices.index[-1] 
			ls = self.historicalPrices['EMA_1Year_Slope'].iloc[-1]
			ss = self.historicalPrices['EMA_3Month_Slope'].iloc[-1]
			deviation = self.historicalPrices['Deviation_15Day'].iloc[-1]/2
			momentum = self.historicalPrices['PC_3Day'].iloc[-1]/2 
			random.seed(42)
			for i in range(0,daysIntoFuture): 	#Add new days to the end for crystal ball predictions
				momentum = (momentum + ls)/2 * (100+random.randint(-3,4))/100
				a = (self.pricePredictions['Predicted_Low'].iloc[-1] + self.pricePredictions['Predicted_High'].iloc[-1])/2
				if ls >= minActionableSlope and ss >= minActionableSlope: #++
					l = a * (1+momentum) + random.randint(round(-deviation*200),round(deviation*10))/100
					h = a * (1+momentum) + random.randint(round(-deviation*10),round(deviation*200))/100
				elif ls >= minActionableSlope and ss < -minActionableSlope: #+-
					l = a * (1+momentum) + random.randint(round(-deviation*200),round(deviation*10))/100
					h = a * (1+momentum) + random.randint(round(-deviation*10),round(deviation*400))/100
				elif ls < -minActionableSlope and ss >= minActionableSlope: #-+
					l = a * (1+momentum) + random.randint(round(-deviation*200),round(deviation*10))/100
					h = a * (1+momentum) + random.randint(round(-deviation*10),round(deviation*400))/100
				elif ls < -minActionableSlope and ss < -minActionableSlope: #--
					l = a * (1+momentum) + random.randint(round(-deviation*200),round(deviation*10))/100
					h = a * (1+momentum) + random.randint(round(-deviation*10),round(deviation*200))/100
				else:	#==
					l = a * (1+momentum) + random.randint(round(-deviation*200),round(deviation*10))/100
					h = a * (1+momentum) + random.randint(round(-deviation*10),round(deviation*200))/100
				self.pricePredictions.loc[d + BDay(i+1), 'Predicted_Low'] = l
				self.pricePredictions.loc[d + BDay(i+1), 'Predicted_High'] = h								
			self.pricePredictions['estAverage']	= (self.pricePredictions['Predicted_Low'] + self.pricePredictions['Predicted_High'])/2
		# elif method==3:	#Use LSTM to predict prices
			# from _classes.SeriesPrediction import StockPredictionNN
			# temporarilyNormalize = False
			# if not self.pricesNormalized:
				# temporarilyNormalize = True
				# self.NormalizePrices()
			# model = StockPredictionNN(base_model_name='Prices', model_type='LSTM')
			# field_list = ['Average']
			# model.LoadSource(sourceDF=self.historicalPrices, field_list=field_list, time_steps=1)
			# model.LoadTarget(targetDF=None, prediction_target_days=daysIntoFuture)
			# model.MakeTrainTest(batch_size=32, train_test_split=.93)
			# model.BuildModel()
			# if (not model.Load() and NNTrainingEpochs == 0): NNTrainingEpochs = TRAINING_EPOCHS
			# if (NNTrainingEpochs > 0): 
				# model.Train(epochs=NNTrainingEpochs)
				# model.Save()
			# model.Predict(True)
			# self.pricePredictions = model.GetTrainingResults(False, False)
			# self.pricePredictions = self.pricePredictions.rename(columns={'Average':'estAverage'})
			# deviation = self.historicalPrices['Deviation_15Day'].iloc[-1]/2
			# self.pricePredictions['Predicted_Low'] = self.pricePredictions['estAverage'] * (1 - deviation)
			# self.pricePredictions['Predicted_High'] = self.pricePredictions['estAverage'] * (1 + deviation)
			# if temporarilyNormalize: 
				# self.predictionsLoaded = True
				# self.NormalizePrices()
		# elif method==4:	#Use CNN to predict prices
			# from _classes.SeriesPrediction import StockPredictionNN
			# temporarilyNormalize = False
			# if not self.pricesNormalized:
				# temporarilyNormalize = True
				# self.NormalizePrices()
			# model = StockPredictionNN(base_model_name='Prices', model_type='CNN')
			# field_list = CONSTANTS.BASE_FIELD_LIST
			# model.LoadSource(sourceDF=self.historicalPrices, field_list=field_list, time_steps=daysIntoFuture*16)
			# model.LoadTarget(targetDF=None, prediction_target_days=daysIntoFuture)
			# model.MakeTrainTest(batch_size=32, train_test_split=.93)
			# model.BuildModel()
			# if (not model.Load() and NNTrainingEpochs == 0): NNTrainingEpochs = TRAINING_EPOCHS
			# if (NNTrainingEpochs > 0): 
				# model.Train(epochs=NNTrainingEpochs)
				# model.Save()
			# model.Predict(True)
			# self.pricePredictions = model.GetTrainingResults(False, False)
			# self.pricePredictions = self.pricePredictions.rename(columns={'Average':'estAverage'})
			# deviation = self.historicalPrices['Deviation_15Day'].iloc[-1]/2
			# self.pricePredictions['Predicted_Low'] = self.pricePredictions['estAverage'] * (1 - deviation)
			# self.pricePredictions['Predicted_High'] = self.pricePredictions['estAverage'] * (1 + deviation)
			# self.pricePredictions = self.pricePredictions[['Predicted_Low','estAverage','Predicted_High']]
			# if temporarilyNormalize: 
				# self.predictionsLoaded = True
				# self.NormalizePrices()
		self.pricePredictions.fillna(0, inplace=True)
		x = self.pricePredictions.join(self.historicalPrices)
		x['PercentageDeviation'] = abs((x['Average']-x['estAverage'])/x['Average'])
		self.predictionDeviation = x['PercentageDeviation'].tail(round(x.shape[0]/4)).mean() #Average of the last 25%, this is being kind as it includes some training data
		self.predictionsLoaded = True
		return True
	
	def PredictFuturePrice(self,fromDate:datetime, daysForward:int=1,method:int=1):
		fromDate=ToTimestamp(fromDate)
		low,high,price,momentum,deviation = self.historicalPrices.loc[fromDate, ['Low','High','Average', 'PC_3Day','Deviation_15Day']]
		#print(p,m,s)
		if method==0:
			futureLow = low
			futureHigh = high
		else:  
			futureLow = price * (1 + daysForward * momentum) - (price * deviation/2)
			futureHigh = price * (1 + daysForward * momentum) + (price * deviation/2)
		return futureLow, futureHigh	

#-------------------------------------------- Gets retrievals -----------------------------------------------
	def GetDateFromIndex(self,indexLocation:int):
		if indexLocation >= self.historicalPrices.shape[0]: indexLocation = self.historicalPrices.shape[0]-1
		d = self.historicalPrices.index.values[indexLocation]
		return d

	def GetPrice(self,forDate:datetime, verbose:bool=False):
		forDate = ToTimestamp(forDate)
		r = 0
		try:
			i = self.historicalPrices.index.get_indexer([forDate], method='ffill')[0] #ffill will effectively look backwards for the first instance
			if i > -1: 
				forDate = self.historicalPrices.index[i]
				r = self.historicalPrices.loc[forDate]['Average']			
		except Exception as e: 
			if verbose or True: 
				print(f" GetDateFromIndex: Unable to get price for {self.ticker} on {forDate} start {self.historyStartDate} end {self.historyEndDate}")	
				print(e)
				print(self.historicalPrices.index.duplicated())
			r = 0
		return r
		
	def GetNearestTradingDate(self,	targetDate: datetime,direction: str = "prior") -> pd.Timestamp | None:
		if self.historicalPrices is None or self.historicalPrices.empty:
			return None
		idx = self.historicalPrices.index
		targetDate = ToTimestamp(targetDate)
		if direction == "prior":
			i = idx.get_indexer([targetDate], method="ffill")[0]
			return idx[i] if i >= 0 else None
		elif direction == "next":
			i = idx.get_indexer([targetDate], method="bfill")[0]
			return idx[i] if i >= 0 else None
		elif direction == "nearest":
			i = idx.get_indexer([targetDate], method="nearest")[0]
			return idx[i] if i >= 0 else None
		else:
			raise ValueError(f"Invalid direction: {direction}")

	def GetPriceData(self, forDate: datetime, field_list: list,	*,	tradingDaysOnly: bool = False, returnType: str = "values", verbose: bool = False):
		if self.historicalPrices is None or self.historicalPrices.empty:
			if verbose: print(f" GetPriceData: {self.ticker}: no historical prices loaded")
			return None
		forDate = ToTimestamp(forDate)
		try:
			if tradingDaysOnly:	# Exact trading-day lookup
				if forDate not in self.historicalPrices.index:
					if verbose: print(f" GetPriceData: {self.ticker}: {FormatDate(forDate)} not a trading day")
					return None
				row = self.historicalPrices.loc[forDate]
			else: # Legacy behavior (calendar-date tolerant)			
				i = self.historicalPrices.index.get_indexer([forDate], method="ffill")[0]
				if i < 0:
					return None
				row = self.historicalPrices.iloc[i]
			data = row[field_list]
			if returnType == "series":
				return data
			elif returnType == "dict":
				return data.to_dict()
			else:
				return data.values
		except Exception as e:
			if verbose:
				print(f" GetPriceData: Unable to get price data for {self.ticker} on {FormatDate(forDate)}")
				print(e)
			return None

	def GetPriceSnapshotDF(self, forDate:datetime, verbose:bool=False):
		forDate = ToTimestamp(forDate)
		df = self.historicalPrices
		try:
			i = df.index.get_indexer([forDate], method='ffill')[0]
			if i > -1: forDate = df.index[i]
			result = df[df.index == forDate]
			result = result.reset_index()
		except:
			result = pd.DataFrame(columns=self.historicalPrices)
			result.fillna(0, inplace=True)
			if verbose: print(f" GetPriceSnapshotDF: Unable to get price snapshot for {self.ticker} on {FormatDate(forDate)}")	
		result['Ticker'] = self.ticker
		result.set_index('Ticker', inplace=True)
		result = result.fillna(0)
		return result

	def GetPriceSnapshot(self, forDate:datetime, verbose:bool=False):
		snapDF = self.GetPriceSnapshotDF(forDate)
		sn = PriceSnapshot.generate(self.ticker, columns=snapDF.columns, valuesDF=snapDF)	
		sn.Comments = ''
		if sn.Low > sn.Channel_High: 
			sn.Comments += 'Overbought; '
		if sn.High < sn.Channel_Low: 
			sn.Comments += 'Oversold; '
		if sn.Deviation_5Day > .0275: 
			sn.Comments += 'HighDeviation; '
		return sn

	def GetCurrentPriceSnapshot(self, regime:str="CONVEX"): return self.GetPriceSnapshot(self.historyEndDate, regime)

	def GetPriceHistory(self, field_list:list = None, includePredictions:bool = False):
		if field_list == None:
			r = self.historicalPrices.copy() #best to pass back copies instead of reference.
		else:
			r = self.historicalPrices[field_list].copy() #best to pass back copies instead of reference.			
		if includePredictions: r = r.join(self.pricePredictions, how='outer')
		return r
		
	def GetPricePredictions(self):
		return self.pricePredictions.copy()  #best to pass back copies instead of reference.

#-------------------------------------------- Graphing -----------------------------------------------
	def GraphData(self, endDate:datetime=None, daysToGraph:int=90, graphTitle:str=None, includePredictions:bool=False, saveToFile:bool=False, fileNameSuffix:str=None, saveToFolder:str='', dpi:int=600, trimHistoricalPredictions:bool = True, verbose: bool=False):
		if not self.statsLoaded:
			self.CalculateStats(fullStats=True)

		if includePredictions:
			if not self.predictionsLoaded:
				self.PredictPrices()
			if endDate is None:
				endDate = self.pricePredictions.index.max()
			endDate = ToTimestamp(endDate)
			startDate = endDate - BDay(daysToGraph)
			fieldSet = ['High', 'Low', 'Channel_High', 'Channel_Low', 'Predicted_High', 'Predicted_Low', 'EMA_3Month', 'EMA_1Year']
			if trimHistoricalPredictions:
				preds = self.pricePredictions[self.pricePredictions.index >= self.historyEndDate]
				x = self.historicalPrices.join(preds, how='outer')
			else:
				fieldSet = ['High', 'Low', 'Predicted_High', 'Predicted_Low']
				x = self.historicalPrices.join(self.pricePredictions, how='outer')
			if daysToGraph > 1800:
				fieldSet = ['Average', 'Predicted_High', 'Predicted_Low']
		else:
			if endDate is None:
				endDate = self.historyEndDate
			endDate = ToTimestamp(endDate)
			startDate = endDate - BDay(daysToGraph)
			fieldSet = ['High', 'Low', 'Channel_High', 'Channel_Low', 'EMA_3Month', 'EMA_1Year']
			if daysToGraph > 1800:
				fieldSet = ['Average']
			x = self.historicalPrices.copy()
		if fileNameSuffix is None:
			fileNameSuffix = f"{str(endDate)[:10]}_{daysToGraph}days"
		if graphTitle is None:
			graphTitle = f"{self.ticker} {fileNameSuffix}"
		x = x[(x.index >= startDate) & (x.index <= endDate)]
		save_path = None
		if saveToFile:
			if saveToFolder == '':
				saveToFolder = self._dataFolderCharts
			if not saveToFolder.endswith('/'):
				saveToFolder += '/'
			if CreateFolder(saveToFolder):
				save_path = f"{saveToFolder}{self.ticker}_{fileNameSuffix}.png"
		PlotHelper.PlotTimeSeries(df=x, fields=fieldSet, start_date=startDate, end_date=endDate, title=graphTitle, colors=['blue', 'red', 'purple', 'purple', 'mediumseagreen', 'seagreen'], dpi=dpi, save=saveToFile, save_path=save_path, show=not saveToFile)
		if verbose and save_path !=None: print(f" GraphData: Saved to {save_path}")
			
#-------------------------------------------- IO Helper functions -----------------------------------------------

	def _save_to_csv(self, df):
		csvFile = os.path.join(self._dataFolderhistoricalPrices, f"{self.ticker}.csv")
		df.reset_index().to_csv(csvFile, index=False)

	def _save_to_sql(self, df):
		if self.database is None:
			return
		if not self.database.Open():
			return
		out = df.reset_index()
		out['Ticker'] = self.ticker    
		startDate = out['Date'].min() # Delete overlapping dates only
		self.database.ExecSQL("DELETE FROM PricesDaily WHERE Ticker=:Ticker AND [Date]>=:StartDate", {"Ticker": self.ticker, "StartDate": startDate})
		self.database.DataFrameToSQL(df=out, tableName='PricesDaily', indexAsColumn=False, clearExistingData=False)
		self.database.Close()

	def _load_history_sql(self, loadStartDate: pd.Timestamp | None, loadEndDate: pd.Timestamp | None, 	verbose: bool = False) -> pd.DataFrame | None:
		if self.database is None:
			return None
		if not self.database.Open():
			return None
		sql = "SELECT [Date], [Open], [High], [Low], [Close], [Volume] FROM PricesDaily WHERE Ticker = :ticker"
		params = {"ticker": self.ticker}
		if loadStartDate is not None:
			sql += " AND [Date] >= :start_date"
			params["start_date"] = loadStartDate.to_pydatetime()

		if loadEndDate is not None:
			sql += " AND [Date] <= :end_date"
			params["end_date"] = loadEndDate.to_pydatetime()
		sql += " ORDER BY [Date]"
		if verbose and loadStartDate: print(f"{self.ticker}: loading from SQL dates [{FormatDate(loadStartDate)}, {FormatDate(loadEndDate)}]")
		df = self.database.DataFrameFromSQL(sql=sql, params=params, indexName="Date")
		self.database.Close()
		if df is None or df.empty or len(df) < 2:
			return None
		return _standardize_datetime_index(df)

	def _load_history_csv(self, verbose: bool = False) -> pd.DataFrame | None:
		if verbose: print(f"{self.ticker}: loading from CSV")
		csvFile = os.path.join(self._dataFolderhistoricalPrices, f"{self.ticker}.csv")
		if not os.path.isfile(csvFile):
			return None
		df = pd.read_csv(csvFile, index_col=0, parse_dates=True, na_values=['nan'])
		missing = set(CONSTANTS.BASE_FIELD_LIST) - set(df.columns)
		if missing:
			logging.warning(f"{self.ticker} CSV missing required fields: {missing}")
			return None
		df = df[CONSTANTS.BASE_FIELD_LIST].copy()
		if df.empty:
			return None
		return _standardize_datetime_index(df)
		
	def _load_history_yahoo(self, verbose: bool = False) -> pd.DataFrame | None:
		if self.ticker in self._failed_tickers:
			if verbose: print(f"{self.ticker}: Yahoo refresh blocked by cooldown")
			return None
		if verbose:	print(f"{self.ticker}: downloading full history from Yahoo")
		dd = DataDownload()
		df = dd.DownloadPriceDataYahooFinance(self.ticker)
		if df is None or df.empty or len(df) < 2:
			self._failed_tickers.add(self.ticker)
			return None
		return _standardize_datetime_index(df)

	def _save_history(self, df: pd.DataFrame, verbose: bool = False):
		if self.useDatabase:
			if verbose:	print(f"{self.ticker}: saving refreshed data to SQL")
			self._save_to_sql(df)
		else:
			if verbose: print(f"{self.ticker}: saving refreshed data to CSV")
			self._save_to_csv(df)

	def _apply_date_window(self, df: pd.DataFrame, requestedStartDate: pd.Timestamp | None,	requestedEndDate: pd.Timestamp | None):
		if df is None or df.empty:
			self.historicalPrices = df
			self._lookbackBuffer = None
			self.pricesLoaded = False
			return
		requestedStartDate = ToTimestamp(requestedStartDate) if requestedStartDate else None
		requestedEndDate   = ToTimestamp(requestedEndDate) if requestedEndDate else None
		if requestedStartDate is None and requestedEndDate is None:
			self.historicalPrices = df.copy()
			self._lookbackBuffer = None
			self.historyStartDate = df.index.min()
			self.historyEndDate   = df.index.max()
			self.pricesLoaded = True
			return
		lookbackStart = (requestedStartDate - timedelta(days=CONSTANTS.REQUIRED_LOOKBACK)	if requestedStartDate else None	)
		core = df
		if lookbackStart is not None:
			core = core.loc[core.index >= lookbackStart]
		if requestedEndDate is not None:
			core = core.loc[core.index <= requestedEndDate]
		window = core
		if requestedStartDate is not None:
			window = window.loc[window.index >= requestedStartDate]
		if requestedStartDate is not None:
			self._lookbackBuffer = core.loc[core.index < requestedStartDate].copy()
		else:
			self._lookbackBuffer = None
		self.historicalPrices = window.copy()
		if self.historicalPrices.empty:
			self._lookbackBuffer = None
			self.pricesLoaded = False
			return
		self.historyStartDate = self.historicalPrices.index.min()
		self.historyEndDate   = self.historicalPrices.index.max()
		self._assert_lookback_validity()
		self.pricesLoaded = True

	def _generate_cash_history(self, start_date: pd.Timestamp | None, end_date: pd.Timestamp | None) -> pd.DataFrame:
		"""Generates a synthetic DataFrame for the CASH ticker."""
		self.ticker = '.INX' #Try .INX is best dates
		if self.LoadHistory(requestedStartDate=start_date, requestedEndDate=end_date):
			date_index = self.historicalPrices.index
		else:
			#Not ideal as it will include some dates the market is closed
			if start_date is None: start_date = pd.Timestamp('1980-01-01') + pd.offsets.BusinessDay(1)
			if end_date is None: end_date = pd.Timestamp.now().normalize() - pd.offsets.BusinessDay(1)
			cal = USFederalHolidayCalendar()		
			bday = CustomBusinessDay(calendar=cal)
			date_index = pd.date_range(start=start_date, end=end_date, freq=bday, name='Date')
		df = pd.DataFrame(1.0, index=date_index, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
		self.ticker = CONSTANTS.CASH_TICKER
		return df
	#-------------------------------------------- IO Funcitons -----------------------------------------------

	def LoadHistory(self, requestedStartDate:datetime=None, requestedEndDate:datetime=None, verbose:bool=False)-> bool:
		#Note: specifying an requestedEndDate will automatically force a data update if required.  Keep that in mind for backtesting
		if requestedEndDate:
			requestedEndDate = ToTimestamp(requestedEndDate)
			latestBusinessDay = (pd.Timestamp.now().normalize() - pd.offsets.BusinessDay(1))
			requestedEndDate -= pd.offsets.BDay(0)
			requestedEndDate = min(requestedEndDate, latestBusinessDay)
		if requestedStartDate: 
			requestedStartDate = ToTimestamp(requestedStartDate)
			requestedStartDate -= pd.offsets.BDay(0)
		self.historicalPrices = None
		self._lookbackBuffer = None
		self.statsLoaded = False
		self.predictionsLoaded = False
		self.pricesLoaded = False
		if requestedStartDate and requestedEndDate and requestedStartDate > requestedEndDate:
			if verbose: print(f" LoadHistory: invalid date range start date {FormatDate(requestedStartDate)} > end date {FormatDate(requestedEndDate)}")
			return False
		if self.ticker == CONSTANTS.CASH_TICKER:
			df = self._generate_cash_history(requestedStartDate, requestedEndDate)
		elif self.useDatabase:
			loadStartDate = None 
			loadEndDate = requestedEndDate
			if requestedStartDate is not None:
				loadStartDate = requestedStartDate - timedelta(days=CONSTANTS.REQUIRED_LOOKBACK)
			df = self._load_history_sql(loadStartDate, loadEndDate, verbose)
		else:
			df = self._load_history_csv(verbose)
		needs_refresh = False
		if df is None:
			gap_bdays = -1
			needs_refresh = True
		elif requestedEndDate is not None:
			dataEnd = df.index.max()
			if dataEnd < requestedEndDate: #Requested date was after the date retrieve, was it significant
				gap_bdays = len(pd.bdate_range(start=dataEnd, end=requestedEndDate)) - 1 #How many business days
				is_recent = requestedEndDate >= (latestBusinessDay - pd.offsets.BusinessDay(5))
				if (gap_bdays > 5) or (is_recent and gap_bdays > 0): #More than 5 busness days or recent
					if gap_bdays > 3: print(f" LoadHistory Warning: historical end date {FormatDate(df.index.max())} is less than requested {FormatDate(requestedEndDate)} for {self.ticker}")
					needs_refresh = True
		if needs_refresh and not CONSTANTS.BLOCK_REFRESHING_FOR_BACKTESTING:
			if verbose:	print(f" {self.ticker}: refreshing from Yahoo to fill gap {gap_bdays}")
			df_yahoo = self._load_history_yahoo(verbose)
			if df_yahoo is None:
				if verbose:	print(f" LoadHistory: {self.ticker}: refresh from Yahoo required and failed.  Proceeding with existing data...")
			else:
				self._save_history(df_yahoo, verbose)
				df = df_yahoo
		if df is None:
			return False
		df = df.copy()
		df['Average'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4.0
		#Apply final visible window (ONLY PLACE)
		self._apply_date_window(df,	requestedStartDate,	requestedEndDate)
		return self.pricesLoaded

	def ReLoadHistory(self, verbose:bool=True):
		if verbose:	print(f" ReLoadHistory: {self.ticker}: full refresh requested... downloading from Yahoo")
		df_yahoo = self._load_history_yahoo(verbose)
		if df_yahoo is None:
			if verbose:	print(f" ReLoadHistory: {self.ticker}: refresh from Yahoo required and failed")
			return False  
		else:
			self._save_history(df_yahoo, verbose)
		self.LoadHistory(verbose=verbose)

	def SavePricesWithStats(self, includePredictions:bool=False, toDatebase:bool = False, verbose:bool=False):
		fileName = self.ticker + '_stats.csv'
		tableName = 'PricesDailyWithStats'
		r = self.historicalPrices
		if includePredictions:
			fileName = self.ticker + '_stats_predictions.csv'
			tableName = 'PricesDailyWithPredictions'
			r = self.historicalPrices.join(self.pricePredictions, how='outer') #, rsuffix='_Predicted'		
		if self.useDatabase and toDatebase:
			if self.database.Open():
				self.database.ExecSQL("if OBJECT_ID('" + tableName + "') is not null Delete FROM " + tableName + " WHERE Ticker='" + self.ticker + "'")
				r['Ticker'] = self.ticker
				self.database.DataFrameToSQL(r, tableName, indexAsColumn=True)
				self.database.Close()
				print(f"Statistics for {self.ticker} saved to {tableName}" )
		else:			
			filePath = self._dataFolderhistoricalPrices + fileName
			r.to_csv(filePath)
			print(f"Statistics for {self.ticker} saved to {filePath}" )

	def SyncSQLAndCSV(self, verbose: bool = True) -> str | None:
		#Load both SQL and CSV datasets, determine which is more complete, 	and overwrite the other source with the authoritative data.
		df_csv = self._load_from_csv()
		df_sql = None
		if self.database is not None and self.database.Open():
			try:
				sql = "SELECT [Date], [Open], [High], [Low], [Close] FROM PricesDaily WHERE Ticker = :ticker ORDER BY [Date]"
				df_sql = self.database.DataFrameFromSQL(sql, params=[self.ticker], indexName="Date")
			finally:
				self.database.Close()
		if df_csv is not None and df_csv.empty:
			df_csv = None
		if df_sql is not None and df_sql.empty:
			df_sql = None

		if df_csv is None and df_sql is None:
			if verbose:	print(f"{self.ticker}: no data found in SQL or CSV")
			return None

		if df_csv is None:
			if verbose:	print(f"{self.ticker}: CSV missing, syncing SQL → CSV")
			self._save_to_csv(df_sql)
			return "SQL"

		if df_sql is None:
			if verbose:	print(f"{self.ticker}: SQL missing, syncing CSV → SQL")
			self._save_to_sql(df_csv)
			return "CSV"
		csv_start, csv_end = df_csv.index.min(), df_csv.index.max()
		sql_start, sql_end = df_sql.index.min(), df_sql.index.max()
		csv_span = (csv_end - csv_start).days
		sql_span = (sql_end - sql_start).days

		if csv_span > sql_span:
			authoritative = "CSV"
		elif sql_span > csv_span:
			authoritative = "SQL"
		else:
			authoritative = "CSV" if len(df_csv) > len(df_sql) else "SQL"
		if authoritative == "CSV":
			if verbose:	print(f"{self.ticker}: CSV authoritative, syncing → SQL")
			self._save_to_sql(df_csv)
		else:
			if verbose:	print(f"{self.ticker}: SQL authoritative, syncing → CSV")
			self._save_to_csv(df_sql)
		return authoritative

	def LoadTickerFromCSVToSQL(self, verbose: bool = True) -> bool:
		if verbose:	print(f"Loading {self.ticker} CSV into SQL...")
		if self.database is None:
			self.database = PTADatabase()
		df_csv = self._load_from_csv()

		if df_csv is None or df_csv.empty:
			if verbose:	print(f"No CSV data found for {self.ticker}")
			return False
		if verbose:	print(f"Saving {len(df_csv)} records to SQL")
		self._save_to_sql(df_csv)
		if verbose:	print(f"CSV → SQL import complete for {self.ticker}")
		return True

	def ExportFromSQLToCSV(self, verbose: bool = True, minAgeHours: int = 12) -> bool:
		if self.database is None:
			if verbose:	print(f"{self.ticker}: no database configured")
			return False
		csvFile = os.path.join(self._dataFolderhistoricalPrices, f"{self.ticker}.csv")
		needsUpdating = True
		if os.path.isfile(csvFile):
			minAgeToRefresh = datetime.now() - timedelta(hours=minAgeHours)
			needsUpdating = datetime.fromtimestamp(os.path.getmtime(csvFile)) < minAgeToRefresh
		if not needsUpdating:
			if verbose: print(f"{self.ticker}: CSV is up to date")
			return True
		df_sql = self._load_history_sql(loadStartDate=None, loadEndDate=None, verbose=verbose)
		if df_sql is None or df_sql.empty:
			if verbose:	print(f"{self.ticker}: no SQL data to export")
			return False
		self._save_to_csv(df_sql)
		if verbose:	print(f"{self.ticker}: exported SQL → CSV ({len(df_sql)} rows)")
		return True
