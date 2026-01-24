import time
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from _classes.Utility import *
from _classes.DataIO import PTADatabase
from _classes.Prices import PriceSnapshot, PricingData

TRADING_MONTH = 21
TRADING_YEAR = 252

class StockPicker():
	def __init__(self, startDate:datetime=None, endDate:datetime=None, useDatabase:bool=True): 
		self.current_regime = "OFF"   # CONVEX | LINEAR | OFF
		self.priceData = []
		self._tickerList = []
		self._startDate = startDate
		self._endDate = endDate
		if useDatabase:
			db = PTADatabase()
			if db.database_configured:
				self.database = db
			else:
				useDatabase = False
		self.useDatabase = useDatabase
		
	def __del__(self): 
		self.priceData = None
		self._tickerList = None
		
	def set_regime(self, dispersion, autocorr):
		if dispersion > 0.35 and autocorr > 0:
			self.current_regime = "CONVEX"
		elif dispersion > 0.25:
			self.current_regime = "LINEAR"
		else:
			self.current_regime = "OFF"

		return self.current_regime

	def AddTicker(self, ticker:str):
		if not ticker in self._tickerList:
			p = PricingData(ticker, useDatabase=self.useDatabase)
			if p.LoadHistory(self._startDate, self._endDate, verbose=False): 
				p.CalculateStats(fullStats=True)
				self.priceData.append(p)
				self._tickerList.append(ticker)

	def RemoveTicker(self, ticker:str, verbose:bool=False):
		i=len(self.priceData)-1
		while i >= 0:
			if ticker == self.priceData[i].ticker:
				if verbose: print(" Removing ticker " + ticker)
				self.priceData.pop(i)
				self._tickerList.remove(ticker)
			i -=1
		if ticker in self._tickerList: 
			print(" Error removing ticker " + ticker)
			print(len(self.priceData))	
			print(self._tickerList)	
			assert(False)

	def AlignToList(self, newList:list, verbose:bool=False):
		#Add/Remove tickers until they match the given list
		i=len(self.priceData)-1
		while i >= 0:
			ticker = self.priceData[i].ticker
			if not ticker in newList:
				if verbose: print(" Removing ticker " + ticker)
				self.priceData.pop(i)
				self._tickerList.remove(ticker)
			i -=1
		pbar = tqdm(total=len(newList), desc=" AlignToList adding tickers")
		for t in newList:
			self.AddTicker(t)
			pbar.update(1)
		pbar.close()
		
	def SaveStats(self):
		for p in self.priceData:
			p.SavePricesWithStats()
			
	def TickerExists(self, ticker:str):
		return ticker in self._tickerList
	
	def TickerCount(self):
		return len(self._tickerList)

	def NormalizePrices(self):
		for i in range(len(self.priceData)):
			self.priceData[i].NormalizePrices()
			
	def FindOpportunities(self, currentDate:datetime, stocksToReturn:int = 5, filterOption:int = 3, minPercentGain=0.00): 
		result = []
		for i in range(len(self.priceData)):
			ticker = self.priceData[i].ticker
			sn = self.priceData[i].GetPriceSnapshot(currentDate)
			if  ((sn.EMA_Short/sn.EMA_Long)-1 > minPercentGain):
				if filterOption ==0: #Overbought
					if sn.Low > sn.Channel_High: result.append(ticker)
				if filterOption ==1: #Oversold
					if sn.High < sn.Channel_Low: result.append(ticker)
				if filterOption ==1: #High price deviation
					if sn.Deviation_5Day > .0275: result.append(ticker)
		return result

	def GetAdaptivePV(self, currentDate, dispersion, autocorr, stockCount=10, longHistoryDays=365, shortHistoryDays=90):
		if dispersion > 0.35 and autocorr > 0:
			regime = "CONVEX"
		elif dispersion > 0.25:
			regime = "LINEAR"
		else:
			regime = "OFF"
		if regime == "OFF":
			return pd.DataFrame(columns=["TargetHoldings"])
		if regime == "CONVEX":
			picks = self.GetHighestPriceMomentum(currentDate=currentDate, longHistoryDays=longHistoryDays, shortHistoryDays=shortHistoryDays, stocksToReturn=stockCount, filterOption=5	)
			# Allocate by Point_Value
			return (pd.DataFrame(picks).groupby("Ticker")["Point_Value"].sum().to_frame("TargetHoldings"))
		# LINEAR regime
		picks = self.GetHighestPriceMomentum(currentDate=currentDate, longHistoryDays=longHistoryDays, shortHistoryDays=shortHistoryDays, stocksToReturn=stockCount, filterOption=3)
		return (pd.DataFrame(picks).groupby("Ticker").size().to_frame("TargetHoldings"))
		
	def compute_cross_sectional_dispersion(self, currentDate, min_valid=20):
		"""
		Cross-sectional standard deviation of 1Y momentum (PC_1Year).
		Measures opportunity dispersion for convex strategies.
		"""
		values = []
		for pdata in self.priceData:
			if not pdata.statsLoaded:
				continue
			df = pdata.historicalPrices
			if currentDate not in df.index:
				continue
			pc1y = df.loc[currentDate, 'PC_1Year']
			if pd.notna(pc1y):
				values.append(pc1y)
		if len(values) < min_valid:
			return 0.0
		return float(np.std(values, ddof=1))

	def compute_momentum_autocorr(self,	currentDate, lookback_months=6, min_valid=20):
		"""
		Average autocorrelation of 1M momentum across universe.
		Positive = momentum persistence.
		Negative = mean reversion.
		"""

		autocorrs = []
		for pdata in self.priceData:
			if not pdata.statsLoaded:
				continue
			df = pdata.historicalPrices
			if currentDate not in df.index:
				continue
			series = df.loc[:currentDate, 'PC_1Month'].dropna()
			if len(series) < lookback_months + 1:
				continue
			s1 = series[-lookback_months:]
			s2 = series.shift(1)[-lookback_months:]
			if s1.std() == 0 or s2.std() == 0:
				continue
			corr = s1.corr(s2)
			if pd.notna(corr):
				autocorrs.append(corr)
		if len(autocorrs) < min_valid:
			return 0.0
		return float(np.mean(autocorrs))

	def compute_downside_volatility(self, currentDate, min_valid=20):
		"""
		Median downside volatility (LossStd_1Year) across universe.
		Confirms whether volatility is being rewarded.
		"""
		losses = []
		for pdata in self.priceData:
			if not pdata.statsLoaded:
				continue
			df = pdata.historicalPrices
			if currentDate not in df.index:
				continue
			loss_std = df.loc[currentDate, 'LossStd_1Year']
			if pd.notna(loss_std):
				losses.append(loss_std)
		if len(losses) < min_valid:
			return 0.0
		return float(np.median(losses))

	def GetHighestPriceMomentum(self, currentDate:datetime, longHistoryDays:int=365, shortHistoryDays:int=30, stocksToReturn:int=5, filterOption:int=3, minPercentGain=0.05, verbose:bool=False): 
		stocksToReturn = int(stocksToReturn)
		currentDate = ToTimestamp(currentDate)
		max_allowed_date = (pd.Timestamp.now().normalize() - pd.offsets.BusinessDay(1)).to_pydatetime()
		if currentDate > max_allowed_date: currentDate = max_allowed_date
		minPC_1Day = minPercentGain / TRADING_YEAR
		candidates = []
		#fields = ['HP_2Yr','HP_1Yr','HP_6Mo','HP_3Mo','HP_2Mo','HP_1Mo','Average_5Day','Average_2Day','Channel_High','Channel_Low','PC_2Year','PC_1Year','PC_6Month','PC_3Month','PC_1Month','PC_1Month3WeekEMA','PC_3Day','PC_1Day','Deviation_15Day','Deviation_10Day','Deviation_5Day','Deviation_1Day','Gain_Monthly','LossStd_1Year','Point_Value','Comments','HasFullLookback']
		for pd_obj in self.priceData:
			ticker = pd_obj.ticker
			df = pd_obj.historicalPrices
			if currentDate not in df.index:
				if verbose:
					print(f"GetHighestPriceMomentum: {ticker} missing date {currentDate}")
				continue
			row = df.loc[currentDate]
			hp2Year = row['HP_2Yr']
			hp1Year = row['HP_1Yr']
			hp6mo   = row['HP_6Mo']
			hp3mo   = row['HP_3Mo']
			hp2mo   = row['HP_2Mo']
			hp1mo   = row['HP_1Mo']
			Price_Current = row['Average_5Day']
			if min(hp2Year, hp1Year, hp6mo, hp2mo, hp1mo, Price_Current) <= 0:
				print(f" GetHighestPriceMomentum min value lookup failed for ticker {ticker} date {currentDate} ")
				continue
			PC_ShortTerm = row['PC_1Month3WeekEMA'] / TRADING_MONTH
			PC_LongTerm  = row['PC_1Year'] / TRADING_YEAR
			pc2mo = ((Price_Current / hp2mo) - 1) * 5.952  # annualized
			candidates.append({
				'Ticker': ticker,
				'hp2Year': hp2Year,
				'hp1Year': hp1Year,
				'hp6mo': hp6mo,
				'hp3mo': hp3mo,
				'hp2mo': hp2mo,
				'hp1mo': hp1mo,
				'Price_Current': Price_Current,
				'Average_5Day': row['Average_5Day'],
				'Average_2Day': row['Average_2Day'],
				'Channel_High': row['Channel_High'],
				'Channel_Low': row['Channel_Low'],
				'PC_2Year': row['PC_2Year'],
				'PC_1Year': row['PC_1Year'],
				'PC_6Month': row['PC_6Month'],
				'PC_3Month': row['PC_3Month'],
				'PC_2Month': pc2mo,
				'PC_1Month': row['PC_1Month'],
				'PC_3Day': row['PC_3Day'],
				'PC_1Day': row['PC_1Day'],
				'Deviation_15Day': row['Deviation_15Day'],
				'Deviation_10Day': row['Deviation_10Day'],
				'Deviation_5Day': row['Deviation_5Day'],
				'Deviation_1Day': row['Deviation_1Day'],
				'Gain_Monthly': row['Gain_Monthly'],
				'LossStd_1Year': row['LossStd_1Year'],
				'Point_Value': row['Point_Value'],
				'Comments': row.get('Comments', ''),
				'latestEntry': pd_obj.historyEndDate,
				'PC_LongTerm': PC_LongTerm,
				'PC_ShortTerm': PC_ShortTerm
			})
		if len(candidates) ==0:
			print(f" GetHighestPriceMomentum no candidates found for date {currentDate} ")
			return pd.DataFrame()
		candidates = pd.DataFrame(candidates).set_index('Ticker')

		#More complex filters that I have tried have all decreased performance which is why these are simple
		#Greatest factors for improvement are high 1yr return and a very low selection of stocks, like 1-3
		#Best way to compensate for few stocks is to blend filters of different strengths
		if filterOption ==1: #high performer, recently at a discount or slowing down but not negative
			candidates.sort_values('PC_LongTerm', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') #Most critical factor sorting by largest long term gain
			filter = (candidates['PC_LongTerm'] > candidates['PC_ShortTerm']) & (candidates['PC_LongTerm'] > minPC_1Day) & (candidates['PC_ShortTerm'] > 0) 
		elif filterOption ==2: #Long term gain meets min requirements
			candidates.sort_values('PC_LongTerm', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') #Most critical factor sorting by largest long term gain
			filter = (candidates['PC_LongTerm'] > minPC_1Day)  
		elif filterOption ==3: #Best overall returns 25% average yearly over 36 years which choosing top 5 sorted by best yearly average
			candidates.sort_values('PC_LongTerm', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') #Most critical factor sorting by largest long term gain
			filter = (candidates['PC_LongTerm'] > minPC_1Day) & (candidates['PC_ShortTerm'] > 0) 
		elif filterOption ==4: #Short term gain meets min requirements
			candidates.sort_values('PC_ShortTerm', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') #Most critical factor sorting by largest short term gain which is not effective
			filter =  (candidates['PC_ShortTerm'] > minPC_1Day) 
		elif filterOption ==44: #Short term gain meets min requirements, sort long value
			candidates.sort_values('PC_LongTerm', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') #Most critical factor sorting by largest long term gain
			filter = (candidates['PC_ShortTerm'] > minPC_1Day) 
		elif filterOption ==5: #Point Value
			candidates.sort_values('Point_Value', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') 
			filter = (candidates['PC_1Year'] > minPC_1Day) & (candidates['Point_Value'] > 0)
		elif filterOption ==6: #Hard year, will often not find cadidates
			candidates.sort_values('LossStd_1Year', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') 
			filter = (candidates['PC_1Year'] > 0) & (candidates['LossStd_1Year'] > .06) & (candidates['LossStd_1Year'] < .15) & (candidates['PC_3Month'] > 0) & (candidates['PC_1Month'] > 0)
		else: #no filter
			candidates.sort_values('PC_LongTerm', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') #Most critical factor, sorting by largest long term gain
			filter = (candidates['Price_Current'] > 0)
		candidates = candidates[filter]
		candidates.drop(columns=['PC_LongTerm','PC_ShortTerm'], inplace=True, axis=1)
		candidates = candidates[:stocksToReturn]
		return candidates

	def ToDataFrame(self, intervalInWeeks:int = 1, pivotOnTicker:bool = False, showGain:bool = True, normalizeValues:bool = False):
		#returns DataFrame of historical prices of all tickers in the picker
		#intervalInWeeks > 0 will drop daily values keeping only those that fall on this weekly interval
		#pivotOnTicker returns the tickers as columns with either the price 'Average' or 'Gain' as a value
		#showGain returns only the percentage change from prior value
		#normalizeValues can be used if not returning gains	
		r = pd.DataFrame()
		for i in range(len(self.priceData)):
			t = self.priceData[i].ticker
			temporarilyNormalize = False
			if normalizeValues and not showGain:
				if not self.priceData[i].pricesNormalized:
					temporarilyNormalize = True
					self.priceData[i].NormalizePrices()
			if len(self.priceData[i].historicalPrices) > 0:
				x = self.priceData[i].historicalPrices.copy()
				if intervalInWeeks > 1:
					startDate = x.index[0]
					x['DayOffset'] = (x.index - startDate).days
					x = x[x['DayOffset'] % (intervalInWeeks * 7) ==0] #Drop all data except specified weekly interval of dates
					x.drop(columns=['DayOffset'], inplace=True, axis=1)
				if pivotOnTicker:
					if r.empty: r = pd.DataFrame(index=x.index)
					if showGain:
						r[t] = (x['Average']/x['Average'].shift(-1))
					else:
						r[t] = x['Average']				
				else:
					if showGain:
						x['Gain'] = x['Average']/x['Average'].shift(-1)
						x = x[['Gain']]
					x['Ticker'] = t
					if r.empty: 
						r = x
					else:
						r = r.append(x)
			if temporarilyNormalize: self.priceData[i].NormalizePrices() #undo price normalization
		r.fillna(value=0, inplace=True)
		r.sort_index
		return r

	def LoadTickerFromCSVToSQL(self, ticker:str):
		for i in range(len(self.priceData)):
			if ticker == self.priceData[i].ticker: self.priceData[i].LoadTickerFromCSVToSQL()
			
	