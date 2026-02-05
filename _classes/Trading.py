import time, json
import numpy as np, pandas as pd
import _classes.Constants as CONSTANTS
from tqdm import tqdm
from math import floor
from typing import Dict, Union, Optional
from dataclasses import dataclass, asdict, field, fields, is_dataclass
from datetime import datetime, timedelta
from _classes.DataIO import PTADatabase
from _classes.Prices import PriceSnapshot, PricingData
from _classes.Utility import *

class SQLFriendlyMixin:
    def to_sql_dict(self):
        """Converts dataclass to a flat dict with SQL-compatible types."""
        fields = getattr(self, 'export_fields', [k for k in self.__dict__ if not k.startswith('_')])       
        final_data = {}
        for field_name in fields:
            val = getattr(self, field_name, None)           
            if isinstance(val, (dict, list)):
                continue
            if isinstance(val, (pd.Timestamp, datetime)):
                final_data[field_name] = val.to_pydatetime() if hasattr(val, 'to_pydatetime') else val
            elif isinstance(val, (np.float64, np.float32)):
                final_data[field_name] = float(val)
            elif isinstance(val, (np.int64, np.int32)):
                final_data[field_name] = int(val)
            elif isinstance(val, bool):
                final_data[field_name] = 1 if val else 0
            else:
                final_data[field_name] = val                
        return final_data
		
@dataclass
class TradeModelParams(SQLFriendlyMixin):
	_startDate: pd.Timestamp = field(init=False, repr=False)
	modelName: str = ''
	init_startDate: str = '1/1/1980'
	durationInYears: int = 12
	portfolioSize: int = 100000
	trancheSize: int = 8181
	reEvaluationInterval: int = 20
	longHistory: int = 365
	shortHistory: int = 90
	
	# Filters & Constraints
	stockCount: int = 9
	filterOption: int = 3
	filterByFundamentals: bool = False
	minPercentGain: float = 0.05
	SP500Only: bool = False
	marketCapMin: int = 0
	marketCapMax: int = 0
	
	# System & Execution
	shopBuyPercent: int = 0
	shopSellPercent: int = 0
	trimProfitsPercent: int = 0
	allocateByPointValue: bool = False
	rateLimitTransactions: bool = False
	saveTradeHistory: bool = True
	use_sql: bool = True
	saveResults: bool = False
	verbose: bool = False
	
	batchName: str = ''
	processing_minutes: int = 0	
	export_fields = [
		'modelName', 'startDate', 'endDate', 'durationInYears', 'stockCount', 'reEvaluationInterval', 'SP500Only', 	'longHistory', 'shortHistory', 'minPercentGain', 'startValue', 'endValue',
		'trancheSize', 'shopBuyPercent', 'shopSellPercent', 'trimProfitsPercent', 'allocateByPointValue', 'filterOption', 'filterByFundamentals', 'rateLimitTransactions','marketCapMin', 'marketCapMax', 'processing_minutes', 'batchName'
	]		
	def __post_init__(self):		
		self.startDate = self.init_startDate # Trigger the setters to convert types immediately on startup
	@property
	def startDate(self) -> pd.Timestamp:
		return self._startDate
	@startDate.setter
	def startDate(self, value):
		self._startDate = pd.to_datetime(value)	
		self.init_startDate = self._startDate
	@property
	def endDate(self) -> pd.Timestamp: return self.startDate + pd.Timedelta(days=CONSTANTS.CALENDAR_YEAR * self.durationInYears)	

@dataclass
class TradeModelPerformanceMetrics(SQLFriendlyMixin):
	# --- Core ---
	batchName: str = ''
	startValue: int = 0
	endValue: int  = 0
	startDate:  pd.Timestamp = None 
	endDate:  pd.Timestamp = None 
	durationInYears: int = 0
	total_gain: float = 0.0
	cagr: float = 0.0
	annualized_vol: float = 0.0
	sharpe_ratio: float = 0.0
	calmar_ratio: float = 0.0

	# --- Drawdowns ---
	max_drawdown: float = 0.0
	max_drawdown_start: pd.Timestamp = None 
	max_drawdown_low: pd.Timestamp = None 
	max_drawdown_end: Optional[pd.Timestamp] = None 
	max_drawdown_duration: int = 0

	avg_drawdown: float = 0.0
	dd_95pct: float = 0.0
	dd_99pct: float = 0.0
	pct_time_in_drawdown: float = 0.0

	avg_drawdown_duration: float = 0.0
	median_drawdown_duration: float = 0.0

	avg_recovery_days: float = 0.0
	median_recovery_days: float = 0.0
	max_recovery_days: int = 0

	# --- Calendar effects ---
	worst_jan_drawdown_year: int = 0
	worst_jan_drawdown_value: float = 0.0
	years_negative: int = 0

	# --- Convexity / tail behavior ---
	return_concentration_10pct: float = 0.0
	return_concentration_20pct: float = 0.0
	best_1yr_return: float = 0.0
	best_3yr_cagr: float = 0.0

	# --- Adaptive diagnostics ---
	avg_convex_weight: float = 0.0
	std_convex_weight: float = 0.0
	pct_time_convex_dominant: float = 0.0
	export_fields = [
		'startDate', 'endDate', 'durationInYears', 'total_gain', 'cagr', 'annualized_vol', 'sharpe_ratio', 'calmar_ratio',
		'max_drawdown', 'max_drawdown_start', 'max_drawdown_low', 'max_drawdown_end', 'max_drawdown_duration', 'avg_drawdown','dd_95pct','dd_99pct', 'pct_time_in_drawdown', 'avg_drawdown_duration', 'median_drawdown_duration',
		'avg_recovery_days','median_recovery_days', 'max_recovery_days', 'worst_jan_drawdown_year', 'worst_jan_drawdown_value', 'years_negative',
		'return_concentration_10pct','return_concentration_20pct','best_1yr_return', 'best_3yr_cagr', 'avg_convex_weight','std_convex_weight', 'pct_time_convex_dominant', 'batchName'
	]		

def load_convex_weight_series(db, modelName: str) -> Optional[pd.Series]:
	sql = f"SELECT asOfDate, convex_weight FROM {CONSTANTS.ADAPTIVE_CONVEX_STATE_TABLE} WHERE modelName = :modelName ORDER BY asOfDate"
	df = db.DataFrameFromSQL(sql, params={"modelName": modelName}, indexName="asOfDate")
	if df is None or df.empty:
		return None
	df.index = pd.to_datetime(df.index)
	return df["convex_weight"].astype(float)

def analyze_portfolio_performance( df: pd.DataFrame, risk_free_rate: float = 0.0, convex_weight_series: Optional[pd.Series] = None) -> TradeModelPerformanceMetrics:
	df = df.copy()
	df['Daily_Return'] = (df['TotalValue'].pct_change().replace([np.inf, -np.inf], 0).fillna(0))
	# --- 1. Returns ---
	start_val = df['TotalValue'].iloc[0]
	end_val   = df['TotalValue'].iloc[-1]
	total_gain = (end_val / start_val - 1) if start_val != 0 else 0
	num_days = (df.index[-1] - df.index[0]).days
	years = max(num_days / CONSTANTS.CALENDAR_YEAR, 1 / CONSTANTS.TRADING_YEAR)
	cagr = (1 + total_gain) ** (1 / years) - 1 if total_gain > -1 else 0

	# --- 2. Drawdown series ---
	df['Cumulative_Max'] = df['TotalValue'].cummax()
	df['Drawdown'] = ((df['TotalValue'] - df['Cumulative_Max']) / df['Cumulative_Max']).fillna(0)
	drawdown_series = df['Drawdown']
	mdd_value = drawdown_series.min() # Worst drawdown
	low_date = drawdown_series.idxmin()
	start_date = df.loc[:low_date, 'TotalValue'].idxmax()
	peak_val = df.loc[start_date, 'TotalValue']
	recovery_df = df.loc[low_date:]
	recovered = recovery_df[recovery_df['TotalValue'] >= peak_val]
	if not recovered.empty:
		end_date = recovered.index[0]
		max_dd_duration = (end_date - start_date).days
	else:
		end_date = None
		max_dd_duration = (df.index[-1] - start_date).days

	# --- 3. Drawdown segments & recovery metrics ---
	in_dd = drawdown_series < 0
	pct_time_in_drawdown = float(in_dd.mean())
	dd_groups = (in_dd != in_dd.shift()).cumsum()
	drawdown_durations = []
	recovery_durations = []
	for _, seg in df[in_dd].groupby(dd_groups):
		start = seg.index[0]
		trough = seg['Drawdown'].idxmin()
		trough_val = seg.loc[trough, 'TotalValue']
		after = df.loc[trough:]
		recovery = after[after['TotalValue'] >= df.loc[start, 'TotalValue']]
		drawdown_durations.append((seg.index[-1] - start).days)
		if not recovery.empty:
			recovery_durations.append((recovery.index[0] - trough).days)
	avg_drawdown_duration = float(np.mean(drawdown_durations)) if drawdown_durations else 0.0
	median_drawdown_duration = float(np.median(drawdown_durations)) if drawdown_durations else 0.0
	avg_recovery_days = float(np.mean(recovery_durations)) if recovery_durations else 0.0
	median_recovery_days = float(np.median(recovery_durations)) if recovery_durations else 0.0
	max_recovery_days = int(max(recovery_durations)) if recovery_durations else 0
	avg_drawdown = float(drawdown_series[drawdown_series < 0].mean()) if (drawdown_series < 0).any() else 0.0
	dd_95pct = float(np.percentile(drawdown_series, 5))
	dd_99pct = float(np.percentile(drawdown_series, 1))

	# --- 4. Yearly drawdowns ---
	year_first = df.groupby(df.index.year)['TotalValue'].first()
	yoy_change = year_first.pct_change().fillna(0)
	years_negative = int((yoy_change < 0).sum())
	df['Year_Baseline'] = df.groupby(df.index.year)['TotalValue'].transform('first')
	df['Year_Relative_DD'] = (
		(df['TotalValue'] - df['Year_Baseline']) / df['Year_Baseline']
	).fillna(0)

	yearly_series = df.groupby(df.index.year)['Year_Relative_DD'].min().clip(upper=0)
	worst_year = int(yearly_series.idxmin())
	worst_year_val = float(yearly_series.min())

	# --- 5. Risk ratios ---
	vol = df['Daily_Return'].std() * np.sqrt(CONSTANTS.TRADING_YEAR)
	vol = 0 if np.isnan(vol) else vol

	daily_rf = risk_free_rate / CONSTANTS.TRADING_YEAR
	dr_std = df['Daily_Return'].std()
	sharpe = ((df['Daily_Return'].mean() - daily_rf) / dr_std * np.sqrt(CONSTANTS.TRADING_YEAR)) if dr_std != 0 else 0
	calmar = cagr / abs(mdd_value) if mdd_value != 0 else 0

	# --- 6. Convexity / return concentration ---
	monthly = df['TotalValue'].resample('ME').last().pct_change().dropna()
	top_10pct = int(len(monthly) * 0.10)
	top_20pct = int(len(monthly) * 0.20)

	return_concentration_10pct = monthly.nlargest(top_10pct).sum() / monthly.sum() if monthly.sum() != 0 else 0
	return_concentration_20pct = monthly.nlargest(top_20pct).sum() / monthly.sum() if monthly.sum() != 0 else 0

	rolling_1y = df['TotalValue'].pct_change(CONSTANTS.TRADING_YEAR).dropna()
	best_1yr_return = float(rolling_1y.max()) if not rolling_1y.empty else 0.0

	rolling_3y = (df['TotalValue'].pct_change(CONSTANTS.TRADING_YEAR * 3) + 1) ** (1 / 3) - 1
	best_3yr_cagr = float(rolling_3y.max()) if not rolling_3y.empty else 0.0

	# --- 7. Adaptive diagnostics ---
	if convex_weight_series is not None and not convex_weight_series.empty:
		aligned = convex_weight_series.reindex(df.index).fillna(0).astype(float)
		avg_convex_weight = float(aligned.mean())
		std_convex_weight = float(aligned.std())
		pct_time_convex_dominant = float((aligned > 0.5).mean())
	else:
		avg_convex_weight = 0.0
		std_convex_weight = 0.0
		pct_time_convex_dominant = 0.0

	# --- 8. Return ---
	return TradeModelPerformanceMetrics(
		total_gain = total_gain,
		cagr = cagr,
		annualized_vol = vol,
		sharpe_ratio = sharpe,
		calmar_ratio = calmar,
		max_drawdown = mdd_value,
		max_drawdown_start = start_date,
		max_drawdown_low = low_date,
		max_drawdown_end = end_date,
		max_drawdown_duration = max_dd_duration,
		avg_drawdown = avg_drawdown,
		dd_95pct = dd_95pct,
		dd_99pct = dd_99pct,
		pct_time_in_drawdown = pct_time_in_drawdown,
		avg_drawdown_duration = avg_drawdown_duration,
		median_drawdown_duration = median_drawdown_duration,
		avg_recovery_days = avg_recovery_days,
		median_recovery_days = median_recovery_days,
		max_recovery_days = max_recovery_days,
		worst_jan_drawdown_year = worst_year,
		worst_jan_drawdown_value = worst_year_val,
		years_negative = years_negative,
		return_concentration_10pct = return_concentration_10pct,
		return_concentration_20pct = return_concentration_20pct,
		best_1yr_return = best_1yr_return,
		best_3yr_cagr = best_3yr_cagr,
		avg_convex_weight = avg_convex_weight,
		std_convex_weight = std_convex_weight,
		pct_time_convex_dominant = pct_time_convex_dominant
	)

def UpdateTradeModelComparisonsFromDailyValue(batchName: str):
	db = PTADatabase()
	if not db.Open():
		return
	sql_models = "SELECT DISTINCT TradeModel FROM TradeModel_DailyValue WHERE BatchName = :batchName"
	model_names = db.ScalarListFromSQL(sql_models, params={"batchName": batchName}, column="TradeModel")
	for model_name in model_names:
		sql_dv = "SELECT Date, TotalValue FROM TradeModel_DailyValue WHERE BatchName = :batchName ORDER BY Date"
		df = db.DataFrameFromSQL(sql_dv, params={"batchName": batchName}, indexName="Date")        
		if df is None or df.empty or len(df) < 2:
			continue           
		df.index = pd.to_datetime(df.index)
		convex_weight_series = load_convex_weight_series(db, model_name)
		metrics = analyze_portfolio_performance(df=df, convex_weight_series=convex_weight_series)        
		metrics.startDate = df.index.min()
		metrics.endDate = df.index.max()
		metrics.startValue = df['TotalValue'].iloc[0]
		metrics.endValue = df['TotalValue'].iloc[-1]
		metrics.modelName = model_name
		metrics.batchName = batchName       
		days = (metrics.endDate - metrics.startDate).days
		metrics.durationInYears = int(round(days / CONSTANTS.CALENDAR_YEAR))
		update_params = metrics.to_sql_dict()
		update_params = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in update_params.items()}
		cols_to_update = [k for k in update_params.keys() if k not in ['modelName', 'batchName']]
		set_clause = ", ".join([f"{col} = :{col}" for col in cols_to_update])  
		sql_update = f"UPDATE TradeModelComparisons SET {set_clause} WHERE batchName = :batchName"
		db.ExecSQL(sql_update, params=update_params)
	db.Close()
	
class Tranche: #interface for handling actions on a chunk of funds
	def __init__(self, size:int=1000):
		self.ticker = ''
		self.size = size
		self.units = 0
		self.available = True
		self.purchased = False
		self.marketOrder = False
		self.sold = False
		self.expired = False
		self.dateBuyOrderPlaced = None
		self.dateBuyOrderFilled = None
		self.dateSellOrderPlaced = None
		self.dateSellOrderFilled = None
		self.buyOrderPrice = 0
		self.purchasePrice = 0
		self.sellOrderPrice = 0
		self.sellPrice = 0
		self.latestPrice = 0
		self.expireAfterDays = 0
		self._verbose = False
		
	def AdjustBuyUnits(self, newValue:int):	
		if self._verbose: print(' Adjusting Buy from ' + str(self.units) + ' to ' + str(newValue) + ' units (' + self.ticker + ')')
		self.units=newValue

	def CancelOrder(self, verbose:bool=False): 
		self.marketOrder=False
		self.expireAfterDays=0
		if self.purchased:
			if verbose: print(f" Sell order for {self.ticker} was canceled.")
			self.dateSellOrderPlaced = None
			self.sellOrderPrice = 0
			self.expired=False
		else:
			if verbose: print(f" Buy order for {self.ticker} was canceled.")
			self.Recycle()
		
	def Expire(self):
		if not self.purchased:  #cancel buy
			if self._verbose: print(f" Buy order from {self.dateBuyOrderPlaced} has expired. ticker: {self.ticker}")
			self.Recycle()
		else: #cancel sell
			if self._verbose: print(f" Sell order from {self.dateSellOrderPlaced} has expired. ticker: {self.ticker}")
			self.dateSellOrderPlaced=None
			self.sellOrderPrice=0
			self.marketOrder = False
			self.expireAfterDays = 0
			self.expired=False

	def PlaceBuy(self, ticker:str, price:float, datePlaced:datetime, marketOrder:bool=False, expireAfterDays:int=10, verbose:bool=False):
		#returns amount taken out of circulation by the order
		r = 0
		self._verbose = verbose
		if self.available and price > 0:
			self.available = False
			self.ticker=ticker
			self.marketOrder = marketOrder
			self.dateBuyOrderPlaced = datePlaced
			self.buyOrderPrice=price
			self.units = floor(self.size/price)
			self.purchased = False
			self.expireAfterDays=expireAfterDays
			r=(price*self.units)
			if self._verbose: 
				if marketOrder:
					print(datePlaced, f" Buy placed ticker: {self.ticker} price: Market units:{self.units}")
				else:
					print(datePlaced, f" Buy placed ticker: {self.ticker} price: ${price} units:{self.units}")
		return r
		
	def PlaceSell(self, price, datePlaced, marketOrder:bool=False, expireAfterDays:int=10, verbose:bool=False):
		r = False
		self._verbose = verbose
		if self.purchased and price > 0:
			self.sold = False
			self.dateSellOrderPlaced = datePlaced
			self.sellOrderPrice = price
			self.marketOrder = marketOrder
			self.expireAfterDays=expireAfterDays
			if self._verbose: 
				if marketOrder: 
					print(datePlaced, f" Sell placed ticker: {self.ticker} price: Market units:{self.units}")
				else:
					print(datePlaced, f" Sell placed ticker: {self.ticker} price: ${price} units:{self.units}")
			r=True
		return r

	def PrintDetails(self):
		if not self.ticker =='' or True:
			print("Stock: " + self.ticker)
			print("units: " + str(self.units))
			print("available: " + str(self.available))
			print("purchased: " + str(self.purchased))
			print("dateBuyOrderPlaced: " + str(self.dateBuyOrderPlaced))
			print("dateBuyOrderFilled: " + str(self.dateBuyOrderFilled))
			print("buyOrderPrice: " + str(self.buyOrderPrice))
			print("purchasePrice: " + str(self.purchasePrice))
			print("dateSellOrderPlaced: " + str(self.dateSellOrderPlaced))
			print("dateSellOrderFilled: " + str(self.dateSellOrderFilled))
			print("sellOrderPrice: " + str(self.sellOrderPrice))
			print("sellPrice: " + str(self.sellPrice))
			print("latestPrice: " + str(self.latestPrice))
			print("\n")

	def Recycle(self):
		self.ticker = ""
		self.units = 0
		self.available = True
		self.purchased = False
		self.sold = False
		self.expired = False
		self.marketOrder = False
		self.dateBuyOrderPlaced = None
		self.dateBuyOrderFilled = None
		self.dateSellOrderPlaced = None
		self.dateSellOrderFilled = None
		self.latestPrice=None
		self.buyOrderPrice = 0
		self.purchasePrice = 0
		self.sellOrderPrice = 0
		self.sellPrice = 0
		self.expireAfterDays = 0
		self._verbose=False
	
	def UpdateStatus(self, price, dateChecked):
		#Returns True if the order had action: filled or expired.
		r = False
		if price > 0: 
			self.latestPrice = price
			if self.buyOrderPrice > 0 and not self.purchased:
				if self.buyOrderPrice >= price or self.marketOrder:
					self.dateBuyOrderFilled = dateChecked
					self.purchasePrice = price
					self.purchased=True
					if self._verbose: print(dateChecked, ' Buy ordered on ' + str(self.dateBuyOrderPlaced) + ' filled for ' + str(price) + ' (' + self.ticker + ')')
					r=True
				else:
					self.expired = (DateDiffDays(self.dateBuyOrderPlaced , dateChecked) > self.expireAfterDays)
					if self.expired and self._verbose: print(dateChecked, ' Buy order from ' + str(self.dateBuyOrderPlaced) + ' expired.')
					r = self.expired
			elif self.sellOrderPrice > 0 and not self.sold:
				if self.sellOrderPrice <= price or self.marketOrder:
					self.dateSellOrderFilled = dateChecked
					self.sellPrice = price
					self.sold=True
					self.expired=False
					if self._verbose: print(dateChecked, ' Sell ordered on ' + str(self.dateSellOrderPlaced) + ' filled for ' + str(price) + ' (' + self.ticker + ')') 
					r=True
				else:
					self.expired = (DateDiffDays(self.dateSellOrderPlaced, dateChecked) > self.expireAfterDays)
					if self.expired and self._verbose: print(dateChecked, ' Sell order from ' + str(self.dateSellOrderPlaced) + ' expired.')
					r = self.expired
			else:
				r=False
		return r

class Position:	#Simple interface for open positions
	def __init__(self, t:Tranche):
		self._t = t
		self.ticker = t.ticker
	def CancelSell(self): 
		if self._t.purchased: self._t.CancelOrder(verbose=True)
	def CurrentValue(self): return self._t.units * self._t.latestPrice
	def Sell(self, price:float, datePlaced:datetime, marketOrder:bool=False, expireAfterDays:int=90): self._t.PlaceSell(price=price, datePlaced=datePlaced, marketOrder=marketOrder, expireAfterDays=expireAfterDays, verbose=False)
	def SellPending(self): return (self._t.sellOrderPrice >0) and not (self._t.sold or  self._t.expired)
	def LatestPrice(self): return self._t.latestPrice

class Portfolio:
	def __init__(self, portfolioName:str, startDate:datetime, totalFunds:int=10000, trancheSize:int=1000, trackHistory:bool=True, useDatabase:bool=True, verbose:bool=False):
		self.tradeHistory = None #DataFrame of trades.  Note: though you can trade more than once a day it is only going to keep one entry per day per stock
		self._commisionCost = 0
		self.portfolioName = portfolioName
		self._initialValue = totalFunds
		self._cash = totalFunds
		self.assetValue = 0
		self._fundsCommittedToOrders = 0
		self._verbose = verbose
		self._trancheCount = floor(totalFunds/trancheSize)
		self._tranches = [Tranche(trancheSize) for x in range(self._trancheCount)]
		self.dailyValue = pd.DataFrame([[startDate,totalFunds,0,totalFunds,'','','','','','','','','','','']], columns=list(['Date','CashValue','AssetValue','TotalValue','Stock00','Stock01','Stock02','Stock03','Stock04','Stock05','Stock06','Stock07','Stock08','Stock09','Stock10']))
		self.dailyValue.set_index(['Date'], inplace=True)
		self.database = None
		if useDatabase:
			db = PTADatabase()
			if db.database_configured:
				self.database = db
			else:
				useDatabase = False
		self.useDatabase = useDatabase
		self.trackHistory = trackHistory
		if trackHistory: 
			self.tradeHistory = pd.DataFrame(columns=['dateBuyOrderPlaced','ticker','dateBuyOrderFilled','dateSellOrderPlaced','dateSellOrderFilled','units','buyOrderPrice','purchasePrice','sellOrderPrice','sellPrice','NetChange'])
			self.tradeHistory.set_index(['dateBuyOrderPlaced','ticker'], inplace=True)

	#----------------------  Status and position info  ---------------------------------------
	def AccountingError(self):
		r = False
		if not self.ValidateFundsCommittedToOrders() == 0: 
			print(' Accounting error: inaccurcy in funds committed to orders!')
			r=True
		if self.FundsAvailable() + self._trancheCount*self._commisionCost < -10: #Over-committed funds		
			OrdersAdjusted = False		
			for t in self._tranches:
				if not t.purchased and t.units > 0:
					print(' Reducing purchase of ' + t.ticker + ' by one unit due to overcommitted funds.')
					t.units -= 1
					OrdersAdjusted = True
					break
			if OrdersAdjusted: self.ValidateFundsCommittedToOrders(True)
			if self.FundsAvailable() + self._trancheCount*self._commisionCost < -10: 
				OrdersAdjusted = False		
				for t in self._tranches:
					if not t.purchased and t.units > 1:
						print(' Reducing purchase of ' + t.ticker + ' by two units due to overcommitted funds.')
						t.units -= 2
						OrdersAdjusted = True
						break
			if self.FundsAvailable() + self._trancheCount*self._commisionCost < -10: #Over-committed funds						
				print(' Accounting error: negative cash balance.  (Cash, CommittedFunds, AvailableFunds) ', self._cash, self._fundsCommittedToOrders, self.FundsAvailable())
				r=True
		return r

	def FundsAvailable(self): return (self._cash - self._fundsCommittedToOrders)
	
	def PendingOrders(self):
		a, b, s, l = self.PositionSummary()
		return (b+s > 0)

	def _ActiveTickers(self):
		# Returns positions or open orders
		result = set()
		for t in self._tranches:
			if not t.available:
				result.add(t.ticker)
		return result

	def GetPositions(self, ticker:str='', asDataFrame:bool=False):	#returns reference to the tranche of active positions or a dataframe with counts
		r = []
		for t in self._tranches:
			if t.purchased and (t.ticker==ticker or ticker==''): 
				p = Position(t)
				r.append(p)
		if asDataFrame:
			y=[]
			for x in r: y.append(x.ticker)
			r = pd.DataFrame(y,columns=list(['Ticker']))
			r = r.groupby(['Ticker']).size().reset_index(name='CurrentHoldings')
			r.set_index(['Ticker'], inplace=True)
			TotalHoldings = r['CurrentHoldings'].sum()
			r['Percentage'] = r['CurrentHoldings']/TotalHoldings
		return r

	def PositionSummary(self):
		available=0
		buyPending=0
		sellPending=0
		longPostition = 0
		for t in self._tranches:
			if t.available:
				available +=1
			elif not t.purchased:
				buyPending +=1
			elif t.purchased and t.dateSellOrderPlaced==None:
				longPostition +=1
			elif t.dateSellOrderPlaced:
				sellPending +=1
		return available, buyPending, sellPending, longPostition			

	def PrintPositions(self):
		i=0
		for t in self._tranches:
			if not t.ticker =='' or True:
				print('Set: ' + str(i))
				t.PrintDetails()
			i=i+1
		print('Funds committed to orders: ' + str(self._fundsCommittedToOrders))
		print('available funds: ' + str(self._cash - self._fundsCommittedToOrders))

	def TranchesAvailable(self):
		a, b, s, l = self.PositionSummary()
		return a

	def ValidateFundsCommittedToOrders(self, SaveAdjustments:bool=True):
		#Returns difference between recorded value and actual
		x=0
		for t in self._tranches:
			if not t.available and not t.purchased: 
				x = x + (t.units*t.buyOrderPrice) + self._commisionCost
		if round(self._fundsCommittedToOrders, 5) == round(x,5): self._fundsCommittedToOrders=x
		if not (self._fundsCommittedToOrders - x) ==0:
			if SaveAdjustments: 
				self._fundsCommittedToOrders = x
			else:
				print( 'Committed funds variance actual/recorded', x, self._fundsCommittedToOrders)
		return (self._fundsCommittedToOrders - x)

	def _calculate_asset_value(self):
		assetValue=0
		for t in self._tranches:
			if t.purchased:
				assetValue = assetValue + (t.units*t.latestPrice)
		self.assetValue = assetValue

	def Value(self):
		self._calculate_asset_value()
		return self._cash, self.assetValue
		
	def ReEvaluateTrancheCount(self, verbose:bool=False):
		#Portfolio performance may require adjusting the available Tranches
		trancheSize = self._tranches[0].size
		c = self._trancheCount
		availableTranches,_,_,_ = self.PositionSummary()
		availableFunds = self._cash - self._fundsCommittedToOrders
		targetAvailable = int(availableFunds/trancheSize)
		if targetAvailable > availableTranches:
			if verbose: 
				print(' Available Funds: ', availableFunds, availableTranches * trancheSize)
				print(' Adding ' + str(targetAvailable - availableTranches) + ' new Tranches to portfolio..')
			for i in range(targetAvailable - availableTranches):
				self._tranches.append(Tranche(trancheSize))
				self._trancheCount +=1
		elif targetAvailable < availableTranches:
			if verbose: print( 'Removing ' + str(availableTranches - targetAvailable) + ' tranches from portfolio..')
			#print(targetAvailable, availableFunds, trancheSize, availableTranches)
			i = self._trancheCount-1
			while i > 0:
				if self._tranches[i].available and targetAvailable < availableTranches:
					if verbose: 
						print(' Available Funds: ', availableFunds, availableTranches * trancheSize)
						print(' Removing tranch at ', i)
					self._tranches.pop(i)	#remove last available
					self._trancheCount -=1
					availableTranches -=1
				i -=1

	#--------------------------------------  Order interface  ---------------------------------------
	def CancelAllOrders(self, currentDate:datetime):
		for t in self._tranches:
			t.CancelOrder()

	def PlaceBuy(self, ticker:str, price:float, datePlaced:datetime, marketOrder:bool=False, expireAfterDays:int=10, verbose:bool=False):
		#Place with first available tranch, returns True if order was placed
		r=False
		price = round(price, 3)
		oldestExistingOrder = None
		availableCash = self.FundsAvailable()
		units=0
		if price > 0: units = int(self._tranches[0].size/price)
		cost = units*price + self._commisionCost
		if availableCash < cost and units > 2:
			units -=1
			cost = units*price + self._commisionCost
		if units == 0 or availableCash < cost:
			if verbose: 
				if price==0:
					print( 'Unable to purchase ' + ticker + '.  Price lookup failed.', datePlaced)
				else:
					print( 'Unable to purchase ' + ticker + '.  Price (' + str(price) + ') exceeds available funds ' + str(availableCash) + ' Traunche Size: ' + str(self._tranches[0].size))
		else:	
			for t in self._tranches: #Find available 
				if t.available :	#Place new order
					self._fundsCommittedToOrders = self._fundsCommittedToOrders + cost 
					x = self._commisionCost + t.PlaceBuy(ticker=ticker, price=price, datePlaced=datePlaced, marketOrder=marketOrder, expireAfterDays=expireAfterDays, verbose=verbose) 
					if not x == cost: #insufficient funds for full purchase
						if verbose: print(' Expected cost changed from', cost, 'to', x)
						self._fundsCommittedToOrders = self._fundsCommittedToOrders - cost + x + self._commisionCost
					r=True
					break
				elif not t.purchased and t.ticker == ticker:	#Might have to replace existing order
					if oldestExistingOrder == None:
						oldestExistingOrder=t.dateBuyOrderPlaced
					else:
						if oldestExistingOrder > t.dateBuyOrderPlaced: oldestExistingOrder=t.dateBuyOrderPlaced
		if not r and units > 0 and False:	#We could allow replacing oldest existing order
			if oldestExistingOrder == None:
				if self.TranchesAvailable() > 0:
					if verbose: print(' Unable to buy ' + str(units) + ' of ' + ticker + ' with funds available: ' + str(FundsAvailable))
				else: 
					if verbose: print(' Unable to buy ' + ticker + ' no tranches available')
			else:
				for t in self._tranches:
					if not t.purchased and t.ticker == ticker and oldestExistingOrder==t.dateBuyOrderPlaced:
						if verbose: print(' No tranch available... replacing order from ' + str(oldestExistingOrder))
						oldCost = t.buyOrderPrice * t.units + self._commisionCost
						if verbose: print(' Replacing Buy order for ' + ticker + ' from ' + str(t.buyOrderPrice) + ' to ' + str(price))
						t.units = units
						t.buyOrderPrice = price
						t.dateBuyOrderPlaced = datePlaced
						t.marketOrder = marketOrder
						self._fundsCommittedToOrders = self._fundsCommittedToOrders - oldCost + cost 
						r=True
						break		
		return r

	def PlaceSell(self, ticker:str, price:float, datePlaced:datetime, marketOrder:bool=False, expireAfterDays:int=10, datepurchased:datetime=None, verbose:bool=False):
		#Returns True if order was placed
		r=False
		price = round(price, 3)
		for t in self._tranches:
			if t.ticker == ticker and t.purchased and t.sellOrderPrice==0 and (datepurchased is None or t.dateBuyOrderFilled == datepurchased):
				t.PlaceSell(price=price, datePlaced=datePlaced, marketOrder=marketOrder, expireAfterDays=expireAfterDays, verbose=verbose)
				r=True
				break
		if not r:	#couldn't find one without a sell, try to update an existing sell order
			for t in self._tranches:
				if t.ticker == ticker and t.purchased:
					if verbose: print(' Updating existing sell order ')
					t.PlaceSell(price=price, datePlaced=datePlaced, marketOrder=marketOrder, expireAfterDays=expireAfterDays, verbose=verbose)
					r=True
					break					
		return r

	def SellAllPositions(self, datePlaced:datetime, ticker:str='', verbose:bool=False, allowWeekEnd:bool=False):
		for t in self._tranches:
			if t.purchased and (t.ticker==ticker or ticker==''): 
				t.PlaceSell(price=t.latestPrice, datePlaced=datePlaced, marketOrder=True, expireAfterDays=5, verbose=verbose)
		self.ProcessDay(withIncrement=False, allowWeekEnd=allowWeekEnd)

	#--------------------------------------  Order Processing ---------------------------------------
	def _CheckOrders(self, ticker, price, dateChecked):
		#check if there was action on any pending orders and update current price of tranche
		price = round(price, 3)
		#self._verbose = True
		for t in self._tranches:
			if t.ticker == ticker:
				r = t.UpdateStatus(price, dateChecked)
				if r:	#Order was filled, update account
					if t.expired:
						if not t.purchased: 
							if self._verbose: print(f" Buy order from {t.dateBuyOrderPlaced} has expired on {dateChecked}. ticker: {t.ticker}")
							self._fundsCommittedToOrders -= (t.units*t.buyOrderPrice)	#remove from funds committed to orders
							self._fundsCommittedToOrders -= self._commisionCost
						else:
							if self._verbose: print(f" Sell order from {t.dateSellOrderPlaced} has expired on {dateChecked}. ticker: {t.ticker}")
						t.Expire()
					elif t.sold:
						if self._verbose: print(t.ticker, " sold for ",t.sellPrice, dateChecked)
						self._cash = self._cash + (t.units*t.sellPrice) - self._commisionCost
						if self._verbose and self._commisionCost > 0: print(' Commission charged for Sell: ' + str(self._commisionCost))
						if self.trackHistory:
							self.tradeHistory.loc[(t.dateBuyOrderPlaced, t.ticker)]=[t.dateBuyOrderFilled,t.dateSellOrderPlaced,t.dateSellOrderFilled,t.units,t.buyOrderPrice,t.purchasePrice,t.sellOrderPrice,t.sellPrice,((t.sellPrice - t.purchasePrice)*t.units)-self._commisionCost*2] 
						t.Recycle()
					elif t.purchased:
						self._fundsCommittedToOrders -= (t.units*t.buyOrderPrice)	#remove from funds committed to orders
						self._fundsCommittedToOrders -= self._commisionCost
						fundsavailable = self._cash - abs(self._fundsCommittedToOrders)
						if t.marketOrder:
							actualCost = t.units*price
							if self._verbose: print(f" CheckOrders: {t.ticker} purchased for {price} on {dateChecked}")
							if (fundsavailable - actualCost - self._commisionCost) < 25:	#insufficient funds
								unitsCanAfford = max(floor((fundsavailable - self._commisionCost)/price)-1, 0)
								if self._verbose:
									print(f" CheckOrders: Ajusting units on market order for {ticker} price {price} units {t.units},  can afford {unitsCanAfford} units")
									print(f" CheckOrders: Cash: {self._cash} Committed Funds: {self._fundsCommittedToOrders} Available: {fundsavailable}")
								if unitsCanAfford ==0:
									t.Recycle()
								else:
									t.AdjustBuyUnits(unitsCanAfford)
						if t.units == 0:
							if self._verbose: print(f" CheckOrders: Cannot afford any {ticker} at market {price}... canceling Buy on {dateChecked}")
							t.Recycle()
						else:
							self._cash = self._cash - (t.units*price) - self._commisionCost 
							if self._verbose and self._commisionCost > 0: print(' CheckOrders: Commission charged for Buy: ' + str(self._commisionCost))		
							if self.trackHistory:							
								self.tradeHistory.loc[(t.dateBuyOrderPlaced,t.ticker), 'dateBuyOrderFilled']=t.dateBuyOrderFilled #Create the row
								self.tradeHistory.loc[(t.dateBuyOrderPlaced,t.ticker)]=[t.dateBuyOrderFilled,t.dateSellOrderPlaced,t.dateSellOrderFilled,t.units,t.buyOrderPrice,t.purchasePrice,t.sellOrderPrice,t.sellPrice,''] 
						
	def _CheckPriceSequence(self, ticker, p1, p2, dateChecked):
		#approximate a price sequence between given prices
		steps=40
		if p1==p2:
			self._CheckOrders(ticker, p1, dateChecked)		
		else:
			step = (p2-p1)/steps
			for i in range(steps):
				p = round(p1 + i * step, 3)
				self._CheckOrders(ticker, p, dateChecked)
			self._CheckOrders(ticker, p2, dateChecked)	

	def ProcessDaysOrders(self, ticker, open, high, low, close, dateChecked):
		#approximate a sequence of the day's prices for given ticker, check orders for each, update price value
		if self.PendingOrders() > 0:
			p2=low
			p3=high
			if (high - open) < (open - low):
				p2=high
				p3=low
			#print(' Given price sequence      ' + str(open) + ' ' + str(high) + ' ' + str(low) + ' ' + str(close))
			#print(' Estimating price sequence ' + str(open) + ' ' + str(p2) + ' ' + str(p3) + ' ' + str(close))
			self._CheckPriceSequence(ticker, open, p2, dateChecked)
			self._CheckPriceSequence(ticker, p2, p3, dateChecked)
			self._CheckPriceSequence(ticker, p3, close, dateChecked)
		else:
			self._CheckOrders(ticker, close, dateChecked)	#No open orders but still need to update last prices
		self.ValidateFundsCommittedToOrders(True)

	def UpdateDailyValue(self):
		_cashValue, assetValue = self.Value()
		positions = self.GetPositions(asDataFrame=True)	
		x = (positions.index.astype(str) + ':' + positions['Percentage'].astype(str))	
		x = x.str[:12].tolist()		
		padding_needed = max(0, 11 - len(x))
		x += [''] * padding_needed		
		new_row = [_cashValue, assetValue, _cashValue + assetValue] + x[:11]
		self.dailyValue.loc[self.currentDate] = new_row	

	#--------------------------------------  Closing Reporting ---------------------------------------
	def SaveTradeHistory(self, foldername:str, addTimeStamp:bool = False):
		if self.trackHistory:
			if self.useDatabase:
				if self.database.Open():
					df = self.tradeHistory.copy()
					df['TradeModel'] = self.modelName 
					df['BatchName'] = self.batchName 				
					self.database.DataFrameToSQL(df, 'TradeModel_Trades', indexAsColumn=True)
					self.database.Close()
			if CreateFolder(foldername):
				filePath = foldername + self.portfolioName 
				if addTimeStamp: filePath += '_' + GetDateTimeStamp()
				filePath += '_trades.csv'
				self.tradeHistory.to_csv(filePath)

	def SaveDailyValue(self, foldername:str, addTimeStamp:bool = False):
		if self.useDatabase:
			if self.database.Open():
				df = self.dailyValue.copy()
				df['TradeModel'] = self.modelName 
				df['BatchName'] = self.batchName 				
				self.database.DataFrameToSQL(df, 'TradeModel_DailyValue', indexAsColumn=True)
				self.database.Close()
		if CreateFolder(foldername):
			filePath = foldername + self.portfolioName 
			if addTimeStamp: filePath += '_' + GetDateTimeStamp()
			filePath+= '_dailyvalue.csv'
			self.dailyValue.to_csv(filePath)
		
class TradingModel(Portfolio):
	#Extends Portfolio to trading environment for testing models
	def __init__(self, modelName:str, startingTicker:str, startDate:datetime, durationInYears:int, totalFunds:int, trancheSize:int=1000, trackHistory:bool=True, useDatabase:bool=True, useFullStats:bool=True, verbose:bool=False):
		#pricesAsPercentages:bool=False would be good but often results in NaN values
		#expects date format in local format, from there everything will be converted to database format				
		self.modelStartDate  = None	
		self.modelEndDate = None
		self.modelReady = False
		self.currentDate = None
		self.priceHistory = []  #list of price histories for each stock in _tickerList
		self._priceMap = {} #Mapping between tickers and priceHistory elements
		self.startingValue = 0 
		self.verbose = verbose
		self._tickerList = []	#list of stocks currently held
		self._dataFolderTradeModel = 'data/trademodel/'
		self.Custom1 = None	#can be used to store custom values when using the model
		self.Custom2 = None
		self._NormalizePrices = False
		self.useFullStats = useFullStats
		self.startingValue = totalFunds
		startDate = ToDateTime(startDate)
		endDate = startDate + timedelta(days=CONSTANTS.CALENDAR_YEAR * durationInYears)
		CreateFolder(self._dataFolderTradeModel)
		self.start_processing = datetime.today()
		self.end_processing = None
		self.modelName = modelName
		self.batchName = modelName[:30] + '_' + self.start_processing.strftime('%Y%m%d-%H%M%S')
		self.database = None
		self.pbar = None
		#total_days = (endDate - startDate).days    	
		total_days = len(pd.bdate_range(start=startDate, end=endDate))
		if useDatabase:
			db = PTADatabase()
			if db.database_configured:
				self.database = db
			else:
				useDatabase = False
		self.useDatabase = useDatabase
		p = PricingData(startingTicker, useDatabase=self.useDatabase)
		if p.LoadHistory(requestedStartDate=startDate, requestedEndDate=endDate, verbose=verbose): 
			if verbose: print(' Loading ' + startingTicker)
			p.CalculateStats(fullStats=self.useFullStats)
			valid_dates = p.historicalPrices.index[(p.historicalPrices.index >= startDate) & (p.historicalPrices.index <= endDate)]
			if not valid_dates.empty:
				self.priceHistory = [p] #add to list
				self._priceMap[p.ticker] = p
				self.modelStartDate = valid_dates[0]
				self.modelEndDate = valid_dates[-1]
				self.currentDate = self.modelStartDate
				self._tickerList = [startingTicker]
				self.pbar = tqdm(total=total_days, desc=f"Running {self.modelName} from {self.modelStartDate} to {self.modelEndDate}", unit="day")
				self.modelReady = len(p.historicalPrices) > 30
			else:
				self.modelReady = False #We don't have enough data
		super(TradingModel, self).__init__(portfolioName=modelName, startDate=startDate, totalFunds=totalFunds, trancheSize=trancheSize, trackHistory=trackHistory, useDatabase=useDatabase, verbose=verbose)	

	def AddTicker(self, ticker:str):
		r = False
		if not ticker in self._tickerList:
			p = PricingData(ticker, useDatabase=self.useDatabase)
			if self.verbose: print(' Loading price history for ' + ticker)
			if p.LoadHistory(requestedStartDate=self.modelStartDate, requestedEndDate=self.modelEndDate): 
				p.CalculateStats(fullStats=self.useFullStats)
				if len(p.historicalPrices) > len(self.priceHistory[0].historicalPrices): #first element is used for trading day indexing, replace if this is a better match
					self.priceHistory.insert(0, p)
					self._tickerList.insert(0, ticker)
				else:
					self.priceHistory.append(p)
					self._tickerList.append(ticker)
				r = True
				self._priceMap[p.ticker] = p
				print(' AddTicker: Added ticker ' + ticker)
			else:
				print( ' AddTicker: Unable to download price history for ticker ' + ticker)
		return r

	def RemoveTicker(self, ticker):
		self.priceHistory = [p for p in self.priceHistory if p.ticker != ticker]
		self._priceMap.pop(ticker, None)
		self._tickerList.remove(ticker)

	def AlignPositions(self, targetPositions: pd.DataFrame, rateLimitTransactions: bool = False, shopBuyPercent: int = 0, shopSellPercent: int = 0, trimProfitsPercent: int = 0, verbose: bool = False): 
		#Performs necessary Buy/Sells to get from current positions to target positions
		#Input ['Ticker']['TargetHoldings'] combo which indicates proportion of desired holdings
		#rateLimitTransactions will limit number of buys/sells per day to one per ticker
		#if not tradeAtMarket then will shop for a good buy and sell price, so far all attempts at shopping or trimming profits yield 3%-13% less average profit
		expireAfterDays = 3
		tradeAtMarket = (shopBuyPercent == 0) and (shopSellPercent == 0) 
		TotalTranches = self._trancheCount
		TotalTargets = targetPositions['TargetHoldings'].sum() 
		scale = 1.0
		if TotalTargets > 0:
			scale = TotalTranches / TotalTargets
			targetPositions.loc[:, 'TargetHoldings'] = (targetPositions['TargetHoldings'] * scale).astype(float).round()
		if verbose: print(f" AlignPositions: Target Positions Scaled. Scale: {scale}")
		currentPositions = self.GetPositions(asDataFrame=True)
		targetPositions = targetPositions.join(currentPositions, how='outer')
		targetPositions.fillna(value=0, inplace=True)		
		targetPositions['Difference'] = targetPositions['TargetHoldings'] - targetPositions['CurrentHoldings']
		
		# Sort Sells (Difference < 0) and Buys (Difference > 0)
		sells = targetPositions[targetPositions['Difference'] < 0].copy()
		sells.sort_values(by=['Difference'], ascending=True, inplace=True) 
		buys = targetPositions[targetPositions['Difference'] > 0].copy()
		buys.sort_values(by=['Difference'], ascending=False, inplace=True) 	
		executionList = pd.concat([sells, buys])
		for ticker, row in executionList.iterrows():
			if ticker == CONSTANTS.CASH_TICKER:
				shopBuyPercent = 0
				shopSellPercent = 0
				trimProfitsPercent = 0
				rateLimitTransactions = False
			orders = int(row['Difference'])
			sn = self.GetPriceSnapshot(ticker)
			if sn is not None:
				target_buy = round(min(sn.Average, sn.Target) * (1 - shopBuyPercent / 100), 2)
				target_sell = round(max(sn.Average, sn.Target) * (1 + shopSellPercent / 100), 2)
				trim_sell = round(max(sn.Average, sn.Target) * (1 + trimProfitsPercent / 100), 2)				
				if orders < 0:  # --- EXECUTE SELLS ---
					price = target_sell if not tradeAtMarket else sn.Average
					if rateLimitTransactions: orders = -1					
					if verbose: print(f" AlignPositions: Sell {ticker} for ${price} (Mkt: {tradeAtMarket})")
					for _ in range(abs(orders)): 
						self.PlaceSell(ticker=ticker, price=price, marketOrder=tradeAtMarket, expireAfterDays=expireAfterDays, verbose=verbose)				
				elif orders > 0 and self.TranchesAvailable() > 0: # --- EXECUTE BUYS ---
					price = target_buy if not tradeAtMarket else sn.Average
					if rateLimitTransactions: orders = 1					
					if verbose: print(f" AlignPositions: Buy {ticker} for ${price} (Mkt: {tradeAtMarket})")
					for _ in range(orders):
						self.PlaceBuy(ticker=ticker, price=price, marketOrder=tradeAtMarket, expireAfterDays=expireAfterDays, verbose=verbose)								
				elif trimProfitsPercent > 0 and row['CurrentHoldings'] > 0: # --- TRIM PROFITS (Only if no other orders pending for this ticker) ---
					print(f"Sell for trim profits {ticker} for ${trim_sell}")
					self.PlaceSell(ticker=ticker, price=trim_sell, marketOrder=False, expireAfterDays=expireAfterDays, verbose=verbose)
		if verbose: print(self.PositionSummary())	

	def CancelAllOrders(self): super(TradingModel, self).CancelAllOrders(self.currentDate)
	
	def _recordTradeModelComparisonToSQL(self, params: TradeModelParams):
		flat_data = params.to_sql_dict()    	
		columns = ", ".join(flat_data.keys())
		placeholders = ", ".join([f":{key}" for key in flat_data.keys()])   
		sql = f"INSERT INTO TradeModelComparisons ({columns}) VALUES ({placeholders})"		
		db = PTADatabase()
		if db.Open():
			db.ExecSQL(sql, params=flat_data)
			db.Close()	

	def _recordTradeModelComparisonToCSV(self, params: TradeModelParams, perf: TradeModelPerformanceMetrics, csv_filename="TradeModelComparisons.csv"):
		flat_params = params.to_flat_dict()
		flat_perf = perf.to_flat_dict()
		combined_data = flat_params | flat_perf
		df = pd.DataFrame([combined_data])
		csvFile = os.path.join(self._dataFolderTradeModel, csv_filename)
		file_exists = os.path.isfile(csvFile)
		df.to_csv(csvFile, mode='a', index=False, header=not file_exists)
		print(f"Results appended to {csvFile}")

	def _recordResults(self, params: TradeModelParams):
		df = self.dailyValue.copy()
		params.startValue = df['TotalValue'].iloc[0]
		params.endValue = df['TotalValue'].iloc[-1]
		params.modelName = self.modelName
		params.batchName = self.batchName
		params.processing_minutes = int((self.end_processing - self.start_processing).seconds /60)
		perf = analyze_portfolio_performance(df)
		perf.batchName = self.batchName		
		perf.startDate = df.index.min()
		perf.endDate = df.index.max()
		perf.startValue = df['TotalValue'].iloc[0]
		perf.endValue = df['TotalValue'].iloc[-1]
		perf.modelName = self.modelName
		perf.batchName = self.batchName       
		days = (perf.endDate - perf.startDate).days
		perf.durationInYears = int(round(days / CONSTANTS.CALENDAR_YEAR))		
		if params.use_sql:
			self._recordTradeModelComparisonToSQL(params)
			UpdateTradeModelComparisonsFromDailyValue(self.batchName)
		else:
			self._recordTradeModelComparisonToCSV(params, perf)	
			
	def CloseModel(self, params: TradeModelParams = None):	
		if not params: params = TradeModelParams()
		cashValue, assetValue = self.Value()
		self.CancelAllOrders()
		if assetValue > 0:
			self.SellAllPositions(self.currentDate, allowWeekEnd=True)
		self.UpdateDailyValue()
		cashValue, assetValue = self.Value()
		netChange = cashValue + assetValue - self.startingValue 
		if self.pbar: self.pbar.close()
		self.end_processing = datetime.today()
		params.processing_minutes = int((self.end_processing - self.start_processing).seconds /60)
		print('Model ' + self.modelName + ' from ' + str(self.modelStartDate)[:10] + ' to ' + str(self.modelEndDate)[:10])
		print('Cash: ' + str(round(cashValue)) + ' asset: ' + str(round(assetValue)) + ' total: ' + str(round(cashValue + assetValue)))
		print('Net change: ' + str(round(netChange)), str(round((netChange/self.startingValue) * 100, 2)) + '%')
		print(f"Processing time: {params.processing_minutes} minutes")
		print('')
		if params.saveTradeHistory:
			self._recordResults(params)
			self.SaveDailyValue(self._dataFolderTradeModel)
			self.SaveTradeHistory(self._dataFolderTradeModel)
		return cashValue + assetValue
		
	def CalculateGain(self, startDate:datetime, endDate:datetime):
		try:
			startValue = self.dailyValue['TotalValue'].at[startDate]
			endValue = self.dailyValue['TotalValue'].at[endDate]
			gain = endValue = startValue
			percentageGain = endValue/startValue
		except:
			gain = -1
			percentageGain = -1
			print('Unable to calculate gain for ', startDate, endDate)
		return gain, percentageGain
			
	def GetCustomValues(self): return self.Custom1, self.Custom2
		
	def GetDailyValue(self): 
		return self.dailyValue.copy() #returns dataframe with daily value of portfolio

	def GetValueAt(self, date): 
		try:
			i = self.dailyValue.index.get_indexer([date], method='nearest')[0]
			if i > -1: r = self.dailyValue.iloc[i]['TotalValue']
		except:
			print('Unable to return value at ', date)
			r=-1
		return r

	def GetPrice(self, ticker:str=''): 
		#returns snapshot object of yesterday's pricing info to help make decisions today
		forDate = self.currentDate + timedelta(days=-1)
		r = None
		if ticker =='':
			r = self.priceHistory[0].GetPrice(forDate)
		else:
			if not ticker in self._tickerList:	self.AddTicker(ticker)
			if ticker in self._tickerList:
				for ph in self.priceHistory:
					if ph.ticker == ticker: r = ph.GetPrice(forDate) 
		return r

	def GetPriceSnapshot(self, ticker:str=''): 
		#returns snapshot object of yesterday's pricing info to help make decisions today
		forDate = self.currentDate + timedelta(days=-1)
		r = None
		if ticker =='':
			r = self.priceHistory[0].GetPriceSnapshot(forDate)
		else:
			if not ticker in self._tickerList:	self.AddTicker(ticker)
			if ticker in self._tickerList:
				for ph in self.priceHistory:
					if ph.ticker == ticker: r = ph.GetPriceSnapshot(forDate) 
		return r

	def ModelCompleted(self) -> bool:
		if not self.modelReady or self.currentDate is None or self.modelEndDate is None:
			print(f" TradeModel: Warning model stop triggered by invalid state. Ready: {self.modelReady}, Date: {self.currentDate}, End: {self.modelEndDate}")
			return True
		is_completed = self.currentDate >= self.modelEndDate		
		if is_completed:
			print(f" TradeModel: Backtest successfully reached end date: {self.modelEndDate}")		
		return is_completed				

	def NormalizePrices(self):
		self._NormalizePrices =  not self._NormalizePrices
		for p in self.priceHistory:
			if not p.pricesNormalized: p.NormalizePrices()
		
	def PlaceBuy(self, ticker:str, price:float, marketOrder:bool=False, expireAfterDays:bool=10, verbose:bool=False):
		if not ticker in self._tickerList: self.AddTicker(ticker)	
		if ticker in self._tickerList:	
			if marketOrder or price ==0: price = self.GetPrice(ticker)
			super(TradingModel, self).PlaceBuy(ticker, price, self.currentDate, marketOrder, expireAfterDays, verbose)
		else:
			print(' Unable to add ticker ' + ticker + ' to portfolio.')

	def PlaceSell(self, ticker:str, price:float, marketOrder:bool=False, expireAfterDays:bool=10, datepurchased:datetime=None, verbose:bool=False): 
		if marketOrder or price ==0: price = self.GetPrice(ticker)
		super(TradingModel, self).PlaceSell(ticker=ticker, price=price, datePlaced=self.currentDate, marketOrder=marketOrder, expireAfterDays=expireAfterDays, verbose=verbose)

	def PlotTradeHistoryAgainstHistoricalPrices(self, tradeHist:pd.DataFrame, priceHist:pd.DataFrame, modelName:str):
		buys = tradeHist.loc[:,['dateBuyOrderFilled','purchasePrice']]
		buys = buys.rename(columns={'dateBuyOrderFilled':'Date'})
		buys.set_index(['Date'], inplace=True)
		sells  = tradeHist.loc[:,['dateSellOrderFilled','sellPrice']]
		sells = sells.rename(columns={'dateSellOrderFilled':'Date'})
		sells.set_index(['Date'], inplace=True)
		dfTemp = priceHist.loc[:,['High','Low', 'Channel_High', 'Channel_Low']]
		dfTemp = dfTemp.join(buys)
		dfTemp = dfTemp.join(sells)
		PlotDataFrame(dfTemp, modelName, 'Date', 'Value')

	def ProcessDay(self, withIncrement: bool = True, allowWeekEnd: bool = False):
		#Process current day and increment the current date, allowWeekEnd is for model closing only
		if self.currentDate.weekday() < 5 or allowWeekEnd:
			active_tickers = self._ActiveTickers()
			for ticker in active_tickers:
				ph = self._priceMap.get(ticker)
				if ph is None:
					continue 
				sn = ph.GetPriceSnapshot(self.currentDate)
				for t in self._tranches:
					if t.ticker != ticker or not t.purchased:
						continue
					if sn.Close == 0:# Delisted: forced liquidation
						if t.sellOrderPrice == 0:
							print(f"ProcessDay: Forcing sell of {ticker} due to delisting.")
							forced_price = round(t.latestPrice * 0.9, 3)
							sn.Open = forced_price, sn.Close = forced_price, sn.High = forced_price, sn.Low = forced_price
							t.PlaceSell(price=forced_price, datePlaced=self.currentDate, marketOrder=True, expireAfterDays=5, verbose=self.verbose)
					else:						
						t.latestPrice = sn.Close # Normal price update
				self.ProcessDaysOrders(ticker, sn.Open, sn.High, sn.Low, sn.Close, self.currentDate)
		self.UpdateDailyValue()
		if self.pbar:
			c = self._cash
			a = self.assetValue
			self.pbar.set_description(f"Model: {self.modelName} from {self.modelStartDate} to {self.modelEndDate} currentDate: {self.currentDate} Cash: ${int(c)} Assets: ${int(a)} Total: ${int(c+a)} Return: {round(100*(((c+a)/self._initialValue)-1), 2)}%")
		self.ReEvaluateTrancheCount()
		if withIncrement:
			idx = self.priceHistory[0].historicalPrices.index		
			pos = idx.searchsorted(self.currentDate, side='right')				
			if pos < len(idx):
				next_date = idx[pos]
				while next_date <= self.currentDate and pos < len(idx) - 1:
					pos += 1
					next_date = idx[pos]					
				self.currentDate = next_date
			else:
				self.currentDate = self.modelEndDate + pd.Timedelta(days=1)
			if self.pbar: self.pbar.update(1)
	
	def SetCustomValues(self, v1, v2):
		self.Custom1 = v1
		self.custom2 = v2
		
class ForcastModel():	#used to forecast the effect of a series of trade actions, one per day, and return the net change in value.  This will mirror the given model.  Can also be used to test alternate past actions 
	def __init__(self, mirroredModel:TradingModel, daysToForecast:int = 10):
		modelName = 'Forcaster for ' + mirroredModel.modelName
		self.daysToForecast = daysToForecast
		self.daysToForecast = daysToForecast
		self.startDate = mirroredModel.modelStartDate 
		durationInYears = (mirroredModel.modelEndDate-mirroredModel.modelStartDate).days/CONSTANTS.CALENDAR_YEAR
		self.tm = TradingModel(modelName=modelName, startingTicker=mirroredModel._tickerList[0], startDate=mirroredModel.modelStartDate, durationInYears=durationInYears, totalFunds=mirroredModel.startingValue, verbose=False, trackHistory=False)
		self.savedModel = TradingModel(modelName=modelName, startingTicker=mirroredModel._tickerList[0], startDate=mirroredModel.modelStartDate, durationInYears=durationInYears, totalFunds=mirroredModel.startingValue, verbose=False, trackHistory=False)
		self.mirroredModel = mirroredModel
		self.tm._tickerList = mirroredModel._tickerList
		self.tm.priceHistory = mirroredModel.priceHistory
		self.savedModel._tickerList = mirroredModel._tickerList
		self.savedModel.priceHistory = mirroredModel.priceHistory

	def Reset(self, updateSavedModel:bool=True):
		if updateSavedModel:
			c, a = self.mirroredModel.Value()
			self.savedModel.currentDate = self.mirroredModel.currentDate
			self.savedModel._cash=self.mirroredModel._cash
			self.savedModel._fundsCommittedToOrders=self.mirroredModel._fundsCommittedToOrders
			self.savedModel.dailyValue = pd.DataFrame([[self.mirroredModel.currentDate,c,a,c+a]], columns=list(['Date','CashValue','AssetValue','TotalValue']))
			self.savedModel.dailyValue.set_index(['Date'], inplace=True)

			if len(self.savedModel._tranches) != len(self.mirroredModel._tranches):
				#print(len(self.savedModel._tranches), len(self.mirroredModel._tranches))
				trancheSize = self.mirroredModel._tranches[0].size
				tc = len(self.mirroredModel._tranches)
				while len(self.savedModel._tranches) < tc:
					self.savedModel._tranches.append(Tranche(trancheSize))
				while len(self.savedModel._tranches) > tc:
					self.savedModel._tranches.pop(-1)
				self.savedModel._trancheCount = len(self.savedModel._tranches)			
			for i in range(len(self.savedModel._tranches)):
				self.savedModel._tranches[i].ticker = self.mirroredModel._tranches[i].ticker
				self.savedModel._tranches[i].available = self.mirroredModel._tranches[i].available
				self.savedModel._tranches[i].size = self.mirroredModel._tranches[i].size
				self.savedModel._tranches[i].units = self.mirroredModel._tranches[i].units
				self.savedModel._tranches[i].purchased = self.mirroredModel._tranches[i].purchased
				self.savedModel._tranches[i].marketOrder = self.mirroredModel._tranches[i].marketOrder
				self.savedModel._tranches[i].sold = self.mirroredModel._tranches[i].sold
				self.savedModel._tranches[i].dateBuyOrderPlaced = self.mirroredModel._tranches[i].dateBuyOrderPlaced
				self.savedModel._tranches[i].dateBuyOrderFilled = self.mirroredModel._tranches[i].dateBuyOrderFilled
				self.savedModel._tranches[i].dateSellOrderPlaced = self.mirroredModel._tranches[i].dateSellOrderPlaced
				self.savedModel._tranches[i].dateSellOrderFilled = self.mirroredModel._tranches[i].dateSellOrderFilled
				self.savedModel._tranches[i].buyOrderPrice = self.mirroredModel._tranches[i].buyOrderPrice
				self.savedModel._tranches[i].purchasePrice = self.mirroredModel._tranches[i].purchasePrice
				self.savedModel._tranches[i].sellOrderPrice = self.mirroredModel._tranches[i].sellOrderPrice
				self.savedModel._tranches[i].sellPrice = self.mirroredModel._tranches[i].sellPrice
				self.savedModel._tranches[i].latestPrice = self.mirroredModel._tranches[i].latestPrice
				self.savedModel._tranches[i].expireAfterDays = self.mirroredModel._tranches[i].expireAfterDays
		c, a = self.savedModel.Value()
		self.startingValue = c + a
		self.tm.currentDate = self.savedModel.currentDate
		self.tm._cash=self.savedModel._cash
		self.tm._fundsCommittedToOrders=self.savedModel._fundsCommittedToOrders
		self.tm.dailyValue = pd.DataFrame([[self.savedModel.currentDate,c,a,c+a]], columns=list(['Date','CashValue','AssetValue','TotalValue']))
		self.tm.dailyValue.set_index(['Date'], inplace=True)
		if len(self.tm._tranches) != len(self.savedModel._tranches):
			#print(len(self.tm._tranches), len(self.savedModel._tranches))
			trancheSize = self.savedModel._tranches[0].size
			tc = len(self.savedModel._tranches)
			while len(self.tm._tranches) < tc:
				self.tm._tranches.append(Tranche(trancheSize))
			while len(self.tm._tranches) > tc:
				self.tm._tranches.pop(-1)
			self.tm._trancheCount = len(self.tm._tranches)			
		for i in range(len(self.tm._tranches)):
			self.tm._tranches[i].ticker = self.savedModel._tranches[i].ticker
			self.tm._tranches[i].available = self.savedModel._tranches[i].available
			self.tm._tranches[i].size = self.savedModel._tranches[i].size
			self.tm._tranches[i].units = self.savedModel._tranches[i].units
			self.tm._tranches[i].purchased = self.savedModel._tranches[i].purchased
			self.tm._tranches[i].marketOrder = self.savedModel._tranches[i].marketOrder
			self.tm._tranches[i].sold = self.savedModel._tranches[i].sold
			self.tm._tranches[i].dateBuyOrderPlaced = self.savedModel._tranches[i].dateBuyOrderPlaced
			self.tm._tranches[i].dateBuyOrderFilled = self.savedModel._tranches[i].dateBuyOrderFilled
			self.tm._tranches[i].dateSellOrderPlaced = self.savedModel._tranches[i].dateSellOrderPlaced
			self.tm._tranches[i].dateSellOrderFilled = self.savedModel._tranches[i].dateSellOrderFilled
			self.tm._tranches[i].buyOrderPrice = self.savedModel._tranches[i].buyOrderPrice
			self.tm._tranches[i].purchasePrice = self.savedModel._tranches[i].purchasePrice
			self.tm._tranches[i].sellOrderPrice = self.savedModel._tranches[i].sellOrderPrice
			self.tm._tranches[i].sellPrice = self.savedModel._tranches[i].sellPrice
			self.tm._tranches[i].latestPrice = self.savedModel._tranches[i].latestPrice
			self.tm._tranches[i].expireAfterDays = self.savedModel._tranches[i].expireAfterDays		
		c, a = self.tm.Value()
		if self.startingValue != c + a:
			print( 'Forcast model accounting error.  ', self.startingValue, self.mirroredModel.Value(), self.savedModel.Value(), self.tm.Value())
			assert(False)
			
	def GetResult(self):
		dayCounter = len(self.tm.dailyValue)
		while dayCounter <= self.daysToForecast:  
			self.tm.ProcessDay()
			dayCounter +=1
		c, a = self.tm.Value()
		endingValue = c + a
		return endingValue - self.startingValue
		

class ExtensiveTesting:
	"""
	SQL-backed FIFO queue for TradeModelParams used in extensive model testing.
	"""

	QUEUE_TABLE = "dbo.TradeModelQueue"

	def __init__(self, verbose: bool = False):
		self.db = PTADatabase()
		if not self.db.Open():
			print(" ExtensiveTesting: database initialization failed.")
			assert False
		self.verbose = verbose

	# ---------- Table Setup ----------

	def ensure_queue_table(self):
		sql = f"""
		IF OBJECT_ID('{self.QUEUE_TABLE}', 'U') IS NULL
		BEGIN
			CREATE TABLE {self.QUEUE_TABLE} (
				QueueID     INT IDENTITY(1,1) PRIMARY KEY,
				EnqueuedAt  DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
				ParamsJson  NVARCHAR(MAX) NOT NULL,
				Processing  BIT NOT NULL DEFAULT 0
			)
		END
		"""
		self.db.ExecSQL(sql)
		if self.verbose:
			print("ExtensiveTesting: Queue table verified")

	# ---------- Serialization ----------



	@staticmethod
	def _serialize(params: TradeModelParams) -> str:
		data = {}
		for f in fields(params):
			if not f.init:
				continue  # skip _startDate and any future non-init fields
			value = getattr(params, f.name)
			if isinstance(value, pd.Timestamp):
				value = value.isoformat()
			data[f.name] = value
			data["startDate"] = params.startDate.isoformat()
		return json.dumps(data)


	@staticmethod
	def _deserialize(json_str: str) -> TradeModelParams:
		data = json.loads(json_str)
		start_date = data.pop("startDate", None)
		params = TradeModelParams(**data)
		if start_date is not None:
			params.startDate = start_date  # uses setter → pd.Timestamp
		return params

	# ---------- Queue Operations ----------

	def add_to_queue(self, params: TradeModelParams):
		sql = f"""
			INSERT INTO {self.QUEUE_TABLE} (ParamsJson)
			VALUES (:params)
		"""
		self.db.ExecSQL(sql, {"params": self._serialize(params)})
		if self.verbose:
			print(f"ExtensiveTesting: Enqueued {params.modelName}")

	def pop_from_queue(self) -> TradeModelParams | None:
		"""
		Atomically dequeue the oldest unprocessed TradeModelParams.
		Safe for multiple workers.
		"""
		sql = f"""
		;WITH cte AS (
			SELECT TOP (1) *
			FROM {self.QUEUE_TABLE} WITH (ROWLOCK, READPAST, UPDLOCK)
			WHERE Processing = 0
			ORDER BY QueueID
		)
		UPDATE cte
		SET Processing = 1
		OUTPUT inserted.QueueID, inserted.ParamsJson;
		"""

		df = self.db.DataFrameFromSQL(sql)
		if df.empty:
			return None

		queue_id = int(df.iloc[0]["QueueID"])
		params = self._deserialize(df.iloc[0]["ParamsJson"])
		
		# Remove after successful pop
		self.db.ExecSQL(
			f"DELETE FROM {self.QUEUE_TABLE} WHERE QueueID = :id",
			{"id": queue_id}
		)

		if self.verbose:
			print(f"ExtensiveTesting: Dequeued job {queue_id}")
		return params

	# ---------- Utilities ----------

	def queue_depth(self) -> int:
		sql = f"SELECT COUNT(*) FROM {self.QUEUE_TABLE} WHERE Processing = 0"
		return int(self.db.ScalarListFromSQL(sql)[0])

	def clear_queue(self):
		self.db.ExecSQL(f"DELETE FROM {self.QUEUE_TABLE}")
		if self.verbose:
			print("ExtensiveTesting: Queue cleared")
