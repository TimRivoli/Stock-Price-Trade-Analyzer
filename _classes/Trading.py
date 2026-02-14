import time, json, socket
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
	shopBuyPercent: float = 0.0
	shopSellPercent: float = 0.0
	trimProfitsPercent: float = 0.0
	allocateByPointValue: bool = True
	rateLimitTransactions: bool = False
	saveTradeHistory: bool = True
	useSQL: bool = True
	saveResults: bool = False
	verbose: bool = False
	
	batchName: str = ''
	processing_minutes: int = 0	
	export_fields = [
		'modelName', 'startDate', 'endDate', 'durationInYears', 'stockCount', 'reEvaluationInterval', 'SP500Only', 	'longHistory', 'shortHistory', 'minPercentGain', 'startValue', 'endValue',
		'shopBuyPercent', 'shopSellPercent', 'trimProfitsPercent', 'allocateByPointValue', 'filterOption', 'useSQL','filterByFundamentals', 'rateLimitTransactions','marketCapMin', 'marketCapMax', 'processing_minutes', 'batchName'
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
	def AddModelNameModifiers(self):
		mn = self.modelName
		if mn == '':
			if self.filterOption == 99:
				mn = f"PM_Blended"
				if self.useSQL: mn += "_SQL" 
			elif self.filterOption == 98:
				mn = f"AdaptiveConvex"
			else: 
				mn = f"PM_filter{self.filterOption}"
		if self.filterByFundamentals and not "_FF" in mn: mn += "_FF"
		if self.SP500Only and not "_SP500" in mn: mn += "_SP500"
		if self.allocateByPointValue and not "_PVAlloc" in mn: mn += "_PVAlloc"
		if self.rateLimitTransactions and not "_RateLimit" in mn:  mn += "_RateLimit"
		self.modelName = mn

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

def load_convex_weight_series(db, reEvaluationInterval: int) -> Optional[pd.Series]:
	sql = f"SELECT asOfDate, convex_weight FROM {CONSTANTS.ADAPTIVE_CONVEX_STATE_TABLE} WHERE reevaluation_interval = :reevaluation_interval ORDER BY asOfDate"
	df = db.DataFrameFromSQL(sql, params={"reevaluation_interval": reEvaluationInterval}, indexName="asOfDate")
	if df is None or df.empty:
		return None
	df.index = pd.to_datetime(df.index)
	print(f" load_convex_weight_series for reEvaluationInterval {reEvaluationInterval} records {len(df)}")
	return df["convex_weight"].astype(float)

def analyze_portfolio_performance(df: pd.DataFrame, risk_free_rate: float = 0.0, convex_weight_series: Optional[pd.Series] = None) -> TradeModelPerformanceMetrics:
	df = df.copy()
	startDate = df.index.min()
	endDate = df.index.max()
	startValue = df['TotalValue'].iloc[0]
	endValue = df['TotalValue'].iloc[-1]
	days = (endDate - startDate).days
	durationInYears = int(round(days / CONSTANTS.CALENDAR_YEAR))

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
		startDate = startDate,
		endDate = endDate,
		startValue = startValue,
		endValue = endValue,
		durationInYears = durationInYears,
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

def UpdateTradeModelComparisonsFromDailyValue(batchName: str, reEvaluationInterval:int, metrics: TradeModelPerformanceMetrics=None):
	db = PTADatabase()
	if not db.Open():
		return
	if metrics:
		update_params = metrics.to_sql_dict()
		update_params = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in update_params.items()}
		cols_to_update = [k for k in update_params.keys() if k not in ['modelName', 'batchName']]
		set_clause = ", ".join([f"{col} = :{col}" for col in cols_to_update])  
		sql_update = f"UPDATE TradeModelComparisons SET {set_clause} WHERE batchName = :batchName"
		db.ExecSQL(sql_update, params=update_params)
		return
	sql_models = "SELECT DISTINCT TradeModel FROM TradeModel_DailyValue WHERE BatchName = :batchName"
	model_names = db.ScalarListFromSQL(sql_models, params={"batchName": batchName}, column="TradeModel")
	for model_name in model_names:
		sql_dv = "SELECT Date, TotalValue FROM TradeModel_DailyValue WHERE BatchName = :batchName ORDER BY Date"
		df = db.DataFrameFromSQL(sql_dv, params={"batchName": batchName}, indexName="Date")        
		if df is None or df.empty or len(df) < 2:
			continue           
		df.index = pd.to_datetime(df.index)
		convex_weight_series = load_convex_weight_series(db, reEvaluationInterval)
		metrics = analyze_portfolio_performance(df=df, convex_weight_series=convex_weight_series)        
		metrics.modelName = model_name
		metrics.batchName = batchName       
		update_params = metrics.to_sql_dict()
		update_params = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in update_params.items()}
		cols_to_update = [k for k in update_params.keys() if k not in ['modelName', 'batchName']]
		set_clause = ", ".join([f"{col} = :{col}" for col in cols_to_update])  
		sql_update = f"UPDATE TradeModelComparisons SET {set_clause} WHERE batchName = :batchName"
		#print(batchName, sql_update)
		db.ExecSQL(sql_update, params=update_params)
	db.Close()

@dataclass
class BuyOrder:
	ticker: str
	datePlaced: datetime
	units: int
	orderPrice: float
	marketOrder: bool = False
	reserved_cash: float = 0.0
	expireAfterDays: int = 10
	fillPrice: float = 0.0
	def is_expired(self, currentDate: datetime) -> bool: return (currentDate - self.datePlaced).days > self.expireAfterDays
	def is_executable(self, price: float) -> bool:
		if price <= 0: return False
		return self.marketOrder or price <= self.orderPrice
	def execute(self, price: float, dateFilled: datetime):
		if price <= 0: 	return None
		if not (self.marketOrder or price <= self.orderPrice): return None
		return Position(
			ticker=self.ticker,
			units=self.units,
			dateBuyOrderPlaced=self.datePlaced,
			dateBuyOrderFilled=dateFilled,
			buyOrderPrice=self.orderPrice,
			purchasePrice=price,
			latestPrice=price
		)
	def cost_at_price(self, price: float, commission: float = 0.0) -> float: return self.units * price + commission
	def adjust_units_for_cash(self, price: float, availableCash: float, commission: float = 0.0):
		if price <= 0:
			self.units = 0
			return
		max_units = int((availableCash - commission) / price)
		self.units = max(0, max_units)

@dataclass
class Position:
    ticker: str
    units: int
    dateBuyOrderPlaced: datetime
    dateBuyOrderFilled: datetime
    buyOrderPrice: float
    purchasePrice: float
    latestPrice: float = 0.0

    # Sell order status
    dateSellOrderPlaced: Optional[datetime] = None
    dateSellOrderFilled: Optional[datetime] = None
    sellOrderPrice: float = 0.0
    sellPrice: float = 0.0
    expireAfterDays: int = 0
    marketOrder: bool = False

    def current_value(self) -> float: return self.units * self.latestPrice
    def is_sell_pending(self) -> bool: 
        return self.dateSellOrderPlaced is not None and self.dateSellOrderFilled is None
    def is_sell_expired(self, currentDate: datetime) -> bool:
        if not self.is_sell_pending(): return False
        if self.marketOrder: return False
        return (currentDate - self.dateSellOrderPlaced).days > self.expireAfterDays
    def expire_sell(self):
        self.dateSellOrderPlaced = None
        self.sellOrderPrice = 0.0
        self.marketOrder = False
        self.expireAfterDays = 0
    def is_sell_executable(self, price: float) -> bool:
        if not self.is_sell_pending(): return False
        if price <= 0: return False
        return self.marketOrder or price >= self.sellOrderPrice
    def execute_sell(self, price: float, dateFilled: datetime, commission: float = 0.0) -> float:
        if not self.is_sell_executable(price): return 0.0
        self.dateSellOrderFilled = dateFilled
        self.sellPrice = price
        proceeds = self.units * price - commission
        return proceeds

class Portfolio:
	def __init__(self, portfolioName:str, startDate:datetime, totalFunds:int=10000, trackHistory:bool=True, useDatabase:bool=True, verbose:bool=False):
		self.pbar = None
		self.database = None
		self.dailyValue = None 
		self.tradeHistory = None 
		self._portfolioName = portfolioName
		self._initialValue = totalFunds
		self._commision_cost = 0
		self._asset_value = 0
		self._total_cash = totalFunds
		self._cash_committed_to_orders = 0
		self._pendingBuys: list[BuyOrder] = []
		self._positions: list[Position] = []
		self._verbose = verbose
		if useDatabase:
			db = PTADatabase()
			if db.database_configured:
				self.database = db
			else:
				useDatabase = False
		self.useDatabase = useDatabase
		self.trackHistory = trackHistory
		if trackHistory: 
			self.dailyValue = pd.DataFrame([[startDate,totalFunds,0,totalFunds,'','','','','','','','','','','']], columns=list(['Date','CashValue','AssetValue','TotalValue','Stock00','Stock01','Stock02','Stock03','Stock04','Stock05','Stock06','Stock07','Stock08','Stock09','Stock10']))
			self.dailyValue.set_index(['Date'], inplace=True)
			self.tradeHistory = pd.DataFrame(columns=['dateBuyOrderPlaced','ticker','dateBuyOrderFilled','dateSellOrderPlaced','dateSellOrderFilled','units','buyOrderPrice','purchasePrice','sellOrderPrice','sellPrice','NetChange'])
			self.tradeHistory.set_index(['dateBuyOrderPlaced','ticker'], inplace=True)

	#----------------------  Internal Utilities ---------------------------------------
	def _calculate_asset_value(self):
		assetValue=0
		for pos in self._positions:
			assetValue += (pos.units * pos.latestPrice)
		self._asset_value = assetValue

	def _cancel_order(self, order: BuyOrder):
		self._print(f" Buy order for {order.ticker} from {order.datePlaced} has been canceled.")
		self._cash_committed_to_orders -= order.reserved_cash
		self._pendingBuys.remove(order)
	
	def _check_orders(self, ticker: str, price: float, dateChecked: datetime):
		price = round(price, 4)
		if price <= 0: 	return

		# ---------------- BUY ORDERS ----------------
		for order in list(self._pendingBuys):
			if order.ticker != ticker:
				continue
			committed_this_order = order.reserved_cash
			if order.is_expired(dateChecked):
				self._print(f" Buy order from {order.datePlaced} has expired on {dateChecked}. ticker: {order.ticker}")
				self._cancel_order(order)
				continue
			if not order.is_executable(price):
				continue

			availableCash = (self._total_cash - self._cash_committed_to_orders) + committed_this_order
			actualCost = order.cost_at_price(price, self._commision_cost)			
			if actualCost > availableCash and order.marketOrder: # Market orders may need to shrink units to avoid negative cash
				self._print(f" CheckOrders: Adjusting market buy units for {ticker} at {price}")
				order.adjust_units_for_cash(price, availableCash, self._commision_cost)
				actualCost = order.cost_at_price(price, self._commision_cost)

			if order.units <= 0 or actualCost > availableCash:
				self._print(f" CheckOrders: Cannot afford any {ticker} at {price}... canceling Buy on {dateChecked}")
				self._cash_committed_to_orders -= committed_this_order
				self._pendingBuys.remove(order)
				continue

			# Execute Buy -> returns a Position
			pos = order.execute(price, dateChecked)
			if pos is not None:
				self._print(f" CheckOrders: {ticker} purchased for {price} on {dateChecked}")
				self._cash_committed_to_orders -= committed_this_order
				self._total_cash -= actualCost

				if self._commision_cost > 0: self._print(' _check_orders: Commission charged for Buy: ' + str(self._commision_cost))
				self._positions.append(pos)
				self._pendingBuys.remove(order)

				if self.trackHistory:
					idx = (pos.dateBuyOrderPlaced, pos.ticker)
					self.tradeHistory.loc[idx, 'dateBuyOrderFilled'] = pos.dateBuyOrderFilled
					self.tradeHistory.loc[idx, 'dateSellOrderPlaced'] = pos.dateSellOrderPlaced
					self.tradeHistory.loc[idx, 'dateSellOrderFilled'] = pos.dateSellOrderFilled
					self.tradeHistory.loc[idx, 'units'] = pos.units
					self.tradeHistory.loc[idx, 'buyOrderPrice'] = pos.buyOrderPrice
					self.tradeHistory.loc[idx, 'purchasePrice'] = pos.purchasePrice
					self.tradeHistory.loc[idx, 'sellOrderPrice'] = pos.sellOrderPrice
					self.tradeHistory.loc[idx, 'sellPrice'] = pos.sellPrice

		# ---------------- POSITIONS / SELL ORDERS ----------------
		for pos in list(self._positions):
			if pos.ticker != ticker:
				continue
			pos.latestPrice = price
			if not pos.is_sell_pending(): continue
			if pos.is_sell_expired(dateChecked):
				self._print(f" Ticker: {pos.ticker} sell order from {pos.dateSellOrderPlaced} has expired on {dateChecked}")
				pos.expire_sell()
				continue
			if pos.is_sell_executable(price):
				proceeds = pos.execute_sell(price, dateChecked, self._commision_cost)
				if proceeds > 0:
					self._total_cash += proceeds
					net_profit = ((pos.sellPrice - pos.purchasePrice) * pos.units) - self._commision_cost * 2
					self._print(f" Ticker: {pos.ticker} sold for {pos.sellPrice} on {dateChecked} net profit ${net_profit})")
					if self._commision_cost > 0: self._print(' _check_orders: Commission charged for Sell: ' + str(self._commision_cost))
					if self.trackHistory:
						self.tradeHistory.loc[(pos.dateBuyOrderPlaced, pos.ticker), :] = [pos.dateBuyOrderFilled,pos.dateSellOrderPlaced,pos.dateSellOrderFilled,pos.units,pos.buyOrderPrice,pos.purchasePrice,pos.sellOrderPrice,pos.sellPrice,net_profit]
					self._positions.remove(pos)
						
	def _check_price_sequence(self, ticker: str, p1: float, p2: float, dateChecked: datetime):
		if p1 == p2:
			self._check_orders(ticker, round(p1, 4), dateChecked)
			return
		steps = 40
		step = (p2 - p1) / steps
		prices = []
		for i in range(steps + 1): prices.append(round(p1 + i * step, 4))
		unique_prices = list(dict.fromkeys(prices))
		for p in unique_prices:
			self._check_orders(ticker, p, dateChecked)

	def _print(self, value:str):
		if self.pbar:
			tqdm.write(value)
		elif self._verbose:
			print(value)

	def _process_days_orders(self, ticker, open, high, low, close, dateChecked):
		# approximate a sequence of the day's prices for given ticker, check orders for each, update price value
		has_buy_orders = any(o.ticker == ticker for o in self._pendingBuys)
		has_sell_orders = any(p.ticker == ticker and p.dateSellOrderPlaced is not None and p.dateSellOrderFilled is None for p in self._positions)
		if has_buy_orders or has_sell_orders:
			p2 = low
			p3 = high
			if (high - open) < (open - low):
				p2 = high
				p3 = low
			self._check_price_sequence(ticker, open, p2, dateChecked)
			self._check_price_sequence(ticker, p2, p3, dateChecked)
			self._check_price_sequence(ticker, p3, close, dateChecked)

		close = round(close, 4) #Update latest price of each position
		for pos in self._positions:
			if pos.ticker == ticker:
				pos.latestPrice = close

	def _update_daily_value(self):
		self._calculate_asset_value()
		if self.trackHistory:
			total_value = self._total_cash + self._asset_value
			# Build value allocation per ticker
			value_map = {}
			for pos in self._positions:
				value_map[pos.ticker] = value_map.get(pos.ticker, 0.0) + (pos.units * pos.latestPrice)
			if len(value_map) > 0:
				positions = pd.DataFrame.from_dict(value_map, orient='index', columns=['Value'])
				positions.index.name = 'Ticker'
				positions['Percentage'] = positions['Value'] / positions['Value'].sum()
				positions.sort_values(by='Percentage', ascending=False, inplace=True)
			else:
				positions = pd.DataFrame(columns=['Value', 'Percentage'])
				positions.index.name = 'Ticker'
			x = (positions.index.astype(str) + ':' + positions['Percentage'].astype(str))
			x = x.str[:12].tolist()
			padding_needed = max(0, 11 - len(x))
			x += [''] * padding_needed
			new_row = [self._total_cash, self._asset_value, total_value] + x[:11]
			self.dailyValue.loc[self.currentDate] = new_row
			
	#----------------------  Status and position info  ---------------------------------------

	def GetAvailableCash(self): return (self._total_cash - self._cash_committed_to_orders)

	def GetCashValue(self): return self._total_cash

	def GetAssetValue(self): return self._asset_value	

	def GetCommittedCash(self): return self._cash_committed_to_orders	

	def GetBuyOrders(self): return self._pendingBuys
	
	def GetPositions(self): return self._positions
	
	def GetPositionSummary(self):
		available_cash = (self._total_cash - self._cash_committed_to_orders)
		buy_pending_count = len(self._pendingBuys)
		sell_pending_count = 0
		long_position_count = 0
		for pos in self._positions:
			if pos.is_sell_pending():
				sell_pending_count += 1
			else:
				long_position_count += 1
		return available_cash, buy_pending_count, sell_pending_count, long_position_count

	def GetSellOrders(self): 
		result = []
		for pos in self._positions:
			if pos.is_sell_pending():
				result +- pos
		return result

	def GetValue(self): return self._total_cash, self._asset_value

	def PrintPositions(self):
		print("----- OPEN POSITIONS -----")
		for i, pos in enumerate(self._positions):
			print(f"Pos {i}: {pos.ticker} units={pos.units} purchase={pos.purchasePrice} latest={pos.latestPrice}")
			if pos.dateSellOrderPlaced:
				print(f"   SellPlaced={pos.dateSellOrderPlaced} SellOrderPrice={pos.sellOrderPrice}")
		print("\n----- PENDING BUYS -----")
		for i, order in enumerate(self._pendingBuys):
			print(f"Order {i}: {order.ticker} units={order.units} limit={order.orderPrice} placed={order.datePlaced}")
		print("\nFunds committed to orders:", self._cash_committed_to_orders)
		print("Available funds:", self._total_cash - self._cash_committed_to_orders)

	#--------------------------------------  Order interface  ---------------------------------------
	def CancelAllOrders(self, currentDate: datetime = None):
		# Cancel all pending buy orders (release committed funds)
		for order in list(self._pendingBuys):
			self._cash_committed_to_orders -= order.reserved_cash
			self._pendingBuys.remove(order)

		# Cancel all pending sells (positions remain open)
		for pos in self._positions:
			if pos.dateSellOrderPlaced is not None and pos.dateSellOrderFilled is None:
				pos.sellOrderPrice = 0
				pos.dateSellOrderPlaced = None
				pos.marketOrder = False
				pos.expireAfterDays = 0

	def PlaceBuy(self, ticker: str, price: float, units: int, datePlaced: datetime, marketOrder: bool = False, expireAfterDays: int = 10, verbose: bool = False):
		if price <= 0 or price is None:
			self._print(f" PlaceBuy: Invalid price ${price} for {ticker}")
			return False
		price = round(price, 4)
		if units <= 0: return False
		cash_required = units * price + self._commision_cost
		available = (self._total_cash - self._cash_committed_to_orders)
		if available < cash_required:
			self._print(f" PlaceBuy: Insufficient funds available {available} to place buy for {ticker} units={units} price={price}")
			return False
		order = BuyOrder(ticker=ticker,	datePlaced=datePlaced, units=units, orderPrice=price, marketOrder=marketOrder, expireAfterDays=expireAfterDays, reserved_cash=cash_required)
		self._pendingBuys.append(order)
		self._cash_committed_to_orders += cash_required
		if marketOrder:
			self._print(f" {datePlaced} Buy placed ticker: {ticker} price: Market units:{units}")
		else:
			self._print(f" {datePlaced} Buy placed ticker: {ticker} price: ${price} units:{units}")
		return True

	def SellPosition(self, position, price: float, datePlaced: datetime, marketOrder: bool = False, expireAfterDays: int = 10):
		if position is None: return False
		if position.dateSellOrderFilled is not None: return False
		if position.units <= 0:	return False
		position.sellOrderPrice = round(price, 4)
		position.dateSellOrderPlaced = datePlaced
		position.marketOrder = marketOrder
		position.expireAfterDays = expireAfterDays
		return True

	def SellPositionPartial(self, position, units: int, price: float, datePlaced: datetime, marketOrder: bool = False, expireAfterDays: int = 10):
		if position is None: return False
		if position.dateSellOrderFilled is not None: return False
		if position.units <= 0: return False
		if units <= 0: return False
		if units > position.units: units = position.units

		# Create a new position representing the portion being sold
		sell_pos = Position(ticker=position.ticker, units=units, dateBuyOrderPlaced=position.dateBuyOrderPlaced, dateBuyOrderFilled=position.dateBuyOrderFilled, buyOrderPrice=position.buyOrderPrice, purchasePrice=position.purchasePrice, latestPrice=position.latestPrice)
		position.units -= units
		sell_pos.sellOrderPrice = round(price, 3)
		sell_pos.dateSellOrderPlaced = datePlaced
		sell_pos.marketOrder = marketOrder
		sell_pos.expireAfterDays = expireAfterDays
		self._positions.append(sell_pos)
		if position.units == 0: self._positions.remove(position)
		return True

	def PlaceSell(self, ticker: str, units: int, price: float, datePlaced: datetime, marketOrder: bool = False, expireAfterDays: int = 10):
		if units <= 0: return False
		eligible = [p for p in self._positions if p.ticker == ticker and p.dateSellOrderPlaced is None and p.dateSellOrderFilled is None]
		eligible.sort(key=lambda p: p.dateBuyOrderFilled)
		units_to_sell = units
		placed_any = False
		for pos in eligible:
			if units_to_sell <= 0:
				break
			if pos.units <= units_to_sell:
				if self.SellPosition(pos, price, datePlaced, marketOrder, expireAfterDays):
					units_to_sell -= pos.units
					placed_any = True
			else:
				if self.SellPositionPartial(pos, units_to_sell, price, datePlaced, marketOrder, expireAfterDays):
					units_to_sell = 0
					placed_any = True
		return placed_any

	def SellAllPositions(self, datePlaced: datetime, ticker: str = '', allowWeekEnd: bool = False):
		for pos in list(self._positions):
			if ticker == '' or pos.ticker == ticker:
				self.SellPosition(position=pos, price=pos.latestPrice, datePlaced=datePlaced, marketOrder=True, expireAfterDays=5)

	#--------------------------------------  Closing Reporting ---------------------------------------
	def SaveTradeHistory(self, foldername:str, addTimeStamp:bool = False):
		if self.trackHistory:
			if self.useDatabase and self.database:
				if self.database.Open():
					df = self.tradeHistory.copy()
					df['TradeModel'] = self.modelName 
					df['BatchName'] = self.batchName 				
					self.database.DataFrameToSQL(df, 'TradeModel_Trades', indexAsColumn=True)
					self.database.Close()
			elif CreateFolder(foldername):
				filePath = foldername + self._portfolioName 
				if addTimeStamp: filePath += '_' + GetDateTimeStamp()
				filePath += '_trades.csv'
				self.tradeHistory.to_csv(filePath)

	def SaveDailyValue(self, foldername:str, addTimeStamp:bool = False):
		if self.useDatabase and self.database:
			if self.database.Open():
				df = self.dailyValue.copy()
				df['TradeModel'] = self.modelName 
				df['BatchName'] = self.batchName 				
				self.database.DataFrameToSQL(df, 'TradeModel_DailyValue', indexAsColumn=True)
				self.database.Close()
		elif CreateFolder(foldername):
			filePath = foldername + self._portfolioName 
			if addTimeStamp: filePath += '_' + GetDateTimeStamp()
			filePath+= '_dailyvalue.csv'
			self.dailyValue.to_csv(filePath)
		
class TradingModel(Portfolio):
	#Extends Portfolio to trading environment for testing models
	def __init__(self, modelName:str, startingTicker:str, startDate:datetime, durationInYears:int, totalFunds:int, trackHistory:bool=True, useDatabase:bool=True, useFullStats:bool=True, verbose:bool=False):
		#pricesAsPercentages:bool=False would be good but often results in NaN values
		#expects date format in local format, from there everything will be converted to database format				
		startDate = ToDateTime(startDate)
		endDate = startDate + timedelta(days=CONSTANTS.CALENDAR_YEAR * durationInYears)
		super(TradingModel, self).__init__(portfolioName=modelName, startDate=startDate, totalFunds=totalFunds, trackHistory=trackHistory, useDatabase=useDatabase, verbose=verbose)	
		self.start_processing = datetime.today()
		self.end_processing = None
		self.modelReady = False
		self.modelName = modelName
		self.batchName = modelName[:30] + '_' + self.start_processing.strftime('%Y%m%d-%H%M%S')
		self.modelStartDate  = None	
		self.modelEndDate = None
		self.currentDate = None
		self.priceHistory = []  #list of price histories for each stock in _tickerList
		self._priceMap = {} #Mapping between tickers and priceHistory elements
		self._tickerList = []	#list of stocks currently held
		self.customValue1 = None	#can be used to store custom values when using the model
		self.customValue2 = None
		self._NormalizePrices = False
		self.useFullStats = useFullStats
		self.startingValue = totalFunds
		self._dataFolderTradeModel = 'data/trademodel/'
		CreateFolder(self._dataFolderTradeModel)
		total_days = len(pd.bdate_range(start=startDate, end=endDate))
		p = PricingData(startingTicker, useDatabase=self.useDatabase)
		if p.LoadHistory(requestedStartDate=startDate, requestedEndDate=endDate, verbose=verbose): 
			self._print(' TradingModel: Loading ' + startingTicker)
			p.CalculateStats(fullStats=self.useFullStats)
			valid_dates = p.historicalPrices.index[(p.historicalPrices.index >= startDate) & (p.historicalPrices.index <= endDate)]
			if not valid_dates.empty:
				self.priceHistory = [p] #add to list
				self._priceMap[p.ticker] = p
				self.modelStartDate = valid_dates[0]
				self.modelEndDate = valid_dates[-1]
				self.currentDate = self.modelStartDate
				self._tickerList = [startingTicker]
				self.pbar = tqdm(total=total_days, desc=f"Running {self.modelName} from {FormatDate(self.modelStartDate)} to {FormatDate(self.modelEndDate)}", unit="day")
				self.modelReady = len(p.historicalPrices) > 30
			else:
				self.modelReady = False #We don't have enough data

	#--------------------------------------  Internal Utilities  ---------------------------------------
	def _active_tickers(self):
		# Returns tickers with open positions or pending buy orders
		result = set()
		for o in self._pendingBuys:
			result.add(o.ticker)
		for p in self._positions:
			result.add(p.ticker)
		return result

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
		flat_params = params.to_sql_dict()
		flat_perf = perf.to_sql_dict()
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
		perf.modelName = self.modelName
		if self.useDatabase:
			self._recordTradeModelComparisonToSQL(params)
			UpdateTradeModelComparisonsFromDailyValue(perf.batchName, params.reEvaluationInterval, perf)
		else:
			self._recordTradeModelComparisonToCSV(params, perf)	

	#--------------------------------------  Interface for Tickers and BuySell Operations  ---------------------------------------
	def AddTicker(self, ticker:str):
		r = False
		if not ticker in self._tickerList:
			p = PricingData(ticker, useDatabase=self.useDatabase)
			if self._verbose: print(' Loading price history for ' + ticker)
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

	def GetPrice(self, ticker:str = None): 
		#returns snapshot object of yesterday's pricing info to help make decisions today
		forDate = self.currentDate + timedelta(days=-1)
		r = None
		if ticker:
			if not ticker in self._tickerList: self.AddTicker(ticker)
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

	def NormalizePrices(self):
		self._NormalizePrices =  not self._NormalizePrices
		for p in self.priceHistory:
			if not p.pricesNormalized: p.NormalizePrices()
		
	def PlaceBuy(self, ticker: str, price: float, units: int, marketOrder: bool = False, expireAfterDays: int = 10, verbose: bool = False):
		if not ticker in self._tickerList: self.AddTicker(ticker)	
		if ticker in self._tickerList:	
			if marketOrder or price ==0: 
				sn = self.GetPriceSnapshot(ticker)
				price = sn.Average
			super(TradingModel, self).PlaceBuy(ticker=ticker, price=price, units=units, datePlaced=self.currentDate, marketOrder=marketOrder, expireAfterDays=expireAfterDays, verbose=verbose)
		else:
			print(' Unable to add ticker ' + ticker + ' to portfolio.')

	def SellPosition(self, position, price: float, datePlaced: datetime, marketOrder: bool = False, expireAfterDays: int = 10):
		#datePlaced is not used in TradeModel				
		if not position.ticker in self._tickerList: self.AddTicker(position.ticker)	
		if position.ticker in self._tickerList:	
			if marketOrder or price ==0: 
				sn = self.GetPriceSnapshot(position.ticker)
				price = sn.Average
		super(TradingModel, self).SellPosition(position=position, price=price, datePlaced=self.currentDate, marketOrder=marketOrder, expireAfterDays=expireAfterDays)

	# def SellPositionPartial(self, position, units: int, price: float, marketOrder: bool = False, expireAfterDays: int = 10):
		# if marketOrder or price ==0: price = self.GetPrice(position.ticker)
		# super(TradingModel, self).SellPositionPartial(position=position, units=units, price=price, datePlaced=self.currentDate, marketOrder=marketOrder, expireAfterDays=expireAfterDays)

	def PlaceSell(self, ticker:str, units: int, price:float, marketOrder:bool=False, expireAfterDays:int=10): 
		if not ticker in self._tickerList: self.AddTicker(ticker)	
		if ticker in self._tickerList:	
			if marketOrder or price ==0: 
				sn = self.GetPriceSnapshot(ticker)
				price = sn.Average
		super(TradingModel, self).PlaceSell(ticker=ticker, units=units, price=price, datePlaced=self.currentDate, marketOrder=marketOrder, expireAfterDays=expireAfterDays)

	def SellAllPositions(self, ticker: str = '', allowWeekEnd: bool = False):
		for position in list(self._positions):
			if ticker == '' or position.ticker == ticker:
				super(TradingModel, self).SellPosition(position=position, price=position.latestPrice, datePlaced=self.currentDate, marketOrder=True)

	def AlignPositions(self, targetPositions: pd.DataFrame, rateLimitTransactions:bool = False, shopBuyPercent:float = 0.0, shopSellPercent:float = 0.0):
		assert isinstance(shopBuyPercent, float), f"Expected float, got {type(shopBuyPercent).__name__}"
		assert isinstance(shopSellPercent, float), f"Expected float, got {type(shopSellPercent).__name__}"
		expireAfterDays = 3
		tradeAtMarket = (shopBuyPercent == 0.0) and (shopSellPercent == 0.0)
		assetValue = sum(p.units * p.latestPrice for p in self._positions)
		totalValue = self._total_cash + assetValue
		if totalValue <= 0:
			print(f" Accounting error assertion: total value cannot be negative ${totalValue}")
			assert(False)
		if targetPositions is None or targetPositions.empty:
			targetPositions = pd.DataFrame({"TargetHoldings":[1.0]}, index=[CONSTANTS.CASH_TICKER])
		else: 
			targetPositions = targetPositions.copy()
		for position in self._positions: #add existing positions, if they were not in target then they need to be sold
			if position.ticker not in targetPositions.index:
				targetPositions.loc[position.ticker, "TargetHoldings"] = 0.0
		for order in list(self._pendingBuys):
			if order.ticker not in targetPositions.index:
				self._cancel_order(order)
		
		TotalTargets = targetPositions['TargetHoldings'].sum()
		if TotalTargets > 0: 
			targetPositions['Weight'] = targetPositions['TargetHoldings'] / TotalTargets
			targetPositions['TargetValue'] = targetPositions['Weight'] * totalValue
		else:
			targetPositions['TargetValue'] = targetPositions['TargetHoldings'] 
			
		current_value_map = {}
		for p in self._positions:
			current_value_map[p.ticker] = current_value_map.get(p.ticker, 0) + (p.units * p.latestPrice)
		for o in self._pendingBuys:
			current_value_map[o.ticker] = current_value_map.get(o.ticker, 0) + (o.units * o.orderPrice)

		targetPositions['CurrentValue'] = 0.0
		for ticker in targetPositions.index:
			targetPositions.loc[ticker, 'CurrentValue'] = current_value_map.get(ticker, 0.0)
		targetPositions['DeltaValue'] = targetPositions['TargetValue'] - targetPositions['CurrentValue']

		sells = targetPositions[targetPositions['DeltaValue'] < 0].copy()
		buys = targetPositions[targetPositions['DeltaValue'] > 0].copy()
		sells.sort_values(by=['DeltaValue'], ascending=True, inplace=True)
		buys.sort_values(by=['DeltaValue'], ascending=False, inplace=True)

		# ---------------- RATE LIMITING ----------------
		if rateLimitTransactions:
			daily_budget = totalValue * CONSTANTS.RATE_LIMITING_MAX_TURNOVER_PCT
			sell_budget = daily_budget
			buy_budget = daily_budget
			sell_needed = abs(sells['DeltaValue']).sum()
			if sell_needed > 0:
				sell_scale = min(1.0, sell_budget / sell_needed)
				sells.loc[:, 'DeltaValue'] *= sell_scale

			buy_needed = buys['DeltaValue'].sum()
			if buy_needed > 0:
				buy_scale = min(1.0, buy_budget / buy_needed)
				buys.loc[:, 'DeltaValue'] *= buy_scale
			self._print(f" AlignPositions: Rate limiting enabled. Total budget ${daily_budget:,.0f}")

		# ---------------- CASH LIMIT ON BUYS ----------------
		availableFunds = self._total_cash - self._cash_committed_to_orders
		buy_needed = buys['DeltaValue'].sum()
		if buy_needed > availableFunds and buy_needed > 0:
			cash_scale = availableFunds / buy_needed
			buys.loc[:, 'DeltaValue'] *= cash_scale
			self._print(f" AlignPositions: Buy scaling due to cash. Scale={cash_scale:.3f} available: {availableFunds} needed: {buy_needed}")

		# ---------------- EXECUTE SELLS ----------------
		for ticker, row in sells.iterrows():
			if ticker == CONSTANTS.CASH_TICKER:
				continue
			sn = self.GetPriceSnapshot(ticker)
			if sn is None or sn.Average <= 0:
				self._print(f" AlignPositions: Unable to get sell price for {ticker} on {self.currentDate}")
				continue
			target_sell = round(max(sn.Average, sn.Target) * (1 + shopSellPercent), 4)
			price = target_sell if not tradeAtMarket else sn.Average
			deltaValue = float(row['DeltaValue'])
			unitsToSell = int(abs(deltaValue) / price)
			if unitsToSell > 0:
				self._print(f" AlignPositions: Sell {ticker} units={unitsToSell} @ ${price} (Mkt: {tradeAtMarket})")
				self.PlaceSell(ticker=ticker, units=unitsToSell, price=price, marketOrder=tradeAtMarket, expireAfterDays=expireAfterDays)


		# ---------------- EXECUTE BUYS ----------------
		for ticker, row in buys.iterrows():
			if ticker == CONSTANTS.CASH_TICKER:
				continue
			sn = self.GetPriceSnapshot(ticker)
			if sn is None or sn.Average <= 0:
				self._print(f" AlignPositions: Unable to get buy price for {ticker} on {self.currentDate}")
				continue
			target_buy = round(min(sn.Average, sn.Target) * shopBuyPercent, 4)
			price = target_buy if not tradeAtMarket else sn.Average
			deltaValue = float(row['DeltaValue'])
			availableFunds = self._total_cash - self._cash_committed_to_orders
			maxUnits = int((availableFunds - self._commision_cost) / price)
			unitsToBuy = int(deltaValue / price)
			unitsToBuy = min(unitsToBuy, maxUnits)
			if unitsToBuy > 0:
				self._print(f" AlignPositions: Buy {ticker} units={unitsToBuy} @ ${price} (Mkt: {tradeAtMarket})")
				self.PlaceBuy(ticker=ticker, price=price, units=unitsToBuy, marketOrder=tradeAtMarket, expireAfterDays=expireAfterDays)

	def TrimProfits(self, trimProfitsPercent: float = 0.03, maxTrimPctPortfolio: float = 0.10, expireAfterDays: int = 3, verbose: bool = False):
		#If we gained trimProfitsPercent then sell up to maxTrimPctPortfolio
		assert isinstance(trimProfitsPercent, float), f"Expected float, got {type(trimProfitsPercent).__name__}"
		assert isinstance(maxTrimPctPortfolio, float), f"Expected float, got {type(maxTrimPctPortfolio).__name__}"
		if trimProfitsPercent <= 0: return
		assetValue = sum(p.units * p.latestPrice for p in self._positions)
		totalValue = self._total_cash + assetValue
		if totalValue <= 0: return
		maxTrimValue = totalValue * maxTrimPctPortfolio
		trimmedValue = 0.0

		# Only positions with profit >= threshold, and no sell order currently placed
		eligible = []
		for pos in self._positions:
			if pos.units <= 0: continue
			if pos.latestPrice is None or pos.latestPrice <= 0: continue
			if pos.purchasePrice <= 0: continue
			if pos.dateSellOrderPlaced is not None: continue
			gainPct = ((pos.latestPrice - pos.purchasePrice) / pos.purchasePrice) * 100
			if gainPct >= trimProfitsPercent:
				eligible.append(pos)
		# Trim highest gain first
		eligible.sort(key=lambda p: (p.latestPrice - p.purchasePrice) / p.purchasePrice, reverse=True)

		for pos in eligible:
			if trimmedValue >= maxTrimValue: break
			remainingTrimCapacity = maxTrimValue - trimmedValue
			unitsToSell = int(remainingTrimCapacity / pos.latestPrice)
			unitsToSell = max(unitsToSell, 1)
			unitsToSell = min(unitsToSell, pos.units)
			if unitsToSell <= 0: continue
			self._print(f" TrimProfits: Selling {unitsToSell} units of {pos.ticker} @ {pos.latestPrice} (purchase={pos.purchasePrice})")
			super(TradingModel, self).SellPositionPartial(position=pos, units=unitsToSell, price=pos.latestPrice, datePlaced=self.currentDate, marketOrder=True, expireAfterDays=expireAfterDays)
			trimmedValue += unitsToSell * pos.latestPrice

	#--------------------------------------  Interface for global operations  ---------------------------------------
	def CalculateGain(self, startDate:datetime, endDate:datetime):
		try:
			startValue = self.dailyValue['TotalValue'].at[startDate]
			endValue = self.dailyValue['TotalValue'].at[endDate]
			gain = endValue - startValue
			percentageGain = (endValue / startValue) - 1
		except:
			gain = -1
			percentageGain = -1
			print('Unable to calculate gain for ', startDate, endDate)
		return gain, percentageGain

	def CloseModel(self, params: TradeModelParams = None):	
		if not params: params = TradeModelParams()		
		self.CancelAllOrders()
		if  self._asset_value > 0:
			self.SellAllPositions(allowWeekEnd=True)
		self.ProcessDay(withIncrement=False, allowWeekEnd=True)
		self._update_daily_value()
		netChange = self._total_cash + self._asset_value - self.startingValue 
		if self.pbar: self.pbar.close()
		self.end_processing = datetime.today()
		params.processing_minutes = int((self.end_processing - self.start_processing).seconds /60)
		print('Model ' + self.modelName + ' from ' + str(self.modelStartDate)[:10] + ' to ' + str(self.modelEndDate)[:10])
		c = self._total_cash
		a = self._asset_value
		gain = 100 * (((c + a) / self._initialValue) - 1)
		print(f"Cash: ${c:,.0f} ' assets: ${a:,.0f} total: ${(c+a):,.0f} gain: {gain:,.0f}%")
		print(f"Processing time: {params.processing_minutes} minutes")
		print('')
		if params.saveTradeHistory:
			self.SaveDailyValue(self._dataFolderTradeModel)
			self.SaveTradeHistory(self._dataFolderTradeModel)
			self._recordResults(params)
		return self._total_cash + self._asset_value	
				
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

	def ModelCompleted(self) -> bool:
		if not self.modelReady or self.currentDate is None or self.modelEndDate is None:
			print(f" TradeModel: Warning model stop triggered by invalid state. Ready: {self.modelReady}, Date: {self.currentDate}, End: {self.modelEndDate}")
			return True
		is_completed = self.currentDate >= self.modelEndDate		
		if is_completed:
			print(f" TradeModel: Backtest successfully reached end date: {self.modelEndDate}")		
		return is_completed				

	def _HandleMissingPriceData(self, ticker: str):
		# Cancel pending buys
		for o in list(self._pendingBuys):
			if o.ticker == ticker:
				committed = o.reserved_cash
				self._cash_committed_to_orders -= committed
				self._pendingBuys.remove(o)
		# Force liquidate open positions
		for pos in list(self._positions):
			if pos.ticker == ticker:
				if pos.latestPrice is None or pos.latestPrice <= 0:
					continue
				if pos.dateSellOrderPlaced is None:
					forced_price = pos.latestPrice #Most likely got aquired
					print(f"ProcessDay: Missing price data for {ticker}. Forcing liquidation at {forced_price}")
					super(TradingModel, self).SellPosition(position=pos, price=forced_price, datePlaced=self.currentDate, marketOrder=True, expireAfterDays=5)

	def ProcessDay(self, withIncrement: bool = True, allowWeekEnd: bool = False):
		# Process current day and increment the current date, allowWeekEnd is for model closing only
		if self.currentDate.weekday() < 5 or allowWeekEnd:
			for ticker in self._active_tickers():
				ph = self._priceMap.get(ticker)
				sn = None
				if ph is not None: sn = ph.GetPriceSnapshot(self.currentDate)
				if ph is None or sn is None:
					try:
						self.AddTicker(ticker)
						ph = self._priceMap.get(ticker)
						sn = ph.GetPriceSnapshot(self.currentDate)
					except Exception as e:
						print(f"ProcessDay: Failed to load ticker {ticker}: {e}")
						self._HandleMissingPriceData(ticker)
						continue
				if sn.Close == 0:
					self._HandleMissingPriceData(ticker)
					continue

				self._process_days_orders(ticker, sn.Open, sn.High, sn.Low, sn.Close, self.currentDate)
				closePrice = round(sn.Close, 4)
				for pos in self._positions:
					if pos.ticker == ticker:
						pos.latestPrice = closePrice
		self._update_daily_value()
		if self.pbar:
			c = self._total_cash
			a = self._asset_value
			gain = 100 * (((c + a) / self._initialValue) - 1)
			self.pbar.set_description(f"Model: {self.modelName} from {FormatDate(self.modelStartDate)} to {FormatDate(self.modelEndDate)} currentDate: {FormatDate(self.currentDate)} Cash: ${c:,.0f} Assets: ${a:,.0f} Total: ${(c + a):,.0f} Return: {gain:,.0f}%")
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

	#--------------------------------------  Custom value get/set ---------------------------------------
	def GetCustomValues(self): return self.customValue1, self.customValue2

	def SetCustomValues(self, v1, v2):
		self.customValue1 = v1
		self.customValue2 = v2
	
	#--------------------------------------  Plotting/Reporting ---------------------------------------
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
			c, a = self.mirroredModel.GetValue()
			self.savedModel.currentDate = self.mirroredModel.currentDate
			self.savedModel._total_cash = self.mirroredModel._total_cash
			self.savedModel._cash_committed_to_orders=self.mirroredModel._cash_committed_to_orders
			self.savedModel.dailyValue = pd.DataFrame([[self.mirroredModel.currentDate,c,a,c+a]], columns=list(['Date','CashValue','AssetValue','TotalValue']))
			self.savedModel.dailyValue.set_index(['Date'], inplace=True)
		c, a = self.savedModel.GetValue()
		self.startingValue = c + a
		self.tm.currentDate = self.savedModel.currentDate
		self.tm._total_cash=self.savedModel._total_cash
		self.tm._cash_committed_to_orders=self.savedModel._cash_committed_to_orders
		self.tm.dailyValue = pd.DataFrame([[self.savedModel.currentDate,c,a,c+a]], columns=list(['Date','CashValue','AssetValue','TotalValue']))
		self.tm.dailyValue.set_index(['Date'], inplace=True)
		c, a = self.tm.GetValue()
		if self.startingValue != c + a:
			print( 'Forcast model accounting error.  ', self.startingValue, self.mirroredModel.GetValue(), self.savedModel.GetValue(), self.tm.GetValue())
			assert(False)
			
	def GetResult(self):
		dayCounter = len(self.tm.dailyValue)
		while dayCounter <= self.daysToForecast:  
			self.tm.ProcessDay()
			dayCounter +=1
		c, a = self.tm.GetValue()
		endingValue = c + a
		return endingValue - self.startingValue
		

class ExtensiveTesting:
	QUEUE_TABLE = "dbo.TradeModelQueue"
	def __init__(self, verbose: bool = False):
		self.db = PTADatabase()
		if not self.db.Open():
			print(" ExtensiveTesting: database initialization failed.")
			assert False
		self.worker_name = socket.gethostname()
		self._verbose = verbose

	def ensure_queue_table(self):
		sql = f"""
		IF OBJECT_ID('{self.QUEUE_TABLE}', 'U') IS NULL
		BEGIN
			CREATE TABLE {self.QUEUE_TABLE} (
				QueueID     INT IDENTITY(1,1) PRIMARY KEY,
				Processing  BIT NOT NULL DEFAULT 0,
				EnqueuedAt  DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
				ParamsJson  NVARCHAR(MAX) NOT NULL,
				WorkerName  NVARCHAR(MAX) NULL
			)
		END
		"""
		self.db.ExecSQL(sql)
		if self._verbose: print("ExtensiveTesting: Queue table verified")
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

	def add_to_queue(self, params: TradeModelParams):
		sql = f"INSERT INTO {self.QUEUE_TABLE} (ParamsJson) VALUES (:params)"
		self.db.ExecSQL(sql, {"params": self._serialize(params)})
		if self._verbose: print(f"ExtensiveTesting: Enqueued {params.modelName}")

	def claim_from_queue(self) -> tuple[int, TradeModelParams | None]:
		claim_sql = f"WITH NextJob AS (SELECT TOP (1) Processing, WorkerName, QueueID, ParamsJson FROM {self.QUEUE_TABLE} WITH (UPDLOCK, ROWLOCK, READPAST) WHERE Processing = 0 ORDER BY QueueID ASC) UPDATE NextJob SET Processing = 1, WorkerName = :worker OUTPUT INSERTED.QueueID, INSERTED.ParamsJson; "	
		rows = self.db.ExecSQL(claim_sql, {"worker": self.worker_name})	
		if not rows: return 0, None
		queue_id, params_json = rows[0][0], rows[0][1]
		if self._verbose: print(f"Worker {self.worker_name}: Successfully claimed job {queue_id}")			
		return queue_id, self._deserialize(params_json)

	def complete_job(self, queue_id: int):
		sql = f"DELETE FROM {self.QUEUE_TABLE} WHERE QueueID = :id"
		self.db.ExecSQL(sql, {"id": queue_id})		
		if self._verbose: print(f"ExtensiveTesting: Job {queue_id} deleted.")	

	def queue_depth(self) -> int:
		sql = f"SELECT COUNT(*) FROM {self.QUEUE_TABLE} WHERE Processing = 0"
		return int(self.db.ScalarListFromSQL(sql)[0])

	def clear_queue(self):
		self.db.ExecSQL(f"DELETE FROM {self.QUEUE_TABLE}")
		if self._verbose:
			print("ExtensiveTesting: Queue cleared")
