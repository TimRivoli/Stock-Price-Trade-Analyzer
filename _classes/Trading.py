import time, json, socket, re
import numpy as np, pandas as pd
import _classes.Constants as CONSTANTS
from tqdm import tqdm
from math import floor
from typing import Dict, Union, Optional
from dataclasses import dataclass, asdict, field, fields, is_dataclass
from datetime import datetime
from _classes.DataIO import PTADatabase
from _classes.Prices import PriceSnapshot, PricingData
from _classes.Utility import *

from _classes.Params import SQLFriendlyMixin, TradeModelParams, TradeModelPerformanceMetrics, SwingFeatures	#11/21/26: moved to Params.py; re-exported here for existing imports

@dataclass
class BuyOrder:
	position_id: int
	ticker: str
	datePlaced: datetime
	units: int
	orderPrice: float
	marketOrder: bool = False
	reserved_cash: float = 0.0
	expireAfterDays: int = 10
	fillPrice: float = 0.0
	tag: str = ''	#6.15: 'Swing' = a trade-around-core buy-back, 6.17: 'Pull' = a pullback buy (TradingModel.SwingTrade)
	unfunded: bool = False	#6.19: fills even when cash is short (a pullback limit buy, funded after it fills)
	def is_expired(self, currentDate: datetime) -> bool: return (currentDate - self.datePlaced).days > self.expireAfterDays
	def is_executable(self, price: float) -> bool:
		if price <= 0: return False
		return self.marketOrder or price <= self.orderPrice
	def execute(self, price: float, dateFilled: datetime):
		if price <= 0: 	return None
		if not (self.marketOrder or price <= self.orderPrice): return None
		return Position(
			position_id=self.position_id,
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
	position_id: int
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
	reasonForSale: str = ''
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
		self._current_position_id = 0
		self._commision_cost = 0
		self._asset_value = 0
		self._total_cash = totalFunds
		self._cash_committed_to_orders = 0
		self._pendingBuys: list[BuyOrder] = []
		self._positions: list[Position] = []
		self.rolloverPending = False	#set by TradingModel.AlignPositions when a realignment is cut short and must continue the next day
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
			self.tradeHistory = pd.DataFrame(columns=['position_id','dateBuyOrderPlaced','ticker','dateBuyOrderFilled','dateSellOrderPlaced','dateSellOrderFilled','units','buyOrderPrice','purchasePrice','sellOrderPrice','sellPrice','netChange','reasonForSale'])
			self.tradeHistory.set_index(['position_id'], inplace=True)

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
	
	def _save_position_to_history(self, pos: Position, netChange: float = None):
		if not self.trackHistory: return
		idx = pos.position_id
		row = {
			'ticker': pos.ticker,
			'dateBuyOrderPlaced': pos.dateBuyOrderPlaced,
			'dateBuyOrderFilled': pos.dateBuyOrderFilled,
			'buyOrderPrice': pos.buyOrderPrice,
			'purchasePrice': pos.purchasePrice,
			'units': pos.units,
			'dateSellOrderPlaced': pos.dateSellOrderPlaced,
			'dateSellOrderFilled': pos.dateSellOrderFilled,
			'sellOrderPrice': pos.sellOrderPrice,
			'sellPrice': pos.sellPrice,
			'reasonForSale': pos.reasonForSale,
			'netChange': netChange,
		}
		if idx not in self.tradeHistory.index:
			new_row = pd.DataFrame([row], index=pd.Index([idx], name=self.tradeHistory.index.name))
			self.tradeHistory = pd.concat([self.tradeHistory, new_row])
		else:
			for col, val in row.items():
				self.tradeHistory.at[idx, col] = val
		
	def _check_orders(self, ticker: str, price: float, dateChecked: datetime):
		#price = round(price, 4)
		if price <= 0: 	return

		# ---------------- BUY ORDERS ----------------
		for order in list(self._pendingBuys):
			if order.ticker != ticker:
				continue
			committed_this_order = order.reserved_cash
			if order.is_expired(dateChecked):
				self._print(f" Buy order from {FormatDate(order.datePlaced)} has expired on {FormatDate(dateChecked)} ticker: {order.ticker}")
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

			if order.units <= 0 or (actualCost > availableCash and not order.unfunded):
				self._print(f" CheckOrders: Cannot afford any {ticker} at {price}... canceling Buy on {FormatDate(dateChecked)}")
				self._cash_committed_to_orders -= committed_this_order
				self._pendingBuys.remove(order)
				continue

			# Execute Buy -> returns a Position
			pos = order.execute(price, dateChecked)
			if pos is not None:
				self._print(f" CheckOrders: {ticker} purchased {pos.units} for {price} on {FormatDate(dateChecked)}")
				self._cash_committed_to_orders -= committed_this_order
				self._total_cash -= actualCost

				if self._commision_cost > 0: self._print(' _check_orders: Commission charged for Buy: ' + str(self._commision_cost))
				self._positions.append(pos)
				self._pendingBuys.remove(order)
				self._save_position_to_history(pos)

		# ---------------- POSITIONS / SELL ORDERS ----------------
		for pos in list(self._positions):
			if pos.ticker != ticker:
				continue
			pos.latestPrice = price
			if not pos.is_sell_pending(): continue
			if pos.is_sell_expired(dateChecked):
				self._print(f" Ticker: {pos.ticker} sell order from {FormatDate(pos.dateSellOrderPlaced)} has expired on {FormatDate(dateChecked)}")
				pos.expire_sell()
				continue
			if pos.is_sell_executable(price):
				proceeds = pos.execute_sell(price, dateChecked, self._commision_cost)
				if proceeds > 0:
					self._total_cash += proceeds
					net_profit = ((pos.sellPrice - pos.purchasePrice) * pos.units) - self._commision_cost * 2
					self._print(f" Ticker: {pos.ticker} sold for {pos.sellPrice} on {FormatDate(dateChecked)} net profit ${net_profit})")
					if self._commision_cost > 0: self._print(' _check_orders: Commission charged for Sell: ' + str(self._commision_cost))
					self._save_position_to_history(pos, net_profit)
					self._positions.remove(pos)
						
	def _check_price_sequence(self, ticker: str, p1: float, p2: float, dateChecked: datetime):
		if p1 == p2:
			self._check_orders(ticker, p1, dateChecked)
			return
		steps = 40
		step = (p2 - p1) / steps
		prices = []
		for i in range(steps + 1): prices.append(round(p1 + i * step, 10))
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

		#close = round(close, 4) #Update latest price of each position
		for pos in self._positions:
			if pos.ticker == ticker:
				pos.latestPrice = close


	def _update_daily_value(self):
		def fmt(ticker, pct): return f'{ticker}:{pct:.4f}'[:12]
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
			x = [fmt(t, p) for t, p in zip(positions.index.astype(str), positions['Percentage'])]
			#x = (positions.index.astype(str) + ':' + positions['Percentage'].astype(str))
			#x = x.str[:12].tolist()
			padding_needed = max(0, 11 - len(x))
			x += [''] * padding_needed
			new_row = [self._total_cash, self._asset_value, total_value] + x[:11]
			self.dailyValue.loc[self.currentDate] = new_row
			
	#----------------------  Status and position info  ---------------------------------------

	def GetAvailableCash(self): return (self._total_cash - self._cash_committed_to_orders)

	def GetCashValue(self): return self._total_cash

	def GetCAGR(self):
		days_elapsed = (self.currentDate - self.modelStartDate).days
		years = days_elapsed / 365.25
		c = self._total_cash
		a = self._asset_value
		if days_elapsed ==0 or self._initialValue == 0: return 0
		total_ratio = (c + a) / self._initialValue
		daily_rate = (total_ratio ** (1 / days_elapsed)) - 1
		if days_elapsed < 365:
			return (total_ratio - 1) * 100
		else:
			return (((1 + daily_rate) ** 365.25) - 1) * 100
		
	def GetAssetValue(self): return self._asset_value	

	def GetCommittedCash(self): return self._cash_committed_to_orders
	
	def GetIdleCashPCT(self):
		available_cash = (self._total_cash - self._cash_committed_to_orders)
		asset_value = self._asset_value
		total_value = available_cash + asset_value
		return (available_cash / total_value) if total_value > 0 else 0

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
				pos.reasonForSale = ''

	def PlaceBuy(self, ticker: str, price: float, units: int, datePlaced: datetime, marketOrder: bool = False, expireAfterDays: int = 10, verbose: bool = False, tag: str = '', allowUnfunded: bool = False):
		if price <= 0 or price is None:
			self._print(f" PlaceBuy: Invalid price ${price} for {ticker}")
			return False
		#price = round(price, 4)
		if units <= 0: return False
		cash_required = units * price + self._commision_cost
		available = (self._total_cash - self._cash_committed_to_orders)
		if available < cash_required and not allowUnfunded:	#6.17 allowUnfunded: a pullback buy paid by sales filling at the same open
			self._print(f" PlaceBuy: Insufficient funds available {available} to place buy for {ticker} units={units} price={price}")
			return False
		order = BuyOrder(position_id=self._current_position_id, ticker=ticker,	datePlaced=datePlaced, units=units, orderPrice=price, marketOrder=marketOrder, expireAfterDays=expireAfterDays, reserved_cash=cash_required, tag=tag, unfunded=allowUnfunded and not marketOrder)
		self._current_position_id +=1
		self._pendingBuys.append(order)
		self._cash_committed_to_orders += cash_required
		if marketOrder:
			self._print(f" {FormatDate(datePlaced)} Buy placed ticker: {ticker} price: Market units:{units}")
		else:
			self._print(f" {FormatDate(datePlaced)} Buy placed ticker: {ticker} price: ${price} units:{units}")
		return True

	def SellPosition(self, position, price: float, datePlaced: datetime, marketOrder: bool = False, expireAfterDays: int = 10, reasonForSale:str=''):
		if position is None: return False
		if position.dateSellOrderFilled is not None: return False
		if position.units <= 0:	return False
		#position.sellOrderPrice = round(price, 4)
		position.sellOrderPrice = price
		position.dateSellOrderPlaced = datePlaced
		position.marketOrder = marketOrder
		position.expireAfterDays = expireAfterDays
		position.reasonForSale = reasonForSale[:20]
		return True

	def SellPositionPartial(self, position, units: int, price: float, datePlaced: datetime, marketOrder: bool = False, expireAfterDays: int = 10, reasonForSale:str=''):
		if position is None: return False
		if position.dateSellOrderFilled is not None: return False
		if position.units <= 0 or units <= 0: return False
		if units >= position.units:
			return self.SellPosition(position, price, datePlaced, marketOrder, expireAfterDays, reasonForSale)
		# Create a new position representing the portion being sold
		sell_pos = Position(position_id = self._current_position_id, ticker=position.ticker, units=units, dateBuyOrderPlaced=position.dateBuyOrderPlaced, dateBuyOrderFilled=position.dateBuyOrderFilled, buyOrderPrice=position.buyOrderPrice, purchasePrice=position.purchasePrice, latestPrice=position.latestPrice)
		self._current_position_id +=1
		position.units -= units
		sell_pos.sellOrderPrice = price
		sell_pos.dateSellOrderPlaced = datePlaced
		sell_pos.marketOrder = marketOrder
		sell_pos.expireAfterDays = expireAfterDays
		sell_pos.reasonForSale = reasonForSale[:20]
		self._save_position_to_history(sell_pos)
		self._positions.append(sell_pos)
		if position.units == 0: self._positions.remove(position)
		return True

	def PlaceSell(self, ticker: str, units: int, price: float, datePlaced: datetime, marketOrder: bool = False, expireAfterDays: int = 10, reasonForSale:str=''):
		if units <= 0: return False
		eligible = [p for p in self._positions if p.ticker == ticker and p.dateSellOrderPlaced is None and p.dateSellOrderFilled is None]
		eligible.sort(key=lambda p: p.dateBuyOrderFilled)
		units_to_sell = units
		placed_any = False
		for pos in eligible:
			if units_to_sell <= 0:
				break
			if pos.units <= units_to_sell:
				if self.SellPosition(pos, price, datePlaced, marketOrder, expireAfterDays, reasonForSale):
					units_to_sell -= pos.units
					placed_any = True
			else:
				if self.SellPositionPartial(pos, units_to_sell, price, datePlaced, marketOrder, expireAfterDays, reasonForSale):
					units_to_sell = 0
					placed_any = True
		return placed_any

	def SellAllPositions(self, datePlaced: datetime, ticker: str = '', allowWeekEnd: bool = False):
		for pos in list(self._positions):
			if ticker == '' or pos.ticker == ticker:
				self.SellPosition(position=pos, price=pos.latestPrice, datePlaced=datePlaced, marketOrder=True, expireAfterDays=5, reasonForSale='Liquidation')

	#--------------------------------------  Closing Reporting ---------------------------------------
	def SaveTradeHistory(self, foldername:str, addTimeStamp:bool = False):
		if self.trackHistory:
			if self.useDatabase and self.database:
				if self.database.Open():
					df = self.tradeHistory.copy()
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
		endDate = startDate + pd.DateOffset(years=durationInYears)
		super(TradingModel, self).__init__(portfolioName=modelName, startDate=startDate, totalFunds=totalFunds, trackHistory=trackHistory, useDatabase=useDatabase, verbose=verbose)	
		self.start_processing = datetime.today()
		self.end_processing = None
		self.modelReady = False
		self.modelName = modelName
		#3 random base-36 chars: two workers starting in the same second got the same batchName and mixed their results (11/18/26). Max 50 chars.
		self.batchName = modelName[:30] + '_' + self.start_processing.strftime('%Y%m%d-%H%M%S') + '-' + ''.join(np.random.default_rng().choice(list('0123456789abcdefghijklmnopqrstuvwxyz'), 3))
		self.modelStartDate  = None	
		self.modelEndDate = None
		self.currentDate = None
		self.priceHistory = []  #list of price histories for each stock in _tickerList
		self._priceMap = {} #Mapping between tickers and priceHistory elements
		self._tickerList = []	#list of stocks currently held
		self.customValue1 = None	#can be used to store custom values when using the model
		self.customValue2 = None
		self._NormalizePrices = False
		self._swings = {}	#6.15 SwingTrade: ticker -> open trade-around-core round trip
		self._pulls = {}	#6.17 SwingTrade: ticker -> open pullback lot
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
		forDate = self.currentDate - pd.offsets.BDay(1) 
		r = None
		if ticker:
			if not ticker in self._tickerList: self.AddTicker(ticker)
			if ticker in self._tickerList:
				for ph in self.priceHistory:
					if ph.ticker == ticker: r = ph.GetPrice(forDate) 
		return r

	def GetPriceSnapshot(self, ticker:str=''): 
		#returns snapshot object of yesterday's pricing info to help make decisions today
		forDate = self.currentDate - pd.offsets.BDay(1) 
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

	def SellPosition(self, position, price: float, datePlaced: datetime, marketOrder: bool = False, expireAfterDays: int = 10, reasonForSale: str = ''):
		#datePlaced is not used in TradeModel				
		if not position.ticker in self._tickerList: self.AddTicker(position.ticker)	
		if position.ticker in self._tickerList:	
			if marketOrder or price ==0: 
				sn = self.GetPriceSnapshot(position.ticker)
				price = sn.Average
		#11/18/26: must return the result - the base PlaceSell counts the units sold with it; returning None made a partial sell liquidate every lot smaller than the request
		return super(TradingModel, self).SellPosition(position=position, price=price, datePlaced=self.currentDate, marketOrder=marketOrder, expireAfterDays=expireAfterDays, reasonForSale=reasonForSale)

	def PlaceSell(self, ticker:str, units: int, price:float, marketOrder:bool=False, expireAfterDays:int=10, reasonForSale: str = ''): 
		if not ticker in self._tickerList: self.AddTicker(ticker)	
		if ticker in self._tickerList:	
			if marketOrder or price ==0: 
				sn = self.GetPriceSnapshot(ticker)
				price = sn.Average
		super(TradingModel, self).PlaceSell(ticker=ticker, units=units, price=price, datePlaced=self.currentDate, marketOrder=marketOrder, expireAfterDays=expireAfterDays, reasonForSale=reasonForSale)

	def SellAllPositions(self, ticker: str = '', allowWeekEnd: bool = False):
		for position in list(self._positions):
			if ticker == '' or position.ticker == ticker:
				super(TradingModel, self).SellPosition(position=position, price=position.latestPrice, datePlaced=self.currentDate, marketOrder=True, reasonForSale='Liquidation')

	def AlignPositions(self, targetPositions: pd.DataFrame, rateLimitTransactions:bool = False, shopBuyPercent:float = 0.0, shopSellPercent:float = 0.0, rateLimitPct: float = None):
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
			for ticker in targetPositions.index: self.AddTicker(ticker) #Should be handled by Buy but just in case
		# Pandas 3.x raises TypeError when adding new rows via df.loc[new_label, col] = scalar
		# into a typed float64 block.  Collect missing tickers first, then concat once.
		missing_tickers = [p.ticker for p in self._positions if p.ticker not in targetPositions.index]
		if missing_tickers:
			targetPositions = pd.concat([
				targetPositions,
				pd.DataFrame({'TargetHoldings': 0.0}, index=missing_tickers)
			])
		target_by_ticker = targetPositions['TargetHoldings'].groupby(level=0).sum()	#6.16: a ticker can appear more than once (one row per lot not in the picks)
		for order in list(self._pendingBuys):
			if order.ticker not in targetPositions.index:
				self._cancel_order(order)
			elif order.tag == 'Swing' and float(target_by_ticker.loc[order.ticker]) <= 0:
				self._cancel_order(order)	#6.15: the stock was dropped - don't buy the swing shares back
			elif order.tag == 'Pull' and float(target_by_ticker.loc[order.ticker]) <= 0:
				self._cancel_order(order)	#6.19: a pending pullback limit buy for a dropped stock
		for p in self._positions:
			if p.reasonForSale == 'Pull' and p.is_sell_pending() and not p.marketOrder and float(target_by_ticker.get(p.ticker, 0.0)) <= 0: p.marketOrder = True	#6.17: dropped - sell the pullback lot now

		TotalTargets = targetPositions['TargetHoldings'].sum()
		if TotalTargets > 0:
			targetPositions['Weight'] = targetPositions['TargetHoldings'] / TotalTargets
			targetPositions['TargetValue'] = targetPositions['Weight'] * totalValue
		else:
			targetPositions['TargetValue'] = targetPositions['TargetHoldings']

		current_value_map = {}
		for p in self._positions:
			if p.is_sell_pending(): continue	#already being sold: not part of what we hold going forward (else it is sold twice)
			current_value_map[p.ticker] = current_value_map.get(p.ticker, 0) + (p.units * p.latestPrice)
		latest = {p.ticker: p.latestPrice for p in self._positions if p.latestPrice > 0}
		for o in self._pendingBuys:
			if o.tag == 'Pull': continue	#6.19: a pending pullback limit buy is an extra lot, not part of the target
			price = latest.get(o.ticker, o.orderPrice) if o.tag == 'Swing' else o.orderPrice	#6.15: swing shares count as held at today's price (no market buy-back)
			current_value_map[o.ticker] = current_value_map.get(o.ticker, 0) + (o.units * price)

		targetPositions['CurrentValue'] = targetPositions.index.map(
			lambda t: current_value_map.get(t, 0.0)
		)
		targetPositions['DeltaValue'] = targetPositions['TargetValue'] - targetPositions['CurrentValue']

		sells = targetPositions[targetPositions['DeltaValue'] < 0].copy()
		buys = targetPositions[targetPositions['DeltaValue'] > 0].copy()
		sells.sort_values(by=['DeltaValue'], ascending=True, inplace=True)
		buys.sort_values(by=['DeltaValue'], ascending=False, inplace=True)

		# ---------------- RATE LIMITING ----------------
		#rolloverPending: the rate limit cut this realignment short, so the run loops call AlignPositions again the next
		#day, whatever the reEvaluationInterval, until the rollover completes
		self.rolloverPending = False
		if rateLimitTransactions:
			daily_budget = totalValue * (rateLimitPct if rateLimitPct else CONSTANTS.RATE_LIMITING_MAX_TURNOVER_PCT)	#6.13: TradeModelParams.rateLimitPct
			sell_budget = daily_budget
			buy_budget = daily_budget
			sell_needed = abs(sells['DeltaValue']).sum()
			if sell_needed > 0:
				sell_scale = min(1.0, sell_budget / sell_needed)
				sells.loc[:, 'DeltaValue'] *= sell_scale
				if sell_scale < 1.0: self.rolloverPending = True

			buy_needed = buys['DeltaValue'].sum()
			if buy_needed > 0:
				buy_scale = min(1.0, buy_budget / buy_needed)
				buys.loc[:, 'DeltaValue'] *= buy_scale
				if buy_scale < 1.0: self.rolloverPending = True
			self._print(f" AlignPositions: Rate limiting enabled. Total budget ${daily_budget:,.0f}")

		# ---------------- CASH LIMIT ON BUYS ----------------
		availableFunds = self._total_cash - self._cash_committed_to_orders
		buy_needed = buys['DeltaValue'].sum()
		if buy_needed > availableFunds and buy_needed > 0:
			cash_scale = availableFunds / buy_needed
			buys.loc[:, 'DeltaValue'] *= cash_scale	#buys waiting for today's sell proceeds go out the next day through the run loops' idle-cash check
			self._print(f" AlignPositions: Buy scaling due to cash. Scale={cash_scale:.3f} available: {availableFunds} needed: {buy_needed}")

		# ---------------- EXECUTE SELLS ----------------
		for ticker, row in sells.iterrows():
			if ticker == CONSTANTS.CASH_TICKER:
				continue
			sn = self.GetPriceSnapshot(ticker)
			if sn is None or sn.Average <= 0:
				self._print(f" AlignPositions: Unable to get sell price for {ticker} on {self.currentDate}")
				continue
			target_sell = round(max(sn.Average, sn.Target) * (1 + shopSellPercent), 10)
			price = target_sell if not tradeAtMarket else sn.Average
			deltaValue = float(row['DeltaValue'])
			unitsToSell = int(abs(deltaValue) / price)
			if unitsToSell > 0:
				self._print(f" AlignPositions: Sell {ticker} units={unitsToSell} @ ${price} (Mkt: {tradeAtMarket})")
				self.PlaceSell(ticker=ticker, units=unitsToSell, price=price, marketOrder=tradeAtMarket, expireAfterDays=expireAfterDays, reasonForSale='ReAlignment')

		# ---------------- EXECUTE BUYS ----------------
		for ticker, row in buys.iterrows():
			if ticker == CONSTANTS.CASH_TICKER:
				continue
			sn = self.GetPriceSnapshot(ticker)
			if sn is None or sn.Average <= 0:
				self._print(f" AlignPositions: Unable to get buy price for {ticker} on {self.currentDate}")
				continue
			target_buy = round(min(sn.Average, sn.Target) * (1-shopBuyPercent), 10)
			price = target_buy if not tradeAtMarket else sn.Average
			deltaValue = float(row['DeltaValue'])
			availableFunds = self._total_cash - self._cash_committed_to_orders
			maxUnits = int((availableFunds - self._commision_cost) / price)
			unitsToBuy = int(deltaValue / price)
			unitsToBuy = min(unitsToBuy, maxUnits)
			if unitsToBuy > 0:
				self._print(f" AlignPositions: Buy {ticker} units={unitsToBuy} @ ${price} (Mkt: {tradeAtMarket})")
				self.PlaceBuy(ticker=ticker, price=price, units=unitsToBuy, marketOrder=tradeAtMarket, expireAfterDays=expireAfterDays)

	def TrimProfits(self, trimProfitsPercent: float = 6.0, maxTrimPctPortfolio: float = 0.35, expireAfterDays: int = 3, accelThreshold: float = 0.05, dmaThreshold: float = 0.07, minWeightToTrim: float = 0.035, trimFraction: float = 0.33):
		if trimProfitsPercent <= 0:
			return
		#latest_price = self.GetPrice(p.ticker)
		#if not latest_price: latest_price = p.latestPrice
		assetValue = sum(p.units * p.latestPrice for p in self._positions)
		totalValue = self._total_cash + assetValue
		if totalValue <= 0:
			return
		maxTrimValue = totalValue * maxTrimPctPortfolio
		trimmedValue = 0.0
		eligible = []
		for pos in self._positions:
			if pos.units <= 0: continue
			if pos.latestPrice is None or pos.latestPrice <= 0: continue
			if pos.purchasePrice <= 0: 				continue
			if pos.dateSellOrderPlaced is not None: continue
			gainPct = ((pos.latestPrice - pos.purchasePrice) / pos.purchasePrice) * 100

			# ----- Conditional Volatility Harvesting Filters -----
			sn = self.GetPriceSnapshot(pos.ticker)
			if sn is None: continue
			if sn.PC_1Year is None or sn.PC_1Month is None or sn.Distance200DMA is None or sn.Vol20 is None or sn.Vol60 is None: continue
			if sn.PC_1Year == 0 or sn.Vol60 <= 0: continue
			if sn.PC_1Year <= 3.0: continue
			accelerationRatio = sn.PC_1Month / sn.PC_1Year
			positionValue = pos.units * pos.latestPrice
			positionWeight = positionValue / totalValue
			if accelerationRatio <= accelThreshold or sn.Distance200DMA <= dmaThreshold or positionWeight <= minWeightToTrim: continue
			eligible.append((pos, gainPct))
		# Highest gain first
		eligible.sort(key=lambda x: x[1], reverse=True)
		for pos, gainPct in eligible:
			if trimmedValue >= maxTrimValue: break
			remainingTrimCapacity = maxTrimValue - trimmedValue
			unitsToSell = int(pos.units * trimFraction)
			unitsToSell = max(unitsToSell, 1)
			maxUnitsByPortfolioCap = int(remainingTrimCapacity / pos.latestPrice)
			unitsToSell = min(unitsToSell, maxUnitsByPortfolioCap)
			unitsToSell = min(unitsToSell, pos.units)
			if unitsToSell <= 0: continue
			targetPrice = pos.latestPrice * (1+trimProfitsPercent)
			gainPct += trimProfitsPercent
			self._print(
				f" TrimProfits (Conditional): Selling {unitsToSell} units of {pos.ticker} "
				f"@ targetPrice={targetPrice} target_gain={gainPct:.2f}%"
			)
			super(TradingModel, self).SellPositionPartial(position=pos, units=unitsToSell, price=targetPrice, datePlaced=self.currentDate, marketOrder=False, expireAfterDays=expireAfterDays, reasonForSale='TrimProfit')
			trimmedValue += unitsToSell * pos.latestPrice

	#6.15 _SW: trade around the core position (docs/experiment/trade_around_core_check.py, 11/22/26: +12-20 bp of the position
	#per trade after a spike, ~1981-2026). After a spike in a held stock - close at the top of its 40-day regression channel
	#(z > swingTriggerZ) or a 5-day rise over swingTriggerSpike x the 20-day daily volatility x sqrt 5, both on the prior day's
	#close - sell swingFraction of it at the open (market), then a limit buy for the same units at sale price x (1 - swingSigma
	#x vol x sqrt swingDays). Unfilled after swingDays trading days, it turns into a market buy (next open). Realignment counts
	#the swing shares as held, cancels the buy-back when the stock's target goes to 0, and a ticker with any order pending is
	#left alone. 6.17: trigger levels are parameters; swingRiskOffOnly trims only while the .INX switch is risk-off.
	#6.17 _PB: buy the pullbacks (docs/experiment/swing_tuning_grid.py, 11/22/26). After a drop in a held stock - z <
	#-pullTriggerZ or a 5-day fall over pullTriggerDrop sigma - buy pullFraction more at the open, funded by market sales of
	#the other held stocks pro rata (filled at the same open: ProcessDay handles the pull ticker after them), and put a limit
	#sell on the extra lot at its fill x (1 + pullSigma x vol x sqrt pullDays); unfilled after pullDays it sells at market.
	#Realignment doesn't count the extra lot (its sell is pending), so the next rebalance restores the other holdings from its
	#proceeds; a stock whose target goes to 0 has its lot sold at market. Skipped while risk-off (pullSkipRiskOff), when other
	#buy orders are pending, or past PULL_MAX_PCT of the portfolio.
	#6.19 (docs/experiment/swing_tuning_grid2.py): pullEntrySigma > 0 = a limit buy at the prior close x (1 - k x vol), live
	#PULL_ENTRY_DAYS days, else canceled; nothing is sold until it fills, then the funding sells at the next open (cash is
	#short overnight). Realignment doesn't count a pending pullback buy and cancels it when the target goes to 0.
	#pullFundStretched: the funding sells the other holdings with the highest channel z first instead of pro rata.
	SWING_WINDOW = 40
	PULL_MAX_PCT = 0.15		#most of the portfolio that open pullback lots may hold
	PULL_ENTRY_DAYS = 2		#6.19: days a pullback limit buy is live
	def _swing_signal(self, ticker: str):
		#(20-day daily vol, 40-day channel z, 5-day move in sigma) on the closes before today, or None (Params.SwingFeatures)
		ph = self._priceMap.get(ticker)
		if ph is None or ph.historicalPrices is None or 'Close' not in ph.historicalPrices.columns: return None
		hp = ph.historicalPrices
		i = hp.index.searchsorted(self.currentDate, side='left')	#closes before today
		if i < self.SWING_WINDOW + 1: return None
		return SwingFeatures(hp['Close'].to_numpy(dtype=float)[i - self.SWING_WINDOW - 1:i])

	def SwingTrade(self, targetPositions: pd.DataFrame, params: TradeModelParams, riskOff: bool = False):
		trim = params.swingFraction > 0 and (riskOff or not params.swingRiskOffOnly)
		pull = params.pullFraction > 0 and not (riskOff and params.pullSkipRiskOff)
		if not (trim or pull) or targetPositions is None or targetPositions.empty or 'TargetHoldings' not in targetPositions.columns: return
		if self.currentDate.weekday() >= 5: return
		held = {}
		for p in self._positions:
			if not p.is_sell_pending(): held[p.ticker] = held.get(p.ticker, 0) + p.units
		busy = {p.ticker for p in self._positions if p.is_sell_pending()} | {o.ticker for o in self._pendingBuys}
		targets = targetPositions['TargetHoldings'].groupby(level=0).sum()
		for ticker, units in held.items():
			if ticker == CONSTANTS.CASH_TICKER or ticker in self._swings or ticker in self._pulls or ticker in busy: continue
			if ticker not in targets.index or float(targets.loc[ticker]) <= 0: continue
			sig = self._swing_signal(ticker)
			if sig is None: continue
			vol, z, move5 = sig
			if trim and (z > params.swingTriggerZ or move5 > params.swingTriggerSpike):
				n = int(units * params.swingFraction)
				if n <= 0: continue
				self.PlaceSell(ticker=ticker, units=n, price=0, marketOrder=True, expireAfterDays=5, reasonForSale='Swing')
				refs = [p for p in self._positions if p.ticker == ticker and p.is_sell_pending() and p.reasonForSale == 'Swing']
				if refs:
					self._swings[ticker] = {'sells': refs, 'factor': max(0.5, 1 - params.swingSigma * vol * np.sqrt(params.swingDays)), 'days': params.swingDays, 'order': None, 'age': 0}
					self._print(f" SwingTrade: {FormatDate(self.currentDate)} sell {n} {ticker} at the open, buy back below x{self._swings[ticker]['factor']:.3f}")
				busy.add(ticker)
			elif pull and (z < -params.pullTriggerZ or move5 < -params.pullTriggerDrop):
				if self._place_pull(ticker, units, vol, held, busy, params): busy.add(ticker)

	def _place_pull(self, ticker: str, units: int, vol: float, held: dict, busy: set, params: TradeModelParams) -> bool:
		limit = params.pullEntrySigma > 0	#6.19
		if any(o.tag != 'Swing' and not (limit and o.tag == 'Pull') for o in self._pendingBuys): return False	#other buys would compete for the funding
		sn = self.GetPriceSnapshot(ticker)
		if sn is None or not sn.Close > 0: return False
		price = float(sn.Close) * (1 - params.pullEntrySigma * vol if limit else 1)
		if not price > 0: return False
		latest = {p.ticker: p.latestPrice for p in self._positions if p.latestPrice > 0}
		totalValue = self._total_cash + sum(p.units * p.latestPrice for p in self._positions)
		open_pull = sum(s['pos'].units * s['pos'].latestPrice if s['pos'] is not None else s['order'].units * s['order'].orderPrice for s in self._pulls.values())
		n = min(int(units * params.pullFraction), int((totalValue * self.PULL_MAX_PCT - open_pull) / price))
		if n <= 0: return False
		if not limit:
			need = n * price + self._commision_cost - max(0.0, self.GetAvailableCash())
			if need > 0 and not self._fund_pull(ticker, need, busy, params.pullFundStretched): return False
		if not Portfolio.PlaceBuy(self, ticker=ticker, price=price, units=n, datePlaced=self.currentDate, marketOrder=not limit, expireAfterDays=3650, tag='Pull', allowUnfunded=True): return False
		self._pulls[ticker] = {'order': self._pendingBuys[-1], 'pos': None, 'factor': 1 + params.pullSigma * vol * np.sqrt(params.pullDays), 'days': params.pullDays, 'age': 0,
			'limit': limit, 'eage': 0, 'stretched': params.pullFundStretched}
		self._print(f" SwingTrade: {FormatDate(self.currentDate)} pullback buy {n} {ticker} {'limit ' + format(price, '.2f') if limit else 'at the open'}, sell above x{self._pulls[ticker]['factor']:.3f}")
		return True

	def _fund_pull(self, ticker: str, need: float, busy: set, stretched: bool) -> bool:
		#market sales of the other holdings worth need x 1.02 (2% extra: the open differs from the last close), pro rata or
		#6.19 stretched: highest channel z first. False (nothing sold) when they hold too little.
		held = {}
		for p in self._positions:
			if not p.is_sell_pending(): held[p.ticker] = held.get(p.ticker, 0) + p.units
		latest = {p.ticker: p.latestPrice for p in self._positions if p.latestPrice > 0}
		funders = {t: u for t, u in held.items() if t != ticker and t != CONSTANTS.CASH_TICKER and t not in busy and t not in self._swings and t not in self._pulls and latest.get(t, 0) > 0}
		pool = sum(u * latest[t] for t, u in funders.items())
		if pool <= need * 1.05: return False
		if stretched:
			zs = {}
			for t in funders:
				sig = self._swing_signal(t)
				zs[t] = sig[1] if sig is not None else -1e9
			left = need * 1.02
			for t in sorted(funders, key=lambda t: (-zs[t], t)):
				k = min(funders[t], int(np.ceil(left / latest[t])))
				if k > 0: self.PlaceSell(ticker=t, units=k, price=0, marketOrder=True, expireAfterDays=5, reasonForSale='PullFund')
				left -= k * latest[t]
				if left <= 0: break
		else:
			for t, u in funders.items():
				k = min(u, int(np.ceil(u * need * 1.02 / pool)))
				if k > 0: self.PlaceSell(ticker=t, units=k, price=0, marketOrder=True, expireAfterDays=5, reasonForSale='PullFund')
		return True

	def _update_swings(self):
		#after the day's orders: place the buy-back for a filled swing sale; turn an unfilled buy-back into a market order after `days`
		for ticker, s in list(self._swings.items()):
			if s['order'] is None:
				if any(p.is_sell_pending() for p in s['sells']): continue
				filled = [p for p in s['sells'] if p.dateSellOrderFilled is not None and p.sellPrice > 0]
				units = sum(p.units for p in filled)
				if units <= 0:
					del self._swings[ticker]
					continue
				limit = sum(p.units * p.sellPrice for p in filled) / units * s['factor']
				units = min(units, int((self.GetAvailableCash() - self._commision_cost) / limit))
				if units > 0 and Portfolio.PlaceBuy(self, ticker=ticker, price=limit, units=units, datePlaced=self.currentDate, marketOrder=False, expireAfterDays=3650, tag='Swing'):
					s['order'] = self._pendingBuys[-1]
				else:
					del self._swings[ticker]
			else:
				o = s['order']
				if not any(q is o for q in self._pendingBuys):	#filled, or canceled by realignment
					del self._swings[ticker]
					continue
				s['age'] += 1
				if s['age'] >= s['days'] - 1 and not o.marketOrder:
					o.marketOrder = True	#limit active on days 2..days; market at the next open
					self._print(f" SwingTrade: {FormatDate(self.currentDate)} {ticker} buy-back not filled, buying at market")
		#6.17 pullback lots: once bought, a limit sell on the lot; market after `days`
		for ticker, s in list(self._pulls.items()):
			if s['pos'] is None:
				o = s['order']
				if any(q is o for q in self._pendingBuys):	#a market buy that didn't fill today (no price), 6.19: a limit buy past its days: give up
					s['eage'] += 1
					if s['limit'] and s['eage'] < self.PULL_ENTRY_DAYS: continue
					self._cancel_order(o)
					del self._pulls[ticker]
					continue
				pos = next((p for p in self._positions if p.position_id == o.position_id and p.dateSellOrderPlaced is None), None)
				if pos is None:
					del self._pulls[ticker]
					continue
				Portfolio.SellPosition(self, pos, price=pos.purchasePrice * s['factor'], datePlaced=self.currentDate, marketOrder=False, expireAfterDays=3650, reasonForSale='Pull')
				s['pos'] = pos
				if s['limit']:	#6.19: filled - sell the funding at the next open
					need = self._commision_cost - self.GetAvailableCash()
					if need > 0:
						busy = {p.ticker for p in self._positions if p.is_sell_pending()} | {q.ticker for q in self._pendingBuys}
						self._fund_pull(ticker, need, busy, s['stretched'])
			else:
				pos = s['pos']
				if pos.dateSellOrderFilled is not None or not any(q is pos for q in self._positions):
					del self._pulls[ticker]
					continue
				s['age'] += 1
				if s['age'] >= s['days'] - 1 and not pos.marketOrder:
					pos.marketOrder = True	#limit active on days 2..days; market at the next open

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
		gain = self.GetCAGR()
		print(f"Cash: ${c:,.0f} ' assets: ${a:,.0f} total: ${(c+a):,.0f} CAGR: {gain:,.0f}%")
		print(f"Processing time: {params.processing_minutes} minutes.  Completion Time: {self.end_processing}")
		print('')
		if params is not None:
			if params.saveResults:
				self.SaveDailyValue(self._dataFolderTradeModel)	#before _recordResults: it reads the daily values back from the table
				self._recordResults(params)
			if params.saveTradeHistory: self.SaveTradeHistory(self._dataFolderTradeModel)
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
					super(TradingModel, self).SellPosition(position=pos, price=forced_price, datePlaced=self.currentDate, marketOrder=True, expireAfterDays=5, reasonForSale='MissingPrice')

	def ProcessDay(self, withIncrement: bool = True, allowWeekEnd: bool = False):
		# Process current day and increment the current date, allowWeekEnd is for model closing only
		if self.currentDate.weekday() < 5 or allowWeekEnd:
			active = sorted(self._active_tickers())	#6.18: a fixed order - a set's order changes with Python's hash seed, and the order same-day orders fill in changes results
			pull_buys = {o.ticker for o in self._pendingBuys if o.tag == 'Pull'}
			if pull_buys: active = [t for t in active if t not in pull_buys] + [t for t in active if t in pull_buys]	#6.17: the funding sales fill first
			for ticker in active:
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
				for pos in self._positions:
					if pos.ticker == ticker:
						pos.latestPrice = sn.Close
			if self._swings or self._pulls: self._update_swings()	#6.15, 6.17
		self._update_daily_value()
		if self.pbar:
			c = self._total_cash
			a = self._asset_value
			gain = self.GetCAGR()
			self.pbar.set_description(f"Model: {self.modelName} Date: {FormatDate(self.currentDate)} CAGR: {gain:,.0f}% Cash: ${c:,.0f} Assets: ${a:,.0f} Total: ${(c + a):,.0f} ({FormatDate(self.modelStartDate)} to {FormatDate(self.modelEndDate)})")
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
				self.currentDate = self.modelEndDate + pd.offsets.BDay(1) 
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
	HOLD_WORKER = "HOLD"	#a row with Processing = 1 and WorkerName 'HOLD' is parked: no worker (old or new code) claims it until released
	def __init__(self, verbose: bool = False, codeVersion: str = ''):
		self.db = PTADatabase()
		if not self.db.Open():
			print(" ExtensiveTesting: database initialization failed.")
			assert False
		self.worker_name = socket.gethostname()
		self._verbose = verbose
		self.codeVersion = codeVersion	#the worker's Selection.SELECTION_VERSION: it only claims jobs whose MinVersion <= this

	def ensure_queue_table(self):
		sql = f"""
		IF OBJECT_ID('{self.QUEUE_TABLE}', 'U') IS NULL
		BEGIN
			CREATE TABLE {self.QUEUE_TABLE} (
				QueueID     INT IDENTITY(1,1) PRIMARY KEY,
				Processing  BIT NOT NULL DEFAULT 0,
				EnqueuedAt  DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
				StartedProcessingAt  DATETIME2 NULL,
				ParamsJson  NVARCHAR(MAX) NOT NULL,
				WorkerName  NVARCHAR(MAX) NULL,
				MinVersion  VARCHAR(20) NULL
			)
		END
		IF COL_LENGTH('{self.QUEUE_TABLE}', 'MinVersion') IS NULL ALTER TABLE {self.QUEUE_TABLE} ADD MinVersion VARCHAR(20) NULL
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
		if data.get("riskFilter", True): data.pop("riskFilter", None)	#only sent when off, so workers on older code can still read default jobs
		if not data.get("rebuildMarketStates", False): data.pop("rebuildMarketStates", None)	#only sent when on
		if not data.get("rebuildBlendedPicks", False): data.pop("rebuildBlendedPicks", None)	#only sent when on
		if not data.get("damageFilter", False): data.pop("damageFilter", None)	#only sent when on
		if not data.get("volAdjustedRank", False): data.pop("volAdjustedRank", None)	#only sent when on
		if not data.get("blendFilters", ""): data.pop("blendFilters", None)	#only sent when set
		if data.get("adaptiveFilterSet", 1) == 1: data.pop("adaptiveFilterSet", None)	#only sent when not the default
		if not data.get("dmaWeight", False): data.pop("dmaWeight", None)	#only sent when on
		if data.get("riskOffExposure", 1.0) >= 1.0: data.pop("riskOffExposure", None)	#5.6 fields: only sent when not the default
		if not data.get("adaptiveStockCount", 0): data.pop("adaptiveStockCount", None)
		#marketStateVersion: always sent (6.2: the default is 4.8, so a dropped 4.6/4.7 would read as 4.8)
		if data.get("riskOffSignal", "index") == "index": data.pop("riskOffSignal", None)	#5.9 fields: only sent when not the default
		if not data.get("postManiaDays", 0): data.pop("postManiaDays", None)
		if not data.get("riskOffSmooth", 0): data.pop("riskOffSmooth", None)	#6.12: only sent when on
		if abs(data.get("rateLimitPct", 0.20) - 0.20) < 1e-9: data.pop("rateLimitPct", None)	#6.13: only sent when not the default
		if not data.get("swingFraction", 0):	#6.15 fields: only sent when on
			for k in ("swingFraction", "swingDays", "swingSigma", "swingTriggerZ", "swingTriggerSpike", "swingRiskOffOnly"): data.pop(k, None)
		else:	#6.17 trigger fields: only sent when not the default
			if abs(data.get("swingTriggerZ", 1.5) - 1.5) < 1e-9: data.pop("swingTriggerZ", None)
			if abs(data.get("swingTriggerSpike", 1.5) - 1.5) < 1e-9: data.pop("swingTriggerSpike", None)
			if not data.get("swingRiskOffOnly", False): data.pop("swingRiskOffOnly", None)
		if not data.get("pullFraction", 0):	#6.17 fields: only sent when on
			for k in ("pullFraction", "pullDays", "pullSigma", "pullTriggerZ", "pullTriggerDrop", "pullSkipRiskOff", "pullEntrySigma", "pullFundStretched"): data.pop(k, None)
		else:	#6.19 fields: only sent when on
			if not data.get("pullEntrySigma", 0): data.pop("pullEntrySigma", None)
			if not data.get("pullFundStretched", False): data.pop("pullFundStretched", None)
		if not data.get("volTarget", 0): data.pop("volTarget", None)	#6.8 fields: only sent when on
		if data.get("volLookback", 21) == 21: data.pop("volLookback", None)
		if not data.get("volTargetOff", ""): data.pop("volTargetOff", None)	#6.11: only sent when set
		if not data.get("profitScreen", 0): data.pop("profitScreen", None)	#6.9: only sent when on
		if not data.get("strictData", ""): data.pop("strictData", None)	#6.10: only sent when set
		if not data.get("saveTradeHistory", False): data.pop("saveTradeHistory", None)	#11/21/26: only sent when on (older workers default it to True and gate everything on it)
		#pointInTimeUniverse: always sent (6.2: the default is True, so a dropped False would read as True)
		data.pop("selectionVersion", None)	#the worker sets its own version; never queued
		return json.dumps(data)
	@staticmethod
	def _deserialize(json_str: str) -> TradeModelParams:
		data = json.loads(json_str)
		start_date = data.pop("startDate", None)
		if "useDatabase" in data: data.setdefault("useStoredPicks", data.pop("useDatabase"))	#renamed 11/18/26; jobs queued by older code
		known = {f.name for f in fields(TradeModelParams) if f.init}
		unknown = [k for k in data if k not in known]
		if unknown:
			print(f" ExtensiveTesting: ignoring unknown queued parameters {unknown} (job queued by newer code?)")
			for k in unknown: data.pop(k)
		params = TradeModelParams(**data)
		if start_date is not None:
			params.startDate = start_date  # uses setter → pd.Timestamp
		return params

	@staticmethod
	def version_number(v) -> int:
		#'5.3' -> 5003, same as the claim SQL's PARSENAME arithmetic (versions are major.minor)
		try:
			parts = [int(x) for x in str(v).strip().split('.')] + [0]
			return parts[0] * 1000 + parts[1]
		except ValueError: return 0
	def add_to_queue(self, params: TradeModelParams, minVersion: str = None, hold: bool = False):
		#minVersion: only workers whose code is at least this version claim the job. hold: park it (see HOLD_WORKER)
		if hold:
			sql = f"INSERT INTO {self.QUEUE_TABLE} (ParamsJson, MinVersion, Processing, WorkerName) VALUES (:params, :mv, 1, '{self.HOLD_WORKER}')"
		else:
			sql = f"INSERT INTO {self.QUEUE_TABLE} (ParamsJson, MinVersion) VALUES (:params, :mv)"
		self.db.ExecSQL(sql, {"params": self._serialize(params), "mv": minVersion})
		if self._verbose: print(f"ExtensiveTesting: Enqueued {params.modelName}")

	MIN_VERSION_SQL = "(ISNULL(TRY_CAST(PARSENAME(MinVersion, 2) AS int), 0) * 1000 + ISNULL(TRY_CAST(PARSENAME(MinVersion, 1) AS int), 0))"	#'6.12' -> 6012, same as version_number
	def newer_job_waiting(self) -> int:
		#QueueID of the first waiting job that needs newer code than this worker's (0 if none): the worker is out of date and should stop
		sql = f"SELECT TOP (1) QueueID FROM {self.QUEUE_TABLE} WHERE Processing = 0 AND MinVersion IS NOT NULL AND {self.MIN_VERSION_SQL} > :myversion ORDER BY QueueID"
		rows = self.db.ExecSQL(sql, {"myversion": self.version_number(self.codeVersion)})
		return int(rows[0][0]) if rows else 0

	def claim_from_queue(self) -> tuple[int, TradeModelParams | None]:
		#Skips jobs needing newer code than this worker's (MinVersion), so a worker that wasn't restarted can't run them wrongly
		version_ok = f"(MinVersion IS NULL OR {self.MIN_VERSION_SQL} <= :myversion)"
		claim_sql = f"WITH NextJob AS (SELECT TOP (1) Processing, WorkerName, StartedProcessingAt, QueueID, ParamsJson FROM {self.QUEUE_TABLE} WITH (UPDLOCK, ROWLOCK, READPAST) WHERE Processing = 0 AND {version_ok} ORDER BY QueueID ASC) UPDATE NextJob SET Processing = 1, StartedProcessingAt = GETDATE(), WorkerName = :worker OUTPUT INSERTED.QueueID, INSERTED.ParamsJson;"
		rows = self.db.ExecSQL(claim_sql, {"worker": self.worker_name, "myversion": self.version_number(self.codeVersion)})
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