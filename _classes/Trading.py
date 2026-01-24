import time
import numpy as np, pandas as pd
from tqdm import tqdm
from math import floor
from datetime import datetime, timedelta
from _classes.DataIO import PTADatabase
from _classes.Prices import PriceSnapshot, PricingData
from _classes.Utility import *

class Tranche: #interface for handling actions on a chunk of funds
	ticker = ''
	size = 0
	units = 0
	available = True
	purchased = False
	marketOrder = False
	sold = False
	expired = False
	dateBuyOrderPlaced = None
	dateBuyOrderFilled = None
	dateSellOrderPlaced = None
	dateSellOrderFilled = None
	buyOrderPrice = 0
	purchasePrice = 0
	sellOrderPrice = 0
	sellPrice = 0
	latestPrice = 0
	expireAfterDays = 0
	_verbose = False
	
	def __init__(self, size:int=1000):
		self.size = size
		
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
	def Sell(self, price:float, datePlaced:datetime, marketOrder:bool=False, expireAfterDays:int=90): self._t.PlaceSell(price=price, datePlaced=datePlaced, marketOrder=marketOrder, expireAfterDays=expireAfterDays, verbose=True)
	def SellPending(self): return (self._t.sellOrderPrice >0) and not (self._t.sold or  self._t.expired)
	def LatestPrice(self): return self._t.latestPrice

class Portfolio:
	def __init__(self, portfolioName:str, startDate:datetime, totalFunds:int=10000, trancheSize:int=1000, trackHistory:bool=True, useDatabase:bool=True, verbose:bool=False):
		self.tradeHistory = None #DataFrame of trades.  Note: though you can trade more than once a day it is only going to keep one entry per day per stock
		self.dailyValue = None	  #DataFrame for the value at the end of each day
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
		#for t in self._tranches:						self.CheckOrders(t.ticker, t.latestPrice, currentDate) 

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
							if self._verbose: print(t.ticker, " purchased for ",price, dateChecked)
							if (fundsavailable - actualCost - self._commisionCost) < 25:	#insufficient funds
								unitsCanAfford = max(floor((fundsavailable - self._commisionCost)/price)-1, 0)
								if self._verbose:
									print(' Ajusting units on market order for ' + ticker + ' Price: ', price, ' Requested Units: ', t.units,  ' Can afford:', unitsCanAfford)
									print(' Cash: ', self._cash, ' Committed Funds: ', self._fundsCommittedToOrders, ' Available: ', fundsavailable)
								if unitsCanAfford ==0:
									t.Recycle()
								else:
									t.AdjustBuyUnits(unitsCanAfford)
						if t.units == 0:
							if self._verbose: print( 'Can not afford any ' + ticker + ' at market ' + str(price) + ' canceling Buy', dateChecked)
							t.Recycle()
						else:
							self._cash = self._cash - (t.units*price) - self._commisionCost 
							if self._verbose and self._commisionCost > 0: print(' Commission charged for Buy: ' + str(self._commisionCost))		
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
		x = positions.index.to_numpy() + ':' + positions['Percentage'].to_numpy(dtype=str)
		for i in range(len(x)): x[i] = x[i][:12]
		while len(x) < 11: x = np.append(x, [''])
		self.dailyValue.loc[self.currentDate]=[_cashValue,assetValue,_cashValue + assetValue, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10]]
		#print(self.dailyValue)

	#--------------------------------------  Closing Reporting ---------------------------------------
	def SaveTradeHistory(self, foldername:str, addTimeStamp:bool = False):
		if self.trackHistory:
			if self.useDatabase:
				if self.database.Open():
					df = self.tradeHistory
					df['TradeModel'] = self.portfolioName 
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
				df['TradeModel'] = self.portfolioName 
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
		self.startingValue = 0 
		self.verbose = False
		self._tickerList = []	#list of stocks currently held
		self._dataFolderTradeModel = 'data/trademodel/'
		self.Custom1 = None	#can be used to store custom values when using the model
		self.Custom2 = None
		self._NormalizePrices = False
		self.useFullStats = False
		self.startingValue = totalFunds
		startDate = ToDateTime(startDate)
		endDate = startDate + timedelta(days=365 * durationInYears)
		CreateFolder(self._dataFolderTradeModel)
		self.modelName = modelName
		self.database = None
		self.days_passed = 0
		total_days = (endDate - startDate).days    	
		self.pbar = tqdm(total=total_days, desc=f"Running {modelName}", unit="day")
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
			self.useFullStats = useFullStats
			p.CalculateStats(fullStats=self.useFullStats)
			valid_dates = p.historicalPrices.index[(p.historicalPrices.index >= startDate) & (p.historicalPrices.index <= endDate)]
			if not valid_dates.empty:
				self.priceHistory = [p] #add to list
				self.modelStartDate = valid_dates[0]
				self.modelEndDate = valid_dates[-1]
				self.currentDate = self.modelStartDate
				self._tickerList = [startingTicker]
				self.modelReady = len(p.historicalPrices) > 30
			else:
				self.modelReady = False #We don't have enough data
		super(TradingModel, self).__init__(portfolioName=modelName, startDate=startDate, totalFunds=totalFunds, trancheSize=trancheSize, trackHistory=trackHistory, useDatabase=useDatabase, verbose=verbose)
		
	def __del__(self):
		self._tickerList = None
		del self.priceHistory[:] 
		self.priceHistory = None
		self.modelStartDate  = None	
		self.modelEndDate = None
		self.modelReady = False

	def Addticker(self, ticker:str):
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
				print(' Addticker: Added ticker ' + ticker)
			else:
				print( ' Addticker: Unable to download price history for ticker ' + ticker)
		return r

	def AlignPositions(self, targetPositions:pd.DataFrame, rateLimitTransactions:bool=False, shopBuyPercent:int=0, shopSellPercent:int=0, trimProfitsPercent:int=0, verbose:bool=False): 
		#Performs necessary Buy/Sells to get from current positions to target positions
		#Input ['Ticker']['TargetHoldings'] combo which indicates proportion of desired holdings
		#rateLimitTransactions will limit number of buys/sells per day to one per ticker
		#if not tradeAtMarket then will shop for a good buy and sell price, so far all attempts at shopping or trimming profits yield 3%-13% less average profit

		expireAfterDays=3
		tradeAtMarket = (shopBuyPercent ==0) and (shopSellPercent ==0) 
		TotalTranches = self._trancheCount
		TotalTargets = targetPositions['TargetHoldings'].sum()  #Sum the TargetHoldings, allocate by Rount(TotalTranches/TargetHoldings)
		scale = 1
		if TotalTargets > 0:
			scale = TotalTranches/TotalTargets
			targetPositions.loc[:, 'TargetHoldings'] = (targetPositions['TargetHoldings'] * scale).astype(float).round()
			#targetPositions.TargetHoldings = (targetPositions.TargetHoldings * scale).astype(float).round() 
		if verbose:
			print('AlignPositions: Target Positions Scaled')
			print(' Scale (TotalTargets, TotalTranches, Scale):', TotalTargets, TotalTranches, scale)
			print(targetPositions)
		currentPositions = self.GetPositions(asDataFrame=True)
		
		#evaluate the difference between current holdings and target, act accordingly, sorts sells ascending, buys descending
		targetPositions = targetPositions.join(currentPositions, how='outer')
		targetPositions.fillna(value=0, inplace=True)				
		if len(currentPositions) ==0: #no current positions
			targetPositions['Difference'] = targetPositions['TargetHoldings'] 
		else:
			targetPositions['Difference'] = targetPositions['TargetHoldings'] - targetPositions['CurrentHoldings']
		sells = targetPositions[targetPositions['Difference'] < 0].copy()
		sells.sort_values(by=['Difference'], axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last') 
		buys = targetPositions[targetPositions['Difference'] >= 0].copy()
		buys.sort_values(by=['Difference'], axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') 
		if verbose: print(self.currentDate)
		if verbose: print('AlignPositions: Target Positions with Current Positions')
		targetPositions = pd.concat([sells, buys]) #Re-ordered sells ascending, buys descending
		if verbose: print(targetPositions)
		for i in range(len(targetPositions)): #for each ticker
			orders = int(targetPositions.iloc[i]['Difference'])
			t = targetPositions.index.values[i]
			if t != 'CASH':
				sn = self.GetPriceSnapshot(t)
				if sn != None:
					price = sn.Average
					target_buy = round(min(sn.Average, sn.Target)*(1-shopBuyPercent/100),2)
					target_sell = round(max(sn.Average, sn.Target)*(1+shopSellPercent/100), 2)
					trim_sell = round(max(sn.Average, sn.Target)*(1+trimProfitsPercent/100), 2)
					#print(f"Ticker: {t} Average: {sn.Average} Target: {sn.Target} Range: {target_buy} to {target_sell} ")
					if orders < 0:
						if not tradeAtMarket: 
							price = target_sell
							#if abs(orders) > 2: price = sn.Average
						if rateLimitTransactions and abs(orders) > 1: orders = -1
						print(f"Sell {t} for ${price} vs ${sn.Average}")
						for _ in range(abs(orders)): 
							self.PlaceSell(ticker=t, price=price, marketOrder=tradeAtMarket, expireAfterDays=expireAfterDays, verbose=verbose)
					elif orders > 0 and self.TranchesAvailable() > 0:
						if not tradeAtMarket: 
							price = target_buy
							#if abs(orders) > 1: price = sn.Target
							#if abs(orders) > 3: price = sn.Average
						if rateLimitTransactions and abs(orders) > 1: orders = 1
						print(f"Buy {t} for ${price} vs ${sn.Average}")
						for _ in range(orders):
							self.PlaceBuy(ticker=t, price=price, marketOrder=tradeAtMarket, expireAfterDays=expireAfterDays, verbose=verbose)											
					elif trimProfitsPercent > 0:
						price = trim_sell
						print(f"Sell for trim profits {t} for ${price} vs ${sn.Average}")
						self.PlaceSell(ticker=t, price=price, marketOrder=False, expireAfterDays=expireAfterDays, verbose=verbose)			
					self.ProcessDay(withIncrement=False)
		self.ProcessDay(withIncrement=False)
		if verbose: print(self.PositionSummary())

	def CancelAllOrders(self): super(TradingModel, self).CancelAllOrders(self.currentDate)
	
	def CloseModel(self, plotResults:bool=True, saveHistoryToFile:bool=True, folderName:str='data/trademodel/', dpi:int=600):	
		cashValue, assetValue = self.Value()
		self.CancelAllOrders()
		if assetValue > 0:
			self.SellAllPositions(self.currentDate, allowWeekEnd=True)
		self.UpdateDailyValue()
		cashValue, assetValue = self.Value()
		netChange = cashValue + assetValue - self.startingValue 
		if self.pbar: self.pbar.close()
		if saveHistoryToFile:
			self.SaveDailyValue(folderName)
			self.SaveTradeHistory(folderName)
		print('Model ' + self.modelName + ' from ' + str(self.modelStartDate)[:10] + ' to ' + str(self.modelEndDate)[:10])
		print('Cash: ' + str(round(cashValue)) + ' asset: ' + str(round(assetValue)) + ' total: ' + str(round(cashValue + assetValue)))
		print('Net change: ' + str(round(netChange)), str(round((netChange/self.startingValue) * 100, 2)) + '%')
		print('')
		if plotResults and self.trackHistory: 
			self.PlotTradeHistoryAgainstHistoricalPrices(self.tradeHistory, self.priceHistory[0].GetPriceHistory(), self.modelName)
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
			if not ticker in self._tickerList:	self.Addticker(ticker)
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
			if not ticker in self._tickerList:	self.Addticker(ticker)
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
		if not ticker in self._tickerList: self.Addticker(ticker)	
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

	def ProcessDay(self, withIncrement:bool=True, allowWeekEnd:bool=False):
		#Process current day and increment the current date, allowWeekEnd is for model closing only
		if self.currentDate.weekday() < 5 or allowWeekEnd:
			for ph in self.priceHistory:
				sn = ph.GetPriceSnapshot(self.currentDate)
				self.ProcessDaysOrders(ph.ticker, sn.Open, sn.High, sn.Low, sn.Close, self.currentDate)
		self.UpdateDailyValue()
		if self.pbar and self.verbose:
			c = self._cash
			a = self.assetValue
			tqdm.write(f"Model: {self.modelName} Date: {self.currentDate} Cash: ${int(c)} Assets: ${int(a)} Total: ${int(c+a)} Return: {round(100*(((c+a)/self._initialValue)-1), 2)}%")
		self.ReEvaluateTrancheCount()
		if withIncrement and self.currentDate <= self.modelEndDate:
			idx = self.priceHistory[0].historicalPrices.index		
			pos = idx.searchsorted(self.currentDate, side='right')
			if self.pbar: self.pbar.update(self.days_passed)
			self.days_passed+=1
			if pos < len(idx):
				self.currentDate = idx[pos]
			else:
				self.currentDate = self.modelEndDate
	
	def SetCustomValues(self, v1, v2):
		self.Custom1 = v1
		self.custom2 = v2
		
class ForcastModel():	#used to forecast the effect of a series of trade actions, one per day, and return the net change in value.  This will mirror the given model.  Can also be used to test alternate past actions 
	def __init__(self, mirroredModel:TradingModel, daysToForecast:int = 10):
		modelName = 'Forcaster for ' + mirroredModel.modelName
		self.daysToForecast = daysToForecast
		self.daysToForecast = daysToForecast
		self.startDate = mirroredModel.modelStartDate 
		durationInYears = (mirroredModel.modelEndDate-mirroredModel.modelStartDate).days/365
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
		

