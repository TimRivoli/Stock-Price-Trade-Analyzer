#Each trading function should define what actions should be taken in the given day from the TradingModel, Buy/Sell/Hold, given the current and recent price information
#Then give it a date range, some money, and a stock, it will execute a strategy and return the results
#WARNING: "Far more money has been lost by investors preparing for corrections or trying to anticipate corrections than has been lost in corrections themselves." - Peter Lynch.
import pandas as pd
from _classes.Graphing import PlotHelper
from _classes.PriceTradeAnalyzer import TradingModel, PriceSnapshot, Position
from _classes.Utility import *

#------------------------------------------- Global model functions  ----------------------------------------------		
def RecordPerformance(ModelName, StartDate, EndDate, StartValue, EndValue, TradeCount, Ticker):
	#Record trade performance to output file, each model run will append to the same files so you can easily compare the results
	filename = 'data/trademodel/PerfomanceComparisons.csv'
	try:
		if FileExists(filename):
			f = open(filename,"a")
		else:
			f = open(filename,"w+")
			f.write('ModelName, StartDate, EndDate, StartValue, EndValue, TotalPercentageGain, TradeCount, Ticker, TimeStamp\n')
		TotalPercentageGain = (EndValue/StartValue)-1
		f.write(ModelName + ',' + str(StartDate) + ',' + str(EndDate) + ',' + str(StartValue) + ',' + str(EndValue) + ',' + str(TotalPercentageGain) + ',' + str(TradeCount) + ',' + Ticker + ',' + str(GetLatestBDay()) + '\n')
		f.close() 
	except:
		print('Unable to write performance report to ' + filename)

def RunModel(modelName:str, modelFunction, ticker:str, startDate:str, durationInYears:int, portfolioSize:int, saveHistoryToFile:bool=True, returndailyValues:bool=False, verbose:bool=False):	
	#Performs the logic of the given model over a period of time to evaluate the performance
	modelName = modelName + '_' + ticker
	print('Running model ' + modelName)
	tm = TradingModel(modelName=modelName, startingTicker=ticker, startDate=startDate, durationInYears=durationInYears, totalFunds=portfolioSize, verbose=verbose)
	startDate = tm.modelStartDate
	endDate = tm.modelEndDate
	if not tm.modelReady:
		print('Unable to initialize price history for model for ' + str(startDate))
		if returndailyValues: return pd.DataFrame()
		else:return portfolioSize
	else:
		while not tm.ModelCompleted():
			modelFunction(tm, ticker)
			tm.ProcessDay()
		cash, asset = tm.GetValue()
		tradeCount = len(tm.tradeHistory)
		ticker = tm.priceHistory[0].ticker
		RecordPerformance(ModelName=modelName, StartDate=startDate, EndDate=tm.currentDate, StartValue=portfolioSize, EndValue=(cash + asset), TradeCount=tradeCount, Ticker=ticker)
		closing_value = tm.CloseModel()
		if returndailyValues:
			return tm.GetDailyValue()   							#return daily value for model comparisons
		else:
			return closing_value

def ExtendedDurationTest(modelName:str, modelFunction, ticker:str, portfolioSize:int=30000):
	#test the model over an extended range of periods and years, output result to .csv file
	TestResults = pd.DataFrame([['1/1/1982','1/1/1982',0,0]], columns=list(['StartDate','EndDate','Duration','EndingValue']))
	TestResults.set_index(['StartDate'], inplace=True)		
	for duration in range(1,10,2):
		for year in range(1982,2017):
			for month in range(1,12,3):
				startDate = str(month) + '/1/' + str(year)
				endDate = str(month) + '/1/' + str(year + duration)
				endValue = RunModel(modelName, modelFunction, ticker, startDate, duration, portfolioSize, saveHistoryToFile=False, returndailyValues=False, verbose=False)
				y = pd.DataFrame([[startDate,endDate,duration,endValue]], columns=list(['StartDate','EndDate','Duration','EndingValue']))
				TestResults = TestResults.append(y, ignore_index=True)
	TestResults.to_csv('data/trademodel/' + modelName + '_' + ticker +'_extendedTest.csv')
	print(TestResults)

def PlotModeldailyValue(modelName:str, modelFunction, ticker:str, startDate:str, durationInYears:int, portfolioSize:int=30000):
	#Plot daily returns of the given model
	m1 = RunModel(modelName, modelFunction, ticker, startDate, durationInYears, portfolioSize, saveHistoryToFile=True, returndailyValues=True, verbose=False)
	if m1.shape[0] > 0:
		print(m1)
		plot = PlotHelper()
		plot.PlotDataFrame(m1, modelName + ' Daily Value (' + ticker + ')', 'Date', 'Value') 

def CompareModels(modelOneName:str, modelOneFunction, modelTwoName:str, modelTwoFunction, ticker:str, startDate:str, durationInYears:int, portfolioSize:int=30000):
	#Compare two models to measure the difference in returns
	m1 = RunModel(modelOneName, modelOneFunction, ticker, startDate, durationInYears, portfolioSize, saveHistoryToFile=False, returndailyValues=True, verbose=False)
	m2 = RunModel(modelTwoName, modelTwoFunction, ticker, startDate, durationInYears, portfolioSize, saveHistoryToFile=False, returndailyValues=True, verbose=False)
	if m1.shape[0] > 0 and m2.shape[0] > 0:
		m1 = m1.join(m2, lsuffix='_' + modelOneName, rsuffix='_' + modelTwoName)
		plot = PlotHelper()
		plot.PlotDataFrame(m1, ticker + ' Model Comparison', 'Date', 'Value') 

#------------------------------------------- Your models go here ----------------------------------------------		
#Each trading function should define what actions should be taken in the given day from the TradingModel, Buy/Sell/Hold, given the current and recent price information

def RunTradingModelBuyHold(tm: TradingModel, ticker:str):
#Baseline model, buy and hold
	cash = tm.GetAvailableCash()
	price = tm.GetPrice(ticker)
	if price:
		units = int(cash/price)
		if units > 0:
			tm.PlaceBuy(ticker=ticker, units=units, price=price, marketOrder=True, expireAfterDays=5, verbose=False)

def RunTradingModelSeasonal(tm: TradingModel, ticker:str):
	#Buy in March, sell in September
	buy_date  = ToDateTime('03/15/' + str(tm.currentDate.year))
	sell_date = ToDateTime('9/10/' + str(tm.currentDate.year))
	#available_cash, buy_pending_count, sell_pending_count, long_position_count  = tm.GetPositionSummary()
	if (tm.currentDate >= sell_date) or (tm.currentDate <= buy_date):
		tm.SellAllPositions()
	else:
		cash = tm.GetAvailableCash()
		price = tm.GetPrice(ticker)			
		if price:
			units = int(cash/price)
			if units > 0:
				tm.PlaceBuy(ticker=ticker, units=units, price=price, marketOrder=True, expireAfterDays=3, verbose=False)

def RunTradingModelFirstHalfOfMonth(tm: TradingModel, ticker: str):
	# From Robert Ariel's observations, most gains are in the first half of the month
	buy_day = 25   # Buy at the end of the month, after the 25th
	sell_day = 15  # Sell mid month, after the 15th
	sn = tm.GetPriceSnapshot(ticker)
	if sn is None:
		return
	d = tm.currentDate.day
	if d >= buy_day or d < 3:
		cash = tm.GetAvailableCash()
		price = sn.High
		if price and price > 0:
			units = int(cash / price)
			if units > 0:
				tm.PlaceBuy(ticker=ticker, units=units, price=price, marketOrder=True, expireAfterDays=3, verbose=False)
	elif d >= sell_day:
		tm.SellAllPositions(ticker=ticker)
				
def RunTradingTestTrading(tm: TradingModel, ticker: str):
	# Test effect of just trading back and forth on profit
	sn = tm.GetPriceSnapshot(ticker)
	if sn is None:
		return
	low = sn.Low
	high = sn.High
	d = tm.currentDate.day
	available_cash, buy_pending_count, sell_pending_count, long_position_count = tm.GetPositionSummary()
	# Buy every 5th day
	if (d % 5) == 0:
		if buy_pending_count == 0 and tm.GetAvailableCash() > high:
			cash = tm.GetAvailableCash()
			price = high
			units = int(cash / price)
			if units > 0:
				tm.PlaceBuy(ticker=ticker, price=price, units=units, marketOrder=True, expireAfterDays=3)
	# Sell all other days
	else:
		if long_position_count > 0 and sell_pending_count == 0:
			total_units = sum(p.units for p in tm._positions if p.ticker == ticker and p.dateSellOrderPlaced is None)
			if total_units > 0:
				tm.PlaceSell(ticker=ticker, units=total_units, price=high, marketOrder=True, expireAfterDays=3)

def RunTradingModelTrending(tm: TradingModel, ticker: str):
	# Buy on positive trends, sell on negative trends
	minActionableSlope = 0.002
	prevTrendState, trendDuration = tm.GetCustomValues()
	if prevTrendState is None: prevTrendState = ''
	if trendDuration is None: trendDuration = 0
	p = tm.GetPriceSnapshot(ticker)
	if p is None:
		return
	available_cash, buy_pending_count, sell_pending_count, long_position_count = tm.GetPositionSummary()
	targetBuy = p.Target * (1 + p.Deviation_5Day / 2)
	targetSell = p.Target * (1 - p.Deviation_5Day / 2)
	trendState = prevTrendState

	# ---- Determine trend state ----
	if p.EMA_LongSlope >= minActionableSlope and p.EMA_ShortSlope >= minActionableSlope:
		trendState = '++'
	elif p.EMA_LongSlope >= minActionableSlope and p.EMA_ShortSlope < minActionableSlope:
		trendState = '+-'
	elif p.EMA_LongSlope < -minActionableSlope and p.EMA_ShortSlope < -minActionableSlope:
		trendState = '--'
	elif p.EMA_LongSlope < (-1 * minActionableSlope) and p.EMA_ShortSlope < (-1 * minActionableSlope):
		trendState = '-+'
	else:
		trendState = 'Flat'

	# ---- Execute trading action ----
	if trendState == '++':
		# go long aggressively
		if buy_pending_count == 0 and available_cash > p.High:
			units = int(available_cash / p.High)
			if units > 0:
				tm.PlaceBuy(ticker=ticker, price=targetBuy, units=units, marketOrder=True, expireAfterDays=3)
	elif trendState == '--':
		# liquidate
		if long_position_count > 0 and sell_pending_count == 0:
			total_units = sum(pos.units for pos in tm._positions if pos.ticker == ticker and pos.dateSellOrderPlaced is None)
			if total_units > 0:
				tm.PlaceSell(ticker=ticker, units=total_units, price=targetSell, marketOrder=True, expireAfterDays=3)
	elif trendState == '-+':
		# early recovery: buy if not already fully invested
		if buy_pending_count == 0 and available_cash > p.High:
			units = int(available_cash / p.High)
			if units > 0:
				tm.PlaceBuy(ticker=ticker, price=targetBuy, units=units, marketOrder=True, expireAfterDays=3)
	elif trendState == 'Flat':
		# mild swing trading behavior
		if p.Low > p.Channel_High:
			# overbought -> sell
			if long_position_count > 0 and sell_pending_count < 2:
				total_units = sum(pos.units for pos in tm._positions if pos.ticker == ticker and pos.dateSellOrderPlaced is None)
				if total_units > 0:
					tm.PlaceSell(ticker=ticker, units=total_units, price=targetSell, marketOrder=False, expireAfterDays=5)
		elif p.High < p.Channel_Low:
			# oversold -> buy
			if available_cash > p.High and buy_pending_count < 2:
				units = int((available_cash * 0.50) / p.High)
				if units > 0:
					tm.PlaceBuy(ticker=ticker, price=targetBuy, units=units, marketOrder=False, expireAfterDays=5)

	# ---- Store trend state duration ----
	if trendState == prevTrendState:
		trendDuration += 1
	else:
		trendDuration = 0
	tm.SetCustomValues(trendState, trendDuration)
					
def RunTradingModelSwingTrend(tm: TradingModel, ticker: str):
	# Combines Trending with Swing Trade in an attempt to increase profits during flat period in particular
	# 70% long, 30% play money
	minActionableSlope = 0.002

	prevTrendState, trendDuration = tm.GetCustomValues()
	if prevTrendState is None: prevTrendState = ''
	if trendDuration is None: trendDuration = 0

	p = tm.GetPriceSnapshot(ticker)
	if p is None:
		return

	available_cash, buy_pending_count, sell_pending_count, long_position_count = tm.GetPositionSummary()

	targetBuy = p.Target * (1 + p.Deviation_5Day / 2)
	targetSell = p.Target * (1 - p.Deviation_5Day / 2)

	trendState = prevTrendState

	def BuyPctCash(pct: float, price: float, market: bool, expire: int):
		cash = tm.GetAvailableCash()
		budget = cash * pct
		if price <= 0 or budget <= 0:
			return
		units = int(budget / p.High)
		if units > 0:
			tm.PlaceBuy(ticker=ticker, price=price, units=units, marketOrder=market, expireAfterDays=expire)

	def SellPctHoldings(pct: float, price: float, market: bool, expire: int):
		open_units = sum(pos.units for pos in tm._positions if pos.ticker == ticker and pos.dateSellOrderPlaced is None)
		if open_units <= 0:
			return
		units = int(open_units * pct)
		units = max(units, 1)
		units = min(units, open_units)
		tm.PlaceSell(ticker=ticker, units=units, price=price, marketOrder=market, expireAfterDays=expire)

	# ---------------- TREND LOGIC ----------------
	if p.EMA_LongSlope >= minActionableSlope and p.EMA_ShortSlope >= minActionableSlope:
		# ++ Positive trend, stay long
		trendState = '++'
		if buy_pending_count < 2 and tm.GetAvailableCash() > p.High:
			BuyPctCash(pct=0.35, price=targetBuy, market=True, expire=3)

	elif p.EMA_LongSlope >= minActionableSlope and p.EMA_ShortSlope < minActionableSlope:
		# +- Correction / early downturn, trim highs
		trendState = '+-'
		if p.Low > p.Channel_High:
			if sell_pending_count < 3 and long_position_count > 0:
				SellPctHoldings(pct=0.20, price=targetSell * 0.98, market=False, expire=3)

		elif p.Low < p.Channel_Low and p.High > p.Channel_Low:
			if sell_pending_count < 3 and long_position_count > 0:
				SellPctHoldings(pct=0.20, price=targetSell * 0.98, market=False, expire=3)

	elif p.EMA_LongSlope < -minActionableSlope and p.EMA_ShortSlope < -minActionableSlope:
		# -- Negative trend, get out
		trendState = '--'

		if p.High < p.Channel_Low:
			# oversold: maybe nibble
			if buy_pending_count < 2 and long_position_count < 3:
				BuyPctCash(pct=0.15, price=targetBuy * 0.95, market=False, expire=2)

		elif p.Low < p.Channel_Low and p.High > p.Channel_Low:
			# straddle low: do nothing
			pass

		else:
			# exit aggressively
			if sell_pending_count < 5 and long_position_count > 0:
				SellPctHoldings(pct=0.50, price=targetSell, market=True, expire=3)

			if trendDuration > 2 and sell_pending_count < 5 and long_position_count > 0:
				SellPctHoldings(pct=0.50, price=targetSell, market=True, expire=3)

	elif p.EMA_LongSlope < (-1 * minActionableSlope) and p.EMA_ShortSlope < (-1 * minActionableSlope):
		# -+ Bounce or early recovery
		trendState = '-+'

		if p.High < p.Channel_Low:
			if buy_pending_count < 2 and long_position_count < 4:
				BuyPctCash(pct=0.20, price=targetBuy * 0.95, market=False, expire=2)

		elif p.Low < p.Channel_Low and p.High > p.Channel_Low:
			if buy_pending_count < 2 and long_position_count < 4:
				BuyPctCash(pct=0.20, price=targetBuy * 0.95, market=False, expire=2)

	else:
		# Flat, aim for 70% long with swing buys
		trendState = 'Flat'

		if p.High < p.Channel_Low:
			if buy_pending_count < 4 and tm.GetAvailableCash() > p.High:
				BuyPctCash(pct=0.15, price=targetBuy, market=False, expire=5)

		# always try to maintain some long exposure
		if buy_pending_count < 3 and tm.GetAvailableCash() > p.High:
			BuyPctCash(pct=0.10, price=targetBuy, market=False, expire=5)

	# ---------------- STORE TREND DURATION ----------------
	if trendState == prevTrendState:
		trendDuration += 1
	else:
		trendDuration = 0
	tm.SetCustomValues(trendState, trendDuration)


def RunTradingModelSwingTrade(tm: TradingModel, ticker: str):
	# Breaks funds into four sets and attempts to make money trading the spread
	# Takes current trend into consideration
	minActionableSlope = 0.002
	prevTrendState, trendDuration = tm.GetCustomValues()
	if prevTrendState is None: prevTrendState = ''
	if trendDuration is None: trendDuration = 0
	p = tm.GetPriceSnapshot(ticker)
	if p is None:
		return
	available_cash, buy_pending_count, sell_pending_count, long_position_count = tm.GetPositionSummary()
	targetBuy = p.Target * (1 + p.Deviation_5Day / 2)
	targetSell = p.Target * (1 - p.Deviation_5Day / 2)
	trendState = prevTrendState
	marketBuy = False
	marketSell = False

	def BuyChunk(pctCash: float, price: float, market: bool, expire: int):
		cash = tm.GetAvailableCash()
		if cash <= 0 or price <= 0:
			return
		budget = cash * pctCash
		units = int(budget / p.High)
		if units > 0:
			tm.PlaceBuy(ticker=ticker, units=units, price=price, marketOrder=market, expireAfterDays=expire)

	def SellChunk(pctUnits: float, price: float, market: bool, expire: int):
		open_units = sum(pos.units for pos in tm._positions if pos.ticker == ticker and pos.dateSellOrderPlaced is None)
		if open_units <= 0:
			return
		units = int(open_units * pctUnits)
		units = max(units, 1)
		units = min(units, open_units)
		tm.PlaceSell(ticker=ticker, units=units, price=price, marketOrder=market, expireAfterDays=expire)

	# ---------------- DETERMINE TREND STATE ----------------
	if p.EMA_LongSlope >= minActionableSlope and p.EMA_ShortSlope >= minActionableSlope:
		# ++ Positive trend
		trendState = '++'
		marketBuy = True
		marketSell = False
		if p.Low > p.Channel_High:
			# overbought: allow selling chunks instead of buying
			marketBuy = False
		elif p.Low < p.Channel_Low:
			# very early in trend, buy aggressively
			marketBuy = True
	elif p.EMA_LongSlope >= minActionableSlope and p.EMA_ShortSlope < minActionableSlope:
		# +- Correction / downturn
		trendState = '+-'
		marketBuy = False
		marketSell = False
		if p.Low < p.Channel_Low and p.High > p.Channel_Low:
			# deep correction: buy
			marketBuy = True
	elif p.EMA_LongSlope < -minActionableSlope and p.EMA_ShortSlope < -minActionableSlope:
		# -- Negative trend
		trendState = '--'
		marketSell = True
		marketBuy = False
	elif p.EMA_LongSlope < (-1 * minActionableSlope) and p.EMA_ShortSlope < (-1 * minActionableSlope):
		# -+ Bounce / early recovery
		trendState = '-+'
		marketBuy = False
		marketSell = False
	else:
		# Flat
		trendState = 'Flat'
		marketBuy = False
		marketSell = False

	# ---------------- EXECUTE SWING ACTIONS ----------------
	# Target behavior: try to maintain ~4 chunks and trade around the target

	# Buy logic: allow up to 3 pending buys
	if buy_pending_count < 3 and tm.GetAvailableCash() > p.High:
		BuyChunk(pctCash=0.25, price=targetBuy, market=marketBuy, expire=3)
	# Sell logic: allow up to 3 pending sells
	if sell_pending_count < 3 and long_position_count > 0:
		SellChunk(pctUnits=0.25, price=targetSell, market=marketSell, expire=3)
	# ---------------- STORE TREND DURATION ----------------
	if trendState == prevTrendState:
		trendDuration += 1
	else:
		trendDuration = 0
	tm.SetCustomValues(trendState, trendDuration)

def TestAllModels(ticker:str, startDate:str, duration:int, portfolioSize:int=30000):
	RunModel('BuyAndHold', RunTradingModelBuyHold, ticker, startDate, duration, portfolioSize, verbose=False)
	RunModel('Seasonal', RunTradingModelSeasonal, ticker, startDate, duration, portfolioSize, verbose=False)
	RunModel('FirstHalfOfMonth', RunTradingModelFirstHalfOfMonth, ticker, startDate, duration, portfolioSize, verbose=False)
	RunModel('RunTradingTestTrading', RunTradingTestTrading, ticker, startDate, duration, portfolioSize, verbose=False)
	RunModel('Trending', RunTradingModelTrending, ticker, startDate, duration, portfolioSize, verbose=False)
	RunModel('SwingTrend', RunTradingModelSwingTrend, ticker, startDate, duration, portfolioSize, verbose=False)
	RunModel('SwingTrade', RunTradingModelSwingTrade, ticker, startDate, duration, portfolioSize, verbose=False)

def TestAllTickers(tickerList:list, startDate:str, duration:int, portfolioSize:int=30000):
	for ticker in tickerList:
		RunModel('Trending', RunTradingModelTrending, ticker, startDate, duration, portfolioSize, verbose=False)
		RunModel('Swing', RunTradingModelSwingTrade, ticker, startDate, duration, portfolioSize, verbose=False)

if __name__ == '__main__':
	startDate = '1/1/1999'
	duration = 10
	tickerList=['BAC','XOM','JNJ','GOOGL','F','MSFT'] 
	TestAllModels('BAC', startDate, duration)
	TestAllTickers(tickerList, startDate, duration)
	CompareModels('BuyHold',RunTradingModelBuyHold,'Trending', RunTradingModelTrending, '.INX','1/1/1987',20)
	RunModel('Seasonal', RunTradingModelSeasonal, ticker, startDate, duration, portfolioSize, verbose=False)
	PlotModeldailyValue('Trending',RunTradingModelTrending, 'GOOGL','1/1/2005',15)
	RunModel('BuyAndHold', RunTradingModelBuyHold, '.INX', '1/1/2020', 1, 100000, verbose=False)
	CompareModels('BuyHold',RunTradingModelBuyHold,'Seasonal', RunTradingModelSeasonal, '.INX','1/1/1990',30)



