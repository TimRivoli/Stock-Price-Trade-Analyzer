#Each trading function should define what actions should be taken in the given day from the TradingModel, Buy/Sell/Hold, given the current and recent price information
#Then give it a date range, some money, and a stock, it will execute a strategy and return the results
#WARNING: "Far more money has been lost by investors preparing for corrections or trying to anticipate corrections than has been lost in corrections themselves." - Peter Lynch.
import pandas as pd
import _classes.Constants as CONSTANTS
from _classes.Graphing import PlotHelper
from _classes.Prices import PricingData, PriceSnapshot
from _classes.Trading import TradingModel, TradeModelParams, Position
from _classes.Selection import StockPicker
from _classes.TickerLists import TickerLists
from _classes.Utility import *

#------------------------------------------- Global model functions  ----------------------------------------------		
def RunModel(ticker:str, modelFunction, params: TradeModelParams):
	#Performs the logic of the given model over a period of time to evaluate the performance
	modelName = params.modelName + '_' + ticker
	print('Running model ' + modelName)
	tm = TradingModel(modelName=modelName, startingTicker=ticker, startDate = params.startDate, durationInYears = params.durationInYears, totalFunds = params.portfolioSize, trancheSize=params.trancheSize, verbose=params.verbose)
	if not tm.modelReady:
		print('Unable to initialize price history for model for ' + str(params.startDate))
		return params.portfolioSize
	else:
		while not tm.ModelCompleted():
			modelFunction(tm, ticker)
			tm.ProcessDay()
			if tm.AccountingError(): 
				print('Accounting error.  Negative cash balance.  Terminating model run.', tm.currentDate)
				tm.PositionSummary()
				#tm.PrintPositions()
				break
		return tm.CloseModel(params)	#return simple closing value to view net effect, detail gets saved with saveHistoryToFile

def ExtendedDurationTest(ticker:str, modelFunction, params: TradeModelParams):
	#test the model over an extended range of periods and years, output result to .csv file
	TestResults = pd.DataFrame([['1/1/1982','1/1/1982',0,0]], columns=list(['StartDate','EndDate','Duration','EndingValue']))
	TestResults.set_index(['StartDate'], inplace=True)		
	params.verbose = False
	for duration in range(1,10,2):
		for year in range(1982,2017):
			for month in range(1,12,3):
				startDate = str(month) + '/1/' + str(year)
				endDate = str(month) + '/1/' + str(year + duration)
				params.startDate = startDate
				params.durationInYears = duration
				endValue = RunModel(ticker, modelFunction, params)
				y = pd.DataFrame([[startDate,endDate,duration,endValue]], columns=list(['StartDate','EndDate','Duration','EndingValue']))
				TestResults = TestResults.append(y, ignore_index=True)
	TestResults.to_csv('data/trademodel/' + modelName + '_' + ticker +'_extendedTest.csv')
	print(TestResults)

def CompareModels(modelOneName:str, modelOneFunction, modelTwoName:str, modelTwoFunction, ticker:str, params: TradeModelParams):
	#Compare two models to measure the difference in returns
	params.modelName = modelOneName
	m1 = RunModel(ticker, modelOneFunction, params)
	params.modelName = modelTwoName
	m2 = RunModel(ticker, modelTwoFunction, params)
	print(f"{modelOneName}:{m1} {modelTwoName}:{m2}")

#------------------------------------------- Your models go here ----------------------------------------------		
#Each trading function should define what actions should be taken in the given day from the TradingModel, Buy/Sell/Hold, given the current and recent price information

def ModelSP500(startDate: str = '1/1/2000', durationInYears:int = 10):
	#Baseline model to compare against.  Buy on day one, hold for the duration and then sell
	ticker = '.INX'
	modelName = 'ModelSP500_' + str(startDate)[:10]
	params = TradeModelParams()
	params.startDate = startDate
	params.durationInYears = durationInYears
	params.saveResults = True
	tm = TradingModel(modelName=modelName, startingTicker=ticker, startDate=params.startDate, durationInYears=params.durationInYears, totalFunds=params.portfolioSize, trancheSize=params.trancheSize, verbose=False)
	if not tm.modelReady:
		print(' ModelSP500: Unable to initialize price history for date ' + str(startDate))
		return 0
	else:
		dayCounter =0
		while not tm.ModelCompleted():
			if dayCounter ==0:
				i=0
				while tm.TranchesAvailable() and i < 100: 
					tm.PlaceBuy(ticker=ticker, price=1, marketOrder=True, expireAfterDays=5, verbose=params.verbose)
					i +=1
			dayCounter+=1
			if dayCounter >= params.reEvaluationInterval: dayCounter=0
			tm.ProcessDay()
		cash, asset = tm.Value()
		if params.verbose: print(' ModelSP500: Ending Value: ', cash + asset, '(Cash', cash, ', Asset', asset, ')')
		params = TradeModelParams()
		return tm.CloseModel(params)		

def RunTradingModelBuyHold(tm: TradingModel, ticker:str):
#Baseline model, buy and hold
	sn = tm.GetPriceSnapshot()
	if tm.verbose: print(sn.Date, sn.Target)
	if not sn == None:
		for i in range(tm._trancheCount):
			available, buyPending, sellPending, longPositions = tm.PositionSummary()				
			if tm.TranchesAvailable() > 0 and tm.FundsAvailable() > sn.High: tm.PlaceBuy(ticker=ticker, price=sn.Low, marketOrder=True)
			if available ==0: break

def RunTradingModelSeasonal(tm: TradingModel, ticker:str):
	#Buy in November, sell in May
	#BuyMonth = 11
	#SellMonth = 5

	#Buy in May, sell in November
	BuyMonth = 5
	SellMonth = 11

	#Buy in March, sell in February, does really poorly
	#BuyMonth = 3
	#SellMonth = 2

	sn = tm.GetPriceSnapshot()
	if not sn == None:
		low = sn.Low
		high = sn.High
		m = tm.currentDate.month
		for i in range(tm._trancheCount):
			available, buyPending, sellPending, longPositions = tm.PositionSummary()				
			if m >= SellMonth and m < BuyMonth:
				if longPositions > 0: 
					tm.PlaceSell(ticker=ticker, price=high, marketOrder=True)
				else:
					break
			else:
				if available > 0 and tm.FundsAvailable() > high: 
					tm.PlaceBuy(ticker=ticker, price=low, marketOrder=True)
				else:
					break

def RunTradingModelFirstHalfOfMonth(tm: TradingModel, ticker:str):
#From Robert Ariel's observations, most gains are in the first half of the month
	BuyDay = 25	 #Buy at the end of the month, after the 25th
	SellDay = 15 #Sell mid month, after the 15th	
	sn = tm.GetPriceSnapshot()
	if not sn == None:
		low = sn.Low
		high = sn.High
		d = tm.currentDate.day
		for i in range(tm._trancheCount):
			available, buyPending, sellPending, longPositions = tm.PositionSummary()				
			if d >= BuyDay or d < 3:
				if available > 0 and tm.FundsAvailable() > high: 
					tm.PlaceBuy(ticker=ticker, price=low, marketOrder=True)
				else:
					break
			elif d >= SellDay:
				if longPositions > 0:
					tm.PlaceSell(ticker=ticker, price=high, marketOrder=True)
				else:
					break
			else:
				break
				
def RunTradingTestTrading(tm: TradingModel, ticker:str):
#Test effect of just trading back and forth on profit
	sn = tm.GetPriceSnapshot()
	if not sn == None:
		low = sn.Low
		high = sn.High
		d = tm.currentDate.day
		for i in range(tm._trancheCount):
			available, buyPending, sellPending, longPositions = tm.PositionSummary()				
			if (d % 5) == 0:
				if available > 0 and tm.FundsAvailable() > high: 
					tm.PlaceBuy(ticker=ticker, price=low, marketOrder=True)
				else:
					break
			else:
				if longPositions > 0:
					tm.PlaceSell(ticker=ticker, price=high, marketOrder=True)
				else:
					break

def RunTradingModelTrending(tm: TradingModel, ticker:str):
	#This compares the slope of short term (6 day) and long term (18 day) exponential moving averages to determine buying opportunities.  Positive, negative, or flat slopes
	#trend states: ++,--,+-,-+,Flat
	#Buy on positive trends, sell on negative
	minActionableSlope = 0.002	#Less than this is considered flat
	prevTrendState, trendDuration = tm.GetCustomValues()
	if prevTrendState == None: prevTrendState = ''
	if trendDuration == None: trendDuration = 0
	p = tm.GetPriceSnapshot()
	if not p == None:
		available, buyPending, sellPending, longPositions = tm.PositionSummary()
		maxPositions = available + buyPending + sellPending + longPositions
		targetBuy = p.Target * (1 + p.Deviation_5Day/2)
		targetSell = p.Target * (1 - p.Deviation_5Day/2)
		for i in range(tm._trancheCount):
			if p.EMA_LongSlope >= minActionableSlope and p.EMA_ShortSlope >= minActionableSlope:	 
				trendState='++' #++	Positive trend, 100% long
				if available > 0 :tm.PlaceBuy(ticker=ticker, price=targetBuy, marketOrder=True)	
			elif p.EMA_LongSlope >= minActionableSlope and p.EMA_ShortSlope < minActionableSlope:  
				trendState='+-' #+- Correction or early downturn, recent price drop
				if p.Low > p.Channel_High:   #Over Bought
					pass
				elif p.Low < p.Channel_Low and p.High > p. Channel_Low: #Deep correction
					pass
				else:
					pass
			elif p.EMA_LongSlope < -minActionableSlope and p.EMA_ShortSlope < -minActionableSlope: 
				trendState='--' #-- Negative trend, get out
				if p.High < p.Channel_Low: #Over sold
					pass
				elif p.Low < p.Channel_Low and p.High > p.Channel_Low: #Low below channel, possible early up or continuation of trend
					pass
				tm.PlaceSell(ticker=ticker, price=targetSell, marketOrder=True)
			elif p.EMA_LongSlope < (-1 * minActionableSlope) and p.EMA_ShortSlope < (-1 * minActionableSlope): #-+ Bounce or early recovery
				trendState='-+' #Short term positive, long term not yet
				if p.High < p.Channel_Low: #Over sold
					pass
				elif p.Low < p.Channel_Low and p.High > p.Channel_Low: #Straddle Low
					pass
				else:
					pass
				if available > 0 :tm.PlaceBuy(ticker=ticker, price=targetBuy, marketOrder=True)	
			else:																	    			 
				trendState='Flat' #flat, target buy and sell to pass the time
				if p.Low > p.Channel_High:   #Over Bought, targeted sell
					if longPositions > 0 and sellPending < 2 :tm.PlaceSell(ticker=ticker, price=targetSell, marketOrder=False)	
				elif p.High < p.Channel_Low: #Over sold, targeted buy
					if available > 0 and buyPending < 2 :tm.PlaceBuy(ticker=ticker, price=targetBuy, marketOrder=False)	
				else:
					pass
		tm.SetCustomValues(trendState, trendDuration)
		if trendState == prevTrendState: 
			trendDuration = trendDuration + 1
		else:
			trendDuration=0
		tm.SetCustomValues(prevTrendState, trendDuration)
					
def RunTradingModelSwingTrend(tm: TradingModel, ticker:str):
	#Combines Trending with Swing Trade in an attempt to increase profits during flat period in particular
	#70% long, 30% play money
	#trend states: ++,--,+-,-+,Flat
	minActionableSlope = 0.002
	prevTrendState, trendDuration = tm.GetCustomValues()
	if prevTrendState == None: prevTrendState = ''
	if trendDuration == None: trendDuration = 0
	p = tm.GetPriceSnapshot()
	if not p == None:
		available, buyPending, sellPending, longPositions = tm.PositionSummary()
		maxPositions = available + buyPending + sellPending + longPositions
		targetBuy = p.Target * (1 + p.Deviation_5Day/2)
		targetSell = p.Target * (1 - p.Deviation_5Day/2)
		for i in range(tm._trancheCount):
			if p.EMA_LongSlope >= minActionableSlope and p.EMA_ShortSlope >= minActionableSlope:	 
				trendState='++' #++	Positive trend, 100% long
				if p.Low > p.Channel_High:   #Over Bought
					tm.PlaceBuy(ticker=ticker, price=targetBuy, marketOrder=True)	
				elif p.Low < p.Channel_Low:	#Still early
					tm.PlaceBuy(ticker=ticker, price=targetBuy, marketOrder=True)	
				else:
					tm.PlaceBuy(ticker=ticker, price=targetBuy, marketOrder=True)	
			elif p.EMA_LongSlope >= minActionableSlope and p.EMA_ShortSlope < minActionableSlope:  
				trendState='+-' #+- Correction or early downturn, recent price drop
				if p.Low > p.Channel_High:   #Over Bought, sell profit
					if sellPending < 3 and longPositions > 7: tm.PlaceSell(ticker=ticker, price=targetSell * .98, marketOrder=False, expireAfterDays=3)
				elif p.Low < p.Channel_Low and p.High > p. Channel_Low: #Deep correction
					if sellPending < 3 and longPositions > 7: tm.PlaceSell(ticker=ticker, price=targetSell * .98, marketOrder=False, expireAfterDays=3)
				else:
					pass
			elif p.EMA_LongSlope < -minActionableSlope and p.EMA_ShortSlope < -minActionableSlope: #-- Negative trend, get out
				trendState='--'
				if p.High < p.Channel_Low: #Over sold
					if buyPending < 3 and longPositions < 6: tm.PlaceBuy(ticker, targetBuy * .95, False, 2)
				elif p.Low < p.Channel_Low and p.High > p.Channel_Low: #Straddle Low, possible early up
					pass
				else:
					tm.PlaceSell(ticker=ticker, price=targetSell, marketOrder=True)
					if trendDuration > 2: 
						if sellPending < 5 and longPositions > 5: tm.PlaceSell(ticker=ticker, price=targetSell, marketOrder=True)
						if sellPending < 5 and longPositions > 0: tm.PlaceSell(ticker=ticker, price=targetSell, marketOrder=True)
				if sellPending < 5 and longPositions > 3:
					tm.PlaceSell(ticker=ticker, price=targetSell, marketOrder=True)
					tm.PlaceSell(ticker=ticker, price=targetSell, marketOrder=True)
			elif p.EMA_LongSlope < (-1 * minActionableSlope) and p.EMA_ShortSlope < (-1 * minActionableSlope): #-+ Bounce or early recovery
				trendState='-+' #Short term positive, long term not yet
				if p.High < p.Channel_Low: #Over sold
					if buyPending < 3 and longPositions < 6: tm.PlaceBuy(ticker, targetBuy * .95, False, 2)
				elif p.Low < p.Channel_Low and p.High > p.Channel_Low: #Straddle Low
					if buyPending < 3 and longPositions < 6: tm.PlaceBuy(ticker, targetBuy * .95, False, 2)
				else:
					pass
			else:																	    			 #flat, aim for 70% long
				trendState='Flat'
				if p.Low > p.Channel_High:   #Over Bought
					pass
				elif p.High < p.Channel_Low: #Over sold
					if buyPending < 3 and longPositions < 8: tm.PlaceBuy(ticker=ticker, price=targetBuy, marketOrder=False, expireAfterDays=5)
					if buyPending < 4: tm.PlaceBuy(ticker=ticker, price=targetBuy, marketOrder=False, expireAfterDays=5)
				else:
					pass
				if buyPending < 3 and longPositions < maxPositions: tm.PlaceBuy(ticker=ticker, price=targetBuy, marketOrder=False, expireAfterDays=5)
		tm.SetCustomValues(trendState, trendDuration)
		if trendState == prevTrendState: 
			trendDuration = trendDuration + 1
		else:
			trendDuration=0
		tm.SetCustomValues(prevTrendState, trendDuration)

def RunTradingModelSwingTrade(tm: TradingModel, ticker:str):
	#Breaks funds into four sets and attempts to make money trading the spread
	#Takes current trend into consideration
	#trend states: ++,--,+-,-+,Flat (normal, negative, correction, resumption, flat)
	
	#ActionableSpread = 3%
	#BigMovement = 2.5 x 5dev
	#trend states: 
	#Normal Trending, generally stick with it, trim highs
	#Flat, Consolidating
	#Correction, Initial shock, Bounce, New Low, Resuming: Track recent high/lows, approach of those values
		
	minActionableSlope = 0.002
	prevTrendState, trendDuration = tm.GetCustomValues()
	if prevTrendState == None: prevTrendState = ''
	if trendDuration == None: trendDuration = 0
	p = tm.GetPriceSnapshot()
	if not p == None:
		available, buyPending, sellPending, longPositions = tm.PositionSummary()
		maxPositions = available + buyPending + sellPending + longPositions
		targetBuy = p.Target * (1 + p.Deviation_5Day/2)
		targetSell = p.Target * (1 - p.Deviation_5Day/2)
		MarketBuy = False
		MarketSell = False
		for i in range(tm._trancheCount):
			if p.EMA_LongSlope >= minActionableSlope and p.EMA_ShortSlope >= minActionableSlope:	 
				#++	Positive trend
				#Actions: stick with it, sell 1/4 if it gets too high, repurchase at 3% discount
				trendState='++' 
				MarketBuy = (longPositions < maxPositions * .7) 
				if p.Low > p.Channel_High:   #Over Bought
					pass
				elif p.Low < p.Channel_Low:	#Very early in trend, possibly not possible
					MarketBuy = True
				else:
					pass
			elif p.EMA_LongSlope >= minActionableSlope and p.EMA_ShortSlope < minActionableSlope:  
				#+- Correction or early downturn, recent price drop
				#Actions: wait for bounce, sell and repurchase near recent low, need to consider state of chunks
				trendState='+-' 
				if p.Low > p.Channel_High:   #Over Bought, sell profit
					pass
				elif p.Low < p.Channel_Low and p.High > p. Channel_Low: #Deep correction
					MarketBuy = True
				else:
					pass
			elif p.EMA_LongSlope < -minActionableSlope and p.EMA_ShortSlope < -minActionableSlope: 
				#-- Negative trend
				#Actions: wait for bounce, sell and repurchase near recent low, need to consider state of chunks
				trendState='--' 
				MarketSell = True
				if p.High < p.Channel_Low: #Over sold
					pass
				elif p.Low < p.Channel_Low and p.High > p.Channel_Low: #Straddle Low, possible early up
					pass
				else:
					pass
			elif p.EMA_LongSlope < (-1 * minActionableSlope) and p.EMA_ShortSlope < (-1 * minActionableSlope): #-+ Bounce or early recovery
				#Short term positive, long term not yet, early return to trend, expect large upward movement to resume trend
				trendState='-+' 
				if p.High < p.Channel_Low: #Over sold
					pass
				elif p.Low < p.Channel_Low and p.High > p.Channel_Low: #Straddle Low
					pass
				else:
					pass
			else:
				#Flat
				#Action: plot high/low and swing trade them
				trendState='Flat'
				if p.Low > p.Channel_High:   #Over Bought
					pass
				elif p.High < p.Channel_Low: #Over sold
					pass
				else:
					pass
			if buyPending <= (maxPositions/3): tm.PlaceBuy(ticker=ticker, price=targetBuy, marketOrder=MarketBuy, expireAfterDays=3)	
			if sellPending <= (maxPositions/3): tm.PlaceSell(ticker=ticker, price=targetSell, marketOrder=MarketSell, expireAfterDays=3)	
		tm.SetCustomValues(trendState, trendDuration)
		if trendState == prevTrendState: 
			trendDuration = trendDuration + 1
		else:
			trendDuration=0
		tm.SetCustomValues(prevTrendState, trendDuration)	

def TestAllModels(tickerList:list, params: TradeModelParams):
	for ticker in tickerList:
		params.modelName = 'BuyAndHold'
		RunModel(ticker, RunTradingModelBuyHold, params)
		params.modelName = 'Seasonal'
		RunModel(ticker, RunTradingModelSeasonal, params)
		params.modelName = 'FirstHalfOfMonth'
		RunModel(ticker, RunTradingModelFirstHalfOfMonth, params)
		params.modelName = 'RunTradingTestTrading'
		RunModel(ticker, RunTradingTestTrading, params)
		params.modelName = 'Trending'
		RunModel(ticker, RunTradingModelTrending, params)
		params.modelName = 'SwingTrend'
		RunModel(ticker, RunTradingModelSwingTrend, params)
		params.modelName = 'SwingTrade'
		RunModel(ticker, RunTradingModelSwingTrade, params)

def TestAllTickers(tickerList:list, params: TradeModelParams):
	for ticker in tickerList:
		params.modelName = 'Swing'
		RunModel(ticker, RunTradingModelSwingTrade, params)
	
if __name__ == '__main__':
	params = TradeModelParams()
	params.startDate = '1/1/1999'
	params.durationInYears = 20
	params.reEvaluationInterval = 20
	params.stockCount = 10
	params.saveResults = True
	params.verbose = False

	ModelSP500(params.startDate, params.durationInYears)
	params.modelName = 'BuyAndHold'
	RunModel('XOM', RunTradingModelBuyHold, params=params)
	tickerList=['BAC','XOM','JNJ','GOOGL','F','MSFT'] 
	TestAllModels(tickerList, params=params)
	TestAllTickers(tickerList, params=params)
	CompareModels('BuyHold',RunTradingModelBuyHold,'Trending', RunTradingModelTrending, '.INX', params=params)



