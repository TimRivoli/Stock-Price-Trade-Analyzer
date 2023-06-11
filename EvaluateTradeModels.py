#Each trading function should define what actions should be taken in the given day from the TradingModel, Buy/Sell/Hold, given the current and recent price information
#Then give it a date range, some money, and a stock, it will execute a strategy and return the results
#WARNING: "Far more money has been lost by investors preparing for corrections or trying to anticipate corrections than has been lost in corrections themselves." - Peter Lynch.
import pandas as pd
from _classes.PriceTradeAnalyzer import TradingModel, PlotHelper, PriceSnapshot, Position
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
		f.write(ModelName + ',' + str(StartDate) + ',' + str(EndDate) + ',' + str(StartValue) + ',' + str(EndValue) + ',' + str(TotalPercentageGain) + ',' + str(TradeCount) + ',' + Ticker + ',' + str(GetTodaysDate()) + '\n')
		f.close() 
	except:
		print('Unable to write performance report to ' + filename)

def RunModel(modelName:str, modelFunction, ticker:str, startDate:str, durationInYears:int, portfolioSize:int, saveHistoryToFile:bool=True, returndailyValues:bool=False, verbose:bool=False):	
	#Performs the logic of the given model over a period of time to evaluate the performance
	modelName = modelName + '_' + ticker
	print('Running model ' + modelName)
	tm = TradingModel(modelName=modelName, startingTicker=ticker, startDate=startDate, durationInYears=durationInYears, totalFunds=portfolioSize, tranchSize=round(portfolioSize/10), verbose=verbose)
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
			if tm.AccountingError(): 
				print('Accounting error.  Negative cash balance.  Terminating model run.', tm.currentDate)
				tm.PositionSummary()
				#tm.PrintPositions()
				break
		cash, asset = tm.Value()
		#print('Ending Value: ', cash + asset, '(Cash', cash, ', Asset', asset, ')')
		tradeCount = len(tm.tradeHistory)
		ticker = tm.priceHistory[0].ticker
		RecordPerformance(ModelName=modelName, StartDate=startDate, EndDate=tm.currentDate, StartValue=portfolioSize, EndValue=(cash + asset), TradeCount=tradeCount, Ticker=ticker)

		if returndailyValues:
			tm.CloseModel(verbose, saveHistoryToFile)
			return tm.GetDailyValue()   							#return daily value for model comparisons
		else:
			return tm.CloseModel(verbose, saveHistoryToFile)		#return simple closing value to view net effect

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
	sn = tm.GetPriceSnapshot()
	if tm.verbose: print(sn.date, sn.Target_1Day)
	if not sn == None:
		for i in range(tm._tranchCount):
			available, buyPending, sellPending, longPositions = tm.PositionSummary()				
			if tm.TranchesAvailable() > 0 and tm.FundsAvailable() > sn.high: tm.PlaceBuy(ticker=ticker, price=sn.low, marketOrder=True)
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
		low = sn.low
		high = sn.high
		m = tm.currentDate.month
		for i in range(tm._tranchCount):
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
		low = sn.low
		high = sn.high
		d = tm.currentDate.day
		for i in range(tm._tranchCount):
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
		low = sn.low
		high = sn.high
		d = tm.currentDate.day
		for i in range(tm._tranchCount):
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
		targetBuy = p.Target_1Day * (1 + p.Deviation_5Day/2)
		targetSell = p.Target_1Day * (1 - p.Deviation_5Day/2)
		for i in range(tm._tranchCount):
			if p.EMA_LongSlope >= minActionableSlope and p.EMA_ShortSlope >= minActionableSlope:	 
				trendState='++' #++	Positive trend, 100% long
				if available > 0 :tm.PlaceBuy(ticker=ticker, price=targetBuy, marketOrder=True)	
			elif p.EMA_LongSlope >= minActionableSlope and p.EMA_ShortSlope < minActionableSlope:  
				trendState='+-' #+- Correction or early downturn, recent price drop
				if p.low > p.Channel_High:   #Over Bought
					pass
				elif p.low < p.Channel_Low and p.high > p. Channel_Low: #Deep correction
					pass
				else:
					pass
			elif p.EMA_LongSlope < -minActionableSlope and p.EMA_ShortSlope < -minActionableSlope: 
				trendState='--' #-- Negative trend, get out
				if p.high < p.Channel_Low: #Over sold
					pass
				elif p.low < p.Channel_Low and p.high > p.Channel_Low: #Low below channel, possible early up or continuation of trend
					pass
				tm.PlaceSell(ticker=ticker, price=targetSell, marketOrder=True)
			elif p.EMA_LongSlope < (-1 * minActionableSlope) and p.EMA_ShortSlope < (-1 * minActionableSlope): #-+ Bounce or early recovery
				trendState='-+' #Short term positive, long term not yet
				if p.high < p.Channel_Low: #Over sold
					pass
				elif p.low < p.Channel_Low and p.high > p.Channel_Low: #Straddle Low
					pass
				else:
					pass
				if available > 0 :tm.PlaceBuy(ticker=ticker, price=targetBuy, marketOrder=True)	
			else:																	    			 
				trendState='Flat' #flat, target buy and sell to pass the time
				if p.low > p.Channel_High:   #Over Bought, targeted sell
					if longPositions > 0 and sellPending < 2 :tm.PlaceSell(ticker=ticker, price=targetSell, marketOrder=False)	
				elif p.high < p.Channel_Low: #Over sold, targeted buy
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
		targetBuy = p.Target_1Day * (1 + p.Deviation_5Day/2)
		targetSell = p.Target_1Day * (1 - p.Deviation_5Day/2)
		for i in range(tm._tranchCount):
			if p.EMA_LongSlope >= minActionableSlope and p.EMA_ShortSlope >= minActionableSlope:	 
				trendState='++' #++	Positive trend, 100% long
				if p.low > p.Channel_High:   #Over Bought
					tm.PlaceBuy(ticker=ticker, price=targetBuy, marketOrder=True)	
				elif p.low < p.Channel_Low:	#Still early
					tm.PlaceBuy(ticker=ticker, price=targetBuy, marketOrder=True)	
				else:
					tm.PlaceBuy(ticker=ticker, price=targetBuy, marketOrder=True)	
			elif p.EMA_LongSlope >= minActionableSlope and p.EMA_ShortSlope < minActionableSlope:  
				trendState='+-' #+- Correction or early downturn, recent price drop
				if p.low > p.Channel_High:   #Over Bought, sell profit
					if sellPending < 3 and longPositions > 7: tm.PlaceSell(ticker=ticker, price=targetSell * .98, marketOrder=False, expireAfterDays=3)
				elif p.low < p.Channel_Low and p.high > p. Channel_Low: #Deep correction
					if sellPending < 3 and longPositions > 7: tm.PlaceSell(ticker=ticker, price=targetSell * .98, marketOrder=False, expireAfterDays=3)
				else:
					pass
			elif p.EMA_LongSlope < -minActionableSlope and p.EMA_ShortSlope < -minActionableSlope: #-- Negative trend, get out
				trendState='--'
				if p.high < p.Channel_Low: #Over sold
					if buyPending < 3 and longPositions < 6: tm.PlaceBuy(ticker, targetBuy * .95, False, 2)
				elif p.low < p.Channel_Low and p.high > p.Channel_Low: #Straddle Low, possible early up
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
				if p.high < p.Channel_Low: #Over sold
					if buyPending < 3 and longPositions < 6: tm.PlaceBuy(ticker, targetBuy * .95, False, 2)
				elif p.low < p.Channel_Low and p.high > p.Channel_Low: #Straddle Low
					if buyPending < 3 and longPositions < 6: tm.PlaceBuy(ticker, targetBuy * .95, False, 2)
				else:
					pass
			else:																	    			 #flat, aim for 70% long
				trendState='Flat'
				if p.low > p.Channel_High:   #Over Bought
					pass
				elif p.high < p.Channel_Low: #Over sold
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
		targetBuy = p.Target_1Day * (1 + p.Deviation_5Day/2)
		targetSell = p.Target_1Day * (1 - p.Deviation_5Day/2)
		MarketBuy = False
		MarketSell = False
		for i in range(tm._tranchCount):
			if p.EMA_LongSlope >= minActionableSlope and p.EMA_ShortSlope >= minActionableSlope:	 
				#++	Positive trend
				#Actions: stick with it, sell 1/4 if it gets too high, repurchase at 3% discount
				trendState='++' 
				MarketBuy = (longPositions < maxPositions * .7) 
				if p.low > p.Channel_High:   #Over Bought
					pass
				elif p.low < p.Channel_Low:	#Very early in trend, possibly not possible
					MarketBuy = True
				else:
					pass
			elif p.EMA_LongSlope >= minActionableSlope and p.EMA_ShortSlope < minActionableSlope:  
				#+- Correction or early downturn, recent price drop
				#Actions: wait for bounce, sell and repurchase near recent low, need to consider state of chunks
				trendState='+-' 
				if p.low > p.Channel_High:   #Over Bought, sell profit
					pass
				elif p.low < p.Channel_Low and p.high > p. Channel_Low: #Deep correction
					MarketBuy = True
				else:
					pass
			elif p.EMA_LongSlope < -minActionableSlope and p.EMA_ShortSlope < -minActionableSlope: 
				#-- Negative trend
				#Actions: wait for bounce, sell and repurchase near recent low, need to consider state of chunks
				trendState='--' 
				MarketSell = True
				if p.high < p.Channel_Low: #Over sold
					pass
				elif p.low < p.Channel_Low and p.high > p.Channel_Low: #Straddle Low, possible early up
					pass
				else:
					pass
			elif p.EMA_LongSlope < (-1 * minActionableSlope) and p.EMA_ShortSlope < (-1 * minActionableSlope): #-+ Bounce or early recovery
				#Short term positive, long term not yet, early return to trend, expect large upward movement to resume trend
				trendState='-+' 
				if p.high < p.Channel_Low: #Over sold
					pass
				elif p.low < p.Channel_Low and p.high > p.Channel_Low: #Straddle Low
					pass
				else:
					pass
			else:
				#Flat
				#Action: plot high/low and swing trade them
				trendState='Flat'
				if p.low > p.Channel_High:   #Over Bought
					pass
				elif p.high < p.Channel_Low: #Over sold
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

def TestAllModels(tickerList:str, startDate:str, duration:int, portfolioSize:int=30000):
	for ticker in tickerList:
		RunModel('BuyAndHold', RunTradingModelBuyHold, ticker, startDate, duration, portfolioSize, verbose=False)
		RunModel('RunTradingTestTrading', RunTradingTestTrading, ticker, startDate, duration, portfolioSize, verbose=False)
		#RunModel('Seasonal', RunTradingModelSeasonal, ticker, startDate, duration, portfolioSize, verbose=False)
		#RunModel('FirstHalfOfMonth', RunTradingModelFirstHalfOfMonth, ticker, startDate, duration, portfolioSize, verbose=False)
		#RunModel('Trending', RunTradingModelTrending, ticker, startDate, duration, portfolioSize, verbose=False)

def TestAllTickers(tickerList:list, startDate:str, duration:int, portfolioSize:int=30000):
	for ticker in tickerList:
		#RunModel('Trending', RunTradingModelTrending, ticker, startDate, duration, portfolioSize, verbose=False)
		RunModel('Swing', RunTradingModelSwingTrade, ticker, startDate, duration, portfolioSize, verbose=False)
	
if __name__ == '__main__':
	#tickerList=['GOOGL']
	startDate = '1/1/1999'
	duration = 20
	tickerList=['BAC','XOM','JNJ','GOOGL','F','MSFT'] 
	TestAllModels(tickerList, startDate, duration)
	TestAllTickers(tickerList, startDate, duration)
	CompareModels('BuyHold',RunTradingModelBuyHold,'Trending', RunTradingModelTrending, '.INX','1/1/1987',20)
	PlotModeldailyValue('Trending',RunTradingModelTrending, 'GOOGL','1/1/2005',15)
	RunModel('BuyAndHold', RunTradingModelBuyHold, '.INX', '1/1/2020', 1, 100000, verbose=False)



