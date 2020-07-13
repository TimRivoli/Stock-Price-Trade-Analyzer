#Each trading function should define what actions should be taken in the given day from the TradingModel, Buy/Sell/Hold, given the current and recent price information
#Then give it a date range, some money, and a stock, it will execute a strategy and return the results

#WARNING: "Far more money has been lost by investors preparing for corrections or trying to anticipate corrections  
#than has been lost in corrections themselves." - Peter Lynch.
import pandas as pd
from _classes.PriceTradeAnalyzer import TradingModel, PlotHelper, PriceSnapshot, Position

#------------------------------------------- Global model functions  ----------------------------------------------		
def RunModel(modelName:str, modelFunction, ticker:str, startDate:str, durationInYears:int, portfolioSize:int, saveHistoryToFile:bool=True, returndailyValues:bool=False, verbose:bool=False):	
	#Performs the logic of the given model over a period of time to evaluate the performance
	modelName = modelName + '_' + ticker
	print('Running model ' + modelName)
	tm = TradingModel(modelName=modelName, startingTicker=ticker, startDate=startDate, durationInYears=durationInYears, totalFunds=portfolioSize, tranchSize=round(portfolioSize/10), verbose=verbose)
	if not tm.modelReady:
		print('Unable to initialize price history for model for ' + str(startDate))
		if returndailyValues: return pd.DataFrame()
		else:return portfolioSize
	else:
		while not tm.ModelCompleted():
			modelFunction(tm, ticker)
			tm.ProcessDay()
			if tm.AccountingError(): 
				print('Accounting error.  The numbers do not add up correctly.  Terminating model run.', tm.currentDate)
				tm.PositionSummary()
				#tm.PrintPositions()
				break
		if returndailyValues:
			tm.CloseModel(verbose, saveHistoryToFile)
			return tm.GetDailyValue()   							#return daily value for model comparisons
		else:
			return tm.CloseModel(verbose, saveHistoryToFile)		#return simple closing value to view net effect

def ExtendedDurationTest(modelName:str, modelFunction, ticker:str, portfolioSize:int=10000):
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

def PlotModeldailyValue(modelName:str, modelFunction, ticker:str, startDate:str, durationInYears:int, portfolioSize:int=10000):
	#Plot daily returns of the given model
	m1 = RunModel(modelName, modelFunction, ticker, startDate, durationInYears, portfolioSize, saveHistoryToFile=True, returndailyValues=True, verbose=False)
	if m1.shape[0] > 0:
		plot = PlotHelper()
		plot.PlotDataFrame(m1, modelName + ' Daily Value (' + ticker + ')', 'Date', 'Value') 

def CompareModels(modelOneName:str, modelOneFunction, modelTwoName:str, modelTwoFunction, ticker:str, startDate:str, durationInYears:int, portfolioSize:int=10000):
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
	currentPrices = tm.GetPriceSnapshot()
	if tm.verbose: print(currentPrices.snapShotDate, currentPrices.nextDayTarget)
	if not currentPrices == None:
		for i in range(tm._tranchCount):
			available, buyPending, sellPending, longPositions = tm.PositionSummary()				
			if tm.TranchesAvailable() > 0 and tm.FundsAvailable() > currentPrices.high: tm.PlaceBuy(ticker, currentPrices.low, True)
			if available ==0: break

def RunTradingModelSeasonal(tm: TradingModel, ticker:str):
#Buy in November, sell in May
	SellMonth = 5
	BuyMonth = 11
	currentPrices = tm.GetPriceSnapshot()
	if not currentPrices == None:
		low = currentPrices.low
		high = currentPrices.high
		m = tm.currentDate.month
		for i in range(tm._tranchCount):
			available, buyPending, sellPending, longPositions = tm.PositionSummary()				
			if m >= SellMonth and m <= BuyMonth:
				if longPositions > 0: 
					tm.PlaceSell(ticker, high, True)
				else:
					break
			else:
				if available > 0 and tm.FundsAvailable() > high: 
					tm.PlaceBuy(ticker, low, True)
				else:
					break

def RunTradingModelFirstHalfOfMonth(tm: TradingModel, ticker:str):
#From Robert Ariel's observations, most gains are in the first half of the month
	BuyDay = 25	 #Buy at the end of the month, after the 25th
	SellDay = 15 #Sell mid month, after the 15th	
	currentPrices = tm.GetPriceSnapshot()
	if not currentPrices == None:
		low = currentPrices.low
		high = currentPrices.high
		d = tm.currentDate.day
		for i in range(tm._tranchCount):
			available, buyPending, sellPending, longPositions = tm.PositionSummary()				
			if d >= BuyDay or d < 3:
				if available > 0 and tm.FundsAvailable() > high: 
					tm.PlaceBuy(ticker, low, True)
				else:
					break
			elif d >= SellDay:
				if longPositions > 0:
					tm.PlaceSell(ticker, high, True)
				else:
					break
			else:
				break
				
def RunTradingModelTrending(tm: TradingModel, ticker:str):
	#This compares the slope of short term (6 day) and long term (18 day) exponential moving averages to determine buying opportunities.  Positive, negative, or flat slopes
	#trend states: ++,--,+-,-+,Flat
	minActionableSlope = 0.002
	prevTrendState, trendDuration = tm.GetCustomValues()
	if prevTrendState == None: prevTrendState = ''
	if trendDuration == None: trendDuration = 0
	p = tm.GetPriceSnapshot()
	if not p == None:
		available, buyPending, sellPending, longPositions = tm.PositionSummary()
		maxPositions = available + buyPending + sellPending + longPositions
		targetBuy = p.nextDayTarget * (1 + p.fiveDayDeviation/2)
		targetSell = p.nextDayTarget * (1 - p.fiveDayDeviation/2)
		for i in range(tm._tranchCount):
			if p.longEMASlope >= minActionableSlope and p.shortEMASlope >= minActionableSlope:	 #++	Positive trend, 100% long
				trendState='++'
				if p.low > p.channelHigh:   #Over Bought
					pass
				elif p.low < p.channelLow:	#Still early
					if buyPending < 3 and longPositions < 6: tm.PlaceBuy(ticker, targetBuy, True)	
					if trendDuration > 1 and  buyPending < 3: tm.PlaceBuy(ticker, targetBuy, True)	
				else:
					if buyPending < 3 and longPositions < 6: tm.PlaceBuy(ticker, targetBuy, True)
				if buyPending < 5 and longPositions < maxPositions: tm.PlaceBuy(ticker, targetBuy, False, 3)
			elif p.longEMASlope >= minActionableSlope and p.shortEMASlope < minActionableSlope:  #+- Correction or early downturn
				trendState='+-'
				if p.low > p.channelHigh:   #Over Bought, try to get out
					if sellPending < 3 and longPositions > 7: tm.PlaceSell(ticker, targetSell * .98, False,3)
				elif p.low < p.channelLow and p.high > p. channelLow: #Deep correction
					if sellPending < 3 and longPositions > 7: tm.PlaceSell(ticker, targetSell, False,3)
				else:
					pass
			elif p.longEMASlope < -minActionableSlope and p.shortEMASlope < -minActionableSlope: #-- Negative trend, get out
				trendState='--'
				if p.high < p.channelLow: #Over sold
					if buyPending < 3 and longPositions < 6: tm.PlaceBuy(ticker, targetBuy * .95, False, 2)
				elif p.low < p.channelLow and p.high > p.channelLow: #Straddle Low, possible early up
					pass
				else:
					if trendDuration > 2: 
						if sellPending < 5 and longPositions > 5:tm.PlaceSell(ticker, targetSell, True)
						if sellPending < 5 and longPositions > 0:tm.PlaceSell(ticker, targetSell, True)
				if sellPending < 5 and longPositions > 3:
					tm.PlaceSell(ticker, targetSell, False, 2)
					tm.PlaceSell(ticker, targetSell, False, 2)
			elif p.longEMASlope < (-1 * minActionableSlope) and p.shortEMASlope < (-1 * minActionableSlope): #-+ Bounce or early recovery
				trendState='-+'
				if p.high < p.channelLow: #Over sold
					if buyPending < 3 and longPositions < 6: tm.PlaceBuy(ticker, targetBuy * .95, False, 2)
				elif p.low < p.channelLow and p.high > p.channelLow: #Straddle Low
					if buyPending < 3 and longPositions < 6: tm.PlaceBuy(ticker, targetBuy * .95, False, 2)
				else:
					pass
			else:																	    			 #flat, aim for 70% long
				trendState='Flat'
				if p.low > p.channelHigh:   #Over Bought
					pass
				elif p.high < p.channelLow: #Over sold
					if buyPending < 3 and longPositions < 8: tm.PlaceBuy(ticker, targetBuy, False,5)
					if buyPending < 4: tm.PlaceBuy(ticker, targetBuy, False,5)
				else:
					pass
				if buyPending < 3 and longPositions < maxPositions: tm.PlaceBuy(ticker, targetBuy, False,5)

		tm.SetCustomValues(trendState, trendDuration)
		if trendState == prevTrendState: 
			trendDuration = trendDuration + 1
		else:
			trendDuration=0
		tm.SetCustomValues(prevTrendState, trendDuration)
					
def RunTradingModelSwingTrend(tm: TradingModel, ticker:str):
	#Combines trending model with targeted "swing" buys, attempting to gain better deals by anticipating daily price fluctuations
	minActionableSlope = 0.002
	prevTrendState, trendDuration = tm.GetCustomValues()
	if prevTrendState == None: prevTrendState = ''
	if trendDuration == None: trendDuration = 0
	p = tm.GetPriceSnapshot()
	if not p == None:
		for i in range(tm._tranchCount):
			available, buyPending, sellPending, longPositions = tm.PositionSummary()
			maxPositions = available + buyPending + sellPending + longPositions
			targetBuy = p.nextDayTarget * (1 + p.fiveDayDeviation/2)
			targetSell = p.nextDayTarget * (1 - p.fiveDayDeviation/2)
			if p.longEMASlope >= minActionableSlope and p.shortEMASlope >= minActionableSlope:	 #++	Positive trend, 70% long
				trendState='++'
				if p.low > p.channelHigh:   #Over Bought
					if sellPending < 3 and longPositions > 7: tm.PlaceSell(ticker, targetSell * (1.03), False,10)
				elif p.low < p.channelLow:	#Still early
					if buyPending < 3 and longPositions < 6: tm.PlaceBuy(ticker, targetBuy, True)	
					if trendDuration > 1 and  buyPending < 3: tm.PlaceBuy(ticker, targetBuy, True)	
				else:
					if buyPending < 3 and longPositions < 6: tm.PlaceBuy(ticker, targetBuy, False)
				if buyPending < 5 and longPositions < maxPositions: tm.PlaceBuy(ticker, targetBuy, False)
			elif p.longEMASlope >= minActionableSlope and p.shortEMASlope < minActionableSlope:  #+- Correction or early downturn
				trendState='+-'
				if p.low > p.channelHigh:   #Over Bought, try to get out
					if sellPending < 3 and longPositions > 7: tm.PlaceSell(ticker, targetSell, False,3)
				elif p.low < p.channelLow and p.high > p. channelLow: #Deep correction
					if sellPending < 3 and longPositions > 7: tm.PlaceSell(ticker, targetSell, False,3)
				else:
					pass
			elif p.longEMASlope < -minActionableSlope and p.shortEMASlope < -minActionableSlope: #-- Negative trend, aim for < 30% long
				trendState='--'
				if p.high < p.channelLow: #Over sold
					if buyPending < 3 and longPositions < 6: tm.PlaceBuy(ticker, targetBuy * .95, False, 2)
				elif p.low < p.channelLow and p.high > p.channelLow: #Straddle Low, early down or up
					pass
				else:
					if sellPending < 5 and longPositions > 3: 
						tm.PlaceSell(ticker, targetSell, True)
						if trendDuration > 1: tm.PlaceSell(ticker, targetSell, True)
				if sellPending < 5 and longPositions > 3:
					tm.PlaceSell(ticker, targetSell, False, 2)
					tm.PlaceSell(ticker, targetSell, False, 2)
			elif p.longEMASlope < (-1 * minActionableSlope) and p.shortEMASlope < (-1 * minActionableSlope): #-+ Bounce or early recovery
				trendState='-+'
				if p.high < p.channelLow: #Over sold
					pass
				elif p.low < p.channelLow and p.high > p.channelLow: #Straddle Low
					if sellPending < 3 and longPositions > 3: tm.PlaceSell(ticker, targetSell, False,3)
				else:
					pass
			else:																				 #flat, aim for 70% long
				trendState='Flat'
				if p.low > p.channelHigh:   #Over Bought
					if sellPending < 3 and longPositions > 7: tm.PlaceSell(ticker, targetSell * (1.03), False,10)
				elif p.high < p.channelLow: #Over sold
					if buyPending < 3 and longPositions < 8: tm.PlaceBuy(ticker, targetBuy, False,5)
					if buyPending < 4: tm.PlaceBuy(ticker, targetBuy, False,5)
				else:
					pass
				if sellPending < 3 and longPositions > 7: tm.PlaceSell(ticker, targetSell, False,5)
				if buyPending < 3 and longPositions < maxPositions: tm.PlaceBuy(ticker, targetBuy, False,5)
			if trendState == prevTrendState: 
				trendDuration = trendDuration + 1
			else:
				trendDuration=0
		tm.SetCustomValues(prevTrendState, trendDuration)
		

if __name__ == '__main__':
	RunModel('BuyAndHold', RunTradingModelBuyHold, 'BAC','1/1/1987', 30, 30000, verbose=False)
	RunModel('Seasonal', RunTradingModelSeasonal, 'BAC','1/1/1987', 30, 30000, verbose=False)
	RunModel('FirstHalfOfMonth', RunTradingModelFirstHalfOfMonth, 'BAC','1/1/1987', 30, 30000, verbose=False)
	RunModel('Trending', RunTradingModelTrending, 'BAC','1/1/1987', 30, 30000, verbose=False)
	RunModel('SwingTrend', RunTradingModelSwingTrend, 'BAC','1/1/1987', 30, 30000, verbose=False)
	#RunModel('Trending', RunTradingModelTrending, '^Spx','1/1/1982', 35, 30000, verbose=False)

	#ExtendedDurationTest('SwingTrend', RunTradingModelSwingTrend,'^SPX')
	#CompareModels('BuyHold',RunTradingModelBuyHold,'Trending', RunTradingModelTrending, '^SPX','1/1/1987',20)
	PlotModeldailyValue('Trending',RunTradingModelTrending, 'Googl','1/1/2005',15)



