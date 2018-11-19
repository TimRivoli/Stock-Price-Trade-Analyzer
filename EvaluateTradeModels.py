#WARNING: "Far more money has been lost by investors preparing for corrections or trying to anticipate corrections  
#than has been lost in corrections themselves." - Peter Lynch.
import pandas
from _classes.PriceTradeAnalyzer import TradingModel, PlotHelper, PriceSnapshot, Position
SPTop70MinusDogs=['AAPL','MSFT','AMZN','FB','BRK-B','JPM','JNJ','XOM','GOOG','GOOGL','BAC','WFC','CVX','UNH','HD','INTC','PFE','T','V','PG','VZ','CSCO','C','CMCSA','ABBV','BA','KO','DWDP','PEP','MRK','DIS','WMT','ORCL','MA','MMM','NVDA','IBM','AMGN','MCD','MO','NFLX','HON','MDT','GILD','TXN','ABT','UNP','SLB','BMY','UTX','AVGO','ACN','QCOM','ADBE','CAT','GS','PYPL','PCLN','USB','UPS','LOW','NKE','TMO','LMT','COST','CVS','LLY','CELG']
SomeStocks=['AAPL','MSFT','AMZN','FB','BRK-B','JPM','JNJ','XOM','BAC','WFC','CVX','UNH','HD','INTC','PFE','T','V','PG','VZ','CSCO','C','CMCSA','ABBV','BA','KO','DWDP','PEP','MRK','DIS','WMT','ORCL','MA','MMM','NVDA','IBM','AMGN','MCD','MO','NFLX','HON','MDT','GILD','TXN','ABT','UNP','SLB','BMY','UTX','AVGO','ACN','QCOM','ADBE','CAT','GS','PYPL','PCLN','USB','UPS','LOW','NKE','TMO','LMT','COST','CVS','LLY','CELG']
TradeRunnerStocks = ['scg','wec']	#TradeRunner 90,000 profit in 14 years, buy and hold would be more like 400,000 on wec.  7% return would be 157,853 profit instead of 90,000, spx would be 83,000

#------------------------------------------- Global helper functions  ----------------------------------------------		
def ExtendedDurationTest(modelName:str, modelFunction, ticker:str):
	PortfolioSize=10000
	#test the model over and extended range of periods and years
	TestResults = pandas.DataFrame([['1/1/1950','1/1/1950',0,0]], columns=list(['StartDate','EndDate','Duration','EndingValue']))
	TestResults.set_index(['StartDate'], inplace=True)		
	for duration in range(1,10,2):
		for year in range(1950,2017):
			for month in range(1,12,3):
				startDate = str(month) + '/1/' + str(year)
				endDate = str(month) + '/1/' + str(year + duration)
				endValue = RunModel(modelName, modelFunction, ticker, startDate, duration, PortfolioSize, saveHistoryToFile=False, returndailyValues=False, verbose=False)
				y = pandas.DataFrame([[startDate,endDate,duration,endValue]], columns=list(['StartDate','EndDate','Duration','EndingValue']))
				TestResults = TestResults.append(y, ignore_index=True)
	TestResults.to_csv('data/trademodel/' + modelName + '_' + ticker +'_extendedTest.csv')
	print(TestResults)

def PlotModeldailyValue(modelName:str, modelFunction, ticker:str, startDate:str, durationInYears:int):
	PortfolioSize=10000
	m1 = RunModel(modelName, modelFunction, ticker, startDate, durationInYears, PortfolioSize, saveHistoryToFile=True, returndailyValues=True, verbose=False)
	if m1.shape[0] > 0:
		plot = PlotHelper()
		plot.PlotDataFrame(m1, modelName + ' Daily Value (' + ticker + ')', 'Date', 'Value') 

def CompareModels(modelOneName:str, modelOneFunction, modelTwoName:str, modelTwoFunction, ticker:str, startDate:str, durationInYears:int):
	PortfolioSize=10000
	m1 = RunModel(modelOneName, modelOneFunction, ticker, startDate, durationInYears, PortfolioSize, saveHistoryToFile=False, returndailyValues=True, verbose=False)
	m2 = RunModel(modelTwoName, modelTwoFunction, ticker, startDate, durationInYears, PortfolioSize, saveHistoryToFile=False, returndailyValues=True, verbose=False)
	if m1.shape[0] > 0 and m2.shape[0] > 0:
		m1 = m1.join(m2, lsuffix='_' + modelOneName, rsuffix='_' + modelTwoName)
		plot = PlotHelper()
		plot.PlotDataFrame(m1, ticker + ' Model Comparison', 'Date', 'Value') 

def RunModel(modelName:str, modelFunction, ticker:str, startDate:str, durationInYears:int, totalFunds:int, saveHistoryToFile:bool=True, returndailyValues:bool=False, verbose:bool=False):	
	modelName = modelName + '_' + ticker 
	tm = TradingModel(modelName=modelName, startingTicker=ticker, startDate=startDate, durationInYears=durationInYears, totalFunds=totalFunds, verbose=verbose)
	if not tm.modelReady:
		print('Unable to initialize price history for model for ' + str(startDate))
		if returndailyValues: return pandas.DataFrame()
		else:return totalFunds
	else:
		while not tm.ModelCompleted():
			tm.ProcessDay()
			modelFunction(tm, ticker)
			if tm.AccountingError(): 
				print('Accounting error.  The numbers do not add up correctly.')
				break
		if returndailyValues:
			tm.CloseModel(verbose, saveHistoryToFile)
			return tm.GetDailyValue()   #return daily value
		else:
			return tm.CloseModel(verbose, saveHistoryToFile)		#return closing value

#------------------------------------------- Your models go here ----------------------------------------------		
#	Each function should define what actions should be taken in the given day from the TradingModel, Buy/Sell/Hold

def RunTradingModelBuyHold(tm: TradingModel, ticker:str):
	currentPrices = tm.GetPriceSnapshot()
	if tm.verbose: print(currentPrices.snapShotDate, currentPrices.nextDayTarget)
	if not currentPrices == None:
		if tm.TraunchesAvailable() > 0 and tm.FundsAvailable() > currentPrices.high: tm.PlaceBuy(ticker, currentPrices.low, True)

def RunTradingModelSeasonal(tm: TradingModel, ticker:str):
	SellMonth = 4	#April
	BuyMonth = 10	#October
	currentPrices = tm.GetPriceSnapshot()
	if not currentPrices == None:
		low = currentPrices.low
		high = currentPrices.high
		m = tm.currentDate.month
		available, buyPending, sellPending, longPositions = tm.PositionSummary()				
		if m >= SellMonth and m <= BuyMonth:
			if longPositions > 0: tm.PlaceSell(ticker, high, True)
		else:
			if available > 0 and tm.FundsAvailable() > high: tm.PlaceBuy(ticker, low, True)

def RunTradingModelTrending(tm: TradingModel, ticker:str):
	#Give it a date range, some money, and a stock, it will execute a strategy and return the results
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
	#Give it a date range, some money, and a stock, it will execute a strategy and return the results
	#minDeviationToTrade = .025
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
	RunModel('BuyAndHold', RunTradingModelBuyHold, '^Spx','1/1/2000', 15, 50000, verbose=False)
	RunModel('BuyAndHold', RunTradingModelBuyHold, 'MRK','1/2/2008', 5, 50000, verbose=False)
	ExtendedDurationTest('BuyHold', RunTradingModelBuyHold,'^SPX')
	RunModel('Trending', RunTradingModelTrending, 'GOOGL','1/2/2008', 5, 50000, verbose=False)
	CompareModels('BuyHold',RunTradingModelBuyHold,'Trending', RunTradingModelTrending, 'Googl','1/1/2000',15)
	#RunModel('Trending', RunTradingModelTrending, '^Spx','1/2/2008', 5, 50000, verbose=False)
	#RunModel('Trending', RunTradingModelTrending, '^Spx','1/1/1990', 27, 50000, verbose=False)
	#RunModel('SwingTrend', RunTradingModelSwingTrend, '^Spx','1/1/1990', 27, 50000, verbose=False)
	#RunModel('BuyAndHold', RunTradingModelBuyHold, '^Spx','1/1/1980', 1, 10000, verbose=False)
	PlotModeldailyValue('Trending',RunTradingModelTrending, 'Googl','1/1/2005',10)



