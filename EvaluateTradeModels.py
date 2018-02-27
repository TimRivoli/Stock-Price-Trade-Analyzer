#WARNING: "Far more money has been lost by investors preparing for corrections or trying to anticipate corrections  
#than has been lost in corrections themselves." - Peter Lynch.
import pandas
from _classes.PriceTradeAnalyzer import TradingModel, PlotHelper, PriceSnapshot, Position
SPTop70MinusDogs=['AAPL','MSFT','AMZN','FB','BRK-B','JPM','JNJ','XOM','GOOG','GOOGL','BAC','WFC','CVX','UNH','HD','INTC','PFE','T','V','PG','VZ','CSCO','C','CMCSA','ABBV','BA','KO','DWDP','PEP','MRK','DIS','WMT','ORCL','MA','MMM','NVDA','IBM','AMGN','MCD','MO','NFLX','HON','MDT','GILD','TXN','ABT','UNP','SLB','BMY','UTX','AVGO','ACN','QCOM','ADBE','CAT','GS','PYPL','PCLN','USB','UPS','LOW','NKE','TMO','LMT','COST','CVS','LLY','CELG']
SomeStocks=['AAPL','MSFT','AMZN','FB','BRK-B','JPM','JNJ','XOM','BAC','WFC','CVX','UNH','HD','INTC','PFE','T','V','PG','VZ','CSCO','C','CMCSA','ABBV','BA','KO','DWDP','PEP','MRK','DIS','WMT','ORCL','MA','MMM','NVDA','IBM','AMGN','MCD','MO','NFLX','HON','MDT','GILD','TXN','ABT','UNP','SLB','BMY','UTX','AVGO','ACN','QCOM','ADBE','CAT','GS','PYPL','PCLN','USB','UPS','LOW','NKE','TMO','LMT','COST','CVS','LLY','CELG']

#------------------------------------------- Global helper functions  ----------------------------------------------		
def ExtendedDurationTest(ticker:str, modelName:str, modelFunction):
	#test the model over and extended range of periods and years
	TestResults = pandas.DataFrame([['1/1/1998','1/1/1998',0,0]], columns=list(['StartDate','EndDate','Duration','EndingValue']))
	TestResults.set_index(['StartDate'], inplace=True)		
	for duration in range(1,10,2):
		for year in range(1999,2017):
			for month in range(1,12,3):
				startDate = str(month) + '/1/' + str(year)
				endDate = str(month) + '/1/' + str(year + duration)
				endValue = modelFunction(ticker,startDate, duration, 10000, verbose=False, saveHistoryToFile=False, returndailyValues=False)
				y = pandas.DataFrame([[startDate,endDate,duration,endValue]], columns=list(['StartDate','EndDate','Duration','EndingValue']))
				TestResults = TestResults.append(y, ignore_index=True)
	TestResults.to_csv('data/trademodel/' + modelName + '_' + ticker +'_extendedTest.csv')
	print(TestResults)

def PlotModeldailyValue(ticker:str, startDate:str, durationInYears:int, modelName:str, modelFunction):
	m1 = modelFunction(ticker, startDate, durationInYears, 10000,  verbose=False, saveHistoryToFile=True, returndailyValues=True)
	if m1.shape[0] > 0:
		plot = PlotHelper()
		plot.PlotDataFrame(m1, modelName + ' Daily Value (' + ticker + ')', 'Date', 'Value') 

def CompareModels(ticker:str, startDate:str, durationInYears:int, modelOneName:str, modelTwoName:str, modelOne, modelTwo):
	m1 = modelOne(ticker, startDate, durationInYears, 10000, verbose=False, saveHistoryToFile=False, returndailyValues=True)
	m2 = modelTwo(ticker, startDate, durationInYears, 10000, verbose=False, saveHistoryToFile=False, returndailyValues=True)
	if m1.shape[0] > 0 and m2.shape[0] > 0:
		m1 = m1.join(m2, lsuffix='_' + modelOneName, rsuffix='_' + modelTwoName)
		plot = PlotHelper()
		plot.PlotDataFrame(m1, ticker + ' Model Comparison', 'Date', 'Value') 

#------------------------------------------- Your models go here ----------------------------------------------		

def RunTradingModelBuyHold(ticker:str, startDate:str, durationInYears:int, totalFunds:int, verbose:bool=False, saveHistoryToFile:bool=True, returndailyValues:bool=False):
	modelName = 'BuyHold' + '_' + ticker 
	tm = TradingModel(modelName, ticker, startDate, durationInYears, totalFunds, verbose)
	if not tm.modelReady:
		print('Unable to initialize price history for model for ' + str(startDate))
		if returndailyValues: return pandas.DataFrame()
		else:return totalFunds
	else:
		while not tm.ModelCompleted():
			tm.ProcessDay()
			currentPrices = tm.GetPriceSnapshot()
			if not currentPrices == None:
				if tm.TraunchesAvailable() and tm.FundsAvailable() > currentPrices.high: tm.PlaceBuy(ticker, currentPrices.low, True)
			if tm.AccountingError(): break
		if returndailyValues:
			tm.CloseModel(verbose, saveHistoryToFile)
			return tm.GetdailyValue()   #return daily value
		else:
			return tm.CloseModel(verbose, saveHistoryToFile)		#return closing value

def RunTradingModelBuyHoldMultipleStocks(tickerList:list, startDate:str, durationInYears:int, totalFunds:int, verbose:bool=False, saveHistoryToFile:bool=True, returndailyValues:bool=False):
	modelName = 'BuyHold_MultipleStocks'
	tm = TradingModel(modelName, tickerList[0], startDate, durationInYears, totalFunds, verbose)
	if not tm.modelReady:
		print('Unable to initialize price history for model for ' + str(startDate))
		if returndailyValues: return pandas.DataFrame()
		else:return totalFunds
	else:
		while not tm.ModelCompleted():
			tm.ProcessDay()
			if tm.TraunchesAvailable():
				for t in tickerList:
					if not tm.TraunchesAvailable(): break
					currentPrices = tm.GetPriceSnapshot(t)
					if not currentPrices==None:
						if tm.TraunchesAvailable() and tm.FundsAvailable() > currentPrices.high: tm.PlaceBuy(t, currentPrices.low, True)
			if tm.AccountingError(): break

		if returndailyValues:
			tm.CloseModel(verbose, saveHistoryToFile)
			return tm.GetdailyValue()   #return daily value
		else:
			return tm.CloseModel(verbose, saveHistoryToFile)		#return closing value

def RunTradingModelSeasonal(ticker:str, startDate:str, durationInYears:int, totalFunds:int, verbose:bool=False, saveHistoryToFile:bool=True, returndailyValues:bool=False):
	modelName = 'Seasonal' + '_' + ticker 
	tm = TradingModel(modelName, ticker, startDate, durationInYears, totalFunds, verbose)
	if not tm.modelReady:
		print('Unable to initialize price history for model for ' + str(startDate))
		if returndailyValues: return pandas.DataFrame()
		else:return totalFunds
	else:
		while not tm.ModelCompleted():
			tm.ProcessDay()
			currentPrices = tm.GetPriceSnapshot()
			if not currentPrices == None:
				low = currentPrices.low
				high = currentPrices.high
				m = tm.currentDate.month
				available, buyPending, sellPending, longPositions = tm.GetPositionSummary()				
				if m >= 11 or m <=4:	#Buy if Nov through April, else sell
					if available > 0 and tm.FundsAvailable() > high: tm.PlaceBuy(ticker, low, True)
				else:
					if longPositions > 0: tm.PlaceSell(ticker, high, True)
			if tm.AccountingError(): break
		if returndailyValues:
			tm.CloseModel(verbose, saveHistoryToFile)
			return tm.GetdailyValue()   #return daily value
		else:
			return tm.CloseModel(verbose, saveHistoryToFile)		#return closing value

def RunTradingModelTrending(ticker:str,startDate:str, durationInYears:int, totalFunds:int, verbose:bool=False, saveHistoryToFile:bool=True, returndailyValues:bool=False):
	#Give it a date range, some money, and a stock, it will execute a strategy and return the results
	minActionableSlope = 0.002
	trendState='Flat'	#++,--,+-,-+,Flat
	prevTrendState = ''
	trendDuration = 0

	modelName = 'Trending' + '_' + ticker 
	tm = TradingModel(modelName, ticker,startDate, durationInYears, totalFunds, verbose)
	if not tm.modelReady:
		print('Unable to initialize price history for model for ' + str(startDate))
		if returndailyValues: return pandas.DataFrame()
		else:return totalFunds
	else:
		while not tm.ModelCompleted():
			tm.ProcessDay()
			p = tm.GetPriceSnapshot()
			if not p == None:
				available, buyPending, sellPending, longPositions = tm.GetPositionSummary()
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
				else:																				 #flat, aim for 70% long
					trendState='Flat'
					if p.low > p.channelHigh:   #Over Bought
						pass
					elif p.high < p.channelLow: #Over sold
						if buyPending < 3 and longPositions < 8: tm.PlaceBuy(ticker, targetBuy, False,5)
						if buyPending < 4: tm.PlaceBuy(ticker, targetBuy, False,5)
					else:
						pass
					if buyPending < 3 and longPositions < maxPositions: tm.PlaceBuy(ticker, targetBuy, False,5)
				if trendState == prevTrendState: 
					trendDuration = trendDuration + 1
				else:
					trendDuration=0
			if tm.AccountingError(): break
		if returndailyValues:
			tm.CloseModel(verbose, saveHistoryToFile)
			return tm.GetdailyValue()   #return daily value
		else:
			return tm.CloseModel(verbose, saveHistoryToFile)		#return closing value
	
def RunTradingModelSwingTrend(ticker:str,startDate:str, durationInYears:int, totalFunds:int, verbose:bool=False, saveHistoryToFile:bool=True, returndailyValues:bool=False):
	#Give it a date range, some money, and a stock, it will execute a strategy and return the results
	#minDeviationToTrade = .025
	minActionableSlope = 0.002
	trendState='Flat'	#++,--,+-,-+,Flat
	prevTrendState = ''
	trendDuration = 0

	modelName = 'SwingTrend' + '_' + ticker 
	tm = TradingModel(modelName, ticker,startDate, durationInYears, totalFunds, verbose)
	if not tm.modelReady:
		print('Unable to initialize price history for model for ' + str(startDate))
		if returndailyValues: return pandas.DataFrame()
		else:return totalFunds
	else:
		while not tm.ModelCompleted():
			tm.ProcessDay()
			p = tm.GetPriceSnapshot()
			if not p == None:
				available, buyPending, sellPending, longPositions = tm.GetPositionSummary()
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
			if tm.AccountingError(): break
		if returndailyValues:
			tm.CloseModel(verbose, saveHistoryToFile)
			return tm.GetdailyValue()   #return daily value
		else:
			return tm.CloseModel(verbose, saveHistoryToFile)		#return closing value

			
def RunTradingModelQLearn(ticker:str,startDate:str, durationInYears:int, totalFunds:int, verbose:bool=False, saveHistoryToFile:bool=True, returndailyValues:bool=False):
	Actions = ['Hold','BuyMarket','BuyAgressiveLeve10','BuyAgressiveLeve11','BuyAgressiveLeve12','SellMarket','SellAgressiveLeve10','SellAgressiveLeve11','SellAgressiveLeve12']
	modelName = 'QLearn' + '_' + ticker 
	tm = TradingModel(modelName, ticker, startDate, durationInYears, totalFunds, verbose)
	if not tm.modelReady:
		print('Unable to initialize price history for model for ' + str(startDate))
		if returndailyValues: return pandas.DataFrame()
		else:return totalFunds
	else:
		while not tm.ModelCompleted():
			tm.ProcessDay()
			currentPrices = tm.GetPriceSnapshot()
			if not currentPrices == None:
				pass
				#do stuff
		if returndailyValues:
			tm.CloseModel(verbose, saveHistoryToFile)
			return tm.GetdailyValue()   #return daily value
		else:
			return tm.CloseModel(verbose, saveHistoryToFile)		#return closing value

#ExtendedDurationTest('GOOGL', 'BuyHold', RunTradingModelBuyHold)
#RunTradingModelBuyHold('MRK','1/2/2008',1, 10000, verbose=True)
#RunTradingModelTrending('MRK','1/2/2010',1, 10000, verbose=True)
#RunTradingModelTrending('GOOGL','1/2/2008',4, 10000, verbose=True)
#CompareModels('Googl','1/1/2000',17, 'BuyHold','Trending',RunTradingModelBuyHold, RunTradingModelTrending)
#CompareModels('^SPX','1/1/2000',17, 'BuyHold','Seasonal',RunTradingModelBuyHold, RunTradingModelSeasonal)
#RunTradingModelTrending('^Spx','1/1/1990',27, 10000, verbose=False)
#RunTradingModelBuyHoldMultipleStocks(SomeStocks,'1/2/1980', 10, 50000, verbose=False)
RunTradingModelBuyHold('^Spx','1/2/2000', 10, 50000, verbose=False)
RunTradingModelBuyHold('^Spx','1/2/2008', 10, 50000, verbose=False)
RunTradingModelSwingTrend('^Spx','1/1/1990',27, 10000, verbose=False)
RunTradingModelBuyHoldMultipleStocks(SomeStocks,'1/2/2000', 10, 50000, verbose=False)
PlotModeldailyValue('Googl','1/1/2005',1, 'Trending',RunTradingModelTrending)
