import pandas as pd
from _classes.PriceTradeAnalyzer import TradingModel, PlotHelper, PriceSnapshot
from _classes.Utility import *

#------------------------------------------- Global model functions  ----------------------------------------------		
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

if __name__ == '__main__':
	CompareModels('BuyHold',RunTradingModelBuyHold,'Seasonal', RunTradingModelSeasonal, '.INX','1/1/1990',30)
