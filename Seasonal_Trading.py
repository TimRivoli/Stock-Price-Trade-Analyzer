import pandas as pd
from _classes.PriceTradeAnalyzer import TradingModel, PlotHelper, PriceSnapshot
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
	tm = TradingModel(modelName=modelName, startingTicker=ticker, startDate=startDate, durationInYears=durationInYears, totalFunds=portfolioSize, trancheSize=round(portfolioSize/10), verbose=verbose)
	startDate = tm.modelStartDate
	endDate = tm.modelEndDate
	dayCounter = 0
	currentYear = 0
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

def PlotModeldailyValue(modelName:str, modelFunction, ticker:str, startDate:str, durationInYears:int, portfolioSize:int=30000):
	#Plot daily returns of the given model
	m1 = RunModel(modelName, modelFunction, ticker, startDate, durationInYears, portfolioSize, saveHistoryToFile=True, returndailyValues=True, verbose=False)
	if m1.shape[0] > 0:
		print(m1)
		plot = PlotHelper()
		plot.PlotDataFrame(m1, modelName + ' Daily Value (' + ticker + ')', 'Date', 'Value') 

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
	#Buy in March, sell in September
	sn = tm.GetPriceSnapshot()
	if not sn == None:
		BuyDate  = ToDateTime('03/01/' + str(tm.currentDate.year))
		SellDate = ToDateTime('10/01/' + str(tm.currentDate.year))
		for i in range(tm._trancheCount):
			available, buyPending, sellPending, longPositions = tm.PositionSummary()				
			if not((tm.currentDate >= SellDate) or (tm.currentDate <= BuyDate)):
				if longPositions > 0: 
					tm.PlaceSell(ticker=ticker, price=sn.High, marketOrder=True)
				else:
					break
			else:
				if available > 0 and tm.FundsAvailable() > sn.High: 
					tm.PlaceBuy(ticker=ticker, price=sn.Low, marketOrder=True)
				else:
					break

def CompareModels(modelOneName:str, modelOneFunction, modelTwoName:str, modelTwoFunction, ticker:str, startDate:str, durationInYears:int, portfolioSize:int=30000):
	#Compare two models to measure the difference in returns
	m1 = RunModel(modelOneName, modelOneFunction, ticker, startDate, durationInYears, portfolioSize, saveHistoryToFile=False, returndailyValues=True, verbose=False)
	m2 = RunModel(modelTwoName, modelTwoFunction, ticker, startDate, durationInYears, portfolioSize, saveHistoryToFile=False, returndailyValues=True, verbose=False)
	if m1.shape[0] > 0 and m2.shape[0] > 0:
		m1 = m1.join(m2, lsuffix='_' + modelOneName, rsuffix='_' + modelTwoName)
		plot = PlotHelper()
		plot.PlotDataFrame(m1, ticker + ' Model Comparison', 'Date', 'Value') 

if __name__ == '__main__':
	CompareModels('BuyHold',RunTradingModelBuyHold,'Seasonal', RunTradingModelSeasonal, '.INX','1/1/1990',30)
