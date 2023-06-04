import sys
import pandas as pd
from _classes.PriceTradeAnalyzer import TradingModel, PricingData, StockPicker
from _classes.TickerLists import TickerLists
from _classes.Utility import *

def RunBuyHold(ticker: str, startDate:str, durationInYears:int, ReEvaluationInterval:int=20, portfolioSize:int=30000, verbose:bool=False):
	#Baseline model to compare against.  Buy on day one, hold for the duration and then sell
	modelName = 'BuyHold_' + (ticker) + '_' + startDate[-4:]
	tm = TradingModel(modelName=modelName, startingTicker=ticker, startDate=startDate, durationInYears=durationInYears, totalFunds=portfolioSize, tranchSize=portfolioSize/10, verbose=verbose)
	if not tm.modelReady:
		print('Unable to initialize price history for model BuyHold date ' + str(startDate))
		return 0
	else:
		dayCounter =0
		while not tm.ModelCompleted():
			if dayCounter ==0:
				i=0
				while tm.TranchesAvailable() and i < 100: 
					tm.PlaceBuy(ticker=ticker, price=1, marketOrder=True, expireAfterDays=10, verbose=verbose)
					i +=1
			dayCounter+=1
			if dayCounter >= ReEvaluationInterval: dayCounter=0
			tm.ProcessDay()
		cash, asset = tm.Value()
		print('Ending Value: ', cash + asset, '(Cash', cash, ', Asset', asset, ')')
		return tm.CloseModel(plotResults=False, saveHistoryToFile=verbose)	

def RunBuyHoldList(tickerList:list, startDate:str, durationInYears:int, portfolioSize:int=30000, verbose:bool=False):
	#Alternative option to use Buy Hold strategy with a list of tickers
	c = len(tickerList)
	modelName = 'BuyHold_tickerList_count' + str(c) + '_' + startDate[-4:]
	tm = TradingModel(modelName=modelName, startingTicker=tickerList[0], startDate=startDate, durationInYears=durationInYears, totalFunds=portfolioSize, tranchSize=portfolioSize/c)
	if not tm.modelReady:
		print('Unable to initialize price history for BuyHoldList date ' + str(startDate))
		return 0
	else:
		for t in tickerList:
			tm.PlaceBuy(ticker=t, price=1, marketOrder=True, expireAfterDays=10, verbose=verbose)
		while not tm.ModelCompleted():
			tm.ProcessDay()
		cash, asset = tm.Value()
		print('Ending Value: ', cash + asset, '(Cash', cash, ', Asset', asset, ')')
		return tm.CloseModel(plotResults=False, saveHistoryToFile=verbose)	

def AlignPositions(tm:TradingModel, targetPositions:pd.DataFrame, stockCount:int, allocateByPointValue:bool= False, verbose:bool = False):
	#Helper function.  Performs necessary Buy/Sells to get from current positions to target positions
	print(targetPositions)
	TotalTranches = tm._tranchCount
	targetPositions=pd.DataFrame(targetPositions.groupby(['Ticker','Point_Value']).size()).reset_index()
	targetPositions.set_index(['Ticker'], inplace=True)
	targetPositions.rename(columns={0:'TargetHoldings'}, inplace=True)
	targetPositions.sort_values(by=['TargetHoldings', 'Point_Value'], axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') 
	targetPositions = targetPositions[:stockCount]
	print(targetPositions)
	if allocateByPoint_Value:
		TotalPoints = targetPositions['Point_Value'].sum()  
		scale = TotalTranches/TotalPoints
		print('scale', TotalPoints, TotalTranches, scale)
		targetPositions.loc[:, 'TargetHoldings'] =  round(targetPositions.loc[:, 'TargetHoldings'] * targetPositions.loc[:, 'Point_Value'] * scale)
	else:
		TotalTargets = targetPositions['TargetHoldings'].sum()  
		scale = TotalTranches/TotalTargets
		print('scale', TotalTargets, TotalTranches, scale)
		targetPositions.loc[:, 'TargetHoldings'] =  round(targetPositions.loc[:, 'TargetHoldings'] * scale)
	print(targetPositions)
	currentPositions = tm.GetPositions(asDataFrame=True)
	if len(currentPositions) > 0:	#evaluate the difference between current holdings and target, act accordingly
		targetPositions = targetPositions.join(currentPositions, how='outer')
		targetPositions.fillna(value=0, inplace=True)				
		targetPositions['Difference'] = targetPositions['TargetHoldings'] - targetPositions['CurrentHoldings']
		print(targetPositions)
		for i in range(len(targetPositions)):
			sells = int(targetPositions.iloc[i]['Difference'])
			if sells < 0:
				t = targetPositions.index.values[i]
				print('Sell ' + str(abs(sells)) + ' ' + t)
				for _ in range(abs(sells)): 
					tm.PlaceSell(ticker=t, price=1, marketOrder=True, expireAfterDays=10, verbose=verbose)
		tm.ProcessDay(withIncrement=False)
		for i in range(len(targetPositions)):
			buys = int(targetPositions.iloc[i]['Difference'])
			if buys > 0:
				t = targetPositions.index.values[i]
				print('Buy ' + str(buys) + ' ' + t)
				for _ in range(buys):
					tm.PlaceBuy(ticker=t, price=1, marketOrder=True, expireAfterDays=10, verbose=verbose)								
		tm.ProcessDay(withIncrement=False)
	elif len(targetPositions) > 0:	
		for i in range(len(targetPositions)):
			buys = int(targetPositions.iloc[i]['TargetHoldings'])
			if buys > 0:
				t = targetPositions.index.values[i]
				print(t)
				print(buys)
				print('Buy ' + str(buys) + ' ' + t)
				for _ in range(buys):
					tm.PlaceBuy(ticker=t, price=1, marketOrder=True, expireAfterDays=10, verbose=verbose)								
		tm.ProcessDay(withIncrement=False)
	print(tm.GetPositions(asDataFrame=True))
	print(tm.PositionSummary())


def RunPriceMomentum(tickerList:list, startDate:str='1/1/1982', durationInYears:int=36, stockCount:int=9, ReEvaluationInterval:int=20, filterOption:int=3, longHistory:int=365, shortHistory:int=90, minPercentGain=0.05, maxVolatility=.12, portfolioSize:int=30000, returndailyValues:bool=False, verbose:bool=False):
	#Choose stockCount stocks with the greatest long term (longHistory days) price appreciation, using different filter options defined in the StockPicker class
	#shortHistory is a shorter time frame (like 90 days) used differently by different filters
	#ReEvaluationInterval is how often to re-evaluate our choices, ideally this should be very short and not matter, otherwise the date selection is biased.
	startDate = ToDate(startDate)
	endDate =  AddDays(startDate, 365 * durationInYears)
	picker = StockPicker(AddDays(startDate, -730), endDate) #Include earlier dates for statistics
	for t in tickerList:
		picker.AddTicker(t)
	tm = TradingModel(modelName='PriceMomentumShort_longHistory_' + str(longHistory) +'_shortHistory_' + str(shortHistory) + '_reeval_' + str(ReEvaluationInterval) + '_stockcount_' + str(stockCount) + '_filter' + str(filterOption) + '_' + str(minPercentGain) + str(maxVolatility), startingTicker='.INX', startDate=startDate, durationInYears=durationInYears, totalFunds=portfolioSize, tranchSize=portfolioSize/stockCount, verbose=verbose)
	dayCounter = 0
	if not tm.modelReady:
		print('Unable to initialize price history for PriceMomentum date ' + str(startDate))
		return 0
	else:
		while not tm.ModelCompleted():
			currentDate =  tm.currentDate
			if dayCounter ==0:
				print('\n')
				print(currentDate)
				c, a = tm.Value()
				print(tm.modelName, int(c), int(a), int(c+a))
				print('available/buy/sell/long',tm.PositionSummary())
				candidates = picker.GetHighestPriceMomentum(currentDate, longHistoryDays=longHistory, shortHistoryDays=shortHistory, stocksToReturn=stockCount, filterOption=filterOption, minPercentGain=minPercentGain, maxVolatility=maxVolatility)
				AlignPositions(tm=tm, targetPositions=candidates, stockCount=stockCount, allocateByPoint_Value=False)
			tm.ProcessDay()
			dayCounter+=1
			if dayCounter >= ReEvaluationInterval: dayCounter=0

		cv1 = tm.CloseModel(plotResults=False, saveHistoryToFile=((durationInYears>1) or verbose))
		if returndailyValues:
			return tm.GetDailyValue()
		else:
			return cv1

def RunPriceMomentumBlended(tickerList:list, startDate:str='1/1/1980', durationInYears:int=29, ReEvaluationInterval:int=20, longHistory:int=365, shortHistory:int=90, portfolioSize:int=30000, returndailyValues:bool=False, verbose:bool=False):
	#Uses blended option for selecting stocks using three different filters, produces the best overall results.
	#1 long term performer at short term discount
	#2 long term performer
	#4 short term performer
	minPercentGain=.05
	BlendDesc = '3.3.44.PV'
	startDate = ToDate(startDate)
	endDate =  AddDays(startDate, 365 * durationInYears)
	picker = StockPicker(AddDays(startDate, -730), endDate) #Include earlier dates for statistics
	stockCount = 11
	for t in tickerList:
		picker.AddTicker(t)
	tm = TradingModel(modelName='PriceMomentum_Blended' + BlendDesc + '_longHistory_' + str(longHistory) +'_shortHistory_' + str(shortHistory) + '_reeval_' + str(ReEvaluationInterval) + '_stockcount_' + str(stockCount), startingTicker='.INX', startDate=startDate, durationInYears=durationInYears, totalFunds=portfolioSize, tranchSize=portfolioSize/stockCount, verbose=verbose)
	dayCounter = 0
	if not tm.modelReady:
		print('Unable to initialize price history for PriceMomentum date ' + str(startDate))
		return 0
	else:
		while not tm.ModelCompleted():
			currentDate =  tm.currentDate
			if dayCounter == 0:
				print('\n')
				print(currentDate)
				c, a = tm.Value()
				print(tm.modelName, int(c), int(a), int(c+a))
				print('available/buy/sell/long',tm.PositionSummary())
				list1 = picker.GetHighestPriceMomentum(currentDate, longHistoryDays=longHistory, shortHistoryDays=shortHistory, stocksToReturn=2, filterOption=3, minPercentGain=minPercentGain)
				list2 = picker.GetHighestPriceMomentum(currentDate, longHistoryDays=longHistory, shortHistoryDays=shortHistory, stocksToReturn=2, filterOption=3, minPercentGain=minPercentGain)
				list3 = picker.GetHighestPriceMomentum(currentDate, longHistoryDays=longHistory, shortHistoryDays=shortHistory, stocksToReturn=2, filterOption=44, minPercentGain=minPercentGain)
				list4 = picker.GetHighestPriceMomentum(currentDate=currentDate, stocksToReturn=5, filterOption=5)
				candidates = pd.concat([list1, list2, list3], sort=True)
				AlignPositions(tm=tm, targetPositions=candidates, stockCount=stockCount, allocateByPoint_Value=False) #Each group given equal weight
			tm.ProcessDay()
			dayCounter+=1
			if dayCounter >= ReEvaluationInterval: dayCounter=0

		cv1 = tm.CloseModel(plotResults=False, saveHistoryToFile=((durationInYears>1) or verbose))
		if returndailyValues:
			return tm.GetDailyValue()
		else:
			return cv1

def RunPoint_Value(tickerList:list, startDate:str='1/1/1982', durationInYears:int=36, stockCount:int=9, ReEvaluationInterval:int=20, minPercentGain=0.05, portfolioSize:int=30000, returndailyValues:bool=False, verbose:bool=False):
	startDate = ToDate(startDate)
	endDate =  AddDays(startDate, 365 * durationInYears)
	picker = StockPicker(AddDays(startDate, -730), endDate) #Include earlier dates for statistics
	for t in tickerList:
		picker.AddTicker(t)
	tm = TradingModel(modelName='Point_Value_reeval_' + str(ReEvaluationInterval) + '_stockcount_' + str(stockCount) + '_' + str(minPercentGain), startingTicker='.INX', startDate=startDate, durationInYears=durationInYears, totalFunds=portfolioSize, tranchSize=2500, verbose=verbose)
	dayCounter = 0
	if not tm.modelReady:
		print('Unable to initialize price history for Point_Value date ' + str(startDate))
		return 0
	else:
		while not tm.ModelCompleted():
			currentDate =  tm.currentDate
			if dayCounter ==0:
				print('\n')
				print(currentDate)
				c, a = tm.Value()
				print(tm.modelName, int(c), int(a), int(c+a))
				print('available/buy/sell/long',tm.PositionSummary())
				candidates = picker.GetHighestPriceMomentum(currentDate, stocksToReturn=stockCount, minPercentGain=minPercentGain, filterOption=5)
				AlignPositions(tm=tm, targetPositions=candidates, stockCount=stockCount, allocateByPoint_Value=True)
			tm.ProcessDay()
			dayCounter+=1
			if dayCounter >= ReEvaluationInterval: dayCounter=0
		cv1 = tm.CloseModel(plotResults=False, saveHistoryToFile=((durationInYears>1) or verbose))
		if returndailyValues:
			return tm.GetDailyValue()
		else:
			return cv1
			
def ComparePMToBH(startYear:int=1982, endYear:int=2018, durationInYears:int=1, stockCount:int=9, ReEvaluationInterval:int=20, filterOption:int=3, longHistory:int=365, shortHistory:int=90):
	#Compares the PriceMomentum strategy to BuyHold in one year intervals, outputs the returns to .csv file
	modelOneName = 'BuyHold'
	modelTwoName = 'PriceMomentum_longHistory_' + str(longHistory) +'_shortHistory_' + str(shortHistory) + '_ReEval_' + str(ReEvaluationInterval) + '_stockcount_' + str(stockCount) + '_filter' + str(filterOption)
	portfolioSize=30000
	TestResults = pd.DataFrame(columns=list(['StartDate','Duration', modelOneName + 'EndingValue',  'ModelEndingValue', modelOneName + 'Gain', 'ModelGain', 'Difference']))
	TestResults.set_index(['StartDate'], inplace=True)		
	trials = int((endYear - startYear)/durationInYears) 
	for i in range(trials):
		startDate = '1/2/' + str(startYear + i * durationInYears)
		m1ev = RunBuyHold('.INX', startDate=startDate, durationInYears=durationInYears, ReEvaluationInterval=ReEvaluationInterval, portfolioSize=portfolioSize)
		m2ev = RunPriceMomentum(tickerList = TickerLists.SPTop70(), startDate=startDate, durationInYears=durationInYears, stockCount=stockCount, ReEvaluationInterval=ReEvaluationInterval, filterOption=filterOption,  longHistory=longHistory, shortHistory=shortHistory, portfolioSize=portfolioSize, returndailyValues=False, verbose=False)
		m1pg = (m1ev/portfolioSize) - 1 
		m2pg = (m2ev/portfolioSize) - 1
		TestResults.loc[startDate] = [durationInYears, m1ev, m2ev, m1pg, m2pg, m2pg-m1pg]
	TestResults.sort_values(['Difference'], axis=0, ascending=True, inplace=True)
	TestResults.to_csv('data/trademodel/Compare' + modelOneName + '_to_' + modelTwoName + '_year ' + str(startYear) + '_duration' + str(durationInYears) +'.csv')
	print(TestResults)

def CompareBlendedToBH(startYear:int=1982, endYear:int=2018, durationInYears:int = 1, ReEvaluationInterval:int=20, longHistory:int=365, shortHistory:int=90):
	#Compares the BlendedPriceMomentum strategy to BuyHold in one year intervals, outputs the returns to .csv file
	stockCount = 11
	modelOneName = 'BuyHold'
	BlendDesc = '3.w3.44.PV'
	modelTwoName = 'PriceMomentumBlended' + BlendDesc
	modelTwoName += '_longHistory_' + str(longHistory) +'_shortHistory_' + str(shortHistory) + '_ReEval_' + str(ReEvaluationInterval) + '_stockcount_' + str(stockCount)
	portfolioSize=30000
	TestResults = pd.DataFrame(columns=list(['StartDate','Duration', modelOneName + 'EndingValue',  'ModelEndingValue', modelOneName + 'Gain', 'ModelGain', 'Difference']))
	TestResults.set_index(['StartDate'], inplace=True)		
	trials = int((endYear - startYear)/durationInYears) 
	for i in range(trials):
		startDate = '1/2/' + str(startYear + i * durationInYears)
		m1ev = RunBuyHold('.INX', startDate=startDate, durationInYears=durationInYears, ReEvaluationInterval=ReEvaluationInterval, portfolioSize=portfolioSize)
		m2ev = RunPriceMomentumBlended(tickerList = TickerLists.SPTop70(), startDate=startDate, durationInYears=durationInYears,  ReEvaluationInterval=ReEvaluationInterval, longHistory=longHistory, shortHistory=shortHistory, portfolioSize=portfolioSize, returndailyValues=False, verbose=False)
		m1pg = (m1ev/portfolioSize) - 1 
		m2pg = (m2ev/portfolioSize) - 1
		TestResults.loc[startDate] = [durationInYears, m1ev, m2ev, m1pg, m2pg, m2pg-m1pg]
	TestResults.sort_values(['Difference'], axis=0, ascending=True, inplace=True)
	TestResults.to_csv('data/trademodel/Compare' + modelOneName + '_to_' + modelTwoName + '_year ' + str(startYear) + '_duration' + str(durationInYears) +'.csv')
	print(TestResults)

def ComparePVToBH(startYear:int=1982, endYear:int=2018, durationInYears:int=1, stockCount:int=9, ReEvaluationInterval:int=20):
	modelOneName = 'BuyHold'
	modelTwoName = 'Point_Value_ReEval_' + str(ReEvaluationInterval) + '_stockcount_' + str(stockCount) 
	portfolioSize=30000
	TestResults = pd.DataFrame(columns=list(['StartDate','Duration', modelOneName + 'EndingValue',  'ModelEndingValue', modelOneName + 'Gain', 'ModelGain', 'Difference']))
	TestResults.set_index(['StartDate'], inplace=True)		
	trials = int((endYear - startYear)/durationInYears) 
	for i in range(trials):
		startDate = '1/2/' + str(startYear + i * durationInYears)
		m1ev = RunBuyHold('.INX', startDate=startDate, durationInYears=durationInYears, ReEvaluationInterval=ReEvaluationInterval, portfolioSize=portfolioSize)
		m2ev = RunPoint_Value(tickerList = TickerLists.SPTop70(), startDate=startDate, durationInYears=durationInYears, stockCount=stockCount, ReEvaluationInterval=ReEvaluationInterval, portfolioSize=portfolioSize, returndailyValues=False, verbose=False)
		m1pg = (m1ev/portfolioSize) - 1 
		m2pg = (m2ev/portfolioSize) - 1
		TestResults.loc[startDate] = [durationInYears, m1ev, m2ev, m1pg, m2pg, m2pg-m1pg]
	TestResults.sort_values(['Difference'], axis=0, ascending=True, inplace=True)
	TestResults.to_csv('data/trademodel/Compare' + modelOneName + '_to_' + modelTwoName + '_year ' + str(startYear) + '_duration' + str(durationInYears) +'.csv')
	print(TestResults)

if __name__ == '__main__':
	switch = 0
	if len(sys.argv[1:]) > 0: switch = sys.argv[1:][0]
	tickers = TickerLists.SPTop70()
	if switch == '1':
		print('Running option: ', switch)
		RunPriceMomentum(tickerList = tickers, startDate='1/1/1982', durationInYears=36, stockCount=5, ReEvaluationInterval=20, filterOption=2, longHistory=365, shortHistory=90) 
		CompareBlendedToBH(startYear=2000,endYear=2010, durationInYears=1, ReEvaluationInterval=5) 
		#CompareBlendedToBH(startYear=1982,endYear=2018, durationInYears=1, ReEvaluationInterval=30) 
		#ComparePMToBH(startYear=1982,endYear=2018, durationInYears=1, ReEvaluationInterval=20, stockCount=2, filterOption=3, longHistory=365, shortHistory=90) 
	elif switch == '2':
		print('Running option: ', switch)
		ComparePMToBH(startYear=2000,endYear=2010, durationInYears=1, ReEvaluationInterval=20, stockCount=5, filterOption=2, longHistory=120, shortHistory=60) 
		#ComparePMToBH(startYear=1982,endYear=2018, durationInYears=1, ReEvaluationInterval=20, stockCount=5, filterOption=2, longHistory=180, shortHistory=60) 
		#ComparePMToBH(startYear=1982,endYear=2018, durationInYears=1, ReEvaluationInterval=20, stockCount=5, filterOption=2, longHistory=240, shortHistory=60) 
	elif switch == '3':
		print('Running option: ', switch)
		#ComparePMToBH(startYear=1982,endYear=2018, durationInYears=1, ReEvaluationInterval=20, stockCount=2, filterOption=0, longHistory=365, shortHistory=90) 
		#ComparePMToBH(startYear=1982,endYear=2018, durationInYears=1, ReEvaluationInterval=20, stockCount=2, filterOption=2, longHistory=365, shortHistory=90) 
		#CompareBlendedToBH(startYear=1982,endYear=2018, durationInYears=1, ReEvaluationInterval=5, longHistory=365, shortHistory=90)
		#CompareBlendedToBH(startYear=1982,endYear=2018, durationInYears=1, ReEvaluationInterval=10, longHistory=365, shortHistory=90)
		CompareBlendedToBH(startYear=1982,endYear=2018, durationInYears=1, ReEvaluationInterval=15, longHistory=365, shortHistory=90)
	else:
		tickers = TickerLists.SPTop70()
		print('Running default option on ' + str(len(tickers)) + ' stocks.')
		RunBuyHold('.INX', startDate='1/1/1982', durationInYears=36, ReEvaluationInterval=5, portfolioSize=30000, verbose=False)	#Baseline
		RunPriceMomentum(tickerList = tickers, startDate='1/1/1982', durationInYears=36, stockCount=5, ReEvaluationInterval=20, filterOption=4, longHistory=365, shortHistory=90) #Shows how the strategy works over a long time period
		ComparePMToBH(startYear=1982,endYear=2018, durationInYears=1, ReEvaluationInterval=20, stockCount=5, filterOption=1, longHistory=365, shortHistory=60) #Runs the model in one year intervals, comparing each to BuyHold
		RunPoint_Value(tickerList = tickers, startDate='1/1/1982', durationInYears=36, stockCount=5, ReEvaluationInterval=30)
