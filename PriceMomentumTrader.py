import sys
import pandas as pd
import _classes.Constants as CONSTANTS
from _classes.Prices import PricingData, PriceSnapshot
from _classes.Trading import TradingModel, TradeModelParams, Position
from _classes.Selection import StockPicker
from _classes.TickerLists import TickerLists
from _classes.Utility import *

def RunBuyHold(ticker: str, startDate:str, durationInYears:int, reEvaluationInterval:int=20, portfolioSize:int=100000, verbose:bool=False):
	#Baseline model to compare against.  Buy on day one, hold for the duration and then sell
	startDate = ToDate(startDate)
	modelName = 'BuyHold_' + (ticker) + '_' + str(startDate)[:10]
	startDate = ToDate(startDate)
	endDate =  AddDays(startDate, 365 * durationInYears)
	tm = TradingModel(modelName=modelName, startingTicker=ticker, startDate=startDate, durationInYears=durationInYears, totalFunds=portfolioSize, verbose=verbose)
	if not tm.modelReady:
		print(' RunBuyHold: Unable to initialize price history for model BuyHold date ' + str(startDate))
		return 0
	else:
		dayCounter =0
		while not tm.ModelCompleted():
			if dayCounter ==0:
				cash = tm.GetAvailableCash()
				price = tm.GetPrice(ticker)
				if price:
					units = int(cash/price)
					if units > 0:
						tm.PlaceBuy(ticker=ticker, units=units, price=price, marketOrder=True, expireAfterDays=5, verbose=verbose)
			dayCounter+=1
			if dayCounter >= reEvaluationInterval: dayCounter=0
			tm.ProcessDay()
		cash, asset = tm.GetValue()
		if verbose: print(' RunBuyHold: Ending Value: ', cash + asset, '(Cash', cash, ', Asset', asset, ')')
		return tm.CloseModel()	

def RunPriceMomentum(tickerList:list, startDate:str='1/1/1982', durationInYears:int=36, stockCount:int=9, reEvaluationInterval:int=20, filterOption:int=3, longHistory:int=365, shortHistory:int=90, minPercentGain=0.05, portfolioSize:int=30000, returndailyValues:bool=False, verbose:bool=False):
	#Choose stockCount stocks with the greatest long term (longHistory days) price appreciation, using different filter options defined in the StockPicker class
	#shortHistory is a shorter time frame (like 90 days) used differently by different filters
	#reEvaluationInterval is how often to re-evaluate our choices, ideally this should be very short and not matter, otherwise the date selection is biased.
	startDate = ToDate(startDate)
	endDate =  AddDays(startDate, 365 * durationInYears)
	picker = StockPicker(startDate, endDate) 
	picker.AlignToList(tickerList)
	modelName = f"PriceMomentumShort_longHistory_{longHistory}_shortHistory_{shortHistory}_reeval_{reEvaluationInterval}_stockcount_{stockCount}_filter{filterOption}_{minPercentGain}"
	tm = TradingModel(modelName=modelName, startingTicker='.INX', startDate=startDate, durationInYears=durationInYears, totalFunds=portfolioSize, verbose=verbose)
	dayCounter = 0
	currentYear = 0
	if not tm.modelReady:
		print('Unable to initialize price history for PriceMomentum date ' + str(startDate))
		return 0
	else:
		while not tm.ModelCompleted():
			currentDate =  tm.currentDate
			if dayCounter ==0:
				c, a = tm.GetValue()
				candidates = picker.GetHighestPriceMomentum(currentDate, stocksToReturn=stockCount, filterOption=filterOption, minPercentGain=minPercentGain)
				candidates = candidates.groupby(level=0)[['Point_Value']].sum().rename(columns={'Point_Value': 'TargetHoldings'})
				tm.AlignPositions(targetPositions=candidates)
			tm.ProcessDay()
			dayCounter+=1
			if dayCounter >= reEvaluationInterval: dayCounter=0
		cv1 = tm.CloseModel()
		if returndailyValues:
			return tm.GetDailyValue()
		else:
			return cv1
			
def ComparePMToBH(startYear:int=1982, endYear:int=2018, durationInYears:int=1, stockCount:int=9, reEvaluationInterval:int=20, filterOption:int=3, longHistory:int=365, shortHistory:int=90):
	#Compares the PriceMomentum strategy to BuyHold in one year intervals, outputs the returns to .csv file
	modelOneName = 'BuyHold'
	modelTwoName = 'PriceMomentum_longHistory_' + str(longHistory) +'_shortHistory_' + str(shortHistory) + '_ReEval_' + str(reEvaluationInterval) + '_stockcount_' + str(stockCount) + '_filter' + str(filterOption)
	portfolioSize=30000
	TestResults = pd.DataFrame(columns=list(['StartDate','Duration', modelOneName + 'EndingValue',  'ModelEndingValue', modelOneName + 'Gain', 'ModelGain', 'Difference']))
	TestResults.set_index(['StartDate'], inplace=True)		
	trials = int((endYear - startYear)/durationInYears) 
	for i in range(trials):
		startDate = '1/2/' + str(startYear + i * durationInYears)
		m1ev = RunBuyHold('.INX', startDate=startDate, durationInYears=durationInYears, reEvaluationInterval=reEvaluationInterval, portfolioSize=portfolioSize)
		m2ev = RunPriceMomentum(tickerList = TickerLists.SPTop70(), startDate=startDate, durationInYears=durationInYears, stockCount=stockCount, reEvaluationInterval=reEvaluationInterval, filterOption=filterOption,  longHistory=longHistory, shortHistory=shortHistory, portfolioSize=portfolioSize, returndailyValues=False, verbose=False)
		m1pg = (m1ev/portfolioSize) - 1 
		m2pg = (m2ev/portfolioSize) - 1
		TestResults.loc[startDate] = [durationInYears, m1ev, m2ev, m1pg, m2pg, m2pg-m1pg]
	TestResults.sort_values(['Difference'], axis=0, ascending=True, inplace=True)
	TestResults.to_csv('data/trademodel/Compare' + modelOneName + '_to_' + modelTwoName + '_year ' + str(startYear) + '_duration' + str(durationInYears) +'.csv')
	print(TestResults)

def ExtensiveTesting1():
	#Helper subroutine for running multiple tests
	RunPriceMomentum(tickerList = tickers, startDate='1/1/1982', durationInYears=36, stockCount=5, reEvaluationInterval=20, filterOption=2, longHistory=365, shortHistory=30) 
	ComparePMToBH(startYear=1982,endYear=2018, durationInYears=1, reEvaluationInterval=20, stockCount=9, filterOption=3, longHistory=365, shortHistory=90) 

def ExtensiveTesting2():
	#Helper subroutine for running multiple tests
	ComparePMToBH(startYear=1982,endYear=2018, durationInYears=1, reEvaluationInterval=20, stockCount=9, filterOption=2, longHistory=120, shortHistory=90) 
	ComparePMToBH(startYear=1982,endYear=2018, durationInYears=1, reEvaluationInterval=20, stockCount=9, filterOption=2, longHistory=180, shortHistory=90) 
	ComparePMToBH(startYear=1982,endYear=2018, durationInYears=1, reEvaluationInterval=20, stockCount=9, filterOption=2, longHistory=240, shortHistory=90) 

def ExtensiveTesting3():
	#Helper subroutine for running multiple tests
	ComparePMToBH(startYear=1982,endYear=2018, durationInYears=1, reEvaluationInterval=20, stockCount=9, filterOption=1, longHistory=365, shortHistory=90) 
	ComparePMToBH(startYear=1982,endYear=2018, durationInYears=1, reEvaluationInterval=20, stockCount=9, filterOption=3, longHistory=365, shortHistory=90) 
	ComparePMToBH(startYear=1982,endYear=2018, durationInYears=1, reEvaluationInterval=20, stockCount=9, filterOption=4, longHistory=365, shortHistory=90) 
	
def ModelPastYear():
	#Show how each strategy performs on the past years data
	startDate = AddDays(GetLatestBDay(), -370)
	RunPriceMomentum(tickerList = tickers, startDate=startDate, durationInYears=1, stockCount=5, reEvaluationInterval=20, verbose=True)
	RunBuyHold(ticker='.INX', startDate=startDate, durationInYears=1)

if __name__ == '__main__':
	switch = 0
	if len(sys.argv[1:]) > 0: switch = sys.argv[1:][0]
	tickers = TickerLists.SPTop70()
	if switch == '1':
		print('Running option: ', switch)
		ExtensiveTesting1()
	elif switch == '2':
		print('Running option: ', switch)
		ExtensiveTesting2()
	elif switch == '3':
		print('Running option: ', switch)
		ExtensiveTesting3()
	elif switch == '4':
		print('Running option: ', switch)
		ModelPastYear()
	else:
		tickers = TickerLists.SPTop70()
		print('Running default option on ' + str(len(tickers)) + ' stocks.')
		#RunBuyHold('.INX', startDate='1/1/2000', durationInYears=10, reEvaluationInterval=5, portfolioSize=30000, verbose=False)	#Baseline
		RunPriceMomentum(tickerList = tickers, startDate='1/1/2000', durationInYears=10, stockCount=5, reEvaluationInterval=20, filterOption=4, longHistory=365, shortHistory=90) #Shows how the strategy works over a long time period
		ComparePMToBH(startYear=2000,endYear=2018, durationInYears=1, reEvaluationInterval=20, stockCount=5, filterOption=1, longHistory=365, shortHistory=60) #Runs the model in one year intervals, comparing each to BuyHold
		#ComparePMToBH(startYear=1982,endYear=2018, durationInYears=1, reEvaluationInterval=20, stockCount=5, filterOption=2, longHistory=365, shortHistory=60) #Runs the model in one year intervals, comparing each to BuyHold
		#ComparePMToBH(startYear=1982,endYear=2018, durationInYears=1, reEvaluationInterval=20, stockCount=5, filterOption=4, longHistory=365, shortHistory=60) #Runs the model in one year intervals, comparing each to BuyHold
