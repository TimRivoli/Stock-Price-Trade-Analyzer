#The forces that propelled a price to do well for 12 months, will continue to do so for another month, for a top SP500 stock that rate will average about 2.66%/mo or 32%/year
#Extending this through low cap stocks (<5BUSD) increases average yield by about 20% and decreases the worst year loss by about the same.
#Some of that may be a result of not having market capitalization data on historical stocks, than can be tested by lowering the market cap to $1
#reEvaluationInterval of 20 or 40 work best, much better than 30, the filters often do little, most significant factor is sort by 1 year price change descending, some gain for filtering out high stdev
import sys
import pandas as pd
import _classes.Constants as CONSTANTS
from _classes.Prices import PricingData
from _classes.Trading import TradingModel, TradeModelParams, ExtensiveTesting
from _classes.Selection import StockPicker
from _classes.TickerLists import TickerLists
from _classes.Utility import *
from datetime import datetime

#------------------------------------------- Specific model functions  ----------------------------------------------		

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

def RunPriceMomentum(params: TradeModelParams):
	#Choose stockCount stocks with the greatest long term (longHistory days) price appreciation, using different filter options defined in the StockPicker class
	#shortHistory is a shorter time frame (like 90 days) used differently by different filters
	#reEvaluationInterval is how often to reevaluate our choices, ideally this should be very short and not matter, otherwise the date selection is biased.

	modelName = 'PM_filter' + str(params.filterOption)
	if params.allocateByPointValue: modelName += "_PVAlloc"
	tm = TradingModel(modelName=modelName, startingTicker=CONSTANTS.CASH_TICKER, startDate=params.startDate, durationInYears=params.durationInYears, totalFunds=params.portfolioSize, trancheSize=params.trancheSize, verbose=params.verbose)
	dayCounter = 0
	currentYear = 0
	tickerList = TickerLists.SPTop70()
	picker = StockPicker(params.startDate, params.endDate) 
	picker.AlignToList(tickerList)
	if not tm.modelReady:
		print(f" RunPriceMomentum: Unable to initialize price history for PriceMomentum date {startDate}")
	else:
		while not tm.ModelCompleted(): 
			currentDate =  tm.currentDate #These are calendar days but trades are only processed on weekdays
			if dayCounter ==0: 
				c, a = tm.Value()
				available, buyPending, sellPending, longPostitions = tm.PositionSummary()
				percentAvailable = (available + buyPending)/(available + buyPending + sellPending + longPostitions)
				candidates = picker.GetHighestPriceMomentum(currentDate, stocksToReturn=params.stockCount, filterOption=params.filterOption, minPercentGain=params.minPercentGain)
				if len(candidates) > 0:
					if params.allocateByPointValue:
						candidates = candidates.groupby(level=0)[['Point_Value']].sum().rename(columns={'Point_Value': 'TargetHoldings'})
					else:
						candidates = candidates.groupby(level=0).size().to_frame(name='TargetHoldings')
					tm.AlignPositions(targetPositions=candidates, rateLimitTransactions=params.rateLimitTransactions, shopBuyPercent=params.shopBuyPercent, shopSellPercent=params.shopSellPercent, trimProfitsPercent=params.trimProfitsPercent, verbose=False) 
			tm.ProcessDay()
			dayCounter+=1
			if dayCounter >= params.reEvaluationInterval: dayCounter=0
	closing_value = tm.CloseModel(params)
	return closing_value

def RunPriceMomentumAdaptiveConvex(params: TradeModelParams):
	modelName =  'AdaptiveConvex_v2.4'
	if params.filterByFundamentals:
		modelName += "_filterByFundamentals"
	elif params.SP500Only: 
		modelName += "_SP500"
	if params.rateLimitTransactions:  modelName += "_RateLimit"
	tm = TradingModel(modelName=modelName, startingTicker = CONSTANTS.CASH_TICKER, startDate=params.startDate, durationInYears=params.durationInYears, totalFunds=params.portfolioSize, trancheSize=params.trancheSize, verbose=params.verbose)
	dayCounter = 0
	currentYear = 0
	tickerList = TickerLists.SPTop70()
	picker = StockPicker(params.startDate, params.endDate) 
	picker.AlignToList(tickerList)
	while not tm.ModelCompleted():
		currentDate = tm.currentDate
		if dayCounter == 0:
			c, a = tm.Value()
			candidates = picker.GetAdaptiveConvex(currentDate, modelName)
			if len(candidates) > 0:
				if params.allocateByPointValue:
					candidates = candidates.groupby(level=0)[['Point_Value']].sum().rename(columns={'Point_Value': 'TargetHoldings'})
				else:
					candidates = candidates.groupby(level=0).size().to_frame(name='TargetHoldings')
				tm.AlignPositions(targetPositions = candidates, rateLimitTransactions = params.rateLimitTransactions, shopBuyPercent = params.shopBuyPercent, shopSellPercent = params.shopSellPercent, trimProfitsPercent = params.trimProfitsPercent, verbose = False) 
		tm.ProcessDay()
		dayCounter += 1
		if dayCounter >= params.reEvaluationInterval: dayCounter = 0
	closing_value = tm.CloseModel(params)
	return closing_value
	
def RunPriceMomentumBlended(params: TradeModelParams):
	#Uses blended option for selecting stocks using three different filters, produces the best overall results.
	#1 long term performer at short term discount
	#2 long term performer
	#4 short term performer
	#44 Short term gain meets min requirements, sort long value
	#5 point value, aka PV
	modelName = 'PM_Blended_3.3.44.PV' 
	if params.filterByFundamentals:
		modelName += "_filterByFundamentals"
	elif params.SP500Only: 
		modelName += "_SP500"
	if params.rateLimitTransactions:  modelName += "_RateLimit"
	modelName += '_reeval_' + str(params.reEvaluationInterval)
	tradeLengthDays = 3 #This is the expireAfterDays and also how often to AlignPositions so as not to execute multiple trades with trades pending (would require evalution of pending)
	print('RunPriceMomentumBlended ', params.startDate, params.endDate)
	tm = TradingModel(modelName=modelName , startingTicker=CONSTANTS.CASH_TICKER, startDate=params.startDate, durationInYears=params.durationInYears, totalFunds=params.portfolioSize, trancheSize=params.trancheSize, verbose=params.verbose)
	dayCounter = 0
	currentYear = 0
	tickerList = TickerLists.SPTop70()
	picker = StockPicker(params.startDate, params.endDate) 
	picker.AlignToList(tickerList)
	if not tm.modelReady:
		print('Unable to initialize price history for PriceMomentum date ' + str(startDate))
	else:
		while not tm.ModelCompleted():
			currentDate =  tm.currentDate #These are calendar days but trades are only processed on weekdays
			if dayCounter == 0: #New picks on reevaluation interval
				c, a = tm.Value()
				print(f"Model: {tm.modelName} Date: {currentDate} Cash: ${int(c)} Assets: ${int(a)} Total: ${int(c+a)} Return: {round(100*(((c+a)/int(params.portfolioSize))-1), 2)}%")
				available, buyPending, sellPending, longPostitions = tm.PositionSummary()
				percentAvailable = (available + buyPending)/(available + buyPending + sellPending + longPostitions)
				candidates = picker.GetPicksBlended(currentDate=currentDate)
				tm.AlignPositions(targetPositions=candidates, rateLimitTransactions=params.rateLimitTransactions, shopBuyPercent=params.shopBuyPercent, shopSellPercent=params.shopSellPercent, trimProfitsPercent=params.trimProfitsPercent, verbose=False) 
			tm.ProcessDay() #Adjust prices and trades every day
			dayCounter+=1
			if dayCounter >= params.reEvaluationInterval: dayCounter=0
	closing_value = tm.CloseModel(params)
	return closing_value

#-------------------------------------------  Specific Tests ----------------------------------------------		

def ModelPastFewYears(params: TradeModelParams):
	#Show how each strategy performs on the recent years 
	years = 4
	endDate = GetLatestBDay()
	startDate = AddDays(endDate, -365 * years)
	params.startDate = startDate
	params.durationInYears = years
	params.verbose = True
	#RunPriceMomentumBlended(params)
	#RunPriceMomentum(params)
	params.modelName = 'AdaptiveConvexRecent'
	params.reEvaluationInterval = 5
	RunPriceMomentumAdaptiveConvex(params)
	
def ExtensiveTestingAddTests():
	tester = ExtensiveTesting(verbose=False)
	tester.ensure_queue_table()
	params = TradeModelParams()
	params.startDate = '1/1/1980'
	params.durationInYears = 45
	params.marketCapMin = 0
	params.shopBuyPercent = 0
	params.shopSellPercent = 0
	params.trimProfitsPercent = 0
	params.SP500Only = False
	params.filterOption = 3
	for i in [5, 12, 20]:
		params.stockCount = i
		tester.add_to_queue(params)
	# for i in [1, 2, 3, 4, 5, 44]:
	# for i in [0, 55]:
		# params.marketCapMin = 0
		# params.filterOption = i
		# params.SP500Only = True
		# tester.add_to_queue(params)
		# params.SP500Only = False
		# tester.add_to_queue(params)
		# params.marketCapMin = 1000
		# tester.add_to_queue(params)

def ExtensiveTestingRun():
	max_jobs = 20
	job_count = 0
	tester = ExtensiveTesting(verbose=True)
	while job_count < max_jobs:
		params = tester.pop_from_queue()
		if params is None:
			break
		if params.filterOption ==0:
			RunPriceMomentumBlended(params)
		elif params.filterOption ==55:
			RunPriceMomentumAdaptiveConvex(params)
		else:
			RunPriceMomentum(params)
		job_count += 1

#-------------------------------------------  Past Year Function----------------------------------------------		

if __name__ == '__main__':
	switch = 0
	if len(sys.argv[1:]) > 0: switch = sys.argv[1:][0]
	params = TradeModelParams()
	params.startDate = '1/1/1999'
	params.durationInYears = 15
	params.SP500Only = False
	params.filterByFundamentals = False
	params.marketCapMin = 0	 #Million USD, so Tiny < 750 < Small < 2,000 < Mid < 10,000 < Large
	params.marketCapMax = 0	 
	params.shopBuyPercent = 0
	params.shopSellPercent = 0
	params.trimProfitsPercent = 0
	params.reEvaluationInterval = 20
	params.stockCount = 10
	params.filterOption = 5
	params.saveResults = True
	params.verbose = False

	if switch == '1':
		print('Running option 1 RunPriceMomentum')
		params.filterOption = 2
		RunPriceMomentum(params)
	elif switch == '2':
		print('Running option 2 RunPriceMomentumAdaptiveConvex')		
		params.filterOption = 55
		RunPriceMomentumAdaptiveConvex(params)
	elif switch == '3':
		print('Running option 3 RunPriceMomentumBlended')		
		RunPriceMomentumBlended(params)
	elif switch == '4':
		print('Running option 4 ModelPastFewYears')
		ModelPastFewYears(params) 
	elif switch == '5':
		ExtensiveTestingRun()
	else:
		params.filterOption = 3
		RunPriceMomentum(params)
		params.filterOption = 55
		RunPriceMomentumAdaptiveConvex(params)
		RunPriceMomentumBlended(params)
