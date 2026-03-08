import sys, time
import pandas as pd
import _classes.Constants as CONSTANTS
from _classes.DataIO import PTADatabase
from _classes.Prices import PricingData
from _classes.Trading import TradingModel, TradeModelParams, ExtensiveTesting, UpdateTradeModelComparisonsFromDailyValue
from _classes.Selection import StockPicker
from _classes.TickerLists import TickerLists
from _classes.Utility import *
from datetime import datetime
use_sql = PTADatabase().database_configured

#------------------------------------------- Specific model functions  ----------------------------------------------		
def ModelSP500(startDate: str = '1/1/2000', durationInYears:int = 10):
	#Baseline model to compare against.  Buy on day one, hold for the duration and then sell
	ticker = '.INX'
	modelName = 'ModelSP500_' + str(startDate)[:10]
	params = TradeModelParams()
	params.startDate = startDate
	params.durationInYears = durationInYears
	params.saveResults = True
	tm = TradingModel(modelName=modelName, startingTicker=ticker, startDate=params.startDate, durationInYears=params.durationInYears, totalFunds=params.portfolioSize, verbose=False)
	tickerList = TickerLists.SP500_2026()
	if not tm.modelReady:
		print(' ModelSP500: Unable to initialize price history for date ' + str(startDate))
		return 0
	else:
		dayCounter = 0
		while not tm.ModelCompleted():
			if dayCounter == 0:
				cash = tm.GetAvailableCash()
				price = tm.GetPrice(ticker)
				if price:
					units = int(cash/price)
					if units > 0:
						tm.PlaceBuy(ticker=ticker, units=units, price=price, marketOrder=True, expireAfterDays=5, verbose=params.verbose)
			dayCounter+=1
			if dayCounter >= params.reEvaluationInterval: dayCounter=0
			tm.ProcessDay()
		params = TradeModelParams()
		return tm.CloseModel(params)		

def RunPriceMomentum(params: TradeModelParams):
	#Choose stockCount stocks with the greatest long term (longHistory days) price appreciation, using different filter options defined in the StockPicker class
	#shortHistory is a shorter time frame (like 90 days) used differently by different filters
	#reEvaluationInterval is how often to reevaluate our choices, ideally this should be very short and not matter, otherwise the date selection is biased.
	
	params.modelName = ''
	if params.pickHistoryWindow > 0: params.allocateByPointValue = False #Much better performance with this
	params.AddModelNameModifiers()
	tm = TradingModel(modelName=params.modelName, startingTicker=CONSTANTS.CASH_TICKER, startDate=params.startDate, durationInYears=params.durationInYears, totalFunds=params.portfolioSize, verbose=params.verbose)
	tickerList = TickerLists.SP500_2026()
	dayCounter = 0
	currentYear = 0
	picker = StockPicker(startDate=params.startDate, endDate=params.endDate, pickHistoryWindow=params.pickHistoryWindow) 
	if not tm.modelReady:
		print(f" RunPriceMomentum: Unable to initialize price history for PriceMomentum date {startDate}")
	else:
		while not tm.ModelCompleted(): 
			currentDate = tm.currentDate 
			if currentYear != currentDate.year:
				currentYear = currentDate.year
				if use_sql: tickerList = TickerLists.GetTickerListSQL(currentYear, SP500Only=params.SP500Only, filterByFundamentals=params.filterByFundamentals,  marketCapMax=params.marketCapMax, marketCapMin=params.marketCapMin)
				picker.AlignToList(tickerList)
			if dayCounter ==0: 				
				candidates = picker.GetHighestPriceMomentum(currentDate, stocksToReturn=params.stockCount, filterOption=params.filterOption, minPercentGain=params.minPercentGain, allocateByTargetHoldings=(not params.allocateByPointValue), allocateByPointValue=params.allocateByPointValue, useRollingWindow=True)
				tm.AlignPositions(targetPositions=candidates, rateLimitTransactions=params.rateLimitTransactions, shopBuyPercent=params.shopBuyPercent, shopSellPercent=params.shopSellPercent) 
			elif (tm.GetIdleCashPCT() >= .005) or (dayCounter % 3 == 1) and (params.rateLimitTransactions or params.trimProfitsPercent > 0):
				tm.AlignPositions(targetPositions=candidates, rateLimitTransactions=params.rateLimitTransactions, shopBuyPercent=params.shopBuyPercent, shopSellPercent=params.shopSellPercent) 
				if params.trimProfitsPercent: tm.TrimProfits(params.trimProfitsPercent)
			tm.ProcessDay()
			dayCounter+=1
			if dayCounter >= params.reEvaluationInterval: dayCounter=0
	closing_value = tm.CloseModel(params)
	return closing_value

def RunPriceMomentumAdaptiveConvex(params: TradeModelParams):
	params.filterOption = 98
	params.allocateByPointValue = False
	params.modelName = f"AdaptiveConvex_v4"
	params.pickHistoryWindow = -1
	params.AddModelNameModifiers()
	tm = TradingModel(modelName=params.modelName, startingTicker = CONSTANTS.CASH_TICKER, startDate=params.startDate, durationInYears=params.durationInYears, totalFunds=params.portfolioSize, verbose=params.verbose)
	tickerList = TickerLists.SP500_2026()
	dayCounter = 0
	currentYear = 0
	picker = StockPicker(startDate=params.startDate, endDate=params.endDate, pickHistoryWindow=params.pickHistoryWindow) 
	while not tm.ModelCompleted():
		currentDate = tm.currentDate
		if currentYear != currentDate.year:
			currentYear = currentDate.year
			if use_sql: tickerList = TickerLists.GetTickerListSQL(currentYear, SP500Only=params.SP500Only, filterByFundamentals=params.filterByFundamentals,  marketCapMax=params.marketCapMax, marketCapMin=params.marketCapMin)
			#tickerList = tickerList[:10]
			picker.AlignToList(tickerList)
		if dayCounter == 0:
			candidates = picker.GetAdaptiveConvexPicks(currentDate)
			tm.AlignPositions(targetPositions = candidates, rateLimitTransactions = params.rateLimitTransactions, shopBuyPercent = params.shopBuyPercent, shopSellPercent = params.shopSellPercent) 
		elif (tm.GetIdleCashPCT() >= .005) or (dayCounter % 3 == 1) and (params.rateLimitTransactions or params.trimProfitsPercent > 0):
			tm.AlignPositions(targetPositions = candidates, rateLimitTransactions = params.rateLimitTransactions, shopBuyPercent = params.shopBuyPercent, shopSellPercent = params.shopSellPercent) 
			if params.trimProfitsPercent: tm.TrimProfits(params.trimProfitsPercent)
		tm.ProcessDay()
		dayCounter += 1
		if dayCounter >= params.reEvaluationInterval: dayCounter = 0
	closing_value = tm.CloseModel(params)
	return closing_value
	
def RunPriceMomentumBlended(params: TradeModelParams, filter_options: dict = None):
	#Uses blended option for selecting stocks using three different filters, produces the best overall results.
	if not filter_options: filter_options = {3:3, 3:3, 1:3, 5:4, 9:2} #CAGR 52.00, MadDD -52.03
	filters = list(filter_options.keys())	
	filter_string = ".".join(str(f) for f in filters)
	params.filterOption = 99
	params.allocateByPointValue = False
	params.modelName = "PM_Blended" 
	if not params.useDatabase: 
		params.modelName +=f"_({filter_string})" #Note: if using the database records, you are using whatever filters they were created with
	params.AddModelNameModifiers()
	print(f" Running {params.modelName} ", params.startDate, params.endDate)
	picker = StockPicker(startDate=params.startDate, endDate=params.endDate, pickHistoryWindow=params.pickHistoryWindow) 
	tm = TradingModel(modelName=params.modelName , startingTicker=CONSTANTS.CASH_TICKER, startDate=params.startDate, durationInYears=params.durationInYears, totalFunds=params.portfolioSize, verbose=params.verbose)
	tickerList = TickerLists.SP500_2026()
	dayCounter = 0
	currentYear = 0
	if not tm.modelReady:
		print('Unable to initialize price history for PriceMomentum date ' + str(startDate))
	else:
		while not tm.ModelCompleted():
			currentDate =  tm.currentDate #These are calendar days but trades are only processed on weekdays
			if currentYear != currentDate.year and not params.useDatabase:
				currentYear = currentDate.year
				if use_sql: tickerList = TickerLists.GetTickerListSQL(currentYear, SP500Only=params.SP500Only, filterByFundamentals=params.filterByFundamentals, marketCapMin=params.marketCapMin, marketCapMax=params.marketCapMax)
				picker.AlignToList(tickerList)
			if dayCounter == 0: #New picks on reevaluation interval
				if params.useDatabase:
					candidates = picker.GetPicksBlendedSQL(currentDate=currentDate, sqlHistory=params.pickHistoryWindow)
				else:
					candidates = picker.GetPicksBlended(currentDate=currentDate, filter_options=filter_options, useRollingWindow=(params.pickHistoryWindow>0) )
				tm.AlignPositions(targetPositions = candidates, rateLimitTransactions = params.rateLimitTransactions, shopBuyPercent = params.shopBuyPercent, shopSellPercent = params.shopSellPercent) 
			elif (tm.GetIdleCashPCT() >= .005) or (dayCounter % 3 == 1) and (params.rateLimitTransactions or params.trimProfitsPercent > 0):
				tm.AlignPositions(targetPositions = candidates, rateLimitTransactions = params.rateLimitTransactions, shopBuyPercent = params.shopBuyPercent, shopSellPercent = params.shopSellPercent) 
				if params.trimProfitsPercent: tm.TrimProfits(params.trimProfitsPercent)
			tm.ProcessDay() #Adjust prices and trades every day
			dayCounter+=1
			if dayCounter >= params.reEvaluationInterval: dayCounter=0
	closing_value = tm.CloseModel(params)
	return closing_value

#-------------------------------------------  Specific Tests ----------------------------------------------		

def ModelPastFewYears(params: TradeModelParams):
	#Show how each strategy performs on the recent years 
	years = 5
	endDate = GetLatestBDay()
	startDate = AddDays(endDate, -365 * years)
	params.startDate = startDate
	params.durationInYears = years
	params.reEvaluationInterval = 5
	params.rateLimitTransactions = True
	params.verbose = True	
	params.filterOption = 5
	RunPriceMomentum(params)
	params.useDatabase = False
	RunPriceMomentumBlended(params)
	params.modelName = ''
	RunPriceMomentumAdaptiveConvex(params)
	
def ExtensiveTestingAddTests():
	tester = ExtensiveTesting(verbose=False)
	tester.ensure_queue_table()
	params = TradeModelParams()
	params.startDate = '1/1/1980'
	params.durationInYears = 45
	params.SP500Only = False
	params.marketCapMin = 0
	params.shopBuyPercent = 0.0
	params.shopSellPercent = 0.0
	params.trimProfitsPercent = 0.0
	params.useDatabase = False
	params.rateLimitTransactions = True
	for fo in [1,2,3,4,5,6,98,99]:
		for i in [5, 10, 20]:
			params.filterOption = fo
			params.reEvaluationInterval = i
			tester.add_to_queue(params)
	
def ExtensiveTestingRun():
	#max_jobs = 20
	job_count = 0
	tester = ExtensiveTesting(verbose=True)
	while True:
		queue_id, params = tester.claim_from_queue()
		if params is None:
			print(f"Queue empty. Waiting 30 seconds... ({job_count} completed)")
			time.sleep(30)
			continue  # Jump back to the start of the while loop
		if params.filterOption == 99:
			RunPriceMomentumBlended(params)
		elif params.filterOption == 98:
			RunPriceMomentumAdaptiveConvex(params)
		else:
			RunPriceMomentum(params)
		tester.complete_job(queue_id)
		job_count += 1

#-------------------------------------------  Past Year Function----------------------------------------------		

if __name__ == '__main__':
	switch = 0
	if len(sys.argv[1:]) > 0: switch = sys.argv[1:][0]
	params = TradeModelParams()
	params.startDate = '1/1/1980'
	params.durationInYears = 45
	params.SP500Only = False
	params.filterByFundamentals = False
	params.marketCapMin = 0	 #Million USD, so Tiny < 750 < Small < 2,000 < Mid < 10,000 < Large
	params.marketCapMax = 0	 
	params.shopBuyPercent = 0.0
	params.shopSellPercent = 0.0
	params.trimProfitsPercent = 0.0
	params.reEvaluationInterval = 5
	params.stockCount = 15
	params.pickHistoryWindow = 26	#26 days is optimal
	params.rateLimitTransactions = True
	params.allocateByPointValue = False
	params.saveResults = True
	params.verbose = True

	if switch == '1':
		print('Running option 1 RunPriceMomentum')
		params.filterOption = 5
		RunPriceMomentum(params)
	elif switch == '2':
		print('Running option 2 RunPriceMomentumAdaptiveConvex')		
		RunPriceMomentumAdaptiveConvex(params)
	elif switch == '3':
		print('Running option 3 RunPriceMomentumBlended')		
		params.useDatabase = False
		RunPriceMomentumBlended(params)
	else:
		print('Running default option ModelPastFewYears')
		ModelPastFewYears(params) 
