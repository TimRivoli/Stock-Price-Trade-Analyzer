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
	params.AddModelNameModifiers()
	tm = TradingModel(modelName=params.modelName, startingTicker=CONSTANTS.CASH_TICKER, startDate=params.startDate, durationInYears=params.durationInYears, totalFunds=params.portfolioSize, verbose=params.verbose)
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
				candidates = picker.GetHighestPriceMomentum(currentDate, stocksToReturn=params.stockCount, filterOption=params.filterOption, minPercentGain=params.minPercentGain, allocateByTargetHoldings=(not params.allocateByPointValue), allocateByPointValue=params.allocateByPointValue)
				tm.AlignPositions(targetPositions=candidates, rateLimitTransactions=params.rateLimitTransactions, shopBuyPercent=params.shopBuyPercent, shopSellPercent=params.shopSellPercent) 
			elif (dayCounter % 3 == 1) and (params.rateLimitTransactions or params.trimProfitsPercent > 0): #Every three days run these two
				tm.AlignPositions(targetPositions=candidates, rateLimitTransactions=params.rateLimitTransactions, shopBuyPercent=params.shopBuyPercent, shopSellPercent=params.shopSellPercent) 
				if params.trimProfitsPercent: tm.TrimProfits(params.trimProfitsPercent)
			tm.ProcessDay()
			dayCounter+=1
			if dayCounter >= params.reEvaluationInterval: dayCounter=0
	closing_value = tm.CloseModel(params)
	return closing_value

def RunPriceMomentumAdaptiveConvex(params: TradeModelParams, convex_filter:int = 6, linear_filter:int = 2, defense_filter:int = 1):
	params.filterOption = 98
	params.allocateByPointValue = False
	params.modelName = f"AdaptiveConvex_v3.5_({convex_filter}.{linear_filter}.{defense_filter})"
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
			picker.AlignToList(tickerList)
		if dayCounter == 0:
			candidates = picker.GetAdaptiveConvexPicks(currentDate, convex_filter, linear_filter, defense_filter)
			tm.AlignPositions(targetPositions = candidates, rateLimitTransactions = params.rateLimitTransactions, shopBuyPercent = params.shopBuyPercent, shopSellPercent = params.shopSellPercent) 
		elif (dayCounter % 3 == 1) and (params.rateLimitTransactions or params.trimProfitsPercent > 0):
			tm.AlignPositions(targetPositions = candidates, rateLimitTransactions = params.rateLimitTransactions, shopBuyPercent = params.shopBuyPercent, shopSellPercent = params.shopSellPercent) 
			if params.trimProfitsPercent: tm.TrimProfits(params.trimProfitsPercent)
		tm.ProcessDay()
		dayCounter += 1
		if dayCounter >= params.reEvaluationInterval: dayCounter = 0
	closing_value = tm.CloseModel(params)
	return closing_value
	
def RunPriceMomentumBlended(params: TradeModelParams, filter1:int = 3, filter2: int = 3, filter3: int = 1, filter4: int = 5):
	#Uses blended option for selecting stocks using three different filters, produces the best overall results.
	#2 long term performer
	#4 short term performer
	#1 Short term gain meets min requirements, sort long value
	#5 point value, aka PV
	params.filterOption = 99
	params.allocateByPointValue = False
	params.modelName = f"PM_Blended_({filter1}.{filter2}.{filter3}.{filter4})"
	params.AddModelNameModifiers()
	print('RunPriceMomentumBlended ', params.startDate, params.endDate)
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
					candidates = picker.GetPicksBlended(currentDate=currentDate, filter1=filter1, filter2=filter2, filter3=filter3, filter4=filter4, useRollingWindow=(params.pickHistoryWindow>0) )
				tm.AlignPositions(targetPositions = candidates, rateLimitTransactions = params.rateLimitTransactions, shopBuyPercent = params.shopBuyPercent, shopSellPercent = params.shopSellPercent) 
			elif (dayCounter % 3 == 1) and (params.rateLimitTransactions or params.trimProfitsPercent > 0):
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
	for i in [98]:
		for ii in [5, 10, 20]:
			params.filterOption = i
			params.reEvaluationInterval = ii
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
			RunPriceMomentumBlended(params, 3, 3, 1, 5)
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
	params.filterOption = 5
	params.rateLimitTransactions = True
	params.allocateByPointValue = True
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
