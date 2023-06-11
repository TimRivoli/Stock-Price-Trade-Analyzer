import os, pandas as pd
import sys
from datetime import datetime, timedelta, date
from _classes.PriceTradeAnalyzer import PricingData, PriceSnapshot, PlotHelper, GetTodaysDate
from _classes.TickerLists import TickerLists
from _classes.Utility import *
IndexList=['.INX','^DJI', '^NDQ']

def PlotAnnualPerformance(ticker:str='.INX'):
	print('Annual performance rate for ' + ticker)
	prices = PricingData(ticker)
	if prices.LoadHistory():
		x = prices.GetPriceHistory(['Average'])
		yearly=x.groupby([(x.index.year)]).first()
		yearlyChange = yearly.pct_change(1)
		monthly=x.groupby([(x.index.year),(x.index.month)]).first()
		plot = PlotHelper()
		plot.PlotDataFrame(yearly, title='Yearly', adjustScale=False)
		plot.PlotDataFrame(monthly, title='Monthly', adjustScale=False)
		plot.PlotDataFrame(yearlyChange, title='Yearly Percentage Change', adjustScale=False)
		print('Average annual change from ', prices.historyStartDate, ' to ', prices.historyEndDate, ': ', yearlyChange.mean().values * 100, '%')
		
def PlotPrediction(ticker:str='.INX', predictionMethod:int=0, daysToGraph:int=60, daysForward:int=5, learnhingEpochs:int=500):
	print('Plotting predictions for ' + ticker)
	prices = PricingData(ticker)
	if prices.LoadHistory():
		prices.NormalizePrices()
		prices.PredictPrices(predictionMethod, daysForward, learnhingEpochs) #1,2 both static trend estimations, 3 LSTM, 4 CNN
		prices.NormalizePrices()
		prices.GraphData(None, daysToGraph, ticker + ' ' + str(daysToGraph) + 'days', True, True, str(daysToGraph) + 'days')
		prices.SaveStatsToFile(includePredictions=True, verbose=True)

def DownloadAndSaveStocks(tickerList:list):
	for ticker in tickerList:
		prices = PricingData(ticker)
		print('Loading ' + ticker)
		if prices.LoadHistory(requestedEndDate=GetTodaysDate()):
			print("Loaded", ticker)

def DownloadAndSaveStocksWithStats(tickerList:list):
	for ticker in tickerList:
		prices = PricingData(ticker)
		print('Loading ' + ticker)
		if prices.LoadHistory(requestedEndDate=GetTodaysDate()):
			print('Calcualting stats ' + ticker)
			prices.CalculateStats()
			prices.SaveStatsToFile(includePredictions=False, verbose=True)

def DownloadAndGraphStocks(tickerList:list, includePredictions:bool = False):
	for ticker in tickerList:
		prices = PricingData(ticker)
		print('Loading ' + ticker)
		if prices.LoadHistory(requestedEndDate=GetTodaysDate()):
			print('Calcualting stats ' + ticker)
			prices.NormalizePrices()
			prices.CalculateStats()
			prices.PredictPrices(method=2, daysIntoFuture=5, NNTrainingEpochs=750) #1,2 both static trend estimations, 3 LSTM, 4 CNN
			prices.NormalizePrices()
			#prices.SaveStatsToFile(includePredictions=True, verbose=True)
			psnap = prices.GetCurrentPriceSnapshot()
			titleStatistics =' 5/15 dev: ' + str(round(psnap.Deviation_5Day*100, 2)) + '/' + str(round(psnap.Deviation_15Day*100, 2)) + '% ' + str(psnap.low) + '/' + str(psnap.Target_1Day) + '/' + str(psnap.high) + ' ' + str(psnap.date)[:10]
			print('Graphing ' + ticker + ' ' + str(psnap.date)[:10])
			for days in [30,90,180,365]: #,2190,4380
				includePredictions2 = includePredictions and (days < 1000)
				prices.GraphData(endDate=None, daysToGraph=days, graphTitle=ticker + '_days' + str(days) + ' ' + titleStatistics, includePredictions=includePredictions2, saveToFile=True, fileNameSuffix=str(days).rjust(4, '0') + 'd', trimHistoricalPredictions=False)

def GraphTimePeriod(ticker:str, endDate:str, days:int):
	prices = PricingData(ticker)
	print('Loading ' + ticker)
	if prices.LoadHistory():
		prices.GraphData(endDate=endDate, daysToGraph=days, graphTitle=None , includePredictions=False, saveToFile=True, fileNameSuffix=None)
		print('Chart saved to \data\charts')

def CalculatePriceCorrelation(tickerList:list):
	datafileName = 'data/_priceCorrelations.csv'
	summaryfileName = 'data/_priceCorrelationTop10.txt'
	result = None
	startDate = str(AddDays(GetTodaysDate(), -365))
	endDate = str(GetTodaysDate())
	for ticker in tickerList:
		prices = PricingData(ticker)
		print('Loading ' + ticker)
		if prices.LoadHistory(requestedEndDate=GetTodaysDate()):
			prices.TrimToDateRange(startDate, endDate)
			prices.NormalizePrices()
			x = prices.GetPriceHistory(['Average'])
			x.rename(index=str, columns={"Average": ticker}, inplace=True)
			if result is None:
				result = x
			else:
				result = result.join(x, how='outer')
	result = result.corr()
	result.to_csv(datafileName)

	f = open(summaryfileName,'w')
	for ticker in tickerList:
		topTen = result.nsmallest(10,ticker)
		print(topTen[ticker])
		f.write(ticker + '\n')
		f.write(topTen[ticker].to_string(header=True,index=True) + '\n')
		f.write('\n')
	f.close()
	print('Intended to create stability, in practice, this is a great way to pair well performing stocks with poor performing or volatile stocks.')

def OpportunityFinder(tickerList:list):
	outputFolder = 'data/dailypicks/'
	summaryFile = '_DailyPicks.csv'
	candidates = pd.DataFrame(columns=list(['Ticker','hp2Year','hp1Year','hp6mo','hp3mo','hp2mo','hp1mo','price_current','Channel_High','Channel_Low','EMA_Short','EMA_Long','2yearPriceChange','1yearPriceChange','6moPriceChange','3moPriceChange','2moPriceChange','1moPriceChange','PC_1Day','Gain_Monthly','LossStd_1Year','Comments']))
	candidates.set_index(['Ticker'], inplace=True)
	for root, dirs, files in os.walk(outputFolder):
		for f in files:
			if f.endswith('.png'): os.unlink(os.path.join(root, f))

	for ticker in tickerList:
		prices = PricingData(ticker)
		currentDate = GetTodaysDate()
		print('Checking ' + ticker)
		if prices.LoadHistory(requestedEndDate=currentDate):
			prices.CalculateStats()
			psnap = prices.GetPriceSnapshot(AddDays(currentDate,-730))
			hp2Year = psnap.Average_5Day
			psnap = prices.GetPriceSnapshot(AddDays(currentDate, -365))
			hp1Year = psnap.Average_5Day
			psnap = prices.GetPriceSnapshot(AddDays(currentDate, -180))
			hp6mo = psnap.Average_5Day
			psnap = prices.GetPriceSnapshot(AddDays(currentDate, -90))
			hp3mo = psnap.Average_5Day
			psnap = prices.GetPriceSnapshot(AddDays(currentDate, -60))
			hp2mo = psnap.Average_5Day
			psnap = prices.GetPriceSnapshot(AddDays(currentDate, -30))
			hp1mo = psnap.Average_5Day
			psnap = prices.GetCurrentPriceSnapshot()
			price_current = psnap.average	
			Comments = ''
			if psnap.low > psnap.Channel_High: 
				Comments += 'OverBought; '
			if psnap.high < psnap.Channel_Low: 
				Comments += 'OverSold; '
			if psnap.Deviation_5Day > .0275: 
				Comments += 'HighDeviation; '
			if Comments !='': 
				titleStatistics =' 5/15 dev: ' + str(round(psnap.Deviation_5Day*100, 2)) + '/' + str(round(psnap.Deviation_15Day*100, 2)) + '% ' + str(psnap.low) + '/' + str(psnap.Target_1Day) + '/' + str(psnap.high) + str(psnap.date)
				prices.GraphData(None, 60, ticker + ' 60d ' + titleStatistics, False, True, '60d', outputFolder)
				if (price_current > 0 and hp2Year > 0 and hp1Year > 0 and hp6mo > 0 and hp2mo > 0 and hp1mo > 0): #values were loaded
					candidates.loc[ticker] = [hp2Year,hp1Year,hp6mo,hp3mo,hp2mo,hp1mo,price_current,psnap.Channel_High,psnap.Channel_Low,psnap.EMA_Short,psnap.EMA_Long,(price_current/hp2Year)-1,(price_current/hp1Year)-1,(price_current/hp6mo)-1,(price_current/hp3mo)-1,(price_current/hp2mo)-1,(price_current/hp1mo)-1,psnap.PC_1Day, psnap.Gain_Monthly, psnap.LossStd_1Year,Comments]
				else:
					print(ticker, price_current,hp2Year,hp1Year, hp6mo, hp2mo ,hp1mo )
	print(candidates)
	candidates.to_csv(outputFolder + summaryFile)
	
def PriceCheck(startDate: str, Ticker:str):
	startDate = ToDate(startDate)
	endDate = AddDays(startDate, 30)
	prices = PricingData(ticker)
	prices.LoadHistory()
	sn = prices.GetPriceSnapshot(startDate, True)
	startPrice = sn.average
	sn = prices.GetPriceSnapshot(endDate, True)
	endPrice = sn.average
	print('From', startDate, ' to ', endDate)
	print(ticker, startPrice, endPrice, (endPrice/startPrice-1)*100)
	
if __name__ == '__main__': #Choose your adventure.
	switch = 0
	if len(sys.argv[1:]) > 0: switch = sys.argv[1:][0]
	if switch == '1': #Price check for day
		startDate = sys.argv[1:][1]
		ticker = sys.argv[1:][2]
		PriceCheck(startDate, ticker)
	else:
		CalculatePriceCorrelation(TickerLists.SPTop70())
		PlotAnnualPerformance('TSLA')
		PlotAnnualPerformance('VIGRX')
		PlotPrediction('.INX', 1, 120, 15)
		for year in range(1930,1980,2):	GraphTimePeriod('.INX', '1/3/' + str(year), 600)
		for year in range(1980,2020,2): GraphTimePeriod('.INX', '1/3/' + str(year), 600)
		GraphTimePeriod('NVDA', '1/1/2003',400)
		OpportunityFinder(TickerLists.SPTop70())
		CalculatePriceCorrelation(TickerLists.SPTop70())
		PlotPrediction('.INX', predictionMethod=3, daysToGraph=60, daysForward=5, learnhingEpochs=750) #LSTM
		PlotPrediction('.INX', predictionMethod=4, daysToGraph=60, daysForward=5, learnhingEpochs=750) #CNN
		DownloadAndSaveStocksWithStats(['TSLA'])
		