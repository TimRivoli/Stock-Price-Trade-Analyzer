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
		prices.SavePricesWithStats(includePredictions=True, verbose=True)

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
			prices.SavePricesWithStats(includePredictions=False, verbose=True)

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
			#prices.SavePricesWithStats(includePredictions=True, verbose=True)
			psnap = prices.GetCurrentPriceSnapshot()
			titleStatistics =' 5/15 dev: ' + str(round(psnap.Deviation_5Day*100, 2)) + '/' + str(round(psnap.Deviation_15Day*100, 2)) + '% ' + str(psnap.Low) + '/' + str(psnap.Target) + '/' + str(psnap.High) + ' ' + str(psnap.Date)[:10]
			print('Graphing ' + ticker + ' ' + str(psnap.Date)[:10])
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
		if prices.LoadHistory(requestedStartDate=startDate, requestedEndDate=endDate):
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
		if ticker not in result.columns:
					print(f"Skipping {ticker}: No data found in results.")
					continue
		topTen = result.nsmallest(10,ticker)
		print(topTen[ticker])
		f.write(ticker + '\n')
		f.write(topTen[ticker].to_string(header=True,index=True) + '\n')
		f.write('\n')
	f.close()
	print('Intended to create stability, in practice, this is a great way to pair well performing stocks with poor performing or volatile stocks.')
	
def PriceCheck(startDate: str, Ticker:str):
	startDate = ToDate(startDate)
	endDate = AddDays(startDate, 30)
	prices = PricingData(ticker)
	prices.LoadHistory()
	sn = prices.GetPriceSnapshot(startDate, True)
	startPrice = sn.Average
	sn = prices.GetPriceSnapshot(endDate, True)
	endPrice = sn.Average
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
		DownloadAndSaveStocksWithStats(['.INX'])
		PlotAnnualPerformance('TSLA')
		PlotAnnualPerformance('VIGRX')
		PlotPrediction('.INX', 1, 120, 15)
		#for year in range(1930,1980,2):	GraphTimePeriod('.INX', '1/3/' + str(year), 600)
		for year in range(1980,2020,2): GraphTimePeriod('.INX', '1/3/' + str(year), 600)
		GraphTimePeriod('NVDA', '1/1/2003',400)
		CalculatePriceCorrelation(TickerLists.SPTop70())
		PlotPrediction('.INX', predictionMethod=3, daysToGraph=60, daysForward=5, learnhingEpochs=750) #LSTM
		PlotPrediction('.INX', predictionMethod=4, daysToGraph=60, daysForward=5, learnhingEpochs=750) #CNN
		DownloadAndSaveStocksWithStats(['TSLA'])
		CalculatePriceCorrelation(TickerLists.SPTop70())
		