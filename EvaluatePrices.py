from datetime import datetime
from dateutil.relativedelta import relativedelta
from _classes.Graphing import PlotHelper
from _classes.Prices import PricingData, PriceSnapshot
from _classes.Selection import StockPicker
from _classes.TickerLists import TickerLists
from _classes.Utility import *

def PlotAnnualPerformance(ticker:str='.INX'):
	print('Annual performance rate for ' + ticker)
	prices = PricingData(ticker)
	if prices.LoadHistory():
		x = prices.GetPriceHistory(['Average'])
		yearly=x.groupby([(x.index.year)]).first()
		yearlyChange = yearly.pct_change(1)
		monthly=x.groupby([(x.index.year),(x.index.month)]).first()
		plot = PlotHelper()
		plot.PlotDataFrame(yearly, title=f"Annual Performance for {ticker}", adjustScale=False)
		plot.PlotDataFrame(monthly, title=f"Monthly Performance for {ticker}", adjustScale=False)
		plot.PlotDataFrame(yearlyChange, title="Yearly Percentage Gain for {ticker}", adjustScale=False)
		print('Average annual change from ', prices.historyStartDate, ' to ', prices.historyEndDate, ': ', yearlyChange.mean().values * 100, '%')	

def DownloadAndSaveStocks(tickerList:list):
	for ticker in tickerList:
		prices = PricingData(ticker=ticker)
		prices.LoadHistory(requestedEndDate=GetLatestBDay(), verbose=True)

def DownloadAndSaveStocksWithStats(tickerList:list, startDate:str = None, endDate:str=GetLatestBDay()):
	for ticker in tickerList:
		prices = PricingData(ticker)
		if prices.LoadHistory(requestedStartDate=startDate, requestedEndDate=endDate, verbose=True):
			prices.CalculateStats()
			prices.SavePricesWithStats(includePredictions=False, toDatebase=False, verbose=True)

def GraphTimePeriod(ticker:str, endDate:str, days:int):
	prices = PricingData(ticker)
	if prices.LoadHistory():
		prices.GraphData(endDate=endDate, daysToGraph=days, graphTitle=None, includePredictions=False, saveToFile=True, fileNameSuffix=None, verbose=False)
	
def GraphHistoricalDrops():
	ticker = '.INX'
	crash_milestones = [
		(datetime(1987, 8, 25), datetime(1987, 12, 4)),   # Black Monday
		(datetime(2000, 3, 24), datetime(2002, 10, 9)),   # Dot-Com Bust
		(datetime(2007, 10, 9), datetime(2009, 3, 9)),    # Financial Crisis
		(datetime(2020, 2, 19), datetime(2020, 3, 23))    # COVID-19
	]
	crash_troughs = [m[1] for m in crash_milestones]
	print(f"Trough Dates (The bottom): {crash_troughs}")
	prices = PricingData(ticker)
	if prices.LoadHistory():
		for d in crash_troughs:
			endDate = d
			for duration in [60, 120, 365]:
				prices.GraphData(endDate=endDate, daysToGraph=duration, graphTitle=None, includePredictions=False, saveToFile=True, fileNameSuffix=None, verbose=False)
			endDate += relativedelta(months=6)
			for duration in [60, 120, 365]:
				prices.GraphData(endDate=endDate, daysToGraph=duration, graphTitle=None, includePredictions=False, saveToFile=True, fileNameSuffix=None, verbose=False)

def PriceCheck(startDate: str, Ticker:str):
	startDate = ToDate(startDate)
	endDate = AddDays(startDate, 30)
	prices = PricingData(ticker)
	prices.LoadHistory(startDate, endDate)
	sn = prices.GetPriceSnapshot(startDate, True)
	startPrice = sn.Average
	sn = prices.GetPriceSnapshot(endDate, True)
	endPrice = sn.Average
	print('From', startDate, ' to ', endDate)
	print(ticker, startPrice, endPrice, (endPrice/startPrice-1)*100)

def ShowPicks(tickerList: list):
	forDate = GetLatestBDay()
	picker = StockPicker()
	picker.AlignToList(tickerList)
	for option in [2,3,4,5]:
		picks = picker.GetHighestPriceMomentum(forDate, stocksToReturn=10, filterOption=option)
		print(f"Picks for filter option {option}")
		print(picks)
	print(f"Picks for Blended option")
	picks = picker.GetPicksBlended(forDate)
	print(picks)
	print(f"Picks for AdaptiveConvex option")
	picks = picker.GetAdaptiveConvex(forDate)
	print(picks)
		
if __name__ == '__main__': 
	# tickerList = TickerLists.StarterList()
	# print(f"Loading {len(tickerList)} stocks..")
	# DownloadAndSaveStocks(tickerList)
	# print(f"Saving stats to _dataFolderhistoricalPrices")
	# DownloadAndSaveStocksWithStats(tickerList)
	# for year in range(1980,2020,2): GraphTimePeriod(ticker = '.INX', endDate = '1/3/' + str(year), days=120)
	# PlotAnnualPerformance('TSLA')
	# PlotAnnualPerformance('VIGRX')
	# ShowPicks(tickerList)
	GraphHistoricalDrops()