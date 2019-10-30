import datetime, os, pandas
from _classes.PriceTradeAnalyzer import PricingData, PriceSnapshot, PlotHelper, GetTodaysDate
from _classes.TickerLists import TickerLists
IndexList=['^SPX','^DJI', '^NDQ']

def PlotAnnualPerformance(ticker:str='^SPX'):
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
		
def PlotPrediction(ticker:str='^SPX', predictionMethod:int=0, daysToGraph:int=60, daysForward:int=5, learnhingEpochs:int=500):
	print('Plotting predictions for ' + ticker)
	prices = PricingData(ticker)
	if prices.LoadHistory():
		prices.NormalizePrices()
		prices.PredictPrices(predictionMethod, daysForward, learnhingEpochs)
		prices.NormalizePrices()
		prices.GraphData(None, daysToGraph, ticker + ' ' + str(daysToGraph) + 'days', True, True, str(daysToGraph) + 'days')
		prices.SaveStatsToFile(includePredictions=True, verbose=True)

def DownloadAndSaveStocksWithStats(tickerList:list):
	for ticker in tickerList:
		prices = PricingData(ticker)
		print('Loading ' + ticker)
		if prices.LoadHistory(requestedEndDate=GetTodaysDate()):
			print('Calcualting stats ' + ticker)
			prices.CalculateStats()
			prices.SaveStatsToFile(includePredictions=False, verbose=True)

def DownloadAndGraphStocks(tickerList:list):
	for ticker in tickerList:
		prices = PricingData(ticker)
		print('Loading ' + ticker)
		if prices.LoadHistory(requestedEndDate=GetTodaysDate()):
			print('Calcualting stats ' + ticker)
			prices.NormalizePrices()
			prices.CalculateStats()
			prices.PredictPrices(2, 15)
			prices.NormalizePrices()
			#prices.SaveStatsToFile(includePredictions=True, verbose=True)
			psnap = prices.GetCurrentPriceSnapshot()
			titleStatistics =' 5/15 dev: ' + str(round(psnap.fiveDayDeviation*100, 2)) + '/' + str(round(psnap.fifteenDayDeviation*100, 2)) + '% ' + str(psnap.low) + '/' + str(psnap.nextDayTarget) + '/' + str(psnap.high) + ' ' + str(psnap.snapShotDate)[:10]
			print('Graphing ' + ticker + ' ' + str(psnap.snapShotDate)[:10])
			for days in [90,180,365,2190,4380]:
				prices.GraphData(None, days, ticker + '_days' + str(days) + ' ' + titleStatistics, (days < 1000), True, str(days).rjust(4, '0') + 'd', trimHistoricalPredictions=False)

def GraphTimePeriod(ticker:str, endDate:datetime, days:int):
	prices = PricingData(ticker)
	print('Loading ' + ticker)
	if prices.LoadHistory():
		prices.GraphData(endDate, days, None , False, True, None)
		print('Chart saved to \data\charts')

def CalculatePriceCorrelation(tickerList:list):
	datafileName = 'data/_priceCorrelations.csv'
	summaryfileName = 'data/_priceCorrelationTop10.txt'
	result = None
	startDate = str(datetime.datetime.now().date()  + datetime.timedelta(days=-365))
	endDate = str(datetime.datetime.now().date())
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

def OpportunityFinder(tickerList:list):
	outputFolder = 'data/dailypicks/'
	summaryFile = '_summary.txt'
	overBoughtList = []
	oversoldList = []
	highDeviationList = []
	for root, dirs, files in os.walk(outputFolder):
		for f in files:
			if f.endswith('.txt') or f.endswith('.png'): os.unlink(os.path.join(root, f))

	for ticker in tickerList:
		prices = PricingData(ticker)
		print('Checking ' + ticker)
		if prices.LoadHistory(requestedEndDate=datetime.datetime.now()):
			prices.CalculateStats()
			psnap = prices.GetCurrentPriceSnapshot()
			titleStatistics =' 5/15 dev: ' + str(round(psnap.fiveDayDeviation*100, 2)) + '/' + str(round(psnap.fifteenDayDeviation*100, 2)) + '% ' + str(psnap.low) + '/' + str(psnap.nextDayTarget) + '/' + str(psnap.high) + str(psnap.snapShotDate)
			if psnap.low > psnap.channelHigh: 
				overBoughtList.append(ticker)
			if psnap.high < psnap.channelLow: 
				oversoldList.append(ticker)
				prices.GraphData(None, 60, ticker + ' 60d ' + titleStatistics, False, True, '60d', outputFolder)
			if psnap.fiveDayDeviation > .0275: 
				highDeviationList.append(ticker)
				prices.GraphData(None, 60, ticker + ' 60d ' + titleStatistics, False, True, '60d', outputFolder)
	print('Over bought:')
	print(overBoughtList)
	print('Over sold:')
	print(oversoldList)
	print('High deviation:')
	print(highDeviationList)
	f = open(outputFolder + summaryFile,'w')
	f.write('Over bought:\n')
	for t in overBoughtList: f.write(t + '\n')
	f.write('\nOver sold:\n')
	for t in oversoldList: f.write(t + '\n')
	f.write('\nHigh deviation:\n')
	for t in highDeviationList: f.write(t + '\n')
	f.close()
	
if __name__ == '__main__': #Choose your adventure.
	#CalculatePriceCorrelation(TickerLists.SPTop70())
	CalculatePriceCorrelation(TickerLists.DogsOfDOW())
	PlotAnnualPerformance('TSLA')
	#PlotAnnualPerformance('VIGRX')
	PlotPrediction('^SPX', 1, 120, 15)
	CalculatePriceCorrelation(TickerLists.SPTop70())
	DownloadAndGraphStocks(TickerLists.SPTop70())
	OpportunityFinder(TickerLists.SPTop70())
	DownloadAndGraphStocks(TickerLists.DogsOfDOW())
	for i in range(30,40,2):	GraphTimePeriod('^SPX', '1/3/19' + str(i), 600)
	PlotPrediction('^SPX', predictionMethod=3, daysToGraph=60, daysForward=5, learnhingEpochs=750) #LSTM
	#PlotPrediction('^SPX', predictionMethod=4, daysToGraph=60, daysForward=5, learnhingEpochs=750) #CNN
