import datetime, os, pandas as pd
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
				prices.GraphData(endDate=None, daysToGraph=days, graphTitle=ticker + '_days' + str(days) + ' ' + titleStatistics, includePredictions=(days < 1000), saveToFile=True, fileNameSuffix=str(days).rjust(4, '0') + 'd', trimHistoricalPredictions=False)

def GraphTimePeriod(ticker:str, endDate:datetime, days:int):
	prices = PricingData(ticker)
	print('Loading ' + ticker)
	if prices.LoadHistory():
		prices.GraphData(endDate=endDate, daysToGraph=days, graphTitle=None , includePredictions=False, saveToFile=True, fileNameSuffix=None)
		print('Chart saved to \data\charts')

def CalculatePriceCorrelation(tickerList:list):
	datafileName = 'data/_priceCorrelations.csv'
	summaryfileName = 'data/_priceCorrelationTop10.txt'
	result = None
	startDate = str(GetTodaysDate() + datetime.timedelta(days=-365))
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
	candidates = pd.DataFrame(columns=list(['Ticker','hp2Year','hp1Year','hp6mo','hp3mo','hp2mo','hp1mo','currentPrice','channelHigh','channelLow','shortEMA','longEMA','2yearPriceChange','1yearPriceChange','6moPriceChange','3moPriceChange','2moPriceChange','1moPriceChange','dailyGain','monthlyGain','monthlyLossStd','Comments']))
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
			psnap = prices.GetPriceSnapshot(currentDate + datetime.timedelta(days=-730))
			hp2Year = psnap.fiveDayAverage
			psnap = prices.GetPriceSnapshot(currentDate + datetime.timedelta(days=-365))
			hp1Year = psnap.fiveDayAverage
			psnap = prices.GetPriceSnapshot(currentDate + datetime.timedelta(days=-180))
			hp6mo = psnap.fiveDayAverage
			psnap = prices.GetPriceSnapshot(currentDate + datetime.timedelta(days=-90))
			hp3mo = psnap.fiveDayAverage
			psnap = prices.GetPriceSnapshot(currentDate + datetime.timedelta(days=-60))
			hp2mo = psnap.fiveDayAverage
			psnap = prices.GetPriceSnapshot(currentDate + datetime.timedelta(days=-30))
			hp1mo = psnap.fiveDayAverage
			psnap = prices.GetCurrentPriceSnapshot()
			currentPrice = psnap.twoDayAverage	
			Comments = ''
			if psnap.low > psnap.channelHigh: 
				Comments += 'OverBought; '
			if psnap.high < psnap.channelLow: 
				Comments += 'OverSold; '
			if psnap.fiveDayDeviation > .0275: 
				Comments += 'HighDeviation; '
			if Comments !='': 
				titleStatistics =' 5/15 dev: ' + str(round(psnap.fiveDayDeviation*100, 2)) + '/' + str(round(psnap.fifteenDayDeviation*100, 2)) + '% ' + str(psnap.low) + '/' + str(psnap.nextDayTarget) + '/' + str(psnap.high) + str(psnap.snapShotDate)
				prices.GraphData(None, 60, ticker + ' 60d ' + titleStatistics, False, True, '60d', outputFolder)
				if (currentPrice > 0 and hp2Year > 0 and hp1Year > 0 and hp6mo > 0 and hp2mo > 0 and hp1mo > 0): #values were loaded
					candidates.loc[ticker] = [hp2Year,hp1Year,hp6mo,hp3mo,hp2mo,hp1mo,currentPrice,psnap.channelHigh,psnap.channelLow,psnap.shortEMA,psnap.longEMA,(currentPrice/hp2Year)-1,(currentPrice/hp1Year)-1,(currentPrice/hp6mo)-1,(currentPrice/hp3mo)-1,(currentPrice/hp2mo)-1,(currentPrice/hp1mo)-1,psnap.dailyGain, psnap.monthlyGain, psnap.monthlyLossStd,Comments]
				else:
					print(ticker, currentPrice,hp2Year,hp1Year, hp6mo, hp2mo ,hp1mo )
	print(candidates)
	candidates.to_csv(outputFolder + summaryFile)
	
if __name__ == '__main__': #Choose your adventure.
	DownloadAndGraphStocks(IndexList)
	CalculatePriceCorrelation(TickerLists.SPTop70())
	#CalculatePriceCorrelation(TickerLists.DogsOfDOW())
	PlotAnnualPerformance('TSLA')
	PlotAnnualPerformance('VIGRX')
	PlotPrediction('^SPX', 1, 120, 15)
	#for year in range(1930,1980,2):	GraphTimePeriod('^SPX', '1/3/' + str(year), 600)
	for year in range(1980,2020,2): GraphTimePeriod('^SPX', '1/3/' + str(year), 600)
	GraphTimePeriod('NVDA', '1/1/2003',400)
	OpportunityFinder(TickerLists.TopPerformers())
	CalculatePriceCorrelation(TickerLists.TopPerformers())
	PlotPrediction('^SPX', predictionMethod=3, daysToGraph=60, daysForward=5, learnhingEpochs=750) #LSTM
	PlotPrediction('^SPX', predictionMethod=4, daysToGraph=60, daysForward=5, learnhingEpochs=750) #CNN
