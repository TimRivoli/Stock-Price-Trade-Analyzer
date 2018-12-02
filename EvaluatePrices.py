import datetime, os, pandas
from _classes.PriceTradeAnalyzer import PricingData, PriceSnapshot, PlotHelper

AShortList=['^SPX','AAPL','GOOGL','F','CVX','XOM','MRK','FCX','NEM','BA']
SPTop70=['^SPX','AAPL','MSFT','AMZN','FB','BRK-B','JPM','JNJ','XOM','GOOG','GOOGL','BAC','WFC','CVX','UNH','HD','INTC','PFE','T','V','PG','VZ','CSCO','C','CMCSA','ABBV','BA','KO','DWDP','PEP','PM','MRK','DIS','WMT','ORCL','MA','MMM','NVDA','IBM','AMGN','MCD','GE','MO','NFLX','HON','MDT','GILD','TXN','ABT','UNP','SLB','BMY','UTX','AVGO','ACN','QCOM','ADBE','CAT','GS','PYPL','PCLN','USB','UPS','LOW','NKE','TMO','LMT','COST','CVS','LLY','CELG']
DogsOfDOW=['VZ','IBM','XOM','PFE','CVX','PG','MRK','KO','GE','CSCO']
SPTop70MinusDogs=['^SPX','AAPL','MSFT','AMZN','FB','BRK-B','JPM','JNJ','XOM','GOOG','GOOGL','BAC','WFC','CVX','UNH','HD','INTC','PFE','T','V','PG','VZ','CSCO','C','CMCSA','ABBV','BA','KO','DWDP','PEP','MRK','DIS','WMT','ORCL','MA','MMM','NVDA','IBM','AMGN','MCD','MO','NFLX','HON','MDT','GILD','TXN','ABT','UNP','SLB','BMY','UTX','AVGO','ACN','QCOM','ADBE','CAT','GS','PYPL','PCLN','USB','UPS','LOW','NKE','TMO','LMT','COST','CVS','LLY','CELG']
IndexList=['^SPX','^DJI', '^NDQ']

def PlotAnnualPerformance(ticker:str='^SPX'):
	print('Annual performance rate for ' + ticker)
	prices = PricingData(ticker)
	if prices.LoadHistory(True):
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
	if prices.LoadHistory(True):
		prices.NormalizePrices()
		prices.PredictPrices(predictionMethod, daysForward, learnhingEpochs)
		prices.NormalizePrices()
		prices.GraphData(None, daysToGraph, ticker + ' ' + str(daysToGraph) + 'days', True, True, str(daysToGraph) + 'days')
		prices.SaveStatsToFile(True)

def DownloadAndGraphStocks(tickerList:list):
	for ticker in tickerList:
		prices = PricingData(ticker)
		print('Loading ' + ticker)
		if prices.LoadHistory(True):
			print('Calcualting stats ' + ticker)
			prices.NormalizePrices()
			prices.CalculateStats()
			prices.PredictPrices(2, 15)
			prices.NormalizePrices()
			#prices.SaveStatsToFile(True)
			psnap = prices.GetCurrentPriceSnapshot()
			titleStatistics =' 5/15 dev: ' + str(round(psnap.fiveDayDeviation*100, 2)) + '/' + str(round(psnap.fifteenDayDeviation*100, 2)) + '% ' + str(psnap.low) + '/' + str(psnap.nextDayTarget) + '/' + str(psnap.high) + ' ' + str(psnap.snapShotDate)[:10]
			print('Graphing ' + ticker + ' ' + str(psnap.snapShotDate)[:10])
			for days in [90,180,365,2190,4380]:
				prices.GraphData(None, days, ticker + '_days' + str(days) + ' ' + titleStatistics, (days < 1000), True, str(days).rjust(4, '0') + 'd', trimHistoricalPredictions=False)

def GraphTimePeriod(ticker:str, endDate:datetime, days:int):
	prices = PricingData(ticker)
	print('Loading ' + ticker)
	if prices.LoadHistory(True):
		prices.GraphData(endDate, days, None , False, True, None)

def CalculatePriceCorrelation(tickerList:list):
	datafileName = 'data/_priceCorrelations.csv'
	summaryfileName = 'data/_priceCorrelationTop10.txt'
	result = pandas.DataFrame()
	startDate = str(datetime.datetime.now().date()  + datetime.timedelta(days=-365))
	endDate = str(datetime.datetime.now().date())
	for ticker in tickerList:
		prices = PricingData(ticker)
		print('Loading ' + ticker)
		if prices.LoadHistory(True):
			prices.TrimToDateRange(startDate, endDate)
			prices.NormalizePrices()
			result[ticker] = prices.GetPriceHistory(['Average'])
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
	#return result

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
		if prices.LoadHistory(True):
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
	#CalculatePriceCorrelation(SPTop70)
	#CalculatePriceCorrelation(DogsOfDOW)
	#PlotAnnualPerformance('TSLA')
	PlotAnnualPerformance('^SPX')
	#PlotPrediction('^SPX', 1, 120, 15)
	#OpportunityFinder(SPTop70)
	#DownloadAndGraphStocks(IndexList)
	#DownloadAndGraphStocks(SPTop70)
	#DownloadAndGraphStocks(DogsOfDOW)
	#for i in range(30,40,2):	GraphTimePeriod('^SPX', '1/3/19' + str(i), 600)
	#PlotPrediction('^SPX', predictionMethod=3, daysToGraph=60, daysForward=5, learnhingEpochs=750) #LSTM
	#PlotPrediction('^SPX', predictionMethod=4, daysToGraph=60, daysForward=5, learnhingEpochs=750) #CNN
