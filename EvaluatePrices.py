import datetime
import os
import pandas
from _classes.PriceTradeAnalyzer import PricingData, PriceSnapshot, PlotHelper

AShortList=['^SPX','AAPL','GOOGL','F','CVX','XOM','MRK','FCX','NEM','BA']
SPTop70=['^SPX','AAPL','MSFT','AMZN','FB','BRK-B','JPM','JNJ','XOM','GOOG','GOOGL','BAC','WFC','CVX','UNH','HD','INTC','PFE','T','V','PG','VZ','CSCO','C','CMCSA','ABBV','BA','KO','DWDP','PEP','PM','MRK','DIS','WMT','ORCL','MA','MMM','NVDA','IBM','AMGN','MCD','GE','MO','NFLX','HON','MDT','GILD','TXN','ABT','UNP','SLB','BMY','UTX','AVGO','ACN','QCOM','ADBE','CAT','GS','PYPL','PCLN','USB','UPS','LOW','NKE','TMO','LMT','COST','CVS','LLY','CELG']
DogsOfDOW=['VZ','IBM','XOM','PFE','CVX','PG','MRK','KO','GE','CSCO']
SPTop70MinusDogs=['^SPX','AAPL','MSFT','AMZN','FB','BRK-B','JPM','JNJ','XOM','GOOG','GOOGL','BAC','WFC','CVX','UNH','HD','INTC','PFE','T','V','PG','VZ','CSCO','C','CMCSA','ABBV','BA','KO','DWDP','PEP','MRK','DIS','WMT','ORCL','MA','MMM','NVDA','IBM','AMGN','MCD','MO','NFLX','HON','MDT','GILD','TXN','ABT','UNP','SLB','BMY','UTX','AVGO','ACN','QCOM','ADBE','CAT','GS','PYPL','PCLN','USB','UPS','LOW','NKE','TMO','LMT','COST','CVS','LLY','CELG']

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
		
def DownloadAndGraphStocks(tickerList:list, simpleMode:bool=False):
	today = datetime.datetime.now().date()
	for ticker in tickerList:
		prices = PricingData(ticker)
		print('Loading ' + ticker)
		if prices.LoadHistory(True):
			if simpleMode:
				prices.TrimToDateRange('2/20/2017', '2/20/2018')
				plot = PlotHelper()
				plot.PlotDataFrame(prices.GetPriceHistory(['Open','High','Low','Close']), 'Sample', 'Date', 'Price', 'sciencefair/sample' ) 
			else:
				print('Calcualting stats ' + ticker)
				prices.CalculateStats()
				prices.SaveStatsToFile()
				prices.PredictPrices()
				psnap = prices.GetCurrentPriceSnapshot()
				titleStatistics =' 5/15 dev: ' + str(round(psnap.fiveDayDeviation*100, 2)) + '/' + str(round(psnap.fifteenDayDeviation*100, 2)) + '% ' + str(psnap.low) + '/' + str(psnap.nextDayTarget) + '/' + str(psnap.high)
				print('Graphing ' + ticker)	
				prices.GraphData(today + datetime.timedelta(days=-60), today, ticker + ' 60d ' + titleStatistics, True, True, '60d')
				prices.GraphData(today + datetime.timedelta(days=-100), today, ticker + ' 100d ' + titleStatistics, True, True, '100d')
				prices.GraphData(today + datetime.timedelta(days=-365), today, ticker + ' 1Year', False, True, '1Year')
				prices.GraphData(today + datetime.timedelta(days=-730), today, ticker + ' 2Year', False, True,  '2Year')
				prices.GraphData(today + datetime.timedelta(days=-3650),today, ticker + ' 10Year', False, True, '10Year')

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
	today = datetime.datetime.now().date()
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
			titleStatistics =' 5/15 dev: ' + str(round(psnap.fiveDayDeviation*100, 2)) + '/' + str(round(psnap.fifteenDayDeviation*100, 2)) + '% ' + str(psnap.low) + '/' + str(psnap.nextDayTarget) + '/' + str(psnap.high)
			if psnap.low > psnap.channelHigh: 
				overBoughtList.append(ticker)
			if psnap.high < psnap.channelLow: 
				oversoldList.append(ticker)
				prices.GraphData(today + datetime.timedelta(days=-60), today, ticker + ' 60d ' + titleStatistics, False, True, '60d', outputFolder)
			if psnap.fiveDayDeviation > .0275: 
				highDeviationList.append(ticker)
				prices.GraphData(today + datetime.timedelta(days=-60), today, ticker + ' 60d ' + titleStatistics, False, True, '60d', outputFolder)
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
			
def Test(ticker:str, epochs:int = 350):
	today = datetime.datetime.now().date()
	prices = PricingData(ticker)
	print('Loading ' + ticker)
	futureDates = 16
	if prices.LoadHistory(True):
		print('Calcualting ' + ticker)
		prices.NormalizePrices()
		prices.CalculateStats()
		for i in range(0,5): #5
			prices.PredictPrices(i, futureDates, epochs)
			#print(prices.pricePredictions)
			prices.NormalizePrices()
			prices.SaveStatsToFile(True)
			print('Deviation: ' + str(round(prices.predictionDeviation*100,2)) + '%')
			modelDescription = ticker + 'futureDates' + str(futureDates) + '_mode' + str(i) + '_deviation' + str(round(prices.predictionDeviation*100, 2)) + '%'
			prices.GraphData(today + datetime.timedelta(days=-60), today + datetime.timedelta(days=futureDates), modelDescription, True, False, modelDescription)

DownloadAndGraphStocks(SPTop70)
#OpportunityFinder(SPTop70)
#CalculatePriceCorrelation(SPTop70)
#DownloadAndGraphStocks(['CSCO'])
#CalculatePriceCorrelation(DogsOfDOW)
#PlotAnnualPerformance('TSLA')
#Test('AAPL', 500)
