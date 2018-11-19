import pandas, os
from pandas.tseries.offsets import BDay
from _classes.PriceTradeAnalyzer import PricingData, PlotHelper
from _classes.SeriesPrediction import StockPredictionNN
dataFolder = 'experiment/'

def CreateFolder(p:str):
	r = True
	if not os.path.exists(p):
		try:
			os.mkdir(p)	
		except Exception as e:
			print('Unable to create folder: ' + p)
			f = False
	return r

def SampleGraphs(ticker:str, daysInGraph:int):
	plot = PlotHelper()
	prices = PricingData(ticker)
	print('Loading ' + ticker)
	if prices.LoadHistory(True):
		prices.NormalizePrices()
		sampleData = prices.GetPriceHistory()
		d = sampleData.index[-1]  
		for i in range(0,200, 10): 	 #Add new days to the end for crystal ball predictions
			sampleDate = d - BDay(i) #pick business day to plot
			plot.PlotDataFrameDateRange(sampleData[['Open','High', 'Low','Close']], sampleDate, daysInGraph, 'Sample window ' + str(daysInGraph), 'Date', 'Price', dataFolder + 'samples/sample' + str(i) + '_' + str(daysInGraph)) 

def SampleLSTM(ticker:str):
	plot = PlotHelper()
	prices = PricingData(ticker)
	print('Loading ' + ticker)
	CreateFolder(dataFolder + 'samples')
	if prices.LoadHistory(True):
		prices.NormalizePrices()
		daysInTarget = 15
		daysInTraining = 200
		sampleData = prices.GetPriceHistory()
		endDate  = sampleData.index.max()
		cuttoffDate = endDate - BDay(daysInTarget)
		startDate = cuttoffDate - BDay(daysInTraining)
		print(dataFolder + 'samples/LSTMsampleLearning', startDate, cuttoffDate, endDate)
		plot.PlotDataFrameDateRange(sampleData[['Average']], cuttoffDate, daysInTraining, 'Learn from this series of days', 'Date', 'Price', dataFolder + 'samples/LSTMLearning') 
		plot.PlotDataFrameDateRange(sampleData[['Average']], endDate, daysInTarget, 'Predict what happens after this series of days', 'Date', 'Price', dataFolder + 'samples/LSTMTarget') 

def SampleCNN(ticker:str):
	plot = PlotHelper()
	prices = PricingData(ticker)
	print('Loading ' + ticker)
	if prices.LoadHistory(True):
		prices.NormalizePrices()
		window_size = 80
		target_size = 10
		daysInTraining = 800
		sampleData = prices.GetPriceHistory()
		endDate  = sampleData.index.max()
		cuttoffDate = endDate - BDay(window_size)
		startDate = cuttoffDate - BDay(daysInTraining)
		print(dataFolder + 'samples/LSTMsampleLearning', startDate, cuttoffDate, endDate)
		for i in range(0,10):
			ii = i * window_size
			d1 = startDate + BDay(ii)
			d2 = d1 + BDay(target_size)
			print(d1, d2, window_size, target_size)
			plot.PlotDataFrameDateRange(sampleData[['Average']], d1, window_size, 'Sample image ' + str(i), 'Date', 'Price', dataFolder + 'samples/CNN' + str(i) + 'Sample') 
			plot.PlotDataFrameDateRange(sampleData[['Average']], d2, target_size, 'Target image ' + str(i), 'Date', 'Price', dataFolder + 'samples/CNN' + str(i) + 'Target') 

def CreateAdditionalGraph():
	plot = PlotHelper()
	for root, dirs, files in os.walk(dataFolder):
		for f in files:
			if f.endswith('.csv'):
				predDF = pandas.read_csv(os.path.join(root, f), index_col=0, parse_dates=True, na_values=['nan'])
				modelDescription = f[:len(f)-4]
				print(modelDescription)
				plot.PlotDataFrameDateRange(predDF[['Average','Average_Predicted']], None, 500, modelDescription + '_last500ays', 'Date', 'Price', dataFolder + modelDescription + '_last500Days') 
				plot.PlotDataFrameDateRange(predDF[['Average','Average_Predicted']], None, 1000, modelDescription + '_last1000ays', 'Date', 'Price', dataFolder + modelDescription + '_last1000Days') 

def PredictPrices(prices:PricingData, predictionMethod:int=0, daysForward:int = 5, numberOfLearningPasses:int = 500):
	#Simple procedure to test different prediction methods
	assert(0 <= predictionMethod <= 2)
	plot = PlotHelper()
	if predictionMethod ==0:		#Linear projection
		print('Running Linear Projection model predicting ' + str(daysForward) + ' days...')
		modelDescription = prices.stockTicker + '_Linear_daysforward' + str(daysForward) 
		predDF = prices.GetPriceHistory()
		predDF['Average'] = (predDF['Open'] + predDF['High'] + predDF['Low'] + predDF['Close'])/4
		d = predDF.index[-1]  
		for i in range(0,daysForward): 	#Add new days to the end for crystal ball predictions
			predDF.loc[d + BDay(i+1), 'Average_Predicted'] = 0
		predDF['PastSlope']  = predDF['Average'].shift(daysForward) / predDF['Average'].shift(daysForward*2)
		predDF['Average_Predicted'] = predDF['Average'].shift(daysForward) * predDF['PastSlope'] 
		predDF['PercentageDeviation'] = abs((predDF['Average']-predDF['Average_Predicted'])/predDF['Average'])
	else:
		SourceFieldList = ['High','Low','Open','Close']
		if predictionMethod ==1:	#LSTM learning
			print('Running LSTM model predicting ' + str(daysForward) + ' days...')
			SourceFieldList = None
			UseLSTM = True
			window_size = 10
			modelDescription = prices.stockTicker + '_LSTM' + '_epochs' + str(numberOfLearningPasses) + '_histwin' + str(window_size) + '_daysforward' + str(daysForward) 
		elif predictionMethod ==2: 	#CNN Learning
			print('Running CNN model predicting ' + str(daysForward) + ' days...')
			UseLSTM = False
			window_size = 16 * daysForward
			modelDescription = prices.stockTicker + '_CNN' + '_epochs' + str(numberOfLearningPasses) + '_histwin' + str(window_size) + '_daysforward' + str(daysForward) 
		learningModule = StockPredictionNN(modelName=prices.stockTicker, UseLSTM=UseLSTM)
		learningModule.LoadSource(prices.GetPriceHistory(), SourceFieldList=SourceFieldList, window_size=window_size)
		learningModule.LoadTarget(targetDF=None, prediction_target_days=daysForward)
		learningModule.MakeBatches(batch_size=32, train_test_split=.93)
		learningModule.Train(epochs=numberOfLearningPasses)
		model.Predict(True)
		predDF = learningModule.GetTrainingResults(True, True)
	averageDeviation = predDF['PercentageDeviation'].tail(round(predDF.shape[0]/4)).mean() #Average of the last 25% to account for training.
	print('Average deviation: ', averageDeviation * 100, '%')
	predDF = predDF.reindex(sorted(predDF.columns), axis=1) #Sort columns alphabetical
	predDF.to_csv(dataFolder + modelDescription + '.csv')
	plot.PlotDataFrame(predDF[['Average','Average_Predicted']], modelDescription, 'Date', 'Price', True, 'experiment/' + modelDescription) 
	plot.PlotDataFrameDateRange(predDF[['Average','Average_Predicted']], None, 160, modelDescription + '_last160ays', 'Date', 'Price', dataFolder + modelDescription + '_last160Days') 
	plot.PlotDataFrameDateRange(predDF[['Average','Average_Predicted']], None, 1000, modelDescription + '_last1000ays', 'Date', 'Price', dataFolder + modelDescription + '_last1000Days') 

def RunPredictions(ticker:str='^SPX', numberOfLearningPasses:int = 750):
	prices = PricingData(ticker)
	CreateFolder(dataFolder)
	print('Loading ' + ticker)
	if prices.LoadHistory(True):
		prices.TrimToDateRange('1/1/1950', '3/1/2018')
		prices.NormalizePrices()
		for ii in [4,20,60]:
			for i in range(0,3):
				PredictPrices(prices,i, ii, numberOfLearningPasses)

				
StockTicker='NFLX' #'NFLX', 'AMZN', 'GOOGL', '^SPX'
SampleGraphs('^SPX', 15)
SampleLSTM('^SPX')
SampleCNN('^SPX')
RunPredictions(StockTicker, 10)
CreateAdditionalGraph()