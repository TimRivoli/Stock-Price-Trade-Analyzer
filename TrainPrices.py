import pandas as pd
import matplotlib
#matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt
from _classes.PriceTradeAnalyzer import PricingData, PriceSnapshot, PlotHelper
from _classes.SeriesPrediction import StockPredictionNN

def PredictPrices(ticker:str = '^SPX', predictionMethod:int=0, daysForward:int = 4):
	#Simple procedure to test different prediction methods
	assert(0 <= predictionMethod <= 2)
	prices = PricingData(ticker)
	print('Loading ' + ticker)
	if prices.LoadHistory(True):
		plot = PlotHelper()
		#prices.TrimToDateRange('1/1/2000', '2/1/2018')
		#prices.ConvertToPercentages()	#This doesn't help
		prices.NormalizePrices()
		if predictionMethod ==0:		#Linear projection
			modelDescription = ticker + '_Linear_daysforward' + str(daysForward) 
			predDF = prices.GetPriceHistory()
			predDF['CurrentAveragePrice'] =  (predDF['Open'] + predDF['High'] + predDF['Low'] + predDF['Close'])/4
			predDF['Slope']  = predDF['CurrentAveragePrice'] / predDF['CurrentAveragePrice'].shift(daysForward)
			predDF['FuturePrice_Predicted'] = predDF['CurrentAveragePrice'] * predDF['Slope'] 
			predDF['FuturePrice_Actual'] = predDF['CurrentAveragePrice'].shift(-daysForward)
			predDF['PercentageDeviation'] = abs((predDF['FuturePrice_Actual']-predDF['FuturePrice_Predicted'])/predDF['FuturePrice_Actual'])
		else:
			learningModule = StockPredictionNN()
			SourceFieldList = ['High','Low','Open','Close']
			numberOfLearningPasses = 350
			if predictionMethod ==1:	#LSTM learning
				window_size = 1
				modelDescription = ticker + '_LSTM' + '_epochs' + str(numberOfLearningPasses) + '_histwin' + str(window_size) + '_daysforward' + str(daysForward) 
				learningModule.LoadData(prices.GetPriceHistory(), window_size=window_size, prediction_target_days=daysForward, UseLSTM=True, SourceFieldList=SourceFieldList, batch_size=10, train_test_split=.93)
				learningModule.TrainLSTM(epochs=numberOfLearningPasses, learning_rate=0.001, dropout_rate=0.8, gradient_clip_margin=4)
			elif predictionMethod ==2: 	#CNN Learning
				window_size = 16 * daysForward
				modelDescription = ticker + '_CNN' + '_epochs' + str(numberOfLearningPasses) + '_histwin' + str(window_size) + '_daysforward' + str(daysForward) 
				learningModule.LoadData(prices.GetPriceHistory(), window_size=window_size, prediction_target_days=daysForward, UseLSTM=False, SourceFieldList=SourceFieldList, batch_size=32, train_test_split=.93)
				learningModule.TrainCNN(epochs=numberOfLearningPasses)
			predDF = learningModule.GetTrainingResults(True, True)
		averageDeviation = predDF['PercentageDeviation'].mean() 
		print('Average deviation: ', averageDeviation * 100, '%')
		predDF.to_csv('data/prediction/' + modelDescription + '.csv')
		plot.PlotDataFrame(predDF[['FuturePrice_Predicted','FuturePrice_Actual']], modelDescription, 'Date', 'Price', True, 'data/prediction/' + modelDescription) 

def TestPredictionModels():
	#Test three different prediction models for distance 4,20,60 days in the future
	StockTicker='GOOGL' #'NFLX', 'AMZN', 'GOOGL', '^SPX'
	for ii in [4,20,60]: 
		for i in range(0,3):
			PredictPrices(StockTicker,i, ii)

def TrainMultipletickers(tickerList:list, prediction_target_days:int = 16, epochs:int = 1000):
	#This is a disaster, adding related tickers does not improve accuracy
	modelDescription = 'Multiticker_CNN'
	modelDescription = modelDescription + '_epochs' + str(epochs) + '_histwin' + str(window_size) + '_daysforward' + str(prediction_target_days) 
	firstTime = True
	for ticker in tickerList:
		print('Loading ' + ticker)
		prices = PricingData(ticker)
		if prices.LoadHistory(True):
			prices.TrimToDateRange('1/1/2000', '2/1/2018')
			prices.NormalizePrices()
			if firstTime:
				sourceDF = prices.GetPriceHistory(list(['High','Low','Open','Close']))
				firstTime = False
			else:
				sourceDF = sourceDF.join(prices.GetPriceHistory(list(['High','Low','Open','Close'])), rsuffix='_' + ticker)
	model = StockPredictionNN()
	SourceFieldList = None
	#SourceFieldList = ['High','Low','Open','Close']
	window_size = 1
	model.LoadData(sourceDF, window_size=window_size, prediction_target_days=prediction_target_days, UseLSTM=False, SourceFieldList=SourceFieldList, batch_size=32, train_test_split=.93)
	model.TrainCNN(epochs=epochs)
	resultDF = model.GetTrainingResults()
	model.PredictionResultsSave(modelDescription)
	model.PredictionResultsPlot(modelDescription)

def Trainticker(ticker:str = 'Googl', UseLSTM:bool=True, prediction_target_days:int = 5, epochs:int = 100, usePercentages:bool=False):
	prices = PricingData(ticker)
	print('Loading ' + ticker)
	if prices.LoadHistory(True):
		prices.TrimToDateRange('1/1/2000', '2/1/2018')
		if usePercentages: 
			prices.ConvertToPercentages()
		else:
			prices.NormalizePrices()
		model = StockPredictionNN()
		if UseLSTM:
			window_size = 1
			modelDescription = ticker + '_LSTM'
			modelDescription += '_epochs' + str(epochs) + '_histwin' + str(window_size) + '_daysforward' + str(prediction_target_days) 
			SourceFieldList = None
			#Note: LSTM doesn't benefit from a window size of > 1 since it is inherent in the model it just add noise
			model.LoadData(prices.GetPriceHistory(), window_size=window_size, prediction_target_days=prediction_target_days, UseLSTM=UseLSTM, SourceFieldList=SourceFieldList, batch_size=10, train_test_split=.93)
			model.TrainLSTM(epochs=epochs, learning_rate=0.001, dropout_rate=0.8, gradient_clip_margin=4)
			#model.PredictLSTM(epochs=epochs)
		else: #CNN
			window_size = 16 * prediction_target_days
			modelDescription = ticker + '_CNN'
			modelDescription += '_epochs' + str(epochs) + '_histwin' + str(window_size) + '_daysforward' + str(prediction_target_days) 
			SourceFieldList = ['High','Low','Open','Close']
			model.LoadData(prices.GetPriceHistory(), window_size=window_size, prediction_target_days=prediction_target_days, UseLSTM=UseLSTM, SourceFieldList=SourceFieldList, batch_size=32, train_test_split=.93)
			model.TrainCNN(epochs=epochs)
			#model.PredictCNN(epochs=epochs)
		if usePercentages: 
			resultDF = model.GetTrainingResults(True, False)
			basePrice = prices.CTPFactor['Average']
			for i in range(resultDF.shape[1]):  resultDF.iloc[0,i] = basePrice
			for i in range(1, 20): resultDF.iloc[i] = resultDF.iloc[i-1]
			for i in (range(20, resultDF.shape[0])):
				resultDF.iloc[i] = (resultDF.iloc[i]+1) * resultDF.iloc[i-1]
			if resultDF.shape[1] == 1:
				resultDF['PercentageDeviation'] = abs((resultDF['FuturePrice_Actual']-resultDF['FuturePrice_Predicted'])/resultDF['FuturePrice_Actual'])
			elif resultDF.shape[1] == 4:
				resultDF['FuturePrice_Actual'] = (resultDF['Open'] + resultDF['High'] + resultDF['Low'] + resultDF['Close'])/4
				resultDF['FuturePrice_Predicted'] = (resultDF['Open_Predicted'] + resultDF['High_Predicted'] + resultDF['Low_Predicted'] + resultDF['Close_Predicted'])/4
				resultDF['PercentageDeviation'] = abs((resultDF['FuturePrice_Actual']-resultDF['FuturePrice_Predicted'])/resultDF['FuturePrice_Actual'])
			resultDF.to_csv(modelDescription +'.csv')
			resultDF.plot()
			plt.show()
		else:
			model.PredictionResultsSave(modelDescription, True, True)
			model.PredictionResultsPlot(modelDescription, True, False)

		
#TrainMultipletickers(['GOOGL','AAPL','TSLA'])
#TestPredictionModels()
Trainticker('^SPX', UseLSTM=True, prediction_target_days = 5, epochs = 750)
Trainticker('^SPX', UseLSTM=False, prediction_target_days = 5, epochs = 750)
Trainticker('^SPX', UseLSTM=True, prediction_target_days = 10, epochs = 750)
Trainticker('^SPX', UseLSTM=False, prediction_target_days = 10, epochs = 750)
Trainticker('^SPX', UseLSTM=True, prediction_target_days = 16, epochs = 750)
Trainticker('^SPX', UseLSTM=False, prediction_target_days = 16, epochs = 750)
Trainticker('^SPX', UseLSTM=True, prediction_target_days = 30, epochs = 750)
Trainticker('^SPX', UseLSTM=False, prediction_target_days = 30, epochs = 750)
