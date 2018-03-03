import pandas
from _classes.PriceTradeAnalyzer import PricingData, PriceSnapshot, PlotHelper
from _classes.SeriesPrediction import StockPredictionNN
dataFolder = 'data/prediction/'

def TestPredictionModels(ticker:str='^SPX', numberOfLearningPasses:int = 500):
	#Simple procedure to test different prediction methods 4,20,60 days in the future
	plot = PlotHelper()
	prices = PricingData(ticker)
	print('Loading ' + ticker)
	for daysForward in [4,20,60]: 
		for predictionMethod in range(0,5):
			usePercentages = False
			normalizePrices = False
			if predictionMethod ==3 or predictionMethod ==4:  usePercentages = True
			modelDescription = ticker + '_method' + str(predictionMethod) + '_epochs' + str(numberOfLearningPasses) + '_daysforward' + str(daysForward) 
			if usePercentages: 
				modelDescription += '_percentages'
				prices.ConvertToPercentages()
			elif normalizePrices:
				prices.NormalizePrices()
			print('Predicting ' + str(daysForward) + ' days using method ' + modelDescription)
			prices.PredictPrices(predictionMethod, daysForward, numberOfLearningPasses)
			if usePercentages: 
				modelDescription += '_percentages'
				prices.ConvertToPercentages()
			elif normalizePrices:
				prices.NormalizePrices()
			predDF = prices.pricePredictions.copy()
			predDF = predDF.join(prices.GetPriceHistory())
			predDF['PercentageDeviation'] = abs((predDF['Average']-predDF['estAverage'])/predDF['Average'])
			averageDeviation = predDF['PercentageDeviation'].tail(round(predDF.shape[0]/4)).mean() #Average of the last 25% to account for training.
			print('Average deviation: ', averageDeviation * 100, '%')
			predDF.to_csv(dataFolder + modelDescription + '.csv')
			plot.PlotDataFrame(predDF[['estAverage','Average']], modelDescription, 'Date', 'Price', True, dataFolder + modelDescription) 

def TraintickerRaw(ticker:str = '^SPX', UseLSTM:bool=True, prediction_target_days:int = 5, epochs:int = 500, usePercentages:bool=False):
	plot = PlotHelper()
	prices = PricingData(ticker)
	print('Loading ' + ticker)
	if prices.LoadHistory(True):
		prices.TrimToDateRange('1/1/2000', '3/1/2018')
		if usePercentages: 
			prices.ConvertToPercentages()
		else:
			prices.NormalizePrices()
		model = StockPredictionNN()
		if UseLSTM:
			window_size = 1
			modelDescription = ticker + '_LSTM'
			modelDescription += '_epochs' + str(epochs) + '_histwin' + str(window_size) + '_daysforward' + str(prediction_target_days) 
			if usePercentages: modelDescription += '_percentages'
			SourceFieldList = None
			#Note: LSTM doesn't benefit from a window size of > 1 since it is inherent in the model it just add noise
			model.LoadData(prices.GetPriceHistory(), window_size=window_size, prediction_target_days=prediction_target_days, UseLSTM=UseLSTM, SourceFieldList=SourceFieldList, batch_size=10, train_test_split=.93)
			model.TrainLSTM(epochs=epochs, learning_rate=0.001, dropout_rate=0.8, gradient_clip_margin=4)
			#model.PredictLSTM(epochs=epochs)
		else: #CNN
			window_size = 16 * prediction_target_days
			modelDescription = ticker + '_CNN'
			modelDescription += '_epochs' + str(epochs) + '_histwin' + str(window_size) + '_daysforward' + str(prediction_target_days) 
			if usePercentages: modelDescription += '_percentages'
			SourceFieldList = ['High','Low','Open','Close']
			model.LoadData(prices.GetPriceHistory(), window_size=window_size, prediction_target_days=prediction_target_days, UseLSTM=UseLSTM, SourceFieldList=SourceFieldList, batch_size=32, train_test_split=.93)
			model.TrainCNN(epochs=epochs)
			#model.PredictCNN(epochs=epochs, learning_rate=2e-5)
		if usePercentages: 
			predDF = model.GetTrainingResults(True, True)
			predDF.fillna(method='bfill', inplace=True)	
			basePrice = prices.CTPFactor['Average']
			predDF.iloc[0] = prices.CTPFactor['Average']
			#for i in range(0,predDF.shape[1]):  predDF.iloc[0,i] = basePrice
			for i in range(1, predDF.shape[0]):
				predDF.iloc[i] = (1 + predDF.iloc[i]) * predDF.iloc[i-1]
			predDF['PercentageDeviation'] = abs((predDF['Average']-predDF['Average_Predicted'])/predDF['Average'])
			predDF.to_csv(dataFolder + modelDescription +'.csv')
			plot.PlotDataFrame(predDF[['Average','Average_Predicted']], modelDescription, 'Date', 'Price', True, dataFolder + modelDescription) 
			plot.PlotDataFrameDateRange(predDF[['Average','Average_Predicted']], None, 160, modelDescription + '_last160ays', 'Date', 'Price', dataFolder + modelDescription + '_last160Days') 
		else:
			model.PredictionResultsSave(modelDescription, True, True)
			model.PredictionResultsPlot(modelDescription, True, False)
		
#TraintickerRaw('^SPX', UseLSTM=False, prediction_target_days = 30, epochs = 50)
#TraintickerRaw('^SPX', UseLSTM=True, prediction_target_days = 10, epochs = 750)
#TraintickerRaw('^SPX', UseLSTM=False, prediction_target_days = 10, epochs = 750)
#TraintickerRaw('^SPX', UseLSTM=True, prediction_target_days = 16, epochs = 750)
#TraintickerRaw('^SPX', UseLSTM=False, prediction_target_days = 16, epochs = 750)
#TraintickerRaw('^SPX', UseLSTM=True, prediction_target_days = 30, epochs = 750)
#TraintickerRaw('^SPX', UseLSTM=False, prediction_target_days = 30, epochs = 750)
#TraintickerRaw('^SPX', UseLSTM=True, prediction_target_days = 5, epochs = 1000, usePercentages=True)
#TraintickerRaw('^SPX', UseLSTM=True, prediction_target_days = 5, epochs = 1000, usePercentages=False)
#TraintickerRaw('^SPX', UseLSTM=False, prediction_target_days = 5, epochs = 1000, usePercentages=True)
#TraintickerRaw('^SPX', UseLSTM=False, prediction_target_days = 5, epochs = 1000, usePercentages=False)
TraintickerRaw('TSLA', UseLSTM=True, prediction_target_days = 5, epochs = 500)
TraintickerRaw('TSLA', UseLSTM=False, prediction_target_days = 5, epochs = 500)
#TestPredictionModels('TSLA', 5)
