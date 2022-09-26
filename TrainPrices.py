import pandas
from _classes.PriceTradeAnalyzer import PricingData, PriceSnapshot, PlotHelper
from _classes.SeriesPrediction import StockPredictionNN
dataFolder = 'data/prediction/'

def TestPredictionModels(ticker:str='^SPX', numberOfLearningPasses:int = 300):
	#Simple procedure to test different prediction methods 4,20,60 days in the future
	plot = PlotHelper()
	prices = PricingData(ticker)
	if prices.LoadHistory():
		prices.TrimToDateRange('1/1/2000', '3/1/2022')
		print('Loading ' + ticker)
		for daysForward in [4,15,25]: 
			for predictionMethod in range(0,5):
				modelDescription = ticker + '_method' + str(predictionMethod) + '_epochs' + str(numberOfLearningPasses) + '_daysforward' + str(daysForward) 
				print('Predicting ' + str(daysForward) + ' days using method ' + modelDescription)
				prices.PredictPrices(predictionMethod, daysForward, numberOfLearningPasses)
				predDF = prices.GetPriceHistory(includePredictions=True)
				predDF['PercentageDeviation'] = abs((predDF['Average']-predDF['estAverage'])/predDF['Average'])
				averageDeviation = predDF['PercentageDeviation'].tail(round(predDF.shape[0]/4)).mean() #Average of the last 25% to account for training.
				print('Average deviation: ', averageDeviation * 100, '%')
				predDF.to_csv(dataFolder + modelDescription + '.csv')
				plot.PlotDataFrame(predDF[['estAverage','Average']], modelDescription, 'Date', 'Price', True, dataFolder + modelDescription) 
				plot.PlotDataFrameDateRange(predDF[['Average','estAverage']], None, 160, modelDescription + '_last160ays', 'Date', 'Price', dataFolder + modelDescription + '_last160Days') 
				plot.PlotDataFrameDateRange(predDF[['Average','estAverage']], None, 500, modelDescription + '_last500Days', 'Date', 'Price', dataFolder + modelDescription + '_last500Days') 
			
def TrainTickerRaw(ticker:str = '.INX', UseLSTM:bool=True, useGenericModel:bool=True, prediction_target_days:int = 5, epochs:int = 300, usePercentages:bool=False, hidden_layer_size:int=512, dropout:bool=True, dropout_rate:float=0.01, learning_rate:float=2e-5):
	plot = PlotHelper()
	prices = PricingData(ticker)
	print('Loading ' + ticker)
	if prices.LoadHistory():
		#prices.TrimToDateRange('1/1/2000', '10/1/2021')
		if usePercentages: 
			prices.ConvertToPercentages() #Percentages don't work well I suspect because small errors have a huge impact when you revert back to the original prices and they roll forward
		else:
			prices.NormalizePrices()
		prices.CalculateStats()
		modelDescription = ticker
		if useGenericModel: modelDescription = 'Prices'
		model = StockPredictionNN(baseModelName=modelDescription, UseLSTM=UseLSTM)
		if UseLSTM:
			window_size = 1
			modelDescription += '_LSTM_epochs' + str(epochs) + '_histwin' + str(window_size) + '_daysforward' + str(prediction_target_days) 
			FieldList = ['Average']
		else: #CNN
			window_size = 16 * prediction_target_days
			modelDescription += '_CNN_epochs' + str(epochs) + '_histwin' + str(window_size) + '_daysforward' + str(prediction_target_days) 
			FieldList = ['High','Low','Open','Close']
		if usePercentages: modelDescription += '_percentages'
		model.LoadSource(sourceDF=prices.GetPriceHistory(), FieldList=FieldList, window_size=window_size)
		model.LoadTarget(targetDF=None, prediction_target_days=prediction_target_days)
		model.MakeBatches(batch_size=32, train_test_split=.93)
		model.BuildModel(hidden_layer_size=hidden_layer_size, dropout=dropout, dropout_rate=dropout_rate, learning_rate=learning_rate)
		if epochs == 0:
			if (not model.Load()): epochs=300
		elif useGenericModel:
			model.Load()
		if epochs > 0:
			model.Train(epochs=epochs)
			print('train accuracy: ' + str(model.trainAccuracy), 'test accuracy: ' + str(model.testAccuracy), 'loss: ' + str(model.trainLoss))
			model.DisplayTrainingSummary()
		model.Predict(True)
		if epochs >= 10: model.Save()
		if usePercentages: 
			predDF = model.GetTrainingResults(True, True)
			predDF = predDF.loc[:,['Average', 'Average_Predicted']]
			print('Unraveling percentages..')
			predDF['Average_Predicted'].fillna(0, inplace=True)
			predDF.iloc[0] = prices.CTPFactor['Average']
			for i in range(1, predDF.shape[0]):
				predDF.iloc[i] = (1 + predDF.iloc[i]) * predDF.iloc[i-1]
			print(predDF)
			predDF['PercentageDeviation'] = abs((predDF['Average']-predDF['Average_Predicted'])/predDF['Average'])
			predDF.to_csv(dataFolder + ticker + '_' + modelDescription +'.csv')
			plot.PlotDataFrame(predDF[['Average','Average_Predicted']], ticker + '_' + modelDescription, 'Date', 'Price', True, dataFolder + modelDescription) 
			plot.PlotDataFrameDateRange(predDF[['Average','Average_Predicted']], None, 160, ticker + '_' + modelDescription + '_last160ays', 'Date', 'Price', dataFolder + modelDescription + '_last160Days') 
			plot.PlotDataFrameDateRange(predDF[['Average','Average_Predicted']], None, 500, ticker + '_' + modelDescription + '_last500Days', 'Date', 'Price', dataFolder + modelDescription + '_last500Days') 
		else:
			model.PredictionResultsSave(ticker + '_' + modelDescription, True, True)
			model.PredictionResultsPlot(ticker + '_' + modelDescription, True, False)
		
if __name__ == '__main__':
	TrainTickerRaw('MSFT', UseLSTM=True, useGenericModel=True, prediction_target_days = 5, epochs = 400)
	TrainTickerRaw('XOM', UseLSTM=True, useGenericModel=True, prediction_target_days = 5, epochs = 400)	
	TestPredictionModels('.INX', numberOfLearningPasses=400)
	TestPredictionModels('MSFT', numberOfLearningPasses=100)
	TestPredictionModels('TSLA', numberOfLearningPasses=400)
