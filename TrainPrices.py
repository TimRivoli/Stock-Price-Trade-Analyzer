import sys, pandas
from _classes.PriceTradeAnalyzer import PricingData, PriceSnapshot, PlotHelper
from _classes.SeriesPrediction import StockPredictionNN
dataFolder = 'data/prediction/'

def TestPredictionModels(ticker:str='^SPX', numberOfLearningPasses:int = 300):
	#Simple procedure to test different prediction methods 4,20,60 days in the future
	plot = PlotHelper()
	prices = PricingData(ticker)
	if prices.LoadHistory():
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
			
def TrainTickerRaw(ticker:str = '.INX', model_type:str='LSTM', use_generic_model:bool=False, prediction_target_days:int = 5, time_steps:int=30, epochs:int = 300):
	use_percentage_change=False
	#, hidden_layer_size:int=256, dropout_rate:float=0.01, learning_rate:float=0.00001
	plot = PlotHelper()
	prices = PricingData(ticker)
	print('Loading ' + ticker)
	if prices.LoadHistory():
		if use_percentage_change: 
			prices.ConvertToPercentages() #Percentages don't work well I suspect because small errors have a huge impact when you revert back to the original prices and they roll forward
		else:
			prices.NormalizePrices()
		prices.CalculateStats()
		base_model_name = ticker
		if use_generic_model: base_model_name = 'Prices'
		model = StockPredictionNN(base_model_name=base_model_name, model_type=model_type)
		#time_steps = 16 * prediction_target_days
		if model_type =='CNN': #For CNN the field order is significant since it is treated as an image
			field_list = ['Average','EMA_Short','EMA_Long','Deviation_5Day','Deviation_15Day','LossStd_1Year','PC_1Month','PC_6Month','PC_1Year','PC_2Year']
		else:
			#field_list = ['Average']
			field_list = ['Average','EMA_Short','EMA_Long','Deviation_5Day','Deviation_15Day','LossStd_1Year','PC_1Month','PC_6Month','PC_1Year','PC_2Year']
		df = prices.GetPriceHistory()
		df['Average']=df['Average_5Day'] #Too much noise in the daily
		model.LoadSource(sourceDF=prices.GetPriceHistory(), field_list=field_list, time_steps=time_steps, use_percentage_change=use_percentage_change)
		model.LoadTarget(targetDF=None, prediction_target_days=prediction_target_days)
		model.MakeTrainTest(batch_size=32, train_test_split=.93)
		#model.DisplayDataSample()
		model.BuildModel(dropout_rate=0, use_BatchNormalization=False)
		if epochs == 0:
			if (not model.Load()): epochs=300
		elif use_generic_model:
			model.Load()
		if epochs > 0:
			model.Train(epochs=epochs)
		model.Predict(use_full_data_set=True)
		#if epochs >= 10: model.Save()
		model.PredictionResultsSave(filename=model.model_name, include_target=True, include_accuracy=False, include_input=True)
		model.PredictionResultsSave(filename=model.model_name + '_Accuracy', include_target=True, include_accuracy=True, include_input=False)
		model.PredictionResultsPlot(filename=model.model_name, include_target=True, include_accuracy=False)
		
if __name__ == '__main__':
	switch = 1
	if len(sys.argv[1:]) > 0: switch = sys.argv[1:][0]
	if switch == '1':
		for t in [60, 90, 120]:
			TrainTickerRaw('XOM', model_type='LSTM', use_generic_model=False, prediction_target_days=10, time_steps=t, epochs=350)
			TrainTickerRaw('XOM', model_type='BiLSTM', use_generic_model=False, prediction_target_days=10, time_steps=t, epochs=350)
	elif switch == '2':
		for t in [60, 90, 120]:
			TrainTickerRaw('XOM', model_type='GRU', use_generic_model=False, prediction_target_days=1, time_steps=t, epochs=350)
	elif switch == '3':
		for t in [90, 120, 160]:
			TrainTickerRaw('XOM', model_type='CNN_LSTM', use_generic_model=False, prediction_target_days=1, time_steps=t, epochs=350)
			TrainTickerRaw('XOM', model_type='CNN', use_generic_model=False, prediction_target_days=1, time_steps=t, epochs=350)
	elif switch == '4':
		for t in [1, 10, 30, 90, 120]:
			TrainTickerRaw('XOM', model_type='GRU', use_generic_model=False, prediction_target_days=10, time_steps=t, epochs=350)
	elif switch == '5':
		TrainTickerRaw('XOM', model_type='Simple', use_generic_model=False, prediction_target_days=10, epochs=350)
	TrainTickerRaw('MSFT', model_type='LSTN', use_generic_model=True, prediction_target_days = 5, epochs = 400)
	#TrainTickerRaw('XOM', model_type='LSTN', use_generic_model=True, prediction_target_days = 5, epochs = 400)	
	#TrainTickerRaw('.INX', model_type='CNN', prediction_target_days = 5, epochs = 4)
	#TestPredictionModels('.INX', numberOfLearningPasses=400)
	TestPredictionModels('MSFT', numberOfLearningPasses=100)
	#TestPredictionModels('TSLA', numberOfLearningPasses=400)
	#TrainTickerRaw('CEIX', model_type='LSTN', use_generic_model=True, prediction_target_days = 10, epochs = 750)
	#TestPredictionModels('CEIX', numberOfLearningPasses=400)
