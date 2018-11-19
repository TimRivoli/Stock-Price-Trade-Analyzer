nonGUIEnvironment = False	#hosted environments often have no GUI so matplotlib won't be outputting to display
import os, numpy, pandas, matplotlib, keras
from pandas.tseries.offsets import BDay
from _classes.Utility import *

#-------------------------------------------- Global settings -----------------------------------------------
nonGUIEnvironment = ReadConfigBool('Settings', 'nonGUIEnvironment')
if nonGUIEnvironment: matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt

#-------------------------------------------- Classes -----------------------------------------------
class SeriesPredictionNN(object):
	_dataFolderTensorFlowModels = 'data/tfmodels/'
	_dataFolderPredictionResults = 'data/prediction/'
	sourceDataLoaded = False
	targetDataLoaded = False
	batchesCreated = False
	predictClasses = False
	model = None

	def __init__(self, modelName:str='', UseLSTM:bool=True, PredictionResultsDataFolder:str='', TensorFlowModelsDataFolder:str=''):
		self.baseModelName = modelName
		self.UseLSTM = UseLSTM
		if UseLSTM:
			self.baseModelName += '_LSTM'
		else:
			self.baseModelName += '_CNN'
		self.modelName = self.baseModelName
		keras.backend.clear_session()
		if not PredictionResultsDataFolder =='':
			if CreateFolder(PredictionResultsDataFolder):
				if not PredictionResultsDataFolder[-1] =='/': PredictionResultsDataFolder += '/'
				self._dataFolderPredictionResults = PredictionResultsDataFolder
		if not TensorFlowModelsDataFolder =='':
			if CreateFolder(TensorFlowModelsDataFolder):
				if not TensorFlowModelsDataFolder[-1] =='/': TensorFlowModelsDataFolder += '/'
				self._dataFolderTensorFlowModels = TensorFlowModelsDataFolder
		CreateFolder(self._dataFolderTensorFlowModels)
		CreateFolder(self._dataFolderPredictionResults)	

#  ----------------------------------------------------  Data preparation  -------------------------------------------------------------------
	def _CustomSourceOperations(self, SourceFieldList:list = None):	pass
	def _CustomTargetOperations(self): pass
	
	def LoadSource(self, sourceDF:pandas.DataFrame, SourceFieldList:list=None, window_size:int=1):
		self.sourceDF = sourceDF.copy()	
		self.sourceDF.sort_index(inplace=True)				#Expecting date descending index
		self.sourceDF.fillna(method='bfill', inplace=True)	#TF will start producing Nan results if it encounters Nan values
		self._CustomSourceOperations(SourceFieldList)
		self.daysInDataSet = self.sourceDF.shape[0]
		self.feature_count =  self.sourceDF.shape[1]
		if self.sourceDF.isnull().values.any(): 
			print('Nan values in source input.  This will break the training.\n')
			assert(False)
		self.window_size = window_size	
		X = [] #create dimension for window of past days, 0 position is most recent
		i = self.sourceDF.shape[0]-1	#start at the last element, move back to first element beyond the window
		while i >= window_size:
			v = self.sourceDF.iloc[i-window_size:i].values
			v = v[::-1] #flip it so that 0 is most recent
			X.insert(0,v)
			i -= 1
		self.X = X
		self.sourceDF.drop(self.sourceDF.index[:self.window_size], inplace=True) #Forget anything that occurs before the history window.  Data is not part of the training source or target
		print('Features in source dataset: {}'.format(self.feature_count))
		print('Days in the source features data set: {}'.format(self.daysInDataSet))
		print('Window size: ', self.window_size)
		self.sourceDataLoaded = True
		
	def LoadTarget(self, targetDF:pandas.DataFrame=None, prediction_target_days:int=1):
		if not self.sourceDataLoaded:
			print('Load source data before target data.')
			assert(False)
		if targetDF is None: 
			self.targetDF = self.sourceDF.copy()
		else:
			self.targetDF = targetDF.copy()
			self.targetDF[self.sourceDF.index.min():]	#trim any entries prior to start of source
		self.prediction_target_days = prediction_target_days
		self._CustomTargetOperations(prediction_target_days)
		print('Classes in target values: {}'.format(self.number_of_classes))
		print('Days in target value data: {}'.format(self.targetDF.shape[0]))
		if prediction_target_days > 0: print('Targets will be pushed ' + str(prediction_target_days) + ' forward for prediction.')
		if self.targetDF.isnull().values.any(): 
			print('Nan values in target input.  This will break the training.\n')
			assert(False)
		if not len(self.X) ==  len(self.y):
			print('X shape: ', len(self.X))
			print('y shape: ', len(self.y))
			assert(False)
		if not self.UseLSTM and not self.number_of_classes == self.feature_count:
			print('CNN model requires feature count to equal class count.')
			assert(False)	
		if self.targetDF.shape[1] == 1:
			self.predictionDF = pandas.DataFrame(index=self.targetDF.index) 
		else:
			self.predictionDF = pandas.DataFrame(columns=list(self.targetDF.columns.values), index=self.targetDF.index)
		self.predictionDF = self.predictionDF[prediction_target_days:]	#drop the first n days, not predicted
		d = self.predictionDF.index[-1] 
		for i in range(0,prediction_target_days): 	#Add new days to the end for crystal ball predictions
			self.predictionDF.loc[d + BDay(i+1), self.targetDF.columns.values[0]] = numpy.nan	
		assert(self.predictionDF.shape[0] == self.targetDF.shape[0] and self.predictionDF.shape[0] == self.sourceDF.shape[0])	#This is key to aligning the results since train data has not dates we use rows
		self.modelName = self.baseModelName + '_win' + str(self.window_size) + '_days' + str(prediction_target_days) + '_feat' + str(self.feature_count) + '_class' + str(self.number_of_classes)
		self.targetDataLoaded = True

	def MakeBatches(self, batch_size=32, train_test_split=.90):
		if not (self.sourceDataLoaded):
			print('Source data needs to be loaded before batching.')
			assert(False)
		elif not self.targetDataLoaded: 
			print('Target data not specified.  Initializing with copy of source.')
			self.LoadTarget()
		self.batch_size = batch_size
		print('Batching data...')
		daysOfData=len(self.X)
		self.train_start_offset = daysOfData % batch_size
		train_test_cuttoff = round((daysOfData // batch_size) * train_test_split) * batch_size + batch_size + self.train_start_offset
		self.X_train  = numpy.array(self.X[:train_test_cuttoff]) #to train_test_cuttoff
		self.X_test = numpy.array(self.X[train_test_cuttoff:])   #after train_test_cuttoff
		if self.UseLSTM:
			self.y_train = numpy.array(self.y[:train_test_cuttoff])  #to train_test_cuttoff
			self.y_test = numpy.array(self.y[train_test_cuttoff:])   #after train_test_cuttoff, this is never used
		else:
			self.y_train = numpy.array(self.y[:train_test_cuttoff])  #to train_test_cuttoff
			self.y_test = numpy.array(self.y[train_test_cuttoff:])   #after train_test_cuttoff, this is never used
		print('train_start_offset: ', self.train_start_offset)
		print('train_test_cuttoff: ', train_test_cuttoff)
		print('(Days, Window size, Features)')
		print('X_train size: {}'.format(self.X_train.shape)) 
		print('X_test size: {}'.format(self.X_test.shape))   
		print('(Days, Classes)')
		print('y_train size: {}'.format(self.y_train.shape))
		print('\n')
		self.batchesCreated = True

#  ----------------------------------------------------  Model Builds  -----------------------------------------------------------------
	def _BuildCNNModel(self, normalizeInput:bool=False, layer_count:int=1, hidden_layer_size:int=256, dropout:bool=True, dropout_rate=0.8, optimizer:str = 'adam', learning_rate=2e-5, metrics:list=['accuracy','val_loss']):
		model = keras.models.Sequential()
		model.add(keras.layers.InputLayer(name='input', input_shape=(self.window_size, self.feature_count))) 
		model.add(keras.layers.Conv1D(64*3, int(self.window_size-15), activation='relu'))
		model.add(keras.layers.Conv1D(64*2, 10, activation='relu'))
		model.add(keras.layers.Conv1D(64*1, 5, activation='relu'))
		model.add(keras.layers.Conv1D(32, 3, activation='relu'))
		model.add(keras.layers.Dense(self.number_of_classes))
		model.compile(loss='mean_squared_error',  optimizer=optimizer,  metrics=metrics)
		self.model = model	

	def _BuildLSTMModel(self, normalizeInput:bool=False, layer_count:int=1, hidden_layer_size:int=256, dropout:bool=True, dropout_rate=0.8, optimizer:str = 'adam', learning_rate=2e-5, metrics:list=['accuracy','val_loss']):
		if self.predictClasses:
			loss_function = 'categorical_crossentropy'
			output_activation = 'softmax'
		else:
			loss_function = 'mean_squared_error'
			output_activation = 'linear'		
		model = keras.models.Sequential()
		model.add(keras.layers.InputLayer(input_shape=(self.window_size, self.feature_count)))
		if normalizeInput: 
			model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
		for i in range(layer_count):
			model.add(keras.layers.LSTM(hidden_layer_size, return_sequences=(i < layer_count-1)))
			if dropout: model.add(keras.layers.Dropout(dropout_rate))
		print('(optimizer, loss, activation) {', optimizer,',', loss_function, ',', output_activation, '}')
		model.add(keras.layers.Dense(self.number_of_classes, activation=output_activation))
		if optimizer == 'adam' :
			opt_function = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
			model.compile(optimizer=opt_function, loss=loss_function, metrics=metrics, )
		else:
			model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics, )		
		self.model = model

	def BuildModel(self, normalizeInput=False, layer_count=2, hidden_layer_size=256, dropout=True, dropout_rate=0.8, optimizer = 'adam', learning_rate=2e-5, metrics=['accuracy']):
		if not (self.sourceDataLoaded):
			print('Source data needs to be loaded before building model.')
			assert(False)
		if not self.batchesCreated: self.MakeBatches()
		if self.batchesCreated:
			keras.backend.clear_session()
			if self.UseLSTM:
				self._BuildLSTMModel(normalizeInput=normalizeInput, layer_count=layer_count, hidden_layer_size=hidden_layer_size, dropout=dropout, dropout_rate=dropout_rate, optimizer = optimizer, learning_rate=learning_rate, metrics=metrics)
			else:
				self._BuildCNNModel(normalizeInput=normalizeInput, layer_count=layer_count, hidden_layer_size=hidden_layer_size, dropout=dropout, dropout_rate=dropout_rate, optimizer = optimizer, learning_rate=learning_rate, metrics=metrics)
		
#  ----------------------------------------------------  Training / Prediction / Utility -----------------------------------------------------------------
	def Train(self, epochs=100):
		if self.model is None: self.BuildModel()		
		if self.model is not None: 
			if not self.batchesCreated: self.MakeBatches()
			hist = self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, epochs=epochs, callbacks=[keras.callbacks.EarlyStopping(patience=2)])
			val_loss, val_accuracy = self.model.evaluate(self.X_test, self.y_test)
			print('loss, accurracy', val_loss, val_accuracy)

	def _RecordPredictedValue(self, rowIndex, value):
		self.predictionDF.iloc[rowIndex] = value

	def Predict(self, useFullDataSet:bool=False):	
		if self.model is None: self.BuildModel()		
		if self.model is not None: 
			if not self.batchesCreated: self.MakeBatches()
			print('Running predictions...')
			d = [self.X_test]
			if useFullDataSet: d = [numpy.array(self.X)]
			if self.predictClasses:
				predictions = self.model.predict_classes(d)	
			else:
				predictions = self.model.predict(d)	
			start_index = len(self.predictionDF) - len(predictions) #getting this number correct is key to aligning predictions with the correct date
			for i in range(len(predictions)):
				self._RecordPredictedValue(start_index + i, predictions[i])
			print('Predictions complete.')

	def Load(self):
		keras.backend.clear_session()
		if not self.batchesCreated: self.MakeBatches()
		f = self._dataFolderTensorFlowModels + self.modelName + '.h5'
		if FileExists(f):
			keras.backend.clear_session()
			self.model = keras.models.load_model(f)
			print('Model restored from disk')

	def Save(self):
		if self.model is None:
			print('No model loaded.')
		else:
			f = self._dataFolderTensorFlowModels + self.modelName + '.h5'
			self.model.save(f)

	def GetTrainingResults(self, includeTrainingTargets:bool = False, includeAccuracy:bool = False):
		if includeTrainingTargets:
			r = self.targetDF.join(self.predictionDF, how='outer', rsuffix='_Predicted')
			r = r.reindex(sorted(r.columns), axis=1)
			return r
		else:
			return self.predictionDF

	def PredictionResultsSave(self, filename:str, includeTrainingTargets:bool = False, includeAccuracy:bool = False):
		r = self.GetTrainingResults(includeTrainingTargets, includeAccuracy)
		if not filename[-4] =='.': filename += '.csv'
		r.to_csv(self._dataFolderPredictionResults + filename)

	def PredictionResultsPlot(self, filename:str='', includeTrainingTargets:bool = False, includeAccuracy:bool = False, daysToPlot:int=0):
		r = self.GetTrainingResults(includeTrainingTargets, includeAccuracy)
		if daysToPlot==0: daysToPlot = len(self.X_test)
		r.iloc[-daysToPlot:].plot()
		plt.legend()
		if not filename=='': 
			if not filename[-4] =='.': filename += '.png'
			plt.savefig(self._dataFolderPredictionResults + filename, dpi=600)			
		else:
			plt.show()
		plt.close('all')

	def DisplayDataSample(self):
		print('Source:')
		print(self.sourceDF[:-10])
		print('Target:')
		print(self.targetDF[:-10])
		print('Predictions:')
		print(self.predictionDF[:-10])
		print('X_train:')
		print(self.X_train[:-10])
		print('y_train:')
		print(self.y_train[:-10])
		
	def DisplayModel(self, IncludeDetail:bool=False):
		print('Model')
		print(self.model.summary())
		if IncludeDetail: print(self.model.to_json())
		#print('Model Summary: ', self.modelName)
		#model_vars = tf.trainable_variables()
		#tf.contrib.slim.model_analyzer.analyze_vars(model_vars, print_info=True)

class StockPredictionNN(SeriesPredictionNN): #Price
	predictClasses = False
	def _CustomSourceOperations(self, SourceFieldList:list = None):
		if list(self.sourceDF.columns.values).count('Average') == 0:
			RoundingPrecision = 2
			MaxHigh = self.sourceDF['High'].max()
			if MaxHigh < 5: RoundingPrecision=6
			self.sourceDF['Average'] = round((self.sourceDF['Low'] + self.sourceDF['High'] + self.sourceDF['Open'] + self.sourceDF['Close'])/4, RoundingPrecision)
		if SourceFieldList == None: 
			self.sourceDF = self.sourceDF[['Average']]
		else:
			if SourceFieldList.count('Average') ==0: SourceFieldList.append('Average')
			self.sourceDF = self.sourceDF[SourceFieldList]

	def _CustomTargetOperations(self, prediction_target_days:int=1):
		if self.UseLSTM:
			self.number_of_classes = 1
			self.targetDF['Average'] = self.sourceDF['Average'].copy()
			y = self.targetDF.shift(-prediction_target_days).values	
			y = y.reshape(-1, 1)
		else:
			self.number_of_classes = self.sourceDF.shape[1] #CNN is going to predict an image in the same shape as the original	
			self.targetDF = self.sourceDF.copy() 
			y = self.targetDF.shift(-prediction_target_days).values	
			y = y.reshape(-1,1,self.number_of_classes)
		self.y = y

	def _RecordPredictedValue(self, rowIndex, value):
		if self.UseLSTM:
			self.predictionDF['Average'].iloc[rowIndex] = value
		else:
			self.predictionDF.iloc[rowIndex] = value

	def GetTrainingResults(self, includeTrainingTargets:bool = False, includeAccuracy:bool = False):
		if includeTrainingTargets:
			r = self.targetDF.join(self.predictionDF, how='outer', rsuffix='_Predicted')
			if includeAccuracy:
				if self.targetDF.shape[1] == 1:
					r['PercentageDeviation'] = abs((r['Average']-r['Average_Predicted'])/r['Average'])
				elif self.targetDF.shape[1] == 4:
					r['Average'] = (r['Open'] + r['High'] + r['Low'] + r['Close'])/4
					r['Average_Predicted'] = (r['Open_Predicted'] + r['High_Predicted'] + r['Low_Predicted'] + r['Close_Predicted'])/4
					r['PercentageDeviation'] = abs((r['Average']-r['Average_Predicted'])/r['Average'])
			r = r.reindex(sorted(r.columns), axis=1)
			return r
		else:
			return self.predictionDF

class TradePredictionNN(SeriesPredictionNN): 
	#Given a windows X of states, predict best actions for the next Y days or the best expected value
	predictClasses = True
	def _CustomTargetOperations(self, prediction_target_days:int=0):
		y = self.targetDF.values
		y = keras.utils.to_categorical(y)
		self.number_of_classes = self.targetDF['0'].max() + 1	#Categories 0 to max
		self.y = y

	def _RecordPredictedValue(self, rowIndex, value):
		self.predictionDF['0'].iloc[rowIndex] = int(round(value))
		
