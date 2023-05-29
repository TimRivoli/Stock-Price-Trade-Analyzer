nonGUIEnvironment = False	#hosted environments often have no GUI so matplotlib won't be outputting to display
useTensorBoard = False
useSQL = False
import os, numpy, pandas, matplotlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import keras, tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, MaxPooling1D, Conv1D, LSTM, GRU, Activation, Flatten, MultiHeadAttention, LayerNormalization, Bidirectional, BatchNormalization
from keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from keras.callbacks import EarlyStopping
from pandas.tseries.offsets import BDay
from _classes.PriceTradeAnalyzer import PTADatabase
from _classes.Utility import *
from time import time
if useTensorBoard: from keras.callbacks import TensorBoard
 
#-------------------------------------------- Global settings -----------------------------------------------
nonGUIEnvironment = ReadConfigBool('Settings', 'nonGUIEnvironment')
if nonGUIEnvironment: matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt

def ToSQL(df:pandas.DataFrame, tableName:str, indexAsColumn:bool=True, clearExistingData:bool=False):
		db = PTADatabase()
		if db.Open():
			db.DataFrameToSQL(df=df, tableName=tableName, indexAsColumn=indexAsColumn, clearExistingData=clearExistingData)
			db.Close()

#-------------------------------------------- Classes -----------------------------------------------
class SeriesPredictionNN(object):
	_dataFolderTensorFlowModels = 'data/tfmodels/'
	_dataFolderPredictionResults = 'data/prediction/'
	source_data_loaded = False
	target_data_loaded = False
	batchesCreated = False
	predictClasses = False
	model = None
	use_BatchNormalization = False
	train_start_accuracy = 0
	train_end_accuracy = 0
	train_start_accuracy2 = 0
	train_end_accuracy2 = 0
	train_test_accuracy = 0
	train_start_loss = 0
	train_end_loss = 0

	pcFieldWatch = [] #These are fields which will automatically be converted to percentage change from prior value

	def __init__(self, baseModelName:str='', model_type:str='', PredictionResultsDataFolder:str='', TensorFlowModelsDataFolder:str=''):	
		if not model_type in ['CNN','LSTM','CNN_LSTM','BiLSTM','GRU','ActorCritic','Attention','Simple']: model_type='LSTM'
		self.model_type = model_type
		baseModelName += '_' + model_type
		self.baseModelName = baseModelName
		self.modelName = baseModelName
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
#  ----------------------------------------------------  Model Builds  -----------------------------------------------------------------

	def _InputLayer(self):
		inputs = []	#inputs are treated as individual characteristics, should improve discernment
		for i in range(self.num_features):
			inputs.append(Input(shape=(self.time_steps, 1)))
		return inputs

	def _Build_Simple_Model(self):
		inputs = self._InputLayer()
		inputs_joined = tf.concat(inputs, axis=-1)
		x = Dense(64, activation='relu')(inputs_joined)
		x = Dense(128, activation='relu')(x)
		x = Dense(128, activation='relu')(x)
		x = Dense(128, activation='relu')(x)
		if self.use_BatchNormalization: x = BatchNormalization()(x)
		if self.dropout_rate > 0: x = Dropout(self.dropout_rate)(x)
		x = Dense(128, activation='relu')(x)
		x = Dropout(0.2)(x)
		output = Dense(1)(x)
		model = Model(inputs=inputs, outputs=output)
		model.compile(optimizer=self.optimizer_function, loss=self.loss_function, metrics=self.metrics)
		self.num_layers = len(model.layers) 
		self.model = model

	def _Build_CNN_Model(self):
		inputs = self._InputLayer()
		inputs_joined = tf.concat(inputs, axis=-1)
		#Dilated convolutional 
		if self.time_steps < 30:
			x = Conv1D(filters=64, kernel_size=10, dilation_rate=1, activation='relu')(inputs_joined)
		else:	
			x = Conv1D(filters=64, kernel_size=10, dilation_rate=1, activation='relu')(inputs_joined)
			s = LayerNormalization()(x)
			x = Conv1D(filters=128, kernel_size=8, dilation_rate=2, activation='relu')(x)
			x = Conv1D(filters=256, kernel_size=6, dilation_rate=4, activation='relu')(x)
			x = MaxPooling1D(pool_size=2)(x)
			x = Flatten()(x)
			x = Dense(256, activation='relu')(x)
		if self.use_BatchNormalization: x = BatchNormalization()(x)
		if self.dropout_rate > 0: x = Dropout(self.dropout_rate)(x)
		x = Dense(128, activation='relu')(x)
		output = Dense(self.num_classes, activation='linear')(x)
		model = Model(inputs=inputs, outputs=output)
		model.compile(optimizer=self.optimizer_function, loss=self.loss_function, metrics=self.metrics)
		self.num_layers = len(model.layers) - len(inputs)
		self.model = model

	def _Build_LSTM_Model(self):
		#LSTM should use a time_window, could add a batchnormalization layer, but doesn't need an activation layer because such is done within LSTM
		#hidden_layer_size:int, dropout_rate:float, optimizer:str, learning_rate:float, metrics:list
		lstm_units = 128
		inputs = self._InputLayer()
		x = tf.concat(inputs, axis=-1)
		#x = LSTM(units=lstm_units, return_sequences=True)(x)
		#x = LayerNormalization()(x)
		#x = LSTM(units=lstm_units, return_sequences=True)(x)
		x = LSTM(units=lstm_units, return_sequences=False)(x)
		if self.use_BatchNormalization: x = BatchNormalization()(x)
		if self.dropout_rate > 0: x = Dropout(self.dropout_rate)(x)
		x = Dense(units=64, activation='relu')(x)
		x = Dense(units=21, activation='relu')(x)
		output = Dense(units=self.num_classes, activation=self.output_activation)(x)
		model = Model(inputs=inputs, outputs=output)
		model.compile(optimizer=self.optimizer_function, loss=self.loss_function, metrics=self.metrics)
		self.num_layers = len(model.layers) - len(inputs)
		self.model = model

	def _Build_BiLSTM_Model(self):
		lstm_units = 128
		inputs = self._InputLayer()
		x = Bidirectional(LSTM(units=lstm_units, return_sequences=True))(tf.concat(inputs, axis=-1))
		#x = LayerNormalization()(x)
		#x = Bidirectional(LSTM(units=lstm_units, return_sequences=True))(x)
		x = Bidirectional(LSTM(units=lstm_units, return_sequences=False))(x)
		#x = Flatten()(x)
		if self.use_BatchNormalization: x = BatchNormalization()(x)
		if self.dropout_rate > 0: x = Dropout(self.dropout_rate)(x)
		x = Dense(units=128, activation='relu')(x)
		x = Dense(units=64, activation='relu')(x)
		x = Dense(units=21, activation='relu')(x)
		output = Dense(units=self.num_classes, activation=self.output_activation)(x)
		model = Model(inputs=inputs, outputs=output)
		model.compile(optimizer=self.optimizer_function, loss=self.loss_function, metrics=self.metrics)
		self.num_layers = len(model.layers) - len(inputs)
		self.model = model

	def _Build_CNN_LSTM_Model(self):
		lstm_units = 128
		inputs = self._InputLayer()
		inputs_joined = tf.concat(inputs, axis=-1)
		x = Conv1D(filters=64, kernel_size=10, dilation_rate=1, activation='relu')(inputs_joined)
		s = LayerNormalization()(x)
		x = Conv1D(filters=128, kernel_size=8, dilation_rate=2, activation='relu')(x)
		s = LayerNormalization()(x)
		x = Conv1D(filters=256, kernel_size=6, dilation_rate=4, activation='relu')(x)
		if self.use_BatchNormalization: x = BatchNormalization()(x)
		x = LSTM(units=lstm_units, return_sequences=True)(x)
		x = LSTM(units=lstm_units, return_sequences=False)(x)
		if self.dropout_rate > 0: x = Dropout(self.dropout_rate)(x)
		x = Dense(units=64, activation='relu')(x)
		x = Dense(units=21, activation='relu')(x)
		output = Dense(units=self.num_classes, activation=self.output_activation)(x)
		model = Model(inputs=inputs, outputs=output)
		model.compile(optimizer=self.optimizer_function, loss=self.loss_function, metrics=self.metrics)
		self.num_layers = len(model.layers) - len(inputs)
		self.model = model

	def _Build_GRU_Model(self):
		inputs = self._InputLayer()
		x = GRU(units=64, return_sequences=True)(tf.concat(inputs, axis=-1))
		x = LayerNormalization()(x)
		x = GRU(units=64, return_sequences=True)(x)
		x = LayerNormalization()(x)
		x = GRU(units=64, return_sequences=True)(x)
		x = Flatten()(x)
		if self.dropout_rate > 0: x = Dropout(self.dropout_rate)(x)
		x = Dense(units=128, activation='relu')(x)
		x = Dense(units=64, activation='relu')(x)
		x = Dense(units=32, activation='relu')(x)
		output = Dense(units=self.num_classes, activation=self.output_activation)(x)
		model = Model(inputs=inputs, outputs=output)
		model.compile(optimizer=self.optimizer_function, loss=self.loss_function, metrics=self.metrics)
		self.num_layers = len(model.layers) - len(inputs)
		#model.summary()
		self.model = model

	def _Build_Attention_Model(self):
		inputs = self._InputLayer()
		x = MultiHeadAttention(num_heads=8, key_dim=64)((tf.concat(inputs, axis=-1)), (tf.concat(inputs, axis=-1)), (tf.concat(inputs, axis=-1)))
		x = LayerNormalization(epsilon=1e-6)(x)
		ffn = Dense(units=256, activation='relu')(x)
		if self.dropout_rate > 0: x = Dropout(self.dropout_rate)(ffn)
		ffn = Dense(units=128, activation='relu')(ffn)
		outputs = Dense(units=self.num_classes, activation='linear')(ffn)
		model = Model(inputs=inputs, outputs=outputs)
		model.compile(optimizer=self.optimizer_function, loss=self.loss_function, metrics=self.metrics)
		self.num_layers = len(model.layers) - len(inputs)
		self.model = model

	def _Build_ActorCritic_Model(self):
		inputs = self._InputLayer()
		model = keras.models.Sequential()
		actor = LSTM(units=64, return_sequences=True)(inputs)
		actor = Dense(units=64, activation='relu')(actor)
		actor = Flatten()(actor) #If I don't flatten then the output layers are still windows in the output
		actor = Dense(units=32, activation='relu')(actor)
		actor = Dense(units=self.num_classes, activation=self.output_activation)(actor)
		critic = LSTM(units=64, return_sequences=True)(inputs)
		critic = Dense(units=64, activation='relu')(critic)
		critic = Flatten()(critic)
		critic = Dense(units=32, activation='relu')(critic)
		critic = Dense(units=self.num_classes, activation=self.output_activation)(critic)
		advantage = actor - critic
		critic_with_advantage = Dense(units=1, activation='linear')(advantage)
		model = Model(inputs=inputs, outputs=[actor, critic_with_advantage])
		model.compile(optimizer=self.optimizer_function, loss=self.loss_function, metrics=self.metrics)
		self.num_layers = len(model.layers) - len(inputs)
		self.model = model

	def BuildModel(self, dropout_rate:float=0.01, use_BatchNormalization:bool=False):
		#hidden_layer_size:int=256, dropout_rate:float=0.01, optimizer:str='adam', learning_rate:float=0.00001, metrics:list=['accuracy'], dropout:bool=True
		#0.01 dropout rate is low, could be closer to .2
		#0.000020 learning rate is low, 0.001 is considered deep, current implementations use adaptive
		self.dropout_rate = dropout_rate
		self.use_BatchNormalization = use_BatchNormalization
		if not (self.source_data_loaded):
			print('Source data needs to be loaded before building model.')
			assert(False)
		if not self.batchesCreated: self.MakeTrainTest()
		self.optimizer_function='adam' #adaptive learning rate, starts at 0.001 
		if self.predictClasses:
			self.loss_function = 'categorical_crossentropy'
			self.output_activation = 'softmax'
			self.metrics = ['accuracy']
		else:
			self.loss_function = 'mean_squared_error'
			self.output_activation = 'linear'
			self.metrics = [RootMeanSquaredError()]#, MeanAbsoluteError()
		if self.batchesCreated:
			keras.backend.clear_session()
			if self.model_type == "LSTM":
				self._Build_LSTM_Model() 
			elif self.model_type == "BiLSTM":
				self._Build_BiLSTM_Model()				
			elif self.model_type == "CNN":
				self._Build_CNN_Model()
			elif self.model_type == "GRU":
				self._Build_GRU_Model()			
			elif self.model_type == "ActorCritic":
				self._Build_ActorCritic_Model()
			elif self.model_type == "Attention":
				self._Build_Attention_Model()
			elif self.model_type=='CNN_LSTM':
				self._Build_CNN_LSTM_Model()
			elif self.model_type=='Simple':
				self._Build_Simple_Model()
		
	def DisplayModel(self, IncludeDetail:bool=False):
		print(self.modelName +  'Model')
		if self.model is None: self.BuildModel()
		if self.model is not None: print(self.model.summary())
		if IncludeDetail: print(self.model.to_json())

	def Load(self):
		filename = self._dataFolderTensorFlowModels + self.modelName 
		print('Restoring model from ' + filename + '.h5')
		if FileExists(filename + '.h5'):
			keras.backend.clear_session()
			self.model = keras.models.load_model(filename + '.h5')
			self.model.load_weights(filename + 'weights.h5')	#This is redundant but if verifies the restore
			print('Model restored from disk')
			self.model.summary()
			r = True
		else:
			print('Model backup not found: ', filename + '.ht5')
			r = False
		return r		

	def Save(self):
		if self.model is None:
			print('No model loaded.')
		else:
			filename = self._dataFolderTensorFlowModels + self.modelName
			self.model.save(filename + '.h5')
			self.model.save_weights(filename + 'weights.h5')
			j = self.model.to_json()
			with open(filename + '.json', "w") as json_file:
				json_file.write(j)

	def SavedModelDelete(self):
		r = True
		filename = self._dataFolderTensorFlowModels + self.modelName 
		print('Deleting saved model... ' + filename)
		if FileExists(filename + '.h5'):
			keras.backend.clear_session()
			print('Deleting ' + filename + '.h5')
			self.model = keras.models.load_model(filename + '.h5')
			try:
				os.remove(filename+ '.h5')
				os.remove(filename+ 'weights.h5')
			except:
				r = False
		else:
			print('Model backup not found: ', filename + '.ht5')
		return r

	def SetModelName(self): #used for backups, restores, and logging generated automatically when you load target data
		self.modelName = self.baseModelName + '_win' + str(self.time_steps) + '_days' + str(self.prediction_target_days) + '_feat' + str(self.feature_count) + '_class' + str(self.num_classes)

	def SetModelParams(self, feature_count:int=0, num_classes:int=0, time_steps:int=0, prediction_target_days:int=0):
		#These are normally set when you load data, but if you want to restore a model and do some PredictOne operations then you will need set these parameters before Load
		if (feature_count > 0): self.feature_count = feature_count
		if (num_classes > 0): self.num_classes = num_classes
		if (time_steps > 0): self.time_steps = time_steps
		if (prediction_target_days > 0): self.prediction_target_days = prediction_target_days
		self.SetModelName(time_steps, prediction_target_days, feature_count, num_classes)

#  ----------------------------------------------------  Data preparation  -------------------------------------------------------------------
	def DisplayDataSample(self):
		print('Source:')
		print(self.sourceDF)
		print('Target:')
		print(self.targetDF)
		print('Predictions:')
		print(self.predictionDF[-1:])
		print('input_data:')
		print(self.input_data[:1])
		print('target_data:')
		print(self.target_data[:1])
		print('input_data_train:')
		print(self.input_data_train[:1])
		print('target_data_train:')
		print(self.target_data[:1])
		print('target_data_test:')
		print(self.target_data_test[:1])
	
	def _AddTimeSteps(self, d:numpy.array, time_steps:int):
		# Reshape the array to have timesteps
		d_reshaped = numpy.zeros((len(d) - time_steps + 1, time_steps))
		for i in range(time_steps):
			d_reshaped[:, i] = d[i:len(d) - time_steps + i + 1]
		return d_reshaped

	def _CustomSourceOperations(self):	pass	
		#Placeholder to allow custom operations on source data setup for subclasses

	def LoadSource(self, sourceDF:pandas.DataFrame, field_list:list=None, time_steps:int=10, use_percentage_change:bool=False):
		#X values are set to a window of historical values for each row of sourceDF starting at row time_steps, it has no dates so the row order must match y
		#sourceDF is trimmed to start at time_steps, which is the beginning of X
		self.source_data_loaded = False
		self.use_percentage_change = use_percentage_change
		self.sourceDF = sourceDF.copy()			#Work off a copy just in case it was passed by ref
		self.sourceDF.sort_index(inplace=True)	#Expecting date descending index
		self.sourceDF.fillna(method='bfill', inplace=True)	#TF will start producing Nan results if it encounters Nan values
		self.source_field_list = field_list
		self._CustomSourceOperations()
		if self.source_field_list == None: self.source_field_list = self.sourceDF.columns.values
		for c in self.source_field_list:
			if not c in self.sourceDF.columns.values:
				print('Source field ' + c + ' not in provided dataframe: ', self.sourceDF.columns.values)
				assert(False)
		self.feature_count = len(self.source_field_list)
		self.sourceDF = self.sourceDF[self.source_field_list]
		if self.sourceDF.isnull().values.any(): 
			print('Nan values in source input.  This may break the training.\n')
			assert(False)
		self.time_steps = time_steps

		pcFields = [] #Fields we are going to convert to percentage change
		self.input_data = []
		for c in self.source_field_list:
			if c in self.pcFieldWatch and use_percentage_change:
				pcFields.append(c)
			else:
				x = self.sourceDF[c].values
				x =  self._AddTimeSteps(x, time_steps)
				self.input_data.append(x)
		if len(pcFields) > 0:
			self.sourceDF = self.sourceDF[pcFields].pct_change(1)
			self.sourceDF.fillna(method='bfill', inplace=True) #first row would be Nan
			for c in pcFields:
				x = self.sourceDF[c].values
				x = self._AddTimeSteps(x, time_steps)
				self.input_data.append(x)
		self.num_features = len(self.input_data)
		self.sourceDF.drop(self.sourceDF.index[:time_steps-1],inplace=True) #Forget anything that occurs before the history window.  Data is not part of the training source or target
		self.num_source_days = self.sourceDF.shape[0]
		print('Features in source dataset: {}'.format(self.num_features))
		print('Days in the source dataframe: {}'.format(self.num_source_days))
		print('Timestep Window Size: ', self.time_steps)
		self.source_data_loaded = True
		
	def _CustomTargetOperations(self): pass
		#Placeholder to allow custom operations on target data setup for subclasses

	def LoadTarget(self, targetDF:pandas.DataFrame=None, prediction_target_days:int=1, shift_target_values:bool=True):
		#loads the target data and enforces sanity checks on the provided data for source and target
		#targetDF can be specified or can be derived from the sourceDF, y and y_train/test are derived from it to test accuracy of training
		#actual training data X and y numpy arrays with no dates so they are aligned by row
		#shift_target_values=True, since X and y have no dates, they are aligned by rows, the y=X.shift(prediction_target_days) or y.shift(-prediction_target_days) = X
		#predictionDF is created as empty date index to store prediced values for all values of sourceDF, prediction_target_days are added to the end to hold projections
		self.prediction_target_days = prediction_target_days
		if not self.source_data_loaded:
			print('Load source data before target data.')
			assert(False)
		if targetDF is None:
			self.targetDF = self.sourceDF.copy()
		else:
			self.targetDF = targetDF.copy()
		self._CustomTargetOperations()
		self.targetDF = self.targetDF[self.sourceDF.index.min():self.sourceDF.index.max()]	#trim any entries outside range of source
		if self.targetDF.isnull().values.any(): 
			print('Nan values in target input.  This will break the training.\n')
			assert(False)
		if shift_target_values and prediction_target_days > 0:
			self.targetDF = self.targetDF.shift(-prediction_target_days)
			print('Targets have been pushed ' + str(prediction_target_days) + ' backward for prediction.')

		self.num_classes = self.targetDF.shape[1] 
		self.target_data = self.targetDF.values
		if self.num_classes ==1: self.target_data = numpy.squeeze(self.target_data)
		print('Classes in target values: {}'.format(self.num_classes))
		print('Days in target dataframe: {}'.format(self.targetDF.shape[0]))
		if len(self.input_data)==0 or len(self.target_data) ==0:
			print('SourceDF',  self.sourceDF.index.min(), self.sourceDF.index.max())
			print('TargetDF',  self.targetDF.index.min(), self.targetDF.index.max())
			print('input_data size: ', len(self.input_data))
			print('target_data size: ', len(self.target_data))
			print('Empty data sets given.')
			assert(False)
		if not len(self.input_data[0]) ==  len(self.target_data):
			print('input_data and target_data should have the same number of rows', len(self.input_data[0]), len(self.target_data))
			#print('Missing target dates')
			#print(self.sourceDF.index.difference(self.targetDF.index))
			#print('Missing source dates')
			#print(self.targetDF.index.difference(self.sourceDF.index))
			assert(False)
		#if self.model_type=='CNN' and not self.num_classes == self.num_features and not self.predictClasses:
		#	print('CNN model requires feature count to equal class count.')
		#	assert(False)
		if self.targetDF.shape[1] == 1:
			self.predictionDF = pandas.DataFrame(index=self.targetDF.index) 
		else:
			self.predictionDF = pandas.DataFrame(columns=list(self.targetDF.columns.values), index=self.targetDF.index)
		self.predictionDF = self.predictionDF[prediction_target_days:]	#drop the first n days, not predicted
		d = self.predictionDF.index[-1] 
		for i in range(0,prediction_target_days): 	#Add new days to the end for predictions beyond the date of data provided
			self.predictionDF.loc[d + BDay(i+1), self.targetDF.columns.values[0]] = numpy.nan	
		assert(self.predictionDF.shape[0] == self.targetDF.shape[0] and self.predictionDF.shape[0] == self.sourceDF.shape[0])	#This is key to aligning the results since train data has no dates we use rows
		self.modelFileName = self.SetModelName()
		self.target_data_loaded = True
		
	def MakeTrainTest(self, batch_size=32, train_test_split=.90):
		#splits data sets into training and testing
		if not (self.source_data_loaded):
			print('Source data needs to be loaded before batching.')
			assert(False)
		elif not self.target_data_loaded: 
			print('Target data not specified.  Initializing with copy of source.')
			self.LoadTarget()
		self.batch_size = batch_size
		print('Batching data...')
		days_of_data=len(self.input_data[0])
		train_start_offset = days_of_data % batch_size
		train_test_cuttoff = round((days_of_data // batch_size) * train_test_split) * batch_size + train_start_offset
		self.input_data_train = []
		self.input_data_test = []
		for l in self.input_data: #for each feature, split it on train_test_cuttoff
			self.input_data_train.append(l[:train_test_cuttoff]) #to train_test_cuttoff
			self.input_data_test.append(l[train_test_cuttoff:])   #after train_test_cuttoff			
		self.target_data_train = self.target_data[:train_test_cuttoff]  #to train_test_cuttoff
		self.target_data_test = self.target_data[train_test_cuttoff:]   #after train_test_cuttoff
		self.train_start_date = self.sourceDF.index.min()
		self.test_start_date = self.sourceDF.index[train_test_cuttoff-1]
		self.test_end_date = self.sourceDF.index.max()		
		print('(train, test, end)', self.train_start_date, self.test_start_date, self.test_end_date)
		print('train_test_cuttoff: ', train_test_cuttoff)
		print('(Days, Window size, Features)')
		print('input_data_train size: ', len(self.input_data_train[0]), len(self.input_data_train[0][0]), len(self.input_data_train)) 
		print('input_data_test size: ',  len(self.input_data_test[0]),  len(self.input_data_test[0][0]),  len(self.input_data_test))
		print('(Days, Classes)')
		print('target_data_train size: {}'.format(self.target_data_train.shape))
		print(self.modelName)
		print('\n')
		self.batchesCreated = True

#  ----------------------------------------------------  Training / Prediction  -----------------------------------------------------------------
	def DisplayTrainingSummary(self):
		if self.train_start_accuracy is not None:
			print('Result of training: accuracy ' + str(self.train_start_accuracy) + ' -> ' + str(self.train_end_accuracy) + ' (' + str((self.train_end_accuracy-self.train_start_accuracy)*100) + '%) loss ' + str(self.train_start_loss) + ' -> ' + str(self.train_end_loss))

	def Train(self, epochs=100):
		if self.model is None: self.BuildModel()		
		if self.model is not None: 
			if not self.batchesCreated: self.MakeTrainTest()
			callBacks = [EarlyStopping(monitor='loss', patience=5)]
			if useTensorBoard: 	callBacks.append(TensorBoard(log_dir="data/tensorboard/{}".format(time()), histogram_freq=0, write_graph=True, write_images=True))
			hist = self.model.fit(self.input_data_train, self.target_data_train, batch_size=self.batch_size, epochs=epochs, callbacks=callBacks) 
			epochs_completed = callBacks[0].stopped_epoch
			if epochs_completed == 0: epochs_completed = epochs
			print(epochs_completed, ' epochs completed')
			#print(hist.history.keys())
			k = list(hist.history.keys())
			if 'accuracy' in hist.history.keys():
				self.train_start_accuracy = hist.history['accuracy'][0]
				self.train_end_accuracy = hist.history['accuracy'][-1]
			elif self.model_type=='ActorCritic': #multiple models returned in the metrics, so keys are dynamic
				self.train_start_accuracy = hist.history[k[-2]][0] #'dense_1_root_mean_squared_error'
				self.train_end_accuracy =   hist.history[k[-2]][-1]
				self.train_start_accuracy2 =hist.history[k[-1]][0] #ActorCritic returns results for both actor and critic
				self.train_end_accuracy2 =  hist.history[k[-1]][-1]
			else:
				self.train_start_accuracy = hist.history['root_mean_squared_error'][0]
				self.train_end_accuracy =   hist.history['root_mean_squared_error'][-1]
			self.train_start_loss = hist.history['loss'][0]
			self.train_end_loss = hist.history['loss'][-1]
			if self.model_type=='ActorCritic':
				print('ActorCritic accuracy')
				print(self.model.evaluate(self.input_data_test, self.target_data_test))
			else:
				val_loss, val_accuracy = self.model.evaluate(self.input_data_test, self.target_data_test)
				print('Test accuracy: ' + str(val_accuracy) + ' loss: ' + str(val_loss))
				self.train_test_accuracy = val_accuracy
			self.DisplayTrainingSummary()
			if self.model_type=='ActorCritic': print(' Critic Accuracy ' + str(self.train_start_accuracy2) + ' -> ' + str(self.train_end_accuracy2) + ' (' + str((self.train_end_accuracy2-self.train_start_accuracy2)*100) + '%)')

			TestResults = pandas.DataFrame(columns=list(['ModelName','train_start_accuracy', 'train_end_accuracy', 'train_start_accuracy2', 'train_end_accuracy2', 'train_start_loss', 'train_end_loss', 'num_features','num_classes','num_layers','prediction_target_days','time_steps','num_source_days','batch_size','epochs','dropout_rate','batch_normalization']))
			TestResults.set_index(['ModelName'], inplace=True)	
			TestResults.loc[self.modelName] = [self.train_start_accuracy, self.train_end_accuracy, self.train_start_accuracy2, self.train_end_accuracy2, self.train_start_loss, self.train_end_loss, self.num_features, self.num_classes, self.num_layers, self.prediction_target_days, self.time_steps, self.num_source_days, self.batch_size, epochs_completed, self.dropout_rate, self.use_BatchNormalization]
			TestResults['source_field_list'] = str(self.source_field_list)
			if useSQL:
				try:
					ToSQL(TestResults, 'NNTrainingResults')
				except:
					print('Unable to write training history to SQL')
			filename = self._dataFolderTensorFlowModels + self.modelName + '_trainhist.csv'
			try:
				TestResults.to_csv(filename)
			except:
				print('Unable to write training history to ' + filename)

	def Predict(self, use_full_data_set:bool=False):
		#Populate predictionDF with the models predicted, date is shifted because X value at d is predicted value at d + prediction_target_days
		#This can either be done on the full data set or just the test portion of it
		#Aligning the predicted values with the predictionDF is important.  The source was stripped of dates so it is matched by row onto the date indexed dataframe
		if self.model is None: self.BuildModel()		
		if self.model is not None: 
			if not self.batchesCreated: self.MakeTrainTest()
			print('Running predictions...')
			d = self.input_data_test
			if use_full_data_set: d = self.input_data
			if self.predictClasses:
				predictions = self.model.predict_classes(d)	
			else:
				predictions = self.model.predict(d)
				print(len(predictions))
				if self.model_type=='ActorCritic': predictions = predictions[1] #results are list [actor, advantage]
					#print(predictions[0].shape) #(13303, 160, 1)
					#print(predictions[0][0].shape) #1
					#print(predictions[0][1].shape) #1
					#print(predictions[1].shape) #(13303, 160, 1)
					#print(predictions[1][0].shape) #1
					#print(predictions[1][1].shape) #1
			if len(d[0]) != len(predictions):
				print(predictions.shape)
				print('The number of predictions returned does not match the number of requested values.', len(d[0]), len(predictions)) #10, 13303
				assert(False)
			start_index = len(self.predictionDF) - len(predictions) #getting this number correct is key to aligning predictions with the correct date, predicting to end date of predictionDF so, number of predictions minus the end
			columnName = self.predictionDF.columns[0]
			for i in range(len(predictions)):
				if self.num_classes == 1:
					self.predictionDF[columnName].iloc[start_index + i] = predictions[i][0]
				else:
					self.predictionDF.iloc[start_index + i] = predictions[i]
			print('Predictions complete.')

	def PredictOne(self, X):	
		r = 0
		if self.model is None: 
			print('Model is not ready for predictions')
		else: 
			d = numpy.array([X])
			if self.predictClasses:
				predictions = self.model.predict_classes(d)	
			else:
				predictions = self.model.predict(d)	
				if self.model_type=='ActorCritic': predictions = predictions[0] #results are for actor and critic, picked actor
			r = predictions[0][0]
		return r

	def GetTrainingResults(self, include_target:bool = False, include_accuracy:bool=False, include_input:bool = False):
		if include_input:
			r = self.sourceDF.join(self.predictionDF, how='outer', rsuffix='_Predicted')
			if include_target: r = r.join(self.targetDF, how='outer', rsuffix='_Target')
			if include_accuracy: r['PercentageDeviation'] = abs((r['Average']-r['Average_Predicted'])/r['Average'])
			r = r.reindex(sorted(r.columns), axis=1)
		elif include_target:
			r = self.targetDF.join(self.predictionDF, how='outer', rsuffix='_Predicted')
			if include_accuracy: r['PercentageDeviation'] = abs((r['Average']-r['Average_Predicted'])/r['Average'])
			r = r.reindex(sorted(r.columns), axis=1)

		else:
			r = self.predictionDF
		return r.copy()

	def PredictionResultsSave(self, filename:str, include_target:bool = False, include_accuracy:bool = False, include_input:bool = False):
		r = self.GetTrainingResults(include_target=include_target, include_accuracy=include_accuracy, include_input=include_input)
		if not filename[-4] =='.': filename += '.csv'
		print('Saving predictions to', self._dataFolderPredictionResults + filename)
		r.to_csv(self._dataFolderPredictionResults + filename)

	def PredictionResultsPlot(self, filename:str='', include_target:bool = False, include_accuracy:bool = False, days_to_plot:int=0):
		r = self.GetTrainingResults(include_target=include_target, include_accuracy=include_accuracy)
		if days_to_plot==0: days_to_plot = len(self.input_data_test) 
		if len(r) > days_to_plot + 90: days_to_plot+=90
		r.iloc[-days_to_plot:].plot()
		plt.legend()
		if not filename=='': 
			if not filename[-4] =='.': filename += '.png'
			plt.savefig(self._dataFolderPredictionResults + filename, dpi=600)			
		else:
			plt.show()
		plt.close('all')


class StockPredictionNN(SeriesPredictionNN): #Series prediction: Predicts future prices, input price data, estimated future price
	predictClasses = False
	pcFieldWatch = ['Average','shortEMA','longEMA'] #These are fields which will automatically be converted to percentage change from prior value
	
	def _CustomSourceOperations(self):
		if not 'Average' in self.sourceDF.columns:
			#RoundingPrecision = 2
			#MaxHigh = self.sourceDF['High'].max()
			#if MaxHigh < 5: RoundingPrecision=6
			self.sourceDF['Average'] = (self.sourceDF['Low'] + self.sourceDF['High'])/2
		if self.source_field_list == None: self.source_field_list = ['Average']
		if self.source_field_list.count('Average') == 0: self.source_field_list.append('Average')

	def _CustomTargetOperations(self):
		if 'Average' in self.targetDF.columns:
			self.targetDF = self.targetDF[['Average']]
		elif 'High' in self.targetDF.columns and 'Low' in self.targetDF.columns:
			self.targetDF['Average'] = (self.targetDF['High'] + self.targetDF['Low'])/2
			self.targetDF = self.targetDF[['Average']]

class TradePredictionNN(SeriesPredictionNN): #Categorical: Predicts best trade actions, input price data, output best action from list of actions
	#Given a windows X of states, predict best actions for the next Y days or the best expected value
	#_defaultTargetDays = 0
	predictClasses = True
	
	def _CustomTargetOperations(self):
		y = self.targetDF.values
		self.num_classes = 7
		y = keras.utils.to_categorical(y, num_classes=self.num_classes)
		print('y', len(y))
		#self.num_classes = self.targetDF['actionID'].max() + 1	#Categories 0 to max
		if self.model_type=='CNN':
			y = y.reshape(-1,1,self.num_classes)
		self.target_data = y

	def _PredictedValueRecord(self, rowIndex, value):
		if self.model_type=='CNN':
			self.predictionDF['actionID'].iloc[rowIndex] = value[0]
		else:
			self.predictionDF['actionID'].iloc[rowIndex] = int(round(value))
		
class StockPickerNN(SeriesPredictionNN): #Series prediction: Predicts future price gain, input current price gain, estimate future gain
	predictClasses = False

	def _CustomTargetOperations(self):
		self.num_classes = self.num_features	#Number of tickers
		y = self.targetDF.values
		y.fillna(method='ffill', inplace=True)
		y = y.values	
		if self.predictClasses: y = keras.utils.to_categorical(y, num_classes=self.num_classes)
		self.target_data = y
