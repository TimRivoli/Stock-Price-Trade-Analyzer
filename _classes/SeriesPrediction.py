#LSTM Adapted from Luka Anicin's https://github.com/lucko515/tesla-stocks-prediction
#CNN adapted from https://nicholastsmith.wordpress.com/ currency estimator and using his TFANN class
nonGUIEnvironment = False	#hosted environments often have no GUI so matplotlib won't be outputting to display
import os, numpy, pandas
from pandas.tseries.offsets import BDay
import matplotlib
if nonGUIEnvironment: matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt
import tensorflow as tf
from TFANN import ANNR

def CreateFolder(p:str):
	r = True
	if not os.path.exists(p):
		try:
			os.mkdir(p)	
		except Exception as e:
			print('Unable to create folder: ' + p)
			f = False
	return r

class StockPredictionNN(object): #aka CrystalBall
	_dataFolderTensorFlowModels = 'data/tfmodels/'
	_dataFolderPredictionResults = 'data/prediction/'

	def __init__(self, dataFolderRoot:str=''):
		if not dataFolderRoot =='':
			if CreateFolder(dataFolderRoot):
				if not dataFolderRoot[-1] =='/': dataFolderRoot += '/'
				self._dataFolderTensorFlowModels = dataFolderRoot + 'tfmodels/'
				self._dataFolderPredictionResults = dataFolderRoot + 'prediction/'
		else: CreateFolder('data')
		CreateFolder(self._dataFolderTensorFlowModels)
		CreateFolder(self._dataFolderPredictionResults)
		
	def LSTM_cell(self, hidden_layer_size, batch_size, number_of_layers, dropout=True, dropout_rate=0.8):   
		layer = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size)
		if dropout:
			layer = tf.contrib.rnn.DropoutWrapper(layer, output_keep_prob=dropout_rate)
		cell = tf.contrib.rnn.MultiRNNCell([layer] * number_of_layers)   
		init_state = cell.zero_state(batch_size, tf.float32)
		return cell, init_state

	def LSTM_output_layer(self, lstm_output, in_size, out_size):   
		x = lstm_output[:, -1, :]
		weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.05), name='output_layer_weights')
		bias = tf.Variable(tf.zeros([out_size]), name='output_layer_bias')
		output = tf.matmul(x, weights) + bias
		return output

	def LSTM_opt_loss(self, logits, targets, learning_rate, grad_clip_margin):   
		losses = []
		for i in range(targets.get_shape()[0]):
			losses.append([(tf.pow(logits[i] - targets[i], 2))])	   
		loss = tf.reduce_sum(losses)/(2 * self.batch_size)
		gradients = tf.gradients(loss, tf.trainable_variables())
		clipper_, _ = tf.clip_by_global_norm(gradients, grad_clip_margin)
		optimizer = tf.train.AdamOptimizer(learning_rate)
		train_optimizer = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
		return loss, train_optimizer

	def LoadData(self, sourceDF:pandas.DataFrame, window_size:int=7, prediction_target_days:int=5, UseLSTM:bool=True, SourceFieldList:list=None, batch_size=32, train_test_split=.90):
		#Prepares source DataFrames calculates features, shapes, batches, etc.  Pass it what you want to process.

		self.window_size = window_size
		sourceDF.sort_index(inplace=True)				#Expecting date descending index
		sourceDF.fillna(method='bfill', inplace=True)	#TF will start producing Nan results if it encounters Nan values
		RoundingPrecision = 2
		MaxHigh = sourceDF['High'].max()
		if MaxHigh < 5: RoundingPrecision=6
		self.targetDF = pandas.DataFrame(index=sourceDF.index)
		if SourceFieldList == None: 
			sourceDF['Average'] = round((sourceDF['Low'] + sourceDF['High'] + sourceDF['Open'] + sourceDF['Close'])/4, RoundingPrecision)
			sourceDF = sourceDF[['Average']]			
			self.targetDF['Average'] = sourceDF['Average'] #I don't need to shift the target dates because I am comparing on index not the date
		else:
			if UseLSTM:
				self.targetDF['Average'] = round((sourceDF['Low'] + sourceDF['High'] + sourceDF['Open'] + sourceDF['Close'])/4, RoundingPrecision)
			else:
				sourceDF = sourceDF[SourceFieldList].copy()
				self.targetDF = sourceDF.copy() #CNN is going to predict an image in the same shape as the original
		#last prediction_target_days of data will be nan, that should be OK since they won't be used for testing.

		self.daysInDataSet = sourceDF.shape[0]
		self.number_of_features =  sourceDF.shape[1]
		self.number_of_classes = self.targetDF.shape[1]
		self.modelName = 'LSTM'
		if not UseLSTM:	self.modelName = 'CNN'
		self.modelName += '_win' + str(window_size) + '_days' + str(prediction_target_days) + '_feat' + str(self.number_of_features) + '_class' + str(self.number_of_classes)
		print('Model name: ', self.modelName)
		print('Features in source dataset: {}'.format(self.number_of_features))
		print('Classes in target values: {}'.format(self.number_of_classes))
		print('Days in the source features data set: {}'.format(self.daysInDataSet))
		print('Days in target value data: {}'.format(self.targetDF.shape[0]))
		X = [] #create dimension for window of past days, 0 position is most recent
		i = sourceDF.shape[0]-1
		while i >= window_size:
			v = sourceDF.iloc[i-window_size:i].values
			v = v[::-1] #flip it so that 0 is most recent
			X.insert(0,v)
			i -= 1
		if UseLSTM:
			y = self.targetDF.iloc[-len(X):].values
			y = y.reshape(-1, self.number_of_classes)
		else:
			y = [] #create dimension for window of past days
			i = self.targetDF.shape[0]-1
			while i >= window_size:
				v = self.targetDF.iloc[i-prediction_target_days:i].values
				v = v[::-1] #flip it so that 0 is most recent
				y.insert(0,v)
				i -= 1	
		print('X shape: ', len(X))
		print('y shape: ', len(y))
		self.X= X
		self.y = y
		if not len(X) ==  len(y):
			print(self.targetDF)
			assert len(X) ==  len(y)
		self.targetDF.drop(self.targetDF.index[:self.window_size], inplace=True) #X,y don't have the first Window_Size values, targetDF and predictionDF shouldn't either

		if self.targetDF.shape[1] ==1:
			self.predictionDF = pandas.DataFrame(index=self.targetDF.index) # + BDay(prediction_target_days)
		else:
			self.predictionDF = pandas.DataFrame(columns=list(self.targetDF.columns.values), index=self.targetDF.index)
		self.predictionDF = self.predictionDF[prediction_target_days:]	#drop the first n days, not predicted
		d = self.predictionDF.index[-1] 
		for i in range(0,prediction_target_days): 	#Add new days to the end for crystal ball predictions
			self.predictionDF.loc[d + BDay(i+1), self.targetDF.columns.values[0]] = numpy.nan	
		assert(self.predictionDF.shape[0] == self.targetDF.shape[0])

		self.batch_size = batch_size
		daysOfData=len(self.X)
		self.train_start_offset = daysOfData % batch_size
		train_test_cuttoff = round((daysOfData // batch_size) * train_test_split) * batch_size + batch_size + self.train_start_offset
		self.X_train  = numpy.array(self.X[:train_test_cuttoff]) #to train_test_cuttoff
		self.y_train = numpy.array(self.y[:train_test_cuttoff])  #to train_test_cuttoff
		self.X_test = numpy.array(self.X[train_test_cuttoff:])   #after train_test_cuttoff
		#self.y_test = numpy.array(self.y[train_test_cuttoff:])   #after train_test_cuttoff, this is never used
		print('train_start_offset: ', self.train_start_offset)
		print('train_test_cuttoff: ', train_test_cuttoff)
		print('(Days, Window size, Features)')
		print('X_train size: {}'.format(self.X_train.shape)) 
		print('X_test size: {}'.format(self.X_test.shape))   
		print('(Days, ForcastedDays, Classes)')
		print('y_train size: {}'.format(self.y_train.shape))
		print('\n')

	def TrainCNN(self, epochs=100, learning_rate=2e-5, saveResults:bool = True):
		#creates the model, trains it for the given number of epochs
		print('Training CNN...')
		#2 1-D conv layers with relu followed by 1-d conv output layer
		networkArchitecture = [('C1d', [8, self.number_of_features, self.number_of_features * 2], 4), ('AF', 'relu'), 
			  ('C1d', [8, self.number_of_features * 2, self.number_of_features * 2], 2), ('AF', 'relu'), 
			  ('C1d', [8, self.number_of_features * 2, self.number_of_features], 2)]
		cnnr = ANNR(self.X_train[0].shape, networkArchitecture, batchSize=self.batch_size, learnRate = learning_rate, maxIter = epochs, reg = 1e-5, tol = 1e-5, verbose = True)
		cnnr.fit(self.X_train, self.y_train)

		#print('Saving model...')
		#saver = tf.train.Saver()
		#saver.save(cnnr.GetSes(), self._dataFolderTensorFlowModels + self.modelName)
		#builder = tf.saved_model.builder.SavedModelBuilder(self._dataFolderTensorFlowModels  + self.modelName)
		#builder.add_meta_graph_and_variables(cnnr.GetSes(), [tf.saved_model.tag_constants.TRAINING], signature_def_map=None, assets_collection=None)
		#builder.save()  	
		self.PredictCNN(epochs, learning_rate, cnnr)
	
	def PredictCNN(self, epochs=100, learning_rate=2e-5, cnnr:ANNR = None):
		#The accuracy of this is zero
		if cnnr ==None:
			print('Restoring model...')
			networkArchitecture = [('C1d', [8, self.number_of_features, self.number_of_features * 2], 4), ('AF', 'relu'), 
				  ('C1d', [8, self.number_of_features * 2, self.number_of_features * 2], 2), ('AF', 'relu'), 
				  ('C1d', [8, self.number_of_features * 2, self.number_of_features], 2)]
			cnnr = ANNR(self.X_train[0].shape, networkArchitecture, batchSize=self.batch_size, learnRate = learning_rate, maxIter = epochs, reg = 1e-5, tol = 1e-5, verbose = True)
			saver = tf.train.import_meta_graph(self._dataFolderTensorFlowModels + self.modelName +'.meta')
			saver.restore(cnnr.GetSes(), self._dataFolderTensorFlowModels + self.modelName)
			#tf.saved_model.loader.load(cnnr.GetSes(), [tf.saved_model.tag_constants.TRAINING], self._dataFolderTensorFlowModels + self.modelName)
		
		print('Predicting...')
		X = numpy.array(self.X)
		numberOfPredictions = len(X) 
		assert(len(X)==len(self.predictionDF))
		for i in range(0,numberOfPredictions):	
			P = X[[i]]	
			YH = cnnr.predict(P)
			for ii in range(self.predictionDF.shape[1]):
				self.predictionDF.iloc[i, ii] = YH[0,0,ii]
		print('Training session closed')

	def TrainLSTM(self, epochs=100, learning_rate=0.001, dropout_rate=0.8, gradient_clip_margin=4):
		#creates the model, trains it for the given number of epochs	
		hidden_layer_size=512 #512
		number_of_layers=1 #1
		dropout=True
		tf.reset_default_graph()
		self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.window_size, self.number_of_features], name='input_data')
		self.targets = tf.placeholder(tf.float32, [self.batch_size, self.number_of_classes], name='targets')
		cell, init_state = self.LSTM_cell(hidden_layer_size, self.batch_size, number_of_layers, dropout, dropout_rate)
		outputs, states = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=init_state)
		self.logits = self.LSTM_output_layer(outputs, hidden_layer_size, self.number_of_classes)
		self.loss, self.opt = self.LSTM_opt_loss(self.logits, self.targets, learning_rate, gradient_clip_margin)
		print('LSTM Cell: ', hidden_layer_size, ' hidden layers of ', hidden_layer_size, ' size, ', number_of_layers, ' layers')

		print('Training LSTM...')
		session =  tf.Session()
		session.run(tf.global_variables_initializer())
		for i in range(epochs):
			ii = self.train_start_offset
			epoch_loss = []
			while(ii + self.batch_size) <= len(self.X_train):
				X_batch = self.X_train[ii:ii+self.batch_size]
				y_batch = self.y_train[ii:ii+self.batch_size]	   
				o, c, _ = session.run([self.logits, self.loss, self.opt], feed_dict={self.inputs:X_batch, self.targets:y_batch})
				epoch_loss.append(c)
				for iii in range(self.batch_size):
					if numpy.isnan(o[iii]).any():
						print('x', X_batch)
						print('y', y_batch)
						print('o', o[iii])
						print('NaN values in prediction.  TF has given up')
						assert(False)
					rowIndex = self.predictionDF.index[[ii + iii ]]	
					self.predictionDF.loc[rowIndex,'Average'] = o[iii]
				ii += self.batch_size
			if (i % 10) == 0:
				print('Epoch {}/{}'.format(i, epochs), ' Current loss: {}'.format(numpy.mean(epoch_loss)))

		saver = tf.train.Saver()
		saver.save(session, self._dataFolderTensorFlowModels + self.modelName)
		self.PredictLSTM(epochs=epochs, learning_rate=learning_rate, dropout_rate=dropout_rate, gradient_clip_margin=gradient_clip_margin, session=session)

	def PredictLSTM(self, epochs=100, learning_rate=0.001, dropout_rate=0.8, gradient_clip_margin=4, session = None):
		if session == None:
			hidden_layer_size=512 #512
			number_of_layers=1 #1
			dropout=True
			tf.reset_default_graph()
			self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.window_size, self.number_of_features], name='input_data')
			self.targets = tf.placeholder(tf.float32, [self.batch_size, self.number_of_classes], name='targets')
			cell, init_state = self.LSTM_cell(hidden_layer_size, self.batch_size, number_of_layers, dropout, dropout_rate)
			outputs, states = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=init_state)
			self.logits = self.LSTM_output_layer(outputs, hidden_layer_size, self.number_of_classes)
			self.loss, self.opt = self.LSTM_opt_loss(self.logits, self.targets, learning_rate, gradient_clip_margin)
			print('LSTM Cell: ', hidden_layer_size, ' hidden layers of ', hidden_layer_size, ' size, ', number_of_layers, ' layers')
			print('Restoring model...')
			session =  tf.Session()
			session.run(tf.global_variables_initializer())
			saver = tf.train.import_meta_graph(self._dataFolderTensorFlowModels + self.modelName +'.meta')
			saver.restore(session, self._dataFolderTensorFlowModels + self.modelName)
			
		print('Predicting ...')
		i = 0
		while i + self.batch_size <= len(self.X_test):   
			o = session.run([self.logits], feed_dict={self.inputs:self.X_test[i:i+self.batch_size]})
			for iii in range(self.batch_size):
				rowIndex = self.predictionDF.index[[self.X_train.shape[0] + i + iii]] #+ self.window_sizefirst window_size values are not in training data
				self.predictionDF.loc[rowIndex,'Average'] = o[0][iii]
			i += self.batch_size
		session.close()
		print('Training session closed')

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

	def PredictionResultsSave(self, filename:str, includeTrainingTargets:bool = False, includeAccuracy:bool = False):
		r = self.GetTrainingResults(includeTrainingTargets, includeAccuracy)
		if not filename[-4] =='.': filename += '.csv'
		r.to_csv(self._dataFolderPredictionResults + filename)

	def PredictionResultsPlot(self, filename:str='', includeTrainingTargets:bool = False, includeAccuracy:bool = False):
		r = self.GetTrainingResults(includeTrainingTargets, includeAccuracy)
		r.plot()
		plt.legend()
		if not filename=='': 
			if not filename[-4] =='.': filename += '.png'
			plt.savefig(self._dataFolderPredictionResults + filename, dpi=600)			
		else:
			plt.show()
		plt.close('all')

	def RestoreModel(self, modelName:str): 
		try:
			saver = tf.train.Saver()
			saver.restore(self.GetSes(), p + n)
		except Exception as e:
			print('Error restoring: ' + p + n)
			return False
		return True

	def SaveModel(self, modelName:str):
		saver = tf.train.Saver()
		saver.save(self.GetSes(), p)
		return False	


