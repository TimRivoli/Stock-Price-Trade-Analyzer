useWebProxyServer = False	#If you need a web proxy to browse the web
nonGUIEnvironment = False	#hosted environments often have no GUI so matplotlib won't be outputting to display

import time, datetime, random, os, ssl
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
import urllib.request as webRequest
import matplotlib
if nonGUIEnvironment: matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from _classes.SeriesPrediction import StockPredictionNN
#pip install any for these if they are missing

#pricingData and TradingModel are the two intended exportable classes
#user input dates are expected to be in local format, after that they should be in database format

#-------------------------------------------- General Utilities -----------------------------------------------
#datetime.datetime.fromtimestamp(dt).date()
def GetMyDateFormat(): return '%m/%d/%Y'

def DateFormatDatabase(givenDate:datetime):
#returns datetime object
	if type(givenDate) == str:
		if givenDate.find('-') > 0 :
			r = datetime.datetime.strptime(givenDate, '%Y-%m-%d')
		else:
			r = datetime.datetime.strptime(givenDate, GetMyDateFormat())
	elif type(givenDate) == datetime:
		r = datetime.datetime.fromtimestamp(givenDate).date()
	else:
		r = givenDate
	return r

def GetDateTimeStamp():
	d = datetime.datetime.now()
	return d.strftime('%Y%m%d%H%M')

def DateDiffDays(startDate:datetime, endDate:datetime):
	delta = endDate-startDate
	return delta.days

def CreateFolder(p:str):
	r = True
	if not os.path.exists(p):
		try:
			os.mkdir(p)	
		except Exception as e:
			print('Unable to create folder: ' + p)
			f = False
	return r
	
def PlotInitDefaults():
	#'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'
	params = {'legend.fontsize': 'xx-small', 'axes.labelsize': 'xx-small','axes.titlesize':'xx-small','xtick.labelsize':'xx-small','ytick.labelsize':'xx-small'}
	plt.rcParams.update(params)

def PlotScalerDateAdjust(minDate:datetime, maxDate:datetime, ax):
	if type(minDate)==str:
		daysInGraph = DateDiffDays(minDate,maxDate)
	else:
		daysInGraph = (maxDate-minDate).days
	if daysInGraph >= 365*3:
		majorlocator =  mdates.YearLocator()
		minorLocator = mdates.MonthLocator()
		majorFormatter = mdates.DateFormatter('%m/%d/%Y')
	elif daysInGraph >= 365:
		majorlocator =  mdates.MonthLocator()
		minorLocator = mdates.WeekdayLocator()
		majorFormatter = mdates.DateFormatter('%m/%d/%Y')
	elif daysInGraph < 90:
		majorlocator =  mdates.DayLocator()
		minorLocator = mdates.DayLocator()
		majorFormatter =  mdates.DateFormatter('%m/%d/%Y')
	else:
		majorlocator =  mdates.WeekdayLocator()
		minorLocator = mdates.DayLocator()
		majorFormatter =  mdates.DateFormatter('%m/%d/%Y')
	ax.xaxis.set_major_locator(majorlocator)
	ax.xaxis.set_major_formatter(majorFormatter)
	ax.xaxis.set_minor_locator(minorLocator)
	#ax.xaxis.set_minor_formatter(daysFmt)
	ax.set_xlim(minDate, maxDate)

def PlotDataFrame(df:pd.DataFrame, title:str, xlabel:str, ylabel:str, adjustScale:bool=True, fileName:str = '', dpi:int = 600):
	if df.shape[0] >= 4:
		PlotInitDefaults()
		ax=df.plot(title=title, linewidth=.75)
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.tick_params(axis='x', rotation=70)
		ax.grid(b=True, which='both', color='0.65', linestyle='-')
		if adjustScale:
			dates= df.index.get_level_values('Date')
			minDate = dates.min()
			maxDate = dates.max()
			PlotScalerDateAdjust(minDate, maxDate, ax)
		if not fileName =='':
			if not fileName[-4] == '.': fileName+= '.png'
			plt.savefig(fileName, dpi=dpi)			
		else:
			fig = plt.figure(1)
			fig.canvas.set_window_title(title)
			plt.show()
		plt.close('all')

currentProxyServer = None
def GetProxiedOpener():
	testURL = 'https://stooq.com'
	proxyList =['173.192.128.238:9999','161.202.226.194:25','173.192.21.89:8080','208.69.113.165:80','161.202.226.194:8123','173.192.21.89:8080','173.192.128.238:8123','45.6.216.66:8080','173.192.128.238:9999','192.210.170.200:1080','47.75.0.253:8081','74.207.254.183:80','173.249.15.107:3128','71.191.75.67:3128','47.88.20.189:80','168.128.29.75:80','52.43.233.218:80']
	userName, password = 'Bill', 'test'
	context = ssl._create_unverified_context()
	handler = webRequest.HTTPSHandler(context=context)
	i = -1
	functioning = False
	global currentProxyServer
	while not functioning and i < len(proxyList):
		if i >=0 or currentProxyServer==None: currentProxyServer = proxyList[i]
		proxy = webRequest.ProxyHandler({'https': r'http://' + userName + ':' + password + '@' + currentProxyServer})
		auth = webRequest.HTTPBasicAuthHandler()
		opener = webRequest.build_opener(proxy, auth, handler) 
		opener.addheaders = [('User-agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.1 Safari/603.1.30')]
		try:
			conn = opener.open(testURL)
			print('Proxy ' + currentProxyServer + ' is functioning')
			functioning = True
		except:
			print('Proxy ' + currentProxyServer + ' is not responding')
		i+=1
	return opener

#-------------------------------------------- Classes -----------------------------------------------
class PlotHelper:
	def PlotDataFrame(self, df:pd.DataFrame, title:str='', xlabel:str='', ylabel:str='', adjustScale:bool=True, fileName:str = '', dpi:int=600): PlotDataFrame(df, title, xlabel, ylabel, adjustScale, fileName, dpi)

	def PlotDataFrameDateRange(self, df:pd.DataFrame, endDate:datetime=None, historyDays:int=90, title:str='', xlabel:str='', ylabel:str='', fileName:str = '', dpi:int=600):
		if df.shape[0] > 10: 
			if endDate==None: endDate=df.index[-1] 	#The latest date in the dataframe assuming ascending order				
			endDate = DateFormatDatabase(endDate)
			startDate = endDate - BDay(historyDays)
			df = df[df.index >= startDate]
			df = df[df.index <= endDate]
			PlotDataFrame(df, title, xlabel, ylabel, True, fileName, dpi)
	
class PriceSnapshot:
	high=0
	low=0
	open=0
	close=0
	oneDayAverage=0
	twoDayAverage=0
	shortEMA=0
	shortEMASlope=0
	longEMA=0
	longEMASlope=0
	channelHigh=0
	channelLow=0
	oneDayDeviation=0
	fiveDayDeviation=0
	fifteenDayDeviation=0
	nextDayTarget=0
	estHigh=0
	estLow=0
	snapshotDate=None
	
class PricingData:
	#Historical prices for a given stock, along with statistics, and future estimated prices
	stockTicker = ''
	historicalPrices = None	#dataframe with price history indexed on date
	pricePredictions = None #dataframe with price predictions indexed on date
	historyStartDate = None
	historyEndDate = None
	pricesLoaded = False
	statsLoaded = False
	predictionsLoaded = False
	predictionDeviation = 0	#Average percentage off from target
	pricesNormalized = False
	pricesInPercentages = False
	_dataFolderCurrentPrices = 'data/current/'
	_dataFolderhistoricalPrices = 'data/historical/'
	_dataFolderCharts = 'data/charts/'
	_dataFolderDailyPicks = 'data/dailypicks/'
	
	def __init__(self, ticker:str, dataFolderRoot:str=''):
		self.stockTicker = ticker
		if not dataFolderRoot =='':
			if CreateFolder(dataFolderRoot):
				if not dataFolderRoot[-1] =='/': dataFolderRoot += '/'
				self._dataFolderCharts = dataFolderRoot + 'charts/'
				self._dataFolderCurrentPrices = dataFolderRoot + 'current/'
				self._dataFolderhistoricalPrices = dataFolderRoot + 'historical/'
				self._dataFolderDailyPicks = dataFolderRoot + 'dailypicks/'
		else: CreateFolder('data')
		CreateFolder(self._dataFolderCurrentPrices)
		CreateFolder(self._dataFolderhistoricalPrices)
		CreateFolder(self._dataFolderCharts)
		CreateFolder(self._dataFolderDailyPicks)
		self.pricesLoaded = False
		self.statsLoaded = False
		self.predictionsLoaded = False
		self.historicalPrices = None	
		self.pricePredictions = None
		self.historyStartDate = None
		self.historyEndDate = None

	def __del__(self):
		self.pricesLoaded = False
		self.statsLoaded = False
		self.predictionsLoaded = False
		self.historicalPrices = None	
		self.pricePredictions = None
		self.historyStartDate = None
		self.historyEndDate = None

	def PrintStatus(self):
		print("pricesLoaded=" + str(self.pricesLoaded))
		print("statsLoaded=" + str(self.statsLoaded))
		print("historyStartDate=" + str(self.historyStartDate))
		print("historyEndDate=" + str(self.historyEndDate))
		
	def DownloadPriceData(self,daysToGoBack:int=0):
		url = "https://stooq.com/q/d/l/?i=d&s=" + self.stockTicker + '.us'
		if self.stockTicker.upper() =='^SPX': url = "https://stooq.com/q/d/l/?i=d&s=" + self.stockTicker 
		filePath = self._dataFolderhistoricalPrices + self.stockTicker + '.csv'
		d = datetime.datetime.now()
		s1 = ''
		EndDate = d.strftime(GetMyDateFormat())
		if daysToGoBack>0:
			d=d+datetime.timedelta(days=-daysToGoBack)
			StartDate = d.strftime(GetMyDateFormat())
			Interval = "&d1=" + StartDate +"&d2=" + EndDate
			url = url + Interval
			if CreateFolder(self._dataFolderCurrentPrices): filePath = self._dataFolderCurrentPrices + '/'+ self.stockTicker + '.csv'
		try:
			if useWebProxyServer:
				opener = GetProxiedOpener()
				opener.addheaders = [('User-agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.1 Safari/603.1.30')]
				openUrl = opener.open(url)
			else:
				openUrl = webRequest.urlopen(url) 
			r = openUrl.read()
			openUrl.close()
			s1=r.decode()
			s1 = s1.replace(chr(13),'')
		except:
			print('Web connection error.')
		if len(s1) < 1024:
			print('No data found online for ticker ' + self.stockTicker)
		else:
			print('Downloaded new data for ticker ' + self.stockTicker)
			f = open(filePath,'w')
			f.write(s1)
			f.close()

	def LoadHistory(self, refreshIfOld:bool=False, dropVolume:bool = True):
		filePath = self._dataFolderhistoricalPrices + self.stockTicker + '.csv'
		if not os.path.isfile(filePath):
			self.DownloadPriceData(0)
		elif refreshIfOld:
			x = round((int(time.time()) - round(os.path.getmtime(filePath))) / 60, 2) # how many minutes old it is
			if x > 720: self.DownloadPriceData(0)		

		if not os.path.isfile(filePath):
			print('No data found for ' + self.stockTicker)
			self.pricesLoaded = False
		else:
			df = pd.read_csv(filePath, index_col=0, parse_dates=True, na_values=['nan'])
			if dropVolume: df = df[['Open','Close','High','Low']]
			self.historicalPrices = df
			self.historicalPrices['Average'] = self.historicalPrices.loc[:,['Open','Close','High','Low']].mean(axis=1) #select those rows, calculate the mean value
			self.historyStartDate = self.historicalPrices.index.min()
			self.historyEndDate = self.historicalPrices.index.max()
			self.pricesLoaded = True
		return self.pricesLoaded 
		
	def TrimToDateRange(self,startDate:datetime, endDate:datetime):
		startDate = DateFormatDatabase(startDate)
		endDate = DateFormatDatabase(endDate)
		self.historicalPrices = self.historicalPrices[self.historicalPrices.index >= startDate]
		self.historicalPrices = self.historicalPrices[self.historicalPrices.index <= endDate]
		self.historyStartDate = self.historicalPrices.index.min()
		self.historyEndDate = self.historicalPrices.index.max()
	
	def ConvertToPercentages(self):
		if self.pricesInPercentages:
			self.historicalPrices.iloc[0] = self.CTPFactor
			for i in range(1, self.historicalPrices.shape[0]):
				self.historicalPrices.iloc[i] = (1 + self.historicalPrices.iloc[i]) * self.historicalPrices.iloc[i-1]
			if self.predictionsLoaded:
				self.pricePredictions.iloc[0] = self.CTPFactor['Average']
				for i in range(1, self.pricePredictions.shape[0]):
					self.pricePredictions.iloc[i] = (1 + self.pricePredictions.iloc[i]) * self.pricePredictions.iloc[i-1]
			self.pricesInPercentages = False
		else:
			self.CTPFactor = self.historicalPrices.iloc[0]
			self.historicalPrices = self.historicalPrices[['Open','High','Low','Close','Average']].pct_change(1)
			#for i in range(self.historicalPrices.shape[0]-1, 0, -1):
			#	self.historicalPrices.iloc[i] = (self.historicalPrices.iloc[i] / self.historicalPrices.iloc[i-1])-1
			if self.predictionsLoaded:
				self.pricePredictions = self.pricePredictions.pct_change(1)
			self.statsLoaded = False
			self.pricesInPercentages = True
		
	def NormalizePrices(self):
		#(x-min(x))/(max(x)-min(x))
		if not self.pricesNormalized:
			low = self.historicalPrices['Low'].min(axis=0)
			high = self.historicalPrices['High'].max(axis=0)
			diff = high-low
			self.historicalPrices['Open'] = (self.historicalPrices['Open']-low)/diff
			self.historicalPrices['Close'] = (self.historicalPrices['Close']-low)/diff
			self.historicalPrices['High'] = (self.historicalPrices['High']-low)/diff
			self.historicalPrices['Low'] = (self.historicalPrices['Low']-low)/diff
			self.historicalPrices['Average'] = self.historicalPrices.loc[:,['Open','Close','High','Low']].mean(axis=1) #select those rows, calculate the mean value
			if self.predictionsLoaded:
				self.pricePredictions['estLow'] = (self.pricePredictions['estLow']-low)/diff
				self.pricePredictions['estAverage'] = (self.pricePredictions['estAverage']-low)/diff
				self.pricePredictions['estHigh'] = (self.pricePredictions['estHigh']-low)/diff
			self.PreNormalizationLow = low
			self.PreNormalizationHigh = high
			self.PreNormalizationDiff = diff
			self.pricesNormalized = True
			if self.statsLoaded: self.CalculateStats()		
			print('Prices have been normalized.')
		else:
			low = self.PreNormalizationLow
			high = self.PreNormalizationHigh 
			diff = self.PreNormalizationDiff
			self.historicalPrices['Open'] = (self.historicalPrices['Open'] * diff) + low
			self.historicalPrices['Close'] = (self.historicalPrices['Close'] * diff) + low
			self.historicalPrices['High'] = (self.historicalPrices['High'] * diff) + low
			self.historicalPrices['Low'] = (self.historicalPrices['Low'] * diff) + low
			self.historicalPrices['Average'] = self.historicalPrices.loc[:,['Open','Close','High','Low']].mean(axis=1) #select those rows, calculate the mean value
			if self.predictionsLoaded:
				self.pricePredictions['estLow'] = (self.pricePredictions['estLow'] * diff) + low
				self.pricePredictions['estAverage'] = (self.pricePredictions['estAverage'] * diff) + low
				self.pricePredictions['estHigh'] = (self.pricePredictions['estHigh'] * diff) + low
			self.pricesNormalized = False
			if self.statsLoaded: self.CalculateStats()

	def CalculateStats(self):
		if not self.pricesLoaded: self.LoadHistory()
		twodav = self.historicalPrices['Average'].rolling(window=2, center=False).mean()
		self.historicalPrices['2DayAv'] = twodav
		self.historicalPrices['shortEMA'] =  self.historicalPrices['Average'].ewm(com=3,min_periods=0,freq='B',adjust=True,ignore_na=False).mean()
		self.historicalPrices['shortEMASlope'] = (self.historicalPrices['shortEMA']/self.historicalPrices['shortEMA'].shift(1))-1
		self.historicalPrices['longEMA'] = self.historicalPrices['Average'].ewm(com=9,min_periods=0,freq='B',adjust=True,ignore_na=False).mean()
		self.historicalPrices['longEMASlope'] = (self.historicalPrices['longEMA']/self.historicalPrices['longEMA'].shift(1))-1
		self.historicalPrices['45dEMA'] = self.historicalPrices['Average'].ewm(com=22,min_periods=0,freq='B',adjust=True,ignore_na=False).mean()
		self.historicalPrices['45dEMASlope'] = (self.historicalPrices['45dEMA']/self.historicalPrices['45dEMA'].shift(1))-1
		self.historicalPrices['1DayDeviation'] = (self.historicalPrices['High'] - self.historicalPrices['Low'])/self.historicalPrices['Low']
		self.historicalPrices['5DavDeviation'] = self.historicalPrices['1DayDeviation'].rolling(window=5, center=False).mean()
		self.historicalPrices['15DavDeviation'] = self.historicalPrices['1DayDeviation'].rolling(window=15, center=False).mean()
		self.historicalPrices['1DayApc'] = ((self.historicalPrices['Average'] - self.historicalPrices['Average'].shift(1)) / self.historicalPrices['Average'].shift(1))
		self.historicalPrices['3DayApc'] = self.historicalPrices['1DayApc'].rolling(window=3, center=False).mean()
		self.historicalPrices['1DayMomentum'] = (self.historicalPrices['Average'] / self.historicalPrices['Average'].shift(1))-1
		self.historicalPrices['3DayMomentum'] = (self.historicalPrices['Average'] / self.historicalPrices['Average'].shift(3))-1
		self.historicalPrices['5DayMomentum'] = (self.historicalPrices['Average'] / self.historicalPrices['Average'].shift(5))-1
		self.historicalPrices['10DayMomentum'] = (self.historicalPrices['Average'] / self.historicalPrices['Average'].shift(10))-1
		self.historicalPrices['channelHigh'] = self.historicalPrices['longEMA'] + (self.historicalPrices['Average']*self.historicalPrices['15DavDeviation'])
		self.historicalPrices['channelLow'] = self.historicalPrices['longEMA'] - (self.historicalPrices['Average']*self.historicalPrices['15DavDeviation'])
		self.historicalPrices.fillna(method='ffill', inplace=True)
		self.historicalPrices.fillna(method='bfill', inplace=True)
		self.statsLoaded = True
		return True
	
	def SaveStatsToFile(self, includePredictions:bool=False):
		if includePredictions:
			filePath = self._dataFolderhistoricalPrices + self.stockTicker + '_stats_predictions.csv'
			r = self.historicalPrices.join(self.pricePredictions, how='outer') #, rsuffix='_Predicted'
			r.to_csv(filePath)
		else:
			filePath = self._dataFolderhistoricalPrices + self.stockTicker + '_stats.csv'
			self.historicalPrices.to_csv(filePath)
		
	def PredictPrices(self, method:int=1, daysIntoFuture:int=1, NNTrainingEpochs:int=350):
		#Predict current prices from previous days info
		self.pricePredictions = pd.DataFrame()	#Clear any previous data
		if not self.statsLoaded: self.CalculateStats()
		if method < 3:
			minActionableSlope = 0.001
			if method==0:	#Same as previous day
				self.pricePredictions = pd.DataFrame()
				self.pricePredictions['estLow'] =  self.historicalPrices['Low'].shift(1)
				self.pricePredictions['estHigh'] = self.historicalPrices['High'].shift(1)
			elif method==1:	#Slope plus momentum with some consideration for trend.
					#++,+-,-+,==
				bucket = self.historicalPrices.copy()
				bucket['estLow']  = bucket['Average'].shift(1) * (1-bucket['15DavDeviation']/2) + (abs(bucket['1DayApc'].shift(1)))
				bucket['estHigh'] = bucket['Average'].shift(1) * (1+bucket['15DavDeviation']/2) + (abs(bucket['1DayApc'].shift(1)))
				bucket = bucket.query('longEMASlope >= -' + str(minActionableSlope) + ' or shortEMASlope >= -' + str(minActionableSlope)) #must filter after rolling calcuations
				bucket = bucket[['estLow','estHigh']]
				self.pricePredictions = bucket
					#-- downward trend
				bucket = self.historicalPrices.copy()
				bucket['estLow'] = bucket['Low'].shift(1).rolling(3).min() * .99
				bucket['estHigh'] = bucket['High'].shift(1).rolling(3).min() 
				bucket = bucket.query('not (longEMASlope >= -' + str(minActionableSlope) + ' or shortEMASlope >= -' + str(minActionableSlope) +')')
				bucket = bucket[['estLow','estHigh']]
				self.pricePredictions = self.pricePredictions.append(bucket)
				self.pricePredictions.sort_index(inplace=True)	
			elif method==2:	#Slope plus momentum with full consideration for trend.
					#++ Often over bought, strong momentum
				bucket = self.historicalPrices.copy() 
				#bucket['estLow']  = bucket['Low'].shift(1).rolling(4).max()  * (1 + abs(bucket['shortEMASlope'].shift(1)))
				#bucket['estHigh'] = bucket['High'].shift(1).rolling(4).max()  * (1 + abs(bucket['shortEMASlope'].shift(1)))
				bucket['estLow']  = bucket['Low'].shift(1).rolling(4).max()  + (abs(bucket['1DayApc'].shift(1)))
				bucket['estHigh'] = bucket['High'].shift(1).rolling(4).max() + (abs(bucket['1DayApc'].shift(1)))
				bucket = bucket.query('longEMASlope >= ' + str(minActionableSlope) + ' and shortEMASlope >= ' + str(minActionableSlope)) #must filter after rolling calcuations
				bucket = bucket[['estLow','estHigh']]
				self.pricePredictions = bucket
					#+- correction or early down turn, loss of momentum
				bucket = self.historicalPrices.copy()
				bucket['estLow']  = bucket['Low'].shift(1).rolling(2).min() 
				bucket['estHigh'] = bucket['High'].shift(1).rolling(3).max()  * (1.005 + abs(bucket['shortEMASlope'].shift(1)))
				bucket = bucket.query('longEMASlope >= ' + str(minActionableSlope) + ' and shortEMASlope < -' + str(minActionableSlope))
				bucket = bucket[['estLow','estHigh']]
				self.pricePredictions = self.pricePredictions.append(bucket)
					 #-+ bounce or early recovery, loss of momentum
				bucket = self.historicalPrices.copy()
				bucket['estLow']  = bucket['Low'].shift(1)
				bucket['estHigh'] = bucket['High'].shift(1).rolling(3).max() * 1.02 
				bucket = bucket.query('longEMASlope < -' + str(minActionableSlope) + ' and shortEMASlope >= ' + str(minActionableSlope))
				bucket = bucket[['estLow','estHigh']]
					#-- Often over sold
				self.pricePredictions = self.pricePredictions.append(bucket)
				bucket = self.historicalPrices.copy() 
				bucket['estLow'] = bucket['Low'].shift(1).rolling(3).min() * .99
				bucket['estHigh'] = bucket['High'].shift(1).rolling(3).min() 
				bucket = bucket.query('longEMASlope < -' + str(minActionableSlope) + ' and shortEMASlope < -' + str(minActionableSlope))
				bucket = bucket[['estLow','estHigh']]
				self.pricePredictions = self.pricePredictions.append(bucket)
					#== no significant slope
				bucket = self.historicalPrices.copy() 
				bucket['estLow']  = bucket['Low'].shift(1).rolling(4).mean()
				bucket['estHigh'] = bucket['High'].shift(1).rolling(4).mean()
				bucket = bucket.query(str(minActionableSlope) + ' > longEMASlope >= -' + str(minActionableSlope) + ' or ' + str(minActionableSlope) + ' > shortEMASlope >= -' + str(minActionableSlope))
				bucket = bucket[['estLow','estHigh']]
				self.pricePredictions = self.pricePredictions.append(bucket)
				self.pricePredictions.sort_index(inplace=True)	

			d = self.historicalPrices.index[-1] 
			ls = self.historicalPrices['longEMASlope'][-1]
			ss = self.historicalPrices['shortEMASlope'][-1]
			deviation = self.historicalPrices['15DavDeviation'][-1]/2
			momentum = self.historicalPrices['3DayMomentum'][-1]/2 
			for i in range(0,daysIntoFuture): 	#Add new days to the end for crystal ball predictions
				momentum = (momentum + ls)/2 * (100+random.randint(-3,4))/100
				a = (self.pricePredictions['estLow'][-1] + self.pricePredictions['estHigh'][-1])/2
				if ls >= minActionableSlope and ss >= minActionableSlope: #++
					l = a * (1+momentum) + random.randint(round(-deviation*200),round(deviation*10))/100
					h = a * (1+momentum) + random.randint(round(-deviation*10),round(deviation*200))/100
				elif ls >= minActionableSlope and ss < -minActionableSlope: #+-
					l = a * (1+momentum) + random.randint(round(-deviation*200),round(deviation*10))/100
					h = a * (1+momentum) + random.randint(round(-deviation*10),round(deviation*400))/100
				elif ls < -minActionableSlope and ss >= minActionableSlope: #-+
					l = a * (1+momentum) + random.randint(round(-deviation*200),round(deviation*10))/100
					h = a * (1+momentum) + random.randint(round(-deviation*10),round(deviation*400))/100
				elif ls < -minActionableSlope and ss < -minActionableSlope: #--
					l = a * (1+momentum) + random.randint(round(-deviation*200),round(deviation*10))/100
					h = a * (1+momentum) + random.randint(round(-deviation*10),round(deviation*200))/100
				else:	#==
					l = a * (1+momentum) + random.randint(round(-deviation*200),round(deviation*10))/100
					h = a * (1+momentum) + random.randint(round(-deviation*10),round(deviation*200))/100
				self.pricePredictions.loc[d + BDay(i+1), 'estLow'] = l
				self.pricePredictions.loc[d + BDay(i+1), 'estHigh'] = h								
			self.pricePredictions['estAverage']	= (self.pricePredictions['estLow'] + self.pricePredictions['estHigh'])/2
		elif method==3:	#Use LSTM to predict prices
			model = StockPredictionNN()
			SourceFieldList = None
			#SourceFieldList = ['Low','High','Open','Close']
			model.LoadData(self.historicalPrices.copy(), window_size=1, prediction_target_days=daysIntoFuture, UseLSTM=True, SourceFieldList=SourceFieldList, batch_size=10, train_test_split=.93)
			model.TrainLSTM(epochs=NNTrainingEpochs, learning_rate=2e-5, dropout_rate=0.8, gradient_clip_margin=4)
			self.pricePredictions = model.GetTrainingResults(False, False)
			print(self.pricePredictions)
			self.pricePredictions = self.pricePredictions.rename(columns={'Average':'estAverage'})
			deviation = self.historicalPrices['15DavDeviation'][-1]/2
			self.pricePredictions['estLow'] = self.pricePredictions['estAverage'] * (1 - deviation)
			self.pricePredictions['estHigh'] = self.pricePredictions['estAverage'] * (1 + deviation)
		elif method==4:	#Use CNN to predict prices
			model = StockPredictionNN()
			SourceFieldList = ['Low','High','Open','Close']
			model.LoadData(self.historicalPrices.copy(), window_size=daysIntoFuture*16, prediction_target_days=daysIntoFuture, UseLSTM=False, SourceFieldList=SourceFieldList, batch_size=32, train_test_split=.93)
			model.TrainCNN(epochs=NNTrainingEpochs)
			self.pricePredictions = model.GetTrainingResults(False, False)
			self.pricePredictions['estAverage'] = (self.pricePredictions['Low'] + self.pricePredictions['High'] + self.pricePredictions['Open'] + self.pricePredictions['Close'])/4
			deviation = self.historicalPrices['15DavDeviation'][-1]/2
			self.pricePredictions['estLow'] = self.pricePredictions['estAverage'] * (1 - deviation)
			self.pricePredictions['estHigh'] = self.pricePredictions['estAverage'] * (1 + deviation)
			self.pricePredictions = self.pricePredictions[['estLow','estAverage','estHigh']]
		self.pricePredictions.fillna(method='bfill', inplace=True)
		x = self.pricePredictions.join(self.historicalPrices)
		x['PercentageDeviation'] = abs((x['Average']-x['estAverage'])/x['Average'])
		self.predictionDeviation = x['PercentageDeviation'].tail(round(x.shape[0]/4)).mean() #Average of the last 25%, this is being kind as it includes some training data
		self.predictionsLoaded = True
		return True
	
	def PredictFuturePrice(self,fromDate:datetime, daysForward:int=1,method:int=1):
		fromDate=DateFormatDatabase(fromDate)
		low,high,price,momentum,deviation = self.historicalPrices.loc[fromDate, ['Low','High','Average', '3DayMomentum','15DavDeviation']]
		#print(p,m,s)
		if method==0:
			futureLow = low
			futureHigh = high
		else:  
			futureLow = price * (1 + daysForward * momentum) - (price * deviation/2)
			futureHigh = price * (1 + daysForward * momentum) + (price * deviation/2)
		return futureLow, futureHigh	

	def GetPrice(self,forDate:datetime):
		forDate=DateFormatDatabase(forDate)
		return self.historicalPrices.loc[forDate,['High', 'Low']]
		
	def GetPriceSnapshot(self,forDate:datetime):
		forDate=DateFormatDatabase(forDate)
		sn = PriceSnapshot()
		sn.snapshotDate = forDate
		try:
			sn.high, sn.low, sn.open,sn.close,sn.oneDayAverage,sn.twoDayAverage,sn.shortEMA,sn.shortEMASlope,sn.longEMA,sn.longEMASlope,sn.channelHigh,sn.channelLow,sn.oneDayDeviation,sn.fiveDayDeviation,sn.fifteenDayDeviation =self.historicalPrices.loc[forDate,['High','Low','Open','Close','Average','2DayAv','shortEMA','shortEMASlope','longEMA','longEMASlope','channelHigh', 'channelLow','1DayDeviation','5DavDeviation','15DavDeviation']]
			#1dTarget = IF(15dapc<0, Min(1dav, 1dav.step(1)),Max(1dav, 1dav.step(1)) *(1+15dapc)) old model from Excel
			if sn.longEMASlope < 0:
				if sn.shortEMASlope > 0:	#bounce or early recovery
					sn.nextDayTarget = min(sn.oneDayAverage, sn.twoDayAverage)
				else:
					sn.nextDayTarget = min(sn.low, sn.twoDayAverage)			
			else:
				if sn.shortEMASlope < 0:	#correction or early downturn
					sn.nextDayTarget = max(sn.oneDayAverage, (sn.twoDayAverage*2)-sn.oneDayAverage) + (sn.oneDayAverage * (sn.longEMASlope))
				else:
					sn.nextDayTarget = max(sn.oneDayAverage, sn.twoDayAverage) + (sn.oneDayAverage * sn.longEMASlope)

			sn.nextDayTarget = max(sn.oneDayAverage, sn.twoDayAverage) + (sn.oneDayAverage * sn.longEMASlope)
			if not self.predictionsLoaded or forDate >= self.historyEndDate:
				sn.estLow,sn.estHigh= self.PredictFuturePrice(forDate,1)
			else:
				tomorrow =  forDate.date() + datetime.timedelta(days=1) 
				sn.estLow,sn.estHigh= self.pricePredictions.loc[tomorrow,['estLow','estHigh']]
		except:
			print('Unable to get price snapshot for ' + self.stockTicker + ' on ' + str(forDate))
		sn.high = round(sn.high,2)
		sn.low = round(sn.low, 2)
		sn.open = round(sn.open, 2)
		sn.close = round(sn.close, 2)
		sn.oneDayAverage = round(sn.oneDayAverage, 2)
		sn.twoDayAverage = round(sn.twoDayAverage, 2)
		sn.channelHigh = round(sn.channelHigh, 2)
		sn.channelLow = round(sn.channelLow, 2)
		sn.nextDayTarget = round(sn.nextDayTarget, 2)
		sn.estHigh = round(sn.estHigh, 2)
		sn.estLow = round(sn.estLow, 2)
		return sn

	def GetCurrentPriceSnapshot(self): return self.GetPriceSnapshot(self.historyEndDate)

	def GetPriceHistory(self, fieldList:list = None, includePredictions:bool = False):
		if fieldList == None:
			r = self.historicalPrices.copy() #best to pass back copies instead of reference.
		else:
			r = self.historicalPrices[fieldList].copy() #best to pass back copies instead of reference.			
		if includePredictions: r = r.join(self.pricePredictions, how='outer')
		return r
		
	def GetPricePredictions(self):
		return self.pricePredictions.copy()  #best to pass back copies instead of reference.

	def GraphData(self, endDate:datetime=None, daysToGraph:int=90, graphTitle:str='', includePredictions:bool=False, saveToFile:bool=False, fileNameSuffix:str='', saveToFolder:str='', dpi:int=600, trimHistoricalPredictions:bool = True):
		PlotInitDefaults()
		if graphTitle=='': 
			graphTitle = self.stockTicker
			if not fileNameSuffix =='': graphTitle = graphTitle + ' ' + fileNameSuffix
		if includePredictions:
			if not self.predictionsLoaded: self.PredictPrices()
			if endDate == None: endDate = self.pricePredictions.index.max()
			endDate = DateFormatDatabase(endDate)
			startDate = endDate - BDay(daysToGraph) 
			fieldSet = ['High','Low', 'channelHigh', 'channelLow', 'estHigh','estLow']
			if trimHistoricalPredictions: 
				y = self.pricePredictions[self.pricePredictions.index >= self.historyEndDate]
				x = self.historicalPrices.join(y, how='outer')
			else: 
				fieldSet = ['High','Low', 'estHigh','estLow']
				x = self.historicalPrices.join(self.pricePredictions, how='outer')
			if daysToGraph > 1800:	fieldSet = ['Average', 'estHigh','estLow']
			ax=x.loc[startDate:endDate,fieldSet].plot(title=graphTitle, linewidth=.75)			
		else:
			if endDate == None: endDate = self.historyEndDate
			endDate = DateFormatDatabase(endDate)
			startDate = endDate - BDay(daysToGraph) 
			if daysToGraph > 1800:
				ax=self.historicalPrices.loc[startDate:endDate,['Average']].plot(title=graphTitle, linewidth=.75)
			else:
				ax=self.historicalPrices.loc[startDate:endDate,['High','Low', 'channelHigh', 'channelLow']].plot(title=graphTitle, linewidth=.75)
		ax.set_xlabel('Date')
		ax.set_ylabel('Price')
		ax.tick_params(axis='x', rotation=70)
		ax.grid(b=True, which='both', color='0.65', linestyle='-')

		PlotScalerDateAdjust(startDate, endDate, ax)
		if saveToFile:
			if not fileNameSuffix =='': fileNameSuffix = '_' + fileNameSuffix
			if saveToFolder=='': saveToFolder = self._dataFolderCharts
			if not saveToFolder.endswith('/'): saveToFolder = saveToFolder + '/'
			if CreateFolder(saveToFolder): 	plt.savefig(saveToFolder + self.stockTicker + fileNameSuffix + '.png', dpi=dpi)			
		else:
			plt.show()
		plt.close('all')

class Traunch:
	ticker = ''
	size = 0
	units = 0
	available = True
	purchased = False
	marketOrder = False
	sold = False
	expired = False
	dateBuyOrderPlaced = None
	dateBuyOrderFilled = None
	dateSellOrderPlaced = None
	dateSellOrderFilled = None
	buyOrderPrice = 0
	purchasePrice = 0
	sellOrderPrice = 0
	sellPrice = 0
	latestPrice = 0
	expireAfterDays = 0
	_verbose = True
	
	def __init__(self, size:int=1000, _verbose:bool=True):
		self.size = size
		self._verbose = _verbose
		
	def PlaceBuy(self, ticker:str, price:float, datePlaced:datetime, marketOrder:bool=False, expireAfterDays:bool=90):
		#returns amount taken out of circulation by the order
		r = 0
		if self.available and price > 0:
			self.available = False
			self.ticker=ticker
			self.dateBuyOrderPlaced = datePlaced
			self.buyOrderPrice=price
			self.units = round(self.size/price)
			self.purchased = False
			self.marketOrder = marketOrder
			self.expireAfterDays=expireAfterDays
			r=(price*self.units)
			if self._verbose: print(' Buy placed for ' + str(self.units) + ' at ' + str(price) + ' Cost ' + str(r) + '(' + self.ticker + ')')
			if marketOrder and self._verbose: print(' Market Order')
		return r
		
	def PlaceSell(self, price, datePlaced, marketOrder:bool=False, expireAfterDays:bool=90):
		r = False
		if self.purchased and price > 0:
			self.sold = False
			self.dateSellOrderPlaced = datePlaced
			self.sellOrderPrice = price
			self.marketOrder = marketOrder
			self.expireAfterDays=expireAfterDays
			if self._verbose: print(' Sell placed for ' + str(self.units) + ' at ' + str(price) + ' (' + self.ticker + ')')
			if marketOrder and self._verbose: print(' Market Order')
			r=True
		return r

	def CancelOrder(self): 
		self.marketOrder=False
		self.expireAfterDays=0
		if self.purchased:
			self.sellOrderPrice = min(10000, self.latestPrice * 3)
		else:
			self.buyOrderPrice = 0
		
	def UpdateOrder(self, price, dateChecked):
		#Returns True if the order had action: filled or expired.
		self.latestPrice = price
		r = False
		if self.buyOrderPrice > 0 and not self.purchased:
			if self.buyOrderPrice >= price or self.marketOrder:
				self.dateBuyOrderFilled = dateChecked
				self.purchasePrice = price
				self.purchased=True
				if self._verbose: print(' Buy ordered on ' + str(self.dateBuyOrderPlaced) + ' filled for ' + str(price) + ' (' + self.ticker + ')')
				r=True
			else:
				self.expired = (DateDiffDays(self.dateBuyOrderPlaced , dateChecked) > self.expireAfterDays)
				if self.expired and self._verbose: print('Buy order expired.')
				r = self.expired
		elif self.sellOrderPrice > 0 and not self.sold:
			if self.sellOrderPrice <= price or self.marketOrder:
				self.dateSellOrderFilled = dateChecked
				self.sellPrice = price
				self.sold=True
				self.expired=False
				if self._verbose: print(' Sell ordered on ' + str(self.dateSellOrderPlaced) + ' filled for ' + str(price) + ' (' + self.ticker + ')')
				r=True
			else:
				self.expired = (DateDiffDays(self.dateSellOrderPlaced, dateChecked) > self.expireAfterDays)
				r = self.expired
		else:
			r=False
		return r
	
	def AdjustBuyUnits(self, newValue:int):	
		if self._verbose: print(' Adjusting Buy from ' + str(self.units) + ' to ' + str(newValue) + ' units (' + self.ticker + ')')
		self.units=newValue

	def Recycle(self):
		self.ticker = ""
		self.units = 0
		self.available = True
		self.purchased = False
		self.sold = False
		self.expired = False
		self.marketOrder = False
		self.dateBuyOrderPlaced = None
		self.dateBuyOrderFilled = None
		self.dateSellOrderPlaced = None
		self.dateSellOrderFilled = None
		self.latestPrice=None
		self.buyOrderPrice = 0
		self.purchasePrice = 0
		self.sellOrderPrice = 0
		self.sellPrice = 0
		self.expireAfterDays = 0
	
	def Expire(self):
		if not self.purchased:
			if self._verbose: print(' Buy order from ' + str(self.dateBuyOrderPlaced) + ' has expired (' + self.ticker + ')')
			self.Recycle()
		else:
			if self._verbose: print(' Sell order from ' + str(self.dateSellOrderPlaced) + ' has expired (' + self.ticker + ')')
			self.dateSellOrderPlaced=None
			self.sellOrderPrice=0
			self.marketOrder = False
			self.expireAfterDays = 0
			self.expired=False

	def PrintDetails(self):
		if not self.ticker =='':
			print("Stock: " + self.ticker)
			print("units: " + str(self.units))
			print("purchased: " + str(self.purchased))
			print("dateBuyOrderPlaced: " + str(self.dateBuyOrderPlaced))
			print("dateBuyOrderFilled: " + str(self.dateBuyOrderFilled))
			print("buyOrderPrice: " + str(self.buyOrderPrice))
			print("purchasePrice: " + str(self.purchasePrice))
			print("dateSellOrderPlaced: " + str(self.dateSellOrderPlaced))
			print("dateSellOrderFilled: " + str(self.dateSellOrderFilled))
			print("sellOrderPrice: " + str(self.sellOrderPrice))
			print("sellPrice: " + str(self.sellPrice))
			print("latestPrice: " + str(self.latestPrice))
			print("\n")

class Position:
	ticker = ''
	dateBuyOrderFilled = None
	dateBuyOrderPlaced = None
	purchasePrice = 0
	latestPrice = 0
	units = 0

	def __init__(self, t:Traunch):
		#ticker:str, dateOrdered:datetime, datepurchased:datetime, purchasePrice:float, units:float
		self._t = t
		self.ticker= t.ticker
		self.dateBuyOrderPlaced = t.dateBuyOrderPlaced
		self.dateBuyOrderFilled = t.dateBuyOrderFilled
		self.purchasePrice = t.purchasePrice
		self.latestPrice = t.latestPrice
		self.units = t.units
		
	def Sell(self, datePlaced:datetime, price:float, marketOrder:bool=False, expireAfterDays:bool=90):
		self._t.PlaceSell(price, datePlaced, marketOrder, expireAfterDays)

class Portfolio:
	portfolioName = ''
	tradeHistory = [] #DataFrame of trades.  Note: though you can trade more than once a day it is only going to keep one entry per day per stock
	dailyValue = []	  #DataFrame for the value at the end of each day
	_cash=0
	_fundsCommittedToOrders=0
	_commisionCost = 4
	_traunches = []			#Sets of funds for investing, rather than just a pool of cash I feel it is better to use chunks of funds
	_traunchSize = 0
	_traunchCount = 0
	_verbose = True
	
	def __del__(self):
		self._cash = 0
		self._traunches = None

	def __init__(self, startDate:datetime, initialFunds:int=10000, _traunchSizeRequested:int=1000, portfolioName:str='', _verbose:bool=True):
		self.portfolioName = portfolioName
		self._cash = initialFunds
		self._fundsCommittedToOrders = 0
		self._verbose = _verbose
		self._traunchSize = _traunchSizeRequested
		self._traunchCount = round(initialFunds/_traunchSizeRequested)
		self._traunches = [Traunch(_traunchSizeRequested, _verbose) for x in range(self._traunchCount)]
		self.dailyValue = pd.DataFrame([[startDate,initialFunds,0,initialFunds]], columns=list(['Date','CashValue','AssetValue','TotalValue']))
		self.dailyValue.set_index(['Date'], inplace=True)
		self.tradeHistory = pd.DataFrame(columns=['dateBuyOrderPlaced','ticker','dateBuyOrderFilled','dateSellOrderPlaced','dateSellOrderFilled','units','buyOrderPrice','purchasePrice','sellOrderPrice','sellPrice','NetChange'])
		self.tradeHistory.set_index(['dateBuyOrderPlaced','ticker'], inplace=True)

	def AccountingError(self):
		r = False
		if not self.ValidateFundsCommittedToOrders() == 0: 
			print('Accounting error: inaccurcy in funds committed to orders!')
			r=True
		if self.FundsAvailable() + self._traunchCount*self._commisionCost < 0: 
			print('Accounting error: negative cash balance!', self._cash, self._fundsCommittedToOrders, self.FundsAvailable())
			r=True
		return r

	def CancelAllOrders(self, currentDate):
		for t in self._traunches:
			t.CancelOrder
		p.CheckOrders(ticker, 10000, currentDate)

	def CheckOrders(self, ticker, price, dateChecked):
		#check if there was action on any pending orders
		price = round(price, 3)
		for t in self._traunches:
			if t.ticker == ticker:
				r = t.UpdateOrder(price, dateChecked)
				if r:	#Order was filled, update account
					if t.expired:
						if not t.purchased: self._fundsCommittedToOrders = self._fundsCommittedToOrders - (t.units*t.buyOrderPrice) - self._commisionCost	#return ear marked order fund
						t.Expire()
					elif t.sold:
						self._cash = self._cash + (t.units*t.sellPrice) - self._commisionCost
						if self._verbose: print(' Commission charged for Sell: ' + str(self._commisionCost))
						self.tradeHistory.loc[(t.dateBuyOrderPlaced, t.ticker)]=[t.dateBuyOrderFilled,t.dateSellOrderPlaced,t.dateSellOrderFilled,t.units,t.buyOrderPrice,t.purchasePrice,t.sellOrderPrice,t.sellPrice,((t.sellPrice - t.purchasePrice)*t.units)-self._commisionCost*2] 
						t.Recycle()
					elif t.purchased:
						self._fundsCommittedToOrders = self._fundsCommittedToOrders - (t.units*t.buyOrderPrice) - self._commisionCost	#return ear marked order fund
						fundsavailable = self._cash - abs(self._fundsCommittedToOrders)
						if t.marketOrder:
							actualCost = t.units*price
							if (fundsavailable - actualCost - self._commisionCost) < 25:	#insufficient funds
								unitsCanAfford = max(round((fundsavailable - self._commisionCost)/price)-1, 0)
								print('Ajusting units on market order for ' + ticker + ' Price: ', price, ' Requested Units: ', t.units,  ' Can afford:', unitsCanAfford)
								print(' Cash: ', self._cash, ' Committed Funds: ', self._fundsCommittedToOrders, ' Available: ', fundsavailable)
								if unitsCanAfford ==0:
									t.Recycle()
								else:
									t.AdjustBuyUnits(unitsCanAfford)
						if t.units == 0:
							if self._verbose: print('Can not afford any ' + ticker + ' at market ' + str(price) + ' canceling Buy')
							t.Recycle()
						else:
							self._cash = self._cash - (t.units*price) - self._commisionCost 
							if self._verbose: print(' Commission charged for Buy: ' + str(self._commisionCost))						
							self.tradeHistory.loc[(t.dateBuyOrderPlaced,t.ticker), 'dateBuyOrderFilled']=t.dateBuyOrderFilled #Create the row
							self.tradeHistory.loc[(t.dateBuyOrderPlaced,t.ticker)]=[t.dateBuyOrderFilled,t.dateSellOrderPlaced,t.dateSellOrderFilled,t.units,t.buyOrderPrice,t.purchasePrice,t.sellOrderPrice,t.sellPrice,''] 
						
	def CheckPriceSequence(self, ticker, p1, p2, dateChecked):
		#approximate a price sequence between given prices
		steps=40
		if p1==p2:
			self.CheckOrders(ticker, p1, dateChecked)
		else:
			step = (p2-p1)/steps
			for i in range(steps):
				p = round(p1 + i * step, 3)
				self.CheckOrders(ticker, p, dateChecked)
			self.CheckOrders(ticker, p2, dateChecked)
	
	def FundsAvailable(self): return (self._cash - self._fundsCommittedToOrders)

	def GetValue(self):
		assetValue=0
		for t in self._traunches:
			if t.purchased:
				assetValue = assetValue + (t.units*t.latestPrice)
		return self._cash, assetValue

	def GetPositions(self):
		r = []
		for t in self._traunches:
			if t.purchased: 
				p = Position(t)
				r.append(p)
		return r
		
	def GetPositionSummary(self):
		available=0
		buyPending=0
		sellPending=0
		longPostition = 0
		for t in self._traunches:
			if t.available:
				available=available+1
			elif  t.dateBuyOrderPlaced==None and not t.purchased:
				buyPending=buyPending+1
			elif t.purchased and t.dateSellOrderPlaced==None:
				longPostition=longPostition+1
			elif t.dateBuyOrderPlaced:
				sellPending=sellPending+1
		return available, buyPending, sellPending, longPostition			
	
	def HasOpenOrders(self):
		r = False
		for t in self._traunches:
			r=(t.available == False) and ((t.purchased == False) or not (t.dateSellOrderPlaced == None))
			if r: break
		return r

	def PlaceBuy(self, ticker:str, price:float, datePlaced:datetime, marketOrder:bool=False, expireAfterDays:bool=90):
		#Place with first available traunch, returns True if order was placed
		price = round(price, 3)
		r=False
		oldestExistingOrder = None
		FundsAvailable = self.FundsAvailable()
		for t in self._traunches:
			units = round(t.size/price)
			cost = units*price  + self._commisionCost
			if cost + self._commisionCost > FundsAvailable:
				units = round((FundsAvailable - self._commisionCost)/price)		
				cost = units*price + self._commisionCost
			if units > 0:
				if t.available and FundsAvailable > cost:	#Place new order
					self._fundsCommittedToOrders = self._fundsCommittedToOrders + cost 
					x = t.PlaceBuy(ticker, price, datePlaced, marketOrder)
					if not x ==cost: #insufficient funds for full purchase
						t.units = units			
					r=True
					break
				elif not t.purchased and t.ticker == ticker:	#Might have to replace existing order
					if oldestExistingOrder == None:
						oldestExistingOrder=t.dateBuyOrderPlaced
					else:
						if oldestExistingOrder > t.dateBuyOrderPlaced: oldestExistingOrder=t.dateBuyOrderPlaced
		if not r and units > 0 and False:	#Should we allow replacing existing order or require canceling existing first?  Going with require Cancel
			if oldestExistingOrder == None:
				if self.TraunchesAvailable():
					if self._verbose: print(' Unable to buy ' + str(units) + ' of ' + ticker + ' with funds available: ' + str(FundsAvailable))
				else: 
					if self._verbose: print(' Unable to buy ' + ticker + ' no traunches available')
			else:
				for t in self._traunches:
					if not t.purchased and t.ticker == ticker and oldestExistingOrder==t.dateBuyOrderPlaced:
						if self._verbose: print(' No traunch available... replacing order from ' + str(oldestExistingOrder))
						units = round(t.size/price)
						cost = units*price + self._commisionCost
						oldCost = t.buyOrderPrice * t.units + self._commisionCost
						if self._verbose: print(' Replacing Buy order for ' + ticker + ' from ' + str(t.buyOrderPrice) + ' to ' + str(price))
						if cost + self._commisionCost > FundsAvailable:
							units = round((FundsAvailable - self._commisionCost)/price)		
							cost = units*price + self._commisionCost
						t.units = units
						t.buyOrderPrice = price
						t.dateBuyOrderPlaced = datePlaced
						t.marketOrder = marketOrder
						self._fundsCommittedToOrders = self._fundsCommittedToOrders + cost - oldCost
						r=True
						break		
		return r

	def PlaceSell(self, ticker:str, price:float, datePlaced:datetime, marketOrder:bool=False, expireAfterDays:bool=10, datepurchased:datetime=None):
		#Returns True if order was placed
		r=False
		price = round(price, 3)
		for t in self._traunches:
			if t.ticker == ticker and t.purchased and t.sellOrderPrice==0 and (datepurchased==None or t.purchase ==datepurchased):
				t.PlaceSell(price, datePlaced, marketOrder,expireAfterDays)
				r=True
				break
		if not r:	#couldn't find one without a sell, try to update an existing sell order
			for t in self._traunches:
				if t.ticker == ticker and t.purchased:
					if self._verbose: print(' Updating existing sell order ')
					t.PlaceSell(price, datePlaced, marketOrder,expireAfterDays)
					r=True
					break					
		return r

	def PrintPositions(self):
		i=0
		for t in self._traunches:
			if not t.ticker =='':
				print('Set: ' + str(i))
				t.PrintDetails()
			i=i+1
		print('Funds committed to orders: ' + str(self._fundsCommittedToOrders))
		print('available funds: ' + str(self._cash - self._fundsCommittedToOrders))

	def ProcessDaysOrders(self, ticker, open, high, low, close, dateChecked):
		#approximate a sequence of the days prices for given ticker and check orders for each
		if self.HasOpenOrders():
			#print(' Processing orders for ' + ticker + ' on ' + str(dateChecked))
			p2=low
			p3=high
			if (high - open) < (open - low):
				p2=high
				p3=low
			#print(' Given price sequence      ' + str(open) + ' ' + str(high) + ' ' + str(low) + ' ' + str(close))
			#print(' Estimating price sequence ' + str(open) + ' ' + str(p2) + ' ' + str(p3) + ' ' + str(close))
			self.CheckPriceSequence(ticker, open, p2, dateChecked)
			self.CheckPriceSequence(ticker, p2, p3, dateChecked)
			self.CheckPriceSequence(ticker, p3, close, dateChecked)
		else:
			self.CheckOrders(ticker, close, dateChecked)	#No open orders but still need to update last prices
		self.UpdatedailyValue(dateChecked)
					
	def SaveTradeHistoryToFile(self, foldername:str, addTimeStamp:bool = False):
		if CreateFolder(foldername):
			filePath = foldername + self.portfolioName 
			if addTimeStamp: filePath += '_' + GetDateTimeStamp()
			filePath += '_trades.csv'
			self.tradeHistory.to_csv(filePath)

	def SaveDailyValueToFile(self, foldername:str, addTimeStamp:bool = False):
		if CreateFolder(foldername):
			filePath = foldername + self.portfolioName 
			if addTimeStamp: filePath += '_' + GetDateTimeStamp()
			filePath+= '_dailyvalue.csv'
			self.dailyValue.to_csv(filePath)

	def SellAllPositions(self, ticker, price, currentDate):
		for i in range(0,self._traunchCount):
			self.PlaceSell(ticker, price, currentDate, True)
			self.CheckOrders(ticker, price, currentDate)
		self.UpdatedailyValue(currentDate)

	def TraunchesAvailable(self):
		r = False
		for t in self._traunches:
			if t.available: 
				r = True
				break
		return r

	def UpdatedailyValue(self, dateChecked):
		_cashValue, assetValue = self.GetValue()
		self.dailyValue.loc[dateChecked]=[_cashValue,assetValue,_cashValue + assetValue] 

	def ValidateFundsCommittedToOrders(self):
		x=0
		for t in self._traunches:
			if not t.available and not t.purchased: 
				x = x + (t.units*t.buyOrderPrice) + self._commisionCost
		if round(self._fundsCommittedToOrders, 5) == round(x,5): self._fundsCommittedToOrders=x
		if not (self._fundsCommittedToOrders - x) ==0:
			print('Committed funds variance actual/recorded', x, self._fundsCommittedToOrders)
		return (self._fundsCommittedToOrders - x)
		
class TradingModel:
	#Implements trading environment for testing models
	modelName = None
	modelStartDate  = None	
	modelEndDate = None
	modelReady = False
	currentDate = None
	portfolio = None	
	priceHistory = []  #list of price histories for each stock in _stockTickerList
	startingValue = 0 
	_verbose = False
	_stockTickerList = []	#list of stocks currently held
	_dataFolderTradeModel = 'data/trademodel/'

	def __init__(self, modelName:str, startingTicker:str, startDate:datetime, durationInYears:int, totalFunds:int, verbose:bool=False, dataFolderRoot: str=''):
		#expects date format in local format, from there everything will be converted to database format				
		startDate = DateFormatDatabase(startDate)
		endDate = startDate + datetime.timedelta(days=365 * durationInYears)
		self.modelReady = False
		if not dataFolderRoot =='':
			if CreateFolder(dataFolderRoot):
				if not dataFolderRoot[-1] =='/': dataFolderRoot += '/'
				self._dataFolderTradeModel = dataFolderRoot + 'trademodel/'
		else: CreateFolder('data')
		CreateFolder(self._dataFolderTradeModel)
		p = PricingData(startingTicker)
		if p.LoadHistory(True): 
			print('Loading ' + startingTicker)
			p.CalculateStats()
			p.TrimToDateRange(startDate, endDate)
			self.priceHistory = [p]
			self.modelStartDate = p.historyStartDate
			self.modelEndDate = p.historyEndDate
			self.currentDate = self.modelStartDate
			modelName += '_' + str(startDate)[:10] + '_' + str(durationInYears) + 'year'
			self.modelName = modelName
			self.portfolio=Portfolio(startDate, totalFunds, 1000, modelName, verbose)
			self._stockTickerList = [startingTicker]
			self.startingValue = totalFunds
			self.modelReady = not(pd.isnull(self.modelStartDate))
		self._verbose = verbose
		
	def __del__(self):
		self._stockTickerList = None
		del self.priceHistory[:] 
		self.priceHistory = None
		self.modelStartDate  = None	
		self.modelEndDate = None
		self.modelReady = False

	def AccountingError(self): return self.portfolio.AccountingError() #Break if the dollars don't add up

	def AddStockTicker(self, ticker:str):
		r = False
		if not ticker in self._stockTickerList:
			p = PricingData(ticker)
			if self._verbose: print(' Loading price history for ' + ticker)
			if p.LoadHistory(True): 
				p.CalculateStats()
				if p.historyStartDate > self.modelStartDate or p.historyEndDate < self.modelEndDate:
					print('Unable to add ' + ticker + ' to the trading model because the price history does not match the model.')
				else:
					p.TrimToDateRange(self.modelStartDate, self.modelEndDate)
					self.priceHistory.append(p)
					self._stockTickerList.append(ticker)
					r = True
			else:
				print('Unable to download price history for ' + ticker)
		return r

	def CancelAllOrders(self): self.portfolio.CancelAllOrders(self.currentDate)
	def CloseModel(self, plotResults:bool=True, saveHistoryToFile:bool=True, folderName:str='data/trademodel/', dpi:int=600):	
		_cashValue, assetValue = self.portfolio.GetValue()
		if assetValue > 0:
			self.SellAllPositions()
			self.ProcessDay()
		if saveHistoryToFile:
			self.portfolio.SaveDailyValueToFile(folderName)
			self.portfolio.SaveTradeHistoryToFile(folderName)
		_cashValue, assetValue = self.portfolio.GetValue()
		print('Model ' + self.modelName + ' from ' + str(self.modelStartDate)[:10] + ' to ' + str(self.modelEndDate)[:10])
		print('Cash: ' + str(round(_cashValue)) + ' asset: ' + str(round(assetValue)) + ' total: ' + str(round(_cashValue + assetValue)))
		print('Net change: ' + str(round(_cashValue + assetValue - self.startingValue)))
		if plotResults: 
			self.PlotTradeHistoryAgainstHistoricalPrices(self.portfolio.tradeHistory, self.priceHistory[0].GetPriceHistory(), self.modelName, dpi)
		return _cashValue
		
	def FundsAvailable(self): return self.portfolio.FundsAvailable()
	def GetValue(self): return self.portfolio.GetValue()		
	def GetPriceSnapshot(self, ticker:str=''): 
		#returns snapshot object of pricing info to help make decisions
		r = None
		if ticker =='':
			r = self.priceHistory[0].GetPriceSnapshot(self.currentDate)
		else:
			if not ticker in self._stockTickerList:	self.AddStockTicker(ticker)
			if ticker in self._stockTickerList:
				for ph in self.priceHistory:
					if ph.stockTicker == ticker: r = ph.GetPriceSnapshot(self.currentDate) 
		return r
	def GetPositionSummary(self): return self.portfolio.GetPositionSummary()
	def GetdailyValue(self): return self.portfolio.dailyValue #returns dataframe with daily value of portfolio
	def GetPositions(self): return self.portfolio.GetPositions()
	def ModelCompleted(self):	return(self.currentDate == self.modelEndDate)
				
	def PlaceBuy(self, ticker:str, price:float, marketOrder:bool=False, expireAfterDays:bool=10):
		if not ticker in self._stockTickerList: self.AddStockTicker(ticker)
		if ticker in self._stockTickerList:	self.portfolio.PlaceBuy(ticker, price, self.currentDate, marketOrder, expireAfterDays)

	def PlaceSell(self, ticker:str, price:float, marketOrder:bool=False, expireAfterDays:bool=10): self.portfolio.PlaceSell(ticker, price, self.currentDate, marketOrder, expireAfterDays)

	def ProcessDay(self):
		#Process current day and increment the current date
		if self.currentDate <= self.modelEndDate: 
			if self._verbose: 
				c, a = self.portfolio.GetValue()
				if self._verbose: print(str(self.currentDate) + ' model: ' + self.modelName + ' _cash: ' + str(c) + ' Assets: ' + str(a))
			for ph in self.priceHistory:
				currentPrices=ph.GetPriceSnapshot(self.currentDate)
				self.portfolio.ProcessDaysOrders(ph.stockTicker, currentPrices.open, currentPrices.high, currentPrices.low, currentPrices.close, self.currentDate)
		if self.currentDate < self.modelEndDate:
			try:
				loc = self.priceHistory[0].historicalPrices.index.get_loc(self.currentDate) + 1
			except:
				#print(self.priceHistory[0].historicalPrices)
				print('Unable to set current date to ', self.currentDate)
			if loc < self.priceHistory[0].historicalPrices.shape[0]:
				nextDay = self.priceHistory[0].historicalPrices.index.values[loc]
				self.currentDate = DateFormatDatabase(str(nextDay)[:10])
			else:
				print('The end: ' + str(self.modelEndDate))
				self.currentDate=self.modelEndDate
			
	def PlotTradeHistoryAgainstHistoricalPrices(self, tradeHist:pd.DataFrame, priceHist:pd.DataFrame, modelName:str):
		buys = tradeHist.loc[:,['dateBuyOrderFilled','purchasePrice']]
		buys = buys.rename(columns={'dateBuyOrderFilled':'Date'})
		buys.set_index(['Date'], inplace=True)
		sells  = tradeHist.loc[:,['dateSellOrderFilled','sellPrice']]
		sells = sells.rename(columns={'dateSellOrderFilled':'Date'})
		sells.set_index(['Date'], inplace=True)
		dfTemp = priceHist.loc[:,['High','Low', 'channelHigh', 'channelLow']]
		dfTemp = dfTemp.join(buys)
		dfTemp = dfTemp.join(sells)
		PlotDataFrame(dfTemp, modelName, 'Date', 'Value')

	def SellAllPositions(self, ticker:str = ''):
		for p in self.GetPositions():
			if (p.ticker ==ticker or ticker ==''): p.Sell(self.currentDate, p.latestPrice, True)
			
	def TraunchesAvailable(self): return self.portfolio.TraunchesAvailable() 

#------------------------------------------- End Classes ----------------------------------------------		
