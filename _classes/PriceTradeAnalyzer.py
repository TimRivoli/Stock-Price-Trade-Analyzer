#These settings can be configured in a global config.ini in the program root directory under [Settings]
useWebProxyServer = False	#If you need a web proxy to browse the web
nonGUIEnvironment = False	#hosted environments often have no GUI so matplotlib won't be outputting to display

#pip install any of these if they are missing
import time, datetime, random, os, ssl, matplotlib
import numpy as np,  pandas as pd
from pandas.tseries.offsets import BDay
import urllib.request as webRequest
from _classes.Utility import *

#pricingData and TradingModel are the two intended exportable classes
#user input dates are expected to be in local format, after that they should be in database format
#-------------------------------------------- Global settings -----------------------------------------------
nonGUIEnvironment = ReadConfigBool('Settings', 'nonGUIEnvironment')
if nonGUIEnvironment: matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
currentProxyServer = None
proxyList = ['173.232.228.25:8080']
useWebProxyServer = ReadConfigBool('Settings', 'useWebProxyServer')
if useWebProxyServer: 
	x =  ReadConfigList('Settings', 'proxyList')
	if not x == None: proxyList = x		

BaseFieldList = ['Open','Close','High','Low']
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
		ax.grid(b=True, which='major', color='black', linestyle='solid', linewidth=.5)
		ax.grid(b=True, which='minor', color='0.65', linestyle='solid', linewidth=.3)
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

def GetProxiedOpener():
	testURL = 'https://stooq.com'
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
	ticker=''
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
	oneDayApc = 0
	oneDayDeviation=0
	fiveDayDeviation=0
	fifteenDayDeviation=0
	estLow=0
	nextDayTarget=0
	estHigh=0
	snapShotDate=None
	
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
		if self.stockTicker[0] == '^': url = "https://stooq.com/q/d/l/?i=d&s=" + self.stockTicker 
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
			s1 = r.decode()
			s1 = s1.replace(chr(13),'')
		except Exception as e:
			print('Web connection error: ', e)
		if len(s1) < 1024:
			print('No data found online for ticker ' + self.stockTicker)
			if useWebProxyServer:
				global currentProxyServer
				global proxyList
				if not currentProxyServer==None and len(proxyList) > 3: 
					print('Removing proxy: ', currentProxyServer)
					proxyList.remove(currentProxyServer)
					currentProxyServer = None
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
			if dropVolume: df = df[BaseFieldList]
			self.historicalPrices = df
			self.historicalPrices['Average'] = self.historicalPrices.loc[:,BaseFieldList].mean(axis=1) #select those rows, calculate the mean value
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
			print('Prices have been converted back from percentages.')
		else:
			self.CTPFactor = self.historicalPrices.iloc[0]
			self.historicalPrices = self.historicalPrices[['Open','Close','High','Low','Average']].pct_change(1)
			self.historicalPrices[:1] = 0
			if self.predictionsLoaded:
				self.pricePredictions = self.pricePredictions.pct_change(1)
			self.statsLoaded = False
			self.pricesInPercentages = True
			print('Prices have been converted to percentage change from previous day.')
		
	def NormalizePrices(self, verbose:bool=False):
		#(x-min(x))/(max(x)-min(x))
		x = self.historicalPrices
		if not self.pricesNormalized:
			low = x['Low'].min(axis=0) - .000001 #To prevent div zero calculation errors.
			high = x['High'].max(axis=0)
			diff = high-low
			x['Open'] = (x['Open']-low)/diff
			x['Close'] = (x['Close']-low)/diff
			x['High'] = (x['High']-low)/diff
			x['Low'] = (x['Low']-low)/diff
			if self.predictionsLoaded:
				self.pricePredictions['estLow'] = (self.pricePredictions['estLow']-low)/diff
				self.pricePredictions['estAverage'] = (self.pricePredictions['estAverage']-low)/diff
				self.pricePredictions['estHigh'] = (self.pricePredictions['estHigh']-low)/diff
			self.PreNormalizationLow = low
			self.PreNormalizationHigh = high
			self.PreNormalizationDiff = diff
			self.pricesNormalized = True
			print('Prices have been normalized.')
		else:
			low = self.PreNormalizationLow
			high = self.PreNormalizationHigh 
			diff = self.PreNormalizationDiff
			x['Open'] = (x['Open'] * diff) + low
			x['Close'] = (x['Close'] * diff) + low
			x['High'] = (x['High'] * diff) + low
			x['Low'] = (x['Low'] * diff) + low
			if self.predictionsLoaded:
				self.pricePredictions['estLow'] = (self.pricePredictions['estLow'] * diff) + low
				self.pricePredictions['estAverage'] = (self.pricePredictions['estAverage'] * diff) + low
				self.pricePredictions['estHigh'] = (self.pricePredictions['estHigh'] * diff) + low
			self.pricesNormalized = False
			print('Prices have been un-normalized.')
		x['Average'] = (x['Open'] + x['Close'] + x['High'] + x['Low'])/4
		#x['Average'] = x.loc[:,BaseFieldList].mean(axis=1, skipna=True) #Wow, this doesn't work.
		if (x['Average'] < x['Low']).any() or (x['Average'] > x['High']).any(): 
			print('WTF?, averages not computed correctly.')
			print(x)
			print(x.loc[:,BaseFieldList].mean(axis=1))
			assert(False)
		self.historicalPrices = x
		if self.statsLoaded: self.CalculateStats()
		if verbose: print(self.historicalPrices[:1])

	def CalculateStats(self):
		if not self.pricesLoaded: self.LoadHistory()
		twodav = self.historicalPrices['Average'].rolling(window=2, center=False).mean()
		self.historicalPrices['2DayAv'] = twodav
		self.historicalPrices['shortEMA'] =  self.historicalPrices['Average'].ewm(com=3,min_periods=0,adjust=True,ignore_na=False).mean()
		self.historicalPrices['shortEMASlope'] = (self.historicalPrices['shortEMA']/self.historicalPrices['shortEMA'].shift(1))-1
		self.historicalPrices['longEMA'] = self.historicalPrices['Average'].ewm(com=9,min_periods=0,adjust=True,ignore_na=False).mean()
		self.historicalPrices['longEMASlope'] = (self.historicalPrices['longEMA']/self.historicalPrices['longEMA'].shift(1))-1
		self.historicalPrices['45dEMA'] = self.historicalPrices['Average'].ewm(com=22,min_periods=0,adjust=True,ignore_na=False).mean()
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
		
	def PredictPrices(self, method:int=1, daysIntoFuture:int=1, NNTrainingEpochs:int=0):
		#Predict current prices from previous days info
		self.predictionsLoaded = False
		self.pricePredictions = pd.DataFrame()	#Clear any previous data
		if not self.statsLoaded: self.CalculateStats()
		if method < 3:
			minActionableSlope = 0.001
			if method==0:	#Same as previous day
				self.pricePredictions = pd.DataFrame()
				self.pricePredictions['estLow'] =  self.historicalPrices['Low'].shift(1)
				self.pricePredictions['estHigh'] = self.historicalPrices['High'].shift(1)
			elif method==1 :	#Slope plus momentum with some consideration for trend.
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
			from _classes.SeriesPrediction import StockPredictionNN
			temporarilyNormalize = False
			if not self.pricesNormalized:
				temporarilyNormalize = True
				self.NormalizePrices()
			model = StockPredictionNN(baseModelName='Prices', UseLSTM=True)
			FieldList = None
			#FieldList = BaseFieldList
			model.LoadSource(sourceDF=self.historicalPrices, FieldList=FieldList, window_size=1)
			model.LoadTarget(targetDF=None, prediction_target_days=daysIntoFuture)
			model.MakeBatches(batch_size=64, train_test_split=.93)
			model.BuildModel()
			if (not model.Load() and NNTrainingEpochs == 0): NNTrainingEpochs = 250
			if (NNTrainingEpochs > 0): 
				model.Train(epochs=NNTrainingEpochs)
				model.Save()
			model.Predict(True)
			self.pricePredictions = model.GetTrainingResults(False, False)
			self.pricePredictions = self.pricePredictions.rename(columns={'Average':'estAverage'})
			deviation = self.historicalPrices['15DavDeviation'][-1]/2
			self.pricePredictions['estLow'] = self.pricePredictions['estAverage'] * (1 - deviation)
			self.pricePredictions['estHigh'] = self.pricePredictions['estAverage'] * (1 + deviation)
			if temporarilyNormalize: 
				self.predictionsLoaded = True
				self.NormalizePrices()
		elif method==4:	#Use CNN to predict prices
			from _classes.SeriesPrediction import StockPredictionNN
			temporarilyNormalize = False
			if not self.pricesNormalized:
				temporarilyNormalize = True
				self.NormalizePrices()
			model = StockPredictionNN(baseModelName='Prices', UseLSTM=False)
			FieldList = BaseFieldList
			model.LoadSource(sourceDF=self.historicalPrices, FieldList=FieldList, window_size=daysIntoFuture*16)
			model.LoadTarget(targetDF=None, prediction_target_days=daysIntoFuture)
			model.MakeBatches(batch_size=64, train_test_split=.93)
			model.BuildModel()
			if (not model.Load() and NNTrainingEpochs == 0): NNTrainingEpochs = 250
			if (NNTrainingEpochs > 0): 
				model.Train(epochs=NNTrainingEpochs)
				model.Save()
			model.Predict(True)
			self.pricePredictions = model.GetTrainingResults(False, False)
			self.pricePredictions['estAverage'] = (self.pricePredictions['Low'] + self.pricePredictions['High'] + self.pricePredictions['Open'] + self.pricePredictions['Close'])/4
			deviation = self.historicalPrices['15DavDeviation'][-1]/2
			self.pricePredictions['estLow'] = self.pricePredictions['estAverage'] * (1 - deviation)
			self.pricePredictions['estHigh'] = self.pricePredictions['estAverage'] * (1 + deviation)
			self.pricePredictions = self.pricePredictions[['estLow','estAverage','estHigh']]
			if temporarilyNormalize: 
				self.predictionsLoaded = True
				self.NormalizePrices()
		self.pricePredictions.fillna(0, inplace=True)
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

	def GetDateFromIndex(self,indexLocation:int):
		d = self.historicalPrices.index.values[indexLocation]
		return DateFormatDatabase(str(d)[:10])

	def GetPrice(self,forDate:datetime):
		forDate=DateFormatDatabase(forDate)
		return self.historicalPrices.loc[forDate,['High', 'Low']]
		
	def GetPriceSnapshot(self,forDate:datetime, yesterday:bool=False):
		forDate = DateFormatDatabase(forDate)
		if yesterday:
			i = self.historicalPrices.index.get_loc(forDate) - 1
			if i > 0: forDate = self.historicalPrices.index.values[i]
		sn = PriceSnapshot()
		sn.ticker = self.stockTicker
		sn.snapShotDate = forDate
		try:
			sn.high, sn.low, sn.open,sn.close,sn.oneDayAverage,sn.twoDayAverage,sn.shortEMA,sn.shortEMASlope,sn.longEMA,sn.longEMASlope,sn.channelHigh,sn.channelLow,sn.oneDayApc,sn.oneDayDeviation,sn.fiveDayDeviation,sn.fifteenDayDeviation =self.historicalPrices.loc[forDate,['High','Low','Open','Close','Average','2DayAv','shortEMA','shortEMASlope','longEMA','longEMASlope','channelHigh', 'channelLow','1DayApc','1DayDeviation','5DavDeviation','15DavDeviation']]
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

	def GraphData(self, endDate:datetime=None, daysToGraph:int=90, graphTitle:str=None, includePredictions:bool=False, saveToFile:bool=False, fileNameSuffix:str=None, saveToFolder:str='', dpi:int=600, trimHistoricalPredictions:bool = True):
		PlotInitDefaults()
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
		else:
			if endDate == None: endDate = self.historyEndDate
			endDate = DateFormatDatabase(endDate)
			startDate = endDate - BDay(daysToGraph) 
			fieldSet = ['High','Low', 'channelHigh', 'channelLow']
			if daysToGraph > 1800: fieldSet = ['Average']
			x = self.historicalPrices
		if fileNameSuffix == None: fileNameSuffix = str(endDate)[:10] + '_' + str(daysToGraph) + 'days'
		if graphTitle==None: graphTitle = self.stockTicker + ' ' + fileNameSuffix 
		ax=x.loc[startDate:endDate,fieldSet].plot(title=graphTitle, linewidth=.75)			
		ax.set_xlabel('Date')
		ax.set_ylabel('Price')
		ax.tick_params(axis='x', rotation=70)
		ax.grid(b=True, which='major', color='black', linestyle='solid', linewidth=.5)
		ax.grid(b=True, which='minor', color='0.65', linestyle='solid', linewidth=.3)

		PlotScalerDateAdjust(startDate, endDate, ax)
		if saveToFile:
			if not fileNameSuffix =='': fileNameSuffix = '_' + fileNameSuffix
			if saveToFolder=='': saveToFolder = self._dataFolderCharts
			if not saveToFolder.endswith('/'): saveToFolder = saveToFolder + '/'
			if CreateFolder(saveToFolder): 	plt.savefig(saveToFolder + self.stockTicker + fileNameSuffix + '.png', dpi=dpi)			
		else:
			plt.show()
		plt.close('all')

class Traunch: #interface for handling actions on a chunk of funds
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
	_verbose = False
	
	def __init__(self, size:int=1000):
		self.size = size
		
	def AdjustBuyUnits(self, newValue:int):	
		if self._verbose: print(' Adjusting Buy from ' + str(self.units) + ' to ' + str(newValue) + ' units (' + self.ticker + ')')
		self.units=newValue

	def CancelOrder(self, verbose:bool=False): 
		self.marketOrder=False
		self.expireAfterDays=0
		if self.purchased:
			if verbose: print('Sell order on ', self.ticker, ' canceled.')
			self.dateSellOrderPlaced = None
			self.sellOrderPrice = 0
			self.expired=False
		else:
			if verbose: print('Buy order for ', self.ticker, ' canceled.')
			self.Recycle()
		
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

	def PlaceBuy(self, ticker:str, price:float, datePlaced:datetime, marketOrder:bool=False, expireAfterDays:int=10, verbose:bool=False):
		#returns amount taken out of circulation by the order
		r = 0
		self._verbose = verbose
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
			if self._verbose: 
				if marketOrder:
					print(datePlaced, ' Buy placed at Market for ' + str(self.units) + ' Cost ' + str(r) + '(' + self.ticker + ')')
				else:
					print(datePlaced, ' Buy placed at ' + str(price) + ' for ' + str(self.units) + ' Cost ' + str(r) + '(' + self.ticker + ')')
		return r
		
	def PlaceSell(self, price, datePlaced, marketOrder:bool=False, expireAfterDays:int=10, verbose:bool=False):
		r = False
		self._verbose = verbose
		if self.purchased and price > 0:
			self.sold = False
			self.dateSellOrderPlaced = datePlaced
			self.sellOrderPrice = price
			self.marketOrder = marketOrder
			self.expireAfterDays=expireAfterDays
			if self._verbose: 
				if marketOrder: 
					print(datePlaced, ' Sell placed at Market for ' + str(self.units) + ' (' + self.ticker + ')')
				else:
					print(datePlaced, ' Sell placed at ' + str(price) + ' for ' + str(self.units) + ' (' + self.ticker + ')')
			r=True
		return r

	def PrintDetails(self):
		if not self.ticker =='' or True:
			print("Stock: " + self.ticker)
			print("units: " + str(self.units))
			print("available: " + str(self.available))
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
		self._verbose=False
	
	def UpdateStatus(self, price, dateChecked):
		#Returns True if the order had action: filled or expired.
		self.latestPrice = price
		r = False
		if self.buyOrderPrice > 0 and not self.purchased:
			if self.buyOrderPrice >= price or self.marketOrder:
				self.dateBuyOrderFilled = dateChecked
				self.purchasePrice = price
				self.purchased=True
				if self._verbose: print(dateChecked, ' Buy ordered on ' + str(self.dateBuyOrderPlaced) + ' filled for ' + str(price) + ' (' + self.ticker + ')')
				r=True
			else:
				self.expired = (DateDiffDays(self.dateBuyOrderPlaced , dateChecked) > self.expireAfterDays)
				if self.expired and self._verbose: print(dateChecked, 'Buy order from ' + str(self.dateBuyOrderPlaced) + ' expired.')
				r = self.expired
		elif self.sellOrderPrice > 0 and not self.sold:
			if self.sellOrderPrice <= price or self.marketOrder:
				self.dateSellOrderFilled = dateChecked
				self.sellPrice = price
				self.sold=True
				self.expired=False
				if self._verbose: print(dateChecked, ' Sell ordered on ' + str(self.dateSellOrderPlaced) + ' filled for ' + str(price) + ' (' + self.ticker + ')') 
				r=True
			else:
				self.expired = (DateDiffDays(self.dateSellOrderPlaced, dateChecked) > self.expireAfterDays)
				if self.expired and self._verbose: print(dateChecked, 'Sell order from ' + str(self.dateSellOrderPlaced) + ' expired.')
				r = self.expired
		else:
			r=False
		return r

class Position:	#Simple interface for open positions
	def __init__(self, t:Traunch): 	self._t = t	
	def CancelSell(self): 
		if self._t.purchased: self._t.CancelOrder(verbose=True)
	def CurrentValue(self): return self._t.units * self._t.latestPrice
	def Sell(self, price:float, datePlaced:datetime, marketOrder:bool=False, expireAfterDays:int=90): self._t.PlaceSell(price=price, datePlaced=datePlaced, marketOrder=marketOrder, expireAfterDays=expireAfterDays, verbose=True)
	def SellPending(self): return (self._t.sellOrderPrice >0) and not (self._t.sold or  self._t.expired)
	def LatestPrice(self): return self._t.latestPrice
	
class Portfolio:
	portfolioName = ''
	tradeHistory = [] #DataFrame of trades.  Note: though you can trade more than once a day it is only going to keep one entry per day per stock
	dailyValue = []	  #DataFrame for the value at the end of each day
	_cash=0
	_fundsCommittedToOrders=0
	_commisionCost = 4
	_traunches = []			#Sets of funds for investing, rather than just a pool of cash I feel it is better to use chunks of funds
	_traunchCount = 0
	_verbose = False
	
	def __del__(self):
		self._cash = 0
		self._traunches = None

	def __init__(self, portfolioName:str, startDate:datetime, totalFunds:int=10000, traunchSize:int=1000, trackHistory:bool=True, verbose:bool=True):
		self.portfolioName = portfolioName
		self._cash = totalFunds
		self._fundsCommittedToOrders = 0
		self._verbose = verbose
		self._traunchCount = round(totalFunds/traunchSize)
		self._traunches = [Traunch(traunchSize) for x in range(self._traunchCount)]
		self.dailyValue = pd.DataFrame([[startDate,totalFunds,0,totalFunds]], columns=list(['Date','CashValue','AssetValue','TotalValue']))
		self.dailyValue.set_index(['Date'], inplace=True)
		self.trackHistory = trackHistory
		if trackHistory: 
			self.tradeHistory = pd.DataFrame(columns=['dateBuyOrderPlaced','ticker','dateBuyOrderFilled','dateSellOrderPlaced','dateSellOrderFilled','units','buyOrderPrice','purchasePrice','sellOrderPrice','sellPrice','NetChange'])
			self.tradeHistory.set_index(['dateBuyOrderPlaced','ticker'], inplace=True)

	#----------------------  Status and position info  ---------------------------------------
	def AccountingError(self):
		r = False
		if not self.ValidateFundsCommittedToOrders() == 0: 
			print('Accounting error: inaccurcy in funds committed to orders!')
			r=True
		if self.FundsAvailable() + self._traunchCount*self._commisionCost < 0: 
			print('Accounting error: negative cash balance.  (Cash, CommittedFunds, AvailableFunds) ', self._cash, self._fundsCommittedToOrders, self.FundsAvailable())
			r=True
		return r

	def FundsAvailable(self): return (self._cash - self._fundsCommittedToOrders)
	
	def PendingOrders(self):
		a, b, s, l = self.PositionSummary()
		return (b+s > 0)

	def Positions(self, ticker:str=''):
		r = []
		for t in self._traunches:
			if t.purchased and (t.ticker==ticker or ticker==''): 
				p = Position(t)
				r.append(p)
		return r

	def PositionSummary(self):
		available=0
		buyPending=0
		sellPending=0
		longPostition = 0
		for t in self._traunches:
			if t.available:
				available +=1
			elif  not t.purchased:
				buyPending +=1
			elif t.purchased and t.dateSellOrderPlaced==None:
				longPostition +=1
			elif t.dateBuyOrderPlaced:
				sellPending +=1
		return available, buyPending, sellPending, longPostition			

	def PrintPositions(self):
		i=0
		for t in self._traunches:
			if not t.ticker =='' or True:
				print('Set: ' + str(i))
				t.PrintDetails()
			i=i+1
		print('Funds committed to orders: ' + str(self._fundsCommittedToOrders))
		print('available funds: ' + str(self._cash - self._fundsCommittedToOrders))

	def TraunchesAvailable(self):
		a, b, s, l = self.PositionSummary()
		return a

	def ValidateFundsCommittedToOrders(self, FixIt:bool=False):
		#Returns difference between recorded value and actual
		x=0
		for t in self._traunches:
			if not t.available and not t.purchased: 
				x = x + (t.units*t.buyOrderPrice) + self._commisionCost
		if round(self._fundsCommittedToOrders, 5) == round(x,5): self._fundsCommittedToOrders=x
		if not (self._fundsCommittedToOrders - x) ==0:
			if FixIt: 
				self._fundsCommittedToOrders = x
			else:
				print('Committed funds variance actual/recorded', x, self._fundsCommittedToOrders)
		return (self._fundsCommittedToOrders - x)

	def Value(self):
		assetValue=0
		for t in self._traunches:
			if t.purchased:
				assetValue = assetValue + (t.units*t.latestPrice)
		return self._cash, assetValue

	#--------------------------------------  Order interface  ---------------------------------------
	def CancelAllOrders(self, currentDate:datetime):
		for t in self._traunches:
			t.CancelOrder()
		#for t in self._traunches:						self.CheckOrders(t.ticker, t.latestPrice, currentDate) 

	def PlaceBuy(self, ticker:str, price:float, datePlaced:datetime, marketOrder:bool=False, expireAfterDays:int=10, verbose:bool=False):
		#Place with first available traunch, returns True if order was placed
		price = round(price, 3)
		r=False
		oldestExistingOrder = None
		FundsAvailable = self.FundsAvailable()
		units = round(self._traunches[0].size/price)
		cost = units*price  + self._commisionCost
		if units == 0 or FundsAvailable < cost:
			if verbose: print('Unable to purchase.  Price exceeds available funds')
		else:	
			for t in self._traunches: #Find available 
				if t.available :	#Place new order
					self._fundsCommittedToOrders = self._fundsCommittedToOrders + cost 
					x = self._commisionCost + t.PlaceBuy(ticker=ticker, price=price, datePlaced=datePlaced, marketOrder=marketOrder, expireAfterDays=expireAfterDays, verbose=verbose) 
					if not x == cost: #insufficient funds for full purchase
						print('Expected cost changed from', cost, 'to', x)
						self._fundsCommittedToOrders = self._fundsCommittedToOrders - cost + x + self._commisionCost
					r=True
					break
				elif not t.purchased and t.ticker == ticker:	#Might have to replace existing order
					if oldestExistingOrder == None:
						oldestExistingOrder=t.dateBuyOrderPlaced
					else:
						if oldestExistingOrder > t.dateBuyOrderPlaced: oldestExistingOrder=t.dateBuyOrderPlaced
		if not r and units > 0 and False:	#We could allow replacing oldest existing order
			if oldestExistingOrder == None:
				if self.TraunchesAvailable() > 0:
					if self._verbose: print(' Unable to buy ' + str(units) + ' of ' + ticker + ' with funds available: ' + str(FundsAvailable))
				else: 
					if self._verbose: print(' Unable to buy ' + ticker + ' no traunches available')
			else:
				for t in self._traunches:
					if not t.purchased and t.ticker == ticker and oldestExistingOrder==t.dateBuyOrderPlaced:
						if self._verbose: print(' No traunch available... replacing order from ' + str(oldestExistingOrder))
						oldCost = t.buyOrderPrice * t.units + self._commisionCost
						if self._verbose: print(' Replacing Buy order for ' + ticker + ' from ' + str(t.buyOrderPrice) + ' to ' + str(price))
						t.units = units
						t.buyOrderPrice = price
						t.dateBuyOrderPlaced = datePlaced
						t.marketOrder = marketOrder
						self._fundsCommittedToOrders = self._fundsCommittedToOrders - oldCost + cost 
						r=True
						break		
		return r

	def PlaceSell(self, ticker:str, price:float, datePlaced:datetime, marketOrder:bool=False, expireAfterDays:int=10, datepurchased:datetime=None, verbose:bool=False):
		#Returns True if order was placed
		r=False
		price = round(price, 3)
		for t in self._traunches:
			if t.ticker == ticker and t.purchased and t.sellOrderPrice==0 and (datepurchased is None or t.dateBuyOrderFilled == datepurchased):
				t.PlaceSell(price=price, datePlaced=datePlaced, marketOrder=marketOrder, expireAfterDays=expireAfterDays, verbose=verbose)
				r=True
				break
		if not r:	#couldn't find one without a sell, try to update an existing sell order
			for t in self._traunches:
				if t.ticker == ticker and t.purchased:
					if self._verbose: print(' Updating existing sell order ')
					t.PlaceSell(price=price, datePlaced=datePlaced, marketOrder=marketOrder, expireAfterDays=expireAfterDays, verbose=verbose)
					r=True
					break					
		return r

	def SellAllPositions(self, datePlaced:datetime, ticker:str='', verbose:bool=False):
		for t in self._traunches:
			if t.purchased and (t.ticker==ticker or ticker==''): 
				t.PlaceSell(price=t.latestPrice, datePlaced=datePlaced, marketOrder=True, expireAfterDays=5, verbose=verbose)

	#--------------------------------------  Order Processing ---------------------------------------
	def _CheckOrders(self, ticker, price, dateChecked):
		#check if there was action on any pending orders and update current price of traunche
		price = round(price, 3)
		for t in self._traunches:
			if t.ticker == ticker:
				r = t.UpdateStatus(price, dateChecked)
				if r:	#Order was filled, update account
					if t.expired:
						if not t.purchased: self._fundsCommittedToOrders = self._fundsCommittedToOrders - (t.units*t.buyOrderPrice) - self._commisionCost	#return ear marked order fund
						t.Expire()
					elif t.sold:
						self._cash = self._cash + (t.units*t.sellPrice) - self._commisionCost
						if self._verbose: print(' Commission charged for Sell: ' + str(self._commisionCost))
						if self.trackHistory:
							self.tradeHistory.loc[(t.dateBuyOrderPlaced, t.ticker)]=[t.dateBuyOrderFilled,t.dateSellOrderPlaced,t.dateSellOrderFilled,t.units,t.buyOrderPrice,t.purchasePrice,t.sellOrderPrice,t.sellPrice,((t.sellPrice - t.purchasePrice)*t.units)-self._commisionCost*2] 
						t.Recycle()
					elif t.purchased:
						self._fundsCommittedToOrders = self._fundsCommittedToOrders - (t.units*t.buyOrderPrice) - self._commisionCost	#return ear marked order fund
						fundsavailable = self._cash - abs(self._fundsCommittedToOrders)
						if t.marketOrder:
							actualCost = t.units*price
							if (fundsavailable - actualCost - self._commisionCost) < 25:	#insufficient funds
								unitsCanAfford = max(round((fundsavailable - self._commisionCost)/price)-1, 0)
								if self._verbose:
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
							if self.trackHistory:							
								self.tradeHistory.loc[(t.dateBuyOrderPlaced,t.ticker), 'dateBuyOrderFilled']=t.dateBuyOrderFilled #Create the row
								self.tradeHistory.loc[(t.dateBuyOrderPlaced,t.ticker)]=[t.dateBuyOrderFilled,t.dateSellOrderPlaced,t.dateSellOrderFilled,t.units,t.buyOrderPrice,t.purchasePrice,t.sellOrderPrice,t.sellPrice,''] 
						
	def _CheckPriceSequence(self, ticker, p1, p2, dateChecked):
		#approximate a price sequence between given prices
		steps=40
		if p1==p2:
			self._CheckOrders(ticker, p1, dateChecked)		
		else:
			step = (p2-p1)/steps
			for i in range(steps):
				p = round(p1 + i * step, 3)
				self._CheckOrders(ticker, p, dateChecked)
			self._CheckOrders(ticker, p2, dateChecked)	

	def ProcessDaysOrders(self, ticker, open, high, low, close, dateChecked):
		#approximate a sequence of the days prices for given ticker and check orders for each
		if self.PendingOrders() > 0:
			p2=low
			p3=high
			if (high - open) < (open - low):
				p2=high
				p3=low
			#print(' Given price sequence      ' + str(open) + ' ' + str(high) + ' ' + str(low) + ' ' + str(close))
			#print(' Estimating price sequence ' + str(open) + ' ' + str(p2) + ' ' + str(p3) + ' ' + str(close))
			self._CheckPriceSequence(ticker, open, p2, dateChecked)
			self._CheckPriceSequence(ticker, p2, p3, dateChecked)
			self._CheckPriceSequence(ticker, p3, close, dateChecked)
		else:
			self._CheckOrders(ticker, close, dateChecked)	#No open orders but still need to update last prices
		self.ValidateFundsCommittedToOrders(True)
		_cashValue, assetValue = self.Value()
		self.dailyValue.loc[dateChecked]=[_cashValue,assetValue,_cashValue + assetValue] 

	#--------------------------------------  Closing Reporting ---------------------------------------
	def SaveTradeHistoryToFile(self, foldername:str, addTimeStamp:bool = False):
		if CreateFolder(foldername):
			filePath = foldername + self.portfolioName 
			if addTimeStamp: filePath += '_' + GetDateTimeStamp()
			filePath += '_trades.csv'
			if self.trackHistory:
				self.tradeHistory.to_csv(filePath)

	def SaveDailyValueToFile(self, foldername:str, addTimeStamp:bool = False):
		if CreateFolder(foldername):
			filePath = foldername + self.portfolioName 
			if addTimeStamp: filePath += '_' + GetDateTimeStamp()
			filePath+= '_dailyvalue.csv'
			self.dailyValue.to_csv(filePath)
		
class TradingModel(Portfolio):
	#Extends Portfolio to trading environment for testing models
	modelName = None
	modelStartDate  = None	
	modelEndDate = None
	modelReady = False
	currentDate = None
	priceHistory = []  #list of price histories for each stock in _stockTickerList
	startingValue = 0 
	verbose = False
	_stockTickerList = []	#list of stocks currently held
	_dataFolderTradeModel = 'data/trademodel/'
	Custom1 = None	#can be used to store custom values when using the model
	Custom2 = None
	_NormalizePrices = False

	def __init__(self, modelName:str, startingTicker:str, startDate:datetime, durationInYears:int, totalFunds:int, traunchSize:int=1000,verbose:bool=False, trackHistory:bool=True):
		#pricesAsPercentages:bool=False would be good but often results in Nan values
		#expects date format in local format, from there everything will be converted to database format				
		startDate = DateFormatDatabase(startDate)
		endDate = startDate + datetime.timedelta(days=365 * durationInYears)
		self.modelReady = False
		CreateFolder(self._dataFolderTradeModel)
		p = PricingData(startingTicker)
		if p.LoadHistory(True): 
			if verbose: print('Loading ' + startingTicker)
			p.CalculateStats()
			p.TrimToDateRange(startDate, endDate)
			self.priceHistory = [p]
			self.modelStartDate = p.historyStartDate
			self.modelEndDate = p.historyEndDate
			self.currentDate = self.modelStartDate
			modelName += '_' + str(startDate)[:10] + '_' + str(durationInYears) + 'year'
			self.modelName = modelName
			super
			self._stockTickerList = [startingTicker]
			self.startingValue = totalFunds
			self.modelReady = not(pd.isnull(self.modelStartDate))
		super(TradingModel, self).__init__(portfolioName=modelName, startDate=startDate, totalFunds=totalFunds, traunchSize=traunchSize, trackHistory=trackHistory, verbose=verbose)
		
	def __del__(self):
		self._stockTickerList = None
		del self.priceHistory[:] 
		self.priceHistory = None
		self.modelStartDate  = None	
		self.modelEndDate = None
		self.modelReady = False

	def AddStockTicker(self, ticker:str):
		r = False
		if not ticker in self._stockTickerList:
			p = PricingData(ticker)
			if self.verbose: print(' Loading price history for ' + ticker)
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

	def CancelAllOrders(self): super(TradingModel, self).CancelAllOrders(self.currentDate)
	
	def CloseModel(self, plotResults:bool=True, saveHistoryToFile:bool=True, folderName:str='data/trademodel/', dpi:int=600):	
		cashValue, assetValue = self.Value()
		netChange = cashValue + assetValue - self.startingValue 		
		if assetValue > 0:
			self.SellAllPositions(self.currentDate, ticker='')
			self.ProcessDay()
		if saveHistoryToFile:
			self.SaveDailyValueToFile(folderName)
			self.SaveTradeHistoryToFile(folderName)
		print('Model ' + self.modelName + ' from ' + str(self.modelStartDate)[:10] + ' to ' + str(self.modelEndDate)[:10])
		print('Cash: ' + str(round(cashValue)) + ' asset: ' + str(round(assetValue)) + ' total: ' + str(round(cashValue + assetValue)))
		print('Net change: ' + str(round(netChange)), str(round((netChange/self.startingValue) * 100, 2)) + '%')
		if plotResults and self.trackHistory: 
			self.PlotTradeHistoryAgainstHistoricalPrices(self.tradeHistory, self.priceHistory[0].GetPriceHistory(), self.modelName)
		return cashValue
		
	def CalculateGain(self, startDate:datetime, endDate:datetime):
		try:
			startValue = self.dailyValue['TotalValue'].at[startDate]
			endValue = self.dailyValue['TotalValue'].at[endDate]
			gain = endValue = startValue
			percentageGain = endValue/startValue
		except:
			gain = -1
			percentageGain = -1
			print('Unable to calculate gain for ', startDate, endDate)
		return gain, percentageGain
			
	def GetCustomValues(self): return self.Custom1, self.Custom2
	def GetDailyValue(self): return self.dailyValue #returns dataframe with daily value of portfolio
	def GetValueAt(self, date): 
		try:
			r = self.dailyValue['TotalValue'].at[date]
		except:
			print('Unable to return value at ', date)
			r=-1
		return r
	def GetPriceSnapshot(self, ticker:str=''): 
		#returns snapshot object of yesterday's pricing info to help make decisions today
		r = None
		if ticker =='':
			r = self.priceHistory[0].GetPriceSnapshot(self.currentDate, yesterday=True)
		else:
			if not ticker in self._stockTickerList:	self.AddStockTicker(ticker)
			if ticker in self._stockTickerList:
				for ph in self.priceHistory:
					if ph.stockTicker == ticker: r = ph.GetPriceSnapshot(self.currentDate, yesterday=True) 
		return r

	def ModelCompleted(self):	return(self.currentDate == self.modelEndDate)

	def NormalizePrices(self):
		self._NormalizePrices =  not self._NormalizePrices
		for p in self.priceHistory:
			if not p.pricesNormalized: p.NormalizePrices()
		
	def PlaceBuy(self, ticker:str, price:float, marketOrder:bool=False, expireAfterDays:bool=10, verbose:bool=False):
		if not ticker in self._stockTickerList: self.AddStockTicker(ticker)
		if ticker in self._stockTickerList:	super(TradingModel, self).PlaceBuy(ticker, price, self.currentDate, marketOrder, expireAfterDays, verbose)

	def PlaceSell(self, ticker:str, price:float, marketOrder:bool=False, expireAfterDays:bool=10, datepurchased:datetime=None, verbose:bool=False): 
		super(TradingModel, self).PlaceSell(ticker=ticker, price=price, datePlaced=self.currentDate, marketOrder=marketOrder, expireAfterDays=expireAfterDays, verbose=verbose)

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

	def ProcessDay(self):
		#Process current day and increment the current date
		if self.currentDate <= self.modelEndDate: 
			if self.verbose: 
				c, a = self.Value()
				if self.verbose: print(str(self.currentDate) + ' model: ' + self.modelName + ' _cash: ' + str(c) + ' Assets: ' + str(a))
			for ph in self.priceHistory:
				p = ph.GetPriceSnapshot(self.currentDate)
				self.ProcessDaysOrders(ph.stockTicker, p.open, p.high, p.low, p.close, self.currentDate)
		if self.currentDate <= self.modelEndDate:
			try:
				loc = self.priceHistory[0].historicalPrices.index.get_loc(self.currentDate) + 1
			except:
				#print(self.priceHistory[0].historicalPrices)
				print('Unable to set current date to ', self.currentDate)
			if loc < self.priceHistory[0].historicalPrices.shape[0]:
				nextDay = self.priceHistory[0].historicalPrices.index.values[loc]
				self.currentDate = DateFormatDatabase(str(nextDay)[:10])
			else:
				#print('The end: ' + str(self.modelEndDate))
				self.currentDate=self.modelEndDate		
	
	def SetCustomValues(self, v1, v2):
		self.Custom1 = v1
		self.custom2 = v2
		
class ForcastModel():	#used to forecast the effect of a series of trade actions, one per day, and return the net change in value.  This will mirror the given model.  Can also be used to test alternate past actions 
	def __init__(self, mirroredModel:TradingModel, daysToForecast:int = 10):
		modelName = 'Forcaster for ' + mirroredModel.modelName
		self.daysToForecast = daysToForecast
		self.startDate = mirroredModel.modelStartDate 
		durationInYears = (mirroredModel.modelEndDate-mirroredModel.modelStartDate).days/365
		self.tm = TradingModel(modelName=modelName, startingTicker=mirroredModel._stockTickerList[0], startDate=mirroredModel.modelStartDate, durationInYears=durationInYears, totalFunds=mirroredModel.startingValue, verbose=False, trackHistory=False)
		self.savedModel = TradingModel(modelName=modelName, startingTicker=mirroredModel._stockTickerList[0], startDate=mirroredModel.modelStartDate, durationInYears=durationInYears, totalFunds=mirroredModel.startingValue, verbose=False, trackHistory=False)
		self.mirroredModel = mirroredModel
		self.tm._stockTickerList = mirroredModel._stockTickerList
		self.tm.priceHistory = mirroredModel.priceHistory
		self.savedModel._stockTickerList = mirroredModel._stockTickerList
		self.savedModel.priceHistory = mirroredModel.priceHistory

	def Reset(self, updateSavedModel:bool=True):
		if updateSavedModel:
			c, a = self.mirroredModel.Value()
			self.savedModel.currentDate = self.mirroredModel.currentDate
			self.savedModel._cash=self.mirroredModel._cash
			self.savedModel._fundsCommittedToOrders=self.mirroredModel._fundsCommittedToOrders
			self.savedModel.dailyValue = pd.DataFrame([[self.mirroredModel.currentDate,c,a,c+a]], columns=list(['Date','CashValue','AssetValue','TotalValue']))
			self.savedModel.dailyValue.set_index(['Date'], inplace=True)
			for i in range(len(self.savedModel._traunches)):
				self.savedModel._traunches[i].ticker = self.mirroredModel._traunches[i].ticker
				self.savedModel._traunches[i].available = self.mirroredModel._traunches[i].available
				self.savedModel._traunches[i].size = self.mirroredModel._traunches[i].size
				self.savedModel._traunches[i].units = self.mirroredModel._traunches[i].units
				self.savedModel._traunches[i].purchased = self.mirroredModel._traunches[i].purchased
				self.savedModel._traunches[i].marketOrder = self.mirroredModel._traunches[i].marketOrder
				self.savedModel._traunches[i].sold = self.mirroredModel._traunches[i].sold
				self.savedModel._traunches[i].dateBuyOrderPlaced = self.mirroredModel._traunches[i].dateBuyOrderPlaced
				self.savedModel._traunches[i].dateBuyOrderFilled = self.mirroredModel._traunches[i].dateBuyOrderFilled
				self.savedModel._traunches[i].dateSellOrderPlaced = self.mirroredModel._traunches[i].dateSellOrderPlaced
				self.savedModel._traunches[i].dateSellOrderFilled = self.mirroredModel._traunches[i].dateSellOrderFilled
				self.savedModel._traunches[i].buyOrderPrice = self.mirroredModel._traunches[i].buyOrderPrice
				self.savedModel._traunches[i].purchasePrice = self.mirroredModel._traunches[i].purchasePrice
				self.savedModel._traunches[i].sellOrderPrice = self.mirroredModel._traunches[i].sellOrderPrice
				self.savedModel._traunches[i].sellPrice = self.mirroredModel._traunches[i].sellPrice
				self.savedModel._traunches[i].latestPrice = self.mirroredModel._traunches[i].latestPrice
				self.savedModel._traunches[i].expireAfterDays = self.mirroredModel._traunches[i].expireAfterDays
		c, a = self.savedModel.Value()
		self.startingValue = c + a
		self.tm.currentDate = self.savedModel.currentDate
		self.tm._cash=self.savedModel._cash
		self.tm._fundsCommittedToOrders=self.savedModel._fundsCommittedToOrders
		self.tm.dailyValue = pd.DataFrame([[self.savedModel.currentDate,c,a,c+a]], columns=list(['Date','CashValue','AssetValue','TotalValue']))
		self.tm.dailyValue.set_index(['Date'], inplace=True)
		for i in range(len(self.tm._traunches)):
			self.tm._traunches[i].ticker = self.savedModel._traunches[i].ticker
			self.tm._traunches[i].available = self.savedModel._traunches[i].available
			self.tm._traunches[i].size = self.savedModel._traunches[i].size
			self.tm._traunches[i].units = self.savedModel._traunches[i].units
			self.tm._traunches[i].purchased = self.savedModel._traunches[i].purchased
			self.tm._traunches[i].marketOrder = self.savedModel._traunches[i].marketOrder
			self.tm._traunches[i].sold = self.savedModel._traunches[i].sold
			self.tm._traunches[i].dateBuyOrderPlaced = self.savedModel._traunches[i].dateBuyOrderPlaced
			self.tm._traunches[i].dateBuyOrderFilled = self.savedModel._traunches[i].dateBuyOrderFilled
			self.tm._traunches[i].dateSellOrderPlaced = self.savedModel._traunches[i].dateSellOrderPlaced
			self.tm._traunches[i].dateSellOrderFilled = self.savedModel._traunches[i].dateSellOrderFilled
			self.tm._traunches[i].buyOrderPrice = self.savedModel._traunches[i].buyOrderPrice
			self.tm._traunches[i].purchasePrice = self.savedModel._traunches[i].purchasePrice
			self.tm._traunches[i].sellOrderPrice = self.savedModel._traunches[i].sellOrderPrice
			self.tm._traunches[i].sellPrice = self.savedModel._traunches[i].sellPrice
			self.tm._traunches[i].latestPrice = self.savedModel._traunches[i].latestPrice
			self.tm._traunches[i].expireAfterDays = self.savedModel._traunches[i].expireAfterDays		
		c, a = self.tm.Value()
		if self.startingValue != c + a:
			print('Forcast model accounting error.  ', self.startingValue, self.mirroredModel.Value(), self.savedModel.Value(), self.tm.Value())
			assert(False)
			
	def GetResult(self):
		dayCounter = len(self.tm.dailyValue)
		while dayCounter <= self.daysToForecast:  
			self.tm.ProcessDay()
			dayCounter +=1
		c, a = self.tm.Value()
		endingValue = c + a
		#print('Start Value: ', self.startingValue)
		#print('End Value: ', endingValue)
		#print(self.tm.dailyValue)
		return endingValue - self.startingValue
		
