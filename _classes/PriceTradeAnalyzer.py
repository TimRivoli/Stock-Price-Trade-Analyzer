#PriceSnapshot, PricingData, Portfolio, TradingModel, ForcastModel, StockPicker, and PTADatabase are the intended exportable classes
#PriceSnapshot, PricingData, Portfolio, TradingModel, ForcastModel, StockPicker, and PTADatabase are the intended exportable classes
#user input dates are expected to be in local format
#These settings can be configured in a global config.ini in the program root directory under [Settings]
#There are also optional settings for using a database
suspendPriceLoads = True	#Use to prevent over burdening of provider with requests while testing
globalUseDatabase=False 	#The global default option for using a database instead of CSV files, to enable populate database settings in the .ini file
displayPythonWarnings = False #Sometimes these are important, sometimes just annoying
useWebProxyServer = False	#Turn on and populate proxy settings if you need a web proxy to browse the web
nonGUIEnvironment = False	#Turn on for hosted environments often have no GUI to prevent matplotlib load fail

#pip install any of these if they are missing
import time, random, os, ssl, matplotlib, warnings, requests
import numpy as np, pandas as pd
import urllib.error, urllib.request as webRequest
from math import floor
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
from _classes.Utility import *
from yahoofinancials import YahooFinancials
from contextlib import suppress

#-------------------------------------------- Global settings -----------------------------------------------
nonGUIEnvironment = ReadConfigBool('Settings', 'nonGUIEnvironment')
if nonGUIEnvironment: matplotlib.use('agg',warn=False, force=True)
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
if not displayPythonWarnings: warnings.filterwarnings("ignore")
base_field_list = ['Open','Close','High','Low']

globalUseDatabase = Is_sql_configured()
useWebProxyServer = ReadConfigBool('Settings', 'useWebProxyServer')

#-------------------------------------------- General Utilities -----------------------------------------------
def PandaIsInIndex(df:pd.DataFrame, value):
	try:
		x = df.loc[value]
		r = True
	except:
		r = False
	return r

def PlotSetDefaults():
	#params = {'legend.fontsize': 4, 'axes.labelsize': 4,'axes.titlesize':4,'xtick.labelsize':4,'ytick.labelsize':4}
	#plt.rcParams.update(params)
	plt.rcParams['font.size'] = 4

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
		PlotSetDefaults()
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


#-------------------------------------------- Classes -----------------------------------------------
class PlotHelper:
	def PlotDataFrame(self, df:pd.DataFrame, title:str='', xlabel:str='', ylabel:str='', adjustScale:bool=True, fileName:str = '', dpi:int=600): PlotDataFrame(df, title, xlabel, ylabel, adjustScale, fileName, dpi)

	def PlotDataFrameDateRange(self, df:pd.DataFrame, endDate:datetime=None, historyDays:int=90, title:str='', xlabel:str='', ylabel:str='', fileName:str = '', dpi:int=600):
		if df.shape[0] > 10: 
			if endDate==None: endDate=df.index[-1] 	#The latest date in the dataframe assuming ascending order				
			endDate = ToDateTime(endDate)
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
	average=0
	average=0
	Average_2Day=0
	Average_5Day=0
	EMA_Short=0
	EMA_ShortSlope=0
	EMA_Long=0
	EMA_LongSlope=0
	EMA_12Day=0
	EMA_26Day=0
	MACD_Line=0
	MACD_Signal=0
	MACD_Histogram=0
	Channel_High=0
	Channel_Low=0
	Deviation_1Day=0
	Deviation_5Day=0
	Deviation_10Day=0
	Deviation_15Day=0
	Gain_Monthly=0
	LossStd_1Year=0
	Target_1Day=0
	Predicted_Low=0
	Predicted_High=0
	PC_1Day=0
	PC_1Month=0
	PC_1Month3WeekEMA=0
	PC_2Month=0
	PC_3Month=0
	PC_6Month=0
	PC_1Year=0
	PC_18Month=0
	PC_2Year=0
	Point_Value=0
	Comments=''
	date=None
	
class DataDownload:
	def _DownLoadGoogleFinancePage(self, ticker:str, stockExchange:str ="NYSE"):
		print("Downloading ticker infor for " + ticker)
		url = "https://www.google.com/finance/quote/" + ticker + ":" + stockExchange + "?window=5D"  #5 Day, (data1: 1M Daily, data2: 1D Minutely)
		try:
			openUrl = webRequest.urlopen(url, timeout=60) 
			r = openUrl.read()
			openUrl.close()
			result = r.decode()
		except:
			print('Failed to open ' + url)
			result=""
		if result !="":
			startCompanyNameDelimiter = 'class="zzDege">'
			startIndex = result.find(startCompanyNameDelimiter,0)
			if startIndex >= len(startCompanyNameDelimiter): 
				result = result[startIndex-50:]
			startDataSectionDelimiter = '[[[["' + ticker + '","' + stockExchange + '"]'
			startIndex = result.find(startDataSectionDelimiter,0)
			if startIndex < 0: result=""
		return result
			
	def _CleanScrapedTextEntry(self, v:str):
		x =  v.find("</td")
		if x > 0: v = v[:x]
		x =  v.find(" ")
		if x > 0: v = v[:x]
		x =  v.find("<")
		if x > 0: v = v[:x]
		v=v.replace(",","")
		if len(v) > 1:
			if v[0] =="$": v = v[1:]
			if v[0] =="£": v = v[1:]
		if len(v) > 1:
			if v[-1:] =="%":
				v = v[:-1]
			elif v[-1:] =="M":
				v = v[:-1]
			elif v[-1:] =="B":
				v = str(float(v[:-1]) * 1000)
			elif v[-1:] =="T":
				v = str(float(v[:-1]) * 1000 * 1000)
			elif v[-1:] =="K":
				v = str(float(v[:-1]) / 1000)
		if v == "—" or v=="-" or v =="": v=0
		if isfloat(v):  v = float(v)
		#print('v',v)
		return v

	def _ScrapeGoogleFinanceTickerInfoAndFinancials(self, ticker:str, pageData:str):
		#Updates missing About and other info
		print(" Parsing ticker infor for " + ticker)
		result = False
		currentDate = GetTodaysDate()
		currentYear = currentDate.year
		
		db = PTADatabase()
		if db.Open():
			cursor = db.GetCursor()
			startCompanyNameDelimiter = 'class="zzDege">'
			endCompanyNameDelimiter = "</div>"
			startIndex = pageData.find(startCompanyNameDelimiter,0)
			if startIndex > 0:
				startIndex+=len(startCompanyNameDelimiter)
				endIndex = pageData.find(endCompanyNameDelimiter, startIndex +1)
				if endIndex > 0: 
					CompanyName = (pageData[startIndex:endIndex])
					if len(CompanyName) > 50: CompanyName = CompanyName[:50]
					#print("Company: " + CompanyName)
					cursor.execute("Update Tickers Set CompanyName=? WHERE Ticker=?",CompanyName, ticker)
			startRecordDelimiter = 'class="P6K39c">' 
			endRecordDelimiter = "</div>"
			startFinancialsDelimiter = 'class="QXDnM">' 
			startAboutDelimiter = 'class="bLLb2d">'
			values = []
			startIndex = pageData.find(startRecordDelimiter,0)
			if startIndex > 0:
				#pageData = pageData[startIndex:]
				#startIndex = 0
				while startIndex >= 0:
					startIndex += len(startRecordDelimiter)
					endIndex = pageData.find(endRecordDelimiter, startIndex +1)
					if endIndex > 0: 
						values.append(pageData[startIndex:endIndex])
					startIndex = pageData.find(startRecordDelimiter, startIndex+1)
				if len(values) > 6:
					#if not values[5].isnumeric():
					#	assert(False)
					#else:
					values[3] = self._CleanScrapedTextEntry(values[3])/1000 #Market Cap
					values[5] = self._CleanScrapedTextEntry(values[5]) #P/E Ratio
					values[6] = self._CleanScrapedTextEntry(values[6]) #Dividend
					if isinstance(values[5], (float, int)) and isinstance(values[6], (float, int)):
						print('  Market Cap:', values[3]) #Market Cap
						print('  P/E:', values[5]) #P/E Ratio
						print('  Dividend:', values[6]) #Dividend
						print('  Exchange:', values[7]) #Exchange
						cursor.execute("Update Tickers Set MarketCap=?, PE_Ratio=?, Dividend=? WHERE Ticker=?",values[3], values[5], values[6], ticker)
					else:
						cursor.execute("Update Tickers Set MarketCap=? WHERE Ticker=?",values[3], ticker)
			values = []
			startIndex = pageData.find(startFinancialsDelimiter,0)
			if startIndex > 0:
				while startIndex >= 0:
					startIndex +=  len(startFinancialsDelimiter)
					endIndex = pageData.find(endRecordDelimiter, startIndex +1)
					if endIndex > 0: 
						values.append(pageData[startIndex:endIndex])
					startIndex = pageData.find(startFinancialsDelimiter, startIndex+1)
				#print('Financial value entry length:',len(values))
				for i in range(len(values)):
					#print(i, values[i][:30], "before")
					values[i]=self._CleanScrapedTextEntry(values[i])
					#print(i, values[i])
				while len(values) < 24: values.append(0)
				with suppress(Exception):
					cursor.execute("DELETE FROM TickerFinancials WHERE Ticker=? AND Year=?", ticker, currentYear)
					SQL = "INSERT INTO TickerFinancials (Ticker, Year, Revenue, OperatingExpense, NetIncome, NetProfitMargin, EarningsPerShare, EBITDA, EffectiveTaxRate) Values(?,?,?,?,?,?,?,?,?)"
					cursor.execute(SQL,ticker,currentYear,values[0],values[1],values[2],values[3],values[4],values[5],values[6])
					SQL = "UPDATE TickerFinancials SET CashShortTermInvestments=?, TotalAssets=?, TotalLiabilities=?, TotalEquity=?, SharesOutstanding=?, PriceToBook=?, ReturnOnAssetts=?, ReturnOnCapital=? WHERE Ticker=? AND Year=?"
					cursor.execute(SQL,values[7],values[8],values[9],values[10],values[11],values[12],values[13],values[14],ticker,currentYear)
					SQL = "UPDATE TickerFinancials SET CashFromOperations=?, CashFromInvesting=?, CashFromFinancing=?, NetChangeInCash=?, FreeCashFlow=? WHERE Ticker=? AND Year=?"
					cursor.execute(SQL,values[16],values[17],values[18],values[19],values[20],ticker,currentYear)	
			startIndex = pageData.find(startAboutDelimiter,0)
			if startIndex > 0:
				startIndex += len(startAboutDelimiter)
				endIndex = pageData.find(endRecordDelimiter, startIndex+1)
				if endIndex > 0: 
					AboutComment = pageData[startIndex:endIndex]
					result=True
				AboutComment = AboutComment[:500]
				#print(AboutComment, len(AboutComment))
				cursor.execute("Update Tickers Set About=? WHERE Ticker=? and About is null", AboutComment, ticker)
			cursor.close()
			db.Close()

	def _ParseAndUpdatePriceHistory(self, ticker:str, stockExchange:str, pageData:str, IntraDayValues:bool=False, verbose:bool=False):
		result = False
		startDate = ""
		endDate = ""
		yearCurrent = datetime.now().year
		yearPrior = yearCurrent-1
		startDataSectionDelimiter = '[[[["' + ticker + '","' + stockExchange + '"]'
		startDataSectionDelimiter2 = '[[[' 
		endDataSectionDelimiter = 'sideChannel:' 	
		startRecordDelimiter = '[['
		endDatePartDelimiter = '[-14400]],['
		endDatePartDelimiter2 = '[-18000]],['
		startIndex = pageData.find(startDataSectionDelimiter,0)
		startIndex += len(startRecordDelimiter) 
		if IntraDayValues: startIndex = pageData.find(startDataSectionDelimiter,startIndex+5) #first data should be monthly, second daily, not always
		if startIndex > 0: 
			startIndex = pageData.find(startDataSectionDelimiter2, startIndex+5)
			if startIndex > 0: 
				startIndex += len(startDataSectionDelimiter2) + 1
				endIndex = pageData.find(endDataSectionDelimiter, startIndex)
				theMeat = pageData[startIndex:endIndex]
				startIndex = theMeat.find(startRecordDelimiter,0)
				db = PTADatabase()
				if db.Open():
					cursor = db.GetCursor()
					currentTimeStamp = datetime.now() + timedelta(minutes=-5) #Local time might be different than SQL and script takes miliseconds so..
					IntraDayValues = False #not consistent with data sets so set this when minute>0
					valideRecords=0
					errors=0
					while startIndex >= 0:
						startIndex += len(startRecordDelimiter) 
						endIndex = theMeat.find(startRecordDelimiter,startIndex+1) 
						dataRecord = theMeat[startIndex:endIndex]
						theMeat=theMeat[endIndex:]
						startIndex = dataRecord.find(endDatePartDelimiter,0)
						x = dataRecord.find(endDatePartDelimiter2,0)
						if x > 10 and (x < startIndex or startIndex < 10): startIndex = x
						datePart = dataRecord[:startIndex]
						if datePart[:10].find("[") == 0: datePart = datePart[1:]
						pricePart = dataRecord[startIndex + len(endDatePartDelimiter):]
						dateParts = datePart.split(",")
						validRecord = dateParts[0].isnumeric() and dateParts[1].isnumeric() and len(dateParts[0]) <=4
						if validRecord: validRecord=(int(dateParts[0])==yearCurrent or int(dateParts[0])==yearPrior)
						if not '"' in datePart: #Skip the section of dates wrapped in quotes, different format and not needed
							if not validRecord: 
								print('Invalid date record: ', datePart)
								print('Price:', pricePart)
								errors +=1
							else:
								priceParts = pricePart.split(",")
								validRecord = priceParts[0].replace(".","").isnumeric()
								if not validRecord:
									print('Invalid price record: ', pricePart)
									errors +=1
								else:
									year = int(dateParts[0])
									month = int(dateParts[1])
									day = int(dateParts[2])
									hour = int(dateParts[3])
									minute = 0
									if dateParts[4] !='null': 
										minute = int(dateParts[4])
									if minute > 0: IntraDayValues=True	
									theDate = datetime(year, month, day, hour, minute)
									valideRecords +=1
									price = priceParts[0]
									price = float(price)
									if len(priceParts) > 5: volume = priceParts[6].replace(']','')
									if not volume.isnumeric():volume = 0
									if volume =="": volume = 0
									if startDate =="": startDate = theDate
									endDate=theDate
									#print(theDate.strftime("%m/%d/%Y, %H:%M:%S"), '$' + price, 'volume: ' + str(volume))
									if price > 0: cursor.execute("INSERT INTO PricesIntraday (Ticker, Year, Month, Day, Hour, Minute, DateTime, Price, Volume) values(?,?,?,?,?,?,?,?,?)", ticker, year, month, day, hour, minute, theDate, price, volume)
						startIndex = theMeat.find(startRecordDelimiter,0)
					print(' Valid records: ' + str(valideRecords), " Errors: " + str(errors))
					result = errors < 10 and valideRecords > 10
					if not result and ticker[0] != '.' and False: 
						print('Failed to load price data for ' + ticker)
						assert(False)
					if result:
						if IntraDayValues: #Updating daily values, date range should be today
							cursor.execute("DELETE FROM PricesIntraday WHERE Ticker=? AND TimeStamp < ? AND [DateTime] BETWEEN ? AND ?", ticker, currentTimeStamp, startDate, endDate)
						else:
							#Only delete the daily (hour=16, minute=0) 
							cursor.execute("DELETE FROM PricesIntraday WHERE Ticker=? AND [Hour]=16 and [Minute]=0 AND TimeStamp < ? AND [DateTime] BETWEEN ? AND ?", ticker, currentTimeStamp, startDate, endDate)
					cursor.close()
					db.Close()
		if result:
			print(" Parse price data successful for ticker: " + ticker, stockExchange, startDate, endDate)
		else:
			print(" Parse price data failed for ticker: " + ticker, stockExchange)
			#assert(False)

	def DownloadIntradayPriceGoogleFinance(self, ticker:str, Exchange:str='NYSE'):
		Result = False
		Exchanges = ['NYSE','NASDAQ','Delisted']
		if Exchange == "":
			i = 0
			while i <= 2 and pageData =="":				
				Exchange = Exchanges[i]
				print("Testing exchange: " + Exchange)
				pageData = ""
				if i < 2: pageData = self._DownLoadGoogleFinancePage(ticker, Exchange)
				i +=1
			if pageData!="": cursor.execute("IF NOT EXISTS(SELECT Ticker FROM tickers WHERE ticker=?) INSERT INTO Tickers (Ticker, Exchange) values(?,?)", ticker, ticker, Exchange)
			cursor.execute("UPDATE t SET Exchange=? FROM Tickers t WHERE Ticker=?", Exchange, ticker)		
		else:
			pageData = self._DownLoadGoogleFinancePage(ticker, Exchange)				
		if pageData != "":
			self._ParseAndUpdatePriceHistory(ticker, Exchange, pageData, True)
			self._ParseAndUpdatePriceHistory(ticker, Exchange, pageData, False)
			self._ScrapeGoogleFinanceTickerInfoAndFinancials(ticker, pageData)
			Result = True
		return Result
	
	
	def DownloadTickerGoogleFinance(self, ticker:str, exchange:str='NYSE'):
		#Updates missing About and other info
		pageData = self._DownLoadGoogleFinancePage(ticker, exchange)
		if pageData != '': self._ScrapeGoogleFinanceTickerInfoAndFinancials(ticker, pageData)

	def DownloadPriceDataYahooFinance(self, ticker_list:list, download_folder:str):
		end_date = GetTodaysDateString()
		for t in ticker_list:
			if t !='':
				t2 = t.upper().replace('.INX', '^SPX')
				if t2[0]== '.': t2 = t2.replace('.', '^')
				t2 = t2.replace('.', '-')
				yf = YahooFinancials([t2], concurrent=True, max_workers=8, country="US")
				hpd = yf.get_historical_price_data('1980-01-01', end_date, 'daily')
				for x in hpd.keys():
					#print(hpd[x].keys())
					if 'prices' in hpd[x].keys():
						df = pd.DataFrame(hpd[x]['prices'])
						df['Date'] = pd.to_datetime(df['formatted_date'])
						df.set_index('Date', inplace=True)
						df.sort_index(inplace=True)
						df = df[['open','high','low','close','volume']]
						df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
						df.to_csv(download_folder + t + '.csv')

	def _UnpackNestedDictionary(self, data):
    #From https://github.com/JECSand/yahoofinancials/issues/98
		outputdict = {}
		dates = []
		for dic in data:
			for key, value in dic.items():
				dates.append(key)
				if isinstance(value, dict):
					for k2, v2, in value.items():
						outputdict[k2] = outputdict.get(k2, []) + [v2]
				else:
					outputdict[key] = outputdict.get(key, []) + [value]   
		return outputdict, dates 

	def DownloadFinanceDataYahooFinance(self, ticker_list:list, download_folder:str, balance_sheet:bool=True, income_statement:bool=True, cash_flow:bool=False):
		yf = YahooFinancials(ticker_list, concurrent=True, max_workers=8, country="US")
		if balance_sheet:
			data = yf.get_financial_stmts('quarterly', 'balance', reformat=True)
			keys = list(data.keys())
			data = data[keys[0]] #'balanceSheetHistoryQuarterly'
			for ticker in ticker_list:
				file_name = ticker + '_quarterly_balance-sheet.csv'
				data_dictionary, dates = self._UnpackNestedDictionary(data[ticker])
				max_length = 0
				for key in data_dictionary.keys():
					if len(data_dictionary[key]) > max_length: max_length = len(data_dictionary[key])
					#print(key, len(data_dictionary[key]))
				incomplete_keys = []
				for key in data_dictionary.keys():
					if len(data_dictionary[key]) < max_length: incomplete_keys.append(key)
				for key in incomplete_keys: del data_dictionary[key]
				df = pd.DataFrame.from_dict(data_dictionary).apply(pd.to_numeric)
				dates = pd.Series(dates, name='Date')
				df = pd.concat([df, dates], axis=1)
				df.set_index('Date', inplace=True)
				df.sort_index(inplace=True)	
				print(df)
				print(download_folder + file_name)
				if len(df) > 0: df.to_csv(download_folder + file_name)

		if income_statement:
			data = yf.get_financial_stmts('quarterly', 'income', reformat=True)
			keys = list(data.keys())
			data = data[keys[0]] #'incomeStatementHistoryQuarterly'
			df =  pd.DataFrame()
			for ticker in ticker_list:
				file_name = ticker + '_quarterly_financials.csv'
				data_dictionary, dates = self._UnpackNestedDictionary(data[ticker])
				max_length = 0
				for key in data_dictionary.keys():
					if len(data_dictionary[key]) > max_length: max_length = len(data_dictionary[key])
					#print(key, len(data_dictionary[key]))			
				incomplete_keys = []
				for key in data_dictionary.keys():
					if len(data_dictionary[key]) < max_length: incomplete_keys.append(key)
				for key in incomplete_keys: del data_dictionary[key]
				df = pd.DataFrame.from_dict(data_dictionary).apply(pd.to_numeric)
				dates = pd.Series(dates, name='Date')
				df = pd.concat([df, dates], axis=1)
				df.set_index('Date', inplace=True)
				df.sort_index(inplace=True)	
				print(df)
				if len(df) > 0: df.to_csv(download_folder + file_name)

		if cash_flow:
			data = yf.get_financial_stmts('quarterly', 'cash', reformat=True)
			keys = list(data.keys())
			data = data[keys[0]] #'cashflowStatementHistoryQuarterly'
			df =  pd.DataFrame()
			for ticker in ticker_list:
				file_name = ticker + '_quarterly_cash-flow.csv'
				data_dictionary, dates = self._UnpackNestedDictionary(data[ticker])
				max_length = 0
				for key in data_dictionary.keys():
					if len(data_dictionary[key]) > max_length: max_length = len(data_dictionary[key])
					#print(key, len(data_dictionary[key]))
				incomplete_keys = []
				for key in data_dictionary.keys():
					if len(data_dictionary[key]) < max_length: incomplete_keys.append(key)
				for key in incomplete_keys: del data_dictionary[key]
				df = pd.DataFrame.from_dict(data_dictionary).apply(pd.to_numeric)
				dates = pd.Series(dates, name='Date')
				df = pd.concat([df, dates], axis=1)
				df.set_index('Date', inplace=True)
				df.sort_index(inplace=True)	
				print(df)
				if len(df) > 0: df.to_csv(download_folder + file_name)

	def DownloadPriceDataStooq(self, ticker:str, download_folder:str, verbose:bool=False):
		url = "https://stooq.com/q/d/l/?i=d&s=" + ticker + '.us'
		if ticker[0] == '^': 
			url = "https://stooq.com/q/d/l/?i=d&s=" + ticker 
		elif ticker == '.INX': 
			url = "https://stooq.com/q/d/l/?i=d&s=^SPX" #This isn't available anymore
		elif ticker == '.DJI': 
			url = "https://stooq.com/q/d/l/?i=d&s=^DJI"
		elif ticker == '.IXIC': 
			url = "https://stooq.com/q/d/l/?i=d&s=^ndq"
		elif "." in ticker:
			url = "https://stooq.com/q/d/l/?i=d&s=" + ticker.replace(".", "-") + '.us'
		filePath = download_folder + ticker + '.csv'
		s1 = ''
		if CreateFolder(download_folder): filePath = download_folder + ticker + '.csv'
		try:
			if useWebProxyServer:
				#opener = GetProxiedOpener()
				#openUrl = opener.open(url)
				proxySet, headerSet = GetWorkingProxy()
				openUrl = requests.get(url, headers=headerSet, proxy=proxySet)
			else:
				openUrl = webRequest.urlopen(url, timeout=45) 
			r = openUrl.read()
			openUrl.close()
			s1 = r.decode()
			s1 = s1.replace(chr(13),'')
		except Exception as e:
			if verbose: print(' Web connection error: ', e)
		if len(s1) < 1024:
			if verbose: print(' No data found online for ticker ' + ticker, url)
			if useWebProxyServer:
				proxySet, headerSet = GetWorkingProxy(True)
		else:
			if verbose: print(' Downloaded new data for ticker ' + ticker)
			f = open(filePath,'w')
			f.write(s1)
			f.close()

class PricingData:
	#Historical prices for a given stock, along with statistics, and future estimated prices
	ticker = ''
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
	_dataFolderhistoricalPrices = 'data/historical/'
	_dataFolderCharts = 'data/charts/'
	_dataFolderDailyPicks = 'data/dailypicks/'
	
	def __init__(self, ticker:str, dataFolderRoot:str='', useDatabase:bool=None):
		self.ticker = ticker
		if not dataFolderRoot =='':
			if CreateFolder(dataFolderRoot):
				if not dataFolderRoot[-1] =='/': dataFolderRoot += '/'
				self._dataFolderCharts = dataFolderRoot + 'charts/'
				self._dataFolderhistoricalPrices = dataFolderRoot + 'historical/'
				self._dataFolderDailyPicks = dataFolderRoot + 'dailypicks/'
		else: CreateFolder('data')
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
		self.database = None
		if useDatabase==None and globalUseDatabase:
			useDatabase = globalUseDatabase
			self.database = PTADatabase()
			if not self.database.Open():
				print("Default option to use database failed, database connection failed.")
				assert(False)
		elif useDatabase:
			self.database = PTADatabase()
		self.useDatabase = useDatabase	

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

	def _LoadHistory(self, refreshPrices:bool=False,verbose:bool=False):
		self.pricesLoaded = False
		filePath = self._dataFolderhistoricalPrices + self.ticker + '.csv'
		if (refreshPrices or (not os.path.isfile(filePath) and not self.useDatabase)): 
			dd = DataDownload()
			#dd.DownloadPriceDataStooq(self.ticker, self._dataFolderhistoricalPrices, verbose=verbose)
			dd.DownloadPriceDataYahooFinance([self.ticker], self._dataFolderhistoricalPrices)
			if os.path.isfile(filePath):
				df = pd.read_csv(filePath, index_col=0, parse_dates=True, na_values=['nan'])
				if (df.shape[0] > 0) and all([item in df.columns for item in base_field_list]): #Rows more than zero and base fields all exist
					df = df[base_field_list]
				self.pricesLoaded = len(df) > 1
				if self.useDatabase: self.LoadTickerFromCSVToSQL()

		if not self.pricesLoaded:
			if self.useDatabase:
				if self.database.Open():
					df = self.database.DataFrameFromSQL("SELECT [Date], [Open], [High], [Low], [Close], [Volume] FROM PricesDaily WHERE ticker='" + self.ticker + "' ORDER BY Date", indexName='Date')
					df.sort_index(inplace=True)	
					df = df[~df.index.duplicated()]
					self.pricesLoaded = len(df) > 1
					self.database.Close()
			elif os.path.isfile(filePath):
				df = pd.read_csv(filePath, index_col=0, parse_dates=True, na_values=['nan'])
				if (df.shape[0] > 0) and all([item in df.columns for item in base_field_list]): #Rows more than zero and base fields all exist
					df = df[base_field_list]
				self.pricesLoaded = len(df) > 1			
		if self.pricesLoaded:
			if (df['Open'] < df['Low']).any() or (df['Close'] < df['Low']).any() or (df['High'] < df['Low']).any() or (df['Open'] > df['High']).any() or (df['Close'] > df['High']).any(): 
				if verbose and False:
					print(self.ticker)
					print(df.loc[df['Low'] > df['High']])
					print(' Data validation error, Low > High.  Dropping values..')
				df = df.loc[df['Low'] <= df['High']]
				df = df.loc[df['Low'] <= df['Open']]
				df = df.loc[df['Low'] <= df['Close']]
				df = df.loc[df['High'] >= df['Open']]
				df = df.loc[df['High'] >= df['Close']]
			df['Average'] = (df['Open'] + df['Close'] + df['High'] + df['Low'])/4
			self.historicalPrices = df
			self.historyStartDate = self.historicalPrices.index.min()
			self.historyEndDate = self.historicalPrices.index.max()
			self.pricesLoaded = True
		if not self.pricesLoaded:
			print(' No data found for ' + self.ticker)
			#badTickerLog = open(self._dataFolderhistoricalPrices + "badTickerLog.txt","a")
			#badTickerLog.write("'" + self.ticker + "',\n")
			#badTickerLog.close() 
		return self.pricesLoaded 
		
	def LoadHistory(self, requestedStartDate:datetime=None, requestedEndDate:datetime=None, verbose:bool=False):
		self._LoadHistory(refreshPrices=False, verbose=verbose)
		if self.pricesLoaded:
			requestNewData = False
			filePath = self._dataFolderhistoricalPrices + self.ticker + '.csv'
			lastUpdated = datetime.now() - timedelta(days=10950)
			if os.path.isfile(filePath): lastUpdated = datetime.fromtimestamp(os.path.getmtime(filePath))
			if DateDiffHours(lastUpdated, datetime.now()) > 12 and not suspendPriceLoads:	#Limit how many times per hour we refresh the data to prevent abusing the source
				if not(requestedEndDate==None): requestNewData = (requestNewData or (self.historyEndDate < requestedEndDate))
				if (requestedStartDate==None and requestedEndDate==None):
					requestNewData = (DateDiffDays(startDate=lastUpdated, endDate=datetime.now()) > 1)
			if requestNewData: 
				if verbose: print(' Requesting new data for ' + self.ticker + ' (requestedStart, historyStart, historyEnd, requestedEnd)', requestedStartDate, self.historyStartDate, self.historyEndDate, requestedEndDate)
				self._LoadHistory(refreshPrices=True, verbose=verbose)
				refreshSuccessfull = self.pricesLoaded
				if not(requestedStartDate==None): refreshSuccessfull = (requestedStartDate >= self.historyStartDate)
				if not(requestedEndDate==None): refreshSuccessfull = (refreshSuccessfull and (self.historyEndDate >= requestedEndDate))
				self.pricesLoaded = refreshSuccessfull
				if not(refreshSuccessfull) and True: print(' Data refresh failed for requested date range (requestedStart, historyStart, historyEnd, requestedEnd)' + self.ticker + ' (requestedStart, historyStart, historyEnd, requestedEnd)', requestedStartDate, self.historyStartDate, self.historyEndDate, requestedEndDate)
			if not(requestedStartDate==None): 
				if (requestedEndDate==None): requestedEndDate = self.historyEndDate
				self.TrimToDateRange(requestedStartDate, requestedEndDate)
		return(self.pricesLoaded)

	def TrimToDateRange(self,startDate:datetime, endDate:datetime):
		startDate = ToDateTime(startDate)
		startDate -= timedelta(days=750) #If we do not include earlier dates we can not calculate all the stats
		endDate = ToDateTime(endDate) + timedelta(days=10)
		self.historicalPrices = self.historicalPrices[(self.historicalPrices.index >= startDate) & (self.historicalPrices.index <= endDate)]
		self.historyStartDate = self.historicalPrices.index.min()
		self.historyEndDate = self.historicalPrices.index.max()
		#print(self.ticker, 'trim to ', startDate, endDate, self.historyStartDate, self.historyEndDate, len(self.historicalPrices))
		
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
			print(' Prices have been converted back from percentages.')
		else:
			self.CTPFactor = self.historicalPrices.iloc[0]
			self.historicalPrices = self.historicalPrices[['Open','Close','High','Low','Average']].pct_change(1)
			self.historicalPrices[:1] = 0
			if self.predictionsLoaded:
				self.pricePredictions = self.pricePredictions.pct_change(1)
			self.statsLoaded = False
			self.pricesInPercentages = True
			print(' Prices have been converted to percentage change from previous day.')
		
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
				self.pricePredictions[['Predicted_Low']] = (self.pricePredictions[['Predicted_Low']]-low)/diff
				self.pricePredictions[['estAverage']] = (self.pricePredictions[['estAverage']]-low)/diff
				self.pricePredictions[['Predicted_High']] = (self.pricePredictions[['Predicted_High']]-low)/diff
			self.PreNormalizationLow = low
			self.PreNormalizationHigh = high
			self.PreNormalizationDiff = diff
			self.pricesNormalized = True
			if verbose: print(' Prices have been normalized.')
		else:
			low = self.PreNormalizationLow
			high = self.PreNormalizationHigh 
			diff = self.PreNormalizationDiff
			x['Open'] = (x['Open'] * diff) + low
			x['Close'] = (x['Close'] * diff) + low
			x['High'] = (x['High'] * diff) + low
			x['Low'] = (x['Low'] * diff) + low
			if self.predictionsLoaded:
				self.pricePredictions[['Predicted_Low']] = (self.pricePredictions[['Predicted_Low']] * diff) + low
				self.pricePredictions[['estAverage']] = (self.pricePredictions[['estAverage']] * diff) + low
				self.pricePredictions[['Predicted_High']] = (self.pricePredictions[['Predicted_High']] * diff) + low
			self.pricesNormalized = False
			if verbose: print(' Prices have been un-normalized.')
		x['Average'] = (x['Open'] + x['Close'] + x['High'] + x['Low'])/4
		#x['Average'] = x.loc[:,base_field_list].mean(axis=1, skipna=True) #Wow, this doesn't work.
		if (x['Average'] < x['Low']).any() or (x['Average'] > x['High']).any(): 
			print(x.loc[x['Average'] < x['Low']])
			print(x.loc[x['Average'] > x['High']])
			print(x.loc[x['Low'] > x['High']])
			print(self.PreNormalizationLow, self.PreNormalizationHigh, self.PreNormalizationDiff, self.PreNormalizationHigh-self.PreNormalizationLow)
			#print(x.loc[:,base_field_list].mean(axis=1))
			print('Stop: averages not computed correctly.')
			assert(False)
		self.historicalPrices = x
		if self.statsLoaded: self.CalculateStats()
		if verbose: print(self.historicalPrices[:1])

	def CalculateStats(self):
		if not self.pricesLoaded: self.LoadHistory()
		self.historicalPrices['Average_2Day'] = self.historicalPrices['Average'].rolling(window=2, center=False).mean()
		self.historicalPrices['Average_5Day'] = self.historicalPrices['Average'].rolling(window=5, center=False).mean() #5 day moving average, eliminates noise daily price changes
		self.historicalPrices['PC_1Day'] = (self.historicalPrices['Average'] / (self.historicalPrices['Average'].shift(1))-1)*250    #1 day price change, annualized
		self.historicalPrices['PC_3Day'] = (self.historicalPrices['Average'] / (self.historicalPrices['Average'].shift(3))-1)*83.33  #3 day price change, annualized
		self.historicalPrices['PC_1Month'] = (self.historicalPrices['Average_5Day'] / (self.historicalPrices['Average_5Day'].shift(20))-1)*12.5  #1 month price change, based on 5 day moving average, annualized
		self.historicalPrices['PC_1Month3WeekEMA'] = self.historicalPrices['PC_1Month'].ewm(span=15, adjust=True, ignore_na=False).mean() #3 week EMA of PC_1Month, performance degradation of short re-eval periods likely due to noise in PC_1Month
		self.historicalPrices['PC_2Month'] = (self.historicalPrices['Average_5Day'] / (self.historicalPrices['Average_5Day'].shift(41))-1)*6.097  #2 month price change, annualized
		self.historicalPrices['PC_3Month'] = (self.historicalPrices['Average_5Day'] / (self.historicalPrices['Average_5Day'].shift(62))-1)*4.03  #3 month price change, annualized
		self.historicalPrices['PC_6Month'] = (self.historicalPrices['Average_5Day'] / (self.historicalPrices['Average_5Day'].shift(125))-1)*2 #6 month price change, annualized
		self.historicalPrices['PC_1Year'] = (self.historicalPrices['Average_5Day'] / (self.historicalPrices['Average_5Day'].shift(250))-1)  #1 year price change, 250 periods
		self.historicalPrices['PC_18Month'] = (self.historicalPrices['Average_5Day'] / (self.historicalPrices['Average_5Day'].shift(375))-1) * 0.667  #1.5 year price change
		self.historicalPrices['PC_2Year'] = (self.historicalPrices['Average_5Day'] / (self.historicalPrices['Average_5Day'].shift(500))-1)/2  #2 year price change, annualized
		self.historicalPrices['EMA_12Day'] =  self.historicalPrices['Average'].ewm(span=12, adjust=True, ignore_na=False).mean()
		self.historicalPrices['EMA_26Day'] =  self.historicalPrices['Average'].ewm(span=26, adjust=True, ignore_na=False).mean()
		self.historicalPrices['MACD_Line'] =  self.historicalPrices['EMA_12Day'] - self.historicalPrices['EMA_26Day']
		self.historicalPrices['MACD_Signal'] =  self.historicalPrices['MACD_Line'].ewm(span=9, adjust=True, ignore_na=False).mean()
		self.historicalPrices['MACD_Histogram'] =  self.historicalPrices['MACD_Line'] - self.historicalPrices['MACD_Signal']
		self.historicalPrices['EMA_1Month'] =  self.historicalPrices['Average'].ewm(span=21, adjust=True, ignore_na=False).mean()
		self.historicalPrices['EMA_3Month'] =  self.historicalPrices['Average'].ewm(span=62, adjust=True, ignore_na=False).mean()
		self.historicalPrices['EMA_6Month'] =  self.historicalPrices['Average'].ewm(span=126, adjust=True, ignore_na=False).mean()
		self.historicalPrices['EMA_1Year'] =  self.historicalPrices['Average'].ewm(span=252, adjust=True, ignore_na=False).mean()
		self.historicalPrices['EMA_Short'] =  self.historicalPrices['EMA_1Month']
		self.historicalPrices['EMA_ShortSlope'] = (self.historicalPrices['EMA_Short']/self.historicalPrices['EMA_Short'].shift(1))-1
		self.historicalPrices['EMA_Long'] = self.historicalPrices['EMA_1Year']
		self.historicalPrices['EMA_LongSlope'] = (self.historicalPrices['EMA_Long']/self.historicalPrices['EMA_Long'].shift(1))-1
		self.historicalPrices['Deviation_1Day'] = (self.historicalPrices['High'] - self.historicalPrices['Low'])/self.historicalPrices['Low']
		self.historicalPrices['Deviation_5Day'] = self.historicalPrices['Deviation_1Day'].rolling(window=5, center=False).mean()
		self.historicalPrices['Deviation_10Day'] = self.historicalPrices['Deviation_1Day'].rolling(window=10, center=False).mean()
		self.historicalPrices['Deviation_15Day'] = self.historicalPrices['Deviation_1Day'].rolling(window=15, center=False).mean()
		self.historicalPrices['Gain_Monthly'] = (self.historicalPrices['Average_5Day'] / self.historicalPrices['Average_5Day'].shift(20))-1
		self.historicalPrices['Gain_Monthly'] = self.historicalPrices['Gain_Monthly'].replace(np.NaN, 0) #test not sure what getting NaN here
		self.historicalPrices['Losses_Monthly'] = self.historicalPrices['Gain_Monthly']
		self.historicalPrices['Losses_Monthly'].loc[self.historicalPrices['Losses_Monthly'] > 0] = 0 #zero out the positives
		self.historicalPrices['LossStd_1Year'] = self.historicalPrices['Losses_Monthly'].rolling(window=252, center=False).std()	#Stdev of negative values, these are the negative monthly price drops in the past year
		self.historicalPrices['Channel_High'] = self.historicalPrices['EMA_Long'] + (self.historicalPrices['Average']*self.historicalPrices['Deviation_15Day'])
		self.historicalPrices['Channel_Low'] = self.historicalPrices['EMA_Long'] - (self.historicalPrices['Average']*self.historicalPrices['Deviation_15Day'])
		self.historicalPrices.fillna(method='ffill', inplace=True)
		self.historicalPrices.fillna(method='bfill', inplace=True)
		self.statsLoaded = True
		return True

	def MonthyReturnVolatility(self): return self.historicalPrices['Gain_Monthly'].rolling(window=253, center=False).std() #of the past year

	def SaveStatsToFile(self, includePredictions:bool=False, verbose:bool=False):
		fileName = self.ticker + '_stats.csv'
		tableName = 'PricesDailyWithStats'
		r = self.historicalPrices
		if includePredictions:
			fileName = self.ticker + '_stats_predictions.csv'
			tableName = 'PricesDailyWithPredictions'
			r = self.historicalPrices.join(self.pricePredictions, how='outer') #, rsuffix='_Predicted'		

		if self.useDatabase:
			if self.database.Open():
				self.database.ExecSQL("if OBJECT_ID('" + tableName + "') is not null Delete FROM " + tableName + " WHERE Ticker='" + self.ticker + "'")
				r['Ticker'] = self.ticker
				self.database.DataFrameToSQL(r, tableName, indexAsColumn=True)
				self.database.Close()
				print('Statistics saved to database: ' + self.ticker)
		else:			
			filePath = self._dataFolderhistoricalPrices + fileName
			r.to_csv(filePath)
			print('Statistics saved to: ' + filePath)
		
	def PredictPrices(self, method:int=1, daysIntoFuture:int=1, NNTrainingEpochs:int=0):
		#Predict current prices from previous days info
		self.predictionsLoaded = False
		self.pricePredictions = pd.DataFrame()	#Clear any previous data
		if not self.statsLoaded: self.CalculateStats()
		if method < 3:
			minActionableSlope = 0.001
			if method==0:	#Same as previous day
				self.pricePredictions = pd.DataFrame()
				self.pricePredictions['Predicted_Low'] =  self.historicalPrices['Low'].shift(1)
				self.pricePredictions['Predicted_High'] = self.historicalPrices['High'].shift(1)
			elif method==1 :	#Slope plus momentum with some consideration for trend.
					#++,+-,-+,==
				bucket = self.historicalPrices.copy()
				bucket['Predicted_Low']  = bucket['Average'].shift(1) * (1-bucket['Deviation_15Day']/2) + (abs(bucket['PC_1Day'].shift(1)))
				bucket['Predicted_High'] = bucket['Average'].shift(1) * (1+bucket['Deviation_15Day']/2) + (abs(bucket['PC_1Day'].shift(1)))
				bucket = bucket.query('EMA_LongSlope >= -' + str(minActionableSlope) + ' or EMA_ShortSlope >= -' + str(minActionableSlope)) #must filter after rolling calcuations
				bucket = bucket[['Predicted_Low','Predicted_High']]
				self.pricePredictions = bucket
					#-- downward trend
				bucket = self.historicalPrices.copy()
				bucket['Predicted_Low'] = bucket['Low'].shift(1).rolling(3).min() * .99
				bucket['Predicted_High'] = bucket['High'].shift(1).rolling(3).min() 
				bucket = bucket.query('not (EMA_LongSlope >= -' + str(minActionableSlope) + ' or EMA_ShortSlope >= -' + str(minActionableSlope) +')')
				bucket = bucket[['Predicted_Low','Predicted_High']]
				self.pricePredictions = self.pricePredictions.append(bucket)
				self.pricePredictions.sort_index(inplace=True)	
			elif method==2:	#Slope plus momentum with full consideration for trend.
					#++ Often over bought, strong momentum
				bucket = self.historicalPrices.copy() 
				#bucket['Predicted_Low']  = bucket['Low'].shift(1).rolling(4).max()  * (1 + abs(bucket['EMA_ShortSlope'].shift(1)))
				#bucket['Predicted_High'] = bucket['High'].shift(1).rolling(4).max()  * (1 + abs(bucket['EMA_ShortSlope'].shift(1)))
				bucket['Predicted_Low']  = bucket['Low'].shift(1).rolling(4).max()  + (abs(bucket['PC_1Day'].shift(1)))
				bucket['Predicted_High'] = bucket['High'].shift(1).rolling(4).max() + (abs(bucket['PC_1Day'].shift(1)))
				bucket = bucket.query('EMA_LongSlope >= ' + str(minActionableSlope) + ' and EMA_ShortSlope >= ' + str(minActionableSlope)) #must filter after rolling calcuations
				bucket = bucket[['Predicted_Low','Predicted_High']]
				self.pricePredictions = bucket
					#+- correction or early down turn, loss of momentum
				bucket = self.historicalPrices.copy()
				bucket['Predicted_Low']  = bucket['Low'].shift(1).rolling(2).min() 
				bucket['Predicted_High'] = bucket['High'].shift(1).rolling(3).max()  * (1.005 + abs(bucket['EMA_ShortSlope'].shift(1)))
				bucket = bucket.query('EMA_LongSlope >= ' + str(minActionableSlope) + ' and EMA_ShortSlope < -' + str(minActionableSlope))
				bucket = bucket[['Predicted_Low','Predicted_High']]
				self.pricePredictions = self.pricePredictions.append(bucket)
					 #-+ bounce or early recovery, loss of momentum
				bucket = self.historicalPrices.copy()
				bucket['Predicted_Low']  = bucket['Low'].shift(1)
				bucket['Predicted_High'] = bucket['High'].shift(1).rolling(3).max() * 1.02 
				bucket = bucket.query('EMA_LongSlope < -' + str(minActionableSlope) + ' and EMA_ShortSlope >= ' + str(minActionableSlope))
				bucket = bucket[['Predicted_Low','Predicted_High']]
					#-- Often over sold
				self.pricePredictions = self.pricePredictions.append(bucket)
				bucket = self.historicalPrices.copy() 
				bucket['Predicted_Low'] = bucket['Low'].shift(1).rolling(3).min() * .99
				bucket['Predicted_High'] = bucket['High'].shift(1).rolling(3).min() 
				bucket = bucket.query('EMA_LongSlope < -' + str(minActionableSlope) + ' and EMA_ShortSlope < -' + str(minActionableSlope))
				bucket = bucket[['Predicted_Low','Predicted_High']]
				self.pricePredictions = self.pricePredictions.append(bucket)
					#== no significant slope
				bucket = self.historicalPrices.copy() 
				bucket['Predicted_Low']  = bucket['Low'].shift(1).rolling(4).mean()
				bucket['Predicted_High'] = bucket['High'].shift(1).rolling(4).mean()
				bucket = bucket.query(str(minActionableSlope) + ' > EMA_LongSlope >= -' + str(minActionableSlope) + ' or ' + str(minActionableSlope) + ' > EMA_ShortSlope >= -' + str(minActionableSlope))
				bucket = bucket[['Predicted_Low','Predicted_High']]
				self.pricePredictions = self.pricePredictions.append(bucket)
				self.pricePredictions.sort_index(inplace=True)	
			d = self.historicalPrices.index[-1] 
			ls = self.historicalPrices['EMA_LongSlope'][-1]
			ss = self.historicalPrices['EMA_ShortSlope'][-1]
			deviation = self.historicalPrices['Deviation_15Day'][-1]/2
			momentum = self.historicalPrices['PC_3Day'][-1]/2 
			for i in range(0,daysIntoFuture): 	#Add new days to the end for crystal ball predictions
				momentum = (momentum + ls)/2 * (100+random.randint(-3,4))/100
				a = (self.pricePredictions['Predicted_Low'][-1] + self.pricePredictions['Predicted_High'][-1])/2
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
				self.pricePredictions.loc[d + BDay(i+1), 'Predicted_Low'] = l
				self.pricePredictions.loc[d + BDay(i+1), 'Predicted_High'] = h								
			self.pricePredictions['estAverage']	= (self.pricePredictions['Predicted_Low'] + self.pricePredictions['Predicted_High'])/2
		elif method==3:	#Use LSTM to predict prices
			from _classes.SeriesPrediction import StockPredictionNN
			temporarilyNormalize = False
			if not self.pricesNormalized:
				temporarilyNormalize = True
				self.NormalizePrices()
			model = StockPredictionNN(base_model_name='Prices', model_type='LSTM')
			field_list = ['Average']
			#field_list = base_field_list
			model.LoadSource(sourceDF=self.historicalPrices, field_list=field_list, time_steps=1)
			model.LoadTarget(targetDF=None, prediction_target_days=daysIntoFuture)
			model.MakeTrainTest(batch_size=32, train_test_split=.93)
			model.BuildModel()
			if (not model.Load() and NNTrainingEpochs == 0): NNTrainingEpochs = 250
			if (NNTrainingEpochs > 0): 
				model.Train(epochs=NNTrainingEpochs)
				model.Save()
			model.Predict(True)
			self.pricePredictions = model.GetTrainingResults(False, False)
			self.pricePredictions = self.pricePredictions.rename(columns={'Average':'estAverage'})
			deviation = self.historicalPrices['Deviation_15Day'][-1]/2
			self.pricePredictions['Predicted_Low'] = self.pricePredictions['estAverage'] * (1 - deviation)
			self.pricePredictions['Predicted_High'] = self.pricePredictions['estAverage'] * (1 + deviation)
			if temporarilyNormalize: 
				self.predictionsLoaded = True
				self.NormalizePrices()
		elif method==4:	#Use CNN to predict prices
			from _classes.SeriesPrediction import StockPredictionNN
			temporarilyNormalize = False
			if not self.pricesNormalized:
				temporarilyNormalize = True
				self.NormalizePrices()
			model = StockPredictionNN(base_model_name='Prices', model_type='CNN')
			field_list = base_field_list
			model.LoadSource(sourceDF=self.historicalPrices, field_list=field_list, time_steps=daysIntoFuture*16)
			model.LoadTarget(targetDF=None, prediction_target_days=daysIntoFuture)
			model.MakeTrainTest(batch_size=32, train_test_split=.93)
			model.BuildModel()
			if (not model.Load() and NNTrainingEpochs == 0): NNTrainingEpochs = 250
			if (NNTrainingEpochs > 0): 
				model.Train(epochs=NNTrainingEpochs)
				model.Save()
			model.Predict(True)
			self.pricePredictions = model.GetTrainingResults(False, False)
			self.pricePredictions = self.pricePredictions.rename(columns={'Average':'estAverage'})
			deviation = self.historicalPrices['Deviation_15Day'][-1]/2
			self.pricePredictions['Predicted_Low'] = self.pricePredictions['estAverage'] * (1 - deviation)
			self.pricePredictions['Predicted_High'] = self.pricePredictions['estAverage'] * (1 + deviation)
			self.pricePredictions = self.pricePredictions[['Predicted_Low','estAverage','Predicted_High']]
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
		fromDate=ToDateTime(fromDate)
		low,high,price,momentum,deviation = self.historicalPrices.loc[fromDate, ['Low','High','Average', 'PC_3Day','Deviation_15Day']]
		#print(p,m,s)
		if method==0:
			futureLow = low
			futureHigh = high
		else:  
			futureLow = price * (1 + daysForward * momentum) - (price * deviation/2)
			futureHigh = price * (1 + daysForward * momentum) + (price * deviation/2)
		return futureLow, futureHigh	

	def GetDateFromIndex(self,indexLocation:int):
		if indexLocation >= self.historicalPrices.shape[0]: indexLocation = self.historicalPrices.shape[0]-1
		d = self.historicalPrices.index.values[indexLocation]
		return d

	def GetPrice(self,forDate:datetime, verbose:bool=False):
		forDate = ToDateTime(forDate)
		try:
			i = self.historicalPrices.index.get_loc(forDate, method='ffill') #ffill will effectively look backwards for the first instance
			forDate = self.historicalPrices.index[i]
			r = self.historicalPrices.loc[forDate]['Average']			
		except Exception as e: 
			if verbose or True: 
				print(' Unable to get price for ' + self.ticker + ' on ' + str(forDate))	
				print(' ', self.historyStartDate, self.historyEndDate)
				print(e)
				print(self.historicalPrices.index.duplicated())
			r = 0
		#if r ==0:
		#	print(forDate)
		#	print(self.historicalPrices.loc[forDate])
		#	assert(False)
		return r
		
	def GetPriceData(self,forDate:datetime, field_list:list, verbose:bool=False):
		r = None
		forDate = ToDateTime(forDate)
		try:
			i = self.historicalPrices.index.get_loc(forDate, method='ffill') #ffill will effectively look backwards for the first instance
			forDate = self.historicalPrices.index[i]
			r = self.historicalPrices.loc[forDate, field_list]
		except Exception as e: 
			if verbose or True: 
				print(' Unable to get price for ' + self.ticker + ' on ' + str(forDate))	
				print(' ', self.historyStartDate, self.historyEndDate)
				print(e)
				print(self.historicalPrices.index.duplicated())
		return r.values

	def GetPriceSnapshot(self,forDate:datetime, verbose:bool=False, pvmethod:int=0):
	#Returns: high, low, open, close, average, date
	#if stats are already calculated then also poplulates:
	#	Average_2Day, Average_5Day, EMA_Short, EMA_ShortSlope, EMA_Long, EMA_LongSlope, Channel_High, Channel_Low, PC_1Day, Deviation_1Day, Deviation_5Day, Deviation_15Day, PC_1Day, Gain_Monthly, LossStd_1Year, Predicted_Low, Target_1Day, Predicted_High
	#	and uses some simple logic to populate Target_1Day
	#used in EvaluatePrices, EvaluateTradeModels, RoboTrader, and internally
	#Not used much: PC_1Day Deviation_1Day PC_1Day Gain_Monthly
		forDate = ToDateTime(forDate)
		sn = PriceSnapshot()
		sn.ticker = self.ticker
		sn.high, sn.low, sn.open, sn.close = 0,0,0,0
		try:
			i = self.historicalPrices.index.get_loc(forDate, method='ffill')
			forDate = self.historicalPrices.index[i]
		except:
			i = 0
			if verbose: print('Unable to get price snapshot for ' + self.ticker + ' on ' + str(forDate))	
		sn.date = forDate 
		if i > 0:
			if not self.statsLoaded:
				sn.high,sn.low,sn.open,sn.close,sn.average=self.historicalPrices.loc[forDate,['High','Low','Open','Close','Average']]
			else:
				sn.high,sn.low,sn.open,sn.close,sn.average,sn.Average_2Day,sn.Average_5Day,sn.EMA_Short,sn.EMA_ShortSlope,sn.EMA_Long,sn.EMA_LongSlope,sn.Channel_High,sn.Channel_Low = self.historicalPrices.loc[forDate,['High','Low','Open','Close','Average','Average_2Day','Average_5Day','EMA_Short','EMA_ShortSlope','EMA_Long','EMA_LongSlope','Channel_High', 'Channel_Low']]
				sn.PC_1Day,sn.Deviation_1Day,sn.Deviation_5Day,sn.Deviation_10Day,sn.Deviation_15Day,sn.Gain_Monthly,sn.LossStd_1Year,sn.PC_1Month,sn.PC_1Month3WeekEMA,sn.PC_2Month,sn.PC_3Month,sn.PC_6Month,sn.PC_1Year,sn.PC_18Month,sn.PC_2Year = self.historicalPrices.loc[forDate,['PC_1Day','Deviation_1Day','Deviation_5Day','Deviation_10Day','Deviation_15Day','Gain_Monthly','LossStd_1Year','PC_1Month','PC_1Month3WeekEMA','PC_2Month','PC_3Month','PC_6Month','PC_1Year','PC_18Month','PC_2Year']]
				sn.EMA_12Day,sn.EMA_26Day,sn.MACD_Line,sn.MACD_Signal,sn.MACD_Histogram = self.historicalPrices.loc[forDate,['EMA_12Day','EMA_26Day','MACD_Line','MACD_Signal','MACD_Histogram']]
				if pd.isna(sn.LossStd_1Year):
					print(sn.ticker, sn.PC_1Year, sn.PC_6Month, sn.PC_3Month, sn.PC_1Month, sn.LossStd_1Year)
					sn.LossStd_1Year = 0
				if pvmethod ==0:
					if sn.PC_1Month > 0: 
						try:
							sn.Point_Value = round((10*sn.PC_1Year) - (6-100*sn.LossStd_1Year))  #New golden: 44% average year -19% worst year, 10 stocks, improves if MarketCap < 3000
						except:
							print('Failed to calculate point value!')
							print(self.ticker, sn.PC_1Year, sn.LossStd_1Year, forDate)
				elif pvmethod ==1:
					sn.Point_Value = round((10*sn.PC_1Year) + (10*sn.PC_6Month) + (10*sn.PC_3Month) + (10*sn.PC_1Month) - (3-10*sn.LossStd_1Year)) #This was my golden, 36%	-32%
				elif pvmethod ==2:
					if sn.PC_1Month > 0: sn.Point_Value = round((10*sn.PC_1Year) + (100*sn.LossStd_1Year))  #38, -29 worse than above, filter 3 testing now 37% -46%
				elif pvmethod ==3:
					if sn.PC_1Month > 0: sn.Point_Value = round((3*sn.PC_18Month) + (12*sn.PC_1Year) + (4*sn.PC_6Month) + (3*sn.PC_3Month) + (2*sn.PC_1Month) - (3-10*sn.LossStd_1Year)) #38%, -33
					#if sn.PC_1Month > 0: sn.Point_Value = round(10*sn.PC_1Year)   #Testing now, 36% -31%
				if False: #37%, -35
					sn.Point_Value = sn.PC_1Year*.1  
					sn.Point_Value = sn.Point_Value + (sn.PC_2Year*.07)/2
					sn.Point_Value = sn.Point_Value + (sn.PC_3Month*.03)
					sn.Point_Value = sn.Point_Value + (sn.PC_1Month*.07)/2
					if sn.PC_6Month<0 and sn.PC_1Year> 0: sn.Point_Value = sn.PC_1Year*.22 
					if sn.LossStd_1Year >= .12 and sn.LossStd_1Year <= .15: sn.Point_Value = (sn.PC_1Month*.30)+(sn.PC_3Month*.18) 
					if sn.PC_1Year > 0 and (sn.LossStd_1Year >= .13 and sn.LossStd_1Year <= .15) and (sn.PC_3Month > 0 and sn.PC_1Month > 0): sn.Point_Value = sn.PC_1Year*.91 					

				if sn.Point_Value < 0: sn.Point_Value=0
				if sn.Point_Value > 100: sn.Point_Value=100
				#Parameter Testing, these all decrease performance by about 5%
				#sn.Point_Value = round((10*sn.PC_2Year) + (10*sn.PC_1Year6mo) + (10*sn.PC_1Year) + (10*sn.PC_6Month) + (10*sn.PC_3Month) + (10*sn.PC_1Month) - (3-10*sn.LossStd_1Year)) #-5% average yield
				#if (sn.PC_2Year < 0 or sn.sn.PC_1Year6mo < 0): sn.Point_Value = round((5*sn.PC_2Year) + (5*sn.PC_1Year6mo) + (10*sn.PC_1Year) + (10*sn.PC_6Month) + (10*sn.PC_3Month) + (10*sn.PC_1Month) - (3-10*sn.LossStd_1Year))  #-5% average yield
				#if (sn.PC_2Year > 0 and sn.sn.PC_1Year6mo > 0): sn.Point_Value = round((5*sn.PC_2Year) + (5*sn.PC_1Year6mo) + (10*sn.PC_1Year) + (10*sn.PC_6Month) + (10*sn.PC_3Month) + (10*sn.PC_1Month) - (3-10*sn.LossStd_1Year))  #-5% average yield
				#if (sn.PC_1Year > .3 and sn.PC2mo*6.0833 > .1 and sn.PC_1Month*12.1666 > .12):# This does NOT improve performance
				#	sn.Point_Value = round((5*sn.PC_1Year) + (5*sn.PC_6Month) + (7*sn.PC_3Month) + (13*sn.PC2mo) + (15*sn.PC_1Month) - (3-10*sn.LossStd_1Year))# This does NOT improve performance

				if sn.EMA_LongSlope < 0:
					if sn.EMA_ShortSlope > 0:	#bounce or early recovery
						sn.Target_1Day = min(sn.average, sn.Average_2Day)
					else:
						sn.Target_1Day = min(sn.low, sn.Average_2Day)			
				else:
					if sn.EMA_ShortSlope < 0:	#correction or early downturn
						sn.Target_1Day = max(sn.average, (sn.Average_2Day*2)-sn.average) + (sn.average * (sn.EMA_LongSlope))
					else:
						sn.Target_1Day = max(sn.average, sn.Average_2Day) + (sn.average * sn.EMA_LongSlope)
					#sn.Target_1Day = max(sn.average, sn.Average_2Day) + (sn.average * sn.EMA_LongSlope)
				sn.Comments = ''
				if sn.low > sn.Channel_High: 
					sn.Comments += 'Overbought; '
				if sn.high < sn.Channel_Low: 
					sn.Comments += 'Oversold; '
				if sn.Deviation_5Day > .0275: 
					sn.Comments += 'HighDeviation; '
				if not self.predictionsLoaded or forDate >= self.historyEndDate:
					sn.Predicted_Low,sn.Predicted_High= self.PredictFuturePrice(forDate,1)
				else:
					tomorrow =  forDate.date() + timedelta(days=1) 
					sn.Predicted_Low,sn.Predicted_High= self.pricePredictions.loc[tomorrow,['Predicted_Low','Predicted_High']]
		sn.average = sn.average
		return sn

	def GetCurrentPriceSnapshot(self): return self.GetPriceSnapshot(self.historyEndDate)

	def GetPriceHistory(self, field_list:list = None, includePredictions:bool = False):
		if field_list == None:
			r = self.historicalPrices.copy() #best to pass back copies instead of reference.
		else:
			r = self.historicalPrices[field_list].copy() #best to pass back copies instead of reference.			
		if includePredictions: r = r.join(self.pricePredictions, how='outer')
		return r
		
	def GetPricePredictions(self):
		return self.pricePredictions.copy()  #best to pass back copies instead of reference.

	def GraphData(self, endDate:datetime=None, daysToGraph:int=90, graphTitle:str=None, includePredictions:bool=False, saveToFile:bool=False, fileNameSuffix:str=None, saveToFolder:str='', dpi:int=600, trimHistoricalPredictions:bool = True):
		PlotSetDefaults()
		if not self.statsLoaded: self.CalculateStats()
		if includePredictions:
			if not self.predictionsLoaded: self.PredictPrices()
			if endDate == None: endDate = self.pricePredictions.index.max()
			endDate = ToDateTime(endDate)
			startDate = endDate - BDay(daysToGraph) 
			fieldSet = ['High','Low', 'Channel_High', 'Channel_Low', 'Predicted_High','Predicted_Low', 'EMA_Short','EMA_Long']
			if trimHistoricalPredictions: 
				y = self.pricePredictions[self.pricePredictions.index >= self.historyEndDate]
				x = self.historicalPrices.join(y, how='outer')
			else: 
				fieldSet = ['High','Low', 'Predicted_High','Predicted_Low']
				x = self.historicalPrices.join(self.pricePredictions, how='outer')
			if daysToGraph > 1800:	fieldSet = ['Average', 'Predicted_High','Predicted_Low']
		else:
			if endDate == None: endDate = self.historyEndDate
			endDate = ToDateTime(endDate)
			startDate = endDate - BDay(daysToGraph) 
			fieldSet = ['High','Low', 'Channel_High', 'Channel_Low','EMA_Short','EMA_Long']
			if daysToGraph > 1800: fieldSet = ['Average']
			x = self.historicalPrices.copy()
		if fileNameSuffix == None: fileNameSuffix = str(endDate)[:10] + '_' + str(daysToGraph) + 'days'
		if graphTitle==None: graphTitle = self.ticker + ' ' + fileNameSuffix
		x = x[(x.index >= startDate) & (x.index <= endDate)]
		if x.shape[0] == 0:
			print('Empty source data')
		else:
			ax=x.loc[startDate:endDate,fieldSet].plot(title=graphTitle, linewidth=.75, color = ['blue', 'red','purple','purple','mediumseagreen','seagreen'])			
			ax.set_xlabel('Date')
			ax.set_ylabel('Price')
			ax.tick_params(axis='x', rotation=70)
			ax.grid(b=True, which='major', color='black', linestyle='solid', linewidth=.5)
			ax.grid(b=True, which='minor', color='0.65', linestyle='solid', linewidth=.1)
			PlotScalerDateAdjust(startDate, endDate, ax)
			if saveToFile:
				if not fileNameSuffix =='': fileNameSuffix = '_' + fileNameSuffix
				if saveToFolder=='': saveToFolder = self._dataFolderCharts
				if not saveToFolder.endswith('/'): saveToFolder = saveToFolder + '/'
				if CreateFolder(saveToFolder): 	plt.savefig(saveToFolder + self.ticker + fileNameSuffix + '.png', dpi=dpi)
				print(' Saved to ' + saveToFolder + self.ticker + fileNameSuffix + '.png')
			else:
				plt.show()
			plt.close('all')
			
	def LoadTickerFromCSVToSQL(self):
		print(" Loading " + self.ticker + " CSV into SQL...")
		csvFile = self._dataFolderhistoricalPrices + self.ticker + '.CSV'
		if not os.path.isfile(csvFile):
			print(" File doesn't exist: " + csvFile)
		else:
			if self.database == None: self.database = PTADatabase()
			if self.database.Open():
				data = pd.read_csv(csvFile)# , index_col=0, parse_dates=True, na_values=['NaN']
				df = pd.DataFrame(data)	
				if df['Date'][0] > df['Date'][1]: #Make sure it is sorted ascending
					print('changing sort order')
					df.sort_values('Date', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last')
				if 'Adj Close' in df.columns: df.drop(columns=['Adj Close'], inplace=True, axis=1)
				df.fillna(method='ffill', inplace=True)
				df.fillna(method='bfill', inplace=True)
				startDate = df['Date'][0]
				sourceRecordCount = len(df)
				cursor = self.database.GetCursor()
				cursor.execute("DELETE FROM PricesDaily WHERE Ticker=? AND [Date]>=?", self.ticker, startDate)
				if True:
					df['Ticker'] = self.ticker
					self.database.DataFrameToSQL(df=df, tableName='PricesDaily', indexAsColumn=False, clearExistingData=False)
					#quoted = urllib.parse.quote_plus(SQL_ConString())
					#engine = create_engine('mssql+pyodbc:///?odbc_connect={}'.format(quoted))
					#df.to_sql('PricesDaily', schema='dbo', con = engine, if_exists='append', index=False)
				else:
					for row in df.itertuples():
						cursor.execute("INSERT INTO PricesDaily ([Ticker],[Date],[Open],[High],[Low],[Close],[Volume]) Values(?,?,?,?,?,?,?)", self.ticker, row.Date,row.Open,row.High, row.Low, row.Close, row.Volume)
				cursor.execute("SELECT COUNT(*) AS RecordCount FROM PricesDaily WHERE Ticker=?", self.ticker)
				for row in cursor.fetchall():
					destRecordCount = row.RecordCount
				self.database.Close()
				print("Imported source records: " + str(sourceRecordCount) + " Destination records: " + str(destRecordCount))
				if destRecordCount != sourceRecordCount and False:
					print("Import to SQL failed. Counts don't match")
					assert(False)

	def ExportFromSQLToCSV(self):
		needsUpdating = True
		csvFile = self._dataFolderhistoricalPrices + self.ticker + '.csv'
		if os.path.isfile(csvFile):
			minAgeToRefresh = datetime.now() - timedelta(hours=12)
			needsUpdating = (datetime.fromtimestamp(os.path.getmtime(csvFile)) < minAgeToRefresh)
		if needsUpdating:
			#print('Updating CSV for ' + self.ticker)
			if self.database != None:	
				if self.database.Open():
					cursor = self.database.GetCursor()
					SQL = "select [Date], [Open], [High], [Low], [Close], [Volume] from PricesDaily WHERE Ticker='" + self.ticker + "' ORDER By Date"
					#print("Exporting " + self.ticker + " to " + csvFile + " ...")
					df = self.database.DataFrameFromSQL(SQL)
					#print(df)
					df.to_csv(csvFile, index=False)
				else:
					print('No database connection')

class Tranche: #interface for handling actions on a chunk of funds
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
			if verbose: print(' Sell order on ', self.ticker, ' canceled.')
			self.dateSellOrderPlaced = None
			self.sellOrderPrice = 0
			self.expired=False
		else:
			if verbose: print(' Buy order for ', self.ticker, ' canceled.')
			self.Recycle()
		
	def Expire(self):
		if not self.purchased:  #cancel buy
			if self._verbose: print(' Buy order from ' + str(self.dateBuyOrderPlaced) + ' has expired (' + self.ticker + ')')
			self.Recycle()
		else: #cancel sell
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
			self.marketOrder = marketOrder
			self.dateBuyOrderPlaced = datePlaced
			self.buyOrderPrice=price
			self.units = floor(self.size/price)
			self.purchased = False
			self.expireAfterDays=expireAfterDays
			r=(price*self.units)
			if self._verbose: 
				if marketOrder:
					print(datePlaced, ' Buy placed at Market (' + str(price) + ') for ' + str(self.units) + ' Cost ' + str(r) + '(' + self.ticker + ')')
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
		r = False
		if price > 0: 
			self.latestPrice = price
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
	def __init__(self, t:Tranche):
		self._t = t
		self.ticker = t.ticker
	def CancelSell(self): 
		if self._t.purchased: self._t.CancelOrder(verbose=True)
	def CurrentValue(self): return self._t.units * self._t.latestPrice
	def Sell(self, price:float, datePlaced:datetime, marketOrder:bool=False, expireAfterDays:int=90): self._t.PlaceSell(price=price, datePlaced=datePlaced, marketOrder=marketOrder, expireAfterDays=expireAfterDays, verbose=True)
	def SellPending(self): return (self._t.sellOrderPrice >0) and not (self._t.sold or  self._t.expired)
	def LatestPrice(self): return self._t.latestPrice
	
class Portfolio:
	portfolioName = ''
	tradeHistory = None #DataFrame of trades.  Note: though you can trade more than once a day it is only going to keep one entry per day per stock
	dailyValue = None	  #DataFrame for the value at the end of each day
	_cash=0
	_fundsCommittedToOrders=0
	_commisionCost = 0
	_tranches = []			#Sets of funds for investing, rather than just a pool of cash.  Simplifies accounting.
	_tranchCount = 0
	_verbose = False
	
	def __del__(self):
		self._cash = 0
		self._tranches = None

	def __init__(self, portfolioName:str, startDate:datetime, totalFunds:int=10000, tranchSize:int=1000, trackHistory:bool=True, useDatabase:bool=None, verbose:bool=True):
		self.portfolioName = portfolioName
		self._cash = totalFunds
		self._fundsCommittedToOrders = 0
		self._verbose = verbose
		self._tranchCount = floor(totalFunds/tranchSize)
		self._tranches = [Tranche(tranchSize) for x in range(self._tranchCount)]
		self.dailyValue = pd.DataFrame([[startDate,totalFunds,0,totalFunds,'','','','','','','','','','','']], columns=list(['Date','CashValue','AssetValue','TotalValue','Stock00','Stock01','Stock02','Stock03','Stock04','Stock05','Stock06','Stock07','Stock08','Stock09','Stock10']))
		self.dailyValue.set_index(['Date'], inplace=True)
		if useDatabase==None and globalUseDatabase:
			useDatabase = globalUseDatabase
			self.database = PTADatabase()
			if not self.database.Open():
				print("Default option to use database failed, database connection failed.")
				assert(False)
		elif useDatabase:
			self.database = PTADatabase()
		self.useDatabase = useDatabase
		self.trackHistory = trackHistory
		if trackHistory: 
			self.tradeHistory = pd.DataFrame(columns=['dateBuyOrderPlaced','ticker','dateBuyOrderFilled','dateSellOrderPlaced','dateSellOrderFilled','units','buyOrderPrice','purchasePrice','sellOrderPrice','sellPrice','NetChange'])
			self.tradeHistory.set_index(['dateBuyOrderPlaced','ticker'], inplace=True)

	#----------------------  Status and position info  ---------------------------------------
	def AccountingError(self):
		r = False
		if not self.ValidateFundsCommittedToOrders() == 0: 
			print(' Accounting error: inaccurcy in funds committed to orders!')
			r=True
		if self.FundsAvailable() + self._tranchCount*self._commisionCost < -10: #Over-committed funds		
			OrdersAdjusted = False		
			for t in self._tranches:
				if not t.purchased and t.units > 0:
					print(' Reducing purchase of ' + t.ticker + ' by one unit due to overcommitted funds.')
					t.units -= 1
					OrdersAdjusted = True
					break
			if OrdersAdjusted: self.ValidateFundsCommittedToOrders(True)
			if self.FundsAvailable() + self._tranchCount*self._commisionCost < -10: 
				OrdersAdjusted = False		
				for t in self._tranches:
					if not t.purchased and t.units > 1:
						print(' Reducing purchase of ' + t.ticker + ' by two units due to overcommitted funds.')
						t.units -= 2
						OrdersAdjusted = True
						break
			if self.FundsAvailable() + self._tranchCount*self._commisionCost < -10: #Over-committed funds						
				print(' Accounting error: negative cash balance.  (Cash, CommittedFunds, AvailableFunds) ', self._cash, self._fundsCommittedToOrders, self.FundsAvailable())
				r=True
		return r

	def FundsAvailable(self): return (self._cash - self._fundsCommittedToOrders)
	
	def PendingOrders(self):
		a, b, s, l = self.PositionSummary()
		return (b+s > 0)

	def GetPositions(self, ticker:str='', asDataFrame:bool=False):	#returns reference to the tranche of active positions or a dataframe with counts
		r = []
		for t in self._tranches:
			if t.purchased and (t.ticker==ticker or ticker==''): 
				p = Position(t)
				r.append(p)
		if asDataFrame:
			y=[]
			for x in r: y.append(x.ticker)
			r = pd.DataFrame(y,columns=list(['Ticker']))
			r = r.groupby(['Ticker']).size().reset_index(name='CurrentHoldings')
			r.set_index(['Ticker'], inplace=True)
			TotalHoldings = r['CurrentHoldings'].sum()
			r['Percentage'] = r['CurrentHoldings']/TotalHoldings
		return r

	def PositionSummary(self):
		available=0
		buyPending=0
		sellPending=0
		longPostition = 0
		for t in self._tranches:
			if t.available:
				available +=1
			elif not t.purchased:
				buyPending +=1
			elif t.purchased and t.dateSellOrderPlaced==None:
				longPostition +=1
			elif t.dateSellOrderPlaced:
				sellPending +=1
		return available, buyPending, sellPending, longPostition			

	def PrintPositions(self):
		i=0
		for t in self._tranches:
			if not t.ticker =='' or True:
				print('Set: ' + str(i))
				t.PrintDetails()
			i=i+1
		print('Funds committed to orders: ' + str(self._fundsCommittedToOrders))
		print('available funds: ' + str(self._cash - self._fundsCommittedToOrders))

	def TranchesAvailable(self):
		a, b, s, l = self.PositionSummary()
		return a

	def ValidateFundsCommittedToOrders(self, SaveAdjustments:bool=True):
		#Returns difference between recorded value and actual
		x=0
		for t in self._tranches:
			if not t.available and not t.purchased: 
				x = x + (t.units*t.buyOrderPrice) + self._commisionCost
		if round(self._fundsCommittedToOrders, 5) == round(x,5): self._fundsCommittedToOrders=x
		if not (self._fundsCommittedToOrders - x) ==0:
			if SaveAdjustments: 
				self._fundsCommittedToOrders = x
			else:
				print( 'Committed funds variance actual/recorded', x, self._fundsCommittedToOrders)
		return (self._fundsCommittedToOrders - x)

	def Value(self):
		assetValue=0
		for t in self._tranches:
			if t.purchased:
				assetValue = assetValue + (t.units*t.latestPrice)
		return self._cash, assetValue
		
	def ReEvaluateTrancheCount(self, verbose:bool=False):
		#Portfolio performance may require adjusting the available Tranches
		tranchSize = self._tranches[0].size
		c = self._tranchCount
		availableTranches,_,_,_ = self.PositionSummary()
		availableFunds = self._cash - self._fundsCommittedToOrders
		targetAvailable = int(availableFunds/tranchSize)
		if targetAvailable > availableTranches:
			if verbose: 
				print(' Available Funds: ', availableFunds, availableTranches * tranchSize)
				print(' Adding ' + str(targetAvailable - availableTranches) + ' new Tranches to portfolio..')
			for i in range(targetAvailable - availableTranches):
				self._tranches.append(Tranche(tranchSize))
				self._tranchCount +=1
		elif targetAvailable < availableTranches:
			if verbose: print( 'Removing ' + str(availableTranches - targetAvailable) + ' tranches from portfolio..')
			#print(targetAvailable, availableFunds, tranchSize, availableTranches)
			i = self._tranchCount-1
			while i > 0:
				if self._tranches[i].available and targetAvailable < availableTranches:
					if verbose: 
						print(' Available Funds: ', availableFunds, availableTranches * tranchSize)
						print(' Removing tranch at ', i)
					self._tranches.pop(i)	#remove last available
					self._tranchCount -=1
					availableTranches -=1
				i -=1


	#--------------------------------------  Order interface  ---------------------------------------
	def CancelAllOrders(self, currentDate:datetime):
		for t in self._tranches:
			t.CancelOrder()
		#for t in self._tranches:						self.CheckOrders(t.ticker, t.latestPrice, currentDate) 

	def PlaceBuy(self, ticker:str, price:float, datePlaced:datetime, marketOrder:bool=False, expireAfterDays:int=10, verbose:bool=False):
		#Place with first available tranch, returns True if order was placed
		r=False
		price = round(price, 3)
		oldestExistingOrder = None
		availableCash = self.FundsAvailable()
		units=0
		if price > 0: units = int(self._tranches[0].size/price)
		cost = units*price + self._commisionCost
		if availableCash < cost and units > 2:
			units -=1
			cost = units*price + self._commisionCost
		if units == 0 or availableCash < cost:
			if verbose: 
				if price==0:
					print( 'Unable to purchase ' + ticker + '.  Price lookup failed.', datePlaced)
				else:
					print( 'Unable to purchase ' + ticker + '.  Price (' + str(price) + ') exceeds available funds ' + str(availableCash) + ' Traunche Size: ' + str(self._tranches[0].size))
		else:	
			for t in self._tranches: #Find available 
				if t.available :	#Place new order
					self._fundsCommittedToOrders = self._fundsCommittedToOrders + cost 
					x = self._commisionCost + t.PlaceBuy(ticker=ticker, price=price, datePlaced=datePlaced, marketOrder=marketOrder, expireAfterDays=expireAfterDays, verbose=verbose) 
					if not x == cost: #insufficient funds for full purchase
						if verbose: print(' Expected cost changed from', cost, 'to', x)
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
				if self.TranchesAvailable() > 0:
					if verbose: print(' Unable to buy ' + str(units) + ' of ' + ticker + ' with funds available: ' + str(FundsAvailable))
				else: 
					if verbose: print(' Unable to buy ' + ticker + ' no tranches available')
			else:
				for t in self._tranches:
					if not t.purchased and t.ticker == ticker and oldestExistingOrder==t.dateBuyOrderPlaced:
						if verbose: print(' No tranch available... replacing order from ' + str(oldestExistingOrder))
						oldCost = t.buyOrderPrice * t.units + self._commisionCost
						if verbose: print(' Replacing Buy order for ' + ticker + ' from ' + str(t.buyOrderPrice) + ' to ' + str(price))
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
		for t in self._tranches:
			if t.ticker == ticker and t.purchased and t.sellOrderPrice==0 and (datepurchased is None or t.dateBuyOrderFilled == datepurchased):
				t.PlaceSell(price=price, datePlaced=datePlaced, marketOrder=marketOrder, expireAfterDays=expireAfterDays, verbose=verbose)
				r=True
				break
		if not r:	#couldn't find one without a sell, try to update an existing sell order
			for t in self._tranches:
				if t.ticker == ticker and t.purchased:
					if verbose: print(' Updating existing sell order ')
					t.PlaceSell(price=price, datePlaced=datePlaced, marketOrder=marketOrder, expireAfterDays=expireAfterDays, verbose=verbose)
					r=True
					break					
		return r

	def SellAllPositions(self, datePlaced:datetime, ticker:str='', verbose:bool=False, allowWeekEnd:bool=False):
		for t in self._tranches:
			if t.purchased and (t.ticker==ticker or ticker==''): 
				t.PlaceSell(price=t.latestPrice, datePlaced=datePlaced, marketOrder=True, expireAfterDays=5, verbose=verbose)
		self.ProcessDay(withIncrement=False, allowWeekEnd=allowWeekEnd)

	#--------------------------------------  Order Processing ---------------------------------------
	def _CheckOrders(self, ticker, price, dateChecked):
		#check if there was action on any pending orders and update current price of tranche
		price = round(price, 3)
		self._verbose = True
		for t in self._tranches:
			if t.ticker == ticker:
				r = t.UpdateStatus(price, dateChecked)
				if r:	#Order was filled, update account
					if t.expired:
						if self._verbose: print(t.ticker, " expired ", dateChecked)
						if not t.purchased: 
							self._fundsCommittedToOrders -= (t.units*t.buyOrderPrice)	#return funds committed to order
							self._fundsCommittedToOrders -= self._commisionCost
						t.Expire()
					elif t.sold:
						if self._verbose: print(t.ticker, " sold for ",t.sellPrice, dateChecked)
						self._cash = self._cash + (t.units*t.sellPrice) - self._commisionCost
						if self._verbose and self._commisionCost > 0: print(' Commission charged for Sell: ' + str(self._commisionCost))
						if self.trackHistory:
							self.tradeHistory.loc[(t.dateBuyOrderPlaced, t.ticker)]=[t.dateBuyOrderFilled,t.dateSellOrderPlaced,t.dateSellOrderFilled,t.units,t.buyOrderPrice,t.purchasePrice,t.sellOrderPrice,t.sellPrice,((t.sellPrice - t.purchasePrice)*t.units)-self._commisionCost*2] 
						t.Recycle()
					elif t.purchased:
						self._fundsCommittedToOrders -= (t.units*t.buyOrderPrice)	#return funds committed to order
						self._fundsCommittedToOrders -= self._commisionCost
						fundsavailable = self._cash - abs(self._fundsCommittedToOrders)
						if t.marketOrder:
							actualCost = t.units*price
							if self._verbose: print(t.ticker, " purchased for ",price, dateChecked)
							if (fundsavailable - actualCost - self._commisionCost) < 25:	#insufficient funds
								unitsCanAfford = max(floor((fundsavailable - self._commisionCost)/price)-1, 0)
								if self._verbose:
									print(' Ajusting units on market order for ' + ticker + ' Price: ', price, ' Requested Units: ', t.units,  ' Can afford:', unitsCanAfford)
									print(' Cash: ', self._cash, ' Committed Funds: ', self._fundsCommittedToOrders, ' Available: ', fundsavailable)
								if unitsCanAfford ==0:
									t.Recycle()
								else:
									t.AdjustBuyUnits(unitsCanAfford)
						if t.units == 0:
							if self._verbose: print( 'Can not afford any ' + ticker + ' at market ' + str(price) + ' canceling Buy', dateChecked)
							t.Recycle()
						else:
							self._cash = self._cash - (t.units*price) - self._commisionCost 
							if self._verbose and self._commisionCost > 0: print(' Commission charged for Buy: ' + str(self._commisionCost))		
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
		#approximate a sequence of the day's prices for given ticker, check orders for each, update price value
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

	def UpdateDailyValue(self):
		_cashValue, assetValue = self.Value()
		positions = self.GetPositions(asDataFrame=True)
		x = positions.index.to_numpy() + ':' + positions['Percentage'].to_numpy(dtype=str)
		for i in range(len(x)): x[i] = x[i][:12]
		while len(x) < 11: x = numpy.append(x, [''])
		self.dailyValue.loc[self.currentDate]=[_cashValue,assetValue,_cashValue + assetValue, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10]]
		#print(self.dailyValue)

	#--------------------------------------  Closing Reporting ---------------------------------------
	def SaveTradeHistoryToFile(self, foldername:str, addTimeStamp:bool = False):
		if self.trackHistory:
			if self.useDatabase:
				if self.database.Open():
					df = self.tradeHistory
					df['TradeModel'] = self.portfolioName 
					self.database.DataFrameToSQL(df, 'TradeModel_Trades', indexAsColumn=True)
					self.database.Close()
			elif CreateFolder(foldername):
				filePath = foldername + self.portfolioName 
				if addTimeStamp: filePath += '_' + GetDateTimeStamp()
				filePath += '_trades.csv'
				self.tradeHistory.to_csv(filePath)

	def SaveDailyValueToFile(self, foldername:str, addTimeStamp:bool = False):
		if self.useDatabase:
			if self.database.Open():
				df = self.dailyValue.copy()
				df['TradeModel'] = self.portfolioName 
				self.database.DataFrameToSQL(df, 'TradeModel_DailyValue', indexAsColumn=True)
				self.database.Close()
		elif CreateFolder(foldername):
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
	priceHistory = []  #list of price histories for each stock in _tickerList
	startingValue = 0 
	verbose = False
	_tickerList = []	#list of stocks currently held
	_dataFolderTradeModel = 'data/trademodel/'
	Custom1 = None	#can be used to store custom values when using the model
	Custom2 = None
	_NormalizePrices = False

	def __init__(self, modelName:str, startingTicker:str, startDate:datetime, durationInYears:int, totalFunds:int, tranchSize:int=1000, trackHistory:bool=True, useDatabase:bool=None, verbose:bool=False):
		#pricesAsPercentages:bool=False would be good but often results in NaN values
		#expects date format in local format, from there everything will be converted to database format				
		startDate = ToDateTime(startDate)
		endDate = startDate + timedelta(days=365 * durationInYears)
		self.modelReady = False
		CreateFolder(self._dataFolderTradeModel)
		if useDatabase==None and globalUseDatabase:
			useDatabase = globalUseDatabase
			self.database = PTADatabase()
			if not self.database.Open():
				print("Default option to use database failed, database connection failed.")
				assert(False)
		elif useDatabase:
			self.database = PTADatabase()
		self.useDatabase = useDatabase
		p = PricingData(startingTicker, useDatabase=self.useDatabase)
		if p.LoadHistory(requestedStartDate=startDate, requestedEndDate=endDate, verbose=verbose): 
			if verbose: print(' Loading ' + startingTicker)
			p.CalculateStats()
			p.TrimToDateRange(startDate, endDate) # - timedelta(days=730), + timedelta(days=10)
			self.priceHistory = [p] #add to list
			i = p.historicalPrices.index.get_loc(startDate, method='nearest')
			startDate = p.historicalPrices.index[i]
			i = p.historicalPrices.index.get_loc(endDate, method='nearest')
			endDate = p.historicalPrices.index[i]
			if not PandaIsInIndex(p.historicalPrices, startDate): startDate += timedelta(days=1)
			if not PandaIsInIndex(p.historicalPrices, startDate): startDate += timedelta(days=1)
			if not PandaIsInIndex(p.historicalPrices, startDate): startDate += timedelta(days=1)
			if not PandaIsInIndex(p.historicalPrices, endDate): endDate -= timedelta(days=1)
			if not PandaIsInIndex(p.historicalPrices, endDate): endDate -= timedelta(days=1)
			if not PandaIsInIndex(p.historicalPrices, endDate): endDate -= timedelta(days=1)
			self.modelStartDate = startDate
			self.modelEndDate = endDate
			self.currentDate = self.modelStartDate
			#modelName += '_' + str(startDate)[:10] + '_' + str(durationInYears) + 'year'
			self.modelName = modelName
			self._tickerList = [startingTicker]
			self.startingValue = totalFunds
			self.modelReady = not(pd.isnull(self.modelStartDate))
		super(TradingModel, self).__init__(portfolioName=modelName, startDate=startDate, totalFunds=totalFunds, tranchSize=tranchSize, trackHistory=trackHistory, useDatabase=useDatabase, verbose=verbose)
		
	def __del__(self):
		self._tickerList = None
		del self.priceHistory[:] 
		self.priceHistory = None
		self.modelStartDate  = None	
		self.modelEndDate = None
		self.modelReady = False

	def Addticker(self, ticker:str):
		r = False
		if not ticker in self._tickerList:
			p = PricingData(ticker, useDatabase=self.useDatabase)
			if self.verbose: print(' Loading price history for ' + ticker)
			if p.LoadHistory(requestedStartDate=self.modelStartDate, requestedEndDate=self.modelEndDate): 
				p.CalculateStats()
				p.TrimToDateRange(self.modelStartDate, self.modelEndDate)# - timedelta(days=750),  + timedelta(days=10)
				if len(p.historicalPrices) > len(self.priceHistory[0].historicalPrices): #first element is used for trading day indexing, replace if this is a better match
					self.priceHistory.insert(0, p)
					self._tickerList.insert(0, ticker)
				else:
					self.priceHistory.append(p)
					self._tickerList.append(ticker)
				r = True
				print(' Added ticker ' + ticker)
			else:
				print( 'Unable to download price history for ticker ' + ticker)
		return r

	def AlignPositions(self, targetPositions:pd.DataFrame, rateLimitTransactions:bool=False, shopBuyPercent:int=0, shopSellPercent:int=0, trimProfitsPercent:int=0, verbose:bool=False): 
		#Performs necessary Buy/Sells to get from current positions to target positions
		#Input ['Ticker']['TargetHoldings'] combo which indicates proportion of desired holdings
		#rateLimitTransactions will limit number of buys/sells per day to one per ticker
		#if not tradeAtMarket then will shop for a good buy and sell price, so far all attempts at shopping or trimming profits yield 3%-13% less average profit

		expireAfterDays=3
		tradeAtMarket = (shopBuyPercent ==0) and (shopSellPercent ==0) 
		TotalTranches = self._tranchCount
		TotalTargets = targetPositions['TargetHoldings'].sum()  #Sum the TargetHoldings, allocate by Rount(TotalTranches/TargetHoldings)
		scale = 1
		if TotalTargets > 0:
			scale = TotalTranches/TotalTargets
			targetPositions.TargetHoldings = (targetPositions.TargetHoldings * scale).astype(float).round() #Round fails if you don't cast the data type beforehand
		if verbose:
			print('Target Positions Scaled')
			print(' Scale (TotalTargets, TotalTranches, Scale):', TotalTargets, TotalTranches, scale)
			print(targetPositions)
		currentPositions = self.GetPositions(asDataFrame=True)
		
		#evaluate the difference between current holdings and target, act accordingly, sorts sells ascending, buys descending
		targetPositions = targetPositions.join(currentPositions, how='outer')
		targetPositions.fillna(value=0, inplace=True)				
		if len(currentPositions) ==0: #no current positions
			targetPositions['Difference'] = targetPositions['TargetHoldings'] 
		else:
			targetPositions['Difference'] = targetPositions['TargetHoldings'] - targetPositions['CurrentHoldings']
		sells = targetPositions[targetPositions['Difference'] < 0]
		sells.sort_values(by=['Difference'], axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last') 
		buys = targetPositions[targetPositions['Difference'] >= 0]
		buys.sort_values(by=['Difference'], axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') 
		if verbose: print(self.currentDate)
		if verbose: print('Target Positions with Current Positions')
		targetPositions = pd.concat([sells, buys]) #Re-ordered sells ascending, buys descending
		if verbose: print(targetPositions)
		for i in range(len(targetPositions)): #for each ticker
			orders = int(targetPositions.iloc[i]['Difference'])
			t = targetPositions.index.values[i]
			if t != 'CASH':
				sn = self.GetPriceSnapshot(t)
				if sn != None:
					price = sn.close
					if orders < 0:
						if not tradeAtMarket: 
							price = sn.Average_5Day * (1+shopSellPercent) 
							if abs(orders) > 2: price = sn.average
						if rateLimitTransactions and abs(orders) > 1: orders = -1
						print('Sell ' + str(abs(orders)) + ' ' + t, '$' + str(price))
						for _ in range(abs(orders)): 
							self.PlaceSell(ticker=t, price=price, marketOrder=tradeAtMarket, expireAfterDays=expireAfterDays, verbose=verbose)
					elif orders > 0 and self.TranchesAvailable() > 0:
						if not tradeAtMarket: 
							price = sn.Average_5Day * (1+shopBuyPercent)
							if abs(orders) > 1: price = sn.Average_5Day
							if abs(orders) > 3: price = sn.close
						if rateLimitTransactions and abs(orders) > 1: orders = 1
						print('Buy ' + str(orders) + ' ' + t, '$' + str(price))
						for _ in range(orders):
							self.PlaceBuy(ticker=t, price=price, marketOrder=tradeAtMarket, expireAfterDays=expireAfterDays, verbose=verbose)											
					elif trimProfitsPercent > 0:
						price = sn.Average_5Day * (1+trimProfitsPercent)
						self.PlaceSell(ticker=t, price=price, marketOrder=False, expireAfterDays=expireAfterDays, verbose=verbose)			
					self.ProcessDay(withIncrement=False)
		self.ProcessDay(withIncrement=False)
		if verbose: print(self.PositionSummary())

	def CancelAllOrders(self): super(TradingModel, self).CancelAllOrders(self.currentDate)
	
	def CloseModel(self, plotResults:bool=True, saveHistoryToFile:bool=True, folderName:str='data/trademodel/', dpi:int=600):	
		cashValue, assetValue = self.Value()
		if assetValue > 0:
			self.SellAllPositions(self.currentDate, allowWeekEnd=True)
		self.UpdateDailyValue()
		cashValue, assetValue = self.Value()
		netChange = cashValue + assetValue - self.startingValue 		
		if saveHistoryToFile:
			self.SaveDailyValueToFile(folderName)
			self.SaveTradeHistoryToFile(folderName)
		print('Model ' + self.modelName + ' from ' + str(self.modelStartDate)[:10] + ' to ' + str(self.modelEndDate)[:10])
		print('Cash: ' + str(round(cashValue)) + ' asset: ' + str(round(assetValue)) + ' total: ' + str(round(cashValue + assetValue)))
		print('Net change: ' + str(round(netChange)), str(round((netChange/self.startingValue) * 100, 2)) + '%')
		print('')
		if plotResults and self.trackHistory: 
			self.PlotTradeHistoryAgainstHistoricalPrices(self.tradeHistory, self.priceHistory[0].GetPriceHistory(), self.modelName)
		return cashValue + assetValue
		
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
	def GetDailyValue(self): 
		return self.dailyValue.copy() #returns dataframe with daily value of portfolio

	def GetValueAt(self, date): 
		try:
			i = self.dailyValue.index.get_loc(date, method='nearest')
			r = self.dailyValue.iloc[i]['TotalValue']
		except:
			print('Unable to return value at ', date)
			r=-1
		return r

	def GetPrice(self, ticker:str=''): 
		#returns snapshot object of yesterday's pricing info to help make decisions today
		forDate = self.currentDate + timedelta(days=-1)
		r = None
		if ticker =='':
			r = self.priceHistory[0].GetPrice(forDate)
		else:
			if not ticker in self._tickerList:	self.Addticker(ticker)
			if ticker in self._tickerList:
				for ph in self.priceHistory:
					if ph.ticker == ticker: r = ph.GetPrice(forDate) 
		return r

	def GetPriceSnapshot(self, ticker:str=''): 
		#returns snapshot object of yesterday's pricing info to help make decisions today
		forDate = self.currentDate + timedelta(days=-1)
		r = None
		if ticker =='':
			r = self.priceHistory[0].GetPriceSnapshot(forDate)
		else:
			if not ticker in self._tickerList:	self.Addticker(ticker)
			if ticker in self._tickerList:
				for ph in self.priceHistory:
					if ph.ticker == ticker: r = ph.GetPriceSnapshot(forDate) 
		return r

	def ModelCompleted(self):
		if self.currentDate ==None or self.modelEndDate == None: 
			r = True
			print('Model start or end date is None', self.currentDate, self.modelEndDate)
		else:
			r = self.currentDate >= self.modelEndDate
		return(r)

	def NormalizePrices(self):
		self._NormalizePrices =  not self._NormalizePrices
		for p in self.priceHistory:
			if not p.pricesNormalized: p.NormalizePrices()
		
	def PlaceBuy(self, ticker:str, price:float, marketOrder:bool=False, expireAfterDays:bool=10, verbose:bool=False):
		if not ticker in self._tickerList: self.Addticker(ticker)	
		if ticker in self._tickerList:	
			if marketOrder or price ==0: price = self.GetPrice(ticker)
			super(TradingModel, self).PlaceBuy(ticker, price, self.currentDate, marketOrder, expireAfterDays, verbose)
		else:
			print(' Unable to add ticker ' + ticker + ' to portfolio.')

	def PlaceSell(self, ticker:str, price:float, marketOrder:bool=False, expireAfterDays:bool=10, datepurchased:datetime=None, verbose:bool=False): 
		if marketOrder or price ==0: price = self.GetPrice(ticker)
		super(TradingModel, self).PlaceSell(ticker=ticker, price=price, datePlaced=self.currentDate, marketOrder=marketOrder, expireAfterDays=expireAfterDays, verbose=verbose)

	def PlotTradeHistoryAgainstHistoricalPrices(self, tradeHist:pd.DataFrame, priceHist:pd.DataFrame, modelName:str):
		buys = tradeHist.loc[:,['dateBuyOrderFilled','purchasePrice']]
		buys = buys.rename(columns={'dateBuyOrderFilled':'Date'})
		buys.set_index(['Date'], inplace=True)
		sells  = tradeHist.loc[:,['dateSellOrderFilled','sellPrice']]
		sells = sells.rename(columns={'dateSellOrderFilled':'Date'})
		sells.set_index(['Date'], inplace=True)
		dfTemp = priceHist.loc[:,['High','Low', 'Channel_High', 'Channel_Low']]
		dfTemp = dfTemp.join(buys)
		dfTemp = dfTemp.join(sells)
		PlotDataFrame(dfTemp, modelName, 'Date', 'Value')

	def ProcessDay(self, withIncrement:bool=True, allowWeekEnd:bool=False):
		#Process current day and increment the current date, allowWeekEnd is for model closing only
		if self.verbose: 
			c, a = self.Value()
			if self.verbose: print(str(self.currentDate) + ' model: ' + self.modelName + ' _cash: ' + str(c) + ' Assets: ' + str(a))
		if self.currentDate.weekday() < 5 or allowWeekEnd:
			for ph in self.priceHistory:
				p = ph.GetPriceSnapshot(self.currentDate)
				self.ProcessDaysOrders(ph.ticker, p.open, p.high, p.low, p.close, self.currentDate)
		self.UpdateDailyValue()
		self.ReEvaluateTrancheCount()
		if withIncrement and self.currentDate <= self.modelEndDate: #increment the date
			try:
				loc = self.priceHistory[0].historicalPrices.index.get_loc(self.currentDate) + 1
				if loc < self.priceHistory[0].historicalPrices.shape[0]:
					nextDay = self.priceHistory[0].historicalPrices.index.values[loc]
					self.currentDate = ToDateTime(str(nextDay)[:10])
				else:
					print('The end: ' + str(self.modelEndDate))
					self.currentDate=self.modelEndDate		
			except:
				#print(self.priceHistory[0].historicalPrices)
				print('Unable to find next date in index from ', self.currentDate,  self.priceHistory[0].ticker)
				self.currentDate += timedelta(days=1)
	
	def SetCustomValues(self, v1, v2):
		self.Custom1 = v1
		self.custom2 = v2
		
class ForcastModel():	#used to forecast the effect of a series of trade actions, one per day, and return the net change in value.  This will mirror the given model.  Can also be used to test alternate past actions 
	def __init__(self, mirroredModel:TradingModel, daysToForecast:int = 10):
		modelName = 'Forcaster for ' + mirroredModel.modelName
		self.daysToForecast = daysToForecast
		self.daysToForecast = daysToForecast
		self.startDate = mirroredModel.modelStartDate 
		durationInYears = (mirroredModel.modelEndDate-mirroredModel.modelStartDate).days/365
		self.tm = TradingModel(modelName=modelName, startingTicker=mirroredModel._tickerList[0], startDate=mirroredModel.modelStartDate, durationInYears=durationInYears, totalFunds=mirroredModel.startingValue, verbose=False, trackHistory=False)
		self.savedModel = TradingModel(modelName=modelName, startingTicker=mirroredModel._tickerList[0], startDate=mirroredModel.modelStartDate, durationInYears=durationInYears, totalFunds=mirroredModel.startingValue, verbose=False, trackHistory=False)
		self.mirroredModel = mirroredModel
		self.tm._tickerList = mirroredModel._tickerList
		self.tm.priceHistory = mirroredModel.priceHistory
		self.savedModel._tickerList = mirroredModel._tickerList
		self.savedModel.priceHistory = mirroredModel.priceHistory

	def Reset(self, updateSavedModel:bool=True):
		if updateSavedModel:
			c, a = self.mirroredModel.Value()
			self.savedModel.currentDate = self.mirroredModel.currentDate
			self.savedModel._cash=self.mirroredModel._cash
			self.savedModel._fundsCommittedToOrders=self.mirroredModel._fundsCommittedToOrders
			self.savedModel.dailyValue = pd.DataFrame([[self.mirroredModel.currentDate,c,a,c+a]], columns=list(['Date','CashValue','AssetValue','TotalValue']))
			self.savedModel.dailyValue.set_index(['Date'], inplace=True)

			if len(self.savedModel._tranches) != len(self.mirroredModel._tranches):
				#print(len(self.savedModel._tranches), len(self.mirroredModel._tranches))
				tranchSize = self.mirroredModel._tranches[0].size
				tc = len(self.mirroredModel._tranches)
				while len(self.savedModel._tranches) < tc:
					self.savedModel._tranches.append(Tranche(tranchSize))
				while len(self.savedModel._tranches) > tc:
					self.savedModel._tranches.pop(-1)
				self.savedModel._tranchCount = len(self.savedModel._tranches)			
			for i in range(len(self.savedModel._tranches)):
				self.savedModel._tranches[i].ticker = self.mirroredModel._tranches[i].ticker
				self.savedModel._tranches[i].available = self.mirroredModel._tranches[i].available
				self.savedModel._tranches[i].size = self.mirroredModel._tranches[i].size
				self.savedModel._tranches[i].units = self.mirroredModel._tranches[i].units
				self.savedModel._tranches[i].purchased = self.mirroredModel._tranches[i].purchased
				self.savedModel._tranches[i].marketOrder = self.mirroredModel._tranches[i].marketOrder
				self.savedModel._tranches[i].sold = self.mirroredModel._tranches[i].sold
				self.savedModel._tranches[i].dateBuyOrderPlaced = self.mirroredModel._tranches[i].dateBuyOrderPlaced
				self.savedModel._tranches[i].dateBuyOrderFilled = self.mirroredModel._tranches[i].dateBuyOrderFilled
				self.savedModel._tranches[i].dateSellOrderPlaced = self.mirroredModel._tranches[i].dateSellOrderPlaced
				self.savedModel._tranches[i].dateSellOrderFilled = self.mirroredModel._tranches[i].dateSellOrderFilled
				self.savedModel._tranches[i].buyOrderPrice = self.mirroredModel._tranches[i].buyOrderPrice
				self.savedModel._tranches[i].purchasePrice = self.mirroredModel._tranches[i].purchasePrice
				self.savedModel._tranches[i].sellOrderPrice = self.mirroredModel._tranches[i].sellOrderPrice
				self.savedModel._tranches[i].sellPrice = self.mirroredModel._tranches[i].sellPrice
				self.savedModel._tranches[i].latestPrice = self.mirroredModel._tranches[i].latestPrice
				self.savedModel._tranches[i].expireAfterDays = self.mirroredModel._tranches[i].expireAfterDays
		c, a = self.savedModel.Value()
		self.startingValue = c + a
		self.tm.currentDate = self.savedModel.currentDate
		self.tm._cash=self.savedModel._cash
		self.tm._fundsCommittedToOrders=self.savedModel._fundsCommittedToOrders
		self.tm.dailyValue = pd.DataFrame([[self.savedModel.currentDate,c,a,c+a]], columns=list(['Date','CashValue','AssetValue','TotalValue']))
		self.tm.dailyValue.set_index(['Date'], inplace=True)
		if len(self.tm._tranches) != len(self.savedModel._tranches):
			#print(len(self.tm._tranches), len(self.savedModel._tranches))
			tranchSize = self.savedModel._tranches[0].size
			tc = len(self.savedModel._tranches)
			while len(self.tm._tranches) < tc:
				self.tm._tranches.append(Tranche(tranchSize))
			while len(self.tm._tranches) > tc:
				self.tm._tranches.pop(-1)
			self.tm._tranchCount = len(self.tm._tranches)			
		for i in range(len(self.tm._tranches)):
			self.tm._tranches[i].ticker = self.savedModel._tranches[i].ticker
			self.tm._tranches[i].available = self.savedModel._tranches[i].available
			self.tm._tranches[i].size = self.savedModel._tranches[i].size
			self.tm._tranches[i].units = self.savedModel._tranches[i].units
			self.tm._tranches[i].purchased = self.savedModel._tranches[i].purchased
			self.tm._tranches[i].marketOrder = self.savedModel._tranches[i].marketOrder
			self.tm._tranches[i].sold = self.savedModel._tranches[i].sold
			self.tm._tranches[i].dateBuyOrderPlaced = self.savedModel._tranches[i].dateBuyOrderPlaced
			self.tm._tranches[i].dateBuyOrderFilled = self.savedModel._tranches[i].dateBuyOrderFilled
			self.tm._tranches[i].dateSellOrderPlaced = self.savedModel._tranches[i].dateSellOrderPlaced
			self.tm._tranches[i].dateSellOrderFilled = self.savedModel._tranches[i].dateSellOrderFilled
			self.tm._tranches[i].buyOrderPrice = self.savedModel._tranches[i].buyOrderPrice
			self.tm._tranches[i].purchasePrice = self.savedModel._tranches[i].purchasePrice
			self.tm._tranches[i].sellOrderPrice = self.savedModel._tranches[i].sellOrderPrice
			self.tm._tranches[i].sellPrice = self.savedModel._tranches[i].sellPrice
			self.tm._tranches[i].latestPrice = self.savedModel._tranches[i].latestPrice
			self.tm._tranches[i].expireAfterDays = self.savedModel._tranches[i].expireAfterDays		
		c, a = self.tm.Value()
		if self.startingValue != c + a:
			print( 'Forcast model accounting error.  ', self.startingValue, self.mirroredModel.Value(), self.savedModel.Value(), self.tm.Value())
			assert(False)
			
	def GetResult(self):
		dayCounter = len(self.tm.dailyValue)
		while dayCounter <= self.daysToForecast:  
			self.tm.ProcessDay()
			dayCounter +=1
		c, a = self.tm.Value()
		endingValue = c + a
		return endingValue - self.startingValue
		
class StockPicker():
	def __init__(self, startDate:datetime=None, endDate:datetime=None, useDatabase:bool=None): 
		if startDate!=None:
			startDate = ToDate(startDate)
			startDate -= timedelta(days=750) #We will use past two years data for statistics, so make sure that is in the range
		self.priceData = []
		self._tickerList = []
		self._startDate = startDate
		self._endDate = endDate
		if useDatabase==None and globalUseDatabase:
			useDatabase = globalUseDatabase
			temp = PTADatabase()
			if not temp.Open():
				print("Default option to use database failed, database connection failed.")
				assert(False)
		self.useDatabase = useDatabase
		
	def __del__(self): 
		self.priceData = None
		self._tickerList = None
		
	def AddTicker(self, ticker:str):
		if not ticker in self._tickerList:
			p = PricingData(ticker, useDatabase=self.useDatabase)
			if p.LoadHistory(self._startDate, self._endDate, verbose=True): 
				p.CalculateStats()
				self.priceData.append(p)
				self._tickerList.append(ticker)

	def RemoveTicker(self, ticker:str, verbose:bool=False):
#		if ticker in self._tickerList:
		i=len(self.priceData)-1
		while i > 0:
			if ticker == self.priceData[i].ticker:
				if verbose: print(" Removing ticker " + ticker)
				self.priceData.pop(i)
			i -=1
		if ticker in self._tickerList: self._tickerList.remove(ticker)	

	def AlignToList(self, newList:list, verbose:bool=False):
		#Add/Remove tickers until they match the given list
		for t in self._tickerList:
			if not t in newList:
				if verbose: print(" Removing ticker " + t)
				self.RemoveTicker(t)
		for t in newList:
			self.AddTicker(t)

	def SaveStats(self):
		for p in self.priceData:
			p.SaveStatsToFile()
			
	def TickerExists(self, ticker:str):
		return ticker in self._tickerList
	
	def TickerCount(self):
		return len(self._tickerList)

	def NormalizePrices(self):
		for i in range(len(self.priceData)):
			self.priceData[i].NormalizePrices()
			
	def FindOpportunities(self, currentDate:datetime, stocksToReturn:int = 5, filterOption:int = 3, minPercentGain=0.00): 
		result = []
		for i in range(len(self.priceData)):
			ticker = self.priceData[i].ticker
			psnap = self.priceData[i].GetPriceSnapshot(currentDate)
			if  ((psnap.EMA_Short/psnap.EMA_Long)-1 > minPercentGain):
				if filterOption ==0: #Overbought
					if psnap.low > psnap.Channel_High: result.append(ticker)
				if filterOption ==1: #Oversold
					if psnap.high < psnap.Channel_Low: result.append(ticker)
				if filterOption ==1: #High price deviation
					if psnap.Deviation_5Day > .0275: result.append(ticker)
		return result

	def GetHighestPriceMomentum(self, currentDate:datetime, longHistoryDays:int = 365, shortHistoryDays:int = 30, stocksToReturn:int = 5, filterOption:int = 3, minPercentGain=0.05, maxVolatility=.1, pvmethod:int=0, verbose:bool=False): 
		minPC_1Day = minPercentGain/365
		candidates = pd.DataFrame(columns=list(['Ticker','hp2Year','hp1Year','hp6mo','hp3mo','hp2mo','hp1mo','Price_Current','PC_2Year','PC_1Year','PC_6Month','PC_3Month','PC_2Month','PC_1Month','PC_1Day','Gain_Monthly','LossStd_1Year','longHistoricalValue','shortHistoricalValue','PC_LongTerm','PC_ShortTerm','Point_Value','Comments','latestEntry']))
		candidates.set_index(['Ticker'], inplace=True)
		lookBackDateLT = currentDate + timedelta(days=-longHistoryDays)
		lookBackDateST = currentDate + timedelta(days=-shortHistoryDays)
		for i in range(len(self.priceData)):
			ticker = self.priceData[i].ticker
			if (lookBackDateLT >= self.priceData[i].historyStartDate and currentDate <= self.priceData[i].historyEndDate + timedelta(days=20)):		
				longHistoricalValue = self.priceData[i].GetPrice(lookBackDateLT)
				shortHistoricalValue = self.priceData[i].GetPrice(lookBackDateST)
				sn = self.priceData[i].GetPriceSnapshot(currentDate + timedelta(days=-730))
				hp2Year = sn.Average_5Day #Looking at 30/90/365 day prices, recent changes are just noise
				sn = self.priceData[i].GetPriceSnapshot(currentDate + timedelta(days=-547))
				hp1Year = sn.Average_5Day
				sn = self.priceData[i].GetPriceSnapshot(currentDate + timedelta(days=-180))
				hp6mo = sn.Average_5Day
				sn = self.priceData[i].GetPriceSnapshot(currentDate + timedelta(days=-90))
				hp3mo = sn.Average_5Day
				sn = self.priceData[i].GetPriceSnapshot(currentDate + timedelta(days=-60))
				hp2mo = sn.Average_5Day
				sn = self.priceData[i].GetPriceSnapshot(currentDate + timedelta(days=-30))
				hp1mo = sn.Average_5Day
				sn = self.priceData[i].GetPriceSnapshot(forDate=currentDate, verbose=True, pvmethod=pvmethod)
				Price_Current = sn.Average_5Day #Looking at 30/90/365 day prices, recent changes are just noise
				PC_ShortTerm = sn.PC_1Month3WeekEMA/20 #Converted to daily
				PC_LongTerm = sn.PC_1Year/250		   #Converted to daily
				Point_Value  = sn.Point_Value
				if (longHistoricalValue > 0 and Price_Current > 0 and shortHistoricalValue > 0 and hp2Year > 0 and hp1Year > 0 and hp6mo > 0 and hp2mo > 0 and hp1mo > 0): #values were loaded
					#print(sn.PC_1Month/30,sn.PC_1Month3WeekEMA/30, ((Price_Current/shortHistoricalValue)-1)/shortHistoricalValue,'PC_1Month','PC_1Month3WeekEMA','((Price_Current/shortHistoricalValue)-1)/shortHistoricalValue')
					pc2mo=((Price_Current/hp2mo)-1)*5.952 #Annualized
					candidates.loc[ticker] = [hp2Year,hp1Year,hp6mo,hp3mo,hp2mo,hp1mo,Price_Current,sn.PC_2Year,sn.PC_1Year,sn.PC_6Month,sn.PC_3Month,pc2mo,sn.PC_1Month,sn.PC_1Day,sn.Gain_Monthly,sn.LossStd_1Year,longHistoricalValue,shortHistoricalValue,PC_LongTerm, PC_ShortTerm, sn.Point_Value, sn.Comments, self.priceData[i].historyEndDate]
				else:
					if Price_Current > 0 and verbose:
						if len(self.priceData[i].historicalPrices) > 0:
							print('Price load failed for ticker: ' + ticker, 'requested, history start, history end', currentDate, self.priceData[i].historyStartDate, self.priceData[i].historyEndDate, hp2Year,hp1Year,hp6mo,hp2mo,hp1mo)
			elif verbose:
				print(self.priceData[i].ticker, lookBackDateLT, currentDate, self.priceData[i].historyStartDate, self.priceData[i].historyEndDate, "Dropped by date range filter (lookBackDateLT, currentDate, historyStartDate, historyEndDate)")

		#More complex filters that I have tried have all decreased performance which is why these are simple
		#Greatest factors for improvement are high 1yr return and a very low selection of stocks, like 1-3
		#Best way to compensate for few stocks is to blend filters of different strengths
		if filterOption ==1: #high performer, recently at a discount or slowing down but not negative
			filter = (candidates['PC_LongTerm'] > candidates['PC_ShortTerm']) & (candidates['PC_LongTerm'] > minPC_1Day) & (candidates['PC_ShortTerm'] > 0) 
			candidates.sort_values('PC_LongTerm', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') #Most critical factor sorting by largest long term gain
		elif filterOption ==2: #Long term gain meets min requirements
			filter = (candidates['PC_LongTerm'] > minPC_1Day)  
			candidates.sort_values('PC_LongTerm', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') #Most critical factor sorting by largest long term gain
		elif filterOption ==3: #Best overall returns 25% average yearly over 36 years which choosing top 5 sorted by best yearly average
			filter = (candidates['PC_LongTerm'] > minPC_1Day) & (candidates['PC_ShortTerm'] > 0) 
			candidates.sort_values('PC_LongTerm', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') #Most critical factor sorting by largest long term gain
		elif filterOption ==4: #Short term gain meets min requirements
			filter =  (candidates['PC_ShortTerm'] > minPC_1Day) 
			candidates.sort_values('PC_ShortTerm', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') #Most critical factor sorting by largest short term gain which is not effective
		elif filterOption ==44: #Short term gain meets min requirements, sort long value
			filter =  (candidates['PC_ShortTerm'] > minPC_1Day) 
			candidates.sort_values('PC_LongTerm', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') #Most critical factor sorting by largest long term gain
		elif filterOption ==5: #Point Value
			filter = (candidates['PC_1Year'] > minPC_1Day) & (candidates['Point_Value'] > 0)
			candidates.sort_values('Point_Value', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') 
		elif filterOption ==6: #Hard year, will often not find cadidates
			filter = (candidates['PC_1Year'] > 0) & (candidates['LossStd_1Year'] > .06) & (candidates['LossStd_1Year'] < .15) & (candidates['PC_3Month'] > 0) & (candidates['PC_1Month'] > 0)
			candidates.sort_values('LossStd_1Year', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') 
		else: #no filter
			filter = (candidates['Price_Current'] > 0)
			candidates.sort_values('PC_LongTerm', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') #Most critical factor, sorting by largest long term gain
		candidates = candidates[filter]
		candidates.drop(columns=['longHistoricalValue','shortHistoricalValue','PC_LongTerm','PC_ShortTerm'], inplace=True, axis=1)
		candidates = candidates[:stocksToReturn]
		return candidates

	def ToDataFrame(self, intervalInWeeks:int = 1, pivotOnTicker:bool = False, showGain:bool = True, normalizeValues:bool = False):
		#returns DataFrame of historical prices of all tickers in the picker
		#intervalInWeeks > 0 will drop daily values keeping only those that fall on this weekly interval
		#pivotOnTicker returns the tickers as columns with either the price 'Average' or 'Gain' as a value
		#showGain returns only the percentage change from prior value
		#normalizeValues can be used if not returning gains	
		r = pd.DataFrame()
		for i in range(len(self.priceData)):
			t = self.priceData[i].ticker
			temporarilyNormalize = False
			if normalizeValues and not showGain:
				if not self.priceData[i].pricesNormalized:
					temporarilyNormalize = True
					self.priceData[i].NormalizePrices()
			if len(self.priceData[i].historicalPrices) > 0:
				x = self.priceData[i].historicalPrices.copy()
				if intervalInWeeks > 1:
					startDate = x.index[0]
					x['DayOffset'] = (x.index - startDate).days
					x = x[x['DayOffset'] % (intervalInWeeks * 7) ==0] #Drop all data except specified weekly interval of dates
					x.drop(columns=['DayOffset'], inplace=True, axis=1)
				if pivotOnTicker:
					if r.empty: r = pd.DataFrame(index=x.index)
					if showGain:
						r[t] = (x['Average']/x['Average'].shift(-1))
					else:
						r[t] = x['Average']				
				else:
					if showGain:
						x['Gain'] = x['Average']/x['Average'].shift(-1)
						x = x[['Gain']]
					x['Ticker'] = t
					if r.empty: 
						r = x
					else:
						r = r.append(x)
			if temporarilyNormalize: self.priceData[i].NormalizePrices() #undo price normalization
		r.fillna(value=0, inplace=True)
		r.sort_index
		return r

	def LoadTickerFromCSVToSQL(self, ticker:str):
		for i in range(len(self.priceData)):
			if ticker == self.priceData[i].ticker: self.priceData[i].LoadTickerFromCSVToSQL()
			
		
