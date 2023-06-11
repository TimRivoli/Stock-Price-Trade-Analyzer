import numpy, pandas as pd
import os, configparser, ast, pyodbc
import urllib.error, urllib.request as webRequest
from datetime import datetime, timedelta, date
from sqlalchemy import create_engine

currentProxyServer = None
proxyList = ['173.232.228.25:8080']

def isfloat(num):
	try:
		float(num)
		return True
	except ValueError:
		return False

def ReadConfig(section_name:str, value_name:str):
	settingsFile = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))) + '/config.ini'
	r = None
	if os.path.isfile(settingsFile):
		config = configparser.ConfigParser()
		config.read(settingsFile)
		if section_name in config:
			try: 
				r = config.get(section_name, value_name)
				r = ast.literal_eval(r)
			except Exception as e:
				print('Unable to read value ', value_name, ' from settings file.')
				print(e)
	if r==None: WriteConfig(section_name, value_name, '')
	return r

def WriteConfig(section_name:str, value_name:str, value:str):
	settingsFile = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))) + '/config.ini'
	config = configparser.ConfigParser()
	if os.path.isfile(settingsFile): config.read(settingsFile)
	if section_name not in config: config.add_section(section_name)
	config.set(section_name, value_name, str(value))
	try: 
		with open(settingsFile, 'w') as configfile: config.write(configfile)
	except Exception as e:
		print('Unable to write value ', value_name, ' from settings file.')
		print(e)

def ReadConfigBool(section_name:str, value_name:str):
	r = ReadConfig(section_name, value_name)
	if r == None: r = False
	return bool(r)

def ReadConfigInt(section_name:str, value_name:str):
	r = ReadConfig(section_name, value_name)
	if r == None: r = 0
	return int(r)

def ReadConfigString(section_name:str, value_name:str):
	r = ReadConfig(section_name, value_name)
	if r == None: r = ''
	return str(r)

def ReadConfigList(section_name:str, value_name:str):
	r = ReadConfig(section_name, value_name)
	print(value_name, r)
	if r == None: 
		r = []
	return r

def CreateFolder(p:str):
	r = True
	if not os.path.exists(p):
		try:
			os.mkdir(p)	
		except Exception as e:
			print('Unable to create folder: ' + p)
			f = False
	return r
	
def FileExists(f): return 	os.path.isfile(f)

def GetMyDateFormat(): return '%m/%d/%Y'

def ToDate(given_date):
#returns date object, converting from string or datetime if necessary
	if type(given_date) == str:
		if given_date.find('-') > 0 :
			r = datetime.strptime(given_date, '%Y-%m-%d').date()
		else:
			r = datetime.strptime(given_date, GetMyDateFormat()).date()
	elif isinstance(given_date, datetime):
		r = given_date.date()
	elif isinstance(given_date, numpy.datetime64):
		r = pd.Timestamp(given_date).date()
	else:
		r = given_date
	return r

def ToDateTime(given_date):
	#returns datetime object
	if type(given_date) == str: given_date = ToDate(given_date)
	if isinstance(given_date, datetime):
		r = given_date
	elif isinstance(given_date, numpy.datetime64):
		r = pd.Timestamp(given_date).date()
	elif isinstance(given_date, date): 
		r = datetime.combine(given_date, datetime.min.time())
	else:
		r = given_date
	return r

def DateFormatDatabase(given_date):
	#returns datetime object, technically should be numpy.datetime64
	r = ToDateTime(given_date)
	return r

def GetDateTimeStamp():
	d = datetime.now()
	return d.strftime('%Y%m%d%H%M')

def GetTodaysDate():
	d = datetime.now()
	#return d.strftime('%m/%d/%Y')
	return d.date()

def GetTodaysDateString():
	d = datetime.now() #--1980-01-01
	return d.strftime('%Y-%m-%d')

def DateDiffDays(start_date:datetime, end_date:datetime):
	delta = end_date-start_date
	return delta.days
		
def DateDiffHours(start_date:datetime, end_date:datetime):
	delta = end_date-start_date
	return int(delta.total_seconds() / 3600)

def AddDays(start_date, days:int):
	start_date = ToDate(start_date)
	return start_date + timedelta(days=days) 

def CreateFolder(p:str):
	r = True
	if not os.path.exists(p):
		try:
			os.mkdir(p)	
		except Exception as e:
			print('Unable to create folder: ' + p)
			f = False
	return r
	
def CleanOldTickerLists():
	errorFile = 'data/historical/badTickerLog.txt' 
	dataFile = '_classes/TickerLists.py'
	if os.path.isfile(errorFile):
		print("Purging bad tickers from tickerlists")
		df = open(dataFile,'r')
		buffer = df.read()
		df.close()
		ef = open(errorFile,'r')
		badEntries = ef.readlines()
		ef.close()
		for badTicker in badEntries:
			badTicker = badTicker.replace("\n","")
			print("Deleting " + badTicker)
			buffer = buffer.replace(badTicker, "")
			buffer = buffer.replace(badTicker, "")
			buffer = buffer.replace(badTicker, "")
			buffer = buffer.replace(badTicker, "")
		df = open(dataFile,'w')
		df.write(buffer)
		df.close()
		os.remove(errorFile)
		print("Completed purging")

def ListToString(l:list):
	r = "['"
	for i in l:
		r += i + "','"
	r += "']"
	r = r.replace(",''","")
	return r
	
def GetProxiedOpener():
	#testURL = 'https://stooq.com'
	testURL = 'https://www.google.com'
	#userName, password = 'mUser', 'SecureAccess'
	userName, password = '', ''
	
	context = ssl._create_unverified_context()
	#context2 = ssl.create_default_context()
	#context2.check_hostname = False
	#context2.verify_mode = ssl.CERT_NONE
	https_handler = webRequest.HTTPSHandler(context=context)
	i = -1
	functioning = False
	global currentProxyServer
	while not functioning and i < len(proxyList):
		if i >=0 or currentProxyServer==None: currentProxyServer = proxyList[i]
		if userName != '':
			proxySet = {'http':userName + ':' + password + '@' + currentProxyServer, 'https':userName + ':' + password + '@' + currentProxyServer}
		else:
			proxySet = {'http':currentProxyServer, 'https':currentProxyServer}
		proxy_handler = webRequest.ProxyHandler(proxySet)
		authHandler = webRequest.HTTPBasicAuthHandler()
		#opener = webRequest.build_opener(proxy_handler, https_handler, authHandler) 		
		opener = webRequest.build_opener(proxy_handler) 		
		opener.addheaders = [('User-agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.1 Safari/603.1.30')]
		#opener.addheaders = [('Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.1.1 Safari/605.1.15')]	
		#opener.addheaders = [('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/18.17763')]
		#opener.addheaders = [('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.102 Safari/537.36 Edg/104.0.1293.63')]
		try:
			print(' Testing Proxy ' + currentProxyServer + '...')
			#response = webRequest.urlopen(req)
			requests.get(testURL, proxies=proxies)
			conn = opener.open(testURL)
#			r = requests.get(testURL, proxies=proxySet)
			print(' Proxy ' + currentProxyServer + ' is functioning')
			functioning = True
		except urllib.error.URLError as e:
			print('Proxy ' + currentProxyServer + ' is not responding')
			print(e.reason)
			conn.close()
		i+=1
	assert(False)
	return opener

def GetWorkingProxy(current_proxy_nonfunctional: bool = False):
	#testURL = 'https://stooq.com'
	testURL = 'https://www.google.com'
	#userName, password = 'mUser', 'SecureAccess'
	userName, password = '', ''
	i = -1
	functioning = False
	headerSet ={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.102 Safari/537.36 Edg/104.0.1293.63'}
	headerSet ={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.1.1 Safari/605.1.15'}	
	global currentProxyServer
	global proxyList
	if proxyList == None:
		x =  ReadConfigList('Settings', 'proxyList')
		if not x == None: proxyList = x		
	if current_proxy_nonfunctional:
		if not currentProxyServer==None and len(proxyList) > 3: 
			if verbose: print( ' Removing proxy: ', currentProxyServer)
			proxyList.remove(currentProxyServer)
			currentProxyServer = None
	while not functioning and i < len(proxyList):
		if i >=0 or currentProxyServer==None: currentProxyServer = proxyList[i]
		if userName != '':
			proxySet = {'https': 'http://' + userName + ':' + password + '@' + currentProxyServer}
		else:
			proxySet = {'http':currentProxyServer, 'https': currentProxyServer}
			#proxySet = {'https': currentProxyServer}
		try:
			print(' Testing Proxy ' + currentProxyServer + '...')
			requests.get(testURL, headers=headerSet, proxies=proxySet)
			requests.raise_for_status()
			print(' Proxy ' + currentProxyServer + ' is functioning')
			functioning = True
		except requests.exceptions.HTTPError as errh:
			print (" Http Error:",errh)
			print(' Proxy ' + currentProxyServer + ' is not responding')
			proxySet = {}
		except requests.exceptions.ConnectionError as errc:
			print (" Error Connecting:",errc)
			print(' Proxy ' + currentProxyServer + ' is not responding')
			proxySet = {}
		except requests.exceptions.Timeout as errt:
			print (" Timeout Error:",errt)
			print(' Proxy ' + currentProxyServer + ' is not responding')
			proxySet = {}
		except requests.exceptions.RequestException as err:
			print (" ", err)		
			print(' Proxy ' + currentProxyServer + ' is not responding')
			proxySet = {}
		i+=1
	return proxySet, headerSet

globalUseDatabase=False
DatabaseServer = ReadConfigString('Database', 'DatabaseServer')
DatabaseName = ReadConfigString('Database', 'DatabaseName')
if DatabaseServer != '' and DatabaseName !='' and DatabaseServer != None and DatabaseName !=None:
	globalUseDatabase=True
	UseSQLDriver = ReadConfigBool('Database', 'UseSQLDriver')
	DatabaseUsername = ReadConfigString('Database', 'DatabaseUsername')
	DatabasePassword = ReadConfigString('Database', 'DatabasePassword')
	DatabaseConstring = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + DatabaseServer + ';DATABASE=' + DatabaseName 
	if UseSQLDriver:
		DatabaseConstring = 'DRIVER={SQL Server Native Client 11.0};SERVER=' + DatabaseServer + ';DATABASE=' + DatabaseName 
	if DatabaseUsername !="" and DatabaseUsername != None:
		DatabaseConstring += ';UID=' + DatabaseUsername + ';PWD=' + DatabasePassword
	else:
		DatabaseConstring += ';Trusted_Connection=yes;' #';Integrated Security=true;'

def Is_sql_configured(): return globalUseDatabase
def SQL_ConString(): return DatabaseConstring
#-------------------------------------------- SQL Utilities -----------------------------------------------
class PTADatabase():
	def __init__(self, verbose:bool = False):
		self.databaseConnected = False
		self.cursor = None	
		self.verbose = verbose
		self.pyEngine = None
		
	def __del__(self):
		if self.databaseConnected: self.Close()
		self.cursor = None

	def Open(self):
		result = False
		self.databaseConnected = False
		try:
			self.conn = pyodbc.connect(SQL_ConString())
			self.conn.autocommit = True
			self.cursor = self.conn.cursor()
			self.databaseConnected = True
			if self.verbose: print("Database connection established")
			result = True
		except Exception as e:
			self.databaseConnected = False
			print("Database connection attempt failed")
			print(e)
		return result

	def Close(self):
		self.cursor = None
		self.conn.close()
		self.databaseConnected = False
		if self.pyEngine != None: self.pyEngine.dispose()
		if self.verbose: print("Database connection closed")
		
	def Connected(self):
		result = False
		try:
			cursor = conn.cursor()
			result = True
		except e:
			if e.__class__ == pyodbc.ProgrammingError:        
				conn == reinit()
				cursor = conn.cursor()
		return result

	def ExecSQL(self, sqlStatement:str):
		self.cursor.execute(sqlStatement)

	def GetCursor(self):
		return self.cursor
	
	def DataFrameToSQL(self, df:pd.DataFrame, tableName:str, indexAsColumn:bool=False, clearExistingData:bool=False):
		if clearExistingData:
			sqlStatement = "if OBJECT_ID('" + tableName + "') is not null Delete FROM " + tableName 
			if not self.databaseConnected: self.Open()
			if self.databaseConnected: 	self.cursor.execute(sqlStatement)
		if self.pyEngine == None:
			quoted = urllib.parse.quote_plus(DatabaseConstring)
			self.pyEngine = create_engine('mssql+pyodbc:///?odbc_connect={}'.format(quoted))
		if indexAsColumn: df.reset_index(drop=False, inplace=True)
		df.to_sql(tableName, schema='dbo', con = self.pyEngine, if_exists='append', index=False)

	def DataFrameFromSQL(self, SQL:str, indexName:str=None):
		if indexName==None:
			df = pd.read_sql_query(SQL, self.conn)
		else:
			df = pd.read_sql_query(SQL, self.conn, index_col=indexName)
		return df

