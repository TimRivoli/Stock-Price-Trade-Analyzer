import time, ssl, requests, pandas as pd
import urllib.error, urllib.request as webRequest
import pyodbc
import yfinance as yf
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
from contextlib import suppress
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError, OperationalError, ProgrammingError
from curl_cffi.requests.exceptions import HTTPError
from _classes.Utility import ReadConfigString, ReadConfigBool, GetTodaysDate, GetTodaysDateString

#-------------------------------------------- SQL Setup and Helpers  -----------------------------------------------
SQL_NUMERIC_MAX = 999_999.99999 #Safety check for garbage prices

def isfloat(num):
	try:
		float(num)
		return True
	except ValueError:
		return False

def SQLAlchemy_Connection_URL(server: str | None, database: str | None, username: str | None, password: str | None, use_trusted: bool = True):
	if not server or not database:
		return None
	driver = "ODBC+Driver+18+for+SQL+Server"
	if username and password:
		return (f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}")
	if use_trusted:
		return (f"mssql+pyodbc://@{server}/{database}?driver={driver}&trusted_connection=yes")
	return None

def _filter_sql_numeric_overflow(df: pd.DataFrame, ticker: str,	cols: list[str]	) -> pd.DataFrame:
	#Drops rows where any column exceeds NUMERIC(11,5) limits.
	mask = (df[cols].abs() <= SQL_NUMERIC_MAX).all(axis=1)
	dropped = (~mask).sum()
	if dropped > 0:
		print(f" DataDownload: WARNING: {ticker} dropped {dropped} rows exceeding NUMERIC(11,5)")
		print(df[~mask])
	return df.loc[mask]

#-------------------------------------------- DataDownload Class  -----------------------------------------------
class DataDownload:

	def _DownLoadGoogleFinancePage(self, ticker:str, stockExchange:str ="NYSE"):
		print("Downloading ticker info for " + ticker)
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
		if v == "—" or v=="-" or v =="" or v=='∞': v=0
		if isfloat(v):  v = float(v)
		#print('v',v)
		return v

	def _ScrapeGoogleFinanceTickerInfoAndFinancials(self, ticker:str, pageData:str):
		#Parses GF page for financial data, updates database
		print(" Parsing ticker infor for " + ticker)
		result = False
		currentDate = GetTodaysDate()
		currentYear = currentDate.year
		
		db = PTADatabase()
		if db.Open():
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
					db.ExecSQL("IF NOT EXISTS (SELECT 1 FROM Tickers WHERE Ticker=:Ticker) INSERT INTO Tickers (Ticker) VALUES (:Ticker)", {"Ticker": ticker})
					db.ExecSQL("UPDATE Tickers SET CompanyName=:CompanyName WHERE Ticker=:Ticker", {"CompanyName": CompanyName, "Ticker": ticker})
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
						db.ExecSQL("UPDATE Tickers SET MarketCap=:MarketCap, PE_Ratio=:PERatio, Dividend=:Dividend WHERE Ticker=:Ticker", {"MarketCap": values[3], "PERatio": values[5], "Dividend": values[6], "Ticker": ticker})
					else:
						db.ExecSQL("UPDATE Tickers SET MarketCap=:MarketCap WHERE Ticker=:Ticker", {"MarketCap": values[3], "Ticker": ticker})
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
				db.ExecSQL("DELETE FROM TickerFinancials WHERE Ticker=:Ticker AND Year=:Year", {"Ticker": ticker, "Year": currentDate.year})
				db.ExecSQL("INSERT INTO TickerFinancials (Ticker, Year, Month, Revenue, OperatingExpense, NetIncome, NetProfitMargin, EarningsPerShare, EBITDA, EffectiveTaxRate) VALUES (:Ticker, :Year, :Month, :Revenue, :OperatingExpense, :NetIncome, :NetProfitMargin, :EPS, :EBITDA, :EffectiveTaxRate)", {"Ticker": ticker, "Year": currentDate.year, "Month": currentDate.month, "Revenue": values[0], "OperatingExpense": values[1], "NetIncome": values[2], "NetProfitMargin": values[3], "EPS": values[4], "EBITDA": values[5], "EffectiveTaxRate": values[6]})
				db.ExecSQL("UPDATE TickerFinancials SET CashShortTermInvestments=:CashSTI, TotalAssets=:TotalAssets, TotalLiabilities=:TotalLiabilities, TotalEquity=:TotalEquity, SharesOutstanding=:Shares, PriceToBook=:PriceToBook, ReturnOnAssetts=:ROA, ReturnOnCapital=:ROC WHERE Ticker=:Ticker AND Year=:Year AND Month=:Month", {"CashSTI": values[7], "TotalAssets": values[8], "TotalLiabilities": values[9], "TotalEquity": values[10], "Shares": values[11], "PriceToBook": values[12], "ROA": values[13], "ROC": values[14], "Ticker": ticker, "Year": currentDate.year, "Month": currentDate.month})
				db.ExecSQL("UPDATE TickerFinancials SET CashFromOperations=:CFO, CashFromInvesting=:CFI, CashFromFinancing=:CFF, NetChangeInCash=:NetCash, FreeCashFlow=:FCF WHERE Ticker=:Ticker AND Year=:Year AND Month=:Month", {"CFO": values[16], "CFI": values[17], "CFF": values[18], "NetCash": values[19], "FCF": values[20], "Ticker": ticker, "Year": currentDate.year, "Month": currentDate.month})
				SQL = "UPDATE TickerFinancials SET CashFromOperations=?, CashFromInvesting=?, CashFromFinancing=?, NetChangeInCash=?, FreeCashFlow=? WHERE Ticker=? AND Year=? AND MONTH=?"
			startIndex = pageData.find(startAboutDelimiter,0)
			if startIndex > 0:
				startIndex += len(startAboutDelimiter)
				endIndex = pageData.find(endRecordDelimiter, startIndex+1)
				if endIndex > 0: 
					AboutComment = pageData[startIndex:endIndex]
					result=True
				#print(AboutComment, len(AboutComment))
				db.ExecSQL("UPDATE Tickers SET About=:About WHERE Ticker=:Ticker AND About IS NULL", {"About": AboutComment[:500], "Ticker": ticker})
			db.Close()

	def _ParseAndUpdatePriceHistory(self, ticker:str, stockExchange:str, pageData:str, IntraDayValues:bool=False, verbose:bool=False):
		#Parses GF page for price data, updates database
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
									if price > 0: 
										db.ExecSQL("INSERT INTO PricesIntraday (Ticker, Year, Month, Day, Hour, Minute, DateTime, Price, Volume) VALUES (:Ticker, :Year, :Month, :Day, :Hour, :Minute, :DateTime, :Price, :Volume)", {"Ticker": ticker, "Year": year, "Month": month, "Day": day, "Hour": hour, "Minute": minute, "DateTime": theDate, "Price": price, "Volume": volume})
						startIndex = theMeat.find(startRecordDelimiter,0)
					print(' Valid records: ' + str(valideRecords), " Errors: " + str(errors))
					result = errors < 10 and valideRecords > 10
					if not result and ticker[0] != '.' and False: 
						print('Failed to load price data for ' + ticker)
						assert(False)
					if result:
						if IntraDayValues: #Updating daily values, date range should be today
							db.ExecSQL("DELETE FROM PricesIntraday WHERE Ticker=:Ticker AND TimeStamp < :TimeStamp AND [DateTime] BETWEEN :StartDate AND :EndDate", {"Ticker": ticker, "TimeStamp": currentTimeStamp, "StartDate": startDate, "EndDate": endDate})
						else:
							#Only delete the daily (hour=16, minute=0) 
							db.ExecSQL("DELETE FROM PricesIntraday WHERE Ticker=:Ticker AND [Hour]=16 AND [Minute]=0 AND TimeStamp < :TimeStamp AND [DateTime] BETWEEN :StartDate AND :EndDate", {"Ticker": ticker, "TimeStamp": currentTimeStamp, "StartDate": startDate, "EndDate": endDate})
					db.Close()
		if result:
			print(" Parse price data successful for ticker: " + ticker, stockExchange, startDate, endDate)
		else:
			print(" Parse price data failed for ticker: " + ticker, stockExchange)
			#assert(False)

	def _IdentifyAndRegisterExchange(self, ticker: str) -> tuple[str, str]:
		Exchanges = ['NYSE', 'NASDAQ', 'INDEXNASDAQ', 'INDEXSP', 'INDEXDJX', 'NYSEARCA', 'BMV', 'LON']
		found_exchange = ""
		page_data = ""

		# Search for the valid exchange
		for e in Exchanges:
			print(f"Testing exchange: {e}")
			data = self._DownLoadGoogleFinancePage(ticker, e)
			if data != "":
				found_exchange = e
				page_data = data
				break 
		
		# If found, update the database
		if found_exchange:
			db = PTADatabase()
			if db.Open():
				db.ExecSQL("IF NOT EXISTS (SELECT 1 FROM Tickers WHERE Ticker=:Ticker) INSERT INTO Tickers (Ticker, Exchange) VALUES (:Ticker, :Exchange)", {"Ticker": ticker, "Exchange": found_exchange})
				db.ExecSQL("UPDATE Tickers SET Exchange=:Exchange WHERE Ticker=:Ticker", {"Exchange": found_exchange, "Ticker": ticker})
		return found_exchange, page_data

	def DownloadIntradayPriceGoogleFinance(self, ticker: str, Exchange: str = None):
		# Downloads past 30 days intraday price and financial data, saves to database
		Result = False	
		if not Exchange:
			Exchange, pageData = self._IdentifyAndRegisterExchange(ticker)
		else:
			pageData = self._DownLoadGoogleFinancePage(ticker, Exchange)

		if pageData != "":
			self._ParseAndUpdatePriceHistory(ticker, Exchange, pageData, True)
			self._ParseAndUpdatePriceHistory(ticker, Exchange, pageData, False)
			self._ScrapeGoogleFinanceTickerInfoAndFinancials(ticker, pageData)
			Result = True
			
		return Result

	def DownloadTickerGoogleFinance(self, ticker:str, Exchange:str=None):
		#Updates missing About and other info, updates database tables
		if not Exchange:
			Exchange, pageData = self._IdentifyAndRegisterExchange(ticker)
		else:
			pageData = self._DownLoadGoogleFinancePage(ticker, Exchange)
		if pageData != '': self._ScrapeGoogleFinanceTickerInfoAndFinancials(ticker, pageData)

	def DownloadPriceDataYahooFinance(self, ticker: str):
		if not ticker:
			return None
		t2 = ticker.upper().replace('.INX', '^SPX')
		if t2.startswith('.'):
			t2 = t2.replace('.', '^')
		t2 = t2.replace('.', '-')
		try:
			tkr = yf.Ticker(t2)
			df = tkr.history(start="1980-01-01", auto_adjust=False)
		except HTTPError as e:
			print(f"WARNING: yfinance history() failed for {ticker}: {e}")
			df = None
		except Exception as e:
			print(f"WARNING: unexpected yfinance error for {ticker}: {e}")
			df = None
		if df is None or df.empty:
			df = yf.download(
				t2,
				start="1980-01-01",
				progress=False,
				auto_adjust=False,
				group_by="column"
			)

		if df is None or df.empty:
			print(f"ERROR: No Yahoo data returned for {ticker}")
			return None

		if isinstance(df.columns, pd.MultiIndex):
			df.columns = df.columns.get_level_values(-1)
		df.columns = [c.capitalize() for c in df.columns]
		REQUIRED = ['Open', 'High', 'Low', 'Close']
		missing = [c for c in REQUIRED if c not in df.columns]
		if missing:
			raise KeyError(f"Yahoo data missing columns for {ticker}: {list(df.columns)}")
		df = df[REQUIRED]
		df = df.apply(pd.to_numeric, errors="coerce")
		df.dropna(inplace=True)
		df = _filter_sql_numeric_overflow(df, ticker=ticker, cols=REQUIRED)
		if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
			df.index = df.index.tz_convert(None)
		df.index.name = "Date"
		return df

#-------------------------------------------- SQL Utilities -----------------------------------------------
class PTADatabase:
	def _CreateEngine(self, url: str):
		try:
			self.engine = create_engine(url, connect_args={"trusted_connection": "yes",	"Encrypt": "no","TrustServerCertificate": "yes"	},	fast_executemany=True,	pool_pre_ping=True)
			with self.engine.connect() as conn:
				conn.execute(text("SELECT 1"))
			self.Session = sessionmaker(bind=self.engine)
		except OperationalError as e:
			raise RuntimeError(" PTADatabase: Database connection failed. Check server name, database name, network connectivity, and authentication settings.") from e
		except pyodbc.Error as e:
			raise RuntimeError(" PTADatabase: ODBC driver error. Verify ODBC Driver 18 for SQL Server is installed and connection parameters are correct.") from e
		except SQLAlchemyError as e:
			raise RuntimeError(" PTADatabase: SQLAlchemy engine initialization failed.") from e
		except Exception as e:
			raise RuntimeError(" PTADatabase: Unexpected error while initializing database engine.") from e

	def __init__(self, verbose: bool = False):
		self.verbose = verbose
		self.server = ReadConfigString("Database", "DatabaseServer")
		self.database = ReadConfigString("Database", "DatabaseName")
		self.username = ReadConfigString("Database", "DatabaseUsername")
		self.password = ReadConfigString("Database", "DatabasePassword")
		url = ReadConfigString("Database", "ConnectionString")
		self.use_trusted = True
		self.engine = None
		self.Session = None
		self.session = None
		self.database_configured = False
		if url == '': url = SQLAlchemy_Connection_URL(server=self.server, database=self.database, username=self.username, password=self.password, use_trusted=self.use_trusted)
		if url:
			self.engine = create_engine(url, connect_args={'trusted_connection': 'yes', 'Encrypt': 'no',	'TrustServerCertificate': 'yes'	}, fast_executemany=True, pool_pre_ping=True)
			self.Session = sessionmaker(bind=self.engine)
			self.database_configured  = True
			if self.verbose: print(" PTADatabase: SQLAlchemy engine created")
		else:
			if self.verbose: print(" PTADatabase: Database config missing — SQL disabled")

	def Open(self):
		if self.engine:
			self.session = self.Session()
			if self.verbose:
				print("SQLAlchemy session opened")
			return True
		return False

	def Close(self):
		if self.session:
			self.session.close()
			if self.verbose: print(" PTADatabase: SQLAlchemy session closed")

	def ExecSQL(self, sql: str, params: dict | None = None):
		try:
			with self.engine.begin() as conn:
				conn.execute(text(sql), params or {})
		except ProgrammingError as e:
			raise RuntimeError(f"S PTADatabase: QL syntax or parameter error in ExecSQL: {sql}") from e
		except OperationalError as e:
			raise RuntimeError(" PTADatabase: Database operation failed (connection/timeout).") from e
		except SQLAlchemyError as e:
			raise RuntimeError(f" PTADatabase: ExecSQL failed: {sql}") from e

	def ScalarListFromSQL(self, sql: str, params: dict | None = None, column: str | None = None):
		try:
			df = pd.read_sql_query(text(sql), self.engine, params=params)
			if df.empty:
				return []
			if column:
				if column not in df.columns:
					raise KeyError(f"Column '{column}' not found in result set")
				return df[column].tolist()
			return df.iloc[:, 0].tolist()
		except ProgrammingError as e:
			raise RuntimeError(f" PTADatabase: SQL syntax or parameter error in ScalarListFromSQL: {sql}") from e
		except SQLAlchemyError as e:
			raise RuntimeError(f" PTADatabase: ScalarListFromSQL failed: {sql}") from e

	def DataFrameFromSQL(self, sql: str, params=None, indexName=None):
		try:
			return pd.read_sql_query(text(sql), self.engine, params=params, index_col=indexName)
		except ProgrammingError as e:
			raise RuntimeError(f" PTADatabase: SQL syntax or parameter error in DataFrameFromSQL: {sql}") from e
		except SQLAlchemyError as e:
			raise RuntimeError(f" PTADatabase: DataFrameFromSQL failed: {sql}") from e

	def DataFrameToSQL(self, df: pd.DataFrame, tableName: str, indexAsColumn: bool = False, clearExistingData: bool = False):
		try:
			if indexAsColumn:
				df = df.reset_index()
			if clearExistingData:
				with self.engine.begin() as conn:
					conn.execute(text(f"IF OBJECT_ID('{tableName}') IS NOT NULL DELETE FROM {tableName}"))
			if df.empty:
				return  # nothing to insert
			df.to_sql(tableName, con=self.engine, schema="dbo", if_exists="append", index=False)
		except ProgrammingError as e:
			raise RuntimeError(f" PTADatabase: SQL error while writing to table '{tableName}'") from e
		except OperationalError as e:
			raise RuntimeError(f" PTADatabase: Database write failed for table '{tableName}'") from e
		except SQLAlchemyError as e:
			raise RuntimeError(f" PTADatabase: DataFrameToSQL failed for table '{tableName}'") from e

