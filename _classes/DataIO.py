import time, ssl, requests, pandas as pd
import urllib.error, urllib.request as webRequest
import pyodbc
import functools
import yfinance as yf
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
from contextlib import suppress
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError, OperationalError, ProgrammingError
from curl_cffi.requests.exceptions import HTTPError
from _classes.Utility import ReadConfigString, ReadConfigBool, GetLatestBDay

#-------------------------------------------- SQL Setup and Helpers  -----------------------------------------------
SQL_NUMERIC_MAX = 999_999_999_999.999999 #Safety check for garbage prices
BASE_FIELD_LIST = ['Open','Close','High','Low','Volume']

def isfloat(num):
	try:
		float(num)
		return True
	except ValueError:
		return False

def retry_sql_on_timeout(retries=3, delay=5):
	"""Decorator to catch transient SQL connection timeouts and retry."""
	def decorator(func):
		@functools.wraps(func)
		def wrapper(*args, **kwargs):
			last_ex = None
			for i in range(retries):
				try:
					return func(*args, **kwargs)
				except (OperationalError, RuntimeError, SQLAlchemyError) as e:
					err_msg = str(e).lower()
					if "08001" in err_msg or "timeout" in err_msg or "delay" in err_msg:
						print(f" PTADatabase: Timeout detected. Retrying ({i+1}/{retries}) in {delay}s...")
						time.sleep(delay)
						last_ex = e
						continue
					raise e
			raise last_ex
		return wrapper
	return decorator

def SQLAlchemy_Connection_URL(server: str | None, database: str | None, username: str | None, password: str | None, use_trusted: bool = True):
	if not server or not database:
		return None
	driver = "ODBC+Driver+18+for+SQL+Server"
	params = f"driver={driver}&LoginTimeout=60"
	if username and password:
		return f"mssql+pyodbc://{username}:{password}@{server}/{database}?{params}"
	if use_trusted:
		return f"mssql+pyodbc://@{server}/{database}?{params}&trusted_connection=yes"
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
		currentDate = GetLatestBDay()
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
		missing = [c for c in BASE_FIELD_LIST if c not in df.columns]
		if missing:
			raise KeyError(f"Yahoo data missing columns for {ticker}: {list(df.columns)}")
		df = df[BASE_FIELD_LIST]
		df = df.apply(pd.to_numeric, errors="coerce")
		df.dropna(inplace=True)
		df = _filter_sql_numeric_overflow(df, ticker=ticker, cols=BASE_FIELD_LIST)
		if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
			df.index = df.index.tz_convert(None)
		df.index.name = "Date"
		return df

	def get_current_options(ticker):
		from math import log
		def graded_score_ratio(ratio, neutral=1.0, sensitivity=0.5, max_score=2.0):
			if ratio is None or ratio <= 0:
				return 0.0
			score = log(ratio / neutral)
			score = max(-max_score, min(max_score, score / sensitivity))
			return round(score, 3)
		group_weights = { "LT_ITM": 0.6, "ST_ITM": 0.6, "ST_OTM": 1.0, "LT_OTM": 1.0 }
		metric_weights = { "vol_ratio": 0.8, "oi_ratio": 0.8, "iv_skew": 0.5, "vw_price": 1.5 }
		current_date = GetLatestBDay()
		try:
			print(f" get_current_options: Getting options for ticker: {ticker}")
			stock = yf.Ticker(ticker)
			expirations = stock.options
			if not expirations:
				return None
			short_term_days = 14
			long_term_days = 45
			valid_exps = [
				e for e in expirations
				if 0 < (datetime.strptime(e, "%Y-%m-%d").date() - current_date).days <= long_term_days
			]
			if not valid_exps:
				return None
			options_data = []
			for exp in valid_exps:
				dte = (datetime.strptime(exp, "%Y-%m-%d").date() - current_date).days
				chain = stock.option_chain(exp)
				for opt_type, df in zip(["call", "put"], [chain.calls, chain.puts]):
					df = df.copy()
					df["type"] = opt_type
					df["expiration"] = exp
					df["dte"] = dte
					options_data.append(df)
			all_options = pd.concat(options_data)
			spot = stock.history(period="1d")["Close"][-1]
			all_options = all_options[
				(all_options["volume"] > 5) &
				(all_options["openInterest"] > 5) &
				(all_options["lastPrice"] > all_options["strike"] * 0.003) &
				((all_options["bid"] > 0) | (all_options["ask"] > 0)) &
				(all_options["impliedVolatility"] < 6)
			]

			all_options["ITM"] = all_options["inTheMoney"]
			all_options["term"] = all_options["dte"].apply(lambda x: "ST" if x <= short_term_days else "LT")
			final_df = all_options.copy()

			group_stats = {}
			for term in ["ST", "LT"]:
				for itm_status in [True, False]:
					subset_puts = final_df[(final_df["term"] == term) & (final_df["ITM"] == itm_status) & (final_df["type"] == "put")]
					subset_calls = final_df[(final_df["term"] == term) & (final_df["ITM"] == itm_status) & (final_df["type"] == "call")]
					key = f"{term}_{'ITM' if itm_status else 'OTM'}"

					put_vol = subset_puts["volume"].sum()
					call_vol = subset_calls["volume"].sum()
					put_oi = subset_puts["openInterest"].sum()
					call_oi = subset_calls["openInterest"].sum()
					put_iv = subset_puts["impliedVolatility"].mean()
					call_iv = subset_calls["impliedVolatility"].mean()

					def vw_price(df, is_put):
						if df.empty:
							return None
						if is_put:
							return ((df["strike"] - df["lastPrice"]) * df["volume"]).sum() / df["volume"].sum()
						else:
							return ((df["strike"] + df["lastPrice"]) * df["volume"]).sum() / df["volume"].sum()

					group_stats[f"{key}_put_call_ratio_vol"] = put_vol / call_vol if call_vol else None
					group_stats[f"{key}_put_call_ratio_oi"] = put_oi / call_oi if call_oi else None
					group_stats[f"{key}_iv_skew"] = put_iv - call_iv if pd.notnull(put_iv) and pd.notnull(call_iv) else None
					group_stats[f"{key}_volume_weighted_price"] = vw_price(subset_calls, False) if call_vol >= put_vol else vw_price(subset_puts, True)
					group_stats[f"{key}_volume_total"] = put_vol + call_vol

			sentiment_score = 0.0
			sentiment_reasons = []

			for key_base in group_weights:
				vol_ratio = group_stats.get(f"{key_base}_put_call_ratio_vol")
				oi_ratio = group_stats.get(f"{key_base}_put_call_ratio_oi")
				iv_skew = group_stats.get(f"{key_base}_iv_skew")
				vw_price = group_stats.get(f"{key_base}_volume_weighted_price")

				local_score = 0.0
				local_reasons = []
				group_weight = group_weights[key_base]

				if vol_ratio is not None:
					score = graded_score_ratio(-vol_ratio) * metric_weights["vol_ratio"]
					local_score += score
					if abs(score) > 0.1:
						local_reasons.append(f"Volume ratio: {score:+.2f}")

				if oi_ratio is not None:
					score = graded_score_ratio(-oi_ratio) * metric_weights["oi_ratio"]
					local_score += score
					if abs(score) > 0.1:
						local_reasons.append(f"OI ratio: {score:+.2f}")

				if iv_skew is not None:
					score = max(-1.0, min(1.0, -iv_skew / 0.02)) * metric_weights["iv_skew"]
					local_score += score
					if abs(score) > 0.1:
						local_reasons.append(f"IV skew: {score:+.2f}")

				if vw_price and spot:
					price_ratio = (vw_price - spot) / spot
					score = max(-1.0, min(1.0, price_ratio * 10)) * metric_weights["vw_price"]
					local_score += score
					if abs(score) > 0.1:
						local_reasons.append(f"VW price signal: {score:+.2f}")

				weighted_score = local_score * group_weight
				sentiment_score += weighted_score
				if local_reasons:
					sentiment_reasons.append(f"{key_base}: {', '.join(local_reasons)} (×{group_weight})")

			if sentiment_score >= 2.0:
				sentiment_label = "Strong Bullish"
			elif sentiment_score >= 1.0:
				sentiment_label = "Bullish"
			elif sentiment_score <= -2.0:
				sentiment_label = "Strong Bearish"
			elif sentiment_score <= -1.0:
				sentiment_label = "Bearish"
			else:
				sentiment_label = "Neutral"

			result = {
				"ticker": ticker,
				"date": current_date,
				"spot": float(spot),
				"sentiment_score": round(sentiment_score, 3),
				"sentiment_label": sentiment_label,
				"sentiment_explanation": "; ".join(sentiment_reasons)
			}
			result.update(group_stats)
			return result

		except Exception as e:
			print(f" get_current_options: Error fetching {ticker}: {e}")
			return None

	def DownloadCurrentOptions(tickers: list):
		all_data = []
		for ticker in tickers:
			data = get_current_options(ticker)
			if data:
				all_data.append(data)
			wait_time = 1
			print(f' DownloadCurrentOptions: Waiting {wait_time} seconds to prevent API overload...')
			time.sleep(wait_time)

		if not all_data:
			print(" DownloadCurrentOptions: No data retrieved.")
			return pd.DataFrame()
		df = pd.DataFrame(all_data)
		return df

	def UpdateCurrentOptions(tickers: list):
		table_name = "Options_Sentiment_Daily"
		current_options = DownloadCurrentOptions(tickers)
		db = PTADatabase()
		if db.Open() and len(current_options) > 0:
			current_date = GetLatestBDay()  
			SQL = f"DELETE FROM {table_name} WHERE [Date]='{current_date.strftime('%Y-%m-%d')}'"
			#print(SQL)
			db.ExecSQL(SQL)
			db.DataFrameToSQL(current_options, tableName=table_name, indexAsColumn=False, clearExistingData=False)
			db.Close()
			print(f" UpdateCurrentOptions: Appended data for {len(current_options)} tickers to {table_name}")

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
			self.engine = create_engine(
				url, 
				connect_args={'trusted_connection': 'yes', 'Encrypt': 'no', 'TrustServerCertificate': 'yes'}, 
				fast_executemany=True, 
				pool_pre_ping=True, 
				pool_recycle=1800, 
				pool_timeout=60
			)
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

	@retry_sql_on_timeout()
	def ExecSQL(self, sql: str, params: dict | None = None):
		try:
			with self.engine.begin() as conn:
				conn.execute(text(sql), params or {})
		except (ProgrammingError, OperationalError, SQLAlchemyError) as e:
			if isinstance(e, ProgrammingError):
				raise RuntimeError(f" PTADatabase: SQL syntax error: {sql}") from e
			raise e

	@retry_sql_on_timeout()
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
		except (ProgrammingError, OperationalError, SQLAlchemyError) as e:
			raise e

	@retry_sql_on_timeout()
	def DataFrameFromSQL(self, sql: str, params=None, indexName=None):
		try:
			return pd.read_sql_query(text(sql), self.engine, params=params, index_col=indexName)
		except (ProgrammingError, OperationalError, SQLAlchemyError) as e:
			raise e
			
	@retry_sql_on_timeout()
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
		except (ProgrammingError, OperationalError, SQLAlchemyError) as e:
			raise e
			
