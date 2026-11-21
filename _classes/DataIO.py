import time, pandas as pd
import numpy as np
import yfinance as yf
import urllib.parse
import pyodbc
import functools
from datetime import datetime, timedelta
from pandas.tseries.offsets import BDay
from contextlib import suppress
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError, OperationalError, ProgrammingError
from pandas.errors import DatabaseError as PandasDatabaseError
from curl_cffi.requests.exceptions import HTTPError
from _classes.Utility import ReadConfigString, ReadConfigBool, GetLatestBDay
from _classes.PriceQuality import clean_price_history, QUALITY_DDL, MIN_ROWS

#-------------------------------------------- SQL Setup and Helpers  -----------------------------------------------
SQL_NUMERIC_MAX = 999_999_999_999.999999 #Safety check for garbage prices
BASE_FIELD_LIST = ['Open','Close','High','Low','Volume']
RAW_FIELD_LIST = ['CloseRaw','VolumeRaw']	#As-traded close and share volume, for point-in-time filters (price floors, dollar volume)

def isfloat(num):
	try:
		float(num)
		return True
	except ValueError:
		return False

_NET_ERR_TERMS = ("08001", "08s01", "10060", "10054", "timeout", "link failure", "communication link")

def retry_sql_on_timeout(retries=5, delay=3, backoff=2, max_delay=60):
	#pandas.read_sql_query wraps SQLAlchemy errors in pandas.errors.DatabaseError, so that must be caught too
	def decorator(func):
		@functools.wraps(func)
		def wrapper(self, *args, **kwargs):
			wait = delay
			for attempt in range(1, retries + 1):
				try:
					return func(self, *args, **kwargs)
				except (SQLAlchemyError, PandasDatabaseError, pyodbc.Error, RuntimeError) as e:
					err_msg = str(e).lower()
					if attempt == retries or not any(term in err_msg for term in _NET_ERR_TERMS):
						raise
					print(f" PTADatabase: Wi-Fi/Network glitch in {func.__name__} ({attempt}/{retries}). Retrying in {wait}s...")
					if getattr(self, "engine", None) is not None:
						self.engine.dispose()	#drop pooled connections that may share the dead socket
					time.sleep(wait)
					wait = min(wait * backoff, max_delay)
		return wrapper
	return decorator

def SQLAlchemy_Connection_URL(server: str | None, database: str | None, username: str | None, password: str | None, use_trusted: bool = True):
	if not server or not database:
		return None
	driver = "ODBC+Driver+18+for+SQL+Server"
	params = f"driver={driver}&LoginTimeout=30&timeout=120&KeepAlive=30"
	if username and password:
		safe_password = urllib.parse.quote_plus(password)
		url =  f"mssql+pyodbc://{username}:{safe_password}@{server}/{database}?{params}&Encrypt=yes&TrustServerCertificate=yes"
		print(url)
		return url
	if use_trusted:
		return f"mssql+pyodbc://@{server}/{database}?{params}&trusted_connection=yes&Encrypt=yes&TrustServerCertificate=yes"
	return None

def _filter_sql_numeric_overflow(df: pd.DataFrame, ticker: str,	cols: list[str]	) -> pd.DataFrame:
	#Drops rows where any column exceeds NUMERIC(11,5) limits.
	mask = (df[cols].abs() <= SQL_NUMERIC_MAX).all(axis=1)
	dropped = (~mask).sum()
	if dropped > 0:
		print(f" DataDownload: WARNING: {ticker} dropped {dropped} rows exceeding NUMERIC(11,5)")
		print(df[~mask])
	return df.loc[mask]

def _adjust_yahoo_history(df: pd.DataFrame) -> pd.DataFrame:
	#Input: yfinance history(auto_adjust=False). Yahoo's Open/High/Low/Close are split-adjusted only, 'Adj Close' is also
	#dividend-adjusted, and Volume is split-adjusted.
	#Output: Open/High/Low/Close adjusted for splits AND dividends (identical to auto_adjust=True), Volume unchanged,
	#plus CloseRaw/VolumeRaw = the price and share volume as actually traded that day.
	df = df.copy()
	cols = {c.lower().replace(' ', ''): c for c in df.columns}
	close = pd.to_numeric(df[cols['close']], errors='coerce')
	if 'stocksplits' in cols:
		splits = pd.to_numeric(df[cols['stocksplits']], errors='coerce').fillna(0).replace(0, 1.0)
	else:
		splits = pd.Series(1.0, index=df.index)
	f = splits[::-1].cumprod()[::-1].shift(-1).fillna(1.0)	#combined ratio of every split AFTER each date (a split day already trades on the new basis)
	df['CloseRaw'] = close * f
	df['VolumeRaw'] = (pd.to_numeric(df[cols['volume']], errors='coerce') / f).round()
	if 'adjclose' in cols:
		k = (pd.to_numeric(df[cols['adjclose']], errors='coerce') / close).where(close > 0)
		k = k.replace([float('inf'), float('-inf')], float('nan'))	#NaN rows are dropped later, as auto_adjust=True would
	else:
		k = 1.0
	for c in ('open', 'high', 'low', 'close'):
		df[cols[c]] = pd.to_numeric(df[cols[c]], errors='coerce') * k
	return df

_START_OVERRIDES = None
_MANUAL_SPLITS = {}
def _get_start_override(ticker: str):
	#Manual start dates from dbo.PriceStartOverride (ticker reuse the quality checks can't detect). Loaded once per run,
	#which also creates the quality tables and columns if they are missing.
	global _START_OVERRIDES, _MANUAL_SPLITS
	if _START_OVERRIDES is None:
		_START_OVERRIDES = {}
		try:
			db = PTADatabase()
			if db.database_configured:
				#Make sure PricesDaily has CloseRaw/VolumeRaw/Repaired before any save: _save_to_sql deletes
				#before it inserts, so an insert failing on a missing column would lose that ticker's rows
				with db.engine.begin() as conn:
					for ddl in QUALITY_DDL:
						conn.execute(text(ddl))
				o = pd.read_sql_query(text("SELECT Ticker, StartDate FROM dbo.PriceStartOverride"), db.engine)
				_START_OVERRIDES = {str(t).upper(): pd.Timestamp(d) for t, d in zip(o['Ticker'], o['StartDate'])}
				m = pd.read_sql_query(text("SELECT Ticker, SplitDate, Ratio FROM dbo.PriceManualSplits"), db.engine)
				for t, d, r in zip(m['Ticker'], m['SplitDate'], m['Ratio']):
					_MANUAL_SPLITS.setdefault(str(t).upper(), []).append((pd.Timestamp(d), float(r)))
		except Exception:
			pass	#database unavailable: no overrides
	return _START_OVERRIDES.get(ticker.upper())

NEW_LISTING_DAYS = 120				#a series under MIN_ROWS that starts within this many days and is current is a new listing: returned, not rejected

#-------------------------------------------- Yahoo Finance Ticker Info and Financials Helpers -----------------------------------------------
FIN_SCALE = 1_000_000				#TickerFinancials dollar amounts and share counts are stored in millions (same units the old Google scraper used)
MARKETCAP_SCALE = 1_000_000_000		#Tickers.MarketCap is stored in billions (same units the old Google scraper used)
YAHOO_EXCHANGE_MAP = {
	'NMS':'NASDAQ', 'NGM':'NASDAQ', 'NCM':'NASDAQ', 'NAS':'NASDAQ',
	'NYQ':'NYSE', 'NYS':'NYSE', 'PCX':'NYSEARCA', 'ASE':'NYSEMKT', 'BTS':'BATS',
	'PNK':'OTCMKTS', 'OQB':'OTCMKTS', 'OQX':'OTCMKTS', 'OEM':'OTCMKTS',
	'SNP':'INDEXSP', 'DJI':'INDEXDJX', 'NIM':'INDEXNASDAQ', 'WCB':'INDEXCBOE',
	'LSE':'LON', 'MEX':'BMV', 'TOR':'TSE',
}

try:
	from yfinance.exceptions import YFRateLimitError
except Exception:
	class YFRateLimitError(Exception): pass

def _yahoo_symbol(ticker: str) -> str:
	#Converts a database ticker to the Yahoo symbol, same rules as DownloadPriceDataYahooFinance
	ticker = ticker.upper().replace('.INX', '^SPX')
	if ticker.startswith('.'): ticker = ticker.replace('.', '^')
	return ticker.replace('.', '-')

def _yahoo_call(fn, what: str, retries: int = 3, delay: float = 20):
	#Calls a yfinance property/method, backing off on rate limits. Returns None on any other failure.
	for i in range(retries):
		try:
			return fn()
		except YFRateLimitError:
			wait = delay * (i + 1)
			print(f" DataDownload: Yahoo rate limit on {what}, waiting {wait}s ({i+1}/{retries})")
			time.sleep(wait)
		except Exception as e:
			print(f" DataDownload: Yahoo request failed for {what}: {e}")
			return None
	return None

def _num(v):
	#Float or None. Handles None, NaN, inf, numpy types and strings like 'Infinity'.
	if v is None: return None
	try:
		f = float(v)
	except (TypeError, ValueError):
		return None
	return f if np.isfinite(f) else None

def _stmt_frame(df):
	#yfinance statements have line items as rows and period end dates as columns (newest first). Returns oldest->newest, or None.
	if df is None or not isinstance(df, pd.DataFrame) or df.empty: return None
	df = df.copy()
	df.columns = pd.to_datetime(df.columns, errors='coerce')
	df = df.loc[:, df.columns.notna()]
	df = df.loc[:, ~df.columns.duplicated()]
	if df.empty: return None
	return df.sort_index(axis=1)

def _stmt_value(df, labels: list, period):
	#First non-empty value among the candidate line item labels for the period
	if df is None or period not in df.columns: return None
	for label in labels:
		if label in df.index:
			v = df.loc[label, period]
			if isinstance(v, pd.Series): v = v.iloc[0]
			v = _num(v)
			if v is not None: return v
	return None

def _prior_value(s: pd.Series, minDays: int, maxDays: int) -> pd.Series:
	#For each period, the most recent value from a period minDays..maxDays earlier (NaN if that quarter is missing)
	out = pd.Series(np.nan, index=s.index, dtype=float)
	valid = s.dropna()
	for d in s.index:
		m = valid[(valid.index <= d - pd.Timedelta(days=minDays)) & (valid.index >= d - pd.Timedelta(days=maxDays))]
		if len(m): out[d] = m.iloc[-1]
	return out

def _pct_change(cur: pd.Series, prev: pd.Series) -> pd.Series:
	#Percent change using abs(prev) so a move from a loss toward a profit reads as positive
	prev = prev.where(prev != 0)
	return (cur - prev) / prev.abs() * 100

def _ttm(s: pd.Series) -> pd.Series:
	#Trailing twelve months: sum of the last 4 quarters when all 4 are present, else the quarter annualized
	r = s.rolling('350D', min_periods=1)
	full = r.sum().where(r.count() == 4)
	return full.fillna(s * 4)

def _sql_value(v):
	v = _num(v)
	return None if v is None else round(v, 6)

#-------------------------------------------- DataDownload Class  -----------------------------------------------
class DataDownload:
	def _GetYahooInfo(self, ticker: str, maxAgeSeconds: int = 600):
		#Returns (yf.Ticker, info dict). Cached briefly so DownLoadTickerInfo + DownloadTickerFinancials on the same ticker hit Yahoo once.
		ticker = ticker.upper()
		cache = getattr(self, '_yahoo_cache', None)
		if cache is None:
			cache = self._yahoo_cache = {}
		hit = cache.get(ticker)
		if hit and time.time() - hit[0] < maxAgeSeconds:
			return hit[1], hit[2]
		symbol = _yahoo_symbol(ticker)
		yt = yf.Ticker(symbol)
		info = _yahoo_call(lambda: yt.info, f"{symbol} info")
		if not isinstance(info, dict): info = {}
		cache[ticker] = (time.time(), yt, info)
		return yt, info

	def DownLoadTickerInfo(self, ticker: str, verbose: bool = True) -> bool:
		#Updates Tickers with company name, sector, industry, about, market cap (billions), P/E and dividend yield (%) from Yahoo Finance.
		#Exchange is only filled in when it is blank. Existing values are kept when Yahoo doesn't supply one.
		if not ticker: return False
		ticker = ticker.upper()
		if verbose: print(" DownLoadTickerInfo: " + ticker)
		yt, info = self._GetYahooInfo(ticker)
		name = info.get('longName') or info.get('shortName')
		quoteType = str(info.get('quoteType') or '').upper()
		if not name and not quoteType:
			print(f" DownLoadTickerInfo: No Yahoo info returned for {ticker}")
			return False
		sector = info.get('sector')
		industry = info.get('industry')
		if quoteType in ('ETF', 'MUTUALFUND'):
			sector = sector or 'Fund'
			industry = industry or info.get('category') or quoteType
		elif quoteType == 'INDEX':
			sector = sector or 'Index'
			industry = industry or 'Index'
		exchangeCode = str(info.get('exchange') or '').upper()
		exchange = YAHOO_EXCHANGE_MAP.get(exchangeCode, exchangeCode)[:11]
		price = _num(info.get('currentPrice')) or _num(info.get('regularMarketPrice')) or _num(info.get('previousClose'))
		marketCap = _num(info.get('marketCap'))
		pe = _num(info.get('trailingPE'))
		divRate = _num(info.get('trailingAnnualDividendRate')) or _num(info.get('dividendRate'))
		if divRate and price:
			dividend = round(divRate / price * 100, 4)
		elif quoteType in ('ETF', 'MUTUALFUND') and _num(info.get('yield')) is not None:
			dividend = round(_num(info.get('yield')) * 100, 4)
		elif quoteType == 'EQUITY':
			dividend = 0.0
		else:
			dividend = None
		about = info.get('longBusinessSummary') or info.get('description')
		params = {
			"Ticker": ticker,
			"Exchange": exchange,
			"CompanyName": name[:50] if name else None,
			"Sector": sector[:25] if sector else None,
			"Industry": industry[:100] if industry else None,
			"MarketCap": round(marketCap / MARKETCAP_SCALE, 2) if marketCap else None,
			"PE": round(pe, 4) if pe is not None else None,
			"Dividend": dividend,
			"About": about[:500] if about else None,
		}
		db = PTADatabase()
		if not db.Open(): return False
		try:
			db.ExecSQL("IF NOT EXISTS (SELECT 1 FROM Tickers WHERE Ticker=:Ticker) INSERT INTO Tickers (Ticker, Exchange) VALUES (:Ticker, :Exchange)", {"Ticker": ticker, "Exchange": exchange})
			db.ExecSQL("""UPDATE Tickers SET CompanyName=COALESCE(:CompanyName, CompanyName), Sector=COALESCE(:Sector, Sector), Industry=COALESCE(:Industry, Industry),
				MarketCap=COALESCE(:MarketCap, MarketCap), PE_Ratio=:PE, Dividend=COALESCE(:Dividend, Dividend), About=COALESCE(About, :About),
				Exchange=CASE WHEN Exchange IS NULL OR LTRIM(RTRIM(Exchange))='' THEN :Exchange ELSE Exchange END
				WHERE Ticker=:Ticker""", params)
		finally:
			db.Close()
		if verbose: print(f"  {params['CompanyName']} | {params['Sector']} | {params['Industry']} | MktCap(B): {params['MarketCap']} | P/E: {params['PE']} | Div%: {params['Dividend']}")
		return True

	def DownloadTickerFinancials(self, ticker: str, verbose: bool = True) -> bool:
		#Downloads Yahoo quarterly income statement, balance sheet and cash flow and writes one TickerFinancials row per fiscal
		#quarter (Year/Month = quarter end). Dollar amounts and shares in millions, margins/rates/returns in percent.
		#MPC = change vs prior quarter, YPC = change vs same quarter a year earlier. Price and valuation ratios come from the
		#current quote, so they're only written on the most recent quarter's row. Sets Tickers.FinancialsLastUpdated even when
		#Yahoo has no statements (funds, indexes, delisted) so DownloadAllFinancials doesn't retry them every run.
		if not ticker: return False
		ticker = ticker.upper()
		if verbose: print(" DownloadTickerFinancials: " + ticker)
		db = PTADatabase()
		if not db.Open(): return False
		try:
			yt, info = self._GetYahooInfo(ticker)
			quoteType = str(info.get('quoteType') or '').upper()
			if ticker.startswith(('.', '^')) or quoteType in ('ETF', 'MUTUALFUND', 'INDEX', 'CURRENCY', 'CRYPTOCURRENCY', 'FUTURE'):
				if verbose: print(f"  No financial statements for {ticker} ({quoteType or 'index'})")
				db.ExecSQL("UPDATE Tickers SET FinancialsLastUpdated=GETDATE() WHERE Ticker=:Ticker", {"Ticker": ticker})
				return False
			symbol = _yahoo_symbol(ticker)
			inc = _stmt_frame(_yahoo_call(lambda: yt.quarterly_income_stmt, f"{symbol} income statement"))
			bal = _stmt_frame(_yahoo_call(lambda: yt.quarterly_balance_sheet, f"{symbol} balance sheet"))
			cf = _stmt_frame(_yahoo_call(lambda: yt.quarterly_cashflow, f"{symbol} cash flow"))
			periods = sorted(set().union(*[set(d.columns) for d in (inc, bal, cf) if d is not None]))
			if not periods:
				print(f" DownloadTickerFinancials: No Yahoo financial statements for {ticker}")
				db.ExecSQL("UPDATE Tickers SET FinancialsLastUpdated=GETDATE() WHERE Ticker=:Ticker", {"Ticker": ticker})
				return False

			fields = {	#column: (statement, candidate Yahoo line items)
				'Revenue': (inc, ['Total Revenue', 'Operating Revenue']),
				'OperatingExpense': (inc, ['Operating Expense', 'Total Expenses']),
				'NetIncome': (inc, ['Net Income', 'Net Income Common Stockholders', 'Net Income Including Noncontrolling Interests']),
				'EarningsPerShare': (inc, ['Diluted EPS', 'Basic EPS']),
				'EBITDA': (inc, ['EBITDA', 'Normalized EBITDA']),
				'EBIT': (inc, ['EBIT', 'Operating Income']),
				'TaxProvision': (inc, ['Tax Provision']),
				'PretaxIncome': (inc, ['Pretax Income']),
				'TaxRateForCalcs': (inc, ['Tax Rate For Calcs']),
				'CashShortTermInvestments': (bal, ['Cash Cash Equivalents And Short Term Investments', 'Cash And Cash Equivalents']),
				'TotalAssets': (bal, ['Total Assets']),
				'TotalLiabilities': (bal, ['Total Liabilities Net Minority Interest', 'Total Liabilities']),
				'TotalEquity': (bal, ['Stockholders Equity', 'Common Stock Equity', 'Total Equity Gross Minority Interest']),
				'SharesOutstanding': (bal, ['Ordinary Shares Number', 'Share Issued']),
				'InvestedCapital': (bal, ['Invested Capital']),
				'TotalDebt': (bal, ['Total Debt']),
				'CashFromOperations': (cf, ['Operating Cash Flow', 'Cash Flow From Continuing Operating Activities']),
				'CashFromInvesting': (cf, ['Investing Cash Flow', 'Cash Flow From Continuing Investing Activities']),
				'CashFromFinancing': (cf, ['Financing Cash Flow', 'Cash Flow From Continuing Financing Activities']),
				'NetChangeInCash': (cf, ['Changes In Cash']),
				'FreeCashFlow': (cf, ['Free Cash Flow']),
				'CapitalExpenditure': (cf, ['Capital Expenditure']),
			}
			q = pd.DataFrame({col: [_stmt_value(src, labels, p) for p in periods] for col, (src, labels) in fields.items()}, index=pd.DatetimeIndex(periods), dtype=float)
			q['FreeCashFlow'] = q['FreeCashFlow'].fillna(q['CashFromOperations'] + q['CapitalExpenditure'])
			q['InvestedCapital'] = q['InvestedCapital'].fillna(q['TotalDebt'] + q['TotalEquity'])

			#Ratios and changes, computed in dollars before scaling
			q['NetProfitMargin'] = q['NetIncome'] / q['Revenue'].where(q['Revenue'] != 0) * 100
			pretax = q['PretaxIncome'].where(q['PretaxIncome'] > 0)
			q['EffectiveTaxRate'] = (q['TaxProvision'] / pretax * 100).fillna(q['TaxRateForCalcs'] * 100)
			taxFrac = (q['EffectiveTaxRate'] / 100).clip(0, 0.5).fillna(0.21)
			q['ReturnOnAssetts'] = _ttm(q['NetIncome']) / q['TotalAssets'].where(q['TotalAssets'] > 0) * 100
			q['ReturnOnCapital'] = _ttm(q['EBIT']) * (1 - taxFrac) / q['InvestedCapital'].where(q['InvestedCapital'] > 0) * 100
			for col in ('Revenue', 'NetIncome', 'TotalLiabilities'):
				q[col + 'MPC'] = _pct_change(q[col], _prior_value(q[col], 60, 120))
				q[col + 'YPC'] = _pct_change(q[col], _prior_value(q[col], 330, 400))

			#Current quote valuation, latest quarter only
			for col in ('Price', 'MarketCapitalization', 'PriceToEarnings', 'PriceToBook', 'PriceToCashFlow', 'PriceToSales'):
				q[col] = np.nan
			latest = q.index[-1]
			price = _num(info.get('currentPrice')) or _num(info.get('regularMarketPrice')) or _num(info.get('previousClose'))
			marketCap = _num(info.get('marketCap'))
			shares = _num(info.get('sharesOutstanding'))
			if price is not None: q.loc[latest, 'Price'] = round(price)	#TickerFinancials.Price is an int column
			if shares: q.loc[latest, 'SharesOutstanding'] = shares
			if marketCap:
				q.loc[latest, 'MarketCapitalization'] = marketCap
				revTTM = _ttm(q['Revenue']).iloc[-1]
				cfoTTM = _ttm(q['CashFromOperations']).iloc[-1]
				ps = _num(info.get('priceToSalesTrailing12Months'))
				q.loc[latest, 'PriceToSales'] = ps if ps is not None else (marketCap / revTTM if revTTM and revTTM > 0 else np.nan)
				q.loc[latest, 'PriceToCashFlow'] = marketCap / cfoTTM if cfoTTM and cfoTTM > 0 else np.nan
			q.loc[latest, 'PriceToEarnings'] = _num(info.get('trailingPE'))
			q.loc[latest, 'PriceToBook'] = _num(info.get('priceToBook'))

			moneyCols = ['Revenue', 'OperatingExpense', 'NetIncome', 'EBITDA', 'CashShortTermInvestments', 'TotalAssets', 'TotalLiabilities', 'TotalEquity',
				'SharesOutstanding', 'MarketCapitalization', 'CashFromOperations', 'CashFromInvesting', 'CashFromFinancing', 'NetChangeInCash', 'FreeCashFlow']
			netIncomeTTM = _ttm(q['NetIncome']).iloc[-1]
			revenueTTM = _ttm(q['Revenue']).iloc[-1]
			q[moneyCols] = q[moneyCols] / FIN_SCALE

			dbCols = ['Price', 'Revenue', 'RevenueMPC', 'RevenueYPC', 'OperatingExpense', 'NetIncome', 'NetIncomeMPC', 'NetIncomeYPC', 'NetProfitMargin',
				'EarningsPerShare', 'EBITDA', 'EffectiveTaxRate', 'CashShortTermInvestments', 'TotalAssets', 'TotalLiabilities', 'TotalLiabilitiesMPC',
				'TotalLiabilitiesYPC', 'TotalEquity', 'SharesOutstanding', 'MarketCapitalization', 'PriceToEarnings', 'PriceToBook', 'PriceToCashFlow',
				'PriceToSales', 'ReturnOnAssetts', 'ReturnOnCapital', 'CashFromOperations', 'CashFromInvesting', 'CashFromFinancing', 'NetChangeInCash', 'FreeCashFlow']
			records = {}
			for p, r in q.iterrows():
				if r[['Revenue', 'NetIncome', 'TotalAssets', 'CashFromOperations']].isna().all(): continue	#nothing useful for this quarter
				rec = {"Ticker": ticker, "Year": int(p.year), "Month": int(p.month)}
				for c in dbCols: rec[c] = _sql_value(r[c])
				if rec['Price'] is not None: rec['Price'] = int(rec['Price'])
				records[(rec['Year'], rec['Month'])] = rec	#one row per Year/Month, latest period wins
			records = list(records.values())
			if not records:
				print(f" DownloadTickerFinancials: No usable quarters for {ticker}")
				db.ExecSQL("UPDATE Tickers SET FinancialsLastUpdated=GETDATE() WHERE Ticker=:Ticker", {"Ticker": ticker})
				return False

			insertSQL = "INSERT INTO TickerFinancials (Ticker, Year, Month, " + ", ".join(dbCols) + ", TimeStamp) VALUES (:Ticker, :Year, :Month, " + ", ".join(':' + c for c in dbCols) + ", GETDATE())"
			last = records[-1]
			tickerParams = {
				"Ticker": ticker,
				"NetIncome": _sql_value(netIncomeTTM / FIN_SCALE) if _num(netIncomeTTM) is not None else None,
				"NetProfitMargin": _sql_value(netIncomeTTM / revenueTTM * 100) if _num(netIncomeTTM) is not None and _num(revenueTTM) else None,
				"ReturnOnCapital": last['ReturnOnCapital'],
			}
			with db.engine.begin() as conn:	#one transaction: a failed insert rolls back the delete
				conn.execute(text("DELETE FROM TickerFinancials WHERE Ticker=:Ticker AND Year=:Year AND Month=:Month"), [{"Ticker": r["Ticker"], "Year": r["Year"], "Month": r["Month"]} for r in records])
				conn.execute(text(insertSQL), records)
				conn.execute(text("""UPDATE Tickers SET FinancialsLastUpdated=GETDATE(), NetIncome=COALESCE(:NetIncome, NetIncome),
					NetProfitMargin=COALESCE(:NetProfitMargin, NetProfitMargin), ReturnOnCapital=COALESCE(:ReturnOnCapital, ReturnOnCapital) WHERE Ticker=:Ticker"""), tickerParams)
			if verbose:
				print(f"  Saved {len(records)} quarters ({records[0]['Year']}-{records[0]['Month']:02d} to {last['Year']}-{last['Month']:02d}) | Revenue(M): {last['Revenue']} | NetIncome(M): {last['NetIncome']} | EPS: {last['EarningsPerShare']}")
			return True
		except Exception as e:
			print(f" DownloadTickerFinancials: Failed for {ticker}: {e}")
			return False
		finally:
			db.Close()

	def DownloadPriceDataYahooFinance(self, ticker: str):
		#Returns Open/High/Low/Close adjusted for splits and dividends, split-adjusted Volume (same as before),
		#plus CloseRaw/VolumeRaw (as traded). See _adjust_yahoo_history.
		if not ticker: return None
		stored_ticker = ticker.upper()	#symbol as stored in PricesDaily, for quality checks and overrides
		ticker = ticker.upper().replace('.INX', '^SPX')
		if ticker.startswith('.'): ticker = ticker.replace('.', '^')
		ticker = ticker.replace('.', '-')
		print(ticker)
		try:
			df = yf.Ticker(ticker).history(start="1980-01-01", auto_adjust=False, repair=True)
		except HTTPError as e:
			print(f"WARNING: yfinance history() failed for {ticker}: {e}")
			df = None
		except Exception as e:
			print(f"WARNING: unexpected yfinance error for {ticker}: {e}")
			df = None
		if df is None or df.empty:
			print(f"ERROR: No Yahoo data returned for {ticker}")
			return None

		if isinstance(df.columns, pd.MultiIndex):
			df.columns = df.columns.get_level_values(-1)
		df.columns = [c.capitalize() for c in df.columns]
		missing = [c for c in BASE_FIELD_LIST if c not in df.columns]
		if missing:
			raise KeyError(f"Yahoo data missing columns for {ticker}: {list(df.columns)}")
		df = _adjust_yahoo_history(df)
		df = df[BASE_FIELD_LIST + RAW_FIELD_LIST]
		df = df.apply(pd.to_numeric, errors="coerce")
		df.dropna(inplace=True)
		df = _filter_sql_numeric_overflow(df, ticker=ticker, cols=BASE_FIELD_LIST + RAW_FIELD_LIST)
		if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
			df.index = df.index.tz_localize(None).normalize()	#exchange-local trading date (tz_convert(None) gave 04:00/05:00 UTC times)
		df.index.name = "Date"
		res = clean_price_history(df, ticker=stored_ticker, start_override=_get_start_override(stored_ticker),
								  manual_splits=_MANUAL_SPLITS.get(stored_ticker))
		if res.rejected:
			if len(res.df) < MIN_ROWS:	#e.g. Yahoo's single leftover quote for a delisted symbol: never save it
				if len(res.df) > 0 and res.df.index.min() >= pd.Timestamp.now() - pd.Timedelta(days=NEW_LISTING_DAYS) and res.df.index.max() >= pd.Timestamp.now() - pd.Timedelta(days=10):
					print(f"{stored_ticker}: new listing, only {len(res.df)} rows since {res.df.index.min().date()}; returned ({res.summary()})")
					return res.df	#brand-new ticker (IPO / new symbol): short but current, keep it
				print(f"WARNING: {stored_ticker} has too little data, not returned ({res.summary()})")
				return None
			#a ticker you refresh is kept up to date even if it looks like junk: repaired data is returned
			print(f"REVIEW: {stored_ticker} failed the junk checks; repaired data returned ({res.summary()})")
			return res.df
		if res.changed:
			print(f"{stored_ticker}: price quality fixes applied ({res.summary()})")
		return res.df

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
			self.engine = create_engine(url, connect_args={'trusted_connection': 'yes', 'Encrypt': 'no', 'TrustServerCertificate': 'yes'}, 	fast_executemany=True, pool_pre_ping=True, pool_recycle=60, pool_timeout=30)
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

	# def __init__(self, verbose: bool = False):
		# self.verbose = verbose
		# self.server = ReadConfigString("Database", "DatabaseServer")
		# self.database = ReadConfigString("Database", "DatabaseName")
		# self.username = ReadConfigString("Database", "DatabaseUsername")
		# self.password = ReadConfigString("Database", "DatabasePassword")
		# url = ReadConfigString("Database", "ConnectionString")
		# self.use_trusted = True
		# self.engine = None
		# self.Session = None
		# self.session = None
		# self.database_configured = False
		# if url == '': url = SQLAlchemy_Connection_URL(server=self.server, database=self.database, username=self.username, password=self.password, use_trusted=self.use_trusted)
		# if url:
			# self.engine = create_engine(
				# url, 
				# connect_args={'trusted_connection': 'yes', 'Encrypt': 'no', 'TrustServerCertificate': 'yes'}, 
				# fast_executemany=True, 
				# pool_pre_ping=True, 
				# pool_recycle=1800, 
				# pool_timeout=60
			# )
			# self.Session = sessionmaker(bind=self.engine)
			# self.database_configured  = True
			# if self.verbose: print(" PTADatabase: SQLAlchemy engine created")
		# else:
			# if self.verbose: print(" PTADatabase: Database config missing — SQL disabled")

	def __init__(self, verbose: bool = False):
		self.verbose = verbose
		self.server = ReadConfigString("Database", "DatabaseServer")
		self.database = ReadConfigString("Database", "DatabaseName")
		self.username = ReadConfigString("Database", "DatabaseUsername")
		self.password = ReadConfigString("Database", "DatabasePassword")
		url = ReadConfigString("Database", "ConnectionString")	
		self.use_trusted = True if not self.username else False
		self.engine = None
		self.Session = None
		self.database_configured = False
		if url == '':
			url = SQLAlchemy_Connection_URL(server=self.server, database=self.database, username=self.username, password=self.password, use_trusted=self.use_trusted)
		if url:
			c_args = {'Encrypt': 'yes','TrustServerCertificate': 'yes'}
			if self.use_trusted:
				c_args['trusted_connection'] = 'yes'
			else:
				c_args['trusted_connection'] = 'no'
			self.engine = create_engine(url,connect_args=c_args, fast_executemany=True, pool_pre_ping=True, pool_recycle=1800, pool_timeout=60)
			self.Session = sessionmaker(bind=self.engine)
			self.database_configured = True
			if self.verbose: print(f" PTADatabase: Engine created (Auth: {'Trusted' if self.use_trusted else 'SQL'})")
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
				result = conn.execute(text(sql), params or {})
				# If the statement has rows (like OUTPUT or SELECT), fetch them now
				if result.returns_rows:
					return result.all()  # Returns all rows as a list of Row objects
				return None
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

	@retry_sql_on_timeout()
	def ReplacePricesDaily(self, ticker: str, df: pd.DataFrame) -> int:
		#Replaces this ticker's rows from the first date in df onward with df, in ONE transaction: if the insert
		#fails, the delete is rolled back and nothing is lost. Dates are normalized to midnight and duplicate
		#dates dropped (last row wins), so the unique index on (Ticker, Date) can't be violated by the data.
		#df: Date as index or column, plus the PricesDaily columns. Returns rows written.
		if df is None or df.empty: return 0
		out = df.reset_index() if 'Date' not in df.columns else df.copy()
		d = pd.to_datetime(out['Date'])
		if getattr(d.dt, 'tz', None) is not None:
			d = d.dt.tz_convert(None)
		out['Date'] = d.dt.normalize()
		dups = out['Date'].duplicated(keep='last')
		if dups.any():
			days = out.loc[dups, 'Date']
			print(f" ReplacePricesDaily: {ticker} had {int(dups.sum())} duplicate dates ({days.min():%Y-%m-%d} to {days.max():%Y-%m-%d}); kept the last row of each")
			out = out[~dups]
		out['Ticker'] = ticker
		with self.engine.begin() as conn:
			conn.execute(text("DELETE FROM dbo.PricesDaily WHERE Ticker = CAST(:Ticker AS varchar(20)) AND [Date] >= :StartDate"),
						 {"Ticker": ticker, "StartDate": out['Date'].min().to_pydatetime()})
			out.to_sql('PricesDaily', con=conn, schema="dbo", if_exists="append", index=False)
		return len(out)

	@retry_sql_on_timeout()
	def LoadMultiTickerPriceHistory(self, tickers: list, startDate=None, endDate=None) -> pd.DataFrame | None:
		#One round trip for many tickers at once (flat frame, Ticker as a column) instead of a query per ticker.
		#Caller (PricingData.BatchLoadHistory) splits the result and normalizes per ticker.
		if not tickers: return None
		params = {f"t{i}": tk for i, tk in enumerate(tickers)}
		#CAST: pyodbc sends str as nvarchar; against the varchar Ticker column (SQL collation) that converts every row and scans the table instead of seeking
		placeholders = ", ".join(f"CAST(:{k} AS varchar(20))" for k in params)
		sql = f"SELECT [Ticker],[Date],[Open],[High],[Low],[Close],[Volume] FROM dbo.PricesDaily WHERE Ticker IN ({placeholders})"
		if startDate is not None:
			sql += " AND [Date] >= :start_date"
			params["start_date"] = startDate
		if endDate is not None:
			sql += " AND [Date] <= :end_date"
			params["end_date"] = endDate
		sql += " ORDER BY [Ticker], [Date]"
		if not self.Open(): return None
		df = self.DataFrameFromSQL(sql=sql, params=params)
		self.Close()
		return df

	@retry_sql_on_timeout()
	def UpdatePricesDailyRaw(self, ticker: str, df: pd.DataFrame) -> int:
		#Fills CloseRaw/VolumeRaw on rows already in PricesDaily for this ticker. Only empty values are filled:
		#as-traded prices never change, and rows loaded from EODHD already have them.
		#ticker is the symbol as stored in PricesDaily (not the Yahoo form). Returns the number of rows updated.
		if df is None or df.empty or not set(RAW_FIELD_LIST).issubset(df.columns): return 0
		recs = [{"Ticker": ticker, "Date": pd.Timestamp(d).normalize().to_pydatetime(), "CloseRaw": float(c), "VolumeRaw": int(v)}
				for d, c, v in zip(df.index, df['CloseRaw'], df['VolumeRaw'])
				if np.isfinite(c) and np.isfinite(v) and 0 < c <= SQL_NUMERIC_MAX and 0 <= v < 9.2e18]
		if not recs: return 0
		with self.engine.begin() as conn:
			conn.execute(text("IF OBJECT_ID('tempdb..#RawStage') IS NOT NULL DROP TABLE #RawStage"))
			#CloseRaw is staged as float and converted in SQL: fast_executemany can send Python floats as text,
			#which SQL Server refuses to convert straight to numeric (error 8114) but converts to float fine
			conn.execute(text("CREATE TABLE #RawStage (Ticker varchar(20) NOT NULL, [Date] smalldatetime NOT NULL, CloseRaw float NULL, VolumeRaw bigint NULL)"))
			conn.execute(text("INSERT INTO #RawStage (Ticker, [Date], CloseRaw, VolumeRaw) VALUES (:Ticker, :Date, :CloseRaw, :VolumeRaw)"), recs)
			result = conn.execute(text("""UPDATE p SET CloseRaw = CAST(ROUND(r.CloseRaw, 6) AS numeric(18,6)), VolumeRaw = r.VolumeRaw
				FROM dbo.PricesDaily p JOIN #RawStage r ON p.Ticker = r.Ticker AND p.[Date] = r.[Date]
				WHERE p.CloseRaw IS NULL"""))
			updated = result.rowcount
			conn.execute(text("DROP TABLE #RawStage"))
		return updated

#-------------------------------------------- One-time backfill  -----------------------------------------------
def BackfillRawPriceColumns(tickers: list[str] | None = None, pause: float = 0.5):
	#Fills CloseRaw/VolumeRaw for existing PricesDaily rows from Yahoo. Default: every ticker that still has empty
	#raw values, except .DL tickers (delisted, loaded from EODHD with raw values). Safe to stop and re-run.
	db = PTADatabase()
	if tickers is None:
		tickers = db.ScalarListFromSQL("SELECT DISTINCT Ticker FROM dbo.PricesDaily WHERE CloseRaw IS NULL AND Ticker NOT LIKE '%.DL%' ORDER BY Ticker")
	print(f"BackfillRawPriceColumns: {len(tickers)} tickers")
	dd = DataDownload()
	for i, t in enumerate(tickers, 1):
		df = dd.DownloadPriceDataYahooFinance(t)
		if df is None or df.empty:
			print(f" [{i}/{len(tickers)}] {t}: no Yahoo data, skipped")
			continue
		try:
			n = db.UpdatePricesDailyRaw(t, df)
		except Exception as e:	#log and keep going; the ticker is retried on the next run
			print(f" [{i}/{len(tickers)}] {t}: FAILED ({str(e).splitlines()[0][:200]})")
			continue
		print(f" [{i}/{len(tickers)}] {t}: {n} rows filled")
		time.sleep(pause)
