import sys, time, datetime, os, shutil, pandas as pd
import _classes.Constants as CONSTANTS
from multiprocessing.dummy import Pool as ThreadPool
from datetime import datetime, timedelta
from _classes.DataIO import PTADatabase, DataDownload
from _classes.Prices import PricingData
from _classes.Trading import TradingModel, TradeModelParams, ExtensiveTesting, UpdateTradeModelComparisonsFromDailyValue
from _classes.Selection import StockPicker, Generate_Picks_For_SQL_DateRange
from _classes.TickerLists import TickerLists
from _classes.Utility import *

FORCE_INCLUDE_TICKERS = ['GOOGL','TER','JNJ','TECK','ASML','LRCX','CF','FANG','GLW','DLTR','NEM','PWR','THC','DDOG','SII','WPM','AAPL','RCL','HWM','VTR','TEAM','VRT','NVDA','SNOW','MN','GOOGL','CORT','DYN','TPR','MSFT','AVGO','PLTR','CRK','CEG','STX','MRK','PR','WDC','VIAV']

def PopulateDatabaseTickers():
	csvFile = 'database/tickerlist.csv'
	if os.path.isfile(csvFile):
		db = PTADatabase()
		if db.Open():
			count = db.ScalarListFromSQL("SELECT COUNT(*) AS Cnt FROM Tickers")[0]
			if count < 25:
				db.ExecSQL("DELETE FROM Tickers")
				df = pd.read_csv(csvFile)
				db.DataFrameToSQL(df, "Tickers")
		db.Close()
	dd = DataDownload()
	for ticker in TickerLists.SP500_2022():
		TickerFullRefresh(ticker)
		dd.DownLoadTickerInfo(ticker)
		dd.DownloadTickerFinancials(ticker)
	for ticker in TickerLists.BigList2021():
		TickerFullRefresh(ticker)
		dd.DownLoadTickerInfo(ticker)
		dd.DownloadTickerFinancials(ticker)

def DownloadAllTickerInfo():
	# For all Tickers, updates missing About and financial info
	db = PTADatabase()
	dd = DataDownload()
	if db.Open():
		df = db.DataFrameFromSQL("SELECT Ticker, Exchange FROM Tickers WHERE Delisted <> 1 AND (CompanyName IS NULL OR FinancialsLastUpdated IS NULL or Industry is null)")
		for _, row in df.iterrows():
			dd.DownLoadTickerInfo(row.Ticker)
			dd.DownloadTickerFinancials(row.Ticker)
		db.Close()
		
def DownloadAllFinancials():
	db = PTADatabase()
	dd = DataDownload()
	if db.Open():
		df = db.DataFrameFromSQL("SELECT Ticker, Exchange FROM Tickers WHERE Delisted <> 1 AND (FinancialsLastUpdated IS NULL or DateDiff(d, FinancialsLastUpdated, GetDate()) > 90)")
		for _, row in df.iterrows():
			dd.DownLoadTickerInfo(row.Ticker)
			dd.DownloadTickerFinancials(row.Ticker)
		db.Close()
		
def DownloadDaily(tickerList: list):
	# Download daily prices for each ticker using yesterday's date
	print(' DownloadDaily: Downloading daily prices for ' + str(len(tickerList)) + ' tickers')
	yesterday = GetLatestBDay()
	tickersUpdated = []
	for ticker in tickerList:
		p = PricingData(ticker)
		if p.LoadHistory(requestedEndDate=yesterday):
			tickersUpdated.append(ticker)
	print(" DownloadDaily: Requested: " + str(len(tickerList)) + " Completed: " + str(len(tickersUpdated)))

def TickerDataRefresh(Daily: bool = True):
	# Update price history for missing data in the past 30 days
	tickerList = []
	db = PTADatabase()
	if db.Open():
		sql = "SELECT Ticker, Exchange FROM rpt_TickerRefreshMonthly"
		if Daily: sql = "SELECT Ticker, Exchange FROM rpt_TickerRefreshDaily"
		df = db.DataFrameFromSQL(sql)
		tickerList = df["Ticker"].tolist()
		db.Close()
		DownloadDaily(tickerList)

def TickerFullRefresh(ticker:str):
	p = PricingData(ticker)
	current_date = GetLatestBDay() 
	recentDate = current_date - pd.offsets.BDay(5)
	if p.ReLoadHistory(verbose=True):
		if p.historyEndDate >= recentDate:
			print(' TickerFullRefresh: Prices updated for ' + ticker)

def TickersFullRefresh():
	print(" TickeFullRefresh refreshing price data from all in rpt_TickerRefreshFullNeeded...")
	tickerCount = 0
	failureCount = 0
	maxStockCount = 150  # 499
	maxFailures = 75
	db = PTADatabase()
	current_date = GetLatestBDay()
	recentDate = current_date - pd.offsets.BDay(5)
	startDate = current_date - pd.DateOffset(years=3)
	if db.Open():
		df = db.DataFrameFromSQL(f"SELECT TOP {maxStockCount} Ticker, Exchange FROM rpt_TickerRefreshFullNeeded")
		for _, row in df.iterrows():
			if failureCount >= maxFailures:
				print(' TickersFullRefresh: Too many errors... exiting.')
				break
			ticker = row.Ticker
			tickerCount += 1
			p = PricingData(ticker)
			if p.LoadHistory(requestedEndDate=current_date):
				if p.historyEndDate >= recentDate:
					print(' TickersFullRefresh: Prices updated for ' + ticker)
					p.LoadTickerFromCSVToSQL()
				else:
					print(' TickersFullRefresh: Failed to download prices for ' + ticker)
					failureCount += 1
			else:
				print(' TickersFullRefresh: Failed to load history or download prices for ' + ticker)
				failureCount += 1
		db.ExecSQL("EXEC sp_UpdateEverything")  # requires NOCOUNT ON
		db.Close()
	print(" TickersFullRefresh: TotalStocks: " + str(tickerCount) + " Failures: " + str(failureCount))
	return tickerCount >= maxStockCount or failureCount >= maxFailures

def Update_TradeModel_Trade_Analysis_Ticker(ticker: str):
	#print("Checking: ", ticker)
	p = PricingData(ticker)
	db = PTADatabase()
	if not p.LoadHistory(): return
	if not db.Open(): return
	p.CalculateStats()
	prices = p.historicalPrices.copy()
	prices = prices.sort_index()
	# --- Precompute rolling stats ---
	prices["Return"] = prices["Close"].pct_change()
	prices["Vol20"] = prices["Return"].rolling(20).std() * (252 ** 0.5)
	prices["Vol60"] = prices["Return"].rolling(60).std() * (252 ** 0.5)
	prices["MA200"] = prices["Close"].rolling(200).mean()
	prices["Distance200DMA"] = prices["Close"] / prices["MA200"] - 1
	entry_dates = db.ScalarListFromSQL(f"SELECT EntryDate FROM TradeModel_Trade_Analysis WHERE Entry_DamageScore IS NULL AND Ticker='{ticker}'")
	for ed in entry_dates:
		ed = ToTimestamp(ed)
		sn = p.GetPriceSnapshot(forDate=ed)
		#assert(sn is not None)
		if sn is None: 	continue
		#assert(ed in prices.index)
		if ed not in prices.index: continue
		row = prices.loc[ed]
		sql = "UPDATE TradeModel_Trade_Analysis SET Entry_PC_1Year=:PC_1Year, Entry_PC_6Month=:PC_6Month, Entry_PC_1Month=:PC_1Month, Entry_PC_1Month3WeekEMA=:PC_1Month3WeekEMA, Entry_20DayVol=:Vol20,Entry_60DayVol=:Vol60, Entry_Distance_200DMA=:Dist200 WHERE Ticker=:Ticker AND EntryDate=:EntryDate "
		params = {
			"PC_1Year": sn.PC_1Year,
			"PC_6Month": sn.PC_6Month,
			"PC_1Month": sn.PC_1Month,
			"PC_1Month3WeekEMA": sn.PC_1Month3WeekEMA,
			"Vol20": float(row["Vol20"]) if not pd.isna(row["Vol20"]) else None,
			"Vol60": float(row["Vol60"]) if not pd.isna(row["Vol60"]) else None,
			"Dist200": float(row["Distance200DMA"]) if not pd.isna(row["Distance200DMA"]) else None,
			"Ticker": ticker,
			"EntryDate": ed
		}
		db.ExecSQL(sql, params)
		sql = "UPDATE TradeModel_Trade_Analysis SET Entry_DamageScore=:DamageScore, Entry_PC_10Day5DayEMA=:PC_10Day5DayEMA, Entry_PC_9Month=:PC_9Month, Entry_JumpRatio=:JumpRatio, Entry_PathologyScore=:PathologyScore, Entry_LossStd_1Year=:LossStd_1Year, Entry_LossSkew_1Year=:LossSkew_1Year,Entry_MaxLoss_1Year=:MaxLoss_1Year,Entry_LogDrawdown=:LogDrawdown WHERE Ticker=:Ticker AND EntryDate=:EntryDate "
		params = {
			"DamageScore": sn.DamageScore,
			"PC_10Day5DayEMA": sn.PC_10Day5DayEMA,
			"PC_9Month": sn.PC_9Month,
			"JumpRatio": sn.JumpRatio,
			"PathologyScore": sn.PathologyScore,
			"LossStd_1Year": sn.LossStd_1Year,
			"LossSkew_1Year": sn.LossSkew_1Year,
			"MaxLoss_1Year": sn.MaxLoss_1Year,
			"LogDrawdown": sn.LogDrawdown,
			"Ticker": ticker,
			"EntryDate": ed
		}
		db.ExecSQL(sql, params)

	exit_dates = db.ScalarListFromSQL(f"SELECT ExitDate FROM TradeModel_Trade_Analysis WHERE Exit_PC_10Day5DayEMA IS NULL AND Ticker='{ticker}'")
	for xd in exit_dates:
		xd = ToTimestamp(xd)
		sn = p.GetPriceSnapshot(forDate=xd)
		if sn is None: continue
		sql = "UPDATE TradeModel_Trade_Analysis SET Exit_PC_10Day5DayEMA=:PC_10Day5DayEMA, Exit_PC_1Month=:PC_1Month, Exit_PC_1Month3WeekEMA=:PC_1Month3WeekEMA WHERE Ticker=:Ticker AND ExitDate=:ExitDate "
		params = {
			"PC_10Day5DayEMA": sn.PC_10Day5DayEMA,
			"PC_1Month": sn.PC_1Month,
			"PC_1Month3WeekEMA": sn.PC_1Month3WeekEMA,
			"Ticker": ticker,
			"ExitDate": xd
		}
		db.ExecSQL(sql, params)
	trades = db.DataFrameFromSQL(f"SELECT EntryDate, ExitDate, EntryPrice FROM TradeModel_Trade_Analysis WHERE Ticker='{ticker}' AND MaxFavorableExcursion IS NULL ")
	for _, trade in trades.iterrows():
		entry_date = ToTimestamp(trade["EntryDate"])
		exit_date =  ToTimestamp(trade["ExitDate"])
		entry_price = trade["EntryPrice"]
		if entry_date not in prices.index or exit_date not in prices.index:
			continue
		trade_slice = prices.loc[entry_date:exit_date].copy()
		if trade_slice.empty:
			continue
		trade_slice["ReturnFromEntry"] = trade_slice["Close"] / entry_price - 1
		# MFE = max favorable move
		mfe = trade_slice["ReturnFromEntry"].max()
		# MAE = max adverse move
		mae = trade_slice["ReturnFromEntry"].min()
		# Max drawdown during trade
		trade_slice["RollingPeak"] = trade_slice["Close"].cummax()
		trade_slice["Drawdown"] = trade_slice["Close"] / trade_slice["RollingPeak"] - 1
		max_dd = trade_slice["Drawdown"].min()
		sql = "UPDATE TradeModel_Trade_Analysis SET MaxFavorableExcursion=:MFE, MaxAdverseExcursion=:MAE, MaxDrawdownDuringTrade=:MaxDD WHERE Ticker=:Ticker AND EntryDate=:EntryDate"
		params = {
			"MFE": float(mfe),
			"MAE": float(mae),
			"MaxDD": float(max_dd),
			"Ticker": ticker,
			"EntryDate": entry_date
		}
		db.ExecSQL(sql, params)
	print(f"Completed Ticker: {ticker}")
	db.Close()

def Update_TradeModel_Trade_Analysis():
	multi_thread = True
	db = PTADatabase()
	if db.Open():
		tickers = db.ScalarListFromSQL("SELECT TOP 1000 Ticker FROM TradeModel_Trade_Analysis WHERE Entry_DamageScore IS NULL Or Exit_PC_10Day5DayEMA is null GROUP BY Ticker ORDER BY COUNT(*) DESC")
		db.Close() # Close main connection so threads can use the pool freely
		if multi_thread: 
			pool = ThreadPool(6) 
			pool.map(Update_TradeModel_Trade_Analysis_Ticker, tickers)       
			pool.close()
			pool.join()
		else:
			for ticker in tickers: 
				Update_TradeModel_Trade_Analysis_Ticker(ticker)

def RefreshPricesWorkingSet():
	#refreshes PricesWorkingSet stocks with stats
	current_date = GetLatestBDay()
	start_date = current_date - timedelta(days=30)
	tickers = TickerLists.GetTickerListSQL(year=current_date.year, month=current_date.month, TopN=CONSTANTS.MARKET_STATE_TOPN_TICKER_COUNT, SP500Only=False, filterByFundamentals=True, annualReturnMin=0) 
	picker = StockPicker(startDate=start_date)
	current_date = picker._endDate
	picker.AlignToList(tickers)
	picker.GeneratePicksForSQL(replaceExisting=True, adaptiveModel=True, verbose=True) #This will regenerate the past 30 days of blended picks and update the PicksBlended table
	picker.GeneratePicksForSQL(replaceExisting=True, adaptiveModel=False, verbose=True) #This will regenerate the past 30 days of blended picks and update the PicksBlended table
	for ticker in FORCE_INCLUDE_TICKERS:
		picker.AddTicker(ticker)
	watchlist = picker.GetHighestPriceMomentum(currentDate=current_date, stocksToReturn=(CONSTANTS.MARKET_STATE_TOPN_TICKER_COUNT + len(FORCE_INCLUDE_TICKERS)) , filterOption=0, returnRawResults=True)
	print(" RefreshPricesWorkingSet results")
	print(watchlist)
	db = PTADatabase()
	if db.Open():
		if 'latestEntry' in watchlist.columns:
			watchlist['latestEntry'] = pd.to_datetime(watchlist['latestEntry'])
		float_cols = watchlist.select_dtypes(include=[np.float64, np.float32]).columns
		watchlist[float_cols] = watchlist[float_cols].round(4)
		watchlist = watchlist.replace({np.nan: None})
		db.ExecSQL("sp_updateeverything")
		if len(watchlist) > 0: 	db.DataFrameToSQL(watchlist, tableName='PricesWorkingSet', indexAsColumn=True, clearExistingData=True)
		db.Close()
		print(" PricesWorkingSet is updated with " + str(len(watchlist)) + ' stocks')
	
if __name__ == '__main__':
	switch = 0
	if len(sys.argv[1:]) > 0: switch = sys.argv[1:][0]
	if switch == '1':
		print('Running option 1: PopulateDatabaseTickers')
		PopulateDatabaseTickers()
	elif switch == '2':
		print('Running option 2: RefreshPricesWorkingSet')
		RefreshPricesWorkingSet() 
	elif switch == '3':
		print('Running option 3: DownloadAllTickerInfo')
		#TickerFullRefresh('AAPL')
		# tickers = ['TEAM','FROG','SNOW']
		# for ticker in tickers: 
			# TickerFullRefresh(ticker)
		DownloadAllTickerInfo()
	elif switch == '4':
		print('Running option 4: Generate_Picks_For_SQL_DateRange')
		CONSTANTS.BLOCK_REFRESHING_FOR_BACKTESTING = True
		Generate_Picks_For_SQL_DateRange(startYear=1980, years=46, TopN=CONSTANTS.MARKET_STATE_TOPN_TICKER_COUNT, replaceExisting=True, adaptiveModel=True, verbose=False)
		#Update_TradeModel_Trade_Analysis()
	else:
		print('Running default option to update daily prices and create working set')
		#DownloadAllTickerInfo()
		#DownloadDaily(TickerLists.Indexes()) #Update indexes
		TickerDataRefresh(Daily=True)	#Update priority
		TickerDataRefresh(Daily=False)	#Fill in monthly data
		RefreshPricesWorkingSet() #Update PricesWorkingSet table
		DownloadAllFinancials()
