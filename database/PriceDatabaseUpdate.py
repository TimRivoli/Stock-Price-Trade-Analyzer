import sys, time, datetime, os, shutil, pandas as pd
import _classes.Constants as CONSTANTS
from multiprocessing.dummy import Pool as ThreadPool
from datetime import datetime, timedelta
from _classes.DataIO import PTADatabase, DataDownload
from _classes.Prices import PricingData
from _classes.Trading import TradingModel, TradeModelParams, ExtensiveTesting, UpdateTradeModelComparisonsFromDailyValue
from _classes.Selection import StockPicker, Generate_PicksBlendedSQL_DateRange
from _classes.TickerLists import TickerLists
from _classes.Utility import *

FORCE_INCLUDE_TICKERS = ['ADMA','AMR','ANF','ASC','AVGO','CEG','CF','CORT','CRK','DVN','HWM','NVDA','PLTR','RCL','ADBE','AAPL','UAL','IAU','GLD']

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
		dd.DownloadTickerGoogleFinance(ticker)
	for ticker in TickerLists.BigList2021():
		TickerFullRefresh(ticker)
		dd.DownloadTickerGoogleFinance(ticker)

def DownloadAllTickerInfo():
	# For all Tickers, updates missing About and financial info
	db = PTADatabase()
	dd = DataDownload()
	if db.Open():
		df = db.DataFrameFromSQL("SELECT Ticker, Exchange FROM Tickers WHERE Exchange <> 'Delisted' AND (CompanyName IS NULL OR FinancialsLastUpdated IS NULL)")
		for _, row in df.iterrows():
			dd.DownloadTickerGoogleFinance(row.Ticker, row.Exchange)
		db.Close()
	
def DownloadIntraday(tickerList: list):
	# Download current day prices; also fills in any missing from past 30 days
	print(' DownloadIntraday: Downloading intraday prices for ' + str(len(tickerList)) + ' tickers')
	tickersUpdated = []
	db = PTADatabase()
	dd = DataDownload()
	if db.Open():
		for ticker in tickerList:
			df = db.DataFrameFromSQL("SELECT Exchange, CompanyName FROM Tickers WHERE Ticker=:Ticker", {"Ticker": ticker})
			if not df.empty:
				Exchange = df.iloc[0].Exchange
				if dd.DownloadIntradayPriceGoogleFinance(ticker, Exchange):
					tickersUpdated.append(ticker)
		db.ExecSQL("EXEC sp_UpdateDailyFromIntraday")  # requires NOCOUNT ON
		db.Close()
	print(" DownloadIntraday: Requested: " + str(len(tickerList)) + " Completed: " + str(len(tickersUpdated)))
		
def TickerDataRefresh(Daily: bool = True):
	# Update price history for missing data in the past 30 days
	tickerList = []
	db = PTADatabase()
	if db.Open():
		db.ExecSQL("EXEC sp_UpdateDailyFromIntraday")  # requires NOCOUNT ON
		sql = "SELECT Ticker, Exchange FROM rpt_TickerRefreshMonthly"
		if Daily:
			sql = "SELECT Ticker, Exchange FROM rpt_TickerRefreshIntraday"
		df = db.DataFrameFromSQL(sql)
		tickerList = df["Ticker"].tolist()
		db.Close()
		DownloadIntraday(tickerList)

def TickerFullRefresh(ticker:str):
	p = PricingData(ticker)
	current_date = GetLatestBDay() 
	recentDate = current_date - pd.offsets.BDay(5)
	if p.ReLoadHistory(verbose=True):
		if p.historyEndDate >= recentDate:
			print(' TickerFullRefresh: Prices updated for ' + ticker)

def TickersFullRefresh():
	print(" TickeFullRefresh ...")
	tickerCount = 0
	failureCount = 0
	maxStockCount = 150  # 499
	maxFailures = 75
	db = PTADatabase()
	current_date = GetLatestBDay()
	recentDate = current_date - pd.offsets.BDay(5)
	startDate = current_date - pd.DateOffset(years=3)
	if db.Open():
		db.ExecSQL("EXEC sp_UpdateDailyFromIntraday")  # requires NOCOUNT ON
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
	tickers = list(FORCE_INCLUDE_TICKERS)
	picker = StockPicker(startDate=start_date)
	picker.AlignToList(tickers)
	forced_included = picker.GetHighestPriceMomentum(currentDate=current_date, stocksToReturn=30, filterOption=0, returnRawResults=True)
	tickers += TickerLists.GetTickerListSQL(year=current_date.year, month=current_date.month, SP500Only=False, filterByFundamentals=False, annualReturnMin=0.1) 
	picker.AlignToList(tickers)
	additions = picker.GetHighestPriceMomentum(currentDate=current_date, stocksToReturn=300, filterOption=0, returnRawResults=True)
	watchlist = pd.concat([forced_included, additions],	axis=0,	sort=True)
	watchlist = watchlist[~watchlist.index.duplicated()]
	watchlist.sort_index(inplace=True)
	watchlist.index.name = 'Ticker'
	print(" RefreshPricesWorkingSet results")
	print(watchlist)
	picker.GeneratePicksBlendedSQL(replaceExisting=True, verbose=True) #This will regenerate the past 30 days of blended picks and update the PicksBlended table

	db = PTADatabase()
	if db.Open():
		if 'latestEntry' in watchlist.columns:
			watchlist['latestEntry'] = pd.to_datetime(watchlist['latestEntry'])
		float_cols = watchlist.select_dtypes(include=[np.float64, np.float32]).columns
		watchlist[float_cols] = watchlist[float_cols].round(4)
		watchlist = watchlist.replace({np.nan: None})
		db.ExecSQL("sp_updateeverything")
		db.DataFrameToSQL(watchlist, tableName='PricesWorkingSet', indexAsColumn=True, clearExistingData=True)
		db.Close()
		print(" PricesWorkingSet is updated with " + str(len(watchlist)) + ' stocks')
	
PopulateDatabaseTickers()
DownloadAllTickerInfo()
RefreshPricesWorkingSet() 
		