import sys, os, datetime, pandas as pd
import _classes.Constants as CONSTANTS
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
		for ticker in tickersUpdated:
			p = PricingData(ticker=ticker, useDatabase=True)
			p.ExportFromSQLToCSV()
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
	pd = PricingData(ticker)
	current_date = GetLatestBDay() 
	recentDate = AddDays(current_date, -7)
	if pd.ReLoadHistory(verbose=True):
		if pd.historyEndDate >= recentDate:
			print(' TickerFullRefresh: Prices updated for ' + ticker)

def TickersFullRefresh():
	print(" TickeFullRefresh ...")
	tickerCount = 0
	failureCount = 0
	maxStockCount = 150  # 499
	maxFailures = 75
	db = PTADatabase()
	current_date = GetLatestBDay()
	startDate = AddDays(current_date, -800)
	recentDate = AddDays(current_date, -7)
	if db.Open():
		db.ExecSQL("EXEC sp_UpdateDailyFromIntraday")  # requires NOCOUNT ON
		df = db.DataFrameFromSQL(f"SELECT TOP {maxStockCount} Ticker, Exchange FROM rpt_TickerRefreshFullNeeded")
		for _, row in df.iterrows():
			if failureCount >= maxFailures:
				print(' TickersFullRefresh: Too many errors... exiting.')
				break
			ticker = row.Ticker
			tickerCount += 1
			pd = PricingData(ticker)
			if pd.LoadHistory(requestedEndDate=current_date):
				if pd.historyEndDate >= recentDate:
					print(' TickersFullRefresh: Prices updated for ' + ticker)
					pd.LoadTickerFromCSVToSQL()
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

def RefreshPricesWorkingSet():
	#refreshes PricesWorkingSet stocks with stats
	current_date = GetLatestBDay()
	start_date = current_date - timedelta(days=30)
	tickers = list(FORCE_INCLUDE_TICKERS)
	picker = StockPicker(startDate=start_date)
	picker.AlignToList(tickers)
	forced_included = picker.GetHighestPriceMomentum(currentDate=current_date, stocksToReturn=30,  minPercentGain=-.1, filterOption=0)
	tickers += TickerLists.GetTickerListSQL(year=current_date.year, month=current_date.month, SP500Only=False, filterByFundamentals=False) #, annualReturnMin=0.05
	picker.AlignToList(tickers)
	additions = picker.GetHighestPriceMomentum(currentDate=current_date, stocksToReturn=300,  minPercentGain=.05, filterOption=0)
	watchlist = pd.concat([forced_included, additions],	axis=0,	sort=True)
	watchlist = watchlist[~watchlist.index.duplicated()]
	watchlist.sort_index(inplace=True)
	adaptive_alloc = picker.GetAdaptiveConvex(currentDate=current_date, modelName='PricesWorkingSet') 
	watchlist = watchlist.join(adaptive_alloc, how="left")
	print(adaptive_alloc)
	if CONSTANTS.CASH_TICKER in adaptive_alloc.index:
		cash_weight = adaptive_alloc.loc[CONSTANTS.CASH_TICKER, "TargetHoldings"]
		cash_row = pd.DataFrame( {col: 0 for col in watchlist.columns}, index=['Cash'] )
		cash_row["Comments"] = ''
		cash_row["TargetHoldings"] = cash_weight
		watchlist = pd.concat([watchlist, cash_row])
	watchlist.index.name = 'Ticker'
	if 'latestEntry' in watchlist.columns:
		watchlist['latestEntry'] = pd.to_datetime(watchlist['latestEntry']).dt.to_pydatetime()
	float_cols = watchlist.select_dtypes(include=[np.float64, np.float32]).columns
	watchlist[float_cols] = watchlist[float_cols].round(4)
	watchlist = watchlist.replace({np.nan: None})
	watchlist['Comments'] = watchlist['Comments'].fillna('').astype(str)
	print(" RefreshPricesWorkingSet results")
	print(watchlist)

	db = PTADatabase()
	if db.Open():
		db.ExecSQL("sp_updateeverything")
		db.DataFrameToSQL(watchlist, tableName='PricesWorkingSet', indexAsColumn=True, clearExistingData=True)
		db.Close()
		print(" PricesWorkingSet is updated with " + str(len(watchlist)) + ' stocks')
	
if __name__ == '__main__':
	switch = 0
	if len(sys.argv[1:]) > 0: switch = sys.argv[1:][0]
	if switch == '1':
		PopulateDatabaseTickers()
	elif switch == '2':
		RefreshPricesWorkingSet() 
	elif switch == '3':
		Generate_PicksBlendedSQL_DateRange(2020, 6, True)
	elif switch == '4':
		DownloadAllTickerInfo()
		#TickerFullRefresh('ORLY')
	else:
		if GetLatestBDay().weekday() < 5: #Skip weekend intraday
			TickerDataRefresh(Daily=True) #Update priority intraday data
			DownloadIntraday(TickerLists.Indexes()) #Update indexes
		DownloadIntraday(FORCE_INCLUDE_TICKERS) 
		TickerDataRefresh(Daily=False)	#Fill in monthly data
		RefreshPricesWorkingSet() #Update PricesWorkingSet table
		#Update_PicksBlendedSQL() #Update PicksBlended table
		