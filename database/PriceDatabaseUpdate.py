import sys, time, os, shutil
import pandas as pd
import yfinance as yf
import datetime, time
from datetime import datetime, timedelta
from _classes.TickerLists import TickerLists
from _classes.Utility import *
from _classes.PriceTradeAnalyzer import PTADatabase, StockPicker, DataDownload, PricingData
from multiprocessing import Pool

FORCE_INCLUDE_TICKERS = ['ADMA','AMR','ANF','ASC','AVGO','CEG','CF','CORT','CRK','DVN','FRO','HWM','NVDA','PLTR','RCL','ADBE','SE','AAPL','TPL','UAL','IAU','GLD']

def ExportTickersToCSV():
	db = PTADatabase()
	if db.Open():
		df = db.DataFrameFromSQL("SELECT Ticker, Exchange, SP500Listed FROM Tickers WHERE Delisted=0")
		csvFile = 'database/tickerlist.csv'
		df.to_csv(csvFile, index=False)

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

def ScaleMarketCap(year:int, marketCap:int):
	base_year = 1980
	final_year = 2025   
	reduction_factor = 0.9  # 10-fold reduction 90% decrease from 1980 to 2025
	scale = 1 - ((final_year - year) / (final_year - base_year)) * reduction_factor
	r = marketCap * scale
	print(f" ScaleMarketCap: MarketCap {marketCap} scaled to {r} for year {year}")
	return r
	
def GetTickerList(year:int=None, month:int=1, SP500Only:bool=False, filterByFundamtals:bool=False, annualReturnMin:float=0, marketCapMin:int=0, marketCapMax:int=0):
	if year==None: year = datetime.now().year
	if marketCapMin > 0: marketCapMin = ScaleMarketCap(year, marketCapMin)
	if marketCapMax > 0: marketCapMax = ScaleMarketCap(year, marketCapMax)
	db = PTADatabase()
	if db.Open():
		df = db.DataFrameFromSQL("EXEC sp_GetTickerList :Year, :Month, :SP500Only, :FilterByFundamentals, :AnnualReturnMin, :MarketCapMin, :MarketCapMax", {"Year": year,"Month": month,"SP500Only": int(SP500Only), "FilterByFundamentals": int(filterByFundamtals), "AnnualReturnMin": annualReturnMin, "MarketCapMin": marketCapMin, "MarketCapMax": marketCapMax})
		tickers = df["Ticker"].tolist()
		for ticker in tickers:
			result.append(ticker)
		db.Close()
	return result

def get_current_options(ticker):
	from math import log

	def graded_score_ratio(ratio, neutral=1.0, sensitivity=0.5, max_score=2.0):
		if ratio is None or ratio <= 0:
			return 0.0
		score = log(ratio / neutral)
		score = max(-max_score, min(max_score, score / sensitivity))
		return round(score, 3)

	group_weights = {
		"LT_ITM": 0.6,
		"ST_ITM": 0.6,
		"ST_OTM": 1.0,
		"LT_OTM": 1.0
	}

	metric_weights = {
		"vol_ratio": 0.8,
		"oi_ratio": 0.8,
		"iv_skew": 0.5,
		"vw_price": 1.5
	}

	current_date = GetTodaysDate()
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

def download_current_options(tickers: list):
	all_data = []
	for ticker in tickers:
		data = get_current_options(ticker)
		if data:
			all_data.append(data)
		wait_time = 1
		print(f' download_current_options: Waiting {wait_time} seconds to prevent API overload...')
		time.sleep(wait_time)

	if not all_data:
		print(" download_current_options: No data retrieved.")
		return pd.DataFrame()
	df = pd.DataFrame(all_data)
	return df

def update_current_options(tickers: list):
	table_name = "Options_Sentiment_Daily"
	current_options = download_current_options(tickers)
	db = PTADatabase()
	if db.Open() and len(current_options) > 0:
		current_date = GetTodaysDate()  
		SQL = f"DELETE FROM {table_name} WHERE [Date]='{current_date.strftime('%Y-%m-%d')}'"
		#print(SQL)
		db.ExecSQL(SQL)
		db.DataFrameToSQL(current_options, tableName=table_name, indexAsColumn=False, clearExistingData=False)
		db.Close()
		print(f" update_current_options: Appended data for {len(current_options)} tickers to {table_name}")

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
	current_date = GetTodaysDate()  
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
    current_date = GetTodaysDate()
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
	print(" RefreshPricesWorkingSet: Generating PricesWorkingSet...")
	current_date = GetTodaysDate()
	start_date =  current_date - timedelta(days=30) #New LookBack should make this much not even required
	picker = StockPicker(startDate=start_date, useDatabase=True)		
	tickers = GetTickerList(year=current_date.year, month=current_date.month, SP500Only=False, filterByFundamtals=False, annualReturnMin=.25) 
	print(" RefreshPricesWorkingSet: Total stocks to consider for price momentum: " + str(len(tickers)))
	picker.AlignToList(tickers)
	#start with anyting having 25% annual return
	result1 = picker.GetHighestPriceMomentum(currentDate=current_date, longHistoryDays=365, stocksToReturn=250, shortHistoryDays=90, filterOption=0)
	#Add high point value options
	result2 = picker.GetHighestPriceMomentum(currentDate=current_date, longHistoryDays=365, stocksToReturn=50, shortHistoryDays=90, filterOption=5)
	result1.sort_index(inplace=True)	
	result2.sort_index(inplace=True)	
	print(" RefreshPricesWorkingSet: Tickers selected for price momentum: " + str(len(result1)) + " from " + str(picker.TickerCount()))	

	#Adding top 51 large cap stocks and some hard coded tickers
	tickers = GetTickerList(year=current_date.year, month=current_date.month, SP500Only=False, filterByFundamtals=False, annualReturnMin=0, marketCapMin=150000, marketCapMax=0) 
	for t in FORCE_INCLUDE_TICKERS:
		if not t in tickers: tickers.append(t)
	picker.AlignToList(tickers)
	#print(len(tickers), picker.TickerCount())
	result3 = picker.GetHighestPriceMomentum(currentDate=current_date, longHistoryDays=365, stocksToReturn=100, shortHistoryDays=90, filterOption=0)
	result3.sort_index(inplace=True)	
	print(" RefreshPricesWorkingSet: Adding top 51 large cap stocks and some hard coded tickers: " + str(len(result3)) + " from " + str(picker.TickerCount()))	
	result = pd.concat([result1, result2, result3], ignore_index=False) 
	result.sort_index(inplace=True)	
	result.drop_duplicates(inplace=True)	
	print(" RefreshPricesWorkingSet results")
	print(result)

	db = PTADatabase()
	if db.Open():
		db.ExecSQL("sp_updateeverything")
		db.DataFrameToSQL(result, tableName='PricesWorkingSet', indexAsColumn=True, clearExistingData=True)
		db.Close()
		print(" PricesWorkingSet is updated with " + str(len(result)) + ' stocks')
	#update_current_options(tickers)


def _PicksBlended(picker:StockPicker, current_date:date):
	list1 = picker.GetHighestPriceMomentum(currentDate=current_date, longHistoryDays=365, shortHistoryDays=90, stocksToReturn=2, filterOption=3)
	list2 = picker.GetHighestPriceMomentum(currentDate=current_date, longHistoryDays=365, shortHistoryDays=90, stocksToReturn=2, filterOption=3)
	list3 = picker.GetHighestPriceMomentum(currentDate=current_date, longHistoryDays=365, shortHistoryDays=90, stocksToReturn=2, filterOption=44)
	list4 = picker.GetHighestPriceMomentum(currentDate=current_date, longHistoryDays=365, shortHistoryDays=90, stocksToReturn=5, filterOption=5)
	#list1 = pd.DataFrame(list1.index.tolist()) #Dataframe of just the index
	list1 = list1['Point_Value'].reset_index() #Dataframe with Ticker and Point_Value fields
	list2 = list2['Point_Value'].reset_index() #Dataframe with Ticker and Point_Value fields
	list3 = list3['Point_Value'].reset_index() #Dataframe with Ticker and Point_Value fields
	list4 = list4['Point_Value'].reset_index() #Dataframe with Ticker and Point_Value fields
	result = pd.concat([list1, list2, list3, list4], sort=True)
	if len(result) > 0:
		print(result)
		result = result.groupby('Ticker').agg({'Ticker': 'count', 'Point_Value': 'mean'}) #Count the tickers, average the PV
		result.rename(columns={'Ticker':'TargetHoldings'}, inplace=True)
		print(result)
		result.sort_values('TargetHoldings', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') 
	else:
		result = pd.DataFrame(columns=list(['Ticker','TargetHoldings']))
	return result

def Generate_PicksBlended_DateRange(startYear:int=None, years: int=0, replaceExisting:bool=False, verbose:bool=False):
	db = PTADatabase()
	if db.Open():
		today = GetTodaysDate()
		if startYear== None:
			startYear = today.year
			endDate = (today - pd.offsets.BDay(5)).date() #Don't go crazy with recent data or you will force reloads
		else:
			endDate = ToDate('12/31/' + str(startYear))	
		startDate = ToDate('1/1/' + str(startYear-years))
		current_date = endDate
		picker = StockPicker(startDate=startDate, endDate=endDate, useDatabase=True)
		dates = db.ScalarListFromSQL("SELECT Date FROM rpt_PicksBlendedDaily_MissingDates WHERE year>=:StartYear AND year<=:EndYear ORDER BY Date",	{"StartYear": startDate.year, "EndYear": endDate.year},	column="Date")
		missing_dates = [d.date() if isinstance(d, datetime) else datetime.strptime(d, '%Y-%m-%d %H:%M:%S').date() for d in dates]
		#print(current_date)
		#print(missing_dates)
		#missing_dates = [datetime.strptime(row[0], '%Y-%m-%d').date() for row in results]
		prev_month = -1
		while current_date >= startDate:
			if current_date.weekday() < 5: #Python Monday=0, skip weekends
				ExistingDataCount = 0
				if not replaceExisting:					
					if not current_date in missing_dates: ExistingDataCount = 1
				if replaceExisting or ExistingDataCount == 0:
					if current_date.month != prev_month:
						if verbose: print(" Generate_PicksBlended_DateRange: Getting tickers for year " + str(current_date.year))				
						tickers = GetTickerList(year=current_date.year, month=current_date.month, SP500Only=False, filterByFundamtals=False, marketCapMin=100) 
						TotalStocks=len(tickers)
						if verbose: print(" Generate_PicksBlended_DateRange: Total stocks: " + str(TotalStocks))
						picker.AlignToList(tickers)			
						TotalValidCandidates = len(picker._tickerList) 
						if verbose: print(' Generate_PicksBlended_DateRange: Running PicksBlended generation on ' + str(TotalValidCandidates) + ' of ' + str(TotalStocks) + ' stocks from ' + str(startDate) + ' to ' + str(endDate))		
						if TotalValidCandidates==0: assert(False)
						prev_month = current_date.month
					if verbose: print(' Generate_PicksBlended_DateRange: Blended 3.3.44.PV Picks - ' + str(current_date))
					result = _PicksBlended(picker, current_date)
					if verbose: print(result)
					if len(result) == 0:
						if verbose: print(" Generate_PicksBlended_DateRange: No data found.")
					else:
						result['Date'] = current_date 
						result['TotalStocks'] = TotalStocks
						result['TotalValidCandidates'] = TotalValidCandidates
						print(result)
						db.ExecSQL("DELETE FROM PicksBlendedDaily WHERE Date='" + str(current_date) + "'")
						db.DataFrameToSQL(result, tableName='PicksBlendedDaily', indexAsColumn=True, clearExistingData=False)
					result=None
			current_date -= timedelta(days=1) 
	db.ExecSQL("sp_UpdateBlendedPicks")
	db.Close()
				
def Generate_PicksBlended(replaceExisting:bool=False):
	#If replaceExisting then it will do the current YTD, else just what is missing
	print('Updating PicksBlended')
	Generate_PicksBlended_DateRange(replaceExisting=replaceExisting)

def RefreshPricesWithStats():
	#refreshes PricesWithStats table with needed calculated values, this isn't used
	tickers = GetTickerList(SP500Only=False, filterByFundamtals=False) 
	print("Generating PricesWithStats...")
	print(" TotalStocks: " + str(len(tickers)))
	picker = StockPicker(useDatabase=True)
	picker.AlignToList(tickers)
	picker.SaveStats()
	print("PricesWithStats is updated")

if __name__ == '__main__':
	switch = 0
	if len(sys.argv[1:]) > 0: switch = sys.argv[1:][0]
	if switch == '1':
		PopulateDatabaseTickers()
	if switch == '2':
		RefreshPricesWorkingSet() 
	else:
		if GetTodaysDate().weekday() < 5: #Skip weekend intraday
			TickerDataRefresh(Daily=True) #Update priority intraday data
			DownloadIntraday(TickerLists.Indexes()) #Update indexes
		DownloadIntraday(FORCE_INCLUDE_TICKERS) 
		TickerDataRefresh(Daily=False)	#Fill in monthly data
		RefreshPricesWorkingSet() #Update PricesWorkingSet table
		Generate_PicksBlended() #Update PicksBlended tableupdate_current_options
		