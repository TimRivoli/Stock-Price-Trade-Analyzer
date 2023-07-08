import sys, time, os, shutil
import pandas as pd
from datetime import datetime, timedelta
from _classes.TickerLists import TickerLists
from _classes.Utility import *
from _classes.PriceTradeAnalyzer import *

def GetTickerList(year:int=None, month:int=1, SP500Only:bool=False, filterByFundamtals:bool=False, annualReturnMin:float=0, marketCapMin:int=0, marketCapMax:int=0):
	if year==None: year = datetime.now().year
	result = []
	db = PTADatabase()
	if db.Open():
		cursor = db.GetCursor()
		SQL = 'sp_GetTickerList ' + str(year) + ',' + str(month) + ',' + str(int(SP500Only)) + ',' + str(int(filterByFundamtals)) + ',' + str(annualReturnMin) + ',' + str(marketCapMin) + ',' + str(marketCapMax) 
		print(SQL)
		cursor.execute(SQL)
		for row in cursor.fetchall():
			result.append(row.Ticker)
		cursor.close()
		db.Close()
	return result

def DownloadAllTickerInfo():
	#For all Tickers, updates missing About and financial info
	db = PTADatabase()
	dd = DataDownload()
	if db.Open():
		cursor = db.GetCursor()
		cursor.execute("select Ticker, Exchange from Tickers where Exchange <> 'Delisted' and (CompanyName is null or FinancialsLastUpdated is null)") # 
		for row in cursor.fetchall():
			dd.DownloadTickerGoogleFinance(row.Ticker, row.Exchange)
		cursor.close()
		db.Close()
	
def DownloadIntraday(tickerList:list):
	#Download current day prices will also fill in any missing from past 30 days
	print('Downloading intraday prices for ' + str(len(tickerList)) + ' tickers')
	tickersUpdated = []
	db = PTADatabase()
	dd = DataDownload()
	if db.Open():
		cursor = db.GetCursor()
		for ticker in tickerList:
			Exchange = ""
			pageData = ""
			CompanyName = ""
			cursor.execute("SELECT Exchange, CompanyName FROM Tickers WHERE Ticker=?", ticker)
			for row in cursor.fetchall():
				Exchange = row.Exchange
				if row.CompanyName != None: CompanyName = row.CompanyName
				if dd.DownloadIntradayPriceGoogleFinance(ticker, Exchange): tickersUpdated.append(ticker)
		cursor.execute("EXEC sp_UpdateDailyFromIntraday")  #This only work if NOCOUNT=ON
		cursor.close()
		db.Close()
		for ticker in tickersUpdated:
			p = PricingData(ticker=ticker, useDatabase=True)
			p.ExportFromSQLToCSV()
	print("Requested: " + str(len(tickerList)) + " Completed: " + str(len(tickersUpdated)))
		
def TickerDataRefresh(Daily:bool=True):
	#Update price history for any missing data in the past 30 day range, or if Daily you can update with intraday values for rpt_TickerRefreshIntraday
	tickerList = []
	db = PTADatabase()
	if db.Open():
		cursor = db.GetCursor()
		cursor.execute("EXEC sp_UpdateDailyFromIntraday") #This only work if NOCOUNT=ON
		SQL = "SELECT Ticker, Exchange from rpt_TickerRefreshMonthly"
		if Daily:
			SQL = "SELECT Ticker, Exchange from rpt_TickerRefreshIntraday"
		cursor.execute(SQL)
		for row in cursor.fetchall():
			tickerList.append(row.Ticker)
		cursor.close()
		db.Close()
		DownloadIntraday(tickerList)

def TickersFullRefresh():
	print("TickeFullRefresh ...")
	tickerCount = 0
	failureCount = 0
	maxStockCount = 150 #499
	maxFailures = 75
	db = PTADatabase()
	currentDate = GetTodaysDate()
	startDate = AddDays(currentDate, -800)
	recentDate = AddDays(currentDate, -7)
	if db.Open():
		cursor = db.GetCursor()
		cursor.execute("EXEC sp_UpdateDailyFromIntraday") #This only work if NOCOUNT=ON
		SQL = "select top " + str(maxStockCount) + " Ticker, Exchange from rpt_TickerRefreshFullNeeded"
		cursor.execute(SQL)
		for row in cursor.fetchall():
			ticker = row.Ticker
			if failureCount >= maxFailures: 
				print(' Too many errors... exiting.')
				break
			tickerCount +=1
			pd = PricingData(ticker, useDatabase=False)
			if pd.LoadHistory(requestedEndDate=currentDate):
				if pd.historyEndDate >= recentDate:
					print(' Prices updated for ' + ticker)
					pd.LoadTickerFromCSVToSQL()
				else:
					print(' Failed to download prices for ' + ticker)
					failureCount+=1
			else:
				print(' Failed to load history or download prices for ' + ticker)
				failureCount+=1
		cursor.execute("EXEC sp_UpdateEverything") #This only work if NOCOUNT=ON
		cursor.close()
		db.Close()
	print(" TotalStocks: " + str(tickerCount) + " Failures: " + str(failureCount))
	return tickerCount >= maxStockCount or failureCount >= maxFailures

def RefreshPricesWorkingSet():
	#refreshes PricesWorkingSet stocks with stats
	print("Generating PricesWorkingSet...")
	db = PTADatabase()
	if db.Open():
		cursor = db.GetCursor()
		currentDate = GetTodaysDate()
		picker = StockPicker(startDate='1/1/2020', endDate=currentDate, useDatabase=True)		
		tickers = GetTickerList(year=currentDate.year, month=currentDate.month, SP500Only=False, filterByFundamtals=False, annualReturnMin=.25) 
		print(" Total stocks to consider for price momentum: " + str(len(tickers)))
		picker.AlignToList(tickers)
		result1 = picker.GetHighestPriceMomentum(currentDate=currentDate, longHistoryDays=365, stocksToReturn=150, shortHistoryDays=90, filterOption=0)
		result2 = picker.GetHighestPriceMomentum(currentDate=currentDate, longHistoryDays=365, stocksToReturn=50, shortHistoryDays=90, filterOption=5)
		result1.sort_index(inplace=True)	
		result2.sort_index(inplace=True)	
		print(" Tickers selected for price momentum: " + str(len(result1)) + " from " + str(picker.TickerCount()))	

		tickers = GetTickerList(year=currentDate.year, month=currentDate.month, SP500Only=False, filterByFundamtals=False, annualReturnMin=0, marketCapMin=150000, marketCapMax=0) 
		picker.AlignToList(tickers)
		print(len(tickers), picker.TickerCount())
		result3 = picker.GetHighestPriceMomentum(currentDate=currentDate, longHistoryDays=365, stocksToReturn=51, shortHistoryDays=90, filterOption=0)
		result3.sort_index(inplace=True)	
		print(" Adding top 51 large cap stocks: " + str(len(result3)) + " from " + str(picker.TickerCount()))	
		result = pd.concat([result1, result2, result3], ignore_index=False) 
		result.sort_index(inplace=True)	
		result.drop_duplicates(inplace=True)	
		db.ExecSQL("sp_updateeverything")
		print(result)
		db.DataFrameToSQL(result, tableName='PricesWorkingSet', indexAsColumn=True, clearExistingData=True)
		db.Close()
		print("PricesWorkingSet is updated with " + str(len(result)) + ' stocks')

def _PicksBlended(picker:StockPicker, currentDate:date):
	list1 = picker.GetHighestPriceMomentum(currentDate=currentDate, longHistoryDays=365, shortHistoryDays=90, stocksToReturn=2, filterOption=3)
	list2 = picker.GetHighestPriceMomentum(currentDate=currentDate, longHistoryDays=365, shortHistoryDays=90, stocksToReturn=2, filterOption=3)
	list3 = picker.GetHighestPriceMomentum(currentDate=currentDate, longHistoryDays=365, shortHistoryDays=90, stocksToReturn=2, filterOption=44)
	list4 = picker.GetHighestPriceMomentum(currentDate=currentDate, longHistoryDays=365, shortHistoryDays=90, stocksToReturn=5, filterOption=5)
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
			endDate = today
		else:
			endDate = ToDate('12/31/' + str(startYear))	
		startDate = ToDate('1/1/' + str(startYear-years))
		currentYear = 0
		currentDate = endDate
		cursor = db.GetCursor()
		picker = StockPicker(startDate=startDate, endDate=endDate, useDatabase=True) 
		while currentDate >= startDate:
			if currentYear != currentDate.year:
				currentYear = currentDate.year
				if verbose: print("Getting tickers for year " + str(currentYear))
					
				tickers = GetTickerList(year=currentDate.year, month=currentDate.month, SP500Only=False, filterByFundamtals=False) 
				TotalStocks=len(tickers)
				if verbose: print(" Total stocks: " + str(TotalStocks))
				picker.AlignToList(tickers)			
				TotalValidCandidates = len(picker._tickerList) 
				if verbose: print('Running PicksBlended generation on ' + str(TotalValidCandidates) + ' of ' + str(TotalStocks) + ' stocks from ' + str(startDate) + ' to ' + str(endDate))		
				if TotalValidCandidates==0: assert(False)
			if currentDate.weekday() < 5: #Python Monday=0, skip weekends
				if verbose: print(' Blended 3.3.44.PV Picks - ' + str(currentDate))
				ExistingDataCount = 0
				if not replaceExisting:
					cursor.execute("SELECT count(*) AS ExistingDataCount FROM [PicksBlendedDaily] WHERE [Date]=?", currentDate)
					for row in cursor.fetchall():
						ExistingDataCount = row.ExistingDataCount
					if verbose: print(' ' + str(ExistingDataCount) + ' rows exist')
				if replaceExisting or ExistingDataCount == 0:
					result = _PicksBlended(picker, currentDate)
					if verbose: print(result)
					if len(result) == 0:
						if verbose: print(" No data found.")
					else:
						result['Date'] = currentDate 
						result['TotalStocks'] = TotalStocks
						result['TotalValidCandidates'] = TotalValidCandidates
						print(result)
						db.ExecSQL("DELETE FROM PicksBlendedDaily WHERE Date='" + str(currentDate) + "'")
						db.DataFrameToSQL(result, tableName='PicksBlendedDaily', indexAsColumn=True, clearExistingData=False)
					result=None
			currentDate -= timedelta(days=1) 
	db.ExecSQL("sp_UpdateBlendedPicks")
	cursor.close()
	db.Close()
		
def Generate_PicksBlended(replaceExisting:bool=False):
	#If replaceExisting then it will do the current YTD, else just what is missing
	print('Updating PicksBlended')
	Generate_PicksBlended_DateRange(replaceExisting=replaceExisting)  

#------------------------------------------------------------ Misc Utility ------------------------------------------------------------------
def ExportStarterTickerList():
	#Exports starter ticker list
	db = PTADatabase()
	if db.Open():
		df = db.DataFrameFromSQL("SELECT Ticker, Exchange, SP500Listed FROM tickers WHERE delisted=0 ORDER BY ticker")
		df.to_csv(path_or_buf='database/tickerlist.csv', index=False)
		db.Close()

def ImportStarterTickerList():
	#Run the database generation script PTA_Generate.sql first to create all the proper tables, queries, stored procedures etc
	db = PTADatabase()
	if db.Open():
		df.df = pd.read_csv('database/tickerlist.csv', index_col=0, parse_dates=True, na_values=['nan'])
		df = db.DataFrameToSQL(df, 'Tickers')	#Run the database generation script PTA_Generate.sql first to create all the proper tables, queries, stored procedures etc
		db.Close()

def LoadFromCSVTOSQL(tickerList:list):
	for t in tickerList:	
		p = PricingData(t)
		p.LoadTickerFromCSVToSQL()

def DownloadWithYahooFinance(tickerList:list):
	for t in tickerList:
		p = PricingData(t)
		p.LoadHistory('1/1/1980', GetTodaysDate(), verbose=True)

def DownloadFinancials(tickerList:list):
	dd = DataDownload()
	dd.DownloadFinanceDataYahooFinance(tickerList, 'data/Financials/', True, True, True)
	
def _ImportFinancialDataCSV(ticker: str, fileName: str, tableName: str, col_list: list, optional_columns: list):
	print("Importing " + fileName, ticker)
	with open(fileName, 'r') as file:	#Remove tabs
		x = file.read().replace('\t', '')
	with open(fileName, 'w') as file:
		file.write(x)
	data = pd.read_csv(fileName)
	df = pd.DataFrame(data)	
	if 'ttm' in df.columns: df.drop(columns=['ttm'], inplace=True, axis=1)
	if not 'Date' in df.columns:
		df = df.transpose()
		df.columns = df. iloc[0] #Set column names to first row
		df = df[1:]				 #drop first row
		df.index.names = ['Date']
	else:
		df.set_index('Date', inplace=True)
	df['Ticker'] = ticker
	SuccessFull = True
	for col in optional_columns:
		if not col in df.columns: df[col] = 0
	df = df.replace(',','', regex=True)
	df.sort_values('Date', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last')
	df.fillna(method='ffill', inplace=True)
	try:
		df = df[col_list]
	except Exception as e: 
		SuccessFull = False
		print(e)
	print(df)
	if SuccessFull: 
		db = PTADatabase()
		if db.Open():
			print('Loading to table ' + tableName)
			db.ExecSQL("IF  EXISTS (select * from sys.objects where name='" + tableName  + "') DELETE FROM " + tableName + " WHERE Ticker='" + ticker + "'")
			db.DataFrameToSQL(df, tableName=tableName, indexAsColumn=True, clearExistingData=False)
			db.Close()
	return SuccessFull
		
def ImportFinanceStatementCSVFiles():
	#Loads the quarterly financial statements and balance sheets into quarterly tables, that then needs to be rolled into a monthly cache separately
	directory = "data\Financials"
	errorFolder = directory + "\Errors"
	for fileName in os.listdir(directory):
		if fileName.endswith("_quarterly_financials.csv"): 
			ticker = fileName.split("_")[0]
			col_list = ['Ticker','TotalRevenue','OperatingRevenue','CostOfRevenue','GrossProfit','OperatingIncome','OperatingExpense','InterestExpense','NonInterestExpense','NetOccupancyExpense','ProfessionalExpenseAndContractServicesExpense','GeneralAndAdministrativeExpense','SellingAndMarketingExpense','OtherNonInterestExpense','OtherNonOperatingIncomeExpenses','RentExpenseSupplemental','OtherIncomeExpense','PretaxIncome','TaxProvision','TaxRateForCalcs','NetIncomeCommonStockholders','NetIncome','NetIncomeIncludingNoncontrollingInterests','NetIncomeContinuousOperations','BasicEPS','DilutedEPS','BasicAverageShares','DilutedAverageShares','TotalExpenses','NetIncomeFromContinuingAndDiscontinuedOperation','NormalizedIncome','EBIT']
			optional_columns = ['TotalRevenue','OperatingRevenue','CostOfRevenue','GrossProfit','OperatingIncome','OperatingExpense','InterestExpense','NonInterestExpense','NetOccupancyExpense','ProfessionalExpenseAndContractServicesExpense','GeneralAndAdministrativeExpense','SellingAndMarketingExpense','OtherNonInterestExpense','OtherNonOperatingIncomeExpenses','RentExpenseSupplemental','OtherIncomeExpense','PretaxIncome','TaxProvision','TaxRateForCalcs','NetIncomeCommonStockholders','NetIncome','NetIncomeIncludingNoncontrollingInterests','NetIncomeContinuousOperations','BasicEPS','DilutedEPS','BasicAverageShares','DilutedAverageShares','TotalExpenses','NetIncomeFromContinuingAndDiscontinuedOperation','NormalizedIncome','EBIT']
			doneFolder = directory + "\IncomeStatements"
			if _ImportFinancialDataCSV(ticker, os.path.join(directory,fileName), 'TickerFinancialsQuarterly', col_list, optional_columns):
				shutil.move(os.path.join(directory,fileName),os.path.join(doneFolder,fileName))
			else:
				shutil.move(os.path.join(directory,fileName),os.path.join(errorFolder,fileName))
		elif fileName.endswith("_quarterly_balance-sheet.csv"): 
			ticker = fileName.split("_")[0]
			col_list=['Ticker','TotalAssets','CashCashEquivalentsAndShortTermInvestments','CashAndCashEquivalents','Inventory','NetPPE','GrossPPE','Properties','GoodwillAndOtherIntangibleAssets','Goodwill','OtherIntangibleAssets','LongTermEquityInvestment','NonCurrentNoteReceivables','CurrentLiabilities','CurrentDebt','CommercialPaper','OtherCurrentBorrowings','TotalNonCurrentLiabilitiesNetMinorityInterest','LongTermDebtAndCapitalLeaseObligation','LongTermDebt','LongTermCapitalLeaseObligation','NonCurrentDeferredLiabilities','NonCurrentDeferredTaxesLiabilities','EmployeeBenefits','NonCurrentPensionAndOtherPostretirementBenefitPlans','PreferredSecuritiesOutsideStockEquity','OtherNonCurrentLiabilities','TotalEquityGrossMinorityInterest','StockholdersEquity','CapitalStock','PreferredStock','CommonStock','AdditionalPaidInCapital','RetainedEarnings','TreasuryStock','GainsLossesNotAffectingRetainedEarnings','MinimumPensionLiabilities','TotalCapitalization','CommonStockEquity','CapitalLeaseObligations','NetTangibleAssets','WorkingCapital','InvestedCapital','TangibleBookValue','TotalDebt','NetDebt','ShareIssued','OrdinarySharesNumber','PreferredSharesNumber','TreasurySharesNumber','TotalLiabilitiesNetMinorityInterest','InterestBearingDepositsLiabilities','TradingLiabilities','DerivativeProductLiabilities','OtherLiabilities']
			optional_columns=['TotalAssets','CashCashEquivalentsAndShortTermInvestments','CashAndCashEquivalents','Inventory','NetPPE','GrossPPE','Properties','GoodwillAndOtherIntangibleAssets','Goodwill','OtherIntangibleAssets','LongTermEquityInvestment','NonCurrentNoteReceivables','CurrentLiabilities','CurrentDebt','CommercialPaper','OtherCurrentBorrowings','TotalNonCurrentLiabilitiesNetMinorityInterest','LongTermDebtAndCapitalLeaseObligation','LongTermDebt','LongTermCapitalLeaseObligation','NonCurrentDeferredLiabilities','NonCurrentDeferredTaxesLiabilities','EmployeeBenefits','NonCurrentPensionAndOtherPostretirementBenefitPlans','PreferredSecuritiesOutsideStockEquity','OtherNonCurrentLiabilities','TotalEquityGrossMinorityInterest','StockholdersEquity','CapitalStock','PreferredStock','CommonStock','AdditionalPaidInCapital','RetainedEarnings','TreasuryStock','GainsLossesNotAffectingRetainedEarnings','MinimumPensionLiabilities','TotalCapitalization','CommonStockEquity','CapitalLeaseObligations','NetTangibleAssets','WorkingCapital','InvestedCapital','TangibleBookValue','TotalDebt','NetDebt','ShareIssued','OrdinarySharesNumber','PreferredSharesNumber','TreasurySharesNumber','TotalLiabilitiesNetMinorityInterest','InterestBearingDepositsLiabilities','TradingLiabilities','DerivativeProductLiabilities','OtherLiabilities']
			doneFolder = directory + "\BalanceSheets"
			if _ImportFinancialDataCSV(ticker, os.path.join(directory, fileName), 'TickerBalanceSheetsQuarterly', col_list, optional_columns):
				shutil.move(os.path.join(directory,fileName),os.path.join(doneFolder,fileName))
			else:
				shutil.move(os.path.join(directory,fileName),os.path.join(errorFolder,fileName))

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
		TickerDataRefresh(Daily=True) #Update priority intraday data
	if switch == '2':
		RefreshPricesWorkingSet()
	elif switch == '3':
		Generate_PicksBlended() #Update PicksBlended table
	else:
		if GetTodaysDate().weekday() < 5: #Skip weekend intraday
			TickerDataRefresh(Daily=True) #Update priority intraday data
			DownloadIntraday(TickerLists.Indexes()) #Update indexes
		TickerDataRefresh(Daily=False)	#Fill in monthly data
		RefreshPricesWorkingSet() #Update PricesWorkingSet table
		Generate_PicksBlended() #Update PicksBlended table