USE [master]
GO
/****** Object:  Database [PTA]    Script Date: 5/29/2023 1:07:35 PM ******/
CREATE DATABASE [PTA]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'PTA', FILENAME = N'D:\Program Files\Microsoft SQL Server\MSSQL15.MSSQLSERVER\MSSQL\DATA\PTA.mdf' , SIZE = 787264KB , MAXSIZE = UNLIMITED, FILEGROWTH = 65536KB )
 LOG ON 
( NAME = N'PTA_log', FILENAME = N'D:\Program Files\Microsoft SQL Server\MSSQL15.MSSQLSERVER\MSSQL\DATA\PTA_log.ldf' , SIZE = 335872KB , MAXSIZE = 2048GB , FILEGROWTH = 65536KB )
 WITH CATALOG_COLLATION = DATABASE_DEFAULT
GO
ALTER DATABASE [PTA] SET COMPATIBILITY_LEVEL = 150
GO
IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [PTA].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO
ALTER DATABASE [PTA] SET ANSI_NULL_DEFAULT OFF 
GO
ALTER DATABASE [PTA] SET ANSI_NULLS OFF 
GO
ALTER DATABASE [PTA] SET ANSI_PADDING OFF 
GO
ALTER DATABASE [PTA] SET ANSI_WARNINGS OFF 
GO
ALTER DATABASE [PTA] SET ARITHABORT OFF 
GO
ALTER DATABASE [PTA] SET AUTO_CLOSE OFF 
GO
ALTER DATABASE [PTA] SET AUTO_SHRINK OFF 
GO
ALTER DATABASE [PTA] SET AUTO_UPDATE_STATISTICS ON 
GO
ALTER DATABASE [PTA] SET CURSOR_CLOSE_ON_COMMIT OFF 
GO
ALTER DATABASE [PTA] SET CURSOR_DEFAULT  GLOBAL 
GO
ALTER DATABASE [PTA] SET CONCAT_NULL_YIELDS_NULL OFF 
GO
ALTER DATABASE [PTA] SET NUMERIC_ROUNDABORT OFF 
GO
ALTER DATABASE [PTA] SET QUOTED_IDENTIFIER OFF 
GO
ALTER DATABASE [PTA] SET RECURSIVE_TRIGGERS OFF 
GO
ALTER DATABASE [PTA] SET  DISABLE_BROKER 
GO
ALTER DATABASE [PTA] SET AUTO_UPDATE_STATISTICS_ASYNC OFF 
GO
ALTER DATABASE [PTA] SET DATE_CORRELATION_OPTIMIZATION OFF 
GO
ALTER DATABASE [PTA] SET TRUSTWORTHY OFF 
GO
ALTER DATABASE [PTA] SET ALLOW_SNAPSHOT_ISOLATION OFF 
GO
ALTER DATABASE [PTA] SET PARAMETERIZATION SIMPLE 
GO
ALTER DATABASE [PTA] SET READ_COMMITTED_SNAPSHOT OFF 
GO
ALTER DATABASE [PTA] SET HONOR_BROKER_PRIORITY OFF 
GO
ALTER DATABASE [PTA] SET RECOVERY SIMPLE 
GO
ALTER DATABASE [PTA] SET  MULTI_USER 
GO
ALTER DATABASE [PTA] SET PAGE_VERIFY CHECKSUM  
GO
ALTER DATABASE [PTA] SET DB_CHAINING OFF 
GO
ALTER DATABASE [PTA] SET FILESTREAM( NON_TRANSACTED_ACCESS = OFF ) 
GO
ALTER DATABASE [PTA] SET TARGET_RECOVERY_TIME = 60 SECONDS 
GO
ALTER DATABASE [PTA] SET DELAYED_DURABILITY = DISABLED 
GO
ALTER DATABASE [PTA] SET QUERY_STORE = OFF
GO
USE [PTA]
GO
/****** Object:  User [PTA]    Script Date: 5/29/2023 1:07:35 PM ******/
CREATE USER [PTA] FOR LOGIN [PTA] WITH DEFAULT_SCHEMA=[dbo]
GO
ALTER ROLE [db_owner] ADD MEMBER [PTA]
GO
/****** Object:  Table [dbo].[HedgeFundHoldings]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[HedgeFundHoldings](
	[HedgeFund] [varchar](50) NOT NULL,
	[Ticker] [varchar](10) NOT NULL,
	[PercentHolding] [float] NOT NULL,
	[TargetPercent] [float] NULL
) ON [PRIMARY]
GO
/****** Object:  View [dbo].[qry_HedgeFundHoldings]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
create view [dbo].[qry_HedgeFundHoldings] AS
select Ticker, Sum(PercentHolding) AS PercentHolding FROM HedgeFundHoldings WHERE PercentHolding is not null and PercentHolding > 0 group by Ticker
GO
/****** Object:  View [dbo].[v_HedgeFundHoldings]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE view [dbo].[v_HedgeFundHoldings] AS
select Ticker, TargetPercent AS HFTarget, RTrim(HedgeFund) + ': ' + ltrim(str(PercentHolding*100)) + '%' AS Holdings from HedgeFundHoldings
GO
/****** Object:  Table [dbo].[Tickers]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[Tickers](
	[CompanyName] [varchar](50) NULL,
	[Ticker] [varchar](10) NOT NULL,
	[StooqTicker] [varchar](10) NULL,
	[Exchange] [varchar](11) NOT NULL,
	[Sector] [varchar](25) NULL,
	[CurrentPrice] [numeric](11, 5) NULL,
	[1MonthReturn] [numeric](5, 3) NULL,
	[1YearReturn] [numeric](5, 3) NULL,
	[EarliestDateDaily] [smalldatetime] NULL,
	[LatestDateDaily] [smalldatetime] NULL,
	[EarliestDateIntraday] [smalldatetime] NULL,
	[LatestDateIntraday] [smalldatetime] NULL,
	[MarketCap] [numeric](11, 3) NULL,
	[NetIncome] [float] NULL,
	[NetProfitMargin] [float] NULL,
	[ReturnOnCapital] [float] NULL,
	[PE_Ratio] [float] NULL,
	[Dividend] [float] NULL,
	[About] [varchar](500) NULL,
	[DaysConstructed] [int] NOT NULL,
	[DaysInPastYear] [int] NOT NULL,
	[DaysTotal] [int] NOT NULL,
	[SP500Listed] [bit] NOT NULL,
	[Delisted] [bit] NOT NULL,
	[CompanySize] [int] NOT NULL,
	[TimeStamp] [smalldatetime] NOT NULL,
	[FinancialsLastUpdated] [smalldatetime] NULL,
	[TradesNetChange] [float] NULL,
	[TradesAvgChange] [float] NULL,
	[TradesMinChange] [float] NULL,
	[TradesMaxChange] [float] NULL,
	[TradesCount] [int] NULL,
	[VolumeAverage] [bigint] NULL
) ON [PRIMARY]
GO
/****** Object:  View [dbo].[rpt_TickerList]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO






CREATE view [dbo].[rpt_TickerList] AS
--Pretty sure this isn't used anymore
--select Ticker, Exchange, CurrentPrice, SP500Listed, CompanySize, MarketCap, NetIncome, NetProfitMargin, ReturnOnCapital, PE_Ratio, [1YearReturn], EarliestDateDaily, LatestDateDaily, EarliestDateIntraday, LatestDateIntraday 
--from Tickers where Exchange not in ('Other','DeListed') and Delisted=0
select * 
FROM 
(select top 120 Ticker, Exchange, CurrentPrice, SP500Listed, [1YearReturn]  from Tickers  where Delisted=0 AND (CurrentPrice > 8 and  [1YearReturn] >  .02) ORDER BY [1YearReturn] DESC) AS x
UNION
select * 
FROM 
(select top 15 Ticker, Exchange, CurrentPrice, SP500Listed, [1YearReturn]  from Tickers WHERE Delisted=0 order by marketcap desc) AS y
UNION
select * 
FROM 
(select Ticker, Exchange, CurrentPrice, SP500Listed, [1YearReturn]  from Tickers WHERE Delisted=0 AND Ticker in ('BAC','JPM','MA','UNH','MRK','OXY','WWE','MCK','HDSN','ENPH','MRO','STLD') ) AS z
--UNION
--select * 
--FROM 
--(select t.Ticker, Exchange, CurrentPrice, SP500Listed, [1YearReturn]  from Tickers t INNER JOIN qry_HedgeFundHoldings h on t.ticker=h.ticker) AS zz WHERE Delisted=0 


GO
/****** Object:  Table [dbo].[TradeModelComparisons]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[TradeModelComparisons](
	[modelName] [varchar](150) NULL,
	[TimeStamp] [smalldatetime] NULL,
	[StartDate] [varchar](12) NULL,
	[Duration] [int] NULL,
	[BuyHoldEndingValue] [numeric](18, 0) NULL,
	[ModelEndingValue] [numeric](18, 0) NULL,
	[BuyHoldGain] [numeric](18, 3) NULL,
	[ModelGain] [numeric](18, 3) NULL,
	[Difference] [numeric](18, 3) NULL,
	[reEvaluationInterval] [int] NULL,
	[shortHistory] [int] NULL,
	[longHistory] [int] NULL,
	[sqlHistory] [int] NULL,
	[stockCount] [int] NULL,
	[stocksBought] [int] NULL,
	[totalTickers] [int] NULL,
	[SP500Only] [bit] NULL,
	[marketCapMin] [int] NULL,
	[marketCapMax] [int] NULL,
	[filterByFundamtals] [bit] NULL,
	[rateLimitTransactions] [bit] NULL,
	[shopBuyPercent] [numeric](12, 3) NULL,
	[shopSellPercent] [numeric](12, 3) NULL,
	[trimProfitsPercent] [numeric](12, 3) NULL,
	[tranchSize] [int] NULL,
	[batchName] [varchar](50) NULL,
	[tickerListName] [varchar](30) NULL
) ON [PRIMARY]
GO
/****** Object:  View [dbo].[rpt_TradeModelGains]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO









CREATE view [dbo].[rpt_TradeModelGains] AS
select ModelName, min(startDate) StartDate, Max(DateAdd(year, Duration, startDate)) EndDate, avg(BuyHoldGain) AvgBuyHoldGain, avg(ModelGain) AvgModelGain, avg([Difference]) [Difference], min(ModelGain) Worst, 
max(ModelGain) Best, Sum(Duration) AS YearCount , [TimeStamp], 
 ReEvaluationInterval AS ReEval, sqlHistory, Duration,  SP500Only, filterByFundamtals, marketCapMin, marketCapMax,  BatchName, TranchSize, stocksBought, stockCount, TotalTickers
from TradeModelComparisons 
GROUP By BatchName, ModelName, [TimeStamp],TranchSize, ReEvaluationInterval, Duration, stocksBought, stockCount, TotalTickers, SP500Only, filterByFundamtals, marketCapMin, marketCapMax, sqlHistory
GO
/****** Object:  Table [dbo].[PicksBlended]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PicksBlended](
	[Ticker] [varchar](10) NULL,
	[TargetHoldings] [numeric](4, 2) NULL,
	[Allocation] [numeric](4, 3) NULL,
	[DateCount] [int] NULL,
	[FirstDate] [smalldatetime] NULL,
	[LastDate] [smalldatetime] NULL
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PricesWorkingSet]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PricesWorkingSet](
	[Ticker] [varchar](10) NULL,
	[hp2Year] [float] NULL,
	[hp1Year] [float] NULL,
	[hp6mo] [float] NULL,
	[hp3mo] [float] NULL,
	[hp2mo] [float] NULL,
	[hp1mo] [float] NULL,
	[currentPrice] [float] NULL,
	[2yearPriceChange] [float] NULL,
	[1yearPriceChange] [float] NULL,
	[6moPriceChange] [float] NULL,
	[3moPriceChange] [float] NULL,
	[2moPriceChange] [float] NULL,
	[1moPriceChange] [float] NULL,
	[dailyGain] [float] NULL,
	[monthlyGain] [float] NULL,
	[monthlyLossStd] [float] NULL,
	[pointValue] [bigint] NULL,
	[Comments] [varchar](255) NULL,
	[latestEntry] [datetime] NULL
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[TickerFinancials]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[TickerFinancials](
	[Ticker] [varchar](10) NOT NULL,
	[Year] [int] NULL,
	[Month] [int] NULL,
	[Revenue] [float] NULL,
	[OperatingExpense] [float] NULL,
	[NetIncome] [float] NULL,
	[NetProfitMargin] [float] NULL,
	[EarningsPerShare] [float] NULL,
	[EBITDA] [float] NULL,
	[EffectiveTaxRate] [float] NULL,
	[CashShortTermInvestments] [float] NULL,
	[TotalAssets] [float] NULL,
	[TotalLiabilities] [float] NULL,
	[TotalEquity] [float] NULL,
	[SharesOutstanding] [float] NULL,
	[MarketCapitalization] [float] NULL,
	[PriceToBook] [float] NULL,
	[PriceToCashFlow] [float] NULL,
	[PriceToSales] [float] NULL,
	[ReturnOnAssetts] [float] NULL,
	[ReturnOnCapital] [float] NULL,
	[CashFromOperations] [float] NULL,
	[CashFromInvesting] [float] NULL,
	[CashFromFinancing] [float] NULL,
	[NetChangeInCash] [float] NULL,
	[FreeCashFlow] [float] NULL,
	[TimeStamp] [smalldatetime] NOT NULL
) ON [PRIMARY]
GO
/****** Object:  View [dbo].[rpt_TickersWorkingSet]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO






CREATE VIEW [dbo].[rpt_TickersWorkingSet]
AS
SELECT t.CompanyName, p.Ticker, t.Sector, t.SP500Listed, hp2Year, hp1Year, hp6mo, hp3mo, hp2mo, hp1mo, 
t.CurrentPrice,
--p.currentPrice,
[2yearPriceChange], [1yearPriceChange], [6moPriceChange], [3moPriceChange], [2moPriceChange], [1moPriceChange], dailyGain, monthlyGain, monthlyLossStd, pointValue, Comments, latestEntry,
bv.Allocation AS TargetHoldings, t.TradesNetChange, t.TradesAvgChange, t.TradesCount, Revenue, t.NetIncome, CompanySize, MarketCap, OperatingExpense, 
t.NetProfitMargin, EarningsPerShare, CashShortTermInvestments, TotalAssets, TotalLiabilities, TotalEquity AS NetWorth, TotalEquity, SharesOutstanding, PriceToBook, ReturnOnAssetts, t.ReturnOnCapital, 
CashFromOperations, CashFromInvesting, CashFromFinancing, NetChangeInCash, FreeCashFlow,
HFTarget, hf.Holdings AS HedgeFundHoldings
FROM PricesWorkingSet p
INNER JOIN Tickers t on p.Ticker=t.Ticker
LEFT JOIN PicksBlended bv ON p.Ticker = bv.Ticker
left join [TickerFinancials] f on p.ticker = f.ticker and f.Year=Year(GetDate())
left join v_HedgeFundHoldings hf on p.ticker=hf.ticker
WHERE p.Ticker not like '.%'
GO
/****** Object:  View [dbo].[rpt_PicksBlended]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE view [dbo].[rpt_PicksBlended] AS
SELECT p.Ticker, TargetHoldings, Allocation, t.CompanyName, t.Sector
FROM PicksBlended p
LEFT JOIN Tickers t on p.Ticker=t.Ticker
GO
/****** Object:  Table [dbo].[PricesDaily]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PricesDaily](
	[Ticker] [varchar](10) NOT NULL,
	[Date] [smalldatetime] NOT NULL,
	[Open] [numeric](11, 5) NULL,
	[High] [numeric](11, 5) NULL,
	[Low] [numeric](11, 5) NULL,
	[Close] [numeric](11, 5) NULL,
	[Volume] [numeric](11, 0) NULL,
	[Constructed] [tinyint] NOT NULL,
	[TimeStamp] [smalldatetime] NOT NULL
) ON [PRIMARY]
GO
/****** Object:  View [dbo].[qry_PriceCurrent]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE View [dbo].[qry_PriceCurrent] AS
select Ticker, AVG((High+Low+[Open]+[Close])/4) AS Price
FROM pricesDaily d
where datediff(d, [Date], getdate()) < 4
GROUP By Ticker
GO
/****** Object:  View [dbo].[qry_Price1Year]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
Create View [dbo].[qry_Price1Year] AS
select Ticker, AVG((High+Low+[Open]+[Close])/4) AS Price
FROM pricesDaily d
where datediff(d, [Date], getdate()) between 360 and 367
GROUP By Ticker
GO
/****** Object:  View [dbo].[qry_Price1Mo]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
Create View [dbo].[qry_Price1Mo] AS
select Ticker, AVG((High+Low+[Open]+[Close])/4) AS Price
FROM pricesDaily d
where datediff(d, [Date], getdate()) between 27 and 33
GROUP By Ticker
GO
/****** Object:  View [dbo].[rpt_TickerRefreshFullNeeded]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO


CREATE View [dbo].[rpt_TickerRefreshFullNeeded] AS
select * FROM 
(select top 150 CompanyName, Ticker, Exchange, StooqTicker, DaysConstructed, DaysInPastYear, DaysTotal, SP500Listed, sector, EarliestDateDaily, LatestDateDaily, EarliestDateIntraday, LatestDateIntraday
from Tickers where (daysinpastyear < 220 or daysinpastyear is null or  datediff(d, latestDateDaily, getDate()) > 30) and Exchange not in ('Other','DeListed','INDEXSP','INDEXDJX','INDEX') 
ORDER BY [timestamp] DESC) AS x
GO
/****** Object:  View [dbo].[rpt_TickerRefreshMonthly]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO




CREATE View [dbo].[rpt_TickerRefreshMonthly] AS
select * FROM 
(select top 450 CompanyName, Ticker, Exchange, StooqTicker, datediff(d, latestDateDaily, getDate()) AS DaysMissing, DaysConstructed, DaysInPastYear, DaysTotal, SP500Listed, sector, EarliestDateDaily, LatestDateDaily, EarliestDateIntraday, LatestDateIntraday
from Tickers 
where 
(datediff(d, latestDateDaily, getDate()) between 15 and 35 and Exchange not in ('Other','DeListed')) or  (Exchange ='' and Delisted=0)
ORDER BY [1YearReturn] DESC) AS x
GO
/****** Object:  View [dbo].[rpt_TickersWorkingSetAnnualized]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO




CREATE VIEW [dbo].[rpt_TickersWorkingSetAnnualized]
AS
SELECT CompanyName, Ticker, Sector, SP500Listed, hp2Year, hp1Year, hp6mo, hp3mo, hp2mo, hp1mo, currentPrice, 
[2yearPriceChange]/2 as [2yearPriceChange], [1yearPriceChange], [6moPriceChange]*2.02776 AS [6moPriceChange], [3moPriceChange] *4.05553 AS [3moPriceChange], [2moPriceChange]*6.0833 AS [2moPriceChange], [1moPriceChange] *12.1666 AS [1moPriceChange], 
dailyGain, monthlyGain, monthlyLossStd, pointValue, Comments, latestEntry,
TargetHoldings, TradesNetChange, TradesAvgChange, TradesCount, Revenue, 
NetIncome, CompanySize, MarketCap, OperatingExpense, NetProfitMargin, EarningsPerShare, CashShortTermInvestments, 
TotalAssets, TotalLiabilities, NetWorth, TotalEquity, SharesOutstanding, PriceToBook, ReturnOnAssetts, ReturnOnCapital, 
CashFromOperations, CashFromInvesting, CashFromFinancing, NetChangeInCash, FreeCashFlow
FROM rpt_TickersWorkingSet
GO
/****** Object:  Table [dbo].[PicksBlendedDaily]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PicksBlendedDaily](
	[Date] [smalldatetime] NOT NULL,
	[Ticker] [varchar](10) NOT NULL,
	[TargetHoldings] [numeric](4, 2) NULL,
	[TotalStocks] [int] NOT NULL,
	[TotalValidCandidates] [int] NOT NULL,
	[TimeStamp] [smalldatetime] NULL
) ON [PRIMARY]
GO
/****** Object:  UserDefinedFunction [dbo].[fn_GetBlendedPicks]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE FUNCTION [dbo].[fn_GetBlendedPicks]
(	
	@RequestedDate datetime, @HistoryDays int
)
RETURNS TABLE 
AS
RETURN 
(
select * from (select top 15 Ticker,  sum(TargetHoldings) AS TargetHoldings, count([date]) AS DateCount, Min([date]) AS FirstDate, max([date]) AS LastDate 
FROM [PicksBlendedDaily] where datediff(d, [date], @RequestedDate) between 0 and @HistoryDays GROUP By Ticker ORDER By sum(TargetHoldings) DESC) AS x
)
GO
/****** Object:  View [dbo].[qry_Ticker_FirstDateInHistory]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
create view [dbo].[qry_Ticker_FirstDateInHistory] AS 
select Ticker, Min(Date) AS FirstDateInHistory FROM PricesDaily group by ticker
GO
/****** Object:  View [dbo].[rpt_TickerMissingDays]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO


CREATE view [dbo].[rpt_TickerMissingDays] AS
select d.Ticker, Year([Date]) AS [Year], count(*) AS DayCount, h.FirstDateInHistory
from PricesDaily d
inner join qry_Ticker_FirstDateInHistory h on d.Ticker=h.Ticker
WHERE Year([Date]) > Year(h.FirstDateInHistory)
GROUP BY d.Ticker, Year([Date]), h.FirstDateInHistory
Having  count(*) < 220 and Year([Date])> 1980 and  Year([Date])< 2022
GO
/****** Object:  Table [dbo].[TickerHistoricalQualityFactors]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[TickerHistoricalQualityFactors](
	[Year] [int] NOT NULL,
	[Ticker] [varchar](15) NOT NULL,
	[SP500Listed] [bit] NULL,
	[InTickers] [bit] NULL,
	[HasPrices] [bit] NOT NULL,
	[PriorYearReturn] [float] NULL,
	[PriorTradesAvgChange] [float] NULL,
	[PriorTradeCount] [int] NULL,
	[LastUpdated] [smalldatetime] NOT NULL
) ON [PRIMARY]
GO
/****** Object:  View [dbo].[rpt_PicksBlendedDailyNonSP500]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
create view [dbo].[rpt_PicksBlendedDailyNonSP500] as
select h.year, d.ticker, count(*) AS Count
from PicksBlendedDaily d
inner join TickerHistoricalQualityFactors  h on d.Ticker=h.ticker and year(d.date)=h.Year
WHERE h.SP500Listed=0
group by h.year, d.ticker
--Order by h.year, d.ticker
GO
/****** Object:  View [dbo].[rpt_PicksBlendedDailyMissingDays]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
create view [dbo].[rpt_PicksBlendedDailyMissingDays] AS
select y as Year, count(*) AS DayCount from 
(select  Year(date) AS y, [date] as d from PicksBlendedDaily   group by Year(date), [date] ) AS x
group by y
having count(*) < 250
GO
/****** Object:  View [dbo].[rpt_PicksBlendedBySector]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE view [dbo].[rpt_PicksBlendedBySector] AS
SELECT t.Sector, Sum(Allocation) AS Allocation
FROM PicksBlended p
LEFT JOIN Tickers t on p.Ticker=t.Ticker
Group by t.Sector
GO
/****** Object:  View [dbo].[rpt_TickerRefreshIntraday]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO


CREATE View [dbo].[rpt_TickerRefreshIntraday] AS
select * 
FROM 
(select top 50 Ticker, Exchange, CurrentPrice, SP500Listed, [1YearReturn], LatestDateDaily, LatestDateIntraday  from Tickers where Exchange not in ('Other','DeListed') and Delisted=0 AND CurrentPrice > 10 and  [1YearReturn] >  .02 ORDER BY [1YearReturn] DESC) AS x
UNION
select * 
FROM 
(select top 15 Ticker, Exchange, CurrentPrice, SP500Listed, [1YearReturn], LatestDateDaily, LatestDateIntraday  from Tickers where Exchange not in ('Other','DeListed') and Delisted=0 AND CurrentPrice > 10 and  [1YearReturn] >  .02 order by marketcap desc) AS y
GO
/****** Object:  View [dbo].[rpt_TickerCandidates]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE view [dbo].[rpt_TickerCandidates] AS
select * from [rpt_TickerRefreshIntraday]
GO
/****** Object:  View [dbo].[rpt_TradeModelGains2010Onward]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE view [dbo].[rpt_TradeModelGains2010Onward] AS
select BatchName, ModelName, avg(BuyHoldGain) AvgBuyHoldGain, avg(ModelGain) AvgModelGain, avg([Difference]) [Difference], min(ModelGain) Worst, max(ModelGain) Best, Sum(Duration) AS YearCount ,
TranchSize, ReEvaluationInterval, Duration, stockCount, TotalTickers, SP500Only, filterByFundamtals
, min(startDate) StartDate, Max(DateAdd(year, Duration, startDate)) EndDate, [TimeStamp]
from TradeModelComparisons 
WHERE cast(StartDate as datetime) > '1/1/2010'
GROUP By BatchName, ModelName, [TimeStamp],TranchSize, ReEvaluationInterval, Duration, stockCount, TotalTickers, SP500Only, filterByFundamtals
GO
/****** Object:  View [dbo].[rpt_SectorPerformanceCurrent]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE View [dbo].[rpt_SectorPerformanceCurrent] AS
select Sector, AVG([1MonthReturn]) MonthReturn , Avg([1YearReturn]) YearReturn, AVG(NetIncome) AS NetIncome, Count(*) AS Tickers 
FROM Tickers 
where currentprice is not null 
Group by Sector --order by sector
GO
/****** Object:  Table [dbo].[TradeModel_Trades]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[TradeModel_Trades](
	[dateBuyOrderPlaced] [varchar](12) NULL,
	[ticker] [varchar](10) NULL,
	[dateBuyOrderFilled] [smalldatetime] NULL,
	[dateSellOrderPlaced] [smalldatetime] NULL,
	[dateSellOrderFilled] [smalldatetime] NULL,
	[units] [int] NULL,
	[buyOrderPrice] [numeric](16, 2) NULL,
	[purchasePrice] [numeric](16, 2) NULL,
	[sellOrderPrice] [numeric](16, 2) NULL,
	[sellPrice] [numeric](16, 2) NULL,
	[NetChange] [numeric](16, 2) NULL,
	[TradeModel] [varchar](150) NULL,
	[TimeStamp] [smalldatetime] NULL,
	[BatchName] [varchar](50) NULL
) ON [PRIMARY]
GO
/****** Object:  View [dbo].[qry_NetTrades]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
create view [dbo].[qry_NetTrades] AS
select Ticker, substring(dateBuyOrderPlaced, 1,4) as Year, NetChange from TradeModel_Trades where NetChange>1000 or  NetChange<-1000 Group By Ticker,  substring(dateBuyOrderPlaced, 1,4) , NetChange
GO
/****** Object:  View [dbo].[rpt_TradeModelNetTradesTicker]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE view [dbo].[rpt_TradeModelNetTradesTicker] AS
select t.Ticker, t.CompanyName, t.Sector, t.SP500Listed, Sum(NetChange) AS NetChange, Avg(NetChange) AvgChange, Min(NetChange) MinChange, Max(NetChange) MaxChange, Count(*) AS TotalTrades, 
Revenue, OperatingExpense, f.NetIncome, f.NetProfitMargin, f.EarningsPerShare,  CashShortTermInvestments, TotalAssets, 
TotalLiabilities, TotalEquity, SharesOutstanding, PriceToBook, ReturnOnAssetts, f.ReturnOnCapital, CashFromOperations, CashFromInvesting, CashFromFinancing, NetChangeInCash, FreeCashFlow
from TradeModel_Trades nt
inner join Tickers t on nt.ticker=t.ticker
LEFT Join TickerFinancials f on nt.Ticker=f.ticker
Group By  t.Ticker, t.CompanyName, t.Sector, t.SP500Listed, Revenue, OperatingExpense, f.NetIncome, f.NetProfitMargin, f.EarningsPerShare, CashShortTermInvestments, TotalAssets, TotalLiabilities, TotalEquity, SharesOutstanding, PriceToBook, ReturnOnAssetts, f.ReturnOnCapital, CashFromOperations, CashFromInvesting, CashFromFinancing, NetChangeInCash, FreeCashFlow
GO
/****** Object:  View [dbo].[rpt_TradeModelNetTrades]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO


CREATE View [dbo].[rpt_TradeModelNetTrades] AS
select t.CompanyName, t.Sector, t.SP500Listed, nt.NetChange, AvgChange,MinChange, MaxChange,TotalTrades,   f.*
from rpt_TradeModelNetTradesTicker nt
inner join Tickers t on nt.ticker=t.ticker
LEFT Join TickerFinancials f on nt.Ticker=f.ticker
GO
/****** Object:  View [dbo].[rpt_TradeModelNetTradesBySP500]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE View [dbo].[rpt_TradeModelNetTradesBySP500] AS
select SP500Listed, Sum(NetChange)NetChange, Avg(AvgChange)AvgChange, Min(MinChange)MinChange, Max(MaxChange)MaxChange, Sum(TotalTrades)TotalTrades
from rpt_TradeModelNetTradesTicker nt
GROUP BY SP500Listed
GO
/****** Object:  View [dbo].[qry_PriceHistoryYearChunks]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
create view [dbo].[qry_PriceHistoryYearChunks] AS
select yearChunk, Ticker, sp500Listed, Year(EarliestDateDaily) FirstYear, Year(LatestDateDaily) LastYear
from tickers 
INNER JOIN (SELECT yearChunk FROM (VALUES (1980),(1985),(1990),(1995),(2000),(2005),(2010),(2015),(2020)) v(yearChunk) ) as x ON x.yearChunk> Year(EarliestDateDaily) and  x.yearChunk < Year(LatestDateDaily)
where EarliestDateDaily is not null and ticker not in ('.DJI','.IXIC','.INX') 
GO
/****** Object:  View [dbo].[rpt_TickerListWithFinancials]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE view [dbo].[rpt_TickerListWithFinancials] AS
select t.*
FROM Tickers t
--Inner join [rpt_TickerList] l on t.ticker=l.ticker
where delisted=0
GO
/****** Object:  View [dbo].[rpt_PriceHistoryYearChunkCounts]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE view [dbo].[rpt_PriceHistoryYearChunkCounts] AS
select Year, sp500Listed, Count(*) AS TickerCount
from TickerHistoricalQualityFactors 
where HasPrices=1
GROUP BY Year, sp500Listed
GO
/****** Object:  View [dbo].[qry_PossibleSplits]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
create view [dbo].[qry_PossibleSplits] AS
select Ticker, Max([Open])/Min([Open]) -1 AS ChangeMaxMin, avg([Open])/Min([Open]) -1 AS ChangeAvgMin
FROM PricesDaily
WHERE DateDiff(d, date, getdate()) < 100 and [Open] > 250
group by ticker
having  avg([Open])/Min([Open]) -1 > .3 
GO
/****** Object:  Table [dbo].[PricesIntraday]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PricesIntraday](
	[Ticker] [varchar](10) NOT NULL,
	[Year] [int] NOT NULL,
	[Month] [int] NOT NULL,
	[Day] [int] NOT NULL,
	[Hour] [int] NOT NULL,
	[Minute] [int] NOT NULL,
	[Date] [date] NULL,
	[DateTime] [smalldatetime] NULL,
	[Price] [numeric](11, 5) NOT NULL,
	[Volume] [numeric](11, 0) NULL,
	[TimeStamp] [smalldatetime] NOT NULL
) ON [PRIMARY]
GO
/****** Object:  View [dbo].[qry_Intraday_HighLow]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE View [dbo].[qry_Intraday_HighLow] AS
select Ticker, [Date], Min(Price) AS [Low], Max(Price) AS [High], Avg(price) AS Average, Sum(Volume) AS Volume FROM [PricesIntraday] GROUP By Ticker, [Date]
GO
/****** Object:  View [dbo].[qry_Intraday_Open]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE View [dbo].[qry_Intraday_Open] AS
select Ticker, [Date], Min(Price) AS [Open] FROM [PricesIntraday] where [Hour]=9 and [Minute]=30 AND Date is not null GROUP BY [Ticker],[Date]
GO
/****** Object:  View [dbo].[qry_Intraday_Close]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE View [dbo].[qry_Intraday_Close] AS
select Ticker, [Date], Max(Price) AS [Close] FROM [PricesIntraday] where ([Hour]=20 or ([Hour]=19 and [Minute]>50)) AND Date is not null GROUP BY [Ticker],[Date]
GO
/****** Object:  View [dbo].[qry_Intraday_OHLC]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE View [dbo].[qry_Intraday_OHLC] AS
select hl.Ticker,hl.[Date], IsNull(o.[Open], hl.Average) AS [Open], hl.[High], hl.[Low], IsNull(c.[Close], hl.Average) as [Close], hl.Volume
FROM qry_Intraday_HighLow hl
LEFT JOIN qry_Intraday_Open o on o.Ticker=hl.Ticker and o.Date=hl.Date
LEFT JOIN qry_Intraday_Close c on c.Ticker=hl.Ticker and c.Date=hl.Date
GO
/****** Object:  Table [dbo].[TickerFinancialsHistorical]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[TickerFinancialsHistorical](
	[Ticker] [varchar](10) NOT NULL,
	[Year] [int] NOT NULL,
	[Month] [int] NOT NULL,
	[Price] [float] NULL,
	[PriceMPC] [float] NULL,
	[PriceYPC] [float] NULL,
	[Revenue] [float] NULL,
	[RevenueMPC] [float] NULL,
	[RevenueYPC] [float] NULL,
	[OperatingExpense] [float] NULL,
	[NetIncome] [float] NULL,
	[NetIncomeMPC] [float] NULL,
	[NetIncomeYPC] [float] NULL,
	[NetProfitMargin] [float] NULL,
	[EarningsPerShare] [float] NULL,
	[EBITDA] [float] NULL,
	[EffectiveTaxRate] [float] NULL,
	[CashShortTermInvestments] [float] NULL,
	[TotalAssets] [float] NULL,
	[TotalLiabilities] [float] NULL,
	[TotalLiabilitiesMPC] [float] NULL,
	[TotalLiabilitiesYPC] [float] NULL,
	[TotalEquity] [float] NULL,
	[SharesOutstanding] [float] NULL,
	[MarketCapitalization] [float] NULL,
	[PriceToBook] [float] NULL,
	[PriceToCashFlow] [float] NULL,
	[PriceToSales] [float] NULL,
	[ReturnOnAssetts] [float] NULL,
	[ReturnOnCapital] [float] NULL,
	[CashFromOperations] [float] NULL,
	[CashFromInvesting] [float] NULL,
	[CashFromFinancing] [float] NULL,
	[NetChangeInCash] [float] NULL,
	[FreeCashFlow] [float] NULL,
	[SP500Listed] [bit] NOT NULL,
	[TimeStamp] [smalldatetime] NOT NULL
) ON [PRIMARY]
GO
/****** Object:  View [dbo].[rpt_TickerWithHistoricalFinancials]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO


CREATE view [dbo].[rpt_TickerWithHistoricalFinancials] AS
select t.CompanyName, t.Sector, t.Exchange, t.About, h.*
FROM Tickers t
Inner join TickerFinancialsHistorical h on t.ticker=h.ticker
where h.Month=1 and t.Delisted=0
GO
/****** Object:  View [dbo].[qry_TickerAnnualChange]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE view [dbo].[qry_TickerAnnualChange] AS
select s.Year, s.Ticker, s.StartPrice, e.EndPrice, (e.EndPrice/s.StartPrice)-1 AS AnnualChange
FROM (select ticker, Year([date]) AS Year, Avg([Open]) AS StartPrice from PricesDaily where Month([Date])=1 and Day([Date]) <=4 group by ticker, Year([date])) AS s
INNER JOIN (select ticker, Year([date]) AS Year, Avg([Open]) AS EndPrice from PricesDaily where Month([Date])=12 and Day([Date]) >=26 group by ticker, Year([date])) as e
ON s.ticker=e.Ticker and s.Year=e.Year
GO
/****** Object:  View [dbo].[qry_TickerHasYearData]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
Create view [dbo].[qry_TickerHasYearData] AS
select Ticker, Year([Date]) AS Year
FROM PricesDaily
Group By Ticker, Year([Date])
GO
/****** Object:  View [dbo].[rpt_TickerCurrentFundamentals]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO




CREATE VIEW [dbo].[rpt_TickerCurrentFundamentals]
AS
SELECT t.Ticker, sp500Listed AS SP500, CompanySize, t.MarketCap, CurrentPrice, bv.TargetHoldings, Revenue, t.NetIncome,  EarningsPerShare, TotalAssets- TotalLiabilities AS NetWorth,
t.TradesNetChange, t.TradesAvgChange, t.TradesCount, VolumeAverage,
t.NetProfitMargin, OperatingExpense, CashShortTermInvestments, TotalAssets, TotalLiabilities, TotalEquity, 
SharesOutstanding,  PriceToBook, ReturnOnAssetts, t.ReturnOnCapital, CashFromOperations, CashFromInvesting, CashFromFinancing, 
NetChangeInCash, FreeCashFlow
FROM Tickers t 
LEFT JOIN PicksBlended bv ON t.Ticker = bv.Ticker
left join [TickerFinancials] f on t.ticker = f.ticker and f.Year=2022
WHERE t.Ticker not like '.%'
GO
/****** Object:  Table [dbo].[TickerBalanceSheetsQuarterly]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[TickerBalanceSheetsQuarterly](
	[Date] [smalldatetime] NULL,
	[Ticker] [nchar](10) NULL,
	[TotalAssets] [float] NULL,
	[CurrentAssets] [float] NULL,
	[CashCashEquivalentsAndShortTermInvestments] [float] NULL,
	[CashAndCashEquivalents] [float] NULL,
	[CashFinancial] [float] NULL,
	[CashEquivalents] [float] NULL,
	[Inventory] [float] NULL,
	[TotalNonCurrentAssets] [float] NULL,
	[NetPPE] [float] NULL,
	[GrossPPE] [float] NULL,
	[Properties] [float] NULL,
	[ConstructionInProgress] [float] NULL,
	[AccumulatedDepreciation] [float] NULL,
	[GoodwillAndOtherIntangibleAssets] [float] NULL,
	[Goodwill] [float] NULL,
	[OtherIntangibleAssets] [float] NULL,
	[InvestmentsAndAdvances] [float] NULL,
	[LongTermEquityInvestment] [float] NULL,
	[InvestmentsinSubsidiariesatCost] [float] NULL,
	[InvestmentsinAssociatesatCost] [float] NULL,
	[InvestmentsinJointVenturesatCost] [float] NULL,
	[NonCurrentNoteReceivables] [float] NULL,
	[DefinedPensionBenefit] [float] NULL,
	[CurrentLiabilities] [float] NULL,
	[CurrentDebt] [float] NULL,
	[CurrentNotesPayable] [float] NULL,
	[CommercialPaper] [float] NULL,
	[LineOfCredit] [float] NULL,
	[OtherCurrentBorrowings] [float] NULL,
	[CurrentCapitalLeaseObligation] [float] NULL,
	[CurrentDeferredLiabilities] [float] NULL,
	[CurrentDeferredRevenue] [float] NULL,
	[OtherCurrentLiabilities] [float] NULL,
	[TotalNonCurrentLiabilitiesNetMinorityInterest] [float] NULL,
	[LongTermDebtAndCapitalLeaseObligation] [float] NULL,
	[LongTermDebt] [float] NULL,
	[LongTermCapitalLeaseObligation] [float] NULL,
	[NonCurrentDeferredLiabilities] [float] NULL,
	[NonCurrentDeferredTaxesLiabilities] [float] NULL,
	[EmployeeBenefits] [float] NULL,
	[NonCurrentPensionAndOtherPostretirementBenefitPlans] [float] NULL,
	[PreferredSecuritiesOutsideStockEquity] [float] NULL,
	[OtherNonCurrentLiabilities] [float] NULL,
	[TotalEquityGrossMinorityInterest] [float] NULL,
	[StockholdersEquity] [float] NULL,
	[CapitalStock] [float] NULL,
	[PreferredStock] [float] NULL,
	[CommonStock] [float] NULL,
	[AdditionalPaidInCapital] [float] NULL,
	[RetainedEarnings] [float] NULL,
	[TreasuryStock] [float] NULL,
	[GainsLossesNotAffectingRetainedEarnings] [float] NULL,
	[UnrealizedGainLoss] [float] NULL,
	[MinimumPensionLiabilities] [float] NULL,
	[TotalCapitalization] [float] NULL,
	[CommonStockEquity] [float] NULL,
	[CapitalLeaseObligations] [float] NULL,
	[NetTangibleAssets] [float] NULL,
	[WorkingCapital] [float] NULL,
	[InvestedCapital] [float] NULL,
	[TangibleBookValue] [float] NULL,
	[TotalDebt] [float] NULL,
	[NetDebt] [float] NULL,
	[ShareIssued] [float] NULL,
	[OrdinarySharesNumber] [float] NULL,
	[PreferredSharesNumber] [float] NULL,
	[TreasurySharesNumber] [float] NULL,
	[TotalLiabilitiesNetMinorityInterest] [float] NULL,
	[InterestBearingDepositsLiabilities] [float] NULL,
	[TradingLiabilities] [float] NULL,
	[DerivativeProductLiabilities] [float] NULL,
	[OtherLiabilities] [float] NULL
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[TickerFinancialsQuarterly]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[TickerFinancialsQuarterly](
	[Date] [smalldatetime] NULL,
	[Ticker] [varchar](10) NULL,
	[TotalRevenue] [float] NULL,
	[OperatingRevenue] [float] NULL,
	[CostOfRevenue] [float] NULL,
	[GrossProfit] [float] NULL,
	[OperatingIncome] [float] NULL,
	[OperatingExpense] [float] NULL,
	[InterestExpense] [float] NULL,
	[NonInterestExpense] [float] NULL,
	[NetOccupancyExpense] [float] NULL,
	[ProfessionalExpenseAndContractServicesExpense] [float] NULL,
	[GeneralAndAdministrativeExpense] [float] NULL,
	[SellingAndMarketingExpense] [float] NULL,
	[OtherNonInterestExpense] [float] NULL,
	[OtherNonOperatingIncomeExpenses] [float] NULL,
	[RentExpenseSupplemental] [float] NULL,
	[OtherIncomeExpense] [float] NULL,
	[PretaxIncome] [float] NULL,
	[TaxProvision] [float] NULL,
	[TaxRateForCalcs] [float] NULL,
	[NetIncomeCommonStockholders] [float] NULL,
	[NetIncome] [float] NULL,
	[NetIncomeIncludingNoncontrollingInterests] [float] NULL,
	[NetIncomeContinuousOperations] [float] NULL,
	[BasicEPS] [float] NULL,
	[DilutedEPS] [float] NULL,
	[BasicAverageShares] [float] NULL,
	[DilutedAverageShares] [float] NULL,
	[TotalExpenses] [float] NULL,
	[NetIncomeFromContinuingAndDiscontinuedOperation] [float] NULL,
	[NormalizedIncome] [float] NULL,
	[EBIT] [float] NULL,
	[Price] [float] NULL,
	[MarketCap] [float] NULL,
	[CAPEI] [float] NULL,
	[PriceCashFlow] [float] NULL,
	[PriceToBook] [float] NULL,
	[PriceToSales] [float] NULL,
	[PE_op_basic] [float] NULL
) ON [PRIMARY]
GO
/****** Object:  View [dbo].[rpt_TickersWithFullPerformanceData]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE view [dbo].[rpt_TickersWithFullPerformanceData] AS
select q.[Date], q.Ticker, q.Price, pd.[Open] AS SP500Price, 
TotalRevenue, NetIncome, BasicAverageShares, BasicEPS, MarketCap, CAPEI, PriceCashFlow, PriceToBook, 
TotalAssets, TotalEquityGrossMinorityInterest, StockHoldersEquity, TotalCapitalization, CommonStockEquity, NetTangibleAssets, InvestedCapital, TangibleBookvalue, TotalDebt, ShareIssued
FROM TickerFinancialsQuarterly q
inner join TickerBalanceSheetsQuarterly bs on q.Ticker=bs.Ticker and q.Date=bs.Date
inner join PricesDaily pd on pd.Date=q.Date and pd.Ticker='.INX'
WHERE q.Price is not null
GO
/****** Object:  View [dbo].[rpt_TickerDuplicateDailyValues]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
create view [dbo].[rpt_TickerDuplicateDailyValues] as
select Ticker, Year([Date]) AS Year, count(*) AS DayCount, Min(Timestamp) FirstStamp, max(TimeStamp) SecondStamp
FROM PricesDaily WHERE ticker <> '.IXIC' group by Ticker, Year([Date]) having count(*) > 300
GO
/****** Object:  View [dbo].[rpt_TradeModelComparisons]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE view [dbo].[rpt_TradeModelComparisons] as
select modelName, StartDate, Duration, BuyHoldEndingValue, ModelEndingValue, BuyHoldGain, ModelGain, Difference, 
reEvaluationInterval, shortHistory, longHistory, TimeStamp,
SP500Only, marketCapMin, marketCapMax, filterByFundamtals, 
rateLimitTransactions, shopBuyPercent, shopSellPercent, trimProfitsPercent, tranchSize, 
sqlHistory, stockCount, stocksBought, totalTickers, 
batchName, tickerListName
 from TradeModelComparisons 
GO
/****** Object:  View [dbo].[rpt_TickerFinancialsQuarterly]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE View [dbo].[rpt_TickerFinancialsQuarterly] AS 
SELECT fq.[Ticker], Year(fq.[Date]) AS Year, Month(fq.[Date]) AS Month, [TotalRevenue]/1000000 AS Revenue,
	  [GrossProfit]/1000000 AS GrossProfit,
	  [OperatingExpense]/1000000 AS OperatingExpense,
	  [OperatingIncome]/1000000 AS [OperatingIncome],
      [NetIncome]/1000000 AS [NetIncome] , --[NormalizedIncome],
	  case when [GrossProfit] =0 then 0 else [NetIncome]/[GrossProfit] * 100 end AS NetProfitMargin, 
	  [BasicEPS] AS EarningsPerShare,      --,[DilutedEPS]     
      [TotalExpenses]/1000000 AS [TotalExpenses],
      [EBIT]/1000000 AS EBITDA,  --Not exact
	  TaxRateForCalcs * 100 as EffectiveTaxRate,	  
	  CashCashEquivalentsAndShortTermInvestments/1000000 AS CashShortTermInvestments,
	  TotalAssets/1000000 AS TotalAssets,
	  TotalLiabilitiesNetMinorityInterest /1000000 AS TotalLiabilities,
	  StockholdersEquity /1000000 AS TotalEquity,
	  [BasicAverageShares]/1000000 AS SharesOutstanding, 	  --,[DilutedAverageShares]
	  PriceToBook,
	  case when TotalAssets=0 then 0 else NetIncome /TotalAssets *100 end AS ReturnOnAssetts,
	  case when (TotalLiabilitiesNetMinorityInterest + StockholdersEquity)=0 then 0 else [EBIT] /(TotalLiabilitiesNetMinorityInterest + StockholdersEquity) *100 end AS ReturnOnCapital
	  --0 AS CashFromOperations,	  0 AS CashFromInvesting,	  0 AS CashFromFinancing,	  0 AS NetChangeInCash, 0 AS FreeCashFlow
  FROM [TickerFinancialsQuarterly] fq  INNER JOIN TickerBalanceSheetsQuarterly bs on fq.Ticker=bs.Ticker and fq.date=bs.date
  

  
  
  


GO
/****** Object:  View [dbo].[qry_PriceCurrentBroad]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE View [dbo].[qry_PriceCurrentBroad] AS
select Ticker, AVG((High+Low+[Open]+[Close])/4) AS Price
FROM pricesDaily d
where datediff(d, [Date], getdate()) < 15
GROUP By Ticker
GO
/****** Object:  Table [dbo].[NNTrainingResults]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[NNTrainingResults](
	[ModelName] [varchar](max) NULL,
	[train_start_accuracy] [float] NULL,
	[train_end_accuracy] [float] NULL,
	[train_start_accuracy2] [bigint] NULL,
	[train_end_accuracy2] [bigint] NULL,
	[train_start_loss] [float] NULL,
	[train_end_loss] [float] NULL,
	[num_features] [int] NULL,
	[num_classes] [int] NULL,
	[num_layers] [int] NULL,
	[prediction_target_days] [int] NULL,
	[time_steps] [int] NULL,
	[num_source_days] [int] NULL,
	[batch_size] [int] NULL,
	[epochs] [int] NULL,
	[dropout_rate] [float] NULL,
	[batch_normalization] [bit] NULL,
	[source_field_list] [varchar](max) NULL,
	[TimeStamp] [smalldatetime] NULL
) ON [PRIMARY] TEXTIMAGE_ON [PRIMARY]
GO
/****** Object:  Table [dbo].[PricesDailyWithStats]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[PricesDailyWithStats](
	[Date] [smalldatetime] NULL,
	[Open] [float] NULL,
	[High] [float] NULL,
	[Low] [float] NULL,
	[Close] [float] NULL,
	[Volume] [float] NULL,
	[Average] [float] NULL,
	[2DayAv] [float] NULL,
	[5DayAv] [float] NULL,
	[shortEMA] [float] NULL,
	[shortEMASlope] [float] NULL,
	[longEMA] [float] NULL,
	[longEMASlope] [float] NULL,
	[45dEMA] [float] NULL,
	[45dEMASlope] [float] NULL,
	[1DayDeviation] [float] NULL,
	[5DavDeviation] [float] NULL,
	[15DavDeviation] [float] NULL,
	[1DayApc] [float] NULL,
	[3DayApc] [float] NULL,
	[dailyGain] [float] NULL,
	[monthlyGain] [float] NULL,
	[monthlyLosses] [float] NULL,
	[monthlyLossStd] [float] NULL,
	[1DayMomentum] [float] NULL,
	[3DayMomentum] [float] NULL,
	[5DayMomentum] [float] NULL,
	[10DayMomentum] [float] NULL,
	[channelHigh] [float] NULL,
	[channelLow] [float] NULL,
	[1DayPC] [float] NULL,
	[20DayPC] [float] NULL,
	[62DayPC] [float] NULL,
	[125DayPC] [float] NULL,
	[250DayPC] [float] NULL,
	[Ticker] [varchar](10) NULL
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[SP500Constituents2022]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[SP500Constituents2022](
	[Ticker] [nvarchar](50) NOT NULL,
	[CompanyName] [nvarchar](50) NOT NULL,
	[Sector] [nvarchar](50) NOT NULL
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[SP500ConstituentsDaily]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[SP500ConstituentsDaily](
	[Date] [smalldatetime] NOT NULL,
	[Ticker] [varchar](15) NOT NULL
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[SP500ConstituentsYearly]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[SP500ConstituentsYearly](
	[Year] [int] NOT NULL,
	[Ticker] [varchar](15) NOT NULL
) ON [PRIMARY]
GO
/****** Object:  Table [dbo].[TradeModel_DailyValue]    Script Date: 5/29/2023 1:07:35 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[TradeModel_DailyValue](
	[TradeModel] [varchar](150) NULL,
	[Date] [datetime] NULL,
	[CashValue] [bigint] NULL,
	[AssetValue] [bigint] NULL,
	[TotalValue] [bigint] NULL,
	[Stock00] [varchar](12) NULL,
	[Stock01] [varchar](12) NULL,
	[Stock02] [varchar](12) NULL,
	[Stock03] [varchar](12) NULL,
	[Stock04] [varchar](12) NULL,
	[Stock05] [varchar](12) NULL,
	[Stock06] [varchar](12) NULL,
	[Stock07] [varchar](12) NULL,
	[Stock08] [varchar](12) NULL,
	[Stock09] [varchar](12) NULL,
	[Stock10] [varchar](12) NULL,
	[TimeStamp] [smalldatetime] NULL,
	[BatchName] [varchar](50) NULL
) ON [PRIMARY]
GO
ALTER TABLE [dbo].[NNTrainingResults] ADD  CONSTRAINT [DF_NNTrainingResults_TimeStamp_1]  DEFAULT (getdate()) FOR [TimeStamp]
GO
ALTER TABLE [dbo].[PicksBlendedDaily] ADD  CONSTRAINT [DF_PicksBlendedDaily_TimeStamp]  DEFAULT (getdate()) FOR [TimeStamp]
GO
ALTER TABLE [dbo].[PricesDaily] ADD  CONSTRAINT [DF_PricesDaily_Volume]  DEFAULT ((0)) FOR [Volume]
GO
ALTER TABLE [dbo].[PricesDaily] ADD  CONSTRAINT [DF_PricesDaily_Constructed]  DEFAULT ((0)) FOR [Constructed]
GO
ALTER TABLE [dbo].[PricesDaily] ADD  CONSTRAINT [DF_PricesDaily_TimeStamp]  DEFAULT (getdate()) FOR [TimeStamp]
GO
ALTER TABLE [dbo].[PricesIntraday] ADD  CONSTRAINT [DF_PricesIntraday_Hour]  DEFAULT ((0)) FOR [Hour]
GO
ALTER TABLE [dbo].[PricesIntraday] ADD  CONSTRAINT [DF_PricesIntraday_Minute]  DEFAULT ((0)) FOR [Minute]
GO
ALTER TABLE [dbo].[PricesIntraday] ADD  CONSTRAINT [DF_PricesIntraday_TimeStamp]  DEFAULT (getdate()) FOR [TimeStamp]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_TotalAssets]  DEFAULT ((0)) FOR [TotalAssets]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_CurrentAssets]  DEFAULT ((0)) FOR [CurrentAssets]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_CashCashEquivalentsAndShortTermInvestments]  DEFAULT ((0)) FOR [CashCashEquivalentsAndShortTermInvestments]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_CashAndCashEquivalents]  DEFAULT ((0)) FOR [CashAndCashEquivalents]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_CashFinancial]  DEFAULT ((0)) FOR [CashFinancial]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_CashEquivalents]  DEFAULT ((0)) FOR [CashEquivalents]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_Inventory]  DEFAULT ((0)) FOR [Inventory]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_TotalNonCurrentAssets]  DEFAULT ((0)) FOR [TotalNonCurrentAssets]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_NetPPE]  DEFAULT ((0)) FOR [NetPPE]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_GrossPPE]  DEFAULT ((0)) FOR [GrossPPE]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_Properties]  DEFAULT ((0)) FOR [Properties]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_ConstructionInProgress]  DEFAULT ((0)) FOR [ConstructionInProgress]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_AccumulatedDepreciation]  DEFAULT ((0)) FOR [AccumulatedDepreciation]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_GoodwillAndOtherIntangibleAssets]  DEFAULT ((0)) FOR [GoodwillAndOtherIntangibleAssets]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_Goodwill]  DEFAULT ((0)) FOR [Goodwill]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_OtherIntangibleAssets]  DEFAULT ((0)) FOR [OtherIntangibleAssets]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_InvestmentsAndAdvances]  DEFAULT ((0)) FOR [InvestmentsAndAdvances]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_LongTermEquityInvestment]  DEFAULT ((0)) FOR [LongTermEquityInvestment]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_InvestmentsinSubsidiariesatCost]  DEFAULT ((0)) FOR [InvestmentsinSubsidiariesatCost]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_InvestmentsinAssociatesatCost]  DEFAULT ((0)) FOR [InvestmentsinAssociatesatCost]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_InvestmentsinJointVenturesatCost]  DEFAULT ((0)) FOR [InvestmentsinJointVenturesatCost]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_NonCurrentNoteReceivables]  DEFAULT ((0)) FOR [NonCurrentNoteReceivables]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_DefinedPensionBenefit]  DEFAULT ((0)) FOR [DefinedPensionBenefit]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_CurrentLiabilities]  DEFAULT ((0)) FOR [CurrentLiabilities]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_CurrentDebt]  DEFAULT ((0)) FOR [CurrentDebt]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_CurrentNotesPayable]  DEFAULT ((0)) FOR [CurrentNotesPayable]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_CommercialPaper]  DEFAULT ((0)) FOR [CommercialPaper]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_LineOfCredit]  DEFAULT ((0)) FOR [LineOfCredit]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_OtherCurrentBorrowings]  DEFAULT ((0)) FOR [OtherCurrentBorrowings]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_CurrentCapitalLeaseObligation]  DEFAULT ((0)) FOR [CurrentCapitalLeaseObligation]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_CurrentDeferredLiabilities]  DEFAULT ((0)) FOR [CurrentDeferredLiabilities]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_CurrentDeferredRevenue]  DEFAULT ((0)) FOR [CurrentDeferredRevenue]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_OtherCurrentLiabilities]  DEFAULT ((0)) FOR [OtherCurrentLiabilities]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_TotalNonCurrentLiabilitiesNetMinorityInterest]  DEFAULT ((0)) FOR [TotalNonCurrentLiabilitiesNetMinorityInterest]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_LongTermDebtAndCapitalLeaseObligation]  DEFAULT ((0)) FOR [LongTermDebtAndCapitalLeaseObligation]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_LongTermDebt]  DEFAULT ((0)) FOR [LongTermDebt]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_LongTermCapitalLeaseObligation]  DEFAULT ((0)) FOR [LongTermCapitalLeaseObligation]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_NonCurrentDeferredLiabilities]  DEFAULT ((0)) FOR [NonCurrentDeferredLiabilities]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_NonCurrentDeferredTaxesLiabilities]  DEFAULT ((0)) FOR [NonCurrentDeferredTaxesLiabilities]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_EmployeeBenefits]  DEFAULT ((0)) FOR [EmployeeBenefits]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_NonCurrentPensionAndOtherPostretirementBenefitPlans]  DEFAULT ((0)) FOR [NonCurrentPensionAndOtherPostretirementBenefitPlans]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_PreferredSecuritiesOutsideStockEquity]  DEFAULT ((0)) FOR [PreferredSecuritiesOutsideStockEquity]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_OtherNonCurrentLiabilities]  DEFAULT ((0)) FOR [OtherNonCurrentLiabilities]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_TotalEquityGrossMinorityInterest]  DEFAULT ((0)) FOR [TotalEquityGrossMinorityInterest]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_StockholdersEquity]  DEFAULT ((0)) FOR [StockholdersEquity]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_CapitalStock]  DEFAULT ((0)) FOR [CapitalStock]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_PreferredStock]  DEFAULT ((0)) FOR [PreferredStock]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_CommonStock]  DEFAULT ((0)) FOR [CommonStock]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_AdditionalPaidInCapital]  DEFAULT ((0)) FOR [AdditionalPaidInCapital]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_RetainedEarnings]  DEFAULT ((0)) FOR [RetainedEarnings]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_TreasuryStock]  DEFAULT ((0)) FOR [TreasuryStock]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_GainsLossesNotAffectingRetainedEarnings]  DEFAULT ((0)) FOR [GainsLossesNotAffectingRetainedEarnings]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_UnrealizedGainLoss]  DEFAULT ((0)) FOR [UnrealizedGainLoss]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_MinimumPensionLiabilities]  DEFAULT ((0)) FOR [MinimumPensionLiabilities]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_TotalCapitalization]  DEFAULT ((0)) FOR [TotalCapitalization]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_CommonStockEquity]  DEFAULT ((0)) FOR [CommonStockEquity]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_CapitalLeaseObligations]  DEFAULT ((0)) FOR [CapitalLeaseObligations]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_NetTangibleAssets]  DEFAULT ((0)) FOR [NetTangibleAssets]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_WorkingCapital]  DEFAULT ((0)) FOR [WorkingCapital]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_InvestedCapital]  DEFAULT ((0)) FOR [InvestedCapital]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_TangibleBookValue]  DEFAULT ((0)) FOR [TangibleBookValue]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_TotalDebt]  DEFAULT ((0)) FOR [TotalDebt]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_NetDebt]  DEFAULT ((0)) FOR [NetDebt]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_ShareIssued]  DEFAULT ((0)) FOR [ShareIssued]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_OrdinarySharesNumber]  DEFAULT ((0)) FOR [OrdinarySharesNumber]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_PreferredSharesNumber]  DEFAULT ((0)) FOR [PreferredSharesNumber]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_TreasurySharesNumber]  DEFAULT ((0)) FOR [TreasurySharesNumber]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_TotalLiabilitiesNetMinorityInterest]  DEFAULT ((0)) FOR [TotalLiabilitiesNetMinorityInterest]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_InterestBearingDepositsLiabilities]  DEFAULT ((0)) FOR [InterestBearingDepositsLiabilities]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_TradingLiabilities]  DEFAULT ((0)) FOR [TradingLiabilities]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_DerivativeProductLiabilities]  DEFAULT ((0)) FOR [DerivativeProductLiabilities]
GO
ALTER TABLE [dbo].[TickerBalanceSheetsQuarterly] ADD  CONSTRAINT [DF_TickerBalanceSheetsQuarterly_OtherLiabilities]  DEFAULT ((0)) FOR [OtherLiabilities]
GO
ALTER TABLE [dbo].[TickerFinancials] ADD  CONSTRAINT [DF_TickerFinancials_TimeStamp]  DEFAULT (getdate()) FOR [TimeStamp]
GO
ALTER TABLE [dbo].[TickerFinancialsHistorical] ADD  CONSTRAINT [DF_TickerFinancialsHistorical_SP500Listed]  DEFAULT ((0)) FOR [SP500Listed]
GO
ALTER TABLE [dbo].[TickerFinancialsHistorical] ADD  CONSTRAINT [DF_TickerFinancialsHistorical_TimeStamp]  DEFAULT (getdate()) FOR [TimeStamp]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_TotalRevenue]  DEFAULT ((0)) FOR [TotalRevenue]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_OperatingRevenue]  DEFAULT ((0)) FOR [OperatingRevenue]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_CostOfRevenue]  DEFAULT ((0)) FOR [CostOfRevenue]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_GrossProfit]  DEFAULT ((0)) FOR [GrossProfit]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_OperatingIncome]  DEFAULT ((0)) FOR [OperatingIncome]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_OperatingExpense]  DEFAULT ((0)) FOR [OperatingExpense]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_InterestExpense]  DEFAULT ((0)) FOR [InterestExpense]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_NonInterestExpense]  DEFAULT ((0)) FOR [NonInterestExpense]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_NetOccupancyExpense]  DEFAULT ((0)) FOR [NetOccupancyExpense]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_ProfessionalExpenseAndContractServicesExpense]  DEFAULT ((0)) FOR [ProfessionalExpenseAndContractServicesExpense]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_GeneralAndAdministrativeExpense]  DEFAULT ((0)) FOR [GeneralAndAdministrativeExpense]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_SellingAndMarketingExpense]  DEFAULT ((0)) FOR [SellingAndMarketingExpense]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_OtherNonInterestExpense]  DEFAULT ((0)) FOR [OtherNonInterestExpense]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_OtherNonOperatingIncomeExpenses]  DEFAULT ((0)) FOR [OtherNonOperatingIncomeExpenses]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_RentExpenseSupplemental]  DEFAULT ((0)) FOR [RentExpenseSupplemental]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_OtherIncomeExpense]  DEFAULT ((0)) FOR [OtherIncomeExpense]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_PretaxIncome]  DEFAULT ((0)) FOR [PretaxIncome]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_TaxProvision]  DEFAULT ((0)) FOR [TaxProvision]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_TaxRateForCalcs]  DEFAULT ((0)) FOR [TaxRateForCalcs]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_NetIncomeCommonStockholders]  DEFAULT ((0)) FOR [NetIncomeCommonStockholders]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_NetIncome]  DEFAULT ((0)) FOR [NetIncome]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_NetIncomeIncludingNoncontrollingInterests]  DEFAULT ((0)) FOR [NetIncomeIncludingNoncontrollingInterests]
GO
ALTER TABLE [dbo].[TickerFinancialsQuarterly] ADD  CONSTRAINT [DF_TickerFinancialsQuarterly_NetIncomeContinuousOperations]  DEFAULT ((0)) FOR [NetIncomeContinuousOperations]
GO
ALTER TABLE [dbo].[TickerHistoricalQualityFactors] ADD  CONSTRAINT [DF_TickerHistoricalQualityFactors_HasPrices]  DEFAULT ((0)) FOR [HasPrices]
GO
ALTER TABLE [dbo].[TickerHistoricalQualityFactors] ADD  CONSTRAINT [DF_TickerHistoricalQualityFactors_LastUpdated]  DEFAULT (getdate()) FOR [LastUpdated]
GO
ALTER TABLE [dbo].[Tickers] ADD  CONSTRAINT [DF_TickerStatus_DaysConstructed]  DEFAULT ((0)) FOR [DaysConstructed]
GO
ALTER TABLE [dbo].[Tickers] ADD  CONSTRAINT [DF_TickerStatus_DaysInPastYear]  DEFAULT ((0)) FOR [DaysInPastYear]
GO
ALTER TABLE [dbo].[Tickers] ADD  CONSTRAINT [DF_TickerStatus_DaysTotal]  DEFAULT ((0)) FOR [DaysTotal]
GO
ALTER TABLE [dbo].[Tickers] ADD  CONSTRAINT [DF_Tickers_SP500Listed]  DEFAULT ((0)) FOR [SP500Listed]
GO
ALTER TABLE [dbo].[Tickers] ADD  CONSTRAINT [DF_Tickers_Delisted]  DEFAULT ((0)) FOR [Delisted]
GO
ALTER TABLE [dbo].[Tickers] ADD  CONSTRAINT [DF_Tickers_ComanySize]  DEFAULT ((0)) FOR [CompanySize]
GO
ALTER TABLE [dbo].[Tickers] ADD  CONSTRAINT [DF_TickersExchanges_TimeStamp]  DEFAULT (getdate()) FOR [TimeStamp]
GO
ALTER TABLE [dbo].[TradeModel_DailyValue] ADD  CONSTRAINT [DF_TradeModel_DailyValue_TimeStamp]  DEFAULT (getdate()) FOR [TimeStamp]
GO
ALTER TABLE [dbo].[TradeModel_Trades] ADD  CONSTRAINT [DF_TradeModel_Trades_TimeStamp]  DEFAULT (getdate()) FOR [TimeStamp]
GO
ALTER TABLE [dbo].[TradeModelComparisons] ADD  CONSTRAINT [DF_ModelComparisons_TimeStamp]  DEFAULT (getdate()) FOR [TimeStamp]
GO
ALTER TABLE [dbo].[TradeModelComparisons] ADD  CONSTRAINT [DF_TradeModelComparisons_totalTickers]  DEFAULT ((0)) FOR [totalTickers]
GO
/****** Object:  StoredProcedure [dbo].[sp_ClearAllCachesAndHistory]    Script Date: 5/29/2023 1:07:36 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO


CREATE  procedure [dbo].[sp_ClearAllCachesAndHistory] AS
delete from PricesWorkingSet
delete from PicksBlended
exec sp_ClearPriceStats
exec sp_PruneShortTermTradeHistory
GO
/****** Object:  StoredProcedure [dbo].[sp_ClearPriceStats]    Script Date: 5/29/2023 1:07:36 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
create procedure [dbo].[sp_ClearPriceStats] AS
DELETE FROM PricesWithPredictions
DELETE FROM PricesWithStats 
GO
/****** Object:  StoredProcedure [dbo].[sp_FixInvalidPrices]    Script Date: 5/29/2023 1:07:36 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO




CREATE procedure [dbo].[sp_FixInvalidPrices] AS

delete from PricesIntraday where [price]=0

--Fix Bad High
--select *, (select Max(v) FROM (values ([Open]),([Low]),([Close])) AS value(v))
update d set High=(select Max(v) FROM (values ([Open]),([Low]),([Close])) AS value(v))
FROM pricesDaily d
where low>High and (high< [Open] or high< [Close])

--Fix Bad Open
--select *, (select AVG(v) FROM (values ([Low]),([High]),([Close])) AS value(v))
update d set [Open]=(select AVG(v) FROM (values ([Low]),([High]),([Close])) AS value(v))
FROM pricesDaily d
where [Open] < Low or [Open] > High 

--Fix Bad Close
--select *, (select AVG(v) FROM (values ([Open]),([High]),([Low])) AS value(v))
update d set [Close]=(select AVG(v) FROM (values ([Open]),([High]),([Low])) AS value(v))
FROM pricesDaily d
where [Close] > High or [Close] < Low 

--Fix Bad Low
--select *, (select Min(v) FROM (values ([Open]),(High),([Close])) AS value(v))
update d set Low=(select Min(v) FROM (values ([Open]),([High]),([Close])) AS value(v))
FROM pricesDaily d
where [Low] > High or [Low] > [Open] or [Low] > [Close] or [Low]=0

Delete  FROM pricesDaily  where low=0
--select *, (select Min(v) FROM (values ([Open]),(High),([Close])) AS value(v))
--Delete  FROM pricesDaily  where (High/Low)-1 > 1.5
GO
/****** Object:  StoredProcedure [dbo].[sp_GetBlendedPicks]    Script Date: 5/29/2023 1:07:36 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO





CREATE procedure [dbo].[sp_GetBlendedPicks] @RequestedDate datetime, @ReEvalPeriod int AS
SET NOCOUNT ON	
Declare @SqlHistory int, @TotalHoldings float
select @SqlHistory=30-@ReEvalPeriod
select @SqlHistory=1 WHERE @SqlHistory < 1
--delete FROM [PicksBlendedDaily] WHERE  DATEPART (weekday, [Date]) in (0,7)
--select * from fn_GetBlendedPicks(@RequestedDate, @SqlHistory)
select @TotalHoldings = Sum(TargetHoldings) from [fn_GetBlendedPicks](@RequestedDate, @SqlHistory)
select Ticker, (TargetHoldings/@TotalHoldings) AS TotalHoldings, TargetHoldings as TargetCount, DateCount, FirstDate, LastDate from [fn_GetBlendedPicks](@RequestedDate, @SqlHistory)
SET NOCOUNT OFF
GO
/****** Object:  StoredProcedure [dbo].[sp_PopulateTickerFinancialsHistoricalByForwardFill]    Script Date: 5/29/2023 1:07:36 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE procedure [dbo].[sp_PopulateTickerFinancialsHistoricalByForwardFill] AS
--populate missing month f1 with prior month f2
insert into TickerFinancialsHistorical (Ticker, Year, Month,  Revenue, OperatingExpense, NetIncome, NetProfitMargin, EarningsPerShare, EBITDA, EffectiveTaxRate, CashShortTermInvestments, TotalAssets, TotalLiabilities,  TotalEquity, SharesOutstanding, PriceToBook, ReturnOnAssetts, ReturnOnCapital )
select f2.Ticker, case when f2.month=12 then f2.Year+1 else f2.Year end, case when f2.month=12 then 1 else f2.Month+1 end, 
 f2.Revenue, f2.OperatingExpense, f2.NetIncome, f2.NetProfitMargin, f2.EarningsPerShare, f2.EBITDA, f2.EffectiveTaxRate, f2.CashShortTermInvestments, f2.TotalAssets, f2.TotalLiabilities,  f2.TotalEquity, f2.SharesOutstanding, f2.PriceToBook, f2.ReturnOnAssetts, f2.ReturnOnCapital 
FROM [TickerFinancialsHistorical] f1
right join [TickerFinancialsHistorical] f2 on f1.ticker=f2.ticker and ((f1.Year=f2.year and f1.Month-1=f2.Month) or (f1.Year-1=f2.year and f1.Month=1 and f2.Month=12))
where f1.ticker is null and (f2.year < year(getDate()) or f2.Month+1 < month(getDate()))
order by f2.ticker, f2.year, f2.month
GO
/****** Object:  StoredProcedure [dbo].[sp_PopulateTickerFinancialsHistoricalFromQuarterly]    Script Date: 5/29/2023 1:07:36 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
create procedure [dbo].[sp_PopulateTickerFinancialsHistoricalFromQuarterly] AS 
insert into TickerFinancialsHistorical (Ticker, Year, Month,  Revenue, OperatingExpense, NetIncome, NetProfitMargin, EarningsPerShare, EBITDA, EffectiveTaxRate, CashShortTermInvestments, TotalAssets, TotalLiabilities,  TotalEquity, SharesOutstanding, PriceToBook, ReturnOnAssetts, ReturnOnCapital )
select q.Ticker, q.Year, q.Month,  q.Revenue, q.OperatingExpense, q.NetIncome, q.NetProfitMargin, q.EarningsPerShare, q.EBITDA, q.EffectiveTaxRate, q.CashShortTermInvestments, q.TotalAssets, q.TotalLiabilities,  q.TotalEquity, q.SharesOutstanding, q.PriceToBook, q.ReturnOnAssetts, q.ReturnOnCapital 
--Price, PriceToCashFlow, PriceToTangibleBook, PriceToSales,
from [rpt_TickerFinancialsQuarterly] q
left join TickerFinancialsHistorical h on q.ticker=h.ticker and q.year=h.year and q.month=h.month
where h.ticker is null --and q.ticker ='googl'
order by q.year, q.month

GO
/****** Object:  StoredProcedure [dbo].[sp_PopulateTickerHistoricalQualityFactors]    Script Date: 5/29/2023 1:07:36 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE procedure [dbo].[sp_PopulateTickerHistoricalQualityFactors] AS
INSERT INTO TickerHistoricalQualityFactors ([Year], Ticker, SP500Listed)
select year(c.Date),  c.Ticker, 1
from SP500HistoricalConstituents c
left join TickerHistoricalQualityFactors h on c.ticker=h.ticker and h.Year=year(c.Date)
where h.ticker is null
Group By c.Ticker, year(Date)

update h set h.SP500Listed=1
from SP500HistoricalConstituents c
INNER join TickerHistoricalQualityFactors h on c.ticker=h.ticker and h.Year=year(c.Date)
where h.SP500Listed=0

INSERT INTO TickerHistoricalQualityFactors ([Year], Ticker, SP500Listed)
select Year(pd.date), pd.Ticker, 0
FROM PricesDaily pd
LEFT JOIN TickerHistoricalQualityFactors h on pd.Ticker=h.ticker and h.Year= Year(pd.date)
WHERE h.Ticker is null and year(pd.Date) > 1980 and pd.Ticker not in ('.INX', '.DJI','.IXIC')
Group By pd.Ticker, year(pd.Date)
ORDER By pd.Ticker, year(pd.Date)

update TickerHistoricalQualityFactors set InTickers=1 where InTickers is null
update h set InTickers=0
--select h.ticker
FROM TickerHistoricalQualityFactors h
left join tickers t on h.ticker=t.ticker
where t.ticker is null and InTickers<>0

update h set SP500Listed=1
from TickerHistoricalQualityFactors h
inner join (select Ticker FROM TickerHistoricalQualityFactors where [year] =1996 and SP500Listed=1) as x on x.ticker=h.ticker
where h.Year < 1996 --my data only goes back to 1996

select h.Ticker, h.Year, Sum(NetChange) NetChange, Count(t.TimeStamp) TradeCount
INTO #Temp
from TradeModel_Trades t
inner join TickerHistoricalQualityFactors h on t.Ticker=h.ticker and Year(t.DateSellOrderFilled) < h.Year
GROUP By h.Ticker, h.Year

update h set PriorTradesAvgChange=NetChange, PriorTradeCount=TradeCount, LastUpdated=getdate()
from TickerHistoricalQualityFactors h
INNER JOIN #Temp t on t.ticker=h.ticker and t.Year=h.year

Update h set PriorYearReturn=c.AnnualChange, LastUpdated=getdate()
from TickerHistoricalQualityFactors h
inner join qry_TickerAnnualChange c on h.ticker=c.ticker and h.year=c.year+1

Update h set HasPrices=1, LastUpdated=getdate()
from TickerHistoricalQualityFactors h
inner join [qry_TickerHasYearData] c on h.ticker=c.ticker and h.year=c.year+1
WHERE HasPrices=0 or HasPrices is null

GO
/****** Object:  StoredProcedure [dbo].[sp_PruneShortTermTradeHistory]    Script Date: 5/29/2023 1:07:36 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE procedure [dbo].[sp_PruneShortTermTradeHistory] AS
select batchname
into #Temp 
from TradeModelcomparisons tm 
WHERE DateDiff(d, TimeStamp, getdate()) > 2
group by batchname HAVING COUNT(*) < 10

update c set BuyHoldEndingValue =0
from TradeModelComparisons c
INNER JOIN #Temp t on c.BatchName=t.BatchName
delete from TradeModelComparisons where BuyHoldEndingValue =0

update c set TotalValue =0
from TradeModel_DailyValue c
INNER JOIN #Temp t on c.BatchName=t.BatchName
delete from TradeModel_DailyValue where TotalValue =0

update c set Units =0
from TradeModel_Trades c
INNER JOIN #Temp t on c.BatchName=t.BatchName
delete from TradeModel_Trades where units=0
GO
/****** Object:  StoredProcedure [dbo].[sp_RenameTicker]    Script Date: 5/29/2023 1:07:36 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE procedure [dbo].[sp_RenameTicker](@OldName varchar(20), @NewName varchar(20)) AS
--select * from tickers where ticker in (@OldName, @NewName)
update tickers set ticker=@NewName, Exchange='' where ticker=@OldName
update [PricesDaily] set ticker=@NewName where ticker=@OldName
update SP500HistoricalConstituents  set ticker=@NewName where ticker=@OldName

GO
/****** Object:  StoredProcedure [dbo].[sp_UpdateBlendedPicks]    Script Date: 5/29/2023 1:07:36 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO




CREATE procedure [dbo].[sp_UpdateBlendedPicks] AS
SET NOCOUNT ON	
Declare @SqlHistory int, @TotalHoldings float
select @SqlHistory=20

delete from PicksBlended
select @TotalHoldings = Sum(TargetHoldings) from [fn_GetBlendedPicks](getdate(), @SqlHistory)
INSERT INTO PicksBlended (Ticker, TargetHoldings, Allocation, DateCount, FirstDate, LastDate)
select Ticker, (TargetHoldings/@TotalHoldings)*20 AS TotalHoldings, (TargetHoldings/@TotalHoldings) as Allocation, DateCount, FirstDate, LastDate from [fn_GetBlendedPicks](getdate(), @SqlHistory)
--select * from fn_GetBlendedPicks(getdate(), 25)
SET NOCOUNT OFF
GO
/****** Object:  StoredProcedure [dbo].[sp_UpdateDailyFromIntraday]    Script Date: 5/29/2023 1:07:36 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO






CREATE Procedure [dbo].[sp_UpdateDailyFromIntraday] AS
SET NOCOUNT ON	--Not having this breaks pyodbc
delete from PricesDaily WHERE Constructed=1
exec sp_UpdateTickerDates
INSERT INTO PricesDaily (Ticker, [Date], [Open], [High], [Low], [Close], Volume, Constructed)
select ohlc.Ticker, ohlc.Date, ohlc.[Open], ohlc.[High], ohlc.[Low], ohlc.[Close], ohlc.Volume, 1 AS Constructed
FROM qry_Intraday_OHLC ohlc
INNER join Tickers ts on ohlc.Ticker=ts.Ticker
WHERE ohlc.[Date] > ts.LatestDateDaily or ts.LatestDateDaily is null
order by Ticker, [Date]
exec sp_UpdateTickerDates
update PricesDaily set constructed=0 where Constructed=1 and datediff(d, date, getdate()) > 30
SET NOCOUNT OFF
GO
/****** Object:  StoredProcedure [dbo].[sp_UpdateEverything]    Script Date: 5/29/2023 1:07:36 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO










CREATE procedure [dbo].[sp_UpdateEverything] AS
SET NOCOUNT ON	--Not having this breaks pyodbc
Exec sp_UpdateDailyFromIntraday
exec sp_FixInvalidPrices
exec sp_UpdateTickerPrices
exec sp_UpdateTickerFinancials
exec sp_UpdateSector
exec sp_UpdateTickerListing
exec sp_UpdateTickerHistoricalQualityFactors
update h set TargetPercent=PercentHolding/(select Sum(PercentHolding) AS Total from HedgeFundHoldings) FROM HedgeFundHoldings h
SET NOCOUNT OFF
GO
/****** Object:  StoredProcedure [dbo].[sp_UpdateSector]    Script Date: 5/29/2023 1:07:36 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO




CREATE procedure [dbo].[sp_UpdateSector] AS
update t set Sector='Transportation' from Tickers t where About like '%transportation%' or About like '%airline%' 
update t set Sector='Technology' from Tickers t where about like '%software%' or about like '%technology%' or about like '%electronic%' or about like '%eaton%' or about like '%cloud%' or about like '%network inf%'
update t set Sector='Insurance' from Tickers t where about like '%insurance%'
update t set Sector='Health Care' from Tickers t where about like '%Pharmaceutical%' or about like '%biotech%' or about like '%Health%' or about like '%HealthCare%' or about like '%medicine%' or about like '%drug%' or about like '%blood%' or about like '%medical%'
update t set Sector='Retail' from Tickers t where about like '%retail%' or about like '%store%' or about like '%clothing%' or about like '%wholesale%' or about like '%cosmetic%' or about like '%fashion%'  or about like '%appare%' or about like '%ebay%'
update t set Sector='Mining' from Tickers t where About like '%mining%' or About like '%minerals%'
update t set Sector='Travel' from Tickers t where About like '%hotel%' or About like '%resort%'
update t set Sector='Automotive' from Tickers t where About like '%Automotive%' or About like '%cars%' or About like '%vehicle%'
update t set Sector='Aerospace' from Tickers t where About like '%Aerospace%' or About like '%aviation%' or About like '%airport%' 
update t set Sector='Manufacturing' from Tickers t where About like '%Manufactur%' or About like '%steel%'  or About like '%industrial%' 
update t set Sector='Commodities' from Tickers t where About like '%agricultural%' or About like '%commodities%' or  About like '%fertilizer%' or  About like '%chemical%' or  About like '%tobac%'  or  About like '%paper%' 
update t set Sector='Financial' from Tickers t where About like '%bank%' or About like '%financial%' or About like '%card servic%' or About like '%credit%' 
update t set Sector='Retail Food' from Tickers t where About like '%food%' or About like '%beverage%' or About like '%restaur%'  or About like '%drink%' 
update t set Sector='Telecom' from Tickers t where About like '%Telecom%' or About like '%cellular%' 
update t set Sector='Real Estate' from Tickers t where About like '%real estate%' 
update t set Sector='Professional Services' from Tickers t where About like '%staffing%' or  About like '%payroll%'  or  About like '%market measur%'  or  About like '%management services%'  or  About like '%professional services%' 
update t set Sector='Mass Media' from Tickers t where About like '%media%' or  About like '%blogging%' 
update t set Sector='Construction' from Tickers t where About like '%Construction%' or  About like '%home build%' 
update t set Sector='Energy' from Tickers t where About like '%energy%' or About like '%gas%' or About like '%coal%' or About like '%oil%' or About like '%electric%' or About like '%petrol%' or About like '%Hydrocarbon%'
update t set Sector='Other' from Tickers t where sector is null 
update t set CompanyName=h.CompanyName, Sector=h.Sector FROM SP500Constituents2022 h inner join Tickers t on t.Ticker=h.Ticker 
GO
/****** Object:  StoredProcedure [dbo].[sp_UpdateTickerDates]    Script Date: 5/29/2023 1:07:36 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO


CREATE Procedure [dbo].[sp_UpdateTickerDates] AS
SET NOCOUNT ON	--Not having this breaks pyodbc
update [PricesIntraday] set [Date] = DATEFROMPARTS ([year],[month],[day])  WHERE [Date] is null
update ts set EarliestDateIntraday=p.EarliestDate, LatestDateIntraday=p.LatestDate FROM Tickers ts  INNER JOIN (select [Ticker], min(DateTime) AS EarliestDate, Max(DateTime) AS LatestDate FROM  [PricesIntraday] GROUP By Ticker) AS p on p.Ticker=ts.Ticker WHERE LatestDateIntraday <> p.LatestDate or LatestDate is null
update ts set EarliestDateDaily=p.EarliestDate, LatestDateDaily=p.LatestDate FROM Tickers ts INNER JOIN (select [Ticker], min([Date]) AS EarliestDate, Max([Date]) AS LatestDate FROM  PricesDaily GROUP By Ticker) AS p on p.Ticker=ts.Ticker WHERE LatestDateDaily<>p.LatestDate or LatestDateDaily is null
update ts set DaysConstructed=pd.Constructed FROM Tickers ts inner join (select Ticker, Sum(Constructed) AS Constructed from PricesDaily GROUP BY Ticker) AS pd on pd.Ticker=ts.Ticker WHERE DaysConstructed<>pd.Constructed
update ts set DaysTotal=pd.[Days] FROM Tickers ts inner join (select Ticker, Count(Date) AS [Days] from PricesDaily GROUP BY Ticker) AS pd on pd.Ticker=ts.Ticker WHERE DaysTotal <> pd.[Days]
update ts set DaysInPastYear=pd.[Days] FROM Tickers ts inner join (select Ticker, Count(Date) AS [Days] from PricesDaily where datediff(d, [Date], getDate()) < 365 GROUP BY Ticker) AS pd on pd.Ticker=ts.Ticker WHERE DaysInPastYear <> pd.[Days]
update ts set [timestamp] = LatestDateDaily  FROM Tickers ts WHERE  [timestamp] < LatestDateDaily 
update ts set [timestamp] = LatestDateIntraday  FROM Tickers ts WHERE  [timestamp] < LatestDateIntraday 
update t set VolumeAverage=x.volume
FROM tickers t
inner join (select Ticker, avg(volume) AS Volume FROM pricesdaily where datediff(d, [date], getdate()) < 365  group by ticker) as x on x.ticker=t.ticker where VolumeAverage <> x.volume
SET NOCOUNT OFF
GO
/****** Object:  StoredProcedure [dbo].[sp_UpdateTickerFinancials]    Script Date: 5/29/2023 1:07:36 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO






CREATE procedure [dbo].[sp_UpdateTickerFinancials] AS 
update f set [Month]=Month(Timestamp) from TickerFinancials f where [Month] is null
update t set MarketCap = SharesOutstanding*CurrentPrice/1000, NetIncome=f.NetIncome, NetProfitMargin=f.NetProfitMargin, ReturnOnCapital=f.ReturnOnCapital, t.FinancialsLastUpdated=f.TimeStamp
from tickers t inner join TickerFinancials f on t.ticker=f.ticker and [year]=Year(GetDate())
where SharesOutstanding > 0 --or t.NetIncome is null or t.ReturnOnCapital is null or FinancialsLastUpdated is null

update t set PE_Ratio =  t.CurrentPrice / f.EarningsPerShare from tickers t inner join TickerFinancials f on t.ticker=f.ticker and [year]=Year(GetDate()) where  f.EarningsPerShare >0 --PE_Ratio is null and
update t set MarketCap = SharesOutstanding*CurrentPrice/1000, NetIncome=f.NetIncome, NetProfitMargin=f.NetProfitMargin, ReturnOnCapital=f.ReturnOnCapital, t.FinancialsLastUpdated=f.TimeStamp
from tickers t inner join TickerFinancials f on t.ticker=f.ticker and [year]=Year(GetDate()) 
where SharesOutstanding > 0 -- MarketCap=0 and or t.NetIncome is null or t.ReturnOnCapital is null or FinancialsLastUpdated is null

insert into TickerFinancialsHistorical (Ticker, Year, Month, Revenue, OperatingExpense, NetIncome, NetProfitMargin, EarningsPerShare, EBITDA, EffectiveTaxRate, CashShortTermInvestments, TotalAssets, TotalLiabilities, TotalEquity, SharesOutstanding, PriceToBook, ReturnOnAssetts, ReturnOnCapital, CashFromOperations, CashFromInvesting, CashFromFinancing, NetChangeInCash, FreeCashFlow)
select tf.Ticker, tf.Year, tf.Month, tf.Revenue, tf.OperatingExpense, tf.NetIncome, tf.NetProfitMargin, tf.EarningsPerShare, tf.EBITDA, tf.EffectiveTaxRate, tf.CashShortTermInvestments, tf.TotalAssets, tf.TotalLiabilities, tf.TotalEquity, tf.SharesOutstanding, tf.PriceToBook, tf.ReturnOnAssetts, tf.ReturnOnCapital, tf.CashFromOperations, tf.CashFromInvesting, tf.CashFromFinancing, tf.NetChangeInCash, tf.FreeCashFlow
from TickerFinancials tf
LEFT JOIN TickerFinancialsHistorical h on tf.Ticker=h.Ticker and tf.Year=h.Year and tf.Month=h.Month
WHERE h.Ticker is null

update q set Price=pd.[Close] FROM TickerFinancialsHistorical q inner join pricesDaily pd on q.ticker=pd.Ticker and q.Year=Year(pd.Date) and q.month=Month(pd.Date) and Day(pd.Date) between 1 and 4 WHERE q.price is null

update h set PriceToCashFlow=t.CurrentPrice/(CashFromOperations/SharesOutstanding)  from TickerFinancials h  inner join tickers t on h.ticker=t.ticker WHERE CashFromOperations > 0 and SharesOutstanding > 0 and t.CurrentPrice > 0 and (PriceToCashFlow is null or PriceToCashFlow=0)
update h set PriceToCashFlow=price/(CashFromOperations/SharesOutstanding) from TickerFinancialsHistorical h WHERE CashFromOperations > 0 and SharesOutstanding > 0 and price > 0 and (PriceToCashFlow is null or PriceToCashFlow=0)

update h set MarketCapitalization=t.CurrentPrice*SharesOutstanding from TickerFinancials h  inner join tickers t on h.ticker=t.ticker WHERE  SharesOutstanding > 0 and t.CurrentPrice > 0 and (MarketCapitalization is null or MarketCapitalization=0)
update h set MarketCapitalization=Price*SharesOutstanding from TickerFinancialsHistorical h WHERE  SharesOutstanding > 0 and Price > 0 and (MarketCapitalization is null or MarketCapitalization=0)

update h set PriceToSales=MarketCapitalization / Revenue from TickerFinancials h WHERE  MarketCapitalization > 0 and Revenue <> 0 and (PriceToSales is null or PriceToSales=0)
update h set PriceToSales=MarketCapitalization / Revenue from TickerFinancialsHistorical h WHERE  MarketCapitalization > 0 and Revenue <> 0 and (PriceToSales is null or PriceToSales=0)

update f1  set f1.PriceMPC = (f1.Price/f2.Price)-1
FROM [TickerFinancialsHistorical] f1
inner join [TickerFinancialsHistorical] f2 on f1.ticker=f2.ticker and ((f1.Year=f2.year and f1.Month-1=f2.Month) or (f1.Year-1=f2.year and f1.Month=1 and f2.Month=12))
where f1.price <> 0 and f2.price <> 0 and (f1.PriceMPC=0 or f1.PriceMPC is null) and not f1.price = f2.price

update f1  set f1.PriceYPC = (f1.Price/f2.Price)-1
FROM [TickerFinancialsHistorical] f1
inner join [TickerFinancialsHistorical] f2 on f1.ticker=f2.ticker and (f1.Year-1=f2.year and f1.Month=f2.Month)
where f1.price <> 0 and f2.price <> 0 and (f1.PriceYPC=0 or f1.PriceYPC is null) and not f1.price = f2.price

update f1  set f1.PriceYPC = (f1.Price/f2.Price)-1
FROM [TickerFinancialsHistorical] f1
inner join [TickerFinancialsHistorical] f2 on f1.ticker=f2.ticker and (f1.Year-1=f2.year and f1.Month+2=f2.Month)
where f1.price <> 0 and f2.price <> 0 and (f1.PriceYPC=0 or f1.PriceYPC is null) and not f1.price = f2.price

update f1  set f1.PriceYPC = (f1.Price/f2.Price)-1
FROM [TickerFinancialsHistorical] f1
inner join [TickerFinancialsHistorical] f2 on f1.ticker=f2.ticker and (f1.Year-1=f2.year and f1.Month+3=f2.Month)
where f1.price <> 0 and f2.price <> 0 and (f1.PriceYPC=0 or f1.PriceYPC is null) and not f1.price = f2.price

update f1  set f1.PriceYPC = (f1.Price/f2.Price)-1
FROM [TickerFinancialsHistorical] f1
inner join [TickerFinancialsHistorical] f2 on f1.ticker=f2.ticker and ((f1.Year-1=f2.year and f1.Month+4=f2.Month))
where f1.price <> 0 and f2.price <> 0 and (f1.PriceYPC=0 or f1.PriceYPC is null) and not f1.price = f2.price

update f1  set  f1.RevenueMPC = (f1.Revenue /f2.Revenue)-1
FROM [TickerFinancialsHistorical] f1
inner join [TickerFinancialsHistorical] f2 on f1.ticker=f2.ticker and ((f1.Year=f2.year and f1.Month-1=f2.Month) or (f1.Year-1=f2.year and f1.Month=1 and f2.Month=12))
where f1.Revenue <> 0 and f2.Revenue <> 0 and (f1.RevenueMPC=0 or f1.RevenueMPC is null) --and not f1.Revenue = f2.Revenue

update f1  set  f1.RevenueYPC = (f1.Revenue /f2.Revenue)-1
FROM [TickerFinancialsHistorical] f1
inner join [TickerFinancialsHistorical] f2 on f1.ticker=f2.ticker and ((f1.Year-1=f2.year and f1.Month=f2.Month) or (f1.Year-1=f2.year and f1.Month+1=f2.Month))
where f1.Revenue <> 0 and f2.Revenue <> 0 and (f1.RevenueYPC=0 or f1.RevenueYPC is null) --and not f1.Revenue = f2.Revenue

update f1  set f1.NetIncomeMPC = (f1.NetIncome /f2.NetIncome)-1
FROM [TickerFinancialsHistorical] f1
inner join [TickerFinancialsHistorical] f2 on f1.ticker=f2.ticker and ((f1.Year=f2.year and f1.Month-1=f2.Month) or (f1.Year-1=f2.year and f1.Month=1 and f2.Month=12))
where f1.NetIncome <> 0 and f2.NetIncome <> 0 and (f1.NetIncomeMPC=0 or f1.NetIncomeMPC is null)-- and not f1.NetIncome = f2.NetIncome

update f1  set f1.NetIncomeYPC = (f1.NetIncome /f2.NetIncome)-1
FROM [TickerFinancialsHistorical] f1
inner join [TickerFinancialsHistorical] f2 on f1.ticker=f2.ticker and ((f1.Year-1=f2.year and f1.Month=f2.Month) or (f1.Year-1=f2.year and f1.Month+1=f2.Month))
where f1.NetIncome <> 0 and f2.NetIncome <> 0 and (f1.NetIncomeYPC=0 or f1.NetIncomeYPC is null)-- and not f1.NetIncome = f2.NetIncome

update f1  set  f1.TotalLiabilitiesMPC = (f1.TotalLiabilities /f2.TotalLiabilities)-1
FROM [TickerFinancialsHistorical] f1
inner join [TickerFinancialsHistorical] f2 on f1.ticker=f2.ticker and ((f1.Year=f2.year and f1.Month-1=f2.Month) or (f1.Year-1=f2.year and f1.Month=1 and f2.Month=12))
where f1.TotalLiabilities <> 0 and f2.TotalLiabilities <> 0 and (f1.TotalLiabilitiesMPC=0 or f1.TotalLiabilitiesMPC is null)-- and not f1.TotalLiabilities = f2.TotalLiabilities

update f1  set  f1.TotalLiabilitiesYPC = (f1.TotalLiabilities /f2.TotalLiabilities)-1
FROM [TickerFinancialsHistorical] f1
inner join [TickerFinancialsHistorical] f2 on f1.ticker=f2.ticker and ((f1.Year-1=f2.year and f1.Month=f2.Month) or (f1.Year-1=f2.year and f1.Month+1=f2.Month))
where f1.TotalLiabilities <> 0 and f2.TotalLiabilities <> 0 and (f1.TotalLiabilitiesYPC=0 or f1.TotalLiabilitiesYPC is null)-- and not f1.TotalLiabilities = f2.TotalLiabilities

update h set SP500Listed=qf.SP500Listed   from TickerFinancialsHistorical h  inner join [TickerHistoricalQualityFactors] qf on h.Year=qf.year and h.Ticker=qf.ticker  where h.SP500Listed <> qf.SP500Listed
GO
/****** Object:  StoredProcedure [dbo].[sp_UpdateTickerFinancialsQuarterly]    Script Date: 5/29/2023 1:07:36 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO


CREATE procedure [dbo].[sp_UpdateTickerFinancialsQuarterly] AS
Update TickerFinancialsQuarterly set OperatingExpense=NonInterestExpense WHERE OperatingExpense=0 and NonInterestExpense<>0
Update TickerFinancialsQuarterly set GrossProfit=TotalRevenue-CostOfRevenue WHERE GrossProfit=0 and TotalRevenue<>0
update TickerBalanceSheetsQuarterly SET TotalLiabilitiesNetMinorityInterest =CurrentLiabilities  where CurrentLiabilities <> 0 and  TotalLiabilitiesNetMinorityInterest = 0 
Update q set BasicAverageShares =ShareIssued FROM TickerFinancialsQuarterly q inner join TickerBalanceSheetsQuarterly bs on q.Ticker=bs.Ticker and q.Date=bs.Date WHERE BasicAverageShares is null

update q set Price=pd.[Open] FROM TickerFinancialsQuarterly q inner join pricesDaily pd on q.ticker=pd.Ticker and q.Date=pd.Date WHERE q.price is null
update q set Price=pd.[Close] FROM TickerFinancialsQuarterly q inner join pricesDaily pd on q.ticker=pd.Ticker and q.Date=dateadd(d, -1, pd.Date) WHERE q.price is null
update q set Price=pd.[Close] FROM TickerFinancialsQuarterly q inner join pricesDaily pd on q.ticker=pd.Ticker and q.Date=dateadd(d, -2, pd.Date) WHERE q.price is null
update q set Price=pd.[Close] FROM TickerFinancialsQuarterly q inner join pricesDaily pd on q.ticker=pd.Ticker and q.Date=dateadd(d, -3, pd.Date) WHERE q.price is null
update q set Price=pd.[Close] FROM TickerFinancialsQuarterly q inner join pricesDaily pd on q.ticker=pd.Ticker and q.Date=dateadd(d, -4, pd.Date) WHERE q.price is null

update q set  BasicEPS = NetIncome/BasicAverageShares
from TickerFinancialsQuarterly q
WHERE NetIncome <> 0 and BasicAverageShares <> 0 and BasicEPS is null

--select price/BasicEPS
update q set  CAPEI = price/BasicEPS
from TickerFinancialsQuarterly q
WHERE price <> 0 and BasicEPS <> 0 and CAPEI is null

update q set MarketCap= Price * BasicAverageShares
from TickerFinancialsQuarterly q
WHERE price <> 0 and BasicAverageShares <> 0 and MarketCap is null

update q set PriceCashFlow = MarketCap / (OperatingIncome-OperatingExpense)
from TickerFinancialsQuarterly q
WHERE MarketCap <> 0 and (OperatingIncome-OperatingExpense) <> 0 and PriceCashFlow is null

update q set PriceToSales = MarketCap / OperatingIncome
from TickerFinancialsQuarterly q
WHERE MarketCap <> 0 and (OperatingIncome) <> 0 and PriceToSales is null

update q set PriceToBook = Price / TangibleBookValue
from TickerFinancialsQuarterly q
inner join TickerBalanceSheetsQuarterly bs on q.Date=bs.Date and q.Ticker=bs.Ticker
WHERE Price <> 0 and TangibleBookValue <> 0 and PriceToBook is null

GO
/****** Object:  StoredProcedure [dbo].[sp_UpdateTickerHistoricalQualityFactors]    Script Date: 5/29/2023 1:07:36 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

CREATE Procedure [dbo].[sp_UpdateTickerHistoricalQualityFactors] AS
  insert into [TickerHistoricalQualityFactors] (Year, Ticker, [SP500Listed], [InTickers], [HasPrices], [PriorYearReturn])
  select Year(GetDate()), t.Ticker, t.SP500Listed, 1,1, t.[1YearReturn]
  FROM Tickers t
  LEFT JOIN [TickerHistoricalQualityFactors] hf on t.Ticker=hf.Ticker and hf.Year=Year(GetDate())
  WHERE  hf.Ticker is null and t.Delisted=0 and t.currentPrice is not null


update q set Price=pd.[Close] FROM [TickerFinancialsHistorical] q inner join pricesDaily pd on q.ticker=pd.Ticker and q.year=Year(pd.date) and q.month=Month(pd.date) and day(pd.Date) < 2  WHERE q.price is null
update q set Price=pd.[Close] FROM [TickerFinancialsHistorical] q inner join pricesDaily pd on q.ticker=pd.Ticker and q.year=Year(pd.date) and q.month=Month(pd.date) and day(pd.Date) < 4  WHERE q.price is null
update q set Price=pd.[Close] FROM [TickerFinancialsHistorical] q inner join pricesDaily pd on q.ticker=pd.Ticker and q.year=Year(pd.date) and q.month=Month(pd.date) and day(pd.Date) < 6  WHERE q.price is null
update q set Price=pd.[Close] FROM [TickerFinancialsHistorical] q inner join pricesDaily pd on q.ticker=pd.Ticker and q.year=Year(pd.date) and q.month=Month(pd.date) and day(pd.Date) < 10  WHERE q.price is null
update q set Price=pd.[Close] FROM [TickerFinancialsHistorical] q inner join pricesDaily pd on q.ticker=pd.Ticker and q.year=Year(pd.date) and q.month=Month(pd.date) WHERE q.price is null
GO
/****** Object:  StoredProcedure [dbo].[sp_UpdateTickerListing]    Script Date: 5/29/2023 1:07:36 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO


CREATE procedure [dbo].[sp_UpdateTickerListing] AS
IF NOT Exists(select Max([year]) FROM SP500ConstituentsYearly  HAVING Max([year])=Year(GetDate())) 
BEGIN
INSERT INTO SP500ConstituentsYearly (Ticker, [Year]) select ticker, Year(GetDate())  from tickers where SP500Listed = 1
END
update tickers set CompanyName = Replace(CompanyName, '&amp;', '&') where CompanyName is not null and CompanyName like '%&amp;%'
update t set Delisted =1  from Tickers t where Exchange like '%Delisted%' and  Delisted <> 1 
update t set sp500Listed=1 FROM SP500ConstituentsYearly h inner join Tickers t on t.Ticker=h.Ticker and SP500Listed <> 1 and exchange<>'delisted' and h.year >=2022
update tickers set SP500Listed=0 WHERE (MarketCap < 5.5 or exchange='delisted') and SP500Listed=1
update tickers set CompanySize=3 WHERE MarketCap > 10 and CompanySize <> 3
update tickers set CompanySize=2 WHERE MarketCap between 2 and 10 and CompanySize <> 2
update tickers set CompanySize=1 WHERE MarketCap < 2 and CompanySize <> 1
GO
/****** Object:  StoredProcedure [dbo].[sp_UpdateTickerPrices]    Script Date: 5/29/2023 1:07:36 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE procedure [dbo].[sp_UpdateTickerPrices] AS
SET NOCOUNT ON
--select t.Ticker, c.Price, m.price, y.Price
update t set CurrentPrice=IsNull(c.Price, cb.Price) FROM Tickers t  LEFT JOIN qry_PriceCurrent c on c.Ticker=t.Ticker inner join [qry_PriceCurrentBroad] cb on cb.Ticker=t.Ticker

--select t.ticker,  t.CurrentPrice/m.Price-1
update t set  [1MonthReturn]=(t.CurrentPrice/m.Price-1)
FROM Tickers t INNER JOIN qry_Price1Mo m on m.Ticker=t.Ticker

update t set  [1YearReturn]=(t.CurrentPrice/y.Price-1)
FROM Tickers t INNER JOIN qry_Price1Year y on y.Ticker=t.Ticker

SET NOCOUNT OFF
GO
/****** Object:  StoredProcedure [dbo].[sp_UpdateTradeHistory]    Script Date: 5/29/2023 1:07:36 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
create procedure [dbo].[sp_UpdateTradeHistory] AS 
Update t set TradesNetChange=r.NetChange, TradesAvgChange=r.AvgChange, TradesMinChange=r.MinChange, TradesMaxChange=r.MaxChange , TradesCount=r.TotalTrades
from tickers t
inner join rpt_NetTradesByTicker r on t.ticker=r.ticker
GO
/****** Object:  StoredProcedure [dbo].[sp_UpdateTradeModelBatches]    Script Date: 5/29/2023 1:07:36 PM ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE procedure [dbo].[sp_UpdateTradeModelBatches] as 
update t set BatchName=substring(ModelName, 1,30) +'_' + Replace(Replace(convert(varchar,[TimeStamp], 120), '-',''), ':','')  FROM TradeModelComparisons t where batchname is null
update t set BatchName=substring(trademodel, 1,30) +'_' + Replace(Replace(convert(varchar,[TimeStamp], 120), '-',''), ':','')  FROM [TradeModel_DailyValue] t where batchname is null
update t set BatchName=substring(trademodel, 1,30) +'_' + Replace(Replace(convert(varchar,[TimeStamp], 120), '-',''), ':','')  FROM TradeModel_Trades t where batchname is null
update tc set StocksBought=x.StocksBought
from TradeModelComparisons tc
INNER JOIN (select TradeModel, count(distinct ticker) AS StocksBought from TradeModel_Trades group by TradeModel) as x on x.TradeModel=tc.BatchName
WHERE tc.StocksBought is null
GO
USE [master]
GO
ALTER DATABASE [PTA] SET  READ_WRITE 
GO
