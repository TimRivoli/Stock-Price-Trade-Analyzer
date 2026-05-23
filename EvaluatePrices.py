from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import pandas as pd
import matplotlib.pyplot as plt
from _classes.Graphing import PlotHelper
from _classes.MarketEvents import MarketEvent, MarketCrashes
from _classes.Prices import PricingData, PriceSnapshot
from _classes.Selection import StockPicker
from _classes.TickerLists import TickerLists
from _classes.Utility import *

def PlotAnnualPerformance(ticker:str='.INX'):
	print('Annual performance rate for ' + ticker)
	prices = PricingData(ticker)
	if prices.LoadHistory():
		x = prices.GetPriceHistory(['Average'])
		yearly=x.groupby([(x.index.year)]).first()
		yearlyChange = yearly.pct_change(1)
		monthly=x.groupby([(x.index.year),(x.index.month)]).first()
		plot = PlotHelper()
		plot.PlotDataFrame(yearly, title=f"Annual Performance for {ticker}", adjustScale=False)
		plot.PlotDataFrame(monthly, title=f"Monthly Performance for {ticker}", adjustScale=False)
		plot.PlotDataFrame(yearlyChange, title=f"Yearly Percentage Gain for {ticker}", adjustScale=False)
		print('Average annual change from ', prices.historyStartDate, ' to ', prices.historyEndDate, ': ', yearlyChange.mean().values * 100, '%')

def DownloadAndSaveStocks(tickerList:list):
	for ticker in tickerList:
		prices = PricingData(ticker=ticker)
		prices.LoadHistory(requestedEndDate=GetLatestBDay(), verbose=True)

def DownloadAndSaveStocksWithStats(tickerList:list, startDate:str = None, endDate:str=GetLatestBDay()):
	for ticker in tickerList:
		prices = PricingData(ticker)
		if prices.LoadHistory(requestedStartDate=startDate, requestedEndDate=endDate, verbose=True):
			prices.CalculateStats()
			prices.SavePricesWithStats(includePredictions=False, toDatebase=False, verbose=True)

def GraphTimePeriod(ticker:str, endDate:str, days:int):
	prices = PricingData(ticker)
	if prices.LoadHistory():
		prices.GraphData(endDate=endDate, daysToGraph=days, graphTitle=None, includePredictions=False, saveToFile=True, fileNameSuffix=None, verbose=False)

def GraphHistoricalDrops():
	ticker = '.INX'
	crash_troughs = [datetime(e.LowDate.year, e.LowDate.month, e.LowDate.day) for e in MarketCrashes.GetAll()]
	print(f"Trough Dates (The bottom): {crash_troughs}")
	prices = PricingData(ticker)
	if prices.LoadHistory():
		for d in crash_troughs:
			endDate = d
			for duration in [60, 120, 365]:
				prices.GraphData(endDate=endDate, daysToGraph=duration, graphTitle=None, includePredictions=False, saveToFile=True, fileNameSuffix=None, verbose=False)
			endDate += relativedelta(months=6)
			for duration in [60, 120, 365]:
				prices.GraphData(endDate=endDate, daysToGraph=duration, graphTitle=None, includePredictions=False, saveToFile=True, fileNameSuffix=None, verbose=False)

def PriceCheck(startDate: str, Ticker:str):
	startDate = ToDate(startDate)
	endDate = AddDays(startDate, 30)
	prices = PricingData(ticker)
	prices.LoadHistory(startDate, endDate)
	sn = prices.GetPriceSnapshot(startDate, True)
	startPrice = sn.Average
	sn = prices.GetPriceSnapshot(endDate, True)
	endPrice = sn.Average
	print('From', startDate, ' to ', endDate)
	print(ticker, startPrice, endPrice, (endPrice/startPrice-1)*100)

def ShowPicks(tickerList: list):
	forDate = GetLatestBDay()
	picker = StockPicker()
	picker.AlignToList(tickerList)
	for option in [2,3,4,5]:
		picks = picker.GetHighestPriceMomentum(forDate, stocksToReturn=10, filterOption=option)
		print(f"Picks for filter option {option}")
		print(picks)
	print(f"Picks for Blended option")
	picks = picker.GetPicksBlended(forDate)
	print(picks)
	print(f"Picks for AdaptiveConvex option")
	picks = picker.GetAdaptiveConvexPicks(forDate)
	print(picks)

# ──────────────────────────────────────────────────────────────────────────────
# New examples
# ──────────────────────────────────────────────────────────────────────────────

def CrashRecoverySpeed(ticker: str = '.INX'):
	"""For each major crash, print how many trading days it took to fully recover."""
	prices = PricingData(ticker)
	if not prices.LoadHistory():
		return
	hist = prices.GetPriceHistory(['Average'])
	print(f'\nCrash Recovery Speed for {ticker}')
	print(f"{'Crash':<32} {'Drop':>7}  {'Trough Date':<13} {'Recovery Days':>14} {'Recovery Date'}")
	print('-' * 90)
	for crash in MarketCrashes.GetAll():
		peak_label   = hist.index.asof(pd.Timestamp(crash.StartDate))
		trough_label = hist.index.asof(pd.Timestamp(crash.LowDate))
		if pd.isnull(peak_label) or pd.isnull(trough_label):
			print(f"{crash.Label:<32}  -- insufficient data --")
			continue
		peak_price   = hist.loc[peak_label,  'Average']
		trough_price = hist.loc[trough_label, 'Average']
		drop_pct = (trough_price / peak_price - 1) * 100
		post_trough = hist.loc[trough_label:]
		recovered = post_trough[post_trough['Average'] >= peak_price]
		if not recovered.empty:
			recovery_date = recovered.index[0]
			trading_days  = len(hist.loc[trough_label:recovery_date]) - 1
			print(f"{crash.Label:<32} {drop_pct:>6.1f}%  {str(trough_label.date()):<13} {trading_days:>14}  {str(recovery_date.date())}")
		else:
			print(f"{crash.Label:<32} {drop_pct:>6.1f}%  {str(trough_label.date()):<13}  still recovering")


def NormalizedComparison(tickers: list, startDate: str = '1/1/2010', title: str = 'Normalized Price Comparison (Base=100)'):
	"""Overlay multiple tickers normalized to 100 at startDate to compare growth."""
	start = ToDate(startDate)
	frames = {}
	for ticker in tickers:
		p = PricingData(ticker)
		if not p.LoadHistory(requestedStartDate=start):
			print(f"  Could not load {ticker}, skipping.")
			continue
		hist = p.GetPriceHistory(['Average'])
		hist = hist[hist.index >= pd.Timestamp(start)]
		if hist.empty:
			continue
		base = hist['Average'].iloc[0]
		if base > 0:
			frames[ticker] = (hist['Average'] / base) * 100
	if not frames:
		print("No data loaded.")
		return
	combined = pd.DataFrame(frames).ffill()
	plot = PlotHelper()
	plot.PlotDataFrame(combined, title=title, ylabel='Value (Base=100)')
	print(f'\nNormalized Comparison from {startDate}')
	for ticker in combined.columns:
		final = combined[ticker].dropna().iloc[-1]
		print(f"  {ticker:<8}: {final:>7.1f}  ({(final / 100 - 1) * 100:+.1f}%)")


def SeasonalMonthlyReturns(ticker: str = '.INX'):
	"""Show which calendar months historically return the most / least for a ticker."""
	prices = PricingData(ticker)
	if not prices.LoadHistory():
		return
	hist = prices.GetPriceHistory(['Average'])
	monthly     = hist['Average'].resample('MS').first().dropna()
	monthly_ret = monthly.pct_change().dropna() * 100
	avg_by_month = monthly_ret.groupby(monthly_ret.index.month).mean()
	month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
	avg_by_month.index = [month_names[m - 1] for m in avg_by_month.index]
	print(f'\nAverage Monthly Return by Calendar Month for {ticker}')
	for month, ret in avg_by_month.items():
		bar = '|' * max(1, int(abs(ret) * 3))
		print(f"  {month:<4}: {ret:+6.2f}%  {bar}")
	fig, ax = plt.subplots(figsize=(10, 5))
	colors = ['#2ecc71' if r >= 0 else '#e74c3c' for r in avg_by_month.values]
	ax.bar(avg_by_month.index, avg_by_month.values, color=colors)
	ax.axhline(0, color='black', linewidth=0.8)
	ax.set_title(f'Average Monthly Return by Calendar Month — {ticker}')
	ax.set_ylabel('Average Monthly Return (%)')
	ax.set_xlabel('Month')
	plt.tight_layout()
	plt.savefig(f'data/charts/SeasonalReturns_{ticker.replace(".", "_")}.png', dpi=600)
	plt.show()


def MarketStressDashboard(tickerList: list = None):
	"""Compute cross-sectional stress metrics for the current trading universe."""
	if tickerList is None:
		tickerList = TickerLists.StarterList()
	forDate = GetLatestBDay()
	picker = StockPicker()
	picker.AlignToList(tickerList)
	multiverse = picker.GetHighestPriceMomentumMulti(currentDate=forDate, filterOptions=[(0, 500)])
	base = multiverse.get(0, pd.DataFrame())
	if base.empty or 'PC_1Year' not in base.columns:
		print("Insufficient data for stress dashboard.")
		return
	dispersion   = picker.compute_cross_sectional_dispersion(base)
	autocorr     = picker.compute_momentum_autocorr(base)
	downside_vol = picker.compute_downside_volatility(base)
	leadership_tilt, corr_6m_1m, corr_1y_1m = picker.compute_leadership_tilt(base)
	stress       = picker.compute_stress_index(dispersion, autocorr, downside_vol)
	print(f'\nMarket Stress Dashboard — {forDate}  ({len(base)} stocks in universe)')
	print(f"  Cross-sectional Dispersion  : {dispersion:.4f}  (higher = more spread of opportunity)")
	print(f"  Momentum Autocorrelation    : {autocorr:+.4f}  (positive = trends persist)")
	print(f"  Downside Volatility         : {downside_vol:.4f}  (lower = calmer market)")
	print(f"  Leadership Tilt             : {leadership_tilt:.4f}  (1 = stable 6M leaders, 0 = 1Y leaders)")
	print(f"  6M vs 1M Correlation        : {corr_6m_1m:+.4f}")
	print(f"  1Y vs 1M Correlation        : {corr_1y_1m:+.4f}")
	print(f"  Stress Index                : {stress:.4f}  (0 = calm, 1 = high stress; 0 if history unavailable)")
	avg_mom = pd.Series({
		'1M': base['PC_1Month'].mean() * 100,
		'3M': base['PC_3Month'].mean() * 100,
		'6M': base['PC_6Month'].mean() * 100,
		'1Y': base['PC_1Year'].mean()  * 100,
	})
	fig, ax = plt.subplots(figsize=(8, 5))
	colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in avg_mom.values]
	ax.bar(avg_mom.index, avg_mom.values, color=colors)
	ax.axhline(0, color='black', linewidth=0.8)
	ax.set_title(f'Universe Average Momentum — {forDate}')
	ax.set_ylabel('Average Return (%)')
	ax.set_xlabel('Time Horizon')
	plt.tight_layout()
	plt.savefig(f'data/charts/MarketStress_{str(forDate)}.png', dpi=600)
	plt.show()


def RollingDrawdown(ticker: str = '.INX'):
	"""Plot the rolling drawdown from all-time high over the full price history."""
	prices = PricingData(ticker)
	if not prices.LoadHistory():
		return
	hist = prices.GetPriceHistory(['Average']).dropna()
	rolling_max = hist['Average'].cummax()
	drawdown    = (hist['Average'] - rolling_max) / rolling_max * 100
	max_dd      = drawdown.min()
	max_dd_date = drawdown.idxmin()
	avg_dd      = drawdown[drawdown < 0].mean()
	pct_underwater = (drawdown < 0).mean() * 100
	print(f'\nRolling Drawdown Stats for {ticker}')
	print(f"  Max Drawdown       : {max_dd:.1f}%  ({max_dd_date.date()})")
	print(f"  Avg Drawdown       : {avg_dd:.1f}%  (when below peak)")
	print(f"  % Time Underwater  : {pct_underwater:.1f}%")
	dd_df = drawdown.to_frame(name='Drawdown %')
	plot = PlotHelper()
	plot.PlotDataFrame(dd_df, title=f'Rolling Drawdown from All-Time High — {ticker}', ylabel='Drawdown (%)', adjustScale=False)


def AdaptiveRegimeSnapshot(tickerList: list = None):
	"""Print the current adaptive market regime state and resulting stock picks."""
	if tickerList is None:
		tickerList = TickerLists.StarterList()
	forDate = GetLatestBDay()
	print(f'\nAdaptive Regime Snapshot — {forDate}')
	picker = StockPicker()
	picker.AlignToList(tickerList)
	picks = picker.GetAdaptiveConvexPicks(forDate)
	state = getattr(picker, '_last_market_state', None)
	if state:
		print(f"\n  Regime Summary      : {state.GetRegimeSummary()}")
		print(f"  Mode Label          : {state.GetModeLabel()}")
		print(f"  Geometry State      : {state.geometry_state}")
		print(f"  Expansion State     : {state.expansion_state}")
		print(f"  Leadership State    : {state.leadership_state}")
		print(f"  Convex Duration     : {state.convex_duration} days")
		print(f"  State Confidence    : {state.state_confidence:.2f}")
		print(f"  Dispersion          : {state.dispersion:.4f}  (p40={state.disp_p40:.4f}, p75={state.disp_p75:.4f})")
		print(f"  Momentum Autocorr   : {state.momentum_autocorr:+.4f}")
		print(f"  Downside Volatility : {state.downside_volatility:.4f}")
		print(f"  Stress Index        : {state.stress_index:.4f}")
		print(f"  Leadership Tilt     : {state.leadership_tilt:.4f}")
		print(f"  Execution Filters   : {state.GetExecutionFilters()}")
		print(f"  Rolling Window      : {state.GetRollingWindowSize()} days")
		flags = [name for name, val in [
			('REGIME_TENSION',    state.regime_tension_flag),
			('LEADERSHIP_BREAK',  state.leadership_break_flag),
			('PERSISTENCE_BREAK', state.persistence_break_flag),
		] if val]
		print(f"  Active Flags        : {', '.join(flags) if flags else 'None'}")
	else:
		print("  Market state unavailable — check SQL connectivity or use a larger ticker list.")
	print(f"\n  Top Picks:")
	print(picks.to_string())
	hist_df = getattr(picker, '_adaptive_history_df', None)
	if hist_df is not None and not hist_df.empty and len(hist_df) > 5:
		plot = PlotHelper()
		metrics = hist_df[['dispersion', 'momentum_autocorr', 'stress_index']].copy()
		plot.PlotDataFrame(metrics, title='Recent Regime Metrics History', adjustScale=False)


def CompareTickerAcrossMarketEvents(ticker: str = '.INX', events: list = None):
	"""Graph a single ticker across market crash/recovery windows aligned to trading-day 0.
	Useful for comparing how a stock behaved during distinct crash episodes.
	"""
	if events is None: return
	prices = PricingData(ticker)
	if not prices.LoadHistory():
		return
	hist = prices.GetPriceHistory(['Average'])
	plot = PlotHelper()
	safe = ticker.replace('.', '_')
	plot.PlotTickerAcrossMarketEvents(
		hist, events,
		normalize=True,
		title=f'{ticker} — Crash Episode Comparison (Normalized to 100)',
		fileName=f'data/charts/TimePeriodCompare_{safe}.png',
	)


def PastYearComparison(ticker: str = '.INX', use_eoy: bool = True):
	"""Compare ticker across the past 3 calendar years plus the current year.
	Each series is normalized to 100 at Jan 1 so year-over-year progress is
	directly comparable on a shared Jan-Dec x-axis.
	use_eoy=False (default): past years end at the same day-of-year as today,
	showing apples-to-apples progress at the same point in the calendar.
	use_eoy=True: past years run Jan 1 - Dec 31 (full year); current year is YTD.
	"""
	from pandas.tseries.offsets import BDay

	def _first_bday(d: date) -> date:
		return pd.bdate_range(start=pd.Timestamp(d), periods=1)[0].date()

	def _last_bday(d: date) -> date:
		ts = pd.Timestamp(d)
		if ts.weekday() >= 5:
			ts -= BDay(1)
		return ts.date()

	today    = date.today()
	anchor   = _last_bday(today)
	cur_year = today.year
	events   = []
	for yr in range(cur_year - 3, cur_year + 1):
		yr_start = _first_bday(date(yr, 1, 1))
		if yr == cur_year:
			yr_end = anchor
			label  = f'{yr} YTD'
		elif use_eoy:
			yr_end = _last_bday(date(yr, 12, 31))
			label  = str(yr)
		else:
			try:
				same_day = anchor.replace(year=yr)
			except ValueError:
				same_day = anchor.replace(year=yr, day=28)
			yr_end = _last_bday(same_day)
			label  = str(yr)
		events.append(MarketEvent(label, yr_start, yr_start, yr_end))

	prices = PricingData(ticker)
	if not prices.LoadHistory():
		return
	hist = prices.GetPriceHistory(['Average'])
	safe = ticker.replace('.', '_')
	plot = PlotHelper()
	plot.PlotTickerAcrossMarketEvents(
		hist, events,
		normalize=True,
		use_calendar_dates=True,
		title=f'{ticker} — Year-over-Year Comparison (Normalized to 100 at Jan 1)',
		fileName=f'data/charts/PastYearComparison_{safe}.png',
	)


def CompareTickersOverPeriod(tickers: list = None, startDate: str = '1/1/2020', endDate: str = None):
	"""Graph 2-3 tickers on the same date axis, each normalized to 100 at startDate.
	Useful for side-by-side relative performance of different names.
	"""
	if tickers is None: return
	if endDate is None:	endDate = str(GetLatestBDay())
	frames = {}
	for ticker in tickers:
		p = PricingData(ticker)
		if p.LoadHistory(requestedStartDate=ToDate(startDate)):
			frames[ticker] = p.GetPriceHistory(['Average'])
		else:
			print(f"  Could not load {ticker}, skipping.")
	if not frames:
		print("No data loaded.")
		return
	plot = PlotHelper()
	safe_names = '_'.join(t.replace('.', '_') for t in tickers)
	plot.PlotTickersOverPeriod(
		frames,
		start_date=startDate,
		end_date=endDate,
		normalize=True,
		title=f'Ticker Comparison {startDate} – {endDate} (Normalized to 100)',
		fileName=f'data/charts/TickerComparison_{safe_names}.png',
	)


if __name__ == '__main__':
	tickerList = TickerLists.StarterList()
	GraphHistoricalDrops()
	SeasonalMonthlyReturns('.INX')
	MarketStressDashboard(tickerList)
	RollingDrawdown('.INX')
	AdaptiveRegimeSnapshot(tickerList)
	print(f"Loading {len(tickerList)} stocks..")
	DownloadAndSaveStocks(tickerList)
	print(f"Saving stats to _dataFolderhistoricalPrices")
	DownloadAndSaveStocksWithStats(tickerList)
	PlotAnnualPerformance('TSLA')
	PlotAnnualPerformance('VIGRX')
	ShowPicks(tickerList)
	GraphHistoricalDrops()
	NormalizedComparison(['NVDA', 'WDC', 'STX', 'MU'], startDate='1/1/2010')
	for year in range(1980,2020,2022): GraphTimePeriod(ticker = '.INX', endDate = '1/3/' + str(year), days=120)
	for year in range(2019,2025,2026): GraphTimePeriod(ticker = '.INX', endDate = '5/15/' + str(year), days=120)
	CrashRecoverySpeed('.INX')
	CompareTickerAcrossMarketEvents(ticker='.INX', events=MarketCrashes.GetAll())
	CompareTickersOverPeriod(tickers=['NVDA', 'WDC', 'SNDK', 'STX'], startDate='1/1/2020', endDate=None)
	PastYearComparison('.INX')
