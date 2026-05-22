import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.tseries.offsets import BDay
from typing import List, Optional
from datetime import datetime
from _classes.Utility import *

def PlotSetDefaults():
	for style in ('seaborn-v0_8-whitegrid', 'seaborn-whitegrid'):
		try:
			plt.style.use(style)
			break
		except OSError:
			continue
	plt.rcParams.update({
		'font.size':          9,
		'axes.titlesize':     11,
		'axes.titleweight':   'bold',
		'axes.labelsize':     9,
		'xtick.labelsize':    8,
		'ytick.labelsize':    8,
		'legend.fontsize':    8,
		'legend.framealpha':  0.7,
		'lines.linewidth':    0.9,
		'figure.dpi':         100,
		'axes.spines.top':    False,
		'axes.spines.right':  False,
	})

def PlotScalerDateAdjust(minDate: datetime, maxDate: datetime, ax):
	if type(minDate) == str:
		daysInGraph = DateDiffDays(minDate, maxDate)
	else:
		daysInGraph = (maxDate - minDate).days
	if daysInGraph >= 365 * 3:
		ax.xaxis.set_major_locator(mdates.YearLocator())
		ax.xaxis.set_minor_locator(mdates.MonthLocator())
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
	elif daysInGraph >= 365:
		ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
		ax.xaxis.set_minor_locator(mdates.MonthLocator())
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
	elif daysInGraph >= 90:
		ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
		ax.xaxis.set_minor_locator(mdates.DayLocator())
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
	else:
		ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
		ax.xaxis.set_minor_locator(mdates.DayLocator())
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
	ax.set_xlim(minDate, maxDate)

class PlotHelper:
	def PlotDataFrame(self, df: pd.DataFrame, title: str = '', xlabel: str = '', ylabel: str = '', adjustScale: bool = True, fileName: str = '', dpi: int = 600, show: bool = True, legend: bool = True, figsize: tuple = (12, 6)):
		if df is None or df.empty:
			print(" Graphing received empty source data")
			return
		PlotSetDefaults()
		fig, ax = plt.subplots(figsize=figsize)
		df.plot(ax=ax, linewidth=0.9, legend=False)
		if title:
			ax.set_title(title, pad=8)
		if xlabel:
			ax.set_xlabel(xlabel)
		if ylabel:
			ax.set_ylabel(ylabel)
		if legend:
			ax.legend(loc='best', framealpha=0.7)
		if adjustScale:
			PlotScalerDateAdjust(df.index.min(), df.index.max(), ax)
			fig.autofmt_xdate(rotation=45, ha='right')
		plt.tight_layout(pad=1.2)
		if fileName:
			if not fileName.lower().endswith('.png'):
				fileName += '.png'
			plt.savefig(fileName, dpi=dpi)
			print(f" Saved to {fileName}")
		if show:
			plt.show()
		plt.close(fig)

	def PlotDataFrameDateRange(self, df: pd.DataFrame, endDate: datetime = None, historyDays: int = 90, title: str = '', xlabel: str = '', ylabel: str = '', fileName: str = '', dpi: int = 600, show: bool = True, legend: bool = True, figsize: tuple = (12, 6)):
		if df is None or df.shape[0] <= 10:
			print(" Graphing received insufficient data to plot")
			return
		if endDate is None:
			endDate = df.index[-1]
		endDate = ToDateTime(endDate)
		startDate = endDate - BDay(historyDays)
		df_filtered = df[(df.index >= startDate) & (df.index <= endDate)]
		self.PlotDataFrame(df=df_filtered, title=title, xlabel=xlabel, ylabel=ylabel, adjustScale=True, fileName=fileName, dpi=dpi, show=show, legend=legend, figsize=figsize)

	def PlotTickerAcrossTimePeriods(self,df: pd.DataFrame,periods: list,price_col: str = 'Average',normalize: bool = True,title: str = '',ylabel: str = '',fileName: str = '',dpi: int = 600,show: bool = True):
		"""Overlay 2-3 time windows of a single ticker aligned to trading-day 0.
		periods: list of (start_date, end_date, label) tuples.
		normalize=True indexes each period to 100 at its first trading day so
		shapes are comparable regardless of absolute price level.
		"""
		if df is None or df.empty:
			print(" Graphing received empty source data")
			return
		PlotSetDefaults()
		combined = {}
		for start, end, label in periods:
			slc = df.loc[pd.Timestamp(start):pd.Timestamp(end), price_col].dropna()
			if slc.empty:
				continue
			if normalize:
				base = slc.iloc[0]
				if base > 0:
					slc = slc / base * 100
			combined[label] = slc.reset_index(drop=True)
		if not combined:
			print(" No data found for any period.")
			return
		result = pd.DataFrame(combined)
		fig, ax = plt.subplots(figsize=(12, 6))
		result.plot(ax=ax, linewidth=0.9)
		if title:
			ax.set_title(title, pad=8)
		ax.set_xlabel('Trading Days from Period Start')
		ax.set_ylabel(ylabel or ('Value (Base=100)' if normalize else price_col))
		ax.legend(loc='best', framealpha=0.7)
		plt.tight_layout(pad=1.2)
		if fileName:
			if not fileName.lower().endswith('.png'):
				fileName += '.png'
			plt.savefig(fileName, dpi=dpi)
			print(f" Saved to {fileName}")
		if show:
			plt.show()
		plt.close(fig)

	def PlotTickersOverPeriod(self,frames: dict,start_date=None,end_date=None,price_col: str = 'Average',normalize: bool = True,title: str = '',ylabel: str = '',fileName: str = '',dpi: int = 600,show: bool = True):
		"""Overlay 2-3 tickers over a shared calendar date range.
		frames: dict of {label: DataFrame} where each DataFrame has a DatetimeIndex.
		normalize=True indexes every series to 100 at start_date so relative
		performance is directly comparable even when prices differ in magnitude.
		"""
		if not frames:
			print(" No frames provided.")
			return
		PlotSetDefaults()
		combined = {}
		for label, df in frames.items():
			if df is None or df.empty:
				continue
			series = df[price_col].dropna() if price_col in df.columns else df.iloc[:, 0].dropna()
			if start_date is not None:
				series = series[series.index >= pd.Timestamp(start_date)]
			if end_date is not None:
				series = series[series.index <= pd.Timestamp(end_date)]
			if series.empty:
				continue
			if normalize:
				base = series.iloc[0]
				if base > 0:
					series = series / base * 100
			combined[label] = series
		if not combined:
			print(" No data found for any ticker.")
			return
		result = pd.DataFrame(combined).ffill()
		fig, ax = plt.subplots(figsize=(12, 6))
		result.plot(ax=ax, linewidth=0.9)
		if title:
			ax.set_title(title, pad=8)
		ax.set_xlabel('Date')
		ax.set_ylabel(ylabel or ('Value (Base=100)' if normalize else price_col))
		ax.legend(loc='best', framealpha=0.7)
		PlotScalerDateAdjust(result.index.min(), result.index.max(), ax)
		fig.autofmt_xdate(rotation=45, ha='right')
		plt.tight_layout(pad=1.2)
		if fileName:
			if not fileName.lower().endswith('.png'):
				fileName += '.png'
			plt.savefig(fileName, dpi=dpi)
			print(f" Saved to {fileName}")
		if show:
			plt.show()
		plt.close(fig)

	@staticmethod
	def PlotTimeSeries(df, fields: List[str], start_date: datetime, end_date: datetime, title: str, xlabel: str = "Date", ylabel: str = "Price", colors: Optional[List[str]] = None, linewidth: float = 0.9, rotate_xticks: int = 45, dpi: int = 600, save: bool = False, save_path: Optional[str] = None, show: bool = True):
		if df.empty:
			print(" Graphing received empty source data")
			return
		PlotSetDefaults()
		fig, ax = plt.subplots()
		df.loc[start_date:end_date, fields].plot(ax=ax, linewidth=linewidth, color=colors, legend=False)
		if title:
			ax.set_title(title, pad=8)
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.legend(loc='best', framealpha=0.7)
		PlotScalerDateAdjust(start_date, end_date, ax)
		fig.autofmt_xdate(rotation=rotate_xticks, ha='right')
		plt.tight_layout(pad=1.2)
		if save and save_path:
			plt.savefig(save_path, dpi=dpi)
			print(f" Saved to {save_path}")
		if show:
			plt.show()
		plt.close(fig)
