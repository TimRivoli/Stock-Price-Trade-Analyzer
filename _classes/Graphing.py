import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import pyplot as plt
from pandas.tseries.offsets import BDay
from typing import List, Optional
from datetime import datetime
from _classes.Utility import *

def PlotSetDefaults():
	#params = {'legend.fontsize': 4, 'axes.labelsize': 4,'axes.titlesize':4,'xtick.labelsize':4,'ytick.labelsize':4}
	#plt.rcParams.update(params)
	plt.rcParams['font.size'] = 5
	plt.rcParams['figure.dpi'] = 600

def PlotScalerDateAdjust(minDate:datetime, maxDate:datetime, ax):
	if type(minDate)==str:
		daysInGraph = DateDiffDays(minDate,maxDate)
	else:
		daysInGraph = (maxDate-minDate).days
	if daysInGraph >= 365*3:
		majorlocator =  mdates.YearLocator()
		minorLocator = mdates.MonthLocator()
		majorFormatter = mdates.DateFormatter('%m/%d/%Y')
	elif daysInGraph >= 365:
		majorlocator =  mdates.MonthLocator()
		minorLocator = mdates.WeekdayLocator()
		majorFormatter = mdates.DateFormatter('%m/%d/%Y')
	elif daysInGraph < 90:
		majorlocator =  mdates.DayLocator()
		minorLocator = mdates.DayLocator()
		majorFormatter =  mdates.DateFormatter('%m/%d/%Y')
	else:
		majorlocator =  mdates.WeekdayLocator()
		minorLocator = mdates.DayLocator()
		majorFormatter =  mdates.DateFormatter('%m/%d/%Y')
	ax.xaxis.set_major_locator(majorlocator)
	ax.xaxis.set_major_formatter(majorFormatter)
	ax.xaxis.set_minor_locator(minorLocator)
	#ax.xaxis.set_minor_formatter(daysFmt)
	ax.set_xlim(minDate, maxDate)

class PlotHelper:
	def PlotDataFrame(self, df: pd.DataFrame, title: str = '', xlabel: str = '', ylabel: str = '', adjustScale: bool = True, fileName: str = '', dpi: int = 600, show: bool = True, legend: bool = True, figsize: tuple = (12, 6)):
		if df is None or df.empty:
			print(" Graphing received empty source data")
			return
		fig, ax = plt.subplots(figsize=figsize)
		df.plot(ax=ax, title=title)
		if xlabel:
			ax.set_xlabel(xlabel)
		if ylabel:
			ax.set_ylabel(ylabel)
		if legend:
			ax.legend()
		if adjustScale:
			PlotScalerDateAdjust(df.index.min(), df.index.max(), ax)
		if fileName:
			if not fileName.lower().endswith('.png'):
				fileName += '.png'
			plt.savefig(fileName, dpi=dpi)
			print(f" Saved to {fileName}")
		if show and not fileName:
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

	@staticmethod
	def PlotTimeSeries(df, fields: List[str], start_date: datetime, end_date: datetime, title: str, xlabel: str = "Date", ylabel: str = "Price", colors: Optional[List[str]] = None, linewidth: float = 0.75, rotate_xticks: int = 70, dpi: int = 600,save: bool = False, save_path: Optional[str] = None, show: bool = True):
		if df.empty:
			print(" Graphing received empty source data")
			return
		PlotSetDefaults()
		fig, ax = plt.subplots()
		df.loc[start_date:end_date, fields].plot(ax=ax, title=title, linewidth=linewidth, color=colors )
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.tick_params(axis="x", rotation=rotate_xticks)

		ax.grid(visible=True, which="major", color="black", linestyle="solid", linewidth=0.5)
		ax.grid(visible=True, which="minor", color="0.65", linestyle="solid", linewidth=0.1)
		PlotScalerDateAdjust(start_date, end_date, ax)
		if save and save_path:
			plt.savefig(save_path, dpi=dpi)
			print(f" Saved to {save_path}")

		if show: plt.show()
		plt.close(fig)
