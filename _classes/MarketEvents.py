from dataclasses import dataclass
from datetime import date
from typing import List, Optional


@dataclass
class MarketEvent:
	Label:        str
	StartDate:    date           # Pre-crash peak
	LowDate:      date           # Market trough (maximum drawdown)
	RecoveryDate: Optional[date] # First close at or above the pre-crash peak; None if unknown

	@property
	def CrashDays(self) -> int:
		return (self.LowDate - self.StartDate).days

	@property
	def RecoveryDays(self) -> Optional[int]:
		"""Calendar days from trough to full recovery."""
		if self.RecoveryDate is None:
			return None
		return (self.RecoveryDate - self.LowDate).days

	@property
	def TotalDays(self) -> Optional[int]:
		"""Calendar days from peak to full recovery."""
		if self.RecoveryDate is None:
			return None
		return (self.RecoveryDate - self.StartDate).days

	def __str__(self) -> str:
		rec = str(self.RecoveryDate) if self.RecoveryDate else 'still recovering'
		return (f"{self.Label:<32}  peak={self.StartDate}  trough={self.LowDate}"
				f"  recovery={rec}  crash={self.CrashDays}d  total={self.TotalDays or '?'}d")


class MarketCrashes:
	"""Historical S&P 500 significant drawdown events.
	Dates reference the S&P 500 composite (.INX / SPX).
	RecoveryDate is the first trading day the index closed back at or above StartDate's level.
	"""

	_events: List[MarketEvent] = [
		# Great Depression — DJIA proxy; S&P composite recovered ~same timeframe
		MarketEvent('Great Depression 1929',     date(1929,  9,  3), date(1932,  7,  8), date(1954, 11, 23)),
		# 1973–74 oil embargo bear market
		MarketEvent('Oil Embargo Bear 1973',     date(1973,  1, 11), date(1974, 10,  3), date(1980,  7, 17)),
		# Gulf War recession
		MarketEvent('Gulf War Recession 1990',   date(1990,  7, 16), date(1990, 10, 11), date(1991,  2, 13)),
		# Black Monday crash
		MarketEvent('Black Monday 1987',         date(1987,  8, 25), date(1987, 12,  4), date(1989,  7, 26)),
		# Dot-com bust
		MarketEvent('Dot-Com Bust 2000',         date(2000,  3, 24), date(2002, 10,  9), date(2007,  5, 30)),
		# Global financial crisis
		MarketEvent('Financial Crisis 2008',     date(2007, 10,  9), date(2009,  3,  9), date(2013,  3, 28)),
		# COVID-19 crash (fastest recovery on record)
		MarketEvent('COVID-19 2020',             date(2020,  2, 19), date(2020,  3, 23), date(2020,  8, 18)),
		# Inflation / Fed rate-hike bear market
		MarketEvent('Inflation Bear Market 2022',date(2022,  1,  3), date(2022, 10, 12), date(2024,  1, 19)),
	]

	@classmethod
	def GetAll(cls) -> List[MarketEvent]:
		return list(cls._events)

	@classmethod
	def GetByLabel(cls, label: str) -> Optional[MarketEvent]:
		"""Case-insensitive substring match on Label."""
		label_lower = label.lower()
		for event in cls._events:
			if label_lower in event.Label.lower():
				return event
		return None

	@classmethod
	def GetContaining(cls, d: date) -> Optional[MarketEvent]:
		"""Return the first crash whose [StartDate, RecoveryDate] range contains d."""
		for event in cls._events:
			end = event.RecoveryDate or date.today()
			if event.StartDate <= d <= end:
				return event
		return None

	@classmethod
	def GetTroughWindow(cls, d: date, days_before: int = 90, days_after: int = 365) -> Optional[MarketEvent]:
		"""Return a crash whose trough falls within a window around d."""
		from datetime import timedelta
		lo = d - timedelta(days=days_before)
		hi = d + timedelta(days=days_after)
		for event in cls._events:
			if lo <= event.LowDate <= hi:
				return event
		return None

	@classmethod
	def Labels(cls) -> List[str]:
		return [e.Label for e in cls._events]

	@classmethod
	def Summary(cls) -> None:
		print(f"\n{'Label':<32}  {'Peak':<12}  {'Trough':<12}  {'Recovery':<12}  {'Crash(d)':>9}  {'Total(d)':>9}")
		print('-' * 105)
		for e in cls._events:
			rec  = str(e.RecoveryDate) if e.RecoveryDate else 'unknown     '
			tot  = str(e.TotalDays)    if e.TotalDays    else '?'
			print(f"{e.Label:<32}  {str(e.StartDate):<12}  {str(e.LowDate):<12}  {rec:<12}  {e.CrashDays:>9}  {tot:>9}")


def GetPastYears(years: int, use_eoy: bool = False) -> List[MarketEvent]:
	"""Return one MarketEvent per year for the past `years` years.
	Each event spans exactly one calendar year. StartDate and RecoveryDate are
	business days. LowDate equals StartDate (no crash concept). Events are
	ordered oldest-first.
	use_eoy=False (default): anchor is the most recent business day on or before today.
	use_eoy=True: anchor is the last business day of the previous calendar year.
	"""
	from pandas.tseries.offsets import BDay
	import pandas as pd

	def _most_recent_bday(d: date) -> date:
		ts = pd.Timestamp(d)
		if ts.weekday() >= 5:
			ts = ts - BDay(1)
		return ts.date()

	def _sub_years(d: date, n: int) -> date:
		try:
			return d.replace(year=d.year - n)
		except ValueError:  # Feb 29 in non-leap year
			return d.replace(year=d.year - n, day=28)

	if use_eoy:
		anchor = _most_recent_bday(date(date.today().year - 1, 12, 31))
	else:
		anchor = _most_recent_bday(date.today())

	result = []
	for i in range(years, 0, -1):
		end   = _most_recent_bday(_sub_years(anchor, i - 1))
		start = _most_recent_bday(_sub_years(anchor, i))
		label = f"{start.strftime('%b %d, %Y')} – {end.strftime('%b %d, %Y')}"
		result.append(MarketEvent(label, start, start, end))
	return result
