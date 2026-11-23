#Run settings and result records (11/21/26: moved from Trading.py so Selection and the trading engine share them without
#importing each other). TradeModelParams: a backtest's settings, defaults = the standard test settings, Production() the
#live ones, model-name rules (AddModelNameModifiers). TradeModelPerformanceMetrics: the TradeModelComparisons metrics.
#Trading.py re-exports all three, so "from _classes.Trading import TradeModelParams" still works.
import re
import numpy as np, pandas as pd
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime

class SQLFriendlyMixin:
	def to_sql_dict(self):
		"""Converts dataclass to a flat dict with SQL-compatible types."""
		fields = getattr(self, 'export_fields', [k for k in self.__dict__ if not k.startswith('_')])       
		final_data = {}
		for field_name in fields:
			val = getattr(self, field_name, None)           
			if isinstance(val, (dict, list)):
				continue
			if isinstance(val, (pd.Timestamp, datetime)):
				final_data[field_name] = val.to_pydatetime() if hasattr(val, 'to_pydatetime') else val
			elif isinstance(val, (np.float64, np.float32)):
				final_data[field_name] = float(val)
			elif isinstance(val, (np.int64, np.int32)):
				final_data[field_name] = int(val)
			elif isinstance(val, bool):
				final_data[field_name] = 1 if val else 0
			else:
				final_data[field_name] = val                
		return final_data
		
@dataclass
class TradeModelParams(SQLFriendlyMixin):
	_startDate: pd.Timestamp = field(init=False, repr=False)
	modelName: str = ''
	#11/19/26: the defaults are the standard test settings (1995 + 27 years, FF, point-in-time lists, 9 stocks, re-pick every
	#5 days, 28-day pick window, no point-value allocation; AdaptiveConvex FS2 on 4.8 states). Callers set only what differs.
	#Production() gives the stored-picks / live settings; the Run*/Rebuild* methods below hold the per-model rules.
	init_startDate: str = '1/1/1995'
	durationInYears: int = 27
	portfolioSize: int = 100000
	reEvaluationInterval: int = 5
	longHistory: int = 365
	shortHistory: int = 90
	
	# Filters & Constraints
	stockCount: int = 9
	filterOption: int = 5
	filterByFundamentals: bool = True
	minPercentGain: float = 0.05
	SP500Only: bool = False
	marketCapMin: int = 0
	marketCapMax: int = 0
	riskFilter: bool = True		#sp_GetTickerList risk pre-filter (with filterByFundamentals). Model name gets _NoRF when off
	pointInTimeUniverse: bool = True	#Selection 6.0: sp_GetTickerList @PointInTime - month M's list from month M-1 figures only (month-end momentum, prior month's volume/price), no look-ahead. Model name _PIT. 6.2: the default and the canonical universe
	damageFilter: bool = False	#sp_GetTickerList @DamageFilter: drop stocks failing base_filter's damage scores at the prior month end (with filterByFundamentals). Model name gets _DMG
	adaptiveFilterSet: int = 2		#Selection 5.4, AdaptiveConvex only: execution filter lists (1 = 4.6, 2 = FS2: F10 when a mania breaks, F6/9v in convex expansion/contraction). Model name _FS2
	riskOffExposure: float = 1.0	#Selection 5.6, blended/adaptive: share invested while .INX is below its 200-day average (2% band); 1.0 = off. Model name _RO<pct>
	riskOffSignal: str = 'index'	#Selection 5.9, with riskOffExposure: 'index' = .INX 200-day switch; 'leaders' = momentum-leader index (LeaderIndexDaily) below its 100-day average. Model name _RO<pct>L
	riskOffSmooth: int = 0			#Selection 6.12, with riskOffExposure: n > 0 = smoothed switch - invested share 1 - (1 - riskOffExposure) x share of the last n trading days risk-off (rounded to 5%), set at each re-pick, no re-pick on a flip. 0 = the instant switch. Model name _RO<pct>S<n>
	volTarget: float = 0.0			#Selection 6.8, blended/adaptive: annualized volatility target - on each re-pick the stocks are scaled to min(1, volTarget / realized volatility of the fully invested portfolio over volLookback trading days), rounded to 10%, rest in cash; on top of riskOffExposure. 0 = off. Model name _VT<pct>
	volLookback: int = 21			#Selection 6.8, with volTarget: trading days of daily returns the realized volatility is measured over. Model name _VT<pct>L<n> when not 21
	volTargetOff: str = ''			#Selection 6.11, with volTarget: comma list of market state labels (Selection.VOL_TARGET_OFF_LABELS: blowoff / geometry) in which volTarget is not applied, read each day from the stored MarketStates (marketStateVersion) of the prior day; a change re-picks. '' = always applied. Model name _VT<pct>X(<labels>)
	profitScreen: float = 0.0		#Selection 6.9, blended/adaptive: each month drop this share of the universe's non-financial stocks with the lowest gross profit / assets (EDGAR, figures filed before the month; Novy-Marx 2013). Financials and stocks without figures stay. 0 = off. Model name _GP<pct>
	strictData: str = ''	#Selection 6.10, blended/adaptive: strict testing. Comma list of data the universe must have, with or without the filter that uses it (Selection.STRICT_DATA_CODES; 'gp' = gross profit / assets). '' = production: each filter's own missing-data rule. Model name _SD(<codes>)
	postManiaDays: int = 0			#Selection 5.9, AdaptiveConvex only: trading days after the last mania day treated as POST_MANIA (stock floor); 0 = the state's own (126). Model name _PMD<n>
	adaptiveStockCount: int = 0		#Selection 5.6, AdaptiveConvex only: fixed stock count instead of the conviction count; 0 = off. Model name _S<n>
	marketStateVersion: float = 4.8	#Selection 5.6, AdaptiveConvex only: MarketStates version read (4.7 = two-phase build, _MS47; 6.1: 4.8 = on the point-in-time lists, _MS48, the default from 6.2). filterOption 97 jobs build its raw rows
	dmaWeight: bool = False			#Selection 5.5, blended/adaptive: the vote blend's 200DMA weighting really applied (it never was). Model name _DMA
	blendFilters: str = ''			#Selection 5.3, PM_Blended only: custom blend "10:3,9v:3,4:2,6:1" (Selection.ParseBlend); '' = DEFAULT_BLEND
	volAdjustedRank: bool = False	#Selection 5.1: momentum filters rank on PC_x / Volatility_1Year instead of raw PC_x. Model name gets _VA
	selectionVersion: str = ''		#set by the worker from Selection.SELECTION_VERSION when the run starts (not taken from the queued job); stored in TradeModelComparisons.SelectionVersion
	rebuildBlendedPicks: bool = False	#PM_Blended only: write each day's raw blended picks to PicksBlendedDaily while running (needs useStoredPicks off, pickHistoryWindow > 0, default blend, canonical universe). Overwrites the stored picks
	rebuildMarketStates: bool = False	#AdaptiveConvex only: regenerate the MarketStates rows for every date the run covers (only with the canonical universe: FF + risk filter, no SP500Only/market cap limits)

	# System & Execution
	pickHistoryWindow: int = 28	#28 days is optimal (3/16/26 testing); AdaptiveConvex uses -1 (its own window)
	shopBuyPercent: float = 0.0
	shopSellPercent: float = 0.0
	trimProfitsPercent: float = 0.0
	allocateByPointValue: bool = False
	rateLimitTransactions: bool = False
	rateLimitPct: float = 0.20		#Selection 6.13, with rateLimitTransactions: share of total value sold and bought per day (CONSTANTS.RATE_LIMITING_MAX_TURNOVER_PCT before 6.13). Model name _RL<pct> when not 20%
	swingFraction: float = 0.0		#Selection 6.15: trade around the core (TradingModel.SwingTrade) - share of a held stock sold after a spike and bought back lower. 0 = off. Model name _SW<pct>
	swingDays: int = 5				#6.15: trading days the buy-back limit waits before buying at market (_SW<pct>D<n> when not 5)
	swingSigma: float = 1.0			#6.15: buy-back limit = sale price x (1 - swingSigma x 20-day daily vol x sqrt swingDays) (_SW<pct>K<x> when not 1)
	swingTriggerZ: float = 1.5		#6.17: trim when the prior close is above this 40-day channel z (_SW..Z<x> when not 1.5)
	swingTriggerSpike: float = 1.5	#6.17: or the 5-day rise is above this x 20-day vol x sqrt 5 (_SW..J<x> when not 1.5)
	swingRiskOffOnly: bool = False	#6.17: trim only while the .INX 200-day switch is risk-off (_SW..R)
	pullFraction: float = 0.0		#Selection 6.17: buy the pullbacks (TradingModel.SwingTrade) - share of a held stock bought extra after a drop, funded from the other holdings, sold back higher. 0 = off. Model name _PB<pct>
	pullDays: int = 10				#6.17: trading days the extra lot's limit sell waits before selling at market (_PB..D<n> when not 10)
	pullSigma: float = 0.5			#6.17: limit sell = fill x (1 + pullSigma x 20-day vol x sqrt pullDays) (_PB..K<x> when not 0.5)
	pullTriggerZ: float = 2.0		#6.17: buy when the prior close is below -this 40-day channel z (_PB..Z<x> when not 2)
	pullTriggerDrop: float = 2.0	#6.17: or the 5-day fall is beyond this x 20-day vol x sqrt 5 (_PB..J<x> when not 2)
	pullSkipRiskOff: bool = True	#6.17: no pullback buys while the .INX switch is risk-off (_PB..A = buy in risk-off too)
	pullEntrySigma: float = 0.0		#6.19: 0 = buy at the open (market); > 0 = limit buy at the prior close x (1 - this x 20-day vol), live 2 days, else no trade; funded once it fills (_PB..E<x>)
	pullFundStretched: bool = False	#6.19: fund the pullback from the other holdings with the highest channel z first (most stretched), not pro rata (_PB..S)
	saveTradeHistory: bool = False	#11/21/26: TradeModel_Trades rows (one per trade, ~20k a run) only when asked; queue --save-trades. Was True and also gated the daily value and results
	useStoredPicks: bool = True	#PM_Blended only: read stored picks from PicksBlendedDaily (_DB, queue --db). Was useDatabase (renamed 11/18/26; column TradeModelComparisons.useStoredPicks)
	saveResults: bool = True	#11/21/26: TradeModel_DailyValue + the TradeModelComparisons row at CloseModel (was unused; default False)
	verbose: bool = False
	
	batchName: str = ''
	processing_minutes: int = 0	
	export_fields = [
		'modelName', 'startDate', 'endDate', 'durationInYears', 'stockCount', 'reEvaluationInterval', 'SP500Only', 	'longHistory', 'shortHistory', 'minPercentGain', 'startValue', 'endValue',
		'pickHistoryWindow', 'shopBuyPercent', 'shopSellPercent', 'trimProfitsPercent', 'allocateByPointValue', 'filterOption', 'useStoredPicks','filterByFundamentals', 'rateLimitTransactions','marketCapMin', 'marketCapMax', 'processing_minutes', 'batchName', 'selectionVersion'
	]		
	_NAME_BASE_FILTER_SET = 1		#model names mark settings that differ from the original runs (_FS2, _MS47), not from the defaults
	_NAME_BASE_STATE_VERSION = 4.6
	PIT_STATE_VERSION = 4.8			#Selection 6.1: MarketStates built on the point-in-time lists (Selection.MARKET_STATE_VERSION_PIT)

	def __post_init__(self):		
		self.startDate = pd.to_datetime(self.init_startDate)	 # Trigger the setters to convert types immediately on startup
	@property
	def startDate(self) -> pd.Timestamp:
		return self._startDate
	@startDate.setter
	def startDate(self, value):
		self._startDate = pd.to_datetime(value)	
		self.init_startDate = self._startDate
	@property
	def endDate(self) -> pd.Timestamp: return self.startDate + pd.DateOffset(years=self.durationInYears) 
	def AddModelNameModifiers(self):
		mn = self.modelName
		if mn == '':
			if self.filterOption == 99: mn = f"PM_Blended"
			elif self.filterOption == 98: mn = f"AdaptiveConvex"
			else: mn = f"PM_filter{self.filterOption}"
		if self.filterOption == 99 and self.useStoredPicks: mn += "_DB" 
		if self.filterByFundamentals and not "_FF" in mn: mn += "_FF"
		if not "_HW" in mn:
			if self.pickHistoryWindow > 0: mn += f"_HW{self.pickHistoryWindow}"
			if self.pickHistoryWindow < 0: mn += f"_HWAdaptive"
		if self.filterByFundamentals and not self.riskFilter and not "_NoRF" in mn: mn += "_NoRF"
		if self.filterByFundamentals and self.damageFilter and not "_DMG" in mn: mn += "_DMG"
		if self.pointInTimeUniverse and not "_PIT" in mn: mn += "_PIT"
		if self.volAdjustedRank and not "_VA" in mn: mn += "_VA"
		if self.filterOption == 98 and self.adaptiveFilterSet != self._NAME_BASE_FILTER_SET and not "_FS" in mn: mn += f"_FS{self.adaptiveFilterSet}"
		if self.dmaWeight and not "_DMA" in mn: mn += "_DMA"
		if self.filterOption == 98 and self.adaptiveStockCount > 0 and not "_S" + str(self.adaptiveStockCount) in mn: mn += f"_S{self.adaptiveStockCount}"
		if self.filterOption == 98 and self.marketStateVersion != self._NAME_BASE_STATE_VERSION and not "_MS" in mn: mn += "_MS" + str(self.marketStateVersion).replace(".", "")
		if self.filterOption == 98 and self.postManiaDays > 0 and not "_PMD" in mn: mn += f"_PMD{self.postManiaDays}"
		if self.riskOffExposure < 1.0 and not "_RO" in mn: mn += f"_RO{int(round(self.riskOffExposure * 100))}" + ("L" if self.riskOffSignal == 'leaders' else "") + (f"S{self.riskOffSmooth}" if self.riskOffSmooth > 0 else "")
		if self.profitScreen > 0 and not re.search(r"_GP\d", mn): mn += f"_GP{int(round(self.profitScreen * 100))}"
		if self.strictData and not "_SD(" in mn: mn += f"_SD({'.'.join(sorted(c.strip().lower() for c in self.strictData.split(',') if c.strip()))})"
		if self.volTarget > 0 and not "_VT" in mn: mn += f"_VT{int(round(self.volTarget * 100))}" + (f"L{self.volLookback}" if self.volLookback != 21 else "") + (f"X({'.'.join(sorted(c.strip().upper() for c in self.volTargetOff.split(',') if c.strip()))})" if self.volTargetOff else "")
		if self.SP500Only and not "_SP500" in mn: mn += "_SP500"
		if self.allocateByPointValue and not "_PVA" in mn: mn += "_PVA"
		if self.rateLimitTransactions and not "_RL" in mn:  mn += "_RL" + (f"{int(round(self.rateLimitPct * 100))}" if abs(self.rateLimitPct - 0.20) > 1e-9 else "")
		if self.swingFraction > 0 and not "_SW" in mn: mn += f"_SW{int(round(self.swingFraction * 100))}" + (f"D{self.swingDays}" if self.swingDays != 5 else "") + (f"K{self.swingSigma:g}" if abs(self.swingSigma - 1.0) > 1e-9 else "") + (f"Z{self.swingTriggerZ:g}" if abs(self.swingTriggerZ - 1.5) > 1e-9 else "") + (f"J{self.swingTriggerSpike:g}" if abs(self.swingTriggerSpike - 1.5) > 1e-9 else "") + ("R" if self.swingRiskOffOnly else "")
		if self.pullFraction > 0 and not "_PB" in mn: mn += f"_PB{int(round(self.pullFraction * 100))}" + (f"D{self.pullDays}" if self.pullDays != 10 else "") + (f"K{self.pullSigma:g}" if abs(self.pullSigma - 0.5) > 1e-9 else "") + (f"Z{self.pullTriggerZ:g}" if abs(self.pullTriggerZ - 2.0) > 1e-9 else "") + (f"J{self.pullTriggerDrop:g}" if abs(self.pullTriggerDrop - 2.0) > 1e-9 else "") + ("" if self.pullSkipRiskOff else "A") + (f"E{self.pullEntrySigma:g}" if self.pullEntrySigma > 0 else "") + ("S" if self.pullFundStretched else "")
		if self.trimProfitsPercent > 0 and not "_TP" in mn:  mn += f"_TP{self.trimProfitsPercent}"
		if self.shopBuyPercent > 0 and not "_SB" in mn:  mn += f"_SB{self.shopBuyPercent}"
		if self.shopSellPercent > 0 and not "_SS" in mn:  mn += f"_SS{self.shopSellPercent}"
		self.modelName = mn

	@classmethod
	def Production(cls, model: str) -> 'TradeModelParams':
		"""The live settings: what PicksAdaptiveDaily / PicksBlendedDaily are built with (StockPicker.UpdateStoredPicks)
		and ModelPastFewYears runs. adaptive = FS2 on 4.8 states, conviction stock count, 50% invested while .INX is below
		its 200-day average (best Sharpe, 11/18/26 tests), smoothed over 10 trading days (6.14, riskOffSmooth 10: 11/21/26
		tests - level with the instant switch 1980-2024 and 1995-2021, +6 CAGR 2022-26, cash share moves in 5% steps); blended = DEFAULT_BLEND (1.3.9.9.6), 28-day window. Both on the
		point-in-time lists (6.2)."""
		p = cls()
		if model == 'adaptive':
			p.filterOption, p.riskOffExposure, p.riskOffSmooth = 98, 0.5, 10
		elif model == 'blended':
			p.filterOption = 99
		else:
			raise ValueError(f"TradeModelParams.Production: unknown model {model!r} (adaptive / blended)")
		p.useStoredPicks = False
		return p

	@classmethod
	def SwingFlags(cls) -> 'TradeModelParams':
		"""11/22/26: the rules behind the trade flags in PicksAdaptiveDaily / PicksBlendedDaily (Signal, TrimBuyBackPct, PullSellPct;
		StockPicker.UpdateStoredPicks, BackfillSwingFlags). Not a backtest setting: Production() has no swing trading. The
		candidates from docs/experiment/swing_tuning_grid.py until the 6.18 tests (878-897) settle them: trim z > 1.5 or a
		5-day rise > 1.5 sigma, buy back 1.5 vol x sqrt 10 lower; pullback z < -2 or a 5-day fall > 2 sigma, not risk-off,
		sell 0.5 vol x sqrt 10 higher. Change these and run BackfillSwingFlags(replace=True) to restate the history."""
		p = cls()
		p.swingFraction, p.swingDays, p.swingSigma = 0.3, 10, 1.5
		p.pullFraction = 0.3
		return p

	def IsCanonicalUniverse(self) -> bool:
		"""The universe stored states and picks are built on (top 250 by sp_GetTickerList with filter by fundamentals and
		the risk filter, point-in-time lists, nothing narrower). Only runs on it may write PicksBlendedDaily or PicksAdaptiveDaily
		(MarketStates: IsStateUniverse)."""
		return self.filterByFundamentals and self.riskFilter and not self.damageFilter and self.pointInTimeUniverse and not self.SP500Only and self.marketCapMin == 0 and self.marketCapMax == 0	#6.2: _PIT adopted

	def IsStateUniverse(self) -> bool:
		"""6.1: the universe the states of marketStateVersion are built on: the canonical list, point-in-time for 4.8+.
		Only runs on it may save or rebuild those MarketStates rows."""
		pit = self.marketStateVersion >= self.PIT_STATE_VERSION
		return self.filterByFundamentals and self.riskFilter and not self.damageFilter and self.pointInTimeUniverse == pit and not self.SP500Only and self.marketCapMin == 0 and self.marketCapMax == 0

	def RebuildMarketStatesProblem(self) -> str:
		"""Why rebuildMarketStates can't be honoured ('' = it can)."""
		if not self.IsStateUniverse(): return "only a run with filter by fundamentals + risk filter on, no damage filter/SP500Only/market cap limits, and _PIT exactly when the state version is 4.8+ may rebuild the states"
		return ''

	def RebuildBlendedPicksProblem(self, customBlend: bool = False) -> str:
		"""Why rebuildBlendedPicks can't be honoured ('' = it can): PicksBlendedDaily holds the default blend on the
		canonical universe, computed every trading day (rolling window), and can't be rebuilt while being read."""
		if not self.IsCanonicalUniverse() or self.volAdjustedRank: return "needs filter by fundamentals + risk filter on, no damage filter/_VA/SP500Only/market cap limits"
		if self.useStoredPicks: return "can't rebuild the stored picks while reading them (useStoredPicks)"
		if self.pickHistoryWindow <= 0: return "needs pickHistoryWindow > 0 (daily picks)"
		if customBlend: return "needs the default blend"
		return ''

	def PrepareAdaptiveRun(self):
		"""Fixed AdaptiveConvex settings (filterOption 98, its own pick window, no point-value allocation) and its name."""
		self.filterOption, self.allocateByPointValue, self.pickHistoryWindow = 98, False, -1
		self.modelName = "AdaptiveConvex_v4.6_BF"
		self.AddModelNameModifiers()

	def PrepareBlendedRun(self, blendName: str):
		"""Fixed PM_Blended settings and its name; the blend is only named when picks are computed (stored picks carry
		whatever blend they were built with)."""
		self.filterOption, self.allocateByPointValue = 99, False
		self.modelName = "PM_Blended" + ("" if self.useStoredPicks else f"_({blendName})")
		self.AddModelNameModifiers()

	def PreparePriceMomentumRun(self):
		self.modelName = ''
		self.AddModelNameModifiers()

@dataclass
class TradeModelPerformanceMetrics(SQLFriendlyMixin):
	# --- Core ---
	batchName: str = ''
	startValue: int = 0
	endValue: int  = 0
	startDate:  pd.Timestamp = None 
	endDate:  pd.Timestamp = None 
	durationInYears: int = 0
	total_gain: float = 0.0
	cagr: float = 0.0
	annualized_vol: float = 0.0
	sharpe_ratio: float = 0.0
	calmar_ratio: float = 0.0

	# --- Drawdowns ---
	max_drawdown: float = 0.0
	max_drawdown_start: pd.Timestamp = None 
	max_drawdown_low: pd.Timestamp = None 
	max_drawdown_end: Optional[pd.Timestamp] = None 
	max_drawdown_duration: int = 0

	avg_drawdown: float = 0.0
	dd_95pct: float = 0.0
	dd_99pct: float = 0.0
	pct_time_in_drawdown: float = 0.0

	avg_drawdown_duration: float = 0.0
	median_drawdown_duration: float = 0.0

	avg_recovery_days: float = 0.0
	median_recovery_days: float = 0.0
	max_recovery_days: int = 0

	# --- Calendar effects ---
	worst_jan_drawdown_year: int = 0
	worst_jan_drawdown_value: float = 0.0
	years_negative: int = 0

	# --- Convexity / tail behavior ---
	return_concentration_10pct: float = 0.0
	return_concentration_20pct: float = 0.0
	best_1yr_return: float = 0.0
	best_3yr_cagr: float = 0.0

	# --- Adaptive diagnostics ---
	avg_convex_weight: float = 0.0
	std_convex_weight: float = 0.0
	pct_time_convex_dominant: float = 0.0
	export_fields = [
		'startDate', 'endDate', 'durationInYears', 'total_gain', 'cagr', 'annualized_vol', 'sharpe_ratio', 'calmar_ratio',
		'max_drawdown', 'max_drawdown_start', 'max_drawdown_low', 'max_drawdown_end', 'max_drawdown_duration', 'avg_drawdown','dd_95pct','dd_99pct', 'pct_time_in_drawdown', 'avg_drawdown_duration', 'median_drawdown_duration',
		'avg_recovery_days','median_recovery_days', 'max_recovery_days', 'worst_jan_drawdown_year', 'worst_jan_drawdown_value', 'years_negative',
		'return_concentration_10pct','return_concentration_20pct','best_1yr_return', 'best_3yr_cagr', 'avg_convex_weight','std_convex_weight', 'pct_time_convex_dominant', 'batchName'
	]


#11/22/26: swing signal features and trade flags, shared by the trading engine (TradingModel.SwingTrade) and the stored picks
SWING_WINDOW = 40	#channel regression window
def SwingFeatures(closes) -> Optional[tuple]:
	"""(20-day daily vol, 40-day channel z, 5-day move in sigma) from the closes BEFORE the signal day (at least
	SWING_WINDOW + 1, the last one the prior close), or None. z = (log close - 40-day log regression line) / residual sd."""
	c = np.asarray(closes, dtype=float)[-(SWING_WINDOW + 1):]
	if len(c) < SWING_WINDOW + 1 or not np.all(np.isfinite(c)) or np.any(c <= 0): return None
	lc = np.log(c)
	r = np.diff(lc)
	vol = float(np.std(r[-20:], ddof=1))
	if not vol > 0: return None
	y = lc[1:]
	x = np.arange(SWING_WINDOW, dtype=float)
	xm = x.mean()
	b = float(((y - y.mean()) * (x - xm)).sum() / ((x - xm) ** 2).sum())
	resid = y - (y.mean() + b * (x - xm))
	sd = float(resid.std(ddof=2))
	z = float(resid[-1] / sd) if sd > 0 else 0.0
	move5 = float((lc[-1] - lc[-6]) / (vol * np.sqrt(5)))
	return vol, z, move5

def SwingFlag(vol: float, z: float, move5: float, riskOff: bool, p: TradeModelParams) -> Optional[str]:
	"""'TRIM', 'PULL' or None under p's swing rules (the same tests TradingModel.SwingTrade makes)."""
	if p.swingFraction > 0 and (riskOff or not p.swingRiskOffOnly) and (z > p.swingTriggerZ or move5 > p.swingTriggerSpike): return 'TRIM'
	if p.pullFraction > 0 and not (riskOff and p.pullSkipRiskOff) and (z < -p.pullTriggerZ or move5 < -p.pullTriggerDrop): return 'PULL'
	return None

def SwingLimitPcts(vol: float, p: TradeModelParams) -> tuple:
	"""(trim buy-back discount, pullback sell premium) as fractions of the fill price, as the engine sets its limits."""
	return min(0.5, p.swingSigma * vol * np.sqrt(p.swingDays)), p.pullSigma * vol * np.sqrt(p.pullDays)
