# =============================================================================================
# SELECTION VERSION (SELECTION_VERSION below). Each backtest stores the version its worker ran in
# TradeModelComparisons.SelectionVersion (NULL = run before versioning, 11/17/26). Bump it for any change to how
# picks are made (filters, ranking, vote weights, price stats they read) and add a line here and in _ChangeLog.txt.
# Options that are off by default and named in the model (e.g. _VA, _DMG) are listed with the version that added them.
# Data/universe changes are not versioned here; they are logged by time in _BacktestDataChanges.txt.
#   5.0  11/17/26 baseline: DEFAULT_BLEND (1.3.9.9.6) with the vote weights in _blend_from_multiverse,
#        base_filter (PC_9Month > 0, LogDrawdown > -0.5, MaxLoss_1Year > -0.6, PathologyScore < 2,
#        DamageScore < 0.5, PC_1Month3WeekEMA > -0.05 unless loose), LogDrawdown from the 5-year high.
#        Runs 596-600 (started 11/17 20:35, before the version column existed) are 5.0.
#   5.1  11/17/26: Volatility_1Year price stat; optional volAdjustedRank (model name _VA): momentum filters rank on
#        PC_x / max(Volatility_1Year, VOL_RANK_FLOOR) instead of raw PC_x. Off: picks identical to 5.0.
#   5.2  11/18/26: no selection change - bumped because the trading engine changed (Trading.py sell-lot fix and
#        rate-limit rollover). Same picks as 5.1, different trades: compare 5.2 runs only with 5.2 runs.
#   5.3  11/18/26: new options only - existing filters and the default blend pick as in 5.2 (5.2 and 5.3 runs compare):
#        filter 11 = filter 10's mask ranked deepest pullback first (filter 10 ranks LogDrawdown descending =
#        shallowest first, despite its comment); per-filter vol-adjusted ranking: filter id + VA_FILTER_OFFSET
#        (109 = filter 9 ranked like _VA); custom blends via TradeModelParams.blendFilters ("10:3,9v:3,4:2,6:1",
#        ParseBlend), named PM_Blended_(10-3.9v-3.4-2.6-1).
#   5.4  11/18/26: new options only (5.2-5.4 runs compare): filters 12 (12-1 momentum / volatility), 13 (low-volatility
#        trend), 14 (at the 5-year high, lowest volatility first); Adaptive execution filter set 2
#        (TradeModelParams.adaptiveFilterSet, model name _FS2): MANIA_BREAKING -> filter 10, filter 6 and 9v in
#        EXPANDING CONVEX / CONTRACTING. Set 1 (default) = the 4.6 lists unchanged.
#   5.5  11/18/26: new option only (5.2-5.5 runs compare): dmaWeight (TradeModelParams.dmaWeight, model name _DMA) applies
#        the vote blend's 200-day-average weighting as intended. The rows read 'Distance_200DMA', which Prices.py never
#        produced (it names it Distance200DMA, fullStats only), so every vote got 0.85. Off: unchanged (0.85 for all).
#        Prices.py now computes MA200/Distance200DMA without fullStats too (nothing else reads it unless fullStats).
#   5.6  11/18/26: new options only (5.2-5.6 runs compare):
#        - Two-phase MarketStates build, state version 4.7 (TradeModelParams.marketStateVersion, model name _MS47).
#          Phase 1 (BuildMarketStatesRaw, queue job filterOption 97) stores each day's raw measures; chunks of years
#          run in parallel on different workers. Phase 2 (DeriveMarketStates, run once, serially) fills the
#          history-dependent fields from the stored rows with DeriveStateFields. The same function derives a missing
#          day during a 4.7 backtest, so stored and live states follow one rule. Differences from 4.6: the previous
#          day for the expansion state is always the 5-day mean of the stored rows through yesterday (what the 4.6
#          daily build used, every day being a pick day; a 4.6 build inside a backtest used it only on pick days and
#          the raw row otherwise, so its labels depended on reEvaluationInterval); velocity_ewm is stored every day.
#          4.6 quirks kept: state_confidence is computed with today's label at confidence 0 (always TRANSITION) and
#          convex_duration 0 (logging only - no decision reads it).
#        - riskOffExposure (_RO50): when .INX closed more than 2% below its 200-day average (back above +2% to end;
#          prior day's close), blended/adaptive picks are scaled to this share and the rest is held as cash.
#          A change in the signal triggers a re-pick that day (IndexRiskSwitch, ApplyRiskOffExposure).
#        - adaptiveFilterSet 3 (_FS3): set 2, and filter 6 in place of filter 1 in every list set 2 leaves as set 1.
#        - adaptiveStockCount (_S12): fixed adaptive stock count instead of GetStockCount (conviction + blow-off floors).
#   5.7  11/18/26: no change to backtests (5.2-5.7 runs compare). Stored picks (PicksBlendedDaily, PicksAdaptiveDaily) come
#        from one function, UpdateStoredPicks, with the production settings below (PRODUCTION_BLEND, TradeModelParams.Production('adaptive')):
#        RefreshPricesWorkingSet appends from the last stored date (re-doing the last few days), and the full rebuild runs
#        as queue jobs in year chunks (filterOption 96 blended, 95 adaptive). Each month's days use that month's universe
#        (as the state build and backtests do). Adaptive replays the 60 calendar days before the first written date, so
#        the vote window (at most 34 trading days) is full and a chunk or refresh writes what a continuous build would.
#        Adaptive days with the index risk-off signal store a cash row (TradeModelParams.Production('adaptive') riskOffExposure).
#        The refresh also rebuilds the states of the re-done days (late prices); ModelPastFewYears can rebuild 5 years.
#        4.7 states are only built for .INX trading days (_state_calendar), as phase 1 is: a stock with a price on a
#        holiday no longer adds a state row (which shifted every later percentile window by one row), and days after
#        .INX's last price (partial data) get none. 4.6 runs unchanged.
#        - StockPicker.saveGeneratedStates: a run on a narrower universe (SP500Only, market cap limits, damage filter, no
#          risk filter) builds a missing day's state in memory only instead of saving it (AlphaTrader sets it).
#        - stress_index is now a derived field (DeriveStateFields; it reads the prior states). Every 4.7 row built by
#          5.6 phase 1 stored 0, so all 4.7 runs before 5.7 used stress 0 (conviction score, state_confidence): re-run them.
#   5.8  11/19/26: live PM_Blended (computed picks, not --db) starts with its rolling window filled from the trading days
#        before the start date (StockPicker.replayPickWindow), as the stored picks are; before, the window started empty.
#        Live blended runs compare only with 5.8+; adaptive, PM_filter and stored-picks (_DB) runs are unchanged (5.2-5.8
#        compare). Defaults moved to TradeModelParams (Production(), the default test settings): no effect on results.
#   5.9  11/19/26: new options only (results as 5.8; same comparison rules):
#        - riskOffSignal 'leaders' (_RO50L): the risk switch reads the momentum-leader index (LeaderIndexDaily: each day's
#          equal-weight return of the 25 highest PC_6Month stocks of that month's canonical universe, chosen on the prior
#          day's close; BuildLeaderIndex) below its 100-day average (2% band) instead of .INX below its 200-day.
#        - postManiaDays (_PMD252, adaptive): POST_MANIA lasts this many trading days after the last mania day instead of 126.
#   6.0  11/19/26: new option only (results as 5.9 unless _PIT): pointInTimeUniverse (_PIT) - sp_GetTickerList @PointInTime
#        chooses month M's list from month M-1 figures (momentum from month-end prices, the $5 check on the month-end raw
#        price, liquidity from M-1's complete month). The default list ranks on month M's days 1-6 prices and the whole
#        month's volume, which a backtest picking on day 1 couldn't have known. _PIT runs aren't the canonical universe
#        (they don't save states or picks). Adaptive _PIT runs still read the 4.7 states built on the default lists.
#   6.1  11/19/26: MarketStates 4.8 (_MS48) = the 4.7 two-phase build on the point-in-time lists (BuildMarketStatesRaw
#        passes @PointInTime for 4.8+). Adaptive _PIT runs read 4.8 (queue enforces _PIT <-> 4.8) and may save/rebuild
#        4.8 states (TradeModelParams.IsStateUniverse). Other runs unchanged (results as 6.0).
#   6.2  11/19/26: the point-in-time list is the default and the canonical universe: TradeModelParams.pointInTimeUniverse
#        True and marketStateVersion 4.8 (defaults and Production), sp_GetTickerList @PointInTime default 1 (Python always
#        passes it), GetTickerListSQL pointInTime=True. Stored picks (UpdateStoredPicks), the daily refresh and the leader
#        index use the point-in-time lists; PicksBlendedDaily / PicksAdaptiveDaily / LeaderIndexDaily rebuilt on them.
#        Names keep _PIT (default runs read ..._FF_HW28_PIT); the old lists need pointInTimeUniverse False (adaptive: with
#        4.7 states). Jobs always carry pointInTimeUniverse and marketStateVersion. Results as 6.1 for the same settings.
#   6.3  11/20/26: fundamentals reporting lag (sp_GetTickerList_63.sql). Point-in-time lists take MarketCapitalization and
#        NetProfitMargin from the row 3 months earlier (@FundamentalsLagMonths), market cap moved to the M-1 month-end
#        price. Only runs with marketCapMin / marketCapMax change; all other results as 6.2.
#   6.4  11/20/26: duplicate exclusions point-in-time (sp_GetTickerList_64.sql, LiquidityExclusions.KeptTicker). A
#        duplicate copy is skipped only in months its kept ticker is priced; bad-data exclusions still every month.
#        13 of 130 affected lists change (1980-01..04, 1984-11, 1985-01/03/06/07, 1990-04); all others as 6.3.
#   6.5  11/20/26: no back-filled price stats (Prices.CalculateStats ffill only). The bfill copied each stat's first
#        value over the days before it existed: in a stock's first year PC_1Year was its return a year later, in all of
#        1980 (prices start 1980-01-02) for every stock. Now NaN, which fails the masks; filters also skip rows with no
#        ranking value. 1980: cash until Oct 1980 (filters 6/9/10), 1-year filters from Jan 1981. Changes 1980 windows
#        and picks of stocks in their first ~2 years of prices (mostly 1980s-90s); live picks unaffected (nothing later
#        to copy from). Stored picks and 1980-81 MarketStates must be rebuilt.
#   6.6  11/20/26: adaptive warm-up. With 6.5 the 4.8 states start 1981-02-02 (no state while the state universe has
#        no 1-year returns); a day before the first stored state now holds cash instead of asserting on universe_size.
#        Only adaptive runs before 1981-02 are affected (they crashed on 6.5).
#   6.7  11/20/26: no state is generated before the first stored state, whatever the universe size (the 6.6 check only
#        applied under UNIVERSE_SIZE_MIN, and the history warm-up passes the current month's universe for earlier days, so a
#        run starting 1980 built and saved January 1981 states from the February list). UpdateStoredPicks writes a cash row
#        for those days and clears a rebuilt month even when it writes no rows (1980-02..1981-01 kept the 6.4 picks).
#        Only adaptive runs before 1981-02 are affected.
#   6.8  11/21/26: volatility-managed exposure (TradeModelParams.volTarget / volLookback, _VT<pct>, blended/adaptive, off by
#        default). On each re-pick, VolTargetExposure measures the realized volatility of the fully invested portfolio over the
#        last volLookback trading days (daily return / prior day's invested share) and holds min(1, target / volatility) of the
#        picks, rounded to 10%, the rest in cash; applied after the risk switch. Barroso and Santa-Clara (2015), Moreira and
#        Muir (2017). Runs without it are unchanged.
# =============================================================================================
# CHANGE LOG (market state / adaptive convex picks). Each change has an on/off flag below so it
# can be backed out individually for comparison runs.
#   4.6-A  Warmup preloads stored states (STATE_HISTORY_ROWS), resumes convex_duration/velocity.
#   4.6-B  Trading-day calendar from loaded prices (no holidays); holiday toDate -> prior day.
#   4.6-C  History trim and percentiles both row-based (STATE_HISTORY_ROWS).
#   4.6-D  toDate processed exactly once in hydration (no self-comparison).
#   4.6-E  One stored row per date per state_version: save/update/lookup keyed on asOfDate +
#          state_version only; universe_size is stored for context, never used as a key.
#   4.6-F  disp_pct / dvol_z (dispersion percentile, downside-vol z) stored per state.
#   4.6-G  Mania split: MANIA_RUNNING / MANIA_BREAKING / POST_MANIA via disp_off_high and
#          days_since_mania (flag USE_MANIA_SPLIT).
#   4.6-H  Stock-count minimums by blow-off state (flag USE_BLOWOFF_STOCK_FLOORS).
#   4.6-I  Single-name weight cap in mania states (flag USE_MANIA_WEIGHT_CAP).
#   4.6-J  Shorter vote window when a mania breaks (flag USE_BREAKING_SHORT_WINDOW).
#   4.6-K  Defensive filter blend when a mania breaks (flag USE_BREAKING_FILTERS).
#   4.6-L  QUIET_MELTUP is logged only; its filter switch/floor are off (flag USE_MELTUP_RESPONSE).
#   4.6-M  velocity_ewm (computed after smoothing) is written back to SQL and history.
#   4.6-O  AlignToList removes tickers through RemoveTicker (optional index hint) so _tickerList stays in
#          sync and dropped tickers can be re-added (universe decay fix).
#   4.6-P  The state build asserts if the universe (current run or stored row) is under UNIVERSE_SIZE_MIN.
#   Optional, off: ROBUST_DISPERSION (IQR of log returns; re-tune before enabling).
# =============================================================================================
import numpy as np, pandas as pd, time
from collections import OrderedDict
from types import SimpleNamespace
import _classes.Constants as CONSTANTS
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict
from tqdm import tqdm
from _classes.Utility import *
from _classes.Prices import PriceSnapshot, PricingData
from _classes.Trading import TradeModelParams
from _classes.DataIO import PTADatabase
from _classes.TickerLists import TickerLists

ADAPTIVE_WARMUP_DAYS = 60 		# small, deterministic warmup for the convex engine
CASH_RESULT = pd.DataFrame(columns=['TargetHoldings', 'Point_Value'])
CASH_RESULT.loc[CONSTANTS.CASH_TICKER] = {'TargetHoldings': 1.0, 'Point_Value': 0}
CASH_RESULT.index.name = 'Ticker'

MODE_CONVEX_FULL   = "CONVEX_FULL"
MODE_CONVEX_STABLE = "CONVEX_STABLE"
MODE_CONVEX_LATE   = "CONVEX_LATE"
MODE_BLENDED       = "BLENDED"
MODE_DEFENSIVE     = "DEFENSIVE"
MODE_TRANSITION = "TRANSITION"
SELECTION_VERSION = "6.8"		# see the version list at the top; stored with every backtest result
VOL_RANK_FLOOR = 0.10			# volAdjustedRank: volatility below this counts as this (stops near-flat series ranking first)
MARKET_STATE_VERSION_2PHASE = 4.7	# 5.6: two-phase build (BuildMarketStatesRaw + DeriveMarketStates); runs read it with marketStateVersion 4.7
MARKET_STATE_VERSION_PIT = TradeModelParams.PIT_STATE_VERSION	# 6.1: 4.8 = two-phase build on the point-in-time lists
MARKET_STATE_VERSION = 4.6		# 4.6: warmup from stored states, trading-day calendar, row-based history window, blow-off detection
STATE_HISTORY_ROWS = 2520		# ~10 trading years; used for the warmup preload, history trim and all percentiles
STATE_HISTORY_MIN_ROWS = 250	# below this, disp_pct/dvol_z stay neutral (early 1980s)
UNIVERSE_SIZE_WARN = 200		# universe below this is a data flag (missing prices), not a regime signal
UNIVERSE_SIZE_MIN = 170			# 4.6-P: the state build stops (assert) if the universe is ever smaller than this
ROBUST_DISPERSION = False		# True = IQR of log 1y returns; changes every dispersion threshold, re-tune before enabling
BLOWOFF_MELTUP_PCT = 0.90		# QUIET_MELTUP: dispersion percentile >= this ...
BLOWOFF_MELTUP_DVOL_Z = -0.5	# ... and downside-vol z-score <= this
BLOWOFF_MANIA_PCT = 0.85		# VOLATILE_MANIA: dispersion percentile >= this ...
BLOWOFF_MANIA_DVOL_Z = 1.5		# ... and downside-vol z-score >= this
BLOWOFF_MANIA_MIN_STOCKS = 11		# MANIA_RUNNING (and plain mania when the split is off)
BLOWOFF_MELTUP_MIN_STOCKS = 13
BLOWOFF_MELTUP_FILTERS = [1,3,3,9,8,8]	# used only if USE_MELTUP_RESPONSE
# --- 4.6-G..L switches and settings (round numbers on purpose: few episodes, avoid fitting 2000)
USE_MANIA_SPLIT = True
USE_BLOWOFF_STOCK_FLOORS = True
USE_MANIA_WEIGHT_CAP = True
USE_BREAKING_SHORT_WINDOW = True
USE_BREAKING_FILTERS = True
USE_MELTUP_RESPONSE = False		# QUIET_MELTUP did not replicate in v4.6 (1980-2010): log only
MANIA_BREAK_OFF_HIGH = 0.80		# mania + dispersion < 80% of its 63-day high -> MANIA_BREAKING
DISP_HIGH_LOOKBACK_ROWS = 63
POST_MANIA_DAYS = 126			# trading days after the last mania day treated as POST_MANIA
MANIA_BREAKING_MIN_STOCKS = 13
POST_MANIA_MIN_STOCKS = 11
MANIA_RUNNING_MAX_WEIGHT = 0.15
MANIA_BREAKING_MAX_WEIGHT = 0.10
MANIA_BREAKING_WINDOW = 14		# vote window (days) while a mania is breaking; normal is 18-34
MANIA_BREAKING_FILTERS = [1,3,3,9,8]	# existing most-defensive blend (LINEAR/TRANSITIONAL)
NO_MANIA_DAYS = 99999
DEFAULT_BLEND = [(1,3),(3,3),(9,3),(9,3),(6,1)]
#5.7: settings the stored picks tables are built with (UpdateStoredPicks: RefreshPricesWorkingSet and the queued rebuild).
#Changing either one means regenerating that table's history (queue_backtest.py --stored-picks), or old and new rows differ.
PRODUCTION_BLEND = DEFAULT_BLEND		#PicksBlendedDaily (also what --db backtests read)
STORED_PICKS_WARMUP_DAYS = 60		#calendar days replayed before the first written adaptive date (vote window <= 34 trading days * 1.4484)
STORED_PICKS_RECENT_DAYS = 14		#days this recent are only written when STORED_PICKS_MIN_COVERAGE of the universe has a price for them
STORED_PICKS_MIN_COVERAGE = 0.97	#(the refresh downloads the universe's prices first; 2026-11-17: 244 of 250 had one)
VA_FILTER_OFFSET = 100		#5.3: filter id + 100 = that filter ranked on return / volatility (like _VA, for one filter only)
BLEND_FILTER_IDS = {1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14}	#filters a custom blend may use ('v' = vol-adjusted, momentum-sorted filters only)
VA_CAPABLE_FILTERS = {1, 2, 3, 4, 6, 7, 8, 9}

def VersionKey(v) -> tuple:
	#'5.10' > '5.9': compare versions as integer tuples. Empty/None = (0,)
	try: return tuple(int(x) for x in str(v).strip().split('.')) if v not in (None, '') else (0,)
	except ValueError: return (0,)

def ParseBlend(text: str) -> list:
	#"10:3,9v:3,4:2,6:1" -> [(10,3),(109,3),(4,2),(6,1)]. 'v' after a filter id = vol-adjusted ranking for that filter
	result = []
	for part in str(text).replace(' ', '').split(','):
		if not part: continue
		fid, _, count = part.partition(':')
		va = fid.lower().endswith('v')
		fid = int(fid[:-1] if va else fid)
		if fid not in BLEND_FILTER_IDS: raise ValueError(f"blend: unknown filter {fid} (allowed {sorted(BLEND_FILTER_IDS)})")
		if va and fid not in VA_CAPABLE_FILTERS: raise ValueError(f"blend: filter {fid} doesn't rank on return, 'v' doesn't apply")
		n = int(count) if count else 3
		if n <= 0: raise ValueError("blend: stock count must be > 0")
		result.append((fid + VA_FILTER_OFFSET if va else fid, n))
	if not result: raise ValueError("blend: no filters given")
	return result

def BlendName(filterOptions: list) -> str:
	#[(10,3),(109,3),(4,2)] -> "10-3.9v-3.4-2"
	return ".".join(f"{f - VA_FILTER_OFFSET}v-{n}" if f >= VA_FILTER_OFFSET else f"{f}-{n}" for f, n in filterOptions)

@dataclass
class AdaptiveConvexMarketState:
	as_of_date: pd.Timestamp
	convex_duration: int = 0
	dispersion: float = 0.0
	momentum_autocorr: float = 0.0
	downside_volatility: float = 0.0
	stress_index: float = 0.0
	corr_6m_1m: float = 0.0
	corr_1y_1m: float = 0.0
	leadership_tilt: float = 0.0
	disp_p40: float = 0.0
	disp_p75: float = 0.0
	expansion_velocity: float = 0.0
	velocity_ewm: float = 0.0        # 5-day EWM of expansion_velocity — smoothed regime momentum
	expansion_state: Optional[str] = "NO_DATA"    # EXPANDING / STABLE / CONTRACTING
	state_version: Optional[float] = MARKET_STATE_VERSION
	state_confidence: float = 0.0
	universe_size: int = 0
	disp_pct: float = 0.5          # percentile of today's dispersion within the prior STATE_HISTORY_ROWS states (0-1)
	dvol_z: float = 0.0            # z-score of downside_volatility within the same history
	disp_off_high: float = 1.0     # dispersion / max dispersion over the last DISP_HIGH_LOOKBACK_ROWS states (incl. today)
	days_since_mania: int = NO_MANIA_DAYS  # trading days since the last prior mania state (0 = yesterday)

	@property
	def is_mania(self) -> bool:
		return self.disp_pct >= BLOWOFF_MANIA_PCT and self.dvol_z >= BLOWOFF_MANIA_DVOL_Z

	@property
	def blowoff_state(self) -> str:
		"""Late-stage blow-off classification (1980-2024 in-sample, 45-year model equity, 126-day forward):
		  QUIET_MELTUP   extreme dispersion + calm downside vol. Median fwd return 8% vs 26% baseline;
		                 P(fwd drawdown < -40%) 18% vs 4%. 11 episodes (2004, mid-2018, Feb 2020 ...).
		  VOLATILE_MANIA extreme dispersion + very high downside vol. Median fwd return 85%, but
		                 P(fwd drawdown < -30%) 57% vs 14%. 5 episodes (1998-2000, 2020-21 ...).
		                 Keep the aggressive filters, limit concentration.
		  v4.6 split (1980-2010, 63-day forward, delisted model): mania with dispersion near its
		  63-day high -> median +20%, P(dd<-20%) 39%; mania with dispersion >20% below the high
		  -> median +2%, P(dd<-20%) 68%. Labels: MANIA_RUNNING, MANIA_BREAKING, POST_MANIA
		  (up to POST_MANIA_DAYS after the last mania day). QUIET_MELTUP did not replicate in v4.6.
		  NONE           none of the above."""
		if self.is_mania:
			if not USE_MANIA_SPLIT: return "VOLATILE_MANIA"
			return "MANIA_BREAKING" if self.disp_off_high < MANIA_BREAK_OFF_HIGH else "MANIA_RUNNING"
		if USE_MANIA_SPLIT and self.days_since_mania < (getattr(self, "post_mania_days", 0) or POST_MANIA_DAYS):	#5.9: per-run override (not a stored field)
			return "POST_MANIA"
		if self.disp_pct >= BLOWOFF_MELTUP_PCT and self.dvol_z <= BLOWOFF_MELTUP_DVOL_Z:
			return "QUIET_MELTUP"
		return "NONE"

	@property
	def persistence_break_flag(self) -> bool:
		return (self.momentum_autocorr < -0.1 and self.dispersion < self.disp_p40)

	@property
	def geometry_state(self):
		if self.dispersion >= self.disp_p75:
			return "CONVEX"
		elif self.dispersion <= self.disp_p40:
			return "LINEAR"
		return "MIXED"

	@property
	def leadership_state(self):
		if self.leadership_tilt > 0.6:
			return "STABLE"
		elif self.leadership_tilt < 0.4:
			return "ROTATING"
		return "TRANSITIONAL"
	
	@property
	def leadership_break_flag(self):
		return (self.leadership_state == "ROTATING"	and self.momentum_autocorr < 0)

	@property
	def regime_tension_flag(self):
		return (self.geometry_state == "CONVEX"	and self.expansion_state == "CONTRACTING")	
	
	def set_expansion_state(self, prev_state):
		self.expansion_state = "STABLE"
		self.expansion_velocity = 0.0
		if prev_state is None:
			return
		d_autocorr = self.momentum_autocorr - prev_state.momentum_autocorr
		d_disp = self.dispersion - prev_state.dispersion
		disp_range = max(self.disp_p75 - self.disp_p40, 1e-6)
		d_disp_norm = d_disp / disp_range
		velocity = 0.6 * d_autocorr + 0.4 * d_disp_norm
		velocity = max(-1.0, min(1.0, velocity))
		self.expansion_velocity = velocity
		expanding = (d_autocorr > 0 and d_disp > 0)
		contracting = (d_autocorr < 0 and d_disp < 0)
		if expanding:
			self.expansion_state = "EXPANDING"
		elif contracting:
			self.expansion_state = "CONTRACTING"
	
	def GetModeLabel(self) -> str:
		# Label kept for SQL logging and diagnostics only — no longer drives decisions.
		# All execution decisions now use the continuous signal methods below.
		if self.state_confidence < 0.45:
			return MODE_TRANSITION
		if self.regime_tension_flag or self.leadership_break_flag:
			return MODE_TRANSITION
		g = self.geometry_state
		e = self.expansion_state
		l = self.leadership_state
		if self.persistence_break_flag:
			return MODE_DEFENSIVE
		if g == "CONVEX":
			if e == "EXPANDING" and l == "STABLE": return MODE_CONVEX_FULL
			if e == "CONTRACTING": return MODE_CONVEX_LATE
			return MODE_CONVEX_STABLE
		if g == "LINEAR": return MODE_DEFENSIVE
		return MODE_BLENDED

	# ── Continuous signal properties ───────────────────────────────────────────
	# All three execution methods below operate on the raw underlying floats,
	# not the discretised label strings.  This avoids the information loss where
	# a velocity of +0.001 and +0.08 both read as "EXPANDING" and received
	# identical treatment.
	#
	# Key signals from 1980-2023 analysis:
	#   disp_norm  Spearman 0.070 vs 21d return — strongest predictor.
	#              Best periods: median 1.46. Worst periods: median 0.14.
	#   velocity   Spearman 0.147 vs 1d return — regime momentum signal.
	#              Weak at 21 days (0.018). Drives window size only.
	#   stress_idx D10 (highest stress) returns 7.76% vs mid-decile 4-5%.
	#              Peak stress is a recovery signal — do NOT penalise it.
	#   autocorr   Essentially flat across deciles — no standalone predictive
	#              power. Retained only as velocity component.

	@property
	def disp_norm(self) -> float:
		"""Dispersion normalised to its own rolling percentile band.
		0 = at the p40 (LINEAR) boundary; 1 = at the p75 (CONVEX) boundary.
		Negative = unusually compressed; >1 = unusually wide (convex zone).
		Far richer than the three-bucket geometry label."""
		denom = max(self.disp_p75 - self.disp_p40, 1e-6)
		return (self.dispersion - self.disp_p40) / denom

	@property
	def conviction_score(self) -> float:
		"""Composite conviction score in [0, 1].
		High = concentrate positions; low = diversify.

		Components:
		  disp_norm      Primary driver (Spearman 0.070 at 21d). Best periods median 1.46.
		  stress_index   Recovery signal: D10 highest stress returned 7.76% vs 4-5% mid.
		                 Treated as mild positive (opportunity at stress peaks), not a penalty.
		  leadership_tilt Persistent leadership adds moderate conviction.
		  velocity_ewm   5-day EWM of expansion_velocity. Single-day velocity is noise
		                 (Spearman 0.018 at 21d) but sustained negative velocity signals
		                 genuine regime deterioration — reduce concentration proportionally.
		                 This is the fix for v4.2's max_rec regression (636 → 1117 days):
		                 disp_norm can stay elevated while held stocks collapse; the velocity
		                 EWM catches the regime momentum turning negative and pulls back."""
		disp_signal   = float(np.clip(self.disp_norm / 2.0, 0.0, 1.0))
		stress_signal = float(np.clip(self.stress_index * 0.4, 0.0, 0.25))
		lt_signal     = float(np.clip((self.leadership_tilt - 0.5) * 0.5, 0.0, 0.25))
		# velocity_ewm: scale to ±0.20 impact on conviction.
		# Positive sustained velocity → small boost (capped at +0.20).
		# Negative sustained velocity → penalty (floored at -0.20).
		# Asymmetric clip: penalise deterioration harder than rewarding acceleration,
		# since the cost of holding concentrated positions through a regime break is
		# higher than the cost of being slightly underweighted at the start of a run.
		vel_signal = float(np.clip(self.velocity_ewm * 3.0, -0.30, 0.20))
		raw = disp_signal + stress_signal + lt_signal + vel_signal
		return float(np.clip(raw, 0.0, 1.0))

	def GetMaxWeight(self):
		"""Single-name cap on target weight (4.6-I). None = no cap."""
		if not USE_MANIA_WEIGHT_CAP: return None
		b = self.blowoff_state
		if b in ("MANIA_RUNNING", "VOLATILE_MANIA"): return MANIA_RUNNING_MAX_WEIGHT
		if b == "MANIA_BREAKING": return MANIA_BREAKING_MAX_WEIGHT
		return None

	def GetRollingWindowSize(self) -> int:
		"""Window driven by velocity_ewm (smoothed regime momentum).
		Using the 5-day EWM rather than raw single-day velocity reduces noise
		while preserving the directional signal.
		Positive velocity_ewm → shorten window (capture accelerating leaders faster).
		Negative velocity_ewm → lengthen window (hold persistent leaders in decel regime).
		Base 26 ± up to 8 days."""
		v  = float(np.clip(self.velocity_ewm, -1.0, 1.0))
		hw = 26 + int(round(-8.0 * v))
		if USE_BREAKING_SHORT_WINDOW and self.blowoff_state == "MANIA_BREAKING":
			return MANIA_BREAKING_WINDOW   # 4.6-J: let faded leaders' votes expire quickly
		return int(np.clip(hw, 18, 34))

	def GetStockCount(self) -> int:
		"""Position count driven continuously by conviction_score.
		Score 1.0 → 7 stocks (maximum concentration).
		Score 0.0 → 15 stocks (maximum diversification).
		Interpolates linearly. conviction_score incorporates velocity_ewm
		so sustained regime deterioration smoothly reduces concentration.

		Floor in EXPANDING regime: even if disp_norm is compressed (unusual —
		dispersion can lag the expansion label by a day or two), we don't want
		15-stock diversification during an expanding regime. Floor at 10."""
		score = self.conviction_score
		# Apply a conviction floor during expanding regimes so compressed dispersion
		# doesn't accidentally push stock count too high when the regime is clearly bullish
		if self.expansion_state == "EXPANDING":
			score = max(score, 0.40)   # floor → max 11 stocks in expanding regime
		count = 15 - int(round(score * 8.0))
		# Blow-off floors: conviction_score rises with dispersion, which concentrates into 7 stocks exactly
		# when forward crash risk is highest. Spread the risk instead of exiting.
		if USE_BLOWOFF_STOCK_FLOORS:   # 4.6-H
			b = self.blowoff_state
			if b in ("MANIA_RUNNING", "VOLATILE_MANIA"): count = max(count, BLOWOFF_MANIA_MIN_STOCKS)
			elif b == "MANIA_BREAKING": count = max(count, MANIA_BREAKING_MIN_STOCKS)
			elif b == "POST_MANIA": count = max(count, POST_MANIA_MIN_STOCKS)
			elif b == "QUIET_MELTUP" and USE_MELTUP_RESPONSE: count = max(count, BLOWOFF_MELTUP_MIN_STOCKS)
		return int(np.clip(count, 7, 15))

	def GetExecutionFilters(self, filterSet: int = 1) -> list:
		"""Filter blend selected by regime label (v4.1 approach, restored in v4.4).
		filterSet 2 (5.4, model name _FS2) replaces four of the lists (see _execution_filters_set2); 1 = as below.

		Hybrid architecture rationale:
		  GetExecutionFilters  → label-based (decisive, preserves explosive compounding years)
		  GetStockCount        → continuous conviction_score with velocity_ewm (smooth concentration)
		  GetRollingWindowSize → continuous velocity_ewm (smooth window adaptation)

		Backtesting showed continuous filter selection (v4.2/v4.3) applied a mild
		daily discount that compounded against CAGR by ~2.7pp over 43 years.
		Label-based selection makes high-conviction choices that capture the full
		return in expanding regimes without that drag.

		Filter reference (HW26 standalone, 1980-2023):
		  F3  CAGR 67%  DD -55%  4 neg yrs  — core anchor, 1yr momentum
		  F9  CAGR 68%  DD -63%  5 neg yrs  — core anchor, 9mo momentum
		  F1  CAGR 64%  DD -57%  6 neg yrs  — broad discovery (loose mask)
		  F8  CAGR 54%  DD -62%  4 neg yrs  — quality gate (EMA > 3mo)
		  F4  CAGR 44%  DD -66%  6 neg yrs  — acceleration (short signal life)
		"""
		if filterSet == 3:	#5.6: set 2, with filter 6 in place of filter 1 in the lists set 2 leaves as set 1
			lst = self._execution_filters_set2()
			if lst is None: lst = self.GetExecutionFilters(filterSet=1)
			return [6 if f == 1 else f for f in lst]
		if filterSet == 2:
			fs2 = self._execution_filters_set2()
			if fs2 is not None: return fs2
		# MANIA_RUNNING keeps the label-based blend (that is where the big years come from); only
		# concentration is limited (GetStockCount, GetMaxWeight). MANIA_BREAKING turns defensive.
		if USE_BREAKING_FILTERS and self.blowoff_state == "MANIA_BREAKING":
			return list(MANIA_BREAKING_FILTERS)   # 4.6-K
		if USE_MELTUP_RESPONSE and self.blowoff_state == "QUIET_MELTUP":
			return list(BLOWOFF_MELTUP_FILTERS)   # 4.6-L (off by default)
		g = self.geometry_state    # CONVEX / LINEAR / MIXED
		e = self.expansion_state   # EXPANDING / CONTRACTING / STABLE
		l = self.leadership_state  # STABLE / TRANSITIONAL / ROTATING

		# EXPANDING: maximum aggression — data shows 100%+ annualised in all geometries.
		# F4 included in CONVEX expanding; velocity_ewm window shortening suits its short signal life.
		if e == "EXPANDING":
			if g == "CONVEX" and l == "STABLE":
				return [3,3,9,9,1,4,4]   # peak conviction
			if g == "CONVEX":
				return [3,3,9,9,1,4]
			return [1,3,3,9,9,8]          # LINEAR/MIXED expanding: quality-anchored aggression

		# CONTRACTING: stay with blended default — data showed 75-84% ann. in contraction.
		# Continuous stock count (via conviction_score + velocity_ewm) handles concentration
		# reduction; filter blend stays aggressive so we don't give up the return.
		if e == "CONTRACTING":
			return [1,3,3,9,9,8]

		# STABLE CONVEX
		if g == "CONVEX":
			if l == "STABLE":
				return [3,3,9,9,8,1]
			if l == "ROTATING":
				return [3,3,9,9,8]        # drop F1 in churn
			return [3,3,9,9,8,1]          # TRANSITIONAL

		# STABLE LINEAR
		if g == "LINEAR":
			if l == "STABLE":
				return [1,3,3,9,9,8]
			if l == "ROTATING":
				return [3,3,9,9,8]
			return [1,3,3,9,8]            # TRANSITIONAL

		# STABLE MIXED
		if l == "STABLE":
			return [1,3,3,9,9,8]
		if l == "ROTATING":
			return [3,3,9,9,8]
		return [1,3,3,9,8]                # fallback

	def _execution_filters_set2(self):
		"""5.4 filter set 2, from single-filter runs 1995-2021 by execution path (state at the prior close), checked
		in both 1995-2008 and 2009-2021. Annualized returns 95-08 / 09-21:
		  MANIA_BREAKING     momentum filters -55% to -82% in both halves; F10 -20% / +7%
		                     (2000-03-30..05-23: momentum -25..-33%, F10 -2%; 2021-03: -21..-29%, F10 -1%)
		  EXPANDING CONVEX   F6 136% / 44%, F9v 43% / 16%; F1/F3/F9 (the current core) -12/-31/-26% in 09-21
		  EXP CONVEX STABLE  F9 96% / 47%, F6 122% / 31%; F4 (counted twice now) -9% in 09-21
		  CONTRACTING        F6 34% / 53% (top or near); F1 the weakest in both (4% / 39%)
		Other states are unchanged (STABLE CONVEX and ROTATING rankings flip between halves).
		Returns None to fall back to the set 1 list."""
		if self.blowoff_state == "MANIA_BREAKING":
			return [10, 10, 10]
		e = self.expansion_state
		if e == "EXPANDING" and self.geometry_state == "CONVEX":
			if self.leadership_state == "STABLE":
				return [9, 9, 9 + VA_FILTER_OFFSET, 6, 6, 3]
			return [6, 6, 9 + VA_FILTER_OFFSET, 9 + VA_FILTER_OFFSET, 4 + VA_FILTER_OFFSET, 10]
		if e == "CONTRACTING":
			return [3, 9, 9, 6, 6, 8]
		return None

	def GetRegimeSummary(self) -> str: 
		return (f"{self.geometry_state}_{self.expansion_state}_{self.leadership_state}_{self.blowoff_state}"	)

	def SaveToSQL(self):
		db = PTADatabase()
		if not db.Open(): 
			return		
		sql_delete = f"DELETE FROM {CONSTANTS.ADAPTIVE_CONVEX_STATE_TABLE} WHERE asOfDate = :as_of_date AND state_version = :state_version"   # one row per date per version; universe_size is context only
		delete_params = {"as_of_date": self.as_of_date, "state_version": float(self.state_version)}
		db.ExecSQL(sql_delete, delete_params)
		sql_insert = f"INSERT INTO {CONSTANTS.ADAPTIVE_CONVEX_STATE_TABLE} (asOfDate, convex_duration, dispersion, momentum_autocorr, downside_volatility, stress_index, corr_6m_1m, corr_1y_1m, leadership_tilt, disp_p40, disp_p75, expansion_velocity, velocity_ewm, geometry_state, expansion_state, leadership_state, regime_tension_flag, leadership_break_flag, persistence_break_flag, state_confidence, universe_size, state_version, disp_pct, dvol_z, blowoff_state, disp_off_high, days_since_mania ) VALUES (:as_of_date, :convex_duration, :dispersion, :momentum_autocorr, :downside_volatility, :stress_index, :corr_6m_1m, :corr_1y_1m, :leadership_tilt, :disp_p40, :disp_p75, :expansion_velocity, :velocity_ewm, :geometry_state, :expansion_state, :leadership_state, :regime_tension_flag, :leadership_break_flag, :persistence_break_flag, :state_confidence, :universe_size, :state_version, :disp_pct, :dvol_z, :blowoff_state, :disp_off_high, :days_since_mania)"
		insert_params = {
			"as_of_date": self.as_of_date,
			"convex_duration": int(self.convex_duration), 
			"dispersion": float(self.dispersion),
			"momentum_autocorr": float(self.momentum_autocorr),
			"downside_volatility": float(self.downside_volatility),
			"stress_index": float(self.stress_index),
			"corr_6m_1m": float(self.corr_6m_1m),
			"corr_1y_1m": float(self.corr_1y_1m),
			"leadership_tilt": float(self.leadership_tilt),
			"disp_p40": float(self.disp_p40),
			"disp_p75": float(self.disp_p75),
			"expansion_velocity": float(self.expansion_velocity),
			"velocity_ewm": float(self.velocity_ewm),
			"geometry_state": self.geometry_state,
			"expansion_state": self.expansion_state,
			"leadership_state": self.leadership_state,
			"regime_tension_flag": int(self.regime_tension_flag),
			"leadership_break_flag": int(self.leadership_break_flag),
			"persistence_break_flag": int(self.persistence_break_flag),
			"state_confidence": float(self.state_confidence),
			"universe_size": int(self.universe_size),
			"state_version": float(self.state_version),
			"disp_pct": float(self.disp_pct),
			"dvol_z": float(self.dvol_z),
			"blowoff_state": self.blowoff_state,
			"disp_off_high": float(self.disp_off_high),
			"days_since_mania": int(min(self.days_since_mania, NO_MANIA_DAYS))
		}
		try:
			db.ExecSQL(sql_insert, insert_params)
		except Exception as e:
			print(f"Error saving MarketState for {self.as_of_date}: {e}")
		finally:
			db.Close()
		
	def ToDict(self) -> dict:
		return {
			"as_of_date": self.as_of_date,
			"convex_duration": int(self.convex_duration),
			"dispersion": float(self.dispersion),
			"momentum_autocorr": float(self.momentum_autocorr),
			"downside_volatility": float(self.downside_volatility),
			"stress_index": float(self.stress_index),		
			"corr_6m_1m": float(self.corr_6m_1m),
			"corr_1y_1m": float(self.corr_1y_1m),
			"leadership_tilt": float(self.leadership_tilt),		
			"disp_p40": float(self.disp_p40),
			"disp_p75": float(self.disp_p75),
			"expansion_velocity": float(self.expansion_velocity),  # stored for EWM in smoothed state
			"geometry_state": self.geometry_state,
			"expansion_state": self.expansion_state,
			"leadership_state": self.leadership_state,
			"mode_label": self.GetModeLabel(),		
			"regime_tension_flag": int(self.regime_tension_flag),
			"leadership_break_flag": int(self.leadership_break_flag),
			"persistence_break_flag": int(self.persistence_break_flag),			
			"state_confidence": float(self.state_confidence),
			"universe_size": int(self.universe_size),
			"state_version": float(self.state_version),
			"disp_pct": float(self.disp_pct),
			"dvol_z": float(self.dvol_z),
			"blowoff_state": self.blowoff_state,
			"disp_off_high": float(self.disp_off_high),
			"days_since_mania": int(min(self.days_since_mania, NO_MANIA_DAYS)),
			"velocity_ewm": float(self.velocity_ewm)
		}

	def SaveVelocityEwmToSQL(self):
		"""4.6-M: velocity_ewm is only known after smoothing, after the row was inserted."""
		db = PTADatabase()
		if not db.Open(): return
		try:
			db.ExecSQL(f"UPDATE {CONSTANTS.ADAPTIVE_CONVEX_STATE_TABLE} SET velocity_ewm = :v WHERE asOfDate = :d AND state_version = :sv",
				{"v": float(self.velocity_ewm), "d": self.as_of_date, "sv": float(self.state_version)})
		except Exception as e:
			print(f"Error saving velocity_ewm for {self.as_of_date}: {e}")
		finally:
			db.Close()

	def ToDateFrame(self):
		df_new = pd.DataFrame([self.ToDict()])
		df_new["as_of_date"] = pd.to_datetime(df_new["as_of_date"])
		df_new = df_new.set_index("as_of_date")
		return df_new
		
class StockPicker():
	def __init__(self, startDate:pd.Timestamp =None, endDate:pd.Timestamp=None, pickHistoryWindow=60, verbose:bool = False): 
		self.pbar = None
		self.verbose = verbose
		self.priceData = []
		self._tickerList = []
		self._failedList = []
		self._droppedCache = OrderedDict()	#ticker -> PricingData for the most recently removed tickers, reused on re-add instead of reloading from SQL
		self._dropCacheSize = 50
		self._sharedDb = PTADatabase(verbose=False)	#one engine/pool shared by every PricingData this picker loads, instead of one per ticker
		if not self._sharedDb.database_configured:
			self._sharedDb = None
		if startDate:
			startDate = ToTimestamp(startDate) - pd.offsets.BusinessDay(ADAPTIVE_WARMUP_DAYS) #Add days for warmup history
		self._startDate = startDate
		if not endDate: endDate = pd.offsets.BDay().rollback(datetime.now())
		self._endDate = ToTimestamp(endDate)
		self.convex_duration = 0
		self.rebuildMarketStates = False	# True only for the explicit state build (Generate_Picks_For_SQL_DateRange adaptive + replaceExisting)
		self.saveGeneratedStates = True		# 5.7: False = a missing day's state is built in memory only (runs on a narrower universe: SP500Only, market cap, damage filter)
		self._new_state_dates = set()		# dates whose MarketStates row this picker generated (only these get velocity_ewm saved)
		self.replayPickWindow = False		# 5.7: an empty rolling pick window is first filled with the trading days of the window before the first pick day
		self.rebuildBlendedPicks = False	# True: GetPicksBlended (rolling window) writes each day's raw blended picks to PicksBlendedDaily
		self._blended_rebuild_buffer = []
		self.marketStateVersion = MARKET_STATE_VERSION	# 5.6: MarketStates rows this picker reads/writes (4.7 = two-phase build)
		self.adaptiveStockCount = 0			# 5.6: > 0 = fixed adaptive stock count (TradeModelParams.adaptiveStockCount, _S<n>)
		self.postManiaDays = 0				# 5.9: > 0 = POST_MANIA length in trading days (TradeModelParams.postManiaDays, _PMD<n>)
		self.dmaWeight = False				# 5.5: vote blend 200DMA weighting really applied (TradeModelParams.dmaWeight, _DMA)
		self.adaptiveFilterSet = 1			# 5.4: AdaptiveConvex execution filter lists (1 = 4.6 lists, 2 = FS2; TradeModelParams.adaptiveFilterSet)
		self.volAdjustedRank = False		# 5.1: momentum filters rank on PC_x / volatility (TradeModelParams.volAdjustedRank, model name _VA)
		self._adaptive_history_df = None
		self._pick_history = None
		self.pickHistoryWindowSize = pickHistoryWindow
		self._trading_calendar = None
		self._trading_calendar_count = -1

#-------------------------------------------- Housekeeping Load/Unload Tickers -----------------------------------------------
	def PrintWrapper(self, value:str):
		if self.pbar:
			tqdm.write(value)
		elif self.verbose:
			print(value)

	def AddTicker(self, ticker:str, preloadedDf:pd.DataFrame=None):
		if not ticker in self._tickerList:
			cached = self._droppedCache.pop(ticker, None)
			if cached is not None:
				self.priceData.append(cached)
				self._tickerList.append(ticker)
				return
			p = PricingData(ticker, database=self._sharedDb)
			if p.LoadHistory(self._startDate, self._endDate, verbose = self.verbose, preloadedDf=preloadedDf):
				p.CalculateStats(fullStats=False)
				self.priceData.append(p)
				self._tickerList.append(ticker)
			else:
				self._failedList.append(ticker)

	def _batch_load_prices(self, tickers:list) -> dict:
		#One SQL round trip for every ticker AlignToList needs to add this pass, instead of one query per ticker.
		#Delegates to Prices.py/DataIO.py so Selection.py doesn't reach into raw SQL itself.
		if not tickers or self._sharedDb is None:
			return {}
		loadStartDate = self._startDate - timedelta(days=CONSTANTS.REQUIRED_LOOKBACK) if self._startDate is not None else None
		return PricingData.BatchLoadHistory(tickers, loadStartDate, self._endDate, self._sharedDb, verbose=self.verbose)

	def RemoveTicker(self, ticker:str, index:int=None):
		#Owns all removals: drops the ticker from priceData and from _tickerList so it can be re-added later.
		#index is an optional hint (position in priceData); if it doesn't match the ticker the list is rescanned.
		removed = None
		if index is not None and 0 <= index < len(self.priceData) and self.priceData[index].ticker == ticker:
			removed = self.priceData.pop(index)
		else:
			i=len(self.priceData)-1
			while i >= 0:
				if ticker == self.priceData[i].ticker:
					removed = self.priceData.pop(i)
				i -=1
		self.PrintWrapper(" Removing ticker " + ticker)
		while ticker in self._tickerList:
			self._tickerList.remove(ticker)
		if ticker in self._tickerList or any(p.ticker == ticker for p in self.priceData):
			print(" Error removing ticker " + ticker)
			print(len(self.priceData))
			print(self._tickerList)
			assert(False)
		if removed is not None:
			self._droppedCache[ticker] = removed	#cache before evicting, so a ticker that bounces in/out skips the SQL reload
			self._droppedCache.move_to_end(ticker)
			while len(self._droppedCache) > self._dropCacheSize:
				self._droppedCache.popitem(last=False)

	def AlignToList(self, newList:list, currentDate=None):
		#Add/Remove tickers until they match the given list
		i=len(self.priceData)-1
		added_tickers = 0
		removed_tickers = 0
		while i >= 0:
			ticker = self.priceData[i].ticker
			if not ticker in newList:
				self.RemoveTicker(ticker, index=i)	#4.6-O: removal goes through RemoveTicker so _tickerList stays in sync
				removed_tickers +=1
			i -=1
		if len(self._tickerList) != len(self.priceData):	#the two lists must always describe the same tickers
			print(f" AlignToList: _tickerList ({len(self._tickerList)}) and priceData ({len(self.priceData)}) are out of sync")
			assert(False)
		pending = [t for t in newList if t not in self._tickerList and t not in self._failedList and t not in self._droppedCache]
		if pending:
			#Unconditional (not PrintWrapper): this is the one stretch with no pbar and no other feedback, so it needs
			#to be visible even when verbose=False, or a slow/stuck query looks indistinguishable from a hang.
			fetch_start = time.time()
			print(f" AlignToList: batch loading {len(pending)} new tickers ({', '.join(pending[:10])}{', ...' if len(pending) > 10 else ''})...")
		batch = self._batch_load_prices(pending)
		if pending:
			print(f" AlignToList: batch load finished in {time.time() - fetch_start:.1f}s ({len(batch)}/{len(pending)} tickers returned data)")
		dateLabel = f" for date ({FormatDate(currentDate)})" if currentDate is not None else ""
		self.pbar = tqdm(total=len(newList), desc=f" AlignToList adding tickers{dateLabel}")
		for t in newList:
			if t in self._failedList:
				self.PrintWrapper(f" Skipping {t} as it failed to load previously")
			elif not t in self._tickerList:
				self.AddTicker(t, preloadedDf=batch.get(t))
				added_tickers +=1
			self.pbar.update(1)
		self.PrintWrapper(f" AlignToList - Added: {added_tickers}	Removed {removed_tickers}")
		self.pbar.close()
		self.pbar = None
		
	def TickerExists(self, ticker:str):
		return ticker in self._tickerList
	
	def TickerCount(self):
		return len(self._tickerList)

	def NormalizePrices(self):
		for i in range(len(self.priceData)):
			self.priceData[i].NormalizePrices()		
			
#-------------------------------------------- Selection routine -----------------------------------------------
		
	def _rolling_history_append(self, currentDate, todays_picks, max_picks: int = 15, window_size: int = None, normalize: bool = True):
		if not window_size: window_size = self.pickHistoryWindowSize
		currentDate = pd.to_datetime(currentDate)
		cutoff=currentDate-pd.Timedelta(days=int(window_size*1.4484)) #to match SQL if desired
		#cutoff = currentDate - pd.offsets.BDay(window_size)
		if not hasattr(self, "_pick_history") or self._pick_history is None:
			self._pick_history = pd.DataFrame(columns=["as_of_date", "TargetHoldings", "Point_Value"])
		todays_picks = todays_picks[["TargetHoldings", "Point_Value"]].copy()
		todays_picks["as_of_date"] = currentDate
		self._pick_history = self._pick_history[self._pick_history["as_of_date"] != currentDate]
		self._pick_history = pd.concat([self._pick_history, todays_picks])	
		self._pick_history = self._pick_history[self._pick_history["as_of_date"] >= cutoff]
		self._pick_history = self._pick_history.sort_values(["as_of_date"])
		self._pick_history["PV_EMA"] = (self._pick_history.groupby(self._pick_history.index)["Point_Value"].transform(lambda x: x.ewm(span=3, adjust=False).mean()))
		result = self._pick_history.groupby(self._pick_history.index).agg(TargetHoldings=("TargetHoldings", "sum"),Point_Value=("PV_EMA", "last"))
		result = result.sort_values("TargetHoldings", ascending=False).head(max_picks)
		if normalize and not result.empty:
			result["TargetHoldings"] /= result["TargetHoldings"].sum()
		return result[["TargetHoldings", "Point_Value"]].copy()
	
	def _update_pick_history(self, currentDate, compute_daily_picks, max_picks=15, normalize=False):
		if not self.priceData: return CASH_RESULT.copy()	#6.0: empty universe (a _PIT list in January 1980 has no prior month): hold cash
		trading_index = self.priceData[0].historicalPrices.index
		currentDate = pd.to_datetime(currentDate)
		if not hasattr(self, "_pick_history") or self._pick_history is None or len(self._pick_history) == 0:
			if self.replayPickWindow:	#fill the window from the trading days before currentDate (same cutoff as _rolling_history_append)
				cutoff = currentDate - pd.Timedelta(days=int(self.pickHistoryWindowSize * 1.4484))
				for d in trading_index[(trading_index >= cutoff) & (trading_index < currentDate)]:
					self._rolling_history_append(currentDate=d, todays_picks=compute_daily_picks(d), max_picks=max_picks, normalize=normalize)
			todays_picks = compute_daily_picks(currentDate)
			return self._rolling_history_append(currentDate=currentDate, todays_picks=todays_picks, max_picks=max_picks, normalize=normalize)
		last_date = self._pick_history["as_of_date"].max()
		mask = (trading_index > last_date) & (trading_index <= currentDate)
		missing_days = trading_index[mask]
		result = None
		for d in missing_days:
			daily_picks = compute_daily_picks(d)
			result = self._rolling_history_append(currentDate=d, todays_picks=daily_picks, max_picks=max_picks, normalize=normalize)
		return result

	def _blend_from_multiverse(self, multiverse_candidates, filters):
		dfs = []
		for f in filters:
			if f not in multiverse_candidates: continue
			df = multiverse_candidates.get(f, pd.DataFrame())
			if df.empty: continue
			pick = df[["Point_Value"]].copy()
			pick["_vote_weight"] = 1.0

			# ── 1. PathologyScore penalty AND zero-flag boost ──────────────────────
			# Trade analysis finding: PathologyScore is a FLAG not a gradient.
			# D1 (score = exactly 0.0): 73.9% win rate.  D2-D10: 37-53%.
			# Two separate adjustments:
			#   a) Score > 1: soft penalty 0.5x (existing blow-off dampener)
			#   b) Score = 0: boost 1.25x — stocks at all-time highs, no spike history
			if "PathologyScore" in df.columns:
				ps = df["PathologyScore"].reindex(pick.index).fillna(0.5)
				pick["_vote_weight"] *= ps.map(lambda s: 0.5 if s > 1.0 else (1.25 if s == 0.0 else 1.0))

			# ── 2. DamageScore zero-flag boost ────────────────────────────────────
			# Trade analysis: DamageScore D1 (score = 0.0): 78.2% win rate vs 37-53% elsewhere.
			# Same binary pattern as PathologyScore — boost the zero case.
			if "DamageScore" in df.columns:
				dam = df["DamageScore"].reindex(pick.index).fillna(0.1)
				pick["_vote_weight"] *= dam.map(lambda d: 1.20 if d == 0.0 else 1.0)

			# ── 3. LossSkew quality gate ─────────────────────────────────────────
			# Trade analysis: strongest continuous predictor (t=8.01, p<0.001).
			# Win rate: 62.5% at LossSkew>0.5, 44.8% at -1.5/-1.0.
			# Not currently used anywhere in the engine.
			# Less negative / positive = losses were small+frequent (safer profile).
			# Very negative = concentrated large losses (fat left tail = dangerous).
			# Soft penalty only — never hard-exclude, discount approach consistent
			# with the vote architecture.
			if "LossSkew_1Year" in df.columns:
				skew = df["LossSkew_1Year"].reindex(pick.index).fillna(-0.5)
				# Penalty ramps from 1.0 at skew=-0.8 down to 0.6 at skew<=-1.5
				# Boost from 1.0 at skew=0 up to 1.20 at skew>=0.5
				skew_weight = skew.apply(lambda s:
					1.20 if s >= 0.5 else
					1.10 if s >= 0.0 else
					1.00 if s >= -0.8 else
					0.80 if s >= -1.2 else
					0.65
				)
				pick["_vote_weight"] *= skew_weight

			# ── 4. Distance from 200DMA sweet spot ───────────────────────────────
			# Trade analysis: win rate 54.8% and avg return 0.11 at 80-100% above.
			# Below 20% above 200DMA: win rate 41.6%, avg return slightly negative.
			# Apply a mild boost in the sweet spot (60-100% above) and mild
			# discount when too close to the 200DMA (below 20% extension).
			if "Distance_200DMA" in df.columns:
				dma = df["Distance_200DMA"].reindex(pick.index).fillna(0.4)
				dma_weight = dma.apply(lambda d:
					1.15 if 0.6 <= d <= 1.0 else   # sweet spot: 60-100% above
					1.05 if 0.4 <= d < 0.6  else   # reasonable zone
					0.85 if d < 0.2          else   # too close to 200DMA
					1.00                            # >100%: extended, neutral
				)
				pick["_vote_weight"] *= dma_weight

			# ── 5. Pullback vote boost ────────────────────────────────────────────
			# Stocks in a controlled pullback with intact medium trend earn a boost.
			# LossSkew gate applied inside: pullback stocks with dangerous loss
			# skew profiles (very fat left tail) are not boosted.
			# This prevents amplifying stocks that are dipping due to structural damage.
			if all(c in df.columns for c in ["LogDrawdown","PC_3Month","PC_9Month","PathologyScore","LossSkew_1Year"]):
				ld   = df["LogDrawdown"].reindex(pick.index).fillna(0)
				pc3  = df["PC_3Month"].reindex(pick.index).fillna(0)
				pc9  = df["PC_9Month"].reindex(pick.index).fillna(0)
				ps2  = df["PathologyScore"].reindex(pick.index).fillna(99)
				sk2  = df["LossSkew_1Year"].reindex(pick.index).fillna(-2.0)
				in_controlled_pullback = (
					(ld < -0.02)  & (ld > -0.15) &  # in pullback, not a breakdown
					(pc3 > 0)     &                   # 3-month trend still rising
					(pc9 > 0)     &                   # medium trend intact
					(ps2 < 1.5)   &                   # not damaged/spiky
					(sk2 > -1.2)                      # loss skew not dangerously fat-tailed
				)
				pick["_vote_weight"] *= in_controlled_pullback.map({True: 1.3, False: 1.0})
			elif all(c in df.columns for c in ["LogDrawdown","PC_3Month","PC_9Month","PathologyScore"]):
				ld  = df["LogDrawdown"].reindex(pick.index).fillna(0)
				pc3 = df["PC_3Month"].reindex(pick.index).fillna(0)
				pc9 = df["PC_9Month"].reindex(pick.index).fillna(0)
				ps2 = df["PathologyScore"].reindex(pick.index).fillna(99)
				in_controlled_pullback = (
					(ld < -0.02) & (ld > -0.15) &
					(pc3 > 0) & (pc9 > 0) & (ps2 < 1.5)
				)
				pick["_vote_weight"] *= in_controlled_pullback.map({True: 1.3, False: 1.0})

			dfs.append(pick)
		if not dfs: return CASH_RESULT.copy()
		result = pd.concat(dfs, sort=True)
		result = (result.groupby(level=0).agg(
			TargetHoldings=('_vote_weight', 'sum'),
			Point_Value=('Point_Value', 'mean')
		))
		result.sort_values('TargetHoldings', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last')
		result.index.name = 'Ticker'
		return result

	def GetHighestPriceMomentumMulti(self, currentDate: datetime, filterOptions: list):
		if not isinstance(filterOptions, list):
			raise ValueError("filterOptions must be a list of tuples like [(filter_id, stock_count)]")
		candidates = {}
		currentDate = ToTimestamp(currentDate)
		if currentDate is None or not self.priceData:	#6.0: or an empty universe (hold cash)
			for filter_option, _ in filterOptions:
				candidates[filter_option] = CASH_RESULT.copy()
			return candidates
		max_allowed_date = (pd.Timestamp.now().normalize() - pd.offsets.BusinessDay(1)).to_pydatetime()
		if currentDate > max_allowed_date:
			currentDate = max_allowed_date
		rows = []
		for pd_obj in self.priceData:
			ticker = pd_obj.ticker
			df = pd_obj.historicalPrices
			if currentDate not in df.index:
				available = df.index[df.index <= pd.Timestamp(currentDate)]
				if available.empty or (pd.Timestamp(currentDate) - available[-1]).days > 7:
					self.PrintWrapper(f" GetHighestPriceMomentumMulti: {ticker} missing {currentDate}")
					continue
				row = df.loc[available[-1]]
			else:
				row = df.loc[currentDate]
			if min(row['Average'], row['Average_5Day'] ) <= 0:
				continue
			rows.append({'Ticker': ticker,'Average_5Day': row['Average_5Day'],'Average_2Day': row['Average_2Day'],'Average': row['Average'],'PC_2Year': row['PC_2Year'],'PC_1Year': row['PC_1Year'],'PC_9Month': row['PC_9Month'],'PC_6Month': row['PC_6Month'],'PC_3Month': row['PC_3Month'],'PC_2Month': row['PC_2Month'],'PC_1Month': row['PC_1Month'],'PC_3Day': row['PC_3Day'],'PC_1Day': row['PC_1Day'],'PC_1Month3WeekEMA': row['PC_1Month3WeekEMA'],'PC_10Day5DayEMA': row['PC_10Day5DayEMA'], 'LossStd_1Year': row['LossStd_1Year'],'LossSkew_1Year': row.get('LossSkew_1Year', 0.0), 'Point_Value': row['Point_Value'], 'LogDrawdown': row['LogDrawdown'], 'PathologyScore': row['PathologyScore'], 'MaxLoss_1Year': row['MaxLoss_1Year'], 'DamageScore': row['DamageScore'], 'Distance_200DMA': (row.get('Distance200DMA', np.nan) if self.dmaWeight else 0.0), 'Volatility_1Year': row.get('Volatility_1Year', np.nan), 'Comments': row.get('Comments', ''),'latestEntry': pd_obj.historyEndDate})
		if not rows: 
			for filter_option, stocks_to_return in filterOptions:
				candidates[filter_option] = CASH_RESULT.copy()
			return candidates
		base = pd.DataFrame(rows).set_index('Ticker')
		base.index.name = 'Ticker'

		#More complex filters that I have tried have all decreased performance which is why these are simple
		#Greatest factors for improvement are high 6mo-1yr return and a very low selection of stocks, like 1-3
		#Best way to compensate for few stocks is to blend filters of different strengths
		def base_filter(df, keep_loose:bool=False):
			result = (
				(df['PC_9Month'] > 0) & 
				(df['LogDrawdown'] > -.5) & 
				(df['MaxLoss_1Year'] > -.6) & 
				(df['PathologyScore'] < 2) & 
				(df['DamageScore'] < 0.5)
			)
			if not keep_loose: result = result & (df['PC_1Month3WeekEMA'] > -.05)
			return result
		#def basic_quality(df): return ( (df['PC_9Month'] > 0) & (df['PC_1Month3WeekEMA'] > -.05) & (df['LogDrawdown'] > -.5) & (df['PathologyScore'] < 2) & (df['MaxLoss_1Year'] > -.6) & (df['DamageScore'] < 0.5) ) #The reason tight quality filters don't work: I don't buy when the filter is picked, I buy when it accumulates votes.  Letting short term loss accumuate votes is essentially shopping for a discount.
		#def loose_quality(df): return ( (df['PC_9Month'] > 0) & (df['LogDrawdown'] > -.5) & (df['PathologyScore'] < 2) & (df['MaxLoss_1Year'] > -.6) & (df['DamageScore'] < 0.5) ) 
		def f10_mask(df):	#filter 10/11 mask (pullback within an intact trend)
			return (
				(df['LogDrawdown'] < -0.02) &          # Actually in a pullback (not at highs)
				(df['LogDrawdown'] > -0.15) &           # Controlled — not a breakdown
				(df['PC_9Month'] > 0) &                 # Medium trend intact
				(df['PC_3Month'] > 0) &                 # 3-month trend still positive
				(df['PathologyScore'] < 1.5) &          # Tighter damage gate than base
				(df['DamageScore'] < 0.3) &             # No sustained multi-month damage
				(df['MaxLoss_1Year'] > -0.6)            # Max monthly loss gate from base
			)
		def vol_filled(df):	#Volatility_1Year with gaps (under 60 days of history) at the median, floored like _VA
			return df['Volatility_1Year'].fillna(df['Volatility_1Year'].median()).fillna(1.0).clip(lower=VOL_RANK_FLOOR)
		def prep12(df):	#12-1 momentum: the year's return without the last month (PC_1Month is annualized x12.5), per unit of volatility
			ex_month = (1 + df['PC_1Year']) / (1 + df['PC_1Month'] / 12.5) - 1
			return df.assign(Score12=ex_month / vol_filled(df), PC_12m1=ex_month)
		def prep13(df):	#low-volatility trend: ranks return per unit of volatility among the calmer half of today's pool
			v = vol_filled(df)
			return df.assign(Score13=df['PC_1Year'] / v, _below_median_vol=v < v.median())
		filter_map = {
			0: dict(sort='PC_1Year', mask=lambda df: (df['Average_5Day'] > 0) ),
			#trend confirmation and stability
			2: dict(sort='PC_1Year', mask=lambda df: ( (df['PC_1Month3WeekEMA'] < df['PC_1Month']) & base_filter(df, False)) ), #44.44	-61.58 Modified filter 3 recent accel
			3: dict(sort='PC_1Year', mask=lambda df: (  base_filter(df, False)) ), #61.95	-55.64
			9: dict(sort='PC_9Month', mask=lambda df:( base_filter(df, False)) ), #60.85	-64.25 Medium-trend confirmation
			5: dict(sort='Point_Value', mask=lambda df: ( (df['Point_Value'] > 0) )), #31.09	-59.82 Used to get ~46%, formula broken
			#Discovery / early growth
			1: dict(sort='PC_1Year', mask=lambda df: (df['PC_1Month3WeekEMA'] > 0) ), #52.81	-69.44	
			4: dict(sort='PC_1Month3WeekEMA', mask=lambda df: ( (df['PC_1Month3WeekEMA'] > 0) & base_filter(df, True)) ), #43.90	-59.93 Discovery and high growth but volitile
			6: dict(sort='PC_6Month', mask=lambda df: ( base_filter(df, True)) ), #59.93	-75.91
			# F7: discount reversal — never worked well as standalone; replaced by F10 + vote boost in _blend_from_multiverse
			7: dict(sort='PC_9Month', mask=lambda df: ( (df["PC_10Day5DayEMA"] < -0.01) & (df["PC_10Day5DayEMA"] > -0.12) & (df["PC_1Month3WeekEMA"] > df["PC_3Month"]) & base_filter(df, False)) ),
			8: dict(sort='PC_9Month', mask=lambda df: ( (df['PC_1Month3WeekEMA'] > df['PC_3Month'] ) & base_filter(df, True)) ), #50.13	-55.80
			# F10: Shopping/discount filter — finds controlled pullbacks within intact medium-term trends.
			# Design principles:
			#   - Never standalone: only adds votes to stocks already nominated by other filters.
			#     Architecturally safe because the vote window is the true shopping mechanism.
			#   - Sort by LogDrawdown ascending = deepest controlled pullback first (most discounted).
			#   - Pullback band [-0.15, -0.02]: excludes at-highs stocks AND real breakdowns.
			#   - PC_3Month > 0: 3-month trend still rising even as price dips = higher-low setup.
			#   - PathologyScore < 1.5: tighter than base — spike + drawdown combos excluded.
			#   - DamageScore < 0.3: tighter than base — no sustained multi-month damage.
			#   - Keeps loose EMA gate: a pullback will naturally have weak short-term EMA,
			#     that's the point. We gate damage instead.
			10: dict(sort='LogDrawdown', mask=lambda df: f10_mask(df)),
			11: dict(sort='LogDrawdown', ascending=True, mask=lambda df: f10_mask(df)),	#5.3: filter 10 ranked deepest pullback first, as filter 10's comment describes
			#5.4 - F12: 12-1 momentum per unit of volatility (skips the last month: short-term reversal)
			12: dict(sort='Score12', prep=prep12, mask=lambda df: ( (df['PC_12m1'] > 0) & base_filter(df, False) )),
			#5.4 - F13: low-volatility trend, a defensive sleeve unlike the high-volatility momentum filters
			13: dict(sort='Score13', prep=prep13, mask=lambda df: (
				df['_below_median_vol'] & (df['PC_1Year'] > 0) & (df['PC_3Month'] > -0.1) & (df['LogDrawdown'] > -0.15) &
				(df['MaxLoss_1Year'] > -0.25) & (df['PathologyScore'] < 1.5) & (df['DamageScore'] < 0.3) )),
			#5.4 - F14: at the 5-year high (52-week-high effect) with a strong year, calmest first
			14: dict(sort='Volatility_1Year', ascending=True, mask=lambda df: (
				(df['LogDrawdown'] >= -0.02) & (df['PC_1Year'] > 0.2) & (df['MaxLoss_1Year'] > -0.6) &
				(df['PathologyScore'] < 1.5) & (df['DamageScore'] < 0.3) )),
		}
		for filter_option, stocks_to_return in filterOptions:
			spec = filter_map.get(filter_option)
			vol_adjusted = self.volAdjustedRank
			if spec is None and filter_option > VA_FILTER_OFFSET:	#5.3: per-filter vol-adjusted ranking (109 = filter 9, _VA style)
				spec = filter_map.get(filter_option - VA_FILTER_OFFSET)
				vol_adjusted = True
			if spec is None: continue
			df_work = base
			if 'prep' in spec and callable(spec['prep']): df_work = spec['prep'](df_work)
			df = df_work.loc[spec['mask'](df_work)]
			df = df.loc[df[spec['sort']].notna()]	#6.5: no ranking value yet (history shorter than its lookback, no longer back-filled) - can't be ranked
			if df.empty:
				candidates[filter_option] = CASH_RESULT.copy()
				continue
			if vol_adjusted and spec['sort'].startswith('PC_'):
				#5.1 _VA: return per unit of volatility; a stock with no volatility yet (under 60 days) ranks on the median
				vol = df['Volatility_1Year'].fillna(df['Volatility_1Year'].median()).fillna(1.0).clip(lower=VOL_RANK_FLOOR)
				df = df.assign(_rank=df[spec['sort']] / vol).sort_values('_rank', ascending=False).drop(columns='_rank')
			else:
				df = df.sort_values(spec['sort'], ascending=spec.get('ascending', False))
			candidates[filter_option] = df.head(int(stocks_to_return))
		return candidates

	def GetHighestPriceMomentum(self, currentDate, stocksToReturn=10, filterOption=5, allocateByPointValue=False, useRollingWindow=False, returnRawResults=False):
		filterOptions=[(filterOption, stocksToReturn)]
		def compute_daily_picks(d):
			multiverse_candidates=self.GetHighestPriceMomentumMulti(currentDate=d, filterOptions=filterOptions)
			result=multiverse_candidates.get(filterOption, pd.DataFrame())
			if result is None or result.empty: return pd.DataFrame(columns=["TargetHoldings","Point_Value"]).rename_axis("Ticker")
			return result.groupby(level=0).agg(TargetHoldings=("Point_Value","size"), Point_Value=("Point_Value","mean"))
		if returnRawResults:
			multiverse_candidates=self.GetHighestPriceMomentumMulti(currentDate=currentDate, filterOptions=filterOptions)
			return  multiverse_candidates.get(filterOption, pd.DataFrame())
		if not useRollingWindow: 
			result=compute_daily_picks(currentDate)
		else: 
			result=self._update_pick_history(currentDate=currentDate, compute_daily_picks=compute_daily_picks, max_picks=stocksToReturn)
		if allocateByPointValue: 
			result["TargetHoldings"]*=np.log1p(result["Point_Value"])
			result["TargetHoldings"]/=result["TargetHoldings"].sum()
		return result
	
	def GetPicksBlended(self, currentDate, filterOptions=None, useRollingWindow=True, normalize=False):
		if not filterOptions: filterOptions=DEFAULT_BLEND
		filters=[item[0] for item in filterOptions]
		def compute_daily_picks(d):
			multiverse_candidates=self.GetHighestPriceMomentumMulti(currentDate=d, filterOptions=filterOptions)
			picks = self._blend_from_multiverse(multiverse_candidates, filters)
			if self.rebuildBlendedPicks: self._blended_rebuild_buffer.append((pd.Timestamp(d), picks.copy(), len(self._tickerList)))
			return picks
		if not useRollingWindow: return compute_daily_picks(currentDate)
		result = self._update_pick_history(currentDate=currentDate, compute_daily_picks=compute_daily_picks, max_picks=15, normalize=normalize)
		if self.rebuildBlendedPicks: self._flush_blended_rebuild()
		if self.verbose:
			print(result)
			print("Pick history:", currentDate, self._pick_history["as_of_date"].min(), "→", self._pick_history["as_of_date"].max(), "trading_days:", self._pick_history["as_of_date"].nunique(), "rows:", len(self._pick_history))
		return result

	def _flush_blended_rebuild(self):
		"""rebuildBlendedPicks: write the daily (un-windowed) blended picks computed since the last flush to
		PicksBlendedDaily, the same rows GeneratePicksForSQL writes (delete the date, insert its picks)."""
		if not self._blended_rebuild_buffer: return
		rows = []
		for d, picks, total_stocks in self._blended_rebuild_buffer:
			if picks is None or len(picks) == 0: continue
			picks = picks[["TargetHoldings", "Point_Value"]].copy()
			picks["Date"] = d
			picks["TotalStocks"] = total_stocks
			picks.index.name = "Ticker"
			rows.append(picks[["Date", "TargetHoldings", "Point_Value", "TotalStocks"]])
		self._blended_rebuild_buffer = []
		if not rows: return
		df = pd.concat(rows)
		db = PTADatabase()
		if not db.Open(): return
		try:
			for d in df["Date"].unique():
				db.ExecSQL("DELETE FROM PicksBlendedDaily WHERE Date = :d", {"d": pd.Timestamp(d)})
			db.DataFrameToSQL(df, tableName="PicksBlendedDaily", indexAsColumn=True, clearExistingData=False)
		finally:
			db.Close()

	def GetPicksBlendedSQL(self, currentDate:date, sqlHistory:int=90, normalize:bool=False):
		result = None
		db = PTADatabase()
		if db.Open():
			SQL = f"select * from fn_GetBlendedPicks('{currentDate}', {sqlHistory})"
			result = db.DataFrameFromSQL(sql=SQL, indexName='Ticker')
			if normalize: result["TargetHoldings"] /= result["TargetHoldings"].sum()
			#SQL = f"select Ticker, TargetHoldings, Point_Value from PicksBlendedDaily WHERE Date='{currentDate}'"
			#result = db.DataFrameFromSQL(sql=SQL, indexName='Ticker')
			#result = self._rolling_history_append(currentDate=currentDate, todays_picks=result, max_picks=15, normalize=normalize)
		db.Close()
		return result				

	def GeneratePicksForSQL(self, startDate: pd.Timestamp = None, endDate: pd.Timestamp = None, replaceExisting:bool=False, adaptiveModel:bool=False, verbose:bool=False):
		db = PTADatabase()
		filterOptions = DEFAULT_BLEND
		if not db.Open(): return False
		startDate = ToTimestamp(startDate or self._startDate)
		startDate = max(startDate, self._startDate)
		endDate = ToTimestamp(endDate or self._endDate)
		endDate = min(endDate, self._endDate)
		current_date = startDate
		prev_month = -1
		tableName = 'PicksBlendedDaily'
		if adaptiveModel:
			tableName = 'PicksAdaptiveDaily'
		print(f" GeneratePicksForSQL from {startDate} to {endDate}")				
		full_price_index = self.priceData[0].historicalPrices.index.sort_values().unique()
		mask = (full_price_index >= startDate) & (full_price_index <= endDate)
		range_price_index = full_price_index[mask]
		existing_dates = pd.to_datetime(db.ScalarListFromSQL(f"SELECT Date FROM {tableName} WHERE [Date]>=:startDate AND [Date]<=:endDate ORDER BY Date", {"startDate": startDate, "endDate": endDate}, column="Date"))
		missing_dates = range_price_index[~range_price_index.isin(existing_dates)]
		target_dates = range_price_index if replaceExisting else missing_dates
		for current_date in target_dates:
			if adaptiveModel:
				result = self.GetAdaptiveConvexPicks(currentDate=current_date) 
			else:
				result = self.GetPicksBlended(currentDate=current_date, filterOptions=filterOptions, useRollingWindow=False, normalize=False) 
			if len(result) == 0:
				if verbose: print(" GeneratePicksForSQL: No data found.")
			else:
				result['Date'] = current_date 
				result['TotalStocks'] = len(self._tickerList) 
				result = result[['Date', 'TargetHoldings', 'Point_Value', 'TotalStocks']]
				result.index.name='Ticker'
				if verbose: print(result)
				db.ExecSQL(f"DELETE FROM {tableName} WHERE Date='" + str(current_date) + "'")
				db.DataFrameToSQL(result, tableName=tableName, indexAsColumn=True, clearExistingData=False)
			result=None
		db.ExecSQL("sp_UpdateBlendedPicks")
		db.Close()
	
	def UpdateStoredPicks(self, plan: dict, dryRun: bool = False, verbose: bool = False):
		"""5.7: writes PicksBlendedDaily / PicksAdaptiveDaily with the production settings (PRODUCTION_BLEND,
		TradeModelParams.Production('adaptive')) for a StoredPicksPlan, in ONE pass over the days on this picker's prices: each month is aligned
		to that month's universe (top 250, filter by fundamentals, risk filter on) once and both models pick from it.
		Prices still missing recent days are downloaded as the tickers load (unless BLOCK_REFRESHING_FOR_BACKTESTING).
		The picker must start on or before plan['warmStart']; it ends aligned to the last month's universe, so extra
		tickers (FORCE_INCLUDE_TICKERS) are added afterwards. Adaptive: pick and state history start empty and replay from
		warmStart without writing (vote window: runs started on different days write identical picks); 4.7 states are read
		as stored, only days after the last stored state are generated (in order, DeriveStateFields); a cash row on index
		risk-off days. Recent days need STORED_PICKS_MIN_COVERAGE of the universe priced - the run stops at the first one
		that isn't. Each written month is deleted first (from the model's start). Returns {table: rows}."""
		starts, warmStart = plan['starts'], plan['warmStart']
		endDate = min(plan['endDate'], ToTimestamp(self._endDate)) if self._endDate is not None else plan['endDate']
		if self._startDate is not None and ToTimestamp(self._startDate) > warmStart:
			print(f" UpdateStoredPicks: the picker starts {FormatDate(self._startDate)}, after the replay start {FormatDate(warmStart)} - create it with startDate=plan['warmStart']")
			return None
		cfg = TradeModelParams.Production('adaptive')	#PicksAdaptiveDaily settings
		do_a, do_b = 'adaptive' in starts, 'blended' in starts
		saved = (self.adaptiveFilterSet, self.marketStateVersion, self.adaptiveStockCount)
		if do_a:
			self.adaptiveFilterSet, self.marketStateVersion, self.adaptiveStockCount = cfg.adaptiveFilterSet, cfg.marketStateVersion, cfg.adaptiveStockCount
			self._pick_history, self._adaptive_history_df, self._last_market_state, self.convex_duration = None, None, None, 0
		risk = IndexRiskSwitch(warmStart, endDate) if do_a and cfg.riskOffExposure < 1.0 else None
		cal = PricingData(CONSTANTS.CASH_TICKER)
		cal.LoadHistory(requestedStartDate=warmStart, requestedEndDate=endDate)
		dates = cal.historicalPrices.sort_index().index.unique()
		dates = dates[(dates >= warmStart) & (dates <= endDate)]
		if len(dates) and dates.max() < endDate - pd.offsets.BDay(1):
			print(f" UpdateStoredPicks: WARNING .INX prices end {FormatDate(dates.max())} - later days (partial data) are not written until .INX is updated")
		sd = plan.get('state_dates')
		a_ok = pd.Series(dates.isin(sd) | (dates > sd.max()) if (do_a and sd is not None and len(sd)) else True, index=dates)	#stored states define the history's trading days
		a_cash = pd.Series(dates < sd.min() if (do_a and sd is not None and len(sd)) else False, index=dates)	#6.6: before the first stored state (no 1-year returns yet): cash, as _get_market_state
		for m, st in starts.items():
			print(f" UpdateStoredPicks {_STORED_TABLES[m]}: writing {FormatDate(st)} to {FormatDate(endDate)}" + (f" (replay from {FormatDate(warmStart)})" if m == 'adaptive' and warmStart < st else "") + (" [dry run]" if dryRun else ""))
		today = ToTimestamp(GetLatestBDay())
		recent, first_write = today - pd.Timedelta(days=STORED_PICKS_RECENT_DAYS), min(starts.values())
		out, written, stopped = {m: [] for m in starts}, {m: 0 for m in starts}, False
		def add_row(rows, picks, d):
			if picks is None or len(picks) == 0: return
			r = picks[['TargetHoldings', 'Point_Value']].copy()
			r['Date'] = d
			r['TotalStocks'] = len(self._tickerList)
			r.index.name = 'Ticker'
			rows.append(r[['Date', 'TargetHoldings', 'Point_Value', 'TotalStocks']])
		db = PTADatabase()
		for (y, mth), month_dates in dates.to_series().groupby([dates.year, dates.month]):
			tickers = TickerLists.GetTickerListSQL(year=int(y), month=int(mth), TopN=CONSTANTS.MARKET_STATE_TOPN_TICKER_COUNT, filterByFundamentals=True, pointInTime=cfg.pointInTimeUniverse)	#6.2: point-in-time lists
			if not tickers and (int(y), int(mth)) == (1980, 1) and cfg.pointInTimeUniverse:	#6.2: January 1980 has no prior month: no list, no picks
				continue
			if not tickers:
				print(f" UpdateStoredPicks: STOPPED - no universe for {int(y)}-{int(mth):02d} (run sp_UpdateEverything first)")
				break
			self.AlignToList(tickers, currentDate=month_dates.iloc[0])
			rows = {m: [] for m in starts}
			for d in month_dates:
				if d >= recent and d >= first_write:	#recent days: the universe's prices must be in, or the state and picks are wrong
					have = sum(1 for p in self.priceData if d in p.historicalPrices.index)
					if have < STORED_PICKS_MIN_COVERAGE * len(self.priceData):
						print(f" UpdateStoredPicks: STOPPED at {FormatDate(d)} - only {have} of {len(self.priceData)} universe stocks have a price for it (refresh their prices, then run again)")
						stopped = True
						break
				if do_a and a_ok[d]:
					picks = self.GetAdaptiveConvexPicks(currentDate=d)
					if risk is not None and picks is not None and len(picks) and risk.IsRiskOff(d): picks = ApplyRiskOffExposure(picks, cfg.riskOffExposure)
					if d >= starts['adaptive']: add_row(rows['adaptive'], picks, d)
				elif do_a and a_cash[d] and d >= starts['adaptive']:
					add_row(rows['adaptive'], CASH_RESULT, d)
				if do_b and d >= starts['blended']:
					add_row(rows['blended'], self.GetPicksBlended(currentDate=d, filterOptions=PRODUCTION_BLEND, useRollingWindow=False, normalize=False), d)
			for m, rr in rows.items():
				if not rr and stopped: continue	#6.6: a month with no rows is still cleared (old rows must not survive a rebuild), unless the run stopped in it
				df = pd.concat(rr) if rr else None
				if df is not None: out[m].append(df)
				lo, hi = max(starts[m], month_dates.iloc[0].replace(day=1)), min(endDate, month_dates.iloc[0] + pd.offsets.MonthEnd(0))
				if not dryRun and lo <= hi and db.Open():
					db.ExecSQL(f"DELETE FROM {_STORED_TABLES[m]} WHERE [Date] >= :lo AND [Date] <= :hi", {"lo": lo, "hi": hi})
					if df is not None: db.DataFrameToSQL(df, tableName=_STORED_TABLES[m], indexAsColumn=True, clearExistingData=False)
					db.Close()
				if df is not None: written[m] += df['Date'].nunique()
			if verbose or mth == 12: print(f"  {int(y)}-{int(mth):02d}: " + ", ".join(f"{m} {n} days" for m, n in written.items()) + f", {len(self._tickerList)} tickers")
			if stopped: break
		self.adaptiveFilterSet, self.marketStateVersion, self.adaptiveStockCount = saved
		for m, n in written.items(): print(f" UpdateStoredPicks {_STORED_TABLES[m]}: {n} days" + (" (dry run, nothing written)" if dryRun else " written"))
		return {_STORED_TABLES[m]: (pd.concat(v) if v else None) for m, v in out.items()}

#-------------------------------------------- AdaptiveConvex Selection Engine  -----------------------------------------------

	def compute_cross_sectional_dispersion(self, base, min_valid=20):
		series = base["PC_1Year"].dropna()
		if len(series) < min_valid: return 0.0
		if ROBUST_DISPERSION:
			# Spread of log 1y returns via IQR: one outlier or bad split cannot dominate.
			logs = np.log1p(series.clip(lower=-0.95))
			return float((logs.quantile(0.75) - logs.quantile(0.25)) / 1.349)
		return float(series.std(ddof=1))

	def compute_downside_volatility(self, base, min_valid=20):
		series = base["LossStd_1Year"].dropna()
		if len(series) < min_valid: return 0.0
		return float(series.median())

	def compute_momentum_autocorr(self, base, min_valid=20):
		req = ["PC_1Month", "PC_3Month"]
		sub = base[req].dropna()
		if len(sub) < min_valid: return 0.0
		return float(sub["PC_1Month"].corr(sub["PC_3Month"], method="spearman"))

	def compute_leadership_tilt(self, df, min_valid=30):
		"""
		Returns:
			tilt (float)         : [0,1]  (1 = favor 6M, 0 = favor 12M)
			corr_6m_1m (float)   : Spearman corr(rank(PC_6Month), rank(PC_1Month3WeekEMA))
			corr_1y_1m (float)   : Spearman corr(rank(PC_1Year), rank(PC_1Month3WeekEMA))
		"""
		required = ["PC_1Year", "PC_6Month", "PC_1Month3WeekEMA"]
		for c in required:
			if c not in df.columns:
				return 0.5, 0.0, 0.0
		sub = df[required].dropna()
		if len(sub) < min_valid:
			return 0.5, 0.0, 0.0
		r1y = sub["PC_1Year"].rank(pct=True)
		r6m = sub["PC_6Month"].rank(pct=True)
		r1m = sub["PC_1Month3WeekEMA"].rank(pct=True)
		corr_6m_1m = float(r6m.corr(r1m, method="spearman"))
		corr_1y_1m = float(r1y.corr(r1m, method="spearman"))
		if np.isnan(corr_6m_1m): corr_6m_1m = 0.0
		if np.isnan(corr_1y_1m): corr_1y_1m = 0.0
		tilt_raw = corr_6m_1m - corr_1y_1m
		scale = 0.15
		tilt = 1.0 / (1.0 + np.exp(-tilt_raw / scale))
		return float(np.clip(tilt, 0.0, 1.0)), corr_6m_1m, corr_1y_1m

	def compute_stress_index(self, dispersion, autocorr, downside_vol):
		hist = getattr(self, "_adaptive_history_df", None)
		if not (hist is not None and not hist.empty and len(hist) >= 60):
			return 0.0

		# ---------------- VOLATILITY STRESS (High downside vol = stress) ----------------
		vol_mean = hist["downside_volatility"].mean()
		vol_std = hist["downside_volatility"].std()
		if vol_std <= 0 or pd.isna(vol_std):
			vol_stress = 0.0
		else:
			vol_z = (downside_vol - vol_mean) / vol_std
			vol_stress = float(np.clip(vol_z / 2.5, 0.0, 1.0))  # stress ramps if vol is > mean


		# ---------------- DISPERSION STRESS (Low dispersion = stress, high dispersion = opportunity) ----------------
		disp_mean = hist["dispersion"].mean()
		disp_std = hist["dispersion"].std()
		if disp_std <= 0 or pd.isna(disp_std):
			dispersion_stress = 0.0
		else:
			disp_z = (dispersion - disp_mean) / disp_std
			dispersion_stress = float(np.clip((-disp_z) / 2.0, 0.0, 1.0))  # only penalize low dispersion

		# ---------------- AUTOCORR STRESS (Negative autocorr = mean reversion stress) ----------------
		auto_mean = hist["momentum_autocorr"].mean()
		auto_std = hist["momentum_autocorr"].std()
		if auto_std <= 0 or pd.isna(auto_std):
			autocorr_stress = 0.0
		else:
			auto_z = (autocorr - auto_mean) / auto_std
			autocorr_stress = float(np.clip((-auto_z) / 2.0, 0.0, 1.0))  # only penalize weak/negative autocorr
		# Volatility is the "damage potential"
		# Dispersion is the "opportunity surface collapse"
		# Autocorr is the "trend break / mean reversion"
		stress = (
			0.45 * vol_stress +
			0.35 * dispersion_stress +
			0.20 * autocorr_stress
		)
		return float(np.clip(stress, 0.0, 1.0))

	def _rolling_regime_percentiles(self, lookback_days: int = STATE_HISTORY_ROWS):
		#Compute adaptive dispersion thresholds from history. Default = ~10 trading years.
		if self._adaptive_history_df is None or len(self._adaptive_history_df) < 50:
			return 0.25, 0.15  # safe warmup defaults
		hist = self._adaptive_history_df.tail(lookback_days)
		p75 = hist["dispersion"].quantile(0.75)
		p40 = hist["dispersion"].quantile(0.40)
		return float(p75), float(p40)

	def _append_adaptive_state(self, params: AdaptiveConvexMarketState):
		df_new = params.ToDateFrame()
		if self._adaptive_history_df is None or self._adaptive_history_df.empty:
			self._adaptive_history_df = df_new
		else:
			if not pd.api.types.is_datetime64_any_dtype(self._adaptive_history_df.index):
				self._adaptive_history_df.index = pd.to_datetime(self._adaptive_history_df.index)
			self._adaptive_history_df = pd.concat([self._adaptive_history_df, df_new])
		self._adaptive_history_df = (self._adaptive_history_df[~self._adaptive_history_df.index.duplicated(keep="last")].sort_index())
		# Trim by rows so the history window matches the percentile window (STATE_HISTORY_ROWS states).
		self._adaptive_history_df = self._adaptive_history_df.tail(STATE_HISTORY_ROWS)

	def _get_market_state_from_sql(self, forDate, candidate_universe_size):
		db = PTADatabase()
		if not db.Open(): return None
			
		sql = f"SELECT * FROM {CONSTANTS.ADAPTIVE_CONVEX_STATE_TABLE} WHERE asOfDate = :asOfDate AND state_version = :state_version ORDER BY record_timestamp DESC"   # universe_size is context only, not a key
		params = {
			"asOfDate": pd.Timestamp(forDate), 
			"state_version": float(self.marketStateVersion)
		}	
		df = db.DataFrameFromSQL(sql, params)
		db.Close()
		if df is None or df.empty: return None
		return self._state_from_row(df.iloc[0])

	def _first_stored_state_date(self):
		"""6.5: earliest stored state of this run's version (cached), None if there are none."""
		if getattr(self, "_first_state_cache", None) is None:
			db = PTADatabase()
			first = None
			if db.Open():
				v = db.ScalarListFromSQL(f"SELECT MIN(asOfDate) AS d FROM {CONSTANTS.ADAPTIVE_CONVEX_STATE_TABLE} WHERE state_version = :v", {"v": float(self.marketStateVersion)}, column="d")
				db.Close()
				if v and v[0] is not None: first = pd.Timestamp(v[0])
			self._first_state_cache = (first,)
		return self._first_state_cache[0]

	@staticmethod
	def _state_from_row(row) -> AdaptiveConvexMarketState:
		def f(name, default=0.0):
			v = row.get(name, default)
			return default if v is None or (isinstance(v, float) and np.isnan(v)) else v
		return AdaptiveConvexMarketState(
			as_of_date=pd.Timestamp(row["asOfDate"]),
			convex_duration=int(f("convex_duration", 0)),
			dispersion=float(f("dispersion")),
			momentum_autocorr=float(f("momentum_autocorr")),
			downside_volatility=float(f("downside_volatility")),
			stress_index=float(f("stress_index")),
			corr_6m_1m=float(f("corr_6m_1m")),
			corr_1y_1m=float(f("corr_1y_1m")),
			leadership_tilt=float(f("leadership_tilt")),
			disp_p40=float(f("disp_p40")),
			disp_p75=float(f("disp_p75")),
			expansion_velocity=float(f("expansion_velocity")),
			velocity_ewm=float(f("velocity_ewm")),
			expansion_state=str(f("expansion_state", "NO_DATA")),
			state_version=float(f("state_version", MARKET_STATE_VERSION)),
			state_confidence=float(f("state_confidence")),
			universe_size=int(f("universe_size", 0)),
			disp_pct=float(f("disp_pct", 0.5)),
			dvol_z=float(f("dvol_z")),
			disp_off_high=float(f("disp_off_high", 1.0)),
			days_since_mania=int(f("days_since_mania", NO_MANIA_DAYS)),
		)

	def _preload_state_history(self, beforeDate: pd.Timestamp):
		"""Warm up from stored states instead of a short in-memory build, so live runs use the same
		~10-year history as a continuous backtest. Not filtered on universe_size: a smaller universe
		(early 1980s) is a data flag, not a reason to break the warmup."""
		db = PTADatabase()
		if not db.Open(): return
		# One row per date (latest record) so duplicate rows for other universe sizes can't shorten the window.
		sql = (f"SELECT TOP {int(STATE_HISTORY_ROWS)} * FROM ("
			   f"SELECT *, ROW_NUMBER() OVER (PARTITION BY asOfDate ORDER BY record_timestamp DESC) AS rn "
			   f"FROM {CONSTANTS.ADAPTIVE_CONVEX_STATE_TABLE} WHERE asOfDate < :beforeDate AND state_version = :state_version"
			   ") x WHERE rn = 1 ORDER BY asOfDate DESC")
		df = db.DataFrameFromSQL(sql, {"beforeDate": pd.Timestamp(beforeDate), "state_version": float(self.marketStateVersion)})
		db.Close()
		if df is None or df.empty: return
		sort_cols = ["asOfDate"] + (["record_timestamp"] if "record_timestamp" in df.columns else [])
		df = df.sort_values(sort_cols).drop_duplicates(subset="asOfDate", keep="last").tail(STATE_HISTORY_ROWS)  # one row per date
		states = [self._state_from_row(r) for _, r in df.iterrows()]
		hist = pd.DataFrame([st.ToDict() for st in states])
		hist["as_of_date"] = pd.to_datetime(hist["as_of_date"])
		self._adaptive_history_df = hist.set_index("as_of_date").sort_index()
		self._last_market_state = states[-1]                 # so expansion velocity continues correctly
		self.convex_duration = int(states[-1].convex_duration)  # resume, don't restart at 0
		self.PrintWrapper(f" Preloaded {len(states)} stored market states through {states[-1].as_of_date.date()}")

	def _state_calendar(self) -> pd.DatetimeIndex:
		"""5.7: trading days for 4.7+ states = the .INX calendar (CASH ticker), as BuildMarketStatesRaw uses. A stock with a
		price row on a holiday (bad data: ICGT, MCIC, MRIN.DL, PSG.DL on 2026-09-07) must not add a state row - it shifts
		every later percentile window by a row - and days after the index's last price (partial data) get no state."""
		if getattr(self, "_state_cal", None) is None or self._state_cal_end < self._endDate:
			p = PricingData(CONSTANTS.CASH_TICKER)
			p.LoadHistory(requestedStartDate=pd.Timestamp('1975-01-01'), requestedEndDate=self._endDate)
			self._state_cal = p.historicalPrices.sort_index().index.unique()
			self._state_cal_end = self._endDate
		return self._state_cal

	def _trading_dates(self) -> pd.DatetimeIndex:
		"""Union of loaded price dates. Replaces pd.bdate_range, which includes market holidays."""
		if self._trading_calendar is None or self._trading_calendar_count != len(self.priceData):
			idx = pd.DatetimeIndex([])
			for p in self.priceData:
				idx = idx.union(p.historicalPrices.index)
			self._trading_calendar = idx.sort_values().unique()
			self._trading_calendar_count = len(self.priceData)
		return self._trading_calendar

	def _mania_context(self, dispersion):
		"""(disp_off_high, days_since_mania) from prior states. Mania in history is re-derived from the
		stored disp_pct/dvol_z, so it does not depend on how older rows were labeled."""
		hist = getattr(self, "_adaptive_history_df", None)
		if hist is None or hist.empty: return 1.0, NO_MANIA_DAYS
		recent = hist["dispersion"].tail(DISP_HIGH_LOOKBACK_ROWS - 1)
		peak = max(float(recent.max()) if len(recent) else dispersion, dispersion)
		off_high = float(dispersion / peak) if peak > 0 else 1.0
		days = NO_MANIA_DAYS
		if "disp_pct" in hist.columns and "dvol_z" in hist.columns:
			m = ((hist["disp_pct"] >= BLOWOFF_MANIA_PCT) & (hist["dvol_z"] >= BLOWOFF_MANIA_DVOL_Z)).to_numpy()
			idx = np.flatnonzero(m)
			if len(idx): days = int(len(m) - 1 - idx[-1])
		return off_high, days

	def _history_percentile_and_z(self, dispersion, downside_vol):
		"""Percentile of today's dispersion and z-score of downside vol vs prior stored states (excludes today)."""
		hist = getattr(self, "_adaptive_history_df", None)
		if hist is None or len(hist) < STATE_HISTORY_MIN_ROWS: return 0.5, 0.0
		h = hist.tail(STATE_HISTORY_ROWS)
		disp_pct = float((h["dispersion"] < dispersion).mean())
		mu, sd = h["downside_volatility"].mean(), h["downside_volatility"].std()
		dvol_z = 0.0 if (sd is None or pd.isna(sd) or sd <= 0) else float((downside_vol - mu) / sd)
		return disp_pct, dvol_z
	
	def _generate_market_state(self, forDate, universe_size):
		filterOptions = [(0,250)]
		multiverse_candidates = self.GetHighestPriceMomentumMulti(currentDate=forDate, filterOptions=filterOptions)
		candidate_universe = multiverse_candidates[0]
		if candidate_universe.empty or CONSTANTS.CASH_TICKER in candidate_universe.index or 'PC_1Year' not in candidate_universe.columns:
			market_state = None
		else:
			dispersion = self.compute_cross_sectional_dispersion(candidate_universe)
			autocorr = self.compute_momentum_autocorr(candidate_universe)
			downside_volatility = self.compute_downside_volatility(candidate_universe)
			stress = self.compute_stress_index(dispersion,autocorr,downside_volatility)
			leadership_tilt, corr_6m_1m, corr_1y_1m = self.compute_leadership_tilt(candidate_universe)
			disp_p75, disp_p40 = self._rolling_regime_percentiles()
			market_state = AdaptiveConvexMarketState(
				as_of_date=forDate,
				dispersion=dispersion,
				momentum_autocorr=autocorr,
				downside_volatility=downside_volatility,
				stress_index=stress,
				corr_6m_1m=corr_6m_1m,
				corr_1y_1m=corr_1y_1m,
				leadership_tilt=leadership_tilt,
				disp_p40=disp_p40,
				disp_p75=disp_p75,
				universe_size = universe_size or len(candidate_universe), #Note: this was the size requested, not necessarily the size found (0 = use the size found)
				state_version = self.marketStateVersion
				)
		return market_state

	def _compute_state_confidence_from_history(self, new_state, window=5):
		if self._adaptive_history_df is None or self._adaptive_history_df.empty:
			return 0.5  # neutral startup confidence
		hist = self._adaptive_history_df.tail(window - 1)
		modes = list(hist["mode_label"]) + [new_state.GetModeLabel()]
		dispersions = list(hist["dispersion"]) + [new_state.dispersion]
		autocorrs = list(hist["momentum_autocorr"]) + [new_state.momentum_autocorr]
		stresses = list(hist["stress_index"]) + [new_state.stress_index]
		latest_mode = modes[-1]
		label_score = modes.count(latest_mode) / len(modes)
		disp_std = np.std(dispersions)
		auto_std = np.std(autocorrs)
		stress_std = np.std(stresses)
		stability_score = 1.0 / (1.0 + 5.0 * (disp_std + auto_std + stress_std))
		persistence_score = min(new_state.convex_duration / 60.0, 1.0)
		confidence = (
			0.5 * label_score +
			0.3 * stability_score +
			0.2 * persistence_score
		)
		return float(np.clip(confidence, 0.0, 1.0))
	
	def _get_market_state(self, forDate, universe_size):
		prev_state = getattr(self, "_last_market_state", None)
		# Stored rows are read-only during backtests: they come from the explicit build
		# (PriceDatabaseUpdate.py 4 -> Generate_Picks_For_SQL_DateRange) and are used as stored, so every run sees
		# the same states whatever its start date or universe. Only a missing date (new recent days) is generated
		# and saved here. rebuildMarketStates ignores stored rows and regenerates every date.
		market_state = None if self.rebuildMarketStates else self._get_market_state_from_sql(forDate, universe_size)
		if market_state:
			if market_state.universe_size < UNIVERSE_SIZE_MIN:	#stored row was built from a shrunken universe: rebuild the market states
				print(f" Market state {pd.Timestamp(forDate).date()}: stored universe_size {market_state.universe_size} < {UNIVERSE_SIZE_MIN}. Rebuild the market states.")
				assert(False)
			self.convex_duration = int(market_state.convex_duration)
			self._last_market_state = market_state
			self._append_adaptive_state(market_state)
			return market_state
		#6.6: before the first stored state (too little history for 1-year returns: 1980 to 1981-02) - no state, hold cash.
		#Checked before the universe size: the history warm-up passes the current month's universe for earlier days
		#(a 6.6 rebuild saved January 1981 states from the February list). The explicit build is the only source of these days.
		first = None if self.rebuildMarketStates else self._first_stored_state_date()
		if first is not None and pd.Timestamp(forDate) < first:
			if not getattr(self, "_warned_state_warmup", False):
				print(f" Market state {pd.Timestamp(forDate).date()}: before the first stored {self.marketStateVersion} state ({first.date()}): holding cash")
				self._warned_state_warmup = True
			return None
		#4.6-P: a shrunken universe corrupts states; stop instead of building one. 5.9: checked only here, so a narrower
		#run (SP500Only: ~130 stocks in 1995) reads the stored canonical states instead of stopping before it gets to them.
		if universe_size < UNIVERSE_SIZE_MIN:
			print(f" Market state {pd.Timestamp(forDate).date()}: no stored state and universe_size {universe_size} < {UNIVERSE_SIZE_MIN}, can't build one. Loaded tickers: {len(self.priceData)}")
			assert(False)
		market_state = self._generate_market_state(forDate, universe_size)
		if not market_state: return None
		if self.marketStateVersion >= MARKET_STATE_VERSION_2PHASE:	#5.6: same rule as the phase 2 build (DeriveStateFields)
			DeriveStateFields(market_state, self._adaptive_history_df)
			self.convex_duration = market_state.convex_duration
			if self.saveGeneratedStates: market_state.SaveToSQL()
			self._new_state_dates.add(pd.Timestamp(market_state.as_of_date))
			self._last_market_state = market_state
			self._append_adaptive_state(market_state)
			return market_state
		market_state.set_expansion_state(prev_state)
		market_state.disp_pct, market_state.dvol_z = self._history_percentile_and_z(market_state.dispersion, market_state.downside_volatility)
		market_state.disp_off_high, market_state.days_since_mania = self._mania_context(market_state.dispersion)
		if market_state.universe_size < UNIVERSE_SIZE_MIN:	#stored row was built from a shrunken universe: delete and rebuild the states
			print(f" Market state {pd.Timestamp(forDate).date()}: stored universe_size {market_state.universe_size} < {UNIVERSE_SIZE_MIN}. Rebuild the market states.")
			assert(False)
		if market_state.universe_size < UNIVERSE_SIZE_WARN:
			self.PrintWrapper(f" Market state {pd.Timestamp(forDate).date()}: universe_size {market_state.universe_size} < {UNIVERSE_SIZE_WARN} (check for missing prices)")
		confidence = self._compute_state_confidence_from_history(market_state)
		if market_state.geometry_state == "CONVEX":
			self.convex_duration += 1
		else:
			self.convex_duration = 0			
		market_state.convex_duration = self.convex_duration
		market_state.state_confidence = confidence
		if self.saveGeneratedStates: market_state.SaveToSQL()
		self._new_state_dates.add(pd.Timestamp(market_state.as_of_date))
		self._last_market_state = market_state		
		self._append_adaptive_state(market_state)
		return market_state		

	def _hydrate_market_state_history(self, toDate: pd.Timestamp, universe_size):
		if self._adaptive_history_df is None or self._adaptive_history_df.empty:
			self._preload_state_history(toDate)
		lookback_start = toDate - pd.offsets.BDay(ADAPTIVE_WARMUP_DAYS)
		if self._adaptive_history_df is not None and not self._adaptive_history_df.empty:
			last_date = self._adaptive_history_df.index.max()
			start_date = max(last_date + pd.offsets.BDay(1), lookback_start)
		else:
			start_date = lookback_start
		cal = self._state_calendar() if self.marketStateVersion >= MARKET_STATE_VERSION_2PHASE else self._trading_dates()	#5.7
		if len(cal):
			prior = cal[cal <= toDate]
			if len(prior): toDate = prior[-1]          # holiday/weekend -> last trading day
		bdays = cal[(cal >= start_date) & (cal < toDate)]   # trading days only (no holidays), excluding toDate
		for d in bdays:
			if self._adaptive_history_df is not None and d in self._adaptive_history_df.index:
				continue
			state = self._get_market_state(d, universe_size)
			if state is None: continue
		# Process toDate exactly once: a second pass would compare the state with itself
		# (zero expansion velocity) and count convex_duration twice.
		last = getattr(self, "_last_market_state", None)
		if last is not None and pd.Timestamp(last.as_of_date) == pd.Timestamp(toDate):
			return last
		return self._get_market_state(toDate, universe_size)

	def _get_market_state_smoothed(self, forDate, universe_size):
		current_state = self._hydrate_market_state_history(forDate, universe_size)
		if current_state is None or self._adaptive_history_df is None or self._adaptive_history_df.empty:
			return current_state
		window_size = 5
		hist = self._adaptive_history_df.tail(window_size)
		if hist.empty:
			return current_state
		smoothed = current_state
		smoothed.dispersion        = hist["dispersion"].mean()
		smoothed.momentum_autocorr = hist["momentum_autocorr"].mean()
		smoothed.leadership_tilt   = hist["leadership_tilt"].mean()
		smoothed.state_confidence  = hist["state_confidence"].mean()
		# Compute EWM of expansion_velocity over the history window.
		# A single day of negative velocity is noise (Spearman 0.018 at 21d).
		# Sustained negative velocity over 5 days is a genuine regime deterioration
		# signal that should reduce conviction and concentration.
		# span=5 gives ~50% weight to the most recent 2 days, decaying smoothly.
		if "expansion_velocity" in hist.columns:
			vel_series = hist["expansion_velocity"].dropna()
			if len(vel_series) >= 2:
				smoothed.velocity_ewm = float(
					vel_series.ewm(span=5, adjust=False).mean().iloc[-1]
				)
			elif len(vel_series) == 1:
				smoothed.velocity_ewm = float(vel_series.iloc[-1])
			# else: leave at dataclass default 0.0
		# 4.6-M: persist velocity_ewm (row was inserted before smoothing) and keep history in sync.
		if self._adaptive_history_df is not None and smoothed.as_of_date in self._adaptive_history_df.index:
			self._adaptive_history_df.loc[smoothed.as_of_date, "velocity_ewm"] = smoothed.velocity_ewm
		if self.marketStateVersion < MARKET_STATE_VERSION_2PHASE and pd.Timestamp(smoothed.as_of_date) in self._new_state_dates and getattr(self, "_last_saved_velocity_date", None) != smoothed.as_of_date:
			if self.saveGeneratedStates: smoothed.SaveVelocityEwmToSQL()
			self._last_saved_velocity_date = smoothed.as_of_date
		return smoothed
	
	def GetAdaptiveConvexPicks(self, currentDate):
		# Pre-fetch all filters any regime path may need.
		# F4 (acceleration): EXPANDING CONVEX only, fetched at 3 stocks.
		# F8 (accel vs trend quality): replaces F6 across all blends — better DD, similar CAGR.
		# F10 (shopping/discount): controlled pullback + intact trend; never generates cash
		#      because it only amplifies votes already cast by other filters via vote boost in
		#      _blend_from_multiverse. Including it in filterOptions here pre-fetches candidates
		#      so the vote boost logic has PathologyScore/LogDrawdown/PC_3Month available.
		if len(self._tickerList) == 0: return CASH_RESULT.copy()	#6.1: empty list (a _PIT list in January 1980 has no prior month): no state can be built
		filterOptions = [(0,250),(4,3),(8,3),(10,5)] + DEFAULT_BLEND
		if self.adaptiveFilterSet >= 2: filterOptions += [(6,3),(9 + VA_FILTER_OFFSET,3),(4 + VA_FILTER_OFFSET,3)]	#set 2 votes with F6 at 3 stocks (DEFAULT_BLEND has it at 1), 9v and 4v
		multiverse_candidates = self.GetHighestPriceMomentumMulti(currentDate=currentDate, filterOptions=filterOptions)
		universe_size = len(multiverse_candidates[0])
		market_state = self._get_market_state_smoothed(currentDate, universe_size)
		if not market_state:
			return CASH_RESULT.copy()
		if self.postManiaDays > 0: market_state.post_mania_days = self.postManiaDays	#5.9 _PMD
		filters = market_state.GetExecutionFilters(filterSet=self.adaptiveFilterSet)
		window_size = market_state.GetRollingWindowSize()
		stock_count = self.adaptiveStockCount if self.adaptiveStockCount > 0 else market_state.GetStockCount()	#5.6 _S<n>
		todays_picks = self._blend_from_multiverse(multiverse_candidates, filters)
		todays_picks = self._rolling_history_append(
			currentDate=currentDate,
			todays_picks=todays_picks,
			window_size=window_size,
			max_picks=stock_count
		)
		max_w = market_state.GetMaxWeight()
		if max_w is not None:
			todays_picks = self._cap_weights(todays_picks, max_w)   # 4.6-I
		return todays_picks

	@staticmethod
	def _cap_weights(picks, max_w):
		"""Cap each TargetHoldings at max_w and spread the excess over uncapped names (cash row untouched).
		If there are too few names for the cap to sum to 1, falls back to equal weight."""
		if picks is None or picks.empty: return picks
		picks = picks.copy()
		mask = picks.index != CONSTANTS.CASH_TICKER
		w = picks.loc[mask, "TargetHoldings"].astype(float)
		total = w.sum()
		if len(w) == 0 or total <= 0: return picks
		cap = max(max_w, 1.0 / len(w)) * total
		for _ in range(20):
			over = w > cap + 1e-12
			if not over.any(): break
			excess = (w[over] - cap).sum()
			w[over] = cap
			under = ~over & (w < cap)
			if not under.any(): break
			w[under] += excess * w[under] / w[under].sum()
		picks.loc[mask, "TargetHoldings"] = w
		return picks

def Generate_Picks_For_SQL_DateRange(startYear:int=None, years: int=0, TopN=CONSTANTS.MARKET_STATE_TOPN_TICKER_COUNT, replaceExisting:bool=False, adaptiveModel:bool=False, verbose:bool=False):
	db = PTADatabase()
	if db.Open():
		today = ToTimestamp(GetLatestBDay())
		if startYear== None:
			startYear = today.year
			endDate = (today - pd.offsets.BDay(1)).date() 
		else:
			endDate = ToTimestamp('12/31/' + str(startYear+years))	
		startDate = ToTimestamp('1/1/' + str(startYear))		
		endDate = ToTimestamp(endDate)		
		if endDate > today: endDate = today
		if startDate > today: startDate = today
		current_date = startDate
		tickers = []
		tableName = 'PicksBlendedDaily'
		if adaptiveModel:
			tableName = 'PicksAdaptiveDaily'
		params = TradeModelParams()
		params.filterByFundamentals = True
		print(f" Generate_Picks_For_SQL_DateRange from {FormatDate(startDate)} to {FormatDate(endDate)}")				
		picker = StockPicker(startDate=startDate, endDate=endDate)
		picker.rebuildMarketStates = adaptiveModel and replaceExisting	# the one place MarketStates rows are regenerated (backtests only add missing dates)
		p = PricingData(CONSTANTS.CASH_TICKER)
		p.LoadHistory(requestedStartDate=startDate, requestedEndDate=endDate)
		full_price_index = p.historicalPrices.sort_index().index.unique()
		existing_dates = pd.to_datetime(db.ScalarListFromSQL(f"SELECT Date FROM {tableName} WHERE [Date]>=:startDate AND [Date]<=:endDate ORDER BY Date", {"startDate": startDate, "endDate": endDate}, column="Date"))
		missing_dates = full_price_index[~full_price_index.isin(existing_dates)]
		target_dates = full_price_index if replaceExisting else missing_dates
		monthly_starts = target_dates[target_dates.to_series().dt.month != target_dates.to_series().dt.month.shift()]
		for month_start in monthly_starts:
			month_end = month_start + pd.offsets.MonthEnd(0)
			if verbose: print(f" Generate_PicksBlended_DateRange: Getting tickers for month {FormatDate(month_start)}")				
			new_tickers = TickerLists.GetTickerListSQL(year=month_start.year, month=month_start.month, TopN=TopN, SP500Only=params.SP500Only, filterByFundamentals=params.filterByFundamentals, marketCapMin=params.marketCapMin, marketCapMax=params.marketCapMax) 
			if len(new_tickers) > 0:
				if verbose: print(f" Generate_Picks_For_SQL_DateRange: Re-query tickers found {len(new_tickers)} instead of previous {len(tickers)}")
				tickers = new_tickers
			picker.AlignToList(tickers)			
			TotalValidCandidates = len(picker._tickerList) 
			if verbose: print(f" Generate_Picks_For_SQL_DateRange: Running PicksBlended generation on {TotalValidCandidates} stocks {FormatDate(month_start)} to {FormatDate(month_end)}")		
			if TotalValidCandidates==0: assert(False)
			picker.GeneratePicksForSQL(startDate=month_start, endDate=month_end, replaceExisting=replaceExisting, adaptiveModel=adaptiveModel, verbose=verbose)
	db.Close()

#-------------------------------------------- 5.7: stored picks tables (PicksBlendedDaily, PicksAdaptiveDaily) -----------------------------------------------
_STORED_TABLES = {'blended': 'PicksBlendedDaily', 'adaptive': 'PicksAdaptiveDaily'}

def StoredPicksPlan(models=('blended', 'adaptive'), startDate=None, endDate=None, redoDays: int = 3, regenerateStates: bool = False, dryRun: bool = False):
	"""What StockPicker.UpdateStoredPicks will write: {'starts': {model: first written day}, 'warmStart', 'endDate',
	'state_dates'}. Create the picker with startDate=plan['warmStart'] (its prices are trimmed to its own start).
	startDate None (refresh): per model, the first day of the month of its redoDays-th last stored day - a month's universe
	(TickerHistoricalQualityFactors) moves while the month is current, and late prices get picked up - and the adaptive
	4.7 states from there are deleted, to be rebuilt in order as the picks run. regenerateStates: delete them from
	startDate as well. A dry run deletes nothing and stops at the last stored state (it must not save new ones)."""
	cfg = TradeModelParams.Production('adaptive')	#PicksAdaptiveDaily settings
	db = PTADatabase()
	if not db.Open(): return None
	today = ToTimestamp(GetLatestBDay())
	endDate = min(ToTimestamp(endDate), today) if endDate is not None else today
	starts = {}
	for m in models:
		if startDate is not None:
			st = ToTimestamp(startDate)
		else:
			last = db.ScalarListFromSQL(f"SELECT TOP {max(int(redoDays), 1)} Date FROM {_STORED_TABLES[m]} GROUP BY Date ORDER BY Date DESC")
			if not last:
				print(f" UpdateStoredPicks: {_STORED_TABLES[m]} is empty; give a startDate (or queue the rebuild)")
				continue
			st = ToTimestamp(last[-1]).replace(day=1)
		if st > endDate:
			print(f" UpdateStoredPicks {_STORED_TABLES[m]}: nothing to do ({FormatDate(st)} > {FormatDate(endDate)})")
			continue
		starts[m] = st
	state_dates = None
	if 'adaptive' in starts:
		if (regenerateStates or startDate is None) and not dryRun:
			db.ExecSQL(f"DELETE FROM {CONSTANTS.ADAPTIVE_CONVEX_STATE_TABLE} WHERE state_version = :v AND asOfDate >= :s", {"v": float(cfg.marketStateVersion), "s": starts['adaptive']})
			print(f" UpdateStoredPicks: market states {cfg.marketStateVersion} from {FormatDate(starts['adaptive'])} deleted; rebuilt in order as the picks run")
		state_dates = pd.to_datetime(db.ScalarListFromSQL(f"SELECT DISTINCT asOfDate FROM {CONSTANTS.ADAPTIVE_CONVEX_STATE_TABLE} WHERE state_version = :v", {"v": float(cfg.marketStateVersion)}))
	db.Close()
	if not starts: return None
	if 'adaptive' in starts and dryRun and len(state_dates): endDate = min(endDate, state_dates.max())
	warmStart = min(starts.values())
	if 'adaptive' in starts:
		first_state = state_dates.min() if len(state_dates) else starts['adaptive']
		warmStart = min(warmStart, max(starts['adaptive'] - pd.Timedelta(days=STORED_PICKS_WARMUP_DAYS), first_state))
	return dict(starts=starts, warmStart=warmStart, endDate=endDate, state_dates=state_dates)

#-------------------------------------------- 5.6: two-phase MarketStates build (state version 4.7) -----------------------------------------------
_DERIVED_COLUMNS = ["stress_index", "disp_p40", "disp_p75", "expansion_velocity", "velocity_ewm", "geometry_state", "expansion_state", "leadership_state",
	"regime_tension_flag", "leadership_break_flag", "persistence_break_flag", "state_confidence", "disp_pct", "dvol_z", "blowoff_state",
	"disp_off_high", "days_since_mania", "convex_duration"]
_HIST_COLUMNS = ["dispersion", "momentum_autocorr", "downside_volatility", "stress_index", "disp_pct", "dvol_z", "expansion_velocity", "convex_duration", "mode_label"]

def DeriveStateFields(state: AdaptiveConvexMarketState, hist: pd.DataFrame):
	"""Fills a state's history-dependent fields from its raw measures and the PRIOR stored states (hist: rows before
	the state's date, oldest first, derived fields already filled; columns as AdaptiveConvexMarketState.ToDict, mode_label
	included). The one rule for 4.7 states: used by the phase 2 build and for a missing day during a 4.7 backtest.
	Same formulas as the 4.6 daily build (previous day = 5-day smoothed state), applied to every day (see the 5.6 note)."""
	h = hist.tail(STATE_HISTORY_ROWS) if hist is not None and len(hist) else None
	n = 0 if h is None else len(h)
	#5.7: stress_index reads the history too (z-scores vs prior states, 0 under 60 rows). Phase 1 had no history, so every
	#5.6-built 4.7 row stored 0; it feeds the conviction score (stock count) and state_confidence.
	state.stress_index = StockPicker.compute_stress_index(SimpleNamespace(_adaptive_history_df=h), state.dispersion, state.momentum_autocorr, state.downside_volatility) if n else 0.0
	if n < 50:
		state.disp_p75, state.disp_p40 = 0.25, 0.15	#same warmup defaults as _rolling_regime_percentiles
	else:
		disp = h["dispersion"].astype(float)
		state.disp_p75, state.disp_p40 = float(disp.quantile(0.75)), float(disp.quantile(0.40))
	state.expansion_state, state.expansion_velocity = "STABLE", 0.0
	prev = h.iloc[-1] if n else None
	if prev is not None:
		#Yesterday as the 4.6 daily build saw it: the 5-day smoothed state (mean of the 5 rows through yesterday)
		p5 = h.tail(5)
		d_autocorr = state.momentum_autocorr - float(p5["momentum_autocorr"].astype(float).mean())
		d_disp = state.dispersion - float(p5["dispersion"].astype(float).mean())
		d_disp_norm = d_disp / max(state.disp_p75 - state.disp_p40, 1e-6)
		state.expansion_velocity = max(-1.0, min(1.0, 0.6 * d_autocorr + 0.4 * d_disp_norm))
		if d_autocorr > 0 and d_disp > 0: state.expansion_state = "EXPANDING"
		elif d_autocorr < 0 and d_disp < 0: state.expansion_state = "CONTRACTING"
	if n < STATE_HISTORY_MIN_ROWS:
		state.disp_pct, state.dvol_z = 0.5, 0.0
	else:
		dv = h["downside_volatility"].astype(float)
		state.disp_pct = float((h["dispersion"].astype(float) < state.dispersion).mean())
		mu, sd = dv.mean(), dv.std()
		state.dvol_z = 0.0 if (sd is None or pd.isna(sd) or sd <= 0) else float((state.downside_volatility - mu) / sd)
	if n:
		recent = h["dispersion"].astype(float).tail(DISP_HIGH_LOOKBACK_ROWS - 1)
		peak = max(float(recent.max()), state.dispersion)
		state.disp_off_high = float(state.dispersion / peak) if peak > 0 else 1.0
		m = ((h["disp_pct"].astype(float) >= BLOWOFF_MANIA_PCT) & (h["dvol_z"].astype(float) >= BLOWOFF_MANIA_DVOL_Z)).to_numpy()
		idx = np.flatnonzero(m)
		state.days_since_mania = int(len(m) - 1 - idx[-1]) if len(idx) else NO_MANIA_DAYS
	else:
		state.disp_off_high, state.days_since_mania = 1.0, NO_MANIA_DAYS
	#state_confidence as 4.6 computes it: today's label at confidence 0 and convex_duration 0 (logging only)
	state.state_confidence, state.convex_duration = 0.0, 0
	if n:
		t = h.tail(4)
		modes = list(t["mode_label"]) + [state.GetModeLabel()]
		disp_std = np.std([float(x) for x in t["dispersion"]] + [state.dispersion])
		auto_std = np.std([float(x) for x in t["momentum_autocorr"]] + [state.momentum_autocorr])
		stress_std = np.std([float(x) for x in t["stress_index"]] + [state.stress_index])
		label_score = modes.count(modes[-1]) / len(modes)
		stability_score = 1.0 / (1.0 + 5.0 * (disp_std + auto_std + stress_std))
		state.state_confidence = float(np.clip(0.5 * label_score + 0.3 * stability_score, 0.0, 1.0))
	else:
		state.state_confidence = 0.5
	if state.geometry_state == "CONVEX":
		state.convex_duration = (int(prev["convex_duration"]) if prev is not None else 0) + 1
	vel = ([float(x) for x in h["expansion_velocity"].tail(4)] if n else []) + [state.expansion_velocity]
	state.velocity_ewm = float(pd.Series(vel).ewm(span=5, adjust=False).mean().iloc[-1])
	return state

def BuildMarketStatesRaw(startDate, endDate, stateVersion: float = MARKET_STATE_VERSION_2PHASE, verbose: bool = False):
	"""Phase 1: raw measures for every trading day in [startDate, endDate), canonical universe (top 250, filter by
	fundamentals, risk filter on; 6.1: point-in-time lists for 4.8+). Derived fields are placeholders until DeriveMarketStates
	runs. Chunks of dates are independent, so they can run on several workers at once (queue_backtest.py --states-raw)."""
	assert stateVersion >= MARKET_STATE_VERSION_2PHASE, "only two-phase state versions are built in pieces"
	startDate, endDate = ToTimestamp(startDate), ToTimestamp(endDate)
	today = ToTimestamp(GetLatestBDay())
	if endDate > today + pd.offsets.BDay(1): endDate = today + pd.offsets.BDay(1)
	db = PTADatabase()
	if not db.Open(): return 0
	db.ExecSQL(f"DELETE FROM {CONSTANTS.ADAPTIVE_CONVEX_STATE_TABLE} WHERE state_version = :v AND asOfDate >= :s AND asOfDate < :e",
		{"v": float(stateVersion), "s": startDate, "e": endDate})
	db.Close()
	picker = StockPicker(startDate=startDate, endDate=endDate)
	picker.marketStateVersion = stateVersion
	cal = PricingData(CONSTANTS.CASH_TICKER)
	cal.LoadHistory(requestedStartDate=startDate, requestedEndDate=endDate)
	dates = cal.historicalPrices.sort_index().index.unique()
	dates = dates[(dates >= startDate) & (dates < endDate)]
	saved = skipped = 0
	print(f" BuildMarketStatesRaw {stateVersion}: {FormatDate(startDate)} to {FormatDate(endDate)} ({len(dates)} trading days)")
	for (y, mth), month_dates in dates.to_series().groupby([dates.year, dates.month]):
		tickers = TickerLists.GetTickerListSQL(year=int(y), month=int(mth), TopN=CONSTANTS.MARKET_STATE_TOPN_TICKER_COUNT, filterByFundamentals=True, pointInTime=stateVersion >= MARKET_STATE_VERSION_PIT)
		picker.AlignToList(tickers, currentDate=month_dates.iloc[0])
		for d in month_dates:
			st = picker._generate_market_state(d, 0)
			if st is None or st.universe_size < UNIVERSE_SIZE_MIN:
				print(f"  {pd.Timestamp(d).date()}: no state (universe {0 if st is None else st.universe_size}) - skipped")
				skipped += 1
				continue
			st.SaveToSQL()
			saved += 1
		print(f"  {int(y)}-{int(mth):02d}: {saved} saved, {skipped} skipped, {len(picker._tickerList)} tickers loaded")
	print(f" BuildMarketStatesRaw done: {saved} states saved, {skipped} skipped")
	return saved

def DeriveMarketStates(stateVersion: float = MARKET_STATE_VERSION_2PHASE, verbose: bool = True):
	"""Phase 2: one serial pass over every stored row of the version, oldest first, filling the derived fields with
	DeriveStateFields. No price loading. Run after every BuildMarketStatesRaw chunk has finished."""
	db = PTADatabase()
	if not db.Open(): return 0
	df = db.DataFrameFromSQL(f"SELECT * FROM {CONSTANTS.ADAPTIVE_CONVEX_STATE_TABLE} WHERE state_version = :v ORDER BY asOfDate", {"v": float(stateVersion)})
	if df is None or df.empty:
		print(f" DeriveMarketStates: no rows for state version {stateVersion}")
		db.Close(); return 0
	df = df.sort_values(["asOfDate", "record_timestamp"]).drop_duplicates(subset="asOfDate", keep="last").reset_index(drop=True)
	gaps = pd.to_datetime(df["asOfDate"]).diff().dt.days
	big = df.loc[gaps > 7, "asOfDate"]
	if len(big): print(f" DeriveMarketStates: {len(big)} gaps over a week in the raw rows, first ones: {[str(x)[:10] for x in big.head(5)]}")
	hist = pd.DataFrame({c: pd.Series([np.nan] * len(df), dtype=float) for c in _HIST_COLUMNS})
	hist["mode_label"] = ""
	out = []
	for i, row in df.iterrows():
		st = StockPicker._state_from_row(row)
		DeriveStateFields(st, hist.iloc[max(0, i - STATE_HISTORY_ROWS):i])
		d = st.ToDict()
		for c in _HIST_COLUMNS: hist.at[i, c] = d[c]
		out.append({"asOfDate": row["asOfDate"], **{c: d[c] for c in _DERIVED_COLUMNS}})
		if verbose and i % 2000 == 0: print(f"  {str(row['asOfDate'])[:10]} {st.GetRegimeSummary()}")
	upd = pd.DataFrame(out)
	stage = "MarketStates_derive_stage"
	db.ExecSQL(f"IF OBJECT_ID('dbo.{stage}') IS NOT NULL DROP TABLE dbo.{stage}")
	db.DataFrameToSQL(upd, tableName=stage)
	set_clause = ", ".join(f"m.{c} = s.{c}" for c in _DERIVED_COLUMNS)
	db.ExecSQL(f"UPDATE m SET {set_clause} FROM {CONSTANTS.ADAPTIVE_CONVEX_STATE_TABLE} m JOIN dbo.{stage} s ON m.asOfDate = s.asOfDate WHERE m.state_version = :v", {"v": float(stateVersion)})
	db.ExecSQL(f"DROP TABLE dbo.{stage}")
	db.Close()
	print(f" DeriveMarketStates {stateVersion}: {len(upd)} rows, {str(df['asOfDate'].iloc[0])[:10]} to {str(df['asOfDate'].iloc[-1])[:10]}")
	return len(upd)

#-------------------------------------------- 5.6: index 200-day risk-off switch -----------------------------------------------
class IndexRiskSwitch:
	"""Risk-off when the index closes more than band below its window-day average; back on when it closes more than band
	above. IsRiskOff(date) reads the PRIOR trading day's close, so the signal is known before the day's picks."""
	def __init__(self, startDate, endDate, ticker: str = '.INX', window: int = 200, band: float = 0.02):
		p = PricingData(ticker)
		p.LoadHistory(requestedStartDate=ToTimestamp(startDate) - pd.DateOffset(days=int(window * 1.6) + 30), requestedEndDate=ToTimestamp(endDate))
		close = p.historicalPrices['Close'].astype(float).sort_index()
		ma = close.rolling(window).mean()
		state, cur = [], False
		for c, m in zip(close.to_numpy(), ma.to_numpy()):
			if m == m:
				if not cur and c < m * (1 - band): cur = True
				elif cur and c > m * (1 + band): cur = False
			state.append(cur)
		self.state = pd.Series(state, index=close.index)
	def IsRiskOff(self, date) -> bool:
		prior = self.state[self.state.index < ToTimestamp(date)]
		return bool(prior.iloc[-1]) if len(prior) else False

#-------------------------------------------- 5.9: momentum-leader risk-off switch -----------------------------------------------
LEADER_INDEX_TABLE = "LeaderIndexDaily"
LEADER_COUNT = 25			#leaders per day (top PC_6Month of the month's canonical universe)
LEADER_LOOKBACK = 126		#trading days for PC_6Month
LEADER_SWITCH_WINDOW = 100	#moving average the leader index is compared with (leaders move ~2x the index: a shorter average than .INX's 200)
LEADER_RETURN_CLIP = (-0.5, 1.0)	#one day's return per stock, guards bad prices

def BuildLeaderIndex(startYear: int = 1980, endYear: int = None, verbose: bool = True) -> int:
	"""Fills LeaderIndexDaily (Date, LeaderReturn, LeaderCount) for .INX trading days not yet stored: each day the
	LEADER_COUNT stocks of that month's canonical universe (sp_GetTickerList top 250, FF + risk filter, 6.2: point-in-time) with the highest
	PC_6Month at the PRIOR day's close, equal-weighted, and their return from the prior close to today's. Computed once;
	existing dates are left alone. Returns the rows added."""
	endYear = endYear or ToTimestamp(GetLatestBDay()).year
	db = PTADatabase()
	if not db.Open(): return 0
	db.ExecSQL(f"IF OBJECT_ID('dbo.{LEADER_INDEX_TABLE}') IS NULL CREATE TABLE dbo.{LEADER_INDEX_TABLE} ([Date] date NOT NULL PRIMARY KEY, LeaderReturn float NOT NULL, LeaderCount int NOT NULL)")
	have = set(pd.to_datetime(db.ScalarListFromSQL(f"SELECT [Date] FROM dbo.{LEADER_INDEX_TABLE}", column="Date") or []))
	inx = db.DataFrameFromSQL("SELECT [Date] FROM PricesDaily WHERE Ticker IN ('.INX') ORDER BY [Date]")
	cal = pd.DatetimeIndex(pd.to_datetime(inx['Date'])).unique().sort_values()
	added = 0
	for y in range(startYear, endYear + 1):
		days = cal[(cal.year == y) & ~cal.isin(have)]
		if not len(days): continue
		universe = {}
		for m in sorted(set(days.month)):
			universe[m] = TickerLists.GetTickerListSQL(year=y, month=int(m), TopN=CONSTANTS.MARKET_STATE_TOPN_TICKER_COUNT, filterByFundamentals=True, pointInTime=True)	#6.2: point-in-time lists
		tickers = sorted({t for v in universe.values() for t in v})
		if not tickers: continue
		lo = days.min() - pd.Timedelta(days=int(LEADER_LOOKBACK * 1.6) + 20)
		frames = []
		for i in range(0, len(tickers), 300):	#literal IN lists (no ? parameters for Ticker on PricesDaily)
			lst = ",".join("'" + t.replace("'", "''") + "'" for t in tickers[i:i + 300])
			frames.append(db.DataFrameFromSQL(f"SELECT Ticker, [Date], [Close] FROM PricesDaily WHERE Ticker IN ({lst}) AND [Date] >= '{lo:%Y-%m-%d}' AND [Date] <= '{days.max():%Y-%m-%d}'"))
		px = pd.concat(frames)
		px['Date'] = pd.to_datetime(px['Date'])
		close = px.pivot_table(index='Date', columns='Ticker', values='Close', aggfunc='last').reindex(cal[(cal >= lo) & (cal <= days.max())]).ffill(limit=5)
		pc6 = close / close.shift(LEADER_LOOKBACK) - 1
		ret = (close / close.shift(1) - 1).clip(*LEADER_RETURN_CLIP)
		rows = []
		for d in days:
			pos = close.index.get_loc(d)
			if pos == 0: continue
			prev = close.index[pos - 1]
			cand = [t for t in universe.get(d.month, []) if t in pc6.columns]
			r = pc6.loc[prev, cand].dropna().nlargest(LEADER_COUNT)
			r_today = ret.loc[d, r.index].dropna()
			if len(r_today) < LEADER_COUNT // 2: continue
			rows.append({'Date': d, 'LeaderReturn': float(r_today.mean()), 'LeaderCount': int(len(r_today))})
		if rows:
			db.DataFrameToSQL(pd.DataFrame(rows), tableName=LEADER_INDEX_TABLE, clearExistingData=False)
			added += len(rows)
		if verbose: print(f"  BuildLeaderIndex {y}: {len(rows)} days, {len(tickers)} tickers")
	db.Close()
	if verbose: print(f" BuildLeaderIndex: {added} rows added")
	return added

class LeaderRiskSwitch(IndexRiskSwitch):
	"""IndexRiskSwitch on the momentum-leader index (LeaderIndexDaily, compounded) with a LEADER_SWITCH_WINDOW-day average.
	Same rule: off below (1 - band) x average, back on above (1 + band) x, read from the prior trading day."""
	def __init__(self, startDate, endDate, window: int = LEADER_SWITCH_WINDOW, band: float = 0.02):
		db = PTADatabase()
		df = None
		if db.Open():
			df = db.DataFrameFromSQL(f"SELECT [Date], LeaderReturn FROM dbo.{LEADER_INDEX_TABLE} WHERE [Date] <= :e ORDER BY [Date]", {"e": ToTimestamp(endDate)})
			db.Close()
		if df is None or df.empty: raise RuntimeError(f"LeaderRiskSwitch: {LEADER_INDEX_TABLE} is empty (run Selection.BuildLeaderIndex)")
		r = df.set_index(pd.to_datetime(df['Date']))['LeaderReturn'].astype(float)
		if r.index.min() > ToTimestamp(startDate) - pd.DateOffset(days=int(window * 1.6)):
			print(f" LeaderRiskSwitch: WARNING {LEADER_INDEX_TABLE} starts {FormatDate(r.index.min())}, too late for a {window}-day average at {FormatDate(startDate)}")
		level = (1 + r).cumprod()
		ma = level.rolling(window).mean()
		state, cur = [], False
		for c, m in zip(level.to_numpy(), ma.to_numpy()):
			if m == m:
				if not cur and c < m * (1 - band): cur = True
				elif cur and c > m * (1 + band): cur = False
			state.append(cur)
		self.state = pd.Series(state, index=level.index)

def ApplyRiskOffExposure(picks: pd.DataFrame, exposure: float) -> pd.DataFrame:
	"""Scales the non-cash TargetHoldings to exposure of the total and puts the rest in the cash row."""
	if picks is None or picks.empty: return picks
	picks = picks.copy()
	total = float(picks["TargetHoldings"].sum())
	if total <= 0: return picks
	cash = CONSTANTS.CASH_TICKER
	stock_mask = picks.index != cash
	cash_now = float(picks.loc[cash, "TargetHoldings"]) if cash in picks.index else 0.0
	picks.loc[stock_mask, "TargetHoldings"] = picks.loc[stock_mask, "TargetHoldings"] * exposure
	new_cash = cash_now + (total - cash_now) * (1 - exposure)
	if cash in picks.index:
		picks.loc[cash, "TargetHoldings"] = new_cash
	else:
		row = pd.DataFrame({c: [0.0] for c in picks.columns}, index=[cash])
		row["TargetHoldings"] = new_cash
		picks = pd.concat([picks, row])
		picks.index.name = 'Ticker'
	return picks

VOL_TARGET_STEP = 0.1		#6.8: exposure rounded to this step, so small volatility changes don't trade
VOL_TARGET_MIN_INVESTED = 0.2	#days invested less than this are skipped (their return says little about the stocks' volatility)

def VolTargetExposure(dailyValue: pd.DataFrame, target: float, lookback: int = 21) -> float:
	"""6.8: share of the picks to hold so the portfolio's realized volatility matches target (annualized).
	The volatility is that of the fully invested portfolio: each day's TotalValue return divided by the prior day's invested
	share (AssetValue / TotalValue), over the last lookback days at least VOL_TARGET_MIN_INVESTED invested, through the last
	stored close. Returns min(1, target / volatility) rounded to VOL_TARGET_STEP, or 1.0 when off or with fewer than
	max(5, lookback // 2) usable days (the start of a run)."""
	if target <= 0 or dailyValue is None or len(dailyValue) < 3: return 1.0
	d = dailyValue[['AssetValue', 'TotalValue']].astype(float).iloc[-(lookback * 3 + 2):]
	r = d['TotalValue'].pct_change()
	w = (d['AssetValue'] / d['TotalValue']).shift(1)
	u = (r / w)[w >= VOL_TARGET_MIN_INVESTED].dropna().iloc[-lookback:]
	if len(u) < max(5, lookback // 2): return 1.0
	vol = float(u.std()) * np.sqrt(252)
	if not vol > 0: return 1.0
	return float(min(1.0, round(target / vol / VOL_TARGET_STEP) * VOL_TARGET_STEP))
