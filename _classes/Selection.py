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
#   Optional, off: ROBUST_DISPERSION (IQR of log returns; re-tune before enabling).
# =============================================================================================
import numpy as np, pandas as pd
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
MARKET_STATE_VERSION = 4.6		# 4.6: warmup from stored states, trading-day calendar, row-based history window, blow-off detection
STATE_HISTORY_ROWS = 2520		# ~10 trading years; used for the warmup preload, history trim and all percentiles
STATE_HISTORY_MIN_ROWS = 250	# below this, disp_pct/dvol_z stay neutral (early 1980s)
UNIVERSE_SIZE_WARN = 200		# universe below this is a data flag (missing prices), not a regime signal
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
		if USE_MANIA_SPLIT and self.days_since_mania < POST_MANIA_DAYS:
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

	def GetExecutionFilters(self) -> list:
		"""Filter blend selected by regime label (v4.1 approach, restored in v4.4).

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
		if startDate:
			startDate = ToTimestamp(startDate) - pd.offsets.BusinessDay(ADAPTIVE_WARMUP_DAYS) #Add days for warmup history
		self._startDate = startDate
		if not endDate: endDate = pd.offsets.BDay().rollback(datetime.now())
		self._endDate = ToTimestamp(endDate)
		self.convex_duration = 0
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

	def AddTicker(self, ticker:str):
		if not ticker in self._tickerList:
			p = PricingData(ticker)
			if p.LoadHistory(self._startDate, self._endDate, verbose = self.verbose):
				p.CalculateStats(fullStats=False)
				self.priceData.append(p)
				self._tickerList.append(ticker)
			else:
				self._failedList.append(ticker)

	def RemoveTicker(self, ticker:str):
		i=len(self.priceData)-1
		while i >= 0:
			if ticker == self.priceData[i].ticker:
				self.PrintWrapper(" Removing ticker " + ticker)
				self.priceData.pop(i)
				self._tickerList.remove(ticker)
			i -=1
		if ticker in self._tickerList: 
			print(" Error removing ticker " + ticker)
			print(len(self.priceData))	
			print(self._tickerList)	
			assert(False)

	def AlignToList(self, newList:list):
		#Add/Remove tickers until they match the given list
		i=len(self.priceData)-1
		added_tickers = 0
		removed_tickers = 0
		while i >= 0:
			ticker = self.priceData[i].ticker
			if not ticker in newList:
				self.PrintWrapper(" Removing ticker " + ticker)
				self.priceData.pop(i)
				removed_tickers +=1
			i -=1
		self.pbar = tqdm(total=len(newList), desc=" AlignToList adding tickers")
		for t in newList:
			if t in self._failedList:
				print(f" Skipping {t} as it failed to load previously")
			elif not t in self._tickerList:
				self.AddTicker(t)
				added_tickers +=1
			self.pbar.update(1)
		print(f" AlignToList - Added: {added_tickers}	Removed {removed_tickers}")
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
		if not hasattr(self, "_pick_history") or self._pick_history is None or len(self._pick_history) == 0:
			todays_picks = compute_daily_picks(currentDate)
			return self._rolling_history_append(currentDate=currentDate, todays_picks=todays_picks, max_picks=max_picks, normalize=normalize)
		last_date = self._pick_history["as_of_date"].max()
		trading_index = self.priceData[0].historicalPrices.index
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
		if currentDate is None:
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
			rows.append({'Ticker': ticker,'Average_5Day': row['Average_5Day'],'Average_2Day': row['Average_2Day'],'Average': row['Average'],'PC_2Year': row['PC_2Year'],'PC_1Year': row['PC_1Year'],'PC_9Month': row['PC_9Month'],'PC_6Month': row['PC_6Month'],'PC_3Month': row['PC_3Month'],'PC_2Month': row['PC_2Month'],'PC_1Month': row['PC_1Month'],'PC_3Day': row['PC_3Day'],'PC_1Day': row['PC_1Day'],'PC_1Month3WeekEMA': row['PC_1Month3WeekEMA'],'PC_10Day5DayEMA': row['PC_10Day5DayEMA'], 'LossStd_1Year': row['LossStd_1Year'],'LossSkew_1Year': row.get('LossSkew_1Year', 0.0), 'Point_Value': row['Point_Value'], 'LogDrawdown': row['LogDrawdown'], 'PathologyScore': row['PathologyScore'], 'MaxLoss_1Year': row['MaxLoss_1Year'], 'DamageScore': row['DamageScore'], 'Distance_200DMA': row.get('Distance_200DMA', 0.0), 'Comments': row.get('Comments', ''),'latestEntry': pd_obj.historyEndDate})
		if not rows: 
			for filter_option, stocks_to_return in filterOptions:
				candidates[filter_option] = CASH_RESULT.copy()
			return candidates
		base = pd.DataFrame(rows).set_index('Ticker')
		base.index.name = 'Ticker'

		def base_filter(df, keep_loose:bool=False):
			result = ( (df['PC_9Month'] > 0.5) & (df['MaxLoss_1Year'] > .6) & (df['PathologyScore'] < .2) )
			if not keep_loose: result = result & (df['PC_1Month3WeekEMA'] > 0)
			return result
		filter_map = {
			0: dict(sort='PC_1Year', mask=lambda df: (df['Average_5Day'] > 0) ),
			2: dict(sort='PC_1Year', mask=lambda df: ( (df['PC_1Month3WeekEMA'] < df['PC_1Month']) & base_filter(df, False)) ), 
			3: dict(sort='PC_1Year', mask=lambda df: (  base_filter(df, False)) ), 
			9: dict(sort='PC_9Month', mask=lambda df:( base_filter(df, False)) ), 
			5: dict(sort='Point_Value', mask=lambda df: ( (df['Point_Value'] > 0) )), 
			1: dict(sort='PC_1Year', mask=lambda df: (df['PC_1Month3WeekEMA'] > 0) ),
			4: dict(sort='PC_1Month3WeekEMA', mask=lambda df: ( (df['PC_1Month3WeekEMA'] > 0) & base_filter(df, True)) ),
			6: dict(sort='PC_6Month', mask=lambda df: ( base_filter(df, True)) ), 
			7: dict(sort='PC_9Month', mask=lambda df: ( (df["PC_10Day5DayEMA"] < -0.01) & (df["PC_10Day5DayEMA"] > -0.12) & (df["PC_1Month3WeekEMA"] > df["PC_3Month"]) & base_filter(df, False)) ),
			8: dict(sort='PC_9Month', mask=lambda df: ( (df['PC_1Month3WeekEMA'] > df['PC_3Month'] ) & base_filter(df, True)) ), 
			10: dict(sort='LogDrawdown', mask=lambda df: (
				(df['LogDrawdown'] < -0.02) &          # Actually in a pullback (not at highs)
				(df['LogDrawdown'] > -0.15) &           # Controlled — not a breakdown
				(df['PC_9Month'] > 0) &                 # Medium trend intact
				(df['PC_3Month'] > 0) &                 # 3-month trend still positive
				(df['PathologyScore'] < 1.5) &          # Tighter damage gate than base
				(df['DamageScore'] < 0.3) &             # No sustained multi-month damage
				(df['MaxLoss_1Year'] > -0.6)            # Max monthly loss gate from base
			)),
		}
		for filter_option, stocks_to_return in filterOptions:
			spec = filter_map.get(filter_option)
			if spec is None: continue
			df_work = base
			if 'prep' in spec and callable(spec['prep']): df_work = spec['prep'](df_work)
			df = df_work.loc[spec['mask'](df_work)]
			if df.empty:
				candidates[filter_option] = CASH_RESULT.copy()
				continue
			df = df.sort_values(spec['sort'], ascending=False)
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
			return self._blend_from_multiverse(multiverse_candidates, filters)
		if not useRollingWindow: return compute_daily_picks(currentDate)
		result = self._update_pick_history(currentDate=currentDate, compute_daily_picks=compute_daily_picks, max_picks=15, normalize=normalize)
		if self.verbose:
			print(result)
			print("Pick history:", currentDate, self._pick_history["as_of_date"].min(), "→", self._pick_history["as_of_date"].max(), "trading_days:", self._pick_history["as_of_date"].nunique(), "rows:", len(self._pick_history))
		return result

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
			"state_version": float(MARKET_STATE_VERSION)
		}	
		df = db.DataFrameFromSQL(sql, params)
		db.Close()
		if df is None or df.empty: return None
		return self._state_from_row(df.iloc[0])

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
		df = db.DataFrameFromSQL(sql, {"beforeDate": pd.Timestamp(beforeDate), "state_version": float(MARKET_STATE_VERSION)})
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
				universe_size = universe_size #Note: this was the size requested, not necessarily the size found
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
		market_state = self._get_market_state_from_sql(forDate, universe_size)
		if not market_state:
			market_state = self._generate_market_state(forDate, universe_size)
			if not market_state: return None
		market_state.set_expansion_state(prev_state) 
		market_state.disp_pct, market_state.dvol_z = self._history_percentile_and_z(market_state.dispersion, market_state.downside_volatility)
		market_state.disp_off_high, market_state.days_since_mania = self._mania_context(market_state.dispersion)
		if market_state.universe_size < UNIVERSE_SIZE_WARN:
			self.PrintWrapper(f" Market state {pd.Timestamp(forDate).date()}: universe_size {market_state.universe_size} < {UNIVERSE_SIZE_WARN} (check for missing prices)")
		confidence = self._compute_state_confidence_from_history(market_state)
		if market_state.geometry_state == "CONVEX":
			self.convex_duration += 1
		else:
			self.convex_duration = 0			
		market_state.convex_duration = self.convex_duration
		market_state.state_confidence = confidence
		market_state.SaveToSQL()		
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
		cal = self._trading_dates()
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
		if getattr(self, "_last_saved_velocity_date", None) != smoothed.as_of_date:
			smoothed.SaveVelocityEwmToSQL()
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
		filterOptions = [(0,250),(4,3),(8,3),(10,5)] + DEFAULT_BLEND
		multiverse_candidates = self.GetHighestPriceMomentumMulti(currentDate=currentDate, filterOptions=filterOptions)
		universe_size = len(multiverse_candidates[0])
		market_state = self._get_market_state_smoothed(currentDate, universe_size)
		if not market_state:
			return CASH_RESULT.copy()
		filters = market_state.GetExecutionFilters()
		window_size = market_state.GetRollingWindowSize()
		stock_count = market_state.GetStockCount()
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
		