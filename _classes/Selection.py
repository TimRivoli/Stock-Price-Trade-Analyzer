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
MARKET_STATE_VERSION = 4.5
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

	def GetRollingWindowSize(self) -> int:
		"""Window driven by velocity_ewm (smoothed regime momentum).
		Using the 5-day EWM rather than raw single-day velocity reduces noise
		while preserving the directional signal.
		Positive velocity_ewm → shorten window (capture accelerating leaders faster).
		Negative velocity_ewm → lengthen window (hold persistent leaders in decel regime).
		Base 26 ± up to 8 days."""
		v  = float(np.clip(self.velocity_ewm, -1.0, 1.0))
		hw = 26 + int(round(-8.0 * v))
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
		return (f"{self.geometry_state}_{self.expansion_state}_{self.leadership_state}"	)

	def SaveToSQL(self):
		db = PTADatabase()
		if not db.Open(): 
			return		
		sql_delete = f"DELETE FROM {CONSTANTS.ADAPTIVE_CONVEX_STATE_TABLE} WHERE asOfDate = :as_of_date AND universe_size = :universe_size"
		delete_params = {"as_of_date": self.as_of_date, "universe_size": int(self.universe_size)}
		db.ExecSQL(sql_delete, delete_params)
		sql_insert = f"INSERT INTO {CONSTANTS.ADAPTIVE_CONVEX_STATE_TABLE} (asOfDate, convex_duration, dispersion, momentum_autocorr, downside_volatility, stress_index, corr_6m_1m, corr_1y_1m, leadership_tilt, disp_p40, disp_p75, expansion_velocity, velocity_ewm, geometry_state, expansion_state, leadership_state, regime_tension_flag, leadership_break_flag, persistence_break_flag, state_confidence, universe_size, state_version ) VALUES (:as_of_date, :convex_duration, :dispersion, :momentum_autocorr, :downside_volatility, :stress_index, :corr_6m_1m, :corr_1y_1m, :leadership_tilt, :disp_p40, :disp_p75, :expansion_velocity, :velocity_ewm, :geometry_state, :expansion_state, :leadership_state, :regime_tension_flag, :leadership_break_flag, :persistence_break_flag, :state_confidence, :universe_size, :state_version)"
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
			"state_version": float(self.state_version)
		}
		try:
			db.ExecSQL(sql_insert, insert_params)
		except Exception as e:
			print(f"Error saving MarketState for {self.as_of_date}: {e}")
		finally:
			db.Close()
		
	def ToDateFrame(self):
		row = {
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
			"state_version": float(self.state_version)
		}
		df_new = pd.DataFrame([row])
		df_new["as_of_date"] = pd.to_datetime(df_new["as_of_date"])
		df_new = df_new.set_index("as_of_date")
		return df_new
		
class StockPicker():
	def __init__(self, startDate:pd.Timestamp =None, endDate:pd.Timestamp=None, pickHistoryWindow=60, verbose:bool = False): 
		self.pbar = None
		self.verbose = verbose
		self.priceData = []
		self._tickerList = []
		if startDate:
			startDate = ToTimestamp(startDate) - pd.offsets.BusinessDay(ADAPTIVE_WARMUP_DAYS) #Add days for warmup history
		self._startDate = startDate
		if not endDate: endDate = pd.offsets.BDay().rollback(datetime.now())
		self._endDate = ToTimestamp(endDate)
		self.convex_duration = 0
		self._adaptive_history_df = None
		self._pick_history = None
		self.pickHistoryWindowSize = pickHistoryWindow

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
		while i >= 0:
			ticker = self.priceData[i].ticker
			if not ticker in newList:
				self.PrintWrapper(" Removing ticker " + ticker)
				self.priceData.pop(i)
				self._tickerList.remove(ticker)
			i -=1
		self.pbar = tqdm(total=len(newList), desc=" AlignToList adding tickers")
		for t in newList:
			self.AddTicker(t)
			self.pbar.update(1)
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
		max_allowed_date = (pd.Timestamp.now().normalize() - pd.offsets.BusinessDay(1)).to_pydatetime()
		if currentDate > max_allowed_date:
			currentDate = max_allowed_date
		rows = []
		for pd_obj in self.priceData:
			ticker = pd_obj.ticker
			df = pd_obj.historicalPrices
			if currentDate not in df.index:
				self.PrintWrapper(f" GetHighestPriceMomentumMulti: {ticker} missing {currentDate}")
				continue
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

	def _rolling_regime_percentiles(self, lookback_days: int = 2520):
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
		cutoff = pd.to_datetime(params.as_of_date) - pd.Timedelta(days=2520)
		self._adaptive_history_df = (self._adaptive_history_df[self._adaptive_history_df.index >= cutoff])

	def _get_market_state_from_sql(self, forDate, candidate_universe_size):
		db = PTADatabase()
		if not db.Open(): return None
			
		sql = f"SELECT * FROM {CONSTANTS.ADAPTIVE_CONVEX_STATE_TABLE} WHERE asOfDate = :asOfDate AND universe_size = :universe_size AND state_version = :state_version"
		params = {
			"asOfDate": pd.Timestamp(forDate), 
			"universe_size": int(candidate_universe_size),
			"state_version": float(MARKET_STATE_VERSION)
		}	
		df = db.DataFrameFromSQL(sql, params)
		db.Close()
		if df is None or df.empty: return None		
		row = df.iloc[0]
		market_state = AdaptiveConvexMarketState(
			as_of_date=pd.Timestamp(row["asOfDate"]),
			convex_duration=int(row.get("convex_duration", 0)),
			dispersion=float(row.get("dispersion", 0.0)),
			momentum_autocorr=float(row.get("momentum_autocorr", 0.0)),
			downside_volatility=float(row.get("downside_volatility", 0.0)),
			stress_index=float(row.get("stress_index", 0.0)),
			corr_6m_1m=float(row.get("corr_6m_1m", 0.0)),
			corr_1y_1m=float(row.get("corr_1y_1m", 0.0)),
			leadership_tilt=float(row.get("leadership_tilt", 0.0)),
			disp_p40=float(row.get("disp_p40", 0.0)),
			disp_p75=float(row.get("disp_p75", 0.0)),
			expansion_state=str(row.get("expansion_state", "NO_DATA")),
			state_version=float(row.get("state_version", MARKET_STATE_VERSION)),
			state_confidence=float(row.get("state_confidence", 0.0)),
			universe_size=int(row.get("universe_size", 0))
		)	
		return market_state
	
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
		lookback_start = toDate - pd.offsets.BDay(ADAPTIVE_WARMUP_DAYS)
		if self._adaptive_history_df is not None and not self._adaptive_history_df.empty:
			last_date = self._adaptive_history_df.index.max()
			start_date = max(last_date + pd.offsets.BDay(1), lookback_start)
		else:
			start_date = lookback_start
		bdays = pd.bdate_range(start=start_date, end=toDate)	   
		if len(bdays) == 0:
			return self._get_market_state(toDate, universe_size)
		for d in bdays:
			if self._adaptive_history_df is not None and d in self._adaptive_history_df.index:
				continue				
			state = self._get_market_state(d, universe_size)			
			if state is None: continue          
		return self._get_market_state(toDate, universe_size)		
	
	def _get_market_state_smoothed(self, forDate, universe_size):
		current_state = self._hydrate_market_state_history(forDate, universe_size)
		if self._adaptive_history_df is None or self._adaptive_history_df.empty:
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
		return todays_picks

def Generate_Picks_For_SQL_DateRange(startYear:int=None, years: int=0, replaceExisting:bool=False, adaptiveModel:bool=False, verbose:bool=False):
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
			new_tickers = TickerLists.GetTickerListSQL(year=month_start.year, month=month_start.month, SP500Only=params.SP500Only, filterByFundamentals=params.filterByFundamentals, marketCapMin=params.marketCapMin, marketCapMax=params.marketCapMax) 
			if len(new_tickers) > 0:
				if verbose: print(f" Generate_Picks_For_SQL_DateRange: Re-query tickers found {len(new_tickers)} instead of previous {len(tickers)}")
				tickers = new_tickers
			picker.AlignToList(tickers)			
			TotalValidCandidates = len(picker._tickerList) 
			if verbose: print(f" Generate_Picks_For_SQL_DateRange: Running PicksBlended generation on {TotalValidCandidates} stocks {FormatDate(month_start)} to {FormatDate(month_end)}")		
			if TotalValidCandidates==0: assert(False)
			picker.GeneratePicksForSQL(startDate=month_start, endDate=month_end, replaceExisting=replaceExisting, adaptiveModel=adaptiveModel, verbose=verbose)
	db.Close()
		