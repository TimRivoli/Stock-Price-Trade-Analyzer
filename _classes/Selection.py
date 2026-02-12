import time, numpy as np, pandas as pd
import _classes.Constants as CONSTANTS
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict
from tqdm import tqdm
from collections import defaultdict
from _classes.Utility import *
from _classes.Prices import PriceSnapshot, PricingData
from _classes.DataIO import PTADatabase
from _classes.TickerLists import TickerLists

BASE_RAMP_UP_PER_DAY = 0.06     # convex ramps fast
BASE_RAMP_DOWN_PER_DAY = 0.03   # convex decays slower (asymmetry)
ADAPTIVE_WARMUP_DAYS = 60 		# small, deterministic warmup for the convex engine

@dataclass
class AdaptiveConvexMarketState:
	# Identity
	as_of_date: pd.Timestamp
	reevaluation_interval: int
	convex_duration : int = 0
	
	# Regime inputs
	dispersion: float = 0.0
	momentum_autocorr: float = 0.0
	downside_volatility: float = 0.0
	stress_index: float = 0.0
	corr_6m_1m: float = 0.0
	corr_1y_1m: float = 0.0

	# Allocation weights (must sum to 1.0)
	convex_weight: float = 0.0
	linear_weight: float = 0.0
	defensive_weight: float = 0.0
	cash_weight: float = 1.0
	
	# Optional diagnostics / state
	version: Optional[float] = 3.2
	is_warmup: Optional[bool] = False
	regime_label: Optional[str] = 'NO_DATA'
	hysteresis_label: Optional[str] = 'NO_DATA'

	def Validate(self, tol: float = 1e-6):
		total = (self.convex_weight + self.linear_weight + self.defensive_weight + self.cash_weight)
		if abs(total - 1.0) > tol:
			raise ValueError(f" AdaptiveConvex weights do not sum to 1.0 (got {total})")

	def SaveToSQL(self):
		db = PTADatabase()
		if db.Open():
			sql_delete = f"DELETE FROM {CONSTANTS.ADAPTIVE_CONVEX_STATE_TABLE} WHERE asOfDate = :asOfDate and reevaluation_interval = :reevaluation_interval"
			params = {"asOfDate": self.as_of_date, "reevaluation_interval": int(self.reevaluation_interval)}
			db.ExecSQL(sql_delete, params)
			sql_insert = f"INSERT INTO {CONSTANTS.ADAPTIVE_CONVEX_STATE_TABLE} (asOfDate, reevaluation_interval, convex_duration, dispersion, momentum_autocorr, downside_volatility, stress_index, convex_weight, linear_weight, defensive_weight, cash_weight, regime_label, hysteresis_label, version, is_warmup) VALUES (:asOfDate, :reevaluation_interval, :convex_duration , :dispersion, :momentum_autocorr, :downside_volatility, :stress_index, :convex_weight, :linear_weight, :defensive_weight, :cash_weight, :regime_label, :hysteresis_label, :version, :is_warmup)"
			params.update({
				"dispersion": float(self.dispersion),
				"momentum_autocorr": float(self.momentum_autocorr),
				"downside_volatility": float(self.downside_volatility),
				"convex_duration": int(self.convex_duration),
				"stress_index": float(self.stress_index),
				"corr_6m_1m": float(self.corr_6m_1m),
				"corr_1y_1m": float(self.corr_1y_1m),
				"convex_weight": float(self.convex_weight),
				"linear_weight": float(self.linear_weight),
				"defensive_weight": float(self.defensive_weight),
				"cash_weight": float(self.cash_weight),
				"regime_label": self.regime_label,
				"hysteresis_label": self.hysteresis_label, 
				"version": float(self.version),
				"is_warmup": bool(self.is_warmup)
			})
			db.ExecSQL(sql_insert, params)
			db.Close()
			
class StockPicker():
	def __init__(self, startDate:datetime=None, endDate:datetime=None, verbose:bool = False): 
		self.pbar = None
		self.verbose = verbose
		self.priceData = []
		self._tickerList = []
		if startDate:
			startDate = ToTimestamp(startDate) - pd.offsets.BusinessDay(ADAPTIVE_WARMUP_DAYS) #Add days for warmup history
		self._startDate = startDate
		self._endDate = endDate
		self._adaptive_is_warming = False
		self._adaptive_last_date = None
		self.convex_state = False
		self.convex_duration = 0
		self.lockout_days_remaining = 0
		self.hysteresis_label = "NEUTRAL"
		self._adaptive_history_df = pd.DataFrame()

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
				p.CalculateStats(fullStats=True)
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
	
	def GetHighestPriceMomentumMulti(self, currentDate: datetime, filterOptions: dict, minPercentGain: float = 0.05):
		if not isinstance(filterOptions, dict):
			raise ValueError("filterOptions must be a dict like {filter_id: stock_count}")
		EMPTY_RESULT = pd.DataFrame(columns=['TargetHoldings', 'Point_Value'])
		EMPTY_RESULT.index.name = 'Ticker'
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
			if min( row['HP_2Yr'], row['HP_1Yr'], row['HP_6Mo'], row['HP_2Mo'], row['HP_1Mo'], row['Average_5Day'] ) <= 0:
				continue
			rows.append({'Ticker': ticker,'hp2Year': row['HP_2Yr'],'hp1Year': row['HP_1Yr'],'hp6mo': row['HP_6Mo'],'hp3mo': row['HP_3Mo'],'hp2mo': row['HP_2Mo'],'hp1mo': row['HP_1Mo'],'Average_5Day': row['Average_5Day'],'Average_2Day': row['Average_2Day'],'Average': row['Average'],'Channel_High': row['Channel_High'],'Channel_Low': row['Channel_Low'],'PC_2Year': row['PC_2Year'],'PC_1Year': row['PC_1Year'],'PC_6Month': row['PC_6Month'],'PC_3Month': row['PC_3Month'],'PC_2Month': row['PC_2Month'],'PC_1Month': row['PC_1Month'],'PC_3Day': row['PC_3Day'],'PC_1Day': row['PC_1Day'],'PC_1Month3WeekEMA': row['PC_1Month3WeekEMA'],'Deviation_15Day': row['Deviation_15Day'],'Deviation_10Day': row['Deviation_10Day'],'Deviation_5Day': row['Deviation_5Day'],'Deviation_1Day': row['Deviation_1Day'],'Gain_Monthly': row['Gain_Monthly'],'LossStd_1Year': row['LossStd_1Year'],'Point_Value': row['Point_Value'],'Comments': row.get('Comments', ''),'HasFullLookback': row.get('HasFullLookback', True),'latestEntry': pd_obj.historyEndDate})
		if not rows: 
			for filter_option in filterOptions.keys(): candidates[filter_option] = EMPTY_RESULT.copy()
			return candidates
		base = pd.DataFrame(rows).set_index('Ticker')
		base.index.name = 'Ticker'

		minPC_1Day = minPercentGain / CONSTANTS.TRADING_YEAR
		#More complex filters that I have tried have all decreased performance which is why these are simple
		#Greatest factors for improvement are high 5mo-1yr return and a very low selection of stocks, like 1-3
		#Best way to compensate for few stocks is to blend filters of different strengths
		#7, 8, 9 are all very poor 
		filter_map = {
			0: dict(sort='PC_1Year', mask=lambda df: (df['Average_5Day'] > 0) ),
			1: dict(sort='PC_1Year', mask=lambda df: (df['PC_1Month3WeekEMA'] > minPC_1Day) ), #currently the defensive strategy			
			2: dict(sort='PC_1Year', mask=lambda df: (df['PC_1Year'] > minPercentGain)), #very high growth
			3: dict(sort='PC_1Year', mask=lambda df: ((df['PC_1Year'] > minPercentGain) & (df['PC_1Month3WeekEMA'] > 0))), #should be defense
			4: dict(sort='PC_1Month3WeekEMA', mask=lambda df: (df['PC_1Month3WeekEMA'] > minPC_1Day) ),
			5: dict(sort='Point_Value', mask=lambda df: ( (df['PC_1Year'] > minPercentGain) & (df['Point_Value'] > 0) )), #Gets close to performance of blended, which is very good
			6: dict(sort='PC_6Month', mask=lambda df: (df['PC_6Month'] > minPercentGain) ), #6 month version of 2 for faster reaction, even faster growth
			7: dict(prep=lambda df: df.assign(BreakoutScore=(2.0 * df['Deviation_10Day'] + 1.0 * df['Deviation_5Day'] + 1.5 * df['PC_3Month'] + 1.0 * df['PC_1Month3WeekEMA'])),sort='BreakoutScore',mask=lambda df: ((df['PC_1Year'] > minPC_1Day) &(df['PC_1Month3WeekEMA'] > 0) &(df['Deviation_10Day'] > 0) &(df['PC_3Month'] > 0))),
			8: dict(prep=lambda df: df.assign(LowVolScore=(df['PC_1Year'] / (0.0001 + df['LossStd_1Year'])),LossCut=df['LossStd_1Year'].quantile(0.35)),sort='LowVolScore',mask=lambda df: ((df['PC_1Year'] > minPC_1Day) & (df['PC_3Month'] > 0) & (df['LossStd_1Year'] > 0) & (df['LossStd_1Year'] <= df['LossCut']))),
			9: dict(prep=lambda df: df.assign(RiskAdjMomentum=(df['PC_3Month'] / (0.0001 + df['LossStd_1Year'])), BlowoffCut=df['Deviation_10Day'].quantile(0.97)), sort='RiskAdjMomentum', mask=lambda df: ((df['PC_6Month'] > minPercentGain) & (df['PC_3Month'] > 0) & (df['PC_1Month3WeekEMA'] > 0) & (df['LossStd_1Year'] > 0) & (df['Deviation_10Day'] > 0) & (df['Deviation_10Day'] < df['BlowoffCut'] ) ) ),
		}
		#former 1, was a dud. dict(sort='PC_1Year', mask=lambda df: ((df['PC_1Year'] > minPercentGain) & (df['PC_1Month3WeekEMA'] > 0) & ( df['PC_1Year'] / CONSTANTS.TRADING_YEAR > df['PC_1Month3WeekEMA'] / CONSTANTS.TRADING_MONTH ) ) ),
		
		for filter_option, stocks_to_return in filterOptions.items():
			spec = filter_map.get(filter_option)
			if spec is None: continue
			df_work = base
			if 'prep' in spec and callable(spec['prep']): df_work = spec['prep'](df_work)
			df = df_work.loc[spec['mask'](df_work)]
			if df.empty:
				candidates[filter_option] = EMPTY_RESULT.copy()
				continue
			df = df.sort_values(spec['sort'], ascending=False)
			candidates[filter_option] = df.head(int(stocks_to_return))
		return candidates

	def GetHighestPriceMomentum(self, currentDate:datetime, stocksToReturn:int = 5, filterOption:int = 3, minPercentGain:float = 0.05, allocateByTargetHoldings:bool = False, allocateByPointValue=False):
		filter_options = {filterOption:stocksToReturn}
		result = self.GetHighestPriceMomentumMulti(currentDate=currentDate, filterOptions=filter_options, minPercentGain=minPercentGain)
		result = result.get(filterOption, pd.DataFrame())
		if allocateByTargetHoldings:
			result = result.groupby(level=0)[['Point_Value']].sum().rename(columns={'Point_Value': 'TargetHoldings'})
		elif allocateByPointValue:
			result = result.groupby(level=0).size().to_frame(name='TargetHoldings')
		return result

	def GetPicksBlended(self, currentDate:date, filter1:int = 3, filter2: int = 3, filter3: int = 1, filter4: int = 5, minPercentGain:float = 0.05):
		#generates list of tickers with TargetHoldings which indicate proportion of holdings	
		filter_options = {filter1:3, filter2:3, filter3:3, filter4:4}
		multiverse_candidates = self.GetHighestPriceMomentumMulti(currentDate=currentDate, filterOptions=filter_options, minPercentGain=minPercentGain)
		list1 = multiverse_candidates[filter1]
		list2 = multiverse_candidates[filter2]
		list3 = multiverse_candidates[filter3]
		list4 = multiverse_candidates[filter4]
		all_inputs = [list1, list2, list3, list4]
		valid_inputs = [i for i in all_inputs if i is not None and len(i) > 0]
		if not valid_inputs:
			return pd.DataFrame(columns=['TargetHoldings']).rename_axis('Ticker')
		result = pd.concat([list1, list2, list3, list4], sort=True) #append lists together
		result = pd.DataFrame(result.groupby(level=0).size()) #Group by ticker with new colum for TargetHoldings, .size=count; .sum=sum, keeps only the index and the count
		result.index.name='Ticker'
		result.rename(columns={0:'TargetHoldings'}, inplace=True)
		result.sort_values('TargetHoldings', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last') 
		return result

	def GetPicksBlendedSQL(self, currentDate:date, sqlHistory:int=90):
		result = None
		db = PTADatabase()
		if db.Open():
			SQL = "select * from fn_GetBlendedPicks('" + str(currentDate) + "', " + str(sqlHistory) + ")"
			result = db.DataFrameFromSQL(sql=SQL, indexName='Ticker')
		db.Close()
		return result	
			
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


		# ---------------- Weighted aggregation ----------------
		# Volatility is the "damage potential"
		# Dispersion is the "opportunity surface collapse"
		# Autocorr is the "trend break / mean reversion"
		stress = (
			0.45 * vol_stress +
			0.35 * dispersion_stress +
			0.20 * autocorr_stress
		)
		return float(np.clip(stress, 0.0, 1.0))

	def update_convex_state_with_hysteresis(self, dispersion, autocorr, stress_tamper, dt_days=1):
		# ---------------- Lockout handling ----------------
		if getattr(self, "lockout_days_remaining", 0) > 0:
			self.lockout_days_remaining = max(0, self.lockout_days_remaining - dt_days)
			self.convex_state = False
			self.hysteresis_label = f"LOCKED_OUT_{self.lockout_days_remaining}"
			return False
		prev_on = bool(self.convex_state)

		# ---------------- Adaptive threshold learning ----------------
		hist = getattr(self, "_adaptive_history_df", None)
		if hist is not None and not hist.empty and len(hist) >= 60:
			disp_enter = float(hist["dispersion"].quantile(0.75))  # harder to enter
			disp_force = float(hist["dispersion"].quantile(0.90))  # dispersion-only override
			disp_stay  = float(hist["dispersion"].quantile(0.55))  # easier to stay in
			auto_enter = float(hist["momentum_autocorr"].quantile(0.40))
			auto_stay  = float(hist["momentum_autocorr"].quantile(0.30))
			auto_exit  = float(hist["momentum_autocorr"].quantile(0.15))
			# Safety clamps (prevents quantiles drifting into nonsense regimes)
			disp_enter = float(np.clip(disp_enter, 0.20, 0.30))
			disp_force = float(np.clip(disp_force, disp_enter + 0.01, 0.35))
			disp_stay  = float(np.clip(disp_stay, 0.12, disp_enter))
			auto_enter = float(np.clip(auto_enter, -0.08, 0.05))
			auto_stay  = float(np.clip(auto_stay, -0.12, 0.05))
			auto_exit  = float(np.clip(auto_exit, -0.20, -0.05))
		else:
			# Fallback constants (bootstrapping / insufficient history)
			disp_enter = 0.24
			disp_force = 0.27
			disp_stay  = 0.18
			auto_enter = -0.03
			auto_stay  = -0.08
			auto_exit  = -0.10

		# ---------------- Previously OFF ----------------
		if not prev_on:
			if (dispersion >= disp_enter and autocorr > auto_enter) or dispersion >= disp_force:
				self.convex_state = True
				self.hysteresis_label = "EXPANDING"
				return True

			if dispersion >= (disp_enter * 0.85):
				self.hysteresis_label = "RECOVERING"
			else:
				self.hysteresis_label = "NEUTRAL"
			self.convex_state = False
			return False

		# ---------------- Previously ON ----------------
		if dispersion >= disp_stay and autocorr > auto_stay:
			self.convex_state = True
			self.hysteresis_label = "ACTIVE"
			return True
		# Emergency lockout (mean reversion / hostile regime)
		if autocorr < auto_exit or stress_tamper < 0.30:
			self.convex_state = False
			self.hysteresis_label = "LOCKED_OUT"
			self.lockout_days_remaining = int(5 + 20 * (1.0 - stress_tamper))
			return False
		# Soft exit (normal fade)
		self.convex_state = False
		self.hysteresis_label = "CONTRACTING"
		return False
	
	def adaptive_engine_weights(self, dispersion, autocorr, stress_tamper, dt_days: int = 1):
		# --- 1. Hysteresis gate (discrete, intentional) ---
		self.convex_state = self.update_convex_state_with_hysteresis(dispersion, autocorr, stress_tamper, dt_days=dt_days)
		if not self.convex_state:
			def_w = 0.10 * stress_tamper
			cash_w = 1.0 - def_w
			target = {"convex": 0.0, "linear": 0.0, "defensive": def_w, "cash": cash_w}
		else:
			# --- 2. Smooth convex intensity from dispersion (sigmoid) ---
			disp_center = 0.25
			disp_width  = 0.04
			z = (dispersion - disp_center) / disp_width
			disp_intensity = 1.0 / (1.0 + np.exp(-z))

			# --- 3. Smooth momentum penalty ---
			mom_floor = -0.15
			mom_ceiling = 0.05
			momentum_factor = (autocorr - mom_floor) / (mom_ceiling - mom_floor)
			momentum_factor = float(np.clip(momentum_factor, 0.0, 1.0))

			# --- 4. Raw convex weight ---
			convex_w = float(np.clip(disp_intensity * momentum_factor, 0.0, 0.85))
			if stress_tamper < 0.75: convex_w *= stress_tamper
			#convex_w = float(np.clip(convex_raw, 0.0, 0.85))
			#convex_w *= stress_tamper

			# --- 5. Smooth cash response ---
			#min_cash = 0.05 + 0.40 * (1.0 - stress_tamper)
			min_cash = 0.05 + 0.25 * (1.0 - stress_tamper)
			max_cash = 0.90
			cash_power = 1.5
			cash_w = min_cash + (1.0 - convex_w) ** cash_power * (max_cash - min_cash)
			cash_w = float(np.clip(cash_w, min_cash, max_cash))

			# --- 6. Linear absorbs remainder ---
			linear_w = max(0.0, 1.0 - convex_w - cash_w)

			# --- 7. Normalize ---
			total = convex_w + linear_w + cash_w
			if total > 0:
				convex_w /= total
				linear_w /= total
				cash_w   /= total
			target = {"convex": convex_w, "linear": linear_w, "defensive": 0.0, "cash": cash_w}

		# --- 8. Time-scaled ramp smoothing (dt aware) ---
		prev = getattr(self, "_prev_weights", {"convex": 0.0, "linear": 0.10, "defensive": 0.0, "cash": 0.90})
		base_up = 0.10   # per-day speed when convex increases
		base_dn = 0.20 * (1.0 + 1.2 * (1.0 - stress_tamper))

		prev_convex = prev.get("convex", 0.0)
		target_convex = target.get("convex", 0.0)
		alpha_up = 1.0 - np.exp(-base_up * dt_days)
		alpha_dn = 1.0 - np.exp(-base_dn * dt_days)
		alpha = alpha_up if target_convex > prev_convex else alpha_dn

		new_weights = {}
		for k in ("convex", "linear", "defensive", "cash"):
			new_weights[k] = float(prev[k] + alpha * (target[k] - prev[k]))

		total = sum(new_weights.values())
		if total > 0:
			for k in new_weights:
				new_weights[k] /= total
		self._prev_weights = new_weights
		return new_weights

	def _append_adaptive_state(self, params: AdaptiveConvexMarketState):
		row = {"as_of_date": params.as_of_date,"convex_duration": params.convex_duration,"dispersion": params.dispersion,"momentum_autocorr": params.momentum_autocorr,"downside_volatility": params.downside_volatility,"stress_index": params.stress_index,"convex_weight": params.convex_weight,"linear_weight": params.linear_weight,"defensive_weight": params.defensive_weight,"cash_weight": params.cash_weight,"regime_label": params.regime_label,"hysteresis_label": params.hysteresis_label}

		df_new = pd.DataFrame([row])
		df_new["as_of_date"] = pd.to_datetime(df_new["as_of_date"])
		df_new = df_new.set_index("as_of_date")

		if self._adaptive_history_df is None or self._adaptive_history_df.empty:
			self._adaptive_history_df = df_new
		else:
			# 2. Ensure the existing DF is also datetime-indexed before concat
			if not pd.api.types.is_datetime64_any_dtype(self._adaptive_history_df.index):
				self._adaptive_history_df.index = pd.to_datetime(self._adaptive_history_df.index)
			
			self._adaptive_history_df = pd.concat([self._adaptive_history_df, df_new])

		# 3. Clean up duplicates and sort
		self._adaptive_history_df = self._adaptive_history_df[~self._adaptive_history_df.index.duplicated(keep="last")]
		self._adaptive_history_df = self._adaptive_history_df.sort_index()

		# 4. Use a Pandas-native Timestamp for the cutoff comparison
		cutoff = pd.to_datetime(params.as_of_date) - pd.Timedelta(days=365)
		self._adaptive_history_df = self._adaptive_history_df[self._adaptive_history_df.index >= cutoff]

	def _adaptive_convex_warmup(self, currentDate):
		if self._adaptive_is_warming or not self._startDate: return
		bdays = pd.bdate_range(self._startDate, currentDate)
		if len(bdays) <= 1: return
		self._adaptive_is_warming = True
		warmup_days = min(ADAPTIVE_WARMUP_DAYS, len(bdays) - 1)
		for d in bdays[-(warmup_days + 1):-1]:
			self.GetAdaptiveConvexPicks(currentDate=d)
		self._adaptive_is_warming = False

	def GetAdaptiveConvexPicks(self, currentDate, convex_filter:int = 6, linear_filter:int = 2, linear_fast_filter:int = 6, defense_filter:int = 1):		
		def add_block(df, block_weight):
			if df is None or df.empty or block_weight <= 0:
				return
			out = pd.DataFrame(index=df.index)
			out["TargetHoldings"] = block_weight / len(df)
			frames.append(out)

		if self._adaptive_last_date is None:
			self._adaptive_convex_warmup(currentDate)
			dt_days = 1
		else:
			dt_days = Business_Days_Since(self._adaptive_last_date, currentDate)
		self._adaptive_last_date = currentDate
		filters = {0:250, convex_filter:4, linear_filter:5, linear_fast_filter:4, defense_filter:6}
		multiverse_candidates = self.GetHighestPriceMomentumMulti(currentDate=currentDate, filterOptions=filters)
		df = multiverse_candidates[0]
		if df is None or df.empty:
			state = AdaptiveConvexMarketState(as_of_date = currentDate, reevaluation_interval=int(dt_days))
			final = pd.DataFrame({"TargetHoldings":[1.0]}, index=[CONSTANTS.CASH_TICKER])
			final.loc[CONSTANTS.CASH_TICKER] = {'TargetHoldings': 1.0, 'Point_Value': 100}
			final = df
		else:
			dispersion = self.compute_cross_sectional_dispersion(df)
			autocorr = self.compute_momentum_autocorr(df)
			downside_volatility = self.compute_downside_volatility(df)
			stress = self.compute_stress_index(dispersion, autocorr, downside_volatility) 
			tilt, corr_6m_1m, corr_1y_1m = self.compute_leadership_tilt(df)
			stress_tamper = 1.0 - 0.5 * stress
			weights = self.adaptive_engine_weights(dispersion, autocorr, stress_tamper, dt_days=dt_days)
			regime_label = Regime_Label_From_Weights(weights)
			if weights.get("convex", 0) > 0:
				self.convex_duration += 1
			else:
				self.convex_duration = 0
			frames = []
			if weights.get("convex", 0) > 0 and convex_filter in multiverse_candidates:
				add_block(multiverse_candidates[convex_filter], weights["convex"])
			if weights.get("linear", 0) > 0 and linear_filter in multiverse_candidates:
				linear_total = weights.get("linear", 0.0)
				linear_fast_w = linear_total * tilt
				linear_slow_w = linear_total * (1.0 - tilt)
				if linear_fast_w > 0 and linear_fast_filter in multiverse_candidates:
					add_block(multiverse_candidates[linear_fast_filter], linear_fast_w)
				if linear_slow_w > 0 and linear_filter in multiverse_candidates:
					add_block(multiverse_candidates[linear_filter], linear_slow_w)
			if weights.get("defensive", 0) > 0 and defense_filter in multiverse_candidates:
				add_block(multiverse_candidates[defense_filter], weights["defensive"])
			if weights.get("cash", 0) > 0:
				cash_df = pd.DataFrame({"TargetHoldings": [weights["cash"]]}, index=[CONSTANTS.CASH_TICKER])
				frames.append(cash_df)
			if not frames: return pd.DataFrame(columns=["TargetHoldings"]).rename_axis("Ticker")
			final = (pd.concat(frames).groupby(level=0)["TargetHoldings"].sum().to_frame())
			final["TargetHoldings"] /= final["TargetHoldings"].sum()
			state = AdaptiveConvexMarketState(as_of_date = currentDate, reevaluation_interval=int(dt_days), convex_duration  = self.convex_duration, dispersion = dispersion, momentum_autocorr = autocorr, downside_volatility = downside_volatility,  stress_index = stress, corr_6m_1m=corr_6m_1m, corr_1y_1m=corr_1y_1m, convex_weight = weights["convex"], linear_weight = weights["linear"], defensive_weight = weights["defensive"], cash_weight = weights["cash"], regime_label = regime_label, hysteresis_label = self.hysteresis_label, is_warmup=self._adaptive_is_warming)
		state.Validate()
		self._append_adaptive_state(state)
		state.SaveToSQL()
		return final
		
def Regime_Label_From_Weights(weights):
	if weights["convex"] < 0.05 and weights["cash"] > 0.7:
		return "OFF_CASH"
	if weights["convex"] > 0.5:
		return "CONVEX_DOMINANT"
	if weights["convex"] > 0.0:
		return "TRANSITION"
	return "LINEAR_DOMINANT"

def Business_Days_Since(prev_date, current_date):
	if prev_date is None:
		return None
	return max(np.busday_count(prev_date.date(), current_date.date()), 1)

def Generate_PicksBlendedSQL_DateRange(startYear:int=None, years: int=0, replaceExisting:bool=False, verbose:bool=False):
	db = PTADatabase()
	if db.Open():
		today = GetLatestBDay()
		if startYear== None:
			startYear = today.year
			endDate = (today - pd.offsets.BDay(1)).date() 
		else:
			endDate = ToDate('12/31/' + str(startYear))	
		startDate = ToDate('1/1/' + str(startYear-years))
		current_date = endDate
		picker = StockPicker(startDate=startDate, endDate=endDate)
		dates = db.ScalarListFromSQL("SELECT Date FROM rpt_PicksBlendedDaily_MissingDates WHERE year>=:StartYear AND year<=:EndYear ORDER BY Date",	{"StartYear": startDate.year, "EndYear": endDate.year},	column="Date")
		missing_dates = [d.date() if isinstance(d, datetime) else datetime.strptime(d, '%Y-%m-%d %H:%M:%S').date() for d in dates]
		#print(current_date)
		#print(missing_dates)
		#missing_dates = [datetime.strptime(row[0], '%Y-%m-%d').date() for row in results]
		prev_month = -1
		while current_date >= startDate:
			if current_date.weekday() < 5: #Python Monday=0, skip weekends
				ExistingDataCount = 0
				if not replaceExisting:					
					if not current_date in missing_dates: ExistingDataCount = 1
				if replaceExisting or ExistingDataCount == 0:
					if current_date.month != prev_month:
						if verbose: print(" Generate_PicksBlended_DateRange: Getting tickers for year " + str(current_date.year))				
						tickers = TickerLists.GetTickerListSQL(year=current_date.year, month=current_date.month, SP500Only=False, filterByFundamentals=False, marketCapMin=100) 
						TotalStocks=len(tickers)
						if verbose: print(" Generate_PicksBlended_DateRange: Total stocks: " + str(TotalStocks))
						picker.AlignToList(tickers)			
						TotalValidCandidates = len(picker._tickerList) 
						if verbose: print(' Generate_PicksBlended_DateRange: Running PicksBlended generation on ' + str(TotalValidCandidates) + ' of ' + str(TotalStocks) + ' stocks from ' + str(startDate) + ' to ' + str(endDate))		
						if TotalValidCandidates==0: assert(False)
						prev_month = current_date.month
					if verbose: print(' Generate_PicksBlended_DateRange: Blended 3.3.1.PV Picks - ' + str(current_date))
					result = picker.GetPicksBlended(currentDate=current_date)
					if verbose: print(result)
					if len(result) == 0:
						if verbose: print(" Generate_PicksBlended_DateRange: No data found.")
					else:
						result['Date'] = current_date 
						result['TotalStocks'] = TotalStocks
						result['TotalValidCandidates'] = TotalValidCandidates
						print(result)
						db.ExecSQL("DELETE FROM PicksBlendedDaily WHERE Date='" + str(current_date) + "'")
						db.DataFrameToSQL(result, tableName='PicksBlendedDaily', indexAsColumn=True, clearExistingData=False)
					result=None
			current_date -= timedelta(days=1) 
	db.ExecSQL("sp_UpdateBlendedPicks")
	db.Close()
		
def Update_PicksBlendedSQL(replaceExisting:bool=False):
	#If replaceExisting then it will do the current YTD, else just what is missing
	print('Updating PicksBlended')
	Generate_PicksBlended_DateRange(replaceExisting=replaceExisting)	
