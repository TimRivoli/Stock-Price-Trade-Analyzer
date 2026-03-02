import numpy as np, pandas as pd
import _classes.Constants as CONSTANTS
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict
from tqdm import tqdm
from _classes.Utility import *
from _classes.Prices import PriceSnapshot, PricingData
from _classes.DataIO import PTADatabase
from _classes.TickerLists import TickerLists

ADAPTIVE_WARMUP_DAYS = 60 		# small, deterministic warmup for the convex engine
EMPTY_RESULT = pd.DataFrame(columns=['TargetHoldings', 'Point_Value'])
EMPTY_RESULT.loc[CONSTANTS.CASH_TICKER] = {'TargetHoldings': 1.0, 'Point_Value': 10}
EMPTY_RESULT.index.name = 'Ticker'

MODE_CONVEX_FULL   = "CONVEX_FULL"
MODE_CONVEX_STABLE = "CONVEX_STABLE"
MODE_CONVEX_LATE   = "CONVEX_LATE"
MODE_BLENDED       = "BLENDED"
MODE_DEFENSIVE     = "DEFENSIVE"
MODE_TRANSITION = "TRANSITION"
MARKET_STATE_VERSION = 4.0
CASH_FILTER = -1

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

	def GetRollingWindowSize(self):
		# leadership_component = (self.leadership_tilt - 0.5) * 2.0
		# persistence_component = self.momentum_autocorr
		# velocity_component = self.expansion_velocity
		# pressure = (
			# 0.45 * leadership_component +
			# 0.35 * persistence_component +
			# 0.20 * velocity_component
		# )
		# pressure = max(-1.0, min(1.0, pressure))
		# base = 15 + pressure * 5
		# band = 5 * self.state_confidence + 2 # confidence 0 → ±2, confidence 1 → ±7
		# lower = 15 - band
		# upper = 15 + band
		# window = max(lower, min(upper, round(base)))
		return 30
	
	def GetExecutionFilters(self):
		if (self.persistence_break_flag and self.state_confidence < 0.5 and self.stress_index > 0.7 ): return "CASH"
		g = self.geometry_state
		e = self.expansion_state
		l = self.leadership_state
		if l == "STABLE":
			if e == "CONTRACTING":
				return [3]      # slower confirmation
			return [4]          # default continuation
		if l == "TRANSITIONAL":
			return [4]
		if l == "ROTATING":
			if e == "EXPANDING":
				return [4]
			return [5]
		return [4]
	
	def GetRegimeSummary(self) -> str: 
		return (f"{self.geometry_state}_{self.expansion_state}_{self.leadership_state}"	)

	def SaveToSQL(self):
		db = PTADatabase()
		if not db.Open(): 
			return		
		sql_delete = f"DELETE FROM {CONSTANTS.ADAPTIVE_CONVEX_STATE_TABLE} WHERE asOfDate = :as_of_date AND universe_size = :universe_size"
		delete_params = {"as_of_date": self.as_of_date, "universe_size": int(self.universe_size)}
		db.ExecSQL(sql_delete, delete_params)
		sql_insert = f"INSERT INTO {CONSTANTS.ADAPTIVE_CONVEX_STATE_TABLE} (asOfDate, convex_duration, dispersion, momentum_autocorr, downside_volatility, stress_index, corr_6m_1m, corr_1y_1m, leadership_tilt, disp_p40, disp_p75, geometry_state, expansion_state, leadership_state, regime_tension_flag, leadership_break_flag, persistence_break_flag, state_confidence, universe_size, state_version ) VALUES (:as_of_date, :convex_duration, :dispersion, :momentum_autocorr, :downside_volatility, :stress_index, :corr_6m_1m, :corr_1y_1m, :leadership_tilt, :disp_p40, :disp_p75, :geometry_state, :expansion_state, :leadership_state, :regime_tension_flag, :leadership_break_flag, :persistence_break_flag, :state_confidence, :universe_size, :state_version)"
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

	# ---------------- Rolling SQLHist-style aggregation (Adaptive version) ----------------	
	def _rolling_history_append(self, currentDate, todays_picks, max_picks: int = 15):
		if not hasattr(self, "_pick_history") or self._pick_history is None:
			self._pick_history = pd.DataFrame(columns=["as_of_date", "TargetHoldings", "Point_Value"])
		todays_picks = todays_picks[["TargetHoldings", "Point_Value"]].copy()
		todays_picks["as_of_date"] = pd.to_datetime(currentDate)
		self._pick_history = pd.concat([self._pick_history, todays_picks])
		cutoff = pd.to_datetime(currentDate) - pd.offsets.BDay(self.pickHistoryWindowSize)
		self._pick_history = self._pick_history[self._pick_history["as_of_date"] >= cutoff].sort_values("as_of_date")
		hist = self._pick_history.copy()
		hist["PV_EMA"] = hist.groupby(level=0)["Point_Value"].transform(lambda x: x.ewm(span=self.pickHistoryWindowSize, adjust=False).mean())
		agg = hist.groupby(hist.index).agg(
			TargetHoldings=("TargetHoldings", "sum"), 
			DateCount=("as_of_date", "nunique"),
			FirstDate=("as_of_date", "min"),
			LastDate=("as_of_date", "max"),
			Point_Value=("PV_EMA", "last")
		)
		agg = agg.sort_values("TargetHoldings", ascending=False).head(max_picks)
		agg["TargetHoldings"] /= agg["TargetHoldings"].sum()
		return agg[["TargetHoldings", "DateCount", "FirstDate", "LastDate", "Point_Value"]].copy()

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
	def _blend_from_multiverse(self, multiverse_candidates, filters):
		dfs = [
			EMPTY_RESULT.copy() if f == CASH_FILTER
			else multiverse_candidates.get(f, pd.DataFrame())
			for f in filters
			if (f == CASH_FILTER) or (f in multiverse_candidates)
		]

		if not dfs:	return EMPTY_RESULT.copy()
		todays_picks = pd.concat(dfs, sort=True)
		todays_picks = (todays_picks.groupby(level=0).agg(TargetHoldings=('Point_Value', 'size'),Point_Value=('Point_Value', 'last')))
		todays_picks["TargetHoldings"] /= todays_picks["TargetHoldings"].sum()
		todays_picks.sort_values('TargetHoldings', ascending=False,inplace=True)
		todays_picks.index.name = 'Ticker'
		return todays_picks

	def GetHighestPriceMomentumMulti(self, currentDate: datetime, filterOptions: dict, minPercentGain: float = 0.05):
		if not isinstance(filterOptions, dict):
			raise ValueError("filterOptions must be a dict like {filter_id: stock_count}")
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
			1: dict(sort='PC_1Year', mask=lambda df: (df['PC_1Month3WeekEMA'] > minPC_1Day) ), #CAGR ~38–40%, inconsistent behavior, mediocre Sharpe, Broad / Noisy Exposure
			2: dict(sort='PC_1Year', mask=lambda df: (df['PC_1Year'] > minPercentGain)), #very high growth
			3: dict(sort='PC_1Year', mask=lambda df: ((df['PC_1Year'] > minPercentGain) & (df['PC_1Month3WeekEMA'] > 0))), 
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

	def GetHighestPriceMomentum(self, currentDate:datetime, stocksToReturn:int = 10, filterOption:int = 5, minPercentGain:float = 0.05, allocateByTargetHoldings:bool = False, allocateByPointValue=False, useRollingWindow=False):
		filter_options = {filterOption: stocksToReturn}
		multiverse_candidates = self.GetHighestPriceMomentumMulti(currentDate=currentDate, filterOptions=filter_options, minPercentGain=minPercentGain)
		todays_picks = multiverse_candidates.get(filterOption, pd.DataFrame())
		if allocateByTargetHoldings or allocateByPointValue or useRollingWindow: #These group the results, otherwise you get full columns
			if todays_picks is None or todays_picks.empty:
				return pd.DataFrame(columns=["TargetHoldings", "Point_Value"]).rename_axis("Ticker")
			daily = todays_picks.groupby(level=0).agg(TargetHoldings=("Point_Value", "size"), Point_Value=("Point_Value", "mean"))
			if useRollingWindow:
				daily = self._rolling_history_append(currentDate=currentDate, todays_picks=daily, max_picks=stocksToReturn)
			if allocateByPointValue:
				daily["TargetHoldings"] *= daily["Point_Value"]
				daily["TargetHoldings"] /= daily["TargetHoldings"].sum()
			todays_picks = daily
		return todays_picks

	def GetPicksBlended(self, currentDate:date, filter1:int = 3, filter2: int = 3, filter3: int = 1, filter4: int = 5, minPercentGain:float = 0.05, useRollingWindow:bool = True):
		#generates list of tickers with TargetHoldings which indicate proportion of holdings	
		filter_options = {filter1:3, filter2:3, filter3:3, filter4:4}
		multiverse_candidates = self.GetHighestPriceMomentumMulti(currentDate=currentDate, filterOptions=filter_options, minPercentGain=minPercentGain)
		df1 = multiverse_candidates.get(filter1, pd.DataFrame())
		df2 = multiverse_candidates.get(filter2, pd.DataFrame())
		df3 = multiverse_candidates.get(filter3, pd.DataFrame())
		df4 = multiverse_candidates.get(filter4, pd.DataFrame())
		todays_picks = pd.concat([df1, df2, df3, df4], sort=True)
		todays_picks = todays_picks.groupby(level=0).agg(TargetHoldings=('Point_Value', 'size'), Point_Value=('Point_Value', 'last'))
		todays_picks.sort_values('TargetHoldings', axis=0, ascending=False, inplace=True, kind='quicksort', na_position='last')
		todays_picks.index.name='Ticker'
		if useRollingWindow: todays_picks = self._rolling_history_append(currentDate=currentDate, todays_picks=todays_picks, max_picks=15)
		return todays_picks

	def GetPicksBlendedSQL(self, currentDate:date, sqlHistory:int=90):
		result = None
		db = PTADatabase()
		if db.Open():
			SQL = "select * from fn_GetBlendedPicks('" + str(currentDate) + "', " + str(sqlHistory) + ")"
			result = db.DataFrameFromSQL(sql=SQL, indexName='Ticker')
		db.Close()
		return result				
	
	def GeneratePicksBlendedSQL(self, startDate: pd.Timestamp = None, endDate: pd.Timestamp = None, replaceExisting:bool=False, verbose:bool=False):
		db = PTADatabase()
		if not db.Open(): return False
		startDate = ToTimestamp(startDate or self._startDate)
		startDate = max(startDate, self._startDate)
		endDate = ToTimestamp(endDate or self._endDate)
		endDate = max(endDate, self._endDate)
		current_date = startDate
		prev_month = -1
		print(f" GeneratePicksBlendedSQL from {startDate} to {endDate}")				
		existing_dates = pd.to_datetime(db.ScalarListFromSQL("SELECT Date FROM PicksBlendedDaily WHERE [Date]>=:startDate AND [Date]<=:endDate ORDER BY Date", {"startDate": startDate, "endDate": endDate}, column="Date"))
		full_price_index = self.priceData[0].historicalPrices.index.sort_values().unique()
		missing_dates = full_price_index[~full_price_index.isin(existing_dates)]
		target_dates = full_price_index if replaceExisting else missing_dates
		for current_date in target_dates:
			if verbose: print(f" GeneratePicksBlendedSQL: Blended using default filters for date {current_date}")
			result = self.GetPicksBlended(currentDate=current_date)
			if len(result) == 0:
				if verbose: print(" GeneratePicksBlendedSQL: No data found.")
			else:
				result['Date'] = current_date 
				result['TotalStocks'] = len(self._tickerList) 
				result = result[['Date', 'TargetHoldings', 'Point_Value', 'TotalStocks']]
				result.index.name='Ticker'
				if verbose: print(result)
				db.ExecSQL("DELETE FROM PicksBlendedDaily WHERE Date='" + str(current_date) + "'")
				db.DataFrameToSQL(result, tableName='PicksBlendedDaily', indexAsColumn=True, clearExistingData=False)
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
		filters = {0:250}
		multiverse_candidates = self.GetHighestPriceMomentumMulti(currentDate=forDate, filterOptions=filters)
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
		smoothed.dispersion = hist["dispersion"].mean()
		smoothed.momentum_autocorr = hist["momentum_autocorr"].mean()
		smoothed.leadership_tilt = hist["leadership_tilt"].mean()
		smoothed.state_confidence = hist["state_confidence"].mean()
		return smoothed
	
	def GetAdaptiveConvexPicks(self, currentDate):		
		filters = {0:250, 3:6, 4:6, 5:6}
		multiverse_candidates = self.GetHighestPriceMomentumMulti(currentDate=currentDate, filterOptions=filters)
		universe_size = len(multiverse_candidates[0])
		market_state = self._get_market_state_smoothed(currentDate, universe_size)
		if not market_state:
			return EMPTY_RESULT.copy()
		filters = market_state.GetExecutionFilters()
		todays_picks = self._blend_from_multiverse(multiverse_candidates, filters)
		self.pickHistoryWindowSize = market_state.GetRollingWindowSize()
		todays_picks = self._rolling_history_append(currentDate=currentDate, todays_picks=todays_picks)
		return todays_picks	

def Generate_PicksBlendedSQL_DateRange(startYear:int=None, years: int=0, replaceExisting:bool=False, verbose:bool=False):
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
		print(f" Generate_PicksBlendedSQL_DateRange from {FormatDate(startDate)} to {FormatDate(endDate)}")				
		picker = StockPicker(startDate=startDate, endDate=endDate)
		p = PricingData(CONSTANTS.CASH_TICKER)
		p.LoadHistory(requestedStartDate=startDate, requestedEndDate=endDate)
		full_price_index = p.historicalPrices.sort_index().index.unique()
		existing_dates = pd.to_datetime(db.ScalarListFromSQL("SELECT Date FROM PicksBlendedDaily WHERE [Date]>=:startDate AND [Date]<=:endDate ORDER BY Date", {"startDate": startDate, "endDate": endDate}, column="Date"))
		missing_dates = full_price_index[~full_price_index.isin(existing_dates)]
		target_dates = full_price_index if replaceExisting else missing_dates
		monthly_starts = target_dates[target_dates.to_series().dt.month != target_dates.to_series().dt.month.shift()]
		for month_start in monthly_starts:
			month_end = month_start + pd.offsets.MonthEnd(0)
			if verbose: print(f" Generate_PicksBlended_DateRange: Getting tickers for month {FormatDate(month_start)}")				
			new_tickers = TickerLists.GetTickerListSQL(year=month_start.year, month=month_start.month, SP500Only=False, filterByFundamentals=False) 
			if len(new_tickers) > 0:
				if verbose: print(f" Generate_PicksBlendedSQL_DateRange: Re-query tickers found {len(new_tickers)} instead of previous {len(tickers)}")
				tickers = new_tickers
			picker.AlignToList(tickers)			
			TotalValidCandidates = len(picker._tickerList) 
			if verbose: print(f" Generate_PicksBlendedSQL_DateRange: Running PicksBlended generation on {TotalValidCandidates} stocks {FormatDate(month_start)} to {FormatDate(month_end)}")		
			if TotalValidCandidates==0: assert(False)
			picker.GeneratePicksBlendedSQL(startDate=month_start, endDate=month_end, replaceExisting=replaceExisting, verbose=verbose)
	db.Close()
		