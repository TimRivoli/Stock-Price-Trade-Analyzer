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

BASE_RAMP_UP_PER_DAY = 0.06     # convex ramps fast
BASE_RAMP_DOWN_PER_DAY = 0.03   # convex decays slower (asymmetry)
ADAPTIVE_WARMUP_DAYS = 4  # small, deterministic warmup for the convex engine

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
			endDate = (today - pd.offsets.BDay(15)).date() #Don't go crazy with recent data or you will force reloads
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
						tickers = TickerLists.GetTickerListSQL(year=current_date.year, month=current_date.month, SP500Only=False, filterByFundamtals=False, marketCapMin=100) 
						TotalStocks=len(tickers)
						if verbose: print(" Generate_PicksBlended_DateRange: Total stocks: " + str(TotalStocks))
						picker.AlignToList(tickers)			
						TotalValidCandidates = len(picker._tickerList) 
						if verbose: print(' Generate_PicksBlended_DateRange: Running PicksBlended generation on ' + str(TotalValidCandidates) + ' of ' + str(TotalStocks) + ' stocks from ' + str(startDate) + ' to ' + str(endDate))		
						if TotalValidCandidates==0: assert(False)
						prev_month = current_date.month
					if verbose: print(' Generate_PicksBlended_DateRange: Blended 3.3.44.PV Picks - ' + str(current_date))
					result = picker.GPicksBlended(currentDate=currentDate)
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

@dataclass
class AdaptiveConvexParams:
	# Identity
	modelName: str
	as_of_date: pd.Timestamp
	convex_duration : int

	# Regime inputs
	dispersion: float
	momentum_autocorr: float
	downside_volatility: float
	stress_index: float

	# Allocation weights (must sum to 1.0)
	convex_weight: float
	linear_weight: float
	defensive_weight: float
	cash_weight: float

	# Optional diagnostics / state
	regime_label: Optional[str] = None
	hysteresis_state: Optional[str] = None

	def Validate(self, tol: float = 1e-6):
		total = ( self.convex_weight + self.linear_weight + self.defensive_weight + self.cash_weight )
		if abs(total - 1.0) > tol:
			raise ValueError(f" AdaptiveConvex weights do not sum to 1.0 (got {total})")

	def Save(self):
		db = PTADatabase()
		if db.Open():
			if not self.modelName: self.modelName = 'AdaptiveConvex'
			sql_delete = f"DELETE FROM {CONSTANTS.ADAPTIVE_CONVEX_STATE_TABLE} WHERE modelName = :modelName AND asOfDate = :asOfDate"
			params = {"modelName": self.modelName,"asOfDate": self.as_of_date}
			db.ExecSQL(sql_delete, params)
			sql_insert = f"INSERT INTO {CONSTANTS.ADAPTIVE_CONVEX_STATE_TABLE} (modelName, asOfDate, convex_duration , dispersion, momentum_autocorr, downside_volatility, stress_index, convex_weight, linear_weight, defensive_weight, cash_weight, regime_label, hysteresis_state) VALUES (:modelName, :asOfDate, :convex_duration , :dispersion, :momentum_autocorr, :downside_volatility, :stress_index, :convex_weight, :linear_weight, :defensive_weight, :cash_weight, :regime_label, :hysteresis_state)"
			params.update({
				"dispersion": float(self.dispersion),
				"momentum_autocorr": float(self.momentum_autocorr),
				"downside_volatility": float(self.downside_volatility),
				"convex_duration": int(self.convex_duration),
				"stress_index": float(self.stress_index),
				"convex_weight": float(self.convex_weight),
				"linear_weight": float(self.linear_weight),
				"defensive_weight": float(self.defensive_weight),
				"cash_weight": float(self.cash_weight),
				"regime_label": self.regime_label,
				"hysteresis_state": self.hysteresis_state
			})
			db.ExecSQL(sql_insert, params)
			db.Close()
			
class StockPicker():
	def __init__(self, startDate:datetime=None, endDate:datetime=None): 
		self.priceData = []
		self._tickerList = []
		self._startDate = startDate
		self._endDate = endDate
		self._adaptive_is_warming = False
		self._adaptive_last_date = None
		self.convex_state = False
		self.downside_volatility=0
		self.convex_duration = 0
		self.hysteresis_state = "NEUTRAL"

#-------------------------------------------- Housekeeping Load/Unload Tickers -----------------------------------------------
	def AddTicker(self, ticker:str):
		if not ticker in self._tickerList:
			p = PricingData(ticker)
			if p.LoadHistory(self._startDate, self._endDate, verbose=False): 
				p.CalculateStats(fullStats=True)
				self.priceData.append(p)
				self._tickerList.append(ticker)

	def RemoveTicker(self, ticker:str, verbose:bool=False):
		i=len(self.priceData)-1
		while i >= 0:
			if ticker == self.priceData[i].ticker:
				if verbose: print(" Removing ticker " + ticker)
				self.priceData.pop(i)
				self._tickerList.remove(ticker)
			i -=1
		if ticker in self._tickerList: 
			print(" Error removing ticker " + ticker)
			print(len(self.priceData))	
			print(self._tickerList)	
			assert(False)

	def AlignToList(self, newList:list, verbose:bool=False):
		#Add/Remove tickers until they match the given list
		i=len(self.priceData)-1
		while i >= 0:
			ticker = self.priceData[i].ticker
			if not ticker in newList:
				if verbose: print(" Removing ticker " + ticker)
				self.priceData.pop(i)
				self._tickerList.remove(ticker)
			i -=1
		pbar = tqdm(total=len(newList), desc=" AlignToList adding tickers")
		for t in newList:
			self.AddTicker(t)
			pbar.update(1)
		pbar.close()
		
	def TickerExists(self, ticker:str):
		return ticker in self._tickerList
	
	def TickerCount(self):
		return len(self._tickerList)

	def NormalizePrices(self):
		for i in range(len(self.priceData)):
			self.priceData[i].NormalizePrices()		
			
#-------------------------------------------- Selection routine -----------------------------------------------
	
	def GetHighestPriceMomentumMulti(self, currentDate: datetime, stocksToReturn: int = 5, minPercentGain: float = 0.05, verbose: bool = False, filterOptions=(0, 1, 2, 3, 4, 44, 5, 6)):
		EMPTY_RESULT = pd.DataFrame(columns=['TargetHoldings', 'Point_Value'])
		#EMPTY_RESULT.loc[CONSTANTS.CASH_TICKER] = [1, 1]
		EMPTY_RESULT.index.name = 'Ticker'

		candidates = {}
		stocksToReturn = int(stocksToReturn)
		currentDate = ToTimestamp(currentDate)
		max_allowed_date = (pd.Timestamp.now().normalize() - pd.offsets.BusinessDay(1)).to_pydatetime()
		if currentDate > max_allowed_date:
			currentDate = max_allowed_date
		rows = []
		for pd_obj in self.priceData:
			ticker = pd_obj.ticker
			df = pd_obj.historicalPrices
			if currentDate not in df.index:
				if verbose:  print(f" GetHighestPriceMomentumMulti: {ticker} missing {currentDate}")
				continue
			row = df.loc[currentDate]

			if min( row['HP_2Yr'], row['HP_1Yr'], row['HP_6Mo'], row['HP_2Mo'], row['HP_1Mo'], row['Average_5Day'] ) <= 0:
				continue
			rows.append({
				'Ticker': ticker,
				'hp2Year': row['HP_2Yr'],
				'hp1Year': row['HP_1Yr'],
				'hp6mo': row['HP_6Mo'],
				'hp3mo': row['HP_3Mo'],
				'hp2mo': row['HP_2Mo'],
				'hp1mo': row['HP_1Mo'],
				'Average_5Day': row['Average_5Day'],
				'Average_2Day': row['Average_2Day'],
				'Average': row['Average'],
				'Channel_High': row['Channel_High'],
				'Channel_Low': row['Channel_Low'],
				'PC_2Year': row['PC_2Year'],
				'PC_1Year': row['PC_1Year'],
				'PC_6Month': row['PC_6Month'],
				'PC_3Month': row['PC_3Month'],
				'PC_2Month': row['PC_2Month'],
				'PC_1Month': row['PC_1Month'],
				'PC_3Day': row['PC_3Day'],
				'PC_1Day': row['PC_1Day'],
				'PC_1Month3WeekEMA': row['PC_1Month3WeekEMA'],
				'Deviation_15Day': row['Deviation_15Day'],
				'Deviation_10Day': row['Deviation_10Day'],
				'Deviation_5Day': row['Deviation_5Day'],
				'Deviation_1Day': row['Deviation_1Day'],
				'Gain_Monthly': row['Gain_Monthly'],
				'LossStd_1Year': row['LossStd_1Year'],
				'Point_Value': row['Point_Value'],
				'Comments': row.get('Comments', ''),
				'HasFullLookback': row.get('HasFullLookback', True),
				'latestEntry': pd_obj.historyEndDate
			})

		if not rows: 
			for opt in filterOptions: candidates[opt] = EMPTY_RESULT.copy()
			return candidates
		base = pd.DataFrame(rows).set_index('Ticker')
		base.index.name = 'Ticker'

		minPC_1Day = minPercentGain / CONSTANTS.TRADING_YEAR
		#More complex filters that I have tried have all decreased performance which is why these are simple
		#Greatest factors for improvement are high 1yr return and a very low selection of stocks, like 1-3
		#Best way to compensate for few stocks is to blend filters of different strengths
		filter_map = {
			0: dict( sort='PC_1Year', mask=lambda df: (df['Average_5Day'] > 0) ),
			1: dict(sort='PC_1Year', mask=lambda df: ((df['PC_1Year'] > minPercentGain) & (df['PC_1Month3WeekEMA'] > 0) & ( df['PC_1Year'] / CONSTANTS.TRADING_YEAR > df['PC_1Month3WeekEMA'] / CONSTANTS.TRADING_MONTH ) ) ),
			2: dict( sort='PC_1Year', mask=lambda df: (df['PC_1Year'] > minPercentGain) ),
			3: dict( sort='PC_1Year', mask=lambda df: ( (df['PC_1Year'] > minPercentGain) & (df['PC_1Month3WeekEMA'] > 0) ) ),
			4: dict( sort='PC_1Month3WeekEMA', mask=lambda df: (df['PC_1Month3WeekEMA'] > minPC_1Day) ),
			44: dict( sort='PC_1Year', mask=lambda df: (df['PC_1Month3WeekEMA'] > minPC_1Day) ),
			5: dict( sort='Point_Value', mask=lambda df: ( (df['PC_1Year'] > minPercentGain) & (df['Point_Value'] > 0) ) ),
			6: dict( sort='LossStd_1Year', mask=lambda df: ( (df['PC_1Year'] > 0) & (df['LossStd_1Year'].between(0.06, 0.15)) & (df['PC_3Month'] > 0) & (df['PC_1Month'] > 0) ) )
		}
		for opt in filterOptions:
			spec = filter_map.get(opt)
			if spec is None:
				continue
			df = base.loc[spec['mask'](base)]
			if df.empty:
				candidates[opt] = EMPTY_RESULT.copy()
				continue
			df = df.sort_values(spec['sort'], ascending=False)
			candidates[opt] = df.head(stocksToReturn)            
		return candidates

	def GetHighestPriceMomentum(self, currentDate: datetime, stocksToReturn: int = 5, filterOption: int = 3, minPercentGain: float = 0.05, verbose: bool = False):
		candidates = self.GetHighestPriceMomentumMulti(currentDate=currentDate, stocksToReturn=stocksToReturn, minPercentGain=minPercentGain, verbose=verbose, filterOptions=(filterOption,))
		return candidates.get(filterOption, pd.DataFrame())

	def GetPicksBlended(self, currentDate:date):
		#generates list of tickers with TargetHoldings which indicate proportion of holdings	
		list1 = self.GetHighestPriceMomentum(currentDate=currentDate, stocksToReturn=2, filterOption=3)
		list2 = self.GetHighestPriceMomentum(currentDate=currentDate, stocksToReturn=2, filterOption=3)
		list3 = self.GetHighestPriceMomentum(currentDate=currentDate, stocksToReturn=2, filterOption=44)
		list4 = self.GetHighestPriceMomentum(currentDate=currentDate, stocksToReturn=5, filterOption=5)
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

	def compute_cross_sectional_dispersion(self, currentDate, min_valid=20):
		"""
		Cross-sectional standard deviation of 1Y momentum (PC_1Year).
		Measures opportunity dispersion for convex strategies.
		"""
		values = []
		for pdata in self.priceData:
			if not pdata.statsLoaded:
				continue
			df = pdata.historicalPrices
			if currentDate not in df.index:
				continue
			pc1y = df.loc[currentDate, 'PC_1Year']
			if pd.notna(pc1y):
				values.append(pc1y)
		if len(values) < min_valid:
			return 0.0
		return float(np.std(values, ddof=1))

	def compute_momentum_autocorr(self,	currentDate, lookback_months=6, min_valid=20):
		"""
		Average autocorrelation of 1M momentum across universe.
		Positive = momentum persistence.
		Negative = mean reversion.
		"""
		autocorrs = []
		for pdata in self.priceData:
			if not pdata.statsLoaded:
				continue
			df = pdata.historicalPrices
			if currentDate not in df.index:
				continue
			series = df.loc[:currentDate, 'PC_1Month'].dropna()
			if len(series) < lookback_months + 1:
				continue
			s1 = series[-lookback_months:]
			s2 = series.shift(1)[-lookback_months:]
			if s1.std() == 0 or s2.std() == 0:
				continue
			corr = s1.corr(s2)
			if pd.notna(corr):
				autocorrs.append(corr)
		if len(autocorrs) < min_valid:
			return 0.0
		return float(np.mean(autocorrs))

	def compute_downside_volatility(self, currentDate, min_valid=20):
		"""
		Median downside volatility (LossStd_1Year) across universe.
		Confirms whether volatility is being rewarded.
		"""
		losses = []
		for pdata in self.priceData:
			if not pdata.statsLoaded:
				continue
			df = pdata.historicalPrices
			if currentDate not in df.index:
				continue
			loss_std = df.loc[currentDate, 'LossStd_1Year']
			if pd.notna(loss_std):
				losses.append(loss_std)
		if len(losses) < min_valid:
			return 0.0
		return float(np.median(losses))
		
	def compute_stress_index(self, currentDate, dispersion, autocorr):
		self.downside_volatility = self.compute_downside_volatility(currentDate)		
		downside_stress = (self.downside_volatility - 0.08) / (0.25 - 0.08) # Typical historical range ~0.05–0.30 in data
		vol_stress = float(np.clip(downside_stress, 0.0, 1.0))
		dispersion_stress = np.clip((0.30 - dispersion) / 0.30, 0.0, 1.0) # Opportunity decay (low dispersion is hostile)		
		autocorr_stress = np.clip(-autocorr / 0.2, 0.0, 1.0) # Momentum decay
		stress = (
			0.45 * vol_stress +
			0.35 * dispersion_stress +
			0.20 * autocorr_stress
		)
		return float(np.clip(stress, 0.0, 1.0))
	
	def update_convex_state(self, dispersion, autocorr):
		prev_on = bool(self.convex_state)
		if not prev_on:
			if dispersion >= 0.24 and autocorr > -0.05:
				self.convex_state = True
				self.hysteresis_state = "EXPANDING"
				return True
			if dispersion >= 0.22:
				self.convex_state = True
				self.hysteresis_state = "EXPANDING"
				return True
			self.convex_state = False
			self.hysteresis_state = "NEUTRAL"
			return False
		if dispersion >= 0.22:
			self.convex_state = True
			self.hysteresis_state = "ACTIVE"
			return True
		if autocorr < -0.1:
			self.convex_state = False
			self.hysteresis_state = "LOCKED_OUT"
			return False
		self.convex_state = False
		self.hysteresis_state = "CONTRACTING"
		return False
		
	def adaptive_engine_weights(self, dispersion, autocorr, dt_days: int = 1):
		# --- 1. Hysteresis gate (discrete, intentional) ---
		convex_state = self.update_convex_state(dispersion, autocorr)
		if not convex_state:
			target = {"convex": 0.0, "linear": 0.10, "defensive": 0.0, "cash": 0.90}
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
			convex_raw = disp_intensity * momentum_factor
			convex_w = float(np.clip(convex_raw, 0.0, 0.85))

			# --- 5. Smooth cash response ---
			min_cash = 0.05
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
		base_up = 0.20   # per-day speed when convex increases
		base_dn = 0.10   # per-day speed when convex decreases (slower decay)
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

	def _AdaptiveConvex_warmup(self, currentDate):
		if self._adaptive_is_warming or not self._startDate: return
		bdays = pd.bdate_range(self._startDate, currentDate)
		if len(bdays) <= 1: return
		self._adaptive_is_warming = True
		warmup_days = min(ADAPTIVE_WARMUP_DAYS, len(bdays) - 1)
		for d in bdays[-(warmup_days + 1):-1]:
			self.GetAdaptiveConvex(currentDate=d, modelName='Warmup')
		self._adaptive_is_warming = False

	def GetAdaptiveConvex(self, currentDate, modelName: str | None = None):
		def add_block(df, block_weight):
			if df is None or df.empty or block_weight <= 0:
				return
			out = pd.DataFrame(index=df.index)
			out["TargetHoldings"] = block_weight / len(df)
			frames.append(out)
		if self._adaptive_last_date is None:
			self._AdaptiveConvex_warmup(currentDate)
			dt_days = 1
		else:
			dt_days = Business_Days_Since(self._adaptive_last_date, currentDate)
		self._adaptive_last_date = currentDate
		dispersion = self.compute_cross_sectional_dispersion(currentDate)
		autocorr = self.compute_momentum_autocorr(currentDate)
		stress = self.compute_stress_index(currentDate, dispersion, autocorr)   
		if stress > 0.8 and self.convex_state: 
			print(' GetAdaptiveConvex: high stress level {}, putting on the brakes by setting convex_state = False')
			self.convex_state = False 

		weights = self.adaptive_engine_weights(dispersion, autocorr)
		regime_label = Regime_Label_From_Weights(weights)

		filters = {}
		if weights.get("convex", 0) > 0:
			filters[2] = 3    # PM_filter2
			self.convex_duration += 1
		else:
			self.convex_duration = 0
		if weights.get("linear", 0) > 0:
			filters[3] = 3    # PM_filter3
		if weights.get("defensive", 0) > 0:
			filters[44] = 2   # PM_filter44
		candidates = self.GetHighestPriceMomentumMulti(currentDate=currentDate, stocksToReturn=5, filterOptions=tuple(filters))
		frames = []
		if weights.get("convex", 0) > 0 and 2 in candidates:
			add_block(candidates[2], weights["convex"])
		if weights.get("linear", 0) > 0 and 3 in candidates:
			add_block(candidates[3], weights["linear"])
		if weights.get("defensive", 0) > 0 and 44 in candidates:
			add_block(candidates[44], weights["defensive"])
		if weights.get("cash", 0) > 0:
			cash_df = pd.DataFrame({"TargetHoldings": [weights["cash"]]}, index=[CONSTANTS.CASH_TICKER])
			frames.append(cash_df)
		if not frames: return pd.DataFrame(columns=["TargetHoldings"]).rename_axis("Ticker")
		final = (pd.concat(frames).groupby(level=0)["TargetHoldings"].sum().to_frame())
		final["TargetHoldings"] /= final["TargetHoldings"].sum()
		if modelName:	#Save adaptive params to SQL
			params = AdaptiveConvexParams(
				modelName = modelName,
				as_of_date = currentDate,
				convex_duration  = self.convex_duration,
				dispersion = dispersion,
				momentum_autocorr = autocorr,
				downside_volatility = self.downside_volatility,
				stress_index = stress,
				convex_weight = weights["convex"],
				linear_weight = weights["linear"],
				defensive_weight = weights["defensive"],
				cash_weight = weights["cash"],
				regime_label = regime_label,
				hysteresis_state = self.hysteresis_state
			)
			params.Validate()
			params.Save()
		return final