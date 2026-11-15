# File: _classes/RLEnvironment.py
import numpy as np
import pandas as pd
import _classes.Constants as CONSTANTS
from _classes.Trading import TradingModel, TradeModelParams
from _classes.Selection import StockPicker, AdaptiveConvexMarketState
from _classes.Prices import PricingData

HOLDING_PERIOD_DAYS = 40   # Target 30-60 day holds; use 40 as midpoint
REEVAL_INTERVAL     = 5    # Days between agent decisions (weekly, matches AlphaTrader reEvaluationInterval)

# Action decoding table
FILTER_BLENDS = {
    0: [3, 3, 9, 9, 1, 4, 4],   # CONVEX_AGGRESSIVE
    1: [3, 3, 9, 9, 8, 1],       # CONVEX_STANDARD
    2: [1, 3, 3, 9, 9, 8],       # QUALITY_ANCHORED
    3: [3, 3, 9, 9],              # MOMENTUM_PURE
    4: [1, 3, 9, 4, 8],           # DISCOVERY
    5: [(1,3),(3,3),(9,3),(9,3),(6,1)],  # DEFAULT_BLEND (tuple format for GetPicksBlended)
    6: [8, 8, 3, 9],              # DEFENSIVE
    7: None,                      # CASH
}
STOCK_COUNTS = [5, 7, 9, 12, 15]
N_BLENDS = len(FILTER_BLENDS)
N_COUNTS = len(STOCK_COUNTS)


def decode_action(action_idx: int) -> tuple:
    """Convert flat action index → (blend_key, stock_count, rebalance_now)
    Action space: 8 blends × 2 (hold/rebalance) = 16 actions.
    Stock count is fixed at 9 (typical AdaptiveConvex default).
    """
    rebalance = action_idx % 2
    blend_key = (action_idx // 2) % N_BLENDS
    return blend_key, 9, bool(rebalance)


class RLTradingEnvironment:
    """
    Gym-style wrapper around TradingModel.
    
    Episode = one calendar year of daily trading.
    Step    = one re-evaluation interval (REEVAL_INTERVAL trading days).
    Reward  = portfolio value % change over the step period.
    """
    
    def __init__(self, params: TradeModelParams, picker: StockPicker):
        self.params = params
        self.picker = picker
        self.tm = None
        self._last_portfolio_value = None
        self._peak_value = None
        self._step_count = 0
        self._last_action = None
        self._last_market_state = None
    
    def reset(self) -> np.ndarray:
        """Initialize a new trading episode. Returns initial state vector."""
        self.tm = TradingModel(
            modelName=self.params.modelName,
            startingTicker=CONSTANTS.CASH_TICKER,
            startDate=self.params.startDate,
            durationInYears=self.params.durationInYears,
            totalFunds=self.params.portfolioSize,
            verbose=False
        )
        # Reset picker caches so each episode starts with a clean rolling window.
        # Without this, _pick_history and _adaptive_history_df from the previous
        # episode bleed in, causing _update_pick_history to return None whenever
        # the new episode's start date precedes the previous episode's end date
        # (which happens on every pass after the first).
        self.picker._pick_history = None
        self.picker._adaptive_history_df = None
        self._last_portfolio_value = self.params.portfolioSize
        self._peak_value = self.params.portfolioSize
        self._step_count = 0
        self._last_action_encoded = [0.0, 0.0]
        return self._get_state()
    
    def step(self, action_idx: int) -> tuple:
        """
        Execute one agent step (covers REEVAL_INTERVAL trading days).
        Returns: (next_state, reward, done, info)
        """
        blend_key, stock_count, rebalance_now = decode_action(action_idx)
        self._last_action = (blend_key, stock_count, rebalance_now)
        
        cash, assets = self.tm.GetValue()
        value_before = cash + assets
        self._last_portfolio_value = value_before  # snapshot start-of-step value for port_1m feature

        # Align positions, then advance REEVAL_INTERVAL days.
        # With REEVAL_INTERVAL=5 this is one call per 5-day window — identical to
        # AlphaTrader's reEvaluationInterval, so the 20% rate limit executes correctly.
        if rebalance_now or self._step_count == 0:
            if blend_key < 7:
                candidates = self._get_candidates(blend_key, stock_count)
                if candidates is not None:
                    self.tm.AlignPositions(
                        targetPositions=candidates,
                        rateLimitTransactions=self.params.rateLimitTransactions,
                        shopBuyPercent=self.params.shopBuyPercent,
                        shopSellPercent=self.params.shopSellPercent
                    )
            else:
                self.tm.AlignPositions(targetPositions=None)
        for _ in range(REEVAL_INTERVAL):
            if self.tm.ModelCompleted():
                break
            self.tm.ProcessDay()

        cash, assets = self.tm.GetValue()
        value_after = cash + assets
        done = self.tm.ModelCompleted()

        # Reward: log return over the full REEVAL_INTERVAL period
        reward = np.log(value_after / value_before) if value_before > 0 else 0.0

        # Small opportunity-cost drag for >90% idle cash
        cash_pct = cash / max(value_after, 1e-8)
        if cash_pct > 0.9:
            reward -= 0.0005

        # Drawdown penalty relative to episode peak
        self._peak_value = max(self._peak_value or value_after, value_after)
        drawdown = max(0.0, (self._peak_value - value_after) / max(self._peak_value, 1e-8))
        if drawdown > 0.05:
            reward -= 0.003 * drawdown

        self._step_count += 1
        self._last_action_encoded = [blend_key / (N_BLENDS - 1), float(rebalance_now)]

        next_state = self._get_state()
        info = {
            "portfolio_value": value_after,
            "blend_key": blend_key,
            "stock_count": stock_count,
            "rebalanced": rebalance_now,
            "avg_position_age": self._get_avg_position_age(),
        }
        return next_state, reward, done, info
    
    def _get_candidates(self, blend_key: int, stock_count: int) -> pd.DataFrame:
        """Translate blend key + stock count into picker result."""
        blend_filters = FILTER_BLENDS[blend_key]
        current_date = self.tm.currentDate
        if current_date is None:
            return None

        if isinstance(blend_filters[0], tuple):
            result = self.picker.GetPicksBlended(
                currentDate=current_date,
                filterOptions=blend_filters,
                useRollingWindow=(self.params.pickHistoryWindow > 0)
            )
        else:
            filter_tuples = [(f, max(stock_count // len(blend_filters), 1)) for f in blend_filters]
            result = self.picker.GetPicksBlended(
                currentDate=current_date,
                filterOptions=filter_tuples,
                useRollingWindow=(self.params.pickHistoryWindow > 0)
            )

        if result is None or result.empty:
            return None
        # Strip the CASH_RESULT placeholder row — when the picker finds no valid stocks
        # it returns a DataFrame containing only the cash ticker instead of None.
        # Passing that to AlignPositions would leave the portfolio in cash silently.
        result = result[result.index != CONSTANTS.CASH_TICKER]
        return result if not result.empty else None
    
    def _get_state(self) -> np.ndarray:
        """Build the state vector from current market + portfolio conditions."""
        # Market regime signals
        market_state = self._get_market_state()
        
        # Universe momentum summary
        universe_feats = self._get_universe_features()
        
        # Portfolio state
        portfolio_feats = self._get_portfolio_features()
        
        # Calendar features
        d = self.tm.currentDate if (self.tm and hasattr(self.tm, 'currentDate')) else None
        if d is not None:
            day_of_year = d.day_of_year if hasattr(d, 'day_of_year') else d.timetuple().tm_yday
            cal_feats = [
                np.sin(2 * np.pi * day_of_year / 365),
                np.cos(2 * np.pi * day_of_year / 365),
                d.month / 12.0,
            ]
        else:
            cal_feats = [0.0, 1.0, 0.5]
        
        state = np.concatenate([
            market_state,
            universe_feats,
            portfolio_feats,
            cal_feats,
            np.array(self._last_action_encoded, dtype=np.float32),
        ]).astype(np.float32)
        
        # Safety: replace NaN/Inf with 0
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        return state
    
    def _get_market_state(self) -> np.ndarray:
        """Extract regime features from AdaptiveConvexMarketState."""
        try:
            universe_size = len(self.picker._tickerList)
            ms = self.picker._get_market_state_smoothed(
                self.tm.currentDate, universe_size
            )
            if ms is None:
                return np.zeros(14, dtype=np.float32)
            return np.array([
                ms.disp_norm,
                ms.conviction_score,
                ms.momentum_autocorr,
                ms.downside_volatility,
                ms.stress_index,
                ms.velocity_ewm,
                ms.leadership_tilt,
                ms.corr_6m_1m,
                ms.corr_1y_1m,
                ms.state_confidence,
                float(ms.geometry_state == "CONVEX"),
                float(ms.geometry_state == "LINEAR"),
                float(ms.expansion_state == "EXPANDING"),
                float(ms.expansion_state == "CONTRACTING"),
            ], dtype=np.float32)
        except Exception:
            return np.zeros(14, dtype=np.float32)
    
    def _get_universe_features(self) -> np.ndarray:
        """Summarize cross-sectional momentum of the current universe."""
        try:
            rows = []
            for p in self.picker.priceData:
                if p.statsLoaded:
                    snap = p.GetPriceSnapshot(self.tm.currentDate)
                    if snap:
                        rows.append({
                            'pc1y': getattr(snap, 'PC_1Year', np.nan),
                            'pc3m': getattr(snap, 'PC_3Month', np.nan),
                            'pc1m': getattr(snap, 'PC_1Month3WeekEMA', np.nan),
                        })
            if not rows:
                return np.zeros(7, dtype=np.float32)
            df = pd.DataFrame(rows).dropna()
            if df.empty:
                return np.zeros(7, dtype=np.float32)
            pc1y = df['pc1y']
            pc1m = df['pc1m']
            top10 = pc1y.nlargest(10).mean() if len(pc1y) >= 10 else pc1y.mean()
            top3 = pc1y.nlargest(3).sum() / max(pc1y.sum(), 1e-6)
            return np.array([
                float(np.clip(pc1y.mean(),        -1.0, 3.0)),   # 1yr mean; clip bull-market extremes
                float(np.clip(pc1y.std(),          0.0, 1.5)),   # 1yr cross-sectional std
                float(np.clip(top10,              -0.5, 3.0)),   # top-10 1yr mean
                float(np.clip(df['pc3m'].mean(),  -0.5, 1.0)),   # 3-month mean
                float((pc1y > 0).mean()),
                float((pc1m > 0).mean()),
                float(np.clip(top3, 0, 1)),
            ], dtype=np.float32)
        except Exception:
            return np.zeros(7, dtype=np.float32)
    
    def _get_portfolio_features(self) -> np.ndarray:
        """Extract current portfolio state features."""
        try:
            cash_val, assets = self.tm.GetValue()
            total = cash_val + assets
            cash = self.tm.GetAvailableCash()
            positions = self.tm.GetPositions()
            n_pos = len(positions) if positions else 0
            cash_pct = cash / total if total > 0 else 1.0

            if positions and len(positions) > 0:
                ages = [
                    (self.tm.currentDate - p.dateBuyOrderFilled).days
                    for p in positions if hasattr(p, 'dateBuyOrderFilled')
                ]
                pnls = [
                    (p.latestPrice / p.purchasePrice - 1)
                    for p in positions
                    if hasattr(p, 'purchasePrice') and p.purchasePrice > 0
                ]
                avg_age = np.mean(ages) if ages else 0.0
                avg_pnl = np.mean(pnls) if pnls else 0.0
                worst_pnl = np.min(pnls) if pnls else 0.0
            else:
                avg_age, avg_pnl, worst_pnl = 0.0, 0.0, 0.0
            
            port_1m = (total / self._last_portfolio_value - 1) if self._last_portfolio_value else 0.0
            
            return np.array([
                float(np.clip(cash_pct, 0, 1)),
                float(n_pos / 20.0),                         # Normalized by max expected
                float(np.clip(avg_age / 60.0, 0, 2)),        # Normalized to target hold
                float(np.clip(avg_pnl, -0.5, 0.5)),
                float(np.clip(worst_pnl, -0.5, 0.5)),
                float(np.clip(port_1m, -0.3, 0.3)),
                float(np.clip(self._step_count / (CONSTANTS.TRADING_YEAR / REEVAL_INTERVAL), 0, 2.0)),
            ], dtype=np.float32)
        except Exception:
            return np.zeros(7, dtype=np.float32)
    
    def _get_avg_position_age(self) -> float:
        """Average days held across open positions."""
        try:
            positions = self.tm.GetPositions()
            if not positions:
                return 0.0
            ages = [
                (self.tm.currentDate - p.dateBuyOrderFilled).days
                for p in positions if hasattr(p, 'dateBuyOrderFilled')
            ]
            return float(np.mean(ages)) if ages else 0.0
        except Exception:
            return 0.0