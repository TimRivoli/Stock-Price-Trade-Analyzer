import pandas as pd
START_DATE_DEFAULT = pd.Timestamp('1980-01-01')
BASE_FIELD_LIST = ['Open','Close','High','Low','Volume']
REQUIRED_LOOKBACK = (365 * 2) + 15 #Calendar days, for when dates are trimmed this amount will be privetly held for calcuations
CASH_TICKER = '$$$$'
TRADING_MONTH = 21
TRADING_YEAR = 252
CALENDAR_YEAR = 365
TRADING_MONTH = 21
ADAPTIVE_CONVEX_STATE_TABLE = 'MarketStates'
RATE_LIMITING_MAX_TURNOVER_PCT: float = 0.20
TRIM_PROFITS_UNIT_PCT = 0.1
BLOCK_REFRESHING_FOR_BACKTESTING = False

