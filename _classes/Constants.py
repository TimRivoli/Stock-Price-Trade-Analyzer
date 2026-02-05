import pandas as pd
START_DATE_DEFAULT = pd.Timestamp('1980-01-01')
BASE_FIELD_LIST = ['Open','Close','High','Low','Volume']
REQUIRED_LOOKBACK = (365 * 2) + 15 #Calendar days, for when dates are trimmed this amount will be privetly held for calcuations
CASH_TICKER = '$$$$'
TRADING_MONTH = 21
TRADING_YEAR = 252
CALENDAR_YEAR = 365
TRADING_MONTH = 21
ADAPTIVE_CONVEX_STATE_TABLE = 'AdaptivePV_RegimeSeries'
BLOCK_REFRESHING_FOR_BACKTESTING = False
