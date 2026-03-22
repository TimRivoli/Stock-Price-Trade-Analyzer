# Stock Price Trade Analyzer

A Python 3 toolkit for downloading, analyzing, and backtesting stock trading strategies, with optional deep-learning price prediction via TensorFlow/Keras.

---

## Overview

This project gives you two tightly integrated modules:

- **PriceTradeAnalyzer** — historical price retrieval, technical analysis, and trading strategy backtesting
- **SeriesPrediction** — neural-network price forecasting (LSTM, CNN, GRU, BiLSTM, and more)

Whether you want to explore classical technical indicators, stress-test a trading strategy against decades of historical data, or experiment with deep learning for price prediction, this library provides a self-contained Python environment to do it.

---

## Modules

### PriceTradeAnalyzer

#### `PricingData`
Given a stock ticker symbol, `PricingData` fetches historical OHLCV data (via the Yahoo Finance library), caches it to CSV, and loads it into a Pandas DataFrame.

Key methods:

| Method | Description |
|---|---|
| `TrimToDateRange` | Slice data to a specific start/end date |
| `ConvertToPercentages` | Express prices as percentage changes |
| `NormalizePrices` | Normalize price series for comparison or ML input |
| `CalculateStats` | Compute EMA, Bollinger Bands, momentum, price channels, and more |
| `Graph` | Plot price charts for multiple time frames using matplotlib |

Several price-prediction helpers are also built in, including wrappers that delegate to the `SeriesPrediction` module for ML-based forecasts.

#### `Portfolio` and `TradingModel`
These classes simulate portfolio management and strategy execution against real historical prices.

**How it works:**
1. Pick the tickers and date range you want to test.
2. Define your buy/sell logic.
3. Run the backtest — the framework tracks daily portfolio value and every trade.
4. Results are exported as CSV and PNG so you can review performance later.

**Included example strategies** (see `EvaluateTradeModels.py`):
- **Buy and Hold** — baseline benchmark
- **Seasonal** — time-of-year entry and exit signals
- **Trending (2 variants)** — momentum-based approaches

**Additional tools:**
- `ExtendedDurationTest` — run any model across multiple time windows automatically
- `CompareModels` — side-by-side performance comparison of two strategies over a shared period

---

### SeriesPrediction

#### `StockPredictionNN`
A dual-output neural network built with TensorFlow and Keras that predicts both the **direction** and **magnitude** of a price move from a feature vector.

**Architecture:** Two fully connected hidden layers (64 units, ReLU) feeding into two separate output heads:

| Output | Activation | Loss | Description |
|---|---|---|---|
| `direction` | Sigmoid | Binary crossentropy | Probability that the next move is up |
| `magnitude` | ReLU | Huber | Expected size of the price move |

The two losses are combined during training with `direction` weighted at 1.0 and `magnitude` at 0.5, so the model prioritizes getting the direction right.

See `TrainPrices.py` for worked examples of training and evaluating the model.

---

## Database Backend (Optional)

The `database/` folder contains everything needed to swap the flat-file CSV backend for a **Microsoft SQL Server** database, unlocking more sophisticated backtesting and analysis across a large universe of stocks.

### Files

| File | Description |
|---|---|
| `PTA_Generate.sql` | Creates the `PTA` database, all tables, views, and stored procedures |
| `PriceDatabaseUpdate.py` | Orchestrates data downloads and database refreshes |
| `tickerlist.csv` | Seed list of tickers to populate the database on first run |
| `config.ini` | Connection settings (server, database name, credentials) |

### What the database adds

**Broader ticker coverage** — the updater seeds from `tickerlist.csv` and automatically pulls in S&P 500 and extended watchlist tickers, storing full price history, company fundamentals, and financial data.

**Automated price maintenance** — `PriceDatabaseUpdate.py` provides several update routines:

- `TickersFullRefresh` — bulk-downloads missing price history for up to 150 tickers at a time, then calls `sp_UpdateEverything` to recompute all derived stats
- `TickerDataRefresh` — lighter daily/monthly refresh for tickers with recent gaps
- `DownloadIntraday` — pulls current-day prices and merges them into daily history via the `sp_UpdateDailyFromIntraday` stored procedure
- `DownloadAllTickerInfo` — fills in missing company metadata and financials

**Deep trade analysis** — `Update_TradeModel_Trade_Analysis` enriches every historical trade record with entry/exit market context, including momentum scores, volatility measures, distance from the 200-day moving average, max favorable/adverse excursion, and max drawdown during the trade. This runs multi-threaded across up to 6 workers for speed.

**Working set and stock picking** — `RefreshPricesWorkingSet` builds a curated watchlist of up to ~300 momentum stocks (plus a hard-coded set of forced inclusions) and writes it to the `PricesWorkingSet` table. It also regenerates the `PicksBlended` table covering the past 30 days.

**Key database objects** (created by `PTA_Generate.sql`):

- `Tickers` — master ticker list with exchange, market cap, fundamentals, and S&P 500 membership
- `TickerFinancials` / `TickerHistoricalQualityFactors` — per-ticker financial and quality metrics over time
- `TradeModel_Trades` / `TradeModel_DailyValue` / `TradeModelComparisons` — full backtest output storage
- `TradeModel_Trade_Analysis` — enriched per-trade analytics
- `PricesWorkingSet` — current momentum-filtered working universe
- `Options_Sentiment_Daily` — options-derived sentiment scores
- `SP500ConstituentsMonthly` / `SP500ConstituentsYearly` — historical S&P 500 membership snapshots
- Stored procedures: `sp_UpdateEverything`, `sp_UpdateDailyFromIntraday`, `sp_UpdateTickerPrices`, `sp_UpdateTradeHistory`, and more

### Setup

1. Install and start **Microsoft SQL Server** (2019 or later recommended).
2. Run `PTA_Generate.sql` in SSMS or `sqlcmd` to create the `PTA` database and all objects.
3. Edit `database/config.ini` with your server name and credentials:

```ini
[Database]
usesqldriver = True
databaseserver = localhost
databasename = PTA
databaseusername = your_username
databasepassword = your_password
```

4. Run the initial population script:

```bash
python database/PriceDatabaseUpdate.py
```

This will seed the ticker list, download price history, and populate the working set. Subsequent runs will perform incremental updates only.

> **Note:** The database backend requires the `pyodbc` package and a working ODBC driver for SQL Server. The CSV-based workflow works without any database setup.

---

## Project Structure

```
Stock-Price-Trade-Analyzer/
├── _classes/
│   ├── PriceTradeAnalyzer.py   # PricingData, Portfolio, TradingModel
│   └── SeriesPrediction.py     # StockPredictionNN
├── database/
│   ├── PTA_Generate.sql        # Creates the SQL Server database and all objects
│   ├── PriceDatabaseUpdate.py  # Automated price/data refresh orchestration
│   ├── tickerlist.csv          # Seed ticker list for initial population
│   └── config.ini              # Database connection settings
├── EvaluatePrices.py           # Price analysis and charting examples
├── EvaluateTradeModels.py      # Strategy backtesting examples
├── TrainPrices.py              # ML model training examples
├── data/
│   ├── charts/                 # Output PNG charts
│   └── *.csv                   # Cached price data and results
└── README.md
```

---

## Requirements

- Python 3.6+
- `pandas`
- `numpy`
- `matplotlib`
- `yahoofinance` (or compatible Yahoo Finance library)
- `tensorflow` >= 1.5 (tested up to 2.x) *(optional — only needed for SeriesPrediction)*
- `keras` *(bundled with TensorFlow 2.x)*
- `pyodbc` + Microsoft ODBC Driver for SQL Server *(optional — only needed for the database backend)*

To skip loading the ML modules entirely, set `enableTensorFlow=False` in `PriceTradeAnalyzer.py`.

---

## Getting Started

```python
from _classes.PriceTradeAnalyzer import PricingData, TradingModel, Portfolio

# Download and analyze a stock
prices = PricingData('AAPL')
prices.CalculateStats()
prices.Graph(90)   # chart the last 90 days

# Run a backtest
model = TradingModel(ticker='AAPL', startDate='2015-01-01', endDate='2023-12-31')
# ... define your buy/sell logic, then:
model.RunModel()
model.GraphResults()
```

See `EvaluatePrices.py`, `EvaluateTradeModels.py`, and `TrainPrices.py` for complete, runnable examples.

---

## Related Projects

- [**Price-Momentum-Trader**](https://github.com/TimRivoli/Price-Momentum-Trader) — backtests a momentum strategy of holding the top 9 performing S&P 500 stocks, rotated monthly, over 35 years.
- [**RobotTrader**](https://github.com/TimRivoli/RobotTrader) — treats trading as a classification problem using a supervised RNN, built on top of this library.

---

## Acknowledgements

Thanks to the following for their TensorFlow/deep learning tutorials and inspiration:
- [Siraj Raval](https://www.linkedin.com/in/sirajraval/)
- [Magnus Erik Hvass Pedersen](https://github.com/Hvass-Labs/TensorFlow-Tutorials)
- [Nicholas T. Smith](https://nicholastsmith.wordpress.com/)
- [Luka Anicin](https://github.com/lucko515/tesla-stocks-prediction)

---

## Disclaimer

This project is for educational and research purposes only. Nothing here constitutes financial advice. Past backtested performance does not guarantee future results.
