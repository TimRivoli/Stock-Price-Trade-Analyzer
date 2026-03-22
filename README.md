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

## Project Structure

```
Stock-Price-Trade-Analyzer/
├── _classes/
│   ├── PriceTradeAnalyzer.py   # PricingData, Portfolio, TradingModel
│   └── SeriesPrediction.py     # StockPredictionNN
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
