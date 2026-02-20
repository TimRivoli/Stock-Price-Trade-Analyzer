

This is a Python 3 project for analyzing stock prices and methods of stock trading. It uses native Python tools and Google TensorFlow machine learning.

Module: PriceTradeAnalyzer
Class PricingData
Given a stock ticker, this will go out and download historical price data using the yahoofinance library.  Downloaded data is cached to .csv file and stored in a Pandas dataframe which is easy to manipulate.  PricingData helper functions allow you to TrimToDateRange, ConvertToPercentages, NormalizePrices, CalculateStats (EMA, channels, momentum, etc), perform time frame graphing (using matplotlib), and a few other bells and whistles.  There are also several prediction models for predicting future prices. 

EvaluatePrices.py shows how to use the PricingData class to PlotAnnualPerformance of a stock, DownloadAndGraphStocks for a list of stocks such as the S&P 500, CalculatePriceCorrelation of a list of stocks, OpportunityFinder to identify recent drops or over-bought/over-sold conditions from a list of stocks.

Classes Portfolio and TradingModel are used to test emulations of trading strategies.  EvaluateTradeModels.py shows examples of how to the TradingModel class to create and test trading strategies.  The models will use actual historical prices from PricingData. You specify the stocks you want to work with, the time frame you want to test against, and the logic you want to use for buying and selling.  Example strategies are included for BuyAndHold, Seasonal investing, and two different Trending approaches.  The resulting daily value and trade history are dataframes which are graphed and saved to .csv and .png files so you can view the performance details later.  ExtendedDurationTest allows you to test the performance of any model over various time frames and durations.  CompareModels allows you to compare the performance of any two models over a given time period.  It would be great to create a re-enforcement learning module using Deep Q or Policy Gradient.  I'm also thinking of making a stock trading game and using the code to run it against historical time periods.  It could be a good educational tool.  I know it has been for me!

Class ForcastModel has been added to forecast the effect of a series of potential actions on a TradingModel.  I'm using this to create a "best actions" sequence for supervised machine learning in another project.  Given a market state and a sequence of actions (or every possible action) which one produces the best result after X days.  This can then be used to train a robotic trainer with supervised learning.

StateSurprisePredictionNN is a machine-learning model designed to predict surprise deviations from trend rather than predicting raw price. It works by first generating a baseline forecast using a simple trend continuation model (typically a linear extrapolation of recent price slope over a chosen horizon). 
The model then computes the residual: Residual = FuturePrice − BaselineForecast
This residual represents the unexpected component of price movement — effectively the “surprise” change from what the trend would predict. The network is trained on a set of engineered state features (momentum, volatility, deviation from mean/trend, compression, etc.) and learns a nonlinear mapping from the current market state to the expected residual outcome. 
The output is split into:
	Direction likelihood (probability of positive vs negative surprise)
	Magnitude estimate (expected size of the surprise move)
This allows the model to identify when price is likely to break from its expected trend path and estimate how large that divergence may be.

TrainPrices.py shows samples of using the StateSurprisePredictionNN class to train and test PricingData using machine learning techniques.  Results are then statistically analyzed for significance.

I've added support for using SQL as a back-end instead of .csv files.  There is a PTAGenerate.sql file to help you get started.  I use sqlalchemy which supports a lot of ODBC data sources.  I use Microsoft "ODBC Driver 18 for SQL Server". If you populate the Database information in the config.ini then the code will attempt to use the SQL database.  Create a ConnectionString setting in the .ini or if you are using MS SQL with ODBC 18 then DatabaseServer, DatabaseName, and optinal DatabaseUsername and DatabasePasswor.

Special thanks to these people for helping me understand deep neural network machine learning:
Siraj Raval: https://www.linkedin.com/in/sirajraval/
Magnus Erik Hvass Pedersen: https://github.com/Hvass-Labs/TensorFlow-Tutorials
Nicholas T. Smith https://nicholastsmith.wordpress.com/ 
Luka Anicin: https://github.com/lucko515/tesla-stocks-prediction
And of course, special thanks to ChatGPT and Gemini for helping my modernize my code.

I've tested this on both Windows Python 3.10-3.13, TensorFlow 2.13.1, Keras 2.13.1.
Requirements: happily all native Python and no C++ compilers

Windows PIP install requirements with:
pip install tqdm, pandas numpy matplotlib scipy requests pyodbc yfinance sqlalchemy curl_cffi
pip install tensorflow keras

Tensorflow and Keras are picky with their versions.  I'm using these:
tensorflow: 2.13.1
keras: 2.13.1
typing_extensions: 4.15.0
NumPy: 1.26.4
pip install tensorflow==2.13.1 keras==2.13.1 typing-extensions==4.15.0 numpy==1.26.4

Have fun and keep programming!
-Tim
