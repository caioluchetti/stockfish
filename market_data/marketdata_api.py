import os
import pandas as pd
import yfinance as yf

csv_path = os.path.join(os.path.dirname(__file__), "sp500_table.csv")
stocks = pd.read_csv(csv_path)
tickers = stocks["Symbol"].tolist()
tickers_str = " ".join(tickers)
data = yf.download(tickers=tickers_str, period="1d", interval="1m", group_by="ticker", threads=True)
last_prices = {ticker: data[ticker]["Close"].iloc[-1] for ticker in tickers if ticker in data.columns.levels[0]}
print(last_prices)