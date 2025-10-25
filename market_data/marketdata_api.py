import pandas as pd
import yfinance as yf

# Load your S&P 500 table
stocks = pd.read_csv(r'market_data\sp500_table.csv')
tickers = stocks['Symbol'].tolist()
tickers_str = ' '.join(tickers)
data = yf.download(tickers=tickers_str, period='1d', interval='1m', group_by='ticker', threads=True)
last_prices = {ticker: data[ticker]['Close'].iloc[-1] for ticker in tickers if ticker in data.columns.levels[0]}
print(last_prices)