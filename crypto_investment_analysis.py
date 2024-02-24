from bs4 import BeautifulSoup as bs
from scipy import optimize
import pandas as pd
import numpy as np
import requests
import json
import time
import re
debug = 0

'''
Downloads historical data from an exchange based on a frequency and, optionally, given list of the cryptocurrencies to use
'''
def get_exchange_index_json(exchange="gemini", frequency="daily", numds=4, symbols=False, output=""):
  currentSet = 0
  container = []
  fileurls = []

  # encodes the frequency to the necessary format for the data provider
  match frequency:
    case "daily":
      frequency = "d"
    case "hourly":
      frequency = "1h"
    case _:
      raise ValueError("Frequency must either be daily or hourly")
  regex = re.compile(r"{}.csv$".format(re.escape(frequency)))

  # Accesses the api for a given exchange to return data in JSON format
  url = f"https://api.cryptodatadownload.com/v1/data/ohlc/{exchange}/available?format=json"
  req = requests.get(url)
  if (req.status_code != 200):
    print(req.status_code, req.headers)
    req.raise_for_status()

  # Grabs the links to the data for the specified coins
  for element in json.loads(req.text).get("data"):
    if (regex.search(element.get("file")) and currentSet < numds):
      fileloc = element.get("file")
      if symbols:
        numds = len(symbols)
        for symbol in symbols:
          match = re.compile(r"^.*_{}USD.*$".format(re.escape(symbol))).search(fileloc)
          if match:
            fileurls.append(match.group(0))
      else:
        fileurls.append(fileloc)
        currentSet += 1

  # Downloads the data for the specified coins and appends them to a dataframe
  for file in fileurls:
    response = requests.get(file)
    if (response.status_code == 200):
      df = pd.read_csv(file, header=1, parse_dates=['Date'])
      df.set_index('Date', inplace=True)

      # Convert the 'Unix' column to datetime format and filter rows
      # with a timestamp  on or after '2020-09-01'
      df = df[pd.to_datetime(df['Unix'], unit='ms') >= '2010-01-01']  # Adjusted for testing purposes due to issues with SOL (NaN)
      if (debug >= 1): print(f"[200]: {file}{df.head(), df.columns}\n")

      container.append(df)
      time.sleep(0.0125)
    else:
      response.raise_for_status()

  dataset = pd.concat(container, axis=0, join="outer", ignore_index=False)
  if (debug >= 1): print(output)
  return dataset, symbols

# Saves historical crypto data in the dataframe
returns, symbols = get_exchange_index_json(exchange="binance", frequency="daily", symbols=["BTC", "ETH", "XRP"])
yearly_returns = {}
mean_returns = {}
std_deviations = {}
returns_df = pd.DataFrame()

# Calculate yearly returns, mean, and std for each coin and store in DataFrame
for symbol in symbols:
    column_name = f'{symbol} Yearly Returns'
    yearly_returns[symbol] = returns[returns['Symbol'] == f'{symbol}USDC']['Close'].resample('Y').last().pct_change()
    returns_df[column_name] = yearly_returns[symbol]
    mean_returns[symbol] = yearly_returns[symbol].mean()
    std_deviations[symbol] = yearly_returns[symbol].std()

avg_yearly_returns = {symbol: yearly_returns[symbol].mean() for symbol in symbols}

# Calculate the correlation matrix
correl_matrix = pd.DataFrame({symbol: yearly_returns[symbol] for symbol in symbols}).corr()

for symbol in symbols:
    print(f'{symbol} Yearly Returns:')
    print(yearly_returns[symbol])

print(f"\nMean yearly returns: {mean_returns}\nAverage yearly returns: {avg_yearly_returns}\nVolatilities: {std_deviations}\n\nCorrelation matrix of yearly returns:\n{correl_matrix}")

# assets' means
mu = np.array([[mean_returns['BTC'], mean_returns['ETH'], mean_returns['XRP']]]).T

# assets' correlations
corrMatrix = correl_matrix

# assets' volatilities
sigmaVec = np.array([[std_deviations['BTC'], std_deviations['ETH'], std_deviations['XRP']]]).T

# build the covariance matrix
Sigma = np.multiply(corrMatrix, np.dot(sigmaVec, sigmaVec.T))

numberOfAssets = sigmaVec.shape[0]
e = np.ones((numberOfAssets, 1))

# risk aversion parameter
gamma = 5

#Parameters for the constraint (here only a budget constraint):
Aeq = e.T
beq = 1

# function to minimize
objective_1 = lambda w: 0.5 * gamma * np.dot(np.dot(w.T, Sigma), w) - np.dot(w.T, mu)

# equality constraint (e'w = 1)
eq_constraint_1 = lambda w: np.dot(Aeq, w) - beq
cons_1 = ({'type': 'eq', 'fun': eq_constraint_1})

# starting point for the solver (start from EW portfolio)
w0 = np.ones((numberOfAssets, )) / numberOfAssets

# run the optimization
result = optimize.minimize(objective_1, w0, method='SLSQP', tol=1e-8, constraints=cons_1)

# print results
w_1 = result.x.reshape((numberOfAssets, 1))

exp_ret_1 = np.dot(w_1.T, mu)
vol_1 = np.dot(np.dot(w_1.T, Sigma), w_1) ** 0.5
print(f"\nExpected return: {exp_ret_1}\nVolatility: {vol_1} with optimal weights:")
for i, w in zip(symbols, w_1): print(f"{i}: {w_1}")