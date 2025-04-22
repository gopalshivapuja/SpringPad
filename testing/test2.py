import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Download data
data = yf.download("LT.NS", start="2024-01-01", end="2025-03-31")

# Print the first few rows
#print(data.head())

# Plot closing prices
#data['Close'].plot(title='LT.NS Stock Price')
#plt.xlabel('Date')
#plt.ylabel('Price')
#plt.show()

data['daily_return'] = data['Close' ] - data['Close'].shift(1)
print(data.head())