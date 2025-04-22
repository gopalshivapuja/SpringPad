# pair trading.py

# --- Standard Libraries ---
import warnings
import yfinance as yf # For downloading stock data
import pandas as pd     # For data manipulation (DataFrames)
import numpy as np      # For numerical operations (though less used directly here)
import statsmodels.api as sm # For statistical models, specifically OLS regression
import matplotlib.pyplot as plt # For plotting results

# --- Configuration ---

# Suppress general warnings (e.g., from underlying libraries)
# Note: Specific warnings might still appear if emitted before this line runs effectively.
warnings.filterwarnings('ignore')

# Define the pair of tickers to analyze
# IMPORTANT: The order matters for regression (ticker[1] is regressed on ticker[0])
tickers = ['AAPL', 'MSFT'] # Example: Apple vs Microsoft (Predict MSFT based on AAPL)

# Define the historical date range for analysis
start_date = '2020-01-01'
end_date = '2023-01-01'

# Define the rolling window size for Z-score calculation
rolling_window = 30 # Number of trading days

# Define Z-score thresholds for entry and exit signals
entry_threshold = 1.5 # Enter trade if Z-score exceeds this value (positive or negative)
exit_threshold = 0.5  # Exit trade if Z-score comes back within this range around zero

# --- Main Script Logic ---

print(f"--- Pair Trading Analysis: {tickers[0]} vs {tickers[1]} ---")
print(f"Period: {start_date} to {end_date} | Rolling Window: {rolling_window} days")

# 1. Download Historical Data
print(f"Downloading data for {tickers}...")
try:
    # yfinance defaults to auto_adjust=True, meaning 'Close' is the adjusted price
    all_data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    # Check if download was successful and returned data
    if all_data.empty:
        print("Error: No data downloaded. Check ticker symbols and date range.")
        exit()

    # Select only the 'Close' prices (adjusted)
    data = all_data['Close']

except Exception as e:
    print(f"Error during data download: {e}")
    exit()

# Remove any rows with missing price data for either stock
data = data.dropna()

# Check if enough data remains for calculations (at least rolling window + 1)
# We need enough data to calculate rolling mean/std and perform regression
if data.shape[0] < rolling_window + 1:
    print(f"Error: Not enough data points ({data.shape[0]}) after removing NaNs for rolling window ({rolling_window}).")
    exit()

print("Calculating hedge ratio via OLS regression...")
# 2. Calculate Hedge Ratio using Ordinary Least Squares (OLS)
# We regress the price of tickers[1] against the price of tickers[0]
# to find how many units of tickers[0] are needed to hedge one unit of tickers[1].
stock1_with_const = sm.add_constant(data[tickers[0]]) # Independent variable (X) + intercept column
stock2 = data[tickers[1]] # Dependent variable (Y)

# Fit the OLS model: Y ~ const + hedge_ratio * X
try:
    results = sm.OLS(stock2, stock1_with_const).fit()
    # The hedge ratio is the coefficient of the independent variable (tickers[0])
    hedge_ratio = results.params[tickers[0]]
except Exception as e:
    print(f"Error during OLS regression: {e}")
    exit()

print(f"Hedge Ratio ({tickers[1]} / {tickers[0]}): {hedge_ratio:.4f}")

# 3. Calculate the Spread
# Spread = Price(tickers[1]) - hedge_ratio * Price(tickers[0])
# This represents the deviation from the expected relationship based on the hedge ratio.
spread = data[tickers[1]] - hedge_ratio * data[tickers[0]]
data['Spread'] = spread # Store the spread in the DataFrame

print("Calculating rolling Z-Score of the spread...")
# 4. Compute Rolling Z-Score of the Spread
# The Z-score measures how many standard deviations the current spread is away
# from its rolling mean, helping identify potential mean-reversion opportunities.
mean = spread.rolling(window=rolling_window).mean()
std = spread.rolling(window=rolling_window).std()
data['Z-Score'] = (spread - mean) / std

# Remove NaNs created by the rolling window calculation
data = data.dropna(subset=['Z-Score'])
if data.empty:
    print("Error: No data left after Z-score calculation (check rolling window size vs data length).")
    exit()

print("Generating trading signals based on Z-Score...")
# 5. Define Trading Signals based on Z-Score Thresholds
# Long the Spread (Buy tickers[1], Sell tickers[0]) if Z-Score is unusually low
data['Long_Signal'] = data['Z-Score'] < -entry_threshold
# Short the Spread (Sell tickers[1], Buy tickers[0]) if Z-Score is unusually high
data['Short_Signal'] = data['Z-Score'] > entry_threshold
# Exit any open position if Z-Score reverts towards the mean
data['Exit_Signal'] = abs(data['Z-Score']) < exit_threshold

print("Simulating trading positions...")
# 6. Simulate Positions (1 = Long Spread, -1 = Short Spread, 0 = Flat)
# Iterate through the data day by day and determine the position based on signals.
position = 0 # Start with no position (flat)
positions = [] # List to store the position for each day

for i in range(len(data)):
    # Check entry conditions ONLY if currently flat (position == 0)
    if position == 0:
        if data['Long_Signal'].iloc[i]:
            position = 1 # Enter Long spread position
        elif data['Short_Signal'].iloc[i]:
            position = -1 # Enter Short spread position
    # Check exit condition ONLY if currently holding a position (position != 0)
    elif position != 0 and data['Exit_Signal'].iloc[i]:
        position = 0 # Exit position, go flat

    positions.append(position)

# Add the daily position status to the DataFrame
data['Position'] = positions

print("Calculating Profit and Loss (PnL)...")
# 7. Calculate Strategy Returns (PnL)
# Calculate the change in the spread's value from one day to the next
data['Spread_Change'] = data['Spread'].diff()

# The daily PnL is the position held *yesterday* (shift(1)) multiplied by *today's* change in spread value.
# This assumes trades are entered/exited based on signals generated from previous day's close,
# and the profit/loss is realized based on the spread change during the holding day.
data['PnL'] = data['Position'].shift(1) * data['Spread_Change']

# Remove initial NaN generated by shift(1) and diff()
data = data.dropna(subset=['PnL'])
if data.empty:
    print("Error: No data left after PnL calculation.")
    exit()

# 8. Calculate Cumulative PnL
data['Cumulative_PnL'] = data['PnL'].cumsum()

# --- Display Final PnL --- 
final_cumulative_pnl = data['Cumulative_PnL'].iloc[-1] if not data.empty else 0
print(f"\n--- Final Result ---")
print(f"Final Cumulative PnL: {final_cumulative_pnl:.2f}")
print("-" * 50)

# --- Plotting Results ---
print("Plotting results...")

# Plot 1: Spread and Z-Score with Thresholds
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True) # Two subplots sharing the x-axis (Date)

# Top subplot: Spread
ax1.plot(data.index, data['Spread'], label='Spread', linewidth=1.0)
ax1.set_title(f'Calculated Spread ({tickers[1]} - {hedge_ratio:.2f} * {tickers[0]})')
ax1.axhline(0, color='grey', linestyle='--', linewidth=0.8, label='Mean (Theoretical)')
ax1.set_ylabel('Spread Value')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Bottom subplot: Z-Score
ax2.plot(data.index, data['Z-Score'], label='Z-Score', linewidth=1.0)
ax2.set_title(f'Spread Z-Score (Rolling Window: {rolling_window} days)')
# Add threshold lines for entry and exit
ax2.axhline(entry_threshold, color='r', linestyle='--', linewidth=0.8, label=f'Entry Threshold (+{entry_threshold})')
ax2.axhline(-entry_threshold, color='r', linestyle='--', linewidth=0.8, label=f'Entry Threshold (-{entry_threshold})')
ax2.axhline(exit_threshold, color='g', linestyle='--', linewidth=0.8, label=f'Exit Threshold (+/-{exit_threshold})')
ax2.axhline(-exit_threshold, color='g', linestyle='--', linewidth=0.8)
ax2.axhline(0, color='grey', linestyle='--', linewidth=0.8, label='Mean (0 Z-Score)') # Line at Z-score = 0
ax2.set_ylabel('Z-Score Value')
ax2.set_xlabel('Date')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.suptitle(f'Pair Trading Analysis: {tickers[0]} vs {tickers[1]}', y=1.02) # Overall title
plt.tight_layout() # Adjust layout to prevent labels overlapping
plt.show()

# Plot 2: Cumulative PnL
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Cumulative_PnL'], label='Cumulative PnL', color='blue')
plt.title(f'Strategy Cumulative PnL: {tickers[0]} vs {tickers[1]}')
plt.xlabel('Date')
plt.ylabel('Cumulative Profit/Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("--- Analysis Complete ---")
