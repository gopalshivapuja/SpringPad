import warnings
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---

# Suppress general warnings
warnings.filterwarnings('ignore')

# Ticker symbol for the asset to backtest (e.g., Nifty ETF or Index)
ticker = "NIFTYBEES.NS"  # Example: NIFTY BeES ETF
# ticker = "^NSEI"     # Alternative: Nifty 50 Index

# Date range for historical data
start_date = "2020-01-01" # Extended start date for better RSI calculation
end_date = "2024-04-01"

# RSI parameters
rsi_period = 14       # Standard RSI lookback period
rsi_oversold = 30     # RSI level below which asset is considered oversold (Buy signal)
rsi_overbought = 70    # RSI level above which asset is considered overbought (Sell signal)

# --- Main Script Logic ---

print(f"--- RSI Backtest for {ticker} ---")
print(f"Period: {start_date} to {end_date} | RSI Period: {rsi_period}")

# 1. Download Historical Data
print(f"Downloading data for {ticker}...")
try:
    # yfinance defaults to auto_adjust=True
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        print("Error: No data downloaded. Check ticker symbol and date range.")
        exit()
    
    # Ensure columns are using a standard index if only one ticker was downloaded
    # Or explicitly handle MultiIndex if that's guaranteed/preferred
    # For simplicity here, if only one ticker, we flatten the index
    if isinstance(data.columns, pd.MultiIndex) and len(data.columns.levels[1]) == 1:
        data.columns = data.columns.droplevel(1) # Drop the ticker level
        print("Note: Flattened MultiIndex columns for single ticker.")

except Exception as e:
    print(f"Error during data download: {e}")
    exit()

# 2. Calculate RSI (using standard Wilder's Smoothing Method)
print("Calculating RSI...")
def compute_rsi_standard(series, period=14):
    """Calculates RSI using Wilder's smoothing method (equivalent to EMA with alpha=1/period)."""
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0.0) # Ensure gains are float
    loss = -delta.where(delta < 0, 0.0) # Ensure losses are float and positive

    # Use Exponential Weighted Moving Average (EWMA) for both gain and loss
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    # Calculate Relative Strength (RS)
    rs = avg_gain / avg_loss.replace(0, 1e-9) # Avoid division by zero

    # Calculate RSI
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

try:
    # Ensure we are passing the 'Close' Series
    close_series = data['Close'] 
    data['RSI'] = compute_rsi_standard(close_series, period=rsi_period)
except KeyError:
    print("Error: 'Close' column not found. Check downloaded data structure.")
    print("Columns found:", data.columns)
    exit()
except Exception as e:
    print(f"Error during RSI calculation: {e}")
    exit()
    
# Remove initial rows with NaN RSI values (due to lookback period)
data = data.dropna(subset=['RSI'])
if data.empty:
    print("Error: Not enough data to calculate RSI after removing NaNs.")
    exit()

print("Generating Buy/Sell signals...")
# 3. Generate Buy/Sell Signals based on RSI thresholds
data['Buy_Signal'] = data['RSI'] < rsi_oversold
data['Sell_Signal'] = data['RSI'] > rsi_overbought

print("Backtesting strategy...")
# 4. Backtest the Strategy
position = 0            # Current position status: 0 = Flat, 1 = Long
buy_price = 0           # Price at which the current position was entered
trades = []             # List to store details of completed trades
entry_date = None       # To store the entry date of the current trade

# Iterate through the DataFrame
for i in range(len(data) - 1):
    # Check for Buy Entry Signal and if currently flat
    if data['Buy_Signal'].iloc[i] and position == 0:
        buy_price = data['Open'].iloc[i+1]
        position = 1
        entry_date = data.index[i+1]
        print(f"  BUY Signal on {data.index[i].date()} (RSI={data['RSI'].iloc[i]:.2f}). Enter LONG at {buy_price:.2f} on {entry_date.date()}")

    # Check for Sell Exit Signal and if currently Long
    elif data['Sell_Signal'].iloc[i] and position == 1:
        sell_price = data['Open'].iloc[i+1]
        if buy_price > 0 and entry_date is not None:
            profit = sell_price - buy_price
            profit_pct = (profit / buy_price) * 100 if buy_price != 0 else 0
            exit_date = data.index[i+1]
            trades.append({
                'Entry Date': entry_date,
                'Buy Price': buy_price,
                'Exit Date': exit_date,
                'Sell Price': sell_price,
                'Profit': profit,
                'Profit Pct': profit_pct
            })
            print(f"  SELL Signal on {data.index[i].date()} (RSI={data['RSI'].iloc[i]:.2f}). Exit LONG at {sell_price:.2f} on {exit_date.date()} | Profit: {profit:.2f} ({profit_pct:.2f}%)")
            print("  " + "-"*60)
            position = 0
            buy_price = 0
            entry_date = None
        else:
             print(f"  SELL Signal on {data.index[i].date()} but cannot record trade (invalid buy_price or entry_date).")
             position = 0 
             buy_price = 0
             entry_date = None

# 5. Analyze Results
print("\n--- Backtest Summary ---")
if trades:
    trades_df = pd.DataFrame(trades)
    total_profit = trades_df['Profit'].sum()
    num_trades = len(trades_df)
    win_rate = (trades_df['Profit'] > 0).mean() * 100 if num_trades > 0 else 0
    avg_profit = total_profit / num_trades if num_trades > 0 else 0

    print(f"Total Trades Executed: {num_trades}")
    print(f"Overall Profit/Loss: {total_profit:.2f}")
    print(f"Average Profit/Loss per Trade: {avg_profit:.2f}")
    print(f"Win Rate: {win_rate:.2f}%")
else:
    print("No completed trades were executed during this period.")
print("-" * 50)

# 6. Plotting Results
print("Plotting results...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Plot 1: Price with Buy/Sell Markers
ax1.plot(data.index, data['Close'], label='Close Price', color='blue', linewidth=1.0)
ax1.set_title(f'{ticker} Price with RSI Strategy Signals')
ax1.set_ylabel('Price')
ax1.grid(True, alpha=0.3)

if trades:
    trades_df['Entry Date'] = pd.to_datetime(trades_df['Entry Date'])
    trades_df['Exit Date'] = pd.to_datetime(trades_df['Exit Date'])
    ax1.scatter(trades_df['Entry Date'], trades_df['Buy Price'], marker='^', color='lime', s=100, label='Buy Entry', zorder=5)
    ax1.scatter(trades_df['Exit Date'], trades_df['Sell Price'], marker='v', color='red', s=100, label='Sell Exit', zorder=5)

ax1.legend()

# Plot 2: RSI with Overbought/Oversold Levels
ax2.plot(data.index, data['RSI'], label='RSI', color='orange', linewidth=1.0)
ax2.axhline(rsi_overbought, color='red', linestyle='--', linewidth=0.8, label=f'Overbought ({rsi_overbought})')
ax2.axhline(rsi_oversold, color='green', linestyle='--', linewidth=0.8, label=f'Oversold ({rsi_oversold})')
ax2.set_title('Relative Strength Index (RSI)')
ax2.set_ylabel('RSI Value')
ax2.set_xlabel('Date')
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()

print("--- Analysis Complete ---")
