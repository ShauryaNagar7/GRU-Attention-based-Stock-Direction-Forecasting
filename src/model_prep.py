# model_prep.py - PHASE A: FEATURE ENGINEERING (Manual Calculation)
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import time

# --- Configuration ---
RAW_DATA_PATH = 'data/raw'
PROCESSED_DATA_PATH = 'data/processed'
METADATA_PATH = 'data/metadata'

# Periods for Feature Calculation
WINDOW_20D = 20
WINDOW_14D = 14
WINDOW_50D = 50

# The 8 Features and Target Column
FINAL_FEATURE_COLS = [
    'F1_Daily_Return', 'F2_Volume_Momentum', 'F3_RSI', 'F4_MACD_Diff', 
    'F5_BB_Position', 'F6_Price_VWAP_Ratio', 'F7_Volatility_Regime', 
    'F9_Support_Resistance', 'Target_Class'
]

# --- Helper Functions for Technical Indicators ---

def calculate_rsi(series: pd.Series, window: int = WINDOW_14D) -> pd.Series:
    """Calculates the Relative Strength Index (RSI)."""
    delta = series.diff()
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate exponential moving average of gains and losses (Wilder's smoothing)
    avg_gain = gain.ewm(span=window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(span=window, adjust=False, min_periods=window).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculates MACD, Signal Line, and MACD Histogram."""
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_diff = macd_line - signal_line
    
    return pd.DataFrame({
        'MACD_Line': macd_line,
        'MACD_Signal': signal_line,
        'MACD_Diff': macd_diff
    })

def calculate_vwap(df: pd.DataFrame, window: int = WINDOW_20D) -> pd.Series:
    """Calculates the Volume Weighted Average Price (VWAP) over a rolling window."""
    # Typical price: (High + Low + Close) / 3
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    # Cumulative calculation over the window
    tpv = typical_price * df['Volume']
    
    # Calculate Rolling VWAP
    vwap_rolling = tpv.rolling(window=window).sum() / df['Volume'].rolling(window=window).sum()
    return vwap_rolling

# --- Core Feature and Target Engineering ---

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates all 8 defined features."""
    
    # --- Price and Volume Features (F1, F2) ---
    # F1_Daily_Return
    df['F1_Daily_Return'] = df['Close'].pct_change()
    
    # F2_Volume_Momentum (Current Volume / 20-day Average Volume)
    df['F2_Volume_Momentum'] = df['Volume'] / df['Volume'].rolling(window=WINDOW_20D).mean()

    # --- Trend and Momentum Features (F3, F4) ---
    # F3_RSI (14-day)
    df['F3_RSI'] = calculate_rsi(df['Close'], window=WINDOW_14D)
    
    # F4_MACD (We use the MACD Histogram, which is MACD Line - Signal Line, as the feature)
    macd_df = calculate_macd(df['Close'])
    df['F4_MACD_Diff'] = macd_df['MACD_Diff']

    # --- Volatility and Position Features (F5, F7) ---
    # F5_BB_Position (normalized by band width)
    # 1. Calculate Bollinger Bands (BB)
    df['MA_20'] = df['Close'].rolling(window=WINDOW_20D).mean()
    df['STD_20'] = df['Close'].rolling(window=WINDOW_20D).std()
    df['Upper_Band'] = df['MA_20'] + (df['STD_20'] * 2)
    df['Lower_Band'] = df['MA_20'] - (df['STD_20'] * 2)
    # 2. Calculate %B (normalized position): (Close - Lower Band) / (Upper Band - Lower Band)
    with np.errstate(divide='ignore', invalid='ignore'):
        df['F5_BB_Position'] = (df['Close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])
    
    # F7_Volatility_Regime (20-day rolling Standard Deviation of Daily Returns)
    df['F7_Volatility_Regime'] = df['F1_Daily_Return'].rolling(window=WINDOW_20D).std()

    # --- Custom Price/Volume Features (F6, F9) ---
    # F6_Price_VWAP_Ratio (Close Price / 20-day Rolling VWAP)
    df['VWAP_20'] = calculate_vwap(df, window=WINDOW_20D)
    df['F6_Price_VWAP_Ratio'] = df['Close'] / df['VWAP_20']

    # F9_Support_Resistance (Position within 50-day High-Low Range)
    df['Range_50_High'] = df['High'].rolling(window=WINDOW_50D).max()
    df['Range_50_Low'] = df['Low'].rolling(window=WINDOW_50D).min()
    # Normalized position: (Close - 50-day Low) / (50-day High - 50-day Low)
    with np.errstate(divide='ignore', invalid='ignore'):
        df['F9_Support_Resistance'] = (df['Close'] - df['Range_50_Low']) / \
                                      (df['Range_50_High'] - df['Range_50_Low'])

    # Clean up auxiliary columns
    aux_cols_to_drop = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume'] and not col.startswith('F')]
    df.drop(columns=aux_cols_to_drop, inplace=True, errors='ignore')

    return df

def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Creates the 3-Class Classification target variable (Up, Down, Flat)."""
    
    # Calculate the next day's Daily Return (Shifted back by 1)
    df['Next_Day_Return'] = df['Close'].pct_change().shift(-1) * 100
    
    UP_THRESHOLD = 0.5    # >= +0.5%
    DOWN_THRESHOLD = -0.5 # <= -0.5%
    
    def classify(ret):
        if ret >= UP_THRESHOLD:
            return 2 # Up
        elif ret <= DOWN_THRESHOLD:
            return 0 # Down
        else:
            return 1 # Flat
    
    df['Target_Class'] = df['Next_Day_Return'].apply(classify)
    df.drop(columns=['Next_Day_Return'], inplace=True)
    return df

def process_single_stock(file_path: str, ticker: str) -> pd.DataFrame | None:
    """Loads, engineers features, creates target, cleans data, and returns the result."""
    try:
        # Load Raw Data
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        df.index.name = 'Date'
        
        # 1. Feature Engineering (Requires OHLCV data)
        df = engineer_features(df.copy())
        
        # 2. Target Creation (Requires Close price)
        df = create_target_variable(df.copy())
        
        # 3. Final Selection and Cleaning
        # Ensure only required features, OHLCV (for reference), and Target are kept
        df['Ticker'] = ticker 
        
        # Select features and target, then drop all rows with NaN (due to rolling windows)
        required_cols = FINAL_FEATURE_COLS + ['Ticker']
        
        # Add a check for data consistency
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"Error: Missing columns after processing {ticker}: {missing}")
            return None

        df_processed = df[required_cols].dropna()
        
        print(f"Features created for {ticker}. Rows: {len(df_processed)}")
        return df_processed
    
    except Exception as e:
        print(f"ERROR processing {ticker}: {e}")
        return None

def run_feature_engineering_pipeline():
    """Main function to orchestrate the feature engineering process."""
    print("--- Starting Full Feature Engineering (Manual Phase A) ---")
    
    # Ensure directories exist
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(METADATA_PATH, exist_ok=True)
    
    all_processed_dfs = []
    
    # Iterate through all raw data files
    raw_files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith('.csv')]
    
    for i, file_name in enumerate(raw_files):
        # CHANGE THIS LINE: Simplify ticker extraction for original file names (e.g., 'MSFT.csv' -> 'MSFT')
        # This handles both US (MSFT.csv) and India (INFY.NS.csv) formats
        if file_name.endswith('.NS.csv'):
            # Handles Indian tickers
            ticker_dot_notation = file_name.replace('.csv', '')
        else:
            # Handles US tickers
            ticker_dot_notation = file_name.replace('.csv', '')
            
        file_path = os.path.join(RAW_DATA_PATH, file_name)
        
        processed_df = process_single_stock(file_path, ticker_dot_notation)
        
        if processed_df is not None and not processed_df.empty:
            all_processed_dfs.append(processed_df)

        time.sleep(0.1) # Small delay to prevent resource contention

    if not all_processed_dfs:
        print("❌ No processed data frames were generated. Check raw data folder.")
        return

    # 4. Combine all processed data into one DataFrame
    combined_df = pd.concat(all_processed_dfs)
    
    # 5. Save the final combined feature set (un-normalized)
    output_file = os.path.join(PROCESSED_DATA_PATH, 'IT_stocks_features_combined.csv')
    combined_df.to_csv(output_file)
    print(f"\n✅ Combined Features DataFrame saved to: {output_file}")
    print(f"Total rows for pre-training: {len(combined_df)}")
    
    print("\n--- Feature Engineering Complete ---")
    print("Next: Sequence Creation, Normalization, and Saving to data/sequences.")


if __name__ == '__main__':
    run_feature_engineering_pipeline()