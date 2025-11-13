# data_prep.py - PHASE A: DATA COLLECTION SCRIPT (Updated)
import yfinance as yf
import pandas as pd
import os
import time

# --- Configuration ---
TICKERS = [
    'MSFT', 'GOOGL', 'NVDA', 'AMD', 'META', 'AAPL', 'AMZN', 'ACN', 'IBM',
    'TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'LTTS.NS'
]
START_DATE = '2017-01-01'
END_DATE = '2025-01-01'
RAW_DATA_PATH = 'data/raw'
REQUIRED_COLS = ['Open', 'High', 'Low', 'Close', 'Volume'] # Essential columns

def fetch_and_save_data():
    """
    Fetches historical stock data with robust parameters, processes the DataFrame 
    for consistency, and saves individual CSV files.
    """
    print("--- Starting Robust Data Collection (Phase A) ---")
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    print(f"Target directory '{RAW_DATA_PATH}' ensured.")

    # Using yf.download with threads for concurrent download
    data = yf.download(
        TICKERS, 
        start=START_DATE, 
        end=END_DATE,
        # --- CRITICAL Parameters Implemented ---
        auto_adjust=False, # Keep original OHLC prices
        actions=False,     # Exclude dividends and splits
        threads=True,      # Use parallel threads for speed
        timeout=30         # Set a timeout for the request
    )
    
    # Check if download returned a MultiIndex DataFrame
    if isinstance(data.columns, pd.MultiIndex):
        print("\nMultiIndex DataFrame detected. Starting column cleaning...")
        
        # --- MultiIndex Handling and Column Validation ---
        for ticker in TICKERS:
            try:
                # 1. Select the columns for the current ticker
                df = data.loc[:, (slice(None), ticker)]
                df.columns = df.columns.droplevel(1) # Flatten MultiIndex ('Close', 'MSFT') -> 'Close'
                
                # 2. Basic Column Validation & NaN Cleaning
                if not all(col in df.columns for col in REQUIRED_COLS):
                    print(f"WARNING: Skipping {ticker}. Missing required columns.")
                    continue

                # 3. Drop rows with any NaN values (often due to missing volume)
                df.dropna(inplace=True) 

                # 4. Save the cleaned, raw data
                file_path = os.path.join(RAW_DATA_PATH, f'{ticker}.csv')
                df.to_csv(file_path)
                print(f"SUCCESS: Data for {ticker} saved and cleaned to {file_path}")
            
            except Exception as e:
                print(f"ERROR: Could not process data for {ticker}. Error: {e}")
    else:
        # Fallback if yf.download returns a single-ticker DataFrame (unlikely with list input)
        print("ERROR: Expected MultiIndex but received single index. Review input TICKERS.")


    print("\n--- Robust Data Collection Complete ---")

if __name__ == '__main__':
    fetch_and_save_data()