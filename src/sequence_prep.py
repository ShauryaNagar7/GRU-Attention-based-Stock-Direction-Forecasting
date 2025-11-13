# sequence_prep.py - PHASE A: SEQUENCE CREATION AND NORMALIZATION
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import pickle

# --- Configuration ---
PROCESSED_DATA_PATH = 'data/processed'
SEQUENCES_DATA_PATH = 'data/sequences'
MODELS_PATH = 'models'

SEQ_LENGTH = 30 # The look-back window for the GRU model

# The 8 Features (matching your FINAL_FEATURE_COLS from model_prep.py)
FEATURE_COLS = [
    'F1_Daily_Return', 'F2_Volume_Momentum', 'F3_RSI', 'F4_MACD_Diff', 
    'F5_BB_Position', 'F6_Price_VWAP_Ratio', 'F7_Volatility_Regime', 
    'F9_Support_Resistance'
]

def create_sequences(data: np.ndarray, targets: np.ndarray, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforms a 2D array of feature data into a 3D sequence array for RNN input.
    
    X_sequences shape: (N_samples, SEQ_LENGTH, N_features)
    y_sequences shape: (N_samples,)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        # Create the 30-day lookback sequence
        X.append(data[i:(i + seq_length), :])
        # The target is the NEXT day's prediction (i + seq_length day)
        y.append(targets[i + seq_length])
        
    return np.array(X), np.array(y)

def run_sequence_preparation():
    print("--- Starting Sequence Creation and Normalization ---")
    
    # 1. Load Combined Data
    combined_file = os.path.join(PROCESSED_DATA_PATH, 'IT_stocks_features_combined.csv')
    if not os.path.exists(combined_file):
        print(f"❌ Error: Combined file not found at {combined_file}")
        print("Please run model_prep.py first.")
        return

    df_combined = pd.read_csv(combined_file, index_col=0, parse_dates=True)
    df_combined.sort_index(inplace=True) # Sort by date just in case
    
    print(f"✅ Loaded combined data: {len(df_combined)} rows from {df_combined['Ticker'].nunique()} tickers.")
    
    # 2. Fit Scaler on ALL data (Transfer Learning Best Practice)
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Extract feature data for scaling
    feature_data = df_combined[FEATURE_COLS].values
    
    # Fit the scaler
    scaler.fit(feature_data)
    
    print(f"✅ Scaler fitted on all {len(FEATURE_COLS)} features across the dataset.")

    # 3. Save the Scaler (Crucial for Prediction/Fine-Tuning)
    scaler_output_path = os.path.join(MODELS_PATH, 'pretrain_scaler.pkl')
    with open(scaler_output_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler object saved to {scaler_output_path}")

    # 4. Create Sequences Per Ticker
    X_list, y_list = [], []
    
    for ticker in df_combined['Ticker'].unique():
        ticker_data = df_combined[df_combined['Ticker'] == ticker].copy()
        
        # Apply the pre-fitted scaler to the current ticker's features
        scaled_features = scaler.transform(ticker_data[FEATURE_COLS].values)
        
        # Extract targets (already 0, 1, 2)
        targets = ticker_data['Target_Class'].values
        
        # Create 3D sequences
        if len(scaled_features) > SEQ_LENGTH:
            X_ticker, y_ticker = create_sequences(scaled_features, targets, SEQ_LENGTH)
            X_list.append(X_ticker)
            y_list.append(y_ticker)
            print(f"  > Created {len(X_ticker)} sequences for {ticker}.")
        else:
            print(f"  > Skipping {ticker}: Not enough data ({len(scaled_features)} rows) for {SEQ_LENGTH}-day sequence.")

    if not X_list:
        print("❌ Failed to create any sequences. Check data length.")
        return
        
    # 5. Concatenate and Finalize Arrays
    X_pretrain = np.concatenate(X_list, axis=0)
    y_pretrain = np.concatenate(y_list, axis=0)
    
    print(f"\nFinal Pre-training Dataset Shape:")
    print(f"  X_pretrain (Features): {X_pretrain.shape} (N_samples, 30, 8)")
    print(f"  y_pretrain (Targets): {y_pretrain.shape} (N_samples,)")
    
    # 6. Save Arrays
    X_output_path = os.path.join(SEQUENCES_DATA_PATH, 'X_pretrain.npy')
    y_output_path = os.path.join(SEQUENCES_DATA_PATH, 'y_pretrain.npy')
    
    np.save(X_output_path, X_pretrain)
    np.save(y_output_path, y_pretrain)
    
    print(f"✅ Sequences saved to {SEQUENCES_DATA_PATH}/")
    print("--- Sequence Preparation Complete ---")


if __name__ == '__main__':
    # You will need to install scikit-learn and pickle (standard)
    # pip install scikit-learn
    run_sequence_preparation()