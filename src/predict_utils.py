import pandas as pd
import numpy as np
import os
import pickle
import yfinance as yf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Attention, GRU, Dense, Dropout, Input # Ensure all layers are imported

# --- Configuration Constants (Local Path) ---
DRIVE_BASE_PATH = os.path.dirname(os.path.abspath(__file__)) 
DRIVE_BASE_PATH = os.path.dirname(DRIVE_BASE_PATH) 
MODELS_PATH = os.path.join(DRIVE_BASE_PATH, 'models') 
SCALER_PATH = os.path.join(MODELS_PATH, 'pretrain_scaler.pkl')
BEST_MODEL_PATH = os.path.join(MODELS_PATH, 'IT_GRU_Pretrain_Best.h5')

SEQ_LENGTH = 30
LOOKBACK_DAYS = 120 # Increased for better fine-tuning data
WINDOW_20D = 20
WINDOW_50D = 50
WINDOW_14D = 14

FINAL_FEATURE_COLS = [
    'F1_Daily_Return', 'F2_Volume_Momentum', 'F3_RSI', 'F4_MACD_Diff', 
    'F5_BB_Position', 'F6_Price_VWAP_Ratio', 'F7_Volatility_Regime', 
    'F9_Support_Resistance'
]

# ====================================================================
# --- LOGGING FUNCTION ---
# ====================================================================

def log_prediction(ticker: str, confidence: float, direction: str, loss: float, history: dict, output_path: str = 'prediction_log.csv'):
    """Logs the results and the final epoch's metrics to a CSV file."""
    
    # Safely determine the validation accuracy key
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc'
    
    final_ft_loss = history['loss'][-1]
    final_ft_acc = history['accuracy'][-1]
    final_val_loss = history['val_loss'][-1]
    final_val_acc = history[val_acc_key][-1]
    
    log_data = {
        'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Ticker': ticker,
        'Confidence_Score': f"{confidence:.4f}",
        'Predicted_Direction': direction,
        'FT_Final_Loss': f"{final_ft_loss:.4f}",
        'FT_Final_Acc': f"{final_ft_acc:.4f}",
        'FT_Final_Val_Loss': f"{final_val_loss:.4f}",
        'FT_Final_Val_Acc': f"{final_val_acc:.4f}",
    }
    
    new_row_df = pd.DataFrame([log_data])
    log_file_path = os.path.join(DRIVE_BASE_PATH, output_path)

    if not os.path.exists(log_file_path):
        new_row_df.to_csv(log_file_path, index=False)
    else:
        new_row_df.to_csv(log_file_path, mode='a', header=False, index=False)
        
    print(f"✅ Prediction logged to {output_path}")

# ====================================================================
#
# ====================================================================
from tensorflow.keras.models import Model

def get_attention_weights(base_model_path: str, X_pred: np.ndarray) -> np.ndarray:
    """
    Loads the model and creates a new model whose output is the Attention layer,
    then returns the attention weights for the final prediction sequence.
    """
    try:
        # Load the base model (which includes the Attention_Mechanism layer)
        base_model = load_model(base_model_path, custom_objects={'Attention': Attention})
        
        # Define the layer we want to observe
        attention_layer = base_model.get_layer('Attention_Mechanism')
        
        # Create a new Keras Model that takes the same input but outputs the Attention Layer's weights.
        # The Attention_Mechanism layer in Keras typically outputs the weighted context vector, 
        # but the key to attention visualization is the attention scores, which must be extracted 
        # from the layer's internal logic or output sequence before reduction.
        
        # Since your Attention layer is configured as part of a functional graph (using Attention()), 
        # the standard way is to create a model outputting the context vector and visualizing 
        # the attention scores based on the GRU output sequence (the input to the final pooling).
        
        # For simplicity and direct visualization, we will assume the output of the 
        # Attention layer before the Context_Vector_Pool reflects the weighted sequence.
        
        # We will extract the output of the layer *before* tf.reduce_mean (Context_Vector_Pool)
        # to get the sequence where high values represent high attention scores.
        
        # Find the output of the layer named 'Attention_Mechanism' (the 3D sequence output)
        attention_output_tensor = base_model.get_layer('Attention_Mechanism').output
        
        # Create a model that outputs this tensor
        attention_model = Model(inputs=base_model.input, outputs=attention_output_tensor)
        
        # Predict the attention output (shape: 1, 30, 128 (GRU_UNITS))
        weighted_sequence = attention_model.predict(X_pred, verbose=0)
        
        # To get a single score per day (30 scores), we average across the feature dimension (axis 2)
        # and normalize them to create a simple heatmap weight.
        
        # Average across the feature dimension (128 units)
        attention_scores = np.mean(weighted_sequence[0], axis=1) # Shape: (30,)
        
        # Normalize scores to 0-1 range for a heatmap
        normalized_scores = (attention_scores - np.min(attention_scores)) / (np.max(attention_scores) - np.min(attention_scores))
        
        return normalized_scores

    except Exception as e:
        print(f"❌ Error extracting attention weights: {e}")
        return np.zeros(X_pred.shape[1]) # Return zeros on failure

# ====================================================================
# --- 2. Feature Engineering Helpers ---
# ====================================================================

def fetch_realtime_data(ticker: str) -> pd.DataFrame | None:
    """Fetches the most recent 120 trading days of OHLCV data."""
    try:
        df = yf.download(ticker, period=f'{LOOKBACK_DAYS}d', progress=False,
                         auto_adjust=False, actions=False)
        if df.empty: return None

        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        df.columns = df.columns.str.capitalize()
        df.ffill(inplace=True); df.dropna(inplace=True)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    except Exception:
        return None

def calculate_rsi(series: pd.Series, window: int = WINDOW_14D) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(span=window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_diff = (exp1 - exp2) - (exp1 - exp2).ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({'MACD_Diff': macd_diff})

def calculate_vwap(df: pd.DataFrame, window: int = WINDOW_20D) -> pd.Series:
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    tpv = typical_price * df['Volume']
    return tpv.rolling(window=window).sum() / df['Volume'].rolling(window=window).sum()

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df['F1_Daily_Return'] = df['Close'].pct_change()
    df['F2_Volume_Momentum'] = df['Volume'] / df['Volume'].rolling(window=WINDOW_20D).mean()
    df['F3_RSI'] = calculate_rsi(df['Close'], window=WINDOW_14D)
    df['F4_MACD_Diff'] = calculate_macd(df['Close'])['MACD_Diff']
    df['MA_20'] = df['Close'].rolling(window=WINDOW_20D).mean()
    df['STD_20'] = df['Close'].rolling(window=WINDOW_20D).std()
    df['Lower_Band'] = df['MA_20'] - (df['STD_20'] * 2)
    df['Upper_Band'] = df['MA_20'] + (df['STD_20'] * 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        df['F5_BB_Position'] = (df['Close'] - df['Lower_Band']) / (df['Upper_Band'] - df['Lower_Band'])
    df['F7_Volatility_Regime'] = df['F1_Daily_Return'].rolling(window=WINDOW_20D).std()
    df['F6_Price_VWAP_Ratio'] = df['Close'] / calculate_vwap(df, window=WINDOW_20D)
    df['Range_50_Low'] = df['Low'].rolling(window=WINDOW_50D).min()
    df['Range_50_High'] = df['High'].rolling(window=WINDOW_50D).max()
    with np.errstate(divide='ignore', invalid='ignore'):
        df['F9_Support_Resistance'] = (df['Close'] - df['Range_50_Low']) / (df['Range_50_High'] - df['Range_50_Low'])

    return df[FINAL_FEATURE_COLS].dropna()

# ====================================================================
# --- 3. Target Creation Functions (3-Class and 2-Class) ---
# ====================================================================

def create_target_variable_3class(df: pd.DataFrame) -> np.ndarray:
    """Creates the 3-Class Classification target array (0:Down, 1:Flat, 2:Up)."""
    df['Next_Day_Return'] = df['Close'].pct_change().shift(-1) * 100
    UP_THRESHOLD = 0.5
    DOWN_THRESHOLD = -0.5
    
    def classify(ret):
        if ret >= UP_THRESHOLD: return 2 
        elif ret <= DOWN_THRESHOLD: return 0
        else: return 1
    
    df['Target_Class'] = df['Next_Day_Return'].apply(classify)
    df.dropna(subset=['Target_Class'], inplace=True)
    return df['Target_Class'].values.astype(int)

def create_target_variable_2class(df: pd.DataFrame) -> np.ndarray:
    """Creates a 2-Class Classification target array (0: NOT UP, 1: UP)."""
    df['Next_Day_Return'] = df['Close'].pct_change().shift(-1) * 100
    UP_THRESHOLD = 0.5
    
    def classify(ret):
        if ret >= UP_THRESHOLD: return 1 # UP
        else: return 0 # NOT UP
    
    df['Target_Class'] = df['Next_Day_Return'].apply(classify)
    df.dropna(subset=['Target_Class'], inplace=True)
    return df['Target_Class'].values.astype(int)

# ====================================================================
# --- 4. Data Preparation Template (Unified Function) ---
# ====================================================================

def prepare_data_for_ft(raw_df: pd.DataFrame, scaler_path: str, target_fn: callable, seq_length: int = SEQ_LENGTH) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    """
    Unified function to prepare data using a specified target creation function.
    """
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    feature_df = engineer_features(raw_df.copy())
    if len(feature_df) < seq_length + 1: return None, None, None

    raw_aligned = raw_df.loc[feature_df.index].copy()
    targets_ft_full = target_fn(raw_aligned.reset_index())
    
    scaled_data = scaler.transform(feature_df.values)
    
    total_ft_samples = len(scaled_data) - seq_length 
    if total_ft_samples <= 0: return None, None, None

    X_fine_tune_list = []
    for i in range(total_ft_samples):
        X_fine_tune_list.append(scaled_data[i : i + seq_length, :])
        
    X_fine_tune = np.array(X_fine_tune_list)
    y_fine_tune = targets_ft_full[seq_length:].copy() 

    X_prediction = scaled_data[-seq_length:, :].reshape(1, seq_length, scaled_data.shape[1])
    
    if X_fine_tune.shape[0] != y_fine_tune.shape[0]:
        print(f"❌ Critical Alignment Error: X_ft ({X_fine_tune.shape[0]}) != y_ft ({y_fine_tune.shape[0]})")
        return None, None, None
    
    print(f"✅ Data ready. FT sequences: {X_fine_tune.shape[0]} samples.")
    
    return X_prediction, X_fine_tune, y_fine_tune

# ====================================================================
# --- 5. Final Run Functions (3-Class and 2-Class) ---
# ====================================================================

def run_3class_prediction(ticker: str):
    """Runs the Transfer Learning pipeline using the original 3-class output."""
    print(f"\n--- [3-CLASS PREDICTION] Running for {ticker} ---")
    
    raw_df = fetch_realtime_data(ticker)
    if raw_df is None: return

    X_pred, X_ft, y_ft = prepare_data_for_ft(raw_df, SCALER_PATH, create_target_variable_3class)
    if X_pred is None: return

    try:
        base_model = load_model(BEST_MODEL_PATH, custom_objects={'Attention': Attention})
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3-Class Transfer Learning (Only Unfreeze Dense Head)
    for layer in base_model.layers:
        layer.trainable = False 
    base_model.get_layer('Dense_Hidden').trainable = True
    base_model.get_layer('Dense_Dropout').trainable = True
    base_model.get_layer('Output_Classifier').trainable = True

    base_model.compile(
        optimizer=Adam(learning_rate=5e-5), 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = base_model.fit(X_ft, y_ft, epochs=15, batch_size=8, validation_split=0.2, 
                             callbacks=[EarlyStopping(patience=3)], verbose=0)
    
    prediction_probas = base_model.predict(X_pred, verbose=0)[0]
    prediction_map = {0: 'DOWN', 1: 'FLAT', 2: 'UP'}
    predicted_class = np.argmax(prediction_probas)
    
    confidence = prediction_probas[predicted_class]

    # Safely determine keys
    val_loss_key = 'val_loss'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'acc'

    print(f"\n[3-Class] Confidence: {confidence:.2f}")
    print(f"[3-Class] Val Acc: {history.history[val_acc_key][-1]:.2f}")

    log_prediction(ticker, confidence, prediction_map[predicted_class], 
                   history.history[val_loss_key][-1], history.history)


def run_2class_prediction(ticker: str):
    """Runs the Transfer Learning pipeline using the specialized 2-class output (Binary)."""
    print(f"\n--- [2-CLASS PREDICTION] Running for {ticker} ---")

    raw_df = fetch_realtime_data(ticker)
    if raw_df is None: return

    X_pred, X_ft, y_ft = prepare_data_for_ft(raw_df, SCALER_PATH, create_target_variable_2class)
    if X_pred is None: return

    try:
        base_model = load_model(BEST_MODEL_PATH, custom_objects={'Attention': Attention})
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Modify Architecture: Replace the 3-class output with 2-class output
    dense_dropout_output = base_model.get_layer('Dense_Dropout').output
    new_output_layer = Dense(1, activation='sigmoid', name='Output_Classifier_Binary')(dense_dropout_output)
    base_model = Model(inputs=base_model.input, outputs=new_output_layer)

    # 3. Transfer Learning: Implement DEEP UNFREEZING
    for layer in base_model.layers:
        layer.trainable = False 
        
    # UNFREEZE: GRU-Encoder and the Classification Head (allows better specialization)
    base_model.get_layer('GRU_Encoder').trainable = True
    base_model.get_layer('Dense_Hidden').trainable = True
    base_model.get_layer('Dense_Dropout').trainable = True
    base_model.get_layer('Output_Classifier_Binary').trainable = True

    # 4. Re-compile the model for fine-tuning
    base_model.compile(
        optimizer=Adam(learning_rate=5e-5), 
        loss='binary_crossentropy', # CRITICAL: Use binary loss
        metrics=['accuracy']
    )

    # 5. Fine-Tune
    history = base_model.fit(X_ft, y_ft, epochs=15, batch_size=8, validation_split=0.2, 
                             callbacks=[EarlyStopping(patience=3)], verbose=0)
    
    # 6. Predict and Interpret
    prediction_probas = base_model.predict(X_pred, verbose=0)[0] 
    confidence = prediction_probas[0] 
    
    predicted_direction = 'UP' if confidence >= 0.5 else 'NOT UP'
    
    # Safely determine keys
    val_loss_key = 'val_loss'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc'

    final_val_acc = history.history[val_acc_key][-1]
    final_val_loss = history.history[val_loss_key][-1]

    print(f"\n--- Final Prediction for {ticker} (T+1) ---")
    print(f"Predicted Direction: **{predicted_direction}**")
    print(f"Confidence Score (UP Probability): **{confidence:.2f}**")
    print(f"FT Final Val Acc: {final_val_acc:.2f}")
    print(f"FT Final Val Loss: {final_val_loss:.4f}")

    # 7. Log Results (Log code remains here)
    log_prediction(ticker, confidence, predicted_direction, 
                   final_val_loss, history.history)

    # 8. Final Console Output (Optional, but good for debugging)
    print(f"\n--- Final Prediction for {ticker} (T+1) ---")
    print(f"Predicted Direction: **{predicted_direction}**")
    print(f"Confidence Score (UP Probability): **{confidence:.2f}**")
    
    # CRITICAL CHANGE: Return the necessary data for the Streamlit app
    return confidence, predicted_direction, X_pred, raw_df