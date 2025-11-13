import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from src.predict_utils import run_2class_prediction, get_attention_weights, BEST_MODEL_PATH # Import necessary functions and constants
from tensorflow.keras.models import load_model # Used for checking model existence

import random
import tensorflow as tf

SEED_VALUE = 42

# Set seeds for reproducibility
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)
random.seed(SEED_VALUE)


# --- STREAMLIT APP LAYOUT ---

st.set_page_config(layout="wide", page_title="GRU-Attention Stock Predictor")

st.title("📈 GRU-Attention IT Stock Directional Predictor")
st.markdown(
    "This application uses Transfer Learning (pre-trained GRU-Attention model) to predict the next-day price direction (UP vs. NOT UP)."
)
st.markdown("---")

col1, col2 = st.columns([1, 2])

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Select Ticker")
    
    # 🌟 CRITICAL CHANGE: Use text_input instead of selectbox
    ticker = st.text_input(
        "Enter Stock Ticker (e.g., NVDA, TCS.NS):", 
        value="MSFT" # Sets a default value
    ).upper() # Converts input to uppercase for consistency with yfinance
    
    st.caption("Note: The model is only pre-trained on IT stocks.")
    
    # Check if the model file exists locally before running
    # ... (Model check code remains the same) ...
    
    st.header("2. Run Analysis")
    if ticker and st.button(f"Predict & Analyze for {ticker}", type="primary"):
        # Run the core prediction logic
        
        with st.spinner(f"Running Transfer Learning for {ticker}..."):
            
            # --- CALL THE BACKEND PREDICTION FUNCTION ---
            results = run_2class_prediction(ticker)
            
            if results:
                confidence, predicted_direction, X_pred, raw_df = results
                
                # --- Get Attention Weights ---
                attention_weights = get_attention_weights(BEST_MODEL_PATH, X_pred)
                
                # --- Display Results ---
                st.subheader("Final Prediction (T+1)")
                
                color = "green" if predicted_direction == "UP" else "red"
                st.markdown(
                    f"Predicted Direction: **<span style='color:{color}; font-size:24px;'>{predicted_direction}</span>**", 
                    unsafe_allow_html=True
                )
                st.metric("Confidence Score (P[UP])", f"{confidence:.2%}")
            
            else:
                st.warning("Prediction failed. Check console for yfinance data errors.")

# Visualization Column (Col 2)
with col2:
    if 'confidence' in locals() and confidence:
        st.header("3. Model Interpretability: Attention Heatmap")
        
        # --- Prepare Data for Plotting ---
        # Get the dates for the last 30 trading days (required for the X-axis)
        # We need the last 30 dates that were used to form the X_pred sequence
        last_30_dates = raw_df.index[-30:].strftime('%m/%d').tolist()
        
        # Create DataFrame for Visualization
        heatmap_df = pd.DataFrame({
            'Date': last_30_dates,
            'Weight': attention_weights
        })

        # --- Plotting the Attention Scores ---
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Highlight the day with the maximum attention
        max_weight_index = heatmap_df['Weight'].idxmax()
        colors = ['red' if i == max_weight_index else 'skyblue' for i in heatmap_df.index]
        
        ax.bar(heatmap_df['Date'], heatmap_df['Weight'], color=colors, edgecolor='black')
        ax.set_title(f'Attention Focus for {ticker}: Last 30 Trading Days', fontsize=14)
        ax.set_ylabel('Attention Weight (Importance)', fontsize=12)
        
        # Annotate the max bar
        max_weight = heatmap_df['Weight'].max()
        ax.annotate(f'Max Focus ({max_weight:.2f})',
                    xy=(max_weight_index, max_weight), 
                    xytext=(max_weight_index, max_weight + 0.05),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=0.5),
                    ha='center', fontsize=9)
        
        plt.xticks(rotation=45, ha='right', fontsize=10)
        st.pyplot(fig)
        
        st.info(
            "The **Attention Weight** indicates which day's data within the 30-day sequence was most influential in making the final prediction. Taller bars mean greater focus."
        )

# --- How to Run ---
if __name__ == '__main__':
    # You must modify run_2class_prediction to return the 4 variables!
    pass