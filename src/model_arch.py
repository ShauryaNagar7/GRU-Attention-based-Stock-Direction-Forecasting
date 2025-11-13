# model_arch.py - PHASE A: MODEL CONSTRUCTION AND PRE-TRAINING
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# --- Configuration ---
SEQUENCES_DATA_PATH = 'data/sequences'
MODELS_PATH = 'models'

# Model Hyperparameters
SEQ_LENGTH = 30     # Input sequence length (time steps)
N_FEATURES = 8      # Number of features (F1 to F7, F9)
N_CLASSES = 3       # Target classes (0: Down, 1: Flat, 2: Up)
GRU_UNITS = 128     # Number of units in the GRU layer
BATCH_SIZE = 64
EPOCHS = 200        # Set high, but EarlyStopping will prevent overfitting

def build_gru_attention_model(input_shape: tuple, gru_units: int, n_classes: int) -> Model:
    """
    Constructs the GRU-Attention Hybrid model for transfer learning pre-training.
    """
    print("Building GRU-Attention Model...")
    
    # 1. Input Layer: Defines the shape of the time series data (30 days, 8 features)
    inputs = Input(shape=input_shape, name='Input_Layer')
    
    # 2. GRU Layer: Processes the time series, returning sequences for Attention
    gru_out = GRU(
        units=gru_units, 
        return_sequences=True, # MUST be True for Attention layer
        kernel_initializer='he_normal',
        name='GRU_Encoder'
    )(inputs)
    gru_out = Dropout(0.3, name='GRU_Dropout')(gru_out)
    
    # 3. Attention Layer: Learns the importance of each time step (day)
    # The GRU output (sequences) is used as both the query and value/key
    attention_output = Attention(name='Attention_Mechanism')(
        [gru_out, gru_out]
    )
    
    # 4. Global Averaging: Reduces the sequence to a fixed-size vector
    # This averages the attended sequences to prepare for the dense layers
    context_vector = tf.reduce_mean(attention_output, axis=1, name='Context_Vector')
    
    # 5. Output Classifier: Dense layers for the 3-class prediction
    dense_1 = Dense(64, activation='relu', name='Dense_Hidden')(context_vector)
    dense_1 = Dropout(0.3, name='Dense_Dropout')(dense_1)
    
    # Output layer uses softmax for multi-class probability
    outputs = Dense(n_classes, activation='softmax', name='Output_Classifier')(dense_1)
    
    model = Model(inputs=inputs, outputs=outputs, name='GRU_Attention_Pretrain_Model')
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy', # Use for integer targets (0, 1, 2)
        metrics=['accuracy']
    )
    
    print("Model compilation complete.")
    return model

def run_pre_training():
    print("--- Starting Phase A: Model Pre-training ---")
    
    # 1. Load Data
    X_path = os.path.join(SEQUENCES_DATA_PATH, 'X_pretrain.npy')
    y_path = os.path.join(SEQUENCES_DATA_PATH, 'y_pretrain.npy')
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print("❌ Error: Sequences not found. Run sequence_prep.py first.")
        return

    X_pretrain = np.load(X_path)
    y_pretrain = np.load(y_path)
    
    # Ensure y_pretrain is 1D (sparse categorical expects this)
    y_pretrain = y_pretrain.astype(int) 
    
    print(f"✅ Data Loaded: X shape {X_pretrain.shape}, y shape {y_pretrain.shape}")
    
    # 2. Build Model
    input_shape = (SEQ_LENGTH, N_FEATURES)
    model = build_gru_attention_model(input_shape, GRU_UNITS, N_CLASSES)
    model.summary()
    
    # 3. Define Callbacks
    # Save the best model weights during training
    model_output_path = os.path.join(MODELS_PATH, 'IT_GRU_Pretrain_Best.h5')
    checkpoint = ModelCheckpoint(
        model_output_path, 
        monitor='val_loss', 
        save_best_only=True, 
        mode='min', 
        verbose=1
    )
    # Stop training if validation loss stops improving (prevents overfitting)
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=20, # Number of epochs to wait
        restore_best_weights=True
    )
    
    # 4. Train Model
    print("\nStarting Training...")
    history = model.fit(
        X_pretrain, 
        y_pretrain,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15, # Use 15% of data for validation
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    print("\n--- Phase A: Pre-training Complete ---")
    print(f"Final best weights saved to: {model_output_path}")

if __name__ == '__main__':
    # Ensure models directory exists
    os.makedirs(MODELS_PATH, exist_ok=True)
    run_pre_training()