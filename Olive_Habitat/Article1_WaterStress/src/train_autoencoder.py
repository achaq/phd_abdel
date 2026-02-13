"""
LSTM Autoencoder Training Script
================================

This script trains the anomaly detection model using the "Normal" dataset.

Methodology:
------------
1. **Load Data**: Read the extracted CSV (e.g., 'ghafsai_training_data.csv').
2. **Filter**: Keep ONLY healthy samples (stress_label == 0). The model must learn "What is Healthy?".
3. **Scale**: Normalize all features to [0, 1] range (Critical for LSTMs).
4. **Sequence**: Convert flat data into sliding windows (e.g., 30 days history).
5. **Train**: Optimize the model to minimize Reconstruction Error (MSE).
6. **Save**: Export the trained model (.pt) and the scaler (.pkl) for the API.

Usage:
------
Run this from the project root:
$ python src/train_autoencoder.py

Author: OliveGuard Research Team
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from pathlib import Path

# Local imports
from config import ml_config, output_config, MODELS_DIR
from lstm_model import LSTMAutoencoder

# =============================================================================
# 1. DATA PREPARATION
# =============================================================================

def load_and_preprocess_data(csv_path: str):
    """
    Load CSV, filter for healthy data, and scale features.
    """
    print(f"📂 Loading data from: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Data file not found: {csv_path}. Please run GEE extraction first.")
        
    df = pd.read_csv(csv_path)
    print(f"   Original shape: {df.shape}")
    
    # 1. Filter for "Healthy" only (Label 0)
    # We want the Autoencoder to learn the "Normal" distribution.
    if 'stress_label' in df.columns:
        df_healthy = df[df['stress_label'] == 0].copy()
        print(f"   Healthy samples: {len(df_healthy)} ({(len(df_healthy)/len(df))*100:.1f}%)")
    else:
        print("⚠️ 'stress_label' column missing. Assuming ALL data is healthy (Unsupervised).")
        df_healthy = df.copy()
        
    # 2. Select Features
    # We combine Optical + Climate + Radar (if available)
    features = ml_config.optical_features + ml_config.climate_features
    # Check if Radar features exist in CSV before adding them
    available_cols = df.columns.tolist()
    radar_cols = [f for f in ml_config.radar_features if f in available_cols]
    
    feature_cols = [f for f in features if f in available_cols] + radar_cols
    feature_cols = list(set(feature_cols)) # Remove duplicates
    
    print(f"   Using {len(feature_cols)} features: {feature_cols}")
    
    # 3. Handle Missing Values
    # Interpolate time series (linear) then backfill/ffill edges
    data = df_healthy[feature_cols].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    # 4. Scaling (0-1)
    # Critical for Neural Networks to converge
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    return data_scaled, scaler, feature_cols

def create_sequences(data, seq_length):
    """
    Convert flat matrix into 3D chunks (samples, seq_length, features).
    Sliding window approach.
    """
    xs = []
    # Loop through data and cut out sequences
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        xs.append(x)
        
    return np.array(xs)

# =============================================================================
# 2. TRAINING LOOP
# =============================================================================

def train_model(data_scaled, seq_length, epochs=50, batch_size=32):
    """
    Train the LSTM Autoencoder.
    """
    # 1. Create Sequences
    print(f"✂️  Creating sequences (Length: {seq_length})...")
    X = create_sequences(data_scaled, seq_length)
    print(f"   Training shape: {X.shape} (Samples, Timesteps, Features)")
    
    if len(X) == 0:
        raise ValueError("❌ Not enough data to create sequences. Need more rows than seq_length.")

    # 2. Convert to PyTorch Tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Autoencoder: Input = Target (Try to reconstruct Input)
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Initialize Model
    input_dim = X.shape[2]
    hidden_dim = 16 # Latent space size (Compression)
    model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"🤖 Model initialized on {device}")
    
    # 4. Optimizer & Loss
    criterion = nn.MSELoss() # Reconstruction Error
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 5. Training Loop
    print("\n🚀 Starting Training...")
    model.train()
    
    for epoch in range(epochs):
        train_loss = 0.0
        
        for batch_in, batch_target in dataloader:
            batch_in, batch_target = batch_in.to(device), batch_target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch_in)
            
            # Calculate Loss (Input vs Reconstructed)
            loss = criterion(output, batch_target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Log progress
        avg_loss = train_loss / len(dataloader)
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.6f}")
            
    print("✅ Training Complete!")
    return model

# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================

def main():
    # 1. Config
    csv_path = output_config.training_data_csv
    seq_length = ml_config.sequence_length # e.g. 30 days
    
    # 2. Check if data exists, if not create dummy for testing flow
    if not os.path.exists(csv_path):
        print(f"⚠️  Data file {csv_path} not found.")
        print("   Creating DUMMY data for code verification...")
        # Create dummy DF with required cols
        dates = pd.date_range(start="2023-01-01", periods=200)
        dummy_data = {
            'date': dates,
            'NDVI': np.random.uniform(0.3, 0.8, 200),
            'NDWI': np.random.uniform(-0.2, 0.1, 200),
            'Temperature_C': np.random.uniform(10, 35, 200),
            'Precipitation_mm': np.random.uniform(0, 10, 200),
            'stress_label': 0 # All healthy
        }
        pd.DataFrame(dummy_data).to_csv(csv_path, index=False)
        print("   ✅ Dummy csv created.")

    # 3. Load & Preprocess
    data_scaled, scaler, features = load_and_preprocess_data(csv_path)
    
    # 4. Train
    model = train_model(data_scaled, seq_length)
    
    # 5. Save Artifacts
    print("\n💾 Saving Artifacts...")
    
    # Save Model
    model_path = MODELS_DIR / "lstm_autoencoder.pt"
    torch.save(model.state_dict(), model_path)
    print(f"   Model saved: {model_path}")
    
    # Save Scaler
    scaler_path = MODELS_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"   Scaler saved: {scaler_path}")
    
    # Save Feature List (Important for inference ordering)
    feature_path = MODELS_DIR / "features.pkl"
    joblib.dump(features, feature_path)
    print(f"   Features saved: {feature_path}")
    
    print("\n🎉 Training Phase Complete. The 'Coach' has finished.")

if __name__ == "__main__":
    main()
