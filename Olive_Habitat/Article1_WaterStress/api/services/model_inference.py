"""
AI Inference Service
====================
Loads the trained LSTM Autoencoder and runs inference on live GEE data.

Pipeline:
1. Load Artifacts (Model, Scaler, Feature List).
2. Preprocess Live Data (FillNa -> Scale).
3. Sequence (Create 30-day window).
4. Predict (Reconstruct).
5. Error (MSE between Input and Output).
"""

import torch
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache

from src.lstm_model import LSTMAutoencoder
from src.config import MODELS_DIR, ml_config

# =============================================================================
# SINGLETON MODEL LOADER
# =============================================================================

class ModelService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = None
        self.device = torch.device("cpu") # Inference on CPU is fine for small batches
        self._load_artifacts()

    def _load_artifacts(self):
        """Load model, scaler, and features from disk."""
        try:
            # Paths
            model_path = MODELS_DIR / "lstm_autoencoder.pt"
            scaler_path = MODELS_DIR / "scaler.pkl"
            features_path = MODELS_DIR / "features.pkl"

            if not model_path.exists() or not scaler_path.exists():
                print("⚠️  Model artifacts not found. Inference will be skipped.")
                return

            # Load features list first to ensure column order
            self.features = joblib.load(features_path)
            
            # Load Scaler
            self.scaler = joblib.load(scaler_path)
            
            # Load Model
            # We need to know input_dim from features to init model architecture
            input_dim = len(self.features)
            hidden_dim = 16 # Must match training config
            
            self.model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval() # Set to evaluation mode
            
            print("✅ AI Model loaded successfully.")
            
        except Exception as e:
            print(f"❌ Failed to load model artifacts: {e}")
            self.model = None

    def predict_anomaly(self, df: pd.DataFrame) -> float:
        """
        Run inference on the dataframe. Returns Anomaly Score (0-1).
        We use the LAST 30 days of data to determine CURRENT stress.
        """
        if self.model is None or df.empty:
            return 0.0 # Default to healthy if no model
            
        # 1. Prepare Data
        # Ensure all required columns exist (fill missing with 0 or mean)
        for col in self.features:
            if col not in df.columns:
                df[col] = 0.0 # Handle missing features gracefully
                
        # Sort by date
        df = df.sort_values('date')
        
        # Select features in correct order
        data_subset = df[self.features].copy()
        
        # Interpolate & Fill
        data_subset = data_subset.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        
        # 2. Get Last Sequence
        seq_len = ml_config.sequence_length # e.g. 30
        if len(data_subset) < seq_len:
            # Not enough data for a full sequence
            # Pad with the first row repeated? Or just return 0.
            # Let's pad for robustness
            pad_len = seq_len - len(data_subset)
            padding = pd.DataFrame([data_subset.iloc[0]] * pad_len, columns=self.features)
            data_subset = pd.concat([padding, data_subset], ignore_index=True)
            
        # Take last 'seq_len' rows
        last_sequence = data_subset.iloc[-seq_len:].values
        
        # 3. Scale
        last_sequence_scaled = self.scaler.transform(last_sequence)
        
        # 4. To Tensor
        # Shape: (1, seq_len, input_dim)
        input_tensor = torch.tensor(last_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 5. Inference
        with torch.no_grad():
            reconstruction = self.model(input_tensor)
            
        # 6. Calculate Error (MSE)
        # MSE per feature per timestep, then averaged?
        # Or just total MSE.
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(reconstruction, input_tensor).item()
        
        # 7. Normalize Score
        # What is a "High" loss?
        # During training, loss might be around 0.01 - 0.05 for healthy.
        # Stressed might be > 0.1.
        # Let's use a sigmoid-like scaling or simple min/max clamp for MVP.
        # Heuristic: Loss > 0.1 is 100% stress.
        anomaly_score = min(loss * 10, 1.0) 
        
        return float(anomaly_score)

# Global instance
_model_service = ModelService()

def get_model_service():
    return _model_service
