"""
LSTM Autoencoder for Anomaly Detection in Olive Orchards
========================================================

This module defines the neural network architecture used to learn the "Normal"
growth patterns of olive trees.

Concept:
--------
1. We train this model ONLY on healthy/normal data (Dataset A).
2. The model learns to compress (Encode) and reconstruct (Decode) these healthy patterns.
3. When we feed it "Stressed" data, it will fail to reconstruct it accurately.
4. High Reconstruction Error = Water Stress / Anomaly.

Architecture:
-------------
Input Sequence -> [Encoder LSTM] -> Latent Vector (Compressed) -> [Decoder LSTM] -> Output Sequence

Author: OliveGuard Research Team
"""

import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    """
    LSTM-based Autoencoder for time-series anomaly detection.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        """
        Initialize the model architecture.
        
        Args:
            input_dim (int): Number of features per time step (e.g., 5 for NDVI, NDWI, Temp, Precip, VH).
            hidden_dim (int): Number of neurons in the LSTM hidden layer (capacity of the model).
            num_layers (int): Number of stacked LSTM layers (depth).
        """
        super(LSTMAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # =====================================================================
        # 1. ENCODER
        # =====================================================================
        # The Encoder reads the input sequence step-by-step and summarizes it.
        # We use batch_first=True so input shape is (batch_size, seq_len, input_dim).
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0  # Add dropout if deep to prevent overfitting
        )
        
        # =====================================================================
        # 2. DECODER
        # =====================================================================
        # The Decoder takes the summary and tries to rebuild the sequence.
        # It mirrors the encoder structure.
        self.decoder = nn.LSTM(
            input_size=hidden_dim,  # Input to decoder is the latent representation
            hidden_size=hidden_dim, # Keep same hidden size
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Final output layer to map hidden state back to original feature space
        # Maps 'hidden_dim' back to 'input_dim'
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        """
        Forward pass logic.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            x_hat (Tensor): Reconstructed output of shape (batch_size, seq_len, input_dim)
        """
        # =====================================================================
        # STEP 1: ENCODING
        # =====================================================================
        # Pass input x through Encoder LSTM.
        # enc_out: Output features for each time step (we ignore this for the bottleneck)
        # (hidden, cell): The final internal state (h_n, c_n). This IS the compressed "context".
        _, (hidden, cell) = self.encoder(x)
        
        # We use the final hidden state as our "Latent Vector" (compressed representation).
        # hidden shape: (num_layers, batch_size, hidden_dim)
        # We take the last layer's hidden state.
        latent_vector = hidden[-1]  # Shape: (batch_size, hidden_dim)
        
        # =====================================================================
        # STEP 2: PREPARE DECODER INPUT
        # =====================================================================
        # The decoder needs a sequence input. We repeat the latent vector 
        # for each time step to give it context at every point.
        # seq_len is x.size(1)
        seq_len = x.size(1)
        
        # Repeat vector: (batch_size, hidden_dim) -> (batch_size, seq_len, hidden_dim)
        # This tells the decoder: "Here is the summary, now reconstruct step 1, step 2..."
        dec_input = latent_vector.unsqueeze(1).repeat(1, seq_len, 1)
        
        # =====================================================================
        # STEP 3: DECODING
        # =====================================================================
        # Pass the repeated latent vector through the Decoder LSTM.
        # dec_out shape: (batch_size, seq_len, hidden_dim)
        dec_out, _ = self.decoder(dec_input)
        
        # =====================================================================
        # STEP 4: RECONSTRUCTION
        # =====================================================================
        # Map the decoder's hidden states back to the original feature space (e.g., NDVI value).
        # x_hat shape: (batch_size, seq_len, input_dim)
        x_hat = self.output_layer(dec_out)
        
        return x_hat

# =============================================================================
# EXAMPLE USAGE / SELF-CHECK
# =============================================================================
if __name__ == "__main__":
    print("🧪 Testing LSTM Autoencoder Architecture...")
    
    # 1. Define dummy parameters
    BATCH_SIZE = 32   # Number of samples (orchard plots)
    SEQ_LEN = 30      # 30 days history
    INPUT_DIM = 5     # 5 Features (NDVI, NDWI, Temp, Precip, VH)
    HIDDEN_DIM = 16   # Compress info into 16 neurons
    
    # 2. Create model
    model = LSTMAutoencoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM)
    print(f"✅ Model created with Input={INPUT_DIM}, Hidden={HIDDEN_DIM}")
    print(model)
    
    # 3. Create dummy input data (Random noise)
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
    print(f"📄 Input shape: {dummy_input.shape}")
    
    # 4. Forward pass
    output = model(dummy_input)
    print(f"📄 Output shape: {output.shape}")
    
    # 5. Validation
    assert output.shape == dummy_input.shape, "❌ Shape mismatch! Output must match Input."
    print("✅ Test Passed: Input and Output shapes match perfectly.")
