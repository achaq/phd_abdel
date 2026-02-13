import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1, kernel_size=3):
        super(CNNLSTM, self).__init__()
        
        # 1D CNN Layer
        # Input shape: (Batch, Channels/Features, Seq_Len)
        # Note: PyTorch Conv1d expects (Batch, Channels, Seq_Len)
        # But our data is usually (Batch, Seq_Len, Features)
        # We'll transpose in forward
        
        self.conv1 = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=64, 
            kernel_size=kernel_size, 
            padding='same' # Keep sequence length same for simplicity
        )
        self.relu = nn.ReLU()
        # self.pool = nn.MaxPool1d(kernel_size=2) # Optional pooling
        
        # LSTM Layer
        # Input to LSTM: (Batch, Seq_Len, Features)
        self.lstm = nn.LSTM(
            input_size=64, 
            hidden_size=hidden_dim, 
            num_layers=n_layers, 
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)
        
        # Permute for Conv1d: (Batch, Features, Seq_Len)
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.relu(x)
        
        # Permute back for LSTM: (Batch, Seq_Len, Features)
        x = x.permute(0, 2, 1)
        
        out, (hn, cn) = self.lstm(x)
        
        # Use the last hidden state
        last_hidden = out[:, -1, :]
        
        prediction = self.fc(last_hidden)
        return prediction

def train_model(model, X_train, y_train, batch_size=32, num_epochs=50, learning_rate=0.001):
    """
    Trains the CNN-LSTM model.
    """
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    loss_history = []
    
    print("Starting training...")
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
            
    return loss_history

if __name__ == "__main__":
    # Test
    input_dim = 3 # Precip, Temp, NDVI
    seq_len = 12
    hidden_dim = 32
    output_dim = 1
    batch_size = 16
    
    # Mock Data
    X = np.random.randn(100, seq_len, input_dim)
    y = np.random.randn(100, output_dim)
    
    model = CNNLSTM(input_dim, hidden_dim, output_dim)
    print(model)
    
    history = train_model(model, X, y, batch_size=batch_size, num_epochs=20)
