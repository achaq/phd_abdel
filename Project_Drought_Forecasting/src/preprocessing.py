import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class DataPreprocessor:
    def __init__(self, sequence_length=12, forecast_horizon=3):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
    def align_datasets(self, gee_dfs, ground_df, freq='M'):
        """
        Aligns multiple GEE dataframes and merges with ground truth data.
        
        Args:
            gee_dfs (list of pd.DataFrame): List of GEE dataframes (precip, temp, ndvi).
                                            Each must have 'date' column.
            ground_df (pd.DataFrame): Ground truth dataframe.
            freq (str): Resampling frequency (default 'M' for Month).
            
        Returns:
            pd.DataFrame: Merged and aligned dataframe.
        """
        # Ensure date columns are datetime
        for df in gee_dfs:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
        ground_df['Date'] = pd.to_datetime(ground_df['Date'])
        ground_df.set_index('Date', inplace=True)
        
        # Resample GEE data
        resampled_gee = []
        for df in gee_dfs:
            # Resample and take mean (handling daily to monthly)
            # Use numeric_only=True to avoid errors with non-numeric cols if any
            resampled = df.resample(freq).mean()
            resampled_gee.append(resampled)
            
        # Concatenate GEE features
        features = pd.concat(resampled_gee, axis=1)
        
        # Resample ground data (target)
        # Assuming ground data might be sparse, we might need to interpolate or join carefully
        # For simplicity, we'll reindex ground data to match features index
        target = ground_df.resample(freq).mean() # Or sum/max depending on metric
        
        # Merge
        merged = features.join(target, how='inner')
        
        # Drop NaNs created by join or missing data
        # In a real scenario, you might want to impute
        merged.dropna(inplace=True)
        
        return merged

    def fit_transform(self, df, feature_cols, target_col):
        """
        Normalizes the data.
        """
        X = df[feature_cols].values
        y = df[[target_col]].values
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        return X_scaled, y_scaled

    def create_sequences(self, X, y):
        """
        Creates sliding window sequences.
        
        Args:
            X (np.array): Feature matrix.
            y (np.array): Target vector.
            
        Returns:
            np.array, np.array: X_seq, y_seq
        """
        X_seq, y_seq = [], []
        
        total_len = len(X)
        for i in range(total_len - self.sequence_length - self.forecast_horizon + 1):
            # Input sequence
            X_seq.append(X[i : i + self.sequence_length])
            # Target (can be a sequence or a single point at horizon)
            # Here we predict the value at t + sequence_length + forecast_horizon - 1
            # Or average over the horizon. Let's predict the value at the end of horizon.
            y_seq.append(y[i + self.sequence_length + self.forecast_horizon - 1])
            
        return np.array(X_seq), np.array(y_seq)

if __name__ == "__main__":
    # Test
    # Mock GEE data
    dates = pd.date_range('2020-01-01', periods=365, freq='D')
    gee_precip = pd.DataFrame({'date': dates, 'precipitation': np.random.rand(365)*10})
    gee_temp = pd.DataFrame({'date': dates, 'temp': np.random.rand(365)*30 + 10})
    
    # Mock Ground data (monthly)
    ground_dates = pd.date_range('2020-01-01', periods=12, freq='M')
    ground_df = pd.DataFrame({
        'Date': ground_dates,
        'Yield': np.random.rand(12) * 20
    })
    
    preprocessor = DataPreprocessor(sequence_length=3, forecast_horizon=1)
    
    merged = preprocessor.align_datasets([gee_precip, gee_temp], ground_df)
    print("Merged shape:", merged.shape)
    print(merged.head())
    
    X_scaled, y_scaled = preprocessor.fit_transform(merged, ['precipitation', 'temp'], 'Yield')
    X_seq, y_seq = preprocessor.create_sequences(X_scaled, y_scaled)
    
    print("Sequences shape:", X_seq.shape, y_seq.shape)
