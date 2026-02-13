import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(filepath):
    """Loads the dataset and sets the date as index."""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

def feature_engineering(df):
    """
    Creates lag features and rolling averages to capture time-series patterns.
    The AI needs to know 'what happened in the last 30 days' to predict stress.
    """
    df_feat = df.copy()
    
    # 1. Rolling Averages (Trend)
    # ---------------------------
    # "Is it getting hotter over the last week/month?"
    for window in [7, 30]:
        df_feat[f'Temp_mean_{window}d'] = df_feat['Temperature_C'].rolling(window=window).mean()
        df_feat[f'Precip_sum_{window}d'] = df_feat['Precipitation_mm'].rolling(window=window).sum()
        df_feat[f'NDVI_mean_{window}d'] = df_feat['NDVI'].rolling(window=window).mean()

    # 2. Lag Features (History)
    # -------------------------
    # "How much did it rain 10 days ago?" (Delayed effect on roots)
    for lag in [10, 20, 30]:
        df_feat[f'Precip_lag_{lag}d'] = df_feat['Precipitation_mm'].shift(lag)
        df_feat[f'Temp_lag_{lag}d'] = df_feat['Temperature_C'].shift(lag)

    # 3. Target Definition (The "Label")
    # ----------------------------------
    # We want to predict if trees *will be stressed*.
    # Definition: Stress = Significant drop in NDVI (Health) or NDWI (Water)
    # Anomaly Approach: If NDWI drops below (Mean - 1 StdDev), it's a stress event.
    
    ndwi_threshold = df_feat['NDWI'].mean() - (1.0 * df_feat['NDWI'].std())
    
    # Create Binary Target: 1 if Stressed, 0 if Healthy
    df_feat['Is_Stressed'] = (df_feat['NDWI'] < ndwi_threshold).astype(int)
    
    # Drop NaN created by rolling/shifting
    df_feat.dropna(inplace=True)
    
    return df_feat

def visualize_data(df, output_dir):
    """Generates plots to explore the data correlations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Correlation Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()
    
    # 2. Stress Events Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['NDWI'], label='NDWI (Water Content)', color='blue', alpha=0.6)
    plt.scatter(df[df['Is_Stressed'] == 1].index, df[df['Is_Stressed'] == 1]['NDWI'], 
                color='red', label='Detected Stress', s=10)
    plt.axhline(df['NDWI'].mean(), color='black', linestyle='--', label='Mean NDWI')
    plt.legend()
    plt.title('Timeline of Water Stress Events in Ghafsai (2023-2024)')
    plt.savefig(os.path.join(output_dir, 'stress_timeline.png'))
    plt.close()

def main():
    input_csv = "Olive_Habitat/Article1_WaterStress/data/ghafsai_olive_data_2023_2024.csv"
    output_dir = "Olive_Habitat/Article1_WaterStress/notebooks/plots"
    processed_csv = "Olive_Habitat/Article1_WaterStress/data/processed_features.csv"
    
    print("Loading data...")
    df = load_data(input_csv)
    
    print("Engineering features...")
    df_processed = feature_engineering(df)
    
    print(f"Features created: {df_processed.shape[1]} columns.")
    print(f"Stress Events detected: {df_processed['Is_Stressed'].sum()} days.")
    
    print("Generating visualizations...")
    visualize_data(df_processed, output_dir)
    
    print("Saving processed dataset...")
    df_processed.to_csv(processed_csv)
    print(f"Done. Ready for Model Training! Processed data at: {processed_csv}")

if __name__ == "__main__":
    main()
