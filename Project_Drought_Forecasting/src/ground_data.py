import pandas as pd
import numpy as np
import os

REQUIRED_COLUMNS = ['Region_ID', 'Date', 'Crop_Type', 'Yield', 'Drought_Index']

def generate_template(output_path='ground_data_template.csv'):
    """Generates a template CSV file for ground data."""
    df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    df.to_csv(output_path, index=False)
    print(f"Template generated at {output_path}")

def validate_ground_data(file_path):
    """
    Validates the ground data CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Validated dataframe if successful.
        
    Raises:
        ValueError: If validation fails.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")
    
    # Check columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check Date format
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception:
        raise ValueError("Column 'Date' must be convertible to datetime.")
    
    # Check numeric columns
    for col in ['Yield', 'Drought_Index']:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric.")
            
    # Check ranges (warnings/errors)
    if (df['Yield'] < 0).any():
        raise ValueError("Yield cannot be negative.")
        
    print("Validation successful.")
    return df

def generate_mock_data(output_path='data/mock_ground_data.csv'):
    """Generates mock ground data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=24, freq='M')
    regions = ['Souss', 'Haouz', 'Doukkala']
    crops = ['Wheat', 'Barley']
    
    data = []
    for region in regions:
        for crop in crops:
            for date in dates:
                # Synthetic yield with some random variation
                yield_val = 15 + np.random.normal(0, 2) 
                # Synthetic drought index (SPI-like)
                drought_idx = np.random.normal(0, 1)
                
                data.append({
                    'Region_ID': region,
                    'Date': date,
                    'Crop_Type': crop,
                    'Yield': max(0, yield_val),
                    'Drought_Index': drought_idx
                })
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Mock data generated at {output_path}")
    return df

if __name__ == "__main__":
    # Example usage
    try:
        mock_path = 'data/mock_ground_data.csv'
        generate_mock_data(mock_path)
        validate_ground_data(mock_path)
    except Exception as e:
        print(f"Error: {e}")
