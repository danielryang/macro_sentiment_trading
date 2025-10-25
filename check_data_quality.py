import pandas as pd
import numpy as np

# Check EURUSD data
print('=== EURUSD Data Quality ===')
eurusd = pd.read_parquet('results/20240101_20240107/aligned_data_EURUSD.parquet')
print(f'Shape: {eurusd.shape}')
print(f'NaN count per column:')
nan_counts = eurusd.isnull().sum()
print(nan_counts[nan_counts > 0].head(10))

# Check USDJPY data  
print('\n=== USDJPY Data Quality ===')
usdjpy = pd.read_parquet('results/20240101_20240107/aligned_data_USDJPY.parquet')
print(f'Shape: {usdjpy.shape}')
print(f'NaN count per column:')
nan_counts = usdjpy.isnull().sum()
print(nan_counts[nan_counts > 0].head(10))

# Check TNOTE data
print('\n=== TNOTE Data Quality ===')
tnote = pd.read_parquet('results/20240101_20240107/aligned_data_TNOTE.parquet')
print(f'Shape: {tnote.shape}')
print(f'NaN count per column:')
nan_counts = tnote.isnull().sum()
print(nan_counts[nan_counts > 0].head(10))

# Check specific columns that might be causing issues
print('\n=== Key Columns Analysis ===')
for asset, data in [('EURUSD', eurusd), ('USDJPY', usdjpy), ('TNOTE', tnote)]:
    print(f'\n{asset}:')
    # Check price columns
    price_cols = [col for col in data.columns if 'close' in col.lower() or 'price' in col.lower()]
    for col in price_cols[:3]:  # First 3 price columns
        if col in data.columns:
            nan_count = data[col].isnull().sum()
            print(f'  {col}: {nan_count} NaN values')
    
    # Check sentiment columns
    sentiment_cols = [col for col in data.columns if 'sentiment' in col.lower()]
    for col in sentiment_cols[:3]:  # First 3 sentiment columns
        if col in data.columns:
            nan_count = data[col].isnull().sum()
            print(f'  {col}: {nan_count} NaN values')



