#!/usr/bin/env python3
"""
Robust parquet reading utilities to handle dbdate and other compatibility issues
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging

logger = logging.getLogger(__name__)

def read_parquet_robust(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Robustly read parquet files with multiple fallback strategies.
    
    Args:
        file_path: Path to parquet file
        **kwargs: Additional arguments for pandas.read_parquet
        
    Returns:
        pandas.DataFrame
        
    Raises:
        Exception: If all reading methods fail
    """
    strategies = [
        ("pyarrow_direct", _read_with_pyarrow_direct),
        ("pyarrow_table", _read_with_pyarrow_table),
        ("fastparquet", _read_with_fastparquet),
        ("pandas_default", _read_with_pandas_default),
        ("csv_conversion", _read_with_csv_conversion)
    ]
    
    last_error = None
    
    for strategy_name, strategy_func in strategies:
        try:
            logger.info(f"Trying {strategy_name} for {file_path}")
            df = strategy_func(file_path, **kwargs)
            logger.info(f"✅ Successfully read with {strategy_name}")
            return df
        except Exception as e:
            logger.warning(f"❌ {strategy_name} failed: {e}")
            last_error = e
            continue
    
    raise Exception(f"All parquet reading strategies failed. Last error: {last_error}")

def _read_with_pyarrow_direct(file_path: str, **kwargs) -> pd.DataFrame:
    """Read using pyarrow engine directly"""
    return pd.read_parquet(file_path, engine='pyarrow', **kwargs)

def _read_with_pyarrow_table(file_path: str, **kwargs) -> pd.DataFrame:
    """Read using pyarrow table conversion"""
    table = pq.read_table(file_path)
    return table.to_pandas()

def _read_with_fastparquet(file_path: str, **kwargs) -> pd.DataFrame:
    """Read using fastparquet engine"""
    return pd.read_parquet(file_path, engine='fastparquet', **kwargs)

def _read_with_pandas_default(file_path: str, **kwargs) -> pd.DataFrame:
    """Read using pandas default engine"""
    return pd.read_parquet(file_path, **kwargs)

def _read_with_csv_conversion(file_path: str, **kwargs) -> pd.DataFrame:
    """Convert to CSV and back to avoid dbdate issues"""
    import tempfile
    import os
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_csv:
        csv_path = tmp_csv.name
    
    try:
        # Try to read with pyarrow and convert problematic columns
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        # Fix any dbdate or problematic columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Try to convert to datetime if it looks like a date
                    if 'date' in col.lower() or df[col].dtype == 'object':
                        df[col] = pd.to_datetime(df[col], errors='ignore')
                except:
                    pass
        
        # Save as CSV
        df.to_csv(csv_path, index=False)
        
        # Read CSV back
        df_clean = pd.read_csv(csv_path)
        
        # Clean up
        os.unlink(csv_path)
        
        return df_clean
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(csv_path):
            os.unlink(csv_path)
        raise e

def save_parquet_robust(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    """
    Robustly save parquet files with compatibility fixes.
    
    Args:
        df: DataFrame to save
        file_path: Output file path
        **kwargs: Additional arguments for to_parquet
    """
    # Clean data types before saving
    df_clean = df.copy()
    
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                # Try to convert to datetime if it looks like a date
                if 'date' in col.lower():
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='ignore')
            except:
                pass
    
    # Save with pyarrow engine for best compatibility
    df_clean.to_parquet(file_path, engine='pyarrow', index=False, **kwargs)
    logger.info(f"✅ Saved parquet file: {file_path}")

def test_parquet_compatibility(file_path: str) -> bool:
    """
    Test if a parquet file can be read with our robust methods.
    
    Args:
        file_path: Path to parquet file
        
    Returns:
        bool: True if readable, False otherwise
    """
    try:
        df = read_parquet_robust(file_path)
        logger.info(f"✅ Parquet file is compatible: {file_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Parquet file is incompatible: {file_path} - {e}")
        return False

















