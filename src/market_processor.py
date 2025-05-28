"""
Market Data Processing Module

This module handles the collection and processing of market data for trading assets.
It implements the following functionality:

1. Download market data from Yahoo Finance
2. Compute returns and technical features
3. Align market data with sentiment features
4. Generate time-series features and indicators

The module processes data for EUR/USD, USD/JPY, and Treasury futures, computing
various market features such as returns, volatility, moving averages, and momentum
indicators. It also aligns these features with sentiment data for model training.

Inputs:
    - Yahoo Finance ticker symbols
    - Date range for data collection
    - Sentiment features from SentimentAnalyzer

Outputs:
    - Market data with computed features
    - Aligned feature sets for model training
    - Technical indicators and statistics

Reference: arXiv:2505.16136v1
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketProcessor:
    def __init__(self):
        self.assets = {
            'EURUSD': 'EURUSD=X',
            'USDJPY': 'USDJPY=X',
            'TNOTE': 'ZN=F'
        }
        
    def fetch_market_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data for specified assets.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary of DataFrames with market data for each asset
        """
        market_data = {}
        
        for asset_name, ticker in self.assets.items():
            try:
                data = yf.download(ticker, start=start_date, end=end_date)
                data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
                data['target'] = (data['returns'].shift(-1) > 0).astype(int)
                market_data[asset_name] = data
                logger.info(f"Successfully downloaded data for {asset_name}")
            except Exception as e:
                logger.error(f"Error downloading data for {asset_name}: {str(e)}")
                
        return market_data
        
    def compute_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute market features for a single asset.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with added features
        """
        df = data.copy()
        
        # Lagged returns
        for lag in [1, 2, 3, 5]:
            df[f'return_lag_{lag}'] = df['returns'].shift(lag)
            
        # Volatility features
        df['volatility_20d'] = df['returns'].rolling(20).std()
        
        # Moving averages
        df['ma_5d'] = df['Close'].rolling(5).mean()
        df['ma_20d'] = df['Close'].rolling(20).mean()
        
        # Price momentum
        df['momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_20d'] = df['Close'] / df['Close'].shift(20) - 1
        
        return df
        
    def align_features(self, market_data: Dict[str, pd.DataFrame], 
                      sentiment_features: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Align market and sentiment features.
        
        Args:
            market_data: Dictionary of market data DataFrames
            sentiment_features: DataFrame with sentiment features
            
        Returns:
            Dictionary of aligned feature DataFrames for each asset
        """
        aligned_data = {}
        
        for asset_name, asset_data in market_data.items():
            # Merge market and sentiment data
            merged = pd.merge(
                asset_data,
                sentiment_features,
                left_index=True,
                right_on='date',
                how='left'
            )
            
            # Create lagged sentiment features
            sentiment_cols = [col for col in merged.columns if col not in asset_data.columns]
            for col in sentiment_cols:
                for lag in [1, 2, 3]:
                    merged[f'{col}_lag_{lag}'] = merged[col].shift(lag)
                    
            # Create rolling sentiment features
            for col in sentiment_cols:
                merged[f'{col}_ma_5d'] = merged[col].rolling(5).mean()
                merged[f'{col}_ma_20d'] = merged[col].rolling(20).mean()
                merged[f'{col}_std_5d'] = merged[col].rolling(5).std()
                merged[f'{col}_std_10d'] = merged[col].rolling(10).std()
                
            # Compute sentiment acceleration
            merged['sentiment_acceleration'] = (
                merged['mean_sentiment_ma_5d'] - merged['mean_sentiment_ma_20d']
            )
            
            # Drop rows with NaN values
            merged = merged.dropna()
            
            aligned_data[asset_name] = merged
            
        return aligned_data 