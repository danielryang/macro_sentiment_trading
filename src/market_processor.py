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
        
        # Returns and momentum features
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'return_lag_{lag}'] = df['returns'].shift(lag)
            df[f'return_ma_{lag}'] = df['returns'].rolling(lag).mean()
            df[f'return_std_{lag}'] = df['returns'].rolling(lag).std()
            
        # Volatility features
        df['volatility_20d'] = df['returns'].rolling(20).std()
        df['volatility_ratio'] = df['volatility_20d'] / df['volatility_20d'].rolling(60).mean()
        df['volatility_change'] = df['volatility_20d'].pct_change()
        
        # Price levels and moving averages
        for window in [5, 10, 20, 50, 100]:
            df[f'ma_{window}d'] = df['Close'].rolling(window).mean()
            df[f'ma_ratio_{window}d'] = df['Close'] / df[f'ma_{window}d']
            df[f'ma_diff_{window}d'] = df[f'ma_{window}d'].diff()
            
        # Price momentum and trend features
        for window in [5, 10, 20, 50]:
            df[f'momentum_{window}d'] = df['Close'] / df['Close'].shift(window) - 1
            df[f'roc_{window}d'] = df['Close'].pct_change(window)
            df[f'trend_{window}d'] = (df['Close'] > df[f'ma_{window}d']).astype(int)
            
        # Range and volatility features
        df['daily_range'] = (df['High'] - df['Low']) / df['Close']
        df['range_ma_5d'] = df['daily_range'].rolling(5).mean()
        df['range_ratio'] = df['daily_range'] / df['range_ma_5d']
        
        # Volume features
        if 'Volume' in df.columns:
            df['volume_ma_5d'] = df['Volume'].rolling(5).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma_5d']
            df['volume_trend'] = df['Volume'].rolling(20).mean().pct_change()
            
        # Mean reversion features
        df['zscore_20d'] = (df['Close'] - df['ma_20d']) / df['Close'].rolling(20).std()
        df['zscore_50d'] = (df['Close'] - df['ma_50d']) / df['Close'].rolling(50).std()
        
        # Trend strength indicators
        df['adx_14d'] = self.compute_adx(df, 14)
        df['trend_strength'] = abs(df['ma_5d'] - df['ma_20d']) / df['Close']
        
        return df
        
    def compute_adx(self, data: pd.DataFrame, period: int) -> pd.Series:
        """
        Compute Average Directional Index (ADX).
        
        Args:
            data: DataFrame with OHLC data
            period: ADX period
            
        Returns:
            Series with ADX values
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
        
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
                for lag in [1, 2, 3, 5]:
                    merged[f'{col}_lag_{lag}'] = merged[col].shift(lag)
                    
            # Create rolling sentiment features
            for col in sentiment_cols:
                merged[f'{col}_ma_5d'] = merged[col].rolling(5).mean()
                merged[f'{col}_ma_20d'] = merged[col].rolling(20).mean()
                merged[f'{col}_std_5d'] = merged[col].rolling(5).std()
                merged[f'{col}_std_10d'] = merged[col].rolling(10).std()
                
            # Compute sentiment acceleration and momentum
            merged['sentiment_acceleration'] = (
                merged['mean_sentiment_ma_5d'] - merged['mean_sentiment_ma_20d']
            )
            merged['sentiment_momentum'] = merged['mean_sentiment'].diff(5)
            
            # Create interaction features
            merged['sentiment_volatility'] = merged['mean_sentiment'] * merged['volatility_20d']
            merged['sentiment_trend'] = merged['mean_sentiment'] * merged['trend_strength']
            
            # Drop rows with NaN values
            merged = merged.dropna()
            
            aligned_data[asset_name] = merged
            
        return aligned_data 