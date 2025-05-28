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
        Fetch market data for specified assets, strictly following research math.
        Returns DataFrames with log returns, next-day target, and 20-day volatility.
        """
        market_data = {}
        for asset_name, ticker in self.assets.items():
            try:
                data = yf.download(ticker, start=start_date, end=end_date)
                # Log returns
                data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
                # Next-day target: 1 if next day's return > 0, else 0
                data['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
                # 20-day rolling volatility (std of log returns)
                data['vol20'] = data['returns'].rolling(20).std()
                market_data[asset_name] = data
                logger.info(f"Successfully downloaded and processed data for {asset_name}")
            except Exception as e:
                logger.error(f"Error downloading data for {asset_name}: {str(e)}")
        return market_data
        
    def compute_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute additional market features strictly as described in the research.
        Includes lagged returns and volatility, and ensures no look-ahead bias.
        """
        df = data.copy()
        # Lagged returns (1 day)
        df['return_lag_1'] = df['returns'].shift(1)
        # 20-day rolling volatility (already computed as vol20)
        # All features use only info up to and including day t
        return df.dropna().reset_index()
        
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
            merged['sentiment_volatility'] = merged['mean_sentiment'] * merged['vol20']
            merged['sentiment_trend'] = merged['mean_sentiment'] * merged['trend_strength']
            
            # Drop rows with NaN values
            merged = merged.dropna()
            
            aligned_data[asset_name] = merged
            
        return aligned_data 