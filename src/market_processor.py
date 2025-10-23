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
        # Don't dropna here - let alignment function handle NaN removal
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
            # Make a copy of sentiment_features for each asset to avoid cross-contamination
            sentiment_features_copy = sentiment_features.copy()

            # Flatten multi-index columns if present
            if isinstance(asset_data.columns, pd.MultiIndex):
                asset_data = asset_data.copy()
                asset_data.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                                     for col in asset_data.columns.values]

            # Ensure sentiment_features has 'date' as a column, not index
            if 'date' not in sentiment_features_copy.columns and sentiment_features_copy.index.name == 'date':
                sentiment_features_copy = sentiment_features_copy.reset_index()

            # Reset asset_data index to make date a column for merging
            asset_data_reset = asset_data.reset_index()
            # Ensure the date column is named 'date' (lowercase) for consistent merging
            if 'Date' in asset_data_reset.columns:
                asset_data_reset = asset_data_reset.rename(columns={'Date': 'date'})
            elif asset_data_reset.columns[0] not in ['date']:
                asset_data_reset = asset_data_reset.rename(columns={asset_data_reset.columns[0]: 'date'})

            # Ensure both DataFrames have consistent date types
            asset_data_reset['date'] = pd.to_datetime(asset_data_reset['date']).dt.date
            sentiment_features_copy['date'] = pd.to_datetime(sentiment_features_copy['date']).dt.date
            
            # Merge market and sentiment data on date column
            logger.info(f"Merging {asset_name}: asset_data has {asset_data_reset.shape[0]} rows, sentiment has {sentiment_features_copy.shape[0]} rows")
            merged = pd.merge(
                asset_data_reset,
                sentiment_features_copy,
                on='date',
                how='left'
            )
            logger.info(f"After merge {asset_name}: {merged.shape[0]} rows")

            # Create lagged sentiment features (filter out non-numeric columns like 'date')
            sentiment_cols = [col for col in merged.columns
                             if col not in asset_data.columns and col not in ['date', 'Date']]

            # Only process numeric sentiment columns
            numeric_sentiment_cols = [col for col in sentiment_cols
                                     if merged[col].dtype in ['int64', 'float64', 'int32', 'float32']]

            for col in numeric_sentiment_cols:
                for lag in [1, 2, 3, 5]:
                    merged[f'{col}_lag_{lag}'] = merged[col].shift(lag)

            # Create rolling sentiment features
            for col in numeric_sentiment_cols:
                merged[f'{col}_ma_5d'] = merged[col].rolling(5).mean()
                merged[f'{col}_ma_20d'] = merged[col].rolling(20).mean()
                merged[f'{col}_std_5d'] = merged[col].rolling(5).std()
                merged[f'{col}_std_10d'] = merged[col].rolling(10).std()
                
            # Compute sentiment acceleration and momentum
            merged['sentiment_acceleration'] = (
                merged['mean_sentiment_ma_5d'] - merged['mean_sentiment_ma_20d']
            )
            merged['sentiment_momentum'] = merged['mean_sentiment'].diff(5)
            
            # Create interaction features (with safety checks for column existence)
            if 'mean_sentiment' in merged.columns and 'vol20' in merged.columns:
                merged['sentiment_volatility'] = merged['mean_sentiment'] * merged['vol20']
            if 'mean_sentiment' in merged.columns and 'trend_strength' in merged.columns:
                merged['sentiment_trend'] = merged['mean_sentiment'] * merged['trend_strength']

            # DEBUG: Check data before filtering
            logger.info(f"DEBUG - {asset_name} before filtering:")
            logger.info(f"  Shape: {merged.shape}")
            logger.info(f"  Columns: {list(merged.columns)[:10]}")
            if merged.shape[0] > 0:
                logger.info(f"  Sample data:")
                logger.info(f"    Date range: {merged['date'].min()} to {merged['date'].max()}")
                logger.info(f"    Price columns: {[col for col in merged.columns if 'close' in col.lower()]}")
                logger.info(f"    Sentiment columns: {[col for col in merged.columns if 'sentiment' in col.lower()]}")
            
            # Only drop rows where base price or sentiment data is missing
            # Do NOT drop rows just because rolling/lagged features are NaN
            critical_cols = []

            # Find price column
            price_col = None
            for col in merged.columns:
                if 'close' in col.lower() and 'lag' not in col.lower() and 'ma' not in col.lower():
                    price_col = col
                    break

            if price_col:
                critical_cols.append(price_col)
                logger.info(f"DEBUG - Found price column: {price_col}")
            if 'mean_sentiment' in merged.columns:
                critical_cols.append('mean_sentiment')
                logger.info(f"DEBUG - Found sentiment column: mean_sentiment")

            # Only drop rows where critical base data is missing (keep rows with NaN in derived features)
            logger.info(f"Before dropna - {asset_name}: {merged.shape[0]} rows")
            if critical_cols:
                logger.info(f"Dropping NaN for columns: {critical_cols}")
                # DEBUG: Check NaN values before dropping
                for col in critical_cols:
                    if col in merged.columns:
                        nan_count = merged[col].isna().sum()
                        logger.info(f"DEBUG - {col}: {nan_count} NaN values out of {len(merged)}")
                merged = merged.dropna(subset=critical_cols)
                logger.info(f"After dropna - {asset_name}: {merged.shape[0]} rows")
            else:
                logger.warning(f"DEBUG - No critical columns found for {asset_name}")

            # Keep the dataframe even if rolling features have NaN
            # (They will naturally have NaN at the start of the time series)

            aligned_data[asset_name] = merged
            
        return aligned_data 