"""
Feature Engineering Pipeline

This module provides comprehensive feature engineering for the macro sentiment trading system.
It handles feature creation, selection, and preprocessing for both sentiment and market data.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

logger = logging.getLogger(__name__)

class FeatureEngineeringPipeline:
    """
    Comprehensive feature engineering pipeline for sentiment and market data.
    
    This class handles:
    1. Feature creation from raw data
    2. Feature selection and ranking
    3. Feature scaling and normalization
    4. Feature interaction and polynomial features
    5. Time-based feature engineering
    """
    
    def __init__(self, 
                 feature_selection_method: str = 'mutual_info',
                 n_features: int = 20,
                 scaling_method: str = 'standard',
                 include_interactions: bool = True,
                 include_polynomial: bool = False,
                 polynomial_degree: int = 2):
        """
        Initialize the feature engineering pipeline.
        
        Args:
            feature_selection_method: Method for feature selection ('mutual_info', 'f_classif', 'none')
            n_features: Number of features to select
            scaling_method: Scaling method ('standard', 'minmax', 'none')
            include_interactions: Whether to include feature interactions
            include_polynomial: Whether to include polynomial features
            polynomial_degree: Degree of polynomial features
        """
        self.feature_selection_method = feature_selection_method
        self.n_features = n_features
        self.scaling_method = scaling_method
        self.include_interactions = include_interactions
        self.include_polynomial = include_polynomial
        self.polynomial_degree = polynomial_degree
        
        # Initialize components
        self.scaler = None
        self.feature_selector = None
        self.feature_names = []
        self.selected_features = []
        
    def create_sentiment_features(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive sentiment features from raw sentiment data.
        
        Args:
            sentiment_df: DataFrame with sentiment data
            
        Returns:
            DataFrame with engineered sentiment features
        """
        logger.info("Creating sentiment features...")
        
        features_df = sentiment_df.copy()
        
        # Basic sentiment features
        if 'polarity' in features_df.columns:
            features_df['sentiment_score'] = features_df['polarity']
            features_df['sentiment_abs'] = np.abs(features_df['polarity'])
            features_df['sentiment_positive'] = (features_df['polarity'] > 0).astype(int)
            features_df['sentiment_negative'] = (features_df['polarity'] < 0).astype(int)
            features_df['sentiment_neutral'] = (features_df['polarity'] == 0).astype(int)
        
        # Sentiment momentum features
        if 'polarity' in features_df.columns:
            features_df['sentiment_momentum_1d'] = features_df['polarity'].diff(1)
            features_df['sentiment_momentum_3d'] = features_df['polarity'].diff(3)
            features_df['sentiment_momentum_7d'] = features_df['polarity'].diff(7)
            
            # Sentiment acceleration
            features_df['sentiment_acceleration'] = features_df['sentiment_momentum_1d'].diff(1)
            
            # Sentiment volatility
            features_df['sentiment_volatility_3d'] = features_df['polarity'].rolling(3).std()
            features_df['sentiment_volatility_7d'] = features_df['polarity'].rolling(7).std()
            
            # Sentiment trend
            features_df['sentiment_trend_3d'] = features_df['polarity'].rolling(3).mean()
            features_df['sentiment_trend_7d'] = features_df['polarity'].rolling(7).mean()
            features_df['sentiment_trend_14d'] = features_df['polarity'].rolling(14).mean()
        
        # Volume-based features
        if 'num_articles' in features_df.columns:
            features_df['article_volume'] = features_df['num_articles']
            features_df['article_volume_ma_3d'] = features_df['num_articles'].rolling(3).mean()
            features_df['article_volume_ma_7d'] = features_df['num_articles'].rolling(7).mean()
            features_df['article_volume_ratio'] = features_df['num_articles'] / features_df['article_volume_ma_7d']
        
        # Goldstein scale features
        if 'goldstein_mean' in features_df.columns:
            features_df['goldstein_score'] = features_df['goldstein_mean']
            features_df['goldstein_volatility'] = features_df['goldstein_mean'].rolling(7).std()
            features_df['goldstein_trend'] = features_df['goldstein_mean'].rolling(7).mean()
        
        # Time-based features
        if features_df.index.dtype == 'datetime64[ns]':
            features_df['day_of_week'] = features_df.index.dayofweek
            features_df['month'] = features_df.index.month
            features_df['quarter'] = features_df.index.quarter
            features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
            features_df['is_month_end'] = (features_df.index.day >= 28).astype(int)
            features_df['is_quarter_end'] = (features_df['month'] % 3 == 0).astype(int)
        
        # Fill NaN values
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        logger.info(f"Created {len(features_df.columns)} sentiment features")
        return features_df
    
    def create_market_features(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive market features from raw market data.
        
        Args:
            market_df: DataFrame with market data
            
        Returns:
            DataFrame with engineered market features
        """
        logger.info("Creating market features...")
        
        features_df = market_df.copy()
        
        # Price-based features
        if 'close' in features_df.columns:
            # Returns
            features_df['returns_1d'] = features_df['close'].pct_change(1)
            features_df['returns_3d'] = features_df['close'].pct_change(3)
            features_df['returns_7d'] = features_df['close'].pct_change(7)
            features_df['returns_14d'] = features_df['close'].pct_change(14)
            features_df['returns_30d'] = features_df['close'].pct_change(30)
            
            # Volatility
            features_df['volatility_3d'] = features_df['returns_1d'].rolling(3).std()
            features_df['volatility_7d'] = features_df['returns_1d'].rolling(7).std()
            features_df['volatility_14d'] = features_df['returns_1d'].rolling(14).std()
            features_df['volatility_30d'] = features_df['returns_1d'].rolling(30).std()
            
            # Moving averages
            features_df['ma_3d'] = features_df['close'].rolling(3).mean()
            features_df['ma_7d'] = features_df['close'].rolling(7).mean()
            features_df['ma_14d'] = features_df['close'].rolling(14).mean()
            features_df['ma_30d'] = features_df['close'].rolling(30).mean()
            features_df['ma_60d'] = features_df['close'].rolling(60).mean()
            
            # Price ratios
            features_df['price_ma_ratio_3d'] = features_df['close'] / features_df['ma_3d']
            features_df['price_ma_ratio_7d'] = features_df['close'] / features_df['ma_7d']
            features_df['price_ma_ratio_14d'] = features_df['close'] / features_df['ma_14d']
            features_df['price_ma_ratio_30d'] = features_df['close'] / features_df['ma_30d']
            
            # Momentum indicators
            features_df['momentum_3d'] = features_df['close'] / features_df['close'].shift(3)
            features_df['momentum_7d'] = features_df['close'] / features_df['close'].shift(7)
            features_df['momentum_14d'] = features_df['close'] / features_df['close'].shift(14)
            features_df['momentum_30d'] = features_df['close'] / features_df['close'].shift(30)
            
            # RSI-like indicators
            features_df['rsi_14d'] = self._calculate_rsi(features_df['close'], 14)
            features_df['rsi_30d'] = self._calculate_rsi(features_df['close'], 30)
        
        # Volume features
        if 'volume' in features_df.columns:
            features_df['volume_ma_3d'] = features_df['volume'].rolling(3).mean()
            features_df['volume_ma_7d'] = features_df['volume'].rolling(7).mean()
            features_df['volume_ma_14d'] = features_df['volume'].rolling(14).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_ma_7d']
            features_df['volume_trend'] = features_df['volume'].rolling(7).mean() / features_df['volume'].rolling(14).mean()
        
        # High-Low features
        if 'high' in features_df.columns and 'low' in features_df.columns:
            features_df['hl_ratio'] = features_df['high'] / features_df['low']
            features_df['hl_range'] = features_df['high'] - features_df['low']
            features_df['hl_range_pct'] = features_df['hl_range'] / features_df['close']
        
        # Fill NaN values
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        logger.info(f"Created {len(features_df.columns)} market features")
        return features_df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_interaction_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create feature interactions between sentiment and market features.
        
        Args:
            features_df: DataFrame with base features
            
        Returns:
            DataFrame with interaction features
        """
        if not self.include_interactions:
            return features_df
        
        logger.info("Creating interaction features...")
        
        interaction_df = features_df.copy()
        
        # Sentiment-Market interactions
        sentiment_cols = [col for col in features_df.columns if 'sentiment' in col.lower()]
        market_cols = [col for col in features_df.columns if any(x in col.lower() for x in ['returns', 'volatility', 'momentum'])]
        
        for sent_col in sentiment_cols[:3]:  # Limit to top 3 sentiment features
            for market_col in market_cols[:3]:  # Limit to top 3 market features
                if sent_col in features_df.columns and market_col in features_df.columns:
                    interaction_name = f"{sent_col}_x_{market_col}"
                    interaction_df[interaction_name] = features_df[sent_col] * features_df[market_col]
        
        logger.info(f"Created {len(interaction_df.columns) - len(features_df.columns)} interaction features")
        return interaction_df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select the most important features using the specified method.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Tuple of (selected_features_df, selected_feature_names)
        """
        if self.feature_selection_method == 'none':
            return X, list(X.columns)
        
        logger.info(f"Selecting {self.n_features} features using {self.feature_selection_method}...")
        
        # Remove any columns with all NaN values
        X_clean = X.dropna(axis=1, how='all')
        
        # Fill remaining NaN values
        X_clean = X_clean.fillna(0)
        
        if self.feature_selection_method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(self.n_features, X_clean.shape[1]))
        elif self.feature_selection_method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=min(self.n_features, X_clean.shape[1]))
        else:
            raise ValueError(f"Unknown feature selection method: {self.feature_selection_method}")
        
        try:
            X_selected = selector.fit_transform(X_clean, y)
            selected_features = X_clean.columns[selector.get_support()].tolist()
            
            self.feature_selector = selector
            self.selected_features = selected_features
            
            logger.info(f"Selected {len(selected_features)} features")
            return pd.DataFrame(X_selected, columns=selected_features, index=X_clean.index), selected_features
            
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}. Using all features.")
            return X_clean, list(X_clean.columns)
    
    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scale features using the specified method.
        
        Args:
            X: Feature matrix
            
        Returns:
            Scaled feature matrix
        """
        if self.scaling_method == 'none':
            return X
        
        logger.info(f"Scaling features using {self.scaling_method}...")
        
        # Fill NaN values
        X_clean = X.fillna(0)
        
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
        
        try:
            X_scaled = self.scaler.fit_transform(X_clean)
            return pd.DataFrame(X_scaled, columns=X_clean.columns, index=X_clean.index)
        except Exception as e:
            logger.warning(f"Feature scaling failed: {e}. Using original features.")
            return X_clean
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """
        Fit the pipeline and transform the data.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Tuple of (transformed_features, feature_names)
        """
        logger.info("Fitting feature engineering pipeline...")
        
        # Create features
        if 'sentiment' in str(X.columns).lower():
            X_engineered = self.create_sentiment_features(X)
        else:
            X_engineered = self.create_market_features(X)
        
        # Create interactions
        X_engineered = self.create_interaction_features(X_engineered)
        
        # Select features
        X_selected, selected_features = self.select_features(X_engineered, y)
        
        # Scale features
        X_scaled = self.scale_features(X_selected)
        
        self.feature_names = list(X_scaled.columns)
        
        logger.info(f"Feature engineering complete: {len(self.feature_names)} features")
        return X_scaled, self.feature_names
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the fitted pipeline.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed feature matrix
        """
        if self.scaler is None or self.feature_selector is None:
            raise ValueError("Pipeline must be fitted before transforming")
        
        # Create features
        if 'sentiment' in str(X.columns).lower():
            X_engineered = self.create_sentiment_features(X)
        else:
            X_engineered = self.create_market_features(X)
        
        # Create interactions
        X_engineered = self.create_interaction_features(X_engineered)
        
        # Select features
        X_selected = X_engineered[self.selected_features]
        
        # Scale features
        X_scaled = self.scaler.transform(X_selected)
        
        return pd.DataFrame(X_scaled, columns=self.selected_features, index=X_selected.index)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the fitted pipeline.
        
        Returns:
            Dictionary of feature names and their importance scores
        """
        if self.feature_selector is None:
            return {}
        
        importance_scores = self.feature_selector.scores_
        feature_names = self.selected_features
        
        return dict(zip(feature_names, importance_scores))
    
    def get_feature_names(self) -> List[str]:
        """Get the names of the selected features."""
        return self.feature_names.copy()

