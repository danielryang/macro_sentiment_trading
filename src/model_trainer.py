"""
Model Training and Backtesting Module

This module implements the model training and backtesting pipeline for the trading strategy.
It provides the following functionality:

1. Train and evaluate multiple models (Logistic Regression, XGBoost)
2. Implement expanding window backtest methodology
3. Compute performance metrics and statistics
4. Generate SHAP values for model interpretability

The module implements a robust backtesting framework with expanding windows,
transaction costs, and proper feature scaling. It also provides tools for model
interpretation using SHAP values to understand feature importance.

Inputs:
    - Aligned feature sets from MarketProcessor
    - Model configurations and parameters
    - Transaction cost specifications

Outputs:
    - Trained models and predictions
    - Backtest results and performance metrics
    - SHAP values for feature importance
    - Trading signals and returns

Reference: arXiv:2505.16136v1
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from typing import Dict, Tuple, List
import logging
import shap
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            'logistic': LogisticRegression(
                penalty='l2',
                C=1.0,
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ),
            'xgboost': xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                scale_pos_weight=1.0
            )
        }
        
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for training.
        
        Args:
            data: DataFrame with features and target
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Get feature columns (exclude date, returns, and target)
        feature_cols = [col for col in data.columns 
                       if col not in ['date', 'returns', 'target', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        X = data[feature_cols].values
        y = data['target'].values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        return X, y, feature_cols
        
    def train_models(self, data: pd.DataFrame) -> Dict[str, object]:
        """
        Train both logistic regression and XGBoost models.
        
        Args:
            data: DataFrame with features and target
            
        Returns:
            Dictionary of trained models
        """
        X, y, feature_cols = self.prepare_features(data)
        
        # Initialize TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        trained_models = {}
        for name, model in self.models.items():
            # Perform time-series cross-validation
            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Train model
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50 if name == 'xgboost' else None,
                    verbose=False
                )
                
                # Evaluate on validation set
                val_score = model.score(X_val, y_val)
                cv_scores.append(val_score)
            
            # Train final model on full dataset
            model.fit(X, y)
            trained_models[name] = model
            logger.info(f"Trained {name} model with mean CV score: {np.mean(cv_scores):.4f}")
            
        return trained_models
        
    def generate_signals(self, model: object, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from model predictions.
        
        Args:
            model: Trained model
            data: DataFrame with features
            
        Returns:
            Series of trading signals (-1 for short, 1 for long)
        """
        X, _, _ = self.prepare_features(data)
        probas = model.predict_proba(X)[:, 1]
        
        # Use dynamic threshold based on prediction distribution
        threshold = np.percentile(probas, 60)  # More conservative threshold
        
        return pd.Series(np.where(probas > threshold, 1, -1), index=data.index)
        
    def compute_returns(self, signals: pd.Series, data: pd.DataFrame, 
                       transaction_cost: float) -> pd.Series:
        """
        Compute strategy returns including transaction costs.
        
        Args:
            signals: Series of trading signals
            data: DataFrame with market data
            transaction_cost: Transaction cost per round trip
            
        Returns:
            Series of strategy returns
        """
        # Compute position changes
        position_changes = signals.diff().abs()
        
        # Compute strategy returns
        strategy_returns = signals.shift(1) * data['returns']
        
        # Subtract transaction costs
        strategy_returns = strategy_returns - position_changes * transaction_cost
        
        return strategy_returns
        
    def backtest(self, data: pd.DataFrame, transaction_cost: float) -> Dict[str, pd.DataFrame]:
        """
        Perform backtest using expanding window approach.
        
        Args:
            data: DataFrame with features and market data
            transaction_cost: Transaction cost per round trip
            
        Returns:
            Dictionary with backtest results for each model
        """
        # Define fold dates
        start_date = data['date'].min()
        end_date = data['date'].max()
        fold_duration = timedelta(days=365 * 2)  # 2 years per fold
        
        results = {}
        
        for model_name in self.models.keys():
            fold_results = []
            current_start = start_date
            
            while current_start + fold_duration < end_date:
                # Define train and test periods
                train_end = current_start + fold_duration
                test_end = min(train_end + timedelta(days=365), end_date)
                
                # Split data
                train_data = data[(data['date'] >= current_start) & (data['date'] < train_end)]
                test_data = data[(data['date'] >= train_end) & (data['date'] < test_end)]
                
                if len(train_data) > 0 and len(test_data) > 0:
                    # Train model
                    model = self.train_models(train_data)[model_name]
                    
                    # Generate signals and compute returns
                    signals = self.generate_signals(model, test_data)
                    returns = self.compute_returns(signals, test_data, transaction_cost)
                    
                    # Store results
                    fold_results.append(pd.DataFrame({
                        'date': test_data['date'],
                        'returns': returns,
                        'cumulative_returns': (1 + returns).cumprod(),
                        'signals': signals,
                        'position_changes': signals.diff().abs()
                    }))
                    
                current_start = train_end
                
            # Combine results from all folds
            if fold_results:
                results[model_name] = pd.concat(fold_results)
                
        return results
        
    def compute_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Compute performance metrics.
        
        Args:
            returns: Series of strategy returns
            
        Returns:
            Dictionary of performance metrics
        """
        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        
        # Drawdown metrics
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = cumulative_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Win rate and profit factor
        win_rate = (returns > 0).mean()
        profit_factor = abs(returns[returns > 0].sum() / returns[returns < 0].sum())
        
        # Additional metrics
        calmar_ratio = annualized_return / abs(max_drawdown)
        sortino_ratio = np.sqrt(252) * returns.mean() / returns[returns < 0].std()
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio
        }
        
        return metrics
        
    def explain_predictions(self, model: object, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute SHAP values for model predictions.
        
        Args:
            model: Trained XGBoost model
            data: DataFrame with features
            
        Returns:
            DataFrame with SHAP values
        """
        if not isinstance(model, xgb.XGBClassifier):
            raise ValueError("SHAP values can only be computed for XGBoost models")
            
        X, _, feature_names = self.prepare_features(data)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X)
        
        # Create DataFrame with SHAP values
        shap_df = pd.DataFrame(
            shap_values,
            columns=feature_names,
            index=data.index
        )
        
        # Add feature importance summary
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 most important features:")
        logger.info(importance_df.head(10))
        
        return shap_df 