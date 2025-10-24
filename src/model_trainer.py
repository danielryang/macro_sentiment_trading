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
        
    def prepare_features(self, data: pd.DataFrame, scaler=None, fit_scaler=True):
        """
        Prepare features and target for training. Uses ALL available features except metadata.
        Args:
            data: DataFrame with features and target
            scaler: Optional StandardScaler instance
            fit_scaler: If True, fit scaler; else, transform only
        Returns:
            X, y, feature_cols, scaler
        """
        # Exclude only metadata columns - USE ALL OTHER FEATURES (577+)
        exclude_keywords = ['date', 'Date', 'returns', 'target', 'index', 'target_return']

        # Also exclude asset-specific OHLCV columns (they have asset names in them)
        feature_cols = []
        for col in data.columns:
            # Skip if column is in exclude list
            if col in exclude_keywords:
                continue
            # Skip if column contains OHLCV keywords with = (like Close_EURUSD=X)
            if any(kw in col for kw in ['Open_', 'High_', 'Low_', 'Close_', 'Volume_']) and '=' in col:
                continue
            # Otherwise, it's a feature - include it!
            feature_cols.append(col)

        X = data[feature_cols].copy()
        
        # Handle NaN values by forward fill, then backward fill, then fill with 0
        X = X.ffill().bfill().fillna(0)
        
        # Convert to numpy array
        X = X.values
        y = data['target'].values

        # Handle scaling based on scaler and fit_scaler parameters
        if scaler is None:
            if fit_scaler:
                # Create new scaler and fit
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                # No scaler provided and not fitting - return unscaled features
                X_scaled = X
        else:
            # Scaler provided
            if fit_scaler:
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = scaler.transform(X)

        return X_scaled, y, feature_cols, scaler
        
    def train_models(self, data: pd.DataFrame) -> Tuple[Dict[str, object], Dict[str, object], List[str]]:
        """
        Train both logistic regression and XGBoost models using only training data for scaling.
        Returns dict of trained models, scalers, and feature columns.
        """
        X, y, feature_cols, scaler = self.prepare_features(data, fit_scaler=True)
        trained_models = {}
        trained_scalers = {}
        for name, model in self.models.items():
            if name == 'logistic':
                model.fit(X, y)
                trained_scalers[name] = scaler
            else:
                model.fit(X, y)
            trained_models[name] = model
        return trained_models, trained_scalers, feature_cols
        
    def generate_signals(self, model: object, data: pd.DataFrame, scaler=None) -> pd.Series:
        """
        Generate trading signals from model predictions. Use scaler for logistic regression.
        """
        if scaler is not None:
            X, _, _, _ = self.prepare_features(data, scaler=scaler, fit_scaler=False)
        else:
            X, _, _, _ = self.prepare_features(data, fit_scaler=False)
        probas = model.predict_proba(X)[:, 1]
        return pd.Series(np.where(probas > 0.5, 1, -1), index=data.index)
        
    def compute_returns(self, signals: pd.Series, data: pd.DataFrame, transaction_cost: float) -> pd.Series:
        """
        Compute strategy returns including transaction costs, using Rust extension if available.
        Args:
            signals: Series of trading signals
            data: DataFrame with market data
            transaction_cost: Transaction cost per round trip
        Returns:
            Series of strategy returns
        """
        try:
            import rust_trade_sim
            # Align signals and returns as float lists
            sig = signals.astype(float).tolist()
            rets = data['returns'].astype(float).tolist()
            strat_returns, _ = rust_trade_sim.simulate_trades(sig, rets, float(transaction_cost))
            return pd.Series(strat_returns, index=signals.index)
        except ImportError:
            # Fallback to Python implementation
            position_changes = signals.diff().abs()
            strategy_returns = signals.shift(1) * data['returns']
            strategy_returns = strategy_returns - position_changes * transaction_cost
            return strategy_returns
        
    def backtest(self, data: pd.DataFrame, transaction_cost: float, asset_name: str = None, save_results: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Perform backtest using expanding window approach, strictly as in the research.
        """
        # Handle date as either index or column
        # Priority: 1) date/Date column, 2) date index, 3) any datetime column
        date_series = None

        # Check for date column (case-insensitive)
        for col in ['date', 'Date', 'DATE']:
            if col in data.columns:
                # Check if it's actually valid (not all NaT)
                if not data[col].isna().all():
                    date_series = data[col]
                    break

        # If no valid column, try index
        if date_series is None and data.index.name == 'date':
            if not pd.isna(data.index).all():
                date_series = data.index

        # Last resort: find any datetime column
        if date_series is None:
            for col in data.columns:
                if pd.api.types.is_datetime64_any_dtype(data[col]) and not data[col].isna().all():
                    date_series = data[col]
                    break

        if date_series is None:
            raise ValueError("No valid date column or index found in data")

        start_date = date_series.min()
        end_date = date_series.max()
        fold_duration = timedelta(days=365 * 2)
        results = {}
        for model_name in self.models.keys():
            fold_results = []
            current_start = start_date
            scaler = None
            while current_start + fold_duration < end_date:
                train_end = current_start + fold_duration
                test_end = min(train_end + timedelta(days=365), end_date)

                # Filter by date (determine which column/index to use)
                # Use the same date_series we identified earlier
                if isinstance(date_series, pd.Series):
                    # It's a column - find its name
                    date_col_name = None
                    for col in ['date', 'Date', 'DATE']:
                        if col in data.columns and data[col].equals(date_series):
                            date_col_name = col
                            break
                    if date_col_name:
                        train_data = data[(data[date_col_name] >= current_start) & (data[date_col_name] < train_end)]
                        test_data = data[(data[date_col_name] >= train_end) & (data[date_col_name] < test_end)]
                    else:
                        # Fallback: use date_series directly by index matching
                        train_mask = (date_series >= current_start) & (date_series < train_end)
                        test_mask = (date_series >= train_end) & (date_series < test_end)
                        train_data = data[train_mask]
                        test_data = data[test_mask]
                else:
                    # It's an index
                    train_data = data[(data.index >= current_start) & (data.index < train_end)]
                    test_data = data[(data.index >= train_end) & (data.index < test_end)]

                if len(train_data) > 0 and len(test_data) > 0:
                    trained_models, trained_scalers = self.train_models(train_data)
                    model = trained_models[model_name]
                    scaler = trained_scalers.get(model_name, None)
                    signals = self.generate_signals(model, test_data, scaler=scaler)
                    returns = self.compute_returns(signals, test_data, transaction_cost)

                    # Get date series for results - use same logic as above
                    if isinstance(date_series, pd.Series):
                        # Find the date column name
                        date_col_name = None
                        for col in ['date', 'Date', 'DATE']:
                            if col in test_data.columns:
                                date_col_name = col
                                break
                        if date_col_name:
                            result_dates = test_data[date_col_name]
                        else:
                            # Use index positions to extract from date_series
                            result_dates = date_series[test_data.index]
                    else:
                        result_dates = test_data.index

                    fold_results.append(pd.DataFrame({
                        'date': result_dates,
                        'returns': returns,
                        'cumulative_returns': (1 + returns).cumprod(),
                        'signals': signals,
                        'position_changes': signals.diff().abs()
                    }))
                current_start = train_end
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