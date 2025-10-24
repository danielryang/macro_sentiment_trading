#!/usr/bin/env python3

"""
Multi-Timeframe Backtesting System

Provides comprehensive backtesting across multiple prediction horizons:
- Next Day (1-day forward returns)
- Next Week (5-day forward returns)  
- Next Month (21-day forward returns)
- Next Quarter (63-day forward returns)

Allows comparison of model performance across different time horizons and
generates signals for multiple timeframes simultaneously.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import json

from src.model_trainer import ModelTrainer
from src.performance_metrics import PerformanceAnalyzer

logger = logging.getLogger(__name__)

class MultiTimeframeBacktester:
    """
    Comprehensive multi-timeframe backtesting system.
    
    Trains separate models for different prediction horizons and evaluates
    their performance to identify optimal timeframes for trading signals.
    """
    
    def __init__(self, timeframes: Optional[Dict[str, int]] = None):
        """
        Initialize multi-timeframe backtester.
        
        Args:
            timeframes: Dictionary mapping timeframe names to forward-looking days
                       Default: {"1D": 1, "1W": 5, "1M": 21, "1Q": 63}
        """
        self.timeframes = timeframes or {
            "1D": 1,    # Next day
            "1W": 5,    # Next week  
            "1M": 21,   # Next month
            "1Q": 63    # Next quarter
        }
        
        self.models = {}  # Store trained models by timeframe
        self.results = {}  # Store backtest results by timeframe
        self.performance_analyzer = PerformanceAnalyzer()
        self.logger = logging.getLogger(__name__)
        
        # Transaction costs by timeframe (higher costs for longer holding periods)
        self.transaction_costs = {
            "1D": 0.0002,  # 2bp for daily trading
            "1W": 0.0003,  # 3bp for weekly rebalancing
            "1M": 0.0005,  # 5bp for monthly rebalancing  
            "1Q": 0.0008   # 8bp for quarterly rebalancing
        }
    
    def run_multi_timeframe_backtest(self, data: pd.DataFrame, asset_name: str, 
                                   models_to_test: List[str] = None) -> Dict[str, Dict]:
        """
        Run comprehensive multi-timeframe backtest.
        
        Args:
            data: Input data with features and price information
            asset_name: Name of the asset being tested
            models_to_test: List of models to test (default: ["logistic", "xgboost"])
            
        Returns:
            Dictionary containing results for each timeframe and model
        """
        if models_to_test is None:
            models_to_test = ["logistic", "xgboost"]
        
        self.logger.info(f"Starting multi-timeframe backtest for {asset_name}")
        self.logger.info(f"Timeframes: {list(self.timeframes.keys())}")
        self.logger.info(f"Models: {models_to_test}")
        
        # Results structure: {timeframe: {model: results}}
        multi_results = {}
        
        for timeframe_name, horizon_days in self.timeframes.items():
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"PROCESSING TIMEFRAME: {timeframe_name} ({horizon_days} days)")
            self.logger.info(f"{'='*60}")
            
            try:
                # Create target variable for this timeframe
                timeframe_data = self._create_timeframe_targets(data, horizon_days, timeframe_name)
                
                if timeframe_data.empty:
                    self.logger.warning(f"No data available for timeframe {timeframe_name}")
                    continue
                
                # Initialize results for this timeframe
                multi_results[timeframe_name] = {}
                
                # Test each model for this timeframe
                for model_name in models_to_test:
                    self.logger.info(f"\nTraining {model_name} for {timeframe_name} horizon...")
                    
                    # Get appropriate transaction cost
                    transaction_cost = self.transaction_costs.get(timeframe_name, 0.0005)
                    
                    # Run backtest for this model and timeframe
                    model_results = self._run_single_timeframe_backtest(
                        timeframe_data, model_name, timeframe_name, 
                        asset_name, transaction_cost
                    )
                    
                    if model_results:
                        multi_results[timeframe_name][model_name] = model_results
                        self.logger.info(f"[OK] {model_name} completed for {timeframe_name}")
                    else:
                        self.logger.warning(f"[FAIL] {model_name} failed for {timeframe_name}")
                
            except Exception as e:
                self.logger.error(f"Error processing timeframe {timeframe_name}: {e}")
                continue
        
        # Generate comparative analysis
        self._generate_comparative_analysis(multi_results, asset_name)
        
        # Store results
        self.results[asset_name] = multi_results
        
        self.logger.info(f"\n[SUCCESS] Multi-timeframe backtest completed for {asset_name}")
        return multi_results
    
    def _create_timeframe_targets(self, data: pd.DataFrame, horizon_days: int, 
                                timeframe_name: str) -> pd.DataFrame:
        """Create target variables for specific timeframe."""
        
        self.logger.info(f"Creating {horizon_days}-day forward targets for {timeframe_name}")
        
        # Copy data
        timeframe_data = data.copy()
        
        # Find returns column
        returns_col = None
        for col in ['returns', 'Returns', 'return', 'log_returns']:
            if col in timeframe_data.columns:
                returns_col = col
                break
        
        if returns_col is None:
            self.logger.error("No returns column found in data")
            return pd.DataFrame()
        
        # Calculate forward-looking returns
        # For horizon_days > 1, we sum the daily log returns over the period
        if horizon_days == 1:
            forward_returns = timeframe_data[returns_col].shift(-1)
        else:
            # Sum log returns over the horizon (this gives cumulative return)
            forward_returns = pd.Series(index=timeframe_data.index, dtype=float)
            for i in range(len(timeframe_data) - horizon_days):
                period_return = timeframe_data[returns_col].iloc[i+1:i+1+horizon_days].sum()
                forward_returns.iloc[i] = period_return
        
        # Remove rows with NaN forward returns
        timeframe_data[f'target_return_{timeframe_name}'] = forward_returns
        timeframe_data = timeframe_data.dropna(subset=[f'target_return_{timeframe_name}'])
        
        # Create target classes using volatility-based thresholds
        return_std = forward_returns.std()
        threshold = 0.5 * return_std  # +/-0.5 sigma thresholds
        
        target_values = pd.Series(1, index=timeframe_data.index, name=f'target_{timeframe_name}')  # Default: hold
        target_values[timeframe_data[f'target_return_{timeframe_name}'] > threshold] = 2   # Buy
        target_values[timeframe_data[f'target_return_{timeframe_name}'] < -threshold] = 0  # Sell
        
        timeframe_data[f'target_{timeframe_name}'] = target_values
        
        self.logger.info(f"Created targets for {timeframe_name}: {len(timeframe_data)} samples")
        self.logger.info(f"Target distribution: {target_values.value_counts().to_dict()}")
        self.logger.info(f"Threshold: +/-{threshold:.4f} (sigma={return_std:.4f})")
        
        return timeframe_data
    
    def _run_single_timeframe_backtest(self, data: pd.DataFrame, model_name: str, 
                                     timeframe_name: str, asset_name: str, 
                                     transaction_cost: float) -> Optional[Dict]:
        """Run backtest for single model and timeframe."""
        
        try:
            # Initialize model trainer
            trainer = ModelTrainer()
            
            # Set target column for this timeframe
            target_col = f'target_{timeframe_name}'
            
            if target_col not in data.columns:
                self.logger.error(f"Target column {target_col} not found")
                return None
            
            # Run expanding window backtest
            self.logger.info(f"Running expanding window backtest for {model_name}...")
            
            # Use the backtest method with specific target column
            results_tuple = trainer.backtest(
                data=data, 
                transaction_cost=transaction_cost,
                asset_name=f"{asset_name}_{timeframe_name}",
                save_results=False  # Don't save intermediate results
            )
            
            if not results_tuple:
                self.logger.warning(f"No results returned")
                return None
                
            results, metrics, shap_values = results_tuple
            
            if not results or model_name not in results:
                self.logger.warning(f"No results for {model_name}")
                return None
            
            model_results = results[model_name]
            model_metrics = metrics.get(model_name, {})
            
            # Enhanced analysis with comprehensive metrics
            if 'returns' not in model_results.columns:
                self.logger.error(f"No returns column found in model results")
                return None
                
            returns_series = model_results['returns']
            
            # Compute comprehensive metrics
            comprehensive_metrics = self.performance_analyzer.compute_comprehensive_metrics(returns_series)
            
            # Generate performance report
            performance_report = self.performance_analyzer.generate_performance_report(returns_series)
            
            # Combine results
            enhanced_results = {
                'basic_metrics': model_metrics,
                'comprehensive_metrics': comprehensive_metrics,
                'performance_report': performance_report,
                'returns': returns_series,
                'signals': model_results.get('signals', pd.Series(0, index=model_results.index)),  # Default to hold signals
                'timeframe': timeframe_name,
                'horizon_days': self.timeframes[timeframe_name],
                'transaction_cost': transaction_cost,
                'model_name': model_name,
                'asset_name': asset_name,
                'shap_values': shap_values.get(model_name, {}) if shap_values else {}
            }
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Error in single timeframe backtest: {e}")
            return None
    
    def _generate_comparative_analysis(self, multi_results: Dict, asset_name: str):
        """Generate comparative analysis across timeframes."""
        
        self.logger.info(f"\n[ANALYSIS] GENERATING COMPARATIVE ANALYSIS FOR {asset_name}")
        self.logger.info("="*70)
        
        # Collect key metrics for comparison
        comparison_data = []
        
        for timeframe, models in multi_results.items():
            for model_name, results in models.items():
                if 'comprehensive_metrics' in results:
                    metrics = results['comprehensive_metrics']
                    comparison_data.append({
                        'timeframe': timeframe,
                        'model': model_name,
                        'annualized_return': metrics.get('annualized_return', 0),
                        'volatility': metrics.get('volatility', 0),
                        'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                        'max_drawdown': metrics.get('max_drawdown', 0),
                        'calmar_ratio': metrics.get('calmar_ratio', 0),
                        'win_rate': metrics.get('win_rate', 0),
                        'profit_factor': metrics.get('profit_factor', 0),
                        'sortino_ratio': metrics.get('sortino_ratio', 0)
                    })
        
        if not comparison_data:
            self.logger.warning("No data for comparative analysis")
            return
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display results
        print(f"\n{asset_name} - MULTI-TIMEFRAME PERFORMANCE COMPARISON")
        print("="*80)
        
        # Best performers by metric
        print("\nBEST PERFORMERS BY METRIC:")
        print("-" * 40)
        
        key_metrics = ['annualized_return', 'sharpe_ratio', 'calmar_ratio', 'win_rate', 'profit_factor']
        
        for metric in key_metrics:
            if metric in comparison_df.columns:
                best_row = comparison_df.loc[comparison_df[metric].idxmax()]
                print(f"{metric.replace('_', ' ').title():20s}: {best_row['timeframe']:3s} {best_row['model']:8s} ({best_row[metric]:6.2f})")
        
        # Display full comparison table
        print(f"\nCOMPLETE COMPARISON TABLE:")
        print("-" * 80)
        print(f"{'Timeframe':>10s} {'Model':>8s} {'Return':>8s} {'Sharpe':>7s} {'Calmar':>7s} {'WinRate':>8s} {'ProfFact':>8s}")
        print("-" * 80)
        
        for _, row in comparison_df.iterrows():
            print(f"{row['timeframe']:>10s} {row['model']:>8s} "
                  f"{row['annualized_return']:>7.1%} {row['sharpe_ratio']:>7.2f} {row['calmar_ratio']:>7.2f} "
                  f"{row['win_rate']:>7.1%} {row['profit_factor']:>8.2f}")
        
        # Save comparison to CSV
        output_path = Path("results") / f"multi_timeframe_comparison_{asset_name}.csv"
        comparison_df.to_csv(output_path, index=False)
        self.logger.info(f"Saved comparison to {output_path}")
    
    def save_results(self, output_dir: str = "results"):
        """Save all multi-timeframe results."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for asset_name, asset_results in self.results.items():
            
            # Save detailed results for each timeframe/model
            for timeframe, models in asset_results.items():
                for model_name, results in models.items():
                    
                    # Save comprehensive metrics
                    metrics_file = output_path / f"metrics_{asset_name}_{timeframe}_{model_name}.json"
                    with open(metrics_file, 'w') as f:
                        # Convert numpy types to Python types for JSON serialization
                        def convert_numpy_types(obj):
                            if isinstance(obj, dict):
                                return {k: convert_numpy_types(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [convert_numpy_types(item) for item in obj]
                            elif hasattr(obj, 'item'):  # numpy scalar
                                return obj.item()
                            else:
                                return obj
                        
                        converted_metrics = convert_numpy_types(results['comprehensive_metrics'])
                        json.dump(converted_metrics, f, indent=2)
                    
                    # Save returns series
                    returns_file = output_path / f"returns_{asset_name}_{timeframe}_{model_name}.csv"
                    results['returns'].to_csv(returns_file)
                    
                    # Save signals if available
                    if 'signals' in results and not results['signals'].empty:
                        signals_file = output_path / f"signals_{asset_name}_{timeframe}_{model_name}.csv"
                        results['signals'].to_csv(signals_file)
                    
                    # Save performance report
                    report_file = output_path / f"report_{asset_name}_{timeframe}_{model_name}.txt"
                    with open(report_file, 'w') as f:
                        f.write(results['performance_report'])
            
            self.logger.info(f"Saved detailed results for {asset_name}")
    
    def get_best_timeframe(self, asset_name: str, metric: str = 'sharpe_ratio') -> Optional[Tuple[str, str]]:
        """
        Get the best performing timeframe and model for an asset.
        
        Args:
            asset_name: Asset name
            metric: Metric to optimize for
            
        Returns:
            Tuple of (timeframe, model_name) or None if not found
        """
        if asset_name not in self.results:
            return None
        
        best_value = float('-inf')
        best_timeframe = None
        best_model = None
        
        for timeframe, models in self.results[asset_name].items():
            for model_name, results in models.items():
                if 'comprehensive_metrics' in results:
                    value = results['comprehensive_metrics'].get(metric, float('-inf'))
                    if value > best_value:
                        best_value = value
                        best_timeframe = timeframe
                        best_model = model_name
        
        return (best_timeframe, best_model) if best_timeframe else None
    
    def generate_multi_timeframe_signals(self, data: pd.DataFrame, asset_name: str) -> Dict[str, Dict]:
        """
        Generate signals for all timeframes simultaneously.
        
        Args:
            data: Current market data
            asset_name: Asset name
            
        Returns:
            Dictionary of signals by timeframe and model
        """
        if asset_name not in self.models:
            self.logger.error(f"No trained models found for {asset_name}")
            return {}
        
        signals = {}
        
        for timeframe in self.timeframes.keys():
            signals[timeframe] = {}
            
            for model_name, model_info in self.models[asset_name].get(timeframe, {}).items():
                try:
                    # Generate signal using the stored model
                    trainer = ModelTrainer()
                    signal_series = trainer.generate_signals(
                        model_info['model'], 
                        data.tail(1), 
                        scaler=model_info.get('scaler'),
                        feature_columns=model_info.get('feature_columns')
                    )
                    
                    if not signal_series.empty:
                        signals[timeframe][model_name] = {
                            'signal': int(signal_series.iloc[0]),
                            'timestamp': datetime.now().isoformat(),
                            'horizon_days': self.timeframes[timeframe]
                        }
                        
                except Exception as e:
                    self.logger.error(f"Error generating signal for {timeframe} {model_name}: {e}")
        
        return signals