#!/usr/bin/env python3

"""
Multi-Timeframe Backtest Command

Runs comprehensive backtesting across multiple prediction horizons (1D, 1W, 1M, 1Q)
to identify optimal timeframes for trading signals and compare performance.
"""

import sys
import logging
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path  
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cli.commands.base import BaseCommand
from src.multi_timeframe_backtester import MultiTimeframeBacktester
from src.market_processor import MarketProcessor


class MultiTimeframeBacktestCommand(BaseCommand):
    """Command to run multi-timeframe backtesting analysis."""
    
    def __init__(self, config, args):
        super().__init__(config, args)
        self.logger = logging.getLogger(__name__)
        
        # Default assets and models
        self.assets = getattr(args, 'assets', ["EURUSD", "USDJPY", "TNOTE"])
        self.models = getattr(args, 'models', ["logistic", "xgboost"])
        
        # Default to latest models for better performance
        self.best_performance = getattr(args, 'best_performance', True)
        self.performance_metric = getattr(args, 'performance_metric', 'accuracy')
        
        # Date filtering parameters
        self.train_date = getattr(args, 'train_date', None)
        self.training_window = getattr(args, 'training_window', None)
        
        # Timeframes to test
        self.timeframes = {
            "1D": 1,    # Next day
            "1W": 5,    # Next week
            "1M": 21,   # Next month
            "1Q": 63    # Next quarter
        }
        
        # Asset display mapping
        self.asset_display = {
            "EURUSD": "EUR/USD",
            "USDJPY": "USD/JPY", 
            "TNOTE": "Treasury Notes"
        }
    
    def execute(self) -> int:
        """Execute the multi-timeframe backtest command."""
        try:
            self.logger.info("=" * 80)
            self.logger.info("MULTI-TIMEFRAME BACKTESTING ANALYSIS")
            self.logger.info("=" * 80)
            
            # Load data for each asset
            asset_data = self._load_asset_data()
            if not asset_data:
                self.logger.error("No asset data found. Run the pipeline first to generate data.")
                return 1
            
            # Load pre-trained models with selection
            models_info = self._load_pretrained_models()
            if not models_info:
                self.logger.error("No pre-trained models found. Run 'train-models' first.")
                return 1
            
            # Initialize multi-timeframe backtester
            backtester = MultiTimeframeBacktester(timeframes=self.timeframes)
            
            # Run backtests for each asset
            all_results = {}
            for asset in self.assets:
                if asset in asset_data and asset in models_info:
                    self.logger.info(f"\n[START] Starting multi-timeframe analysis for {asset}")
                    
                    # Get available models for this asset
                    available_models = list(models_info[asset].keys())
                    self.logger.info(f"Using models for {asset}: {available_models}")
                    
                    asset_results = backtester.run_multi_timeframe_backtest(
                        data=asset_data[asset],
                        asset_name=asset,
                        models_to_test=available_models
                    )
                    
                    if asset_results:
                        all_results[asset] = asset_results
                        self.logger.info(f"[OK] Completed {asset} analysis")
                    else:
                        self.logger.warning(f"[FAIL] Failed {asset} analysis")
            
            if not all_results:
                self.logger.error("No successful backtests completed")
                return 1
            
            # Save detailed results
            backtester.save_results(output_dir=self.args.output_dir)
            
            # Generate summary report
            self._generate_summary_report(all_results)
            
            # Generate recommendations
            self._generate_recommendations(backtester, all_results)
            
            self.logger.info("Multi-timeframe backtesting completed successfully!")
            return 0
            
        except Exception as e:
            self.logger.error(f"Error in multi-timeframe backtest: {e}", exc_info=True)
            return 1
    
    def _load_asset_data(self) -> Dict[str, pd.DataFrame]:
        """Load aligned data for all assets."""
        asset_data = {}
        
        for asset in self.assets:
            try:
                # Look for aligned data files
                aligned_file = Path(f"results/aligned_data_{asset}.parquet")
                
                if aligned_file.exists():
                    data = pd.read_parquet(aligned_file)
                    
                    # Validate data has required columns
                    required_cols = ['returns', 'target']
                    missing_cols = [col for col in required_cols if col not in data.columns]
                    
                    if missing_cols:
                        self.logger.warning(f"{asset}: Missing columns {missing_cols}")
                        continue
                    
                    if len(data) < 100:  # Minimum data requirement (reduced for testing)
                        self.logger.warning(f"{asset}: Insufficient data ({len(data)} rows)")
                        continue
                    
                    asset_data[asset] = data
                    self.logger.info(f"Loaded {asset} data: {data.shape}")
                    
                else:
                    self.logger.warning(f"No aligned data found for {asset}: {aligned_file}")
                    
            except Exception as e:
                self.logger.error(f"Error loading {asset} data: {e}")
                continue
        
        return asset_data
    
    def _load_pretrained_models(self) -> Dict:
        """Load pre-trained models using advanced selection criteria."""
        from src.model_persistence import ModelPersistence
        
        models_info = {}
        persistence = ModelPersistence()
        
        self.logger.info("Loading pre-trained models with latest model selection...")
        
        for asset in self.assets:
            models_info[asset] = {}
            
            # Get available models for this asset
            available_models = persistence.list_models(asset=asset)
            
            if not available_models:
                self.logger.warning(f"No models found for {asset}")
                continue
            
                # Apply date filtering
                if self.train_date or self.training_window:
                    available_models = self._filter_models_by_date(available_models)
                
                # Apply model type filtering
                if self.models:
                    available_models = [m for m in available_models if m['model_type'] in self.models]
                
                # Apply performance filtering (default to latest/best models)
                if self.best_performance:
                    available_models = self._filter_models_by_performance(available_models)
            
            # Load the selected models
            for model_info in available_models:
                try:
                    model_id = model_info['model_id']
                    model, scaler, feature_columns, metadata = persistence.load_model(model_id)
                    
                    models_info[asset][model_info['model_type']] = {
                        'model': model,
                        'scaler': scaler,
                        'feature_columns': feature_columns,
                        'metadata': metadata,
                        'model_id': model_id
                    }
                    
                    self.logger.info(f"Loaded {asset} {model_info['model_type']} model (ID: {model_id})")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load {asset} {model_info['model_type']} model: {e}")
        
        # Check if we have any models
        total_models = sum(len(asset_models) for asset_models in models_info.values())
        self.logger.info(f"Successfully loaded {total_models} pre-trained models")
        
        return models_info
    
    def _filter_models_by_performance(self, models: List[Dict]) -> List[Dict]:
        """Filter models to keep only the best performing ones."""
        if not models:
            return models
        
        # Group by model type and select best performer for each type
        model_types = {}
        for model in models:
            model_type = model.get('model_type', 'unknown')
            if model_type not in model_types:
                model_types[model_type] = []
            model_types[model_type].append(model)
        
        best_models = []
        for model_type, type_models in model_types.items():
            # Find the best model by performance metric
            best_model = None
            best_score = -1
            
            for model in type_models:
                metrics = model.get('metrics', {})
                score = metrics.get(self.performance_metric, 0)
                
                if score > best_score:
                    best_score = score
                    best_model = model
            
            if best_model:
                best_models.append(best_model)
                self.logger.info(f"Best {model_type} model: {best_model.get('model_id', 'unknown')} "
                               f"({self.performance_metric}={best_score:.3f})")
        
        self.logger.info(f"Performance filtering: {len(models)} -> {len(best_models)} models")
        return best_models
    
    def _filter_models_by_date(self, models: List[Dict]) -> List[Dict]:
        """Filter models by date range using new YYYYMMDD format."""
        from datetime import datetime
        
        filtered_models = []
        
        for model in models:
            if self.train_date:
                # Filter by training date (when model was trained)
                training_date = model.get('training_date', '')
                if not training_date:
                    continue
                    
                try:
                    # Parse training date from model
                    model_date = datetime.fromisoformat(training_date.replace('Z', '+00:00'))
                    model_date_str = model_date.strftime('%Y%m%d')
                    
                    # Parse filter date(s)
                    if len(self.train_date) == 8:  # Single date: YYYYMMDD
                        if model_date_str != self.train_date:
                            continue
                    elif len(self.train_date) == 16:  # Date range: YYYYMMDDYYYYMMDD
                        start_date = self.train_date[:8]
                        end_date = self.train_date[8:]
                        if not (start_date <= model_date_str <= end_date):
                            continue
                    else:
                        self.logger.warning(f"Invalid train-date format: {self.train_date}")
                        continue
                    
                    filtered_models.append(model)
                    
                except ValueError as e:
                    self.logger.warning(f"Invalid date format in model {model.get('model_id', 'unknown')}: {e}")
                    continue
            
            elif self.training_window:
                # Filter by training window (data period model was trained on)
                training_params = model.get('training_params', {})
                start_date = training_params.get('start_date', '')
                end_date = training_params.get('end_date', '')
                
                if not start_date or not end_date:
                    continue
                    
                try:
                    # Parse training window dates
                    model_start = datetime.fromisoformat(start_date)
                    model_end = datetime.fromisoformat(end_date)
                    model_start_str = model_start.strftime('%Y%m%d')
                    model_end_str = model_end.strftime('%Y%m%d')
                    
                    # Parse filter date(s)
                    if len(self.training_window) == 8:  # Single date: YYYYMMDD
                        filter_date = self.training_window
                        # For single date, check if model was trained exactly on that day
                        if model_start_str != filter_date or model_end_str != filter_date:
                            continue
                    elif len(self.training_window) == 16:  # Date range: YYYYMMDDYYYYMMDD
                        filter_start = self.training_window[:8]
                        filter_end = self.training_window[8:]
                        # Check if training windows overlap
                        if not (model_start_str <= filter_end and model_end_str >= filter_start):
                            continue
                    else:
                        self.logger.warning(f"Invalid training-window format: {self.training_window}")
                        continue
                    
                    filtered_models.append(model)
                    
                except ValueError as e:
                    self.logger.warning(f"Invalid date format in model {model.get('model_id', 'unknown')}: {e}")
                    continue
        
        filter_type = "training window" if self.training_window else "training date"
        self.logger.info(f"Date filtering ({filter_type}): {len(models)} -> {len(filtered_models)} models")
        return filtered_models
    
    def _generate_summary_report(self, all_results: Dict):
        """Generate comprehensive summary report."""
        
        print("\n" + "=" * 100)
        print("                         MULTI-TIMEFRAME BACKTESTING SUMMARY")
        print("=" * 100)
        
        # Collect all results for ranking
        performance_data = []
        
        for asset, timeframes in all_results.items():
            for timeframe, models in timeframes.items():
                for model, results in models.items():
                    if 'comprehensive_metrics' in results:
                        metrics = results['comprehensive_metrics']
                        performance_data.append({
                            'asset': asset,
                            'timeframe': timeframe,
                            'model': model,
                            'annualized_return': metrics.get('annualized_return', 0),
                            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                            'calmar_ratio': metrics.get('calmar_ratio', 0),
                            'max_drawdown': metrics.get('max_drawdown', 0),
                            'win_rate': metrics.get('win_rate', 0),
                            'profit_factor': metrics.get('profit_factor', 0)
                        })
        
        if not performance_data:
            print("No performance data available for summary")
            return
        
        df = pd.DataFrame(performance_data)
        
        # Top performers by key metrics
        print("\n[TOP PERFORMERS] BY METRIC:")
        print("-" * 60)
        
        key_metrics = [
            ('sharpe_ratio', 'Sharpe Ratio'),
            ('annualized_return', 'Annualized Return'),
            ('calmar_ratio', 'Calmar Ratio'),
            ('win_rate', 'Win Rate'),
            ('profit_factor', 'Profit Factor')
        ]
        
        for metric_col, metric_name in key_metrics:
            if metric_col in df.columns and not df[metric_col].isna().all():
                top_performer = df.loc[df[metric_col].idxmax()]
                value = top_performer[metric_col]
                
                if metric_col in ['annualized_return', 'win_rate', 'max_drawdown']:
                    value_str = f"{value:.1%}"
                else:
                    value_str = f"{value:.2f}"
                
                print(f"{metric_name:18s}: {top_performer['asset']:7s} {top_performer['timeframe']:3s} "
                      f"{top_performer['model']:8s} ({value_str})")
        
        # Performance matrix by asset and timeframe
        print(f"\n[PERFORMANCE MATRIX] (Sharpe Ratio):")
        print("-" * 80)
        
        # Create pivot table for Sharpe ratios
        sharpe_pivot = df.pivot_table(
            values='sharpe_ratio', 
            index=['asset', 'model'], 
            columns='timeframe',
            aggfunc='mean'
        )
        
        print(sharpe_pivot.round(2))
        
        # Asset-level summary
        print(f"\n[ASSET SUMMARY]:")
        print("-" * 60)
        
        for asset in df['asset'].unique():
            asset_data = df[df['asset'] == asset]
            best_combo = asset_data.loc[asset_data['sharpe_ratio'].idxmax()]
            
            asset_name = self.asset_display.get(asset, asset)
            print(f"{asset_name:15s}: Best {best_combo['timeframe']} {best_combo['model']} "
                  f"(Sharpe: {best_combo['sharpe_ratio']:.2f}, Return: {best_combo['annualized_return']:.1%})")
        
        # Save summary to file
        summary_file = Path(self.args.output_dir) / "multi_timeframe_summary.csv"
        df.to_csv(summary_file, index=False)
        print(f"\n[SAVED] Summary saved to: {summary_file}")
    
    def _generate_recommendations(self, backtester: MultiTimeframeBacktester, all_results: Dict):
        """Generate trading recommendations based on results."""
        
        print("\n" + "=" * 80)
        print("                         TRADING RECOMMENDATIONS")
        print("=" * 80)
        
        recommendations = []
        
        for asset in all_results.keys():
            # Get best timeframe/model combo for this asset
            best_combo = backtester.get_best_timeframe(asset, metric='sharpe_ratio')
            
            if best_combo:
                timeframe, model = best_combo
                
                # Get the actual results
                results = all_results[asset][timeframe][model]
                metrics = results['comprehensive_metrics']
                
                # Determine confidence level based on multiple metrics
                sharpe = metrics.get('sharpe_ratio', 0)
                win_rate = metrics.get('win_rate', 0)
                profit_factor = metrics.get('profit_factor', 0)
                
                confidence = "HIGH" if (sharpe > 1.0 and win_rate > 0.55 and profit_factor > 1.5) else \
                            "MEDIUM" if (sharpe > 0.5 and win_rate > 0.50) else "LOW"
                
                recommendations.append({
                    'asset': asset,
                    'asset_name': self.asset_display.get(asset, asset),
                    'timeframe': timeframe,
                    'model': model,
                    'confidence': confidence,
                    'sharpe_ratio': sharpe,
                    'annualized_return': metrics.get('annualized_return', 0),
                    'win_rate': win_rate,
                    'max_drawdown': metrics.get('max_drawdown', 0)
                })
        
        # Sort by Sharpe ratio
        recommendations.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        
        print("\n[OPTIMAL CONFIGS] TRADING CONFIGURATIONS:")
        print("-" * 80)
        print(f"{'Asset':15s} {'Timeframe':10s} {'Model':10s} {'Confidence':10s} {'Sharpe':>7s} {'Return':>8s} {'WinRate':>8s}")
        print("-" * 80)
        
        for rec in recommendations:
            print(f"{rec['asset_name']:15s} {rec['timeframe']:10s} {rec['model']:10s} "
                  f"{rec['confidence']:10s} {rec['sharpe_ratio']:7.2f} {rec['annualized_return']:7.1%} "
                  f"{rec['win_rate']:7.1%}")
        
        # Usage guidance
        print(f"\n[GUIDANCE] USAGE GUIDANCE:")
        print("-" * 40)
        
        if recommendations:
            best_rec = recommendations[0]
            print(f"[BEST] Best Overall: {best_rec['asset_name']} using {best_rec['timeframe']} {best_rec['model']}")
            
            # Timeframe guidance
            timeframe_counts = {}
            for rec in recommendations:
                tf = rec['timeframe']
                timeframe_counts[tf] = timeframe_counts.get(tf, 0) + 1
            
            most_common_tf = max(timeframe_counts, key=timeframe_counts.get)
            timeframe_names = {"1D": "daily", "1W": "weekly", "1M": "monthly", "1Q": "quarterly"}
            
            print(f"[EFFECTIVE] Most Effective Timeframe: {most_common_tf} ({timeframe_names.get(most_common_tf, 'unknown')} rebalancing)")
            print(f"[NOTE] Longer timeframes generally have lower transaction costs but require more patience")
        
        # Save recommendations
        rec_file = Path(self.args.output_dir) / "trading_recommendations.json"
        import json
        with open(rec_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        print(f"\n[SAVED] Recommendations saved to: {rec_file}")