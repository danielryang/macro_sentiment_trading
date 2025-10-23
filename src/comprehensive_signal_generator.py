#!/usr/bin/env python3
"""
Comprehensive Signal Generator - Professional Trading Signal Delivery System

This module provides comprehensive trading signals for all trained assets using
the completed 10-year BigQuery trained models. It delivers institutional-grade
signal analysis with confidence scoring, model consensus, and professional formatting.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import logging
from pathlib import Path

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from feature_pipeline import FeatureEngineeringPipeline
from data_collector import DataCollector

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ComprehensiveSignalGenerator:
    """
    Professional signal delivery system for institutional-grade trading signals.

    Provides comprehensive coverage of 67+ trained assets with confidence scoring,
    model consensus analysis, and asset class organization.
    """

    def __init__(self, results_dir: str = "results"):
        """Initialize the signal generator with model directory."""
        self.results_dir = Path(results_dir)
        self.models_dir = self.results_dir / "models"
        self.available_assets = []
        self.models = {}
        self.features = {}
        self.scalers = {}

        # Asset classification for professional reporting
        self.asset_classes = {
            'FX_MAJORS': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD'],
            'FX_CROSSES': ['EURGBP', 'EURJPY', 'GBPJPY', 'GBPCHF', 'AUDCHF', 'EURAUD'],
            'FX_EMERGING': ['USDPLN', 'PLNJPY', 'USDBRL', 'USDKRW', 'USDMXN', 'USDTRY', 'USDSEK', 'USDSGD', 'USDINR', 'USDCNY', 'USDHKD', 'HKDJPY', 'NZDUSD'],
            'EQUITIES_US': ['SPY', 'QQQ', 'IWM', 'SP500', 'NASDAQ', 'DOW', 'RUSSELL'],
            'EQUITIES_INTL': ['EEM', 'EWJ', 'EWY', 'EWZ', 'VWO', 'FXI', 'INDA', 'RSX'],
            'EQUITIES_INDICES': ['CAC', 'DAX', 'FTSE', 'NIKKEI', 'HSI', 'ASX', 'KOSPI', 'SENSEX', 'SHANGHAI', 'TSX'],
            'SECTOR_ETFS': ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY'],
            'COMMODITIES': ['GOLD', 'SILVER', 'COPPER', 'PLATINUM', 'CRUDE', 'NATGAS'],
            'AGRICULTURE': ['CORN', 'WHEAT', 'SOYBEAN', 'SUGAR'],
            'BONDS': ['TNOTE', 'TBOND', 'TYNOTE', 'FVNOTE', 'BUND', 'GILT'],
            'CRYPTO': ['BTCUSD', 'ETHUSD', 'ADAUSD', 'LTCUSD', 'SOLUSD', 'DOGEUSD', 'LINKUSD', 'MATICUSD'],
            'VOLATILITY': ['VIX'],
            'GOLD_RELATED': ['GLD']
        }

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Initialize data collector and feature pipeline
        self.data_collector = DataCollector()
        self.feature_pipeline = FeatureEngineeringPipeline()

    def discover_available_assets(self) -> List[str]:
        """Discover all assets with trained models."""
        assets = set()

        if not self.models_dir.exists():
            self.logger.error(f"Models directory not found: {self.models_dir}")
            return []

        # Find all XGBoost models (each asset should have both XGB and Logistic)
        for model_file in self.models_dir.glob("model_*_xgboost.pkl"):
            asset = model_file.stem.replace("model_", "").replace("_xgboost", "")
            # Verify both models exist
            logistic_file = self.models_dir / f"model_{asset}_logistic.pkl"
            if logistic_file.exists():
                assets.add(asset)

        self.available_assets = sorted(list(assets))
        self.logger.info(f"Discovered {len(self.available_assets)} assets with complete model sets")
        return self.available_assets

    def load_models(self, assets: Optional[List[str]] = None) -> Dict[str, Dict]:
        """Load trained models for specified assets."""
        if assets is None:
            assets = self.available_assets

        loaded_models = {}

        for asset in assets:
            asset_models = {}

            # Load XGBoost model
            xgb_path = self.models_dir / f"model_{asset}_xgboost.pkl"
            if xgb_path.exists():
                try:
                    with open(xgb_path, 'rb') as f:
                        asset_models['xgboost'] = pickle.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to load XGBoost model for {asset}: {e}")
                    continue

            # Load Logistic Regression model
            lr_path = self.models_dir / f"model_{asset}_logistic.pkl"
            if lr_path.exists():
                try:
                    with open(lr_path, 'rb') as f:
                        asset_models['logistic'] = pickle.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to load Logistic model for {asset}: {e}")
                    continue

            # Load features
            features_path = self.models_dir / f"features_{asset}_xgboost.pkl"
            if features_path.exists():
                try:
                    with open(features_path, 'rb') as f:
                        asset_models['features'] = pickle.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to load features for {asset}: {e}")
                    continue

            # Load scalers
            scaler_path = self.models_dir / f"scaler_{asset}_logistic.pkl"
            if scaler_path.exists():
                try:
                    with open(scaler_path, 'rb') as f:
                        asset_models['scaler'] = pickle.load(f)
                except Exception as e:
                    self.logger.warning(f"Failed to load scaler for {asset}: {e}")

            if len(asset_models) >= 3:  # Must have both models and features
                loaded_models[asset] = asset_models
                self.logger.debug(f"Successfully loaded models for {asset}")
            else:
                self.logger.warning(f"Incomplete model set for {asset}, skipping")

        self.models = loaded_models
        self.logger.info(f"Successfully loaded models for {len(loaded_models)} assets")
        return loaded_models

    def collect_recent_data(self, days: int = 30) -> pd.DataFrame:
        """Collect recent market and sentiment data for prediction."""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)

        self.logger.info(f"Collecting data from {start_date} to {end_date}")

        try:
            # Collect data using the existing pipeline
            events_data, daily_features = self.data_collector.collect_data(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )

            if daily_features is None or daily_features.empty:
                self.logger.error("No data collected")
                return pd.DataFrame()

            self.logger.info(f"Collected {len(daily_features)} days of features")
            return daily_features

        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            return pd.DataFrame()

    def generate_signal(self, asset: str, features_data: pd.DataFrame) -> Dict:
        """Generate trading signal for a single asset."""
        if asset not in self.models:
            return {'error': f'No models loaded for {asset}'}

        try:
            # Get the most recent feature values
            if features_data.empty:
                return {'error': 'No features data available'}

            # Use the last available row of features
            recent_features = features_data.iloc[-1:].copy()

            # Get the required features for this asset
            required_features = self.models[asset]['features']

            # Ensure we have all required features
            missing_features = [f for f in required_features if f not in recent_features.columns]
            if missing_features:
                return {'error': f'Missing features: {missing_features[:5]}...'}

            # Extract feature values
            X = recent_features[required_features].values

            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                return {'error': 'Invalid feature values (NaN or Inf)'}

            signals = {}

            # XGBoost prediction
            try:
                xgb_model = self.models[asset]['xgboost']
                xgb_proba = xgb_model.predict_proba(X)[0]
                xgb_pred = np.argmax(xgb_proba)
                xgb_confidence = np.max(xgb_proba) - np.sort(xgb_proba)[-2]

                signals['xgboost'] = {
                    'prediction': int(xgb_pred),
                    'probabilities': xgb_proba.tolist(),
                    'confidence': float(xgb_confidence),
                    'signal': self._map_prediction_to_signal(xgb_pred)
                }
            except Exception as e:
                signals['xgboost'] = {'error': str(e)}

            # Logistic Regression prediction
            try:
                lr_model = self.models[asset]['logistic']
                scaler = self.models[asset].get('scaler')

                X_scaled = X.copy()
                if scaler is not None:
                    X_scaled = scaler.transform(X)

                lr_proba = lr_model.predict_proba(X_scaled)[0]
                lr_pred = np.argmax(lr_proba)
                lr_confidence = np.max(lr_proba) - np.sort(lr_proba)[-2]

                signals['logistic'] = {
                    'prediction': int(lr_pred),
                    'probabilities': lr_proba.tolist(),
                    'confidence': float(lr_confidence),
                    'signal': self._map_prediction_to_signal(lr_pred)
                }
            except Exception as e:
                signals['logistic'] = {'error': str(e)}

            # Calculate consensus
            consensus = self._calculate_consensus(signals)

            return {
                'asset': asset,
                'timestamp': datetime.now().isoformat(),
                'models': signals,
                'consensus': consensus
            }

        except Exception as e:
            return {'error': f'Signal generation failed: {str(e)}'}

    def _map_prediction_to_signal(self, prediction: int) -> str:
        """Map numeric prediction to signal name."""
        mapping = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        return mapping.get(prediction, 'UNKNOWN')

    def _calculate_consensus(self, signals: Dict) -> Dict:
        """Calculate consensus between models."""
        if 'error' in signals.get('xgboost', {}) or 'error' in signals.get('logistic', {}):
            return {'signal': 'ERROR', 'agreement': False, 'avg_confidence': 0.0}

        xgb_signal = signals['xgboost']['signal']
        lr_signal = signals['logistic']['signal']
        xgb_conf = signals['xgboost']['confidence']
        lr_conf = signals['logistic']['confidence']

        agreement = xgb_signal == lr_signal
        avg_confidence = (xgb_conf + lr_conf) / 2

        if agreement:
            consensus_signal = xgb_signal
        else:
            # Use higher confidence model
            consensus_signal = xgb_signal if xgb_conf > lr_conf else lr_signal

        return {
            'signal': consensus_signal,
            'agreement': agreement,
            'avg_confidence': float(avg_confidence),
            'strength': 'HIGH' if avg_confidence > 0.3 else 'MEDIUM' if avg_confidence > 0.15 else 'LOW'
        }

    def generate_all_signals(self) -> Dict[str, Dict]:
        """Generate signals for all available assets."""
        if not self.models:
            self.logger.error("No models loaded. Call load_models() first.")
            return {}

        # Collect recent data
        features_data = self.collect_recent_data()
        if features_data.empty:
            self.logger.error("No features data available for signal generation")
            return {}

        signals = {}
        successful = 0
        failed = 0

        self.logger.info(f"Generating signals for {len(self.models)} assets...")

        for asset in self.models:
            signal = self.generate_signal(asset, features_data)
            signals[asset] = signal

            if 'error' in signal:
                failed += 1
                self.logger.debug(f"Failed to generate signal for {asset}: {signal['error']}")
            else:
                successful += 1

        self.logger.info(f"Signal generation complete: {successful} successful, {failed} failed")
        return signals

    def classify_assets_by_class(self, signals: Dict[str, Dict]) -> Dict[str, List]:
        """Organize assets by asset class."""
        classified = {}

        for class_name, assets in self.asset_classes.items():
            class_signals = []
            for asset in assets:
                if asset in signals and 'error' not in signals[asset]:
                    class_signals.append({
                        'asset': asset,
                        'signal': signals[asset]
                    })

            if class_signals:
                classified[class_name] = class_signals

        # Handle unclassified assets
        all_classified = set()
        for assets in self.asset_classes.values():
            all_classified.update(assets)

        unclassified = []
        for asset in signals:
            if asset not in all_classified and 'error' not in signals[asset]:
                unclassified.append({
                    'asset': asset,
                    'signal': signals[asset]
                })

        if unclassified:
            classified['OTHER'] = unclassified

        return classified

    def format_professional_report(self, signals: Dict[str, Dict]) -> str:
        """Generate professional trading signal report."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')

        # Calculate summary statistics
        total_assets = len(signals)
        successful_signals = sum(1 for s in signals.values() if 'error' not in s)
        failed_signals = total_assets - successful_signals

        signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        high_confidence = []
        model_agreements = []

        for asset, signal in signals.items():
            if 'error' not in signal:
                consensus = signal['consensus']
                signal_counts[consensus['signal']] += 1

                if consensus['strength'] == 'HIGH':
                    high_confidence.append((asset, consensus))

                if consensus['agreement']:
                    model_agreements.append(asset)

        # Classify assets
        classified_assets = self.classify_assets_by_class(signals)

        # Generate report
        report = []
        report.append("=" * 80)
        report.append("MACRO SENTIMENT TRADING SIGNALS - COMPREHENSIVE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {timestamp}")
        report.append(f"Data Source: 10-Year BigQuery Historical Training")
        report.append(f"Models: XGBoost + Logistic Regression Ensemble")
        report.append("")

        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Assets Analyzed: {total_assets}")
        report.append(f"Successful Signals: {successful_signals}")
        report.append(f"Failed Signals: {failed_signals}")
        report.append(f"Success Rate: {successful_signals/total_assets*100:.1f}%")
        report.append("")

        # Signal Distribution
        report.append("SIGNAL DISTRIBUTION")
        report.append("-" * 40)
        for signal_type, count in signal_counts.items():
            pct = count/successful_signals*100 if successful_signals > 0 else 0
            report.append(f"{signal_type}: {count} assets ({pct:.1f}%)")
        report.append("")

        # Model Consensus
        agreement_rate = len(model_agreements)/successful_signals*100 if successful_signals > 0 else 0
        report.append(f"Model Agreement Rate: {agreement_rate:.1f}% ({len(model_agreements)}/{successful_signals})")
        report.append("")

        # High Confidence Signals
        if high_confidence:
            report.append("HIGH CONFIDENCE SIGNALS (Top Conviction)")
            report.append("-" * 40)
            # Sort by confidence
            high_confidence.sort(key=lambda x: x[1]['avg_confidence'], reverse=True)
            for asset, consensus in high_confidence[:10]:  # Top 10
                agreement = "AGREE" if consensus['agreement'] else "SPLIT"
                report.append(f"{asset:10} | {consensus['signal']:4} | {consensus['avg_confidence']:.3f} | {agreement}")
            report.append("")

        # Asset Class Analysis
        report.append("SIGNALS BY ASSET CLASS")
        report.append("=" * 80)

        for class_name, asset_signals in classified_assets.items():
            if not asset_signals:
                continue

            report.append(f"\n{class_name.replace('_', ' ')}")
            report.append("-" * 50)

            # Count signals by type for this class
            class_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            class_details = []

            for item in asset_signals:
                asset = item['asset']
                signal_data = item['signal']
                consensus = signal_data['consensus']

                class_counts[consensus['signal']] += 1

                # Model details
                xgb_signal = signal_data['models'].get('xgboost', {}).get('signal', 'ERROR')
                lr_signal = signal_data['models'].get('logistic', {}).get('signal', 'ERROR')
                agreement = "AGREE" if consensus['agreement'] else "SPLIT"

                class_details.append(f"{asset:12} | {consensus['signal']:4} | XGB:{xgb_signal:4} LR:{lr_signal:4} | {consensus['avg_confidence']:.3f} | {agreement}")

            # Class summary
            total_class = len(asset_signals)
            for signal_type, count in class_counts.items():
                if count > 0:
                    pct = count/total_class*100
                    report.append(f"{signal_type}: {count}/{total_class} ({pct:.0f}%)")

            report.append("")
            report.append("Asset          | Signal | Models      | Conf  | Agreement")
            report.append("-" * 65)
            for detail in sorted(class_details):
                report.append(detail)
            report.append("")

        # Risk Warnings
        report.append("RISK WARNINGS & DISCLAIMERS")
        report.append("-" * 40)
        report.append("• Signals based on historical sentiment-price relationships")
        report.append("• Market conditions may differ from training period (2014-2024)")
        report.append("• Consider position sizing based on confidence levels")
        report.append("• Verify signals against fundamental analysis")
        report.append("• Past performance does not guarantee future results")
        report.append("")

        # Technical Details
        report.append("TECHNICAL DETAILS")
        report.append("-" * 40)
        report.append("• Features: 70+ engineered sentiment and market indicators")
        report.append("• Training: Expanding window backtesting (2-year training, 1-year test)")
        report.append("• Signal Classes: SELL (<-0.5σ), HOLD (±0.5σ), BUY (>+0.5σ)")
        report.append("• Confidence: max(P(class)) - second_max(P(class))")
        report.append("• Data: GDELT news events + Yahoo Finance market data")
        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def save_signals_json(self, signals: Dict[str, Dict], filename: str) -> bool:
        """Save signals to JSON file."""
        try:
            import json

            # Make signals JSON serializable
            json_signals = {}
            for asset, signal in signals.items():
                if 'error' not in signal:
                    json_signals[asset] = signal

            filepath = self.results_dir / filename
            with open(filepath, 'w') as f:
                json.dump(json_signals, f, indent=2, default=str)

            self.logger.info(f"Signals saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save signals: {e}")
            return False

    def get_top_recommendations(self, signals: Dict[str, Dict], top_n: int = 10) -> Dict[str, List]:
        """Get top trading recommendations by signal type."""
        recommendations = {'BUY': [], 'SELL': [], 'HOLD': []}

        for asset, signal in signals.items():
            if 'error' in signal:
                continue

            consensus = signal['consensus']
            signal_type = consensus['signal']
            confidence = consensus['avg_confidence']
            agreement = consensus['agreement']

            recommendations[signal_type].append({
                'asset': asset,
                'confidence': confidence,
                'agreement': agreement,
                'strength': consensus['strength']
            })

        # Sort by confidence and return top N
        for signal_type in recommendations:
            recommendations[signal_type].sort(key=lambda x: x['confidence'], reverse=True)
            recommendations[signal_type] = recommendations[signal_type][:top_n]

        return recommendations


def main():
    """Main function for standalone execution."""
    print("Initializing Comprehensive Signal Generator...")

    # Initialize signal generator
    generator = ComprehensiveSignalGenerator()

    # Discover available assets
    assets = generator.discover_available_assets()
    print(f"Found {len(assets)} assets with trained models")

    if not assets:
        print("No trained models found. Please run the training pipeline first.")
        return

    # Load all models
    print("Loading trained models...")
    models = generator.load_models()
    print(f"Successfully loaded models for {len(models)} assets")

    if not models:
        print("Failed to load any models.")
        return

    # Generate signals
    print("Generating comprehensive trading signals...")
    signals = generator.generate_all_signals()

    if not signals:
        print("Failed to generate any signals.")
        return

    # Generate professional report
    report = generator.format_professional_report(signals)
    print(report)

    # Save signals to JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_filename = f"comprehensive_signals_{timestamp}.json"
    generator.save_signals_json(signals, json_filename)

    # Get top recommendations
    recommendations = generator.get_top_recommendations(signals)

    print("\n" + "="*80)
    print("TOP TRADING RECOMMENDATIONS")
    print("="*80)

    for signal_type, recs in recommendations.items():
        if recs:
            print(f"\nTOP {signal_type} SIGNALS:")
            print("-" * 30)
            for i, rec in enumerate(recs[:5], 1):
                agreement = "[OK]" if rec['agreement'] else "[ERROR]"
                print(f"{i}. {rec['asset']:10} | Conf: {rec['confidence']:.3f} | {rec['strength']:6} | {agreement}")


if __name__ == "__main__":
    main()