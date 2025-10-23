"""
Real-Time Trading System Integration
Complete system for generating live trading recommendations based on current news sentiment.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import logging
from pathlib import Path
import schedule
import time
from typing import Dict, List, Optional

from trading_recommender import TradingRecommender, TradingRecommendation
from advanced_features import AdvancedFeatureEngineer
from data_collector import DataCollector
from sentiment_analyzer import SentimentAnalyzer
from market_processor import MarketProcessor

class RealTimeTradingSystem:
    """Complete real-time trading system with live data feeds and recommendations."""
    
    def __init__(self, models_path: str = "results/models/", 
                 config_path: str = ".env"):
        """Initialize the real-time trading system."""
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/realtime_trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.models_path = Path(models_path)
        self.feature_engineer = AdvancedFeatureEngineer()
        self.data_collector = DataCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.market_processor = MarketProcessor()
        
        # Load trained models
        self.models = self._load_models()
        
        # Initialize recommender
        self.recommender = TradingRecommender(
            models=self.models,
            transaction_costs={'EURUSD': 0.0002, 'USDJPY': 0.0002, 'TNOTE': 0.0005},
            risk_params={
                'max_position_size': 0.20,  # Conservative 20% max position
                'confidence_threshold': 0.65,  # Higher threshold for live trading
                'stop_loss_multiplier': 2.0,
                'take_profit_multiplier': 3.0,
                'min_expected_return': 0.008  # 0.8% minimum expected return
            }
        )
        
        self.logger.info("Real-time trading system initialized successfully")
    
    def _load_models(self) -> Dict[str, Dict[str, object]]:
        """Load trained models from disk."""
        models = {}
        
        for asset in ['EURUSD', 'USDJPY', 'TNOTE']:
            models[asset] = {}
            
            for model_type in ['logistic', 'xgboost']:
                model_file = self.models_path / f"model_{asset}_{model_type}.pkl"
                
                if model_file.exists():
                    try:
                        with open(model_file, 'rb') as f:
                            models[asset][model_type] = pickle.load(f)
                        self.logger.info(f"Loaded {asset} {model_type} model")
                    except Exception as e:
                        self.logger.error(f"Failed to load {asset} {model_type} model: {e}")
                else:
                    self.logger.warning(f"Model file not found: {model_file}")
        
        return models
    
    def collect_current_data(self, lookback_days: int = 30) -> Dict[str, pd.DataFrame]:
        """Collect the latest data for analysis."""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=lookback_days)
        
        self.logger.info(f"Collecting data from {start_date} to {end_date}")
        
        try:
            # Collect news and sentiment data
            news_data = self.data_collector.collect_and_process_news(
                start_date=start_date,
                end_date=end_date
            )
            
            # Process sentiment if not already available
            if 'goldstein_mean' not in news_data.columns:
                sentiment_data = self.sentiment_analyzer.process_sentiment(news_data)
            else:
                sentiment_data = news_data
            
            # Create daily features
            daily_features = self.sentiment_analyzer.create_daily_features(sentiment_data)
            
            # Collect market data
            market_data = {}
            for asset in ['EURUSD=X', 'USDJPY=X', 'ZN=F']:
                market_df = self.market_processor.fetch_market_data(
                    symbol=asset,
                    start_date=start_date,
                    end_date=end_date
                )
                if market_df is not None and not market_df.empty:
                    market_data[asset] = market_df
            
            self.logger.info("Data collection completed successfully")
            
            return {
                'sentiment': daily_features,
                'market': market_data
            }
            
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            return {}
    
    def generate_live_recommendations(self) -> List[TradingRecommendation]:
        """Generate live trading recommendations based on current data."""
        
        self.logger.info("Generating live trading recommendations...")
        
        # Collect current data
        current_data = self.collect_current_data()
        
        if not current_data:
            self.logger.error("No data available for recommendations")
            return []
        
        try:
            # Prepare features for each asset
            recommendations = []
            
            for asset_symbol, asset_name in [('EURUSD=X', 'EURUSD'), 
                                           ('USDJPY=X', 'USDJPY'), 
                                           ('ZN=F', 'TNOTE')]:
                
                if asset_symbol not in current_data['market']:
                    self.logger.warning(f"No market data for {asset_name}")
                    continue
                
                # Align sentiment and market data
                aligned_data = self._align_data_for_prediction(
                    current_data['sentiment'],
                    current_data['market'][asset_symbol],
                    asset_name
                )
                
                if aligned_data is None or len(aligned_data) == 0:
                    self.logger.warning(f"No aligned data for {asset_name}")
                    continue
                
                # Generate recommendation for this asset
                rec = self.recommender._generate_asset_recommendation(
                    asset=asset_name,
                    current_data=aligned_data.tail(1),  # Most recent data point
                    market_data=current_data['market'][asset_symbol],
                    sentiment_data=current_data['sentiment']
                )
                
                if rec:
                    recommendations.append(rec)
                    self.logger.info(f"Generated recommendation for {asset_name}: {rec.direction.value}")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {e}")
            return []
    
    def _align_data_for_prediction(self, sentiment_data: pd.DataFrame,
                                 market_data: pd.DataFrame, 
                                 asset_name: str) -> Optional[pd.DataFrame]:
        """Align sentiment and market data for prediction."""
        
        try:
            # Ensure both dataframes have date columns
            if 'Date' not in market_data.columns and market_data.index.name != 'Date':
                market_data = market_data.reset_index()
            
            # Merge on date
            if 'date' in sentiment_data.columns:
                sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
            
            if 'Date' in market_data.columns:
                market_data['Date'] = pd.to_datetime(market_data['Date'])
                merged = pd.merge(sentiment_data, market_data, 
                                left_on='date', right_on='Date', how='inner')
            else:
                # Handle index-based dates
                market_data.index = pd.to_datetime(market_data.index)
                market_data = market_data.reset_index()
                market_data.rename(columns={'Date': 'date'}, inplace=True)
                merged = pd.merge(sentiment_data, market_data, on='date', how='inner')
            
            if len(merged) == 0:
                return None
            
            # Add enhanced features
            enhanced_data = self.feature_engineer.create_regime_features(merged)
            enhanced_data = self.feature_engineer.create_sentiment_interactions(
                enhanced_data, enhanced_data
            )
            enhanced_data = self.feature_engineer.create_macro_calendar_features(enhanced_data)
            enhanced_data = self.feature_engineer.create_technical_features(enhanced_data)
            
            return enhanced_data
            
        except Exception as e:
            self.logger.error(f"Data alignment failed for {asset_name}: {e}")
            return None
    
    def run_scheduled_analysis(self):
        """Run scheduled analysis and generate recommendations."""
        
        try:
            recommendations = self.generate_live_recommendations()
            
            if recommendations:
                # Format and display recommendations
                formatted_recs = self.recommender.format_recommendations_for_display(recommendations)
                print(formatted_recs)
                
                # Save to file
                filename = self.recommender.save_recommendations_to_file(recommendations)
                self.logger.info(f"Recommendations saved to: {filename}")
                
                # Send alerts for high-confidence recommendations
                self._send_high_priority_alerts(recommendations)
                
            else:
                self.logger.info("No trading recommendations generated")
                print("No trading opportunities identified at this time.")
                
        except Exception as e:
            self.logger.error(f"Scheduled analysis failed: {e}")
    
    def _send_high_priority_alerts(self, recommendations: List[TradingRecommendation]):
        """Send alerts for high-confidence/high-return recommendations."""
        
        high_priority_recs = [
            rec for rec in recommendations 
            if (rec.confidence > 0.8 or abs(rec.predicted_return) > 0.02) 
            and rec.direction.value != "HOLD"
        ]
        
        if high_priority_recs:
            alert_message = " HIGH PRIORITY TRADING ALERTS \n\n"
            
            for rec in high_priority_recs:
                alert_message += f"{rec.asset}: {rec.direction.value}\n"
                alert_message += f"Expected Return: {rec.predicted_return:.2%}\n"
                alert_message += f"Confidence: {rec.confidence:.1%}\n"
                alert_message += f"Risk Level: {rec.risk_level.value}\n\n"
            
            # Log high priority alerts
            self.logger.warning(f"HIGH PRIORITY ALERT: {len(high_priority_recs)} recommendations")
            
            # Here you could integrate with email, Slack, SMS, etc.
            print("=" * 60)
            print(alert_message)
            print("=" * 60)
    
    def start_real_time_monitoring(self, analysis_frequency: str = "4h"):
        """Start real-time monitoring with scheduled analysis."""
        
        self.logger.info(f"Starting real-time monitoring with {analysis_frequency} frequency")
        
        # Schedule analysis
        if analysis_frequency == "1h":
            schedule.every().hour.do(self.run_scheduled_analysis)
        elif analysis_frequency == "4h":
            schedule.every(4).hours.do(self.run_scheduled_analysis)
        elif analysis_frequency == "daily":
            schedule.every().day.at("09:00").do(self.run_scheduled_analysis)
        else:
            self.logger.error(f"Invalid frequency: {analysis_frequency}")
            return
        
        # Run initial analysis
        self.run_scheduled_analysis()
        
        print(f"[LAUNCH] Real-time monitoring started!")
        print(f"Analysis frequency: {analysis_frequency}")
        print("Press Ctrl+C to stop...")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("Real-time monitoring stopped by user")
            print("Real-time monitoring stopped.")
    
    def run_single_analysis(self):
        """Run a single analysis and display results."""
        print("[PROCESS] Running single analysis...")
        self.run_scheduled_analysis()

def main():
    """Main entry point for real-time trading system."""
    
    # Initialize system
    trading_system = RealTimeTradingSystem()
    
    # Command line interface
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "monitor":
            frequency = sys.argv[2] if len(sys.argv) > 2 else "4h"
            trading_system.start_real_time_monitoring(frequency)
        elif command == "analyze":
            trading_system.run_single_analysis()
        else:
            print("Usage: python realtime_trading_system.py [monitor|analyze] [frequency]")
    else:
        # Default: run single analysis
        trading_system.run_single_analysis()

if __name__ == "__main__":
    main()