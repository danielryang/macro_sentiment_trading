"""
Real-Time Trading Recommendation System
Provides actionable trading recommendations based on sentiment analysis predictions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import logging
from dataclasses import dataclass, asdict
from enum import Enum

class TradeDirection(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

@dataclass
class TradingRecommendation:
    """Structured trading recommendation with all relevant details."""
    asset: str
    direction: TradeDirection
    confidence: float  # 0-1 scale
    predicted_return: float  # Expected return %
    risk_level: RiskLevel
    position_size: float  # Recommended position size (0-1 scale)
    stop_loss: float  # Stop loss level
    take_profit: float  # Take profit level
    holding_period: int  # Recommended holding period in days
    reasoning: List[str]  # Key factors driving the recommendation
    sentiment_score: float  # Current sentiment reading
    model_predictions: Dict[str, float]  # Individual model predictions
    timestamp: datetime
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

class TradingRecommender:
    """Generate actionable trading recommendations from sentiment predictions."""
    
    def __init__(self, models: Dict[str, object], 
                 transaction_costs: Dict[str, float] = None,
                 risk_params: Dict[str, float] = None):
        """
        Initialize the trading recommender.
        
        Args:
            models: Dictionary of trained models for each asset
            transaction_costs: Transaction costs per asset (default: FX=0.0002, Futures=0.0005)
            risk_params: Risk management parameters
        """
        self.models = models
        self.transaction_costs = transaction_costs or {
            'EURUSD': 0.0002, 'USDJPY': 0.0002, 'TNOTE': 0.0005
        }
        self.risk_params = risk_params or {
            'max_position_size': 0.25,  # Maximum 25% of capital per trade
            'confidence_threshold': 0.6,  # Minimum confidence for trading
            'stop_loss_multiplier': 2.0,  # Stop loss = 2x expected volatility
            'take_profit_multiplier': 3.0,  # Take profit = 3x expected volatility
            'min_expected_return': 0.005  # Minimum 0.5% expected return
        }
        self.logger = logging.getLogger(__name__)
        
    def generate_recommendations(self, 
                               current_data: pd.DataFrame,
                               market_data: pd.DataFrame,
                               sentiment_data: pd.DataFrame) -> List[TradingRecommendation]:
        """
        Generate trading recommendations based on current market conditions.
        
        Args:
            current_data: Latest feature data for prediction
            market_data: Current market prices and volatility
            sentiment_data: Latest sentiment readings
            
        Returns:
            List of TradingRecommendation objects
        """
        recommendations = []
        
        for asset in self.models.keys():
            try:
                rec = self._generate_asset_recommendation(
                    asset, current_data, market_data, sentiment_data
                )
                if rec:
                    recommendations.append(rec)
            except Exception as e:
                self.logger.error(f"Error generating recommendation for {asset}: {e}")
                
        return recommendations
    
    def _generate_asset_recommendation(self, 
                                     asset: str,
                                     current_data: pd.DataFrame,
                                     market_data: pd.DataFrame,
                                     sentiment_data: pd.DataFrame) -> Optional[TradingRecommendation]:
        """Generate recommendation for a specific asset."""
        
        # Get model predictions
        model_preds = self._get_model_predictions(asset, current_data)
        if not model_preds:
            return None
            
        # Calculate ensemble prediction
        ensemble_pred, confidence = self._calculate_ensemble_prediction(model_preds)
        
        # Get current market conditions
        current_price = self._get_current_price(asset, market_data)
        volatility = self._get_current_volatility(asset, market_data)
        sentiment_score = self._get_sentiment_score(asset, sentiment_data)
        
        # Calculate expected return and direction
        predicted_return = self._calculate_predicted_return(ensemble_pred, volatility)
        direction = self._determine_trade_direction(predicted_return, confidence)
        
        # Skip if confidence too low or expected return too small
        if (confidence < self.risk_params['confidence_threshold'] or 
            abs(predicted_return) < self.risk_params['min_expected_return']):
            direction = TradeDirection.HOLD
            
        # Calculate position sizing and risk management
        position_size = self._calculate_position_size(predicted_return, volatility, confidence)
        risk_level = self._assess_risk_level(volatility, sentiment_score, confidence)
        
        # Set stop loss and take profit levels
        stop_loss, take_profit = self._calculate_risk_levels(
            current_price, predicted_return, volatility, direction
        )
        
        # Determine holding period based on prediction strength
        holding_period = self._calculate_holding_period(confidence, predicted_return)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            asset, ensemble_pred, sentiment_score, volatility, confidence
        )
        
        return TradingRecommendation(
            asset=asset,
            direction=direction,
            confidence=confidence,
            predicted_return=predicted_return,
            risk_level=risk_level,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            holding_period=holding_period,
            reasoning=reasoning,
            sentiment_score=sentiment_score,
            model_predictions=model_preds,
            timestamp=datetime.now()
        )
    
    def _get_model_predictions(self, asset: str, data: pd.DataFrame) -> Dict[str, float]:
        """Get predictions from all models for the asset."""
        predictions = {}
        
        if asset in self.models:
            model_dict = self.models[asset]
            
            for model_name, model in model_dict.items():
                try:
                    # Get prediction probabilities
                    proba = model.predict_proba(data)
                    
                    # Convert to expected return signal
                    # Assuming class 0=sell, 1=hold, 2=buy
                    if len(proba[0]) == 3:
                        signal = proba[0][2] - proba[0][0]  # Buy prob - Sell prob
                    else:
                        signal = proba[0][1] - 0.5  # Binary case
                        
                    predictions[model_name] = signal
                    
                except Exception as e:
                    self.logger.warning(f"Model {model_name} prediction failed: {e}")
                    
        return predictions
    
    def _calculate_ensemble_prediction(self, model_preds: Dict[str, float]) -> Tuple[float, float]:
        """Calculate ensemble prediction and confidence."""
        if not model_preds:
            return 0.0, 0.0
            
        predictions = list(model_preds.values())
        
        # Ensemble prediction (simple average)
        ensemble_pred = np.mean(predictions)
        
        # Confidence based on agreement between models
        pred_std = np.std(predictions)
        max_std = 1.0  # Maximum possible standard deviation
        confidence = max(0.0, 1.0 - (pred_std / max_std))
        
        return ensemble_pred, confidence
    
    def _calculate_predicted_return(self, ensemble_pred: float, volatility: float) -> float:
        """Convert ensemble prediction to expected return."""
        # Scale prediction by current volatility
        return ensemble_pred * volatility * 5.0  # 5-day horizon scaling
    
    def _determine_trade_direction(self, predicted_return: float, confidence: float) -> TradeDirection:
        """Determine trade direction based on predicted return and confidence."""
        abs_return = abs(predicted_return)
        
        if abs_return < self.risk_params['min_expected_return']:
            return TradeDirection.HOLD
        elif predicted_return > 0:
            if confidence > 0.8 and abs_return > 0.02:
                return TradeDirection.STRONG_BUY
            else:
                return TradeDirection.BUY
        else:
            if confidence > 0.8 and abs_return > 0.02:
                return TradeDirection.STRONG_SELL
            else:
                return TradeDirection.SELL
    
    def _calculate_position_size(self, predicted_return: float, 
                               volatility: float, confidence: float) -> float:
        """Calculate optimal position size using Kelly criterion-inspired approach."""
        # Kelly fraction: f = (bp - q) / b
        # where b = odds, p = prob of win, q = prob of loss
        
        win_prob = 0.5 + (confidence - 0.5)  # Adjust based on confidence
        expected_return = abs(predicted_return)
        
        # Risk-adjusted position sizing
        kelly_fraction = (win_prob * expected_return - (1 - win_prob) * expected_return) / expected_return
        kelly_fraction = max(0, kelly_fraction)  # No negative positions
        
        # Scale by confidence and cap at maximum
        position_size = kelly_fraction * confidence * 0.5  # Conservative scaling
        position_size = min(position_size, self.risk_params['max_position_size'])
        
        return position_size
    
    def _assess_risk_level(self, volatility: float, sentiment_score: float, 
                          confidence: float) -> RiskLevel:
        """Assess the risk level of the trade."""
        # High volatility = higher risk
        vol_risk = volatility > 0.02  # 2% daily volatility threshold
        
        # Extreme sentiment readings = higher risk
        sentiment_risk = abs(sentiment_score) > 0.8
        
        # Low confidence = higher risk  
        confidence_risk = confidence < 0.7
        
        risk_factors = sum([vol_risk, sentiment_risk, confidence_risk])
        
        if risk_factors >= 3:
            return RiskLevel.EXTREME
        elif risk_factors == 2:
            return RiskLevel.HIGH
        elif risk_factors == 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _calculate_risk_levels(self, current_price: float, predicted_return: float,
                             volatility: float, direction: TradeDirection) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        
        # Stop loss: 2x daily volatility
        stop_distance = volatility * self.risk_params['stop_loss_multiplier']
        
        # Take profit: 3x daily volatility or predicted return, whichever is larger
        profit_distance = max(
            volatility * self.risk_params['take_profit_multiplier'],
            abs(predicted_return)
        )
        
        if direction in [TradeDirection.BUY, TradeDirection.STRONG_BUY]:
            stop_loss = current_price * (1 - stop_distance)
            take_profit = current_price * (1 + profit_distance)
        elif direction in [TradeDirection.SELL, TradeDirection.STRONG_SELL]:
            stop_loss = current_price * (1 + stop_distance)
            take_profit = current_price * (1 - profit_distance)
        else:  # HOLD
            stop_loss = current_price
            take_profit = current_price
            
        return stop_loss, take_profit
    
    def _calculate_holding_period(self, confidence: float, predicted_return: float) -> int:
        """Calculate recommended holding period in days."""
        base_period = 5  # Base 5-day holding period
        
        # Longer holding for higher confidence
        confidence_multiplier = 1 + confidence
        
        # Shorter holding for larger predicted moves (faster resolution)
        return_multiplier = max(0.5, 1 - abs(predicted_return) * 10)
        
        holding_period = int(base_period * confidence_multiplier * return_multiplier)
        return max(1, min(holding_period, 20))  # Cap between 1-20 days
    
    def _generate_reasoning(self, asset: str, ensemble_pred: float, 
                          sentiment_score: float, volatility: float, 
                          confidence: float) -> List[str]:
        """Generate human-readable reasoning for the recommendation."""
        reasoning = []
        
        # Model prediction reasoning
        if ensemble_pred > 0.3:
            reasoning.append(f"Strong bullish signal from sentiment models (score: {ensemble_pred:.2f})")
        elif ensemble_pred > 0.1:
            reasoning.append(f"Moderate bullish signal from sentiment models (score: {ensemble_pred:.2f})")
        elif ensemble_pred < -0.3:
            reasoning.append(f"Strong bearish signal from sentiment models (score: {ensemble_pred:.2f})")
        elif ensemble_pred < -0.1:
            reasoning.append(f"Moderate bearish signal from sentiment models (score: {ensemble_pred:.2f})")
        else:
            reasoning.append(f"Neutral signal from sentiment models (score: {ensemble_pred:.2f})")
        
        # Sentiment reasoning
        if sentiment_score > 0.5:
            reasoning.append(f"Positive news sentiment supporting bullish outlook ({sentiment_score:.2f})")
        elif sentiment_score < -0.5:
            reasoning.append(f"Negative news sentiment supporting bearish outlook ({sentiment_score:.2f})")
        else:
            reasoning.append(f"Neutral news sentiment ({sentiment_score:.2f})")
        
        # Volatility reasoning
        if volatility > 0.025:
            reasoning.append(f"High volatility environment ({volatility:.1%}) - increased risk")
        elif volatility < 0.01:
            reasoning.append(f"Low volatility environment ({volatility:.1%}) - reduced opportunity")
        else:
            reasoning.append(f"Normal volatility environment ({volatility:.1%})")
        
        # Confidence reasoning
        if confidence > 0.8:
            reasoning.append(f"High model confidence ({confidence:.1%}) - strong signal reliability")
        elif confidence < 0.6:
            reasoning.append(f"Low model confidence ({confidence:.1%}) - proceed with caution")
        else:
            reasoning.append(f"Moderate model confidence ({confidence:.1%})")
        
        return reasoning
    
    def _get_current_price(self, asset: str, market_data: pd.DataFrame) -> float:
        """Get current price for the asset."""
        if asset in market_data.columns:
            return market_data[asset].iloc[-1] if len(market_data) > 0 else 1.0
        return 1.0
    
    def _get_current_volatility(self, asset: str, market_data: pd.DataFrame) -> float:
        """Get current volatility for the asset."""
        vol_col = f'{asset}_vol20'
        if vol_col in market_data.columns:
            return market_data[vol_col].iloc[-1] if len(market_data) > 0 else 0.015
        return 0.015  # Default 1.5% daily volatility
    
    def _get_sentiment_score(self, asset: str, sentiment_data: pd.DataFrame) -> float:
        """Get current sentiment score."""
        if 'goldstein_mean' in sentiment_data.columns:
            return sentiment_data['goldstein_mean'].iloc[-1] if len(sentiment_data) > 0 else 0.0
        return 0.0
    
    def format_recommendations_for_display(self, 
                                         recommendations: List[TradingRecommendation]) -> str:
        """Format recommendations for human-readable display."""
        if not recommendations:
            return "No trading recommendations at this time."
        
        output = []
        output.append("=" * 80)
        output.append("[CHART] MACRO SENTIMENT TRADING RECOMMENDATIONS")
        output.append("=" * 80)
        output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("")
        
        for i, rec in enumerate(recommendations, 1):
            output.append(f"{i}. {rec.asset} - {rec.direction.value}")
            output.append(f"   Confidence: {rec.confidence:.1%} | Risk: {rec.risk_level.value}")
            output.append(f"   Expected Return: {rec.predicted_return:.2%}")
            output.append(f"   Position Size: {rec.position_size:.1%} of capital")
            
            if rec.direction != TradeDirection.HOLD:
                output.append(f"   Stop Loss: {rec.stop_loss:.4f} | Take Profit: {rec.take_profit:.4f}")
                output.append(f"   Holding Period: {rec.holding_period} days")
            
            output.append("   Key Factors:")
            for reason in rec.reasoning:
                output.append(f"   • {reason}")
            output.append("")
        
        return "\n".join(output)
    
    def save_recommendations_to_file(self, 
                                   recommendations: List[TradingRecommendation],
                                   filename: str = None) -> str:
        """Save recommendations to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"trading_recommendations_{timestamp}.json"
        
        # Convert recommendations to dict format
        rec_dicts = [rec.to_dict() for rec in recommendations]
        
        # Handle datetime serialization
        for rec_dict in rec_dicts:
            rec_dict['timestamp'] = rec_dict['timestamp'].isoformat()
            rec_dict['direction'] = rec_dict['direction'].value
            rec_dict['risk_level'] = rec_dict['risk_level'].value
        
        with open(filename, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'recommendations': rec_dicts
            }, f, indent=2)
        
        return filename