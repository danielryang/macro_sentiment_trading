"""
Sentiment Analysis Module

This module implements sentiment analysis using the FinBERT model to score news headlines.
It provides the following functionality:

1. Load and initialize FinBERT model and tokenizer
2. Compute sentiment scores for headlines in batches
3. Generate daily sentiment features and statistics
4. Handle GPU acceleration when available

The module uses the ProsusAI/finbert model from HuggingFace to classify headlines
into negative, neutral, and positive categories, and computes a continuous polarity
score for each headline. It also aggregates these scores into daily features.

Inputs:
    - Cleaned headlines from HeadlineProcessor
    - FinBERT model and tokenizer
    - Optional GPU device for acceleration

Outputs:
    - Sentiment probabilities (negative, neutral, positive)
    - Polarity scores (-1 to +1)
    - Daily sentiment features and statistics

Reference: arXiv:2505.16136v1
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.model.to(self.device)
        self.model.eval()
        
    def compute_sentiment(self, headlines: List[str], batch_size: int = 32) -> pd.DataFrame:
        """
        Compute sentiment scores for a list of headlines.
        
        Args:
            headlines: List of headlines to analyze
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with sentiment scores
        """
        results = []
        
        for i in range(0, len(headlines), batch_size):
            batch = headlines[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                
            # Store results
            for j, prob in enumerate(probs):
                results.append({
                    'headline': batch[j],
                    'p_negative': prob[0],
                    'p_neutral': prob[1],
                    'p_positive': prob[2],
                    'polarity': prob[2] - prob[0]  # Ppos - Pneg
                })
                
        return pd.DataFrame(results)
        
    def compute_daily_features(self, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute daily sentiment features.
        
        Args:
            sentiment_df: DataFrame with sentiment scores
            
        Returns:
            DataFrame with daily features
        """
        daily_features = []
        
        for date, group in sentiment_df.groupby('date'):
            features = {
                'date': date,
                'mean_sentiment': group['polarity'].mean(),
                'sentiment_std': group['polarity'].std(),
                'news_volume': len(group),
                'log_volume': np.log(1 + len(group)),
                'article_impact': group['polarity'].mean() * np.log(1 + len(group))
            }
            
            # Add Goldstein features if available
            if 'goldstein' in group.columns:
                features.update({
                    'goldstein_mean': group['goldstein'].mean(),
                    'goldstein_std': group['goldstein'].std()
                })
                
            daily_features.append(features)
            
        return pd.DataFrame(daily_features) 