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
        Compute daily sentiment features strictly following the research math.
        Args:
            sentiment_df: DataFrame with columns ['date', 'polarity', 'goldstein']
        Returns:
            DataFrame with daily features and all required lags, MAs, rolling stds, and sums.
        """
        # Aggregate by date
        grouped = sentiment_df.groupby('date')
        features = []
        for date, group in grouped:
            N_t = len(group)
            S_t = group['polarity'].mean()
            sigma_t = group['polarity'].std(ddof=0)
            V_t = N_t
            logV_t = np.log(1 + N_t)
            AI_t = S_t * logV_t
            if 'goldstein' in group.columns:
                G_t = group['goldstein'].mean()
                sigmaG_t = group['goldstein'].std(ddof=0)
            else:
                G_t = np.nan
                sigmaG_t = np.nan
            features.append({
                'date': date,
                'mean_sentiment': S_t,
                'sentiment_std': sigma_t,
                'news_volume': V_t,
                'log_volume': logV_t,
                'article_impact': AI_t,
                'goldstein_mean': G_t,
                'goldstein_std': sigmaG_t
            })
        df = pd.DataFrame(features)
        df = df.sort_values('date').reset_index(drop=True)

        # Add lags (1,2,3) for all features except date
        for col in ['mean_sentiment', 'sentiment_std', 'news_volume', 'log_volume', 'article_impact', 'goldstein_mean', 'goldstein_std']:
            for lag in [1, 2, 3]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        # Add moving averages (5, 20) for mean_sentiment
        for window in [5, 20]:
            df[f'mean_sentiment_ma_{window}d'] = df['mean_sentiment'].rolling(window).mean()

        # Sentiment acceleration: MA5 - MA20
        df['sentiment_acceleration'] = df['mean_sentiment_ma_5d'] - df['mean_sentiment_ma_20d']

        # Rolling std devs (5, 10) for mean_sentiment
        for window in [5, 10]:
            df[f'mean_sentiment_std_{window}d'] = df['mean_sentiment'].rolling(window).std()

        # Rolling sums (5, 10) for news_volume
        for window in [5, 10]:
            df[f'news_volume_sum_{window}d'] = df['news_volume'].rolling(window).sum()

        return df.dropna().reset_index(drop=True) 