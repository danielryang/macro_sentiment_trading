"""
Process Sentiment Command

This command processes sentiment analysis on news data.
"""

from pathlib import Path
from .base import BaseCommand


class ProcessSentimentCommand(BaseCommand):
    """Command to process sentiment analysis."""
    
    def validate_args(self) -> None:
        """Validate command arguments."""
        # Handle both input_path and data_path for compatibility
        input_path = getattr(self.args, 'input_path', None) or getattr(self.args, 'data_path', None)
        if not input_path:
            raise ValueError("Either input_path or data_path must be provided")
        self.check_file_exists(input_path, required=True)
    
    def execute(self) -> int:
        """Execute sentiment processing."""
        try:
            with self.with_logged_operation("Sentiment Processing"):
                import pandas as pd
                
                # Load input data with BigQuery parquet compatibility
                input_path = getattr(self.args, 'input_path', None) or getattr(self.args, 'data_path', None)
                self.logger.info(f"Loading data from: {input_path}")
                
                # Handle BigQuery parquet format with robust reading
                from src.parquet_utils import read_parquet_robust
                events_df = read_parquet_robust(input_path)
                self.log_data_info("Input Events Data", events_df)
                
                # Process sentiment
                method = getattr(self.args, 'method', 'bigquery')  # Default to bigquery
                if method == "bigquery":
                    sentiment_df, daily_features = self._analyze_sentiment_bigquery(events_df)
                else:
                    raise ValueError(f"Unsupported sentiment method: {method}")
                
                self.log_data_info("Sentiment Data", sentiment_df)
                self.log_data_info("Daily Features", daily_features)
                
                # Extract time range from input filename if available
                input_filename = Path(input_path).stem
                time_range = "unknown_period"
                
                # Try to extract time range from filename (events_data_YYYYMMDD_YYYYMMDD)
                if "events_data_" in input_filename:
                    parts = input_filename.split("_")
                    if len(parts) >= 3 and parts[0] == "events" and parts[1] == "data":
                        start_date_str = parts[2]
                        end_date_str = parts[3] if len(parts) > 3 else parts[2]
                        time_range = f"{start_date_str}_{end_date_str}"
                
                # Save results with time range in filename
                sentiment_path = self.get_output_path(f"sentiment_data_{time_range}", ".parquet")
                daily_features_path = self.get_output_path(f"daily_features_{time_range}", ".parquet")
                
                sentiment_df.to_parquet(sentiment_path)
                daily_features.to_parquet(daily_features_path)
                
                self.logger.info(f"Saved sentiment data ({time_range}) to: {sentiment_path}")
                self.logger.info(f"Saved daily features ({time_range}) to: {daily_features_path}")
                
                return 0
                
        except Exception as e:
            self.logger.error(f"Sentiment processing failed: {e}")
            return 1
    
    def _analyze_sentiment_bigquery(self, events_df):
        """Analyze sentiment using BigQuery data with FinBERT on headlines."""
        import numpy as np
        import pandas as pd

        # Use both GDELT tone AND FinBERT analysis on headlines
        self.logger.info(f"Processing sentiment for {len(events_df)} events")
        self.logger.info(f"  - GDELT tone scores (pre-computed)")
        self.logger.info(f"  - FinBERT analysis on {events_df['headline'].notna().sum()} headlines")
        
        # Parse GDELT tone data (complex comma-separated format)
        # Extract the first value from the comma-separated string as the primary tone
        def parse_gdelt_tone(tone_string):
            if pd.isna(tone_string) or tone_string == '':
                return np.nan
            try:
                # Split by comma and take the first value (primary tone)
                tone_values = str(tone_string).split(',')
                return float(tone_values[0])
            except (ValueError, IndexError):
                return np.nan
        
        tone_numeric = events_df['tone'].apply(parse_gdelt_tone)

        # Process headlines through FinBERT (lazy import to avoid slow startup)
        finbert_scores = self._run_finbert_on_headlines(events_df)

        # Convert date column to date only (remove time component) for proper grouping
        events_df['date'] = pd.to_datetime(events_df['date']).dt.date
        
        # Create sentiment DataFrame with BOTH GDELT tone and FinBERT scores
        sentiment_df = pd.DataFrame({
            'date': events_df['date'],
            # GDELT tone (pre-computed from BigQuery)
            'gdelt_tone': tone_numeric / 100.0,  # Normalize to -1 to +1 range
            'gdelt_label': pd.cut(tone_numeric,
                                bins=[-float('inf'), -1, 1, float('inf')],
                                labels=['negative', 'neutral', 'positive']),
            # FinBERT scores (from transformer analysis)
            'finbert_score': finbert_scores['score'],
            'finbert_label': finbert_scores['label'],
            'finbert_positive': finbert_scores['positive'],
            'finbert_negative': finbert_scores['negative'],
            'finbert_neutral': finbert_scores['neutral'],
            # Combined sentiment (average of GDELT and FinBERT)
            'sentiment_score': (tone_numeric / 100.0 + finbert_scores['score']) / 2.0,
            'sentiment_label': finbert_scores['label']  # Use FinBERT label as primary
        })
        
        # Create daily features with BOTH GDELT and FinBERT aggregates
        daily_sentiment = sentiment_df.groupby('date').agg({
            # Combined sentiment (now that GDELT tone is properly parsed)
            'sentiment_score': ['mean', 'std', 'count'],
            # GDELT tone (now properly parsed)
            'gdelt_tone': ['mean', 'std'],
            # FinBERT scores
            'finbert_score': ['mean', 'std'],
            'finbert_positive': 'mean',
            'finbert_negative': 'mean',
            'finbert_neutral': 'mean'
        }).reset_index()

        # Flatten column names
        daily_sentiment.columns = [
            'date',
            'mean_sentiment', 'sentiment_std', 'news_volume',
            'gdelt_tone_mean', 'gdelt_tone_std',
            'finbert_score_mean', 'finbert_score_std',
            'finbert_positive_mean', 'finbert_negative_mean', 'finbert_neutral_mean'
        ]
        daily_sentiment = daily_sentiment.assign(
            sentiment_std=daily_sentiment['sentiment_std'].fillna(0),
            gdelt_tone_std=daily_sentiment['gdelt_tone_std'].fillna(0),
            finbert_score_std=daily_sentiment['finbert_score_std'].fillna(0)
        )
        
        # Add additional features
        daily_sentiment['log_volume'] = np.log1p(daily_sentiment['news_volume'])

        # Add article_impact feature (from research paper: S_t * log(1 + V_t))
        daily_sentiment['article_impact'] = daily_sentiment['mean_sentiment'] * daily_sentiment['log_volume']

        # Add additional features based on available hybrid data
        # Create BOTH simple goldstein features (for model compatibility)
        # AND aggregated features (for richer representation)
        simple_goldstein_cols = {}
        aggregated_cols = {}

        if 'goldstein_mean' in events_df.columns:
            # Simple goldstein_mean (average of per-event goldstein means)
            simple_goldstein_cols['goldstein_mean'] = 'mean'
            # Aggregated goldstein statistics (for advanced features)
            aggregated_cols['goldstein_mean'] = ['mean', 'std']
        if 'goldstein_std' in events_df.columns:
            # Simple goldstein_std (average of per-event goldstein stds)
            simple_goldstein_cols['goldstein_std'] = 'mean'
            # Aggregated goldstein statistics (for advanced features)
            aggregated_cols['goldstein_std'] = ['mean', 'std']
        if 'num_articles' in events_df.columns:
            aggregated_cols['num_articles'] = 'sum'
        if 'tone' in events_df.columns:
            aggregated_cols['tone'] = ['mean', 'std']

        if simple_goldstein_cols or aggregated_cols:
            # Convert to numeric first
            for col in list(simple_goldstein_cols.keys()) + list(aggregated_cols.keys()):
                if col in events_df.columns:
                    events_df[col] = pd.to_numeric(events_df[col], errors='coerce')

            # Create simple goldstein features (model compatibility)
            if simple_goldstein_cols:
                daily_simple = events_df.groupby('date').agg(simple_goldstein_cols).reset_index()
            else:
                daily_simple = pd.DataFrame({'date': events_df.groupby('date').size().index})

            # Create aggregated features (rich representation)
            if aggregated_cols:
                daily_aggregated = events_df.groupby('date').agg(aggregated_cols).reset_index()
                # Flatten column names for aggregated features
                daily_aggregated.columns = ['date'] + [f"{col}_{stat}" if isinstance(stat, str) else f"{col}_{stat[0]}"
                                                     for col, stat in aggregated_cols.items()
                                                     for stat in (stat if isinstance(stat, list) else [stat])]
            else:
                daily_aggregated = pd.DataFrame({'date': events_df.groupby('date').size().index})

            # Merge simple and aggregated features
            daily_additional = daily_simple.merge(daily_aggregated, on='date', how='outer')
        else:
            # Create default values if no additional columns
            daily_additional = events_df.groupby('date').size().reset_index(name='count')
            daily_additional['goldstein_mean'] = 0
            daily_additional['goldstein_std'] = 0
            daily_additional['num_articles'] = daily_additional['count']
        
        # Merge sentiment and additional features
        daily_features = daily_sentiment.merge(daily_additional, on='date', how='outer')
        
        # Add time-based features
        daily_features = daily_features.sort_values('date').reset_index(drop=True)
        
        # Comprehensive lagged features (lags 1,2,3,5 for ALL key features)
        lag_features = ['mean_sentiment', 'sentiment_std', 'news_volume', 'log_volume',
                       'article_impact', 'goldstein_mean', 'goldstein_std']

        for feature in lag_features:
            if feature in daily_features.columns:
                for lag in [1, 2, 3, 5]:
                    daily_features[f'{feature}_lag_{lag}'] = daily_features[feature].shift(lag)

        # Comprehensive moving averages (5d, 20d for key features)
        ma_features = ['mean_sentiment', 'article_impact', 'goldstein_mean', 'goldstein_std']
        for feature in ma_features:
            if feature in daily_features.columns:
                daily_features[f'{feature}_ma_5d'] = daily_features[feature].rolling(5).mean()
                daily_features[f'{feature}_ma_20d'] = daily_features[feature].rolling(20).mean()

        # Comprehensive rolling std devs (5d, 10d for key features)
        std_features = ['mean_sentiment', 'article_impact', 'goldstein_mean', 'goldstein_std']
        for feature in std_features:
            if feature in daily_features.columns:
                daily_features[f'{feature}_std_5d'] = daily_features[feature].rolling(5).std()
                daily_features[f'{feature}_std_10d'] = daily_features[feature].rolling(10).std()

        # Acceleration (second derivative of sentiment)
        daily_features['sentiment_acceleration'] = daily_features['mean_sentiment'].diff().diff()

        # Volume rolling sums
        daily_features['news_volume_sum_5d'] = daily_features['news_volume'].rolling(5).sum()
        daily_features['news_volume_sum_10d'] = daily_features['news_volume'].rolling(10).sum()
        
        self.logger.info(f"Created daily features for {len(daily_features)} days")

        return sentiment_df, daily_features

    def _run_finbert_on_headlines(self, events_df):
        """Run FinBERT sentiment analysis on headlines."""
        import pandas as pd
        import numpy as np

        # Lazy import to avoid slow startup
        from src.sentiment_analyzer import SentimentAnalyzer

        # Get headlines
        headlines = events_df['headline'].fillna('').astype(str)

        # Filter out empty headlines
        non_empty_mask = headlines.str.len() > 0
        non_empty_headlines = headlines[non_empty_mask]

        self.logger.info(f"Running FinBERT on {len(non_empty_headlines)} non-empty headlines...")

        if len(non_empty_headlines) == 0:
            # No headlines, return neutral scores
            return pd.DataFrame({
                'score': np.zeros(len(headlines)),
                'label': ['neutral'] * len(headlines),
                'positive': np.zeros(len(headlines)),
                'negative': np.zeros(len(headlines)),
                'neutral': np.ones(len(headlines))
            })

        # Initialize sentiment analyzer
        analyzer = SentimentAnalyzer()

        # Process all non-empty headlines at once
        finbert_df = analyzer.compute_sentiment(non_empty_headlines.tolist(), batch_size=32)

        self.logger.info(f"FinBERT analysis complete!")

        # Create results dataframe for all headlines (including empty ones)
        results = pd.DataFrame({
            'score': np.zeros(len(headlines)),
            'label': ['neutral'] * len(headlines),
            'positive': np.zeros(len(headlines)),
            'negative': np.zeros(len(headlines)),
            'neutral': np.ones(len(headlines))
        })

        # Fill in scores for non-empty headlines
        for orig_idx, finbert_idx in zip(non_empty_headlines.index, range(len(finbert_df))):
            row = finbert_df.iloc[finbert_idx]
            results.loc[orig_idx, 'score'] = row['polarity']  # Use polarity as score
            results.loc[orig_idx, 'positive'] = row['p_positive']
            results.loc[orig_idx, 'negative'] = row['p_negative']
            results.loc[orig_idx, 'neutral'] = row['p_neutral']

            # Determine label based on highest probability
            if row['p_positive'] > row['p_negative'] and row['p_positive'] > row['p_neutral']:
                results.loc[orig_idx, 'label'] = 'positive'
            elif row['p_negative'] > row['p_positive'] and row['p_negative'] > row['p_neutral']:
                results.loc[orig_idx, 'label'] = 'negative'
            else:
                results.loc[orig_idx, 'label'] = 'neutral'

        return results


