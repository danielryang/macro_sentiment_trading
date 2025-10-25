"""
News Collection Module

This module handles the collection and filtering of news data from the GDELT v2 API.
It focuses on macro-relevant events (EventCode 100-199) and implements the following functionality:

1. Fetch events from GDELT API for specified date ranges
2. Filter for macro-relevant events (policy statements, economic consultations)
3. Rank events by article coverage and retain top 100 per day
4. Handle API rate limiting and error cases
5. Store collected data permanently for reuse

The module uses the GDELT v2 API's document search endpoint to retrieve event records
and their associated metadata, including article URLs and coverage statistics.

Inputs:
    - Date range (start_date, end_date)
    - GDELT v2 API endpoint

Outputs:
    - DataFrame containing filtered events with metadata
    - Article URLs for headline extraction
    - Coverage statistics for ranking

Reference: arXiv:2505.16136v1
"""

import pandas as pd
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
from gdelt import gdelt
import warnings
from bs4 import GuessedAtParserWarning, XMLParsedAsHTMLWarning
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.headline_util import fetch_headline

# Suppress warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GDELTCollector:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.news_dir = os.path.join(data_dir, "news")
        os.makedirs(self.news_dir, exist_ok=True)
        self.client = gdelt(version=2)
        
    def _get_data_path(self, start_date: str, end_date: str) -> str:
        """Get the data file path for a date range."""
        return os.path.join(self.news_dir, f"gdelt_{start_date}_{end_date}.parquet")
        
    def _filter_and_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for macro events and process columns."""
        logger.info(f"Unique eventcode values before filtering: {df.eventcode.unique()}")
        macro_code_prefixes = tuple(str(i).zfill(3) for i in range(100, 200))
        macro_df = df[df.eventcode.str.startswith(macro_code_prefixes, na=False)]
        logger.info(f"Events remaining after macro filter: {len(macro_df):,}")

        macro_df = (
            macro_df.rename(columns={
                "sqldate": "date",
                "actor1name": "actor1",
                "actor2name": "actor2",
                "eventcode": "event_type",
                "goldsteinscale": "goldstein_scale",
                "nummentions": "num_mentions",
                "numsources": "num_sources",
                "numarticles": "num_articles",
                "avgtone": "tone",
                "sourceurl": "url",
            })
            .loc[:, ["date", "actor1", "actor2", "event_type", "goldstein_scale",
                     "num_mentions", "num_sources", "num_articles", "tone", "url"]]
        )
        macro_df["date"] = pd.to_datetime(macro_df["date"], format="%Y%m%d", errors="coerce")
        macro_df = macro_df.dropna(subset=["date"])
        return macro_df

    def _scope_top_events(self, df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
        """Keep only the top_n events per day by num_articles."""
        df = (
            df.sort_values(["date", "num_articles"], ascending=[True, False])
            .groupby("date")
            .head(top_n)
            .reset_index(drop=True)
        )
        logger.info(f"Scoped to top {top_n} events/day → {len(df):,} rows")
        return df

    def _scrape_headlines(self, urls: pd.Series, max_workers: int = 20) -> pd.Series:
        """Fetch headlines in parallel for a series of URLs."""
        headlines = pd.Series([""] * len(urls), index=urls.index)
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            future_to_idx = {exe.submit(fetch_headline, url): idx 
                           for idx, url in enumerate(urls)}
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    headlines.iat[idx] = fut.result()
                except Exception:
                    headlines.iat[idx] = ""
        return headlines

    def fetch_events(self, start_date: str, end_date: str, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch events from GDELT API for the specified date range.
        Stores data permanently for reuse.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            force_refresh: If True, ignore stored data and fetch fresh data
            
        Returns:
            DataFrame containing filtered events
        """
        data_path = self._get_data_path(start_date, end_date)
        
        # Try to load from stored data if available
        if not force_refresh and os.path.exists(data_path):
            logger.info(f"Loading stored data from {data_path}")
            return pd.read_parquet(data_path)
            
        logger.info("Stored data not found. Fetching fresh data...")
        
        # Fetch all events
        logger.info(f"Requesting GDELT events from {start_date} to {end_date}...")
        raw_df = self.client.Search([start_date, end_date], table="events", normcols=True)
        logger.info(f"Fetched {len(raw_df):,} total events.")
        
        # Process and filter events
        proc_df = self._filter_and_process(raw_df)
        
        # Scope to top 100 events per day
        proc_df = self._scope_top_events(proc_df, top_n=100)
        
        # Scrape headlines in monthly batches
        proc_df = proc_df.sort_values("date").reset_index(drop=True)
        headlines = pd.Series("", index=proc_df.index)
        
        # Group by each year-month period
        periods = proc_df["date"].dt.to_period("M").unique()
        for period in periods:
            try:
                # Mask for this year-month
                mask = proc_df["date"].dt.to_period("M") == period
                urls = proc_df.loc[mask, "url"]
                logger.info(f"Starting headline scrape for {period} ({urls.size} URLs)…")
                sub = self._scrape_headlines(urls, max_workers=50)
                headlines.loc[sub.index] = sub
                logger.info(f"Finished headline scrape for {period}")
                
                # Intermediate checkpointing every quarter
                if period.month % 3 == 0:
                    checkpoint_path = Path(f"data/raw/gdelt/gdelt_macro_events_top100_with_headlines_checkpoint_{period}.csv")
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    proc_df.loc[:headlines.notna().sum(), :].to_csv(checkpoint_path, index=False)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                    
            except Exception as e:
                logger.error(f"Month {period} failed: {e}")
                continue
        
        proc_df["headline"] = headlines
        
        # Drop failures
        before = len(proc_df)
        proc_df = proc_df[proc_df["headline"].str.len() > 0].reset_index(drop=True)
        logger.info(f"Dropped {before - len(proc_df):,} rows without headlines → {len(proc_df):,} remain")
        
        # Save data permanently
        logger.info(f"Saving data to {data_path}")
        proc_df.to_parquet(data_path)
        
        return proc_df
            
    def get_daily_events(self, date: str) -> pd.DataFrame:
        """
        Get events for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            DataFrame containing filtered events for the specified date
        """
        return self.fetch_events(date, date) 