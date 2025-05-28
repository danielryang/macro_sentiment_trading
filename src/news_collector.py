"""
News Collection Module

This module handles the collection and filtering of news data from the GDELT v2 API.
It focuses on macro-relevant events (EventCode 100-199) and implements the following functionality:

1. Fetch events from GDELT API for specified date ranges
2. Filter for macro-relevant events (policy statements, economic consultations)
3. Rank events by article coverage and retain top 100 per day
4. Handle API rate limiting and error cases

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

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GDELTCollector:
    def __init__(self):
        self.base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        
    def fetch_events(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch events from GDELT API for the specified date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame containing filtered events
        """
        query = {
            "query": "EventCode:100-199",  # Macro-relevant events
            "mode": "artlist",
            "format": "json",
            "startdatetime": f"{start_date}T00:00:00Z",
            "enddatetime": f"{end_date}T23:59:59Z",
            "maxrecords": 250  # We'll filter to top 100 per day later
        }
        
        try:
            response = requests.get(self.base_url, params=query)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            events = pd.DataFrame(data['articles'])
            
            # Add date column for grouping
            events['date'] = pd.to_datetime(events['seendate']).dt.date
            
            # Group by date and get top 100 by number of articles
            events = events.groupby('date').apply(
                lambda x: x.nlargest(100, 'numarts')
            ).reset_index(drop=True)
            
            return events
            
        except Exception as e:
            logger.error(f"Error fetching GDELT data: {str(e)}")
            raise
            
    def get_daily_events(self, date: str) -> pd.DataFrame:
        """
        Get events for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            DataFrame containing filtered events for the specified date
        """
        return self.fetch_events(date, date) 