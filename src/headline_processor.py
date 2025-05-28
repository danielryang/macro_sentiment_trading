"""
Headline Processing Module

This module handles the extraction and cleaning of headlines from news articles.
It implements the following functionality:

1. Extract headlines from article URLs using BeautifulSoup
2. Clean and normalize text (lowercase, remove special characters)
3. Truncate headlines to 512 WordPiece tokens for FinBERT
4. Handle extraction failures and invalid URLs

The module uses BeautifulSoup to parse HTML and extract headlines from either
Open Graph meta tags or standard HTML title tags. It also handles text cleaning
and tokenization to prepare headlines for sentiment analysis.

Inputs:
    - Article URLs from GDELT events
    - FinBERT tokenizer for length validation

Outputs:
    - Cleaned and normalized headlines
    - Tokenized text ready for sentiment analysis
    - Extraction success/failure status

Reference: arXiv:2505.16136v1
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Optional
import logging
from transformers import AutoTokenizer
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeadlineProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        
    def clean_headline(self, headline: str) -> str:
        """
        Clean and normalize a headline.
        
        Args:
            headline: Raw headline text
            
        Returns:
            Cleaned headline
        """
        # Convert to lowercase
        headline = headline.lower()
        
        # Remove special characters and extra whitespace
        headline = re.sub(r'[^\w\s]', ' ', headline)
        headline = re.sub(r'\s+', ' ', headline).strip()
        
        # Truncate to 512 WordPiece tokens
        tokens = self.tokenizer.encode(headline, add_special_tokens=False)
        if len(tokens) > 512:
            headline = self.tokenizer.decode(tokens[:512])
            
        return headline
        
    def extract_headline(self, url: str) -> Optional[str]:
        """
        Extract headline from article URL.
        
        Args:
            url: Article URL
            
        Returns:
            Extracted headline or None if extraction fails
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try Open Graph title first
            og_title = soup.find('meta', property='og:title')
            if og_title:
                return og_title.get('content')
                
            # Fall back to regular title
            title = soup.find('title')
            if title:
                return title.text
                
            return None
            
        except Exception as e:
            logger.warning(f"Failed to extract headline from {url}: {str(e)}")
            return None
            
    def process_articles(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of articles to extract and clean headlines.
        
        Args:
            events_df: DataFrame containing article URLs
            
        Returns:
            DataFrame with added headline column
        """
        headlines = []
        for url in events_df['url']:
            headline = self.extract_headline(url)
            if headline:
                headline = self.clean_headline(headline)
            headlines.append(headline)
            
        events_df['headline'] = headlines
        return events_df.dropna(subset=['headline']) 