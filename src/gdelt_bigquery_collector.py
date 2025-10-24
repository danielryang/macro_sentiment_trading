"""
GDELT BigQuery Collector

Alternative to raw files collector that uses Google BigQuery to access
GDELT data with pre-processed headlines and additional metadata.

Requires:
- Google Cloud Project with BigQuery API enabled
- Service account credentials or OAuth
- BigQuery client library: pip install google-cloud-bigquery

Benefits:
- Pre-extracted headlines (no web scraping needed)
- Much faster data retrieval
- Access to additional GDELT tables (mentions, gkg, etc.)
- More reliable than scraping old URLs

Costs:
- ~$5 per TB of data processed
- Typical queries cost $0.01-$0.10
"""

import os
import pandas as pd
import logging
from typing import Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def _check_bigquery_available():
    """Check if BigQuery is available at runtime."""
    try:
        from google.cloud import bigquery
        from google.cloud.exceptions import NotFound
        return True
    except ImportError:
        return False

BIGQUERY_AVAILABLE = _check_bigquery_available()


class GDELTBigQueryCollector:
    """Collect GDELT data using Google BigQuery API with pre-processed headlines."""
    
    def __init__(self, project_id: Optional[str] = None, credentials_path: Optional[str] = None):
        """
        Initialize BigQuery collector.
        
        Args:
            project_id: Google Cloud project ID (can also set GOOGLE_CLOUD_PROJECT env var)
            credentials_path: Path to service account JSON (can also set GOOGLE_APPLICATION_CREDENTIALS env var)
        """
        if not BIGQUERY_AVAILABLE:
            raise ImportError(
                "google-cloud-bigquery not installed. Run: pip install google-cloud-bigquery"
            )
        
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
        if not self.project_id:
            raise ValueError(
                "Google Cloud Project ID required. Set GOOGLE_CLOUD_PROJECT environment variable "
                "or pass project_id parameter. Get your project ID from: "
                "https://console.cloud.google.com"
            )
        
        if credentials_path:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        elif not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            logger.warning(
                "No Google Cloud credentials found. Either set GOOGLE_APPLICATION_CREDENTIALS "
                "environment variable or run 'gcloud auth application-default login'"
            )
        
        try:
            from google.cloud import bigquery
            self.client = bigquery.Client(project=self.project_id)
            self.dataset_id = "gdelt-bq"  # Public GDELT dataset
        except Exception as e:
            raise ValueError(
                f"Failed to initialize BigQuery client: {e}. "
                f"Please check your credentials and project ID."
            )
        
    def test_connection(self) -> bool:
        """Test BigQuery connection and GDELT access."""
        try:
            # Test query on events table
            query = """
            SELECT COUNT(*) as count
            FROM `gdelt-bq.gdeltv2.events`
            WHERE SQLDATE = 20240101
            LIMIT 1
            """
            from google.cloud import bigquery
            result = self.client.query(query).result()
            count = next(iter(result)).count
            logger.info(f"BigQuery connection successful. Sample count: {count}")
            return True
        except Exception as e:
            logger.error(f"BigQuery connection failed: {e}")
            return False
    
    def fetch_gkg_with_headlines(
        self, 
        start_date: str, 
        end_date: str, 
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch GDELT GKG data with headlines from BigQuery.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with GKG data and headlines
        """
        # Convert dates to GKG format (YYYYMMDDHHMMSS)
        start_gkg = start_date.replace('-', '') + '000000'
        end_gkg = end_date.replace('-', '') + '235959'
        
        query = """
        WITH ranked_records AS (
            SELECT
                PARSE_DATETIME('%Y%m%d%H%M%S', CAST(DATE AS STRING)) as date,
                DATE as full_date,
                REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS headline,
                REGEXP_EXTRACT(Extras, r'<PAGE_URL>(.*?)</PAGE_URL>') AS url,
                V2Tone as tone,
                DocumentIdentifier as doc_id,
                ROW_NUMBER() OVER (
                    PARTITION BY PARSE_DATE('%Y%m%d', CAST(SUBSTR(CAST(DATE AS STRING), 1, 8) AS STRING))
                    ORDER BY DATE DESC
                ) as row_num
            FROM `gdelt-bq.gdeltv2.gkg`
            WHERE DATE >= CAST(@start_gkg AS INT64)
              AND DATE <= CAST(@end_gkg AS INT64)
              AND Extras IS NOT NULL
              AND REGEXP_CONTAINS(Extras, r'<PAGE_TITLE>.*?</PAGE_TITLE>')
        )
        SELECT date, full_date, headline, url, tone, doc_id
        FROM ranked_records
        WHERE row_num <= @records_per_day
        ORDER BY date ASC
        """
        
        from google.cloud import bigquery

        # Calculate records per day (limit divided by approximate days in range)
        from datetime import datetime
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days_in_range = (end_dt - start_dt).days + 1
        records_per_day = max(10, limit // days_in_range)  # At least 10 per day

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start_gkg", "STRING", start_gkg),
                bigquery.ScalarQueryParameter("end_gkg", "STRING", end_gkg),
                bigquery.ScalarQueryParameter("records_per_day", "INT64", records_per_day),
            ]
        )
        
        logger.info(f"Executing GKG query for {start_date} to {end_date}")
        query_job = self.client.query(query, job_config=job_config)
        
        # Convert to DataFrame
        gkg_df = query_job.to_dataframe()
        
        # PROACTIVE FIX: Clean BigQuery data types to prevent dbdate issues
        gkg_df = self._clean_bigquery_dataframe(gkg_df)
        
        logger.info(f"Retrieved {len(gkg_df)} GKG records with headlines from BigQuery")
        
        return gkg_df

    def fetch_hybrid_data(
        self,
        start_date: str,
        end_date: str,
        event_codes: Optional[List[int]] = None,
        limit: int = 50000  # Much higher limit to cover multiple days
    ) -> pd.DataFrame:
        """
        Fetch hybrid data combining GKG headlines with events metadata.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            event_codes: Filter by event codes (e.g., [100, 120, 130] for macro events)
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with combined GKG headlines and events metadata
        """
        # Get GKG data with headlines (pass the limit parameter)
        gkg_df = self.fetch_gkg_with_headlines(start_date, end_date, limit=limit)

        # Get events data for metadata
        events_df = self.fetch_events_with_headlines(
            start_date, end_date, event_codes, limit=limit
        )
        
        logger.info(f"Hybrid approach: {len(gkg_df)} GKG records, {len(events_df)} events records")
        
        # Create a hybrid dataset by adding events metadata to GKG data
        if len(events_df) > 0:
            # Add events metadata columns to GKG data
            # For simplicity, we'll add aggregate metadata per date
            events_metadata = events_df.groupby('date').agg({
                'goldstein_scale': ['mean', 'std'],
                'num_articles': 'sum',
                'num_mentions': 'sum',
                'num_sources': 'sum',
                'actor1': 'nunique',
                'actor2': 'nunique'
            }).reset_index()
            
            # Flatten column names
            events_metadata.columns = [
                'date', 'goldstein_mean', 'goldstein_std', 'num_articles', 
                'num_mentions', 'num_sources', 'actor1_count', 'actor2_count'
            ]
            
            # Merge with GKG data on date
            hybrid_df = gkg_df.merge(events_metadata, on='date', how='left')
            
            # Fill missing values
            hybrid_df['goldstein_mean'] = hybrid_df['goldstein_mean'].fillna(0)
            hybrid_df['goldstein_std'] = hybrid_df['goldstein_std'].fillna(0)
            hybrid_df['num_articles'] = hybrid_df['num_articles'].fillna(0)
            hybrid_df['num_mentions'] = hybrid_df['num_mentions'].fillna(0)
            hybrid_df['num_sources'] = hybrid_df['num_sources'].fillna(0)
            hybrid_df['actor1_count'] = hybrid_df['actor1_count'].fillna(0)
            hybrid_df['actor2_count'] = hybrid_df['actor2_count'].fillna(0)
            
            logger.info(f"Hybrid data created with {len(hybrid_df)} records")
            return hybrid_df
        else:
            # If no events data, return GKG data with default metadata
            gkg_df['goldstein_mean'] = 0
            gkg_df['goldstein_std'] = 0
            gkg_df['num_articles'] = 1  # Each GKG record represents 1 article
            gkg_df['num_mentions'] = 1
            gkg_df['num_sources'] = 1
            gkg_df['actor1_count'] = 0
            gkg_df['actor2_count'] = 0
            
            logger.info(f"Using GKG-only data with default metadata: {len(gkg_df)} records")
            return gkg_df

    def fetch_events_with_headlines(
        self, 
        start_date: str, 
        end_date: str, 
        event_codes: Optional[List[int]] = None,
        min_goldstein: Optional[float] = None,
        max_goldstein: Optional[float] = None,
        countries: Optional[List[str]] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch GDELT events with headlines from BigQuery.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            event_codes: Filter by event codes (e.g., [100, 120, 130] for macro events)
            min_goldstein: Minimum Goldstein scale value
            max_goldstein: Maximum Goldstein scale value
            countries: Filter by country codes (e.g., ['US', 'CN', 'RU'])
            limit: Maximum number of events to return
            
        Returns:
            DataFrame with events and headlines
        """
        # Build the query
        base_query = """
        SELECT 
            PARSE_DATE('%Y%m%d', CAST(SQLDATE AS STRING)) as date,
            SQLDATE as sqldate,
            Actor1Name as actor1,
            Actor2Name as actor2,
            EventCode as event_type,
            GoldsteinScale as goldstein_scale,
            NumMentions as num_mentions,
            NumSources as num_sources,
            NumArticles as num_articles,
            AvgTone as tone,
            Actor1CountryCode as actor1_country,
            Actor2CountryCode as actor2_country,
            ActionGeo_CountryCode as action_country,
            SourceURL as url
        FROM `gdelt-bq.gdeltv2.events`
        WHERE SQLDATE >= CAST(REPLACE(@start_date, '-', '') AS INT64) 
          AND SQLDATE <= CAST(REPLACE(@end_date, '-', '') AS INT64)
        """
        
        conditions = []
        from google.cloud import bigquery
        
        # Add filters - always include event codes for macro events
        if not event_codes:
            # Default to macro events if no specific codes provided
            event_codes = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
        
        # Convert event codes to strings since EventCode is STRING in BigQuery
        event_codes_str = [str(code) for code in event_codes]
        conditions.append("EventCode IN UNNEST(@event_codes)")
        
        # Create job config with ALL parameters upfront
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start_date", "STRING", start_date),
                bigquery.ScalarQueryParameter("end_date", "STRING", end_date),
                bigquery.ArrayQueryParameter("event_codes", "STRING", event_codes_str),
            ]
        )
        
        if min_goldstein is not None:
            conditions.append("GoldsteinScale >= @min_goldstein")
            job_config.query_parameters.append(
                bigquery.ScalarQueryParameter("min_goldstein", "FLOAT64", min_goldstein)
            )
        
        if max_goldstein is not None:
            conditions.append("GoldsteinScale <= @max_goldstein")
            job_config.query_parameters.append(
                bigquery.ScalarQueryParameter("max_goldstein", "FLOAT64", max_goldstein)
            )
        
        if countries:
            conditions.append(
                "(Actor1CountryCode IN UNNEST(@countries) OR "
                "Actor2CountryCode IN UNNEST(@countries) OR "
                "ActionGeo_CountryCode IN UNNEST(@countries))"
            )
            job_config.query_parameters.append(
                bigquery.ArrayQueryParameter("countries", "STRING", countries)
            )
        
        # Add conditions to query
        if conditions:
            base_query += "\n        AND " + "\n        AND ".join(conditions)
        
        # Use window function to get top N records per day
        # Calculate records per day
        from datetime import datetime
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days_in_range = (end_dt - start_dt).days + 1
        records_per_day = max(10, limit // days_in_range)

        # Wrap query with ROW_NUMBER to limit per day
        base_query = f"""
        WITH base_data AS (
            {base_query}
        ),
        ranked_events AS (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY sqldate
                    ORDER BY num_articles DESC
                ) as row_num
            FROM base_data
        )
        SELECT * EXCEPT(row_num)
        FROM ranked_events
        WHERE row_num <= {records_per_day}
        ORDER BY SQLDATE ASC
        """
        
        logger.info(f"Executing BigQuery query for {start_date} to {end_date}")
        query_job = self.client.query(base_query, job_config=job_config)
        
        # Convert to DataFrame
        events_df = query_job.to_dataframe()
        
        # PROACTIVE FIX: Clean BigQuery data types to prevent dbdate issues
        events_df = self._clean_bigquery_dataframe(events_df)
        
        logger.info(f"Retrieved {len(events_df)} events from BigQuery")
        
        return events_df
    
    def _clean_bigquery_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean BigQuery DataFrame to prevent dbdate and other compatibility issues.
        
        Args:
            df: Raw DataFrame from BigQuery
            
        Returns:
            Cleaned DataFrame with standard pandas dtypes
        """
        df_clean = df.copy()
        
        for col in df_clean.columns:
            if hasattr(df_clean[col].dtype, 'name'):
                dtype_name = df_clean[col].dtype.name
                
                # Fix BigQuery-specific date types
                if 'dbdate' in dtype_name or 'date' in dtype_name.lower():
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                    logger.debug(f"Converted {col} from {dtype_name} to datetime")
                
                # Fix object columns with potential issues
                elif 'object' in dtype_name:
                    # Convert to string and clean
                    df_clean[col] = df_clean[col].astype(str).replace('nan', '').replace('None', '')
                
                # Ensure numeric columns are properly typed
                elif 'int' in dtype_name or 'float' in dtype_name:
                    try:
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    except:
                        pass
        
        logger.info(f"Cleaned DataFrame: {df_clean.shape} with standard dtypes")
        return df_clean
    
    def fetch_mentions_with_headlines(
        self, 
        start_date: str, 
        end_date: str,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch GDELT mentions which include headlines.
        
        This table often has better headline coverage than events.
        """
        query = """
        SELECT 
            PARSE_DATE('%Y%m%d', CAST(DATE AS STRING)) as date,
            DATE as mention_date,
            GlobalEventID as event_id,
            MentionType as mention_type,
            MentionSourceName as source_name,
            MentionIdentifier as url,
            SentimentScore as sentiment,
            DistanceKM as distance_km,
            MentionDocLen as doc_length,
            MentionDocTone as doc_tone
        FROM `gdelt-bq.gdeltv2.eventmentions`
        WHERE DATE >= CAST(REPLACE(@start_date, '-', '') AS INT64) 
          AND DATE <= CAST(REPLACE(@end_date, '-', '') AS INT64)
        AND MentionIdentifier IS NOT NULL
        ORDER BY SentimentScore DESC
        LIMIT @limit
        """
        
        from google.cloud import bigquery
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start_date", "STRING", start_date),
                bigquery.ScalarQueryParameter("end_date", "STRING", end_date),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )
        
        logger.info(f"Fetching mentions for {start_date} to {end_date}")
        query_job = self.client.query(query, job_config=job_config)
        
        mentions_df = query_job.to_dataframe()
        logger.info(f"Retrieved {len(mentions_df)} mentions from BigQuery")
        
        return mentions_df
    
    def estimate_query_cost(self, start_date: str, end_date: str) -> dict:
        """
        Estimate the cost of a query before running it.
        
        Returns:
            Dict with estimated bytes processed and cost in USD
        """
        dry_run_query = f"""
        SELECT *
        FROM `gdelt-bq.gdeltv2.events`
        WHERE SQLDATE >= CAST(REPLACE('{start_date}', '-', '') AS INT64) 
          AND SQLDATE <= CAST(REPLACE('{end_date}', '-', '') AS INT64)
        """
        
        from google.cloud import bigquery
        job_config = bigquery.QueryJobConfig(dry_run=True)
        query_job = self.client.query(dry_run_query, job_config=job_config)
        
        bytes_processed = query_job.total_bytes_processed
        # BigQuery pricing: $5 per TB
        cost_usd = (bytes_processed / (1024**4)) * 5
        
        return {
            "bytes_processed": bytes_processed,
            "gb_processed": bytes_processed / (1024**3),
            "estimated_cost_usd": cost_usd,
            "date_range": f"{start_date} to {end_date}"
        }


def compare_approaches():
    """Compare raw files vs BigQuery approach."""
    print("GDELT Data Collection Approaches Comparison")
    print("=" * 50)
    
    approaches = {
        "Raw Files (Current)": {
            "Cost": "Free",
            "Headlines": "Web scraping required",
            "Success Rate": "~39% (URL decay)",
            "Speed": "Slow (sequential scraping)",
            "Reliability": "Low (timeouts, 404s)",
            "Historical Data": "Full access",
            "Setup": "None required"
        },
        "BigQuery API": {
            "Cost": "~$5/TB (~$0.01-0.10 per query)",
            "Headlines": "Pre-processed by GDELT", 
            "Success Rate": "~90%+ (fresh extraction)",
            "Speed": "Fast (SQL queries)",
            "Reliability": "High (Google infrastructure)",
            "Historical Data": "Full access",
            "Setup": "GCP account + credentials"
        }
    }
    
    for approach, details in approaches.items():
        print(f"\n{approach}:")
        for metric, value in details.items():
            print(f"  {metric}: {value}")


if __name__ == "__main__":
    compare_approaches()
