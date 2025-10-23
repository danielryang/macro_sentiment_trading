"""
Example: Using GDELT BigQuery for Better Headlines

This example shows how to use the BigQuery collector for more reliable
headline extraction compared to the raw files approach.
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

def estimate_costs():
    """Estimate BigQuery costs for typical use cases."""
    print("GDELT BigQuery Cost Estimates")
    print("=" * 40)
    
    # Typical data sizes (rough estimates based on GDELT documentation)
    daily_events_gb = 0.5  # ~500MB per day of events
    monthly_events_gb = daily_events_gb * 30
    yearly_events_gb = daily_events_gb * 365
    
    cost_per_gb = 5 / 1024  # $5 per TB = ~$0.0049 per GB
    
    scenarios = [
        ("1 day of events", daily_events_gb),
        ("1 month of events", monthly_events_gb), 
        ("1 year of events", yearly_events_gb),
        ("Your use case (120 days)", daily_events_gb * 120),
    ]
    
    for scenario, gb_size in scenarios:
        cost = gb_size * cost_per_gb
        print(f"{scenario:20} | {gb_size:6.1f} GB | ${cost:6.3f}")
    
    print("\nNote: Actual costs may vary based on query complexity and filtering")

def setup_instructions():
    """Print setup instructions for BigQuery."""
    print("\nBigQuery Setup Instructions")
    print("=" * 40)
    
    steps = [
        "1. Create Google Cloud Project at https://console.cloud.google.com",
        "2. Enable BigQuery API in your project",
        "3. Create service account with BigQuery User role",
        "4. Download service account JSON key",
        "5. Install: pip install google-cloud-bigquery",
        "6. Set environment variable: GOOGLE_APPLICATION_CREDENTIALS=path/to/key.json"
    ]
    
    for step in steps:
        print(f"   {step}")

def example_usage():
    """Show example usage if BigQuery is available."""
    print("\nExample Usage (if BigQuery is set up)")
    print("=" * 40)
    
    code_example = '''
# Initialize collector
collector = GDELTBigQueryCollector(project_id="your-project-id")

# Test connection
if collector.test_connection():
    # Fetch macro-relevant events with good headlines
    events_df = collector.fetch_events_with_headlines(
        start_date="2015-03-01",
        end_date="2015-03-02", 
        event_codes=[100, 110, 120, 130, 140, 150],  # Macro events
        min_goldstein=-5.0,  # Negative events (conflicts, etc.)
        limit=100
    )
    
    print(f"Retrieved {len(events_df)} events")
    print(f"Headlines available: {events_df['url'].notna().sum()}")
'''
    
    print(code_example)

def compare_results():
    """Compare expected results between approaches."""
    print("\nExpected Results Comparison")
    print("=" * 40)
    
    comparison = {
        "Metric": ["Headlines Success Rate", "Query Speed", "Cost (120 days)", "Reliability"],
        "Raw Files": ["~39%", "~5-10 minutes", "$0", "Low"],
        "BigQuery": ["~90%+", "~10-30 seconds", "~$0.30", "High"]
    }
    
    # Print table
    col_width = 20
    for i, metric in enumerate(comparison["Metric"]):
        print(f"{metric:<{col_width}} | {comparison['Raw Files'][i]:<{col_width}} | {comparison['BigQuery'][i]}")

def main():
    """Run the examples and explanations."""
    print("GDELT BigQuery vs Raw Files Analysis")
    print("=" * 50)
    
    estimate_costs()
    compare_results()
    setup_instructions()
    example_usage()
    
    print("\nRecommendation")
    print("=" * 40)
    print("For production use with historical data (like your 2015 analysis),")
    print("BigQuery is likely worth the small cost (~$0.30) for:")
    print("  • 2-3x better headline extraction rate")
    print("  • 100x faster data retrieval") 
    print("  • Much more reliable pipeline")
    print("  • Access to additional GDELT features")

if __name__ == "__main__":
    main()

