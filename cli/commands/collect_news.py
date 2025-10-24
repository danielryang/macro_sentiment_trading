"""
Collect News Command

This command collects news data from GDELT.
"""

from .base import BaseCommand


class CollectNewsCommand(BaseCommand):
    """Command to collect news data from GDELT."""
    
    def validate_args(self) -> None:
        """Validate command arguments."""
        # Validate date range
        from datetime import datetime
        try:
            start_date = datetime.strptime(self.args.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(self.args.end_date, "%Y-%m-%d")
            
            if start_date >= end_date:
                raise ValueError("Start date must be before end date")
            
            if end_date > datetime.now():
                raise ValueError("End date cannot be in the future")
                
        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}")
    
    def execute(self) -> int:
        """Execute news collection."""
        try:
            with self.with_logged_operation("News Collection"):
                # Import here to avoid circular imports
                from src.data_collector import collect_and_process_news
                
                self.logger.info(f"Collecting news from {self.args.start_date} to {self.args.end_date}")
                
                # Get method from arguments
                method = getattr(self.args, 'method', 'bigquery')
                self.logger.info(f"Using method: {method}")
                
                events_df = collect_and_process_news(
                    start_date=self.args.start_date,
                    end_date=self.args.end_date,
                    force_refresh=self.args.force_refresh,
                    use_method=method
                )
                
                self.log_data_info("Events Data", events_df)
                
                # Save events data with time range in filename
                start_date_str = self.args.start_date.replace("-", "")
                end_date_str = self.args.end_date.replace("-", "")
                output_path = self.get_output_path(f"events_data_{start_date_str}_{end_date_str}", ".parquet")
                events_df.to_parquet(output_path)
                self.logger.info(f"Saved events data ({self.args.start_date} to {self.args.end_date}) to: {output_path}")
                
                return 0
                
        except Exception as e:
            self.logger.error(f"News collection failed: {e}", exc_info=True)
            return 1


