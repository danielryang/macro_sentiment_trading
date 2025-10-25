"""
Run Pipeline Command

Executes the complete macro sentiment trading pipeline.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cli.commands.base import BaseCommand
from src.main import run_pipeline

logger = logging.getLogger(__name__)


class RunPipelineCommand(BaseCommand):
    """Run the complete trading pipeline."""
    
    def __init__(self, config, args):
        super().__init__(config, args)
        self.start_date = args.start_date
        self.end_date = args.end_date
        self.assets = args.assets
        self.models = args.models
        self.output_dir = args.output_dir
        self.force_refresh = args.force_refresh
        self.method = args.method
        self.skip_news = args.skip_news
        self.skip_sentiment = args.skip_sentiment
        self.skip_market = args.skip_market
        self.skip_training = args.skip_training
    
    def execute(self):
        """Execute the run-pipeline command."""
        try:
            logger.info("=" * 80)
            logger.info("MACRO SENTIMENT TRADING PIPELINE")
            logger.info("=" * 80)
            logger.info(f"Start Date: {self.start_date}")
            logger.info(f"End Date: {self.end_date}")
            logger.info(f"Assets: {self.assets if self.assets else 'All available assets'}")
            logger.info(f"Models: {self.models}")
            logger.info(f"Output Directory: {self.output_dir}")
            logger.info("=" * 80)
            
            # Set up results directory
            results_dir = Path(self.output_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Run the pipeline with error handling
            try:
                # Import and run the main pipeline
                from src.main import run_pipeline
                
                # Determine if we should collect news
                should_collect_news = not self.skip_news
                should_process_headlines = not self.skip_sentiment
                
                logger.info("Starting pipeline execution...")
                
                # Run the pipeline
                result = run_pipeline(
                    start_date=self.start_date,
                    end_date=self.end_date,
                    should_collect_news=should_collect_news,
                    should_process_headlines=should_process_headlines,
                    validate_config=True
                )
                
                if result is None:
                    logger.error("Pipeline execution failed")
                    return 1
                
                logger.info("Pipeline execution completed successfully!")
                return 0
                
            except Exception as e:
                logger.error(f"Pipeline execution failed: {str(e)}")
                logger.error("This may be due to:")
                logger.error("  - Insufficient data for some assets")
                logger.error("  - Date misalignment between market and sentiment data")
                logger.error("  - Missing required columns (target, features)")
                logger.error("  - Data quality issues")
                logger.error("")
                logger.error("Try running with a longer date range or check data quality.")
                return 1
                
        except Exception as e:
            logger.error(f"Command execution failed: {str(e)}")
            return 1