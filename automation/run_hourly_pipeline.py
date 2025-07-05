#!/usr/bin/env python3
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.fetch_and_merge_all_data import ComprehensiveDataPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_hourly_pipeline():
    """Run the complete hourly pipeline"""
    try:
        logger.info("="*50)
        logger.info("Starting hourly pipeline run")
        logger.info(f"Time: {datetime.now()}")
        
        # Step 1: Fetch and merge data
        logger.info("Step 1: Fetching and merging data...")
        data_pipeline = ComprehensiveDataPipeline()
        data_pipeline.run_pipeline()
        
        logger.info("Hourly pipeline completed successfully")
        logger.info("="*50)
        
        except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    run_hourly_pipeline()