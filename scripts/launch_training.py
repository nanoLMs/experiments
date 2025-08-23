#!/usr/bin/env python3
"""
Launch Advanced NanoLM Training
==============================
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from enhanced_config import EnhancedTrainConfig, setup_logging
from monitoring_system import create_monitoring_system

def main():
    # Load configuration
    config = EnhancedTrainConfig()
    config.validate()

    # Setup logging
    setup_logging(config)

    # Create monitoring system
    monitoring = create_monitoring_system(config)

    # Print configuration summary
    config.print_summary()

    logging.info("🚀 Starting Advanced NanoLM Training")

    # Import and run trainer
    try:
        from trainer_bnb_quantized import train
        train(config, monitoring)
    except ImportError:
        logging.error("❌ Trainer not found. Please implement trainer_bnb_quantized.py")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
