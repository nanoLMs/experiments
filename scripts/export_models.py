#!/usr/bin/env python3
"""
Export Trained Models
====================
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from enhanced_config import EnhancedTrainConfig, setup_logging

def main():
    config = EnhancedTrainConfig()
    setup_logging(config)

    logging.info("📦 Starting model export")

    # Import and run export system
    try:
        from enhanced_export_system import enhanced_export_after_training

        model_path = "checkpoints/final_model.pt"
        export_dir = config.infrastructure.export_dir

        results = enhanced_export_after_training(model_path, export_dir)

        logging.info("✅ Export completed")
        for format_name, result in results.items():
            if 'error' in result:
                logging.error(f"❌ {format_name}: {result['error']}")
            else:
                logging.info(f"✅ {format_name}: {result.get('path', 'Success')}")

    except ImportError:
        logging.error("❌ Export system not found")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
