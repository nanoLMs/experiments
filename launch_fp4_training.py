#!/usr/bin/env python3
"""
FP4 FQT Training Launcher
========================

Simple launcher for FP4 Fully Quantized Training with automatic configuration.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are available"""
    required_modules = ['torch', 'transformers']
    missing = []

    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Please install with: pip install torch transformers")
        return False

    return True

def check_fp4_files():
    """Check if FP4 FQT files are present"""
    required_files = [
        'fp4_fqt_core.py',
        'fp4_model_integration.py',
        'trainer_fp4_fqt.py',
        'config.py'
    ]

    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)

    if missing:
        print(f"❌ Missing FP4 files: {', '.join(missing)}")
        return False

    return True

def validate_config():
    """Validate FP4 configuration"""
    try:
        from config import TrainConfig
        cfg = TrainConfig()

        if not getattr(cfg, 'use_pure_4bit', False):
            print("⚠️ FP4 FQT not enabled in config.py")
            print("Set use_pure_4bit = True to enable")
            return False

        print("✅ FP4 FQT configuration validated")
        return True

    except Exception as e:
        print(f"❌ Config validation failed: {e}")
        return False

def run_analysis():
    """Run FP4 analysis and validation"""
    print("🔬 Running FP4 FQT validation...")
    try:
        subprocess.run([sys.executable, 'analyze_fp4_fqt.py'], check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ FP4 validation failed")
        return False
    except FileNotFoundError:
        print("⚠️ analyze_fp4_fqt.py not found, skipping validation")
        return True

def main():
    """Main launcher function"""
    print("🚀 FP4 FQT Training Launcher")
    print("=" * 50)

    # Check dependencies
    if not check_dependencies():
        return 1

    # Check FP4 files
    if not check_fp4_files():
        print("Please ensure all FP4 FQT files are present")
        return 1

    # Validate configuration
    if not validate_config():
        return 1

    # Ask user for options
    print("\nChoose training mode:")
    print("1. Full FP4 FQT Training (recommended)")
    print("2. Validation & Analysis Only")
    print("3. Legacy Optimized Trainer")

    choice = input("Enter choice (1-3): ").strip()

    if choice == "1":
        print("\n🔥 Starting FP4 FQT Training...")
        print("Features enabled:")
        print("  • NVFP4 format (E2M1 + E4M3)")
        print("  • Split rounding strategy")
        print("  • Automatic QAF phase")
        print("  • ~75% memory savings")

        # Optional validation
        if input("\nRun validation first? (y/n): ").lower() == 'y':
            if not run_analysis():
                if input("Continue anyway? (y/n): ").lower() != 'y':
                    return 1

        try:
            subprocess.run([sys.executable, 'trainer_fp4_fqt.py'], check=True)
        except KeyboardInterrupt:
            print("\n⏹️ Training stopped by user")
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
            return 1

    elif choice == "2":
        if not run_analysis():
            return 1
        print("\n✅ Validation complete!")

    elif choice == "3":
        print("\n🔧 Starting legacy optimized trainer...")
        print("Note: Consider upgrading to FP4 FQT for better performance")

        try:
            subprocess.run([sys.executable, 'trainer_optimized.py'], check=True)
        except KeyboardInterrupt:
            print("\n⏹️ Training stopped by user")
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
            return 1

    else:
        print("Invalid choice")
        return 1

    print("\n🎉 Done!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
