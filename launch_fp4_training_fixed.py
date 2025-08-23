#!/usr/bin/env python3
"""
Launch Optimized FP4 FQT Training
==================================

Launch script for the advanced FP4 FQT trainer with all fixes:
- Proper loss scaling and convergence
- Advanced features: MoE + MTP + HRM + Anti-hallucination
- Automatic QAF phase detection
- Memory-efficient training
- Enhanced export system

Usage:
    python launch_fp4_training_fixed.py
"""

import os
import sys
import torch
import logging
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def setup_environment():
    """Setup environment for optimal training"""
    
    # Set environment variables for optimal performance
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA operations
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Enable cuDNN v8 API
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid warnings
    
    # Set PyTorch settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Set optimal thread counts
    torch.set_num_threads(8)  # Optimal for i5-13600K
    
    print("🔧 Environment optimized for FP4 FQT training")

def check_prerequisites():
    """Check system prerequisites"""
    print("🔍 Checking prerequisites...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available! FP4 training requires GPU.")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python: {sys.version.split()[0]}")
    
    # Check PyTorch version
    torch_version = torch.__version__
    if torch.version.cuda is None:
        print("❌ PyTorch CUDA not available")
        return False
    print(f"✅ PyTorch: {torch_version} (CUDA {torch.version.cuda})")
    
    # Check critical imports
    try:
        from transformers import PreTrainedTokenizerFast
        print("✅ Transformers library available")
    except ImportError:
        print("❌ Transformers library not found")
        return False
    
    try:
        from fp4_fqt_core import NVFP4Quantizer
        print("✅ FP4 FQT core available")
    except ImportError:
        print("❌ FP4 FQT core not found")
        return False
    
    try:
        from enhanced_export_system import enhanced_export_after_training
        print("✅ Enhanced export system available")
    except ImportError:
        print("❌ Enhanced export system not found")
        return False
    
    return True

def display_training_info():
    """Display training information"""
    print()
    print("🚀" * 25)
    print("🚀 ADVANCED FP4 FQT TRAINING LAUNCH")  
    print("🚀" * 25)
    print()
    print("📋 FEATURES ENABLED:")
    print("  🔥 FP4 Fully Quantized Training (NVFP4)")
    print("  🧠 Mixture of Experts (MoE)")
    print("  🎯 Multi-Token Prediction (MTP)")
    print("  🧩 Hierarchical Reasoning Module (HRM)")
    print("  🛡️  Anti-Hallucination mechanisms")
    print("  🔄 Automatic QAF phase detection")
    print("  💾 Advanced loss scaling")
    print("  📦 Enhanced model export system")
    print()
    print("📊 TRAINING OPTIMIZATIONS:")
    print("  • Optimized for RTX 3060 Ti (8GB)")
    print("  • ~75% memory reduction vs FP32")
    print("  • 2-4x training speedup expected")
    print("  • Proper convergence with loss scaling")
    print("  • Advanced gradient monitoring")
    print()

def main():
    """Main launch function"""
    print("🎬 Initializing Advanced FP4 FQT Training...")
    
    # Setup environment
    setup_environment()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please install required dependencies.")
        sys.exit(1)
    
    # Display info
    display_training_info()
    
    # Check for existing checkpoints
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pt")):
        checkpoints = list(checkpoint_dir.glob("step_*.pt"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[1]))
            print(f"📂 Found existing checkpoints. Latest: {latest_checkpoint.name}")
            response = input("Continue training from latest checkpoint? [Y/n]: ").strip().lower()
            if response in ['n', 'no']:
                print("🗑️  To start fresh training, remove the 'checkpoints' directory.")
                return
        print()
    
    # Final confirmation
    print("⚠️  IMPORTANT NOTES:")
    print("  • Training will use 4-bit precision (FP4)")
    print("  • Model will automatically switch to QAF phase when gradients stagnate")
    print("  • All models will be automatically exported after training")
    print("  • Training is optimized for your RTX 3060 Ti (8GB)")
    print()
    
    response = input("🚀 Ready to start FP4 FQT training? [Y/n]: ").strip().lower()
    if response in ['n', 'no']:
        print("Training cancelled.")
        return
    
    print("\n" + "🚀" * 50)
    print("🚀 LAUNCHING ADVANCED FP4 FQT TRAINING")
    print("🚀" * 50)
    print()
    
    try:
        # Import and run the optimized trainer
        from trainer_optimized_fixed import train
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('training.log')
            ]
        )
        
        print("📝 Logging enabled: training.log")
        print()
        
        # Start training
        train()
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user.")
        print("💡 Checkpoints are saved regularly. You can resume training later.")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("📝 Check training.log for detailed error information.")
        
        # Print traceback for debugging
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n🧹 GPU memory cleared.")
        print("🎬 Launch script completed.")

if __name__ == "__main__":
    main()
