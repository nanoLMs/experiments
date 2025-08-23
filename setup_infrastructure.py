#!/usr/bin/env python3
"""
Infrastructure Setup Script for Advanced NanoLM
===============================================

Sets up the complete project infrastructure including:
- Directory structure
- Configuration validation
- Logging system
- Monitoring system
- Dependencies check
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Any

# Import our enhanced configuration
from enhanced_config import EnhancedTrainConfig, setup_logging, create_default_config
from monitoring_system import create_monitoring_system


def check_dependencies() -> Dict[str, bool]:
    """Check if required dependencies are available"""
    dependencies = {
        'torch': False,
        'transformers': False,
        'bitsandbytes': False,
        'flash_attn': False,
        'rich': False,
        'matplotlib': False,
        'scipy': False,
        'pynvml': False,
        'psutil': False
    }

    for dep in dependencies:
        try:
            if dep == 'flash_attn':
                import flash_attn
            elif dep == 'pynvml':
                import pynvml
            else:
                __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False

    return dependencies


def install_missing_dependencies(missing: List[str]) -> bool:
    """Install missing dependencies"""
    if not missing:
        return True

    print(f"📦 Installing missing dependencies: {', '.join(missing)}")

    # Map package names to pip install names
    pip_names = {
        'torch': 'torch',
        'transformers': 'transformers',
        'bitsandbytes': 'bitsandbytes',
        'flash_attn': 'flash-attn --no-build-isolation',
        'rich': 'rich',
        'matplotlib': 'matplotlib',
        'scipy': 'scipy',
        'pynvml': 'pynvml',
        'psutil': 'psutil'
    }

    for dep in missing:
        if dep in pip_names:
            try:
                cmd = f"{sys.executable} -m pip install {pip_names[dep]}"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ Successfully installed {dep}")
                else:
                    print(f"❌ Failed to install {dep}: {result.stderr}")
                    return False
            except Exception as e:
                print(f"❌ Error installing {dep}: {e}")
                return False

    return True


def create_directory_structure(base_dir: Path = Path(".")) -> Dict[str, Path]:
    """Create project directory structure"""
    directories = {
        'checkpoints': base_dir / 'checkpoints',
        'loss_tracking': base_dir / 'loss_tracking',
        'monitoring': base_dir / 'monitoring',
        'exported_models': base_dir / 'exported_models',
        'logs': base_dir / 'logs',
        'configs': base_dir / 'configs',
        'data': base_dir / 'data',
        'tests': base_dir / 'tests',
        'profiling': base_dir / 'profiling',
        'debug': base_dir / 'debug',
        'docs': base_dir / 'docs',
        'scripts': base_dir / 'scripts'
    }

    created = []
    for name, path in directories.items():
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created.append(name)

    if created:
        print(f"📁 Created directories: {', '.join(created)}")

    return directories


def setup_git_ignore(base_dir: Path = Path(".")):
    """Setup .gitignore file"""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyTorch
*.pth
*.pt
*.bin
*.safetensors

# Training outputs
checkpoints/
loss_tracking/
monitoring/
exported_models/
logs/
profiling/
debug/
*.log

# Data
data/
*.txt
*.json
*.csv

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Environment
.env
.venv/
env/
venv/
"""

    gitignore_path = base_dir / '.gitignore'
    if not gitignore_path.exists():
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content.strip())
        print("📝 Created .gitignore file")


def validate_system_requirements() -> Dict[str, Any]:
    """Validate system requirements"""
    requirements = {
        'python_version': sys.version_info >= (3, 8),
        'cuda_available': False,
        'gpu_memory_gb': 0,
        'system_memory_gb': 0,
        'disk_space_gb': 0
    }

    # Check CUDA
    try:
        import torch
        requirements['cuda_available'] = torch.cuda.is_available()
        if requirements['cuda_available']:
            requirements['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except ImportError:
        pass

    # Check system memory
    try:
        import psutil
        requirements['system_memory_gb'] = psutil.virtual_memory().total / (1024**3)
        requirements['disk_space_gb'] = psutil.disk_usage('.').free / (1024**3)
    except ImportError:
        pass

    return requirements


def create_example_configs(config_dir: Path):
    """Create example configuration files"""
    # Default config
    default_config = create_default_config()
    default_config.save_config(config_dir / 'default_config.json')

    # Small model config (for testing)
    small_config = create_default_config()
    small_config.n_layers = 8
    small_config.n_heads = 4
    small_config.d_model = 256
    small_config.d_ff = 1024
    small_config.moe_every = 0  # Disable MoE for small model
    small_config.save_config(config_dir / 'small_model_config.json')

    # Large model config (for production)
    large_config = create_default_config()
    large_config.n_layers = 24
    large_config.n_heads = 12
    large_config.d_model = 512
    large_config.d_ff = 2048
    large_config.moe_every = 2
    large_config.n_experts = 8
    large_config.save_config(config_dir / 'large_model_config.json')

    print("📋 Created example configuration files")


def create_launch_scripts(scripts_dir: Path):
    """Create launch scripts"""

    # Training launch script
    train_script = """#!/usr/bin/env python3
\"\"\"
Launch Advanced NanoLM Training
==============================
\"\"\"

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
"""

    with open(scripts_dir / 'launch_training.py', 'w') as f:
        f.write(train_script)

    # Make executable
    os.chmod(scripts_dir / 'launch_training.py', 0o755)

    # Export script
    export_script = """#!/usr/bin/env python3
\"\"\"
Export Trained Models
====================
\"\"\"

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
"""

    with open(scripts_dir / 'export_models.py', 'w') as f:
        f.write(export_script)

    os.chmod(scripts_dir / 'export_models.py', 0o755)

    print("🚀 Created launch scripts")


def create_readme(base_dir: Path):
    """Create comprehensive README"""
    readme_content = """# Advanced NanoLM System

A lightweight, edge-optimized language model with advanced features:

- **Mixture of Experts (MoE)** - Efficient expert routing
- **Multi-Token Prediction (MTP)** - Predict multiple future tokens
- **Hierarchical Reasoning (HRM)** - Multi-timescale processing
- **Anti-Hallucination** - Factual consistency mechanisms
- **Advanced Quantization** - NF4 training, FP4 fine-tuning
- **Comprehensive Monitoring** - Real-time training analytics

## Quick Start

1. **Setup Environment**:
   ```bash
   python setup_infrastructure.py
   ```

2. **Configure Training**:
   ```bash
   # Edit configs/default_config.json or create your own
   ```

3. **Start Training**:
   ```bash
   python scripts/launch_training.py
   ```

4. **Export Models**:
   ```bash
   python scripts/export_models.py
   ```

## Project Structure

```
├── configs/                 # Configuration files
├── scripts/                 # Launch scripts
├── checkpoints/            # Model checkpoints
├── exported_models/        # Exported model formats
├── loss_tracking/          # Training metrics
├── monitoring/             # System monitoring data
├── logs/                   # Log files
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

## Features

### Model Architecture
- **Size**: 100-150MB (quantized)
- **Parameters**: ~30M (configurable)
- **Quantization**: NF4/FP4 with bitsandbytes
- **Attention**: Flash attention support

### Advanced Features
- **MoE**: Configurable expert routing
- **MTP**: Multi-token prediction heads
- **HRM**: Hierarchical reasoning module
- **Anti-Hallucination**: Forbidden token filtering

### Training Pipeline
- **Loss Tracking**: Advanced prediction and analysis
- **Monitoring**: Real-time system metrics
- **Error Recovery**: Automatic handling of OOM and gradient issues
- **Export**: Multi-platform model formats

## Configuration

See `configs/` directory for example configurations:
- `default_config.json` - Standard configuration
- `small_model_config.json` - For testing/development
- `large_model_config.json` - For production use

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory
- 16GB+ system memory

## License

MIT License - see LICENSE file for details.
"""

    with open(base_dir / 'README.md', 'w') as f:
        f.write(readme_content)

    print("📖 Created README.md")


def main():
    """Main setup function"""
    print("🔧 Setting up Advanced NanoLM Infrastructure")
    print("=" * 50)

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return 1

    # Check dependencies
    print("📦 Checking dependencies...")
    deps = check_dependencies()
    missing = [dep for dep, available in deps.items() if not available]

    if missing:
        # Separate critical from optional dependencies
        critical = [dep for dep in missing if dep not in ['flash_attn', 'pynvml']]
        optional = [dep for dep in missing if dep in ['flash_attn', 'pynvml']]

        if critical:
            print(f"⚠️ Missing critical dependencies: {', '.join(critical)}")
            if input("Install critical dependencies? (y/n): ").lower() == 'y':
                if not install_missing_dependencies(critical):
                    print("❌ Failed to install critical dependencies")
                    return 1

        if optional:
            print(f"ℹ️ Optional dependencies not available: {', '.join(optional)}")
            print("  - flash_attn: Requires CUDA toolkit for installation")
            print("  - pynvml: For GPU monitoring (will use torch.cuda instead)")
    else:
        print("✅ All dependencies available")

    # Validate system requirements
    print("\n💻 Validating system requirements...")
    sys_req = validate_system_requirements()

    if not sys_req['python_version']:
        print("❌ Python version too old")
        return 1

    if not sys_req['cuda_available']:
        print("⚠️ CUDA not available - training will be slow")
    else:
        print(f"✅ CUDA available with {sys_req['gpu_memory_gb']:.1f}GB GPU memory")

    print(f"💾 System memory: {sys_req['system_memory_gb']:.1f}GB")
    print(f"💿 Disk space: {sys_req['disk_space_gb']:.1f}GB")

    # Create directory structure
    print("\n📁 Creating directory structure...")
    directories = create_directory_structure()

    # Setup git ignore
    setup_git_ignore()

    # Create example configs
    print("\n📋 Creating configuration files...")
    create_example_configs(directories['configs'])

    # Create launch scripts
    print("\n🚀 Creating launch scripts...")
    create_launch_scripts(directories['scripts'])

    # Create README
    print("\n📖 Creating documentation...")
    create_readme(Path("."))

    # Test configuration
    print("\n🧪 Testing configuration...")
    try:
        config = create_default_config()
        config.validate()
        print("✅ Configuration validation passed")

        # Print summary
        config.print_summary()

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return 1

    print("\n🎉 Infrastructure setup completed!")
    print("\nNext steps:")
    print("1. Review configuration in configs/default_config.json")
    print("2. Run: python scripts/launch_training.py")
    print("3. Monitor training progress in loss_tracking/")

    return 0


if __name__ == "__main__":
    sys.exit(main())