# Advanced NanoLM System

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
