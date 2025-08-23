# 🚀 NanoLM GPU-Accelerated Tokenizer

A high-performance, GPU-accelerated tokenizer designed specifically for the NanoLM architecture with advanced preprocessing capabilities and multimodal token support.

## ✨ Features

### 🔥 GPU Acceleration

- **GPU-accelerated preprocessing**: Utilizes CUDA for batch text processing
- **Memory-optimized batching**: Efficient memory management with configurable batch sizes
- **Automatic fallback**: Seamlessly falls back to parallel CPU processing when GPU is unavailable
- **Real-time monitoring**: GPU memory usage tracking and performance metrics

### 🎯 NanoLM Integration

- **Custom vocabulary**: 32,000 tokens optimized for NanoLM architecture
- **Special tokens**: Comprehensive set of multimodal, reasoning, and formatting tokens
- **MTP support**: Multi-token prediction tokens for advanced language modeling
- **Reasoning tokens**: Built-in support for chain-of-thought and reasoning patterns

### 🛠️ Advanced Preprocessing

- **Parallel processing**: Multi-core CPU utilization for maximum throughput
- **Text normalization**: Advanced cleaning and normalization pipeline
- **Memory efficiency**: Streaming processing for large datasets
- **Progress tracking**: Real-time progress monitoring with detailed statistics

## 🚀 Quick Start

### Installation Requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tokenizers transformers pandas pathlib
```

### Basic Usage

```python
from data.datatokenize import NanoLMGPUTToken

# Initialize tokenizer
tokenizer = NanoLMGPUTToken(
    vocab_size=32000,
    min_frequency=3,
    output_dir="./nanolm_tokenizer",
    batch_size=2000  # Adjust based on GPU memory
)

# Train tokenizer
tokenizer.train_tokenizer()

# Save tokenizer
tokenizer.save_tokenizer()

# Test tokenizer
tokenizer.test_tokenizer("Hello world! This is a test.")
```

### Running Tests

```bash
# Quick test
python test_gpu_tokenizer.py

# Performance benchmark
python benchmark_tokenizer.py
```

## 📊 Performance

### GPU Acceleration Benefits

- **2-5x faster** preprocessing compared to CPU-only processing
- **Memory efficient** batch processing prevents OOM errors
- **Scalable** performance with larger datasets

### Benchmark Results (RTX 4090)

```
📊 GPU vs CPU Speedup: 3.2x
🚀 GPU Speed: 1,250 texts/second
🔄 CPU Speed: 390 texts/second
```

## 🎯 Special Tokens

The tokenizer includes comprehensive special tokens for various use cases:

### Core Tokens

- `[UNK]`, `[PAD]`, `[CLS]`, `[SEP]`, `[MASK]`

### Multimodal Tokens

- `[IMG_START]`, `[IMG_END]`, `[VIS_TOKEN]`
- `[AUDIO_START]`, `[AUDIO_END]`, `[AUD_TOKEN]`

### Reasoning Tokens

- `[THINK]`, `[REASON]`, `[CONCLUDE]`

### MTP Tokens

- `[NEXT_1]`, `[NEXT_2]`, `[NEXT_3]`

### Document Structure

- `[DOC_START]`, `[DOC_END]`, `[SECTION_START]`, `[SECTION_END]`
- `[HEADER_START]`, `[HEADER_END]`, `[TITLE_START]`, `[TITLE_END]`

### Data Types

- `[JSON_START]`, `[JSON_END]`, `[XML_START]`, `[XML_END]`
- `[TABLE_START]`, `[TABLE_END]`, `[ROW_START]`, `[ROW_END]`

## ⚙️ Configuration

### GPU Settings

```python
tokenizer = NanoLMGPUTToken(
    vocab_size=32000,      # Target vocabulary size
    min_frequency=3,       # Minimum token frequency
    batch_size=2000,       # GPU batch size (adjust for your GPU)
    output_dir="./tokenizer"
)
```

### Memory Optimization

- **Batch size**: Start with 1000, increase based on GPU memory
- **Text length**: Longer texts require smaller batch sizes
- **GPU memory**: Monitor usage with built-in tracking

## 🔧 Advanced Usage

### Custom Data Sources

```python
class CustomLoadData(LoadData):
    def __init__(self):
        super().__init__()
        self.textpath = "/path/to/your/text.txt"
        self.jsonpath = "/path/to/your/data.json"
```

### Performance Tuning

```python
# For high-memory GPUs (24GB+)
tokenizer = NanoLMGPUTToken(batch_size=5000)

# For low-memory GPUs (8GB)
tokenizer = NanoLMGPUTToken(batch_size=500)

# CPU-only mode
tokenizer = NanoLMGPUTToken(batch_size=100)  # Will auto-detect and use CPU
```

## 📈 Monitoring

The tokenizer provides real-time monitoring:

```
🚀 NanoLM GPU-Accelerated Tokenizer
============================================================
🔥 Device: cuda
📊 Target vocabulary: 32,000 tokens
📊 Min frequency: 3
📊 Batch size: 2,000
🚀 Using GPU: NVIDIA GeForce RTX 4090
Memory Allocated: 0.00 MB
Max Memory Allocated: 0.00 MB
Memory Cached: 0.00 MB

🔥 GPU-accelerated preprocessing...
📊 Processed 10,000/50,000 texts
🚀 GPU Memory: 245.32 MB

🔥 Training on 45,678 preprocessed texts...
📊 Target vocab: 32,000, Min freq: 3
✅ Training completed in 127.45 seconds
🚀 Final GPU Memory: 512.18 MB
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   ```python
   # Reduce batch size
   tokenizer = NanoLMGPUTToken(batch_size=500)
   ```

2. **Slow Performance**

   ```python
   # Increase batch size (if memory allows)
   tokenizer = NanoLMGPUTToken(batch_size=3000)
   ```

3. **GPU Not Detected**
   ```bash
   # Check CUDA installation
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Performance Tips

- Use SSD storage for faster data loading
- Ensure adequate GPU memory (8GB+ recommended)
- Monitor GPU utilization with `nvidia-smi`
- Use mixed precision for even faster training

## 📝 License

This project is part of the NanoLM ecosystem and follows the same licensing terms.

## 🤝 Contributing

Contributions are welcome! Please ensure:

- GPU compatibility testing
- Performance benchmarks
- Documentation updates
- Test coverage

---

**Built with ❤️ for the NanoLM community**
