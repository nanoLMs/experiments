#!/usr/bin/env python3
"""
Benchmark script to compare GPU vs CPU tokenizer performance
"""

import sys
import os
import time
import torch
sys.path.append('data')

from datatokenize import NanoLMGPUTToken

def benchmark_preprocessing():
    """Benchmark GPU vs CPU preprocessing"""
    print("🏁 Tokenizer Preprocessing Benchmark")
    print("=" * 60)

    # Create test data
    test_texts = [
        "This is a sample text for preprocessing benchmark." * 10,
        "Machine learning models require efficient tokenization." * 15,
        "GPU acceleration can significantly improve performance." * 12,
        "Natural language processing is computationally intensive." * 8,
        "Tokenizers convert text into numerical representations." * 20
    ] * 100  # Create 500 test texts

    print(f"📊 Test data: {len(test_texts)} texts")
    print(f"📊 Average text length: {sum(len(t) for t in test_texts) / len(test_texts):.1f} chars")

    # Initialize tokenizer
    tokenizer = NanoLMGPUTToken(
        vocab_size=16000,
        min_frequency=2,
        batch_size=100
    )

    # Benchmark GPU preprocessing
    if torch.cuda.is_available():
        print("\n🔥 GPU Preprocessing Benchmark")
        torch.cuda.synchronize()
        start_time = time.time()

        gpu_results = tokenizer._gpu_accelerated_preprocessing(test_texts)

        torch.cuda.synchronize()
        gpu_time = time.time() - start_time

        print(f"✅ GPU Time: {gpu_time:.3f} seconds")
        print(f"📊 GPU Processed: {len(gpu_results)} texts")
        print(f"🚀 GPU Speed: {len(test_texts) / gpu_time:.1f} texts/second")
    else:
        print("⚠️  GPU not available for benchmarking")
        gpu_time = float('inf')
        gpu_results = []

    # Benchmark CPU preprocessing
    print("\n🔄 CPU Preprocessing Benchmark")
    start_time = time.time()

    cpu_results = tokenizer._parallel_text_processing(test_texts)

    cpu_time = time.time() - start_time

    print(f"✅ CPU Time: {cpu_time:.3f} seconds")
    print(f"📊 CPU Processed: {len(cpu_results)} texts")
    print(f"🚀 CPU Speed: {len(test_texts) / cpu_time:.1f} texts/second")

    # Compare results
    if torch.cuda.is_available() and gpu_time < float('inf'):
        speedup = cpu_time / gpu_time
        print(f"\n🏆 Performance Summary")
        print(f"📊 GPU vs CPU Speedup: {speedup:.2f}x")

        if speedup > 1:
            print(f"✅ GPU is {speedup:.2f}x faster than CPU")
        else:
            print(f"⚠️  CPU is {1/speedup:.2f}x faster than GPU")

    print(f"\n📊 Memory Usage:")
    if torch.cuda.is_available():
        print(f"🚀 GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"🚀 GPU Max Memory: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")

def benchmark_full_training():
    """Benchmark full tokenizer training"""
    print("\n" + "=" * 60)
    print("🏁 Full Training Benchmark")
    print("=" * 60)

    # Small vocabulary for quick testing
    tokenizer = NanoLMGPUTToken(
        vocab_size=4000,
        min_frequency=1,
        output_dir="./benchmark_tokenizer",
        batch_size=200
    )

    print(f"📊 Training data size: {len(tokenizer.text):,} characters")

    # Time the full training process
    start_time = time.time()
    tokenizer.train_tokenizer()
    training_time = time.time() - start_time

    print(f"\n🏆 Training Results:")
    print(f"⏱️  Total training time: {training_time:.2f} seconds")
    print(f"🚀 Training speed: {len(tokenizer.text) / training_time:.0f} chars/second")

    # Test the trained tokenizer
    test_text = "This is a comprehensive test of our GPU-accelerated tokenizer performance."
    tokenizer.test_tokenizer(test_text)

if __name__ == "__main__":
    benchmark_preprocessing()
    benchmark_full_training()
    print("\n✅ Benchmark completed!")