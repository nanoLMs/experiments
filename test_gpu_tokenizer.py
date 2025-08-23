#!/usr/bin/env python3
"""
Test script for GPU-accelerated tokenizer
"""

import sys
import os
sys.path.append('data')

from datatokenize import NanoLMGPUTToken
import torch

def test_gpu_tokenizer():
    """Test the GPU-accelerated tokenizer"""
    print("🧪 Testing GPU-Accelerated Tokenizer")
    print("=" * 50)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  GPU not available, using CPU")

    # Initialize tokenizer with smaller vocab for testing
    tokenizer = NanoLMGPUTToken(
        vocab_size=8000,  # Smaller for testing
        min_frequency=2,
        output_dir="./test_tokenizer",
        batch_size=500
    )

    print(f"📊 Loaded {len(tokenizer.text_data)} text lines")
    print(f"📊 Loaded {len(tokenizer.json_data)} JSON entries")
    print(f"📊 Total text length: {len(tokenizer.text):,} characters")

    # Train tokenizer
    print("\n🔥 Starting training...")
    tokenizer.train_tokenizer()

    # Save tokenizer
    print("\n💾 Saving tokenizer...")
    tokenizer.save_tokenizer()

    # Test tokenization
    print("\n🧪 Testing tokenization...")
    test_texts = [
        "Hello world! This is a test of the GPU-accelerated tokenizer.",
        "Machine learning and artificial intelligence are revolutionizing technology.",
        "[THINK] Let me analyze this problem step by step. [CONCLUDE] The solution is clear.",
        "The constitution provides fundamental rights to all citizens.",
        "Legal acts and regulations govern our society."
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\n--- Test {i} ---")
        tokenizer.test_tokenizer(text)

    print("\n✅ GPU Tokenizer test completed successfully!")

if __name__ == "__main__":
    test_gpu_tokenizer()