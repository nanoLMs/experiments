#!/usr/bin/env python3
"""
Unit Tests for NanoLM Model Architecture
=======================================

Comprehensive tests for:
- Quantized transformer components
- Multi-head attention with RoPE
- Feed-forward networks
- Model forward pass and generation
- Memory footprint calculation
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanolm_model import (
    NanoLMModel, TransformerBlock, MultiHeadAttention, FeedForward,
    QuantizedLinear, RoPEPositionalEncoding, apply_rotary_pos_emb,
    ModelOutput, create_nanolm_model
)

# Mock config for testing
class MockConfig:
    def __init__(self):
        # Model architecture
        self.vocab_size = 1000
        self.d_model = 256
        self.n_heads = 8
        self.d_head = 32
        self.n_layers = 4
        self.d_ff = 1024
        self.seq_len = 128

        # Dropout settings
        self.dropout = 0.1
        self.attn_dropout = 0.1
        self.ff_dropout = 0.1

        # Attention settings
        self.use_flash_attn = False
        self.rope_base = 10000
        self.rope_scaling = 1.0

        # Model settings
        self.tie_word_embeddings = True

        # Quantization settings
        self.use_quantization = False
        self.use_bnb_4bit = False
        self.bnb_4bit_quant_type = "nf4"
        self.bnb_4bit_compute_dtype = "bfloat16"
        self.bnb_4bit_use_double_quant = True
        self.bnb_4bit_quant_storage = "uint8"
        self.bnb_4bit_quantize_heads = False
        self.bnb_4bit_quantize_router = False

        # FP4 settings
        self.use_fp4 = False
        self.fp4_format = "nvfp4"
        self.fp4_block_size = 16
        self.fp4_split_rounding = True
        self.qaf_threshold = 1e-6
        self.max_qaf_steps = 100
        self.qaf_precision = "bf16"


class TestRoPEPositionalEncoding(unittest.TestCase):
    """Test RoPE positional encoding"""

    def setUp(self):
        self.d_model = 64
        self.max_seq_len = 128
        self.rope = RoPEPositionalEncoding(self.d_model, self.max_seq_len)

    def test_rope_initialization(self):
        """Test RoPE initialization"""
        self.assertEqual(self.rope.d_model, self.d_model)
        self.assertEqual(self.rope.max_seq_len, self.max_seq_len)
        self.assertTrue(hasattr(self.rope, 'inv_freq'))
        self.assertTrue(hasattr(self.rope, 'cos_cached'))
        self.assertTrue(hasattr(self.rope, 'sin_cached'))

    def test_rope_forward(self):
        """Test RoPE forward pass"""
        batch_size, seq_len, d_model = 2, 32, self.d_model
        x = torch.randn(batch_size, seq_len, d_model)

        cos, sin = self.rope(x, seq_len)

        # Check output shapes
        self.assertEqual(cos.shape, (seq_len, d_model))
        self.assertEqual(sin.shape, (seq_len, d_model))

        # Check values are in reasonable range
        self.assertTrue(torch.all(cos >= -1.0) and torch.all(cos <= 1.0))
        self.assertTrue(torch.all(sin >= -1.0) and torch.all(sin <= 1.0))

    def test_rope_cache_extension(self):
        """Test RoPE cache extension for longer sequences"""
        # Test with sequence longer than initial max_seq_len
        long_seq_len = self.max_seq_len + 50
        x = torch.randn(1, long_seq_len, self.d_model)

        cos, sin = self.rope(x, long_seq_len)

        # Check that cache was extended
        self.assertEqual(cos.shape[0], long_seq_len)
        self.assertEqual(sin.shape[0], long_seq_len)
        self.assertGreaterEqual(self.rope.max_seq_len, long_seq_len)

    def test_apply_rotary_pos_emb(self):
        """Test rotary position embedding application"""
        batch_size, n_heads, seq_len, d_head = 2, 8, 32, 32
        q = torch.randn(batch_size, n_heads, seq_len, d_head)
        k = torch.randn(batch_size, n_heads, seq_len, d_head)

        cos = torch.randn(seq_len, d_head)
        sin = torch.randn(seq_len, d_head)

        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)

        # Check output shapes
        self.assertEqual(q_embed.shape, q.shape)
        self.assertEqual(k_embed.shape, k.shape)

        # Check that embeddings are different from original
        self.assertFalse(torch.allclose(q, q_embed))
        self.assertFalse(torch.allclose(k, k_embed))


class TestQuantizedLinear(unittest.TestCase):
    """Test quantized linear layer"""

    def test_quantized_linear_standard(self):
        """Test standard (non-quantized) linear layer"""
        layer = QuantizedLinear(64, 32, bias=True)

        self.assertFalse(layer.is_quantized)
        self.assertIsInstance(layer.linear, nn.Linear)

        # Test forward pass
        x = torch.randn(16, 64)
        output = layer(x)
        self.assertEqual(output.shape, (16, 32))

    def test_quantized_linear_with_config(self):
        """Test quantized linear layer with quantization config"""
        quant_config = {
            'use_bnb_4bit': False,  # Disabled for testing without bitsandbytes
            'compute_dtype': 'bfloat16',
            'use_double_quant': True,
            'quant_type': 'nf4'
        }

        layer = QuantizedLinear(64, 32, bias=True, quantization_config=quant_config)

        # Should fall back to standard linear when bitsandbytes not available
        self.assertFalse(layer.is_quantized)

        # Test forward pass
        x = torch.randn(16, 64)
        output = layer(x)
        self.assertEqual(output.shape, (16, 32))

    def test_quantized_linear_extra_repr(self):
        """Test string representation"""
        layer = QuantizedLinear(64, 32)
        repr_str = layer.extra_repr()

        self.assertIn('in_features=64', repr_str)
        self.assertIn('out_features=32', repr_str)
        self.assertIn('quantized=', repr_str)


class TestMultiHeadAttention(unittest.TestCase):
    """Test multi-head attention"""

    def setUp(self):
        self.config = MockConfig()
        self.attention = MultiHeadAttention(self.config)

    def test_attention_initialization(self):
        """Test attention initialization"""
        self.assertEqual(self.attention.d_model, self.config.d_model)
        self.assertEqual(self.attention.n_heads, self.config.n_heads)
        self.assertEqual(self.attention.d_head, self.config.d_head)

        # Check that projections exist
        self.assertIsInstance(self.attention.q_proj, QuantizedLinear)
        self.assertIsInstance(self.attention.k_proj, QuantizedLinear)
        self.assertIsInstance(self.attention.v_proj, QuantizedLinear)
        self.assertIsInstance(self.attention.o_proj, QuantizedLinear)

        # Check RoPE
        self.assertIsNotNone(self.attention.rope)

    def test_attention_forward(self):
        """Test attention forward pass"""
        batch_size, seq_len = 2, 32
        hidden_states = torch.randn(batch_size, seq_len, self.config.d_model)

        attn_output, attn_weights, present_kv = self.attention(hidden_states)

        # Check output shape
        self.assertEqual(attn_output.shape, (batch_size, seq_len, self.config.d_model))

        # Check attention weights (should be None for flash attention, tensor for standard)
        if not self.attention.use_flash_attn:
            self.assertIsNotNone(attn_weights)
            expected_shape = (batch_size, self.config.n_heads, seq_len, seq_len)
            self.assertEqual(attn_weights.shape, expected_shape)

    def test_attention_with_mask(self):
        """Test attention with attention mask"""
        batch_size, seq_len = 2, 32
        hidden_states = torch.randn(batch_size, seq_len, self.config.d_model)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, seq_len//2:] = 0  # Mask second half

        attn_output, attn_weights, present_kv = self.attention(
            hidden_states, attention_mask=attention_mask
        )

        # Check output shape
        self.assertEqual(attn_output.shape, (batch_size, seq_len, self.config.d_model))

    def test_attention_with_cache(self):
        """Test attention with KV caching"""
        batch_size, seq_len = 2, 32
        hidden_states = torch.randn(batch_size, seq_len, self.config.d_model)

        # First pass with cache
        attn_output1, _, present_kv = self.attention(
            hidden_states, use_cache=True
        )

        self.assertIsNotNone(present_kv)
        self.assertEqual(len(present_kv), 2)  # key and value

        # Second pass with past key values
        new_hidden_states = torch.randn(batch_size, 1, self.config.d_model)
        attn_output2, _, _ = self.attention(
            new_hidden_states, past_key_value=present_kv, use_cache=True
        )

        self.assertEqual(attn_output2.shape, (batch_size, 1, self.config.d_model))


class TestFeedForward(unittest.TestCase):
    """Test feed-forward network"""

    def setUp(self):
        self.config = MockConfig()
        self.ff = FeedForward(self.config)

    def test_feedforward_initialization(self):
        """Test feed-forward initialization"""
        self.assertEqual(self.ff.d_model, self.config.d_model)
        self.assertEqual(self.ff.d_ff, self.config.d_ff)

        # Check SwiGLU components
        self.assertIsInstance(self.ff.gate_proj, QuantizedLinear)
        self.assertIsInstance(self.ff.up_proj, QuantizedLinear)
        self.assertIsInstance(self.ff.down_proj, QuantizedLinear)

    def test_feedforward_forward(self):
        """Test feed-forward forward pass"""
        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, self.config.d_model)

        output = self.ff(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, self.config.d_model))

        # Check that output is different from input (non-identity)
        self.assertFalse(torch.allclose(x, output))


class TestTransformerBlock(unittest.TestCase):
    """Test transformer block"""

    def setUp(self):
        self.config = MockConfig()
        self.block = TransformerBlock(self.config, layer_idx=0)

    def test_transformer_block_initialization(self):
        """Test transformer block initialization"""
        self.assertEqual(self.block.layer_idx, 0)

        # Check components
        self.assertIsInstance(self.block.input_layernorm, nn.LayerNorm)
        self.assertIsInstance(self.block.post_attention_layernorm, nn.LayerNorm)
        self.assertIsInstance(self.block.self_attn, MultiHeadAttention)
        self.assertIsInstance(self.block.mlp, FeedForward)

    def test_transformer_block_forward(self):
        """Test transformer block forward pass"""
        batch_size, seq_len = 2, 32
        hidden_states = torch.randn(batch_size, seq_len, self.config.d_model)

        output, attn_weights, present_kv = self.block(hidden_states)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, self.config.d_model))

        # Check that output is different from input (due to transformations)
        self.assertFalse(torch.allclose(hidden_states, output))

    def test_transformer_block_with_cache(self):
        """Test transformer block with caching"""
        batch_size, seq_len = 2, 32
        hidden_states = torch.randn(batch_size, seq_len, self.config.d_model)

        output, attn_weights, present_kv = self.block(
            hidden_states, use_cache=True
        )

        self.assertIsNotNone(present_kv)
        self.assertEqual(output.shape, (batch_size, seq_len, self.config.d_model))


class TestNanoLMModel(unittest.TestCase):
    """Test complete NanoLM model"""

    def setUp(self):
        self.config = MockConfig()
        self.model = NanoLMModel(self.config)

    def test_model_initialization(self):
        """Test model initialization"""
        self.assertEqual(self.model.vocab_size, self.config.vocab_size)
        self.assertEqual(self.model.d_model, self.config.d_model)
        self.assertEqual(self.model.n_layers, self.config.n_layers)

        # Check components
        self.assertIsInstance(self.model.embed_tokens, nn.Embedding)
        self.assertEqual(len(self.model.layers), self.config.n_layers)
        self.assertIsInstance(self.model.norm, nn.LayerNorm)

        # Check tied embeddings
        if self.config.tie_word_embeddings:
            self.assertIsNone(self.model.lm_head)
        else:
            self.assertIsNotNone(self.model.lm_head)

    def test_model_forward(self):
        """Test model forward pass"""
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

        outputs = self.model(input_ids)

        # Check output structure
        self.assertIsInstance(outputs, ModelOutput)
        self.assertEqual(outputs.logits.shape, (batch_size, seq_len, self.config.vocab_size))
        self.assertIsNotNone(outputs.hidden_states)
        self.assertEqual(len(outputs.hidden_states), self.config.n_layers)

    def test_model_forward_with_attention_mask(self):
        """Test model forward pass with attention mask"""
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, seq_len//2:] = 0  # Mask second half

        outputs = self.model(input_ids, attention_mask=attention_mask)

        self.assertEqual(outputs.logits.shape, (batch_size, seq_len, self.config.vocab_size))

    def test_model_forward_with_cache(self):
        """Test model forward pass with KV caching"""
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

        outputs = self.model(input_ids, use_cache=True)

        self.assertIsNotNone(outputs.past_key_values)
        self.assertEqual(len(outputs.past_key_values), self.config.n_layers)

    def test_model_generation(self):
        """Test model text generation"""
        batch_size, seq_len = 1, 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

        generated = self.model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=False,
            temperature=1.0
        )

        # Check that sequence was extended
        self.assertEqual(generated.shape, (batch_size, seq_len + 5))

        # Check that original tokens are preserved
        self.assertTrue(torch.equal(generated[:, :seq_len], input_ids))

    def test_model_generation_with_sampling(self):
        """Test model generation with sampling"""
        torch.manual_seed(42)  # For reproducibility

        batch_size, seq_len = 1, 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

        generated = self.model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )

        self.assertEqual(generated.shape, (batch_size, seq_len + 5))

    def test_get_num_params(self):
        """Test parameter counting"""
        total_params = self.model.get_num_params()
        non_embedding_params = self.model.get_num_params(non_embedding=True)

        self.assertGreater(total_params, 0)
        self.assertLess(non_embedding_params, total_params)

        # Check that difference is embedding parameters
        embedding_params = self.model.embed_tokens.weight.numel()
        self.assertEqual(total_params - non_embedding_params, embedding_params)

    def test_get_memory_footprint(self):
        """Test memory footprint calculation"""
        memory_info = self.model.get_memory_footprint()

        # Check required fields
        required_fields = [
            'total_params', 'quantized_params', 'quantized_memory_mb',
            'unquantized_memory_mb', 'total_memory_mb', 'quantization_ratio'
        ]

        for field in required_fields:
            self.assertIn(field, memory_info)

        # Check values are reasonable
        self.assertGreater(memory_info['total_params'], 0)
        self.assertGreaterEqual(memory_info['quantized_params'], 0)
        self.assertGreater(memory_info['total_memory_mb'], 0)
        self.assertGreaterEqual(memory_info['quantization_ratio'], 0.0)
        self.assertLessEqual(memory_info['quantization_ratio'], 1.0)

    def test_prepare_attention_mask(self):
        """Test attention mask preparation"""
        batch_size, seq_len = 2, 8
        attention_mask = torch.ones(batch_size, seq_len)

        prepared_mask = self.model._prepare_attention_mask(attention_mask, seq_len)

        # Check shape (should include batch dimension)
        expected_shape = (batch_size, 1, seq_len, seq_len)
        self.assertEqual(prepared_mask.shape, expected_shape)

        # Check causal structure (lower triangular)
        mask_2d = prepared_mask[0, 0]
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    # Future positions should be masked (large negative value)
                    self.assertLess(mask_2d[i, j].item(), -1000.0)


class TestModelFactory(unittest.TestCase):
    """Test model factory function"""

    def test_create_nanolm_model(self):
        """Test model creation factory"""
        config = MockConfig()
        model = create_nanolm_model(config)

        self.assertIsInstance(model, NanoLMModel)
        self.assertEqual(model.config, config)

    def test_create_nanolm_model_with_quantization(self):
        """Test model creation with quantization enabled"""
        config = MockConfig()
        config.use_quantization = True
        config.use_bnb_4bit = True

        model = create_nanolm_model(config)

        self.assertIsInstance(model, NanoLMModel)
        self.assertIsNotNone(model.quantization_controller)


class TestModelIntegration(unittest.TestCase):
    """Integration tests for the complete model"""

    def test_end_to_end_training_step(self):
        """Test a complete training step"""
        config = MockConfig()
        model = create_nanolm_model(config)

        # Create sample data
        batch_size, seq_len = 4, 64
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass
        outputs = model(input_ids)
        logits = outputs.logits

        # Calculate loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, config.vocab_size), labels.view(-1))

        # Backward pass
        loss.backward()

        # Check that gradients were computed
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)

        print(f"✅ Training step test passed (loss: {loss.item():.4f})")

    def test_model_size_estimation(self):
        """Test model size estimation accuracy"""
        config = MockConfig()
        model = create_nanolm_model(config)

        # Get actual parameter count
        actual_params = sum(p.numel() for p in model.parameters())

        # Get estimated parameter count
        estimated_params = model.get_num_params()

        # Should match exactly
        self.assertEqual(actual_params, estimated_params)

        # Test memory footprint
        memory_info = model.get_memory_footprint()

        # Total params should match
        self.assertEqual(memory_info['total_params'], actual_params)

        print(f"✅ Model size: {actual_params:,} parameters")
        print(f"✅ Memory footprint: {memory_info['total_memory_mb']:.1f} MB")

    def test_different_model_sizes(self):
        """Test different model configurations"""
        configs = [
            # Small model
            {'n_layers': 2, 'd_model': 128, 'n_heads': 4, 'd_ff': 512},
            # Medium model
            {'n_layers': 6, 'd_model': 384, 'n_heads': 6, 'd_ff': 1536},
            # Large model
            {'n_layers': 12, 'd_model': 512, 'n_heads': 8, 'd_ff': 2048},
        ]

        for i, config_updates in enumerate(configs):
            with self.subTest(config=i):
                config = MockConfig()
                for key, value in config_updates.items():
                    setattr(config, key, value)

                # Update d_head to match d_model and n_heads
                config.d_head = config.d_model // config.n_heads

                model = create_nanolm_model(config)

                # Test forward pass
                batch_size, seq_len = 2, 32
                input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

                outputs = model(input_ids)
                self.assertEqual(outputs.logits.shape, (batch_size, seq_len, config.vocab_size))

                memory_info = model.get_memory_footprint()
                print(f"✅ Config {i}: {model.get_num_params():,} params, {memory_info['total_memory_mb']:.1f} MB")


if __name__ == '__main__':
    # Setup test environment
    torch.manual_seed(42)
    np.random.seed(42)

    # Run tests
    unittest.main(verbosity=2)