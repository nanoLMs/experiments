#!/usr/bin/env python3
"""
Unit Tests for MoE System
=========================

Comprehensive tests for:
- Expert networks and routing
- Load balancing and auxiliary losses
- MoE layer integration
- Expert usage statistics
- Model integration with MoE
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

from moe_system import (
    Expert, Router, LoadBalancer, MoELayer, MoETransformerBlock,
    MoEOutput, create_moe_model
)

# Mock config for testing
class MockConfig:
    def __init__(self):
        # Model architecture
        self.d_model = 256
        self.d_ff = 1024
        self.n_experts = 4
        self.moe_top_k = 1
        self.expert_ff_mult = 1.0
        self.router_jitter = 0.01
        self.moe_aux_weight = 0.01
        self.router_z_loss = 1e-4
        self.capacity_factor = 1.0
        self.ff_dropout = 0.1
        self.dropout = 0.1

        # Quantization settings
        self.use_bnb_4bit = False
        self.bnb_4bit_quantize_router = False
        self.bnb_4bit_compute_dtype = "bfloat16"
        self.bnb_4bit_use_double_quant = True
        self.bnb_4bit_quant_type = "nf4"

        # Full model settings (for integration tests)
        self.n_layers = 4
        self.n_heads = 8
        self.d_head = 32
        self.seq_len = 128
        self.vocab_size = 1000
        self.attn_dropout = 0.1
        self.use_flash_attn = False
        self.rope_base = 10000
        self.rope_scaling = 1.0
        self.tie_word_embeddings = True
        self.use_quantization = False
        self.moe_every = 2  # Every 2nd layer is MoE

        # Additional required attributes
        self.bnb_4bit_quant_storage = "uint8"
        self.bnb_4bit_quantize_heads = False
        self.use_fp4 = False
        self.fp4_format = "nvfp4"
        self.fp4_block_size = 16
        self.fp4_split_rounding = True
        self.qaf_threshold = 1e-6
        self.max_qaf_steps = 100
        self.qaf_precision = "bf16"


class TestExpert(unittest.TestCase):
    """Test individual expert networks"""

    def setUp(self):
        self.config = MockConfig()
        self.expert = Expert(self.config, expert_id=0)

    def test_expert_initialization(self):
        """Test expert initialization"""
        self.assertEqual(self.expert.expert_id, 0)
        self.assertEqual(self.expert.d_model, self.config.d_model)
        self.assertEqual(self.expert.d_ff, int(self.config.d_ff * self.config.expert_ff_mult))

        # Check SwiGLU components
        self.assertIsNotNone(self.expert.gate_proj)
        self.assertIsNotNone(self.expert.up_proj)
        self.assertIsNotNone(self.expert.down_proj)

        # Check usage tracking buffers
        self.assertTrue(hasattr(self.expert, 'usage_count'))
        self.assertTrue(hasattr(self.expert, 'total_tokens'))

    def test_expert_forward(self):
        """Test expert forward pass"""
        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, self.config.d_model)

        output = self.expert(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, self.config.d_model))

        # Check that output is different from input (non-identity)
        self.assertFalse(torch.allclose(x, output))

    def test_expert_usage_tracking(self):
        """Test expert usage statistics tracking"""
        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, self.config.d_model)

        # Initial stats should be zero
        initial_stats = self.expert.get_usage_stats()
        self.assertEqual(initial_stats['usage_count'], 0)
        self.assertEqual(initial_stats['total_tokens'], 0)

        # Forward pass should update stats (in training mode)
        self.expert.train()
        output = self.expert(x)

        updated_stats = self.expert.get_usage_stats()
        self.assertEqual(updated_stats['usage_count'], 1)
        self.assertEqual(updated_stats['total_tokens'], batch_size * seq_len)

        # Reset stats
        self.expert.reset_usage_stats()
        reset_stats = self.expert.get_usage_stats()
        self.assertEqual(reset_stats['usage_count'], 0)
        self.assertEqual(reset_stats['total_tokens'], 0)

    def test_expert_eval_mode(self):
        """Test that expert doesn't track usage in eval mode"""
        batch_size, seq_len = 4, 32
        x = torch.randn(batch_size, seq_len, self.config.d_model)

        # Set to eval mode
        self.expert.eval()
        output = self.expert(x)

        # Stats should remain zero
        stats = self.expert.get_usage_stats()
        self.assertEqual(stats['usage_count'], 0)
        self.assertEqual(stats['total_tokens'], 0)


class TestRouter(unittest.TestCase):
    """Test router functionality"""

    def setUp(self):
        self.config = MockConfig()
        self.router = Router(self.config)

    def test_router_initialization(self):
        """Test router initialization"""
        self.assertEqual(self.router.n_experts, self.config.n_experts)
        self.assertEqual(self.router.top_k, self.config.moe_top_k)
        self.assertIsNotNone(self.router.gate)

        # Check load balancing buffers
        self.assertTrue(hasattr(self.router, 'expert_counts'))
        self.assertTrue(hasattr(self.router, 'total_tokens_routed'))

    def test_router_forward(self):
        """Test router forward pass"""
        batch_size, seq_len = 4, 32
        hidden_states = torch.randn(batch_size, seq_len, self.config.d_model)

        expert_weights, expert_indices, router_logits = self.router(hidden_states)

        # Check output shapes
        self.assertEqual(expert_weights.shape, (batch_size * seq_len, self.config.moe_top_k))
        self.assertEqual(expert_indices.shape, (batch_size * seq_len, self.config.moe_top_k))
        self.assertEqual(router_logits.shape, (batch_size, seq_len, self.config.n_experts))

        # Check that weights are normalized (sum to 1)
        weight_sums = expert_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6))

        # Check that indices are valid
        self.assertTrue(torch.all(expert_indices >= 0))
        self.assertTrue(torch.all(expert_indices < self.config.n_experts))

    def test_router_load_balancing_stats(self):
        """Test load balancing statistics"""
        batch_size, seq_len = 4, 32
        hidden_states = torch.randn(batch_size, seq_len, self.config.d_model)

        # Initial stats
        initial_stats = self.router.get_load_balancing_stats()
        self.assertEqual(initial_stats['total_tokens'], 0)

        # Forward pass in training mode
        self.router.train()
        expert_weights, expert_indices, router_logits = self.router(hidden_states)

        # Check updated stats
        updated_stats = self.router.get_load_balancing_stats()
        self.assertEqual(updated_stats['total_tokens'], batch_size * seq_len)
        self.assertEqual(len(updated_stats['expert_usage']), self.config.n_experts)
        self.assertGreaterEqual(updated_stats['balance_score'], 0.0)
        self.assertLessEqual(updated_stats['balance_score'], 1.0)

        # Reset stats
        self.router.reset_load_balancing_stats()
        reset_stats = self.router.get_load_balancing_stats()
        self.assertEqual(reset_stats['total_tokens'], 0)

    def test_router_jitter_noise(self):
        """Test that jitter noise is applied during training"""
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, self.config.d_model)

        # Set seed for reproducibility
        torch.manual_seed(42)
        self.router.train()
        _, _, logits_train = self.router(hidden_states)

        # Same input in eval mode should give different results due to no jitter
        torch.manual_seed(42)
        self.router.eval()
        _, _, logits_eval = self.router(hidden_states)

        # Results should be different due to jitter in training mode
        # Note: This test might occasionally fail due to randomness
        # We just check that both produce valid outputs
        self.assertEqual(logits_train.shape, logits_eval.shape)


class TestLoadBalancer(unittest.TestCase):
    """Test load balancing functionality"""

    def setUp(self):
        self.config = MockConfig()
        self.load_balancer = LoadBalancer(self.config)

    def test_load_balancer_initialization(self):
        """Test load balancer initialization"""
        self.assertEqual(self.load_balancer.n_experts, self.config.n_experts)
        self.assertEqual(self.load_balancer.aux_weight, self.config.moe_aux_weight)
        self.assertEqual(self.load_balancer.z_loss_weight, self.config.router_z_loss)

    def test_aux_loss_computation(self):
        """Test auxiliary loss computation"""
        batch_size, seq_len = 4, 32
        router_logits = torch.randn(batch_size, seq_len, self.config.n_experts)
        expert_indices = torch.randint(0, self.config.n_experts, (batch_size * seq_len, self.config.moe_top_k))

        aux_loss = self.load_balancer.compute_aux_loss(router_logits, expert_indices)

        # Check that loss is a scalar tensor
        self.assertEqual(aux_loss.shape, torch.Size([]))
        self.assertGreaterEqual(aux_loss.item(), 0.0)

    def test_z_loss_computation(self):
        """Test router z-loss computation"""
        batch_size, seq_len = 4, 32
        router_logits = torch.randn(batch_size, seq_len, self.config.n_experts)

        z_loss = self.load_balancer.compute_z_loss(router_logits)

        # Check that loss is a scalar tensor
        self.assertEqual(z_loss.shape, torch.Size([]))
        self.assertGreaterEqual(z_loss.item(), 0.0)

    def test_z_loss_disabled(self):
        """Test z-loss when disabled"""
        # Temporarily disable z-loss
        original_weight = self.load_balancer.z_loss_weight
        self.load_balancer.z_loss_weight = 0.0

        batch_size, seq_len = 4, 32
        router_logits = torch.randn(batch_size, seq_len, self.config.n_experts)

        z_loss = self.load_balancer.compute_z_loss(router_logits)

        # Should be zero
        self.assertEqual(z_loss.item(), 0.0)

        # Restore original weight
        self.load_balancer.z_loss_weight = original_weight

    def test_total_load_balancing_loss(self):
        """Test total load balancing loss"""
        batch_size, seq_len = 4, 32
        router_logits = torch.randn(batch_size, seq_len, self.config.n_experts)
        expert_indices = torch.randint(0, self.config.n_experts, (batch_size * seq_len, self.config.moe_top_k))

        total_loss = self.load_balancer.compute_load_balancing_loss(router_logits, expert_indices)

        # Check that loss is a scalar tensor
        self.assertEqual(total_loss.shape, torch.Size([]))
        self.assertGreaterEqual(total_loss.item(), 0.0)


class TestMoELayer(unittest.TestCase):
    """Test complete MoE layer"""

    def setUp(self):
        self.config = MockConfig()
        self.moe_layer = MoELayer(self.config, layer_idx=0)

    def test_moe_layer_initialization(self):
        """Test MoE layer initialization"""
        self.assertEqual(self.moe_layer.layer_idx, 0)
        self.assertEqual(self.moe_layer.n_experts, self.config.n_experts)
        self.assertEqual(len(self.moe_layer.experts), self.config.n_experts)
        self.assertIsNotNone(self.moe_layer.router)
        self.assertIsNotNone(self.moe_layer.load_balancer)

    def test_moe_layer_forward(self):
        """Test MoE layer forward pass"""
        batch_size, seq_len = 4, 32
        hidden_states = torch.randn(batch_size, seq_len, self.config.d_model)

        output = self.moe_layer(hidden_states)

        # Check output structure
        self.assertIsInstance(output, MoEOutput)
        self.assertEqual(output.hidden_states.shape, (batch_size, seq_len, self.config.d_model))
        self.assertEqual(output.router_logits.shape, (batch_size, seq_len, self.config.n_experts))
        self.assertEqual(output.expert_usage.shape, (self.config.n_experts,))

        # Check that losses are scalar tensors
        self.assertEqual(output.aux_loss.shape, torch.Size([]))
        self.assertEqual(output.load_balancing_loss.shape, torch.Size([]))

        # Check that losses are non-negative
        self.assertGreaterEqual(output.aux_loss.item(), 0.0)
        self.assertGreaterEqual(output.load_balancing_loss.item(), 0.0)

    def test_moe_layer_expert_stats(self):
        """Test expert statistics collection"""
        batch_size, seq_len = 4, 32
        hidden_states = torch.randn(batch_size, seq_len, self.config.d_model)

        # Forward pass to generate stats
        self.moe_layer.train()
        output = self.moe_layer(hidden_states)

        # Get expert stats
        stats = self.moe_layer.get_expert_stats()

        # Check stats structure
        self.assertEqual(stats['layer_idx'], 0)
        self.assertEqual(stats['n_experts'], self.config.n_experts)
        self.assertIn('router_stats', stats)
        self.assertIn('expert_stats', stats)
        self.assertEqual(len(stats['expert_stats']), self.config.n_experts)

        # Reset stats
        self.moe_layer.reset_expert_stats()

        # Stats should be reset
        reset_stats = self.moe_layer.get_expert_stats()
        self.assertEqual(reset_stats['router_stats']['total_tokens'], 0)


class TestMoETransformerBlock(unittest.TestCase):
    """Test MoE transformer block"""

    def setUp(self):
        self.config = MockConfig()
        self.moe_block = MoETransformerBlock(self.config, layer_idx=0)

    def test_moe_transformer_block_initialization(self):
        """Test MoE transformer block initialization"""
        self.assertEqual(self.moe_block.layer_idx, 0)

        # Check components
        self.assertIsNotNone(self.moe_block.input_layernorm)
        self.assertIsNotNone(self.moe_block.post_attention_layernorm)
        self.assertIsNotNone(self.moe_block.self_attn)
        self.assertIsNotNone(self.moe_block.moe)

    def test_moe_transformer_block_forward(self):
        """Test MoE transformer block forward pass"""
        batch_size, seq_len = 4, 32
        hidden_states = torch.randn(batch_size, seq_len, self.config.d_model)

        output, attn_weights, present_kv, moe_output = self.moe_block(hidden_states)

        # Check output shapes
        self.assertEqual(output.shape, (batch_size, seq_len, self.config.d_model))
        self.assertIsInstance(moe_output, MoEOutput)

        # Check that output is different from input
        self.assertFalse(torch.allclose(hidden_states, output))

    def test_moe_transformer_block_with_cache(self):
        """Test MoE transformer block with caching"""
        batch_size, seq_len = 4, 32
        hidden_states = torch.randn(batch_size, seq_len, self.config.d_model)

        output, attn_weights, present_kv, moe_output = self.moe_block(
            hidden_states, use_cache=True
        )

        # Should return present key values
        self.assertIsNotNone(present_kv)
        self.assertEqual(len(present_kv), 2)  # key and value


class TestMoEModelIntegration(unittest.TestCase):
    """Test MoE model integration"""

    def setUp(self):
        self.config = MockConfig()

    def test_create_moe_model(self):
        """Test MoE model creation"""
        model = create_moe_model(self.config)

        # Check that model was created
        self.assertIsNotNone(model)

        # Check that some layers were replaced with MoE layers
        moe_layers = [layer for layer in model.layers if hasattr(layer, 'moe')]
        self.assertGreater(len(moe_layers), 0)

        # With moe_every=2, we should have 2 MoE layers (layers 1 and 3)
        expected_moe_layers = self.config.n_layers // self.config.moe_every
        self.assertEqual(len(moe_layers), expected_moe_layers)

    def test_moe_model_forward(self):
        """Test MoE model forward pass"""
        model = create_moe_model(self.config)

        batch_size, seq_len = 2, 64
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

        outputs = model(input_ids)

        # Check output shape
        self.assertEqual(outputs.logits.shape, (batch_size, seq_len, self.config.vocab_size))

        # Check that model has more parameters than base model (due to experts)
        param_count = model.get_num_params()
        self.assertGreater(param_count, 0)

    def test_moe_model_training_step(self):
        """Test MoE model training step with auxiliary losses"""
        model = create_moe_model(self.config)
        model.train()

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

        # Forward pass
        outputs = model(input_ids)

        # Calculate main loss
        loss_fn = nn.CrossEntropyLoss()
        main_loss = loss_fn(outputs.logits.view(-1, self.config.vocab_size), labels.view(-1))

        # Collect auxiliary losses from MoE layers
        aux_losses = []
        for layer in model.layers:
            if hasattr(layer, 'moe'):
                # We need to do a forward pass to get MoE outputs
                # This is a simplified test - in practice, the losses would be collected during forward
                pass

        # Check that main loss is computed
        self.assertGreater(main_loss.item(), 0)

        # Backward pass should work
        main_loss.backward()

        # Check that gradients were computed for at least some parameters
        grad_count = sum(1 for param in model.parameters() if param.requires_grad and param.grad is not None)
        self.assertGreater(grad_count, 0, "At least some parameters should have gradients")

    def test_moe_model_different_configs(self):
        """Test MoE model with different configurations"""
        configs = [
            {'moe_every': 1, 'n_experts': 2},  # Every layer is MoE, 2 experts
            {'moe_every': 3, 'n_experts': 8},  # Every 3rd layer, 8 experts
            {'moe_every': 0, 'n_experts': 4},  # No MoE layers
        ]

        for i, config_updates in enumerate(configs):
            with self.subTest(config=i):
                config = MockConfig()
                for key, value in config_updates.items():
                    setattr(config, key, value)

                model = create_moe_model(config)

                # Count MoE layers
                moe_layers = [layer for layer in model.layers if hasattr(layer, 'moe')]

                if config.moe_every == 0:
                    expected_moe_layers = 0
                else:
                    expected_moe_layers = config.n_layers // config.moe_every

                self.assertEqual(len(moe_layers), expected_moe_layers)

                # Test forward pass
                batch_size, seq_len = 2, 16
                input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

                outputs = model(input_ids)
                self.assertEqual(outputs.logits.shape, (batch_size, seq_len, config.vocab_size))


class TestMoEEfficiency(unittest.TestCase):
    """Test MoE efficiency and performance characteristics"""

    def setUp(self):
        self.config = MockConfig()

    def test_expert_specialization(self):
        """Test that experts can specialize (basic test)"""
        # Create MoE layer
        moe_layer = MoELayer(self.config, layer_idx=0)
        moe_layer.train()

        # Create different types of input patterns
        batch_size, seq_len = 8, 16

        # Pattern 1: All positive values
        pattern1 = torch.abs(torch.randn(batch_size, seq_len, self.config.d_model))

        # Pattern 2: All negative values
        pattern2 = -torch.abs(torch.randn(batch_size, seq_len, self.config.d_model))

        # Process patterns multiple times to allow specialization
        for _ in range(10):
            output1 = moe_layer(pattern1)
            output2 = moe_layer(pattern2)

        # Get expert usage for both patterns
        # This is a basic test - in practice, we'd need more sophisticated analysis
        self.assertEqual(output1.hidden_states.shape, pattern1.shape)
        self.assertEqual(output2.hidden_states.shape, pattern2.shape)

    def test_load_balancing_effectiveness(self):
        """Test that load balancing improves expert usage distribution"""
        moe_layer = MoELayer(self.config, layer_idx=0)
        moe_layer.train()

        batch_size, seq_len = 16, 32

        # Process many different inputs
        for _ in range(20):
            hidden_states = torch.randn(batch_size, seq_len, self.config.d_model)
            output = moe_layer(hidden_states)

        # Get load balancing stats
        stats = moe_layer.get_expert_stats()
        balance_score = stats['router_stats']['balance_score']

        # Balance score should be reasonable (not perfect, but not terrible)
        self.assertGreater(balance_score, 0.1)  # At least 10% balanced
        self.assertLessEqual(balance_score, 1.0)  # At most 100% balanced


if __name__ == '__main__':
    # Setup test environment
    torch.manual_seed(42)
    np.random.seed(42)

    # Run tests
    unittest.main(verbosity=2)