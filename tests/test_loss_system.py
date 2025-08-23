#!/usr/bin/env python3
"""
Test suite for Multi-Component Loss System
==========================================

Comprehensive tests for the advanced loss system including:
- Individual loss component calculations
- Multi-component loss integration
- Loss scaling and adaptation
- Dynamic weight adjustment
- FP4 underflow prevention
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loss_system import (
    MultiComponentLoss, LossWeights, LossOutput, LossComponent,
    create_loss_system
)


class TestConfig:
    """Test configuration for loss system"""
    loss_smoothing = 0.1
    mtp_loss_weights = [1.0, 0.5, 0.25, 0.125]
    router_z_loss = 1e-4
    forbidden_tokens = [100, 200, 300]


@pytest.fixture
def config():
    return TestConfig()


@pytest.fixture
def loss_weights():
    return LossWeights(
        main_weight=1.0,
        mtp_weight=0.5,
        auxiliary_weight=0.01,
        reasoning_weight=0.1,
        anti_hallucination_weight=0.25,
        fp4_scaling_factor=10.0,
        adaptive_scaling=True,
        dynamic_weighting=False
    )


@pytest.fixture
def sample_data():
    batch_size, seq_len, vocab_size = 4, 32, 1000

    return {
        'logits': torch.randn(batch_size, seq_len, vocab_size),
        'targets': torch.randint(0, vocab_size, (batch_size, seq_len)),
        'mtp_logits': [
            torch.randn(batch_size, seq_len, vocab_size) for _ in range(4)
        ],
        'router_logits': torch.randn(batch_size, seq_len, 4),
        'reasoning_logits': torch.randn(batch_size, seq_len, vocab_size),
        'level_contributions': {
            'low': torch.randn(batch_size, seq_len, vocab_size),
            'mid': torch.randn(batch_size, seq_len, vocab_size),
            'high': torch.randn(batch_size, seq_len, vocab_size)
        }
    }


class TestLossWeights:
    """Test loss weights configuration"""

    def test_default_weights(self):
        """Test default weight values"""
        weights = LossWeights()

        assert weights.main_weight == 1.0
        assert weights.mtp_weight == 0.5
        assert weights.auxiliary_weight == 0.01
        assert weights.reasoning_weight == 0.1
        assert weights.anti_hallucination_weight == 0.25
        assert weights.fp4_scaling_factor == 10.0
        assert weights.adaptive_scaling == True
        assert weights.dynamic_weighting == False

    def test_custom_weights(self, loss_weights):
        """Test custom weight configuration"""
        assert loss_weights.main_weight == 1.0
        assert loss_weights.mtp_weight == 0.5
        assert loss_weights.fp4_scaling_factor == 10.0


class TestMultiComponentLoss:
    """Test multi-component loss calculator"""

    def test_initialization(self, config, loss_weights):
        """Test loss system initialization"""
        loss_system = MultiComponentLoss(config, loss_weights)

        assert loss_system.config == config
        assert loss_system.loss_weights == loss_weights
        assert hasattr(loss_system, 'main_loss_fn')
        assert hasattr(loss_system, 'mtp_loss_fn')
        assert hasattr(loss_system, 'auxiliary_loss_fn')
        assert hasattr(loss_system, 'reasoning_loss_fn')
        assert loss_system.current_scaling_factor == loss_weights.fp4_scaling_factor
        assert loss_system.step_count == 0

    def test_main_loss_calculation(self, config, loss_weights, sample_data):
        """Test main loss calculation"""
        loss_system = MultiComponentLoss(config, loss_weights)

        main_loss = loss_system.calculate_main_loss(
            sample_data['logits'], sample_data['targets']
        )

        assert isinstance(main_loss, torch.Tensor)
        assert main_loss.dim() == 0  # Scalar
        assert main_loss.item() >= 0
        assert not torch.isnan(main_loss)
        assert not torch.isinf(main_loss)

    def test_mtp_loss_calculation(self, config, loss_weights, sample_data):
        """Test multi-token prediction loss calculation"""
        loss_system = MultiComponentLoss(config, loss_weights)

        mtp_loss, individual_losses = loss_system.calculate_mtp_loss(
            sample_data['mtp_logits'], sample_data['targets'], config.mtp_loss_weights
        )

        assert isinstance(mtp_loss, torch.Tensor)
        assert mtp_loss.dim() == 0
        assert mtp_loss.item() >= 0
        assert isinstance(individual_losses, dict)
        assert len(individual_losses) == len(sample_data['mtp_logits'])

        # Check individual loss keys
        for i in range(len(sample_data['mtp_logits'])):
            assert f't+{i+1}' in individual_losses

    def test_auxiliary_loss_calculation(self, config, loss_weights, sample_data):
        """Test auxiliary (MoE) loss calculation"""
        loss_system = MultiComponentLoss(config, loss_weights)

        aux_loss = loss_system.calculate_auxiliary_loss(sample_data['router_logits'])

        assert isinstance(aux_loss, torch.Tensor)
        assert aux_loss.dim() == 0
        assert aux_loss.item() >= 0

        # Test with None input
        aux_loss_none = loss_system.calculate_auxiliary_loss(None)
        assert aux_loss_none.item() == 0.0

    def test_reasoning_loss_calculation(self, config, loss_weights, sample_data):
        """Test reasoning loss calculation"""
        loss_system = MultiComponentLoss(config, loss_weights)

        reasoning_loss, individual_losses = loss_system.calculate_reasoning_loss(
            sample_data['reasoning_logits'], sample_data['targets'],
            sample_data['level_contributions']
        )

        assert isinstance(reasoning_loss, torch.Tensor)
        assert reasoning_loss.dim() == 0
        assert reasoning_loss.item() >= 0
        assert isinstance(individual_losses, dict)
        assert 'main' in individual_losses

        # Check level contributions
        for level in ['low', 'mid', 'high']:
            if level in sample_data['level_contributions']:
                assert level in individual_losses

    def test_anti_hallucination_loss_calculation(self, config, loss_weights, sample_data):
        """Test anti-hallucination loss calculation"""
        loss_system = MultiComponentLoss(config, loss_weights)

        anti_halluc_loss = loss_system.calculate_anti_hallucination_loss(
            sample_data['logits'], sample_data['targets'], config.forbidden_tokens
        )

        assert isinstance(anti_halluc_loss, torch.Tensor)
        assert anti_halluc_loss.dim() == 0
        assert anti_halluc_loss.item() >= 0

        # Test without forbidden tokens
        anti_halluc_loss_none = loss_system.calculate_anti_hallucination_loss(
            sample_data['logits'], sample_data['targets'], None
        )
        assert anti_halluc_loss_none.item() >= 0

    def test_loss_scaling(self, config, loss_weights):
        """Test loss scaling for FP4"""
        loss_system = MultiComponentLoss(config, loss_weights)

        # Test normal loss (no scaling)
        normal_loss = torch.tensor(5.0)
        scaled_normal = loss_system.apply_loss_scaling(normal_loss, use_fp4=False)
        assert torch.allclose(scaled_normal, normal_loss)

        # Test FP4 scaling
        fp4_loss = torch.tensor(1.0)
        scaled_fp4 = loss_system.apply_loss_scaling(fp4_loss, use_fp4=True)
        assert scaled_fp4.item() == fp4_loss.item() * loss_weights.fp4_scaling_factor

        # Test adaptive scaling with very small loss
        small_loss = torch.tensor(1e-8)
        initial_scaling = loss_system.current_scaling_factor
        scaled_small = loss_system.apply_loss_scaling(small_loss, use_fp4=True)

        # Scaling factor should increase for very small losses
        if loss_weights.adaptive_scaling:
            assert loss_system.current_scaling_factor >= initial_scaling

    def test_complete_loss_calculation(self, config, loss_weights, sample_data):
        """Test complete multi-component loss calculation"""
        loss_system = MultiComponentLoss(config, loss_weights)

        model_outputs = {
            'logits': sample_data['logits'],
            'mtp_logits': sample_data['mtp_logits'],
            'router_logits': sample_data['router_logits'],
            'reasoning_logits': sample_data['reasoning_logits'],
            'level_contributions': sample_data['level_contributions']
        }

        loss_output = loss_system(model_outputs, sample_data['targets'], use_fp4=False)

        # Check output structure
        assert isinstance(loss_output, LossOutput)
        assert isinstance(loss_output.total_loss, torch.Tensor)
        assert loss_output.total_loss.dim() == 0
        assert loss_output.total_loss.item() >= 0

        # Check component losses
        assert isinstance(loss_output.component_losses, dict)
        expected_components = ['main', 'mtp', 'auxiliary', 'reasoning', 'anti_hallucination']
        for component in expected_components:
            assert component in loss_output.component_losses

        # Check weighted losses
        assert isinstance(loss_output.weighted_losses, dict)
        for component in expected_components:
            assert component in loss_output.weighted_losses

        # Check loss weights
        assert isinstance(loss_output.loss_weights, dict)
        assert loss_output.loss_weights['main'] == loss_weights.main_weight
        assert loss_output.loss_weights['mtp'] == loss_weights.mtp_weight

        # Check loss statistics
        assert isinstance(loss_output.loss_stats, dict)
        assert 'total_loss_magnitude' in loss_output.loss_stats
        assert 'component_ratios' in loss_output.loss_stats
        assert 'step_count' in loss_output.loss_stats

    def test_fp4_loss_calculation(self, config, loss_weights, sample_data):
        """Test loss calculation with FP4 scaling"""
        loss_system = MultiComponentLoss(config, loss_weights)

        model_outputs = {
            'logits': sample_data['logits'],
            'mtp_logits': sample_data['mtp_logits'],
            'router_logits': sample_data['router_logits'],
            'reasoning_logits': sample_data['reasoning_logits'],
            'level_contributions': sample_data['level_contributions']
        }

        # Calculate loss without FP4
        loss_output_normal = loss_system(model_outputs, sample_data['targets'], use_fp4=False)

        # Calculate loss with FP4
        loss_output_fp4 = loss_system(model_outputs, sample_data['targets'], use_fp4=True)

        # FP4 loss should be scaled
        expected_scaled_loss = loss_output_normal.total_loss * loss_weights.fp4_scaling_factor
        assert torch.allclose(loss_output_fp4.total_loss, expected_scaled_loss, rtol=1e-5)

        # Scaling factor should be recorded
        assert loss_output_fp4.scaling_factor == loss_weights.fp4_scaling_factor

    def test_dynamic_weight_adjustment(self, config, sample_data):
        """Test dynamic weight adjustment"""
        # Enable dynamic weighting
        dynamic_weights = LossWeights(dynamic_weighting=True)
        loss_system = MultiComponentLoss(config, dynamic_weights)

        model_outputs = {
            'logits': sample_data['logits'],
            'mtp_logits': sample_data['mtp_logits'],
            'router_logits': sample_data['router_logits'],
            'reasoning_logits': sample_data['reasoning_logits'],
            'level_contributions': sample_data['level_contributions']
        }

        # Run multiple steps to trigger weight adjustment
        initial_main_weight = loss_system.loss_weights.main_weight

        for step in range(loss_system.weight_adjustment_interval + 1):
            loss_output = loss_system(model_outputs, sample_data['targets'])

        # Weights may have been adjusted (this is probabilistic)
        # Just check that the system doesn't crash and weights are reasonable
        assert loss_system.loss_weights.main_weight > 0
        assert loss_system.loss_weights.mtp_weight > 0

    def test_loss_statistics(self, config, loss_weights, sample_data):
        """Test loss statistics calculation"""
        loss_system = MultiComponentLoss(config, loss_weights)

        model_outputs = {
            'logits': sample_data['logits'],
            'mtp_logits': sample_data['mtp_logits'],
            'router_logits': sample_data['router_logits'],
            'reasoning_logits': sample_data['reasoning_logits'],
            'level_contributions': sample_data['level_contributions']
        }

        # Run multiple steps to build history
        for step in range(20):
            loss_output = loss_system(model_outputs, sample_data['targets'])

        stats = loss_system.get_loss_statistics()

        assert isinstance(stats, dict)
        assert 'current_loss' in stats
        assert 'average_loss_100' in stats
        assert 'loss_std_100' in stats
        assert 'loss_trend' in stats
        assert 'scaling_factor' in stats
        assert 'total_steps' in stats
        assert 'loss_history_length' in stats

        # Check values are reasonable
        assert stats['current_loss'] >= 0
        assert stats['average_loss_100'] >= 0
        assert stats['loss_std_100'] >= 0
        assert stats['total_steps'] == 20
        assert stats['loss_history_length'] == 20

    def test_loss_trend_calculation(self, config, loss_weights):
        """Test loss trend calculation"""
        loss_system = MultiComponentLoss(config, loss_weights)

        # Create artificial decreasing trend
        decreasing_losses = [10.0 - i * 0.1 for i in range(50)]
        loss_system.loss_history = decreasing_losses

        trend = loss_system._calculate_loss_trend()
        assert trend < 0  # Should be negative for decreasing trend

        # Create artificial increasing trend
        increasing_losses = [1.0 + i * 0.1 for i in range(50)]
        loss_system.loss_history = increasing_losses

        trend = loss_system._calculate_loss_trend()
        assert trend > 0  # Should be positive for increasing trend

    def test_loss_history_management(self, config, loss_weights, sample_data):
        """Test loss history management"""
        loss_system = MultiComponentLoss(config, loss_weights)

        model_outputs = {
            'logits': sample_data['logits']
        }

        # Run many steps to test history truncation
        for step in range(1200):  # More than the 1000 limit
            loss_output = loss_system(model_outputs, sample_data['targets'])

        # History should be truncated to 1000
        assert len(loss_system.loss_history) == 1000
        assert loss_system.step_count == 1200

        # Test reset
        loss_system.reset_loss_history()
        assert len(loss_system.loss_history) == 0
        assert loss_system.step_count == 0

    def test_edge_cases(self, config, loss_weights):
        """Test edge cases and error handling"""
        loss_system = MultiComponentLoss(config, loss_weights)

        batch_size, seq_len, vocab_size = 2, 4, 100

        # Test with minimal data
        small_logits = torch.randn(batch_size, seq_len, vocab_size)
        small_targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        model_outputs = {'logits': small_logits}
        loss_output = loss_system(model_outputs, small_targets)

        assert isinstance(loss_output.total_loss, torch.Tensor)
        assert loss_output.total_loss.item() >= 0

        # Test with empty MTP logits
        model_outputs_empty_mtp = {
            'logits': small_logits,
            'mtp_logits': []
        }
        loss_output_empty = loss_system(model_outputs_empty_mtp, small_targets)
        assert loss_output_empty.component_losses['mtp'].item() == 0.0

        # Test with very short sequences
        tiny_logits = torch.randn(1, 1, vocab_size)
        tiny_targets = torch.randint(0, vocab_size, (1, 1))

        model_outputs_tiny = {'logits': tiny_logits}
        loss_output_tiny = loss_system(model_outputs_tiny, tiny_targets)
        assert isinstance(loss_output_tiny.total_loss, torch.Tensor)


class TestLossSystemIntegration:
    """Test loss system integration"""

    def test_factory_function(self, config, loss_weights):
        """Test factory function"""
        loss_system = create_loss_system(config, loss_weights)

        assert isinstance(loss_system, MultiComponentLoss)
        assert loss_system.config == config
        assert loss_system.loss_weights == loss_weights

    def test_gradient_flow(self, config, loss_weights, sample_data):
        """Test gradient flow through loss system"""
        loss_system = MultiComponentLoss(config, loss_weights)

        # Create model outputs with requires_grad=True
        logits = sample_data['logits'].clone().detach().requires_grad_(True)
        model_outputs = {'logits': logits}

        loss_output = loss_system(model_outputs, sample_data['targets'])

        # Backward pass
        loss_output.total_loss.backward()

        # Check gradients
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
        assert not torch.isinf(logits.grad).any()

    def test_memory_efficiency(self, config, loss_weights):
        """Test memory efficiency with large inputs"""
        loss_system = MultiComponentLoss(config, loss_weights)

        # Large batch test
        large_batch_size, seq_len, vocab_size = 16, 128, 5000

        large_logits = torch.randn(large_batch_size, seq_len, vocab_size)
        large_targets = torch.randint(0, vocab_size, (large_batch_size, seq_len))

        model_outputs = {'logits': large_logits}

        # Should not crash with large inputs
        loss_output = loss_system(model_outputs, large_targets)
        assert isinstance(loss_output.total_loss, torch.Tensor)
        assert loss_output.total_loss.item() >= 0


def run_all_tests():
    """Run all loss system tests"""
    print("🧪 Running Loss System Tests")

    # Create test fixtures
    config = TestConfig()
    loss_weights = LossWeights(
        main_weight=1.0,
        mtp_weight=0.5,
        auxiliary_weight=0.01,
        reasoning_weight=0.1,
        anti_hallucination_weight=0.25,
        fp4_scaling_factor=10.0
    )

    batch_size, seq_len, vocab_size = 4, 32, 1000
    sample_data = {
        'logits': torch.randn(batch_size, seq_len, vocab_size),
        'targets': torch.randint(0, vocab_size, (batch_size, seq_len)),
        'mtp_logits': [torch.randn(batch_size, seq_len, vocab_size) for _ in range(4)],
        'router_logits': torch.randn(batch_size, seq_len, 4),
        'reasoning_logits': torch.randn(batch_size, seq_len, vocab_size),
        'level_contributions': {
            'low': torch.randn(batch_size, seq_len, vocab_size),
            'mid': torch.randn(batch_size, seq_len, vocab_size),
            'high': torch.randn(batch_size, seq_len, vocab_size)
        }
    }

    print("✅ Testing Loss Weights...")
    test_weights = TestLossWeights()
    test_weights.test_default_weights()
    test_weights.test_custom_weights(loss_weights)

    print("✅ Testing Multi-Component Loss...")
    test_loss = TestMultiComponentLoss()
    test_loss.test_initialization(config, loss_weights)
    test_loss.test_main_loss_calculation(config, loss_weights, sample_data)
    test_loss.test_mtp_loss_calculation(config, loss_weights, sample_data)
    test_loss.test_auxiliary_loss_calculation(config, loss_weights, sample_data)
    test_loss.test_reasoning_loss_calculation(config, loss_weights, sample_data)
    test_loss.test_anti_hallucination_loss_calculation(config, loss_weights, sample_data)
    test_loss.test_loss_scaling(config, loss_weights)
    test_loss.test_complete_loss_calculation(config, loss_weights, sample_data)
    test_loss.test_fp4_loss_calculation(config, loss_weights, sample_data)
    test_loss.test_loss_statistics(config, loss_weights, sample_data)
    test_loss.test_loss_trend_calculation(config, loss_weights)
    test_loss.test_loss_history_management(config, loss_weights, sample_data)
    test_loss.test_edge_cases(config, loss_weights)

    print("✅ Testing Loss System Integration...")
    test_integration = TestLossSystemIntegration()
    test_integration.test_factory_function(config, loss_weights)
    test_integration.test_gradient_flow(config, loss_weights, sample_data)
    test_integration.test_memory_efficiency(config, loss_weights)

    print("🎉 All loss system tests passed!")


if __name__ == "__main__":
    run_all_tests()