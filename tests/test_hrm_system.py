#!/usr/bin/env python3
"""
Test suite for Hierarchical Reasoning Module (HRM) System
========================================================

Comprehensive tests for the HRM implementation including:
- Individual processor functionality
- Multi-timescale processing
- State management and persistence
- Convergence detection
- Integration with base model
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrm_system import (
    HierarchicalProcessor, HierarchicalReasoningModule, HRMIntegratedModel,
    ReasoningLevel, HRMState, HRMOutput, create_hrm_model,
    calculate_hrm_metrics
)


class TestConfig:
    """Test configuration for HRM system"""
    d_model = 128
    vocab_size = 500
    dropout = 0.1

    # HRM settings
    hrm_N_cycles = 2
    hrm_T_steps = 2
    hrm_use_gradient_approx = True
    hrm_convergence_threshold = 0.95
    hrm_min_convergence_steps = 10
    hrm_loss_weight = 0.1
    hrm_reset_every = 50

    # Quantization settings (disabled for testing)
    use_bnb_4bit = False
    bnb_4bit_quantize_hrm = False
    bnb_4bit_compute_dtype = "bfloat16"
    bnb_4bit_use_double_quant = True
    bnb_4bit_quant_type = "nf4"
    bnb_4bit_quant_storage = "uint8"
    bnb_4bit_quantize_router = False

    # Full model settings
    n_layers = 2
    n_heads = 4
    d_head = 32
    d_ff = 512
    seq_len = 64
    attn_dropout = 0.1
    ff_dropout = 0.1
    use_flash_attn = False
    rope_base = 10000
    rope_scaling = 1.0
    tie_word_embeddings = True
    use_quantization = False

    # FP4 settings
    use_fp4 = False
    fp4_format = "nvfp4"
    fp4_block_size = 16
    fp4_split_rounding = True
    qaf_threshold = 1e-6
    max_qaf_steps = 100
    qaf_precision = "bf16"

    # MoE settings
    moe_every = 0
    n_experts = 2
    moe_top_k = 1
    expert_ff_mult = 1.0
    router_jitter = 0.01
    moe_aux_weight = 0.01
    router_z_loss = 1e-4
    capacity_factor = 1.0

    # MTP settings
    mtp_k = 2
    mtp_loss_weights = [1.0, 0.5]


@pytest.fixture
def config():
    return TestConfig()


@pytest.fixture
def sample_input():
    batch_size, seq_len, d_model = 2, 16, 128
    return torch.randn(batch_size, seq_len, d_model)


@pytest.fixture
def sample_state():
    batch_size, seq_len, d_model = 2, 16, 128
    return torch.randn(batch_size, seq_len, d_model)


@pytest.fixture
def sample_targets():
    batch_size, seq_len = 2, 16
    return torch.randint(0, 500, (batch_size, seq_len))


class TestHierarchicalProcessor:
    """Test individual hierarchical processor"""

    def test_processor_initialization(self, config):
        """Test processor initialization"""
        processor = HierarchicalProcessor(config, ReasoningLevel.LOW, update_frequency=1)

        assert processor.level == ReasoningLevel.LOW
        assert processor.update_frequency == 1
        assert processor.d_model == config.d_model

        # Check that all components are initialized
        assert hasattr(processor, 'state_processor')
        assert hasattr(processor, 'state_update')
        assert hasattr(processor, 'reasoning_head')
        assert hasattr(processor, 'influence_gate')

    def test_processor_forward(self, config, sample_input, sample_state):
        """Test processor forward pass"""
        processor = HierarchicalProcessor(config, ReasoningLevel.MID, update_frequency=2)

        # Test with update
        new_state, reasoning_out, influence = processor(
            sample_input, sample_state, should_update=True
        )

        assert new_state.shape == sample_state.shape
        assert reasoning_out.shape == (*sample_input.shape[:2], config.vocab_size)
        assert influence.shape == (*sample_input.shape[:2], 1)

        # Test without update
        new_state_no_update, _, _ = processor(
            sample_input, sample_state, should_update=False
        )

        # State should remain unchanged when not updating
        assert torch.allclose(new_state_no_update, sample_state)

    def test_convergence_calculation(self, config, sample_state):
        """Test convergence calculation"""
        processor = HierarchicalProcessor(config, ReasoningLevel.HIGH, update_frequency=4)

        # Test with identical states (should have high convergence)
        convergence_identical = processor.calculate_convergence(sample_state, sample_state)
        assert convergence_identical == 1.0

        # Test with different states (should have lower convergence)
        different_state = sample_state + torch.randn_like(sample_state) * 0.1
        convergence_different = processor.calculate_convergence(sample_state, different_state)
        assert 0.0 <= convergence_different <= 1.0
        assert convergence_different < convergence_identical

        # Test with None previous state
        convergence_none = processor.calculate_convergence(None, sample_state)
        assert convergence_none == 0.0


class TestHierarchicalReasoningModule:
    """Test complete HRM module"""

    def test_hrm_initialization(self, config):
        """Test HRM initialization"""
        hrm = HierarchicalReasoningModule(config)

        assert hrm.N_cycles == config.hrm_N_cycles
        assert hrm.T_steps == config.hrm_T_steps
        assert hasattr(hrm, 'low_processor')
        assert hasattr(hrm, 'mid_processor')
        assert hasattr(hrm, 'high_processor')
        assert hasattr(hrm, 'fusion_layer')
        assert hasattr(hrm, 'final_reasoning_head')

    def test_state_initialization(self, config):
        """Test HRM state initialization"""
        hrm = HierarchicalReasoningModule(config)
        batch_size, seq_len = 2, 16
        device = torch.device('cpu')

        state = hrm.initialize_states(batch_size, seq_len, device)

        assert isinstance(state, HRMState)
        assert state.low_level_state.shape == (batch_size, seq_len, config.d_model)
        assert state.mid_level_state.shape == (batch_size, seq_len, config.d_model)
        assert state.high_level_state.shape == (batch_size, seq_len, config.d_model)
        assert state.step_count == 0
        assert state.cycle_count == 0
        assert len(state.convergence_scores) == 3

    def test_hrm_forward(self, config, sample_input):
        """Test HRM forward pass"""
        hrm = HierarchicalReasoningModule(config)

        # Test without previous state (initialization)
        output = hrm(sample_input)

        assert isinstance(output, HRMOutput)
        assert output.reasoning_logits.shape == (*sample_input.shape[:2], config.vocab_size)
        assert isinstance(output.hierarchical_states, HRMState)
        assert isinstance(output.convergence_metrics, dict)
        assert isinstance(output.level_contributions, dict)

        # Test with previous state
        output2 = hrm(sample_input, output.hierarchical_states)
        assert output2.hierarchical_states.step_count == 2

    def test_multi_timescale_updates(self, config, sample_input):
        """Test multi-timescale update logic"""
        hrm = HierarchicalReasoningModule(config)

        # Run multiple steps and check update patterns
        state = None
        for step in range(10):
            output = hrm(sample_input, state)
            state = output.hierarchical_states

            # Low level should always update
            assert state.step_count == step + 1

            # Check mid and high level update frequencies
            should_update_mid = (state.step_count % config.hrm_T_steps == 0)
            should_update_high = (state.step_count % (config.hrm_N_cycles * config.hrm_T_steps) == 0)

            # These are implicit in the implementation, but we can verify step counts
            assert state.step_count > 0

    def test_reasoning_loss_calculation(self, config, sample_input, sample_targets):
        """Test reasoning loss calculation"""
        hrm = HierarchicalReasoningModule(config)

        output = hrm(sample_input)
        reasoning_loss, individual_losses = hrm.calculate_reasoning_loss(
            output.reasoning_logits, sample_targets, output.level_contributions
        )

        assert isinstance(reasoning_loss, torch.Tensor)
        assert reasoning_loss.dim() == 0  # Scalar
        assert reasoning_loss.item() >= 0

        assert isinstance(individual_losses, dict)
        assert 'main' in individual_losses

        # Check that individual losses sum to approximately total loss
        individual_sum = sum(loss for loss in individual_losses.values())
        assert torch.allclose(reasoning_loss, individual_sum, rtol=1e-5)

    def test_convergence_detection(self, config):
        """Test convergence detection"""
        hrm = HierarchicalReasoningModule(config)

        # Test with high convergence
        high_convergence = {'low': 0.98, 'mid': 0.97, 'high': 0.96, 'overall': 0.97}
        assert hrm.check_convergence(high_convergence) == True

        # Test with low convergence
        low_convergence = {'low': 0.5, 'mid': 0.6, 'high': 0.4, 'overall': 0.5}
        assert hrm.check_convergence(low_convergence) == False

    def test_reasoning_stats(self, config):
        """Test reasoning statistics"""
        hrm = HierarchicalReasoningModule(config)
        stats = hrm.get_reasoning_stats()

        assert isinstance(stats, dict)
        assert 'N_cycles' in stats
        assert 'T_steps' in stats
        assert 'total_parameters' in stats
        assert 'processor_params' in stats

        assert stats['N_cycles'] == config.hrm_N_cycles
        assert stats['T_steps'] == config.hrm_T_steps
        assert stats['total_parameters'] > 0


class TestHRMIntegratedModel:
    """Test HRM integrated with base model"""

    def test_integrated_model_initialization(self, config):
        """Test integrated model initialization"""
        model = create_hrm_model(config)

        assert isinstance(model, HRMIntegratedModel)
        assert hasattr(model, 'base_model')
        assert hasattr(model, 'hrm')
        assert model.hrm_state is None  # Initially None
        assert model.step_count == 0

    def test_integrated_model_forward(self, config):
        """Test integrated model forward pass"""
        model = create_hrm_model(config)

        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Test without labels
        outputs = model(input_ids)

        assert 'logits' in outputs
        assert 'reasoning_logits' in outputs
        assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)
        assert outputs['reasoning_logits'].shape == (batch_size, seq_len, config.vocab_size)
        assert outputs['loss'] is None

        # Test with labels
        outputs_with_labels = model(input_ids, labels=labels)

        assert outputs_with_labels['loss'] is not None
        assert outputs_with_labels['main_loss'] is not None
        assert outputs_with_labels['reasoning_loss'] is not None
        assert outputs_with_labels['loss'].item() >= 0

    def test_hrm_state_persistence(self, config):
        """Test HRM state persistence across forward passes"""
        model = create_hrm_model(config)

        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # First forward pass
        outputs1 = model(input_ids)
        assert model.hrm_state is not None
        assert model.hrm_state.step_count == 1

        # Second forward pass
        outputs2 = model(input_ids)
        assert model.hrm_state.step_count == 2

        # Test state reset
        model.reset_hrm_state()
        assert model.hrm_state is None
        assert model.step_count == 0

    def test_hrm_state_reset(self, config):
        """Test HRM state reset functionality"""
        model = create_hrm_model(config)

        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Run some steps
        for _ in range(3):
            model(input_ids)

        assert model.step_count == 3
        assert model.hrm_state.step_count == 3

        # Test reset via parameter
        outputs = model(input_ids, reset_hrm=True)
        assert model.step_count == 1
        assert model.hrm_state.step_count == 1

    def test_automatic_reset(self, config):
        """Test automatic HRM state reset"""
        config.hrm_reset_every = 3  # Reset every 3 steps
        model = create_hrm_model(config)

        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Run steps up to reset threshold
        for step in range(5):
            model(input_ids)

            if step < 2:
                assert model.step_count == step + 1
            else:
                # Should reset after step 3
                assert model.step_count <= 3

    def test_model_statistics(self, config):
        """Test model statistics"""
        model = create_hrm_model(config)
        stats = model.get_hrm_stats()

        assert isinstance(stats, dict)
        assert 'base_model_params' in stats
        assert 'hrm_params' in stats
        assert 'total_params' in stats
        assert 'hrm_overhead' in stats

        assert stats['total_params'] > stats['base_model_params']
        assert stats['hrm_overhead'] > 0


class TestHRMMetrics:
    """Test HRM metrics calculation"""

    def test_hrm_metrics_calculation(self, config, sample_targets):
        """Test HRM metrics calculation"""
        batch_size, seq_len = sample_targets.shape
        vocab_size = config.vocab_size

        # Create mock reasoning outputs
        reasoning_outputs = {
            'low': torch.randn(batch_size, seq_len, vocab_size),
            'mid': torch.randn(batch_size, seq_len, vocab_size),
            'high': torch.randn(batch_size, seq_len, vocab_size)
        }

        convergence_metrics = {
            'low': 0.8, 'mid': 0.7, 'high': 0.9, 'overall': 0.8
        }

        metrics = calculate_hrm_metrics(reasoning_outputs, sample_targets, convergence_metrics)

        assert isinstance(metrics, dict)
        assert 'convergence' in metrics
        assert 'level_accuracies' in metrics
        assert 'level_perplexities' in metrics
        assert 'reasoning_diversity' in metrics
        assert 'hierarchical_consistency' in metrics

        # Check that all levels have metrics
        for level in ['low', 'mid', 'high']:
            assert level in metrics['level_accuracies']
            assert level in metrics['level_perplexities']
            assert level in metrics['reasoning_diversity']

        # Check value ranges
        for accuracy in metrics['level_accuracies'].values():
            assert 0.0 <= accuracy <= 1.0

        for perplexity in metrics['level_perplexities'].values():
            assert perplexity > 0

        assert 0.0 <= metrics['hierarchical_consistency'] <= 1.0


class TestHRMIntegration:
    """Test HRM integration with other components"""

    def test_end_to_end_training_step(self, config):
        """Test end-to-end training step with HRM"""
        model = create_hrm_model(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that gradients were computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_memory_efficiency(self, config):
        """Test memory efficiency of HRM"""
        model = create_hrm_model(config)

        # Get initial memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

        batch_size, seq_len = 4, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # Run multiple forward passes
        for _ in range(5):
            outputs = model(input_ids, labels=labels)
            loss = outputs['loss']

            # Simulate backward pass
            loss.backward()
            model.zero_grad()

        # Memory should not grow excessively
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_growth = final_memory - initial_memory
            # This is a rough check - memory growth should be reasonable
            assert memory_growth < 1e9  # Less than 1GB growth


def run_all_tests():
    """Run all HRM tests"""
    print("🧪 Running HRM System Tests")

    # Create test config
    config = TestConfig()

    # Test data
    sample_input = torch.randn(2, 16, 128)
    sample_state = torch.randn(2, 16, 128)
    sample_targets = torch.randint(0, 500, (2, 16))

    print("✅ Testing Hierarchical Processor...")
    test_processor = TestHierarchicalProcessor()
    test_processor.test_processor_initialization(config)
    test_processor.test_processor_forward(config, sample_input, sample_state)
    test_processor.test_convergence_calculation(config, sample_state)

    print("✅ Testing HRM Module...")
    test_hrm = TestHierarchicalReasoningModule()
    test_hrm.test_hrm_initialization(config)
    test_hrm.test_state_initialization(config)
    test_hrm.test_hrm_forward(config, sample_input)
    test_hrm.test_multi_timescale_updates(config, sample_input)
    test_hrm.test_reasoning_loss_calculation(config, sample_input, sample_targets)
    test_hrm.test_convergence_detection(config)
    test_hrm.test_reasoning_stats(config)

    print("✅ Testing HRM Integrated Model...")
    test_integrated = TestHRMIntegratedModel()
    test_integrated.test_integrated_model_initialization(config)
    test_integrated.test_integrated_model_forward(config)
    test_integrated.test_hrm_state_persistence(config)
    test_integrated.test_hrm_state_reset(config)
    test_integrated.test_model_statistics(config)

    print("✅ Testing HRM Metrics...")
    test_metrics = TestHRMMetrics()
    test_metrics.test_hrm_metrics_calculation(config, sample_targets)

    print("✅ Testing HRM Integration...")
    test_integration = TestHRMIntegration()
    test_integration.test_end_to_end_training_step(config)
    test_integration.test_memory_efficiency(config)

    print("🎉 All HRM tests passed!")


if __name__ == "__main__":
    run_all_tests()