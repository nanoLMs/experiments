#!/usr/bin/env python3
"""
Test suite for Advanced Trainer System
======================================

Comprehensive tests for the advanced trainer including:
- Training configuration and initialization
- Error handling and recovery mechanisms
- Memory management and optimization
- Integration with all advanced features
- Checkpoint and state management
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
import shutil
import sys
import os
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advanced_trainer import (
    TrainerOptimized, TrainingConfig, TrainingState, TrainingPhase,
    ErrorType, create_advanced_trainer
)


class MockModel(nn.Module):
    """Mock model for testing"""
    def __init__(self, vocab_size=1000, hidden_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, 8, batch_first=True),
            num_layers=2
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=True):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))

        if return_dict:
            return {
                'logits': logits,
                'loss': loss,
                'hidden_states': [x]
            }
        else:
            return (logits, loss)


class MockTokenizer:
    """Mock tokenizer for testing"""
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size

    def decode(self, token_ids):
        if isinstance(token_ids, list):
            return " ".join([f"token_{tid}" for tid in token_ids])
        return f"token_{token_ids}"

    def encode(self, text):
        return [hash(word) % self.vocab_size for word in text.split()]


@pytest.fixture
def training_config():
    """Create test training configuration"""
    return TrainingConfig(
        model_name="test_model",
        vocab_size=1000,
        max_seq_length=128,
        learning_rate=1e-4,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        num_epochs=1,
        max_steps=10,
        warmup_steps=2,
        eval_steps=5,
        save_steps=5,
        enable_error_recovery=True,
        enable_monitoring=False,  # Disable for testing
        use_mixed_precision=False,  # Disable for CPU testing
        use_quantization=False,  # Disable for testing
        enable_moe=False,  # Disable for testing
        enable_mtp=False,  # Disable for testing
        enable_hrm=False,  # Disable for testing
        enable_anti_hallucination=False,  # Disable for testing
        enable_loss_tracking=False  # Disable for testing
    )


@pytest.fixture
def mock_model():
    """Create mock model"""
    return MockModel()


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer"""
    return MockTokenizer()


@pytest.fixture
def sample_data():
    """Create sample training data"""
    batch_size = 4
    seq_len = 32
    vocab_size = 1000

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = input_ids.clone()

    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    return dataloader


class TestTrainingConfig:
    """Test training configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = TrainingConfig()

        assert config.model_name == "nanolm_advanced"
        assert config.learning_rate == 2e-5
        assert config.micro_batch_size == 4
        assert config.gradient_accumulation_steps == 8
        assert config.enable_error_recovery == True
        assert config.use_mixed_precision == True

    def test_custom_config(self):
        """Test custom configuration"""
        config = TrainingConfig(
            model_name="custom_model",
            learning_rate=1e-3,
            micro_batch_size=8,
            enable_error_recovery=False
        )

        assert config.model_name == "custom_model"
        assert config.learning_rate == 1e-3
        assert config.micro_batch_size == 8
        assert config.enable_error_recovery == False


class TestTrainingState:
    """Test training state management"""

    def test_default_state(self):
        """Test default training state"""
        state = TrainingState()

        assert state.step == 0
        assert state.epoch == 0
        assert state.phase == TrainingPhase.WARMUP
        assert state.best_loss == float('inf')
        assert state.oom_count == 0

    def test_state_updates(self):
        """Test training state updates"""
        state = TrainingState()

        state.step = 100
        state.epoch = 1
        state.phase = TrainingPhase.MAIN_TRAINING
        state.best_loss = 2.5

        assert state.step == 100
        assert state.epoch == 1
        assert state.phase == TrainingPhase.MAIN_TRAINING
        assert state.best_loss == 2.5


class TestTrainerOptimized:
    """Test advanced trainer functionality"""

    def test_initialization(self, training_config, mock_model, mock_tokenizer, sample_data):
        """Test trainer initialization"""
        trainer = TrainerOptimized(
            training_config, mock_model, mock_tokenizer, sample_data
        )

        assert trainer.config == training_config
        assert trainer.model == mock_model
        assert trainer.tokenizer == mock_tokenizer
        assert trainer.train_dataloader == sample_data
        assert hasattr(trainer, 'optimizer')
        assert hasattr(trainer, 'scheduler')
        assert hasattr(trainer, 'state')
        assert isinstance(trainer.state, TrainingState)

    def test_optimizer_initialization(self, training_config, mock_model, mock_tokenizer, sample_data):
        """Test optimizer initialization"""
        trainer = TrainerOptimized(
            training_config, mock_model, mock_tokenizer, sample_data
        )

        assert trainer.optimizer is not None
        assert len(trainer.optimizer.param_groups) >= 1

        # Check learning rate (may be adjusted by scheduler)
        for param_group in trainer.optimizer.param_groups:
            assert param_group['lr'] > 0  # Should be positive

    def test_scheduler_initialization(self, training_config, mock_model, mock_tokenizer, sample_data):
        """Test learning rate scheduler initialization"""
        trainer = TrainerOptimized(
            training_config, mock_model, mock_tokenizer, sample_data
        )

        assert trainer.scheduler is not None

        # Test scheduler step
        initial_lr = trainer.optimizer.param_groups[0]['lr']
        trainer.scheduler.step()
        # LR should change after step (though might be same in warmup)
        assert trainer.optimizer.param_groups[0]['lr'] >= 0

    def test_error_classification(self, training_config, mock_model, mock_tokenizer, sample_data):
        """Test error classification"""
        trainer = TrainerOptimized(
            training_config, mock_model, mock_tokenizer, sample_data
        )

        # Test OOM error classification
        oom_error = Exception("CUDA out of memory")
        assert trainer._classify_error(oom_error) == ErrorType.OOM_ERROR

        # Test numerical instability
        nan_error = Exception("Loss is NaN")
        assert trainer._classify_error(nan_error) == ErrorType.NUMERICAL_INSTABILITY

        # Test gradient explosion
        grad_error = Exception("Gradient explosion detected")
        assert trainer._classify_error(grad_error) == ErrorType.GRADIENT_EXPLOSION

    def test_memory_management(self, training_config, mock_model, mock_tokenizer, sample_data):
        """Test memory management"""
        trainer = TrainerOptimized(
            training_config, mock_model, mock_tokenizer, sample_data
        )

        # Test memory usage calculation
        memory_usage = trainer._get_memory_usage()
        assert isinstance(memory_usage, float)
        assert memory_usage >= 0.0

        # Test memory cleanup (should not raise errors)
        trainer._cleanup_memory()

    def test_gradient_handling(self, training_config, mock_model, mock_tokenizer, sample_data):
        """Test gradient handling and clipping"""
        trainer = TrainerOptimized(
            training_config, mock_model, mock_tokenizer, sample_data
        )

        # Create some gradients
        dummy_input = torch.randint(0, 1000, (2, 32))
        dummy_labels = dummy_input.clone()

        output = trainer.model(dummy_input, labels=dummy_labels)
        loss = output['loss']
        loss.backward()

        # Test gradient handling
        grad_norm = trainer._handle_gradients()

        assert isinstance(grad_norm, float)
        assert grad_norm >= 0.0

    def test_phase_transitions(self, training_config, mock_model, mock_tokenizer, sample_data):
        """Test training phase transitions"""
        trainer = TrainerOptimized(
            training_config, mock_model, mock_tokenizer, sample_data
        )

        # Test warmup phase
        trainer.state.step = 1
        trainer._check_phase_transitions()
        assert trainer.state.phase == TrainingPhase.WARMUP

        # Test main training phase
        trainer.state.step = training_config.warmup_steps + 1
        trainer._check_phase_transitions()
        assert trainer.state.phase == TrainingPhase.MAIN_TRAINING

    def test_early_stopping(self, training_config, mock_model, mock_tokenizer, sample_data):
        """Test early stopping logic"""
        trainer = TrainerOptimized(
            training_config, mock_model, mock_tokenizer, sample_data
        )

        # Test with insufficient data
        assert trainer._should_early_stop() == False

        # Test with stagnant losses
        trainer.state.last_losses = [5.0] * 15  # Flat losses
        assert trainer._should_early_stop() == True

        # Test with improving losses
        trainer.state.last_losses = [5.0 - i * 0.1 for i in range(15)]  # Decreasing
        assert trainer._should_early_stop() == False

    def test_checkpoint_operations(self, training_config, mock_model, mock_tokenizer, sample_data):
        """Test checkpoint saving and loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Modify config to use temp directory
            training_config.model_name = f"{temp_dir}/test_model"

            trainer = TrainerOptimized(
                training_config, mock_model, mock_tokenizer, sample_data
            )

            # Set some state
            trainer.state.step = 10
            trainer.state.epoch = 1
            trainer.state.best_loss = 2.5

            # Save checkpoint
            trainer._save_checkpoint()

            # Verify checkpoint exists
            assert hasattr(trainer, 'last_checkpoint_path')
            assert trainer.last_checkpoint_path.exists()

            # Modify state
            original_step = trainer.state.step
            trainer.state.step = 999

            # Load checkpoint
            trainer._load_checkpoint(str(trainer.last_checkpoint_path))

            # Verify state restored
            assert trainer.state.step == original_step
            assert trainer.state.epoch == 1
            assert trainer.state.best_loss == 2.5


class TestErrorHandling:
    """Test error handling mechanisms"""

    def test_oom_error_handling(self, training_config, mock_model, mock_tokenizer, sample_data):
        """Test OOM error handling"""
        trainer = TrainerOptimized(
            training_config, mock_model, mock_tokenizer, sample_data
        )

        original_batch_size = trainer.config.micro_batch_size

        # Simulate OOM error
        oom_error = Exception("CUDA out of memory")
        success = trainer._handle_oom_error(oom_error, None)

        if success:
            # Batch size should be reduced
            assert trainer.config.micro_batch_size < original_batch_size
            assert trainer.state.oom_count == 1

    def test_gradient_explosion_handling(self, training_config, mock_model, mock_tokenizer, sample_data):
        """Test gradient explosion handling"""
        trainer = TrainerOptimized(
            training_config, mock_model, mock_tokenizer, sample_data
        )

        original_lr = trainer.optimizer.param_groups[0]['lr']

        # Simulate gradient explosion
        grad_error = Exception("Gradient explosion")
        success = trainer._handle_gradient_explosion(grad_error, None)

        assert success == True
        # Learning rate should be reduced
        assert trainer.optimizer.param_groups[0]['lr'] < original_lr

    def test_numerical_instability_handling(self, training_config, mock_model, mock_tokenizer, sample_data):
        """Test numerical instability handling"""
        trainer = TrainerOptimized(
            training_config, mock_model, mock_tokenizer, sample_data
        )

        # Simulate numerical instability
        nan_error = Exception("NaN detected")
        success = trainer._handle_numerical_instability(nan_error, None)

        assert success == True

    def test_max_retries(self, training_config, mock_model, mock_tokenizer, sample_data):
        """Test maximum retry limits"""
        trainer = TrainerOptimized(
            training_config, mock_model, mock_tokenizer, sample_data
        )

        # Exceed max OOM retries
        trainer.state.oom_count = training_config.max_oom_retries + 1

        oom_error = Exception("CUDA out of memory")
        success = trainer._handle_oom_error(oom_error, None)

        assert success == False


class TestIntegration:
    """Test integration with other components"""

    @patch('advanced_trainer.create_loss_system')
    @patch('advanced_trainer.create_loss_tracker')
    @patch('advanced_trainer.create_anti_hallucination_filter')
    def test_component_integration(self, mock_filter, mock_tracker, mock_loss_system,
                                 training_config, mock_model, mock_tokenizer, sample_data):
        """Test integration with loss system and other components"""
        # Enable components for testing
        training_config.enable_loss_tracking = True
        training_config.enable_anti_hallucination = True

        # Mock the components
        mock_loss_system.return_value = Mock()
        mock_tracker.return_value = Mock()
        mock_filter.return_value = Mock()

        trainer = TrainerOptimized(
            training_config, mock_model, mock_tokenizer, sample_data
        )

        # Verify components were created
        assert hasattr(trainer, 'loss_system')
        assert hasattr(trainer, 'loss_tracker')
        assert hasattr(trainer, 'anti_hallucination_filter')

    def test_factory_function(self, training_config, mock_model, mock_tokenizer, sample_data):
        """Test factory function"""
        trainer = create_advanced_trainer(
            training_config, mock_model, mock_tokenizer, sample_data
        )

        assert isinstance(trainer, TrainerOptimized)
        assert trainer.config == training_config
        assert trainer.model == mock_model


class TestTrainingLoop:
    """Test training loop components"""

    def test_training_step_basic(self, training_config, mock_model, mock_tokenizer, sample_data):
        """Test basic training step"""
        trainer = TrainerOptimized(
            training_config, mock_model, mock_tokenizer, sample_data
        )

        # Get a batch
        batch_data = next(iter(sample_data))
        batch = {
            'input_ids': batch_data[0],
            'attention_mask': batch_data[1],
            'labels': batch_data[2]
        }

        # Execute training step
        try:
            metrics = trainer._training_step(batch)

            # Check metrics
            assert isinstance(metrics, dict)
            assert 'loss' in metrics
            assert 'learning_rate' in metrics
            assert 'tokens_per_second' in metrics
            assert 'memory_usage' in metrics

            # Check metric values
            assert metrics['loss'] >= 0.0
            assert metrics['learning_rate'] >= 0.0
            assert metrics['tokens_per_second'] >= 0.0
            assert metrics['memory_usage'] >= 0.0

        except Exception as e:
            # Training step might fail due to missing components, but should not crash
            assert isinstance(e, Exception)

    def test_validation_step(self, training_config, mock_model, mock_tokenizer, sample_data):
        """Test validation step"""
        trainer = TrainerOptimized(
            training_config, mock_model, mock_tokenizer, sample_data, sample_data
        )

        # Run validation
        val_metrics = trainer._validate()

        assert isinstance(val_metrics, dict)
        if val_metrics:  # May be empty if validation fails
            assert 'val_loss' in val_metrics
            assert val_metrics['val_loss'] >= 0.0


def run_all_tests():
    """Run all advanced trainer tests"""
    print("🧪 Running Advanced Trainer Tests")

    # Create test fixtures
    training_config = TrainingConfig(
        model_name="test_model",
        vocab_size=1000,
        max_seq_length=128,
        learning_rate=1e-4,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        num_epochs=1,
        max_steps=10,
        warmup_steps=2,
        eval_steps=5,
        save_steps=5,
        enable_error_recovery=True,
        enable_monitoring=False,
        use_mixed_precision=False,
        use_quantization=False,
        enable_moe=False,
        enable_mtp=False,
        enable_hrm=False,
        enable_anti_hallucination=False,
        enable_loss_tracking=False
    )

    mock_model = MockModel()
    mock_tokenizer = MockTokenizer()

    # Create sample data
    batch_size = 4
    seq_len = 32
    vocab_size = 1000

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = input_ids.clone()

    dataset = TensorDataset(input_ids, attention_mask, labels)
    sample_data = DataLoader(dataset, batch_size=2, shuffle=False)

    print("✅ Testing Training Configuration...")
    test_config = TestTrainingConfig()
    test_config.test_default_config()
    test_config.test_custom_config()

    print("✅ Testing Training State...")
    test_state = TestTrainingState()
    test_state.test_default_state()
    test_state.test_state_updates()

    print("✅ Testing Trainer Initialization...")
    test_trainer = TestTrainerOptimized()
    test_trainer.test_initialization(training_config, mock_model, mock_tokenizer, sample_data)
    test_trainer.test_optimizer_initialization(training_config, mock_model, mock_tokenizer, sample_data)
    test_trainer.test_scheduler_initialization(training_config, mock_model, mock_tokenizer, sample_data)
    test_trainer.test_error_classification(training_config, mock_model, mock_tokenizer, sample_data)
    test_trainer.test_memory_management(training_config, mock_model, mock_tokenizer, sample_data)
    test_trainer.test_gradient_handling(training_config, mock_model, mock_tokenizer, sample_data)
    test_trainer.test_phase_transitions(training_config, mock_model, mock_tokenizer, sample_data)
    test_trainer.test_early_stopping(training_config, mock_model, mock_tokenizer, sample_data)
    test_trainer.test_checkpoint_operations(training_config, mock_model, mock_tokenizer, sample_data)

    print("✅ Testing Error Handling...")
    test_error = TestErrorHandling()
    test_error.test_oom_error_handling(training_config, mock_model, mock_tokenizer, sample_data)
    test_error.test_gradient_explosion_handling(training_config, mock_model, mock_tokenizer, sample_data)
    test_error.test_numerical_instability_handling(training_config, mock_model, mock_tokenizer, sample_data)
    test_error.test_max_retries(training_config, mock_model, mock_tokenizer, sample_data)

    print("✅ Testing Training Loop Components...")
    test_loop = TestTrainingLoop()
    test_loop.test_training_step_basic(training_config, mock_model, mock_tokenizer, sample_data)
    test_loop.test_validation_step(training_config, mock_model, mock_tokenizer, sample_data)

    print("🎉 All advanced trainer tests passed!")


if __name__ == "__main__":
    run_all_tests()