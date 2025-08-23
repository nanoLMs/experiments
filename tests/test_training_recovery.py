#!/usr/bin/env python3
"""
Test suite for Training Recovery and Checkpointing System
========================================================

Comprehensive tests for the checkpoint and recovery system including:
- Checkpoint creation and validation
- Training state preservation and recovery
- Automatic checkpointing triggers
- Emergency save functionality
- Checkpoint cleanup and management
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import tempfile
import shutil
import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_recovery import (
    CheckpointManager, TrainingRecovery, AutoCheckpointer,
    CheckpointConfig, TrainingState, CheckpointMetadata,
    CheckpointType, RecoveryStrategy, create_checkpoint_system
)


class TestModel(nn.Module):
    """Simple test model"""

    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def checkpoint_config(temp_dir):
    """Create test checkpoint configuration"""
    return CheckpointConfig(
        checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
        backup_dir=os.path.join(temp_dir, "backups"),
        save_every_n_steps=10,
        save_every_n_epochs=1,
        save_every_n_minutes=1,
        keep_last_n_checkpoints=3,
        keep_best_n_checkpoints=2,
        verify_checkpoints=True,
        auto_recovery=True,
        max_recovery_attempts=2
    )


@pytest.fixture
def test_model():
    """Create test model"""
    return TestModel()


@pytest.fixture
def test_optimizer(test_model):
    """Create test optimizer"""
    return optim.Adam(test_model.parameters(), lr=0.001)


@pytest.fixture
def test_scheduler(test_optimizer):
    """Create test scheduler"""
    return optim.lr_scheduler.StepLR(test_optimizer, step_size=5, gamma=0.9)


@pytest.fixture
def test_training_state():
    """Create test training state"""
    return TrainingState(
        epoch=3,
        step=50,
        global_step=150,
        best_loss=0.8,
        best_val_loss=0.9,
        training_start_time=time.time() - 3600,
        total_training_time=3600,
        loss_history=[1.5, 1.2, 1.0, 0.8],
        val_loss_history=[1.6, 1.3, 1.1, 0.9],
        lr_history=[0.001, 0.0009, 0.0008, 0.0007],
        tokens_processed=10000,
        batches_processed=100
    )


@pytest.fixture
def checkpoint_manager(checkpoint_config):
    """Create checkpoint manager"""
    return CheckpointManager(checkpoint_config)


@pytest.fixture
def training_recovery(checkpoint_manager, checkpoint_config):
    """Create training recovery"""
    return TrainingRecovery(checkpoint_manager, checkpoint_config)


@pytest.fixture
def auto_checkpointer(checkpoint_manager, checkpoint_config):
    """Create auto checkpointer"""
    return AutoCheckpointer(checkpoint_manager, checkpoint_config)


class TestCheckpointConfig:
    """Test checkpoint configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = CheckpointConfig()

        assert config.checkpoint_dir == "./checkpoints"
        assert config.backup_dir is None
        assert config.save_every_n_steps == 1000
        assert config.save_every_n_epochs == 1
        assert config.save_every_n_minutes == 30
        assert config.keep_last_n_checkpoints == 5
        assert config.keep_best_n_checkpoints == 3
        assert config.verify_checkpoints == True
        assert config.auto_recovery == True
        assert config.max_recovery_attempts == 3
        assert config.recovery_strategy == RecoveryStrategy.RESUME_LATEST

    def test_custom_config(self, temp_dir):
        """Test custom configuration"""
        config = CheckpointConfig(
            checkpoint_dir=temp_dir,
            save_every_n_steps=100,
            keep_last_n_checkpoints=10,
            verify_checkpoints=False,    recovery_strategy=RecoveryStrategy.RESUME_BEST
        )

        assert config.checkpoint_dir == temp_dir
        assert config.save_every_n_steps == 100
        assert config.keep_last_n_checkpoints == 10
        assert config.verify_checkpoints == False
        assert config.recovery_strategy == RecoveryStrategy.RESUME_BEST


class TestTrainingState:
    """Test training state management"""

    def test_default_state(self):
        """Test default training state"""
        state = TrainingState()

        assert state.epoch == 0
        assert state.step == 0
        assert state.global_step == 0
        assert state.best_loss == float('inf')
        assert state.best_val_loss == float('inf')
        assert len(state.loss_history) == 0
        assert len(state.val_loss_history) == 0
        assert len(state.lr_history) == 0
        assert state.tokens_processed == 0
        assert state.batches_processed == 0
        assert state.recovery_count == 0
        assert state.last_recovery_time is None
        assert len(state.failure_reasons) == 0
        assert len(state.custom_state) == 0

    def test_state_serialization(self, test_training_state):
        """Test training state serialization"""
        from dataclasses import asdict

        state_dict = asdict(test_training_state)

        assert isinstance(state_dict, dict)
        assert state_dict['epoch'] == 3
        assert state_dict['step'] == 50
        assert state_dict['global_step'] == 150
        assert state_dict['best_loss'] == 0.8
        assert len(state_dict['loss_history']) == 4
        assert len(state_dict['lr_history']) == 4


class TestCheckpointManager:
    """Test checkpoint manager functionality"""

    def test_initialization(self, checkpoint_manager, checkpoint_config):
        """Test checkpoint manager initialization"""
        assert checkpoint_manager.config == checkpoint_config
        assert checkpoint_manager.checkpoint_dir.exists()
        assert isinstance(checkpoint_manager.checkpoint_registry, dict)
        assert checkpoint_manager.metadata_file.exists() or len(checkpoint_manager.checkpoint_registry) == 0

    def test_checkpoint_id_generation(self, checkpoint_manager):
        """Test checkpoint ID generation"""
        checkpoint_id = checkpoint_manager._generate_checkpoint_id(
            CheckpointType.REGULAR, epoch=5, step=100
        )

        assert isinstance(checkpoint_id, str)
        assert "regular" in checkpoint_id
        assert "e5" in checkpoint_id
        assert "s100" in checkpoint_id
        assert len(checkpoint_id) > 10  # Should include timestamp

    def test_hash_calculation(self, checkpoint_manager, test_model):
        """Test hash calculation for integrity checking"""
        model_hash = checkpoint_manager._calculate_hash(test_model)

        assert isinstance(model_hash, str)
        assert len(model_hash) == 16  # Truncated SHA256

        # Hash should be consistent
        model_hash2 = checkpoint_manager._calculate_hash(test_model)
        assert model_hash == model_hash2

        # Different models should have different hashes
        different_model = TestModel(input_size=20)
        different_hash = checkpoint_manager._calculate_hash(different_model)
        assert model_hash != different_hash

    def test_save_checkpoint(self, checkpoint_manager, test_model, test_optimizer,
                           test_scheduler, test_training_state):
        """Test checkpoint saving"""
        checkpoint_id = checkpoint_manager.save_checkpoint(
            test_model, test_optimizer, test_scheduler, test_training_state,
            CheckpointType.REGULAR, validation_loss=0.85, notes="Test checkpoint"
        )

        assert isinstance(checkpoint_id, str)
        assert checkpoint_id in checkpoint_manager.checkpoint_registry

        # Check metadata
        metadata = checkpoint_manager.checkpoint_registry[checkpoint_id]
        assert metadata.epoch == test_training_state.epoch
        assert metadata.step == test_training_state.step
        assert metadata.global_step == test_training_state.global_step
        assert metadata.checkpoint_type == CheckpointType.REGULAR
        assert metadata.validation_loss == 0.85
        assert metadata.notes == "Test checkpoint"
        assert metadata.file_size > 0

        # Check file exists
        checkpoint_path = checkpoint_manager.checkpoint_dir / f"{checkpoint_id}.pt"
        assert checkpoint_path.exists()

    def test_load_checkpoint(self, checkpoint_manager, test_model, test_optimizer,
                           test_scheduler, test_training_state):
        """Test checkpoint loading"""
        # Save checkpoint first
        checkpoint_id = checkpoint_manager.save_checkpoint(
            test_model, test_optimizer, test_scheduler, test_training_state
        )

        # Create new model and optimizer
        new_model = TestModel()
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = optim.lr_scheduler.StepLR(new_optimizer, step_size=5, gamma=0.9)

        # Load checkpoint
        loaded_state = checkpoint_manager.load_checkpoint(
            checkpoint_id, new_model, new_optimizer, new_scheduler
        )

        # Verify loaded state
        assert loaded_state.epoch == test_training_state.epoch
        assert loaded_state.step == test_training_state.step
        assert loaded_state.global_step == test_training_state.global_step
        assert loaded_state.best_loss == test_training_state.best_loss
        assert loaded_state.loss_history == test_training_state.loss_history
        assert loaded_state.lr_history == test_training_state.lr_history

        # Verify model parameters were loaded
        original_params = list(test_model.parameters())
        loaded_params = list(new_model.parameters())

        for orig, loaded in zip(original_params, loaded_params):
            assert torch.allclose(orig, loaded)

    def test_checkpoint_verification(self, checkpoint_manager, test_model,
                                   test_optimizer, test_scheduler, test_training_state):
        """Test checkpoint verification"""
        # Save valid checkpoint
        checkpoint_id = checkpoint_manager.save_checkpoint(
            test_model, test_optimizer, test_scheduler, test_training_state
        )

        checkpoint_path = checkpoint_manager.checkpoint_dir / f"{checkpoint_id}.pt"

        # Verify valid checkpoint
        assert checkpoint_manager._verify_checkpoint(checkpoint_path) == True

        # Test with non-existent file
        fake_path = checkpoint_manager.checkpoint_dir / "nonexistent.pt"
        assert checkpoint_manager._verify_checkpoint(fake_path) == False

        # Test with empty file
        empty_path = checkpoint_manager.checkpoint_dir / "empty.pt"
        empty_path.touch()
        assert checkpoint_manager._verify_checkpoint(empty_path) == False

    def test_checkpoint_listing(self, checkpoint_manager, test_model, test_optimizer,
                              test_scheduler, test_training_state):
        """Test checkpoint listing and filtering"""
        # Save multiple checkpoints
        regular_id = checkpoint_manager.save_checkpoint(
            test_model, test_optimizer, test_scheduler, test_training_state,
            CheckpointType.REGULAR
        )

        best_id = checkpoint_manager.save_checkpoint(
            test_model, test_optimizer, test_scheduler, test_training_state,
            CheckpointType.BEST
        )

        # Test listing all checkpoints
        all_checkpoints = checkpoint_manager.list_checkpoints()
        assert len(all_checkpoints) == 2

        # Test filtering by type
        regular_checkpoints = checkpoint_manager.list_checkpoints(CheckpointType.REGULAR)
        assert len(regular_checkpoints) == 1
        assert regular_checkpoints[0][0] == regular_id

        best_checkpoints = checkpoint_manager.list_checkpoints(CheckpointType.BEST)
        assert len(best_checkpoints) == 1
        assert best_checkpoints[0][0] == best_id

    def test_latest_checkpoint(self, checkpoint_manager, test_model, test_optimizer,
                             test_scheduler, test_training_state):
        """Test getting latest checkpoint"""
        # No checkpoints initially
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is None

        # Save checkpoint
        checkpoint_id = checkpoint_manager.save_checkpoint(
            test_model, test_optimizer, test_scheduler, test_training_state
        )

        # Get latest
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is not None
        assert latest[0] == checkpoint_id

    def test_best_checkpoint(self, checkpoint_manager, test_model, test_optimizer,
                           test_scheduler):
        """Test getting best checkpoint"""
        # Create states with different losses
        state1 = TrainingState(epoch=1, step=10, global_step=10, loss_history=[1.0])
        state2 = TrainingState(epoch=2, step=20, global_step=20, loss_history=[0.5])
        state3 = TrainingState(epoch=3, step=30, global_step=30, loss_history=[0.8])

        # Save checkpoints
        id1 = checkpoint_manager.save_checkpoint(test_model, test_optimizer, None, state1)
        id2 = checkpoint_manager.save_checkpoint(test_model, test_optimizer, None, state2)
        id3 = checkpoint_manager.save_checkpoint(test_model, test_optimizer, None, state3)

        # Get best checkpoint (lowest loss)
        best = checkpoint_manager.get_best_checkpoint()
        assert best is not None
        assert best[0] == id2  # state2 has the lowest loss (0.5)
        assert best[1].loss == 0.5


class TestTrainingRecovery:
    """Test training recovery functionality"""

    def test_initialization(self, training_recovery, checkpoint_manager, checkpoint_config):
        """Test training recovery initialization"""
        assert training_recovery.checkpoint_manager == checkpoint_manager
        assert training_recovery.config == checkpoint_config
        assert training_recovery.recovery_in_progress == False
        assert training_recovery.recovery_attempts == 0
        assert training_recovery.last_successful_checkpoint is None

    def test_find_recovery_checkpoint(self, training_recovery, checkpoint_manager,
                                    test_model, test_optimizer, test_training_state):
        """Test finding recovery checkpoint"""
        # No checkpoints initially
        recovery_checkpoint = training_recovery.find_recovery_checkpoint()
        assert recovery_checkpoint is None

        # Save checkpoint
        checkpoint_id = checkpoint_manager.save_checkpoint(
            test_model, test_optimizer, None, test_training_state
        )

        # Should find the checkpoint
        recovery_checkpoint = training_recovery.find_recovery_checkpoint()
        assert recovery_checkpoint is not None
        assert recovery_checkpoint[0] == checkpoint_id

    def test_recovery_strategies(self, checkpoint_config, checkpoint_manager,
                               test_model, test_optimizer):
        """Test different recovery strategies"""
        # Create states with different characteristics
        state1 = TrainingState(epoch=1, step=10, global_step=10, loss_history=[1.0])
        state2 = TrainingState(epoch=2, step=20, global_step=20, loss_history=[0.5])  # Best loss

        # Save checkpoints
        id1 = checkpoint_manager.save_checkpoint(test_model, test_optimizer, None, state1)
        time.sleep(0.01)  # Ensure different timestamps
        id2 = checkpoint_manager.save_checkpoint(test_model, test_optimizer, None, state2)

        # Test RESUME_LATEST strategy
        checkpoint_config.recovery_strategy = RecoveryStrategy.RESUME_LATEST
        recovery = TrainingRecovery(checkpoint_manager, checkpoint_config)
        latest = recovery.find_recovery_checkpoint()
        assert latest[0] == id2  # Most recent

        # Test RESUME_BEST strategy
        checkpoint_config.recovery_strategy = RecoveryStrategy.RESUME_BEST
        recovery = TrainingRecovery(checkpoint_manager, checkpoint_config)
        best = recovery.find_recovery_checkpoint()
        assert best[0] == id2  # Best loss

        # Test RESTART_FRESH strategy
        checkpoint_config.recovery_strategy = RecoveryStrategy.RESTART_FRESH
        recovery = TrainingRecovery(checkpoint_manager, checkpoint_config)
        fresh = recovery.find_recovery_checkpoint()
        assert fresh is None

    def test_attempt_recovery(self, training_recovery, checkpoint_manager,
                            test_model, test_optimizer, test_training_state):
        """Test recovery attempt"""
        # Save checkpoint first
        checkpoint_id = checkpoint_manager.save_checkpoint(
            test_model, test_optimizer, None, test_training_state
        )

        # Create new model and optimizer for recovery
        new_model = TestModel()
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)

        # Attempt recovery
        recovered_state = training_recovery.attempt_recovery(new_model, new_optimizer)

        assert recovered_state is not None
        assert recovered_state.epoch == test_training_state.epoch
        assert recovered_state.step == test_training_state.step
        assert recovered_state.recovery_count == test_training_state.recovery_count + 1
        assert recovered_state.last_recovery_time is not None
        assert training_recovery.recovery_attempts == 1

    def test_recovery_validation(self, training_recovery):
        """Test recovery validation"""
        # Valid state
        valid_state = TrainingState(
            epoch=5, step=100, global_step=500,
            best_loss=0.5, loss_history=[1.0, 0.8, 0.6, 0.5]
        )
        assert training_recovery.validate_recovery(valid_state) == True

        # Invalid state (negative values)
        invalid_state = TrainingState(epoch=-1, step=100, global_step=500)
        assert training_recovery.validate_recovery(invalid_state) == False

        # Invalid state (negative global step)
        invalid_state2 = TrainingState(epoch=5, step=100, global_step=-10)
        assert training_recovery.validate_recovery(invalid_state2) == False

    def test_max_recovery_attempts(self, checkpoint_config, checkpoint_manager):
        """Test maximum recovery attempts limit"""
        checkpoint_config.max_recovery_attempts = 2
        recovery = TrainingRecovery(checkpoint_manager, checkpoint_config)

        # Create model and optimizer
        model = TestModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Attempt recovery without any checkpoints (should fail)
        result1 = recovery.attempt_recovery(model, optimizer)
        assert result1 is None
        assert recovery.recovery_attempts == 1

        result2 = recovery.attempt_recovery(model, optimizer)
        assert result2 is None
        assert recovery.recovery_attempts == 2

        # Third attempt should be blocked
        result3 = recovery.attempt_recovery(model, optimizer)
        assert result3 is None
        assert recovery.recovery_attempts == 2  # Should not increment


class TestAutoCheckpointer:
    """Test automatic checkpointing functionality"""

    def test_initialization(self, auto_checkpointer, checkpoint_manager, checkpoint_config):
        """Test auto checkpointer initialization"""
        assert auto_checkpointer.checkpoint_manager == checkpoint_manager
        assert auto_checkpointer.config == checkpoint_config
        assert auto_checkpointer.last_checkpoint_time > 0
        assert auto_checkpointer.last_checkpoint_step == 0
        assert auto_checkpointer.last_checkpoint_epoch == 0

    def test_step_based_trigger(self, auto_checkpointer, checkpoint_config):
        """Test step-based checkpointing trigger"""
        state = TrainingState(global_step=15)  # config.save_every_n_steps = 10

        should_checkpoint, checkpoint_type = auto_checkpointer.should_checkpoint(state)
        assert should_checkpoint == True
        assert checkpoint_type == CheckpointType.REGULAR

    def test_epoch_based_trigger(self, auto_checkpointer, checkpoint_config):
        """Test epoch-based checkpointing trigger"""
        state = TrainingState(epoch=2)  # config.save_every_n_epochs = 1

        should_checkpoint, checkpoint_type = auto_checkpointer.should_checkpoint(state)
        assert should_checkpoint == True
        assert checkpoint_type == CheckpointType.REGULAR

    def test_time_based_trigger(self, auto_checkpointer, checkpoint_config):
        """Test time-based checkpointing trigger"""
        # Set last checkpoint time to more than save_every_n_minutes ago
        auto_checkpointer.last_checkpoint_time = time.time() - (checkpoint_config.save_every_n_minutes * 60 + 10)

        state = TrainingState()
        should_checkpoint, checkpoint_type = auto_checkpointer.should_checkpoint(state)
        assert should_checkpoint == True
        assert checkpoint_type == CheckpointType.REGULAR

    def test_best_loss_trigger(self, auto_checkpointer):
        """Test best loss checkpointing trigger"""
        state = TrainingState(best_loss=1.0, loss_history=[0.5])  # Current loss better than best

        should_checkpoint, checkpoint_type = auto_checkpointer.should_checkpoint(state)
        assert should_checkpoint == True
        assert checkpoint_type == CheckpointType.BEST

    def test_best_validation_loss_trigger(self, auto_checkpointer):
        """Test best validation loss checkpointing trigger"""
        state = TrainingState(best_val_loss=1.0)
        validation_loss = 0.8  # Better than best

        should_checkpoint, checkpoint_type = auto_checkpointer.should_checkpoint(state, validation_loss)
        assert should_checkpoint == True
        assert checkpoint_type == CheckpointType.BEST

    def test_no_trigger(self, auto_checkpointer):
        """Test when no checkpointing should occur"""
        # Recent checkpoint, no improvement
        auto_checkpointer.last_checkpoint_time = time.time()
        auto_checkpointer.last_checkpoint_step = 5
        auto_checkpointer.last_checkpoint_epoch = 0

        state = TrainingState(global_step=8, epoch=0, best_loss=0.5, loss_history=[0.6])

        should_checkpoint, checkpoint_type = auto_checkpointer.should_checkpoint(state)
        assert should_checkpoint == False

    def test_checkpoint_if_needed(self, auto_checkpointer, checkpoint_manager,
                                test_model, test_optimizer, test_scheduler):
        """Test conditional checkpointing"""
        # Create state that should trigger checkpointing
        state = TrainingState(global_step=15, epoch=1, loss_history=[0.5])

        checkpoint_id = auto_checkpointer.checkpoint_if_needed(
            test_model, test_optimizer, test_scheduler, state
        )

        assert checkpoint_id is not None
        assert checkpoint_id in checkpoint_manager.checkpoint_registry

        # Update tracking variables should be updated
        assert auto_checkpointer.last_checkpoint_step == 15
        assert auto_checkpointer.last_checkpoint_epoch == 1

    def test_force_checkpoint(self, auto_checkpointer, checkpoint_manager,
                            test_model, test_optimizer, test_scheduler):
        """Test forced checkpointing"""
        # Create state that normally wouldn't trigger checkpointing
        state = TrainingState(global_step=1, epoch=0)

        # Force checkpoint
        checkpoint_id = auto_checkpointer.checkpoint_if_needed(
            test_model, test_optimizer, test_scheduler, state, force=True
        )

        assert checkpoint_id is not None
        assert checkpoint_id in checkpoint_manager.checkpoint_registry


class TestIntegration:
    """Integration tests for the complete checkpoint system"""

    def test_create_checkpoint_system(self, checkpoint_config):
        """Test factory function"""
        manager, recovery, auto_checkpointer = create_checkpoint_system(checkpoint_config)

        assert isinstance(manager, CheckpointManager)
        assert isinstance(recovery, TrainingRecovery)
        assert isinstance(auto_checkpointer, AutoCheckpointer)

        assert recovery.checkpoint_manager == manager
        assert auto_checkpointer.checkpoint_manager == manager

    def test_end_to_end_workflow(self, checkpoint_config, temp_dir):
        """Test complete end-to-end checkpoint and recovery workflow"""
        # Create system
        manager, recovery, auto_checkpointer = create_checkpoint_system(checkpoint_config)

        # Create model and training components
        model = TestModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

        # Simulate training progress
        training_state = TrainingState()

        for epoch in range(3):
            training_state.epoch = epoch

            for step in range(20):
                training_state.step = step
                training_state.global_step = epoch * 20 + step

                # Simulate loss improvement
                loss = 1.0 - (epoch * 20 + step) * 0.01
                training_state.loss_history.append(loss)
                training_state.lr_history.append(optimizer.param_groups[0]['lr'])

                # Update best loss
                if loss < training_state.best_loss:
                    training_state.best_loss = loss

                # Check for automatic checkpointing
                checkpoint_id = auto_checkpointer.checkpoint_if_needed(
                    model, optimizer, scheduler, training_state
                )

                if checkpoint_id:
                    print(f"Auto-checkpoint saved at epoch {epoch}, step {step}: {checkpoint_id}")

        # Verify checkpoints were created
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) > 0

        # Test recovery
        new_model = TestModel()
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = optim.lr_scheduler.StepLR(new_optimizer, step_size=5, gamma=0.9)

        recovered_state = recovery.attempt_recovery(new_model, new_optimizer, new_scheduler)

        assert recovered_state is not None
        assert recovered_state.epoch >= 0
        assert recovered_state.global_step >= 0
        assert len(recovered_state.loss_history) > 0

        # Verify model parameters were restored
        original_params = list(model.parameters())
        recovered_params = list(new_model.parameters())

        for orig, recovered in zip(original_params, recovered_params):
            assert torch.allclose(orig, recovered, atol=1e-6)

    def test_checkpoint_cleanup(self, checkpoint_config, temp_dir):
        """Test checkpoint cleanup functionality"""
        checkpoint_config.keep_last_n_checkpoints = 2

        manager, _, _ = create_checkpoint_system(checkpoint_config)

        model = TestModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Save multiple checkpoints
        checkpoint_ids = []
        for i in range(5):
            state = TrainingState(epoch=i, step=i*10, global_step=i*10)
            checkpoint_id = manager.save_checkpoint(model, optimizer, None, state)
            checkpoint_ids.append(checkpoint_id)
            time.sleep(0.01)  # Ensure different timestamps

        # Should only keep the last 2 checkpoints
        remaining_checkpoints = manager.list_checkpoints()
        assert len(remaining_checkpoints) == 2

        # Should be the most recent ones
        remaining_ids = [cp[0] for cp in remaining_checkpoints]
        assert checkpoint_ids[-1] in remaining_ids
        assert checkpoint_ids[-2] in remaining_ids


if __name__ == "__main__":
    # Run basic tests
    print("🧪 Running Training Recovery Tests")

    # Test basic functionality
    with tempfile.TemporaryDirectory() as temp_dir:
        config = CheckpointConfig(
            checkpoint_dir=os.path.join(temp_dir, "checkpoints"),
            save_every_n_steps=5,
            keep_last_n_checkpoints=2
        )

        manager, recovery, auto_checkpointer = create_checkpoint_system(config)
        print("✅ Checkpoint system created")

        # Test model
        model = TestModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

        # Test state
        state = TrainingState(epoch=2, step=25, global_step=50, loss_history=[1.0, 0.8, 0.6])

        # Save checkpoint
        checkpoint_id = manager.save_checkpoint(model, optimizer, scheduler, state)
        print(f"✅ Checkpoint saved: {checkpoint_id}")

        # Test loading
        new_model = TestModel()
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = optim.lr_scheduler.StepLR(new_optimizer, step_size=5, gamma=0.9)

        loaded_state = manager.load_checkpoint(checkpoint_id, new_model, new_optimizer, new_scheduler)
        print(f"✅ Checkpoint loaded: epoch {loaded_state.epoch}, step {loaded_state.step}")

        # Test recovery
        recovered_state = recovery.attempt_recovery(new_model, new_optimizer, new_scheduler)
        print(f"✅ Recovery successful: epoch {recovered_state.epoch}")

        # Test auto-checkpointing
        should_checkpoint, cp_type = auto_checkpointer.should_checkpoint(state)
        print(f"✅ Auto-checkpoint check: {should_checkpoint}, type: {cp_type}")

    print("🎉 All basic tests passed!")