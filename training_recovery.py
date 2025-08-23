#!/usr/bin/env python3
"""
Training Recovery and Checkpointing System
==========================================

Robust checkpoint management with automatic recovery, state preservation,
and failure handling for the advanced NanoLM training system.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import time
import shutil
import hashlib
import logging
import pickle
import threading
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import tempfile
import fcntl
import signal
import atexit


class CheckpointType(Enum):
    """Types of checkpoints"""
    REGULAR = "regular"
    BEST = "best"
    EMERGENCY = "emergency"
    MILESTONE = "milestone"
    RECOVERY = "recovery"


class RecoveryStrategy(Enum):
    """Recovery strategies"""
    RESUME_EXACT = "resume_exact"      # Resume from exact state
    RESUME_BEST = "resume_best"        # Resume from best checkpoint
    RESUME_LATEST = "resume_latest"    # Resume from latest checkpoint
    RESTART_FRESH = "restart_fresh"    # Start training from scratch


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint files"""
    timestamp: datetime
    epoch: int
    step: int
    global_step: int
    loss: float
    learning_rate: float
    checkpoint_type: CheckpointType
    model_hash: str
    optimizer_hash: str
    file_size: int
    validation_loss: Optional[float] = None
    training_time: float = 0.0
    gpu_memory_used: float = 0.0
    notes: str = ""


@dataclass
class TrainingState:
    """Complete training state for recovery"""
    epoch: int = 0
    step: int = 0
    global_step: int = 0
    best_loss: float = float('inf')
    best_val_loss: float = float('inf')
    training_start_time: float = 0.0
    total_training_time: float = 0.0

    # Loss history
    loss_history: List[float] = field(default_factory=list)
    val_loss_history: List[float] = field(default_factory=list)

    # Learning rate history
    lr_history: List[float] = field(default_factory=list)

    # Training metrics
    tokens_processed: int = 0
    batches_processed: int = 0

    # Reinformation
    recovery_count: int = 0
    last_recovery_time: Optional[float] = None
    failure_reasons: List[str] = field(default_factory=list)

    # Custom state (for extensions)
    custom_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management"""
    # Paths
    checkpoint_dir: str = "./checkpoints"
    backup_dir: Optional[str] = None

    # Checkpoint frequency
    save_every_n_steps: int = 1000
    save_every_n_epochs: int = 1
    save_every_n_minutes: int = 30

    # Retention policy
    keep_last_n_checkpoints: int = 5
    keep_best_n_checkpoints: int = 3
    keep_milestone_checkpoints: bool = True

    # Validation and integrity
    verify_checkpoints: bool = True
    compress_checkpoints: bool = False

    # Recovery settings
    auto_recovery: bool = True
    max_recovery_attempts: int = 3
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RESUME_LATEST

    # Emergency settings
    emergency_save_on_signal: bool = True
    emergency_save_on_exception: bool = True

    # Backup settings
    enable_backup: bool = True
    backup_frequency_hours: int = 24


class CheckpointManager:
    """Manages checkpoint creation, validation, and cleanup"""

    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.backup_dir = Path(config.backup_dir) if config.backup_dir else None

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self.backup_dir:
            self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint registry
        self.checkpoint_registry = {}
        self.metadata_file = self.checkpoint_dir / "checkpoint_registry.json"

        # Load existing registry
        self._load_registry()

        # Lock file for concurrent access
        self.lock_file = self.checkpoint_dir / ".checkpoint_lock"

        logging.info("✅ Checkpoint Manager initialized")
        logging.info(f"  • Checkpoint directory: {self.checkpoint_dir}")
        logging.info(f"  • Registry contains {len(self.checkpoint_registry)} checkpoints")

    def _load_registry(self):
        """Load checkpoint registry from disk"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)

                # Convert to CheckpointMetadata objects
                for checkpoint_id, metadata_dict in data.items():
                    # Convert timestamp string back to datetime
                    metadata_dict['timestamp'] = datetime.fromisoformat(metadata_dict['timestamp'])
                    metadata_dict['checkpoint_type'] = CheckpointType(metadata_dict['checkpoint_type'])

                    self.checkpoint_registry[checkpoint_id] = CheckpointMetadata(**metadata_dict)

                logging.info(f"Loaded {len(self.checkpoint_registry)} checkpoints from registry")

            except Exception as e:
                logging.error(f"Failed to load checkpoint registry: {e}")
                self.checkpoint_registry = {}

    def _save_registry(self):
        """Save checkpoint registry to disk"""
        try:
            # Convert to serializable format
            data = {}
            for checkpoint_id, metadata in self.checkpoint_registry.items():
                metadata_dict = asdict(metadata)
                metadata_dict['timestamp'] = metadata.timestamp.isoformat()
                metadata_dict['checkpoint_type'] = metadata.checkpoint_type.value
                data[checkpoint_id] = metadata_dict

            # Atomic write
            temp_file = self.metadata_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)

            temp_file.replace(self.metadata_file)

        except Exception as e:
            logging.error(f"Failed to save checkpoint registry: {e}")

    def _generate_checkpoint_id(self, checkpoint_type: CheckpointType,
                               epoch: int, step: int) -> str:
        """Generate unique checkpoint ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{checkpoint_type.value}_{timestamp}_e{epoch}_s{step}"

    def _calculate_hash(self, obj: Any) -> str:
        """Calculate hash of an object for integrity checking"""
        try:
            if hasattr(obj, 'state_dict'):
                # For models and optimizers
                state_dict = obj.state_dict()
                serialized = pickle.dumps(state_dict, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                serialized = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

            return hashlib.sha256(serialized).hexdigest()[:16]
        except Exception as e:
            logging.warning(f"Failed to calculate hash: {e}")
            return "unknown"

    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                       scheduler: Optional[Any], training_state: TrainingState,
                       checkpoint_type: CheckpointType = CheckpointType.REGULAR,
                       validation_loss: Optional[float] = None,
                       notes: str = "") -> str:
        """Save a checkpoint with metadata"""

        # Generate checkpoint ID
        checkpoint_id = self._generate_checkpoint_id(
            checkpoint_type, training_state.epoch, training_state.step
        )

        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"

        try:
            # Acquire lock
            with self._acquire_lock():
                # Prepare checkpoint data
                checkpoint_data = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'training_state': asdict(training_state),
                    'checkpoint_id': checkpoint_id,
                    'save_time': time.time(),
                    'pytorch_version': torch.__version__,
                }

                # Add custom data if present
                if hasattr(model, 'get_checkpoint_data'):
                    checkpoint_data['custom_model_data'] = model.get_checkpoint_data()

                # Save checkpoint
                if self.config.compress_checkpoints:
                    torch.save(checkpoint_data, checkpoint_path,
                             _use_new_zipfile_serialization=True)
                else:
                    torch.save(checkpoint_data, checkpoint_path)

                # Calculate metadata
                file_size = checkpoint_path.stat().st_size
                model_hash = self._calculate_hash(model)
                optimizer_hash = self._calculate_hash(optimizer)

                # Create metadata
                metadata = CheckpointMetadata(
                    timestamp=datetime.now(),
                    epoch=training_state.epoch,
                    step=training_state.step,
                    global_step=training_state.global_step,
                    loss=training_state.loss_history[-1] if training_state.loss_history else 0.0,
                    learning_rate=training_state.lr_history[-1] if training_state.lr_history else 0.0,
                    checkpoint_type=checkpoint_type,
                    model_hash=model_hash,
                    optimizer_hash=optimizer_hash,
                    file_size=file_size,
                    validation_loss=validation_loss,
                    training_time=training_state.total_training_time,
                    notes=notes
                )

                # Verify checkpoint if enabled
                if self.config.verify_checkpoints:
                    if not self._verify_checkpoint(checkpoint_path):
                        raise RuntimeError("Checkpoint verification failed")

                # Add to registry
                self.checkpoint_registry[checkpoint_id] = metadata
                self._save_registry()

                # Cleanup old checkpoints
                self._cleanup_checkpoints()

                logging.info(f"✅ Checkpoint saved: {checkpoint_id}")
                logging.info(f"  • Type: {checkpoint_type.value}")
                logging.info(f"  • Size: {file_size / 1024 / 1024:.2f} MB")
                logging.info(f"  • Path: {checkpoint_path}")

                return checkpoint_id

        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")
            # Clean up partial file
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            raise

    def load_checkpoint(self, checkpoint_id: str, model: nn.Module,
                       optimizer: optim.Optimizer, scheduler: Optional[Any] = None,
                       strict: bool = True) -> TrainingState:
        """Load a checkpoint and restore training state"""

        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            # Verify checkpoint
            if self.config.verify_checkpoints:
                if not self._verify_checkpoint(checkpoint_path):
                    raise RuntimeError("Checkpoint verification failed")

            # Load checkpoint
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            # Load model state
            model.load_state_dict(checkpoint_data['model_state_dict'], strict=strict)

            # Load optimizer state
            optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])

            # Load scheduler state
            if scheduler and checkpoint_data.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])

            # Load training state
            training_state_dict = checkpoint_data['training_state']
            training_state = TrainingState(**training_state_dict)

            # Load custom data if present
            if hasattr(model, 'load_checkpoint_data') and 'custom_model_data' in checkpoint_data:
                model.load_checkpoint_data(checkpoint_data['custom_model_data'])

            logging.info(f"✅ Checkpoint loaded: {checkpoint_id}")
            logging.info(f"  • Epoch: {training_state.epoch}")
            logging.info(f"  • Step: {training_state.step}")
            logging.info(f"  • Best loss: {training_state.best_loss:.6f}")

            return training_state

        except Exception as e:
            logging.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            raise

    def _verify_checkpoint(self, checkpoint_path: Path) -> bool:
        """Verify checkpoint integrity"""
        try:
            # Basic file checks
            if not checkpoint_path.exists():
                return False

            if checkpoint_path.stat().st_size == 0:
                return False

            # Try to load checkpoint with weights_only=False for verification
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            # Check required keys
            required_keys = ['model_state_dict', 'optimizer_state_dict', 'training_state']
            for key in required_keys:
                if key not in checkpoint_data:
                    logging.error(f"Missing key in checkpoint: {key}")
                    return False

            # Verify training state structure
            training_state = checkpoint_data['training_state']
            if not isinstance(training_state, dict):
                return False

            return True

        except Exception as e:
            logging.error(f"Checkpoint verification failed: {e}")
            return False

    def _cleanup_checkpoints(self):
        """Clean up old checkpoints according to retention policy"""
        try:
            # Group checkpoints by type
            checkpoints_by_type = {}
            for checkpoint_id, metadata in self.checkpoint_registry.items():
                checkpoint_type = metadata.checkpoint_type
                if checkpoint_type not in checkpoints_by_type:
                    checkpoints_by_type[checkpoint_type] = []
                checkpoints_by_type[checkpoint_type].append((checkpoint_id, metadata))

            # Clean up regular checkpoints
            if CheckpointType.REGULAR in checkpoints_by_type:
                regular_checkpoints = checkpoints_by_type[CheckpointType.REGULAR]
                regular_checkpoints.sort(key=lambda x: x[1].timestamp, reverse=True)

                # Keep only the most recent N checkpoints
                to_remove = regular_checkpoints[self.config.keep_last_n_checkpoints:]
                for checkpoint_id, metadata in to_remove:
                    self._remove_checkpoint(checkpoint_id)

            # Clean up best checkpoints
            if CheckpointType.BEST in checkpoints_by_type:
                best_checkpoints = checkpoints_by_type[CheckpointType.BEST]
                best_checkpoints.sort(key=lambda x: x[1].loss)

                # Keep only the best N checkpoints
                to_remove = best_checkpoints[self.config.keep_best_n_checkpoints:]
                for checkpoint_id, metadata in to_remove:
                    self._remove_checkpoint(checkpoint_id)

            # Keep milestone and emergency checkpoints if configured
            # (These are not cleaned up automatically)

        except Exception as e:
            logging.error(f"Checkpoint cleanup failed: {e}")

    def _remove_checkpoint(self, checkpoint_id: str):
        """Remove a checkpoint file and its metadata"""
        try:
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
            if checkpoint_path.exists():
                checkpoint_path.unlink()

            if checkpoint_id in self.checkpoint_registry:
                del self.checkpoint_registry[checkpoint_id]

            logging.debug(f"Removed checkpoint: {checkpoint_id}")

        except Exception as e:
            logging.error(f"Failed to remove checkpoint {checkpoint_id}: {e}")

    def _acquire_lock(self):
        """Acquire file lock for concurrent access protection"""
        class FileLock:
            def __init__(self, lock_file):
                self.lock_file = lock_file
                self.lock_fd = None

            def __enter__(self):
                self.lock_fd = open(self.lock_file, 'w')
                fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX)
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.lock_fd:
                    fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
                    self.lock_fd.close()

        return FileLock(self.lock_file)

    def list_checkpoints(self, checkpoint_type: Optional[CheckpointType] = None) -> List[Tuple[str, CheckpointMetadata]]:
        """List available checkpoints"""
        checkpoints = []
        for checkpoint_id, metadata in self.checkpoint_registry.items():
            if checkpoint_type is None or metadata.checkpoint_type == checkpoint_type:
                checkpoints.append((checkpoint_id, metadata))

        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x[1].timestamp, reverse=True)
        return checkpoints

    def get_latest_checkpoint(self, checkpoint_type: Optional[CheckpointType] = None) -> Optional[Tuple[str, CheckpointMetadata]]:
        """Get the latest checkpoint"""
        checkpoints = self.list_checkpoints(checkpoint_type)
        return checkpoints[0] if checkpoints else None

    def get_best_checkpoint(self) -> Optional[Tuple[str, CheckpointMetadata]]:
        """Get the checkpoint with the best loss"""
        best_checkpoint = None
        best_loss = float('inf')

        for checkpoint_id, metadata in self.checkpoint_registry.items():
            if metadata.loss < best_loss:
                best_loss = metadata.loss
                best_checkpoint = (checkpoint_id, metadata)

        return best_checkpoint

    def create_backup(self, checkpoint_id: str) -> bool:
        """Create backup of a checkpoint"""
        if not self.backup_dir:
            return False

        try:
            source_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
            backup_path = self.backup_dir / f"{checkpoint_id}.pt"

            if source_path.exists():
                shutil.copy2(source_path, backup_path)
                logging.info(f"Backup created: {backup_path}")
                return True

        except Exception as e:
            logging.error(f"Failed to create backup for {checkpoint_id}: {e}")

        return False


class TrainingRecovery:
    """Handles training recovery and automatic resumption"""

    def __init__(self, checkpoint_manager: CheckpointManager, config: CheckpointConfig):
        self.checkpoint_manager = checkpoint_manager
        self.config = config

        # Recovery state
        self.recovery_in_progress = False
        self.recovery_attempts = 0
        self.last_successful_checkpoint = None

        # Emergency save handlers
        self.emergency_handlers_registered = False

        logging.info("✅ Training Recovery initialized")

    def setup_emergency_handlers(self, save_callback: Callable[[], str]):
        """Setup emergency save handlers for signals and exceptions"""
        if self.emergency_handlers_registered:
            return

        self.emergency_save_callback = save_callback

        if self.config.emergency_save_on_signal:
            # Register signal handlers
            signal.signal(signal.SIGINT, self._emergency_save_handler)
            signal.signal(signal.SIGTERM, self._emergency_save_handler)

            # Register exit handler
            atexit.register(self._emergency_save_on_exit)

        self.emergency_handlers_registered = True
        logging.info("✅ Emergency save handlers registered")

    def _emergency_save_handler(self, signum, frame):
        """Handle emergency save on signal"""
        logging.warning(f"Received signal {signum}, performing emergency save...")
        try:
            checkpoint_id = self.emergency_save_callback()
            logging.info(f"Emergency checkpoint saved: {checkpoint_id}")
        except Exception as e:
            logging.error(f"Emergency save failed: {e}")

        # Re-raise the signal
        signal.default_int_handler(signum, frame)

    def _emergency_save_on_exit(self):
        """Perform emergency save on exit"""
        if hasattr(self, 'emergency_save_callback'):
            try:
                checkpoint_id = self.emergency_save_callback()
                logging.info(f"Exit checkpoint saved: {checkpoint_id}")
            except Exception as e:
                logging.error(f"Exit save failed: {e}")

    def find_recovery_checkpoint(self) -> Optional[Tuple[str, CheckpointMetadata]]:
        """Find the best checkpoint for recovery"""

        if self.config.recovery_strategy == RecoveryStrategy.RESUME_EXACT:
            # Try to find the exact checkpoint we were training from
            if self.last_successful_checkpoint:
                return self.last_successful_checkpoint

        elif self.config.recovery_strategy == RecoveryStrategy.RESUME_BEST:
            # Find the best checkpoint
            return self.checkpoint_manager.get_best_checkpoint()

        elif self.config.recovery_strategy == RecoveryStrategy.RESUME_LATEST:
            # Find the latest checkpoint
            return self.checkpoint_manager.get_latest_checkpoint()

        elif self.config.recovery_strategy == RecoveryStrategy.RESTART_FRESH:
            # Don't recover, start fresh
            return None

        # Fallback: try latest, then best
        latest = self.checkpoint_manager.get_latest_checkpoint()
        if latest:
            return latest

        return self.checkpoint_manager.get_best_checkpoint()

    def attempt_recovery(self, model: nn.Module, optimizer: optim.Optimizer,
                        scheduler: Optional[Any] = None) -> Optional[TrainingState]:
        """Attempt to recover training from a checkpoint"""

        if self.recovery_attempts >= self.config.max_recovery_attempts:
            logging.error("Maximum recovery attempts exceeded")
            return None

        self.recovery_in_progress = True
        self.recovery_attempts += 1

        try:
            # Find recovery checkpoint
            recovery_checkpoint = self.find_recovery_checkpoint()

            if not recovery_checkpoint:
                logging.warning("No suitable checkpoint found for recovery")
                return None

            checkpoint_id, metadata = recovery_checkpoint

            logging.info(f"Attempting recovery from checkpoint: {checkpoint_id}")
            logging.info(f"  • Recovery attempt: {self.recovery_attempts}")
            logging.info(f"  • Checkpoint epoch: {metadata.epoch}")
            logging.info(f"  • Checkpoint step: {metadata.step}")
            logging.info(f"  • Checkpoint loss: {metadata.loss:.6f}")

            # Load checkpoint
            training_state = self.checkpoint_manager.load_checkpoint(
                checkpoint_id, model, optimizer, scheduler
            )

            # Update recovery information
            training_state.recovery_count += 1
            training_state.last_recovery_time = time.time()

            # Reset recovery state
            self.recovery_in_progress = False
            self.last_successful_checkpoint = (checkpoint_id, metadata)

            logging.info("✅ Training recovery successful")
            return training_state

        except Exception as e:
            logging.error(f"Recovery attempt {self.recovery_attempts} failed: {e}")
            self.recovery_in_progress = False

            # Try next strategy if available
            if self.recovery_attempts < self.config.max_recovery_attempts:
                logging.info("Trying alternative recovery strategy...")
                return self.attempt_recovery(model, optimizer, scheduler)

            return None

    def is_recovery_needed(self, training_state: TrainingState) -> bool:
        """Check if recovery is needed based on training state"""

        # Check if we have any checkpoints
        checkpoints = self.checkpoint_manager.list_checkpoints()
        if not checkpoints:
            return False

        # Check if current state is significantly behind latest checkpoint
        latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
        if latest_checkpoint:
            _, metadata = latest_checkpoint

            # If we're more than 100 steps behind, recovery might be needed
            if training_state.global_step < metadata.global_step - 100:
                return True

            # If training was interrupted (large time gap)
            time_since_checkpoint = time.time() - metadata.timestamp.timestamp()
            if time_since_checkpoint > 3600:  # 1 hour
                return True

        return False

    def validate_recovery(self, training_state: TrainingState) -> bool:
        """Validate that recovery was successful"""
        try:
            # Basic validation checks
            if training_state.epoch < 0 or training_state.step < 0:
                return False

            if training_state.global_step < 0:
                return False

            if not training_state.loss_history:
                logging.warning("No loss history after recovery")

            # Check for reasonable values
            if training_state.best_loss == float('inf'):
                logging.warning("Best loss is infinity after recovery")

            return True

        except Exception as e:
            logging.error(f"Recovery validation failed: {e}")
            return False


class AutoCheckpointer:
    """Automatic checkpointing with configurable triggers"""

    def __init__(self, checkpoint_manager: CheckpointManager, config: CheckpointConfig):
        self.checkpoint_manager = checkpoint_manager
        self.config = config

        # Tracking variables
        self.last_checkpoint_time = time.time()
        self.last_checkpoint_step = 0
        self.last_checkpoint_epoch = 0

        # Background checkpointing
        self.background_thread = None
        self.stop_background = False

        logging.info("✅ Auto Checkpointer initialized")

    def should_checkpoint(self, training_state: TrainingState,
                         validation_loss: Optional[float] = None) -> Tuple[bool, CheckpointType]:
        """Determine if checkpointing should occur"""

        current_time = time.time()

        # Check step-based trigger
        if (training_state.global_step - self.last_checkpoint_step >=
            self.config.save_every_n_steps):
            return True, CheckpointType.REGULAR

        # Check epoch-based trigger
        if (training_state.epoch - self.last_checkpoint_epoch >=
            self.config.save_every_n_epochs):
            return True, CheckpointType.REGULAR

        # Check time-based trigger
        if (current_time - self.last_checkpoint_time >=
            self.config.save_every_n_minutes * 60):
            return True, CheckpointType.REGULAR

        # Check for best model
        if validation_loss is not None and validation_loss < training_state.best_val_loss:
            return True, CheckpointType.BEST

        # Check for best training loss
        current_loss = training_state.loss_history[-1] if training_state.loss_history else float('inf')
        if current_loss < training_state.best_loss:
            return True, CheckpointType.BEST

        return False, CheckpointType.REGULAR

    def checkpoint_if_needed(self, model: nn.Module, optimizer: optim.Optimizer,
                           scheduler: Optional[Any], training_state: TrainingState,
                           validation_loss: Optional[float] = None,
                           force: bool = False) -> Optional[str]:
        """Checkpoint if conditions are met"""

        if not force:
            should_save, checkpoint_type = self.should_checkpoint(training_state, validation_loss)
            if not should_save:
                return None
        else:
            checkpoint_type = CheckpointType.REGULAR

        try:
            # Save checkpoint
            checkpoint_id = self.checkpoint_manager.save_checkpoint(
                model, optimizer, scheduler, training_state,
                checkpoint_type, validation_loss
            )

            # Update tracking variables
            self.last_checkpoint_time = time.time()
            self.last_checkpoint_step = training_state.global_step
            self.last_checkpoint_epoch = training_state.epoch

            return checkpoint_id

        except Exception as e:
            logging.error(f"Auto-checkpoint failed: {e}")
            return None

    def start_background_checkpointing(self, save_callback: Callable[[], Optional[str]]):
        """Start background checkpointing thread"""
        if self.background_thread and self.background_thread.is_alive():
            return

        self.stop_background = False
        self.background_thread = threading.Thread(
            target=self._background_checkpoint_loop,
            args=(save_callback,),
            daemon=True
        )
        self.background_thread.start()

        logging.info("✅ Background checkpointing started")

    def stop_background_checkpointing(self):
        """Stop background checkpointing thread"""
        self.stop_background = True
        if self.background_thread:
            self.background_thread.join(timeout=5.0)

        logging.info("⏹️ Background checkpointing stopped")

    def _background_checkpoint_loop(self, save_callback: Callable[[], Optional[str]]):
        """Background checkpointing loop"""
        while not self.stop_background:
            try:
                # Sleep for a portion of the checkpoint interval
                time.sleep(min(60, self.config.save_every_n_minutes * 60 // 4))

                if self.stop_background:
                    break

                # Attempt checkpoint
                checkpoint_id = save_callback()
                if checkpoint_id:
                    logging.debug(f"Background checkpoint saved: {checkpoint_id}")

            except Exception as e:
                logging.error(f"Background checkpointing error: {e}")
                time.sleep(60)  # Wait before retrying


def create_checkpoint_system(config: CheckpointConfig) -> Tuple[CheckpointManager, TrainingRecovery, AutoCheckpointer]:
    """Factory function to create complete checkpoint system"""

    checkpoint_manager = CheckpointManager(config)
    training_recovery = TrainingRecovery(checkpoint_manager, config)
    auto_checkpointer = AutoCheckpointer(checkpoint_manager, config)

    return checkpoint_manager, training_recovery, auto_checkpointer


def test_checkpoint_system():
    """Test the checkpoint and recovery system"""
    print("🧪 Testing Checkpoint and Recovery System")

    # Create test configuration
    config = CheckpointConfig(
        checkpoint_dir="./test_checkpoints",
        save_every_n_steps=10,
        keep_last_n_checkpoints=3,
        verify_checkpoints=True,
        auto_recovery=True
    )

    # Create system components
    checkpoint_manager, training_recovery, auto_checkpointer = create_checkpoint_system(config)

    print("✅ Checkpoint system created")

    # Create test model and optimizer
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # Create test training state
    training_state = TrainingState(
        epoch=5,
        step=100,
        global_step=500,
        best_loss=0.5,
        loss_history=[1.0, 0.8, 0.6, 0.5],
        lr_history=[0.001, 0.0009, 0.0008, 0.0007]
    )

    print("✅ Test model and state created")

    # Test checkpoint saving
    checkpoint_id = checkpoint_manager.save_checkpoint(
        model, optimizer, scheduler, training_state,
        CheckpointType.REGULAR, notes="Test checkpoint"
    )

    print(f"✅ Checkpoint saved: {checkpoint_id}")

    # Test checkpoint listing
    checkpoints = checkpoint_manager.list_checkpoints()
    print(f"✅ Found {len(checkpoints)} checkpoints")

    # Test checkpoint loading
    new_model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
    new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=10, gamma=0.9)

    loaded_state = checkpoint_manager.load_checkpoint(
        checkpoint_id, new_model, new_optimizer, new_scheduler
    )

    print(f"✅ Checkpoint loaded: epoch {loaded_state.epoch}, step {loaded_state.step}")

    # Test recovery
    recovery_state = training_recovery.attempt_recovery(new_model, new_optimizer, new_scheduler)
    if recovery_state:
        print(f"✅ Recovery successful: epoch {recovery_state.epoch}")

    # Test auto-checkpointing
    should_checkpoint, checkpoint_type = auto_checkpointer.should_checkpoint(training_state)
    print(f"✅ Auto-checkpoint check: {should_checkpoint}, type: {checkpoint_type}")

    # Cleanup
    import shutil
    if os.path.exists("./test_checkpoints"):
        shutil.rmtree("./test_checkpoints")

    print("🎉 Checkpoint system tests completed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    test_checkpoint_system()