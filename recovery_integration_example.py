#!/usr/bin/env python3
"""
Training Recovery Integration Example
====================================

Demonstrates how to integrate the comprehensive checkpoint and recovery system
with the advanced trainer for robust, fault-tolerant training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import logging
import signal
import os
from typing import Dict, Any, Optional

# Import our systems
from training_recovery import (
    CheckpointManager, TrainingRecovery, AutoCheckpointer,
    CheckpointConfig, TrainingState, CheckpointType, RecoveryStrategy,
    create_checkpoint_system
)
# from advanced_trainer import TrainingConfig, AdvancedTrainer  # Not needed for this demo


class RobustModel(nn.Module):
    """Example model with checkpoint data support"""

    def __init__(self, vocab_size=1000, hidden_size=256, num_layers=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)

        # Custom state for checkpointing
        self.training_metadata = {
            'model_version': '1.0',
            'architecture': 'transformer',
            'creation_time': time.time()
        }

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embedding(input_ids)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        logits = self.output(x)

        outputs = {'logits': logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            outputs['loss'] = loss

        return outputs

    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Get custom data to include in checkpoint"""
        return {
            'training_metadata': self.training_metadata,
            'model_config': {
                'vocab_size': self.vocab_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers
            }
        }

    def load_checkpoint_data(self, data: Dict[str, Any]):
        """Load custom data from checkpoint"""
        if 'training_metadata' in data:
            self.training_metadata.update(data['training_metadata'])


class RobustTrainer:
    """Enhanced trainer with comprehensive recovery capabilities"""

    def __init__(self, model: nn.Module, train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 checkpoint_config: Optional[CheckpointConfig] = None):

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Training components
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

        # Checkpoint system
        self.checkpoint_config = checkpoint_config or CheckpointConfig(
            checkpoint_dir="./robust_checkpoints",
            backup_dir="./robust_backups",
            save_every_n_steps=50,
            save_every_n_epochs=1,
            save_every_n_minutes=10,
            keep_last_n_checkpoints=5,
            keep_best_n_checkpoints=3,
            verify_checkpoints=True,
            auto_recovery=True,
            max_recovery_attempts=3,
            recovery_strategy=RecoveryStrategy.RESUME_LATEST,
            emergency_save_on_signal=True,
            emergency_save_on_exception=True
        )

        # Initialize checkpoint system
        self.checkpoint_manager, self.training_recovery, self.auto_checkpointer = \
            create_checkpoint_system(self.checkpoint_config)

        # Training state
        self.training_state = TrainingState()

        # Recovery setup
        self.setup_recovery()

        logging.info("✅ Robust Trainer initialized")
        logging.info(f"  • Device: {self.device}")
        logging.info(f"  • Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logging.info(f"  • Checkpoint directory: {self.checkpoint_config.checkpoint_dir}")

    def setup_recovery(self):
        """Setup recovery and emergency save handlers"""
        # Setup emergency save callback
        def emergency_save() -> str:
            try:
                return self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, self.scheduler, self.training_state,
                    CheckpointType.EMERGENCY, notes="Emergency save"
                )
            except Exception as e:
                logging.error(f"Emergency save failed: {e}")
                return ""

        # Register emergency handlers
        self.training_recovery.setup_emergency_handlers(emergency_save)

        # Attempt recovery if needed
        if self.checkpoint_config.auto_recovery:
            self.attempt_recovery()

    def attempt_recovery(self) -> bool:
        """Attempt to recover from existing checkpoints"""
        try:
            recovered_state = self.training_recovery.attempt_recovery(
                self.model, self.optimizer, self.scheduler
            )

            if recovered_state:
                self.training_state = recovered_state
                logging.info("🔄 Training recovered from checkpoint")
                logging.info(f"  • Resumed at epoch: {self.training_state.epoch}")
                logging.info(f"  • Resumed at step: {self.training_state.step}")
                logging.info(f"  • Best loss: {self.training_state.best_loss:.6f}")
                logging.info(f"  • Recovery count: {self.training_state.recovery_count}")
                return True
            else:
                logging.info("🆕 Starting fresh training (no recovery needed)")
                return False

        except Exception as e:
            logging.error(f"Recovery attempt failed: {e}")
            return False

    def save_checkpoint(self, checkpoint_type: CheckpointType = CheckpointType.REGULAR,
                       validation_loss: Optional[float] = None,
                       notes: str = "") -> Optional[str]:
        """Save checkpoint with current training state"""
        try:
            checkpoint_id = self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.scheduler, self.training_state,
                checkpoint_type, validation_loss, notes
            )

            # Create backup for important checkpoints
            if checkpoint_type in [CheckpointType.BEST, CheckpointType.MILESTONE]:
                self.checkpoint_manager.create_backup(checkpoint_id)

            return checkpoint_id

        except Exception as e:
            logging.error(f"Checkpoint save failed: {e}")
            return None

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Execute a single training step with timing"""
        step_start_time = time.time()

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Forward pass
        forward_start = time.time()
        outputs = self.model(**batch)
        forward_time = time.time() - forward_start

        loss = outputs['loss']

        # Backward pass
        backward_start = time.time()
        loss.backward()
        backward_time = time.time() - backward_start

        # Optimizer step
        optimizer_start = time.time()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        optimizer_time = time.time() - optimizer_start

        total_time = time.time() - step_start_time

        # Update training state
        self.training_state.step += 1
        self.training_state.global_step += 1
        self.training_state.loss_history.append(loss.item())
        self.training_state.lr_history.append(self.optimizer.param_groups[0]['lr'])
        self.training_state.tokens_processed += batch['input_ids'].numel()
        self.training_state.batches_processed += 1

        # Update best loss
        if loss.item() < self.training_state.best_loss:
            self.training_state.best_loss = loss.item()

        return {
            'loss': loss.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'forward_time': forward_time,
            'backward_time': backward_time,
            'optimizer_time': optimizer_time,
            'total_time': total_time
        }

    def validate(self) -> Optional[float]:
        """Run validation and return average loss"""
        if not self.val_dataloader:
            return None

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                try:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}

                    outputs = self.model(**batch)
                    total_loss += outputs['loss'].item()
                    num_batches += 1

                except Exception as e:
                    logging.warning(f"Validation batch failed: {e}")
                    continue

        self.model.train()

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            self.training_state.val_loss_history.append(avg_loss)

            # Update best validation loss
            if avg_loss < self.training_state.best_val_loss:
                self.training_state.best_val_loss = avg_loss

            return avg_loss

        return None

    def train(self, num_epochs: int = 10) -> Dict[str, Any]:
        """Main training loop with comprehensive recovery"""
        logging.info(f"🚀 Starting robust training for {num_epochs} epochs")

        training_start_time = time.time()
        self.training_state.training_start_time = training_start_time

        try:
            for epoch in range(self.training_state.epoch, num_epochs):
                self.training_state.epoch = epoch
                epoch_start_time = time.time()
                epoch_loss = 0.0
                num_batches = 0

                logging.info(f"\n📈 Epoch {epoch + 1}/{num_epochs}")

                for batch_idx, batch_data in enumerate(self.train_dataloader):
                    try:
                        # Prepare batch
                        if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                            input_ids, labels = batch_data
                            batch = {'input_ids': input_ids, 'labels': labels}
                        else:
                            batch = batch_data

                        # Training step
                        step_result = self.train_step(batch)
                        epoch_loss += step_result['loss']
                        num_batches += 1

                        # Logging
                        if batch_idx % 20 == 0:
                            logging.info(
                                f"  Step {self.training_state.step}: "
                                f"Loss = {step_result['loss']:.6f}, "
                                f"LR = {step_result['learning_rate']:.2e}, "
                                f"Time = {step_result['total_time']:.3f}s"
                            )

                        # Auto-checkpointing
                        checkpoint_id = self.auto_checkpointer.checkpoint_if_needed(
                            self.model, self.optimizer, self.scheduler, self.training_state
                        )

                        if checkpoint_id:
                            logging.info(f"  📁 Auto-checkpoint saved: {checkpoint_id}")

                    except Exception as e:
                        logging.error(f"Training step failed: {e}")
                        self.training_state.failure_reasons.append(str(e))

                        # Attempt recovery if configured
                        if self.checkpoint_config.auto_recovery:
                            logging.info("Attempting recovery from training step failure...")
                            if self.attempt_recovery():
                                continue

                        raise

                # End of epoch
                avg_epoch_loss = epoch_loss / max(1, num_batches)
                epoch_time = time.time() - epoch_start_time

                # Validation
                val_loss = self.validate()
                val_info = f", Val Loss = {val_loss:.6f}" if val_loss else ""

                logging.info(
                    f"  ✅ Epoch {epoch + 1} completed: "
                    f"Avg Loss = {avg_epoch_loss:.6f}{val_info}, "
                    f"Time = {epoch_time:.2f}s"
                )

                # Update scheduler
                self.scheduler.step()

                # Save epoch checkpoint
                checkpoint_type = CheckpointType.BEST if (val_loss and val_loss == self.training_state.best_val_loss) else CheckpointType.REGULAR
                checkpoint_id = self.save_checkpoint(
                    checkpoint_type, val_loss, f"End of epoch {epoch + 1}"
                )

                if checkpoint_id:
                    logging.info(f"  📁 Epoch checkpoint saved: {checkpoint_id}")

        except KeyboardInterrupt:
            logging.warning("\n⏹️ Training interrupted by user")
            # Save emergency checkpoint
            emergency_id = self.save_checkpoint(
                CheckpointType.EMERGENCY, notes="User interruption"
            )
            if emergency_id:
                logging.info(f"Emergency checkpoint saved: {emergency_id}")

        except Exception as e:
            logging.error(f"\n❌ Training failed: {e}")
            # Save emergency checkpoint
            emergency_id = self.save_checkpoint(
                CheckpointType.EMERGENCY, notes=f"Training failure: {str(e)}"
            )
            if emergency_id:
                logging.info(f"Emergency checkpoint saved: {emergency_id}")
            raise

        finally:
            # Update total training time
            total_time = time.time() - training_start_time
            self.training_state.total_training_time += total_time

            # Save final checkpoint
            final_id = self.save_checkpoint(
                CheckpointType.MILESTONE, notes="Training completed"
            )
            if final_id:
                logging.info(f"Final checkpoint saved: {final_id}")

        # Generate training report
        return self.generate_training_report()

    def generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        checkpoints = self.checkpoint_manager.list_checkpoints()
        best_checkpoint = self.checkpoint_manager.get_best_checkpoint()

        report = {
            'training_summary': {
                'total_epochs': self.training_state.epoch,
                'total_steps': self.training_state.step,
                'global_steps': self.training_state.global_step,
                'total_training_time': self.training_state.total_training_time,
                'tokens_processed': self.training_state.tokens_processed,
                'batches_processed': self.training_state.batches_processed,
                'best_loss': self.training_state.best_loss,
                'best_val_loss': self.training_state.best_val_loss,
                'recovery_count': self.training_state.recovery_count
            },
            'checkpoint_summary': {
                'total_checkpoints': len(checkpoints),
                'best_checkpoint': best_checkpoint[0] if best_checkpoint else None,
                'best_checkpoint_loss': best_checkpoint[1].loss if best_checkpoint else None,
                'checkpoint_types': {}
            },
            'failure_analysis': {
                'failure_count': len(self.training_state.failure_reasons),
                'failure_reasons': self.training_state.failure_reasons
            }
        }

        # Count checkpoint types
        for checkpoint_id, metadata in checkpoints:
            checkpoint_type = metadata.checkpoint_type.value
            report['checkpoint_summary']['checkpoint_types'][checkpoint_type] = \
                report['checkpoint_summary']['checkpoint_types'].get(checkpoint_type, 0) + 1

        return report


def create_sample_dataset(vocab_size=1000, seq_len=64, num_samples=500):
    """Create sample dataset for demonstration"""
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = torch.randint(0, vocab_size, (num_samples, seq_len))

    dataset = TensorDataset(input_ids, labels)
    return dataset


def demonstrate_robust_training():
    """Demonstrate robust training with recovery capabilities"""
    print("🛡️ Robust Training with Recovery Demonstration")

    # Create dataset
    train_dataset = create_sample_dataset(vocab_size=1000, seq_len=64, num_samples=400)
    val_dataset = create_sample_dataset(vocab_size=1000, seq_len=64, num_samples=100)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print(f"Dataset created: {len(train_dataset)} train, {len(val_dataset)} val samples")

    # Create model
    model = RobustModel(vocab_size=1000, hidden_size=128, num_layers=2)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create checkpoint configuration
    checkpoint_config = CheckpointConfig(
        checkpoint_dir="./demo_robust_checkpoints",
        backup_dir="./demo_robust_backups",
        save_every_n_steps=25,
        save_every_n_epochs=1,
        save_every_n_minutes=5,
        keep_last_n_checkpoints=3,
        keep_best_n_checkpoints=2,
        verify_checkpoints=True,
        auto_recovery=True,
        max_recovery_attempts=2,
        recovery_strategy=RecoveryStrategy.RESUME_LATEST,
        emergency_save_on_signal=True,
        emergency_save_on_exception=True
    )

    # Create robust trainer
    trainer = RobustTrainer(
        model, train_dataloader, val_dataloader, checkpoint_config
    )

    print("✅ Robust trainer initialized")

    # Demonstrate training with recovery
    try:
        print("\n🚀 Starting robust training...")
        report = trainer.train(num_epochs=3)

        print("\n📊 Training Report:")
        print(f"  • Total epochs: {report['training_summary']['total_epochs']}")
        print(f"  • Total steps: {report['training_summary']['total_steps']}")
        print(f"  • Training time: {report['training_summary']['total_training_time']:.2f}s")
        print(f"  • Best loss: {report['training_summary']['best_loss']:.6f}")
        print(f"  • Best val loss: {report['training_summary']['best_val_loss']:.6f}")
        print(f"  • Recovery count: {report['training_summary']['recovery_count']}")
        print(f"  • Total checkpoints: {report['checkpoint_summary']['total_checkpoints']}")
        print(f"  • Failure count: {report['failure_analysis']['failure_count']}")

        # Show checkpoint types
        checkpoint_types = report['checkpoint_summary']['checkpoint_types']
        if checkpoint_types:
            print("  • Checkpoint types:")
            for cp_type, count in checkpoint_types.items():
                print(f"    - {cp_type}: {count}")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")

        # Show that emergency checkpoint was saved
        checkpoints = trainer.checkpoint_manager.list_checkpoints()
        emergency_checkpoints = [cp for cp in checkpoints if cp[1].checkpoint_type == CheckpointType.EMERGENCY]

        if emergency_checkpoints:
            print(f"✅ Emergency checkpoints saved: {len(emergency_checkpoints)}")
            for cp_id, metadata in emergency_checkpoints:
                print(f"  • {cp_id}: {metadata.notes}")

    # Demonstrate recovery from existing checkpoints
    print("\n🔄 Demonstrating recovery...")

    # Create new trainer instance (simulating restart)
    new_model = RobustModel(vocab_size=1000, hidden_size=128, num_layers=2)
    new_trainer = RobustTrainer(
        new_model, train_dataloader, val_dataloader, checkpoint_config
    )

    # Check if recovery occurred
    if new_trainer.training_state.recovery_count > 0:
        print("✅ Recovery successful!")
        print(f"  • Recovered to epoch: {new_trainer.training_state.epoch}")
        print(f"  • Recovered to step: {new_trainer.training_state.step}")
        print(f"  • Best loss: {new_trainer.training_state.best_loss:.6f}")
    else:
        print("ℹ️ No recovery needed (starting fresh)")

    # List available checkpoints
    checkpoints = new_trainer.checkpoint_manager.list_checkpoints()
    print(f"\n📁 Available checkpoints: {len(checkpoints)}")
    for i, (cp_id, metadata) in enumerate(checkpoints[:5]):  # Show first 5
        print(f"  {i+1}. {cp_id}")
        print(f"     • Type: {metadata.checkpoint_type.value}")
        print(f"     • Epoch: {metadata.epoch}, Step: {metadata.step}")
        print(f"     • Loss: {metadata.loss:.6f}")
        print(f"     • Size: {metadata.file_size / 1024 / 1024:.2f} MB")
        print(f"     • Time: {metadata.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n🎉 Robust training demonstration completed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Run demonstration
    demonstrate_robust_training()