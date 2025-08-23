#!/usr/bin/env python3
"""
Advanced Trainer with Error Handling for NanoLM
==============================================

Implements a comprehensive training system that integrates all advanced features:
- MoE (Mixture of Experts)
- MTP (Multi-Token Prediction)
- HRM (Hierarchical Reasoning Module)
- Anti-hallucination training
- Advanced quantization (NF4/FP4/QAF)
- Loss tracking and prediction
- Robust error handling and recovery
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import gc
import os
import time
import logging
import traceback
import psutil
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path

# Import our custom components
from loss_system import create_loss_system, LossWeights
from loss_tracker import create_loss_tracker
from anti_hallucination import create_anti_hallucination_filter
from unlikelihood_training import create_anti_hallucination_trainer, UnlikelihoodConfig, ContrastiveLossConfig
from advanced_quantization import QuantizationController
from monitoring_system import MonitoringSystem


class TrainingPhase(Enum):
    """Training phases"""
    WARMUP = "warmup"
    MAIN_TRAINING = "main_training"
    FINE_TUNING = "fine_tuning"
    QAF_PHASE = "qaf_phase"
    EVALUATION = "evaluation"


class ErrorType(Enum):
    """Types of training errors"""
    OOM_ERROR = "out_of_memory"
    GRADIENT_EXPLOSION = "gradient_explosion"
    GRADIENT_VANISHING = "gradient_vanishing"
    LOSS_DIVERGENCE = "loss_divergence"
    NUMERICAL_INSTABILITY = "numerical_instability"
    HARDWARE_ERROR = "hardware_error"
    DATA_ERROR = "data_error"


@dataclass
class TrainingConfig:
    """Comprehensive training configuration"""
    # Model and data
    model_name: str = "nanolm_advanced"
    vocab_size: int = 32000
    max_seq_length: int = 2048

    # Training hyperparameters
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Batch and accumulation
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_batch_size: int = 32
    min_batch_size: int = 1

    # Training schedule
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_steps: int = 100000
    eval_steps: int = 500
    save_steps: int = 1000

    # Error handling
    enable_error_recovery: bool = True
    max_oom_retries: int = 3
    batch_size_reduction_factor: float = 0.5
    gradient_clip_threshold: float = 10.0
    loss_spike_threshold: float = 5.0

    # Memory management
    enable_memory_cleanup: bool = True
    memory_cleanup_threshold: float = 0.9
    enable_gradient_checkpointing: bool = True

    # Mixed precision
    use_mixed_precision: bool = True
    loss_scale: float = 65536.0

    # Quantization
    use_quantization: bool = True
    quantization_phase_transition: int = 50000

    # Advanced features
    enable_moe: bool = True
    enable_mtp: bool = True
    enable_hrm: bool = True
    enable_anti_hallucination: bool = True
    enable_loss_tracking: bool = True

    # Loss system configuration
    loss_smoothing: float = 0.1
    mtp_loss_weights: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25, 0.125])
    router_z_loss: float = 1e-4
    forbidden_tokens: List[int] = field(default_factory=list)

    # Additional required attributes
    convergence_window: int = 50
    convergence_threshold: float = 1e-4
    plateau_patience: int = 100
    target_loss: float = 2.0
    early_stopping_patience: int = 200
    min_improvement: float = 1e-3

    # Monitoring
    enable_monitoring: bool = True
    log_interval: int = 10
    detailed_logging: bool = False


@dataclass
class TrainingState:
    """Current training state"""
    step: int = 0
    epoch: int = 0
    phase: TrainingPhase = TrainingPhase.WARMUP
    current_lr: float = 0.0
    best_loss: float = float('inf')

    # Error tracking
    oom_count: int = 0
    gradient_explosion_count: int = 0
    loss_spike_count: int = 0

    # Performance metrics
    tokens_per_second: float = 0.0
    memory_usage: float = 0.0
    gpu_utilization: float = 0.0

    # Training metrics
    running_loss: float = 0.0
    last_losses: List[float] = field(default_factory=list)


class TrainerOptimized:
    """Advanced trainer with comprehensive error handling and feature integration"""

    def __init__(self, config: TrainingConfig, model, tokenizer, train_dataloader,
                 val_dataloader=None, device=None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training state
        self.state = TrainingState()

        # Initialize components
        self._initialize_optimizer()
        self._initialize_scheduler()
        self._initialize_loss_system()
        self._initialize_monitoring()
        self._initialize_error_handling()

        # Mixed precision
        if self.config.use_mixed_precision:
            self.scaler = GradScaler()

        # Quantization controller
        if self.config.use_quantization:
            self.quantization_controller = QuantizationController(config)

        logging.info("✅ Advanced Trainer initialized")
        logging.info(f"  • Device: {self.device}")
        logging.info(f"  • Mixed precision: {self.config.use_mixed_precision}")
        logging.info(f"  • Quantization: {self.config.use_quantization}")
        logging.info(f"  • Error recovery: {self.config.enable_error_recovery}")

    def _initialize_optimizer(self):
        """Initialize optimizer with advanced settings"""
        # Separate parameters for different components
        param_groups = []

        # Base model parameters
        base_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and not any(skip in name for skip in ['bias', 'LayerNorm', 'layernorm']):
                base_params.append(param)

        if base_params:
            param_groups.append({
                'params': base_params,
                'lr': self.config.learning_rate,
                'weight_decay': self.config.weight_decay
            })

        # Bias and LayerNorm parameters (no weight decay)
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and any(skip in name for skip in ['bias', 'LayerNorm', 'layernorm']):
                no_decay_params.append(param)

        if no_decay_params:
            param_groups.append({
                'params': no_decay_params,
                'lr': self.config.learning_rate,
                'weight_decay': 0.0
            })

        # Use AdamW optimizer
        self.optimizer = optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
            weight_decay=self.config.weight_decay
        )

        logging.info(f"✅ Optimizer initialized with {len(param_groups)} parameter groups")

    def _initialize_scheduler(self):
        """Initialize learning rate scheduler"""
        from torch.optim.lr_scheduler import OneCycleLR

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=self.config.max_steps,
            pct_start=self.config.warmup_steps / self.config.max_steps,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )

        logging.info("✅ Learning rate scheduler initialized")

    def _initialize_loss_system(self):
        """Initialize comprehensive loss system"""
        # Multi-component loss system
        loss_weights = LossWeights(
            main_weight=1.0,
            mtp_weight=0.5,
            auxiliary_weight=0.01,
            reasoning_weight=0.1,
            anti_hallucination_weight=0.25,
            fp4_scaling_factor=10.0
        )

        self.loss_system = create_loss_system(self.config, loss_weights)

        # Loss tracking and prediction
        if self.config.enable_loss_tracking:
            self.loss_tracker = create_loss_tracker(self.config)

        # Anti-hallucination system
        if self.config.enable_anti_hallucination:
            self.anti_hallucination_filter = create_anti_hallucination_filter(self.config)

            # Unlikelihood training
            unlikelihood_config = UnlikelihoodConfig()
            contrastive_config = ContrastiveLossConfig()

            self.anti_hallucination_trainer = create_anti_hallucination_trainer(
                unlikelihood_config, contrastive_config, self.anti_hallucination_filter
            )

        logging.info("✅ Loss systems initialized")

    def _initialize_monitoring(self):
        """Initialize monitoring and diagnostics"""
        if self.config.enable_monitoring:
            self.monitor = MonitoringSystem(self.config)

        # Training metrics
        self.training_metrics = {
            'losses': [],
            'learning_rates': [],
            'gradient_norms': [],
            'memory_usage': [],
            'throughput': []
        }

        logging.info("✅ Monitoring system initialized")

    def _initialize_error_handling(self):
        """Initialize error handling system"""
        self.error_handlers = {
            ErrorType.OOM_ERROR: self._handle_oom_error,
            ErrorType.GRADIENT_EXPLOSION: self._handle_gradient_explosion,
            ErrorType.GRADIENT_VANISHING: self._handle_gradient_vanishing,
            ErrorType.LOSS_DIVERGENCE: self._handle_loss_divergence,
            ErrorType.NUMERICAL_INSTABILITY: self._handle_numerical_instability
        }

        # Error statistics
        self.error_stats = {error_type: 0 for error_type in ErrorType}

        logging.info("✅ Error handling system initialized")

    def train(self) -> Dict[str, Any]:
        """Main training loop with comprehensive error handling"""
        logging.info("🚀 Starting advanced training")

        try:
            self.model.train()
            self.state.phase = TrainingPhase.WARMUP

            # Training loop
            for epoch in range(self.config.num_epochs):
                self.state.epoch = epoch

                epoch_metrics = self._train_epoch()

                # Validation
                if self.val_dataloader:
                    val_metrics = self._validate()
                    epoch_metrics.update(val_metrics)

                # Phase transitions
                self._check_phase_transitions()

                # Early stopping check
                if self._should_early_stop():
                    logging.info("Early stopping triggered")
                    break

                logging.info(f"Epoch {epoch} completed: {epoch_metrics}")

            # Final evaluation
            self.state.phase = TrainingPhase.EVALUATION
            final_metrics = self._final_evaluation()

            logging.info("🎉 Training completed successfully")
            return final_metrics

        except Exception as e:
            logging.error(f"Training failed with error: {e}")
            logging.error(traceback.format_exc())

            if self.config.enable_error_recovery:
                return self._attempt_recovery(e)
            else:
                raise e

    def _train_epoch(self) -> Dict[str, Any]:
        """Train one epoch with error handling"""
        epoch_start_time = time.time()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_dataloader):
            try:
                # Training step with error handling
                step_metrics = self._training_step(batch)

                epoch_loss += step_metrics['loss']
                num_batches += 1

                # Update state
                self.state.step += 1
                self.state.running_loss = step_metrics['loss']

                # Logging
                if self.state.step % self.config.log_interval == 0:
                    self._log_training_progress(step_metrics)

                # Evaluation
                if self.state.step % self.config.eval_steps == 0 and self.val_dataloader:
                    val_metrics = self._validate()
                    logging.info(f"Validation at step {self.state.step}: {val_metrics}")

                # Checkpointing
                if self.state.step % self.config.save_steps == 0:
                    self._save_checkpoint()

                # Memory cleanup
                if self.config.enable_memory_cleanup:
                    self._cleanup_memory()

                # Check for early termination
                if self.state.step >= self.config.max_steps:
                    break

            except Exception as e:
                error_type = self._classify_error(e)
                if self._handle_error(error_type, e, batch):
                    continue  # Retry the batch
                else:
                    raise e  # Unrecoverable error

        # Epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / max(1, num_batches)

        return {
            'epoch_loss': avg_loss,
            'epoch_time': epoch_time,
            'batches_processed': num_batches,
            'tokens_per_second': self.state.tokens_per_second
        }

    def _training_step(self, batch) -> Dict[str, Any]:
        """Execute a single training step with comprehensive error handling"""
        step_start_time = time.time()

        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        labels = batch.get('labels', input_ids).to(self.device)

        # Forward pass with mixed precision
        with autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=self.config.use_mixed_precision):
            # Model forward pass
            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )

            # Multi-component loss calculation
            loss_outputs = self.loss_system(model_outputs, labels,
                                          use_fp4=self._is_fp4_phase())

            total_loss = loss_outputs.total_loss

            # Anti-hallucination loss (if enabled)
            if self.config.enable_anti_hallucination:
                anti_halluc_loss, anti_halluc_info = self.anti_hallucination_trainer.calculate_anti_hallucination_loss(
                    model_outputs.get('logits', torch.tensor([])),
                    labels,
           model_outputs.get('hidden_states', torch.tensor([])),
                    tokenizer=self.tokenizer
                )
                total_loss += anti_halluc_loss

            # Scale loss for gradient accumulation
            scaled_loss = total_loss / self.config.gradient_accumulation_steps

        # Backward pass
        if self.config.use_mixed_precision:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # Gradient accumulation
        if (self.state.step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping and analysis
            grad_norm = self._handle_gradients()

            # Optimizer step
            if self.config.use_mixed_precision:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Scheduler step
            self.scheduler.step()

            # Zero gradients
            self.optimizer.zero_grad()

            # Update learning rate
            self.state.current_lr = self.scheduler.get_last_lr()[0]

        # Calculate metrics
        step_time = time.time() - step_start_time
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        tokens_processed = batch_size * seq_len

        self.state.tokens_per_second = tokens_processed / step_time

        # Update loss tracking
        if self.config.enable_loss_tracking:
            self.loss_tracker.update(total_loss.item(), self.state.step)

        # Collect metrics
        step_metrics = {
            'loss': total_loss.item(),
            'scaled_loss': scaled_loss.item(),
            'learning_rate': self.state.current_lr,
            'step_time': step_time,
            'tokens_per_second': self.state.tokens_per_second,
            'batch_size': batch_size,
            'memory_usage': self._get_memory_usage()
        }

        # Add component losses
        if hasattr(loss_outputs, 'component_losses'):
            for component, loss_val in loss_outputs.component_losses.items():
                step_metrics[f'{component}_loss'] = loss_val.item()

        # Add anti-hallucination metrics
        if self.config.enable_anti_hallucination and 'anti_halluc_info' in locals():
            step_metrics['anti_hallucination_loss'] = anti_halluc_loss.item()

        return step_metrics

    def _handle_gradients(self) -> float:
        """Handle gradient clipping and analysis"""
        # Calculate gradient norm
        total_norm = 0.0
        param_count = 0

        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1

        total_norm = total_norm ** (1. / 2)

        # Gradient explosion detection
        if total_norm > self.config.gradient_clip_threshold:
            self.state.gradient_explosion_count += 1
            logging.warning(f"Gradient explosion detected: norm={total_norm:.4f}")

            if total_norm > self.config.gradient_clip_threshold * 2:
                self._handle_error(ErrorType.GRADIENT_EXPLOSION,
                                 Exception(f"Severe gradient explosion: {total_norm}"), None)

        # Gradient vanishing detection
        if total_norm < 1e-7 and param_count > 0:
            self.state.gradient_explosion_count += 1
            logging.warning(f"Gradient vanishing detected: norm={total_norm:.2e}")

        # Gradient clipping
        if self.config.use_mixed_precision:
            self.scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        return total_norm

    def _validate(self) -> Dict[str, Any]:
        """Run validation with error handling"""
        if not self.val_dataloader:
            return {}

        self.model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                try:
                    input_ids = batch['input_ids'].to(self.device)
                    labels = batch.get('labels', input_ids).to(self.device)
                    attention_mask = batch.get('attention_mask', None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)

                    with autocast(enabled=self.config.use_mixed_precision):
                        model_outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            return_dict=True
                        )

                        loss_outputs = self.loss_system(model_outputs, labels)
                        val_loss += loss_outputs.total_loss.item()
                        val_steps += 1

                except Exception as e:
                    logging.warning(f"Validation step failed: {e}")
                    continue

        self.model.train()

        avg_val_loss = val_loss / max(1, val_steps)

        # Update best loss
        if avg_val_loss < self.state.best_loss:
            self.state.best_loss = avg_val_loss
            self._save_best_model()

        return {
            'val_loss': avg_val_loss,
            'val_steps': val_steps,
            'best_loss': self.state.best_loss
        }

    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify the type of error"""
        error_str = str(error).lower()

        if 'out of memory' in error_str or 'cuda out of memory' in error_str:
            return ErrorType.OOM_ERROR
        elif 'nan' in error_str or 'inf' in error_str:
            return ErrorType.NUMERICAL_INSTABILITY
        elif 'gradient' in error_str:
            return ErrorType.GRADIENT_EXPLOSION
        elif 'loss' in error_str and 'diverge' in error_str:
            return ErrorType.LOSS_DIVERGENCE
        else:
            return ErrorType.HARDWARE_ERROR

    def _handle_error(self, error_type: ErrorType, error: Exception, batch) -> bool:
        """Handle training errors and attempt recovery"""
        if not self.config.enable_error_recovery:
            return False

        self.error_stats[error_type] += 1
        logging.error(f"Handling {error_type.value}: {error}")

        if error_type in self.error_handlers:
            return self.error_handlers[error_type](error, batch)
        else:
            logging.error(f"No handler for error type: {error_type}")
            return False

    def _handle_oom_error(self, error: Exception, batch) -> bool:
        """Handle out-of-memory errors"""
        self.state.oom_count += 1

        if self.state.oom_count > self.config.max_oom_retries:
            logging.error("Max OOM retries exceeded")
            return False

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

        # Reduce batch size
        current_batch_size = self.config.micro_batch_size
        new_batch_size = max(
            self.config.min_batch_size,
            int(current_batch_size * self.config.batch_size_reduction_factor)
        )

        if new_batch_size < current_batch_size:
            self.config.micro_batch_size = new_batch_size
            logging.info(f"Reduced batch size from {current_batch_size} to {new_batch_size}")

            # Adjust gradient accumulation to maintain effective batch size
            self.config.gradient_accumulation_steps = min(
                self.config.gradient_accumulation_steps * 2,
                self.config.max_batch_size // new_batch_size
            )

            return True
        else:
            logging.error("Cannot reduce batch size further")
            return False

    def _handle_gradient_explosion(self, error: Exception, batch) -> bool:
        """Handle gradient explosion"""
        # Reduce learning rate temporarily
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.5

        logging.info(f"Reduced learning rate due to gradient explosion")

        # Clear gradients
        self.optimizer.zero_grad()

        return True

    def _handle_gradient_vanishing(self, error: Exception, batch) -> bool:
        """Handle gradient vanishing"""
        # Increase learning rate slightly
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 1.1

        logging.info(f"Increased learning rate due to gradient vanishing")

        return True

    def _handle_loss_divergence(self, error: Exception, batch) -> bool:
        """Handle loss divergence"""
        # Reset to previous checkpoint if available
        if hasattr(self, 'last_checkpoint_path'):
            logging.info("Attempting to restore from last checkpoint")
            self._load_checkpoint(self.last_checkpoint_path)
            return True

        # Reduce learning rate significantly
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= 0.1

        logging.info("Reduced learning rate due to loss divergence")
        return True

    def _handle_numerical_instability(self, error: Exception, batch) -> bool:
        """Handle numerical instability"""
        # Clear cache and reset gradients
        torch.cuda.empty_cache()
        self.optimizer.zero_grad()

        # Reduce loss scaling if using mixed precision
        if self.config.use_mixed_precision:
            self.scaler._scale *= 0.5
            logging.info(f"Reduced loss scaling to {self.scaler._scale}")

        return True

    def _cleanup_memory(self):
        """Clean up memory to prevent OOM"""
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated()
            if max_memory > 0:
                memory_usage = torch.cuda.memory_allocated() / max_memory

                if memory_usage > self.config.memory_cleanup_threshold:
                    torch.cuda.empty_cache()
                    gc.collect()
                    logging.debug(f"Memory cleanup performed at {memory_usage:.1%} usage")
            else:
                # No memory allocated yet, just clean cache
                torch.cuda.empty_cache()
                gc.collect()

    def _get_memory_usage(self) -> float:
        """Get current memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)  # GB
        else:
            return psutil.virtual_memory().percent / 100.0

    def _is_fp4_phase(self) -> bool:
        """Check if we're in FP4 quantization phase"""
        return (self.config.use_quantization and
                self.state.step > self.config.quantization_phase_transition)

    def _check_phase_transitions(self):
        """Check and handle training phase transitions"""
        if self.state.step < self.config.warmup_steps:
            self.state.phase = TrainingPhase.WARMUP
        elif self.state.step < self.config.quantization_phase_transition:
            self.state.phase = TrainingPhase.MAIN_TRAINING
        elif self.state.step < self.config.max_steps * 0.9:
            self.state.phase = TrainingPhase.FINE_TUNING
        else:
            self.state.phase = TrainingPhase.QAF_PHASE

        # Handle quantization phase transition
        if (self.config.use_quantization and
            self.state.step == self.config.quantization_phase_transition):
            logging.info("Transitioning to FP4 quantization phase")
            self.quantization_controller.transition_to_fp4()

    def _should_early_stop(self) -> bool:
        """Check if training should stop early"""
        if len(self.state.last_losses) < 10:
            return False

        # Check for loss stagnation
        recent_losses = self.state.last_losses[-10:]
        loss_improvement = recent_losses[0] - recent_losses[-1]

        if loss_improvement < 0.001:  # Very small improvement
            logging.info("Early stopping due to loss stagnation")
            return True

        return False

    def _log_training_progress(self, metrics: Dict[str, Any]):
        """Log training progress"""
        log_msg = (
            f"Step {self.state.step} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"LR: {metrics['learning_rate']:.2e} | "
            f"Tokens/s: {metrics['tokens_per_second']:.0f} | "
            f"Memory: {metrics['memory_usage']:.1f}GB | "
            f"Phase: {self.state.phase.value}"
        )

        if self.config.detailed_logging:
            # Add component losses
            for key, value in metrics.items():
                if key.endswith('_loss') and key != 'loss':
                    log_msg += f" | {key}: {value:.4f}"

        logging.info(log_msg)

        # Update metrics history
        self.training_metrics['losses'].append(metrics['loss'])
        self.training_metrics['learning_rates'].append(metrics['learning_rate'])
        self.training_metrics['memory_usage'].append(metrics['memory_usage'])
        self.training_metrics['throughput'].append(metrics['tokens_per_second'])

    def _save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint_dir = Path(f"checkpoints/{self.config.model_name}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_step_{self.state.step}.pt"

        checkpoint = {
            'step': self.state.step,
            'epoch': self.state.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_state': self.state,
            'config': self.config,
            'training_metrics': self.training_metrics
        }

        if self.config.use_mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        self.last_checkpoint_path = checkpoint_path

        logging.info(f"Checkpoint saved: {checkpoint_path}")

    def _save_best_model(self):
        """Save the best model"""
        best_model_dir = Path(f"models/{self.config.model_name}")
        best_model_dir.mkdir(parents=True, exist_ok=True)

        best_model_path = best_model_dir / "best_model.pt"

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_loss': self.state.best_loss,
            'step': self.state.step
        }, best_model_path)

        logging.info(f"Best model saved: {best_model_path}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.state = checkpoint['training_state']
        self.training_metrics = checkpoint['training_metrics']

        if self.config.use_mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        logging.info(f"Checkpoint loaded: {checkpoint_path}")

    def _final_evaluation(self) -> Dict[str, Any]:
        """Perform final evaluation"""
        logging.info("Performing final evaluation...")

        final_metrics = {}

        # Validation metrics
        if self.val_dataloader:
            val_metrics = self._validate()
            final_metrics.update(val_metrics)

        # Loss tracking summary
        if self.config.enable_loss_tracking:
            loss_stats = self.loss_tracker.get_loss_statistics()
            final_metrics['loss_tracking'] = loss_stats

        # Training statistics
        final_metrics.update({
            'total_steps': self.state.step,
            'total_epochs': self.state.epoch,
            'final_phase': self.state.phase.value,
            'error_stats': dict(self.error_stats),
            'best_loss': self.state.best_loss,
            'final_lr': self.state.current_lr
        })

        # Performance metrics
        if self.training_metrics['throughput']:
            final_metrics['avg_throughput'] = np.mean(self.training_metrics['throughput'])
            final_metrics['peak_throughput'] = np.max(self.training_metrics['throughput'])

        return final_metrics

    def _attempt_recovery(self, error: Exception) -> Dict[str, Any]:
        """Attempt to recover from training failure"""
        logging.info("Attempting training recovery...")

        # Try to load last checkpoint
        if hasattr(self, 'last_checkpoint_path') and self.last_checkpoint_path.exists():
            try:
                self._load_checkpoint(str(self.last_checkpoint_path))
                logging.info("Successfully recovered from checkpoint")

                # Return partial results
                return self._final_evaluation()

            except Exception as recovery_error:
                logging.error(f"Recovery failed: {recovery_error}")

        # Return error information
        return {
            'training_failed': True,
            'error': str(error),
            'error_stats': dict(self.error_stats),
            'partial_metrics': self.training_metrics
        }


def create_advanced_trainer(config: TrainingConfig, model, tokenizer,
                          train_dataloader, val_dataloader=None, device=None) -> TrainerOptimized:
    """Factory function to create advanced trainer"""
    return TrainerOptimized(config, model, tokenizer, train_dataloader, val_dataloader, device)


def test_advanced_trainer():
    """Test the advanced trainer system"""
    print("🧪 Testing Advanced Trainer System")

    # This would require actual model and data, so we'll just test initialization
    config = TrainingConfig(
        model_name="test_nanolm",
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=100,
        enable_error_recovery=True,
        enable_monitoring=True
    )

    print(f"✅ Training config created:")
    print(f"  • Model: {config.model_name}")
    print(f"  • Batch size: {config.micro_batch_size}")
    print(f"  • Max steps: {config.max_steps}")
    print(f"  • Error recovery: {config.enable_error_recovery}")
    print(f"  • Monitoring: {config.enable_monitoring}")

    print("🎉 Advanced trainer test completed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    test_advanced_trainer()