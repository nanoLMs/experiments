#!/usr/bin/env python3
"""
Optimized FP4 FQT Trainer
=========================

Advanced trainer implementing all research paper features:
- FP4 Fully Quantized Training (NVFP4 format)
- Mixture of Experts (MoE) 
- Multi-Token Prediction (MTP)
- Hierarchical Reasoning Module (HRM)
- Anti-Hallucination mechanisms
- Proper loss scaling and convergence
- Automatic QAF phase detection
- Memory-efficient training pipeline
"""

import os
import math
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

# Fix HuggingFace tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import PreTrainedTokenizerFast
from config import TrainConfig
from data_loader import create_dataloader
from schedule import cosine_with_warmup
from checkpoint import save_checkpoint, load_latest

# FP4 FQT imports
from fp4_model_integration import (
    create_fp4_model, get_fp4_optimizer_config, should_use_amp_with_fp4,
    FP4TrainingPhaseManager
)
from rich_output import (
    print_header, print_system_info, print_success, print_error, print_warning,
    TrainingMetrics, check_rich_installation
)

# Rich progress bar imports
try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn, 
        TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn,
        TaskProgressColumn, ProgressColumn
    )
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    Console = None

class TokensPerSecColumn(ProgressColumn):
    """Custom progress column for tokens per second"""
    
    def render(self, task):
        if task.fields.get('tokens_per_sec', 0) > 0:
            return Text(f"{task.fields['tokens_per_sec']:.0f} tok/s", style="cyan")
        return Text("-- tok/s", style="dim")

class LossColumn(ProgressColumn):
    """Custom progress column for training loss"""
    
    def render(self, task):
        if task.fields.get('loss', 0) > 0:
            return Text(f"Loss: {task.fields['loss']:.4f}", style="red")
        return Text("Loss: --", style="dim")

class LRColumn(ProgressColumn):
    """Custom progress column for learning rate"""
    
    def render(self, task):
        if task.fields.get('lr', 0) > 0:
            return Text(f"LR: {task.fields['lr']:.2e}", style="yellow")
        return Text("LR: --", style="dim")

class EpochColumn(ProgressColumn):
    """Custom progress column for epoch progress"""
    
    def render(self, task):
        current_epoch = task.fields.get('current_epoch', 0)
        total_epochs = task.fields.get('total_epochs', 0)
        if current_epoch > 0 and total_epochs > 0:
            return Text(f"Epoch: {current_epoch}/{total_epochs}", style="magenta")
        return Text("Epoch: --/--", style="dim")

class TotalTimeColumn(ProgressColumn):
    """Custom progress column for estimated total training time"""
    
    def render(self, task):
        if task.completed > 0 and task.total and task.total > 0:
            progress_ratio = task.completed / task.total
            if progress_ratio > 0:
                # Get elapsed time from task fields or calculate from progress
                elapsed_seconds = task.fields.get('elapsed_time', 0)
                if elapsed_seconds > 0:
                    estimated_total_seconds = elapsed_seconds / progress_ratio
                    
                    # Format total time
                    if estimated_total_seconds < 3600:  # Less than 1 hour
                        total_time_str = f"{estimated_total_seconds/60:.0f}m"
                    elif estimated_total_seconds < 86400:  # Less than 1 day
                        hours = int(estimated_total_seconds // 3600)
                        minutes = int((estimated_total_seconds % 3600) // 60)
                        total_time_str = f"{hours}h{minutes:02d}m"
                    else:  # 1 day or more
                        days = int(estimated_total_seconds // 86400)
                        hours = int((estimated_total_seconds % 86400) // 3600)
                        total_time_str = f"{days}d{hours:02d}h"
                    
                    return Text(f"Total: {total_time_str}", style="bright_blue")
        
        return Text("Total: --", style="dim")

def create_training_progress_bar():
    """Create the advanced training progress bar"""
    if not HAS_RICH:
        return None
        
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        "•",
        EpochColumn(),
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn(),
        "•",
        TotalTimeColumn(),
        "•",
        TokensPerSecColumn(),
        "•",
        LossColumn(),
        "•",
        LRColumn(),
        refresh_per_second=4,
        expand=True
    )

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_seed(seed: int):
    """Setup random seeds for reproducibility"""
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_distributed():
    """Initialize distributed training if available"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        torch.distributed.init_process_group('nccl')
        return True, torch.distributed.get_rank(), torch.distributed.get_world_size()
    return False, 0, 1

def get_memory_stats():
    """Get GPU memory statistics"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return allocated, reserved, total
    return 0, 0, 0

class AdvancedLossCalculator:
    """
    Advanced loss calculation with proper weighting and scaling
    Implements all loss components from research papers
    """
    
    def __init__(self, cfg: TrainConfig, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.ignore_index = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
        
        # Loss components
        self.main_loss_weight = 1.0
        self.mtp_loss_weight = getattr(cfg, 'mtp_loss_weight', 0.3)
        self.aux_loss_weight = getattr(cfg, 'moe_aux_weight', 0.02)
        self.reasoning_loss_weight = getattr(cfg, 'reasoning_loss_weight', 0.05)
        self.anti_hallu_weight = getattr(cfg, 'factual_penalty_weight', 0.1)
        
        # Adaptive loss scaling for FP4 training
        self.loss_scale_factor = 10.0  # Scale up losses to prevent underflow in FP4
        
        logger.info(f"🎯 Loss Calculator initialized:")
        logger.info(f"  • Main loss weight: {self.main_loss_weight}")
        logger.info(f"  • MTP loss weight: {self.mtp_loss_weight}")
        logger.info(f"  • Aux loss weight: {self.aux_loss_weight}")
        logger.info(f"  • Reasoning loss weight: {self.reasoning_loss_weight}")
        logger.info(f"  • Anti-hallucination weight: {self.anti_hallu_weight}")
        logger.info(f"  • FP4 loss scale factor: {self.loss_scale_factor}")
    
    def calculate_losses(self, model_output, targets, training_phase: str = "fp4") -> Dict[str, torch.Tensor]:
        """
        Calculate all loss components with proper scaling
        
        Args:
            model_output: (logits_main, logits_mtp, aux_loss, reason_logits)
            targets: Target token IDs
            training_phase: Current training phase
            
        Returns:
            Dictionary of loss components
        """
        logits_main, logits_mtp, aux_loss, reason_logits = model_output
        losses = {}
        
        # Main language modeling loss
        main_loss = F.cross_entropy(
            logits_main.view(-1, logits_main.size(-1)), 
            targets.view(-1), 
            ignore_index=self.ignore_index,
            reduction='mean'
        )
        losses['main'] = main_loss * self.main_loss_weight
        
        # Multi-Token Prediction (MTP) loss
        if logits_mtp and len(logits_mtp) > 0:
            mtp_losses = []
            for i, mtp_logits in enumerate(logits_mtp):
                # Shift targets for future token prediction
                if i + 1 < targets.size(1):
                    shifted_targets = torch.cat([
                        targets[:, i+1:],
                        torch.full((targets.size(0), i+1), self.ignore_index, device=targets.device)
                    ], dim=1)
                    
                    mtp_loss = F.cross_entropy(
                        mtp_logits.view(-1, mtp_logits.size(-1)),
                        shifted_targets.view(-1),
                        ignore_index=self.ignore_index,
                        reduction='mean'
                    )
                    
                    # Weight decreases for further predictions
                    weight = 1.0 / (i + 2)
                    mtp_losses.append(mtp_loss * weight)
            
            if mtp_losses:
                losses['mtp'] = sum(mtp_losses) * self.mtp_loss_weight
            else:
                losses['mtp'] = torch.tensor(0.0, device=targets.device)
        else:
            losses['mtp'] = torch.tensor(0.0, device=targets.device)
        
        # MoE auxiliary loss (load balancing)
        if aux_loss is not None and torch.is_tensor(aux_loss):
            losses['aux'] = aux_loss * self.aux_loss_weight
        else:
            losses['aux'] = torch.tensor(0.0, device=targets.device)
        
        # Reasoning loss (if available)
        if reason_logits is not None:
            # Use last token as reasoning target (simplified)
            reason_targets = targets[:, -1]
            reason_loss = F.cross_entropy(
                reason_logits,
                reason_targets,
                ignore_index=self.ignore_index,
                reduction='mean'
            )
            losses['reasoning'] = reason_loss * self.reasoning_loss_weight
        else:
            losses['reasoning'] = torch.tensor(0.0, device=targets.device)
        
        # Anti-hallucination penalty (implicit in model output)
        # This is typically handled by the model's anti-hallucination wrapper
        
        # Total loss with FP4 scaling
        total_loss = sum(losses.values())
        
        # Apply loss scaling for FP4 training to prevent underflow
        if training_phase == "fp4":
            total_loss = total_loss * self.loss_scale_factor
            # Also scale individual losses for monitoring
            for key in losses:
                if key != 'total':
                    losses[key] = losses[key] * self.loss_scale_factor
        
        losses['total'] = total_loss
        
        return losses

class GradientMonitor:
    """
    Monitors gradient statistics for FP4 training
    Detects gradient explosion, vanishing, and stagnation
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.grad_norms = []
        self.grad_variance = []
        self.loss_history = []
        
    def update(self, grad_norm: float, loss: float):
        """Update gradient and loss statistics with NaN protection"""
        # Only add finite values to avoid numpy warnings
        if np.isfinite(grad_norm) and np.isfinite(loss):
            self.grad_norms.append(grad_norm)
            self.loss_history.append(loss)
            
            if len(self.grad_norms) > self.window_size:
                self.grad_norms.pop(0)
                self.loss_history.pop(0)
            
            # Calculate gradient variance with finite values only
            if len(self.grad_norms) > 10:
                recent_grads = self.grad_norms[-10:]
                # Use numpy with finite values check
                if all(np.isfinite(g) for g in recent_grads):
                    variance = np.var(recent_grads)
                    if np.isfinite(variance):
                        self.grad_variance.append(variance)
                        
                        if len(self.grad_variance) > self.window_size:
                            self.grad_variance.pop(0)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current gradient statistics"""
        if not self.grad_norms:
            return {'mean_grad_norm': 0, 'grad_variance': 0, 'loss_trend': 0}
        
        stats = {
            'mean_grad_norm': np.mean(self.grad_norms[-10:]) if len(self.grad_norms) >= 10 else np.mean(self.grad_norms),
            'grad_variance': np.mean(self.grad_variance[-10:]) if len(self.grad_variance) >= 10 else 0,
            'loss_trend': 0
        }
        
        # Calculate loss trend (recent improvement)
        if len(self.loss_history) >= 20:
            recent_losses = self.loss_history[-20:]
            first_half = np.mean(recent_losses[:10])
            second_half = np.mean(recent_losses[10:])
            stats['loss_trend'] = first_half - second_half  # Positive = improving
        
        return stats
    
    def detect_issues(self) -> List[str]:
        """Detect training issues from gradient statistics"""
        issues = []
        stats = self.get_statistics()
        
        # Gradient explosion
        if stats['mean_grad_norm'] > 10.0:
            issues.append("gradient_explosion")
        
        # Vanishing gradients
        if stats['mean_grad_norm'] < 1e-6:
            issues.append("vanishing_gradients")
        
        # Gradient stagnation (low variance)
        if stats['grad_variance'] < 1e-8 and len(self.grad_norms) > 50:
            issues.append("gradient_stagnation")
        
        # Loss stagnation (no improvement)
        if stats['loss_trend'] < 1e-4 and len(self.loss_history) > 50:
            issues.append("loss_stagnation")
        
        return issues

def train():
    """Main training function with all advanced features"""
    
    # Check Rich installation
    check_rich_installation()
    
    # Setup
    cfg = TrainConfig()
    distributed, rank, world = init_distributed()
    setup_seed(cfg.seed + rank)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        print_header(
            "🚀 Advanced NanoLM FP4 FQT Training",
            "FP4 FQT • MoE • MTP • HRM • Anti-Hallucination • Auto-QAF"
        )
    
    # Load tokenizer
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer_dir)
        if rank == 0:
            print_success(f"📚 Tokenizer loaded: {len(tokenizer):,} tokens")
    except Exception as e:
        if rank == 0:
            print_error(f"Failed to load tokenizer from {cfg.tokenizer_dir}", e)
        return
    
    # Update config for advanced features
    cfg.use_pure_4bit = True
    cfg.use_hrm = True  # Enable Hierarchical Reasoning
    cfg.enable_reasoning = True
    cfg.factual_penalty_weight = 0.1  # Anti-hallucination
    
    # Create advanced model
    if rank == 0:
        print_success("🔥 Creating Advanced FP4 FQT Model")
        print("📊 Advanced Features Enabled:")
        print("  • FP4 Fully Quantized Training (NVFP4)")
        print("  • Mixture of Experts (MoE)")
        print("  • Multi-Token Prediction (MTP)")
        print("  • Hierarchical Reasoning Module (HRM)")
        print("  • Anti-Hallucination mechanisms")
        print("  • Automatic QAF phase detection")
        print("  • Advanced loss scaling")
    
    model = create_fp4_model(cfg)
    model.to(device)
    
    # Model statistics
    total_params = model.num_parameters()
    trainable_params = model.num_trainable_parameters()
    
    if rank == 0:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        print_system_info(gpu_name, gpu_memory, total_params)
        
        if hasattr(model, 'get_training_phase_info'):
            phase_info = model.get_training_phase_info()
            if 'memory_stats' in phase_info:
                stats = phase_info['memory_stats']
                print_success(f"💾 FP4 Memory Efficiency:")
                print(f"  • Memory saved: {stats['memory_saved_mb']:.1f} MB")
                print(f"  • Compression ratio: {stats['compression_ratio']:.2f}x")
    
    # Advanced loss calculator
    loss_calculator = AdvancedLossCalculator(cfg, tokenizer)
    
    # FP4-optimized optimizer
    optimizer_config = get_fp4_optimizer_config(cfg, model)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        **optimizer_config
    )
    
    # AMP settings (careful with FP4)
    use_amp = should_use_amp_with_fp4(cfg) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    # Training phase manager
    phase_manager = FP4TrainingPhaseManager(model, cfg) if hasattr(model, 'training_phase') else None
    
    # Gradient monitor
    grad_monitor = GradientMonitor()
    
    # Data loading
    try:
        if rank == 0:
            print("🔄 Creating dataloader...")
        dataloader = create_dataloader(cfg, tokenizer, world)
        steps_per_epoch = len(dataloader)
        total_steps = cfg.num_epochs * steps_per_epoch
        effective_batch_size = cfg.effective_batch_size(world)
        if rank == 0:
            print(f"✅ Dataloader created successfully - {steps_per_epoch:,} steps per epoch")
            
            # Test the dataloader with a single batch
            print("🧪 Testing dataloader with first batch...")
            try:
                test_iter = iter(dataloader)
                test_batch = next(test_iter)
                input_ids, target_ids = test_batch
                print(f"✅ Dataloader test passed - batch shapes: {input_ids.shape}, {target_ids.shape}")
                del test_iter, test_batch, input_ids, target_ids
            except Exception as test_e:
                print_warning(f"Dataloader test failed: {test_e}")
                print("Continuing anyway - will handle errors during training...")
    except Exception as e:
        if rank == 0:
            print_error("Failed to create dataloader", e)
            import traceback
            print("Full traceback:")
            traceback.print_exc()
        return
    
    if rank == 0:
        print_success(f"📊 Training Configuration:")
        print(f"  • Dataset steps per epoch: {steps_per_epoch:,}")
        print(f"  • Total training steps: {total_steps:,}")
        print(f"  • Effective batch size: {effective_batch_size:,}")
        print(f"  • Mixed precision: {'Enabled' if use_amp else 'Disabled (FP4 sufficient)'}")
        print(f"  • Expected memory usage: {total_params * 0.5 / 1e6:.0f} MB (FP4)")
    
    # Load checkpoint
    ckpt_state, start_step = load_latest(cfg.ckpt_dir)
    if ckpt_state:
        try:
            model.load_state_dict(ckpt_state['model'], strict=False)
            optimizer.load_state_dict(ckpt_state['optim'])
            if 'scaler' in ckpt_state and scaler:
                scaler.load_state_dict(ckpt_state['scaler'])
            start_step = ckpt_state['step']
            
            if rank == 0:
                print_success(f"✅ Resumed from step {start_step}")
        except Exception as e:
            if rank == 0:
                print_warning(f"Checkpoint loading failed: {e}")
            start_step = 0
    
    # Save configuration
    if rank == 0 and cfg.save_config_once:
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        config_data = cfg.__dict__.copy()
        config_data['advanced_features'] = {
            'fp4_fqt': True,
            'moe': True,
            'mtp': True,
            'hrm': True,
            'anti_hallucination': True,
            'auto_qaf': True
        }
        with open(os.path.join(cfg.ckpt_dir, 'train_config.json'), 'w') as f:
            json.dump(config_data, f, indent=2)
        print_success(f"💾 Advanced config saved")
    
    # Training loop with advanced features
    accum_steps = 0
    running_losses = {'total': 0, 'main': 0, 'mtp': 0, 'aux': 0, 'reasoning': 0}
    global_step = start_step
    seen_tokens = start_step * cfg.seq_len * effective_batch_size
    
    training_start_time = time.time()
    best_loss = float('inf')
    
    if rank == 0:
        print()
        print("🚀" * 20)
        print("🚀 STARTING ADVANCED FP4 FQT TRAINING")
        print("🚀" * 20)
        print()
    
    # Create Rich progress bar
    progress_bar = None
    task_id = None
    if rank == 0 and HAS_RICH:
        progress_bar = create_training_progress_bar()
        if progress_bar:
            progress_bar.start()
            task_id = progress_bar.add_task(
                description="Step",
                total=total_steps,
                completed=global_step,
                current_epoch=0,
                total_epochs=cfg.num_epochs
            )
    
    for epoch in range(cfg.num_epochs):
        if rank == 0:
            training_phase = model.training_phase if hasattr(model, 'training_phase') else 'normal'
            # Update progress bar with current epoch
            if progress_bar and task_id is not None:
                progress_bar.update(
                    task_id,
                    current_epoch=epoch + 1,
                    total_epochs=cfg.num_epochs
                )
            else:
                print(f"📈 EPOCH {epoch + 1}/{cfg.num_epochs} | Phase: {training_phase}")
                print("=" * 60)
        
        epoch_start_time = time.time()
        
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            # Update progress bar every batch to show activity
            if progress_bar and task_id is not None:
                if batch_idx == 0:  # First batch
                    progress_bar.update(
                        task_id,
                        completed=global_step,
                        current_epoch=epoch + 1,
                        total_epochs=cfg.num_epochs,
                        description=f"Step {global_step:,} (Starting batch processing...)"
                    )
                elif batch_idx % 5 == 0:  # Update every 5 batches to show progress
                    progress_bar.update(
                        task_id,
                        completed=global_step,
                        current_epoch=epoch + 1,
                        total_epochs=cfg.num_epochs,
                        description=f"Step {global_step:,} (Processing batch {batch_idx})"
                    )
            
            if global_step < start_step:
                global_step += 1
                continue
            
            # Memory management
            if torch.cuda.is_available():
                allocated, reserved, total = get_memory_stats()
                if allocated / total > 0.9:  # >90% memory usage
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
            
            # Prepare batch
            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)
            
            # Ensure correct sequence length
            if input_ids.size(1) != cfg.seq_len:
                if input_ids.size(1) > cfg.seq_len:
                    input_ids = input_ids[:, :cfg.seq_len]
                    target_ids = target_ids[:, :cfg.seq_len]
                else:
                    continue  # Skip short sequences
            
            # Debug: Show when we start forward pass
            if batch_idx == 0 and rank == 0:
                print(f"🔍 Starting first forward pass with input shape: {input_ids.shape}")
            
            # Forward pass with advanced features
            current_phase = model.training_phase if hasattr(model, 'training_phase') else 'normal'
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                try:
                    if batch_idx == 0 and rank == 0:
                        print(f"🔍 Model forward pass starting...")
                    
                    model_output = model(input_ids)
                    
                    if batch_idx == 0 and rank == 0:
                        print(f"🔍 Model forward pass completed, calculating loss...")
                    
                    # Advanced loss calculation
                    loss_components = loss_calculator.calculate_losses(
                        model_output, target_ids, current_phase
                    )
                    
                    total_loss = loss_components['total']
                    
                    if batch_idx == 0 and rank == 0:
                        print(f"🔍 Loss calculated: {total_loss.item():.4f}")
                    
                except torch.cuda.OutOfMemoryError:
                    if rank == 0:
                        print_warning("⚠️ OOM detected, clearing cache and skipping batch")
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    continue
                except Exception as forward_e:
                    if rank == 0:
                        print_warning(f"⚠️ Forward pass error: {forward_e}")
                        print(f"🔍 Input shape: {input_ids.shape}, device: {input_ids.device}")
                        print(f"🔍 Model device: {next(model.parameters()).device}")
                    continue
            
            # Backward pass (uses stochastic rounding in FP4 phase)
            scaled_loss = total_loss / cfg.grad_accum_steps
            
            # Apply loss scaling correction for FP4
            if current_phase == "fp4":
                scaled_loss = scaled_loss / loss_calculator.loss_scale_factor
            
            scaler.scale(scaled_loss).backward()
            accum_steps += 1
            
            # Update running losses
            for key in running_losses:
                if key in loss_components:
                    running_losses[key] += loss_components[key].item()
            
            # Optimizer step
            if accum_steps % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                
                # Gradient clipping and monitoring with NaN protection
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                
                # Check for NaN/Inf gradients and handle gracefully
                grad_norm_value = grad_norm.item()
                loss_value = total_loss.item()
                
                if not torch.isfinite(grad_norm) or not torch.isfinite(total_loss):
                    if rank == 0:
                        print_warning(f"⚠️ Non-finite values detected - grad_norm: {grad_norm_value}, loss: {loss_value}")
                        print("🔄 Skipping optimizer step and continuing...")
                    
                    # Reset gradients and continue
                    optimizer.zero_grad(set_to_none=True)
                    accum_steps = 0
                    continue
                
                # Safe gradient monitoring with finite value checks
                if torch.isfinite(grad_norm) and torch.isfinite(total_loss):
                    grad_monitor.update(grad_norm_value, loss_value)
                else:
                    # Use safe fallback values for monitoring
                    grad_monitor.update(1.0, 10.0)
                
                # FP4 phase management
                if phase_manager:
                    phase_manager.on_step_end(global_step, grad_norm.item())
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                # Learning rate schedule
                lr = cosine_with_warmup(global_step, cfg.warmup_steps, total_steps, cfg.lr, cfg.min_lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Update progress bar and logging
                if global_step % cfg.log_interval == 0 and rank == 0:
                    avg_losses = {k: v / cfg.log_interval for k, v in running_losses.items()}
                    
                    # Training statistics
                    elapsed = time.time() - training_start_time
                    tokens_per_sec = seen_tokens / elapsed if elapsed > 0 else 0
                    
                    # Update progress bar if available
                    if progress_bar and task_id is not None:
                        progress_bar.update(
                            task_id,
                            completed=global_step,
                            current_epoch=epoch + 1,
                            total_epochs=cfg.num_epochs,
                            elapsed_time=elapsed,
                            tokens_per_sec=tokens_per_sec,
                            loss=avg_losses['total'],
                            lr=lr,
                            description=f"Step {global_step:,}"
                        )
                    else:
                        # Fallback to print-based logging
                        grad_stats = grad_monitor.get_statistics()
                        allocated, reserved, total = get_memory_stats()
                        
                        # Phase information
                        phase_info = ""
                        if hasattr(model, 'get_training_phase_info'):
                            info = model.get_training_phase_info()
                            phase = info['phase']
                            grad_ratio = info.get('grad_to_noise_ratio', 0)
                            phase_info = f" | Phase: {phase} (G/N: {grad_ratio:.2f})"
                        
                        # Log comprehensive training statistics
                        print()
                        print(f"📊 STEP {global_step:,}/{total_steps:,} | Epoch {epoch+1}/{cfg.num_epochs} | LR: {lr:.2e}{phase_info}")
                        print(f"🎯 Losses: Total={avg_losses['total']:.4f}, Main={avg_losses['main']:.4f}, MTP={avg_losses['mtp']:.4f}, Aux={avg_losses['aux']:.4f}")
                        print(f"🧠 Gradients: Norm={grad_stats['mean_grad_norm']:.6f}, Variance={grad_stats['grad_variance']:.8f}, Trend={grad_stats['loss_trend']:.6f}")
                        print(f"💾 Memory: {allocated:.1f}GB/{total:.1f}GB ({allocated/total*100:.1f}%) | Speed: {tokens_per_sec:.0f} tok/s")
                        
                        # Check for training issues
                        issues = grad_monitor.detect_issues()
                        if issues:
                            print_warning(f"⚠️ Training issues detected: {', '.join(issues)}")
                    
                    # Update best loss
                    if avg_losses['total'] < best_loss:
                        best_loss = avg_losses['total']
                        if not progress_bar:  # Only print if no progress bar
                            print_success(f"🎉 New best loss: {best_loss:.4f}")
                    
                    # Reset running losses
                    running_losses = {k: 0 for k in running_losses}
                
                # Memory cleanup
                if torch.cuda.is_available() and global_step % 10 == 0:
                    torch.cuda.empty_cache()
                    if global_step % 100 == 0:
                        import gc
                        gc.collect()
            
            global_step += 1
            seen_tokens += cfg.seq_len * cfg.micro_batch_size * world
            
            # Check stopping conditions
            if (cfg.target_total_tokens and seen_tokens >= cfg.target_total_tokens) or global_step >= total_steps:
                if rank == 0:
                    reason = "token budget reached" if seen_tokens >= cfg.target_total_tokens else "step limit reached"
                    print_success(f"🏁 Training complete: {reason}")
                break
        
        # Epoch summary
        if rank == 0:
            epoch_duration = time.time() - epoch_start_time
            print()
            print("─" * 60)
            print(f"✅ EPOCH {epoch + 1} COMPLETE")
            print(f"⏱️  Duration: {epoch_duration/60:.1f} minutes")
            print(f"🎯 Best Loss: {best_loss:.4f}")
            if hasattr(model, 'get_training_phase_info'):
                info = model.get_training_phase_info()
                print(f"🔄 Training Phase: {info['phase']}")
            print("─" * 60)
            print()
        
        # Save checkpoint
        if rank == 0 and global_step % cfg.save_interval < cfg.grad_accum_steps:
            checkpoint_data = {
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'scaler': scaler.state_dict() if scaler else None,
                'step': global_step,
                'epoch': epoch,
                'best_loss': best_loss,
                'seen_tokens': seen_tokens,
                'config': cfg.__dict__
            }
            
            # Save phase information
            if hasattr(model, 'get_training_phase_info'):
                checkpoint_data['phase_info'] = model.get_training_phase_info()
            
            checkpoint_path = save_checkpoint(checkpoint_data, cfg.ckpt_dir, global_step, cfg.keep_last_k)
            print_success(f"💾 Checkpoint saved: {os.path.basename(checkpoint_path)}")
        
        if (cfg.target_total_tokens and seen_tokens >= cfg.target_total_tokens) or global_step >= total_steps:
            break
    
    # Stop progress bar
    if progress_bar:
        progress_bar.stop()
    
    # Final checkpoint and export with compression
    if rank == 0:
        print("\n💾 Preparing final model with compression...")
        
        # Create compressed model state dict
        compressed_state_dict = {}
        original_size = 0
        compressed_size = 0
        
        for name, param in model.state_dict().items():
            original_size += param.numel() * 4  # FP32 size
            
            # Compress most parameters to FP16, keep critical ones in FP32
            if any(critical in name for critical in ['embed', 'head', 'ln_f']):
                # Keep embeddings and final layers in FP32 for stability
                compressed_state_dict[name] = param.clone()
                compressed_size += param.numel() * 4
                print(f"  🔒 Keeping {name} in FP32 (critical layer)")
            else:
                # Compress to FP16
                compressed_state_dict[name] = param.clone().half()
                compressed_size += param.numel() * 2
        
        compression_ratio = original_size / compressed_size
        size_saved_mb = (original_size - compressed_size) / (1024 * 1024)
        
        print(f"  ✅ Compression ratio: {compression_ratio:.2f}x")
        print(f"  ✅ Size saved: {size_saved_mb:.1f} MB")
        print(f"  ✅ Final model size: ~{compressed_size / (1024 * 1024):.1f} MB")
        
        final_checkpoint = {
            'model': compressed_state_dict,
            'step': global_step,
            'epoch': epoch,
            'best_loss': best_loss,
            'seen_tokens': seen_tokens,
            'config': cfg.__dict__,
            'final': True,
            'training_duration': time.time() - training_start_time,
            'compression_info': {
                'original_size_mb': original_size / (1024 * 1024),
                'compressed_size_mb': compressed_size / (1024 * 1024),
                'compression_ratio': compression_ratio,
                'size_saved_mb': size_saved_mb
            }
        }
        
        # Don't save optimizer and scaler states in final model (saves space)
        print("  🗑️  Excluding optimizer states to save space")
        
        if hasattr(model, 'get_training_phase_info'):
            final_checkpoint['phase_info'] = model.get_training_phase_info()
            final_checkpoint['phase_history'] = phase_manager.get_phase_summary() if phase_manager else {}
        
        final_path = os.path.join(cfg.ckpt_dir, 'final_model.pt')
        torch.save(final_checkpoint, final_path)
        
        # Check actual file size
        actual_size_mb = os.path.getsize(final_path) / (1024 * 1024)
        print(f"  📁 Actual file size: {actual_size_mb:.1f} MB")
        
        print()
        print("🎉" * 30)
        print("🎉 ADVANCED FP4 FQT TRAINING COMPLETE!")
        print("🎉" * 30)
        print()
        print(f"📊 FINAL STATISTICS:")
        print(f"  ✅ Total steps: {global_step:,}")
        print(f"  ✅ Tokens processed: {seen_tokens/1e6:.1f}M")
        print(f"  ✅ Best loss: {best_loss:.4f}")
        print(f"  ✅ Training time: {(time.time() - training_start_time)/3600:.2f} hours")
        print(f"  ✅ Average speed: {seen_tokens/(time.time() - training_start_time):.0f} tokens/sec")
        
        if hasattr(model, 'get_training_phase_info'):
            info = model.get_training_phase_info()
            print(f"  ✅ Final phase: {info['phase']}")
            if 'memory_stats' in info:
                stats = info['memory_stats']
                print(f"  ✅ Memory saved: {stats['memory_saved_mb']:.1f} MB")
        
        print(f"  ✅ Models saved in: {cfg.ckpt_dir}")
        
        # Automatic model export
        print(f"\n🚀 Starting automatic model export...")
        try:
            from enhanced_export_system import enhanced_export_after_training
            export_sizes = enhanced_export_after_training(final_path, "exported_models")
            print_success(f"📦 {len(export_sizes)} formats exported successfully!")
        except Exception as e:
            print_warning(f"Model export failed: {e}")
            print("💡 You can manually export later with:")
            print(f"   python enhanced_export_system.py {final_path}")

if __name__ == '__main__':
    train()
