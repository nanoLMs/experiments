#!/usr/bin/env python3
"""
Enhanced FP4 FQT Trainer
========================

Trainer with full FP4 FQT integration including:
- Split rounding strategy
- Automatic QAF phase detection
- Gradient tracking and stagnation detection
- Memory-efficient training pipeline
"""

import os, math, time, torch
import json
import torch.nn as nn

# Fix HuggingFace tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.distributed as dist
from torch.optim import AdamW
from transformers import PreTrainedTokenizerFast
from config import TrainConfig
from model_moe import NanoMoEModel

# FP4 FQT imports
from fp4_model_integration import (
    integrate_pure_4bit_model,
    get_fp4_optimizer_config,
    should_use_amp_with_fp4
)
from fp4_fqt_core import get_fp4_memory_stats, set_model_training_phase

from data_loader import create_dataloader
from anti_hallu import LogitConstraint
from schedule import cosine_with_warmup
from checkpoint import save_checkpoint, load_latest
from rich_output import (
    rich_output, print_header, print_system_info, print_config_summary,
    print_training_step, print_validation_results, print_checkpoint_info,
    print_export_summary, print_error, print_warning, print_success,
    TrainingMetrics, check_rich_installation
)

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, BackwardPrefetch, StateDictType, FullStateDictConfig
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    HAS_FSDP = True
except ImportError:
    HAS_FSDP = False


def setup_seed(seed):
    import random, numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group('nccl')
        return True, dist.get_rank(), dist.get_world_size()
    return False, 0, 1


def get_fp4_optimizer(cfg, model):
    """Get optimizer optimized for FP4 training"""
    params = [p for p in model.parameters() if p.requires_grad]

    # Check if this is an FP4 model
    is_fp4_model = hasattr(model, 'training_phase')

    if is_fp4_model:
        print("🔧 Configuring optimizer for FP4 FQT training")
        optimizer_config = get_fp4_optimizer_config(cfg, model)

        # Use standard AdamW with FP4-optimized settings
        optimizer = AdamW(
            params,
            lr=optimizer_config['lr'],
            betas=optimizer_config['betas'],
            weight_decay=optimizer_config['weight_decay'],
            eps=optimizer_config['eps']
        )

        print(f"  • Learning rate: {optimizer_config['lr']}")
        print(f"  • Weight decay: {optimizer_config['weight_decay']}")
        print(f"  • Epsilon: {optimizer_config['eps']}")

    else:
        # Standard optimizer for non-FP4 models
        print("🔧 Using standard AdamW optimizer")
        optimizer = AdamW(
            params,
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
            eps=1e-8
        )

    return optimizer


def maybe_fsdp_wrap(cfg, model):
    if not (cfg.fsdp and HAS_FSDP and dist.is_initialized()):
        return model
    auto_wrap = size_based_auto_wrap_policy(min_num_params=cfg.fsdp_wrap_layer_size)
    model = FSDP(model, auto_wrap_policy=auto_wrap, backward_prefetch=BackwardPrefetch.BACKWARD_PRE, device_id=torch.cuda.current_device())
    return model


def evaluate(model, tokenizer, cfg, device):
    """Enhanced evaluation with FP4 phase awareness"""
    model.eval()

    # Store current phase if FP4 model
    original_phase = None
    if hasattr(model, 'training_phase'):
        original_phase = model.training_phase
        # Ensure evaluation uses RtN rounding
        set_model_training_phase(model, "normal")

    eval_prompts = [
        "The medical treatment for",
        "In this image, we can see",
        "The key principle of",
        "This multimodal example shows"
    ]

    results = []
    total_eval_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100)

    with torch.no_grad():
        # Generation evaluation
        for prompt in eval_prompts:
            ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
            max_new = min(64, cfg.generation_max_new_tokens)

            for _ in range(max_new):
                logits_main, _, _, _ = model(ids)
                next_logits = logits_main[:, -1]
                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.argmax(probs, dim=-1, keepdim=True)
                ids = torch.cat([ids, next_id], dim=1)
                if ids.size(1) >= cfg.seq_len or next_id.item() == tokenizer.eos_token_id:
                    break

            text = tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
            results.append(f"{prompt} → {text[len(prompt):].strip()[:60]}...")

        # Loss evaluation
        try:
            eval_text = "This is a test sequence for evaluation purposes."
            eval_ids = tokenizer(eval_text, return_tensors='pt', max_length=cfg.seq_len, truncation=True)['input_ids'].to(device)
            if eval_ids.size(1) > 1:
                x = eval_ids[:, :-1]
                y = eval_ids[:, 1:]
                logits_main, _, _, _ = model(x)
                eval_loss = criterion(logits_main.view(-1, logits_main.size(-1)), y.view(-1))
                total_eval_loss = eval_loss.item()
        except:
            total_eval_loss = 0.0

    # Restore original phase if FP4 model
    if original_phase is not None:
        set_model_training_phase(model, original_phase)

    model.train()
    return results, total_eval_loss


def train():
    # Set memory management environment variable for fragmentation avoidance
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Check Rich installation
    check_rich_installation()

    cfg = TrainConfig()
    distributed, rank, world = init_distributed()
    setup_seed(cfg.seed + rank)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Beautiful header
    if rank == 0:
        print_header(
            "🚀 NanoLM FP4 FQT Training System",
            "Pure 4-bit Fully Quantized Training • NVFP4 Format • Split Rounding"
        )

    # Load tokenizer
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer_dir)
        if rank == 0:
            print_success(f"Tokenizer loaded: {len(tokenizer):,} tokens")
    except Exception as e:
        if rank == 0:
            print_error(f"Failed to load tokenizer from {cfg.tokenizer_dir}", e)
        return

    if cfg.forbidden_tokens_file and os.path.exists(cfg.forbidden_tokens_file):
        with open(cfg.forbidden_tokens_file) as f:
            cfg.forbidden_tokens = json.load(f)

    # Initialize model with FP4 FQT
    if rank == 0:
        print_success("🔥 Creating FP4 FQT Model")
        print("📊 FP4 Fully Quantized Training (FQT) Features:")
        print("  • NVFP4 format: E2M1 data, E4M3 scale")
        print("  • Block size: 16 (optimal from research)")
        print("  • Split rounding: RtN forward, SR backward")
        print("  • Automatic QAF phase for convergence")
        print("  • Expected: ~75% memory reduction")

    model = integrate_pure_4bit_model(cfg)

    # Check parameter count
    total_params = model.num_parameters()
    trainable_params = model.num_trainable_parameters()

    if rank == 0:
        # Display system information
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

        print_system_info(gpu_name, gpu_memory, total_params)

        # FP4 memory statistics
        if hasattr(model, 'memory_stats'):
            stats = model.memory_stats
            print_success(f"FP4 Memory Efficiency:")
            print(f"  • Memory saved: {stats['memory_saved_mb']:.1f} MB")
            print(f"  • Compression ratio: {stats['compression_ratio']:.2f}x")

        # Configuration summary
        config_dict = {
            'n_layers': cfg.n_layers,
            'd_model': cfg.d_model,
            'n_heads': cfg.n_heads,
            'n_experts': cfg.n_experts,
            'moe_every': cfg.moe_every,
            'learning_rate': cfg.lr,
            'micro_batch_size': cfg.micro_batch_size,
            'seq_len': cfg.seq_len,
            'grad_accum_steps': cfg.grad_accum_steps,
            'use_pure_4bit': cfg.use_pure_4bit,
            'fp4_format': 'NVFP4 (E2M1+E4M3)',
            'split_rounding': True
        }
        print_config_summary(config_dict)

    # Enable gradient checkpointing
    if cfg.gradient_checkpointing:
        model.cfg.gradient_checkpointing = True
        if rank == 0:
            print("✅ Gradient checkpointing enabled")

    model = maybe_fsdp_wrap(cfg, model)
    model.to(device)

    # Model compilation (careful with FP4)
    if cfg.compile_model and hasattr(torch, 'compile'):
        try:
            if rank == 0:
                print(f"⚡ Compiling FP4 model with mode: {cfg.compile_mode}")
                print("   This may take longer for FP4 models...")

            model = torch.compile(
                model,
                mode=cfg.compile_mode,
                fullgraph=cfg.compile_fullgraph,
                dynamic=cfg.compile_dynamic
            )

            if rank == 0:
                print("✅ FP4 model compiled successfully!")

        except Exception as e:
            if rank == 0:
                print(f"⚠️ FP4 model compilation failed: {e}")
                print("🔄 Continuing with eager mode")

    # Setup loss and constraints
    pad_id = tokenizer.pad_token_id
    ignore_index = pad_id if pad_id is not None else -100
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    logit_constraint = LogitConstraint(cfg.forbidden_tokens, cfg.factual_penalty_weight).to(device)

    # FP4-optimized optimizer
    optimizer = get_fp4_optimizer(cfg, model)

    # AMP settings for FP4
    use_amp = should_use_amp_with_fp4(cfg) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    if rank == 0:
        amp_status = "enabled (FP4 compatible)" if use_amp else "disabled (FP4 sufficient)"
        print(f"🔧 Mixed precision: {amp_status}")

    # Data loading
    try:
        dl = create_dataloader(cfg, tokenizer, world)
        steps_per_epoch = len(dl)
    except Exception as e:
        if rank == 0:
            print_error(f"Failed to create dataloader: {e}")
        return

    total_steps = cfg.num_epochs * steps_per_epoch
    effective_batch_size = cfg.effective_batch_size(world)

    if rank == 0:
        print(f"📊 FP4 Training Setup:")
        print(f"  Dataset steps per epoch: {steps_per_epoch:,}")
        print(f"  Total training steps: {total_steps:,}")
        print(f"  Effective batch size: {effective_batch_size:,}")
        print(f"  Expected FP4 speedup: 2-4x")
        estimated_hours = total_steps * 0.15 / 3600  # FP4 should be faster
        print(f"  Estimated training time: {estimated_hours:.1f} hours")

    # Checkpoint loading
    ckpt_state, start_step = load_latest(cfg.ckpt_dir)
    start_epoch = 0
    best_eval_loss = float('inf')
    patience_counter = 0

    if ckpt_state:
        try:
            model.load_state_dict(ckpt_state['model'])
            optimizer.load_state_dict(ckpt_state['optim'])
            if 'scaler' in ckpt_state and scaler:
                scaler.load_state_dict(ckpt_state['scaler'])
            start_step = ckpt_state['step']
            start_epoch = ckpt_state.get('epoch', 0)
            best_eval_loss = ckpt_state.get('best_eval_loss', float('inf'))
            patience_counter = ckpt_state.get('patience_counter', 0)

            # Restore FP4 training phase if available
            if hasattr(model, 'training_phase') and 'fp4_phase' in ckpt_state:
                model.training_phase = ckpt_state['fp4_phase']
                if rank == 0:
                    print(f"✅ Restored FP4 phase: {model.training_phase}")

            if rank == 0:
                print(f"✅ Resumed from step {start_step}, epoch {start_epoch}")
        except Exception as e:
            if rank == 0:
                print(f"⚠️ Checkpoint loading failed: {e}")
            start_step = 0

    # Save config
    if rank == 0 and cfg.save_config_once:
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        config_data = {k: v for k, v in cfg.__dict__.items() if not k.startswith('_')}
        config_data['fp4_enabled'] = True
        config_data['fp4_format'] = 'NVFP4'

        with open(os.path.join(cfg.ckpt_dir, 'train_config.json'), 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"💾 FP4 config saved to {cfg.ckpt_dir}/train_config.json")

    # Memory monitoring helper
    def get_gpu_memory_info():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return allocated, reserved, total
        return 0, 0, 0

    def check_memory_warning():
        allocated, reserved, total = get_gpu_memory_info()
        if allocated / total > 0.85:  # Warn if using >85% memory
            print(f"⚠️ HIGH MEMORY USAGE: {allocated:.1f}/{total:.1f} GB ({allocated/total*100:.1f}%)")
            print(f"   Reserved: {reserved:.1f} GB")
            return True
        return False

    # Training loop
    accum = 0
    running_loss = 0.0
    running_mtp_loss = 0.0
    running_reason_loss = 0.0
    running_aux_loss = 0.0
    global_step = start_step
    seen_tokens = start_step * cfg.seq_len * effective_batch_size

    # Initialize single progress display (avoids Rich multiple live display conflicts)
    progress = None
    task_id = None

    if rank == 0:
        # Single progress display for steps
        progress = rich_output.create_training_progress(total_steps)
        if progress:
            task_id = progress.add_task(
                "🚀 FP4 Training Progress",
                total=total_steps,
                completed=global_step,
                tokens_per_sec=0
            )
            progress.start()

    training_start_time = time.time()

    for epoch in range(start_epoch, cfg.num_epochs):
        epoch_start_time = time.time()
        epoch_steps = 0
        epoch_loss = 0.0

        if rank == 0:
            phase_info = ""
            if hasattr(model, 'get_training_phase_info'):
                info = model.get_training_phase_info()
                phase_info = f" (FP4 phase: {info['phase']})"

            # Enhanced epoch start message
            print()
            print("=" * 80)
            print(f"🚀 EPOCH {epoch + 1}/{cfg.num_epochs}{phase_info}")
            print(f"📊 Progress: {((epoch) / cfg.num_epochs * 100):.1f}% complete")
            print(f"⏱️  Steps this epoch: 0/{steps_per_epoch:,}")
            print("=" * 80)

        for it, (x, y) in enumerate(dl):
            if global_step < start_step:
                global_step += 1
                continue

            # First step debugging
            if global_step == start_step and rank == 0:
                print(f"🔄 Processing first step...")
                print(f"   Batch shape: {x.shape}")
                print(f"   Sequence length: {x.size(1)}")
                print(f"   Batch size: {x.size(0)}")
                print(f"   Expected time: 30-60 seconds for first step")
                first_step_start = time.time()

            # Pre-batch memory check
            if check_memory_warning():
                torch.cuda.empty_cache()
                import gc
                gc.collect()

            # Ensure correct sequence length
            if x.size(1) != cfg.seq_len:
                if x.size(1) > cfg.seq_len:
                    x = x[:, :cfg.seq_len]
                    y = y[:, :cfg.seq_len]
                else:
                    # Skip this batch if too short
                    continue

            x = x.to(device, non_blocking=cfg.pin_memory)
            y = y.to(device, non_blocking=cfg.pin_memory)

            # Aggressive memory cleanup before forward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Forward pass with FP4
            with torch.amp.autocast('cuda', enabled=use_amp):
                try:
                    logits_main, logits_mtp, aux_loss, reason_logits = model(x)
                except torch.cuda.OutOfMemoryError as e:
                    # Emergency memory cleanup and retry with gradient checkpointing
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()

                    # Try with reduced batch size if possible
                    if x.size(0) > 1:
                        print(f"⚠️ OOM detected, trying with smaller batch size")
                        half_batch = x.size(0) // 2
                        x1, y1 = x[:half_batch], y[:half_batch]
                        x2, y2 = x[half_batch:], y[half_batch:]

                        # Process first half
                        logits_main1, logits_mtp1, aux_loss1, reason_logits1 = model(x1)
                        torch.cuda.empty_cache()

                        # Process second half
                        logits_main2, logits_mtp2, aux_loss2, reason_logits2 = model(x2)
                        torch.cuda.empty_cache()

                        # Combine results
                        logits_main = torch.cat([logits_main1, logits_main2], dim=0)
                        logits_mtp = [torch.cat([lm1, lm2], dim=0) for lm1, lm2 in zip(logits_mtp1, logits_mtp2)]
                        aux_loss = aux_loss1 + aux_loss2
                        reason_logits = torch.cat([reason_logits1, reason_logits2], dim=0) if reason_logits1 is not None else None

                        # Update y for loss computation
                        y = torch.cat([y1, y2], dim=0)

                        # Cleanup
                        del logits_main1, logits_main2, logits_mtp1, logits_mtp2
                        del aux_loss1, aux_loss2, reason_logits1, reason_logits2
                        del x1, x2, y1, y2
                        torch.cuda.empty_cache()
                    else:
                        raise e  # Re-raise if batch size is already 1

                # Apply anti-hallucination
                logits_main, penalty = logit_constraint(logits_main)
                loss_main = criterion(logits_main.view(-1, logits_main.size(-1)), y.view(-1))

                # MTP loss computation
                if global_step == start_step and len(cfg.mtp_loss_weights) != len(logits_mtp):
                    if rank == 0:
                        print_warning(f"Adjusting MTP weights: {len(logits_mtp)} heads")
                    cfg.mtp_loss_weights = cfg.mtp_loss_weights[:len(logits_mtp)] + [0.1] * (len(logits_mtp) - len(cfg.mtp_loss_weights))

                mtp_loss = 0.0
                for i, lm in enumerate(logits_mtp):
                    shift_y = torch.cat([
                        y[:, i+1:],
                        torch.full((y.size(0), i+1), ignore_index, device=device)
                    ], dim=1)
                    mtp_loss += cfg.mtp_loss_weights[i] * criterion(lm.view(-1, lm.size(-1)), shift_y.view(-1))

                reason_loss = 0.0
                if reason_logits is not None and cfg.reasoning_loss_weight > 0:
                    target_last = y[:, -1]
                    reason_loss = cfg.reasoning_loss_weight * criterion(reason_logits, target_last)

                total_loss = loss_main + mtp_loss + aux_loss + penalty + reason_loss

            # Store loss values before cleanup
            loss_main_val = loss_main.item()
            mtp_loss_val = mtp_loss.item() if isinstance(mtp_loss, torch.Tensor) else mtp_loss
            reason_loss_val = reason_loss.item() if isinstance(reason_loss, torch.Tensor) else reason_loss
            aux_loss_val = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss

            # --- Progress bar update every batch (real-time) ---
            if rank == 0 and progress and task_id is not None:
                elapsed = time.time() - training_start_time
                tokens_per_sec = seen_tokens / elapsed if elapsed > 0 else 0
                percent = 100.0 * global_step / total_steps
                # Update progress bar description and fields
                progress.update(
                    task_id,
                    completed=global_step,
                    tokens_per_sec=tokens_per_sec,
                    description=f"Step {global_step:,}/{total_steps:,} | Epoch {epoch+1}/{cfg.num_epochs} | {percent:.2f}% | {tokens_per_sec:.0f} tok/s | EpochStep {it+1}/{steps_per_epoch}"
                )

            # Backward pass (uses stochastic rounding in FP4)
            try:
                scaler.scale(total_loss / cfg.grad_accum_steps).backward()
                accum += 1

                # Update running losses
                running_loss += loss_main_val
                running_mtp_loss += mtp_loss_val
                running_reason_loss += reason_loss_val
                running_aux_loss += aux_loss_val

                # Immediate cleanup of loss tensors
                del total_loss, loss_main, penalty
                if isinstance(mtp_loss, torch.Tensor):
                    del mtp_loss
                if isinstance(reason_loss, torch.Tensor):
                    del reason_loss
                if isinstance(aux_loss, torch.Tensor):
                    del aux_loss

                # Clear intermediate activations
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if accum % cfg.grad_accum_steps == 0:
                    scaler.unscale_(optimizer)

                    # Calculate gradient norm for FP4 phase detection
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

                    # Update FP4 gradient tracking
                    if hasattr(model, 'update_gradient_stats'):
                        model.update_gradient_stats(grad_norm.item())
                        model.step_qaf_phase()

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    # Aggressive memory cleanup after optimizer step
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        # Force garbage collection every few steps
                        if global_step % 10 == 0:
                            import gc
                            gc.collect()
                            torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError as oom_error:
                print(f"⚠️ OOM during backward pass at step {global_step}")
                print("💡 Clearing all caches and retrying...")

                # Emergency cleanup
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()

                # Skip this batch and continue
                print("⚠️ Skipping batch due to memory constraints")
                continue

                # Learning rate schedule
                lr = cosine_with_warmup(global_step, cfg.warmup_steps, total_steps, cfg.lr, cfg.min_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                # Logging with FP4 phase info
                if global_step % cfg.log_interval == 0 and (rank == 0):
                    avg_loss = running_loss / cfg.log_interval
                    mem = torch.cuda.memory_allocated()/1e6 if torch.cuda.is_available() else 0
                    tokens_seen = seen_tokens / 1e6

                    elapsed = time.time() - training_start_time
                    tokens_per_sec = seen_tokens / elapsed if elapsed > 0 else 0
                    eta = (total_steps - global_step) * (elapsed / global_step) if global_step > 0 else None

                    if avg_loss < best_eval_loss:
                        best_eval_loss = avg_loss

                    # Create metrics with FP4 info
                    metrics = TrainingMetrics(
                        step=global_step,
                        epoch=epoch,
                        loss=avg_loss,
                        lr=lr,
                        tokens_per_sec=tokens_per_sec,
                        gpu_memory=mem/1000,
                        cpu_usage=0,
                        elapsed_time=elapsed,
                        eta=eta,
                        best_loss=best_eval_loss
                    )

                    # Add FP4 phase info to display
                    if hasattr(model, 'get_training_phase_info'):
                        phase_info = model.get_training_phase_info()
                        if rank == 0:
                            phase_str = f"FP4 Phase: {phase_info['phase']}"
                            if phase_info['phase'] == 'qaf':
                                phase_str += f" ({phase_info['qaf_steps']}/{phase_info['max_qaf_steps']})"
                            print(f"  {phase_str}")

                    print_training_step(metrics, progress, task_id)

                    running_loss = 0.0

                    # Update epoch tracking
                    epoch_loss += avg_loss
                    epoch_steps += 1
                    running_mtp_loss = 0.0
                    running_reason_loss = 0.0
                    running_aux_loss = 0.0

                # Memory reporting
                if cfg.report_memory_interval and global_step % cfg.report_memory_interval == 0 and rank == 0 and torch.cuda.is_available():
                    peak = torch.cuda.max_memory_allocated()/1e6
                    reserved = torch.cuda.memory_reserved()/1e6
                    print_success(f"FP4 Memory: Peak {peak:.0f}MB, Reserved {reserved:.0f}MB")
                    torch.cuda.reset_peak_memory_stats()

                # Evaluation
                if global_step % cfg.eval_interval == 0 and (rank == 0):
                    eval_results, eval_loss = evaluate(model, tokenizer, cfg, device)

                    val_metrics = {
                        'perplexity': math.exp(min(eval_loss, 10)),
                        'quality_score': max(0, 1 - eval_loss/5)
                    }

                    print_validation_results(eval_loss, val_metrics)

                    if eval_results:
                        print_success("FP4 Sample generations:")
                        for i, result in enumerate(eval_results[:2]):
                            rich_output.console.print(f"  [dim]Sample {i+1}:[/dim] {result}")

                    # Best model saving
                    if cfg.save_best and eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        patience_counter = 0

                        best_state = {
                            'model': model.state_dict(),
                            'optim': optimizer.state_dict(),
                            'scaler': scaler.state_dict() if scaler else None,
                            'step': global_step,
                            'epoch': epoch,
                            'eval_loss': eval_loss,
                            'config': cfg.__dict__
                        }

                        # Save FP4 phase info
                        if hasattr(model, 'training_phase'):
                            best_state['fp4_phase'] = model.training_phase

                        best_path = os.path.join(cfg.ckpt_dir, 'best_model.pt')
                        torch.save(best_state, best_path)
                        print_checkpoint_info(best_path, global_step, eval_loss)
                        print_success(f"New best FP4 model! Loss: {eval_loss:.4f}")
                    else:
                        patience_counter += cfg.eval_interval

                # Regular checkpointing
                if global_step % cfg.save_interval == 0 and (rank == 0):
                    state = {
                        'model': model.state_dict(),
                        'optim': optimizer.state_dict(),
                        'scaler': scaler.state_dict() if scaler else None,
                        'step': global_step,
                        'epoch': epoch,
                        'best_eval_loss': best_eval_loss,
                        'patience_counter': patience_counter,
                        'seen_tokens': seen_tokens,
                        'config': cfg.__dict__
                    }

                    # Save FP4 training state
                    if hasattr(model, 'training_phase'):
                        state['fp4_phase'] = model.training_phase

                    checkpoint_path = save_checkpoint(state, cfg.ckpt_dir, global_step, cfg.keep_last_k)
                    current_loss = running_loss / max(1, cfg.log_interval)
                    print_checkpoint_info(checkpoint_path, global_step, current_loss)

            global_step += 1
            seen_tokens += cfg.seq_len * cfg.micro_batch_size * cfg.grad_accum_steps * world

            # First step completion message
            if global_step == start_step + 1 and rank == 0 and 'first_step_start' in locals():
                first_step_time = time.time() - first_step_start
                print(f"✅ First step completed in {first_step_time:.1f} seconds!")
                print(f"🚀 Training is now running normally...")
                print(f"⚡ Subsequent steps should be much faster (~0.2-0.5s each)")

            # Check stopping conditions
            if (cfg.target_total_tokens and seen_tokens >= cfg.target_total_tokens) or global_step >= total_steps:
                if rank == 0:
                    reason = "token budget reached" if seen_tokens >= cfg.target_total_tokens else "step limit reached"
                    print(f"🏁 FP4 training complete: {reason}")
                break

        # Epoch completion summary
        if rank == 0:
            epoch_duration = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / max(1, epoch_steps)

            # Epoch progress is shown in the detailed messages above

            print()
            print("─" * 80)
            print(f"✅ EPOCH {epoch + 1} COMPLETE")
            print(f"📊 Average Loss: {avg_epoch_loss:.4f}")
            print(f"⏱️  Duration: {epoch_duration/60:.1f} minutes")
            print(f"🚀 Steps: {epoch_steps:,}")
            print(f"📈 Progress: {((epoch + 1) / cfg.num_epochs * 100):.1f}% of training complete")

            if epoch + 1 < cfg.num_epochs:
                remaining_epochs = cfg.num_epochs - (epoch + 1)
                estimated_remaining = (epoch_duration * remaining_epochs) / 3600
                print(f"⏳ Estimated remaining: {estimated_remaining:.1f} hours")

            print("─" * 80)
            print()

        if (cfg.target_total_tokens and seen_tokens >= cfg.target_total_tokens) or global_step >= total_steps:
            break

    # Final checkpoint with FP4 state
    if rank == 0:
        final_state = {
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler else None,
            'step': global_step,
            'epoch': epoch,
            'best_eval_loss': best_eval_loss,
            'seen_tokens': seen_tokens,
            'config': cfg.__dict__,
            'final': True
        }

        if hasattr(model, 'training_phase'):
            final_state['fp4_phase'] = model.training_phase

        final_path = os.path.join(cfg.ckpt_dir, 'final_model.pt')
        torch.save(final_state, final_path)

        # Stop progress display
        if progress:
            progress.stop()

        print()
        print("🎉" * 40)
        print("🎉 FP4 FQT TRAINING COMPLETE! 🎉")
        print("🎉" * 40)
        print()
        print(f"📊 FINAL STATISTICS:")
        print(f"  ✅ Total steps: {global_step:,}")
        print(f"  ✅ Tokens processed: {seen_tokens/1e6:.1f}M")
        print(f"  ✅ Best eval loss: {best_eval_loss:.4f}")
        print(f"  ✅ Training time: {(time.time() - training_start_time)/3600:.2f} hours")
        print(f"  ✅ Average speed: {seen_tokens/(time.time() - training_start_time):.0f} tokens/sec")
        print(f"  ✅ Models saved in: {cfg.ckpt_dir}")

        if hasattr(model, 'memory_stats'):
            stats = model.memory_stats
            print(f"  ✅ Memory saved: {stats['memory_saved_mb']:.1f} MB")
            print(f"  ✅ Compression ratio: {stats['compression_ratio']:.2f}x")

        # 🚀 AUTOMATIC MODEL EXPORT FOR ALL DEVICES
        print(f"\n🚀 Starting automatic model export for all devices...")
        print(f"📱 Creating models for: Android, iOS, Windows, Linux, Web, Cloud")

        try:
            from enhanced_export_system import enhanced_export_after_training

            # Use the best model if available, otherwise use final model
            best_path = os.path.join(cfg.ckpt_dir, 'best_model.pt')
            export_checkpoint = best_path if os.path.exists(best_path) else final_path

            print(f"📦 Exporting from: {os.path.basename(export_checkpoint)}")
            export_sizes = enhanced_export_after_training(export_checkpoint, "exported_models")

            print(f"\n✅ Model export completed!")
            print(f"📁 Exported {len(export_sizes)} formats to: exported_models/")

            # Show device-specific recommendations
            print(f"\n📱 Device-Specific Models Created:")
            device_models = {
                "📱 Android Apps": "quantized_mobile.ptl",
                "🍎 iOS Apps": "coreml_ios.mlmodel",
                "🖥️ Windows/Linux": "pytorch_fp16.pt",
                "☁️ Cloud APIs": "huggingface/",
                "🌐 Web Browser": "onnx_optimized.onnx",
                "🔧 Edge Devices": "pytorch_4bit.pt"
            }

            for device, model_file in device_models.items():
                if any(model_file.split('.')[0] in fmt for fmt in export_sizes.keys()):
                    size_mb = next((size for fmt, size in export_sizes.items()
                                  if model_file.split('.')[0] in fmt), 0) / (1024*1024)
                    print(f"  {device}: {model_file} ({size_mb:.1f} MB)")

            print(f"\n📋 Quick Start:")
            print(f"  • Android: Copy quantized_mobile.ptl to your app")
            print(f"  • iOS: Copy model.mlmodel to Xcode project")
            print(f"  • Server: Use pytorch_fp16.pt for production")
            print(f"  • Web: Deploy onnx_optimized.onnx with ONNX.js")
            print(f"  • Edge: Use pytorch_4bit.pt for Raspberry Pi")

        except Exception as e:
            print(f"⚠️ Model export failed: {e}")
            print(f"💡 You can manually export later with:")
            print(f"   python export_models.py {final_path}")
            print(f"   This will create models for all devices")


if __name__ == '__main__':
    train()
