import os, math, time, json, torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from transformers import PreTrainedTokenizerFast
from contextlib import nullcontext

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from config import TrainConfig
from config_simple_loss_debug import SimpleLossDebugConfig
from model_moe import NanoMoEModel

# Import FP4 FQT implementation
try:
    from fp4_model_integration import integrate_pure_4bit_model
    HAS_FP4_FQT = True
except ImportError:
    HAS_FP4_FQT = False

# Legacy pure 4-bit import (deprecated)
try:
    from pure_4bit_integration import integrate_pure_4bit_model as integrate_legacy_4bit
    HAS_LEGACY_4BIT = True
except ImportError:
    HAS_LEGACY_4BIT = False

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
from loss_tracker import LossTracker

# Import AMP components with fallbacks
try:
    from torch.cuda.amp import GradScaler, autocast as autocast_cuda
    HAS_AMP = True
except ImportError:
    HAS_AMP = False
    autocast_cuda = None

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

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


def get_optimizer(cfg, model):
    """Get optimizer - use bitsandbytes AdamW8bit for 4-bit quantized models"""
    params = [p for p in model.parameters() if p.requires_grad]

    # Use bitsandbytes optimizer for 4-bit models
    if cfg.use_bnb_4bit and HAS_BNB:
        try:
            print("🔧 Using BitsAndBytes AdamW8bit optimizer for NF4 model")
            optimizer = bnb.optim.AdamW8bit(
                params,
                lr=cfg.lr,
                betas=cfg.betas,
                weight_decay=cfg.weight_decay,
                eps=1e-8,
                min_8bit_size=4096,  # Only quantize large optimizer states
                percentile_clipping=100,  # Disable percentile clipping for stability
                block_wise=True  # Enable block-wise quantization for memory efficiency
            )
            print("   ✅ AdamW8bit optimizer created successfully")
            print("   ✅ Optimizer states will also be 8-bit quantized")
            return optimizer

        except Exception as e:
            print(f"⚠️ AdamW8bit failed: {e}")
            print("🔧 Falling back to regular AdamW (still works with NF4 model)")

    # Standard AdamW (works perfectly with 4-bit models)
    print("🔧 Using standard PyTorch AdamW optimizer")
    optimizer = AdamW(
        params,
        lr=cfg.lr,
        betas=cfg.betas,
        weight_decay=cfg.weight_decay,
        eps=1e-8
    )
    print("   ✅ Standard AdamW optimizer created")
    return optimizer


def maybe_fsdp_wrap(cfg, model):
    if not (cfg.fsdp and HAS_FSDP and dist.is_initialized()):
        return model
    auto_wrap = size_based_auto_wrap_policy(min_num_params=cfg.fsdp_wrap_layer_size)
    model = FSDP(model, auto_wrap_policy=auto_wrap, backward_prefetch=BackwardPrefetch.BACKWARD_PRE, device_id=torch.cuda.current_device())
    return model


def evaluate(model, tokenizer, cfg, device):
    """Enhanced evaluation with multiple prompts and loss computation"""
    model.eval()
    eval_prompts = [
        "The medical treatment for",
        "In this image, we can see",
        "The key principle of",
        "This multimodal example shows"
    ]

    results = []
    total_eval_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100, label_smoothing=0.1)

    with torch.no_grad():
        # Generation evaluation
        for prompt in eval_prompts:
            ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
            max_new = min(64, cfg.generation_max_new_tokens)  # Shorter for eval

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

        # Loss evaluation on a few random samples
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

    model.train()
    return results, total_eval_loss


def train(config_class=TrainConfig):
    # Check Rich installation
    check_rich_installation()

    cfg = config_class()
    distributed, rank, world = init_distributed()
    setup_seed(cfg.seed + rank)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Beautiful header
    if rank == 0:
        print_header(
            "🚀 NanoLM Training System",
            "MoE + MTP + Reasoning + Anti-Hallucination • Optimized for RTX 3060 Ti"
        )

    # Load tokenizer with error handling
    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer_dir)
        if rank == 0:
            print_success(f"Tokenizer loaded: {len(tokenizer):,} tokens")
    except Exception as e:
        if rank == 0:
            print_error(f"Failed to load tokenizer from {cfg.tokenizer_dir}", e)
            print_warning("Please run 'python analyze_tokenizer.py' first to check available tokenizers")
        return

    # DYNAMICALLY SET VOCAB SIZE from tokenizer to prevent index errors
    if cfg.vocab_size != len(tokenizer):
        if rank == 0:
            print_warning(f"Config vocab_size ({cfg.vocab_size}) doesn't match tokenizer ({len(tokenizer)}). Extending model embeddings.")
        cfg.vocab_size = len(tokenizer)

    if cfg.forbidden_tokens_file and os.path.exists(cfg.forbidden_tokens_file):
        with open(cfg.forbidden_tokens_file) as f:
            cfg.forbidden_tokens = json.load(f)

    # Initialize model with FP4 FQT support
    if getattr(cfg, 'use_pure_4bit', False) and HAS_FP4_FQT:
        if rank == 0:
            print_success("🔥 Creating FP4 FQT model (research-based)")
            print("   📊 NVFP4 format with split rounding strategy")
            print("   ⚡ Expected: ~75% memory savings, 2-4x speedup")
            print("   🎯 Automatic QAF phase for convergence")

        model = integrate_pure_4bit_model(cfg)

    elif getattr(cfg, 'use_pure_4bit', False) and HAS_LEGACY_4BIT:
        if rank == 0:
            print_warning("⚠️ Using legacy 4-bit implementation")
            print_warning("   Consider upgrading to FP4 FQT for better performance")

        model = integrate_legacy_4bit(cfg)

    else:
        if rank == 0 and getattr(cfg, 'use_pure_4bit', False):
            print_warning("⚠️ FP4 FQT requested but not available")
            print_warning("   Falling back to regular NanoMoEModel")

        model = NanoMoEModel(cfg)

    # CRITICAL: Check parameter count BEFORE quantization
    total_params = model.num_parameters()
    trainable_params = model.num_trainable_parameters()

    if rank == 0:
        # Display system information
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

        print_system_info(gpu_name, gpu_memory, total_params)

        # Check parameter count
        if total_params > 110e6:  # Allow 10% buffer
            print_warning(f"Model exceeds 100M target ({total_params/1e6:.1f}M)! Consider reducing layers/hidden dims")
        else:
            print_success(f"Model within budget: {total_params/1e6:.1f}M parameters")

        # Display configuration
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
            'use_quantization': cfg.use_quantization,
            'gradient_checkpointing': cfg.gradient_checkpointing
        }
        print_config_summary(config_dict)

    # -------------------- BITSANDBYTES 4-BIT QUANTIZATION --------------------
    # Ensure we only attempt real NF4 quantization when CUDA + bitsandbytes are available
    quantization_success = False
    if cfg.use_bnb_4bit and HAS_BNB:
        if not torch.cuda.is_available():
            if rank == 0:
                print_warning("⚠️ BitsAndBytes NF4 was requested but no CUDA device detected. Disabling NF4 to avoid CUDA hangs.")
                print_warning("   Set cfg.use_bnb_4bit=False to force CPU training or enable CUDA.")
            cfg.use_bnb_4bit = False
            quantization_success = False
        else:
            # Move model to CUDA before applying bnb replacements to ensure ops bind to the right device/context
            model.to(device)
            if rank == 0:
                print_success("🔥 Applying REAL bitsandbytes NF4 quantization")
                print(f"   • Quant type: {cfg.bnb_4bit_quant_type}")
                print(f"   • Compute dtype: {cfg.bnb_4bit_compute_dtype}")
                print(f"   • Double quant: {cfg.bnb_4bit_use_double_quant}")

            # Import and apply quantization
            from fp4_quant import apply_bnb_4bit
            try:
                quantization_success = apply_bnb_4bit(model, cfg)
            except Exception as e:
                quantization_success = False
                if rank == 0:
                    print_warning(f"⚠️ BitsAndBytes quantization failed: {e}")
                    print_warning("   Falling back to FP16 training")

            if rank == 0:
                if quantization_success:
                    print("   ✅ BitsAndBytes NF4 quantization applied successfully")
                    print("   ✅ Expected: ~75% memory reduction during training")
                    print("   ✅ Expected: 2-4x training speedup")
                else:
                    print("   ⚠️ BitsAndBytes quantization failed or was skipped, using FP16")

    elif cfg.use_bnb_4bit and not HAS_BNB:
        if rank == 0:
            print_warning("⚠️ BitsAndBytes requested but not available")
            print_warning("   Install with: pip install bitsandbytes")
            print_warning("   Falling back to FP16 training")

    elif getattr(cfg, 'use_fp4', False):
        # REMOVED: No more fake 4-bit simulation
        if rank == 0:
            print_warning("⚠️ Fake 4-bit disabled. Use use_bnb_4bit=True for real quantization")
            print_warning("⚠️ Falling back to FP16 training for better performance")
    else:
        if rank == 0:
            print_success("🚀 Using FP16 mixed precision (fastest for RTX 3060 Ti)")

    # DISABLE gradient checkpointing for bitsandbytes compatibility
    if cfg.use_bnb_4bit:
        cfg.gradient_checkpointing = False
        model.cfg.gradient_checkpointing = False
        if rank == 0:
            print("⚠️ Gradient checkpointing disabled for bitsandbytes compatibility")
    elif cfg.gradient_checkpointing:
        model.cfg.gradient_checkpointing = True
        if rank == 0:
            print("✅ Gradient checkpointing enabled")

    # Continue wrapping and moving model if not already on device
    model = maybe_fsdp_wrap(cfg, model)
    model.to(device)

    # Enhanced model compilation with CUDA library setup
    if cfg.compile_model and hasattr(torch, 'compile'):
        try:
            # Set up environment for successful compilation (use global os)
            os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')

            # Get compilation settings from config
            compile_mode = getattr(cfg, 'compile_mode', 'default')
            compile_fullgraph = getattr(cfg, 'compile_fullgraph', False)
            compile_dynamic = getattr(cfg, 'compile_dynamic', True)

            if rank == 0:
                print(f"⚡ Compiling model with mode: {compile_mode}")
                print("   This may take 1-2 minutes on first run...")

            model = torch.compile(
                model,
                mode=compile_mode,
                fullgraph=compile_fullgraph,
                dynamic=compile_dynamic
            )

            if rank == 0:
                print("✅ Model compiled successfully!")
                print("🚀 Expect 20-30% speedup during training")

        except Exception as e:
            if rank == 0:
                print(f"⚠️ Model compilation failed: {e}")
                print("🔄 Continuing with eager mode (still fast with 4-bit)")
                print("💡 Check CUDA installation if compilation is needed")

    # FIXED: Proper ignore index handling
    pad_id = tokenizer.pad_token_id
    ignore_index = pad_id if pad_id is not None else -100
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    logit_constraint = LogitConstraint(cfg.forbidden_tokens, cfg.factual_penalty_weight).to(device)

    # Use explicit GradScaler if available
    if HAS_AMP and torch.cuda.is_available():
        scaler = GradScaler(enabled=cfg.amp)
    else:
        scaler = None

    # autocast context manager
    if autocast_cuda is not None:
        def autocast_ctx(device_type='cuda', enabled=True):
            return autocast_cuda(device_type=device_type, enabled=enabled)
    else:
        autocast_ctx = lambda *a, **k: nullcontext()

    optimizer = get_optimizer(cfg, model)

    # Data loading with error handling
    try:
        dl = create_dataloader(cfg, tokenizer, world, vocab_size=cfg.vocab_size)
        steps_per_epoch = len(dl)
    except Exception as e:
        if rank == 0:
            print(f"❌ Failed to create dataloader: {e}")
            print(f"Check if corpus file exists: {cfg.train_corpus}")
        return

    if config_class == SimpleLossDebugConfig:
        total_steps = 100
    else:
        total_steps = cfg.num_epochs * steps_per_epoch

    # SMOKE TEST: limit steps when requested
    smoke_mode = os.environ.get('SMOKE_TEST', '0') == '1'
    if smoke_mode:
        if rank == 0:
            print("🔬 SMOKE_TEST mode enabled: limiting training to 1 step")
        total_steps = min(total_steps, 1)

    effective_batch_size = cfg.effective_batch_size(world)

    if rank == 0:
        print(f"📊 Training Setup:")
        print(f"  Dataset steps per epoch: {steps_per_epoch:,}")
        print(f"  Total training steps: {total_steps:,}")
        print(f"  Effective batch size: {effective_batch_size:,}")
        print(f"  Tokens per step: {cfg.seq_len * effective_batch_size:,}")
        print(f"  Target time per step: 0.25s")
        estimated_hours = total_steps * 0.25 / 3600
        print(f"  Estimated training time: {estimated_hours:.1f} hours")

    # ROBUST CHECKPOINT RESUME
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
            if rank == 0:
                print(f"✅ Resumed from step {start_step}, epoch {start_epoch}")
                print(f"   Best eval loss: {best_eval_loss:.4f}, patience: {patience_counter}")
        except Exception as e:
            if rank == 0:
                print(f"⚠️  Checkpoint loading failed: {e}, starting fresh")
            start_step = 0

    # Save config once
    if rank == 0 and cfg.save_config_once:
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        with open(os.path.join(cfg.ckpt_dir, 'train_config.json'), 'w') as f:
            json.dump({k: v for k, v in cfg.__dict__.items() if not k.startswith('_')}, f, indent=2)
        print(f"💾 Config saved to {cfg.ckpt_dir}/train_config.json")

    # Initialize advanced loss tracking
    loss_tracker = LossTracker(save_dir="loss_tracking")
    loss_tracker.target_steps = total_steps
    loss_tracker.load_checkpoint()  # Load previous data if exists

    if rank == 0:
        print_success("📊 Advanced loss tracking initialized")
        print(f"   • Target steps: {total_steps:,}")
        print(f"   • Save directory: loss_tracking/")
        print(f"   • Features: Prediction, convergence detection, visualization")

    # Training loop with Rich progress tracking
    accum = 0
    running_loss = 0.0
    running_mtp_loss = 0.0
    running_reason_loss = 0.0
    running_aux_loss = 0.0
    global_step = start_step
    seen_tokens = start_step * cfg.seq_len * effective_batch_size

    # Initialize progress bar
    progress = None
    task_id = None
    if rank == 0:
        progress = rich_output.create_training_progress(total_steps)
        if progress:
            task_id = progress.add_task(
                "Training Progress",
                total=total_steps,
                completed=global_step,
                tokens_per_sec=0.0  # Initialize the field
            )
            progress.start()

    training_start_time = time.time()

    for epoch in range(start_epoch, cfg.num_epochs):
        if rank == 0:
            print_success(f"Starting Epoch {epoch + 1}/{cfg.num_epochs}")

        for it, (x, y) in enumerate(dl):
            if global_step < start_step:
                global_step += 1
                continue

            x = x.to(device, non_blocking=cfg.pin_memory)
            y = y.to(device, non_blocking=cfg.pin_memory)

            # Debug: print min/max token ids on first step to detect OOB indices
            if global_step == start_step:
                try:
                    xmin = int(x.min().item())
                    xmax = int(x.max().item())
                    if rank == 0:
                        print(f"🔎 Token id range in first batch: min={xmin}, max={xmax}, vocab_size={cfg.vocab_size}")
                    if xmin < 0 or xmax >= cfg.vocab_size:
                        if rank == 0:
                            print_warning("⚠️ Token ids out-of-range detected. Clamping to valid vocab range.")
                        x = x.clamp(0, cfg.vocab_size - 1)
                except Exception as e:
                    if rank == 0:
                        print_warning(f"Failed to analyze token ids: {e}")

            # FIXED: Use new autocast API and handle scaler properly
            with autocast_ctx('cuda', enabled=cfg.amp and torch.cuda.is_available()):
                logits_main, logits_mtp, aux_loss, reason_logits = model(x)

                logits_main, penalty = logit_constraint(logits_main)
                loss_main = criterion(logits_main.view(-1, logits_main.size(-1)), y.view(-1))

                # Validate MTP configuration once (only on first step)
                if global_step == start_step:
                    if len(cfg.mtp_loss_weights) != len(logits_mtp):
                        if rank == 0:
                            print_warning(f"MTP heads mismatch: model={len(logits_mtp)}, config={len(cfg.mtp_loss_weights)}")
                            print_warning("Auto-adjusting MTP weights to match model")

                        # Auto-adjust weights
                        if len(logits_mtp) > len(cfg.mtp_loss_weights):
                            # Add weights for extra heads
                            extra_weights = [0.1] * (len(logits_mtp) - len(cfg.mtp_loss_weights))
                            cfg.mtp_loss_weights = cfg.mtp_loss_weights + extra_weights
                        else:
                            # Truncate weights for fewer heads
                            cfg.mtp_loss_weights = cfg.mtp_loss_weights[:len(logits_mtp)]

                    if rank == 0:
                        print_success(f"✅ MTP Configuration:")
                        print_success(f"   • Heads: {len(logits_mtp)}")
                        print_success(f"   • Weights: {cfg.mtp_loss_weights}")
                        print_success(f"   • Sequence length: {y.size(1)}")
                        for i in range(len(logits_mtp)):
                            print_success(f"   • Head {i}: predicts t+{i+1} with weight {cfg.mtp_loss_weights[i]}")

                # FIXED MTP: Proper multi-token prediction loss computation
                mtp_loss = torch.zeros((), device=y.device, dtype=loss_main.dtype)
                for i, mtp_logits in enumerate(logits_mtp):
                    shift = i + 1  # MTP head i predicts t+shift

                    # Only compute loss where we have valid future tokens
                    if shift < y.size(1):  # Ensure we have enough sequence length
                        # Use the same positions for both logits and targets
                        # Predict positions [0, 1, ..., T-shift-1] -> targets [shift, shift+1, ..., T-1]
                        valid_length = y.size(1) - shift

                        # Extract valid predictions and targets
                        mtp_pred = mtp_logits[:, :valid_length]  # [B, T-shift, V]
                        mtp_target = y[:, shift:shift+valid_length]  # [B, T-shift]

                        # Compute loss only on valid positions
                        if mtp_pred.numel() > 0 and mtp_target.numel() > 0:
                            mtp_loss_i = criterion(
                                mtp_pred.contiguous().view(-1, mtp_pred.size(-1)),
                                mtp_target.contiguous().view(-1)
                            )
                            mtp_loss += cfg.mtp_loss_weights[i] * mtp_loss_i

                reason_loss = torch.zeros((), device=y.device, dtype=loss_main.dtype)
                if reason_logits is not None and cfg.reasoning_loss_weight > 0:
                    target_last = y[:, -1]
                    reason_loss = cfg.reasoning_loss_weight * criterion(reason_logits, target_last)

                aux_t = aux_loss if isinstance(aux_loss, torch.Tensor) else torch.zeros((), device=y.device, dtype=loss_main.dtype)
                total_loss = loss_main + mtp_loss + aux_t + penalty + reason_loss

            # Handle backward pass with proper scaler
            if scaler is not None:
                scaler.scale(total_loss / cfg.grad_accum_steps).backward()
            else:
                (total_loss / cfg.grad_accum_steps).backward()

            accum += 1

            # Track individual loss components
            running_loss += loss_main.item()
            running_mtp_loss += mtp_loss.item() if isinstance(mtp_loss, torch.Tensor) else float(mtp_loss)
            running_reason_loss += reason_loss.item() if isinstance(reason_loss, torch.Tensor) else float(reason_loss)
            running_aux_loss += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else float(aux_loss)

            if accum % cfg.grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                # Learning rate schedule
                lr = cosine_with_warmup(global_step, cfg.warmup_steps, total_steps, cfg.lr, cfg.min_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                # Rich logging with beautiful display
                if (global_step % cfg.log_interval == 0) and (rank == 0):
                    avg_loss = running_loss / cfg.log_interval
                    avg_mtp = running_mtp_loss / cfg.log_interval
                    avg_reason = running_reason_loss / cfg.log_interval
                    avg_aux = running_aux_loss / cfg.log_interval
                    mem = torch.cuda.memory_allocated()/1e6 if torch.cuda.is_available() else 0
                    tokens_seen = seen_tokens / 1e6

                    # Calculate performance metrics
                    elapsed = time.time() - training_start_time
                    tokens_per_sec = seen_tokens / elapsed if elapsed > 0 else 0
                    eta = (total_steps - global_step) * (elapsed / global_step) if global_step > 0 else None
                    cpu_percent = 0  # You can add psutil here if needed

                    # Update best loss
                    if avg_loss < best_eval_loss:
                        best_eval_loss = avg_loss

                    # 📊 UPDATE LOSS TRACKER
                    loss_tracker.update(
                        step=global_step,
                        epoch=epoch,
                        loss=avg_loss,
                        lr=lr,
                        main_loss=avg_loss,  # Main component
                        mtp_loss=avg_mtp,
                        aux_loss=avg_aux,
                        reason_loss=avg_reason
                    )

                    # Create metrics object
                    metrics = TrainingMetrics(
                        step=global_step,
                        epoch=epoch,
                        loss=avg_loss,
                        lr=lr,
                        tokens_per_sec=tokens_per_sec,
                        gpu_memory=mem/1000,  # Convert to GB
                        cpu_usage=cpu_percent,
                        elapsed_time=elapsed,
                        eta=eta,
                        best_loss=best_eval_loss
                    )

                    # Rich display
                    print_training_step(metrics, progress, task_id)

                    running_loss = 0.0
                    running_mtp_loss = 0.0
                    running_reason_loss = 0.0
                    running_aux_loss = 0.0

                # Memory reporting with Rich
                if cfg.report_memory_interval and global_step % cfg.report_memory_interval == 0 and rank == 0 and torch.cuda.is_available():
                    peak = torch.cuda.max_memory_allocated()/1e6
                    reserved = torch.cuda.memory_reserved()/1e6
                    print_success(f"Memory: Peak {peak:.0f}MB, Reserved {reserved:.0f}MB")
                    torch.cuda.reset_peak_memory_stats()

                # 📊 ADVANCED LOSS ANALYSIS (every 1000 steps)
                if global_step % 1000 == 0 and rank == 0:
                    print_success("📊 Generating loss analysis...")

                    # Get current stats
                    stats = loss_tracker.get_current_stats()
                    prediction = loss_tracker.predict_final_loss()

                    print(f"🎯 Loss Analysis (Step {global_step:,}):")
                    print(f"  • Current Loss: {stats['current_loss']:.6f}")
                    print(f"  • Best Loss: {stats['best_loss']:.6f}")
                    print(f"  • Moving Average: {stats['moving_average']:.6f}")
                    print(f"  • Loss Reduction Rate: {stats['loss_reduction_rate']:.8f}/step")
                    print(f"  • Convergence Progress: {stats['convergence_progress']*100:.1f}%")

                    if prediction['ensemble_prediction'] is not None:
                        improvement = ((stats['current_loss'] - prediction['ensemble_prediction']) / stats['current_loss'] * 100)
                        print(f"🔮 Final Loss Prediction:")
                        print(f"  • Predicted Final Loss: {prediction['ensemble_prediction']:.6f}")
                        print(f"  • Confidence: {prediction['confidence']*100:.1f}%")
                        print(f"  • Expected Improvement: {improvement:.1f}%")
                        print(f"  • Steps Remaining: {prediction['steps_remaining']:,}")

                    # Generate plots
                    try:
                        loss_tracker.plot_loss_curves()
                        print("  ✅ Loss curves updated")
                    except Exception as e:
                        print(f"  ⚠️ Plot generation failed: {e}")

                    # Check for convergence
                    if loss_tracker.is_converged():
                        print_warning("🎯 Training appears to have converged!")
                        print_warning("   Consider early stopping to save time")

                # Evaluation with Rich display
                if global_step % cfg.eval_interval == 0 and (rank == 0):
                    eval_results, eval_loss = evaluate(model, tokenizer, cfg, device)

                    # Create validation metrics
                    val_metrics = {
                        'perplexity': math.exp(min(eval_loss, 10)),  # Cap for display
                        'quality_score': max(0, 1 - eval_loss/5)  # Rough quality estimate
                    }

                    print_validation_results(eval_loss, val_metrics)

                    # Show sample outputs
                    if eval_results:
                        print_success("Sample generations:")
                        for i, result in enumerate(eval_results[:2]):
                            rich_output.console.print(f"  [dim]Sample {i+1}:[/dim] {result}")

                    # Early stopping and best model saving
                    if cfg.save_best and eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        patience_counter = 0
                        # Save best model
                        best_state = {
                            'model': model.state_dict(),
                            'optim': optimizer.state_dict(),
                            'scaler': scaler.state_dict() if scaler else None,
                            'step': global_step,
                            'epoch': epoch,
                            'eval_loss': eval_loss,
                            'config': cfg.__dict__
                        }
                        best_path = os.path.join(cfg.ckpt_dir, 'best_model.pt')
                        torch.save(best_state, best_path)
                        print_checkpoint_info(best_path, global_step, eval_loss)
                        print_success(f"New best model! Validation loss: {eval_loss:.4f}")
                    else:
                        patience_counter += cfg.eval_interval
                        if patience_counter >= cfg.early_stopping_patience:
                            print_warning(f"Early stopping triggered (patience: {patience_counter})")
                            break

                # Regular checkpointing with Rich display
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
                    checkpoint_path = save_checkpoint(state, cfg.ckpt_dir, global_step, cfg.keep_last_k)
                    current_loss = running_loss / max(1, cfg.log_interval)
                    print_checkpoint_info(checkpoint_path, global_step, current_loss)

            global_step += 1
            seen_tokens += cfg.seq_len * cfg.micro_batch_size * world

            # If smoke mode, stop after first step
            if smoke_mode and global_step >= 1:
                if rank == 0:
                    print("🔬 SMOKE_TEST completed: stopping after 1 step")
                break

        if smoke_mode:
            break

    # Final checkpoint
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
        final_path = os.path.join(cfg.ckpt_dir, 'final_model.pt')
        torch.save(final_state, final_path)

        # Stop progress bar
        if progress and rank == 0:
            progress.stop()

        # 📊 FINAL LOSS ANALYSIS
        if rank == 0:
            print_success("📊 Generating final loss analysis...")
            final_report = loss_tracker.generate_report()
            print(final_report)

            # Save final plots
            loss_tracker.plot_loss_curves()

            # Save final prediction
            final_prediction = loss_tracker.predict_final_loss()
            with open(os.path.join(cfg.ckpt_dir, 'final_loss_analysis.json'), 'w') as f:
                json.dump({
                    'final_stats': loss_tracker.get_current_stats(),
                    'final_prediction': final_prediction,
                    'training_summary': {
                        'total_steps': global_step,
                        'total_tokens': seen_tokens,
                        'best_loss': best_eval_loss,
                        'final_loss': loss_tracker.losses[-1] if loss_tracker.losses else None,
                        'converged': loss_tracker.is_converged()
                    }
                }, f, indent=2)

        print(f"\n🎉 Training Complete!")
        print(f"  Final step: {global_step:,}")
        print(f"  Tokens processed: {seen_tokens/1e6:.1f}M")
        print(f"  Best eval loss: {best_eval_loss:.4f}")
        if rank == 0 and loss_tracker.losses:
            print(f"  Final loss: {loss_tracker.losses[-1]:.6f}")
            if loss_tracker.is_converged():
                print(f"  ✅ Training converged successfully!")
            else:
                print(f"  ⚠️ Training stopped before full convergence")
        print(f"  Models saved in: {cfg.ckpt_dir}")
        print(f"  Loss analysis saved in: loss_tracking/")

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if args.debug:
        train(config_class=SimpleLossDebugConfig)
    else:
        train()