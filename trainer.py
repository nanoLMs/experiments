import os, math, time, json, torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from transformers import PreTrainedTokenizerFast
from config import TrainConfig
from model_moe import NanoMoEModel
from data_loader import create_dataloader
from anti_hallu import LogitConstraint
from schedule import cosine_with_warmup
from checkpoint import save_checkpoint, load_latest
from fp4_quant import apply_fp4

from hrm_bridge import HRMBridge


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
    if cfg.use_fp4 and HAS_BNB:
        optim_cls = bnb.optim.AdamW8bit
    else:
        optim_cls = AdamW
    params = [p for p in model.parameters() if p.requires_grad]
    return optim_cls(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)


def maybe_fsdp_wrap(cfg, model):
    if not (cfg.fsdp and HAS_FSDP and dist.is_initialized()):
        return model
    auto_wrap = size_based_auto_wrap_policy(min_num_params=cfg.fsdp_wrap_layer_size)
    model = FSDP(model, auto_wrap_policy=auto_wrap, backward_prefetch=BackwardPrefetch.BACKWARD_PRE, device_id=torch.cuda.current_device())
    return model


def evaluate(model, tokenizer, cfg, device):
    model.eval()
    with torch.no_grad():
        prompt = "Test:"  # placeholder
        ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
        max_new = cfg.generation_max_new_tokens
        for _ in range(max_new):
            logits_main, _, _, _ = model(ids)
            next_logits = logits_main[:, -1]
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if ids.size(1) >= cfg.seq_len:
                break
        text = tokenizer.batch_decode(ids)[0]
    model.train()
    return text


def train():
    cfg = TrainConfig()
    distributed, rank, world = init_distributed()
    setup_seed(cfg.seed + rank)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer_dir)

    if cfg.forbidden_tokens_file and os.path.exists(cfg.forbidden_tokens_file):
        with open(cfg.forbidden_tokens_file) as f:
            cfg.forbidden_tokens = json.load(f)

    model = NanoMoEModel(cfg)
    # Enable gradient checkpointing if required
    if cfg.gradient_checkpointing:
        model.cfg.gradient_checkpointing = True
    if cfg.use_fp4:
        if HAS_BNB:
            # Defer to bitsandbytes quantization for linear layers automatically (8-bit) + simulate FP4 via custom pass
            apply_fp4(model)
        else:
            apply_fp4(model)
    model = maybe_fsdp_wrap(cfg, model)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id or -100)
    logit_constraint = LogitConstraint(cfg.forbidden_tokens, cfg.factual_penalty_weight).to(device)

    optimizer = get_optimizer(cfg, model)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and torch.cuda.is_available())

    dl = create_dataloader(cfg, tokenizer, world)
    steps_per_epoch = len(dl)
    total_steps = cfg.num_epochs * steps_per_epoch

    ckpt_state, start_step = load_latest(cfg.ckpt_dir)
    if ckpt_state:
        model.load_state_dict(ckpt_state['model'])
        optimizer.load_state_dict(ckpt_state['optim'])
        start_step = ckpt_state['step']
        if rank==0: print(f"Resumed from step {start_step}")

    accum = 0
    running_loss = 0.0
    global_step = start_step

    if rank==0 and cfg.save_config_once:
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        with open(os.path.join(cfg.ckpt_dir, 'train_config.json'), 'w') as f:
            json.dump(cfg.__dict__, f, indent=2)

    seen_tokens = 0
    tokens_per_batch = cfg.seq_len * cfg.micro_batch_size * cfg.grad_accum_steps * world

    for epoch in range(cfg.num_epochs):
        for it, (x, y) in enumerate(dl):
            if global_step < start_step:
                global_step += 1
                continue
            x = x.to(device)
            y = y.to(device)
            with torch.cuda.amp.autocast(enabled=cfg.amp and torch.cuda.is_available()):
                logits_main, logits_mtp, aux_loss, reason_logits = model(x)
                loss_main = criterion(logits_main.view(-1, logits_main.size(-1)), y.view(-1))
                mtp_loss = 0.0
                for i, lm in enumerate(logits_mtp):
                    shift_y = torch.cat([y[:, i+1:], torch.full((y.size(0), i+1), tokenizer.pad_token_id, device=device)], dim=1)
                    mtp_loss += cfg.mtp_loss_weights[i] * criterion(lm.view(-1, lm.size(-1)), shift_y.view(-1))
                reason_loss = 0.0
                if reason_logits is not None:
                    # use last token target as reasoning supervision
                    target_last = y[:, -1]
                    reason_loss = cfg.reasoning_loss_weight * criterion(reason_logits, target_last)
                logits_adjusted, penalty = logit_constraint(logits_main)
                loss = loss_main + mtp_loss + aux_loss + penalty + reason_loss
            scaler.scale(loss / cfg.grad_accum_steps).backward()
            accum += 1
            running_loss += loss.item()
            if accum % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr = cosine_with_warmup(global_step, cfg.warmup_steps, total_steps, cfg.lr, cfg.min_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
                if global_step % cfg.log_interval == 0 and (rank==0):
                    avg_loss = running_loss / cfg.log_interval
                    mem = torch.cuda.memory_allocated()/1e6 if torch.cuda.is_available() else 0
                    print(f"step {global_step} loss {avg_loss:.4f} lr {lr:.2e} memMB {mem:.1f}")
                    running_loss = 0.0
                if cfg.report_memory_interval and global_step % cfg.report_memory_interval == 0 and rank==0 and torch.cuda.is_available():
                    peak = torch.cuda.max_memory_allocated()/1e6
                    print(f"[memory] peak_alloc_MB={peak:.1f}")
                if global_step % cfg.eval_interval == 0 and (rank==0):
                    sample = evaluate(model, tokenizer, cfg, device)
                    print(f"[eval sample] {sample[:120]}")
                if global_step % cfg.save_interval == 0 and (rank==0):
                    state = {'model': model.state_dict(), 'optim': optimizer.state_dict(), 'step': global_step}
                    save_checkpoint(state, cfg.ckpt_dir, global_step, cfg.keep_last_k)
            global_step += 1
            seen_tokens += cfg.seq_len * cfg.micro_batch_size * world
            if (cfg.target_total_tokens and seen_tokens >= cfg.target_total_tokens) or global_step >= total_steps:
                break
        if (cfg.target_total_tokens and seen_tokens >= cfg.target_total_tokens) or global_step >= total_steps:
            break
    if rank==0:
        final_state = {'model': model.state_dict(), 'optim': optimizer.state_dict(), 'step': global_step}
        save_checkpoint(final_state, cfg.ckpt_dir, global_step, cfg.keep_last_k)

        print(f"\n🎉 Training Complete!")
        print(f"  Final step: {global_step:,}")
        print(f"  Models saved in: {cfg.ckpt_dir}")

        # 🚀 AUTOMATIC MODEL EXPORT FOR ALL DEVICES
        print(f"\n🚀 Starting automatic model export for all devices...")
        print(f"📱 Creating models for: Android, iOS, Windows, Linux, Web, Cloud")

        try:
            from enhanced_export_system import enhanced_export_after_training

            # Get the latest checkpoint
            final_path = os.path.join(cfg.ckpt_dir, f'step_{global_step:08d}.pt')

            print(f"📦 Exporting from: {os.path.basename(final_path)}")
            export_sizes = enhanced_export_after_training(final_path, "exported_models")

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


def compute_mtp_loss(cfg, logits, targets):
    # logits: [B,T,V]; targets [B,T]
    if getattr(cfg, "mtp_k", 1) <= 1:
        return nn.functional.cross_entropy(logits.transpose(1,2), targets, reduction='mean')
    # reuse same logits as approximation (parallel MTP heads can be added in model)
    losses = []
    B,T,V = logits.shape
    K = min(cfg.mtp_k, T-1)
    for k in range(1, K+1):
        trunc = T - k
        pred = logits[:, :trunc, :]
        tgt = targets[:, k:k+trunc]
        losses.append(nn.functional.cross_entropy(pred.transpose(1,2), tgt, reduction='mean'))
    return sum(losses)/len(losses)
