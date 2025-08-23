
import os, time, torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import PreTrainedTokenizerFast

from config_simple import SimpleTrainConfig
from model_moe import NanoMoEModel
from data_loader import create_dataloader
from schedule import cosine_with_warmup

def setup_seed(seed):
    import random, numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_simple():
    cfg = SimpleTrainConfig()
    setup_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("--- Simple Trainer ---")
    print(f"Using device: {device}")

    try:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.tokenizer_dir)
        print(f"Tokenizer loaded: {len(tokenizer):,} tokens")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    model = NanoMoEModel(cfg).to(device)
    print(f"Model created: {model.num_parameters()/1e6:.2f}M parameters")

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision == 'bf16')
    criterion = nn.CrossEntropyLoss()

    try:
        dl = create_dataloader(cfg, tokenizer, world_size=1)
        steps_per_epoch = len(dl)
    except Exception as e:
        print(f"Failed to create dataloader: {e}")
        return

    total_steps = cfg.num_epochs * steps_per_epoch
    print(f"Total steps: {total_steps}")

    global_step = 0
    for epoch in range(cfg.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{cfg.num_epochs} ---")
        model.train()
        for x, y in dl:
            x, y = x.to(device), y.to(device)

            lr = cosine_with_warmup(global_step, cfg.warmup_steps, total_steps, cfg.lr, cfg.min_lr)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=cfg.mixed_precision == 'bf16'):
                # We only get logits_main since all other features are disabled
                logits_main, _, _, _ = model(x)
                loss = criterion(logits_main.view(-1, logits_main.size(-1)), y.view(-1))

            scaler.scale(loss / cfg.grad_accum_steps).backward()

            if (global_step + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()

            if global_step % cfg.log_interval == 0:
                print(f"Step {global_step}/{total_steps} | Loss: {loss.item():.4f} | LR: {lr:.6f}")

            global_step += 1

    print("\n--- Training Finished ---")

if __name__ == '__main__':
    train_simple()
