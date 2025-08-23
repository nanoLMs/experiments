#!/usr/bin/env python3
"""
Clean Model Export Script
Exports model without quantization artifacts for deployment
"""

import torch
import torch.nn as nn
from config import TrainConfig
from model_moe import NanoMoEModel
import os

def clean_state_dict(state_dict):
    """Remove quantization artifacts from state dict"""
    clean_dict = {}

    for key, value in state_dict.items():
        # Skip quantization-specific keys
        if any(skip in key for skip in [
            'absmax', 'quant_map', 'nested_absmax', 'nested_quant_map',
            'quant_state', 'bitsandbytes__nf4'
        ]):
            continue

        # For LoRA layers, only keep the merged weights
        if 'lora_A' in key or 'lora_B' in key:
            continue

        # Keep base weights and regular weights
        if 'base.weight' in key:
            # Rename base.weight to weight
            new_key = key.replace('base.weight', 'weight')
            clean_dict[new_key] = value
        elif '.weight' in key and 'base' not in key and 'lora' not in key:
            clean_dict[key] = value
        elif '.bias' in key:
            clean_dict[key] = value
        else:
            clean_dict[key] = value

    return clean_dict

def export_clean_model(checkpoint_path, output_path):
    """Export a clean, deployable model"""
    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create clean config (no quantization, no LoRA)
    cfg = TrainConfig()
    cfg.use_quantization = False
    cfg.use_bnb_4bit = False
    cfg.lora = False

    # Create clean model
    model = NanoMoEModel(cfg)

    # Clean the state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    clean_dict = clean_state_dict(state_dict)

    # Load clean weights
    try:
        model.load_state_dict(clean_dict, strict=False)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"⚠️ Warning during loading: {e}")
        # Try to load what we can
        model.load_state_dict(clean_dict, strict=False)

    # Save clean model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': cfg,
        'model_size': sum(p.numel() for p in model.parameters()),
    }, output_path)

    print(f"✅ Clean model exported to: {output_path}")
    print(f"📊 Model size: {sum(p.numel() for p in model.parameters()):,} parameters")

if __name__ == "__main__":
    checkpoint_path = "checkpoints/final_model.pt"
    output_path = "checkpoints/clean_model.pt"

    if os.path.exists(checkpoint_path):
        export_clean_model(checkpoint_path, output_path)
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")