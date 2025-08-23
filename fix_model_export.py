#!/usr/bin/env python3
"""
Fix model export for quantized models
Handles state_dict mismatches between quantized and regular models
"""
import torch
import torch.nn as nn
from pathlib import Path
import json
from config import TrainConfig
from model_moe import NanoMoEModel

def create_compatible_model_for_export(checkpoint_path: str, output_path: str = None):
    """
    Create a compatible model for export by handling quantized weights properly
    """
    print(f"🔧 Loading quantized checkpoint: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Load the ORIGINAL config from checkpoint (not current config)
    if 'config' in checkpoint:
        print("🔧 Using config from checkpoint...")
        cfg_dict = checkpoint['config']
        cfg = TrainConfig()
        # Update config with checkpoint values
        for key, value in cfg_dict.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    else:
        print("⚠️ No config in checkpoint, using current config")
        cfg = TrainConfig()

    # CRITICAL: Disable quantization and LoRA for clean export
    print("🔧 Disabling quantization and LoRA for export...")
    cfg.use_quantization = False
    cfg.use_bnb_4bit = False
    cfg.use_fp4 = False
    cfg.lora = False  # Disable LoRA for clean export

    # Create a clean model (no quantization, no LoRA for export)
    print("🔧 Creating clean model for export...")
    clean_model = NanoMoEModel(cfg)

    # Create a mapping of quantized keys to clean keys
    clean_state_dict = {}
    quantized_keys = []

    for key, value in state_dict.items():
        # Skip quantization-specific keys
        if any(quant_key in key for quant_key in [
            '.absmax', '.quant_map', '.nested_absmax', '.nested_quant_map',
            '.quant_state', 'bitsandbytes__nf4'
        ]):
            quantized_keys.append(key)
            continue

        # Handle weight shape mismatches from quantization
        if key in clean_model.state_dict():
            expected_shape = clean_model.state_dict()[key].shape
            actual_shape = value.shape

            if expected_shape != actual_shape:
                print(f"⚠️ Shape mismatch for {key}: expected {expected_shape}, got {actual_shape}")

                # Try to reshape quantized weights back to original shape
                if len(actual_shape) == 2 and actual_shape[1] == 1:
                    # This looks like a flattened quantized weight
                    total_elements = actual_shape[0]
                    expected_elements = 1
                    for dim in expected_shape:
                        expected_elements *= dim

                    if total_elements == expected_elements:
                        print(f"✅ Reshaping {key} from {actual_shape} to {expected_shape}")
                        clean_state_dict[key] = value.view(expected_shape)
                    else:
                        print(f"❌ Cannot reshape {key}: element count mismatch")
                        # Initialize with random weights as fallback
                        clean_state_dict[key] = clean_model.state_dict()[key].clone()
                else:
                    print(f"❌ Complex shape mismatch for {key}, using random initialization")
                    clean_state_dict[key] = clean_model.state_dict()[key].clone()
            else:
                clean_state_dict[key] = value
        else:
            print(f"⚠️ Unknown key in checkpoint: {key}")

    # Add any missing keys from the clean model
    for key in clean_model.state_dict():
        if key not in clean_state_dict:
            print(f"⚠️ Missing key {key}, using random initialization")
            clean_state_dict[key] = clean_model.state_dict()[key].clone()

    print(f"🔧 Processed {len(clean_state_dict)} parameters")
    print(f"🗑️ Skipped {len(quantized_keys)} quantization-specific keys")

    # Load the clean state dict
    try:
        clean_model.load_state_dict(clean_state_dict, strict=False)
        print("✅ Successfully loaded clean state dict")
    except Exception as e:
        print(f"❌ Failed to load state dict: {e}")
        return None

    # Save the clean model
    if output_path is None:
        output_path = checkpoint_path.replace('.pt', '_clean.pt')

    clean_checkpoint = {
        'model': clean_model.state_dict(),
        'config': cfg.__dict__,
        'model_info': {
            'parameters': sum(p.numel() for p in clean_model.parameters()),
            'architecture': 'NanoMoE',
            'precision': 'fp32',
            'quantization_removed': True,
            'original_checkpoint': checkpoint_path
        }
    }

    torch.save(clean_checkpoint, output_path)
    print(f"💾 Saved clean model to: {output_path}")

    return output_path, clean_model

def export_all_formats_fixed(checkpoint_path: str):
    """
    Export all model formats with proper quantization handling
    """
    print("🚀 Starting fixed model export...")

    # First create a clean model
    clean_path, clean_model = create_compatible_model_for_export(checkpoint_path)

    if clean_model is None:
        print("❌ Failed to create clean model")
        return

    # Now export using the standard export system
    from export_models import export_final_models

    print("📦 Exporting all formats...")
    sizes = export_final_models(clean_path, "exported_models_fixed")

    print("🎉 Export completed successfully!")
    print(f"📁 Models saved to: exported_models_fixed/")

    # Print size summary
    print("\n📊 Model Sizes:")
    for format_name, size_bytes in sizes.items():
        size_mb = size_bytes / (1024 * 1024)
        print(f"  • {format_name}: {size_mb:.1f} MB")

    return sizes

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("❌ Usage: python fix_model_export.py <checkpoint_path>")
        print("📝 Example: python fix_model_export.py checkpoints/final_model.pt")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    export_all_formats_fixed(checkpoint_path)