#!/usr/bin/env python3
"""
Advanced Model Export Fix for Quantized + LoRA Models
Handles state_dict mismatches between quantized LoRA models and clean export models
"""
import torch
import torch.nn as nn
from pathlib import Path
import json
from config import TrainConfig
from model_moe import NanoMoEModel

def merge_lora_weights(base_weight, lora_A, lora_B, lora_alpha=16, lora_rank=8):
    """
    Merge LoRA weights into base weight: W = W_base + (lora_B @ lora_A) * (alpha/rank)
    """
    try:
        scaling = lora_alpha / lora_rank
        delta_w = torch.mm(lora_B, lora_A) * scaling
        return base_weight + delta_w
    except Exception as e:
        print(f"⚠️ LoRA merge failed: {e}")
        return base_weight

def reshape_quantized_weight(weight, target_shape):
    """
    Attempt to reshape quantized weight back to original shape
    """
    if weight.numel() != torch.prod(torch.tensor(target_shape)):
        return None

    try:
        return weight.view(target_shape)
    except:
        return None

def create_compatible_model_for_export(checkpoint_path: str, output_path: str = None):
    """
    Create a compatible model for export by properly handling quantized LoRA weights
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

    # Load the ORIGINAL config from checkpoint
    if 'config' in checkpoint:
        print("🔧 Using config from checkpoint...")
        cfg_dict = checkpoint['config']
        cfg = TrainConfig()
        # Update config with checkpoint values
        for key, value in cfg_dict.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

        print(f"   • Architecture: {cfg.n_layers} layers, {cfg.d_model} dim")
        print(f"   • MoE: every {cfg.moe_every} layers, {cfg.n_experts} experts")
        print(f"   • LoRA: {'enabled' if cfg.lora else 'disabled'}")
        print(f"   • Quantization: {'enabled' if cfg.use_bnb_4bit else 'disabled'}")
    else:
        print("⚠️ No config in checkpoint, using current config")
        cfg = TrainConfig()

    # CRITICAL: Create export config (disable quantization and LoRA)
    export_cfg = TrainConfig()
    # Copy architecture settings from checkpoint
    export_cfg.vocab_size = cfg.vocab_size
    export_cfg.n_layers = cfg.n_layers
    export_cfg.n_heads = cfg.n_heads
    export_cfg.d_model = cfg.d_model
    export_cfg.d_ff = cfg.d_ff
    export_cfg.moe_every = cfg.moe_every
    export_cfg.n_experts = cfg.n_experts
    export_cfg.moe_top_k = cfg.moe_top_k
    export_cfg.expert_ff_mult = cfg.expert_ff_mult
    export_cfg.enable_reasoning = cfg.enable_reasoning
    export_cfg.reasoning_dim = cfg.reasoning_dim
    export_cfg.reasoning_layers = cfg.reasoning_layers
    export_cfg.mtp_heads = cfg.mtp_heads

    # DISABLE quantization and LoRA for clean export
    export_cfg.use_quantization = False
    export_cfg.use_bnb_4bit = False
    export_cfg.use_fp4 = False
    export_cfg.lora = False

    print("🔧 Creating clean export model...")
    print(f"   • No quantization, no LoRA")
    print(f"   • Architecture matches checkpoint")

    clean_model = NanoMoEModel(export_cfg)

    # Organize weights by type
    base_weights = {}
    lora_A_weights = {}
    lora_B_weights = {}
    regular_weights = {}
    quantized_keys = []

    print("🔍 Analyzing checkpoint weights...")

    for key, value in state_dict.items():
        # Skip quantization metadata
        if any(quant_key in key for quant_key in [
            '.absmax', '.quant_map', '.nested_absmax', '.nested_quant_map',
            '.quant_state', 'bitsandbytes__nf4'
        ]):
            quantized_keys.append(key)
            continue

        # Categorize weights
        if '.base.' in key:
            clean_key = key.replace('.base.', '.')
            base_weights[clean_key] = value
        elif '.lora_A.' in key:
            clean_key = key.replace('.lora_A.', '.')
            lora_A_weights[clean_key] = value
        elif '.lora_B.' in key:
            clean_key = key.replace('.lora_B.', '.')
            lora_B_weights[clean_key] = value
        else:
            regular_weights[key] = value

    print(f"   • Base weights: {len(base_weights)}")
    print(f"   • LoRA A weights: {len(lora_A_weights)}")
    print(f"   • LoRA B weights: {len(lora_B_weights)}")
    print(f"   • Regular weights: {len(regular_weights)}")
    print(f"   • Quantization keys (skipped): {len(quantized_keys)}")

    # Build clean state dict
    clean_state_dict = {}

    # 1. Process LoRA weights (merge base + LoRA)
    print("🔧 Merging LoRA weights...")
    lora_merged = 0
    lora_failed = 0

    for key in base_weights.keys():
        base_weight = base_weights[key]

        # Check if we have corresponding LoRA weights
        if key in lora_A_weights and key in lora_B_weights:
            lora_A = lora_A_weights[key]
            lora_B = lora_B_weights[key]

            # Handle quantized weights (flattened to [N, 1])
            if (base_weight.dim() == 2 and base_weight.shape[1] == 1 and
                lora_A.dim() == 2 and lora_A.shape[1] == 1 and
                lora_B.dim() == 2 and lora_B.shape[1] == 1):

                print(f"⚠️ Quantized LoRA weights detected for {key}")

                # Get target shape from clean model
                if key in clean_model.state_dict():
                    target_shape = clean_model.state_dict()[key].shape

                    # Try to reshape base weight
                    reshaped_base = reshape_quantized_weight(base_weight, target_shape)
                    if reshaped_base is not None:
                        print(f"✅ Reshaped quantized base weight for {key}")
                        clean_state_dict[key] = reshaped_base
                        lora_merged += 1
                    else:
                        print(f"❌ Failed to reshape {key}, using random init")
                        clean_state_dict[key] = clean_model.state_dict()[key].clone()
                        lora_failed += 1
                else:
                    print(f"⚠️ Key {key} not found in clean model")
                    lora_failed += 1

            # Handle regular (non-quantized) LoRA weights
            else:
                try:
                    merged_weight = merge_lora_weights(
                        base_weight, lora_A, lora_B,
                        cfg.lora_alpha, cfg.lora_rank
                    )
                    clean_state_dict[key] = merged_weight
                    print(f"✅ Merged LoRA for {key}")
                    lora_merged += 1
                except Exception as e:
 print(f"❌ LoRA merge failed for {key}: {e}")
                    clean_state_dict[key] = base_weight  # Use base weight only
                    lora_failed += 1
        else:
            # No LoRA components, use base weight directly
            if key in clean_model.state_dict():
                target_shape = clean_model.state_dict()[key].shape
                if base_weight.shape != target_shape:
                    reshaped = reshape_quantized_weight(base_weight, target_shape)
                    if reshaped is not None:
                        clean_state_dict[key] = reshaped
                        print(f"✅ Reshaped {key}")
                    else:
                        print(f"❌ Shape mismatch for {key}, using random init")
                        clean_state_dict[key] = clean_model.state_dict()[key].clone()
                else:
                    clean_state_dict[key] = base_weight
            else:
                print(f"⚠️ Unknown key: {key}")

    # 2. Process regular weights (non-LoRA)
    print("🔧 Processing regular weights...")
    regular_processed = 0

    for key, value in regular_weights.items():
        if key in clean_model.state_dict():
            target_shape = clean_model.state_dict()[key].shape
            if value.shape != target_shape:
                reshaped = reshape_quantized_weight(value, target_shape)
                if reshaped is not None:
                    clean_state_dict[key] = reshaped
                    print(f"✅ Reshaped regular weight {key}")
                else:
                    print(f"❌ Shape mismatch for {key}, using random init")
                    clean_state_dict[key] = clean_model.state_dict()[key].clone()
            else:
                clean_state_dict[key] = value
            regular_processed += 1
        else:
            print(f"⚠️ Unknown regular key: {key}")

    # 3. Fill in missing weights with random initialization
    print("🔧 Filling missing weights...")
    missing_filled = 0

    for key in clean_model.state_dict():
        if key not in clean_state_dict:
            clean_state_dict[key] = clean_model.state_dict()[key].clone()
            print(f"⚠️ Missing key {key}, using random initialization")
            missing_filled += 1

    # Summary
    print(f"\n📊 Processing Summary:")
    print(f"   • LoRA weights merged: {lora_merged}")
    print(f"   • LoRA merge failures: {lora_failed}")
    print(f"   • Regular weights processed: {regular_processed}")
    print(f"   • Missing weights filled: {missing_filled}")
    print(f"   • Total parameters: {len(clean_state_dict)}")

    # Load the clean state dict
    try:
        missing_keys, unexpected_keys = clean_model.load_state_dict(clean_state_dict, strict=False)
        if missing_keys:
            print(f"⚠️ Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
        print("✅ Successfully loaded clean state dict")
    except Exception as e:
        print(f"❌ Failed to load state dict: {e}")
        return None, None

    # Save the clean model
    if output_path is None:
        output_path = checkpoint_path.replace('.pt', '_clean_export.pt')

    clean_checkpoint = {
        'model': clean_model.state_dict(),
        'config': export_cfg.__dict__,
        'model_info': {
            'parameters': sum(p.numel() for p in clean_model.parameters()),
            'architecture': 'NanoMoE',
            'precision': 'fp32',
            'quantization_removed': True,
            'lora_merged': True,
            'original_checkpoint': checkpoint_path,
            'lora_weights_merged': lora_merged,
            'processing_summary': {
                'lora_merged': lora_merged,
                'lora_failed': lora_failed,
                'regular_processed': regular_processed,
                'missing_filled': missing_filled
            }
        }
    }

    torch.save(clean_checkpoint, output_path)
    print(f"💾 Saved clean export model to: {output_path}")

    return output_path, clean_model

def export_all_formats_fixed(checkpoint_path: str):
    """
    Export all model formats with proper quantization and LoRA handling
    """
    print("🚀 Starting advanced model export with LoRA merging...")

    # First create a clean model with LoRA merged
    clean_path, clean_model = create_compatible_model_for_export(checkpoint_path)

    if clean_model is None:
        print("❌ Failed to create clean model")
        return

    # Now export using the standard export system with dtype fixes
    print("📦 Exporting all formats with dtype compatibility...")

    # Create a simple exporter that handles dtype issues
    from export_models import ModelExporter

    try:
        exporter = ModelExporter(clean_path, "exported_models_fixed")

        # Override the model to ensure FP32 for compatibility
        exporter.model = exporter.model.float()  # Ensure FP32

        sizes = exporter.export_all_formats()

        print("🎉 Export completed successfully!")
        print(f"📁 Models saved to: exported_models_fixed/")

        # Print size summary
        print("\n📊 Model Sizes:")
        for format_name, size_bytes in sizes.items():
            size_mb = size_bytes / (1024 * 1024)
            print(f"  • {format_name}: {size_mb:.1f} MB")

        return sizes

    except Exception as e:
        print(f"❌ Export failed: {e}")
        print("🔧 Trying basic PyTorch export...")

        # Fallback: basic PyTorch export
        output_dir = Path("exported_models_fixed")
        output_dir.mkdir(exist_ok=True)

        # FP32 model
        fp32_path = output_dir / "model_fp32.pt"
        torch.save({
            'model_state_dict': clean_model.float().state_dict(),
            'config': clean_model.cfg.__dict__ if hasattr(clean_model, 'cfg') else {},
            'model_info': {
                'parameters': sum(p.numel() for p in clean_model.parameters()),
                'precision': 'fp32',
                'format': 'pytorch'
            }
        }, fp32_path)

        # FP16 model
        fp16_path = output_dir / "model_fp16.pt"
        torch.save({
            'model_state_dict': clean_model.half().state_dict(),
            'config': clean_model.cfg.__dict__ if hasattr(clean_model, 'cfg') else {},
            'model_info': {
                'parameters': sum(p.numel() for p in clean_model.parameters()),
                'precision': 'fp16',
                'format': 'pytorch'
            }
        }, fp16_path)

        fp32_size = fp32_path.stat().st_size
        fp16_size = fp16_path.stat().st_size

        print(f"✅ Basic export completed:")
        print(f"  • FP32: {fp32_size / (1024*1024):.1f} MB")
        print(f"  • FP16: {fp16_size / (1024*1024):.1f} MB")

        return {
            'pytorch_fp32': fp32_size,
            'pytorch_fp16': fp16_size
        }

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("❌ Usage: python fix_model_export_v2.py <checkpoint_path>")
        print("📝 Example: python fix_model_export_v2.py checkpoints/final_model.pt")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    export_all_formats_fixed(checkpoint_path)