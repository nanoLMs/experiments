# Enhanced NF4 quantization with bitsandbytes integration
import torch
import torch.nn as nn

try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit
    HAS_BNB = True
    print("✅ BitsAndBytes available for NF4 quantization")
except ImportError:
    HAS_BNB = False
    print("❌ BitsAndBytes not available - install with: pip install bitsandbytes")

def quantize_fp4(param: torch.Tensor):
    """Custom FP4 simulation via quantization to 16 levels"""
    with torch.no_grad():
        # Get scale factor
        scale = param.abs().amax() + 1e-8
        # Quantize to 4-bit range (-8 to 7)
        q = torch.clamp((param / scale) * 7.0, -8, 7)
        # Round and dequantize
        q_rounded = torch.round(q)
        deq = (q_rounded / 7.0) * scale
        return deq.half()  # Store as half precision

def apply_real_bnb_nf4(model, cfg):
    """Apply real bitsandbytes NF4 quantization using proper Linear4bit"""
    if not HAS_BNB:
        print("❌ BitsAndBytes not available - cannot apply NF4 quantization")
        return False

    print("🔥 Applying bitsandbytes NF4 quantization to linear layers...")
    print(f"   • Quant type: {cfg.bnb_4bit_quant_type}")
    print(f"   • Compute dtype: {cfg.bnb_4bit_compute_dtype}")
    print(f"   • Double quant: {cfg.bnb_4bit_use_double_quant}")

    quantized_count = 0

    def replace_linear_recursive(module, name=""):
        nonlocal quantized_count

        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            # Quantize linear layers (lower threshold to include routers)
            min_params = 100 if 'router' in full_name else 1000  # Lower threshold for routers
            if isinstance(child, nn.Linear) and child.weight.numel() > min_params:
                # Skip certain layers that might need full precision
                skip_layers = ['tok_emb', 'ln_', 'layernorm']  # Always skip these

                # Configurable skipping for heads and router
                if not getattr(cfg, 'bnb_4bit_quantize_heads', False):
                    skip_layers.extend(['head', 'mtp_heads', 'reason_head'])
                if not getattr(cfg, 'bnb_4bit_quantize_router', False):
                    skip_layers.append('router')

                if any(skip in full_name for skip in skip_layers):
                    print(f"  ⏭️ Skipping {full_name} (critical layer)")
                    continue

                try:
                    # Get compute dtype from config
                    compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype, torch.bfloat16)

                    # Create proper Linear4bit replacement
                    quantized_linear = Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=compute_dtype,
                        compress_statistics=cfg.bnb_4bit_use_double_quant,  # This is the correct parameter name
                        quant_type=cfg.bnb_4bit_quant_type,  # "nf4"
                        quant_storage=getattr(torch, cfg.bnb_4bit_quant_storage, torch.uint8)
                    )

                    # Copy weights and bias properly
                    with torch.no_grad():
                        # Copy weight data
                        quantized_linear.weight.data = child.weight.data.clone()

                        # Copy bias if exists
                        if child.bias is not None and quantized_linear.bias is not None:
                            quantized_linear.bias.data = child.bias.data.clone()

                    # Replace the layer
                    setattr(module, child_name, quantized_linear)
                    quantized_count += 1
                    print(f"  ✅ Quantized {full_name}: {child.weight.numel():,} params")

                except Exception as e:
                    print(f"  ⚠️ Failed to quantize {full_name}: {e}")
                    import traceback
                    print(f"     Error details: {traceback.format_exc()}")
            else:
                # Recurse into child modules
                replace_linear_recursive(child, full_name)

    replace_linear_recursive(model)
    print(f"🎯 Successfully quantized {quantized_count} layers with bitsandbytes NF4")

    if quantized_count > 0:
        print("✅ NF4 quantization applied successfully!")
        print("   Expected memory reduction: ~75%")
        print("   Expected speedup: 2-4x during training")

    return quantized_count > 0

def apply_bnb_4bit(model, cfg):
    """Apply bitsandbytes 4-bit quantization with NF4"""
    if not HAS_BNB:
        print("❌ BitsAndBytes not available")
        print("📦 Install with: pip install bitsandbytes")
        return False

    print("🚀 Applying bitsandbytes NF4 4-bit quantization...")
    success = apply_real_bnb_nf4(model, cfg)

    if success:
        print("✅ BitsAndBytes NF4 quantization completed successfully!")
        estimate_memory_savings(model)
        return True
    else:
        print("❌ BitsAndBytes NF4 quantization failed")
        return False

def estimate_memory_savings(model):
    """Estimate memory savings from quantization"""
    total_params = 0
    quantizable_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.dim() > 1 and param.numel() > 1000:
            if not any(skip in name for skip in ['tok_emb', 'ln_', 'layernorm']):
                quantizable_params += param.numel()

    # Rough estimates:
    # FP32: 4 bytes per param
    # FP16: 2 bytes per param
    # FP4/NF4: ~0.5 bytes per param (with some overhead)

    fp32_size = total_params * 4 / 1e6  # MB
    fp16_size = total_params * 2 / 1e6  # MB
    fp4_size = (total_params - quantizable_params) * 2 / 1e6 + quantizable_params * 0.5 / 1e6  # MB

    print(f"📊 Memory Estimates:")
    print(f"  FP32: {fp32_size:.1f} MB")
    print(f"  FP16: {fp16_size:.1f} MB")
    print(f"  FP4 (mixed): {fp4_size:.1f} MB")
    print(f"  Savings: {((fp32_size - fp4_size) / fp32_size * 100):.1f}%")

    return fp4_size
