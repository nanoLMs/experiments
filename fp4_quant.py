# Enhanced FP4 / NF4 quantization with bitsandbytes integration
import torch
import torch.nn as nn

try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

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

def apply_real_fp4_bnb(model):
    """Apply real bitsandbytes NF4 quantization where possible"""
    if not HAS_BNB:
        return False

    print("🔥 Applying bitsandbytes NF4 quantization to linear layers...")
    quantized_count = 0

    def replace_linear_recursive(module, name=""):
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            if isinstance(child, nn.Linear) and child.weight.numel() > 1000:  # Only quantize large layers
                # Skip certain layers that might need full precision
                skip_layers = ['head', 'tok_emb', 'ln_', 'router']
                if any(skip in full_name for skip in skip_layers):
                    continue

                try:
                    # Create 4-bit replacement
                    quantized_linear = Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=torch.float16,
                        quant_type="nf4",
                        use_double_quant=True
                    )

                    # Copy weights with quantization
                    with torch.no_grad():
                        quantized_linear.weight.data = child.weight.data
                        if child.bias is not None:
                            quantized_linear.bias.data = child.bias.data

                    setattr(module, child_name, quantized_linear)
                    quantized_count += 1
                    print(f"  ✓ Quantized {full_name}: {child.weight.numel():,} params")

                except Exception as e:
                    print(f"  ⚠️ Failed to quantize {full_name}: {e}")
            else:
                replace_linear_recursive(child, full_name)

    replace_linear_recursive(model)
    print(f"🎯 Quantized {quantized_count} layers with bitsandbytes NF4")
    return quantized_count > 0

def apply_fp4(model):
    """Apply FP4 quantization - try bitsandbytes first, fallback to custom"""
    if HAS_BNB:
        success = apply_real_fp4_bnb(model)
        if success:
            return

    # Fallback to custom FP4 simulation
    print("🔧 Applying custom FP4 simulation...")
    quantized_count = 0

    for name, param in model.named_parameters():
        if param.dim() > 1 and param.numel() > 1000:  # Only quantize weight matrices
            # Skip embeddings and layer norms
            if any(skip in name for skip in ['tok_emb', 'ln_', 'layernorm']):
                continue

            param.data = quantize_fp4(param.data)
            quantized_count += 1

    print(f"🎯 Applied custom FP4 to {quantized_count} parameters")

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
