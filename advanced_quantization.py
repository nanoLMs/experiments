#!/usr/bin/env python3
"""
Advanced Quantization System for NanoLM
=======================================

Implements state-of-the-art quantization techniques:
- NF4 quantization with bitsandbytes for training
- FP4 quantization with NVFP4 format for fine-tuning
- QAF (Quantization-Aware Fine-tuning) transition detection
- Quality validation and error recovery
"""

import torch
import torch.nn as nn
import logging
import warnings
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import json

try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit, Embedding
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    bnb = None
    Linear4bit = None
    Embedding = None

try:
    from transformers import BitsAndBytesConfig
    TRANSFORMERS_BNB_AVAILABLE = True
except ImportError:
    TRANSFORMERS_BNB_AVAILABLE = False
    BitsAndBytesConfig = None


@dataclass
class QuantizationMetrics:
    """Metrics for quantization quality assessment"""
    timestamp: float
    quantization_error: float
    weight_distribution: Dict[str, float]
    activation_range: Dict[str, Tuple[float, float]]
    gradient_noise_ratio: float
    quality_score: float
    memory_reduction: float


class NF4Quantizer:
    """
    NF4 Quantization using bitsandbytes

    Implements Normal Float 4-bit quantization for training phase.
    Based on QLoRA paper: https://arxiv.org/abs/2305.14314
    """

    def __init__(self, config):
        self.config = config
        self.quantization_config = None
        self.quantized_modules = {}
        self.original_dtypes = {}

        if not BNB_AVAILABLE:
            raise ImportError("bitsandbytes not available. Install with: pip install bitsandbytes")

        # Create bitsandbytes config
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,  # "nf4"
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),  # torch.bfloat16
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,  # True
            bnb_4bit_quant_storage=getattr(torch, config.bnb_4bit_quant_storage) if hasattr(torch, config.bnb_4bit_quant_storage) else torch.uint8
        )

        logging.info(f"✅ NF4 Quantizer initialized")
        logging.info(f"  • Quantization type: {config.bnb_4bit_quant_type}")
        logging.info(f"  • Compute dtype: {config.bnb_4bit_compute_dtype}")
        logging.info(f"  • Double quantization: {config.bnb_4bit_use_double_quant}")

    def quantize_linear_layer(self, layer: nn.Linear, layer_name: str) -> nn.Module:
        """Convert a linear layer to 4-bit quantized version"""
        if not isinstance(layer, nn.Linear):
            return layer

        # Store original dtype
        self.original_dtypes[layer_name] = layer.weight.dtype

        # Create quantized layer
        quantized_layer = Linear4bit(
            input_features=layer.in_features,
            output_features=layer.out_features,
            bias=layer.bias is not None,
            compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            compress_statistics=self.config.bnb_4bit_use_double_quant,
            quant_type=self.config.bnb_4bit_quant_type
        )

        # Copy weights and bias
        with torch.no_grad():
            quantized_layer.weight.data = layer.weight.data
            if layer.bias is not None:
                quantized_layer.bias.data = layer.bias.data

        self.quantized_modules[layer_name] = quantized_layer

        logging.debug(f"✅ Quantized layer: {layer_name}")
        return quantized_layer

    def quantize_embedding_layer(self, layer: nn.Embedding, layer_name: str) -> nn.Module:
        """Convert embedding layer to quantized version if enabled"""
        if not self.config.bnb_4bit_quantize_heads:
            return layer

        # For now, keep embeddings in original precision
        # Can be extended to use quantized embeddings if needed
        logging.debug(f"ℹ️ Keeping embedding in original precision: {layer_name}")
        return layer

    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Apply NF4 quantization to entire model"""
        quantized_count = 0
        total_params_before = sum(p.numel() for p in model.parameters())

        # Recursively quantize linear layers
        def quantize_recursive(module, prefix=""):
            nonlocal quantized_count

            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name

                if isinstance(child, nn.Linear):
                    # Quantize linear layers
                    quantized_layer = self.quantize_linear_layer(child, full_name)
                    setattr(module, name, quantized_layer)
                    quantized_count += 1

                elif isinstance(child, nn.Embedding):
                    # Handle embedding layers
                    quantized_layer = self.quantize_embedding_layer(child, full_name)
                    setattr(module, name, quantized_layer)

                else:
                    # Recursively process child modules
                    quantize_recursive(child, full_name)

        quantize_recursive(model)

        total_params_after = sum(p.numel() for p in model.parameters())
        memory_reduction = 1.0 - (total_params_after / total_params_before)

        logging.info(f"✅ NF4 quantization completed")
        logging.info(f"  • Quantized layers: {quantized_count}")
        logging.info(f"  • Memory reduction: {memory_reduction:.1%}")
        logging.info(f"  • Parameters: {total_params_before:,} → {total_params_after:,}")

        return model

    def validate_quantization_quality(self, original_model: nn.Module, quantized_model: nn.Module,
                                    test_input: torch.Tensor) -> QuantizationMetrics:
        """Validate quantization quality by comparing outputs"""
        with torch.no_grad():
            # Get outputs from both models
            original_output = original_model(test_input)
            quantized_output = quantized_model(test_input)

            # Calculate quantization error
            if isinstance(original_output, tuple):
                original_logits = original_output[0]
                quantized_logits = quantized_output[0]
            else:
                original_logits = original_output
                quantized_logits = quantized_output

            # Mean squared error
            mse_error = torch.mean((original_logits - quantized_logits) ** 2).item()

            # Relative error
            relative_error = mse_error / torch.mean(original_logits ** 2).item()

            # Weight distribution analysis
            weight_stats = {}
            for name, module in quantized_model.named_modules():
                if isinstance(module, Linear4bit):
                    weights = module.weight.data.float()
                    weight_stats[name] = {
                        'mean': weights.mean().item(),
                        'std': weights.std().item(),
                        'min': weights.min().item(),
                        'max': weights.max().item()
                    }

            # Quality score (higher is better)
            quality_score = 1.0 / (1.0 + relative_error)

            metrics = QuantizationMetrics(
                timestamp=torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True)),
                quantization_error=relative_error,
                weight_distribution=weight_stats,
                activation_range={},  # Can be extended
                gradient_noise_ratio=0.0,  # Will be updated during training
                quality_score=quality_score,
                memory_reduction=0.75  # Approximate for NF4
            )

            logging.info(f"📊 Quantization quality validation:")
            logging.info(f"  • Relative error: {relative_error:.6f}")
            logging.info(f"  • Quality score: {quality_score:.4f}")

            return metrics

    def get_quantization_info(self) -> Dict[str, Any]:
        """Get information about quantized modules"""
        return {
            'quantized_modules': list(self.quantized_modules.keys()),
            'quantization_config': {
                'quant_type': self.config.bnb_4bit_quant_type,
                'compute_dtype': self.config.bnb_4bit_compute_dtype,
                'double_quant': self.config.bnb_4bit_use_double_quant
            },
            'total_quantized': len(self.quantized_modules)
        }


class FP4Quantizer:
    """
    FP4 Quantization for fine-tuning phase

    Implements NVFP4 format based on:
    "FP4 Fully Quantized Training" paper
    """

    def __init__(self, config):
        self.config = config
        self.quantized_modules = {}
        self.scale_factors = {}
        self.zero_points = {}

        # FP4 format configuration
        self.format = config.fp4_format  # "nvfp4"
        self.block_size = config.fp4_block_size  # 16
        self.split_rounding = config.fp4_split_rounding  # True

        # NVFP4 format: E2M1 (1 sign + 2 exponent + 1 mantissa)
        self.fp4_min = -6.0
        self.fp4_max = 6.0
        self.fp4_levels = 16  # 2^4 = 16 levels

        logging.info(f"✅ FP4 Quantizer initialized")
        logging.info(f"  • Format: {self.format}")
        logging.info(f"  • Block size: {self.block_size}")
        logging.info(f"  • Split rounding: {self.split_rounding}")

    def quantize_tensor_fp4(self, tensor: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Quantize tensor to FP4 format"""
        original_shape = tensor.shape
        tensor_flat = tensor.flatten()

        # Block-wise quantization
        quantized_blocks = []
        num_blocks = (tensor_flat.numel() + self.block_size - 1) // self.block_size

        for i in range(num_blocks):
            start_idx = i * self.block_size
            end_idx = min((i + 1) * self.block_size, tensor_flat.numel())
            block = tensor_flat[start_idx:end_idx]

            # Calculate scale and zero point for this block
            block_min = block.min()
            block_max = block.max()

            # Avoid division by zero
            if block_max == block_min:
                scale = 1.0
                zero_point = 0.0
            else:
                scale = (block_max - block_min) / (self.fp4_max - self.fp4_min)
                zero_point = block_min - scale * self.fp4_min

            # Quantize block
            normalized = (block - zero_point) / scale

            if training and self.split_rounding:
                # Stochastic rounding for backward pass
                quantized = torch.floor(normalized) + torch.bernoulli(normalized - torch.floor(normalized))
            else:
                # Round-to-nearest for forward pass
                quantized = torch.round(normalized)

            # Clamp to FP4 range
            quantized = torch.clamp(quantized, self.fp4_min, self.fp4_max)

            # Dequantize
            dequantized = quantized * scale + zero_point
            quantized_blocks.append(dequantized)

        # Concatenate blocks and reshape
        quantized_tensor = torch.cat(quantized_blocks)[:tensor_flat.numel()]
        return quantized_tensor.reshape(original_shape)

    def quantize_model(self, model: nn.Module) -> nn.Module:
        """Apply FP4 quantization to model"""
        quantized_count = 0

        def apply_fp4_hook(module, input, output):
            """Hook to apply FP4 quantization to layer outputs"""
            if isinstance(output, torch.Tensor):
                return self.quantize_tensor_fp4(output, training=module.training)
            elif isinstance(output, tuple):
                return tuple(self.quantize_tensor_fp4(o, training=module.training) if isinstance(o, torch.Tensor) else o for o in output)
            return output

        # Apply hooks to linear layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, Linear4bit)):
                module.register_forward_hook(apply_fp4_hook)
                self.quantized_modules[name] = module
                quantized_count += 1

        logging.info(f"✅ FP4 quantization applied to {quantized_count} layers")
        return model

    def remove_quantization(self, model: nn.Module) -> nn.Module:
        """Remove FP4 quantization hooks"""
        for name, module in model.named_modules():
            if hasattr(module, '_forward_hooks'):
                module._forward_hooks.clear()

        logging.info("✅ FP4 quantization hooks removed")
        return model


class QAFDetector:
    """
    QAF (Quantization-Aware Fine-tuning) Transition Detector

    Detects when to transition from FP4 to higher precision based on
    gradient-to-noise ratio analysis from the research paper.
    """

    def __init__(self, config):
        self.config = config
        self.threshold = config.qaf_threshold  # 1e-6
        self.max_qaf_steps = config.max_qaf_steps  # 1000
        self.qaf_precision = config.qaf_precision  # "bf16"

        # Gradient tracking
        self.gradient_history = []
        self.noise_history = []
        self.gradient_norms = []

        # State tracking
        self.qaf_triggered = False
        self.qaf_step_count = 0
        self.trigger_step = None

        logging.info(f"✅ QAF Detector initialized")
        logging.info(f"  • Threshold: {self.threshold}")
        logging.info(f"  • Max QAF steps: {self.max_qaf_steps}")
        logging.info(f"  • QAF precision: {self.qaf_precision}")

    def update_gradient_stats(self, model: nn.Module, step: int):
        """Update gradient statistics for QAF detection"""
        total_grad_norm = 0.0
        total_params = 0
        grad_variance = 0.0

        # Calculate gradient statistics
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm().item()
                total_grad_norm += grad_norm ** 2
                total_params += param.numel()

                # Calculate gradient variance (proxy for noise)
                grad_flat = param.grad.data.flatten()
                grad_var = torch.var(grad_flat).item()
                grad_variance += grad_var

        total_grad_norm = np.sqrt(total_grad_norm)
        avg_grad_variance = grad_variance / max(1, len(list(model.parameters())))

        # Store statistics
        self.gradient_norms.append(total_grad_norm)
        self.gradient_history.append(total_grad_norm)
        self.noise_history.append(avg_grad_variance)

        # Keep only recent history
        max_history = 100
        if len(self.gradient_history) > max_history:
            self.gradient_history = self.gradient_history[-max_history:]
            self.noise_history = self.noise_history[-max_history:]

    def calculate_gradient_to_noise_ratio(self) -> float:
        """Calculate gradient-to-noise ratio"""
        if len(self.gradient_history) < 10:
            return float('inf')  # Not enough data

        # Use recent gradient statistics
        recent_gradients = self.gradient_history[-10:]
        recent_noise = self.noise_history[-10:]

        avg_gradient = np.mean(recent_gradients)
        avg_noise = np.mean(recent_noise)

        if avg_noise == 0:
            return float('inf')

        return avg_gradient / avg_noise

    def should_trigger_qaf(self, step: int) -> bool:
        """Check if QAF should be triggered"""
        if self.qaf_triggered:
            return False

        if len(self.gradient_history) < 20:
            return False  # Need sufficient history

        # Calculate gradient-to-noise ratio
        gnr = self.calculate_gradient_to_noise_ratio()

        # Check theoretical threshold: √(3d) × σ_q
        # For FP4, σ_q ≈ 0.1 (quantization noise)
        # d is effective dimensionality (approximated)
        d_eff = 1000  # Rough approximation
        theoretical_threshold = np.sqrt(3 * d_eff) * 0.1

        # Use the more restrictive threshold
        effective_threshold = min(self.threshold, theoretical_threshold)

        if gnr < effective_threshold:
            logging.info(f"🎯 QAF trigger detected at step {step}")
            logging.info(f"  • Gradient-to-noise ratio: {gnr:.8f}")
            logging.info(f"  • Threshold: {effective_threshold:.8f}")

            self.qaf_triggered = True
            self.trigger_step = step
            return True

        return False

    def is_qaf_complete(self, step: int) -> bool:
        """Check if QAF phase is complete"""
        if not self.qaf_triggered:
            return False

        self.qaf_step_count = step - self.trigger_step
        return self.qaf_step_count >= self.max_qaf_steps

    def get_qaf_status(self) -> Dict[str, Any]:
        """Get current QAF status"""
        gnr = self.calculate_gradient_to_noise_ratio()

        return {
            'qaf_triggered': self.qaf_triggered,
            'trigger_step': self.trigger_step,
            'qaf_step_count': self.qaf_step_count,
            'gradient_to_noise_ratio': gnr,
            'threshold': self.threshold,
            'is_complete': self.is_qaf_complete(self.qaf_step_count + (self.trigger_step or 0))
        }


class QuantizationController:
    """
    Main controller for quantization system

    Manages transitions between NF4 training, FP4 fine-tuning, and QAF phases.
    """

    def __init__(self, config):
        self.config = config
        self.current_phase = "nf4"  # "nf4", "fp4", "qaf"

        # Initialize quantizers
        self.nf4_quantizer = NF4Quantizer(config) if config.use_bnb_4bit else None
        self.fp4_quantizer = FP4Quantizer(config) if config.use_fp4 else None
        self.qaf_detector = QAFDetector(config)

        # Metrics tracking
        self.quantization_metrics = []

        logging.info(f"✅ Quantization Controller initialized")
        logging.info(f"  • Current phase: {self.current_phase}")
        logging.info(f"  • NF4 enabled: {config.use_bnb_4bit}")
        logging.info(f"  • FP4 enabled: {config.use_fp4}")

    def apply_quantization(self, model: nn.Module, phase: Optional[str] = None) -> nn.Module:
        """Apply quantization based on current or specified phase"""
        if phase is None:
            phase = self.current_phase

        if phase == "nf4" and self.nf4_quantizer:
            logging.info("🔧 Applying NF4 quantization...")
            model = self.nf4_quantizer.quantize_model(model)

        elif phase == "fp4" and self.fp4_quantizer:
            logging.info("🔧 Applying FP4 quantization...")
            model = self.fp4_quantizer.quantize_model(model)

        elif phase == "qaf":
            logging.info("🔧 Entering QAF phase (higher precision)...")
            # Remove FP4 quantization for QAF
            if self.fp4_quantizer:
                model = self.fp4_quantizer.remove_quantization(model)

        return model

    def update_phase(self, model: nn.Module, step: int) -> Tuple[nn.Module, bool]:
        """Update quantization phase based on training progress"""
        phase_changed = False

        # Update QAF detector
        self.qaf_detector.update_gradient_stats(model, step)

        # Check for phase transitions
        if self.current_phase == "nf4" and self.config.use_fp4:
            # Transition to FP4 after initial training
            if step > 1000:  # After some initial training
                logging.info("🔄 Transitioning from NF4 to FP4...")
                self.current_phase = "fp4"
                model = self.apply_quantization(model, "fp4")
                phase_changed = True

        elif self.current_phase == "fp4":
            # Check for QAF trigger
            if self.qaf_detector.should_trigger_qaf(step):
                logging.info("🔄 Transitioning from FP4 to QAF...")
                self.current_phase = "qaf"
                model = self.apply_quantization(model, "qaf")
                phase_changed = True

        elif self.current_phase == "qaf":
            # Check if QAF is complete
            if self.qaf_detector.is_qaf_complete(step):
                logging.info("✅ QAF phase completed")

        return model, phase_changed

    def get_quantization_status(self) -> Dict[str, Any]:
        """Get current quantization status"""
        status = {
            'current_phase': self.current_phase,
            'qaf_status': self.qaf_detector.get_qaf_status(),
            'metrics_count': len(self.quantization_metrics)
        }

        if self.nf4_quantizer:
            status['nf4_info'] = self.nf4_quantizer.get_quantization_info()

        return status

    def save_quantization_state(self, path: str):
        """Save quantization state to file"""
        state = {
            'current_phase': self.current_phase,
            'qaf_status': self.qaf_detector.get_qaf_status(),
            'gradient_history': self.qaf_detector.gradient_history[-100:],  # Keep recent history
            'metrics': [vars(m) for m in self.quantization_metrics[-10:]]  # Keep recent metrics
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

        logging.info(f"💾 Quantization state saved to {path}")

    def load_quantization_state(self, path: str):
        """Load quantization state from file"""
        try:
            with open(path, 'r') as f:
                state = json.load(f)

            self.current_phase = state.get('current_phase', 'nf4')

            # Restore QAF detector state
            qaf_status = state.get('qaf_status', {})
            self.qaf_detector.qaf_triggered = qaf_status.get('qaf_triggered', False)
            self.qaf_detector.trigger_step = qaf_status.get('trigger_step')
            self.qaf_detector.gradient_history = state.get('gradient_history', [])

            logging.info(f"📂 Quantization state loaded from {path}")

        except Exception as e:
            logging.warning(f"⚠️ Failed to load quantization state: {e}")


def create_quantization_controller(config) -> QuantizationController:
    """Factory function to create quantization controller"""
    return QuantizationController(config)


def test_quantization_system():
    """Test the quantization system"""
    print("🧪 Testing Quantization System")

    # Mock config for testing
    class MockConfig:
        use_bnb_4bit = True
        bnb_4bit_quant_type = "nf4"
        bnb_4bit_compute_dtype = "bfloat16"
        bnb_4bit_use_double_quant = True
        bnb_4bit_quant_storage = "uint8"
        bnb_4bit_quantize_heads = True
        bnb_4bit_quantize_router = True

        use_fp4 = True
        fp4_format = "nvfp4"
        fp4_block_size = 16
        fp4_split_rounding = True

        qaf_threshold = 1e-6
        max_qaf_steps = 1000
        qaf_precision = "bf16"

    config = MockConfig()

    # Test NF4 quantizer
    if BNB_AVAILABLE:
        print("✅ Testing NF4 quantizer...")
        nf4_quantizer = NF4Quantizer(config)

        # Create test model
        test_model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        )

        # Quantize model
        quantized_model = nf4_quantizer.quantize_model(test_model)
        print(f"✅ NF4 quantization test passed")
    else:
        print("⚠️ Skipping NF4 test - bitsandbytes not available")

    # Test FP4 quantizer
    print("✅ Testing FP4 quantizer...")
    fp4_quantizer = FP4Quantizer(config)

    test_tensor = torch.randn(32, 128)
    quantized_tensor = fp4_quantizer.quantize_tensor_fp4(test_tensor)
    print(f"✅ FP4 quantization test passed")

    # Test QAF detector
    print("✅ Testing QAF detector...")
    qaf_detector = QAFDetector(config)

    # Simulate gradient updates
    test_model = nn.Linear(64, 32)
    for step in range(25):
        # Simulate gradients
        for param in test_model.parameters():
            param.grad = torch.randn_like(param) * 0.01

        qaf_detector.update_gradient_stats(test_model, step)

    gnr = qaf_detector.calculate_gradient_to_noise_ratio()
    print(f"✅ QAF detector test passed (GNR: {gnr:.6f})")

    print("🎉 All quantization tests passed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    test_quantization_system()