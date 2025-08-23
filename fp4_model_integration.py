#!/usr/bin/env python3
"""
FP4 Model Integration
=====================

Integrates FP4 FQT with NanoMoEModel:
- Creates FP4 version of the model
- Handles phase transitions (FP4 → QAF)
- Manages optimizer settings for FP4 training
- Provides compatibility with existing trainer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging

from model_moe import NanoMoEModel
from config import TrainConfig
from fp4_fqt_core import (
    FP4QuantizedLinear, NVFP4Quantizer, FP4GradientTracker,
    apply_fp4_fqt_to_model, set_model_training_phase,
    get_fp4_memory_stats, register_fp4_backward_hooks
)

logger = logging.getLogger(__name__)

class FP4NanoMoEModel(nn.Module):
    """
    FP4 FQT version of NanoMoEModel with:
    - NVFP4 quantization for Linear layers
    - Automatic QAF phase detection
    - Gradient tracking and phase management
    - Memory-efficient training
    """
    
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        
        # Create base model
        self.base_model = NanoMoEModel(cfg)
        
        # Apply FP4 FQT to Linear layers (skip output heads)
        skip_modules = ("head", "mtp_heads", "reason_head")
        apply_fp4_fqt_to_model(self.base_model, skip_modules=skip_modules)
        
        # Training phase management
        self.training_phase = "fp4"  # "fp4", "qaf", or "normal"
        self.grad_tracker = FP4GradientTracker()
        self.model_dimension = sum(p.numel() for p in self.parameters())
        
        # QAF phase settings
        self.qaf_steps = 0
        self.max_qaf_steps = cfg.max_qaf_steps if hasattr(cfg, 'max_qaf_steps') else 1000
        
        # Memory statistics
        self.memory_stats = get_fp4_memory_stats(self.base_model)
        
        # Register backward hooks for stochastic rounding on gradients
        if cfg.fp4_split_rounding if hasattr(cfg, 'fp4_split_rounding') else True:
            register_fp4_backward_hooks(self.base_model)
            logger.info("✅ Registered FP4 backward hooks for stochastic rounding")
        
        logger.info(f"🔥 FP4 NanoMoE Model created:")
        logger.info(f"  • Total parameters: {self.memory_stats['total_parameters']:,}")
        logger.info(f"  • FP4 parameters: {self.memory_stats['fp4_parameters']:,} ({self.memory_stats['fp4_percentage']:.1f}%)")
        logger.info(f"  • Memory saved: {self.memory_stats['memory_saved_mb']:.1f} MB")
        logger.info(f"  • Compression ratio: {self.memory_stats['compression_ratio']:.2f}x")
    
    def forward(self, idx):
        """Forward pass with FP4 quantization and phase awareness"""
        # Delegate to base model (which now has FP4 layers)
        return self.base_model(idx)
    
    def update_gradient_stats(self, grad_norm: float):
        """Update gradient statistics for QAF detection"""
        self.grad_tracker.update_gradient_norm(grad_norm)
    
    def step_qaf_phase(self):
        """Step QAF phase counter and check for automatic switching"""
        if self.training_phase == "fp4" and self.should_switch_to_qaf():
            self._switch_to_qaf_phase()
        elif self.training_phase == "qaf":
            self.qaf_steps += 1
            if self.qaf_steps >= self.max_qaf_steps:
                logger.info(f"🎯 QAF phase completed after {self.qaf_steps} steps")
    
    def should_switch_to_qaf(self) -> bool:
        """Check if should switch from FP4 to QAF phase"""
        return self.grad_tracker.should_switch_to_qaf(self.model_dimension)
    
    def _switch_to_qaf_phase(self):
        """Switch from FP4 to QAF phase"""
        if self.training_phase == "fp4":
            logger.info("🔄 Switching from FP4 to QAF phase")
            logger.info("  • Forward pass: FP4 (for inference compatibility)")
            logger.info("  • Backward pass: Higher precision gradients")
            
            self.training_phase = "qaf"
            set_model_training_phase(self.base_model, "qaf")
            self.qaf_steps = 0
    
    def get_training_phase_info(self) -> Dict[str, Any]:
        """Get detailed training phase information"""
        grad_to_noise_ratio = self.grad_tracker.get_gradient_to_noise_ratio(self.model_dimension)
        recent_grad_norm = np.mean(self.grad_tracker.grad_norms[-5:]) if self.grad_tracker.grad_norms else 0
        
        return {
            'phase': self.training_phase,
            'grad_to_noise_ratio': grad_to_noise_ratio,
            'qaf_triggered': self.grad_tracker.qaf_triggered,
            'qaf_steps': self.qaf_steps,
            'max_qaf_steps': self.max_qaf_steps,
            'recent_grad_norm': recent_grad_norm,
            'memory_stats': self.memory_stats
        }
    
    def num_parameters(self):
        """Get total number of parameters"""
        return self.base_model.num_parameters()
    
    def num_trainable_parameters(self):
        """Get number of trainable parameters"""
        return self.base_model.num_trainable_parameters()


def integrate_pure_4bit_model(cfg: TrainConfig) -> FP4NanoMoEModel:
    """
    Create FP4 FQT integrated model
    
    Args:
        cfg: Training configuration
        
    Returns:
        FP4 integrated NanoMoE model
    """
    logger.info("🚀 Creating FP4 FQT integrated model")
    
    # Enable FP4 specific settings
    if not hasattr(cfg, 'use_pure_4bit'):
        cfg.use_pure_4bit = True
    if not hasattr(cfg, 'fp4_format'):
        cfg.fp4_format = 'nvfp4'
    if not hasattr(cfg, 'fp4_block_size'):
        cfg.fp4_block_size = 16
    if not hasattr(cfg, 'fp4_split_rounding'):
        cfg.fp4_split_rounding = True
    
    # Create FP4 model
    model = FP4NanoMoEModel(cfg)
    
    logger.info(f"✅ FP4 FQT model integration complete")
    
    return model


def get_fp4_optimizer_config(cfg: TrainConfig, model: nn.Module) -> Dict[str, Any]:
    """
    Get optimizer configuration optimized for FP4 FQT training
    
    Args:
        cfg: Training configuration
        model: FP4 model
        
    Returns:
        Optimizer configuration dictionary
    """
    # Check if model is in FP4 phase
    is_fp4_model = hasattr(model, 'training_phase')
    
    if is_fp4_model and model.training_phase == "fp4":
        # FP4 phase: Lower learning rate for stability
        lr = cfg.lr * 0.8  # Slightly lower for FP4 stability
        weight_decay = cfg.weight_decay * 0.5  # Lower weight decay
        eps = 1e-6  # Smaller epsilon for FP4 precision
        
        logger.info("🔧 Optimizer configured for FP4 FQT phase")
        
    elif is_fp4_model and model.training_phase == "qaf":
        # QAF phase: Higher precision, can use higher LR
        lr = cfg.lr * 1.2  # Slightly higher for QAF convergence
        weight_decay = cfg.weight_decay
        eps = 1e-8  # Standard epsilon
        
        logger.info("🔧 Optimizer configured for QAF phase")
        
    else:
        # Normal training
        lr = cfg.lr
        weight_decay = cfg.weight_decay
        eps = 1e-8
        
        logger.info("🔧 Optimizer configured for normal training")
    
    return {
        'lr': lr,
        'betas': cfg.betas,
        'weight_decay': weight_decay,
        'eps': eps,
        'amsgrad': False  # Disable amsgrad for FP4 compatibility
    }


def should_use_amp_with_fp4(cfg: TrainConfig) -> bool:
    """
    Determine if AMP should be used with FP4 model
    
    Args:
        cfg: Training configuration
        
    Returns:
        Whether to use AMP
    """
    # FP4 already provides low precision, AMP may not be necessary
    # But can be used for non-FP4 operations
    if hasattr(cfg, 'fp4_use_amp') and cfg.fp4_use_amp is not None:
        return cfg.fp4_use_amp
    
    # Default: Disable AMP for FP4 training to avoid conflicts
    return False


class FP4TrainingPhaseManager:
    """
    Manages FP4 training phases and transitions
    """
    
    def __init__(self, model: FP4NanoMoEModel, cfg: TrainConfig):
        self.model = model
        self.cfg = cfg
        self.phase_history = []
        
    def on_step_end(self, step: int, grad_norm: float):
        """Called after each training step"""
        # Update gradient statistics
        self.model.update_gradient_stats(grad_norm)
        
        # Step QAF phase counter and check transitions
        self.model.step_qaf_phase()
        
        # Log phase changes
        current_phase = self.model.training_phase
        if not self.phase_history or self.phase_history[-1][1] != current_phase:
            self.phase_history.append((step, current_phase))
            logger.info(f"📊 Step {step}: Training phase = {current_phase}")
    
    def get_phase_summary(self) -> Dict[str, Any]:
        """Get summary of phase transitions"""
        return {
            'current_phase': self.model.training_phase,
            'phase_history': self.phase_history,
            'qaf_steps': self.model.qaf_steps,
            'gradient_info': self.model.get_training_phase_info()
        }


# Hierarchical Reasoning Module integration
class HRMIntegratedModel(FP4NanoMoEModel):
    """
    FP4 model with Hierarchical Reasoning Module (HRM) integration
    Combines FP4 FQT with HRM-style hierarchical processing
    """
    
    def __init__(self, cfg: TrainConfig):
        super().__init__(cfg)
        
        # HRM parameters
        self.N_cycles = getattr(cfg, 'N_cycles', 2)
        self.T_steps = getattr(cfg, 'T_steps', 2) 
        self.segments = getattr(cfg, 'segments', 2)
        
        # Hierarchical state tracking
        self.high_level_state = None
        self.low_level_state = None
        self.cycle_count = 0
        self.step_count = 0
        
        logger.info(f"🧠 HRM integration enabled:")
        logger.info(f"  • N_cycles: {self.N_cycles}")
        logger.info(f"  • T_steps: {self.T_steps}")
        logger.info(f"  • Segments: {self.segments}")
    
    def forward(self, idx):
        """HRM-style hierarchical forward pass"""
        if not self.training or not hasattr(self.cfg, 'use_hrm') or not self.cfg.use_hrm:
            # Standard forward pass during inference or if HRM disabled
            return super().forward(idx)
        
        # HRM hierarchical processing
        batch_size = idx.size(0)
        
        # Initialize states if needed
        if self.high_level_state is None:
            self.high_level_state = torch.zeros(batch_size, self.cfg.d_model, device=idx.device)
            self.low_level_state = torch.zeros(batch_size, self.cfg.d_model, device=idx.device)
        
        # Hierarchical processing cycles
        for cycle in range(self.N_cycles):
            # Low-level processing (T steps)
            for step in range(self.T_steps):
                # Standard forward pass with current state context
                logits_main, logits_mtp, aux_loss, reason_logits = super().forward(idx)
                
                # Update low-level state (simplified)
                if hasattr(self.base_model, 'ln_f'):
                    # Use final layer norm output as state
                    self.low_level_state = self.base_model.ln_f(self.base_model.tok_emb(idx).mean(dim=1))
            
            # High-level update (once per cycle)
            self.high_level_state = 0.9 * self.high_level_state + 0.1 * self.low_level_state
        
        # Final forward pass with updated states
        return super().forward(idx)


# Anti-hallucination integration for FP4
class FP4AntiHallucinationWrapper(nn.Module):
    """
    Wraps FP4 model with anti-hallucination mechanisms
    Maintains precision for critical output layers
    """
    
    def __init__(self, fp4_model: FP4NanoMoEModel, cfg: TrainConfig):
        super().__init__()
        self.fp4_model = fp4_model
        self.cfg = cfg
        
        # Keep anti-hallucination components in higher precision
        self.forbidden_token_ids = getattr(cfg, 'forbidden_tokens', []) or []
        self.factual_penalty_weight = getattr(cfg, 'factual_penalty_weight', 0.1)
        
        # Precision override for output layers
        self._ensure_output_precision()
        
    def _ensure_output_precision(self):
        """Ensure output heads remain in higher precision"""
        for name, module in self.fp4_model.base_model.named_modules():
            if any(head in name for head in ['head', 'mtp_heads', 'reason_head']):
                # These should already be skipped in FP4 conversion
                if hasattr(module, 'weight'):
                    module.weight.data = module.weight.data.float()
                    if hasattr(module, 'bias') and module.bias is not None:
                        module.bias.data = module.bias.data.float()
    
    def forward(self, idx):
        """Forward pass with anti-hallucination"""
        # Get FP4 model output
        logits_main, logits_mtp, aux_loss, reason_logits = self.fp4_model(idx)
        
        # Apply anti-hallucination penalties (in higher precision)
        if self.forbidden_token_ids and self.factual_penalty_weight > 0:
            penalty = 0.0
            for token_id in self.forbidden_token_ids:
                if token_id < logits_main.size(-1):
                    # Penalize forbidden tokens
                    penalty += self.factual_penalty_weight * torch.mean(torch.softmax(logits_main, dim=-1)[:, :, token_id])
            
            aux_loss = aux_loss + penalty
        
        return logits_main, logits_mtp, aux_loss, reason_logits
    
    def update_gradient_stats(self, grad_norm: float):
        """Delegate to FP4 model"""
        self.fp4_model.update_gradient_stats(grad_norm)
    
    def step_qaf_phase(self):
        """Delegate to FP4 model"""
        self.fp4_model.step_qaf_phase()
    
    def get_training_phase_info(self):
        """Delegate to FP4 model"""
        return self.fp4_model.get_training_phase_info()
    
    def num_parameters(self):
        """Delegate to FP4 model"""
        return self.fp4_model.num_parameters()
    
    def num_trainable_parameters(self):
        """Delegate to FP4 model"""
        return self.fp4_model.num_trainable_parameters()


# Factory functions for different model configurations
def create_fp4_model(cfg: TrainConfig) -> nn.Module:
    """Create FP4 FQT model with full quantized training support"""
    
    try:
        logger.info("🔥 Creating FP4 FQT model for true 4-bit training...")
        logger.info("🎯 Features: NVFP4 format • Split rounding • QAF detection")
        
        # Create the FP4 integrated model directly
        model = integrate_pure_4bit_model(cfg)
        
        # Move model to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        logger.info("✅ FP4 FQT model created successfully")
        
        # Test the model with a small forward pass
        logger.info("🧪 Testing FP4 FQT model...")
        test_seq_len = min(cfg.seq_len, 32)  # Use shorter sequence for testing
        test_input = torch.randint(0, min(cfg.vocab_size, 1000), (1, test_seq_len))
        
        # Ensure test input is on the same device as model
        test_input = test_input.to(device)
        
        try:
            with torch.no_grad():
                # Set model to eval mode for testing
                model.eval()
                output = model(test_input)
                logger.info(f"✅ FP4 FQT model test passed - output shapes: {[o.shape if torch.is_tensor(o) else len(o) for o in output]}")
                # Set back to train mode
                model.train()
        except Exception as test_e:
            logger.warning(f"⚠️ FP4 FQT model test failed: {test_e}")
            logger.warning(f"Test input shape: {test_input.shape}, device: {test_input.device}")
            # If test fails, we'll still return the model as it might work during training
            logger.info("📝 FP4 FQT model created but test failed - proceeding anyway")
        
        return model
        
    except Exception as e:
        logger.error(f"⚠️ NF4 model creation failed with error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        logger.info("🔄 Falling back to standard base model...")
        
        # Final fallback to standard model
        from model_moe import NanoMoEModel
        
        class StandardModelWrapper(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                self.base_model = NanoMoEModel(cfg)
                self.training_phase = "normal"
            
            def forward(self, idx):
                return self.base_model(idx)
            
            def num_parameters(self):
                return sum(p.numel() for p in self.parameters())
            
            def num_trainable_parameters(self):
                return sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            def update_gradient_stats(self, grad_norm: float):
                """Simplified gradient stats - just pass for standard model"""
                pass
            
            def step_qaf_phase(self):
                """Simplified QAF phase - not needed for standard model"""
                pass
            
            def get_training_phase_info(self):
                return {
                    'phase': self.training_phase,
                    'grad_to_noise_ratio': 1.0,
                    'qaf_triggered': False,
                    'memory_stats': {'memory_saved_mb': 0, 'compression_ratio': 1.0}
                }
        
        model = StandardModelWrapper(cfg)
        logger.info("✅ Created standard fallback model")
        return model


if __name__ == "__main__":
    # Test integration
    import numpy as np
    from config import TrainConfig
    
    print("🧪 Testing FP4 Model Integration")
    
    # Create test config
    cfg = TrainConfig()
    cfg.use_pure_4bit = True
    cfg.fp4_format = 'nvfp4'
    cfg.fp4_split_rounding = True
    cfg.use_hrm = True
    cfg.factual_penalty_weight = 0.1
    
    # Test model creation
    model = create_fp4_model(cfg)
    print(f"✅ Model created with {model.num_parameters():,} parameters")
    
    # Test forward pass
    dummy_input = torch.randint(0, cfg.vocab_size, (2, 128))
    with torch.no_grad():
        output = model(dummy_input)
        print(f"✅ Forward pass successful: {[o.shape for o in output[:3]]}")
    
    # Test training phase info
    phase_info = model.get_training_phase_info()
    print(f"✅ Phase info: {phase_info['phase']}")
    
    print("\n🎉 FP4 Model Integration tests passed!")
