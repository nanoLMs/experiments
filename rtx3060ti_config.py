#!/usr/bin/env python3
"""
NanoLM RTX 3060 TI Optimized Configuration
==========================================

Optimized configuration for:
- RTX 3060 TI (8GB VRAM)
- 32GB DDR4 RAM  
- Core i5 13600K
- BitNet 1.58-bit architecture
- Multi-platform export capability

Based on research insights from:
- BitNet b1.58: Ternary {-1, 0, 1} quantization
- T-MAC: CPU optimization for edge deployment
- HRM: Hierarchical reasoning with multi-timescale processing
- Enhanced export system for all target devices
"""

import torch
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import os


@dataclass
class RTX3060TIOptimizedConfig:
    """Optimized configuration for RTX 3060 TI system"""
    
    # ==========================================
    # SYSTEM SPECIFICATIONS
    # ==========================================
    gpu_name: str = "RTX 3060 TI"
    gpu_memory_gb: float = 8.0
    cpu_cores: int = 20  # i5-13600K (6P + 8E cores, 20 threads)
    ram_gb: float = 32.0
    
    # ==========================================
    # MEMORY OPTIMIZATION FOR 8GB VRAM
    # ==========================================
    max_vram_usage: float = 0.85  # 6.8GB usable
    max_ram_usage: float = 0.80   # 25.6GB usable
    
    # Batch sizes optimized for 8GB VRAM
    batch_size: int = 16           # Reduced for 8GB limit
    micro_batch_size: int = 4      # For gradient accumulation
    gradient_accumulation_steps: int = 4  # 16 effective batch size
    
    # Sequence length optimization
    max_sequence_length: int = 512   # Balanced for memory/performance
    block_size: int = 512
    
    # ==========================================
    # BITNET 1.58-BIT QUANTIZATION
    # ==========================================
    # Based on "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"
    use_bitnet_quantization: bool = True
    quantization_bits: str = "1.58-bit"  # Ternary {-1, 0, 1}
    quantize_activations: bool = False    # Keep activations in FP16/BF16
    bitnet_layer_norm_dtype: str = "fp32" # Keep LayerNorm in FP32
    
    # BitNet-specific optimizations
    use_gradient_checkpointing: bool = True
    optimize_for_inference: bool = True
    enable_bitnet_cuda_kernels: bool = True
    
    # ==========================================
    # MODEL ARCHITECTURE (OPTIMIZED FOR EDGE)
    # ==========================================
    # Small but efficient model for 8GB constraint
    vocab_size: int = 32000
    n_layer: int = 24              # Reduced from typical 32+ layers
    n_head: int = 16               # Optimized for RTX 3060 TI
    n_embd: int = 1024             # Balanced embedding size
    n_kv_heads: int = 8            # Grouped query attention
    
    # Estimated parameters: ~30M (much smaller due to 1.58-bit weights)
    
    # ==========================================
    # MIXTURE OF EXPERTS (MOE) - OPTIMIZED
    # ==========================================
    use_moe: bool = True
    moe_num_experts: int = 4       # Reduced for memory constraints
    moe_top_k: int = 2             # Standard for efficiency
    moe_freq: int = 2              # Every 2nd layer
    moe_capacity_factor: float = 1.25
    moe_drop_tokens: bool = True
    moe_aux_loss_coeff: float = 0.01
    
    # ==========================================
    # MULTI-TOKEN PREDICTION (MTP)
    # ==========================================
    use_mtp: bool = True
    mtp_num_tokens: int = 4        # Predict 4 tokens ahead
    mtp_loss_weight: float = 0.5
    mtp_layers: List[int] = field(default_factory=lambda: [12, 18, 24])  # Last few layers
    
    # ==========================================
    # HIERARCHICAL REASONING MODULE (HRM)
    # ==========================================
    use_hrm: bool = True
    hrm_num_layers: int = 2        # Lightweight for memory
    hrm_hidden_size: int = 512     # Reduced size
    hrm_num_reasoning_steps: int = 2
    hrm_convergence_threshold: float = 0.95
    hrm_loss_weight: float = 0.1
    
    # ==========================================
    # TRAINING OPTIMIZATION FOR RTX 3060 TI
    # ==========================================
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Mixed precision for memory efficiency
    use_mixed_precision: bool = True
    precision: str = "bf16"        # Better for RTX 3060 TI
    
    # Optimizer optimizations
    use_fused_adamw: bool = True
    use_cpu_offload: bool = False  # We have enough RAM
    
    # ==========================================
    # MEMORY MANAGEMENT
    # ==========================================
    pin_memory: bool = True
    num_workers: int = 8           # Half of CPU threads
    prefetch_factor: int = 2
    
    # Gradient accumulation strategy
    gradient_checkpointing: bool = True
    offload_optimizer: bool = False  # Keep in GPU for speed
    
    # Cache management
    max_cache_size_gb: float = 2.0
    cache_cleanup_threshold: float = 0.8
    
    # ==========================================
    # ANTI-HALLUCINATION
    # ==========================================
    use_anti_hallucination: bool = True
    forbidden_token_penalty: float = -10.0
    factual_consistency_weight: float = 0.25
    repetition_penalty: float = 1.1
    
    # ==========================================
    # TOKENIZER CONFIGURATION
    # ==========================================
    tokenizer_path: str = "./nanolm_tokenizer"
    vocab_size: int = 32000
    special_tokens_count: int = 100  # Legal domain tokens
    bos_token: str = "[BOS]"
    eos_token: str = "[EOS]"
    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    
    # ==========================================
    # DATASET CONFIGURATION
    # ==========================================
    train_data_path: str = "./legal_corpus.txt"
    val_data_path: str = "./validation_data.txt"
    max_train_samples: Optional[int] = None
    max_val_samples: int = 1000
    
    # ==========================================
    # CHECKPOINTING & SAVING
    # ==========================================
    save_dir: str = "./checkpoints"
    save_every: int = 500
    keep_last_n_checkpoints: int = 3
    save_optimizer_state: bool = False  # Save memory
    
    # ==========================================
    # VALIDATION & EVALUATION
    # ==========================================
    eval_every: int = 100
    eval_max_new_tokens: int = 256
    eval_temperature: float = 0.8
    eval_top_k: int = 50
    eval_top_p: float = 0.9
    
    # ==========================================
    # EXPORT CONFIGURATION
    # ==========================================
    auto_export_on_completion: bool = True
    export_formats: List[str] = field(default_factory=lambda: [
        "torchscript",    # Mobile/Edge deployment
        "onnx",           # Cross-platform inference
        "coreml",         # iOS/macOS optimization
        "tflite",         # Android/Mobile
        "openvino",       # Intel CPU optimization
        "huggingface"     # Ecosystem integration
    ])
    
    # Format-specific optimizations
    torchscript_optimize: bool = True
    onnx_opset_version: int = 11
    coreml_target_ios: str = "13.0"
    tflite_quantize: bool = True
    openvino_precision: str = "FP16"
    
    # ==========================================
    # MONITORING & LOGGING
    # ==========================================
    use_rich_logging: bool = True
    log_file: str = "nanolm_training.log"
    log_level: str = "INFO"
    
    # Performance monitoring
    monitor_gpu_memory: bool = True
    monitor_system_metrics: bool = True
    monitoring_interval: int = 10  # steps
    
    # Loss tracking
    track_component_losses: bool = True
    predict_final_loss: bool = True
    loss_smoothing_window: int = 50
    
    # ==========================================
    # DEPLOYMENT TARGETS
    # ==========================================
    target_devices: List[str] = field(default_factory=lambda: [
        "android",        # ARM/x86 Android devices
        "ios",           # iPhone/iPad
        "macos",         # Apple Silicon Mac
        "windows",       # x86_64 Windows
        "linux",         # x86_64/ARM64 Linux
        "raspberry_pi",  # ARM edge devices
        "jetson",        # NVIDIA edge
        "web_browser",   # WebAssembly
        "cloud_cpu",     # Cloud CPU instances
        "cloud_gpu"      # Cloud GPU instances
    ])
    
    # Device-specific optimizations
    mobile_model_size_mb: int = 100    # Target size for mobile
    edge_latency_ms: int = 50          # Target latency
    web_model_size_mb: int = 50        # Smaller for web
    
    # ==========================================
    # RESEARCH OPTIMIZATIONS
    # ==========================================
    # Based on T-MAC paper for CPU optimization
    use_lookup_tables: bool = True     # For 1-bit inference
    enable_cpu_optimization: bool = True
    
    # Based on BitVLA for multimodal capabilities
    multimodal_ready: bool = True
    vision_encoder_path: Optional[str] = None
    
    # Based on HRM paper
    use_multi_timescale_processing: bool = True
    hierarchical_attention: bool = True
    
    # ==========================================
    # ADVANCED FEATURES
    # ==========================================
    # Dynamic loss scaling for stability
    use_dynamic_loss_scaling: bool = True
    loss_scale_window: int = 1000
    
    # Curriculum learning
    use_curriculum_learning: bool = False
    curriculum_stages: int = 3
    
    # Knowledge distillation
    use_knowledge_distillation: bool = False
    teacher_model_path: Optional[str] = None
    distillation_temperature: float = 3.0
    
    # ==========================================
    # DEBUGGING & DEVELOPMENT
    # ==========================================
    debug_mode: bool = False
    profile_memory: bool = False
    trace_model: bool = False
    validate_gradients: bool = False
    
    # Testing
    dry_run: bool = False
    max_steps_debug: int = 100
    
    def __post_init__(self):
        """Validate and adjust configuration"""
        # Ensure memory constraints are respected
        if self.batch_size * self.micro_batch_size > 64:
            print("⚠️ Reducing batch size for 8GB VRAM constraint")
            self.batch_size = min(self.batch_size, 16)
            
        # Ensure gradient accumulation is properly set
        if self.batch_size % self.micro_batch_size != 0:
            self.gradient_accumulation_steps = self.batch_size // self.micro_batch_size
            
        # Adjust model size if too large
        estimated_params = (
            self.n_layer * self.n_embd * self.n_embd * 4 +  # Attention
            self.n_layer * self.n_embd * self.n_embd * 8    # FFN
        ) // 8  # 1.58-bit quantization saves ~8x memory
        
        if estimated_params > 50_000_000:  # 50M parameter limit
            print("⚠️ Model too large for efficient training, reducing size")
            self.n_layer = min(self.n_layer, 20)
            self.n_embd = min(self.n_embd, 896)
            
        # Ensure export directory exists
        os.makedirs(self.save_dir, exist_ok=True)
        
    def get_memory_usage_estimate(self) -> Dict[str, float]:
        """Estimate memory usage for validation"""
        # Rough estimates in GB
        model_memory = (self.n_layer * self.n_embd * self.n_embd * 8) / (8 * 1e9)  # 1.58-bit savings
        batch_memory = (self.batch_size * self.max_sequence_length * self.n_embd * 4) / 1e9
        optimizer_memory = model_memory * 2  # Adam states
        
        total_gpu = model_memory + batch_memory + optimizer_memory
        
        return {
            "model_gb": model_memory,
            "batch_gb": batch_memory, 
            "optimizer_gb": optimizer_memory,
            "total_gpu_gb": total_gpu,
            "gpu_utilization": total_gpu / self.gpu_memory_gb,
            "within_limits": total_gpu < (self.gpu_memory_gb * self.max_vram_usage)
        }
        
    def print_config_summary(self):
        """Print configuration summary"""
        memory_est = self.get_memory_usage_estimate()
        
        print("🚀 NanoLM RTX 3060 TI Configuration")
        print("=" * 50)
        print(f"GPU: {self.gpu_name} ({self.gpu_memory_gb}GB)")
        print(f"RAM: {self.ram_gb}GB DDR4")
        print(f"CPU: {self.cpu_cores} threads")
        print()
        print("📊 Model Configuration:")
        print(f"  Layers: {self.n_layer}")
        print(f"  Embedding: {self.n_embd}")
        print(f"  Heads: {self.n_head}")
        print(f"  Quantization: {self.quantization_bits}")
        print()
        print("💾 Memory Estimates:")
        print(f"  Model: {memory_est['model_gb']:.1f}GB")
        print(f"  Batch: {memory_est['batch_gb']:.1f}GB") 
        print(f"  Total GPU: {memory_est['total_gpu_gb']:.1f}GB")
        print(f"  GPU Utilization: {memory_est['gpu_utilization']:.1%}")
        print(f"  Within Limits: {'✅' if memory_est['within_limits'] else '❌'}")
        print()
        print("🎯 Training Settings:")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Micro Batch: {self.micro_batch_size}")
        print(f"  Grad Accumulation: {self.gradient_accumulation_steps}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Precision: {self.precision}")
        print()
        print("🔧 Advanced Features:")
        print(f"  MoE: {'✅' if self.use_moe else '❌'}")
        print(f"  MTP: {'✅' if self.use_mtp else '❌'}")
        print(f"  HRM: {'✅' if self.use_hrm else '❌'}")
        print(f"  Anti-Hallucination: {'✅' if self.use_anti_hallucination else '❌'}")
        print()
        print("📱 Export Targets:")
        for fmt in self.export_formats:
            print(f"  ✅ {fmt}")
        print()


# Create default configuration instance
default_config = RTX3060TIOptimizedConfig()

# Compatibility with existing code
class TrainConfig(RTX3060TIOptimizedConfig):
    """Backward compatibility alias"""
    pass

if __name__ == "__main__":
    # Test configuration
    config = RTX3060TIOptimizedConfig()
    config.print_config_summary()
    
    # Validate memory constraints
    memory_est = config.get_memory_usage_estimate()
    if not memory_est['within_limits']:
        print("❌ Configuration exceeds GPU memory limits!")
        print("Consider reducing batch_size, n_layer, or n_embd")
    else:
        print("✅ Configuration is optimized for RTX 3060 TI!")