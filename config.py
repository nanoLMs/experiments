from dataclasses import dataclass, field
from typing import Tuple, List, Optional    # ================================
    # TRAINING SCHEDULE - OPTIMIZED FOR MAXIMUM GPU UTILIZATION
    # ================================
@dataclass
class TrainConfig:

    seq_len: int = 512                 # REDUCED: Faster first step
    micro_batch_size: int = 2          # MINIMAL: Ensure first step completes quickly
    grad_accum_steps: int = 12         # INCREASED: Compensate for smaller micro batch
    num_epochs: int = 10               # REDUCED: Fine-tuning needs fewer epochs
    target_total_tokens: int = 500_000_000  # REDUCED: Fine-tuning phase

    lr: float = 1e-4                   # REDUCED: Lower LR for fine-tuning stability
    min_lr: float = 1e-5               # Proportionally lower
    warmup_steps: int = 100            # SHORTER: Already pre-trained
    weight_decay: float = 0.01         # LOWER weight decay for FP4
    betas: Tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0         # Gradient clipping

    # ================================
    # PERFORMANCE OPTIMIZATIONS - MAXIMUM GPU HUNGRY SETTINGS
    # ================================
    pin_memory: bool = True            # Pin memory for faster transfers
    num_workers: int = 2               # REDUCED: Avoid data loading bottleneck
    persistent_workers: bool = False   # DISABLED: Reduce memory pressure
    prefetch_factor: int = 2           # REDUCED: Less memory usage
    dataloader_drop_last: bool = True  # Drop incomplete batches

    # GPU Memory and Performance
    gradient_checkpointing: bool = True  # Enable for memory efficiency
    compile_model: bool = False        # DISABLED: Avoid first-step compilation delay
    use_fused_adam: bool = True        # Use fused AdamW for speed
    mixed_precision: str = "fp16"      # Use FP16 for non-quantized operations

    # Advanced Performance Settings
    cudnn_benchmark: bool = True       # Enable cuDNN benchmark for speed
    cuda_empty_cache_steps: int = 50   # Clear cache every N steps
    max_split_size_mb: int = 512       # Limit memory fragmentation TrainConfig:
    # ================================
    # DATA PATHS & FILES
    # ================================
    tokenizer_dir: str = "/home/swadhin/experiments/nanolm_tokenizer/hf_tokenizer"
    train_corpus: str = "/home/swadhin/experiments/nanolm_tokenizer/test_corpus.txt"
    ckpt_dir: str = "checkpoints"

    # ================================
    # MODEL ARCHITECTURE - OPTIMIZED FOR RTX 3060 Ti (8GB) WITH MEMORY CONSTRAINTS
    # ================================
    vocab_size: int = 32000
    n_layers: int = 8                  # FURTHER REDUCED: Better fit in 8GB GPU with FP4
    n_heads: int = 8                   # Kept at 8 for good attention
    d_model: int = 512                 # Kept at 512 for balance
    d_head: int = d_model // n_heads   # Auto-calculated = 64
    d_ff: int = 1024                   # FURTHER REDUCED: 2x expansion ratio for memory

    # ================================
    # MIXTURE OF EXPERTS (MoE) - OPTIMIZED FOR 8GB GPU
    # ================================
    moe_every: int = 6                 # Every 6th layer is MoE (less frequent for memory)
    n_experts: int = 4                 # REDUCED: Fewer experts for memory
    moe_top_k: int = 2
    expert_ff_mult: float = 0.75       # REDUCED SIZE: Further memory savings
    router_jitter: float = 0.01
    router_z_loss: float = 1e-4
    capacity_factor: float = 1.0       # FULL CAPACITY: Memory available with 4-bit
    moe_aux_weight: float = 0.02       # Load balance weight

    # ================================
    # REASONING HEADS - OPTIMIZED FOR MEMORY
    # ================================
    enable_reasoning: bool = True
    reasoning_layers: List[int] = field(default_factory=lambda: [8])  # Single reasoning layer
    reasoning_dim: int = 256           # REDUCED: Smaller reasoning dimension
    reasoning_loss_weight: float = 0.05

    # ================================
    # MULTI-TOKEN PREDICTION (MTP) - MEMORY OPTIMIZED
    # ================================
    mtp_heads: int = 3                 # Predict t+1, t+2, t+3 for better learning
    mtp_loss_weights: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])  # Decreasing weights

    # ================================
    # STAGE 2: FP4 FQT FINE-TUNING (CUTTING-EDGE PERFORMANCE)
    # ================================
    use_quantization: bool = True      # General quantization flag
    use_fp4: bool = False              # ❌ DISABLE fake 4-bit simulation
    use_bnb_4bit: bool = True         # ❌ DISABLE bitsandbytes (switch to FP4)
    use_pure_4bit: bool = False         # ✅ ENABLE FP4 FQT methodology

    # BITSANDBYTES NF4 CONFIGURATION (STAGE 1 - STABLE TRAINING):
    bnb_4bit_compute_dtype: str = "bfloat16"  # Compute in BF16 (fast + stable)
    bnb_4bit_quant_type: str = "nf4"          # NormalFloat4 - best 4-bit format
    bnb_4bit_use_double_quant: bool = True    # Extra compression
    bnb_4bit_quant_storage: str = "uint8"     # Storage format

    # FP4 FQT CONFIGURATION (STAGE 2 - FINE-TUNING):
    # - NVFP4 format: E2M1 data precision, E4M3 scale format
    # - Block size: 16 (optimal from research)
    # - Split rounding strategy: RtN forward, SR backward/update
    # - Automatic QAF phase for final convergence
    fp4_format: str = "nvfp4"                 # NVFP4 format (E2M1 + E4M3)
    fp4_block_size: int = 16                  # Block size for quantization (optimal)
    fp4_split_rounding: bool = True           # Split rounding strategy
    fp4_use_amp: bool = False                 # AMP not needed with FP4

    # QAF (Quantization-Aware Finetuning) Configuration:
    qaf_threshold: float = 1e-6               # Gradient stagnation threshold
    max_qaf_steps: int = 1000                 # Brief QAF phase duration
    qaf_precision: str = "bf16"               # Higher precision for QAF gradients

    # ================================
    # LoRA (Low-Rank Adaptation)
    # ================================
    lora: bool = True
    lora_rank: int = 8                 # Reduced for memory efficiency
    lora_alpha: int = 16               # Reduced from 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ("qkv", "out", "w1", "w2", "head", "mtp_")



    # ================================
    # MEMORY OPTIMIZATIONS
    # ================================
    gradient_checkpointing: bool = True     # RE-ENABLED: bitsandbytes is stable
    activation_checkpointing: bool = True   # RE-ENABLED: Memory savings
    compile_model: bool = False           # DISABLED: Avoiding dtype conflicts with 4-bit + compilation
    compile_mode: str = "default"         # Options: default, reduce-overhead, max-autotune
    compile_fullgraph: bool = False       # More stable compilation
    compile_dynamic: bool = True          # Handle variable sequence lengths
    pin_memory: bool = True
    amp: bool = True                      # Automatic Mixed Precision

    # ================================
    # REGULARIZATION
    # ================================
    dropout: float = 0.05
    attn_dropout: float = 0.05
    ff_dropout: float = 0.05

    # ================================
    # FLASHATTENTION
    # ================================
    use_flash_attn: bool = True
    flash_dropout_p: float = 0.0
    rope_base: int = 10000
    rope_scaling: float = 1.0

    # ================================
    # ANTI-HALLUCINATION
    # ================================
    factual_penalty_weight: float = 0.1
    forbidden_tokens: Optional[List[int]] = None
    forbidden_tokens_file: Optional[str] = None

    # ================================
    # CHECKPOINTING & MONITORING
    # ================================
    log_interval: int = 25             # Log every 25 steps
    eval_interval: int = 1000          # Evaluate every 1000 steps
    save_interval: int = 500           # Save every 500 steps
    keep_last_k: int = 5               # Keep last 5 checkpoints
    save_best: bool = True             # Save best model
    early_stopping_patience: int = 10000
    report_memory_interval: int = 250
    save_config_once: bool = True

    # ================================
    # DISTRIBUTED TRAINING - DISABLED FOR SINGLE GPU
    # ================================
    fsdp: bool = False                 # Disabled for single RTX 3060 Ti
    fsdp_wrap_layer_size: int = 1000000
    fsdp_cpu_offload: bool = True      # Offload to CPU when needed
    ddp: bool = False                  # Disabled for single GPU

    # ================================
    # DATA LOADING - OPTIMIZED FOR i5-13600K + 32GB RAM
    # ================================
    dataloader_workers: int = 16       # OPTIMAL: 80% of 20 logical CPUs (i5-13600K)
    pin_memory: bool = True             # Essential for GPU training
    persistent_workers: bool = True     # Keep workers alive between epochs
    prefetch_factor: int = 4            # Aggressive prefetching (32GB RAM allows it)
    drop_last: bool = True              # Consistent batch sizes
    max_dataset_samples: int = 1000000  # Increased dataset size (more RAM available)

    # ================================
    # EVALUATION & GENERATION
    # ================================
    eval_samples: int = 128
    generation_max_new_tokens: int = 128

    # ================================
    # MISC
    # ================================
    seed: int = 42

    def effective_batch_size(self, world_size: int) -> int:
        """Calculate effective batch size across all devices"""
        return self.micro_batch_size * self.grad_accum_steps * world_size


    # ================================
    # MULTI-TOKEN PREDICTION (MTP)
    # ================================
    mtp_k: int = 4                     # predict next K tokens at once (loss across shifts)

    # ================================
    # HIERARCHICAL REASONING (HRM-like)
    # ================================
    use_hrm: bool = True               # enable segmented 1-step-grad training
    segments: int = 2                  # M segments per batch item
    N_cycles: int = 2                  # HRM low/high cycles per segment
    T_steps: int = 2                   # high-level updates every T low-level steps
    use_act: bool = False              # optional adaptive segments

    # ================================
    # 4-bit NormalFloat (NF4) Quantization
    # ================================
    use_fp4: bool = True               # replace Linear with NF4 where safe
    fp4_compute_dtype: str = "fp16"    # compute dtype when using 4-bit weights
    fp4_skip_modules: tuple = ("lm_head",)  # keep output head in higher precision
