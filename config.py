from dataclasses import dataclass, field
from typing import Tuple, List, Optional

    # ================================
    # TRAINING SCHEDULE - OPTIMIZED FOR MAXIMUM GPU UTILIZATION
    # ================================
@dataclass
class TrainConfig:
    """Training and model configuration tuned for building a compact nanoLM

    - Defaults chosen to target a compact model (approx 100-150 MB when quantized)
    - Built-in support for MoE, MTP, hierarchical reasoning (HRM), anti-hallucination losses
    - NF4 (bitsandbytes) training enabled by default, FP4 options for fine-tuning
    """

    # -------------------- dataset / training schedule --------------------
    seq_len: int = 256                 # OPTIMAL: Good balance for convergence
    micro_batch_size: int = 8          # STABLE: Proven batch size
    grad_accum_steps: int = 4          # INCREASED: Better gradient estimation
    num_epochs: int = 10               # FULL: Complete training for convergence
    target_total_tokens: Optional[int] = None  # DISABLED: Train for full epochs without token limit

    lr: float = 2e-5                   # REDUCED: More stable learning rate for convergence
    min_lr: float = 1e-6               # Higher minimum for stability
    warmup_steps: int = 500           # INCREASED: Better warmup for stability
    weight_decay: float = 0.01          # INCREASED: Better regularization
    betas: Tuple[float, float] = (0.9, 0.999)  # Standard Adam betas
    max_grad_norm: float = 1.0         # INCREASED: Allow larger gradients for convergence

    # -------------------- performance / data loader --------------------
    pin_memory: bool = True            # Pin memory for faster transfers
    num_workers: int = 4               # REDUCED: Avoid data loading bottleneck
    dataloader_workers: int = 4        # backward-compatible alias used elsewhere
    persistent_workers: bool = True     # Keep workers alive between epochs
    prefetch_factor: int = 2           # Aggressive prefetching (32GB RAM allows it)
    dataloader_drop_last: bool = True  # Drop incomplete batches

    # GPU Memory and Performance
    gradient_checkpointing: bool = True  # Enable for memory efficiency
    activation_checkpointing: bool = True   # RE-ENABLED: Memory savings
    mixed_precision: str = "bf16"      # Use FP16 for non-quantized operations
    amp: bool = True                      # Automatic Mixed Precision
    compile_model: bool = False
    compile_backend: Optional[str] = None  # e.g., 'inductor', 'nvfuser' (fallback None)

    # -------------------- compact model architecture (defaults) --------------------
    # Defaults chosen to hit ~30M parameters FP32; with 4-bit storage it fits well under 150MB
    vocab_size: int = 29086
    n_layers: int = 16                # INCREASED: More layers for better learning
    n_heads: int = 8                   # INCREASED: More attention heads
    d_model: int = 384                 # INCREASED: Better representation capacity
    # d_head computed in validate() to keep consistent with n_heads/d_model
    d_head: Optional[int] = None
    d_ff: int = 1536                   # INCREASED: 2x expansion ratio

    tie_word_embeddings: bool = True

    # -------------------- MoE (Mixture of Experts) --------------------
    moe_every: int = 0                 # 0 = disabled; set to e.g. 2 or 4 to insert MoE layers
    n_experts: int = 4
    moe_top_k: int = 1                 # top-1 gating for efficient inference
    expert_ff_mult: float = 1.0      # expert FF multiplier (how large experts are)
    router_jitter: float = 0.01
    router_z_loss: float = 1e-4
    capacity_factor: float = 1.0
    moe_aux_weight: float = 0.01

    # -------------------- Multi-Token Prediction (MTP) --------------------
    mtp_k: int = 4                   # predict next K tokens

    mtp_loss_weights: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25, 0.125])

    # -------------------- Hierarchical Reasoning (HRM-like) --------------------
    use_hrm: bool = True
    hrm_segments: int = 2
    hrm_N_cycles: int = 2
    hrm_T_steps: int = 2

    # -------------------- reasoning / explanation head --------------------
    # Enable an optional reasoning / explanation aggregator head (lightweight)
    enable_reasoning: bool = True
    reasoning_dim: int = 256

    # -------------------- attention / flash attention --------------------
    use_flash_attn: bool = True
    attn_dropout: float = 0.0
    rope_base: int = 10000
    rope_scaling: float = 1.0

    # -------------------- anti-hallucination & losses --------------------
    factual_penalty_weight: float = 0.25
    unlikelihood_weight: float = 0.5
    contrastive_loss_weight: float = 0.1
    reasoning_loss_weight: float = 0.1
    forbidden_tokens_file: Optional[str] = None
    forbidden_tokens: Optional[List[int]] = None
    retrieval_augmentation: bool = False

    # -------------------- quantization/training formats --------------------
    use_quantization: bool = True
    use_bnb_4bit: bool = True        # train in NF4 with bitsandbytes
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_storage: str = "uint8"
    bnb_4bit_quantize_heads: bool = True
    bnb_4bit_quantize_router: bool = True

    # FP4 fine-tuning options (post-NF4 training)
    use_fp4: bool = True
    fp4_format: str = "nvfp4"
    fp4_block_size: int = 16
    fp4_split_rounding: bool = True
    qaf_threshold: float = 1e-6
    max_qaf_steps: int = 1000
    qaf_precision: str = "bf16"

    # -------------------- LoRA (optional) --------------------
    lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ()

    # -------------------- regularization and dropout --------------------
    dropout: float = 0.05
    ff_dropout: float = 0.05

    # -------------------- checkpointing / logging --------------------
    tokenizer_dir: str = "/home/swadhin/experiments/lawdataset/legal_tokenizer/hf_tokenizer/"
    train_corpus: str = "/home/swadhin/experiments/lawdataset/legal_tokenizer/test_corpus.txt"
    ckpt_dir: str = "checkpoints"

    log_interval: int = 10
    eval_interval: int = 1000
    save_interval: int = 1000
    keep_last_k: int = 5
    save_best: bool = True
    save_config_once: bool = True
    report_memory_interval: int = 250

    # Early stopping configuration
    early_stopping_patience: Optional[int] = None  # number of evaluations with no improvement
    early_stopping_min_delta: float = 0.0  # minimum change to qualify as improvement

    # -------------------- distributed / hardware --------------------
    fsdp: bool = False
    ddp: bool = False

    # -------------------- evaluation / generation --------------------
    eval_samples: int = 128
    generation_max_new_tokens: int = 128

    # -------------------- meta / targets --------------------
    target_model_size_mb: Tuple[int, int] = (100, 150)  # target range for edge deployment
    seed: int = 42

    # -------------------- derived / helper methods --------------------
    def effective_batch_size(self, world_size: int) -> int:
        return self.micro_batch_size * self.grad_accum_steps * world_size

    def validate(self):
        """Sanity-check and fill derived fields."""
        # compute d_head if missing
        if self.d_head is None:
            if self.n_heads > 0:
                self.d_head = max(1, self.d_model // self.n_heads)
            else:
                self.d_head = self.d_model

        # ensure mtp_k matches loss weights
        if len(self.mtp_loss_weights) < self.mtp_k:
            # pad with geometric decay if not provided
            base = self.mtp_loss_weights[0] if self.mtp_loss_weights else 1.0
            self.mtp_loss_weights = [base * (0.5 ** i) for i in range(self.mtp_k)]

        # disable expensive features automatically for tiny models
        if (self.n_layers * self.d_model) < 4000:
            # too small for MoE
            self.moe_every = 0

    def estimate_model_size_mb(self, quantized: bool = True) -> float:
        """Rough parameter count -> size estimation.

        This is a coarse estimator (ignores optimizer states, adapters, and small heads).
        When quantized==True we assume effective 4-bit storage (0.5 bytes/param) for bulk params.
        """
        # embeddings
        params = int(self.vocab_size) * int(self.d_model)
        # transformer rough params: per layer ~ 4*d_model*d_model (attn proj) + 2*d_model*d_ff (ff)
        per_layer = 4 * self.d_model * self.d_model + 2 * self.d_model * self.d_ff
        params += per_layer * self.n_layers
        # small head and norms
        params += 5 * self.d_model

        if quantized:
            bytes_per_param = 0.5  # 4-bit -> 0.5 bytes
        else:
            bytes_per_param = 4.0
        size_mb = params * bytes_per_param / (1024 * 1024)
        return float(size_mb)

    def to_dict(self) -> dict:
        self.validate()
        return {k: v for k, v in self.__dict__.items()}
