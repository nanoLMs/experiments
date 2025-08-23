
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

@dataclass
class SimpleTrainConfig:
    """A simplified configuration for baseline training."""

    # ================================
    # TRAINING SCHEDULE
    # ================================
    seq_len: int = 256
    micro_batch_size: int = 4
    grad_accum_steps: int = 8
    num_epochs: int = 3
    lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    betas: Tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0

    # ================================
    # DATA & TOKENIZER
    # ================================
    tokenizer_dir: str = "/home/swadhin/experiments/lawdataset/legal_tokenizer/hf_tokenizer/"
    train_corpus: str = "/home/swadhin/experiments/lawdataset/legal_tokenizer/test_corpus.txt"
    ckpt_dir: str = "checkpoints_simple"

    # ================================
    # MODEL ARCHITECTURE (SIMPLE)
    # ================================
    vocab_size: int = 32000
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    d_head: int = d_model // n_heads
    d_ff: int = d_model * 4

    # ================================
    # DISABLED FEATURES
    # ================================
    moe_every: int = 0  # 0 means no MoE
    n_experts: int = 0
    expert_ff_mult: float = 1.0
    moe_top_k: int = 0
    router_jitter: float = 0.0
    capacity_factor: float = 1.0
    router_z_loss: float = 0.0
    moe_aux_weight: float = 0.0
    mtp_heads: int = 0
    enable_reasoning: bool = False
    use_hrm: bool = False
    use_quantization: bool = False
    lora: bool = False

    # ================================
    # PERFORMANCE
    # ================================
    use_flash_attn: bool = True
    gradient_checkpointing: bool = False # Disabled for simplicity
    compile_model: bool = False
    mixed_precision: str = "bf16"

    # ================================
    # LOGGING & SAVING
    # ================================
    log_interval: int = 10
    eval_interval: int = 250
    save_interval: int = 500

    # ================================
    # ATTENTION & DROPOUT
    # ================================
    rope_base: int = 10000
    rope_scaling: float = 1.0
    flash_dropout_p: float = 0.0
    dropout: float = 0.05
    attn_dropout: float = 0.05
    ff_dropout: float = 0.05

    # ================================
    # MISC
    # ================================
    seed: int = 42

    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.grad_accum_steps
