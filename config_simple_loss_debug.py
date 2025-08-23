from dataclasses import dataclass, field
from typing import Tuple, List, Optional

@dataclass
class SimpleLossDebugConfig:

    seq_len: int = 256
    micro_batch_size: int = 2
    grad_accum_steps: int = 4
    num_epochs: int = 1
    target_total_tokens: Optional[int] = None

    lr: float = 1e-4
    min_lr: float = 1e-5
    warmup_steps: int = 50
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.98)
    max_grad_norm: float = 1.0

    pin_memory: bool = True
    num_workers: int = 2
    persistent_workers: bool = False
    prefetch_factor: int = 2
    dataloader_drop_last: bool = True

    gradient_checkpointing: bool = False
    compile_model: bool = False
    use_fused_adam: bool = True
    mixed_precision: str = "bf16"

    cudnn_benchmark: bool = True
    cuda_empty_cache_steps: int = 50
    max_split_size_mb: int = 512

    tokenizer_dir: str = "/home/swadhin/experiments/lawdataset/legal_tokenizer/hf_tokenizer/"
    train_corpus: str = "/home/swadhin/experiments/lawdataset/legal_tokenizer/test_corpus.txt"
    ckpt_dir: str = "checkpoints_loss_debug"

    vocab_size: int = 32000
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    d_head: int = d_model // n_heads
    d_ff: int = d_model * 4

    moe_every: int = 0
    n_experts: int = 0
    moe_top_k: int = 0
    expert_ff_mult: float = 1.0
    router_jitter: float = 0.0
    router_z_loss: float = 0.0
    capacity_factor: float = 1.0
    moe_aux_weight: float = 0.0

    enable_reasoning: bool = False
    reasoning_layers: List[int] = field(default_factory=lambda: [])
    reasoning_dim: int = 256
    reasoning_loss_weight: float = 0.0

    mtp_heads: int = 0
    mtp_loss_weights: List[float] = field(default_factory=lambda: [])

    use_quantization: bool = False
    use_fp4: bool = False
    use_bnb_4bit: bool = False
    use_pure_4bit: bool = False

    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = False
    bnb_4bit_quant_storage: str = "uint8"

    bnb_4bit_quantize_heads: bool = False
    bnb_4bit_quantize_router: bool = False

    fp4_format: str = "nvfp4"
    fp4_block_size: int = 16
    fp4_split_rounding: bool = True
    fp4_use_amp: bool = False

    qaf_threshold: float = 1e-6
    max_qaf_steps: int = 0
    qaf_precision: str = "bf16"

    lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = ()

    activation_checkpointing: bool = False
    compile_mode: str = "default"
    compile_fullgraph: bool = False
    compile_dynamic: bool = True
    amp: bool = True

    dropout: float = 0.0
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0

    use_flash_attn: bool = True
    flash_dropout_p: float = 0.0
    rope_base: int = 10000
    rope_scaling: float = 1.0

    factual_penalty_weight: float = 0.0
    forbidden_tokens: Optional[List[int]] = None
    forbidden_tokens_file: Optional[str] = None

    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 200
    keep_last_k: int = 3
    save_best: bool = True
    early_stopping_patience: int = 1000
    report_memory_interval: int = 100
    save_config_once: bool = True

    fsdp: bool = False
    fsdp_wrap_layer_size: int = 1000000
    fsdp_cpu_offload: bool = False
    ddp: bool = False

    dataloader_workers: int = 2
    drop_last: bool = True
    max_dataset_samples: int = 100000

    eval_samples: int = 64
    generation_max_new_tokens: int = 64

    seed: int = 42

    def effective_batch_size(self, world_size: int) -> int:
        return self.micro_batch_size * self.grad_accum_steps * world_size

    mtp_k: int = 0

    use_hrm: bool = False
    segments: int = 0
    N_cycles: int = 0
    T_steps: int = 0
    use_act: bool = False