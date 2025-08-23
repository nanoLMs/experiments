#!/usr/bin/env python3
"""
Enhanced Configuration System for Advanced NanoLM
================================================

Comprehensive configuration management with validation, logging, and monitoring.
Extends the existing config.py with additional infrastructure features.
"""

import os
import json
import logging
import warnings
from dataclasses import dataclass, field, asdict
from typing import Tuple, List, Optional, Dict, Any, Union
from pathlib import Path
import torch

# Import the existing config
from config import TrainConfig

@dataclass
class InfrastructureConfig:
    """Infrastructure and monitoring configuration"""

    # Logging Configuration
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = "training.log"
    console_logging: bool = True
    rich_logging: bool = True  # Use rich for better console output

    # Monitoring Configuration
    enable_loss_tracking: bool = True
    loss_tracking_dir: str = "loss_tracking"
    enable_memory_monitoring: bool = True
    enable_gpu_monitoring: bool = True
    monitoring_interval: int = 100  # steps

    # Validation Configuration
    validate_config_on_init: bool = True
    strict_validation: bool = True
    warn_on_suboptimal: bool = True

    # Export Configuration
    auto_export_on_completion: bool = True
    export_formats: List[str] = field(default_factory=lambda: [
        "torchscript", "onnx", "huggingface"
    ])
    export_dir: str = "exported_models"

    # Error Handling Configuration
    enable_auto_recovery: bool = True
    max_recovery_attempts: int = 3
    recovery_strategies: List[str] = field(default_factory=lambda: [
        "reduce_batch_size", "clear_cache", "gradient_clipping"
    ])

    # Performance Configuration
    enable_profiling: bool = False
    profile_dir: str = "profiling"
    benchmark_mode: bool = False

    # Development Configuration
    debug_mode: bool = False
    save_debug_info: bool = False
    debug_dir: str = "debug"


@dataclass
class EnhancedTrainConfig(TrainConfig):
    """Enhanced training configuration with infrastructure features"""

    # Infrastructure settings
    infrastructure: InfrastructureConfig = field(default_factory=InfrastructureConfig)

    # Enhanced validation settings
    validate_data_paths: bool = True
    validate_model_size: bool = True
    validate_memory_requirements: bool = True

    # Advanced quantization settings (extending existing)
    quantization_validation: bool = True
    quantization_quality_threshold: float = 0.95  # Minimum quality retention

    # Enhanced MoE settings (extending existing)
    moe_load_balancing_weight: float = 0.01
    moe_expert_dropout: float = 0.1
    moe_capacity_factor_train: float = 1.25
    moe_capacity_factor_eval: float = 2.0

    # Enhanced MTP settings (extending existing)
    mtp_temperature: float = 1.0
    mtp_top_k: int = 50
    mtp_top_p: float = 0.9

    # Enhanced HRM settings (extending existing)
    hrm_gradient_checkpointing: bool = True
    hrm_memory_efficient: bool = True
    hrm_convergence_threshold: float = 0.95
    hrm_N_cycles: int = 2
    hrm_T_steps: int = 2
    hrm_use_gradient_approx: bool = True
    hrm_min_convergence_steps: int = 10
    hrm_loss_weight: float = 0.1
    hrm_reset_every: int = 100
    bnb_4bit_quantize_hrm: bool = False

    # Loss function enhancements
    loss_scaling_factor: float = 10.0  # For FP4 underflow prevention
    adaptive_loss_scaling: bool = True
    loss_smoothing: float = 0.1

    # Multi-component loss weights
    main_loss_weight: float = 1.0
    mtp_loss_weight: float = 0.5
    auxiliary_loss_weight: float = 0.01
    reasoning_loss_weight: float = 0.1
    anti_hallucination_loss_weight: float = 0.25

    # Dynamic loss adjustment
    dynamic_loss_weighting: bool = False
    loss_weight_adjustment_interval: int = 100

    # Anti-hallucination settings
    forbidden_tokens: List[int] = field(default_factory=list)
    enable_unlikelihood_training: bool = True
    enable_contrastive_loss: bool = True

    # Loss tracking and prediction settings
    convergence_window: int = 50
    convergence_threshold: float = 1e-4
    plateau_patience: int = 100
    target_loss: float = 2.0
    early_stopping_patience: int = 200
    min_improvement: float = 1e-3

    # Loss prediction settings
    enable_loss_prediction: bool = True
    prediction_methods: List[str] = field(default_factory=lambda: [
        "exponential_decay", "power_law", "trend_analysis", "learning_curve"
    ])
    ensemble_prediction: bool = True
    loss_visualization: bool = True

    # Anti-hallucination settings
    enable_anti_hallucination: bool = True
    filtering_policy: str = "moderate"  # strict, moderate, lenient, adaptive
    hallucination_confidence_threshold: float = 0.5
    enable_factual_checking: bool = True
    enable_repetition_detection: bool = True
    factual_confidence_threshold: float = 0.7

    # Forbidden token filtering
    enable_forbidden_token_filtering: bool = True
    forbidden_token_sets: List[str] = field(default_factory=lambda: [
        "profanity", "hate_speech", "violence", "illegal_activities",
        "medical_misinformation", "conspiracy_theories", "personal_info"
    ])
    custom_forbidden_tokens: List[int] = field(default_factory=list)
    custom_forbidden_patterns: List[str] = field(default_factory=list)

    # Context-aware filtering
    enable_context_filtering: bool = True
    context_window_size: int = 512
    adaptive_filtering_strength: float = 1.0

    # Unlikelihood training settings
    enable_unlikelihood_training: bool = True
    unlikelihood_alpha: float = 1.0
    sequence_level_weight: float = 0.5
    unlikelihood_context_window: int = 50
    min_sequence_length: int = 3
    use_adaptive_alpha: bool = True
    forbidden_threshold: float = 0.1

    # Contrastive loss settings
    enable_contrastive_loss: bool = True
    contrastive_temperature: float = 0.07
    contrastive_margin: float = 0.5
    negative_samples: int = 5
    use_hard_negatives: bool = True
    factual_weight: float = 1.0
    semantic_weight: float = 0.5

    # Combined anti-hallucination training
    loss_balance_weight: float = 0.5  # Balance between unlikelihood and contrastive

    # Training stability
    gradient_accumulation_dtype: str = "float32"  # Accumulate in higher precision
    optimizer_eps: float = 1e-8
    optimizer_weight_decay_exclude: List[str] = field(default_factory=lambda: [
        "bias", "LayerNorm.weight", "layernorm.weight"
    ])

    def validate(self):
        """Enhanced validation with comprehensive checks"""
        # Call parent validation
        super().validate()

        # Infrastructure validation
        if self.infrastructure.validate_config_on_init:
            self._validate_infrastructure()

        # Path validation
        if self.validate_data_paths:
            self._validate_paths()

        # Model size validation
        if self.validate_model_size:
            self._validate_model_size()

        # Memory requirements validation
        if self.validate_memory_requirements:
            self._validate_memory_requirements()

        # Quantization validation
        if self.quantization_validation:
            self._validate_quantization_settings()

        # Feature compatibility validation
        self._validate_feature_compatibility()

        # Performance warnings
        if self.infrastructure.warn_on_suboptimal:
            self._check_suboptimal_settings()

    def _validate_infrastructure(self):
        """Validate infrastructure settings"""
        # Check log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        if self.infrastructure.log_level not in valid_log_levels:
            raise ValueError(f"Invalid log_level: {self.infrastructure.log_level}. "
                           f"Must be one of {valid_log_levels}")

        # Check export formats
        valid_formats = ["torchscript", "onnx", "huggingface", "coreml", "tflite"]
        for fmt in self.infrastructure.export_formats:
            if fmt not in valid_formats:
                warnings.warn(f"Unknown export format: {fmt}")

        # Check recovery strategies
        valid_strategies = ["reduce_batch_size", "clear_cache", "gradient_clipping",
                          "reduce_sequence_length", "disable_features"]
        for strategy in self.infrastructure.recovery_strategies:
            if strategy not in valid_strategies:
                warnings.warn(f"Unknown recovery strategy: {strategy}")

    def _validate_paths(self):
        """Validate file and directory paths"""
        # Check tokenizer directory
        if not os.path.exists(self.tokenizer_dir):
            raise FileNotFoundError(f"Tokenizer directory not found: {self.tokenizer_dir}")

        # Check training corpus
        if not os.path.exists(self.train_corpus):
            raise FileNotFoundError(f"Training corpus not found: {self.train_corpus}")

        # Create output directories if they don't exist
        for dir_path in [self.ckpt_dir, self.infrastructure.loss_tracking_dir,
                        self.infrastructure.export_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def _validate_model_size(self):
        """Validate model size constraints"""
        estimated_size = self.estimate_model_size_mb(quantized=True)
        target_min, target_max = self.target_model_size_mb

        if estimated_size > target_max:
            if self.infrastructure.strict_validation:
                raise ValueError(f"Estimated model size ({estimated_size:.1f}MB) exceeds "
                               f"target maximum ({target_max}MB)")
            else:
                warnings.warn(f"Model size ({estimated_size:.1f}MB) may exceed target "
                            f"({target_max}MB)")

        if estimated_size < target_min:
            warnings.warn(f"Model size ({estimated_size:.1f}MB) is below target minimum "
                        f"({target_min}MB). Consider increasing model capacity.")

    def _validate_memory_requirements(self):
        """Validate memory requirements against available GPU memory"""
        if not torch.cuda.is_available():
            warnings.warn("CUDA not available - cannot validate GPU memory requirements")
            return

        # Estimate memory requirements
        estimated_params = self._estimate_parameter_count()

        # Rough memory estimation (very approximate)
        if self.use_quantization:
            param_memory = estimated_params * 0.5 / (1024**3)  # 4-bit = 0.5 bytes per param
        else:
            param_memory = estimated_params * 4 / (1024**3)    # FP32 = 4 bytes per param

        # Add overhead for activations, gradients, optimizer states
        total_memory = param_memory * 4  # Rough multiplier

        # Check against available GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        if total_memory > gpu_memory * 0.9:  # Leave 10% headroom
            if self.infrastructure.strict_validation:
                raise RuntimeError(f"Estimated memory requirement ({total_memory:.1f}GB) "
                                 f"exceeds available GPU memory ({gpu_memory:.1f}GB)")
            else:
                warnings.warn(f"High memory usage expected ({total_memory:.1f}GB / "
                            f"{gpu_memory:.1f}GB available)")

    def _validate_quantization_settings(self):
        """Validate quantization configuration"""
        if self.use_quantization:
            if self.use_bnb_4bit and self.bnb_4bit_quant_type not in ["nf4", "fp4"]:
                raise ValueError(f"Invalid quantization type: {self.bnb_4bit_quant_type}")

            if self.use_fp4 and self.fp4_format not in ["nvfp4", "e4m3", "e5m2"]:
                raise ValueError(f"Invalid FP4 format: {self.fp4_format}")

            if self.fp4_block_size not in [8, 16, 32, 64]:
                warnings.warn(f"Unusual FP4 block size: {self.fp4_block_size}. "
                            f"Recommended: 16")

    def _validate_feature_compatibility(self):
        """Validate feature combinations for compatibility"""
        # MoE + Quantization compatibility
        if self.moe_every > 0 and self.use_quantization:
            if not self.bnb_4bit_quantize_router:
                warnings.warn("MoE with quantization should quantize routers for consistency")

        # MTP + HRM compatibility
        if self.mtp_k > 4 and self.use_hrm:
            warnings.warn("High MTP_K with HRM may increase memory usage significantly")

        # Flash attention compatibility
        if self.use_flash_attn and self.seq_len > 2048:
            warnings.warn("Flash attention with very long sequences may have issues")

        # Gradient checkpointing + quantization
        if self.gradient_checkpointing and self.use_bnb_4bit:
            warnings.warn("Gradient checkpointing with bitsandbytes may cause issues")

    def _check_suboptimal_settings(self):
        """Check for suboptimal configuration settings"""
        # Batch size warnings
        effective_batch = self.micro_batch_size * self.grad_accum_steps
        if effective_batch < 16:
            warnings.warn(f"Small effective batch size ({effective_batch}) may slow convergence")

        # Learning rate warnings
        if self.lr > 1e-3:
            warnings.warn(f"High learning rate ({self.lr}) may cause instability")
        if self.lr < 1e-6:
            warnings.warn(f"Very low learning rate ({self.lr}) may slow convergence")

        # Sequence length warnings
        if self.seq_len < 128:
            warnings.warn(f"Short sequence length ({self.seq_len}) may limit model capability")
        if self.seq_len > 1024 and not self.use_flash_attn:
            warnings.warn(f"Long sequences without flash attention may be slow")

        # Quantization warnings
        if self.use_quantization and self.lr > 5e-4:
            warnings.warn("High learning rate with quantization may cause instability")

    def _estimate_parameter_count(self) -> int:
        """Estimate total parameter count"""
        # Embedding parameters
        params = self.vocab_size * self.d_model

        # Transformer layer parameters
        # Attention: 4 * d_model * d_model (Q, K, V, O projections)
        # Feed-forward: 2 * d_model * d_ff
        per_layer = 4 * self.d_model * self.d_model + 2 * self.d_model * self.d_ff

        # MoE adjustment
        if self.moe_every > 0:
            moe_layers = self.n_layers // self.moe_every
            regular_layers = self.n_layers - moe_layers

            # MoE layers have n_experts * expert_size instead of single FF
            moe_ff_params = self.n_experts * self.d_model * int(self.d_ff * self.expert_ff_mult)
            regular_ff_params = self.d_model * self.d_ff

            moe_layer_params = 4 * self.d_model * self.d_model + 2 * moe_ff_params
            regular_layer_params = 4 * self.d_model * self.d_model + 2 * regular_ff_params

            params += moe_layers * moe_layer_params + regular_layers * regular_layer_params
        else:
            params += per_layer * self.n_layers

        # Layer norms and other small components
        params += self.n_layers * 2 * self.d_model  # 2 layer norms per layer
        params += self.d_model  # Final layer norm

        # Output head (if not tied)
        if not self.tie_word_embeddings:
            params += self.vocab_size * self.d_model

        # MTP heads
        params += self.mtp_k * self.d_model * self.vocab_size

        # Reasoning head
        if self.enable_reasoning:
            params += self.d_model * self.reasoning_dim + self.reasoning_dim * self.vocab_size

        return params

    def get_memory_estimate(self) -> Dict[str, float]:
        """Get detailed memory usage estimates"""
        params = self._estimate_parameter_count()

        # Parameter memory
        if self.use_quantization:
            param_memory = params * 0.5  # 4-bit
        else:
            param_memory = params * 4    # FP32

        # Activation memory (rough estimate)
        batch_size = self.micro_batch_size
        activation_memory = batch_size * self.seq_len * self.d_model * self.n_layers * 4

        # Gradient memory (same as parameters if not quantized)
        if self.use_quantization:
            gradient_memory = params * 2  # FP16 gradients
        else:
            gradient_memory = params * 4  # FP32 gradients

        # Optimizer state memory (AdamW has 2 states per parameter)
        optimizer_memory = params * 8  # 2 * FP32

        # Convert to MB
        return {
            "parameters_mb": param_memory / (1024**2),
            "activations_mb": activation_memory / (1024**2),
            "gradients_mb": gradient_memory / (1024**2),
            "optimizer_mb": optimizer_memory / (1024**2),
            "total_mb": (param_memory + activation_memory + gradient_memory + optimizer_memory) / (1024**2)
        }

    def save_config(self, path: str):
        """Save configuration to JSON file"""
        config_dict = asdict(self)

        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj

        config_dict = convert_paths(config_dict)

        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"✅ Configuration saved to {path}")

    def load_config(self, path: str):
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)

        # Update current config with loaded values
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Re-validate after loading
        self.validate()
        print(f"✅ Configuration loaded from {path}")

    def print_summary(self):
        """Print configuration summary"""
        memory_est = self.get_memory_estimate()
        model_size = self.estimate_model_size_mb(quantized=True)

        print("🔧 Enhanced NanoLM Configuration Summary")
        print("=" * 50)
        print(f"Model Architecture:")
        print(f"  • Layers: {self.n_layers}")
        print(f"  • Heads: {self.n_heads}")
        print(f"  • Hidden Size: {self.d_model}")
        print(f"  • FF Size: {self.d_ff}")
        print(f"  • Sequence Length: {self.seq_len}")
        print(f"  • Vocabulary Size: {self.vocab_size:,}")

        print(f"\nAdvanced Features:")
        print(f"  • MoE: {'Enabled' if self.moe_every > 0 else 'Disabled'}")
        if self.moe_every > 0:
            print(f"    - Every {self.moe_every} layers, {self.n_experts} experts")
        print(f"  • MTP: Enabled (K={self.mtp_k})")
        print(f"  • HRM: {'Enabled' if self.use_hrm else 'Disabled'}")
        print(f"  • Anti-Hallucination: Enabled")

        print(f"\nQuantization:")
        print(f"  • NF4 Training: {'Enabled' if self.use_bnb_4bit else 'Disabled'}")
        print(f"  • FP4 Fine-tuning: {'Enabled' if self.use_fp4 else 'Disabled'}")
        if self.use_fp4:
            print(f"    - Format: {self.fp4_format}, Block Size: {self.fp4_block_size}")

        print(f"\nTraining Configuration:")
        print(f"  • Learning Rate: {self.lr}")
        print(f"  • Batch Size: {self.micro_batch_size} (effective: {self.effective_batch_size(1)})")
        print(f"  • Epochs: {self.num_epochs}")
        print(f"  • Gradient Clipping: {self.max_grad_norm}")

        print(f"\nMemory Estimates:")
        print(f"  • Model Size: {model_size:.1f} MB")
        print(f"  • Parameter Memory: {memory_est['parameters_mb']:.1f} MB")
        print(f"  • Total Training Memory: {memory_est['total_mb']:.1f} MB")

        print(f"\nInfrastructure:")
        print(f"  • Loss Tracking: {'Enabled' if self.infrastructure.enable_loss_tracking else 'Disabled'}")
        print(f"  • Auto Export: {'Enabled' if self.infrastructure.auto_export_on_completion else 'Disabled'}")
        print(f"  • Auto Recovery: {'Enabled' if self.infrastructure.enable_auto_recovery else 'Disabled'}")
        print(f"  • Debug Mode: {'Enabled' if self.infrastructure.debug_mode else 'Disabled'}")


def setup_logging(config: EnhancedTrainConfig):
    """Setup logging based on configuration"""
    # Configure logging level
    log_level = getattr(logging, config.infrastructure.log_level.upper())

    # Create formatters
    formatter = logging.Formatter(config.infrastructure.log_format)

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    if config.infrastructure.console_logging:
        if config.infrastructure.rich_logging:
            try:
                from rich.logging import RichHandler
                console_handler = RichHandler(rich_tracebacks=True)
            except ImportError:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
        else:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

        console_handler.setLevel(log_level)
        root_logger.addHandler(console_handler)

    # File handler
    if config.infrastructure.log_file:
        file_handler = logging.FileHandler(config.infrastructure.log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

    # Suppress some noisy loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    logging.info("✅ Logging system initialized")


def create_default_config() -> EnhancedTrainConfig:
    """Create default enhanced configuration"""
    config = EnhancedTrainConfig()

    # Enable key features for advanced nanoLM
    config.moe_every = 2  # Enable MoE every 2 layers
    config.use_hrm = True
    config.enable_reasoning = True

    # Optimize for loss convergence
    config.lr = 2e-5
    config.weight_decay = 0.01
    config.max_grad_norm = 1.0

    # Enable infrastructure features
    config.infrastructure.enable_loss_tracking = True
    config.infrastructure.enable_memory_monitoring = True
    config.infrastructure.auto_export_on_completion = True

    return config


if __name__ == "__main__":
    # Demo usage
    print("🧪 Enhanced Configuration System Demo")

    # Create and validate config
    config = create_default_config()
    config.validate()

    # Print summary
    config.print_summary()

    # Setup logging
    setup_logging(config)

    # Save config
    config.save_config("demo_config.json")

    print("✅ Configuration system demo completed!")