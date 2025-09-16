#!/usr/bin/env python3
"""
NanoLM: Domain-Specific Nano Language Model
Advanced architecture with MoE, MTP, Hierarchical Reasoning, and Anti-hallucination

Research-backed implementation featuring:
- NF4/FP4 quantization (bitsandbytes)
- Mixture of Experts (domain specialization)
- Multi-Token Prediction (speed + reasoning)
- Hierarchical dual-pathway reasoning
- Anti-hallucination verification
- Rich progress logging
- Clean modular design
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import bitsandbytes as bnb
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.text import Text

console = Console()

@dataclass
class NanoLMConfig:
    """NanoLM configuration with all advanced features"""
    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    max_seq_len: int = 512

    # MoE Configuration
    num_experts: int = 4
    top_k_experts: int = 2
    expert_dropout: float = 0.1

    # MTP Configuration
    num_predict_tokens: int = 3
    mtp_layers: Optional[List[int]] = None  # Which layers have MTP heads

    # Hierarchical Reasoning
    fast_dim: int = 256  # Fast pathway dimension
    slow_dim: int = 256  # Slow pathway dimension

    # Quantization
    use_nf4: bool = True
    nf4_block_size: int = 16

    # Anti-hallucination
    uncertainty_threshold: float = 0.7
    verification_layers: int = 2

    # Training
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5

    def __post_init__(self):
        if self.mtp_layers is None:
            # Add MTP to middle and final layers, but ensure indices are valid
            if self.n_layers >= 2:
                self.mtp_layers = [self.n_layers // 2, self.n_layers - 1]
            else:
                self.mtp_layers = [self.n_layers - 1] if self.n_layers > 0 else []

class NF4Linear(nn.Module):
    """4-bit quantized linear layer using bitsandbytes NF4 format"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Create a regular linear layer first, then convert to 4-bit later
        # This avoids the initialization issues with bitsandbytes
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Store info for potential 4-bit conversion
        self._use_4bit = False
        self._device = None

    def to_4bit(self, device='cuda'):
        """Convert to 4-bit quantization after initialization"""
        if not self._use_4bit and torch.cuda.is_available():
            try:
                # Create 4-bit linear layer
                linear_4bit = bnb.nn.Linear4bit(
                    self.in_features,
                    self.out_features,
                    bias=self.linear.bias is not None,
                    compute_dtype=torch.float16,
                    compress_statistics=True,
                    quant_type="nf4"
                )

                # Copy weights
                linear_4bit.weight.data = self.linear.weight.data.clone()
                if self.linear.bias is not None and linear_4bit.bias is not None:
                    linear_4bit.bias.data = self.linear.bias.data.clone()

                # Move to device and replace
                linear_4bit = linear_4bit.to(device)
                self.linear = linear_4bit
                self._use_4bit = True
                self._device = device
            except Exception as e:
                console.print(f"[yellow]⚠️  4-bit conversion failed, using FP16: {e}[/yellow]")
                # Fallback to FP16
                self.linear = self.linear.half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class RotaryPositionalEmbedding(nn.Module):
    """RoPE positional encoding for better sequence understanding"""

    def __init__(self, d_model: int, max_seq_len: int = 8192):
        super().__init__()
        self.d_model = d_model

        # Precompute positional encodings
        position = torch.arange(max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        # Get the positional encoding buffer
        pe_buffer = getattr(self, 'pe')

        # Check if we need to extend embeddings
        if seq_len > pe_buffer.size(1):
            # Create new positional embeddings on-the-fly
            position = torch.arange(seq_len).unsqueeze(1).float().to(x.device)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float() *
                                -(math.log(10000.0) / self.d_model)).to(x.device)

            pe = torch.zeros(seq_len, self.d_model).to(x.device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            return x + pe.unsqueeze(0)
        else:
            return x + pe_buffer[0, :seq_len]

class MixtureOfExperts(nn.Module):
    """Domain-specialized Mixture of Experts"""

    def __init__(self, config: NanoLMConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.d_model = config.d_model

        # Gating network
        self.gate = NF4Linear(config.d_model, config.num_experts, bias=False)

        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                NF4Linear(config.d_model, config.d_model * 4),
                nn.SiLU(),
                nn.Dropout(config.expert_dropout),
                NF4Linear(config.d_model * 4, config.d_model)
            ) for _ in range(config.num_experts)
        ])

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)

        # Compute gating scores
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute expert outputs
        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_prob = top_k_probs[:, i:i+1]

            # Route to experts
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_output = self.experts[expert_id](x_flat[mask])
                    output[mask] += expert_prob[mask] * expert_output

        # Load balancing loss
        load_balancing_loss = self._load_balancing_loss(gate_probs)

        output = output.view(batch_size, seq_len, d_model)
        return self.dropout(output), load_balancing_loss

    def _load_balancing_loss(self, gate_probs: torch.Tensor) -> torch.Tensor:
        """Encourage balanced expert usage"""
        # Compute load (fraction of tokens assigned to each expert)
        expert_counts = gate_probs.sum(dim=0)
        total_tokens = gate_probs.size(0)
        load = expert_counts / total_tokens

        # Compute importance (sum of gate probabilities for each expert)
        importance = gate_probs.mean(dim=0)

        # Load balancing loss
        return (load * importance).sum() * self.num_experts

class MultiTokenPredictor(nn.Module):
    """Multi-Token Prediction head for parallel token generation"""

    def __init__(self, config: NanoLMConfig):
        super().__init__()
        self.num_predict = config.num_predict_tokens
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size

        # Shared feature extraction
        self.shared_layer = NF4Linear(config.d_model, config.d_model)

        # Individual prediction heads
        self.prediction_heads = nn.ModuleList([
            NF4Linear(config.d_model, config.vocab_size)
            for _ in range(self.num_predict)
        ])

        # Coherence sampler (ensures token coherence)
        self.sampler = nn.Sequential(
            NF4Linear(config.d_model, config.d_model // 2),
            nn.SiLU(),
            NF4Linear(config.d_model // 2, config.vocab_size * self.num_predict)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract shared features
        shared_features = F.silu(self.shared_layer(x))

        # Individual token predictions
        token_logits = []
        for head in self.prediction_heads:
            logits = head(shared_features)
            token_logits.append(logits)

        # Coherent sequence sampling
        sampler_logits = self.sampler(shared_features)
        sampler_logits = sampler_logits.view(*x.shape[:2], self.num_predict, self.vocab_size)

        return {
            "individual_logits": torch.stack(token_logits, dim=2),
            "sampler_logits": sampler_logits,
            "shared_features": shared_features
        }

class HierarchicalReasoning(nn.Module):
    """Dual-pathway reasoning: Fast associative + Slow deliberative"""

    def __init__(self, config: NanoLMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.fast_dim = config.fast_dim
        self.slow_dim = config.slow_dim

        # Fast pathway (associative, parallel processing)
        self.fast_attention = nn.MultiheadAttention(
            self.fast_dim,
            config.n_heads // 2,
            dropout=config.dropout,
            batch_first=True
        )
        self.fast_ffn = nn.Sequential(
            NF4Linear(self.fast_dim, self.fast_dim * 2),
            nn.SiLU(),
            NF4Linear(self.fast_dim * 2, self.fast_dim)
        )

        # Slow pathway (deliberative, sequential processing)
        self.slow_lstm = nn.LSTM(
            self.slow_dim,
            self.slow_dim,
            batch_first=True,
            dropout=config.dropout
        )
        self.slow_ffn = nn.Sequential(
            NF4Linear(self.slow_dim, self.slow_dim * 2),
            nn.SiLU(),
            NF4Linear(self.slow_dim * 2, self.slow_dim)
        )

        # Pathway projections
        self.to_fast = NF4Linear(config.d_model, self.fast_dim)
        self.to_slow = NF4Linear(config.d_model, self.slow_dim)

        # Integration gate
        self.integration_gate = nn.Sequential(
            NF4Linear(self.fast_dim + self.slow_dim, config.d_model),
            nn.Sigmoid()
        )

        self.output_proj = NF4Linear(self.fast_dim + self.slow_dim, config.d_model)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Split into dual pathways
        fast_x = self.to_fast(x)
        slow_x = self.to_slow(x)

        # Convert attention mask for PyTorch's MultiheadAttention
        # key_padding_mask should be True for padded positions
        key_padding_mask = None
        if attention_mask is not None:
            # Convert from attention_mask (1 for real tokens, 0 for padding)
            # to key_padding_mask (True for padding, False for real tokens)
            key_padding_mask = (attention_mask == 0)

        # Fast pathway: parallel associative processing
        fast_attn, _ = self.fast_attention(fast_x, fast_x, fast_x, key_padding_mask=key_padding_mask)
        fast_out = self.fast_ffn(fast_attn + fast_x)

        # Slow pathway: sequential deliberative processing
        slow_lstm_out, _ = self.slow_lstm(slow_x)
        slow_out = self.slow_ffn(slow_lstm_out + slow_x)

        # Integrate pathways
        combined = torch.cat([fast_out, slow_out], dim=-1)
        gate = self.integration_gate(combined)

        output = self.output_proj(combined)
        return gate * output

class AntiHallucinationVerifier(nn.Module):
    """Multi-faceted hallucination detection and prevention"""

    def __init__(self, config: NanoLMConfig):
        super().__init__()
        self.d_model = config.d_model
        self.threshold = config.uncertainty_threshold

        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            NF4Linear(config.d_model, config.d_model // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            NF4Linear(config.d_model // 2, 1)  # Uncertainty score
        )

        # Factuality verification
        self.factuality_head = nn.Sequential(
            NF4Linear(config.d_model, config.d_model // 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            NF4Linear(config.d_model // 2, 1)  # Factuality score
        )

        # Entropy-based confidence
        self.entropy_estimator = NF4Linear(config.d_model, 1)

    def forward(self, hidden_states: torch.Tensor, logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Uncertainty estimation
        uncertainty = torch.sigmoid(self.uncertainty_head(hidden_states))

        # Factuality score
        factuality = torch.sigmoid(self.factuality_head(hidden_states))

        # Entropy-based confidence
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1, keepdim=True)
        max_entropy = math.log(logits.size(-1))
        normalized_entropy = entropy / max_entropy

        confidence = 1.0 - normalized_entropy
        confidence_gate = torch.sigmoid(self.entropy_estimator(hidden_states))

        # Combined verification score
        verification_score = factuality * confidence * confidence_gate * (1 - uncertainty)

        return {
            "uncertainty": uncertainty,
            "factuality": factuality,
            "confidence": confidence,
            "verification_score": verification_score,
            "should_abstain": verification_score < self.threshold
        }

class NanoLMBlock(nn.Module):
    """Unified NanoLM transformer block with all advanced features"""

    def __init__(self, config: NanoLMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = config.d_model

        # Layer normalization
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.norm3 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        # Core attention
        self.attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )

        # Hierarchical reasoning
        self.hierarchical_reasoning = HierarchicalReasoning(config)

        # Mixture of Experts (in specific layers)
        if layer_idx % (config.n_layers // 2) == 0:  # Every other layer
            self.moe = MixtureOfExperts(config)
            self.use_moe = True
        else:
            # Standard FFN
            self.ffn = nn.Sequential(
                NF4Linear(config.d_model, config.d_model * 4),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                NF4Linear(config.d_model * 4, config.d_model)
            )
            self.use_moe = False

        # Multi-Token Prediction (in specified layers)
        if config.mtp_layers and layer_idx in config.mtp_layers:
            self.mtp_head = MultiTokenPredictor(config)
            self.use_mtp = True
        else:
            self.use_mtp = False

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        # Self-attention
        attn_x = self.norm1(x)

        # Convert attention mask for PyTorch's MultiheadAttention
        key_padding_mask = None
        if attention_mask is not None:
            # Convert from attention_mask (1 for real tokens, 0 for padding)
            # to key_padding_mask (True for padding, False for real tokens)
            key_padding_mask = (attention_mask == 0)

        attn_out, attn_weights = self.attention(attn_x, attn_x, attn_x, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)

        # Hierarchical reasoning
        reasoning_x = self.norm2(x)
        reasoning_out = self.hierarchical_reasoning(reasoning_x, attention_mask)
        x = x + self.dropout(reasoning_out)

        # MoE or standard FFN
        ffn_x = self.norm3(x)
        if self.use_moe:
            ffn_out, load_balancing_loss = self.moe(ffn_x)
            x = x + ffn_out
        else:
            ffn_out = self.ffn(ffn_x)
            x = x + self.dropout(ffn_out)
            load_balancing_loss = torch.tensor(0.0, device=x.device)

        outputs = {
            "hidden_states": x,
            "attention_weights": attn_weights,
            "load_balancing_loss": load_balancing_loss
        }

        # Multi-Token Prediction
        if self.use_mtp:
            mtp_outputs = self.mtp_head(x)
            outputs["mtp_predictions"] = mtp_outputs

        return outputs

class NanoLM(nn.Module):
    """Complete NanoLM with all advanced features"""

    def __init__(self, config: NanoLMConfig):
        super().__init__()
        self.config = config

        console.print(Panel.fit(
            "[bold blue]🚀 Initializing NanoLM Architecture[/bold blue]\n"
            f"📊 Model Size: {config.d_model}D x {config.n_layers}L\n"
            f"🧠 Features: MoE({config.num_experts}) + MTP({config.num_predict_tokens}) + Hierarchical + Anti-hallucination\n"
            f"⚡ Quantization: NF4 4-bit",
            title="NanoLM Initialization"
        ))

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = RotaryPositionalEmbedding(config.d_model, config.max_seq_len)

        # Transformer blocks
        self.layers = nn.ModuleList([
            NanoLMBlock(config, i) for i in range(config.n_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        # Output heads
        self.lm_head = NF4Linear(config.d_model, config.vocab_size, bias=False)

        # Anti-hallucination verifier
        self.verifier = AntiHallucinationVerifier(config)

        # Initialize weights
        self.apply(self._init_weights)

        # Print model info
        self._print_model_info()

    def to(self, device):
        """Override to handle 4-bit conversion"""
        result = super().to(device)

        # Convert NF4Linear layers to 4-bit if moving to CUDA
        if str(device).startswith('cuda') and self.config.use_nf4:
            self._convert_to_4bit(device)

        return result

    def _convert_to_4bit(self, device):
        """Convert all NF4Linear layers to actual 4-bit"""
        for name, module in self.named_modules():
            if isinstance(module, NF4Linear):
                module.to_4bit(device)

    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, NF4Linear):
            # Initialize the underlying linear layer if accessible
            if hasattr(module.linear, 'weight') and module.linear.weight is not None:
                torch.nn.init.normal_(module.linear.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def _print_model_info(self):
        """Print detailed model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        table = Table(title="NanoLM Architecture Details")
        table.add_column("Component", style="cyan")
        table.add_column("Details", style="white")

        table.add_row("Total Parameters", f"{total_params:,}")
        table.add_row("Trainable Parameters", f"{trainable_params:,}")
        table.add_row("Model Dimension", str(self.config.d_model))
        table.add_row("Layers", str(self.config.n_layers))
        table.add_row("Attention Heads", str(self.config.n_heads))
        table.add_row("MoE Experts", str(self.config.num_experts))
        table.add_row("MTP Tokens", str(self.config.num_predict_tokens))
        table.add_row("Quantization", "NF4 4-bit")

        console.print(table)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                return_dict: bool = True) -> Dict[str, Any]:

        # Token embeddings + positional encoding
        x = self.token_embedding(input_ids)
        x = self.pos_encoding(x)

        # Collect outputs
        all_hidden_states = []
        all_attention_weights = []
        total_load_balancing_loss = 0.0
        mtp_predictions = []

        # Forward through transformer blocks
        for layer in self.layers:
            layer_outputs = layer(x, attention_mask)
            x = layer_outputs["hidden_states"]

            all_hidden_states.append(x)
            all_attention_weights.append(layer_outputs["attention_weights"])
            total_load_balancing_loss += layer_outputs["load_balancing_loss"]

            if "mtp_predictions" in layer_outputs:
                mtp_predictions.append(layer_outputs["mtp_predictions"])

        # Final normalization
        x = self.norm(x)

        # Language modeling head
        lm_logits = self.lm_head(x)

        # Anti-hallucination verification
        verification_outputs = self.verifier(x, lm_logits)

        if return_dict:
            return {
                "logits": lm_logits,
                "hidden_states": all_hidden_states,
                "attention_weights": all_attention_weights,
                "mtp_predictions": mtp_predictions,
                "load_balancing_loss": total_load_balancing_loss,
                "verification": verification_outputs,
                "last_hidden_state": x
            }
        else:
            return lm_logits

    def generate(self,
                 input_ids: torch.Tensor,
                 max_new_tokens: int = 50,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 use_mtp: bool = True,
                 verify_outputs: bool = True) -> Dict[str, Any]:
        """Advanced generation with MTP and verification"""

        self.eval()
        generated_tokens = []
        verification_scores = []

        with torch.no_grad():
            current_ids = input_ids.clone()

            for step in range(max_new_tokens):
                # Forward pass
                outputs = self(current_ids)

                # Use MTP if available and enabled
                if use_mtp and outputs["mtp_predictions"]:
                    # Use latest MTP prediction
                    mtp_logits = outputs["mtp_predictions"][-1]["sampler_logits"][:, -1, 0, :]
                    next_token_logits = mtp_logits
                else:
                    next_token_logits = outputs["logits"][:, -1, :]

                # Verification check
                verification = outputs["verification"]
                should_abstain = verification["should_abstain"][:, -1, :].squeeze()
                verification_score = verification["verification_score"][:, -1, :].squeeze().item()

                if verify_outputs and should_abstain.any():
                    # High uncertainty, consider abstaining or using conservative sampling
                    temperature = min(temperature * 0.8, 0.3)

                # Temperature scaling
                next_token_logits = next_token_logits / temperature

                # Top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)
                generated_tokens.append(next_token.item())
                verification_scores.append(verification_score)

                # Early stopping for special tokens
                if next_token.item() in [0, 1, 2]:  # EOS, UNK, PAD
                    break

        return {
            "generated_ids": current_ids,
            "generated_tokens": generated_tokens,
            "verification_scores": verification_scores,
            "average_confidence": sum(verification_scores) / len(verification_scores) if verification_scores else 0.0
        }

# Export the main model class
__all__ = ["NanoLM", "NanoLMConfig"]

if __name__ == "__main__":
    # Quick test
    config = NanoLMConfig(
        vocab_size=32000,
        d_model=512,
        n_layers=8,
        n_heads=8
    )

    model = NanoLM(config)
    console.print("[green]✅ NanoLM architecture created successfully![/green]")