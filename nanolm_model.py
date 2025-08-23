#!/usr/bin/env python3
"""
Advanced NanoLM Model Architecture
=================================

Implements a lightweight transformer model with:
- Quantized layers (NF4/FP4 support)
- Flash attention optimization
- RoPE positional encoding
- Layer normalization
- Configurable architecture for edge deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_func = None

try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    Linear4bit = None

from advanced_quantization import QuantizationController


@dataclass
class ModelOutput:
    """Model output structure"""
    logits: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None
    attention_weights: Optional[List[torch.Tensor]] = None
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    loss: Optional[torch.Tensor] = None


class RoPEPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE)

    Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    More efficient than absolute positional embeddings for long sequences.
    """

    def __init__(self, d_model: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.base = base

        # Pre-compute frequency matrix
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

        # Pre-compute cos and sin for efficiency
        self._update_cos_sin_cache(max_seq_len)

    def _update_cos_sin_cache(self, seq_len: int):
        """Update cached cos/sin values"""
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len

        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embedding"""
        if seq_len is None:
            seq_len = x.shape[-2]

        if seq_len > self.max_seq_len:
            self._update_cos_sin_cache(seq_len)

        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors"""
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class QuantizedLinear(nn.Module):
    """
    Quantization-aware linear layer

    Automatically uses quantized implementation when available,
    falls back to standard linear layer otherwise.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 quantization_config: Optional[Dict] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quantization_config = quantization_config or {}

        # Use quantized layer if available and enabled
        if (BNB_AVAILABLE and
            self.quantization_config.get('use_bnb_4bit', False)):

            self.linear = Linear4bit(
                input_features=in_features,
                output_features=out_features,
                bias=bias,
                compute_dtype=getattr(torch, self.quantization_config.get('compute_dtype', 'bfloat16')),
                compress_statistics=self.quantization_config.get('use_double_quant', True),
                quant_type=self.quantization_config.get('quant_type', 'nf4')
            )
            self.is_quantized = True
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
            self.is_quantized = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, quantized={self.is_quantized}'


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with optional Flash Attention

    Features:
    - Flash attention for memory efficiency
    - RoPE positional encoding
    - Configurable attention dropout
    - KV caching support
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_head or (config.d_model // config.n_heads)
        self.dropout = config.attn_dropout

        assert self.d_model == self.n_heads * self.d_head, "d_model must equal n_heads * d_head"

        # Quantization config for linear layers
        quant_config = {
            'use_bnb_4bit': config.use_bnb_4bit,
            'compute_dtype': config.bnb_4bit_compute_dtype,
            'use_double_quant': config.bnb_4bit_use_double_quant,
            'quant_type': config.bnb_4bit_quant_type
        }

        # Query, Key, Value projections
        self.q_proj = QuantizedLinear(self.d_model, self.d_model, bias=False, quantization_config=quant_config)
        self.k_proj = QuantizedLinear(self.d_model, self.d_model, bias=False, quantization_config=quant_config)
        self.v_proj = QuantizedLinear(self.d_model, self.d_model, bias=False, quantization_config=quant_config)
        self.o_proj = QuantizedLinear(self.d_model, self.d_model, bias=False, quantization_config=quant_config)

        # RoPE positional encoding
        self.rope = RoPEPositionalEncoding(
            d_model=self.d_head,
            max_seq_len=config.seq_len * 2,  # Allow for longer sequences
            base=config.rope_base
        )

        # Attention dropout
        self.attn_dropout = nn.Dropout(self.dropout)

        # Flash attention availability
        self.use_flash_attn = config.use_flash_attn and FLASH_ATTN_AVAILABLE

        if self.use_flash_attn:
            logging.info("✅ Using Flash Attention")
        else:
            logging.info("ℹ️ Using standard attention (Flash Attention not available)")

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(q, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle past key values for caching
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        # Store for next iteration if using cache
        if use_cache:
            present_key_value = (k, v)
        else:
            present_key_value = None

        # Compute attention
        if self.use_flash_attn and attention_mask is None:
            # Use Flash Attention (more efficient)
            # Reshape for flash attention: (batch, seq_len, n_heads, d_head)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            attn_output = flash_attn_func(q, k, v, dropout_p=self.dropout if self.training else 0.0)
            attn_weights = None  # Flash attention doesn't return weights
        else:
            # Standard attention
            attn_output, attn_weights = self._standard_attention(q, k, v, attention_mask)

        # Reshape and project output
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, present_key_value

    def _standard_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard scaled dot-product attention"""
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Transpose back to (batch, seq_len, n_heads, d_head)
        attn_output = attn_output.transpose(1, 2)

        return attn_output, attn_weights


class FeedForward(nn.Module):
    """
    Feed-forward network with quantization support

    Uses SwiGLU activation (from PaLM paper) for better performance.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.dropout = config.ff_dropout

        # Quantization config
        quant_config = {
            'use_bnb_4bit': config.use_bnb_4bit,
            'compute_dtype': config.bnb_4bit_compute_dtype,
            'use_double_quant': config.bnb_4bit_use_double_quant,
            'quant_type': config.bnb_4bit_quant_type
        }

        # SwiGLU requires 3 linear layers: gate, up, down
        self.gate_proj = QuantizedLinear(self.d_model, self.d_ff, bias=False, quantization_config=quant_config)
        self.up_proj = QuantizedLinear(self.d_model, self.d_ff, bias=False, quantization_config=quant_config)
        self.down_proj = QuantizedLinear(self.d_ff, self.d_model, bias=False, quantization_config=quant_config)

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation: swish(gate) * up
        gate = F.silu(self.gate_proj(x))  # SiLU/Swish activation
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        return self.down_proj(hidden)


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm architecture

    Features:
    - Pre-normalization (more stable training)
    - Residual connections
    - Optional MoE integration point
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Layer normalization (pre-norm)
        self.input_layernorm = nn.LayerNorm(config.d_model, eps=1e-5)
        self.post_attention_layernorm = nn.LayerNorm(config.d_model, eps=1e-5)

        # Multi-head attention
        self.self_attn = MultiHeadAttention(config)

        # Feed-forward network (will be replaced by MoE if configured)
        self.mlp = FeedForward(config)

        # Dropout for residual connections
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        attn_output, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache
        )

        # Residual connection
        hidden_states = residual + self.dropout(attn_output)

        # Pre-norm MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Feed-forward
        mlp_output = self.mlp(hidden_states)

        # Residual connection
        hidden_states = residual + self.dropout(mlp_output)

        return hidden_states, attn_weights, present_key_value


class NanoLMModel(nn.Module):
    """
    Main NanoLM model architecture

    A lightweight transformer optimized for edge deployment with:
    - Quantization support (NF4/FP4)
    - Flash attention
    - RoPE positional encoding
    - Configurable architecture
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_layers = config.n_layers

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(config.d_model, eps=1e-5)

        # Language modeling head
        if config.tie_word_embeddings:
            self.lm_head = None  # Will use embed_tokens.weight
        else:
            # Quantization config for output head
            quant_config = {
                'use_bnb_4bit': config.use_bnb_4bit and config.bnb_4bit_quantize_heads,
                'compute_dtype': config.bnb_4bit_compute_dtype,
                'use_double_quant': config.bnb_4bit_use_double_quant,
                'quant_type': config.bnb_4bit_quant_type
            }
            self.lm_head = QuantizedLinear(
                config.d_model,
                config.vocab_size,
                bias=False,
                quantization_config=quant_config
            )

        # Initialize weights
        self.apply(self._init_weights)

        # Quantization controller
        self.quantization_controller = None
        if config.use_quantization:
            self.quantization_controller = QuantizationController(config)

        logging.info(f"✅ NanoLM model initialized")
        logging.info(f"  • Parameters: {self.get_num_params():,}")
        logging.info(f"  • Layers: {config.n_layers}")
        logging.info(f"  • Heads: {config.n_heads}")
        logging.info(f"  • Hidden size: {config.d_model}")
        logging.info(f"  • Vocab size: {config.vocab_size:,}")
        logging.info(f"  • Quantization: {'Enabled' if config.use_quantization else 'Disabled'}")

    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            # Use Xavier/Glorot initialization for linear layers
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Use normal initialization for embeddings
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # Initialize layer norm
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, QuantizedLinear):
            # Initialize quantized linear layers
            if hasattr(module.linear, 'weight'):
                torch.nn.init.xavier_uniform_(module.linear.weight)
                if hasattr(module.linear, 'bias') and module.linear.bias is not None:
                    torch.nn.init.zeros_(module.linear.bias)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Get number of parameters"""
        params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            params -= self.embed_tokens.weight.numel()
        return params

    def get_memory_footprint(self) -> Dict[str, float]:
        """Get model memory footprint in MB"""
        total_params = 0
        quantized_params = 0

        # Use actual parameter count from the model
        total_params = self.get_num_params()

        # Count quantized parameters
        for name, module in self.named_modules():
            if isinstance(module, QuantizedLinear) and module.is_quantized:
                quantized_params += sum(p.numel() for p in module.parameters())

        # Calculate memory usage
        # Quantized parameters use ~0.5 bytes (4-bit), others use 4 bytes (FP32)
        quantized_memory = quantized_params * 0.5 / (1024 ** 2)  # MB
        unquantized_memory = (total_params - quantized_params) * 4 / (1024 ** 2)  # MB
        total_memory = quantized_memory + unquantized_memory

        return {
            'total_params': total_params,
            'quantized_params': quantized_params,
            'quantized_memory_mb': quantized_memory,
            'unquantized_memory_mb': unquantized_memory,
            'total_memory_mb': total_memory,
            'quantization_ratio': quantized_params / total_params if total_params > 0 else 0.0
        }

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                return_dict: bool = True) -> ModelOutput:

        batch_size, seq_len = input_ids.shape

        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Prepare attention mask
        if attention_mask is not None:
            # Convert attention mask to causal mask format
            attention_mask = self._prepare_attention_mask(attention_mask, seq_len)

        # Initialize past key values if not provided
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        # Forward through transformer layers
        all_hidden_states = []
        all_attentions = []
        present_key_values = []

        for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            all_hidden_states.append(hidden_states)

            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache
            )

            hidden_states = layer_outputs[0]

            if layer_outputs[1] is not None:
                all_attentions.append(layer_outputs[1])

            if use_cache:
                present_key_values.append(layer_outputs[2])

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Language modeling head
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            # Tied embeddings
            logits = F.linear(hidden_states, self.embed_tokens.weight)

        if return_dict:
            return ModelOutput(
                logits=logits,
                hidden_states=all_hidden_states,
                attention_weights=all_attentions if all_attentions else None,
                past_key_values=present_key_values if use_cache else None
            )
        else:
            return (logits, all_hidden_states, all_attentions, present_key_values)

    def _prepare_attention_mask(self, attention_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Prepare attention mask for causal attention"""
        batch_size = attention_mask.shape[0] if attention_mask is not None else 1
        device = attention_mask.device if attention_mask is not None else torch.device('cpu')

        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        # Combine with attention mask
        if attention_mask is not None:
            # Expand attention mask to match causal mask dimensions
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            # Broadcast multiply
            attention_mask = attention_mask * causal_mask  # (batch, 1, seq_len, seq_len)
        else:
            attention_mask = causal_mask.expand(batch_size, 1, seq_len, seq_len)

        # Convert to additive mask (0 for attend, -inf for mask)
        attention_mask = (1.0 - attention_mask) * -10000.0

        return attention_mask

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9,
                do_sample: bool = True) -> torch.Tensor:
        """Simple text generation"""
        self.eval()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(input_ids, use_cache=False)
                logits = outputs.logits

                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature

                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')

                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')

                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids


def create_nanolm_model(config) -> NanoLMModel:
    """Factory function to create NanoLM model"""
    model = NanoLMModel(config)

    # Apply quantization if enabled
    if config.use_quantization and model.quantization_controller:
        model = model.quantization_controller.apply_quantization(model)

    return model


def test_nanolm_model():
    """Test the NanoLM model"""
    print("🧪 Testing NanoLM Model")

    # Mock config for testing
    class MockConfig:
        vocab_size = 1000
        d_model = 256
        n_heads = 8
        d_head = 32
        n_layers = 4
        d_ff = 1024
        seq_len = 128
        dropout = 0.1
        attn_dropout = 0.1
        ff_dropout = 0.1
        use_flash_attn = False
        rope_base = 10000
        rope_scaling = 1.0
        tie_word_embeddings = True

        # Quantization settings
        use_quantization = False
        use_bnb_4bit = False
        bnb_4bit_quant_type = "nf4"
        bnb_4bit_compute_dtype = "bfloat16"
        bnb_4bit_use_double_quant = True
        bnb_4bit_quantize_heads = False

    config = MockConfig()

    # Create model
    model = create_nanolm_model(config)

    # Test forward pass
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    outputs = model(input_ids)

    print(f"✅ Model forward pass successful")
    print(f"  • Input shape: {input_ids.shape}")
    print(f"  • Output logits shape: {outputs.logits.shape}")
    print(f"  • Model parameters: {model.get_num_params():,}")

    # Test memory footprint
    memory_info = model.get_memory_footprint()
    print(f"  • Memory footprint: {memory_info['total_memory_mb']:.1f} MB")

    # Test generation
    print("✅ Testing text generation...")
    generated = model.generate(input_ids[:1, :10], max_new_tokens=5, do_sample=False)
    print(f"  • Generated sequence length: {generated.shape[1]}")

    print("🎉 All NanoLM model tests passed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    test_nanolm_model()