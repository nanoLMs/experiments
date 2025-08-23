#!/usr/bin/env python3
"""
Mixture of Experts (MoE) System for NanoLM
==========================================

Implements efficient MoE with:
- Expert routing with load balancing
- Top-k gating (k=1 for efficiency)
- Auxiliary loss for load balancing
- Quantization support
- Memory-efficient expert selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass

from nanolm_model import QuantizedLinear


@dataclass
class MoEOutput:
    """Output structure for MoE layers"""
    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    expert_usage: torch.Tensor
    aux_loss: torch.Tensor
    load_balancing_loss: torch.Tensor


class Expert(nn.Module):
    """
    Single expert in MoE system

    Each expert is a feed-forward network that can specialize
    in different types of patterns or knowledge.
    """

    def __init__(self, config, expert_id: int = 0):
        super().__init__()
        self.config = config
        self.expert_id = expert_id
        self.d_model = config.d_model
        self.d_ff = int(config.d_ff * config.expert_ff_mult)  # Allow different expert sizes

        # Quantization config
        quant_config = {
            'use_bnb_4bit': config.use_bnb_4bit and config.bnb_4bit_quantize_router,
            'compute_dtype': config.bnb_4bit_compute_dtype,
            'use_double_quant': config.bnb_4bit_use_double_quant,
            'quant_type': config.bnb_4bit_quant_type
        }

        # Expert feed-forward network (SwiGLU)
        self.gate_proj = QuantizedLinear(
            self.d_model, self.d_ff, bias=False, quantization_config=quant_config
        )
        self.up_proj = QuantizedLinear(
            self.d_model, self.d_ff, bias=False, quantization_config=quant_config
        )
        self.down_proj = QuantizedLinear(
            self.d_ff, self.d_model, bias=False, quantization_config=quant_config
        )

        self.dropout = nn.Dropout(config.ff_dropout)

        # Expert usage tracking
        self.register_buffer('usage_count', torch.zeros(1))
        self.register_buffer('total_tokens', torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert"""
        # Update usage statistics
        if self.training:
            batch_size, seq_len = x.shape[:2]
            self.usage_count += 1
            self.total_tokens += batch_size * seq_len

        # SwiGLU activation
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        return self.down_proj(hidden)

    def get_usage_stats(self) -> Dict[str, float]:
        """Get expert usage statistics"""
        return {
            'expert_id': self.expert_id,
            'usage_count': self.usage_count.item(),
            'total_tokens': self.total_tokens.item(),
            'usage_rate': self.usage_count.item() / max(1, self.total_tokens.item())
        }

    def reset_usage_stats(self):
        """Reset usage statistics"""
        self.usage_count.zero_()
        self.total_tokens.zero_()


class Router(nn.Module):
    """
    Router for MoE system

    Determines which experts should process each token.
    Uses top-k gating with load balancing.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_experts = config.n_experts
        self.top_k = config.moe_top_k
        self.jitter_noise = config.router_jitter

        # Quantization config for router
        quant_config = {
            'use_bnb_4bit': config.use_bnb_4bit and config.bnb_4bit_quantize_router,
            'compute_dtype': config.bnb_4bit_compute_dtype,
            'use_double_quant': config.bnb_4bit_use_double_quant,
            'quant_type': config.bnb_4bit_quant_type
        }

        # Router network
        self.gate = QuantizedLinear(
            self.d_model, self.n_experts, bias=False, quantization_config=quant_config
        )

        # Load balancing tracking
        self.register_buffer('expert_counts', torch.zeros(self.n_experts))
        self.register_buffer('total_tokens_routed', torch.zeros(1))

        logging.info(f"✅ Router initialized: {self.n_experts} experts, top-{self.top_k}")

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts

        Returns:
            - expert_weights: Weights for combining expert outputs
            - expert_indices: Which experts to use for each token
            - router_logits: Raw router logits for auxiliary loss
        """
        batch_size, seq_len, d_model = hidden_states.shape
        hidden_states = hidden_states.view(-1, d_model)  # (batch_size * seq_len, d_model)

        # Compute router logits
        router_logits = self.gate(hidden_states)  # (batch_size * seq_len, n_experts)

        # Add jitter noise during training for better load balancing
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise

        # Apply softmax to get probabilities
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k gating
        expert_weights, expert_indices = torch.topk(router_probs, self.top_k, dim=-1)

        # Normalize weights (they should sum to 1 for each token)
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        # Update load balancing statistics
        if self.training:
            self._update_load_balancing_stats(expert_indices, batch_size * seq_len)

        return expert_weights, expert_indices, router_logits.view(batch_size, seq_len, -1)

    def _update_load_balancing_stats(self, expert_indices: torch.Tensor, num_tokens: int):
        """Update load balancing statistics"""
        # Count how many tokens are routed to each expert
        expert_counts = torch.zeros(self.n_experts, device=expert_indices.device)
        for i in range(self.n_experts):
            expert_counts[i] = (expert_indices == i).sum().float()

        # Update running statistics
        self.expert_counts += expert_counts
        self.total_tokens_routed += num_tokens

    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics"""
        total_tokens = self.total_tokens_routed.item()

        if total_tokens == 0:
            return {
                'expert_usage': [0.0] * self.n_experts,
                'balance_score': 1.0,
                'total_tokens': 0
            }

        # Calculate usage percentages
        expert_usage = (self.expert_counts / self.total_tokens_routed).tolist()

        # Calculate balance score (1.0 = perfectly balanced, 0.0 = completely imbalanced)
        ideal_usage = 1.0 / self.n_experts
        balance_score = 1.0 - sum(abs(usage - ideal_usage) for usage in expert_usage) / 2.0

        return {
            'expert_usage': expert_usage,
            'balance_score': balance_score,
            'total_tokens': total_tokens
        }

    def reset_load_balancing_stats(self):
        """Reset load balancing statistics"""
        self.expert_counts.zero_()
        self.total_tokens_routed.zero_()


class LoadBalancer(nn.Module):
    """
    Load balancing component for MoE

    Computes auxiliary losses to encourage balanced expert usage.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_experts = config.n_experts
        self.aux_weight = config.moe_aux_weight
        self.z_loss_weight = config.router_z_loss
        self.capacity_factor = config.capacity_factor

    def compute_aux_loss(self, router_logits: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute auxiliary loss for load balancing

        Based on the Switch Transformer paper: encourages balanced expert usage.
        """
        batch_size, seq_len, n_experts = router_logits.shape
        num_tokens = batch_size * seq_len

        # Compute router probabilities
        router_probs = F.softmax(router_logits.view(-1, n_experts), dim=-1)

        # Compute expert usage frequencies
        expert_usage = torch.zeros(n_experts, device=router_logits.device)
        for i in range(n_experts):
            expert_usage[i] = (expert_indices.view(-1) == i).float().mean()

        # Compute mean router probability for each expert
        mean_router_probs = router_probs.mean(dim=0)

        # Auxiliary loss: encourages uniform expert usage
        aux_loss = self.aux_weight * n_experts * torch.sum(mean_router_probs * expert_usage)

        return aux_loss

    def compute_z_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute router z-loss

        Encourages router logits to have reasonable magnitude.
        """
        if self.z_loss_weight == 0:
            return torch.tensor(0.0, device=router_logits.device)

        # Z-loss: penalizes large router logits
        z_loss = self.z_loss_weight * torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)

        return z_loss

    def compute_load_balancing_loss(self, router_logits: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """Compute total load balancing loss"""
        aux_loss = self.compute_aux_loss(router_logits, expert_indices)
        z_loss = self.compute_z_loss(router_logits)

        return aux_loss + z_loss


class MoELayer(nn.Module):
    """
    Complete MoE layer

    Combines router, experts, and load balancer into a single layer.
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_experts = config.n_experts
        self.top_k = config.moe_top_k

        # Create experts
        self.experts = nn.ModuleList([
            Expert(config, expert_id=i) for i in range(self.n_experts)
        ])

        # Create router
        self.router = Router(config)

        # Create load balancer
        self.load_balancer = LoadBalancer(config)

        logging.info(f"✅ MoE Layer {layer_idx} initialized: {self.n_experts} experts")

    def forward(self, hidden_states: torch.Tensor) -> MoEOutput:
        """Forward pass through MoE layer"""
        batch_size, seq_len, d_model = hidden_states.shape
        original_shape = hidden_states.shape

        # Flatten for expert processing
        hidden_states_flat = hidden_states.view(-1, d_model)  # (batch_size * seq_len, d_model)

        # Route tokens to experts
        expert_weights, expert_indices, router_logits = self.router(hidden_states)

        # Process tokens through selected experts
        expert_outputs = self._process_with_experts(hidden_states_flat, expert_weights, expert_indices)

        # Reshape back to original shape
        expert_outputs = expert_outputs.view(original_shape)

        # Compute auxiliary losses
        aux_loss = self.load_balancer.compute_aux_loss(router_logits, expert_indices)
        load_balancing_loss = self.load_balancer.compute_load_balancing_loss(router_logits, expert_indices)

        # Get expert usage statistics
        expert_usage = self._get_expert_usage_tensor(expert_indices)

        return MoEOutput(
            hidden_states=expert_outputs,
            router_logits=router_logits,
            expert_usage=expert_usage,
            aux_loss=aux_loss,
            load_balancing_loss=load_balancing_loss
        )

    def _process_with_experts(self, hidden_states: torch.Tensor,
                            expert_weights: torch.Tensor,
                            expert_indices: torch.Tensor) -> torch.Tensor:
        """Process tokens through selected experts"""
        num_tokens, d_model = hidden_states.shape
        output = torch.zeros_like(hidden_states)

        # Process each expert
        for expert_id in range(self.n_experts):
            # Find tokens assigned to this expert
            expert_mask = (expert_indices == expert_id).any(dim=-1)

            if expert_mask.sum() == 0:
                continue  # No tokens for this expert

            # Get tokens for this expert
            expert_tokens = hidden_states[expert_mask]

            if expert_tokens.numel() == 0:
                continue

            # Process through expert
            expert_output = self.experts[expert_id](expert_tokens)

            # Get weights for this expert
            expert_token_weights = expert_weights[expert_mask]
            expert_weight_mask = (expert_indices[expert_mask] == expert_id)

            # Apply weights and accumulate
            for i, token_idx in enumerate(expert_mask.nonzero().squeeze(-1)):
                if expert_weight_mask[i].any():
                    weight_idx = expert_weight_mask[i].nonzero().squeeze(-1)
                    if weight_idx.numel() > 0:
                        weight = expert_token_weights[i, weight_idx[0]]
                        output[token_idx] += weight * expert_output[i]

        return output

    def _get_expert_usage_tensor(self, expert_indices: torch.Tensor) -> torch.Tensor:
        """Get expert usage as tensor"""
        usage = torch.zeros(self.n_experts, device=expert_indices.device)
        total_tokens = expert_indices.numel()

        for i in range(self.n_experts):
            usage[i] = (expert_indices == i).sum().float() / total_tokens

        return usage

    def get_expert_stats(self) -> Dict[str, Any]:
        """Get comprehensive expert statistics"""
        stats = {
            'layer_idx': self.layer_idx,
            'n_experts': self.n_experts,
            'router_stats': self.router.get_load_balancing_stats(),
            'expert_stats': [expert.get_usage_stats() for expert in self.experts]
        }

        return stats

    def reset_expert_stats(self):
        """Reset all expert statistics"""
        self.router.reset_load_balancing_stats()
        for expert in self.experts:
            expert.reset_usage_stats()


class MoETransformerBlock(nn.Module):
    """
    Transformer block with MoE instead of regular feed-forward

    Replaces the standard feed-forward network with MoE layer.
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Import here to avoid circular imports
        from nanolm_model import MultiHeadAttention

        # Layer normalization (pre-norm)
        self.input_layernorm = nn.LayerNorm(config.d_model, eps=1e-5)
        self.post_attention_layernorm = nn.LayerNorm(config.d_model, eps=1e-5)

        # Multi-head attention
        self.self_attn = MultiHeadAttention(config)

        # MoE layer instead of regular feed-forward
        self.moe = MoELayer(config, layer_idx)

        # Dropout for residual connections
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]], MoEOutput]:

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

        # Pre-norm MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MoE layer
        moe_output = self.moe(hidden_states)

        # Residual connection
        hidden_states = residual + self.dropout(moe_output.hidden_states)

        return hidden_states, attn_weights, present_key_value, moe_output


def create_moe_model(config):
    """
    Create a model with MoE layers

    Replaces specified transformer blocks with MoE blocks.
    """
    from nanolm_model import NanoLMModel

    # Create base model
    model = NanoLMModel(config)

    # Replace specified layers with MoE layers
    if config.moe_every > 0:
        new_layers = nn.ModuleList()

        for i in range(config.n_layers):
            if (i + 1) % config.moe_every == 0:  # Replace every moe_every-th layer
                new_layers.append(MoETransformerBlock(config, layer_idx=i))
                logging.info(f"✅ Replaced layer {i} with MoE layer")
            else:
                new_layers.append(model.layers[i])

        model.layers = new_layers

    return model


def test_moe_system():
    """Test the MoE system"""
    print("🧪 Testing MoE System")

    # Mock config for testing
    class MockConfig:
        d_model = 256
        d_ff = 1024
        n_experts = 4
        moe_top_k = 1
        expert_ff_mult = 1.0
        router_jitter = 0.01
        moe_aux_weight = 0.01
        router_z_loss = 1e-4
        capacity_factor = 1.0
        ff_dropout = 0.1
        dropout = 0.1

        # Quantization settings
        use_bnb_4bit = False
        bnb_4bit_quantize_router = False
        bnb_4bit_compute_dtype = "bfloat16"
        bnb_4bit_use_double_quant = True
        bnb_4bit_quant_type = "nf4"

        # Model settings
        n_layers = 4
        n_heads = 8
        d_head = 32
        seq_len = 128
        vocab_size = 1000
        attn_dropout = 0.1
        use_flash_attn = False
        rope_base = 10000
        rope_scaling = 1.0
        tie_word_embeddings = True
        use_quantization = False
        moe_every = 2  # Every 2nd layer is MoE

    config = MockConfig()

    # Test individual components
    print("✅ Testing Expert...")
    expert = Expert(config, expert_id=0)
    x = torch.randn(16, 32, config.d_model)
    expert_output = expert(x)
    print(f"  • Expert output shape: {expert_output.shape}")

    print("✅ Testing Router...")
    router = Router(config)
    weights, indices, logits = router(x)
    print(f"  • Router weights shape: {weights.shape}")
    print(f"  • Router indices shape: {indices.shape}")

    print("✅ Testing MoE Layer...")
    moe_layer = MoELayer(config, layer_idx=0)
    moe_output = moe_layer(x)
    print(f"  • MoE output shape: {moe_output.hidden_states.shape}")
    print(f"  • Aux loss: {moe_output.aux_loss.item():.6f}")

    print("✅ Testing MoE Model...")
    model = create_moe_model(config)

    # Test forward pass
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    outputs = model(input_ids)
    print(f"  • Model output shape: {outputs.logits.shape}")
    print(f"  • Model parameters: {model.get_num_params():,}")

    # Test expert statistics
    moe_layers = [layer for layer in model.layers if isinstance(layer, MoETransformerBlock)]
    if moe_layers:
        stats = moe_layers[0].moe.get_expert_stats()
        print(f"  • Expert balance score: {stats['router_stats']['balance_score']:.3f}")

    print("🎉 All MoE tests passed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    test_moe_system()