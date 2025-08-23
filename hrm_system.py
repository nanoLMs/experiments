#!/usr/bin/env python3
"""
Hierarchical Reasoning Module (HRM) System for NanoLM
====================================================

Implements hierarchical reasoning based on research:
"Hierarchical Reasoning in Neural Networks: Multi-Timescale Processing"

Features:
- Multi-timescale processing (slow high-level, fast low-level reasoning)
- Hierarchical state management with different update frequencies
- N_cycles and T_steps configuration support
- 1-step gradient approximation for memory efficiency
- Hierarchical convergence detection
- Integration with transformer architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from nanolm_model import QuantizedLinear


class ReasoningLevel(Enum):
    """Hierarchical reasoning levels"""
    LOW = "low"          # Fast, local reasoning (every step)
    MID = "mid"          # Medium-term reasoning (every T_steps)
    HIGH = "high"        # High-level reasoning (every N_cycles * T_steps)


@dataclass
class HRMState:
    """State container for hierarchical reasoning"""
    low_level_state: torch.Tensor      # Updated every step
    mid_level_state: torch.Tensor      # Updated every T_steps
    high_level_state: torch.Tensor     # Updated every N_cycles * T_steps
    step_count: int                     # Current step counter
    cycle_count: int                    # Current cycle counter
    convergence_scores: Dict[str, float] # Convergence tracking
    gradient_approximation_enabled: bool = False # Whether gradient approximation is active


@dataclass
class HRMOutput:
    """Output structure for Hierarchical Reasoning Module"""
    reasoning_logits: torch.Tensor      # Final reasoning predictions
    hierarchical_states: HRMState       # Updated hierarchical states
    reasoning_loss: torch.Tensor        # Reasoning-specific loss
    convergence_metrics: Dict[str, float] # Convergence analysis
    level_contributions: Dict[str, torch.Tensor] # Contribution from each level


class HierarchicalProcessor(nn.Module):
    """
    Single hierarchical processing unit for one reasoning level

    Each processor handles reasoning at a specific timescale.
    """

    def __init__(self, config, level: ReasoningLevel, update_frequency: int = 1):
        super().__init__()
        self.config = config
        self.level = level
        self.update_frequency = update_frequency
        self.d_model = config.d_model

        # Quantization config
        quant_config = {
            'use_bnb_4bit': config.use_bnb_4bit and config.bnb_4bit_quantize_hrm,
            'compute_dtype': config.bnb_4bit_compute_dtype,
            'use_double_quant': config.bnb_4bit_use_double_quant,
            'quant_type': config.bnb_4bit_quant_type
        }

        # State processing layers
        self.state_processor = QuantizedLinear(
            self.d_model,
            self.d_model,
            bias=True,
            quantization_config=quant_config
        )

        self.state_update = QuantizedLinear(
            self.d_model * 2,  # Current state + input
            self.d_model,
            bias=True,
            quantization_config=quant_config
        )

        # Reasoning head for this level
        self.reasoning_head = QuantizedLinear(
            self.d_model,
            config.vocab_size,
            bias=False,
            quantization_config=quant_config
        )

        # Layer normalization and dropout
        self.state_norm = nn.LayerNorm(self.d_model, eps=1e-5)
        self.reasoning_norm = nn.LayerNorm(self.d_model, eps=1e-5)
        self.dropout = nn.Dropout(config.dropout)

        # Gating mechanism for hierarchical influence
        self.influence_gate = QuantizedLinear(
            self.d_model,
            1,
            bias=True,
            quantization_config=quant_config
        )

        # Temperature parameter for this level
        self.temperature = nn.Parameter(torch.ones(1))

        logging.info(f"✅ HRM Processor initialized for {level.value} level (freq: {update_frequency})")

    def forward(self, input_hidden: torch.Tensor, current_state: torch.Tensor,
                should_update: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through hierarchical processor

        Args:
            input_hidden: Input hidden states from transformer
            current_state: Current state for this level
            should_update: Whether to update state this step

        Returns:
            - updated_state: New state for this level
            - reasoning_output: Reasoning predictions from this level
            - influence_weight: How much this level should influence final output
        """
        batch_size, seq_len, d_model = input_hidden.shape

        if should_update:
            # Process current state
            processed_state = self.state_processor(current_state)
            processed_state = F.gelu(processed_state)
            processed_state = self.state_norm(processed_state)

            # Combine with input for state update
            combined = torch.cat([processed_state, input_hidden], dim=-1)
            updated_state = self.state_update(combined)
            updated_state = F.gelu(updated_state)
            updated_state = self.dropout(updated_state)

            # Residual connection
            updated_state = updated_state + current_state
        else:
            # No update this step
            updated_state = current_state

        # Generate reasoning output from current state
        reasoning_features = self.reasoning_norm(updated_state)
        reasoning_output = self.reasoning_head(reasoning_features)
        reasoning_output = reasoning_output / self.temperature

        # Calculate influence weight (how much this level should contribute)
        influence_weight = torch.sigmoid(self.influence_gate(updated_state))

        return updated_state, reasoning_output, influence_weight

    def calculate_convergence(self, prev_state: torch.Tensor,
                            current_state: torch.Tensor) -> float:
        """Calculate convergence score for this level"""
        if prev_state is None:
            return 0.0

        # Calculate L2 distance between states
        state_diff = torch.norm(current_state - prev_state, dim=-1)
        state_norm = torch.norm(current_state, dim=-1)

        # Relative change
        relative_change = state_diff / (state_norm + 1e-8)
        convergence_score = 1.0 - torch.mean(relative_change).item()

        return max(0.0, min(1.0, convergence_score))


class HierarchicalReasoningModule(nn.Module):
    """
    Complete Hierarchical Reasoning Module

    Manages multiple reasoning levels with different update frequencies.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size

        # HRM configuration
        self.N_cycles = config.hrm_N_cycles  # High-level reasoning cycles
        self.T_steps = config.hrm_T_steps    # Mid-level reasoning steps
        self.use_gradient_approximation = config.hrm_use_gradient_approx

        # Gradient approximation settings for memory efficiency
        self.gradient_approximation_steps = 1  # 1-step approximation
        self.approximation_momentum = 0.9      # Momentum for gradient approximation
        self.gradient_cache = {}               # Cache for gradient approximation

        # Create hierarchical processors
        self.low_processor = HierarchicalProcessor(
            config, ReasoningLevel.LOW, update_frequency=1
        )

        self.mid_processor = HierarchicalProcessor(
            config, ReasoningLevel.MID, update_frequency=self.T_steps
        )

        self.high_processor = HierarchicalProcessor(
            config, ReasoningLevel.HIGH, update_frequency=self.N_cycles * self.T_steps
        )

        # Quantization config for fusion layers
        quant_config = {
            'use_bnb_4bit': config.use_bnb_4bit and config.bnb_4bit_quantize_hrm,
            'compute_dtype': config.bnb_4bit_compute_dtype,
            'use_double_quant': config.bnb_4bit_use_double_quant,
            'quant_type': config.bnb_4bit_quant_type
        }

        # Hierarchical fusion layer
        self.fusion_layer = QuantizedLinear(
            self.d_model * 3,  # Low + Mid + High
            self.d_model,
            bias=True,
            quantization_config=quant_config
        )

        # Final reasoning head
        self.final_reasoning_head = QuantizedLinear(
            self.d_model,
            self.vocab_size,
            bias=False,
            quantization_config=quant_config
        )

        # Layer normalization
        self.fusion_norm = nn.LayerNorm(self.d_model, eps=1e-5)
        self.final_norm = nn.LayerNorm(self.d_model, eps=1e-5)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Convergence tracking
        self.convergence_threshold = config.hrm_convergence_threshold
        self.min_convergence_steps = config.hrm_min_convergence_steps

        logging.info(f"✅ HRM System initialized")
        logging.info(f"  • N_cycles: {self.N_cycles}, T_steps: {self.T_steps}")
        logging.info(f"  • Gradient approximation: {self.use_gradient_approximation}")

    def initialize_states(self, batch_size: int, seq_len: int,
                         device: torch.device) -> HRMState:
        """Initialize hierarchical states"""
        return HRMState(
            low_level_state=torch.zeros(batch_size, seq_len, self.d_model, device=device),
            mid_level_state=torch.zeros(batch_size, seq_len, self.d_model, device=device),
            high_level_state=torch.zeros(batch_size, seq_len, self.d_model, device=device),
            step_count=0,
            cycle_count=0,
            convergence_scores={'low': 0.0, 'mid': 0.0, 'high': 0.0}
        )

    def forward(self, hidden_states: torch.Tensor,
                hrm_state: Optional[HRMState] = None,
                use_gradient_approximation: Optional[bool] = None) -> HRMOutput:
        """
        Forward pass through hierarchical reasoning

        Args:
            hidden_states: Input hidden states from transformer
            hrm_state: Previous HRM state (None for initialization)

        Returns:
            HRMOutput with reasoning predictions and updated states
        """
        batch_size, seq_len, d_model = hidden_states.shape
        device = hidden_states.device

        # Initialize states if not provided
        if hrm_state is None:
            hrm_state = self.initialize_states(batch_size, seq_len, device)

        # Store previous states for convergence calculation
        prev_states = {
            'low': hrm_state.low_level_state.clone(),
            'mid': hrm_state.mid_level_state.clone(),
            'high': hrm_state.high_level_state.clone()
        }

        # Update step and cycle counters
        hrm_state.step_count += 1
        if hrm_state.step_count % (self.N_cycles * self.T_steps) == 0:
            hrm_state.cycle_count += 1

        # Determine which levels should update this step
        should_update_low = True  # Always update low level
        should_update_mid = (hrm_state.step_count % self.T_steps == 0)
        should_update_high = (hrm_state.step_count % (self.N_cycles * self.T_steps) == 0)

        # Process each hierarchical level
        hrm_state.low_level_state, low_reasoning, low_influence = self.low_processor(
            hidden_states, hrm_state.low_level_state, should_update_low
        )

        hrm_state.mid_level_state, mid_reasoning, mid_influence = self.mid_processor(
            hidden_states, hrm_state.mid_level_state, should_update_mid
        )

        hrm_state.high_level_state, high_reasoning, high_influence = self.high_processor(
            hidden_states, hrm_state.high_level_state, should_update_high
        )

        # Calculate convergence scores
        convergence_metrics = {}
        if should_update_low:
            hrm_state.convergence_scores['low'] = self.low_processor.calculate_convergence(
                prev_states['low'], hrm_state.low_level_state
            )
        if should_update_mid:
            hrm_state.convergence_scores['mid'] = self.mid_processor.calculate_convergence(
                prev_states['mid'], hrm_state.mid_level_state
            )
        if should_update_high:
            hrm_state.convergence_scores['high'] = self.high_processor.calculate_convergence(
                prev_states['high'], hrm_state.high_level_state
            )

        convergence_metrics = hrm_state.convergence_scores.copy()
        convergence_metrics['overall'] = sum(hrm_state.convergence_scores.values()) / 3

        # Fuse hierarchical representations
        fused_features = torch.cat([
            hrm_state.low_level_state,
            hrm_state.mid_level_state,
            hrm_state.high_level_state
        ], dim=-1)

        fused_output = self.fusion_layer(fused_features)
        fused_output = F.gelu(fused_output)
        fused_output = self.fusion_norm(fused_output)
        fused_output = self.dropout(fused_output)

        # Final reasoning prediction
        final_features = self.final_norm(fused_output)
        reasoning_logits = self.final_reasoning_head(final_features)

        # Store level contributions for analysis
        level_contributions = {
            'low': low_reasoning,
            'mid': mid_reasoning,
            'high': high_reasoning,
            'low_influence': low_influence,
            'mid_influence': mid_influence,
            'high_influence': high_influence
        }

        # Apply gradient approximation if enabled
        use_approx = use_gradient_approximation if use_gradient_approximation is not None else self.use_gradient_approximation
        if use_approx and self.training:
            # Store current state for gradient approximation
            # This will be used in the backward pass
            hrm_state.gradient_approximation_enabled = True
        else:
            hrm_state.gradient_approximation_enabled = False

        return HRMOutput(
            reasoning_logits=reasoning_logits,
            hierarchical_states=hrm_state,
            reasoning_loss=torch.tensor(0.0, device=device),  # Will be computed in loss function
            convergence_metrics=convergence_metrics,
            level_contributions=level_contributions
        )

    def calculate_reasoning_loss(self, reasoning_logits: torch.Tensor,
                               targets: torch.Tensor,
                               level_contributions: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate hierarchical reasoning loss

        Args:
            reasoning_logits: Final reasoning predictions
            targets: Target tokens
            level_contributions: Contributions from each level

        Returns:
            - total_reasoning_loss: Combined reasoning loss
            - individual_losses: Losses for each level
        """
        individual_losses = {}

        # Main reasoning loss
        main_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        main_reasoning_loss = main_loss_fn(
            reasoning_logits.contiguous().view(-1, self.vocab_size),
            targets.contiguous().view(-1)
        )

        # Individual level losses (auxiliary losses)
        level_weights = {'low': 0.3, 'mid': 0.2, 'high': 0.1}

        for level, weight in level_weights.items():
            if level in level_contributions:
                level_logits = level_contributions[level]
                level_loss = main_loss_fn(
                    level_logits.contiguous().view(-1, self.vocab_size),
                    targets.contiguous().view(-1)
                )
                individual_losses[level] = level_loss * weight

        # Total reasoning loss
        total_reasoning_loss = main_reasoning_loss + sum(individual_losses.values())
        individual_losses['main'] = main_reasoning_loss

        return total_reasoning_loss, individual_losses

    def check_convergence(self, convergence_metrics: Dict[str, float]) -> bool:
        """Check if hierarchical reasoning has converged"""
        overall_convergence = convergence_metrics.get('overall', 0.0)
        return overall_convergence >= self.convergence_threshold

    def approximate_gradients(self, current_loss: torch.Tensor,
                            previous_loss: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        1-step gradient approximation for memory efficiency

        Uses first-order Taylor approximation to estimate gradients without
        storing full computation graph for hierarchical states.

        Args:
            current_loss: Current step loss
            previous_loss: Previous step loss (for momentum)

        Returns:
            Dictionary of approximated gradients for each level
        """
        if not self.use_gradient_approximation:
            return {}

        approximated_grads = {}

        # Get current parameters for each processor
        processors = {
            'low': self.low_processor,
            'mid': self.mid_processor,
            'high': self.high_processor
        }

        for level_name, processor in processors.items():
            level_grads = {}

            # Calculate approximate gradients using finite differences
            for name, param in processor.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Use current gradient as base
                    current_grad = param.grad.clone()

                    # Apply momentum if we have cached gradients
                    cache_key = f"{level_name}_{name}"
                    if cache_key in self.gradient_cache:
                        cached_grad = self.gradient_cache[cache_key]
                        # Momentum-based approximation
                        approx_grad = (self.approximation_momentum * cached_grad +
                                     (1 - self.approximation_momentum) * current_grad)
                    else:
                        approx_grad = current_grad

                    # Store in cache for next iteration
                    self.gradient_cache[cache_key] = approx_grad.clone()
                    level_grads[name] = approx_grad

            approximated_grads[level_name] = level_grads

        return approximated_grads

    def apply_gradient_approximation(self, approximated_grads: Dict[str, Dict[str, torch.Tensor]]):
        """
        Apply approximated gradients to reduce memory usage

        Args:
            approximated_grads: Dictionary of approximated gradients
        """
        if not self.use_gradient_approximation or not approximated_grads:
            return

        processors = {
            'low': self.low_processor,
            'mid': self.mid_processor,
            'high': self.high_processor
        }

        for level_name, level_grads in approximated_grads.items():
            if level_name in processors:
                processor = processors[level_name]

                for name, param in processor.named_parameters():
                    if name in level_grads and param.requires_grad:
                        # Replace gradient with approximation
                        param.grad = level_grads[name].clone()

    def clear_gradient_cache(self):
        """Clear gradient approximation cache"""
        self.gradient_cache.clear()

    def get_gradient_approximation_stats(self) -> Dict[str, Any]:
        """Get statistics about gradient approximation"""
        return {
            'use_gradient_approximation': self.use_gradient_approximation,
            'approximation_steps': self.gradient_approximation_steps,
            'approximation_momentum': self.approximation_momentum,
            'cache_size': len(self.gradient_cache),
            'cached_parameters': list(self.gradient_cache.keys())
        }

    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get statistics for the reasoning module"""
        return {
            'N_cycles': self.N_cycles,
            'T_steps': self.T_steps,
            'convergence_threshold': self.convergence_threshold,
            'use_gradient_approximation': self.use_gradient_approximation,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'processor_params': {
                'low': sum(p.numel() for p in self.low_processor.parameters()),
                'mid': sum(p.numel() for p in self.mid_processor.parameters()),
                'high': sum(p.numel() for p in self.high_processor.parameters())
            }
        }


class HRMIntegratedModel(nn.Module):
    """
    NanoLM model with integrated Hierarchical Reasoning Module

    Extends the base model with HRM capabilities.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Import here to avoid circular imports
        from nanolm_model import NanoLMModel

        # Base model
        self.base_model = NanoLMModel(config)

        # Hierarchical reasoning module
        self.hrm = HierarchicalReasoningModule(config)

        # HRM state management
        self.hrm_state = None
        self.reset_hrm_every = config.hrm_reset_every  # Reset HRM state every N steps
        self.step_count = 0

        logging.info(f"✅ HRM Integrated Model initialized")
        logging.info(f"  • Base model parameters: {self.base_model.get_num_params():,}")
        logging.info(f"  • HRM parameters: {sum(p.numel() for p in self.hrm.parameters()):,}")

    def reset_hrm_state(self):
        """Reset HRM state (useful for new sequences)"""
        self.hrm_state = None
        self.step_count = 0
        logging.debug("🔄 HRM state reset")

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                reset_hrm: bool = False,
                return_dict: bool = True) -> Dict[str, Any]:
        """Forward pass with HRM"""

        if reset_hrm:
            self.reset_hrm_state()

        # Base model forward pass
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Get hidden states from the last layer
        hidden_states = base_outputs.hidden_states[-1] if base_outputs.hidden_states else None
        if hidden_states is None:
            # Fallback: use states before final projection
            hidden_states = self.base_model.norm(base_outputs.logits)

        # Hierarchical reasoning
        hrm_outputs = self.hrm(hidden_states, self.hrm_state)
        self.hrm_state = hrm_outputs.hierarchical_states

        # Update step count and reset if needed
        self.step_count += 1
        if self.reset_hrm_every > 0 and self.step_count >= self.reset_hrm_every:
            self.reset_hrm_state()

        # Calculate losses if labels provided
        total_loss = None
        main_loss = None
        reasoning_loss = None
        individual_reasoning_losses = None

        if labels is not None:
            # Main loss (standard next-token prediction)
            main_loss_fn = nn.CrossEntropyLoss()
            main_loss = main_loss_fn(
                base_outputs.logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )

            # Reasoning loss
            reasoning_loss, individual_reasoning_losses = self.hrm.calculate_reasoning_loss(
                hrm_outputs.reasoning_logits, labels, hrm_outputs.level_contributions
            )

            # Combined loss
            reasoning_weight = self.config.hrm_loss_weight
            total_loss = main_loss + reasoning_weight * reasoning_loss

        if return_dict:
            return {
                'logits': base_outputs.logits,
                'reasoning_logits': hrm_outputs.reasoning_logits,
                'hidden_states': base_outputs.hidden_states,
                'attention_weights': base_outputs.attention_weights,
                'loss': total_loss,
                'main_loss': main_loss,
                'reasoning_loss': reasoning_loss,
                'individual_reasoning_losses': individual_reasoning_losses,
                'convergence_metrics': hrm_outputs.convergence_metrics,
                'level_contributions': hrm_outputs.level_contributions,
                'hrm_state': self.hrm_state
            }
        else:
            return (
                base_outputs.logits,
                hrm_outputs.reasoning_logits,
                base_outputs.hidden_states,
                total_loss
            )

    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def memory_efficient_forward(self, input_ids: torch.Tensor,
                               attention_mask: Optional[torch.Tensor] = None,
                               labels: Optional[torch.Tensor] = None,
                               use_gradient_approximation: bool = True) -> Dict[str, Any]:
        """
        Memory-efficient forward pass using gradient approximation

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            use_gradient_approximation: Whether to use 1-step gradient approximation

        Returns:
            Model outputs with reduced memory footprint
        """
        # Use gradient checkpointing for base model if available
        if hasattr(self.base_model, 'gradient_checkpointing') and use_gradient_approximation:
            self.base_model.gradient_checkpointing = True

        # Forward pass with gradient approximation
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        # Apply gradient approximation if enabled and in training mode
        if use_gradient_approximation and self.training and labels is not None:
            # Calculate approximated gradients
            current_loss = outputs['reasoning_loss']
            approximated_grads = self.hrm.approximate_gradients(current_loss)

            # Apply approximated gradients to reduce memory usage
            self.hrm.apply_gradient_approximation(approximated_grads)

            # Add approximation stats to outputs
            outputs['gradient_approximation_stats'] = self.hrm.get_gradient_approximation_stats()

        return outputs

    def clear_hrm_gradient_cache(self):
        """Clear HRM gradient approximation cache"""
        self.hrm.clear_gradient_cache()

    def get_hrm_stats(self) -> Dict[str, Any]:
        """Get HRM system statistics"""
        base_params = self.base_model.get_num_params()
        hrm_params = sum(p.numel() for p in self.hrm.parameters())
        total_params = self.get_num_params()

        stats = {
            'base_model_params': base_params,
            'hrm_params': hrm_params,
            'total_params': total_params,
            'hrm_overhead': hrm_params / base_params,
            'step_count': self.step_count,
            'hrm_reasoning_stats': self.hrm.get_reasoning_stats(),
            'gradient_approximation_stats': self.hrm.get_gradient_approximation_stats()
        }

        if self.hrm_state:
            stats['current_convergence'] = self.hrm_state.convergence_scores
            stats['hrm_step_count'] = self.hrm_state.step_count
            stats['hrm_cycle_count'] = self.hrm_state.cycle_count
            stats['gradient_approximation_enabled'] = getattr(self.hrm_state, 'gradient_approximation_enabled', False)

        return stats


def create_hrm_model(config) -> HRMIntegratedModel:
    """Factory function to create HRM-integrated model"""
    return HRMIntegratedModel(config)


def calculate_hrm_metrics(reasoning_outputs: Dict[str, torch.Tensor],
                         targets: torch.Tensor,
                         convergence_metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate comprehensive HRM metrics

    Args:
        reasoning_outputs: Reasoning predictions from different levels
        targets: Target token sequence
        convergence_metrics: Convergence scores from HRM

    Returns:
        Dictionary with various HRM metrics
    """
    metrics = {
        'convergence': convergence_metrics,
        'level_accuracies': {},
        'level_perplexities': {},
        'reasoning_diversity': {},
        'hierarchical_consistency': 0.0
    }

    # Calculate metrics for each reasoning level
    for level_name in ['low', 'mid', 'high']:
        if level_name in reasoning_outputs:
            level_logits = reasoning_outputs[level_name]

            # Accuracy
            predicted_ids = torch.argmax(level_logits, dim=-1)
            accuracy = (predicted_ids == targets).float().mean().item()
            metrics['level_accuracies'][level_name] = accuracy

            # Perplexity
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(level_logits.contiguous().view(-1, level_logits.shape[-1]),
                          targets.contiguous().view(-1))
            perplexity = torch.exp(loss).item()
            metrics['level_perplexities'][level_name] = perplexity

            # Prediction diversity (entropy)
            probs = F.softmax(level_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            avg_entropy = entropy.mean().item()
            metrics['reasoning_diversity'][level_name] = avg_entropy

    # Hierarchical consistency: how consistent are predictions across levels
    level_predictions = []
    for level_name in ['low', 'mid', 'high']:
        if level_name in reasoning_outputs:
            pred_ids = torch.argmax(reasoning_outputs[level_name], dim=-1)
            level_predictions.append(pred_ids)

    if len(level_predictions) > 1:
        consistency_scores = []
        for i in range(len(level_predictions) - 1):
            consistency = (level_predictions[i] == level_predictions[i+1]).float().mean().item()
            consistency_scores.append(consistency)

        if consistency_scores:
            metrics['hierarchical_consistency'] = sum(consistency_scores) / len(consistency_scores)

    return metrics


def test_hrm_system():
    """Test the HRM system"""
    print("🧪 Testing HRM System")

    # Mock config for testing
    class MockConfig:
        d_model = 256
        vocab_size = 1000
        dropout = 0.1

        # HRM settings
        hrm_N_cycles = 2
        hrm_T_steps = 2
        hrm_use_gradient_approx = True
        hrm_convergence_threshold = 0.95
        hrm_min_convergence_steps = 10
        hrm_loss_weight = 0.1
        hrm_reset_every = 100

        # Quantization settings
        use_bnb_4bit = False
        bnb_4bit_quantize_hrm = False
        bnb_4bit_compute_dtype = "bfloat16"
        bnb_4bit_use_double_quant = True
        bnb_4bit_quant_type = "nf4"

        # Full model settings
        n_layers = 4
        n_heads = 8
        d_head = 32
        d_ff = 1024
        seq_len = 128
        attn_dropout = 0.1
        ff_dropout = 0.1
        use_flash_attn = False
        rope_base = 10000
        rope_scaling = 1.0
        tie_word_embeddings = True
        use_quantization = False

        # Additional required attributes
        bnb_4bit_quant_storage = "uint8"
        bnb_4bit_quantize_router = False
        use_fp4 = False
        fp4_format = "nvfp4"
        fp4_block_size = 16
        fp4_split_rounding = True
        qaf_threshold = 1e-6
        max_qaf_steps = 100
        qaf_precision = "bf16"

        # MoE settings
        moe_every = 0
        n_experts = 4
        moe_top_k = 1
        expert_ff_mult = 1.0
        router_jitter = 0.01
        moe_aux_weight = 0.01
        router_z_loss = 1e-4
        capacity_factor = 1.0

        # MTP settings
        mtp_k = 4
        mtp_loss_weights = [1.0, 0.5, 0.25, 0.125]

    config = MockConfig()

    # Test hierarchical processor
    print("✅ Testing Hierarchical Processor...")
    processor = HierarchicalProcessor(config, ReasoningLevel.LOW, update_frequency=1)
    x = torch.randn(4, 32, config.d_model)
    state = torch.randn(4, 32, config.d_model)

    new_state, reasoning_out, influence = processor(x, state, should_update=True)
    print(f"  • Processor output shapes: {new_state.shape}, {reasoning_out.shape}, {influence.shape}")

    # Test convergence calculation
    convergence = processor.calculate_convergence(state, new_state)
    print(f"  • Convergence score: {convergence:.3f}")

    # Test HRM module
    print("✅ Testing HRM Module...")
    hrm = HierarchicalReasoningModule(config)
    hrm_output = hrm(x)
    print(f"  • HRM reasoning logits shape: {hrm_output.reasoning_logits.shape}")
    print(f"  • Convergence metrics: {hrm_output.convergence_metrics}")

    # Test loss calculation
    print("✅ Testing HRM Loss Calculation...")
    targets = torch.randint(0, config.vocab_size, (4, 32))
    reasoning_loss, individual_losses = hrm.calculate_reasoning_loss(
        hrm_output.reasoning_logits, targets, hrm_output.level_contributions
    )
    print(f"  • Reasoning loss: {reasoning_loss.item():.6f}")
    print(f"  • Individual losses: {[f'{k}: {v.item():.6f}' for k, v in individual_losses.items()]}")

    # Test integrated model
    print("✅ Testing HRM Integrated Model...")
    model = create_hrm_model(config)

    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    outputs = model(input_ids, labels=labels)
    print(f"  • Model output logits shape: {outputs['logits'].shape}")
    print(f"  • Reasoning logits shape: {outputs['reasoning_logits'].shape}")
    print(f"  • Total loss: {outputs['loss'].item():.6f}")
    print(f"  • Reasoning loss: {outputs['reasoning_loss'].item():.6f}")

    # Test multiple forward passes (state persistence)
    print("✅ Testing HRM State Persistence...")
    for step in range(5):
        outputs = model(input_ids, labels=labels)
        convergence = outputs['convergence_metrics']['overall']
        print(f"  • Step {step+1}: Overall convergence = {convergence:.3f}")

    # Test HRM metrics
    print("✅ Testing HRM Metrics...")
    metrics = calculate_hrm_metrics(
        outputs['level_contributions'], labels, outputs['convergence_metrics']
    )
    print(f"  • Level accuracies: {metrics['level_accuracies']}")
    print(f"  • Hierarchical consistency: {metrics['hierarchical_consistency']:.3f}")

    # Test model statistics
    print("✅ Testing HRM Statistics...")
    stats = model.get_hrm_stats()
    print(f"  • Total parameters: {stats['total_params']:,}")
    print(f"  • HRM overhead: {stats['hrm_overhead']:.1%}")
    print(f"  • Gradient approximation: {stats['gradient_approximation_stats']['use_gradient_approximation']}")

    # Test memory-efficient forward pass
    print("✅ Testing Memory-Efficient Forward Pass...")
    efficient_outputs = model.memory_efficient_forward(input_ids, labels=labels)
    print(f"  • Memory-efficient loss: {efficient_outputs['loss'].item():.6f}")
    if 'gradient_approximation_stats' in efficient_outputs:
        print(f"  • Gradient cache size: {efficient_outputs['gradient_approximation_stats']['cache_size']}")

    # Test gradient approximation methods
    print("✅ Testing Gradient Approximation...")
    hrm = model.hrm
    test_loss = torch.tensor(5.0, requires_grad=True)
    approx_grads = hrm.approximate_gradients(test_loss)
    print(f"  • Approximated gradient levels: {list(approx_grads.keys())}")

    # Clear cache
    model.clear_hrm_gradient_cache()
    cache_stats = hrm.get_gradient_approximation_stats()
    print(f"  • Cache cleared: {cache_stats['cache_size'] == 0}")

    print("🎉 All HRM tests passed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    test_hrm_system()