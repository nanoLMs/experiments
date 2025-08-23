#!/usr/bin/env python3
"""
Advanced Multi-Component Loss System for NanoLM
==============================================

Implements comprehensive loss calculation system including:
- Main loss (standard next-token prediction)
- Multi-Token Prediction (MTP) loss
- Mixture of Experts (MoE) auxiliary loss
- Hierarchical Reasoning Module (HRM) loss
- Anti-hallucination loss
- Proper loss weighting and scaling for FP4 underflow prevention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

# For loss prediction methods (using numpy only)
# from scipy import optimize  # Optional for advanced prediction
# from sklearn.linear_model import LinearRegression  # Optional for advanced prediction


class LossComponent(Enum):
    """Types of loss components"""
    MAIN = "main"                           # Standard next-token prediction
    MTP = "mtp"                            # Multi-token prediction
    AUXILIARY = "auxiliary"                 # MoE load balancing
    REASONING = "reasoning"                 # Hierarchical reasoning
    ANTI_HALLUCINATION = "anti_hallucination"  # Factual consistency


@dataclass
class LossWeights:
    """Configuration for loss component weights"""
    main_weight: float = 1.0
    mtp_weight: float = 0.5
    auxiliary_weight: float = 0.01
    reasoning_weight: float = 0.1
    anti_hallucination_weight: float = 0.25

    # FP4 scaling factor to prevent underflow
    fp4_scaling_factor: float = 10.0
    adaptive_scaling: bool = True

    # Dynamic weight adjustment
    dynamic_weighting: bool = False
    weight_decay_schedule: Optional[Dict[str, float]] = None


@dataclass
class LossOutput:
    """Output structure for multi-component loss calculation"""
    total_loss: torch.Tensor                    # Combined weighted loss
    component_losses: Dict[str, torch.Tensor]   # Individual component losses
    weighted_losses: Dict[str, torch.Tensor]    # Weighted component losses
    loss_weights: Dict[str, float]              # Applied weights
    scaling_factor: float                       # Applied scaling factor
    loss_stats: Dict[str, Any]                 # Loss statistics and metrics


class MultiComponentLoss(nn.Module):
    """
    Advanced multi-component loss calculator

    Handles multiple loss components with proper weighting, scaling,
    and adaptive adjustments for stable training.
    """

    def __init__(self, config, loss_weights: Optional[LossWeights] = None):
        super().__init__()
        self.config = config
        self.loss_weights = loss_weights or LossWeights()

        # Loss functions
        self.main_loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=config.loss_smoothing)
        self.mtp_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.auxiliary_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.reasoning_loss_fn = nn.CrossEntropyLoss(reduction='mean')

        # Anti-hallucination loss components
        self.unlikelihood_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.contrastive_loss_fn = nn.CosineEmbeddingLoss(reduction='mean')

        # Loss scaling and adaptation
        self.current_scaling_factor = self.loss_weights.fp4_scaling_factor
        self.loss_history = []
        self.gradient_history = []

        # Dynamic weight adjustment
        self.step_count = 0
        self.weight_adjustment_interval = 100

        logging.info(f"✅ Multi-Component Loss System initialized")
        logging.info(f"  • Main weight: {self.loss_weights.main_weight}")
        logging.info(f"  • MTP weight: {self.loss_weights.mtp_weight}")
        logging.info(f"  • Auxiliary weight: {self.loss_weights.auxiliary_weight}")
        logging.info(f"  • Reasoning weight: {self.loss_weights.reasoning_weight}")
        logging.info(f"  • Anti-hallucination weight: {self.loss_weights.anti_hallucination_weight}")
        logging.info(f"  • FP4 scaling factor: {self.current_scaling_factor}")

    def calculate_main_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate main next-token prediction l
        Args:
            logits: Model predictions [batch, seq, vocab]
            targets: Target tokens [batch, seq]

        Returns:
            Main loss tensor
        """
        # Reshape for loss calculation
        flat_logits = logits.contiguous().view(-1, logits.size(-1))
        flat_targets = targets.contiguous().view(-1)

        # Calculate cross-entropy loss with label smoothing
        main_loss = self.main_loss_fn(flat_logits, flat_targets)

        return main_loss

    def calculate_mtp_loss(self, mtp_logits: List[torch.Tensor], targets: torch.Tensor,
                          mtp_weights: List[float]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate multi-token prediction loss

        Args:
            mtp_logits: List of prediction logits for each future position
            targets: Target token sequence
            mtp_weights: Weights for each prediction head

        Returns:
            - total_mtp_loss: Combined MTP loss
            - individual_mtp_losses: Individual losses for each prediction head
        """
        if not mtp_logits:
            return torch.tensor(0.0, device=targets.device), {}

        individual_losses = {}
        batch_size, seq_len = targets.shape

        for i, (logits, weight) in enumerate(zip(mtp_logits, mtp_weights)):
            step_name = f't+{i+1}'

            if i + 1 >= seq_len:
                individual_losses[step_name] = torch.tensor(0.0, device=targets.device)
                continue

            # Get target tokens for this prediction step
            target_tokens = targets[:, i+1:]
            pred_logits = logits[:, :target_tokens.shape[1], :]

            if target_tokens.numel() == 0:
                individual_losses[step_name] = torch.tensor(0.0, device=targets.device)
                continue

            # Calculate weighted loss
            step_loss = self.mtp_loss_fn(
                pred_logits.contiguous().view(-1, pred_logits.size(-1)),
                target_tokens.contiguous().view(-1)
            )
            individual_losses[step_name] = step_loss * weight

        # Total MTP loss
        total_mtp_loss = sum(individual_losses.values())

        return total_mtp_loss, individual_losses

    def calculate_auxiliary_loss(self, router_logits: Optional[torch.Tensor],
                               expert_usage: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate MoE auxiliary loss for load balancing

        Args:
            router_logits: Router prediction logits [batch, seq, n_experts]
            expert_usage: Expert usage statistics

        Returns:
            Auxiliary loss tensor
        """
        if router_logits is None:
            return torch.tensor(0.0)

        batch_size, seq_len, n_experts = router_logits.shape

        # Load balancing loss - encourage uniform expert usage
        router_probs = F.softmax(router_logits, dim=-1)

        # Calculate expert usage frequency
        expert_freq = torch.mean(router_probs, dim=(0, 1))  # [n_experts]

        # Target uniform distribution
        uniform_freq = torch.ones_like(expert_freq) / n_experts

        # KL divergence loss for load balancing
        aux_loss = F.kl_div(
            torch.log(expert_freq + 1e-8),
            uniform_freq,
            reduction='batchmean'
        )

        # Add router z-loss for stability (from Switch Transformer paper)
        if hasattr(self.config, 'router_z_loss') and self.config.router_z_loss > 0:
            router_z_loss = torch.mean(torch.square(torch.logsumexp(router_logits, dim=-1)))
            aux_loss = aux_loss + self.config.router_z_loss * router_z_loss

        return aux_loss

    def calculate_reasoning_loss(self, reasoning_logits: torch.Tensor, targets: torch.Tensor,
                               level_contributions: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate hierarchical reasoning loss

        Args:
            reasoning_logits: Final reasoning predictions
            targets: Target tokens
            level_contributions: Contributions from each reasoning level

        Returns:
            - total_reasoning_loss: Combined reasoning loss
            - individual_reasoning_losses: Losses for each reasoning level
        """
        if reasoning_logits is None:
            return torch.tensor(0.0), {}

        # Main reasoning loss
        main_reasoning_loss = self.reasoning_loss_fn(
            reasoning_logits.contiguous().view(-1, reasoning_logits.size(-1)),
            targets.contiguous().view(-1)
        )

        individual_losses = {'main': main_reasoning_loss}

        # Individual level losses if available
        if level_contributions:
            level_weights = {'low': 0.3, 'mid': 0.2, 'high': 0.1}

            for level, weight in level_weights.items():
                if level in level_contributions:
                    level_logits = level_contributions[level]
                    level_loss = self.reasoning_loss_fn(
                        level_logits.contiguous().view(-1, level_logits.size(-1)),
                        targets.contiguous().view(-1)
                    )
                    individual_losses[level] = level_loss * weight

        # Total reasoning loss
        total_reasoning_loss = sum(individual_losses.values())

        return total_reasoning_loss, individual_losses

    def calculate_anti_hallucination_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                                        forbidden_tokens: Optional[List[int]] = None,
                                        factual_consistency_targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate anti-hallucination loss

        Args:
            logits: Model predictions
            targets: Target tokens
            forbidden_tokens: List of forbidden token IDs
            factual_consistency_targets: Targets for factual consistency

        Returns:
            Anti-hallucination loss tensor
        """
        total_anti_halluc_loss = torch.tensor(0.0, device=logits.device)

        # Unlikelihood loss for forbidden tokens
        if forbidden_tokens and len(forbidden_tokens) > 0:
            # Create mask for forbidden tokens
            forbidden_mask = torch.zeros_like(targets, dtype=torch.bool)
            for token_id in forbidden_tokens:
                forbidden_mask |= (targets == token_id)

            if forbidden_mask.any():
                # Calculate unlikelihood loss for forbidden tokens
                token_losses = self.unlikelihood_loss_fn(
                    logits.contiguous().view(-1, logits.size(-1)),
                    targets.contiguous().view(-1)
                )
                token_losses = token_losses.view(targets.shape)

                # Apply unlikelihood loss (negative log likelihood)
                forbidden_losses = token_losses[forbidden_mask]
                if forbidden_losses.numel() > 0:
                    unlikelihood_loss = -torch.mean(torch.log(1 - torch.exp(-forbidden_losses) + 1e-8))
                    total_anti_halluc_loss = total_anti_halluc_loss + unlikelihood_loss

        # Factual consistency loss (contrastive learning)
        if factual_consistency_targets is not None:
            # Extract hidden representations (simplified)
            batch_size, seq_len, vocab_size = logits.shape

            # Use logits as representations for contrastive learning
            pred_representations = torch.mean(logits, dim=1)  # [batch, vocab_size]
            target_representations = torch.mean(factual_consistency_targets, dim=1)

            # Positive pairs (same sequence)
            positive_labels = torch.ones(batch_size, device=logits.device)

            # Contrastive loss
            contrastive_loss = self.contrastive_loss_fn(
                pred_representations, target_representations, positive_labels
            )
            total_anti_halluc_loss = total_anti_halluc_loss + contrastive_loss

        return total_anti_halluc_loss

    def apply_loss_scaling(self, loss: torch.Tensor, use_fp4: bool = False) -> torch.Tensor:
        """
        Apply loss scaling to prevent underflow in FP4 training

        Args:
            loss: Input loss tensor
            use_fp4: Whether FP4 quantization is being used

        Returns:
            Scaled loss tensor
        """
        if not use_fp4:
            return loss

        # Apply scaling factor
        scaled_loss = loss * self.current_scaling_factor

        # Adaptive scaling based on loss magnitude
        if self.loss_weights.adaptive_scaling:
            loss_magnitude = torch.abs(loss).item()

            # Adjust scaling factor if loss is too small or too large
            if loss_magnitude < 1e-7:  # Too small, increase scaling
                self.current_scaling_factor = min(self.current_scaling_factor * 1.1, 100.0)
            elif loss_magnitude > 1e2:  # Too large, decrease scaling
                self.current_scaling_factor = max(self.current_scaling_factor * 0.9, 1.0)

        return scaled_loss

    def adjust_weights_dynamically(self, component_losses: Dict[str, torch.Tensor]):
        """
        Dynamically adjust loss weights based on component magnitudes

        Args:
            component_losses: Dictionary of component losses
        """
        if not self.loss_weights.dynamic_weighting:
            return

        if self.step_count % self.weight_adjustment_interval == 0:
            # Calculate relative magnitudes
            loss_magnitudes = {}
            for component, loss in component_losses.items():
                if isinstance(loss, torch.Tensor) and loss.numel() > 0:
                    loss_magnitudes[component] = loss.detach().item()

            if len(loss_magnitudes) > 1:
                # Normalize weights based on relative magnitudes
                total_magnitude = sum(loss_magnitudes.values())

                for component in loss_magnitudes:
                    relative_magnitude = loss_magnitudes[component] / total_magnitude

                    # Adjust weights (simple heuristic)
                    if component == 'main':
                        self.loss_weights.main_weight = max(0.5, min(2.0, 1.0 / relative_magnitude))
                    elif component == 'mtp':
                        self.loss_weights.mtp_weight = max(0.1, min(1.0, 0.5 / relative_magnitude))
                    # Add more component adjustments as needed

    def forward(self, model_outputs: Dict[str, Any], targets: torch.Tensor,
                use_fp4: bool = False) -> LossOutput:
        """
        Calculate complete multi-component loss

        Args:
            model_outputs: Dictionary containing model outputs
            targets: Target token sequence
            use_fp4: Whether FP4 quantization is being used

        Returns:
            LossOutput with all loss components and statistics
        """
        device = targets.device
        component_losses = {}
        weighted_losses = {}

        # 1. Main loss (always present)
        if 'logits' in model_outputs:
            main_loss = self.calculate_main_loss(model_outputs['logits'], targets)
            component_losses['main'] = main_loss
        else:
            component_losses['main'] = torch.tensor(0.0, device=device)

        # 2. MTP loss
        if 'mtp_logits' in model_outputs and model_outputs['mtp_logits']:
            mtp_weights = getattr(self.config, 'mtp_loss_weights', [1.0, 0.5, 0.25, 0.125])
            mtp_loss, individual_mtp = self.calculate_mtp_loss(
                model_outputs['mtp_logits'], targets, mtp_weights
            )
            component_losses['mtp'] = mtp_loss
            component_losses.update({f'mtp_{k}': v for k, v in individual_mtp.items()})
        else:
            component_losses['mtp'] = torch.tensor(0.0, device=device)

        # 3. Auxiliary loss (MoE)
        if 'router_logits' in model_outputs:
            aux_loss = self.calculate_auxiliary_loss(
                model_outputs['router_logits'],
                model_outputs.get('expert_usage')
            )
            component_losses['auxiliary'] = aux_loss
        else:
            component_losses['auxiliary'] = torch.tensor(0.0, device=device)

        # 4. Reasoning loss (HRM)
        if 'reasoning_logits' in model_outputs:
            reasoning_loss, individual_reasoning = self.calculate_reasoning_loss(
                model_outputs['reasoning_logits'], targets,
                model_outputs.get('level_contributions')
            )
            component_losses['reasoning'] = reasoning_loss
            component_losses.update({f'reasoning_{k}': v for k, v in individual_reasoning.items()})
        else:
            component_losses['reasoning'] = torch.tensor(0.0, device=device)

        # 5. Anti-hallucination loss
        if 'logits' in model_outputs:
            forbidden_tokens = getattr(self.config, 'forbidden_tokens', None)
            anti_halluc_loss = self.calculate_anti_hallucination_loss(
                model_outputs['logits'], targets, forbidden_tokens
            )
            component_losses['anti_hallucination'] = anti_halluc_loss
        else:
            component_losses['anti_hallucination'] = torch.tensor(0.0, device=device)

        # Increment step count
        self.step_count += 1

        # Dynamic weight adjustment
        self.adjust_weights_dynamically(component_losses)

        # Apply weights
        weighted_losses['main'] = component_losses['main'] * self.loss_weights.main_weight
        weighted_losses['mtp'] = component_losses['mtp'] * self.loss_weights.mtp_weight
        weighted_losses['auxiliary'] = component_losses['auxiliary'] * self.loss_weights.auxiliary_weight
        weighted_losses['reasoning'] = component_losses['reasoning'] * self.loss_weights.reasoning_weight
        weighted_losses['anti_hallucination'] = component_losses['anti_hallucination'] * self.loss_weights.anti_hallucination_weight

        # Calculate total loss
        total_loss = sum(weighted_losses.values())

        # Apply scaling for FP4
        scaled_total_loss = self.apply_loss_scaling(total_loss, use_fp4)

        # Calculate loss statistics
        loss_stats = {
            'total_loss_magnitude': total_loss.item(),
            'scaling_factor': self.current_scaling_factor,
            'component_ratios': {
                k: (v.item() / total_loss.item() if total_loss.item() > 0 else 0.0)
                for k, v in component_losses.items()
                if isinstance(v, torch.Tensor) and v.numel() > 0
            },
            'gradient_norm': None,  # Will be filled by training loop
            'step_count': self.step_count
        }

        # Store loss history for tracking
        self.loss_history.append(total_loss.item())
        if len(self.loss_history) > 1000:  # Keep last 1000 steps
            self.loss_history = self.loss_history[-1000:]

        return LossOutput(
            total_loss=scaled_total_loss,
            component_losses=component_losses,
            weighted_losses=weighted_losses,
            loss_weights={
                'main': self.loss_weights.main_weight,
                'mtp': self.loss_weights.mtp_weight,
                'auxiliary': self.loss_weights.auxiliary_weight,
                'reasoning': self.loss_weights.reasoning_weight,
                'anti_hallucination': self.loss_weights.anti_hallucination_weight
            },
            scaling_factor=self.current_scaling_factor,
            loss_stats=loss_stats
        )

    def get_loss_statistics(self) -> Dict[str, Any]:
        """Get comprehensive loss statistics"""
        if not self.loss_history:
            return {}

        recent_losses = self.loss_history[-100:] if len(self.loss_history) >= 100 else self.loss_history

        return {
            'current_loss': self.loss_history[-1] if self.loss_history else 0.0,
            'average_loss_100': np.mean(recent_losses),
            'loss_std_100': np.std(recent_losses),
            'loss_trend': self._calculate_loss_trend(),
            'scaling_factor': self.current_scaling_factor,
            'total_steps': self.step_count,
            'loss_history_length': len(self.loss_history)
        }

    def _calculate_loss_trend(self) -> float:
        """Calculate loss trend (positive = increasing, negative = decreasing)"""
        if len(self.loss_history) < 10:
            return 0.0

        recent_losses = self.loss_history[-50:] if len(self.loss_history) >= 50 else self.loss_history
        x = np.arange(len(recent_losses))

        # Linear regression to find trend
        if len(recent_losses) > 1:
            slope, _ = np.polyfit(x, recent_losses, 1)
            return float(slope)

        return 0.0

    def reset_loss_history(self):
        """Reset loss tracking history"""
        self.loss_history = []
        self.gradient_history = []
        self.step_count = 0
        logging.info("🔄 Loss history reset")


def create_loss_system(config, loss_weights: Optional[LossWeights] = None) -> MultiComponentLoss:
    """Factory function to create multi-component loss system"""
    return MultiComponentLoss(config, loss_weights)


def test_loss_system():
    """Test the multi-component loss system"""
    print("🧪 Testing Multi-Component Loss System")

    # Mock config
    class MockConfig:
        loss_smoothing = 0.1
        mtp_loss_weights = [1.0, 0.5, 0.25, 0.125]
        router_z_loss = 1e-4
        forbidden_tokens = [100, 200, 300]  # Mock forbidden tokens

    config = MockConfig()

    # Create loss system
    print("✅ Testing Loss System Initialization...")
    loss_weights = LossWeights(
        main_weight=1.0,
        mtp_weight=0.5,
        auxiliary_weight=0.01,
        reasoning_weight=0.1,
        anti_hallucination_weight=0.25,
        fp4_scaling_factor=10.0
    )

    loss_system = create_loss_system(config, loss_weights)
    print(f"  • Loss system initialized with {len(loss_weights.__dict__)} weight parameters")

    # Test individual loss components
    print("✅ Testing Individual Loss Components...")

    batch_size, seq_len, vocab_size = 4, 32, 1000
    device = torch.device('cpu')

    # Mock model outputs
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Test main loss
    main_loss = loss_system.calculate_main_loss(logits, targets)
    print(f"  • Main loss: {main_loss.item():.6f}")

    # Test MTP loss
    mtp_logits = [
        torch.randn(batch_size, seq_len, vocab_size),
        torch.randn(batch_size, seq_len, vocab_size),
        torch.randn(batch_size, seq_len, vocab_size),
        torch.randn(batch_size, seq_len, vocab_size)
    ]
    mtp_loss, individual_mtp = loss_system.calculate_mtp_loss(mtp_logits, targets, config.mtp_loss_weights)
    print(f"  • MTP loss: {mtp_loss.item():.6f}")
    print(f"  • Individual MTP losses: {len(individual_mtp)}")

    # Test auxiliary loss
    router_logits = torch.randn(batch_size, seq_len, 4)  # 4 experts
    aux_loss = loss_system.calculate_auxiliary_loss(router_logits)
    print(f"  • Auxiliary loss: {aux_loss.item():.6f}")

    # Test reasoning loss
    reasoning_logits = torch.randn(batch_size, seq_len, vocab_size)
    level_contributions = {
        'low': torch.randn(batch_size, seq_len, vocab_size),
        'mid': torch.randn(batch_size, seq_len, vocab_size),
        'high': torch.randn(batch_size, seq_len, vocab_size)
    }
    reasoning_loss, individual_reasoning = loss_system.calculate_reasoning_loss(
        reasoning_logits, targets, level_contributions
    )
    print(f"  • Reasoning loss: {reasoning_loss.item():.6f}")
    print(f"  • Individual reasoning losses: {len(individual_reasoning)}")

    # Test anti-hallucination loss
    anti_halluc_loss = loss_system.calculate_anti_hallucination_loss(
        logits, targets, config.forbidden_tokens
    )
    print(f"  • Anti-hallucination loss: {anti_halluc_loss.item():.6f}")

    # Test complete loss calculation
    print("✅ Testing Complete Loss Calculation...")

    model_outputs = {
        'logits': logits,
        'mtp_logits': mtp_logits,
        'router_logits': router_logits,
        'reasoning_logits': reasoning_logits,
        'level_contributions': level_contributions
    }

    # Test without FP4
    loss_output = loss_system(model_outputs, targets, use_fp4=False)
    print(f"  • Total loss (no FP4): {loss_output.total_loss.item():.6f}")
    print(f"  • Component losses: {len(loss_output.component_losses)}")
    print(f"  • Scaling factor: {loss_output.scaling_factor}")

    # Test with FP4
    loss_output_fp4 = loss_system(model_outputs, targets, use_fp4=True)
    print(f"  • Total loss (FP4): {loss_output_fp4.total_loss.item():.6f}")
    print(f"  • FP4 scaling factor: {loss_output_fp4.scaling_factor}")

    # Test loss statistics
    print("✅ Testing Loss Statistics...")

    # Run multiple steps to build history
    for step in range(10):
        loss_output = loss_system(model_outputs, targets)

    stats = loss_system.get_loss_statistics()
    print(f"  • Average loss (last 100): {stats['average_loss_100']:.6f}")
    print(f"  • Loss trend: {stats['loss_trend']:.6f}")
    print(f"  • Total steps: {stats['total_steps']}")

    # Test loss scaling
    print("✅ Testing Loss Scaling...")
    small_loss = torch.tensor(1e-8)
    large_loss = torch.tensor(1e3)

    scaled_small = loss_system.apply_loss_scaling(small_loss, use_fp4=True)
    scaled_large = loss_system.apply_loss_scaling(large_loss, use_fp4=True)

    print(f"  • Small loss scaled: {small_loss.item():.2e} -> {scaled_small.item():.2e}")
    print(f"  • Large loss scaled: {large_loss.item():.2e} -> {scaled_large.item():.2e}")

    print("🎉 All loss system tests passed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    test_loss_system()