#!/usr/bin/env python3
"""
Unlikelihood Training and Contrastive Loss for Anti-Hallucination
================================================================

Implements advanced training techniques for reducing hallucinations:
- Unlikelihood training for forbidden sequences
- Contrastive loss for factual accuracy
- Sequence-level anti-hallucination training
- Integration with existing loss systems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from anti_hallucination import AntiHallucinationFilter, HallucinationType


class UnlikelihoodType(Enum):
    """Types of unlikelihood training"""
    TOKEN_LEVEL = "token_level"
    SEQUENCE_LEVEL = "sequence_level"
    CONTEXT_AWARE = "context_aware"
    ADAPTIVE = "adaptive"


@dataclass
class UnlikelihoodConfig:
    """Configuration for unlikelihood training"""
    alpha: float = 1.0                    # Unlikelihood loss weight
    sequence_level_weight: float = 0.5    # Weight for sequence-level loss
    context_window: int = 50               # Context window for sequence evaluation
    min_sequence_length: int = 3           # Minimum sequence length for evaluation
    temperature: float = 1.0               # Temperature for probability calculation
    use_adaptive_alpha: bool = True        # Adaptive alpha based on training progress
    forbidden_threshold: float = 0.1       # Threshold for considering tokens forbidden


@dataclass
class ContrastiveLossConfig:
    """Configuration for contrastive loss"""
    temperature: float = 0.07              # Temperature for contrastive learning
    margin: float = 0.5                    # Margin for contrastive loss
    negative_samples: int = 5              # Number of negative samples
    use_hard_negatives: bool = True        # Use hard negative mining
    factual_weight: float = 1.0            # Weight for factual consistency
    semantic_weight: float = 0.5           # Weight for semantic consistency


class UnlikelihoodTrainer:
    """Implements unlikelihood training for anti-hallucination"""

    def __init__(self, config: UnlikelihoodConfig, anti_hallucination_filter: AntiHallucinationFilter):
        self.config = config
        self.filter = anti_hallucination_filter

        # Training statistics
        self.training_stats = {
            'total_steps': 0,
            'unlikelihood_loss_sum': 0.0,
            'forbidden_sequences_detected': 0,
            'alpha_adjustments': 0
        }

        logging.info("✅ Unlikelihood Trainer initialized")
        logging.info(f"  • Alpha: {config.alpha}")
        logging.info(f"  • Sequence weight: {config.sequence_level_weight}")
        logging.info(f"  • Context window: {config.context_window}")

    def calculate_unlikelihood_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                                  input_text: str = "", context: str = "",
                                  tokenizer = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Calculate unlikelihood loss for forbidden sequences

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            targets: Target token sequence [batch_size, seq_len]
            input_text: Input text for context
            context: Additional context
            tokenizer: Tokenizer for text conversion

        Returns:
            - unlikelihood_loss: Combined unlikelihood loss
            - loss_info: Detailed loss information
        """

        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device

        # Apply anti-hallucination filtering to identify forbidden tokens
        filtering_result = self.filter.filter_logits(
            logits, input_text, context, tokenizer
        )

        # Initialize loss components
        token_level_loss = torch.tensor(0.0, device=device)
        sequence_level_loss = torch.tensor(0.0, device=device)

        # Get probabilities from logits
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Token-level unlikelihood loss
        if filtering_result.blocked_tokens:
            token_level_loss = self._calculate_token_level_loss(
                probs, log_probs, filtering_result.blocked_tokens, targets
            )

        # Sequence-level unlikelihood loss
        sequence_level_loss = self._calculate_sequence_level_loss(
            probs, targets, filtering_result, tokenizer
        )

        # Adaptive alpha adjustment
        current_alpha = self._get_adaptive_alpha()

        # Combine losses
        total_unlikelihood_loss = (
            current_alpha * token_level_loss +
            self.config.sequence_level_weight * sequence_level_loss
        )

        # Update statistics
        self.training_stats['total_steps'] += 1
        self.training_stats['unlikelihood_loss_sum'] += total_unlikelihood_loss.item()
        if filtering_result.blocked_tokens:
            self.training_stats['forbidden_sequences_detected'] += 1

        # Prepare loss information
        loss_info = {
            'token_level_loss': token_level_loss.item(),
            'sequence_level_loss': sequence_level_loss.item(),
            'total_unlikelihood_loss': total_unlikelihood_loss.item(),
            'current_alpha': current_alpha,
            'blocked_tokens': filtering_result.blocked_tokens,
            'hallucination_detections': len(filtering_result.hallucination_detections),
            'filtering_strength': filtering_result.filtering_stats['filtering_strength']
        }

        return total_unlikelihood_loss, loss_info

    def _calculate_token_level_loss(self, probs: torch.Tensor, log_probs: torch.Tensor,
                                   blocked_tokens: List[int], targets: torch.Tensor) -> torch.Tensor:
        """Calculate token-level unlikelihood loss"""
        device = probs.device
        batch_size, seq_len, vocab_size = probs.shape

        # Create mask for blocked tokens
        blocked_mask = torch.zeros(vocab_size, device=device, dtype=torch.bool)
        blocked_mask[blocked_tokens] = True

        # Get probabilities for blocked tokens
        blocked_probs = probs[:, :, blocked_mask]  # [batch, seq, num_blocked]

        if blocked_probs.numel() == 0:
            return torch.tensor(0.0, device=device)

        # Unlikelihood loss: -log(1 - p) for forbidden tokens
        # Use numerical stability: -log(1 - p) ≈ p + p²/2 + ... for small p
        epsilon = 1e-8
        clamped_probs = torch.clamp(blocked_probs, min=epsilon, max=1.0 - epsilon)

        # Calculate -log(1 - p) with numerical stability
        unlikelihood_loss = -torch.log(1.0 - clamped_probs + epsilon)

        # Average over sequence and batch dimensions
        token_loss = torch.mean(unlikelihood_loss)

        return token_loss

    def _calculate_sequence_level_loss(self, probs: torch.Tensor, targets: torch.Tensor,
                                     filtering_result, tokenizer) -> torch.Tensor:
        """Calculate sequence-level unlikelihood loss"""
        device = probs.device
        batch_size, seq_len, vocab_size = probs.shape

        if not tokenizer or seq_len < self.config.min_sequence_length:
            return torch.tensor(0.0, device=device)

        sequence_losses = []

        for batch_idx in range(batch_size):
            # Extract sequences from this batch
            batch_targets = targets[batch_idx]
            batch_probs = probs[batch_idx]

            # Generate text for sequence evaluation
            try:
                generated_text = tokenizer.decode(batch_targets.tolist())
            except:
                continue

            # Check if sequence contains forbidden patterns
            is_forbidden, violations = self.filter.forbidden_db.check_token_forbidden(
                0, generated_text, ""
            )

            if is_forbidden:
                # Calculate sequence probability
                target_probs = batch_probs[range(seq_len), batch_targets]
                sequence_prob = torch.prod(target_probs)

                # Sequence-level unlikelihood: -log(1 - P(sequence))
                epsilon = 1e-8
                clamped_seq_prob = torch.clamp(sequence_prob, min=epsilon, max=1.0 - epsilon)
                seq_loss = -torch.log(1.0 - clamped_seq_prob + epsilon)

                sequence_losses.append(seq_loss)

        if sequence_losses:
            return torch.mean(torch.stack(sequence_losses))
        else:
            return torch.tensor(0.0, device=device)

    def _get_adaptive_alpha(self) -> float:
        """Get adaptive alpha based on training progress"""
        if not self.config.use_adaptive_alpha:
            return self.config.alpha

        # Adaptive alpha based on detection rate
        if self.training_stats['total_steps'] > 0:
            detection_rate = (
                self.training_stats['forbidden_sequences_detected'] /
                self.training_stats['total_steps']
            )

            # Increase alpha if detection rate is high
            if detection_rate > 0.1:
                adaptive_alpha = self.config.alpha * 1.5
                self.training_stats['alpha_adjustments'] += 1
            elif detection_rate < 0.01:
                adaptive_alpha = self.config.alpha * 0.8
            else:
                adaptive_alpha = self.config.alpha

            return min(adaptive_alpha, 5.0)  # Cap at 5.0

        return self.config.alpha

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        if self.training_stats['total_steps'] > 0:
            avg_loss = self.training_stats['unlikelihood_loss_sum'] / self.training_stats['total_steps']
            detection_rate = self.training_stats['forbidden_sequences_detected'] / self.training_stats['total_steps']
        else:
            avg_loss = 0.0
            detection_rate = 0.0

        return {
            'total_steps': self.training_stats['total_steps'],
            'average_unlikelihood_loss': avg_loss,
            'forbidden_detection_rate': detection_rate,
            'alpha_adjustments': self.training_stats['alpha_adjustments'],
            'current_alpha': self._get_adaptive_alpha()
        }


class ContrastiveLossTrainer:
    """Implements contrastive loss for factual accuracy"""

    def __init__(self, config: ContrastiveLossConfig):
        self.config = config

        # Training statistics
        self.training_stats = {
            'total_steps': 0,
            'contrastive_loss_sum': 0.0,
            'positive_pairs': 0,
            'negative_pairs': 0
        }

        logging.info("✅ Contrastive Loss Trainer initialized")
        logging.info(f"  • Temperature: {config.temperature}")
        logging.info(f"  • Margin: {config.margin}")
        logging.info(f"  • Negative samples: {config.negative_samples}")

    def calculate_contrastive_loss(self, hidden_states: torch.Tensor,
                                 factual_targets: torch.Tensor,
                                 factual_labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Calculate contrastive loss for factual accuracy

        Args:
            hidden_states: Hidden representations [batch_size, seq_len, hidden_dim]
            factual_targets: Target factual representations [batch_size, seq_len, hidden_dim]
            factual_labels: Binary labels for factual accuracy [batch_size, seq_len]

        Returns:
            - contrastive_loss: Contrastive loss value
            - loss_info: Detailed loss information
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device

        # Pool representations (mean pooling over sequence)
        pooled_hidden = torch.mean(hidden_states, dim=1)  # [batch_size, hidden_dim]
        pooled_targets = torch.mean(factual_targets, dim=1)  # [batch_size, hidden_dim]
        pooled_labels = torch.mean(factual_labels.float(), dim=1)  # [batch_size]

        # Normalize representations
        pooled_hidden = F.normalize(pooled_hidden, p=2, dim=1)
        pooled_targets = F.normalize(pooled_targets, p=2, dim=1)

        # Calculate similarity matrix
        similarity_matrix = torch.matmul(pooled_hidden, pooled_targets.T) / self.config.temperature

        # Create positive and negative pairs
        positive_mask = (pooled_labels.unsqueeze(1) == pooled_labels.unsqueeze(0)).float()
        negative_mask = 1.0 - positive_mask

        # Remove diagonal (self-similarity)
        identity_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask * (1.0 - identity_mask)

        # InfoNCE-style contrastive loss
        contrastive_loss = self._calculate_infonce_loss(
            similarity_matrix, positive_mask, negative_mask
        )

        # Factual consistency loss
        factual_loss = self._calculate_factual_consistency_loss(
            pooled_hidden, pooled_targets, pooled_labels
        )

        # Combine losses
        total_loss = (
            self.config.factual_weight * contrastive_loss +
            self.config.semantic_weight * factual_loss
        )

        # Update statistics
        self.training_stats['total_steps'] += 1
        self.training_stats['contrastive_loss_sum'] += total_loss.item()
        self.training_stats['positive_pairs'] += torch.sum(positive_mask).item()
        self.training_stats['negative_pairs'] += torch.sum(negative_mask).item()

        # Prepare loss information
        loss_info = {
            'contrastive_loss': contrastive_loss.item(),
            'factual_loss': factual_loss.item(),
            'total_contrastive_loss': total_loss.item(),
            'positive_pairs': torch.sum(positive_mask).item(),
            'negative_pairs': torch.sum(negative_mask).item(),
            'temperature': self.config.temperature
        }

        return total_loss, loss_info

    def _calculate_infonce_loss(self, similarity_matrix: torch.Tensor,
                               positive_mask: torch.Tensor,
                               negative_mask: torch.Tensor) -> torch.Tensor:
        """Calculate InfoNCE contrastive loss"""
        batch_size = similarity_matrix.shape[0]
        device = similarity_matrix.device

        # For each sample, calculate loss
        losses = []

        for i in range(batch_size):
            # Get positive and negative similarities for sample i
            pos_similarities = similarity_matrix[i] * positive_mask[i]
            neg_similarities = similarity_matrix[i] * negative_mask[i]

            # Get valid positive similarities (non-zero)
            valid_pos = pos_similarities[positive_mask[i] > 0]

            if len(valid_pos) == 0:
                continue

            # Get negative similarities
            valid_neg = neg_similarities[negative_mask[i] > 0]

            # Calculate InfoNCE loss for each positive
            for pos_sim in valid_pos:
                # Numerator: exp(positive similarity)
                numerator = torch.exp(pos_sim)

                # Denominator: exp(positive) + sum(exp(negatives))
                denominator = numerator + torch.sum(torch.exp(valid_neg))

                # InfoNCE loss: -log(numerator / denominator)
                loss = -torch.log(numerator / (denominator + 1e-8))
                losses.append(loss)

        if losses:
            return torch.mean(torch.stack(losses))
        else:
            return torch.tensor(0.0, device=device)

    def _calculate_factual_consistency_loss(self, hidden_states: torch.Tensor,
                                          target_states: torch.Tensor,
                                          labels: torch.Tensor) -> torch.Tensor:
        """Calculate factual consistency loss"""
        # Cosine similarity between hidden states and targets
        cosine_sim = F.cosine_similarity(hidden_states, target_states, dim=1)

        # Binary cross-entropy loss for factual accuracy
        # Positive labels should have high similarity, negative labels low similarity
        factual_loss = F.binary_cross_entropy_with_logits(
            cosine_sim, labels, reduction='mean'
        )

        return factual_loss

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        if self.training_stats['total_steps'] > 0:
            avg_loss = self.training_stats['contrastive_loss_sum'] / self.training_stats['total_steps']
        else:
            avg_loss = 0.0

        return {
            'total_steps': self.training_stats['total_steps'],
            'average_contrastive_loss': avg_loss,
            'total_positive_pairs': self.training_stats['positive_pairs'],
            'total_negative_pairs': self.training_stats['negative_pairs'],
            'positive_negative_ratio': (
                self.training_stats['positive_pairs'] /
                max(1, self.training_stats['negative_pairs'])
            )
        }


class AntiHallucinationTrainer:
    """Combined trainer for anti-hallucination techniques"""

    def __init__(self, unlikelihood_config: UnlikelihoodConfig,
                 contrastive_config: ContrastiveLossConfig,
                 anti_hallucination_filter: AntiHallucinationFilter):

        self.unlikelihood_trainer = UnlikelihoodTrainer(unlikelihood_config, anti_hallucination_filter)
        self.contrastive_trainer = ContrastiveLossTrainer(contrastive_config)

        # Combined training configuration
        self.enable_unlikelihood = True
        self.enable_contrastive = True
        self.loss_balance_weight = 0.5  # Balance between unlikelihood and contrastive

        logging.info("✅ Anti-Hallucination Trainer initialized")
        logging.info(f"  • Unlikelihood training: {self.enable_unlikelihood}")
        logging.info(f"  • Contrastive learning: {self.enable_contrastive}")

    def calculate_anti_hallucination_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                                        hidden_states: torch.Tensor,
                                        factual_targets: Optional[torch.Tensor] = None,
                                        factual_labels: Optional[torch.Tensor] = None,
                                        input_text: str = "", context: str = "",
                                        tokenizer = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Calculate combined anti-hallucination loss

        Args:
            logits: Model output logits
            targets: Target tokens
            hidden_states: Hidden representations
            factual_targets: Target factual representations (optional)
            factual_labels: Factual accuracy labels (optional)
            input_text: Input text
            context: Additional context
            tokenizer: Tokenizer

        Returns:
            - total_loss: Combined anti-hallucination loss
            - loss_info: Detailed loss information
        """
        device = logits.device
        total_loss = torch.tensor(0.0, device=device)
        loss_info = {}

        # Unlikelihood training loss
        if self.enable_unlikelihood:
            unlikelihood_loss, unlikelihood_info = self.unlikelihood_trainer.calculate_unlikelihood_loss(
                logits, targets, input_text, context, tokenizer
            )
            total_loss += self.loss_balance_weight * unlikelihood_loss
            loss_info['unlikelihood'] = unlikelihood_info

        # Contrastive learning loss
        if self.enable_contrastive and factual_targets is not None and factual_labels is not None:
            contrastive_loss, contrastive_info = self.contrastive_trainer.calculate_contrastive_loss(
                hidden_states, factual_targets, factual_labels
            )
            total_loss += (1.0 - self.loss_balance_weight) * contrastive_loss
            loss_info['contrastive'] = contrastive_info

        # Combined loss information
        loss_info['total_anti_hallucination_loss'] = total_loss.item()
        loss_info['loss_balance_weight'] = self.loss_balance_weight

        return total_loss, loss_info

    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        stats = {
            'unlikelihood_stats': self.unlikelihood_trainer.get_training_statistics(),
            'contrastive_stats': self.contrastive_trainer.get_training_statistics(),
            'training_config': {
                'enable_unlikelihood': self.enable_unlikelihood,
                'enable_contrastive': self.enable_contrastive,
                'loss_balance_weight': self.loss_balance_weight
            }
        }

        return stats

    def update_training_config(self, **kwargs):
        """Update training configuration"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logging.info(f"Updated {key} to {value}")


def create_anti_hallucination_trainer(unlikelihood_config: UnlikelihoodConfig,
                                    contrastive_config: ContrastiveLossConfig,
                                    anti_hallucination_filter: AntiHallucinationFilter) -> AntiHallucinationTrainer:
    """Factory function to create anti-hallucination trainer"""
    return AntiHallucinationTrainer(unlikelihood_config, contrastive_config, anti_hallucination_filter)


def test_unlikelihood_training():
    """Test the unlikelihood training system"""
    print("🧪 Testing Unlikelihood Training System")

    # Mock config and components
    from anti_hallucination import create_anti_hallucination_filter

    from anti_hallucination import FilteringPolicy

    class MockConfig:
        filtering_policy = FilteringPolicy.MODERATE
        hallucination_confidence_threshold = 0.5
        enable_factual_checking = True
        enable_repetition_detection = True
        factual_confidence_threshold = 0.7

    config = MockConfig()
    filter_system = create_anti_hallucination_filter(config)

    # Test configurations
    unlikelihood_config = UnlikelihoodConfig(
        alpha=1.0,
        sequence_level_weight=0.5,
        context_window=50,
        min_sequence_length=3
    )

    contrastive_config = ContrastiveLossConfig(
        temperature=0.07,
        margin=0.5,
        negative_samples=5
    )

    # Test unlikelihood trainer
    print("✅ Testing Unlikelihood Trainer...")
    unlikelihood_trainer = UnlikelihoodTrainer(unlikelihood_config, filter_system)

    # Create mock data
    batch_size, seq_len, vocab_size = 2, 10, 100
    hidden_dim = 256

    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Mock tokenizer
    class MockTokenizer:
        def decode(self, token_ids):
            return " ".join([f"token_{tid}" for tid in token_ids])

    tokenizer = MockTokenizer()

    # Test unlikelihood loss calculation
    unlikelihood_loss, loss_info = unlikelihood_trainer.calculate_unlikelihood_loss(
        logits, targets, "Test input with some damn words", "Test context", tokenizer
    )

    print(f"  • Unlikelihood loss: {unlikelihood_loss.item():.6f}")
    print(f"  • Token level loss: {loss_info['token_level_loss']:.6f}")
    print(f"  • Sequence level loss: {loss_info['sequence_level_loss']:.6f}")
    print(f"  • Current alpha: {loss_info['current_alpha']:.3f}")

    # Test contrastive trainer
    print("✅ Testing Contrastive Trainer...")
    contrastive_trainer = ContrastiveLossTrainer(contrastive_config)

    # Create mock factual data
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    factual_targets = torch.randn(batch_size, seq_len, hidden_dim)
    factual_labels = torch.randint(0, 2, (batch_size, seq_len))

    contrastive_loss, contrastive_info = contrastive_trainer.calculate_contrastive_loss(
        hidden_states, factual_targets, factual_labels
    )

    print(f"  • Contrastive loss: {contrastive_loss.item():.6f}")
    print(f"  • Factual loss: {contrastive_info['factual_loss']:.6f}")
    print(f"  • Positive pairs: {contrastive_info['positive_pairs']}")
    print(f"  • Negative pairs: {contrastive_info['negative_pairs']}")

    # Test combined trainer
    print("✅ Testing Combined Anti-Hallucination Trainer...")
    combined_trainer = create_anti_hallucination_trainer(
        unlikelihood_config, contrastive_config, filter_system
    )

    total_loss, combined_info = combined_trainer.calculate_anti_hallucination_loss(
        logits, targets, hidden_states, factual_targets, factual_labels,
        "Test input", "Test context", tokenizer
    )

    print(f"  • Total anti-hallucination loss: {total_loss.item():.6f}")
    print(f"  • Loss balance weight: {combined_info['loss_balance_weight']}")

    # Test statistics
    print("✅ Testing Training Statistics...")
    stats = combined_trainer.get_comprehensive_statistics()
    print(f"  • Unlikelihood steps: {stats['unlikelihood_stats']['total_steps']}")
    print(f"  • Contrastive steps: {stats['contrastive_stats']['total_steps']}")
    print(f"  • Detection rate: {stats['unlikelihood_stats']['forbidden_detection_rate']:.3f}")

    print("🎉 All unlikelihood training tests completed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    test_unlikelihood_training()