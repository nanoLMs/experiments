#!/usr/bin/env python3
"""
Multi-Token Prediction (MTP) System for NanoLM
==============================================

Implements multi-token prediction based on research:
"Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential"

Features:
- Simultaneous prediction of multiple future tokens (t+1, t+2, t+3, t+4)
- Weighted loss calculation with geometric decay
- Improved sample efficiency
- Integration with transformer architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from nanolm_model import QuantizedLinear


@dataclass
class MTPOutput:
    """Output structure for Multi-Token Prediction"""
    main_logits: torch.Tensor  # Standard next-token prediction
    mtp_logits: List[torch.Tensor]  # Multi-token predictions [t+1, t+2, t+3, ...]
    mtp_loss: torch.Tensor  # Combined MTP loss
    individual_losses: List[torch.Tensor]  # Individual losses for each prediction head
    prediction_confidence: torch.Tensor  # Confidence scores for predictions


class MultiTokenPredictionHead(nn.Module):
    """
    Single prediction head for one future token position

    Each head predicts tokens at a specific future position (t+k).
    """

    def __init__(self, config, prediction_step: int = 1):
        super().__init__()
        self.config = config
        self.prediction_step = prediction_step  # How many steps ahead this head predicts
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size

        # Quantization config
        quant_config = {            'use_bnb_4bit': config.use_bnb_4bit and config.bnb_4bit_quantize_heads,
            'compute_dtype': config.bnb_4bit_compute_dtype,
            'use_double_quant': config.bnb_4bit_use_double_quant,
            'quant_type': config.bnb_4bit_quant_type
        }

        # Prediction head architecture
        # Use a small intermediate layer to allow specialization for different time steps
        self.intermediate = QuantizedLinear(
            self.d_model,
            self.d_model // 2,
            bias=True,
            quantization_config=quant_config
        )

        self.prediction_head = QuantizedLinear(
            self.d_model // 2,
            self.vocab_size,
            bias=False,
            quantization_config=quant_config
        )

        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model // 2, eps=1e-5)

        # Temperature parameter for this prediction step (learnable)
        self.temperature = nn.Parameter(torch.ones(1))

        logging.info(f"✅ MTP Head initialized for t+{prediction_step}")

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through prediction head

        Returns:
            - logits: Prediction logits for this time step
            - confidence: Confidence score for this prediction
        """
        # Intermediate processing
        intermediate = self.intermediate(hidden_states)
        intermediate = F.gelu(intermediate)  # GELU activation
        intermediate = self.layer_norm(intermediate)
        intermediate = self.dropout(intermediate)

        # Final prediction
        logits = self.prediction_head(intermediate)

        # Apply learnable temperature
        logits = logits / self.temperature

        # Calculate confidence as entropy of prediction distribution
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        max_entropy = math.log(self.vocab_size)
        confidence = 1.0 - (entropy / max_entropy)  # Higher confidence = lower entropy

        return logits, confidence

    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get statistics for this prediction head"""
        return {
            'prediction_step': self.prediction_step,
            'temperature': self.temperature.item(),
            'parameters': sum(p.numel() for p in self.parameters())
        }


class MultiTokenPredictionHeads(nn.Module):
    """
    Complete Multi-Token Prediction system

    Manages multiple prediction heads for different future positions.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mtp_k = config.mtp_k  # Number of future tokens to predict
        self.loss_weights = config.mtp_loss_weights
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size

        # Ensure loss weights match mtp_k
        if len(self.loss_weights) < self.mtp_k:
            # Extend with geometric decay
            base_weight = self.loss_weights[0] if self.loss_weights else 1.0
            self.loss_weights = [base_weight * (0.5 ** i) for i in range(self.mtp_k)]

        # Create prediction heads for each future position
        self.prediction_heads = nn.ModuleList([
            MultiTokenPredictionHead(config, prediction_step=i+1)
            for i in range(self.mtp_k)
        ])

        # Shared feature extractor (optional)
        quant_config = {
            'use_bnb_4bit': config.use_bnb_4bit and config.bnb_4bit_quantize_heads,
            'compute_dtype': config.bnb_4bit_compute_dtype,
            'use_double_quant': config.bnb_4bit_use_double_quant,
            'quant_type': config.bnb_4bit_quant_type
        }

        self.feature_extractor = QuantizedLinear(
            self.d_model,
            self.d_model,
            bias=True,
            quantization_config=quant_config
        )

        self.feature_norm = nn.LayerNorm(self.d_model, eps=1e-5)

        logging.info(f"✅ MTP System initialized: {self.mtp_k} prediction heads")
        logging.info(f"  • Loss weights: {self.loss_weights}")

    def forward(self, hidden_states: torch.Tensor) -> MTPOutput:
        """
        Forward pass through all prediction heads

        Args:
            hidden_states: (batch_size, seq_len, d_model)

        Returns:
            MTPOutput with all predictions and losses
        """
        batch_size, seq_len, d_model = hidden_states.shape

        # Extract features for MTP
        features = self.feature_extractor(hidden_states)
        features = F.gelu(features)
        features = self.feature_norm(features)

        # Get predictions from all heads
        mtp_logits = []
        confidences = []

        for head in self.prediction_heads:
            logits, confidence = head(features)
            mtp_logits.append(logits)
            confidences.append(confidence)

        # Stack confidences
        prediction_confidence = torch.stack(confidences, dim=-1)  # (batch, seq, mtp_k)

        # Main logits are the first prediction (t+1)
        main_logits = mtp_logits[0]

        return MTPOutput(
            main_logits=main_logits,
            mtp_logits=mtp_logits,
            mtp_loss=torch.tensor(0.0, device=hidden_states.device),  # Will be computed in loss function
            individual_losses=[torch.tensor(0.0, device=hidden_states.device)] * self.mtp_k,
            prediction_confidence=prediction_confidence
        )

    def calculate_mtp_loss(self, mtp_logits: List[torch.Tensor], targets: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Calculate multi-token prediction loss

        Args:
            mtp_logits: List of prediction logits for each future position
            targets: Target token sequence (batch_size, seq_len)

        Returns:
            - total_mtp_loss: Weighted sum of all MTP losses
            - individual_losses: List of individual losses for each prediction head
        """
        batch_size, seq_len = targets.shape
        individual_losses = []

        # Calculate loss for each prediction head
        for i, (logits, weight) in enumerate(zip(mtp_logits, self.loss_weights)):
            if i + 1 >= seq_len:
                # Not enough sequence length for this prediction
                individual_losses.append(torch.tensor(0.0, device=targets.device))
                continue

            # Get target tokens for this prediction step
            # For t+k prediction, we need targets shifted by k positions
            target_tokens = targets[:, i+1:]  # Skip first i+1 tokens
            pred_logits = logits[:, :target_tokens.shape[1], :]  # Match sequence length

            if target_tokens.numel() == 0:
                individual_losses.append(torch.tensor(0.0, device=targets.device))
                continue

            # Calculate cross-entropy loss
            loss_fn = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fn(pred_logits.contiguous().view(-1, self.vocab_size),
                          target_tokens.contiguous().view(-1))

            individual_losses.append(loss * weight)

        # Total MTP loss is sum of weighted individual losses
        total_mtp_loss = sum(individual_losses)

        return total_mtp_loss, individual_losses

    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get statistics for all prediction heads"""
        return {
            'mtp_k': self.mtp_k,
            'loss_weights': self.loss_weights,
            'head_stats': [head.get_prediction_stats() for head in self.prediction_heads],
            'total_parameters': sum(p.numel() for p in self.parameters())
        }

    def get_prediction_accuracy(self, mtp_logits: List[torch.Tensor], targets: torch.Tensor) -> Dict[str, float]:
        """
        Calculate prediction accuracy for each head

        Args:
            mtp_logits: List of prediction logits
            targets: Target tokens

        Returns:
            Dictionary with accuracy for each prediction step
        """
        accuracies = {}

        for i, logits in enumerate(mtp_logits):
            if i + 1 >= targets.shape[1]:
                accuracies[f't+{i+1}'] = 0.0
                continue

            # Get predictions and targets
            predictions = torch.argmax(logits, dim=-1)
            target_tokens = targets[:, i+1:]
            pred_tokens = predictions[:, :target_tokens.shape[1]]

            if target_tokens.numel() == 0:
                accuracies[f't+{i+1}'] = 0.0
                continue

            # Calculate accuracy
            correct = (pred_tokens == target_tokens).float()
            accuracy = correct.mean().item()
            accuracies[f't+{i+1}'] = accuracy

        return accuracies


class MTPIntegratedModel(nn.Module):
    """
    NanoLM model with integrated Multi-Token Prediction

    Extends the base model with MTP capabilities.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Import here to avoid circular imports
        from nanolm_model import NanoLMModel

        # Base model
        self.base_model = NanoLMModel(config)

        # Multi-token prediction heads
        self.mtp_heads = MultiTokenPredictionHeads(config)

        logging.info(f"✅ MTP Integrated Model initialized")
        logging.info(f"  • Base model parameters: {self.base_model.get_num_params():,}")
        logging.info(f"  • MTP parameters: {sum(p.numel() for p in self.mtp_heads.parameters()):,}")

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                return_dict: bool = True) -> Dict[str, Any]:
        """Forward pass with MTP"""

        # Base model forward pass
        base_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Get hidden states from the last layer
        hidden_states = base_outputs.hidden_states[-1] if base_outputs.hidden_states else None
        if hidden_states is None:
            # If hidden states not available, use the states before final projection
            # We need to get them from the model's norm layer
            hidden_states = self.base_model.norm(base_outputs.logits)  # This is a hack for testing

        # Multi-token prediction
        mtp_outputs = self.mtp_heads(hidden_states)

        # Calculate losses if labels provided
        total_loss = None
        mtp_loss = None
        individual_mtp_losses = None

        if labels is not None:
            # Main loss (standard next-token prediction)
            main_loss_fn = nn.CrossEntropyLoss()
            main_loss = main_loss_fn(
                base_outputs.logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )

            # MTP loss
            mtp_loss, individual_mtp_losses = self.mtp_heads.calculate_mtp_loss(
                mtp_outputs.mtp_logits, labels
            )

            # Combined loss
            total_loss = main_loss + mtp_loss

        if return_dict:
            return {
                'logits': base_outputs.logits,
                'mtp_logits': mtp_outputs.mtp_logits,
                'hidden_states': base_outputs.hidden_states,
                'attention_weights': base_outputs.attention_weights,
                'loss': total_loss,
                'main_loss': main_loss if labels is not None else None,
                'mtp_loss': mtp_loss,
                'individual_mtp_losses': individual_mtp_losses,
                'prediction_confidence': mtp_outputs.prediction_confidence
            }
        else:
            return (
                base_outputs.logits,
                mtp_outputs.mtp_logits,
                base_outputs.hidden_states,
                total_loss
            )

    def get_num_params(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters())

    def get_mtp_stats(self) -> Dict[str, Any]:
        """Get MTP system statistics"""
        base_params = self.base_model.get_num_params()
        mtp_params = sum(p.numel() for p in self.mtp_heads.parameters())
        total_params = self.get_num_params()

        return {
            'base_model_params': base_params,
            'mtp_params': mtp_params,
            'total_params': total_params,
            'mtp_overhead': mtp_params / base_params,
            'mtp_heads_stats': self.mtp_heads.get_prediction_stats()
        }


def create_mtp_model(config) -> MTPIntegratedModel:
    """Factory function to create MTP-integrated model"""
    return MTPIntegratedModel(config)


def calculate_mtp_metrics(predictions: List[torch.Tensor], targets: torch.Tensor,
                         loss_weights: List[float]) -> Dict[str, Any]:
    """
    Calculate comprehensive MTP metrics

    Args:
        predictions: List of prediction tensors for each future position
        targets: Target token sequence
        loss_weights: Weights for each prediction head

    Returns:
        Dictionary with various MTP metrics
    """
    metrics = {
        'num_predictions': len(predictions),
        'loss_weights': loss_weights,
        'accuracies': {},
        'perplexities': {},
        'prediction_diversity': {},
        'temporal_consistency': 0.0
    }

    # Calculate accuracy and perplexity for each prediction head
    for i, pred_logits in enumerate(predictions):
        step_name = f't+{i+1}'

        if i + 1 >= targets.shape[1]:
            metrics['accuracies'][step_name] = 0.0
            metrics['perplexities'][step_name] = float('inf')
            metrics['prediction_diversity'][step_name] = 0.0
            continue

        # Get target tokens for this prediction step
        target_tokens = targets[:, i+1:]
        pred_tokens = pred_logits[:, :target_tokens.shape[1], :]

        if target_tokens.numel() == 0:
            continue

        # Accuracy
        predicted_ids = torch.argmax(pred_tokens, dim=-1)
        accuracy = (predicted_ids == target_tokens).float().mean().item()
        metrics['accuracies'][step_name] = accuracy

        # Perplexity
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(pred_tokens.contiguous().view(-1, pred_tokens.shape[-1]),
                      target_tokens.contiguous().view(-1))
        perplexity = torch.exp(loss).item()
        metrics['perplexities'][step_name] = perplexity

        # Prediction diversity (entropy of prediction distribution)
        probs = F.softmax(pred_tokens, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        avg_entropy = entropy.mean().item()
        metrics['prediction_diversity'][step_name] = avg_entropy

    # Temporal consistency: how consistent are predictions across time steps
    if len(predictions) > 1:
        consistency_scores = []
        for i in range(len(predictions) - 1):
            pred1 = torch.argmax(predictions[i], dim=-1)
            pred2 = torch.argmax(predictions[i+1], dim=-1)

            # Compare overlapping predictions
            if pred1.shape[1] > 1 and pred2.shape[1] > 0:
                overlap_len = min(pred1.shape[1] - 1, pred2.shape[1])
                if overlap_len > 0:
                    pred1_overlap = pred1[:, 1:1+overlap_len]
                    pred2_overlap = pred2[:, :overlap_len]
                    consistency = (pred1_overlap == pred2_overlap).float().mean().item()
                    consistency_scores.append(consistency)

        if consistency_scores:
            metrics['temporal_consistency'] = sum(consistency_scores) / len(consistency_scores)

    return metrics


def test_mtp_system():
    """Test the MTP system"""
    print("🧪 Testing MTP System")

    # Mock config for testing
    class MockConfig:
        d_model = 256
        vocab_size = 1000
        mtp_k = 4
        mtp_loss_weights = [1.0, 0.5, 0.25, 0.125]
        dropout = 0.1

        # Quantization settings
        use_bnb_4bit = False
        bnb_4bit_quantize_heads = False
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

    config = MockConfig()

    # Test individual prediction head
    print("✅ Testing Single Prediction Head...")
    head = MultiTokenPredictionHead(config, prediction_step=1)
    x = torch.randn(4, 32, config.d_model)
    logits, confidence = head(x)
    print(f"  • Head output shape: {logits.shape}")
    print(f"  • Confidence shape: {confidence.shape}")

    # Test multi-token prediction heads
    print("✅ Testing Multi-Token Prediction Heads...")
    mtp_heads = MultiTokenPredictionHeads(config)
    mtp_output = mtp_heads(x)
    print(f"  • Main logits shape: {mtp_output.main_logits.shape}")
    print(f"  • Number of MTP heads: {len(mtp_output.mtp_logits)}")
    print(f"  • Confidence shape: {mtp_output.prediction_confidence.shape}")

    # Test loss calculation
    print("✅ Testing MTP Loss Calculation...")
    targets = torch.randint(0, config.vocab_size, (4, 32))
    mtp_loss, individual_losses = mtp_heads.calculate_mtp_loss(mtp_output.mtp_logits, targets)
    print(f"  • MTP loss: {mtp_loss.item():.6f}")
    print(f"  • Individual losses: {[loss.item() for loss in individual_losses]}")

    # Test integrated model
    print("✅ Testing MTP Integrated Model...")
    model = create_mtp_model(config)

    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    outputs = model(input_ids, labels=labels)
    print(f"  • Model output logits shape: {outputs['logits'].shape}")
    print(f"  • Total loss: {outputs['loss'].item():.6f}")
    print(f"  • MTP loss: {outputs['mtp_loss'].item():.6f}")

    # Test MTP metrics
    print("✅ Testing MTP Metrics...")
    metrics = calculate_mtp_metrics(outputs['mtp_logits'], labels, config.mtp_loss_weights)
    print(f"  • Accuracies: {metrics['accuracies']}")
    print(f"  • Temporal consistency: {metrics['temporal_consistency']:.3f}")

    print("🎉 All MTP tests passed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    test_mtp_system()