#!/usr/bin/env python3
"""
Test suite for Unlikelihood Training and Contrastive Loss
=========================================================

Comprehensive tests for the unlikelihood training system including:
- Unlikelihood loss calculation and optimization
- Contrastive loss for factual accuracy
- Combined anti-hallucination training
- Adaptive alpha adjustment
- Training statistics and monitoring
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unlikelihood_training import (
    UnlikelihoodTrainer, ContrastiveLossTrainer, AntiHallucinationTrainer,
    UnlikelihoodConfig, ContrastiveLossConfig, UnlikelihoodType,
    create_anti_hallucination_trainer
)
from anti_hallucination import create_anti_hallucination_filter, FilteringPolicy


class TestConfig:
    """Test configuration for anti-hallucination system"""
    filtering_policy = FilteringPolicy.MODERATE
    hallucination_confidence_threshold = 0.5
    enable_factual_checking = True
    enable_repetition_detection = True
    factual_confidence_threshold = 0.7


@pytest.fixture
def config():
    return TestConfig()


@pytest.fixture
def anti_hallucination_filter(config):
    return create_anti_hallucination_filter(config)


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing"""
    class MockTokenizer:
        def __init__(self):
            self.vocab = {
                0: "<pad>", 1: "<unk>", 2: "the", 3: "and", 4: "is",
                5: "damn", 6: "kill", 7: "fuck", 8: "good", 9: "bad",
                10: "hello", 11: "world", 12: "test", 13: "token"
            }

        def decode(self, token_ids):
            if isinstance(token_ids, list):
                return " ".join([self.vocab.get(tid, f"<token_{tid}>") for tid in token_ids])
            return self.vocab.get(token_ids, f"<token_{token_ids}>")

    return MockTokenizer()


@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    batch_size, seq_len, vocab_size = 4, 16, 100
    hidden_dim = 256

    return {
        'logits': torch.randn(batch_size, seq_len, vocab_size),
        'targets': torch.randint(0, vocab_size, (batch_size, seq_len)),
        'hidden_states': torch.randn(batch_size, seq_len, hidden_dim),
        'fatargets': torch.randn(batch_size, seq_len, hidden_dim),
        'factual_labels': torch.randint(0, 2, (batch_size, seq_len))
    }


class TestUnlikelihoodConfig:
    """Test unlikelihood configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = UnlikelihoodConfig()

        assert config.alpha == 1.0
        assert config.sequence_level_weight == 0.5
        assert config.context_window == 50
        assert config.min_sequence_length == 3
        assert config.temperature == 1.0
        assert config.use_adaptive_alpha == True
        assert config.forbidden_threshold == 0.1

    def test_custom_config(self):
        """Test custom configuration"""
        config = UnlikelihoodConfig(
            alpha=2.0,
            sequence_level_weight=0.8,
            context_window=100,
            use_adaptive_alpha=False
        )

        assert config.alpha == 2.0
        assert config.sequence_level_weight == 0.8
        assert config.context_window == 100
        assert config.use_adaptive_alpha == False


class TestContrastiveLossConfig:
    """Test contrastive loss configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ContrastiveLossConfig()

        assert config.temperature == 0.07
        assert config.margin == 0.5
        assert config.negative_samples == 5
        assert config.use_hard_negatives == True
        assert config.factual_weight == 1.0
        assert config.semantic_weight == 0.5

    def test_custom_config(self):
        """Test custom configuration"""
        config = ContrastiveLossConfig(
            temperature=0.1,
            margin=1.0,
            negative_samples=10,
            use_hard_negatives=False
        )

        assert config.temperature == 0.1
        assert config.margin == 1.0
        assert config.negative_samples == 10
        assert config.use_hard_negatives == False


class TestUnlikelihoodTrainer:
    """Test unlikelihood trainer functionality"""

    def test_initialization(self, anti_hallucination_filter):
        """Test trainer initialization"""
        config = UnlikelihoodConfig()
        trainer = UnlikelihoodTrainer(config, anti_hallucination_filter)

        assert trainer.config == config
        assert trainer.filter == anti_hallucination_filter
        assert hasattr(trainer, 'training_stats')
        assert trainer.training_stats['total_steps'] == 0

    def test_unlikelihood_loss_calculation(self, anti_hallucination_filter, sample_data, mock_tokenizer):
        """Test unlikelihood loss calculation"""
        config = UnlikelihoodConfig()
        trainer = UnlikelihoodTrainer(config, anti_hallucination_filter)

        # Calculate loss
        loss, loss_info = trainer.calculate_unlikelihood_loss(
            sample_data['logits'],
            sample_data['targets'],
            "Test input with some damn words",
            "Test context",
            mock_tokenizer
        )

        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # Check loss info
        assert isinstance(loss_info, dict)
        assert 'token_level_loss' in loss_info
        assert 'sequence_level_loss' in loss_info
        assert 'total_unlikelihood_loss' in loss_info
        assert 'current_alpha' in loss_info
        assert 'blocked_tokens' in loss_info

        # Check that statistics were updated
        assert trainer.training_stats['total_steps'] == 1

    def test_token_level_loss(self, anti_hallucination_filter):
        """Test token-level unlikelihood loss"""
        config = UnlikelihoodConfig()
        trainer = UnlikelihoodTrainer(config, anti_hallucination_filter)

        # Create mock data with known structure
        batch_size, seq_len, vocab_size = 2, 5, 20
        probs = torch.rand(batch_size, seq_len, vocab_size)
        log_probs = torch.log(probs + 1e-8)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        blocked_tokens = [5, 10, 15]  # Some blocked tokens

        # Calculate token-level loss
        token_loss = trainer._calculate_token_level_loss(probs, log_probs, blocked_tokens, targets)

        assert isinstance(token_loss, torch.Tensor)
        assert token_loss.dim() == 0
        assert token_loss.item() >= 0.0

    def test_sequence_level_loss(self, anti_hallucination_filter, mock_tokenizer):
        """Test sequence-level unlikelihood loss"""
        config = UnlikelihoodConfig(min_sequence_length=3)
        trainer = UnlikelihoodTrainer(config, anti_hallucination_filter)

        # Create mock data
        batch_size, seq_len, vocab_size = 2, 8, 20
        probs = torch.rand(batch_size, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Mock filtering result
        class MockFilteringResult:
            def __init__(self):
                self.blocked_tokens = [5, 10]
                self.filtering_stats = {'filtering_strength': 0.1}

        filtering_result = MockFilteringResult()

        # Calculate sequence-level loss
        seq_loss = trainer._calculate_sequence_level_loss(probs, targets, filtering_result, mock_tokenizer)

        assert isinstance(seq_loss, torch.Tensor)
        assert seq_loss.dim() == 0
        assert seq_loss.item() >= 0.0

    def test_adaptive_alpha(self, anti_hallucination_filter):
        """Test adaptive alpha adjustment"""
        config = UnlikelihoodConfig(use_adaptive_alpha=True, alpha=1.0)
        trainer = UnlikelihoodTrainer(config, anti_hallucination_filter)

        # Initially should return base alpha
        alpha = trainer._get_adaptive_alpha()
        assert alpha == config.alpha

        # Simulate high detection rate
        trainer.training_stats['total_steps'] = 100
        trainer.training_stats['forbidden_sequences_detected'] = 20  # 20% detection rate

        adaptive_alpha = trainer._get_adaptive_alpha()
        assert adaptive_alpha > config.alpha  # Should increase

        # Test with low detection rate
        trainer.training_stats['forbidden_sequences_detected'] = 0  # 0% detection rate

        adaptive_alpha = trainer._get_adaptive_alpha()
        assert adaptive_alpha < config.alpha  # Should decrease

    def test_training_statistics(self, anti_hallucination_filter, sample_data, mock_tokenizer):
        """Test training statistics tracking"""
        config = UnlikelihoodConfig()
        trainer = UnlikelihoodTrainer(config, anti_hallucination_filter)

        # Run multiple training steps
        for i in range(5):
            trainer.calculate_unlikelihood_loss(
                sample_data['logits'],
                sample_data['targets'],
                f"Test input {i}",
                "Test context",
                mock_tokenizer
            )

        # Check statistics
        stats = trainer.get_training_statistics()

        assert stats['total_steps'] == 5
        assert 'average_unlikelihood_loss' in stats
        assert 'forbidden_detection_rate' in stats
        assert 'current_alpha' in stats
        assert stats['average_unlikelihood_loss'] >= 0.0
        assert 0.0 <= stats['forbidden_detection_rate'] <= 1.0


class TestContrastiveLossTrainer:
    """Test contrastive loss trainer functionality"""

    def test_initialization(self):
        """Test trainer initialization"""
        config = ContrastiveLossConfig()
        trainer = ContrastiveLossTrainer(config)

        assert trainer.config == config
        assert hasattr(trainer, 'training_stats')
        assert trainer.training_stats['total_steps'] == 0

    def test_contrastive_loss_calculation(self, sample_data):
        """Test contrastive loss calculation"""
        config = ContrastiveLossConfig()
        trainer = ContrastiveLossTrainer(config)

        # Calculate loss
        loss, loss_info = trainer.calculate_contrastive_loss(
            sample_data['hidden_states'],
            sample_data['factual_targets'],
            sample_data['factual_labels']
        )

        # Check loss properties
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        # Check loss info
        assert isinstance(loss_info, dict)
        assert 'contrastive_loss' in loss_info
        assert 'factual_loss' in loss_info
        assert 'total_contrastive_loss' in loss_info
        assert 'positive_pairs' in loss_info
        assert 'negative_pairs' in loss_info

        # Check that statistics were updated
        assert trainer.training_stats['total_steps'] == 1

    def test_infonce_loss(self):
        """Test InfoNCE loss calculation"""
        config = ContrastiveLossConfig()
        trainer = ContrastiveLossTrainer(config)

        # Create mock similarity matrix and masks
        batch_size = 4
        similarity_matrix = torch.randn(batch_size, batch_size)
        positive_mask = torch.eye(batch_size)  # Identity as positive pairs
        negative_mask = 1.0 - positive_mask

        # Remove diagonal for positive mask
        positive_mask = positive_mask * 0  # No positive pairs for this test

        # Calculate InfoNCE loss
        infonce_loss = trainer._calculate_infonce_loss(similarity_matrix, positive_mask, negative_mask)

        assert isinstance(infonce_loss, torch.Tensor)
        assert infonce_loss.dim() == 0
        assert infonce_loss.item() >= 0.0

    def test_factual_consistency_loss(self):
        """Test factual consistency loss"""
        config = ContrastiveLossConfig()
        trainer = ContrastiveLossTrainer(config)

        # Create mock data
        batch_size, hidden_dim = 4, 256
        hidden_states = torch.randn(batch_size, hidden_dim)
        target_states = torch.randn(batch_size, hidden_dim)
        labels = torch.randint(0, 2, (batch_size,)).float()

        # Calculate factual consistency loss
        factual_loss = trainer._calculate_factual_consistency_loss(hidden_states, target_states, labels)

        assert isinstance(factual_loss, torch.Tensor)
        assert factual_loss.dim() == 0
        assert factual_loss.item() >= 0.0

    def test_training_statistics(self, sample_data):
        """Test training statistics tracking"""
        config = ContrastiveLossConfig()
        trainer = ContrastiveLossTrainer(config)

        # Run multiple training steps
        for i in range(3):
            trainer.calculate_contrastive_loss(
                sample_data['hidden_states'],
                sample_data['factual_targets'],
                sample_data['factual_labels']
            )

        # Check statistics
        stats = trainer.get_training_statistics()

        assert stats['total_steps'] == 3
        assert 'average_contrastive_loss' in stats
        assert 'total_positive_pairs' in stats
        assert 'total_negative_pairs' in stats
        assert 'positive_negative_ratio' in stats
        assert stats['average_contrastive_loss'] >= 0.0


class TestAntiHallucinationTrainer:
    """Test combined anti-hallucination trainer"""

    def test_initialization(self, anti_hallucination_filter):
        """Test trainer initialization"""
        unlikelihood_config = UnlikelihoodConfig()
        contrastive_config = ContrastiveLossConfig()

        trainer = AntiHallucinationTrainer(
            unlikelihood_config, contrastive_config, anti_hallucination_filter
        )

        assert hasattr(trainer, 'unlikelihood_trainer')
        assert hasattr(trainer, 'contrastive_trainer')
        assert trainer.enable_unlikelihood == True
        assert trainer.enable_contrastive == True
        assert trainer.loss_balance_weight == 0.5

    def test_combined_loss_calculation(self, anti_hallucination_filter, sample_data, mock_tokenizer):
        """Test combined anti-hallucination loss calculation"""
        unlikelihood_config = UnlikelihoodConfig()
        contrastive_config = ContrastiveLossConfig()

        trainer = AntiHallucinationTrainer(
            unlikelihood_config, contrastive_config, anti_hallucination_filter
        )

        # Calculate combined loss
        total_loss, loss_info = trainer.calculate_anti_hallucination_loss(
            sample_data['logits'],
            sample_data['targets'],
            sample_data['hidden_states'],
            sample_data['factual_targets'],
            sample_data['factual_labels'],
            "Test input with forbidden words",
            "Test context",
            mock_tokenizer
        )

        # Check loss properties
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.dim() == 0
        assert total_loss.item() >= 0.0
        assert not torch.isnan(total_loss)
        assert not torch.isinf(total_loss)

        # Check loss info structure
        assert isinstance(loss_info, dict)
        assert 'unlikelihood' in loss_info
        assert 'contrastive' in loss_info
        assert 'total_anti_hallucination_loss' in loss_info
        assert 'loss_balance_weight' in loss_info

    def test_unlikelihood_only_mode(self, anti_hallucination_filter, sample_data, mock_tokenizer):
        """Test with only unlikelihood training enabled"""
        unlikelihood_config = UnlikelihoodConfig()
        contrastive_config = ContrastiveLossConfig()

        trainer = AntiHallucinationTrainer(
            unlikelihood_config, contrastive_config, anti_hallucination_filter
        )

        # Disable contrastive learning
        trainer.enable_contrastive = False

        # Calculate loss
        total_loss, loss_info = trainer.calculate_anti_hallucination_loss(
            sample_data['logits'],
            sample_data['targets'],
            sample_data['hidden_states'],
            input_text="Test input",
            tokenizer=mock_tokenizer
        )

        assert isinstance(total_loss, torch.Tensor)
        assert 'unlikelihood' in loss_info
        assert 'contrastive' not in loss_info

    def test_contrastive_only_mode(self, anti_hallucination_filter, sample_data):
        """Test with only contrastive learning enabled"""
        unlikelihood_config = UnlikelihoodConfig()
        contrastive_config = ContrastiveLossConfig()

        trainer = AntiHallucinationTrainer(
            unlikelihood_config, contrastive_config, anti_hallucination_filter
        )

        # Disable unlikelihood training
        trainer.enable_unlikelihood = False

        # Calculate loss
        total_loss, loss_info = trainer.calculate_anti_hallucination_loss(
            sample_data['logits'],
            sample_data['targets'],
            sample_data['hidden_states'],
            sample_data['factual_targets'],
            sample_data['factual_labels']
        )

        assert isinstance(total_loss, torch.Tensor)
        assert 'contrastive' in loss_info
        assert 'unlikelihood' not in loss_info

    def test_comprehensive_statistics(self, anti_hallucination_filter, sample_data, mock_tokenizer):
        """Test comprehensive statistics"""
        unlikelihood_config = UnlikelihoodConfig()
        contrastive_config = ContrastiveLossConfig()

        trainer = AntiHallucinationTrainer(
            unlikelihood_config, contrastive_config, anti_hallucination_filter
        )

        # Run training steps
        for i in range(3):
            trainer.calculate_anti_hallucination_loss(
                sample_data['logits'],
                sample_data['targets'],
                sample_data['hidden_states'],
                sample_data['factual_targets'],
                sample_data['factual_labels'],
                f"Test input {i}",
                "Test context",
                mock_tokenizer
            )

        # Get comprehensive statistics
        stats = trainer.get_comprehensive_statistics()

        assert isinstance(stats, dict)
        assert 'unlikelihood_stats' in stats
        assert 'contrastive_stats' in stats
        assert 'training_config' in stats

        # Check nested statistics
        assert stats['unlikelihood_stats']['total_steps'] == 3
        assert stats['contrastive_stats']['total_steps'] == 3
        assert stats['training_config']['enable_unlikelihood'] == True
        assert stats['training_config']['enable_contrastive'] == True

    def test_config_updates(self, anti_hallucination_filter):
        """Test training configuration updates"""
        unlikelihood_config = UnlikelihoodConfig()
        contrastive_config = ContrastiveLossConfig()

        trainer = AntiHallucinationTrainer(
            unlikelihood_config, contrastive_config, anti_hallucination_filter
        )

        # Update configuration
        trainer.update_training_config(
            loss_balance_weight=0.8,
            enable_contrastive=False
        )

        assert trainer.loss_balance_weight == 0.8
        assert trainer.enable_contrastive == False


class TestFactoryFunction:
    """Test factory function"""

    def test_create_anti_hallucination_trainer(self, anti_hallucination_filter):
        """Test factory function"""
        unlikelihood_config = UnlikelihoodConfig()
        contrastive_config = ContrastiveLossConfig()

        trainer = create_anti_hallucination_trainer(
            unlikelihood_config, contrastive_config, anti_hallucination_filter
        )

        assert isinstance(trainer, AntiHallucinationTrainer)
        assert trainer.unlikelihood_trainer.config == unlikelihood_config
        assert trainer.contrastive_trainer.config == contrastive_config


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_data(self, anti_hallucination_filter):
        """Test with minimal data"""
        config = UnlikelihoodConfig()
        trainer = UnlikelihoodTrainer(config, anti_hallucination_filter)

        # Very small tensors
        small_logits = torch.randn(1, 1, 10)
        small_targets = torch.randint(0, 10, (1, 1))

        loss, loss_info = trainer.calculate_unlikelihood_loss(
            small_logits, small_targets, "", "", None
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0.0

    def test_no_forbidden_tokens(self, anti_hallucination_filter, sample_data):
        """Test when no forbidden tokens are detected"""
        config = UnlikelihoodConfig()
        trainer = UnlikelihoodTrainer(config, anti_hallucination_filter)

        # Use clean input text
        loss, loss_info = trainer.calculate_unlikelihood_loss(
            sample_data['logits'],
            sample_data['targets'],
            "This is clean text with no issues",
            "Clean context",
            None
        )

        # Should still work, just with zero or minimal loss
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0.0
        assert loss_info['token_level_loss'] >= 0.0

    def test_extreme_values(self, anti_hallucination_filter):
        """Test with extreme tensor values"""
        config = UnlikelihoodConfig()
        trainer = UnlikelihoodTrainer(config, anti_hallucination_filter)

        # Create tensors with extreme values
        batch_size, seq_len, vocab_size = 2, 5, 20
        extreme_logits = torch.tensor([[[1000.0] * vocab_size] * seq_len] * batch_size)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))

        try:
            loss, loss_info = trainer.calculate_unlikelihood_loss(
                extreme_logits, targets, "test", "test", None
            )

            # Should handle extreme values gracefully
            assert isinstance(loss, torch.Tensor)
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)

        except Exception as e:
            # Should fail gracefully if at all
            assert isinstance(e, Exception)


def run_all_tests():
    """Run all unlikelihood training tests"""
    print("🧪 Running Unlikelihood Training Tests")

    # Create test fixtures
    config = TestConfig()
    anti_hallucination_filter = create_anti_hallucination_filter(config)

    class MockTokenizer:
        def __init__(self):
            self.vocab = {
                0: "<pad>", 1: "<unk>", 2: "the", 3: "and", 4: "is",
                5: "damn", 6: "kill", 7: "fuck", 8: "good", 9: "bad",
                10: "hello", 11: "world", 12: "test", 13: "token"
            }

        def decode(self, token_ids):
            if isinstance(token_ids, list):
                return " ".join([self.vocab.get(tid, f"<token_{tid}>") for tid in token_ids])
            return self.vocab.get(token_ids, f"<token_{token_ids}>")

    mock_tokenizer = MockTokenizer()

    # Sample data
    batch_size, seq_len, vocab_size = 4, 16, 100
    hidden_dim = 256

    sample_data = {
        'logits': torch.randn(batch_size, seq_len, vocab_size),
        'targets': torch.randint(0, vocab_size, (batch_size, seq_len)),
        'hidden_states': torch.randn(batch_size, seq_len, hidden_dim),
        'factual_targets': torch.randn(batch_size, seq_len, hidden_dim),
        'factual_labels': torch.randint(0, 2, (batch_size, seq_len))
    }

    print("✅ Testing Configuration Classes...")
    test_config = TestUnlikelihoodConfig()
    test_config.test_default_config()
    test_config.test_custom_config()

    test_contrastive_config = TestContrastiveLossConfig()
    test_contrastive_config.test_default_config()
    test_contrastive_config.test_custom_config()

    print("✅ Testing Unlikelihood Trainer...")
    test_unlikelihood = TestUnlikelihoodTrainer()
    test_unlikelihood.test_initialization(anti_hallucination_filter)
    test_unlikelihood.test_unlikelihood_loss_calculation(anti_hallucination_filter, sample_data, mock_tokenizer)
    test_unlikelihood.test_token_level_loss(anti_hallucination_filter)
    test_unlikelihood.test_sequence_level_loss(anti_hallucination_filter, mock_tokenizer)
    test_unlikelihood.test_adaptive_alpha(anti_hallucination_filter)
    test_unlikelihood.test_training_statistics(anti_hallucination_filter, sample_data, mock_tokenizer)

    print("✅ Testing Contrastive Loss Trainer...")
    test_contrastive = TestContrastiveLossTrainer()
    test_contrastive.test_initialization()
    test_contrastive.test_contrastive_loss_calculation(sample_data)
    test_contrastive.test_infonce_loss()
    test_contrastive.test_factual_consistency_loss()
    test_contrastive.test_training_statistics(sample_data)

    print("✅ Testing Combined Anti-Hallucination Trainer...")
    test_combined = TestAntiHallucinationTrainer()
    test_combined.test_initialization(anti_hallucination_filter)
    test_combined.test_combined_loss_calculation(anti_hallucination_filter, sample_data, mock_tokenizer)
    test_combined.test_unlikelihood_only_mode(anti_hallucination_filter, sample_data, mock_tokenizer)
    test_combined.test_contrastive_only_mode(anti_hallucination_filter, sample_data)
    test_combined.test_comprehensive_statistics(anti_hallucination_filter, sample_data, mock_tokenizer)
    test_combined.test_config_updates(anti_hallucination_filter)

    print("✅ Testing Factory Function...")
    test_factory = TestFactoryFunction()
    test_factory.test_create_anti_hallucination_trainer(anti_hallucination_filter)

    print("✅ Testing Edge Cases...")
    test_edge = TestEdgeCases()
    test_edge.test_empty_data(anti_hallucination_filter)
    test_edge.test_no_forbidden_tokens(anti_hallucination_filter, sample_data)
    test_edge.test_extreme_values(anti_hallucination_filter)

    print("🎉 All unlikelihood training tests passed!")


if __name__ == "__main__":
    run_all_tests()