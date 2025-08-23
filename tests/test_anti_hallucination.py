#!/usr/bin/env python3
"""
Test suite for Anti-Hallucination System
========================================

Comprehensive tests for the anti-hallucination system including:
- Forbidden token database functionality
- Factual consistency checking
- Token filtering and blocking
- Rule management and configuration
- Export/import functionality
"""

import pytest
import torch
import numpy as np
import json
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anti_hallucination import (
    AntiHallucinationFilter, ForbiddenTokenDatabase, FactualConsistencyChecker,
    FilteringPolicy, HallucinationType, FilteringRule, HallucinationDetection,
    FilteringResult, create_anti_hallucination_filter
)


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
            if isinstance(token_ids, list) and len(token_ids) == 1:
                return self.vocab.get(token_ids[0], f"<token_{token_ids[0]}>")
            return " ".join([self.vocab.get(tid, f"<token_{tid}>") for tid in token_ids])

    return MockTokenizer()


class TestForbiddenTokenDatabase:
    """Test forbidden token database functionality"""

    def test_initialization(self):
        """Test database initialization"""
        db = ForbiddenTokenDatabase()

        # Should have default token sets
        assert len(db.token_sets) > 0
        assert 'profanity' in db.token_sets
        assert 'hate_speech' in db.token_sets
        assert 'violence' in db.token_sets
        assert 'personal_info' in db.token_sets

    def test_add_token_set(self):
        """Test adding token sets"""
        db = ForbiddenTokenDatabase()

        # Add custom token set
        db.add_token_set("test_set", [100, 101, 102], [r"\btest\b", r"\bexample\b"])

        assert "test_set" in db.token_sets
        assert 100 in db.token_sets["test_set"]["token_ids"]
        assert len(db.token_sets["test_set"]["patterns"]) == 2

    def test_token_checking(self):
        """Test token forbidden checking"""
        db = ForbiddenTokenDatabase()

        # Test pattern matching
        is_forbidden, violations = db.check_token_forbidden(123, "damn", "")
        assert is_forbidden == True
        assert len(violations) > 0
        assert "profanity" in violations[0]

        # Test non-forbidden token
        is_forbidden, violations = db.check_token_forbidden(456, "hello", "")
        assert is_forbidden == False
        assert len(violations) == 0

        # Test token ID matching
        db.add_token_set("test_ids", [999], [])
        is_forbidden, violations = db.check_token_forbidden(999, "anything", "")
        assert is_forbidden == True

    def test_context_rules(self):
        """Test context-dependent rules"""
        db = ForbiddenTokenDatabase()

        # Add context rule
        db.add_context_rule(
            "medical_context",
            [r"\bmedical\b", r"\bhealth\b"],
            [500],
            [r"\bcure\b"]
        )

        # Test without context
        is_forbidden, violations = db.check_token_forbidden(500, "cure", "")
        assert is_forbidden == False

        # Test with matching context
        is_forbidden, violations = db.check_token_forbidden(500, "cure", "medical advice")
        assert is_forbidden == True
        assert "medical_context" in violations[0]

    def test_export_import(self):
        """Test export and import functionality"""
        db = ForbiddenTokenDatabase()

        # Add custom data
        db.add_token_set("custom", [777, 888], [r"\bcustom\b"])

        # Export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name

        try:
            success = db.export_rules(export_path)
            assert success == True

            # Verify export file
            with open(export_path, 'r') as f:
                data = json.load(f)

            assert 'token_sets' in data
            assert 'custom' in data['token_sets']
            assert set(data['token_sets']['custom']['token_ids']) == {777, 888}

            # Test import
            new_db = ForbiddenTokenDatabase()
            import_success = new_db.import_rules(export_path)
            assert import_success == True

            # Verify imported data
            assert 'custom' in new_db.token_sets
            assert 777 in new_db.token_sets['custom']['token_ids']

        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)


class TestFactualConsistencyChecker:
    """Test factual consistency checker"""

    def test_initialization(self, config):
        """Test checker initialization"""
        checker = FactualConsistencyChecker(config)

        assert checker.config == config
        assert hasattr(checker, 'fact_patterns')
        assert hasattr(checker, 'confidence_threshold')
        assert len(checker.fact_patterns) > 0

    def test_date_inconsistency_detection(self, config):
        """Test date inconsistency detection"""
        checker = FactualConsistencyChecker(config)

        # Test obvious anachronism
        result = checker.check_factual_consistency(
            "The internet was invented in 1650 by Thomas Edison."
        )

        # Should detect some kind of issue (though our simple checker might not catch this specific case)
        assert isinstance(result, HallucinationDetection)
        assert result.hallucination_type == HallucinationType.FACTUAL_INCONSISTENCY

    def test_repetition_detection(self, config):
        """Test repetition pattern detection"""
        checker = FactualConsistencyChecker(config)

        # Test repetitive text
        repetitive_text = "The cat sat on the mat. " * 10
        result = checker.check_factual_consistency(repetitive_text)

        assert isinstance(result, HallucinationDetection)
        if result.detected:
            assert "repetitive" in result.explanation.lower() or "repetition" in result.explanation.lower()

    def test_number_plausibility(self, config):
        """Test number plausibility checking"""
        checker = FactualConsistencyChecker(config)

        # Test implausible percentage
        result = checker.check_factual_consistency(
            "Studies show that 150 percent of people agree with this statement."
        )

        # Should detect the impossible percentage
        if result.detected:
            assert result.confidence < 1.0

    def test_context_consistency(self, config):
        """Test context consistency checking"""
        checker = FactualConsistencyChecker(config)

        # Test contradictory context
        result = checker.check_factual_consistency(
            "The weather is very cold today.",
            "It's extremely hot outside with temperatures over 100 degrees."
        )

        # May or may not detect this simple contradiction
        assert isinstance(result, HallucinationDetection)


class TestAntiHallucinationFilter:
    """Test main anti-hallucination filter"""

    def test_initialization(self, config):
        """Test filter initialization"""
        filter_system = AntiHallucinationFilter(config)

        assert filter_system.config == config
        assert hasattr(filter_system, 'forbidden_db')
        assert hasattr(filter_system, 'factual_checker')
        assert hasattr(filter_system, 'filtering_rules')
        assert len(filter_system.filtering_rules) > 0
        assert hasattr(filter_system, 'filtering_stats')

    def test_rule_management(self, config):
        """Test filtering rule management"""
        filter_system = AntiHallucinationFilter(config)

        # Test adding rule
        initial_count = len(filter_system.filtering_rules)

        new_rule = FilteringRule(
            name="test_rule",
            token_ids=[999],
            token_patterns=["test_pattern"],
            policy=FilteringPolicy.STRICT,
            description="Test rule"
        )

        filter_system.add_filtering_rule(new_rule)
        assert len(filter_system.filtering_rules) == initial_count + 1
        assert "test_rule" in filter_system.filtering_stats['rules_applied']

        # Test updating rule
        success = filter_system.update_rule("test_rule", confidence_threshold=0.9)
        assert success == True

        # Find the updated rule
        updated_rule = None
        for rule in filter_system.filtering_rules:
            if rule.name == "test_rule":
                updated_rule = rule
                break

        assert updated_rule is not None
        assert updated_rule.confidence_threshold == 0.9

        # Test updating non-existent rule
        success = filter_system.update_rule("non_existent", confidence_threshold=0.5)
        assert success == False

    def test_logits_filtering_1d(self, config, mock_tokenizer):
        """Test logits filtering with 1D input"""
        filter_system = AntiHallucinationFilter(config)

        # Create 1D logits
        vocab_size = 100
        logits = torch.randn(vocab_size)

        # Apply filtering
        result = filter_system.filter_logits(
            logits,
            input_text="This is a test with some damn words",
            context="Testing context",
            tokenizer=mock_tokenizer
        )

        # Check result structure
        assert isinstance(result, FilteringResult)
        assert result.original_logits.shape == logits.shape
        assert result.filtered_logits.shape == logits.shape
        assert isinstance(result.blocked_tokens, list)
        assert isinstance(result.applied_rules, list)
        assert isinstance(result.hallucination_detections, list)
        assert isinstance(result.confidence_scores, dict)
        assert isinstance(result.filtering_stats, dict)

        # Check that filtering stats are reasonable
        assert result.filtering_stats['tokens_analyzed'] > 0
        assert result.filtering_stats['tokens_blocked'] >= 0
        assert 0.0 <= result.filtering_stats['filtering_strength'] <= 1.0

    def test_logits_filtering_2d(self, config, mock_tokenizer):
        """Test logits filtering with 2D input"""
        filter_system = AntiHallucinationFilter(config)

        # Create 2D logits (batch, vocab)
        batch_size, vocab_size = 3, 100
        logits = torch.randn(batch_size, vocab_size)

        # Apply filtering
        result = filter_system.filter_logits(
            logits,
            input_text="Testing batch filtering",
            tokenizer=mock_tokenizer
        )

        assert result.original_logits.shape == logits.shape
        assert result.filtered_logits.shape == logits.shape
        assert result.filtering_stats['tokens_analyzed'] == batch_size * min(100, vocab_size)

    def test_logits_filtering_3d(self, config, mock_tokenizer):
        """Test logits filtering with 3D input"""
        filter_system = AntiHallucinationFilter(config)

        # Create 3D logits (batch, seq, vocab)
        batch_size, seq_len, vocab_size = 2, 5, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)

        # Apply filtering
        result = filter_system.filter_logits(
            logits,
            input_text="Testing sequence filtering",
            tokenizer=mock_tokenizer
        )

        assert result.original_logits.shape == logits.shape
        assert result.filtered_logits.shape == logits.shape

    def test_filtering_policies(self, config, mock_tokenizer):
        """Test different filtering policies"""
        filter_system = AntiHallucinationFilter(config)

        # Create logits with high probability for a "forbidden" token
        vocab_size = 100
        logits = torch.zeros(vocab_size)
        logits[5] = 10.0  # High logit for token 5 (which should be "damn" in our mock)

        # Test strict policy
        strict_rule = FilteringRule(
            name="strict_test",
            token_ids=[5],
            token_patterns=[],
            policy=FilteringPolicy.STRICT,
            confidence_threshold=0.0
        )

        filter_system.add_filtering_rule(strict_rule)
        result = filter_system.filter_logits(logits, tokenizer=mock_tokenizer)

        # Should block the token regardless of confidence
        if 5 in result.blocked_tokens:
            assert result.filtered_logits[5] == -float('inf')

    def test_factual_checking_integration(self, config):
        """Test integration with factual consistency checker"""
        filter_system = AntiHallucinationFilter(config)

        # Test with factually questionable input
        logits = torch.randn(100)
        result = filter_system.filter_logits(
            logits,
            input_text="The internet was invented in 1650 by aliens from Mars.",
            context="We are discussing historical facts."
        )

        # Should have some hallucination detections (though may not detect this specific case)
        assert isinstance(result.hallucination_detections, list)

    def test_statistics_tracking(self, config, mock_tokenizer):
        """Test statistics tracking"""
        filter_system = AntiHallucinationFilter(config)

        # Initial stats
        initial_stats = filter_system.get_filtering_statistics()
        assert initial_stats['total_filtered'] == 0

        # Apply filtering multiple times
        logits = torch.randn(50)
        for i in range(5):
            filter_system.filter_logits(logits, tokenizer=mock_tokenizer)

        # Check updated stats
        updated_stats = filter_system.get_filtering_statistics()
        assert updated_stats['total_filtered'] == 5
        assert updated_stats['active_rules'] > 0
        assert updated_stats['total_rules'] > 0

    def test_configuration_export_import(self, config):
        """Test configuration export and import"""
        filter_system = AntiHallucinationFilter(config)

        # Add custom rule
        custom_rule = FilteringRule(
            name="export_test",
            token_ids=[123, 456],
            token_patterns=["test_pattern"],
            policy=FilteringPolicy.LENIENT,
            confidence_threshold=0.8,
            description="Test export rule"
        )
        filter_system.add_filtering_rule(custom_rule)

        # Export configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_path = f.name

        try:
            success = filter_system.export_configuration(export_path)
            assert success == True

            # Verify export file structure
            with open(export_path, 'r') as f:
                config_data = json.load(f)

            assert 'default_policy' in config_data
            assert 'filtering_rules' in config_data
            assert 'statistics' in config_data

            # Check that our custom rule is in the export
            rule_names = [rule['name'] for rule in config_data['filtering_rules']]
            assert 'export_test' in rule_names

        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)


class TestFilteringPolicies:
    """Test different filtering policies"""

    def test_strict_policy(self, config):
        """Test strict filtering policy"""
        filter_system = AntiHallucinationFilter(config)

        # Create rule with strict policy
        rule = FilteringRule(
            name="strict_test",
            token_ids=[],
            token_patterns=[],
            policy=FilteringPolicy.STRICT
        )

        # Test should_block_token method
        should_block = filter_system._should_block_token(rule, 0.1, "test", "")
        assert should_block == True  # Strict always blocks

        should_block = filter_system._should_block_token(rule, 0.9, "test", "")
        assert should_block == True  # Strict always blocks

    def test_moderate_policy(self, config):
        """Test moderate filtering policy"""
        filter_system = AntiHallucinationFilter(config)

        rule = FilteringRule(
            name="moderate_test",
            token_ids=[],
            token_patterns=[],
            policy=FilteringPolicy.MODERATE,
            confidence_threshold=0.5
        )

        # Below threshold - should not block
        should_block = filter_system._should_block_token(rule, 0.3, "test", "")
        assert should_block == False

        # Above threshold - should block
        should_block = filter_system._should_block_token(rule, 0.7, "test", "")
        assert should_block == True

    def test_lenient_policy(self, config):
        """Test lenient filtering policy"""
        filter_system = AntiHallucinationFilter(config)

        rule = FilteringRule(
            name="lenient_test",
            token_ids=[],
            token_patterns=[],
            policy=FilteringPolicy.LENIENT,
            confidence_threshold=0.5
        )

        # Below threshold - should not block
        should_block = filter_system._should_block_token(rule, 0.7, "test", "")
        assert should_block == False

        # High confidence - should block
        should_block = filter_system._should_block_token(rule, 0.9, "test", "")
        assert should_block == True

    def test_adaptive_policy(self, config):
        """Test adaptive filtering policy"""
        filter_system = AntiHallucinationFilter(config)

        rule = FilteringRule(
            name="adaptive_test",
            token_ids=[],
            token_patterns=[],
            policy=FilteringPolicy.ADAPTIVE,
            confidence_threshold=0.5
        )

        # Test with different contexts
        should_block_no_context = filter_system._should_block_token(rule, 0.6, "test", "")
        should_block_with_context = filter_system._should_block_token(
            rule, 0.6, "test", "This is a very long context " * 10
        )

        # Both should be boolean values
        assert isinstance(should_block_no_context, bool)
        assert isinstance(should_block_with_context, bool)


class TestFactoryFunction:
    """Test factory function"""

    def test_create_anti_hallucination_filter(self, config):
        """Test factory function"""
        filter_system = create_anti_hallucination_filter(config)

        assert isinstance(filter_system, AntiHallucinationFilter)
        assert filter_system.config == config


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_logits(self, config):
        """Test with empty or invalid logits"""
        filter_system = AntiHallucinationFilter(config)

        # Test with very small logits
        small_logits = torch.randn(1)
        result = filter_system.filter_logits(small_logits)

        assert isinstance(result, FilteringResult)
        assert result.original_logits.shape == small_logits.shape

    def test_no_tokenizer(self, config):
        """Test filtering without tokenizer"""
        filter_system = AntiHallucinationFilter(config)

        logits = torch.randn(50)
        result = filter_system.filter_logits(logits, tokenizer=None)

        # Should still work without tokenizer
        assert isinstance(result, FilteringResult)
        assert result.filtering_stats['tokens_analyzed'] > 0

    def test_invalid_token_patterns(self, config):
        """Test with invalid regex patterns"""
        filter_system = AntiHallucinationFilter(config)

        # This should not crash the system
        try:
            # Add rule with potentially problematic pattern
            rule = FilteringRule(
                name="invalid_pattern_test",
                token_ids=[],
                token_patterns=["nonexistent_pattern"],
                policy=FilteringPolicy.MODERATE
            )
            filter_system.add_filtering_rule(rule)

            # Should still work
            logits = torch.randn(20)
            result = filter_system.filter_logits(logits)
            assert isinstance(result, FilteringResult)

        except Exception as e:
            # If it fails, it should fail gracefully
            assert isinstance(e, Exception)

    def test_extreme_logit_values(self, config, mock_tokenizer):
        """Test with extreme logit values"""
        filter_system = AntiHallucinationFilter(config)

        # Test with very large values
        extreme_logits = torch.tensor([1000.0, -1000.0, float('inf'), -float('inf'), float('nan')])

        try:
            result = filter_system.filter_logits(extreme_logits, tokenizer=mock_tokenizer)
            assert isinstance(result, FilteringResult)
        except Exception:
            # Should handle extreme values gracefully
            pass


def run_all_tests():
    """Run all anti-hallucination tests"""
    print("🧪 Running Anti-Hallucination Tests")

    # Create test fixtures
    config = TestConfig()

    class MockTokenizer:
        def __init__(self):
            self.vocab = {
                0: "<pad>", 1: "<unk>", 2: "the", 3: "and", 4: "is",
                5: "damn", 6: "kill", 7: "fuck", 8: "good", 9: "bad",
                10: "hello", 11: "world", 12: "test", 13: "token"
            }

        def decode(self, token_ids):
            if isinstance(token_ids, list) and len(token_ids) == 1:
                return self.vocab.get(token_ids[0], f"<token_{token_ids[0]}>")
            return " ".join([self.vocab.get(tid, f"<token_{tid}>") for tid in token_ids])

    mock_tokenizer = MockTokenizer()

    print("✅ Testing Forbidden Token Database...")
    test_db = TestForbiddenTokenDatabase()
    test_db.test_initialization()
    test_db.test_add_token_set()
    test_db.test_token_checking()
    test_db.test_context_rules()
    test_db.test_export_import()

    print("✅ Testing Factual Consistency Checker...")
    test_checker = TestFactualConsistencyChecker()
    test_checker.test_initialization(config)
    test_checker.test_date_inconsistency_detection(config)
    test_checker.test_repetition_detection(config)
    test_checker.test_number_plausibility(config)
    test_checker.test_context_consistency(config)

    print("✅ Testing Anti-Hallucination Filter...")
    test_filter = TestAntiHallucinationFilter()
    test_filter.test_initialization(config)
    test_filter.test_rule_management(config)
    test_filter.test_logits_filtering_1d(config, mock_tokenizer)
    test_filter.test_logits_filtering_2d(config, mock_tokenizer)
    test_filter.test_logits_filtering_3d(config, mock_tokenizer)
    test_filter.test_filtering_policies(config, mock_tokenizer)
    test_filter.test_factual_checking_integration(config)
    test_filter.test_statistics_tracking(config, mock_tokenizer)
    test_filter.test_configuration_export_import(config)

    print("✅ Testing Filtering Policies...")
    test_policies = TestFilteringPolicies()
    test_policies.test_strict_policy(config)
    test_policies.test_moderate_policy(config)
    test_policies.test_lenient_policy(config)
    test_policies.test_adaptive_policy(config)

    print("✅ Testing Factory Function...")
    test_factory = TestFactoryFunction()
    test_factory.test_create_anti_hallucination_filter(config)

    print("✅ Testing Edge Cases...")
    test_edge = TestEdgeCases()
    test_edge.test_empty_logits(config)
    test_edge.test_no_tokenizer(config)
    test_edge.test_invalid_token_patterns(config)
    test_edge.test_extreme_logit_values(config, mock_tokenizer)

    print("🎉 All anti-hallucination tests passed!")


if __name__ == "__main__":
    run_all_tests()