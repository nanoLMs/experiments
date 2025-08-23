#!/usr/bin/env python3
"""
Anti-Hallucination System for NanoLM
===================================

Implements comprehensive anti-hallucination mechanisms including:
- Forbidden token filtering and blocking
- Factual consistency checking
- Confidence-based output validation
- Context-aware hallucination detection
- Configurable filtering policies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class FilteringPolicy(Enum):
    """Filtering policy options"""
    STRICT = "strict"           # Block all forbidden tokens
    MODERATE = "moderate"       # Block with confidence threshold
    LENIENT = "lenient"         # Warn but allow with low confidence
    ADAPTIVE = "adaptive"       # Adjust based on context


class HallucinationType(Enum):
    """Types of hallucination detection"""
    FORBIDDEN_TOKEN = "forbidden_token"
    FACTUAL_INCONSISTENCY = "factual_inconsistency"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    CONTEXT_MISMATCH = "context_mismatch"
    REPETITION_LOOP = "repetition_loop"


@dataclass
class FilteringRule:
    """Configuration for a filtering rule"""
    name: str
    token_ids: List[int]
    token_patterns: List[str]
    policy: FilteringPolicy
    confidence_threshold: float = 0.5
    context_dependent: bool = False
    description: str = ""
    enabled: bool = True


@dataclass
class HallucinationDetection:
    """Result of hallucination detection"""
    detected: bool
    hallucination_type: HallucinationType
    confidence: float
    affected_tokens: List[int]
    explanation: str
    suggested_action: str
    severity: float  # 0.0 to 1.0


@dataclass
class FilteringResult:
    """Result of token filtering"""
    original_logits: torch.Tensor
    filtered_logits: torch.Tensor
    blocked_tokens: List[int]
    applied_rules: List[str]
    hallucination_detections: List[HallucinationDetection]
    confidence_scores: Dict[int, float]
    filtering_stats: Dict[str, Any]


class ForbiddenTokenDatabase:
    """Database of forbidden tokens and patterns"""

    def __init__(self):
        self.token_sets = {}
        self.pattern_rules = {}
        self.context_rules = {}

        # Initialize with common problematic patterns
        self._initialize_default_rules()

        logging.info("✅ Forbidden Token Database initialized")

    def _initialize_default_rules(self):
        """Initialize with common forbidden token patterns"""

        # Harmful content categories
        self.add_token_set("profanity", [], [
            r"\b(fuck|shit|damn|hell|bitch|asshole)\b",
            r"\b(crap|piss|bastard|whore|slut)\b"
        ])

        self.add_token_set("hate_speech", [], [
            r"\b(nigger|faggot|retard|spic|chink)\b",
            r"\b(kike|wetback|towelhead|raghead)\b"
        ])

        self.add_token_set("violence", [], [
            r"\b(kill|murder|assassinate|torture|bomb)\b",
            r"\b(shoot|stab|strangle|poison|execute)\b"
        ])

        self.add_token_set("illegal_activities", [], [
            r"\b(drugs|cocaine|heroin|meth|weed)\b",
            r"\b(steal|rob|fraud|hack|pirate)\b"
        ])

        # Misinformation patterns
        self.add_token_set("medical_misinformation", [], [
            r"\b(cure cancer|miracle cure|big pharma conspiracy)\b",
            r"\b(vaccines cause autism|covid hoax)\b"
        ])

        self.add_token_set("conspiracy_theories", [], [
            r"\b(illuminati|new world order|deep state)\b",
            r"\b(chemtrails|flat earth|moon landing fake)\b"
        ])

        # Personal information patterns
        self.add_token_set("personal_info", [], [
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
            r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b",  # Credit card pattern
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"  # Email pattern
        ])

        # Repetitive patterns
        self.add_token_set("repetition", [], [
            r"\b(\w+)\s+\1\s+\1\b",  # Word repeated 3+ times
            r"(.{10,})\1{2,}"  # Phrase repeated 3+ times
        ])

    def add_token_set(self, name: str, token_ids: List[int], patterns: List[str]):
        """Add a set of forbidden tokens"""
        self.token_sets[name] = {
            'token_ids': set(token_ids),
            'patterns': [re.compile(pattern, re.IGNORECASE) for pattern in patterns],
            'raw_patterns': patterns
        }
        logging.info(f"Added token set '{name}' with {len(token_ids)} tokens and {len(patterns)} patterns")

    def add_context_rule(self, name: str, context_patterns: List[str],
                        forbidden_tokens: List[int], forbidden_patterns: List[str]):
        """Add context-dependent filtering rule"""
        self.context_rules[name] = {
            'context_patterns': [re.compile(pattern, re.IGNORECASE) for pattern in context_patterns],
            'forbidden_tokens': set(forbidden_tokens),
            'forbidden_patterns': [re.compile(pattern, re.IGNORECASE) for pattern in forbidden_patterns]
        }
        logging.info(f"Added context rule '{name}'")

    def check_token_forbidden(self, token_id: int, token_text: str, context: str = "") -> Tuple[bool, List[str]]:
        """Check if a token is forbidden"""
        violations = []

        # Check direct token ID matches
        for set_name, token_set in self.token_sets.items():
            if token_id in token_set['token_ids']:
                violations.append(f"Token ID {token_id} in forbidden set '{set_name}'")

        # Check pattern matches
        for set_name, token_set in self.token_sets.items():
            for pattern in token_set['patterns']:
                if pattern.search(token_text):
                    violations.append(f"Token '{token_text}' matches pattern in set '{set_name}'")

        # Check context-dependent rules
        if context:
            for rule_name, rule in self.context_rules.items():
                # Check if context matches
                context_match = any(pattern.search(context) for pattern in rule['context_patterns'])
                if context_match:
                    # Check if token is forbidden in this context
                    if token_id in rule['forbidden_tokens']:
                        violations.append(f"Token ID {token_id} forbidden in context '{rule_name}'")

                    for pattern in rule['forbidden_patterns']:
                        if pattern.search(token_text):
                            violations.append(f"Token '{token_text}' forbidden in context '{rule_name}'")

        return len(violations) > 0, violations

    def get_forbidden_token_ids(self) -> Set[int]:
        """Get all forbidden token IDs"""
        forbidden_ids = set()
        for token_set in self.token_sets.values():
            forbidden_ids.update(token_set['token_ids'])
        return forbidden_ids

    def export_rules(self, filepath: str) -> bool:
        """Export filtering rules to JSON"""
        try:
            export_data = {
                'token_sets': {
                    name: {
                        'token_ids': list(data['token_ids']),
                        'patterns': data['raw_patterns']
                    }
                    for name, data in self.token_sets.items()
                },
                'context_rules': {
                    name: {
                        'context_patterns': [p.pattern for p in data['context_patterns']],
                        'forbidden_tokens': list(data['forbidden_tokens']),
                        'forbidden_patterns': [p.pattern for p in data['forbidden_patterns']]
                    }
                    for name, data in self.context_rules.items()
                },
                'export_timestamp': datetime.now().isoformat()
            }

            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)

            logging.info(f"Filtering rules exported to {filepath}")
            return True

        except Exception as e:
            logging.error(f"Export failed: {e}")
            return False

    def import_rules(self, filepath: str) -> bool:
        """Import filtering rules from JSON"""
        try:
            with open(filepath, 'r') as f:
                import_data = json.load(f)

            # Import token sets
            for name, data in import_data.get('token_sets', {}).items():
                self.add_token_set(name, data['token_ids'], data['patterns'])

            # Import context rules
            for name, data in import_data.get('context_rules', {}).items():
                self.add_context_rule(
                    name,
                    data['context_patterns'],
                    data['forbidden_tokens'],
                    data['forbidden_patterns']
                )

            logging.info(f"Filtering rules imported from {filepath}")
            return True

        except Exception as e:
            logging.error(f"Import failed: {e}")
            return False


class FactualConsistencyChecker:
    """Checks for factual consistency and plausibility"""

    def __init__(self, config):
        self.config = config
        self.knowledge_base = {}
        self.fact_patterns = {}
        self.confidence_threshold = getattr(config, 'factual_confidence_threshold', 0.7)

        self._initialize_fact_patterns()

        logging.info("✅ Factual Consistency Checker initialized")

    def _initialize_fact_patterns(self):
        """Initialize common factual patterns to check"""

        # Date patterns
        self.fact_patterns['dates'] = [
            r"\b(19|20)\d{2}\b",  # Years
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+(19|20)\d{2}\b"
        ]

        # Number patterns
        self.fact_patterns['numbers'] = [
            r"\b\d+(\.\d+)?\s*(million|billion|trillion)\b",
            r"\b\d+(\.\d+)?\s*(percent|%)\b",
            r"\$\d+(\.\d+)?\s*(million|billion|trillion)?\b"
        ]

        # Geographic patterns
        self.fact_patterns['geography'] = [
            r"\b(capital of|located in|borders|population of)\b",
            r"\b(miles|kilometers|km)\s+(from|to|away)\b"
        ]

        # Scientific patterns
        self.fact_patterns['science'] = [
            r"\b(discovered in|invented in|founded in)\s+(19|20)\d{2}\b",
            r"\b(temperature of|speed of|weight of)\b"
        ]

    def check_factual_consistency(self, text: str, context: str = "") -> HallucinationDetection:
        """Check text for factual inconsistencies"""

        # Simple heuristic-based checking
        inconsistencies = []
        confidence = 1.0

        # Check for obvious contradictions
        if self._check_date_inconsistencies(text):
            inconsistencies.append("Date inconsistency detected")
            confidence -= 0.3

        if self._check_number_plausibility(text):
            inconsistencies.append("Implausible numbers detected")
            confidence -= 0.2

        if self._check_repetition_patterns(text):
            inconsistencies.append("Repetitive patterns detected")
            confidence -= 0.4

        # Check against context if provided
        if context and self._check_context_consistency(text, context):
            inconsistencies.append("Context inconsistency detected")
            confidence -= 0.3

        detected = len(inconsistencies) > 0 and confidence < self.confidence_threshold

        return HallucinationDetection(
            detected=detected,
            hallucination_type=HallucinationType.FACTUAL_INCONSISTENCY,
            confidence=max(0.0, confidence),
            affected_tokens=[],  # Would need tokenizer to identify specific tokens
            explanation="; ".join(inconsistencies) if inconsistencies else "No factual issues detected",
            suggested_action="Review and verify factual claims" if detected else "Continue",
            severity=1.0 - confidence if detected else 0.0
        )

    def _check_date_inconsistencies(self, text: str) -> bool:
        """Check for date-related inconsistencies"""
        # Look for impossible dates or anachronisms
        date_patterns = [
            r"\b(invented|discovered|founded)\s+in\s+(19|20)\d{2}\b",
            r"\b(19|20)\d{2}\s+(invention|discovery)\b"
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Simple check: no inventions before 1800
                for match in matches:
                    if isinstance(match, tuple):
                        year = int(match[1] + match[2] if len(match) > 2 else match[0])
                    else:
                        year = int(re.search(r'\d{4}', str(match)).group())

                    if year < 1800:  # Very basic check
                        return True

        return False

    def _check_number_plausibility(self, text: str) -> bool:
        """Check for implausible numbers"""
        # Look for obviously wrong statistics
        number_patterns = [
            r"\b(\d+(\.\d+)?)\s*percent\b",
            r"\b(\d+(\.\d+)?)\s*(million|billion|trillion)\s+people\b"
        ]

        for pattern in number_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        number = float(match[0])
                    else:
                        number = float(match)

                    # Basic plausibility checks
                    if "percent" in pattern and number > 100:
                        return True
                    if "people" in pattern and number > 8:  # World population ~8 billion
                        return True

                except ValueError:
                    continue

        return False

    def _check_repetition_patterns(self, text: str) -> bool:
        """Check for repetitive patterns that might indicate hallucination"""
        # Check for word repetition
        words = text.lower().split()
        if len(words) > 10:
            # Check for excessive repetition
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            # If any word appears more than 30% of the time, it's likely repetitive
            max_count = max(word_counts.values())
            if max_count > len(words) * 0.3:
                return True

        # Check for phrase repetition
        if len(text) > 50:
            # Simple check for repeated substrings
            for i in range(len(text) - 20):
                substring = text[i:i+20]
                if text.count(substring) > 2:
                    return True

        return False

    def _check_context_consistency(self, text: str, context: str) -> bool:
        """Check consistency with provided context"""
        # Simple keyword-based consistency check
        context_words = set(context.lower().split())
        text_words = set(text.lower().split())

        # Check for contradictory terms (very basic)
        contradictions = [
            ("hot", "cold"), ("big", "small"), ("fast", "slow"),
            ("yes", "no"), ("true", "false"), ("good", "bad")
        ]

        for word1, word2 in contradictions:
            if word1 in context_words and word2 in text_words:
                return True
            if word2 in context_words and word1 in text_words:
                return True

        return False


class AntiHallucinationFilter:
    """Main anti-hallucination filtering system"""

    def __init__(self, config):
        self.config = config

        # Initialize components
        self.forbidden_db = ForbiddenTokenDatabase()
        self.factual_checker = FactualConsistencyChecker(config)

        # Configuration
        self.default_policy = getattr(config, 'filtering_policy', FilteringPolicy.MODERATE)
        self.confidence_threshold = getattr(config, 'hallucination_confidence_threshold', 0.5)
        self.enable_factual_checking = getattr(config, 'enable_factual_checking', True)
        self.enable_repetition_detection = getattr(config, 'enable_repetition_detection', True)

        # Filtering rules
        self.filtering_rules = []

        # Statistics (initialize before rules)
        self.filtering_stats = {
            'total_filtered': 0,
            'tokens_blocked': 0,
            'hallucinations_detected': 0,
            'rules_applied': {}
        }

        # Initialize default rules
        self._initialize_default_rules()

        logging.info("✅ Anti-Hallucination Filter initialized")
        logging.info(f"  • Default policy: {self.default_policy.value}")
        logging.info(f"  • Confidence threshold: {self.confidence_threshold}")
        logging.info(f"  • Factual checking: {self.enable_factual_checking}")

    def _initialize_default_rules(self):
        """Initialize default filtering rules"""

        # Strict filtering for harmful content
        self.add_filtering_rule(FilteringRule(
            name="harmful_content",
            token_ids=[],
            token_patterns=["profanity", "hate_speech", "violence"],
            policy=FilteringPolicy.STRICT,
            confidence_threshold=0.0,
            description="Block harmful and offensive content"
        ))

        # Moderate filtering for misinformation
        self.add_filtering_rule(FilteringRule(
            name="misinformation",
            token_ids=[],
            token_patterns=["medical_misinformation", "conspiracy_theories"],
            policy=FilteringPolicy.MODERATE,
            confidence_threshold=0.3,
            description="Filter potential misinformation"
        ))

        # Lenient filtering for personal information
        self.add_filtering_rule(FilteringRule(
            name="personal_info",
            token_ids=[],
            token_patterns=["personal_info"],
            policy=FilteringPolicy.LENIENT,
            confidence_threshold=0.7,
            description="Warn about personal information"
        ))

        # Adaptive filtering for repetition
        self.add_filtering_rule(FilteringRule(
            name="repetition",
            token_ids=[],
            token_patterns=["repetition"],
            policy=FilteringPolicy.ADAPTIVE,
            confidence_threshold=0.5,
            description="Handle repetitive content adaptively"
        ))

    def add_filtering_rule(self, rule: FilteringRule):
        """Add a new filtering rule"""
        self.filtering_rules.append(rule)
        self.filtering_stats['rules_applied'][rule.name] = 0
        logging.info(f"Added filtering rule: {rule.name}")

    def filter_logits(self, logits: torch.Tensor,
                     input_text: str = "",
                     context: str = "",
                     tokenizer = None) -> FilteringResult:
        """
        Apply anti-hallucination filtering to model logits

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size] or [vocab_size]
            input_text: Input text for context
            context: Additional context for filtering
            tokenizer: Tokenizer for converting token IDs to text

        Returns:
            FilteringResult with filtered logits and detection results
        """
        original_logits = logits.clone()
        filtered_logits = logits.clone()
        blocked_tokens = []
        applied_rules = []
        hallucination_detections = []
        confidence_scores = {}

        # Handle different logit shapes
        if logits.dim() == 3:
            # Batch processing - apply to last position
            batch_size, seq_len, vocab_size = logits.shape
            current_logits = logits[:, -1, :]  # Last position
        elif logits.dim() == 2:
            # Batch of single positions
            batch_size, vocab_size = logits.shape
            current_logits = logits
        else:
            # Single position
            vocab_size = logits.shape[0]
            current_logits = logits.unsqueeze(0)
            batch_size = 1

        # Convert logits to probabilities for confidence calculation
        probs = F.softmax(current_logits, dim=-1)

        # Get top-k tokens for analysis
        top_k = min(100, vocab_size)
        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)

        for batch_idx in range(batch_size):
            batch_blocked = []
            batch_applied_rules = []
            batch_detections = []

            # Analyze top tokens
            for i in range(top_k):
                token_id = top_indices[batch_idx, i].item()
                token_prob = top_probs[batch_idx, i].item()

                # Get token text if tokenizer available
                token_text = ""
                if tokenizer:
                    try:
                        token_text = tokenizer.decode([token_id])
                    except:
                        token_text = f"<token_{token_id}>"

                # Check against forbidden token database
                is_forbidden, violations = self.forbidden_db.check_token_forbidden(
                    token_id, token_text, context
                )

                if is_forbidden:
                    # Apply filtering based on rules
                    for rule in self.filtering_rules:
                        if not rule.enabled:
                            continue

                        # Check if this rule applies
                        rule_applies = False
                        for pattern_name in rule.token_patterns:
                            if pattern_name in [v.split("'")[1] for v in violations if "set '" in v]:
                                rule_applies = True
                                break

                        if rule_applies:
                            should_block = self._should_block_token(
                                rule, token_prob, token_text, context
                            )

                            if should_block:
                                # Block the token by setting very low probability
                                if logits.dim() == 3:
                                    filtered_logits[batch_idx, -1, token_id] = -float('inf')
                                elif logits.dim() == 2:
                                    filtered_logits[batch_idx, token_id] = -float('inf')
                                else:
                                    filtered_logits[token_id] = -float('inf')

                                batch_blocked.append(token_id)
                                if rule.name not in batch_applied_rules:
                                    batch_applied_rules.append(rule.name)
                                    self.filtering_stats['rules_applied'][rule.name] += 1

                                # Create hallucination detection
                                detection = HallucinationDetection(
                                    detected=True,
                                    hallucination_type=HallucinationType.FORBIDDEN_TOKEN,
                                    confidence=1.0 - token_prob,
                                    affected_tokens=[token_id],
                                    explanation=f"Token blocked by rule '{rule.name}': {'; '.join(violations)}",
                                    suggested_action=f"Apply {rule.policy.value} filtering",
                                    severity=self._calculate_severity(rule.policy, token_prob)
                                )
                                batch_detections.append(detection)

                # Store confidence score
                confidence_scores[token_id] = token_prob

            blocked_tokens.extend(batch_blocked)
            applied_rules.extend(batch_applied_rules)
            hallucination_detections.extend(batch_detections)

        # Factual consistency checking
        if self.enable_factual_checking and input_text:
            factual_detection = self.factual_checker.check_factual_consistency(
                input_text, context
            )
            if factual_detection.detected:
                hallucination_detections.append(factual_detection)

        # Update statistics
        self.filtering_stats['total_filtered'] += 1
        self.filtering_stats['tokens_blocked'] += len(blocked_tokens)
        self.filtering_stats['hallucinations_detected'] += len(hallucination_detections)

        # Calculate filtering statistics
        filtering_stats = {
            'tokens_analyzed': top_k * batch_size,
            'tokens_blocked': len(blocked_tokens),
            'rules_applied': len(set(applied_rules)),
            'hallucinations_detected': len(hallucination_detections),
            'average_confidence': np.mean(list(confidence_scores.values())) if confidence_scores else 0.0,
            'filtering_strength': len(blocked_tokens) / (top_k * batch_size) if top_k * batch_size > 0 else 0.0
        }

        return FilteringResult(
            original_logits=original_logits,
            filtered_logits=filtered_logits,
            blocked_tokens=blocked_tokens,
            applied_rules=list(set(applied_rules)),
            hallucination_detections=hallucination_detections,
            confidence_scores=confidence_scores,
            filtering_stats=filtering_stats
        )

    def _should_block_token(self, rule: FilteringRule, token_prob: float,
                           token_text: str, context: str) -> bool:
        """Determine if a token should be blocked based on the rule"""

        if rule.policy == FilteringPolicy.STRICT:
            return True

        elif rule.policy == FilteringPolicy.MODERATE:
            return token_prob > rule.confidence_threshold

        elif rule.policy == FilteringPolicy.LENIENT:
            return token_prob > rule.confidence_threshold and token_prob > 0.8

        elif rule.policy == FilteringPolicy.ADAPTIVE:
            # Adaptive policy considers context and confidence
            base_threshold = rule.confidence_threshold

            # Adjust threshold based on context
            if context and len(context) > 100:
                base_threshold *= 0.8  # More lenient with more context

            # Adjust based on token probability
            if token_prob > 0.9:
                base_threshold *= 0.5  # More strict for high-confidence bad tokens

            return token_prob > base_threshold

        return False

    def _calculate_severity(self, policy: FilteringPolicy, token_prob: float) -> float:
        """Calculate severity score for a blocked token"""
        base_severity = {
            FilteringPolicy.STRICT: 1.0,
            FilteringPolicy.MODERATE: 0.7,
            FilteringPolicy.LENIENT: 0.4,
            FilteringPolicy.ADAPTIVE: 0.6
        }

        # Adjust based on token probability
        prob_multiplier = min(1.0, token_prob * 2)  # Higher prob = higher severity

        return base_severity[policy] * prob_multiplier

    def get_filtering_statistics(self) -> Dict[str, Any]:
        """Get comprehensive filtering statistics"""
        return {
            'total_filtered': self.filtering_stats['total_filtered'],
            'tokens_blocked': self.filtering_stats['tokens_blocked'],
            'hallucinations_detected': self.filtering_stats['hallucinations_detected'],
            'rules_applied': dict(self.filtering_stats['rules_applied']),
            'active_rules': len([r for r in self.filtering_rules if r.enabled]),
            'total_rules': len(self.filtering_rules),
            'forbidden_token_sets': len(self.forbidden_db.token_sets),
            'context_rules': len(self.forbidden_db.context_rules),
            'average_blocks_per_filter': (
                self.filtering_stats['tokens_blocked'] / max(1, self.filtering_stats['total_filtered'])
            )
        }

    def update_rule(self, rule_name: str, **kwargs):
        """Update an existing filtering rule"""
        for rule in self.filtering_rules:
            if rule.name == rule_name:
                for key, value in kwargs.items():
                    if hasattr(rule, key):
                        setattr(rule, key, value)
                logging.info(f"Updated rule '{rule_name}': {kwargs}")
                return True

        logging.warning(f"Rule '{rule_name}' not found")
        return False

    def export_configuration(self, filepath: str) -> bool:
        """Export filter configuration"""
        try:
            config_data = {
                'default_policy': self.default_policy.value,
                'confidence_threshold': self.confidence_threshold,
                'enable_factual_checking': self.enable_factual_checking,
                'enable_repetition_detection': self.enable_repetition_detection,
                'filtering_rules': [
                    {
                        'name': rule.name,
                        'token_ids': rule.token_ids,
                        'token_patterns': rule.token_patterns,
                        'policy': rule.policy.value,
                        'confidence_threshold': rule.confidence_threshold,
                        'context_dependent': rule.context_dependent,
                        'description': rule.description,
                        'enabled': rule.enabled
                    }
                    for rule in self.filtering_rules
                ],
                'statistics': self.get_filtering_statistics(),
                'export_timestamp': datetime.now().isoformat()
            }

            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)

            logging.info(f"Filter configuration exported to {filepath}")
            return True

        except Exception as e:
            logging.error(f"Export failed: {e}")
            return False


def create_anti_hallucination_filter(config) -> AntiHallucinationFilter:
    """Factory function to create anti-hallucination filter"""
    return AntiHallucinationFilter(config)


def test_anti_hallucination_system():
    """Test the anti-hallucination system"""
    print("🧪 Testing Anti-Hallucination System")

    # Mock config
    class MockConfig:
        filtering_policy = FilteringPolicy.MODERATE
        hallucination_confidence_threshold = 0.5
        enable_factual_checking = True
        enable_repetition_detection = True
        factual_confidence_threshold = 0.7

    config = MockConfig()

    # Test forbidden token database
    print("✅ Testing Forbidden Token Database...")
    db = ForbiddenTokenDatabase()

    # Test token checking
    is_forbidden, violations = db.check_token_forbidden(123, "damn", "")
    print(f"  • Token 'damn' forbidden: {is_forbidden}")
    print(f"  • Violations: {violations}")

    # Test pattern matching
    is_forbidden, violations = db.check_token_forbidden(456, "kill someone", "")
    print(f"  • Token 'kill someone' forbidden: {is_forbidden}")

    # Test factual consistency checker
    print("✅ Testing Factual Consistency Checker...")
    checker = FactualConsistencyChecker(config)

    # Test factual checking
    detection = checker.check_factual_consistency(
        "The internet was invented in 1650 by aliens.",
        "We are discussing modern technology."
    )
    print(f"  • Factual inconsistency detected: {detection.detected}")
    print(f"  • Explanation: {detection.explanation}")

    # Test repetition detection
    repetitive_text = "The cat sat on the mat. The cat sat on the mat. The cat sat on the mat."
    detection = checker.check_factual_consistency(repetitive_text)
    print(f"  • Repetition detected: {detection.detected}")

    # Test main anti-hallucination filter
    print("✅ Testing Anti-Hallucination Filter...")
    filter_system = create_anti_hallucination_filter(config)

    # Create mock logits
    vocab_size = 1000
    batch_size = 2
    seq_len = 10

    # Test with 3D logits (batch, seq, vocab)
    logits_3d = torch.randn(batch_size, seq_len, vocab_size)

    # Mock tokenizer
    class MockTokenizer:
        def decode(self, token_ids):
            # Simple mock - return token ID as string
            return f"token_{token_ids[0]}"

    tokenizer = MockTokenizer()

    # Test filtering
    result = filter_system.filter_logits(
        logits_3d,
        input_text="This is a test input with some damn words",
        context="Testing context",
        tokenizer=tokenizer
    )

    print(f"  • Original logits shape: {result.original_logits.shape}")
    print(f"  • Filtered logits shape: {result.filtered_logits.shape}")
    print(f"  • Tokens blocked: {len(result.blocked_tokens)}")
    print(f"  • Rules applied: {result.applied_rules}")
    print(f"  • Hallucinations detected: {len(result.hallucination_detections)}")
    print(f"  • Filtering stats: {result.filtering_stats}")

    # Test with 1D logits
    logits_1d = torch.randn(vocab_size)
    result_1d = filter_system.filter_logits(logits_1d, tokenizer=tokenizer)
    print(f"  • 1D filtering - tokens blocked: {len(result_1d.blocked_tokens)}")

    # Test statistics
    print("✅ Testing Filter Statistics...")
    stats = filter_system.get_filtering_statistics()
    print(f"  • Total filtered: {stats['total_filtered']}")
    print(f"  • Active rules: {stats['active_rules']}")
    print(f"  • Forbidden token sets: {stats['forbidden_token_sets']}")

    # Test rule updates
    print("✅ Testing Rule Updates...")
    success = filter_system.update_rule('harmful_content', confidence_threshold=0.8)
    print(f"  • Rule update success: {success}")

    # Test export/import
    print("✅ Testing Export/Import...")
    export_success = filter_system.export_configuration("test_filter_config.json")
    print(f"  • Export success: {export_success}")

    db_export_success = db.export_rules("test_forbidden_tokens.json")
    print(f"  • Database export success: {db_export_success}")

    print("🎉 All anti-hallucination tests completed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    test_anti_hallucination_system()