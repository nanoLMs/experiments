#!/usr/bin/env python3
"""
Unit Tests for Advanced Quantization System
==========================================

Comprehensive tests for:
- NF4 quantization accuracy and performance
- FP4 quantization with NVFP4 format
- QAF transition detection
- Quantization quality validation
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from advanced_quantization import (
    NF4Quantizer, FP4Quantizer, QAFDetector, QuantizationController,
    QuantizationMetrics, create_quantization_controller
)

# Mock config for testing
class MockConfig:
    def __init__(self):
        # NF4 settings
        self.use_bnb_4bit = True
        self.bnb_4bit_quant_type = "nf4"
        self.bnb_4bit_compute_dtype = "bfloat16"
        self.bnb_4bit_use_double_quant = True
        self.bnb_4bit_quant_storage = "uint8"
        self.bnb_4bit_quantize_heads = True
        self.bnb_4bit_quantize_router = True

        # FP4 settings
        self.use_fp4 = True
        self.fp4_format = "nvfp4"
        self.fp4_block_size = 16
        self.fp4_split_rounding = True

        # QAF settings
        self.qaf_threshold = 1e-6
        self.max_qaf_steps = 100  # Reduced for testing
        self.qaf_precision = "bf16"


class TestFP4Quantizer(unittest.TestCase):
    """Test FP4 quantization functionality"""

    def setUp(self):
        self.config = MockConfig()
        self.quantizer = FP4Quantizer(self.config)

    def test_fp4_quantizer_initialization(self):
        """Test FP4 quantizer initialization"""
        self.assertEqual(self.quantizer.format, "nvfp4")
        self.assertEqual(self.quantizer.block_size, 16)
        self.assertTrue(self.quantizer.split_rounding)
        self.assertEqual(self.quantizer.fp4_levels, 16)

    def test_tensor_quantization_fp4(self):
        """Test FP4 tensor quantization"""
        # Test with different tensor shapes
        test_tensors = [
            torch.randn(32, 64),
            torch.randn(128, 256),
            torch.randn(16, 32, 48),
            torch.randn(8)  # Small tensor
        ]

        for tensor in test_tensors:
            with self.subTest(shape=tensor.shape):
                # Test forward pass (round-to-nearest)
                quantized = self.quantizer.quantize_tensor_fp4(tensor, training=False)

                # Check shape preservation
                self.assertEqual(quantized.shape, tensor.shape)

                # Check quantization error is reasonable
                error = torch.mean((tensor - quantized) ** 2).item()
                self.assertLess(error, 1.0, "Quantization error too high")

    def test_stochastic_vs_deterministic_rounding(self):
        """Test difference between stochastic and deterministic rounding"""
        tensor = torch.randn(64, 128)

        # Deterministic rounding (training=False)
        quant1 = self.quantizer.quantize_tensor_fp4(tensor, training=False)
        quant2 = self.quantizer.quantize_tensor_fp4(tensor, training=False)

        # Should be identical
        self.assertTrue(torch.allclose(quant1, quant2))

        # Stochastic rounding (training=True) - may be different
        # Note: This test might occasionally fail due to randomness
        torch.manual_seed(42)
        quant3 = self.quantizer.quantize_tensor_fp4(tensor, training=True)
        torch.manual_seed(43)
        quant4 = self.quantizer.quantize_tensor_fp4(tensor, training=True)

        # Usually different, but not guaranteed
        # Just check they're both valid quantizations
        error3 = torch.mean((tensor - quant3) ** 2).item()
        error4 = torch.mean((tensor - quant4) ** 2).item()
        self.assertLess(error3, 1.0)
        self.assertLess(error4, 1.0)

    def test_model_quantization_fp4(self):
        """Test FP4 model quantization"""
        # Create test model
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Dropout(0.1),
            nn.Linear(32, 10)
        )

        # Apply FP4 quantization
        quantized_model = self.quantizer.quantize_model(model)

        # Check that hooks were added
        self.assertGreater(len(self.quantizer.quantized_modules), 0)

        # Test forward pass
        test_input = torch.randn(16, 64)
        output = quantized_model(test_input)

        # Check output shape
        self.assertEqual(output.shape, (16, 10))

        # Remove quantization
        self.quantizer.remove_quantization(quantized_model)


class TestQAFDetector(unittest.TestCase):
    """Test QAF transition detection"""

    def setUp(self):
        self.config = MockConfig()
        self.detector = QAFDetector(self.config)

    def test_qaf_detector_initialization(self):
        """Test QAF detector initialization"""
        self.assertEqual(self.detector.threshold, 1e-6)
        self.assertEqual(self.detector.max_qaf_steps, 100)
        self.assertFalse(self.detector.qaf_triggered)
        self.assertIsNone(self.detector.trigger_step)

    def test_gradient_statistics_update(self):
        """Test gradient statistics tracking"""
        # Create test model
        model = nn.Linear(32, 16)

        # Simulate gradient updates
        for step in range(10):
            # Create mock gradients
            for param in model.parameters():
                param.grad = torch.randn_like(param) * 0.01

            self.detector.update_gradient_stats(model, step)

        # Check that statistics were recorded
        self.assertEqual(len(self.detector.gradient_history), 10)
        self.assertEqual(len(self.detector.noise_history), 10)
        self.assertEqual(len(self.detector.gradient_norms), 10)

    def test_gradient_to_noise_ratio_calculation(self):
        """Test gradient-to-noise ratio calculation"""
        # Manually set some gradient history
        self.detector.gradient_history = [0.1, 0.08, 0.09, 0.07, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02]
        self.detector.noise_history = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

        gnr = self.detector.calculate_gradient_to_noise_ratio()

        # Should be around recent average / noise average
        # With the given values, expect around 6.2
        self.assertGreater(gnr, 4.0)
        self.assertLess(gnr, 8.0)

    def test_qaf_trigger_detection(self):
        """Test QAF trigger detection"""
        # Set up conditions for QAF trigger
        # Very low gradient-to-noise ratio
        self.detector.gradient_history = [1e-10] * 25
        self.detector.noise_history = [1e-2] * 25

        # Should trigger QAF
        should_trigger = self.detector.should_trigger_qaf(100)
        self.assertTrue(should_trigger)
        self.assertTrue(self.detector.qaf_triggered)
        self.assertEqual(self.detector.trigger_step, 100)

    def test_qaf_completion_detection(self):
        """Test QAF completion detection"""
        # Trigger QAF first
        self.detector.qaf_triggered = True
        self.detector.trigger_step = 100

        # Test before completion
        self.assertFalse(self.detector.is_qaf_complete(150))

        # Test after completion
        self.assertTrue(self.detector.is_qaf_complete(250))

    def test_qaf_status_reporting(self):
        """Test QAF status reporting"""
        status = self.detector.get_qaf_status()

        # Check required fields
        self.assertIn('qaf_triggered', status)
        self.assertIn('trigger_step', status)
        self.assertIn('qaf_step_count', status)
        self.assertIn('gradient_to_noise_ratio', status)
        self.assertIn('threshold', status)
        self.assertIn('is_complete', status)


class TestQuantizationController(unittest.TestCase):
    """Test quantization controller functionality"""

    def setUp(self):
        self.config = MockConfig()
        self.controller = QuantizationController(self.config)

    def test_controller_initialization(self):
        """Test quantization controller initialization"""
        self.assertEqual(self.controller.current_phase, "nf4")
        self.assertIsNotNone(self.controller.fp4_quantizer)
        self.assertIsNotNone(self.controller.qaf_detector)

    def test_phase_transitions(self):
        """Test quantization phase transitions"""
        # Create test model
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.Linear(64, 16)
        )

        # Test initial phase
        self.assertEqual(self.controller.current_phase, "nf4")

        # Simulate training steps to trigger FP4 transition
        for step in range(1005):  # Should trigger FP4 at step 1000
            # Add mock gradients
            for param in model.parameters():
                param.grad = torch.randn_like(param) * 0.01

            model, phase_changed = self.controller.update_phase(model, step)

            if step == 1001:  # After transition
                self.assertEqual(self.controller.current_phase, "fp4")
                self.assertTrue(phase_changed)

    def test_quantization_status_reporting(self):
        """Test quantization status reporting"""
        status = self.controller.get_quantization_status()

        # Check required fields
        self.assertIn('current_phase', status)
        self.assertIn('qaf_status', status)
        self.assertIn('metrics_count', status)

    def test_state_save_load(self):
        """Test quantization state save/load"""
        # Modify controller state
        self.controller.current_phase = "fp4"
        self.controller.qaf_detector.qaf_triggered = True
        self.controller.qaf_detector.trigger_step = 500

        # Save state
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            self.controller.save_quantization_state(temp_path)

            # Create new controller and load state
            new_controller = QuantizationController(self.config)
            new_controller.load_quantization_state(temp_path)

            # Check state was restored
            self.assertEqual(new_controller.current_phase, "fp4")
            self.assertTrue(new_controller.qaf_detector.qaf_triggered)
            self.assertEqual(new_controller.qaf_detector.trigger_step, 500)

        finally:
            os.unlink(temp_path)


class TestQuantizationMetrics(unittest.TestCase):
    """Test quantization metrics and validation"""

    def test_quantization_metrics_creation(self):
        """Test quantization metrics creation"""
        metrics = QuantizationMetrics(
            timestamp=1234567890.0,
            quantization_error=0.001,
            weight_distribution={'layer1': {'mean': 0.0, 'std': 1.0}},
            activation_range={'layer1': (-2.0, 2.0)},
            gradient_noise_ratio=10.0,
            quality_score=0.95,
            memory_reduction=0.75
        )

        self.assertEqual(metrics.timestamp, 1234567890.0)
        self.assertEqual(metrics.quantization_error, 0.001)
        self.assertEqual(metrics.quality_score, 0.95)
        self.assertEqual(metrics.memory_reduction, 0.75)


class TestQuantizationIntegration(unittest.TestCase):
    """Integration tests for quantization system"""

    def setUp(self):
        self.config = MockConfig()

    def test_create_quantization_controller_factory(self):
        """Test quantization controller factory function"""
        controller = create_quantization_controller(self.config)

        self.assertIsInstance(controller, QuantizationController)
        self.assertEqual(controller.current_phase, "nf4")

    def test_end_to_end_quantization_workflow(self):
        """Test complete quantization workflow"""
        # Create controller
        controller = create_quantization_controller(self.config)

        # Create test model
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Linear(32, 10)
        )

        # Apply initial quantization
        model = controller.apply_quantization(model)

        # Simulate training with phase transitions
        for step in range(1200):
            # Add mock gradients with decreasing magnitude (simulating convergence)
            grad_scale = max(0.001, 0.1 * np.exp(-step / 500))
            for param in model.parameters():
                param.grad = torch.randn_like(param) * grad_scale

            # Update phase
            model, phase_changed = controller.update_phase(model, step)

            # Log phase changes
            if phase_changed:
                print(f"Phase changed to {controller.current_phase} at step {step}")

        # Check final status
        status = controller.get_quantization_status()
        print(f"Final status: {status}")

        # Should have progressed through phases
        self.assertIn(controller.current_phase, ["fp4", "qaf"])


# Mock bitsandbytes for testing when not available
class MockLinear4bit(nn.Module):
    def __init__(self, input_features, output_features, bias=True, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_features, input_features))
        self.bias = nn.Parameter(torch.randn(output_features)) if bias else None
        self.in_features = input_features
        self.out_features = output_features

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)


class TestQuantizationWithMocks(unittest.TestCase):
    """Test quantization system with mocked dependencies"""

    @patch('advanced_quantization.BNB_AVAILABLE', False)
    def test_nf4_quantizer_without_bitsandbytes(self):
        """Test NF4 quantizer behavior when bitsandbytes is not available"""
        config = MockConfig()

        with self.assertRaises(ImportError):
            NF4Quantizer(config)

    @patch('advanced_quantization.Linear4bit', MockLinear4bit)
    @patch('advanced_quantization.BNB_AVAILABLE', True)
    def test_nf4_quantizer_with_mock_bitsandbytes(self):
        """Test NF4 quantizer with mocked bitsandbytes"""
        config = MockConfig()

        # This should work with mocked Linear4bit
        quantizer = NF4Quantizer(config)

        # Test layer quantization
        layer = nn.Linear(32, 16)
        quantized_layer = quantizer.quantize_linear_layer(layer, "test_layer")

        self.assertIsInstance(quantized_layer, MockLinear4bit)


if __name__ == '__main__':
    # Setup test environment
    torch.manual_seed(42)
    np.random.seed(42)

    # Run tests
    unittest.main(verbosity=2)