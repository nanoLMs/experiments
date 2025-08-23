#!/usr/bin/env python3
"""
Test Suite for Memory and Compute Optimizations
==============================================

Comprehensive tests for the memory and compute optimization system.
"""

import unittest
import time
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import the system under test
from memory_compute_optimizations import (
    MemoryMonitor,
    GradientCheckpointer,
    ActivationCheckpointer,
    ComputeOptimizer,
    MemoryComputeOptimizer,
    FusedMultiHeadAttention,
    FusedFeedForward,
    FusedLinearReLU,
    OptimizationConfig,
    MemoryStats,
    ComputeStats,
    CheckpointingStrategy,
    MemoryOptimizationLevel,
    ComputeOptimizationMode
)


class TestOptimizationConfig(unittest.TestCase):
    """Test optimization configuration"""

    def test_config_creation(self):
        """Test creating optimization configuration"""
        config = OptimizationConfig(
            checkpointing_strategy=CheckpointingStrategy.ADAPTIVE,
            memory_optimization_level=MemoryOptimizationLevel.AGGRESSIVE,
            compute_optimization_mode=ComputeOptimizationMode.SPEED,
            target_memory_usage_gb=8.0
        )

        self.assertEqual(config.checkpointing_strategy, CheckpointingStrategy.ADAPTIVE)
        self.assertEqual(config.memory_optimization_level, MemoryOptimizationLevel.AGGRESSIVE)
        self.assertEqual(config.compute_optimization_mode, ComputeOptimizationMode.SPEED)
        self.assertEqual(config.target_memory_usage_gb, 8.0)

    def test_config_defaults(self):
        """Test configuration defaults"""
        config = OptimizationConfig()

        self.assertEqual(config.checkpointing_strategy, CheckpointingStrategy.ADAPTIVE)
        self.assertEqual(config.memory_optimization_level, MemoryOptimizationLevel.MODERATE)
        self.assertEqual(config.compute_optimization_mode, ComputeOptimizationMode.BALANCED)
        self.assertTrue(config.enable_activation_checkpointing)
        self.assertTrue(config.enable_kernel_fusion)


class TestMemoryStats(unittest.TestCase):
    """Test memory statistics"""

    def test_memory_stats_creation(self):
        """Test memory stats creation"""
        stats = MemoryStats(
            allocated_mb=100.0,
            cached_mb=50.0,
            peak_mb=150.0,
            utilization_ratio=0.75
        )

        self.assertEqual(stats.allocated_mb, 100.0)
        self.assertEqual(stats.cached_mb, 50.0)
        self.assertEqual(stats.peak_mb, 150.0)
        self.assertEqual(stats.utilization_ratio, 0.75)


class TestMemoryMonitor(unittest.TestCase):
    """Test memory monitoring system"""

    def setUp(self):
        self.config = OptimizationConfig(
            memory_monitoring_enabled=False,  # Disable background monitoring for tests
            target_memory_usage_gb=2.0,
            memory_pressure_threshold=0.8
        )
        self.monitor = MemoryMonitor(self.config)

    def test_memory_monitor_initialization(self):
        """Test memory monitor initialization"""
        self.assertIsNotNone(self.monitor.config)
        self.assertEqual(len(self.monitor.memory_history), 0)
        self.assertEqual(self.monitor.peak_memory, 0.0)

    def test_get_current_memory_stats(self):
        """Test getting current memory statistics"""
        stats = self.monitor.get_current_memory_stats()

        self.assertIsInstance(stats, MemoryStats)
        self.assertGreaterEqual(stats.allocated_mb, 0.0)
        self.assertGreaterEqual(stats.timestamp, 0.0)

    @patch('memory_compute_optimizations.PSUTIL_AVAILABLE', True)
    @patch('psutil.virtual_memory')
    def test_memory_pressure_detection(self, mock_memory):
        """Test memory pressure detection"""
        # Mock high memory usage
        mock_memory.return_value.percent = 85.0  # 85% usage
        mock_memory.return_value.available = 1024 * 1024 * 1024  # 1GB available

        is_pressure = self.monitor.is_memory_pressure()
        self.assertTrue(is_pressure)

        # Mock low memory usage
        mock_memory.return_value.percent = 50.0  # 50% usage
        is_pressure = self.monitor.is_memory_pressure()
        self.assertFalse(is_pressure)

    def test_memory_cleanup(self):
        """Test memory cleanup functionality"""
        # This should not raise any exceptions
        self.monitor.cleanup_memory(aggressive=False)
        self.monitor.cleanup_memory(aggressive=True)

    def test_memory_recommendations(self):
        """Test memory optimization recommendations"""
        recommendations = self.monitor.get_memory_recommendations()

        self.assertIsInstance(recommendations, list)
        # Recommendations may be empty if no pressure detected


class TestGradientCheckpointer(unittest.TestCase):
    """Test gradient checkpointing system"""

    def setUp(self):
        self.config = OptimizationConfig(
            checkpointing_strategy=CheckpointingStrategy.ADAPTIVE,
            checkpoint_ratio=0.5
        )
        self.checkpointer = GradientCheckpointer(self.config)

    def test_checkpointer_initialization(self):
        """Test gradient checkpointer initialization"""
        self.assertEqual(self.checkpointer.config, self.config)
        self.assertEqual(self.checkpointer.current_checkpoint_ratio, 0.5)
        self.assertEqual(len(self.checkpointer.checkpoint_layers), 0)

    def test_uniform_checkpointing_strategy(self):
        """Test uniform checkinting strategy"""
        uniform_config = OptimizationConfig(
            checkpointing_strategy=CheckpointingStrategy.UNIFORM,
            checkpoint_ratio=0.5  # Every 2nd layer
        )
        uniform_checkpointer = GradientCheckpointer(uniform_config)

        # Test layer checkpointing decisions
        self.assertTrue(uniform_checkpointer.should_checkpoint_layer(0, 10))  # Layer 0
        self.assertFalse(uniform_checkpointer.should_checkpoint_layer(1, 10))  # Layer 1
        self.assertTrue(uniform_checkpointer.should_checkpoint_layer(2, 10))  # Layer 2
        self.assertFalse(uniform_checkpointer.should_checkpoint_layer(3, 10))  # Layer 3

    def test_no_checkpointing_strategy(self):
        """Test no checkpointing strategy"""
        none_config = OptimizationConfig(
            checkpointing_strategy=CheckpointingStrategy.NONE
        )
        none_checkpointer = GradientCheckpointer(none_config)

        # No layers should be checkpointed
        for i in range(10):
            self.assertFalse(none_checkpointer.should_checkpoint_layer(i, 10))

    def test_adaptive_checkpointing(self):
        """Test adaptive checkpointing strategy"""
        # Test with different memory conditions
        low_memory_stats = MemoryStats(utilization_ratio=0.3)
        high_memory_stats = MemoryStats(utilization_ratio=0.9)

        # High memory pressure should increase checkpointing
        high_pressure_decision = self.checkpointer.should_checkpoint_layer(
            1, 10, high_memory_stats
        )

        # Low memory pressure should decrease checkpointing
        low_pressure_decision = self.checkpointer.should_checkpoint_layer(
            1, 10, low_memory_stats
        )

        # The exact decisions depend on implementation, but we can test the method runs
        self.assertIsInstance(high_pressure_decision, bool)
        self.assertIsInstance(low_pressure_decision, bool)

    def test_layer_cost_tracking(self):
        """Test layer cost tracking"""
        # Update layer costs
        self.checkpointer.update_layer_cost(0, compute_time=0.1, memory_usage=100.0)
        self.checkpointer.update_layer_cost(1, compute_time=0.2, memory_usage=50.0)

        # Check costs are recorded
        self.assertIn(0, self.checkpointer.layer_costs)
        self.assertIn(1, self.checkpointer.layer_costs)

        # Check cost ratios
        cost_0 = self.checkpointer.layer_costs[0]
        cost_1 = self.checkpointer.layer_costs[1]

        self.assertEqual(cost_0['compute_time'], 0.1)
        self.assertEqual(cost_0['memory_usage'], 100.0)
        self.assertAlmostEqual(cost_0['cost_ratio'], 0.1 / 100.0)

    def test_checkpoint_function(self):
        """Test checkpoint function wrapper"""
        # Create a simple function to checkpoint
        def test_function(x):
            return x * 2

        # Test with checkpointing enabled
        result = self.checkpointer.checkpoint_function(test_function, torch.tensor(5.0))
        self.assertEqual(result, torch.tensor(10.0))

        # Test with checkpointing disabled
        none_config = OptimizationConfig(checkpointing_strategy=CheckpointingStrategy.NONE)
        none_checkpointer = GradientCheckpointer(none_config)

        result = none_checkpointer.checkpoint_function(test_function, torch.tensor(5.0))
        self.assertEqual(result, torch.tensor(10.0))


class TestActivationCheckpointer(unittest.TestCase):
    """Test activation checkpointing system"""

    def setUp(self):
        self.config = OptimizationConfig(
            target_memory_usage_gb=1.0,  # 1GB for testing
            enable_offloading=False,  # Disable for simpler testing
            enable_compression=False
        )
        self.checkpointer = ActivationCheckpointer(self.config)

    def test_checkpointer_initialization(self):
        """Test activation checkpointer initialization"""
        self.assertEqual(len(self.checkpointer.saved_activations), 0)
        self.assertEqual(self.checkpointer.current_activation_memory_mb, 0.0)
        self.assertGreater(self.checkpointer.max_activation_memory_mb, 0)

    def test_save_and_get_activation(self):
        """Test saving and retrieving activations"""
        # Create test activation
        activation = torch.randn(10, 20)

        # Save activation
        success = self.checkpointer.save_activation("test_key", activation)
        self.assertTrue(success)

        # Retrieve activation
        retrieved = self.checkpointer.get_activation("test_key")
        self.assertIsNotNone(retrieved)
        self.assertTrue(torch.equal(activation, retrieved))

        # Test non-existent key
        non_existent = self.checkpointer.get_activation("non_existent")
        self.assertIsNone(non_existent)

    def test_activation_removal(self):
        """Test activation removal"""
        activation = torch.randn(5, 10)

        # Save and verify
        self.checkpointer.save_activation("test_key", activation)
        self.assertIsNotNone(self.checkpointer.get_activation("test_key"))

        # Remove and verify
        self.checkpointer.remove_activation("test_key")
        self.assertIsNone(self.checkpointer.get_activation("test_key"))

    def test_clear_all_activations(self):
        """Test clearing all activations"""
        # Save multiple activations
        for i in range(3):
            activation = torch.randn(5, 10)
            self.checkpointer.save_activation(f"key_{i}", activation)

        # Verify they exist
        self.assertEqual(len(self.checkpointer.saved_activations), 3)

        # Clear all
        self.checkpointer.clear_all_activations()

        # Verify they're gone
        self.assertEqual(len(self.checkpointer.saved_activations), 0)
        self.assertEqual(self.checkpointer.current_activation_memory_mb, 0.0)

    def test_memory_usage_tracking(self):
        """Test memory usage tracking"""
        initial_usage = self.checkpointer.get_memory_usage()
        self.assertEqual(initial_usage['current_mb'], 0.0)
        self.assertEqual(initial_usage['num_activations'], 0)

        # Save activation
        activation = torch.randn(100, 100)  # Larger activation
        self.checkpointer.save_activation("large_key", activation)

        # Check updated usage
        updated_usage = self.checkpointer.get_memory_usage()
        self.assertGreater(updated_usage['current_mb'], 0.0)
        self.assertEqual(updated_usage['num_activations'], 1)

    def test_memory_pressure_handling(self):
        """Test handling of memory pressure"""
        # Fill up memory with large activations
        large_activation = torch.randn(1000, 1000)  # Very large activation

        # This should succeed initially
        success1 = self.checkpointer.save_activation("large_1", large_activation)
        self.assertTrue(success1)

        # This might trigger eviction or fail due to memory limits
        success2 = self.checkpointer.save_activation("large_2", large_activation)
        # Don't assert success/failure as it depends on memory management


class TestComputeOptimizer(unittest.TestCase):
    """Test compute optimization system"""

    def setUp(self):
        self.config = OptimizationConfig(
            compute_optimization_mode=ComputeOptimizationMode.BALANCED,
            enable_kernel_fusion=True,
            enable_mixed_precision=True,
            enable_graph_optimization=True
        )
        self.optimizer = ComputeOptimizer(self.config)

    def test_optimizer_initialization(self):
        """Test compute optimizer initialization"""
        self.assertEqual(self.optimizer.config, self.config)
        self.assertEqual(len(self.optimizer.fused_operations), 0)
        self.assertEqual(len(self.optimizer.compute_stats), 0)

    def test_module_optimization(self):
        """Test module optimization"""
        # Create test module
        module = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )

        # Optimize module
        optimized_module = self.optimizer.optimize_module(module)

        # Should return a module (may be the same or modified)
        self.assertIsInstance(optimized_module, nn.Module)

        # Should be marked as optimized
        self.assertIn(optimized_module, self.optimizer.optimized_modules)

    def test_fused_attention_creation(self):
        """Test fused attention creation"""
        embed_dim, num_heads = 512, 8

        fused_attention = self.optimizer.create_fused_attention(embed_dim, num_heads)

        self.assertIsInstance(fused_attention, FusedMultiHeadAttention)
        self.assertEqual(fused_attention.embed_dim, embed_dim)
        self.assertEqual(fused_attention.num_heads, num_heads)

    def test_fused_feedforward_creation(self):
        """Test fused feedforward creation"""
        input_dim, hidden_dim = 256, 1024

        fused_ff = self.optimizer.create_fused_feedforward(input_dim, hidden_dim)

        self.assertIsInstance(fused_ff, FusedFeedForward)
        self.assertEqual(fused_ff.input_dim, input_dim)
        self.assertEqual(fused_ff.hidden_dim, hidden_dim)

    def test_compute_stats_recording(self):
        """Test compute statistics recording"""
        # Record some stats
        self.optimizer.record_compute_stats(
            forward_time=0.1,
            backward_time=0.2,
            flops=1e12,
            memory_bandwidth=100.0
        )

        # Check stats were recorded
        self.assertEqual(len(self.optimizer.compute_stats), 1)

        stats = self.optimizer.compute_stats[0]
        self.assertEqual(stats.forward_time_ms, 100.0)  # 0.1s = 100ms
        self.assertEqual(stats.backward_time_ms, 200.0)  # 0.2s = 200ms
        self.assertAlmostEqual(stats.total_time_ms, 300.0, places=1)

    def test_optimization_recommendations(self):
        """Test optimization recommendations"""
        # Add some performance data
        for i in range(15):
            self.optimizer.record_compute_stats(
                forward_time=0.05 + i * 0.01,  # Increasing forward time
                backward_time=0.15 + i * 0.02,  # Increasing backward time
                flops=1e11,  # Low FLOPS
                memory_bandwidth=50.0
            )

        recommendations = self.optimizer.get_optimization_recommendations()

        self.assertIsInstance(recommendations, list)
        # Should have recommendations due to poor performance
        self.assertGreater(len(recommendations), 0)


class TestFusedOperations(unittest.TestCase):
    """Test fused operation modules"""

    def test_fused_multi_head_attention(self):
        """Test fused multi-head attention"""
        embed_dim, num_heads = 256, 8
        batch_size, seq_len = 2, 32

        config = OptimizationConfig(
            memory_optimization_level=MemoryOptimizationLevel.MODERATE
        )

        attention = FusedMultiHeadAttention(embed_dim, num_heads, config)

        # Test forward pass
        x = torch.randn(batch_size, seq_len, embed_dim)
        output, _ = attention(x, x, x)  # Self-attention: query, key, value are the same

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, embed_dim))

    def test_fused_multi_head_attention_with_mask(self):
        """Test fused attention with attention mask"""
        embed_dim, num_heads = 128, 4
        batch_size, seq_len = 1, 16

        config = OptimizationConfig()
        attention = FusedMultiHeadAttention(embed_dim, num_heads, config)

        x = torch.randn(batch_size, seq_len, embed_dim)

        # Create causal mask
        attn_mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dims

        output, _ = attention(x, x, x, attn_mask)
        self.assertEqual(output.shape, (batch_size, seq_len, embed_dim))

    def test_fused_feedforward(self):
        """Test fused feedforward network"""
        input_dim, hidden_dim = 128, 512
        batch_size, seq_len = 2, 16

        config = OptimizationConfig(
            memory_optimization_level=MemoryOptimizationLevel.MODERATE
        )

        ff = FusedFeedForward(input_dim, hidden_dim, config)

        # Test forward pass
        x = torch.randn(batch_size, seq_len, input_dim)
        output = ff(x)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, seq_len, input_dim))

    def test_fused_linear_relu(self):
        """Test fused Linear + ReLU operation"""
        in_features, out_features = 64, 128
        batch_size = 4

        fused_layer = FusedLinearReLU(in_features, out_features)

        # Test forward pass
        x = torch.randn(batch_size, in_features)
        output = fused_layer(x)

        # Check output shape and non-negativity (ReLU effect)
        self.assertEqual(output.shape, (batch_size, out_features))
        self.assertTrue(torch.all(output >= 0))  # ReLU should make all values non-negative


class TestMemoryComputeOptimizer(unittest.TestCase):
    """Test main memory and compute optimizer"""

    def setUp(self):
        self.config = OptimizationConfig(
            checkpointing_strategy=CheckpointingStrategy.ADAPTIVE,
            memory_optimization_level=MemoryOptimizationLevel.MODERATE,
            compute_optimization_mode=ComputeOptimizationMode.BALANCED,
            enable_dynamic_optimization=False,  # Disable for simpler testing
            memory_monitoring_enabled=False  # Disable background monitoring
        )
        self.optimizer = MemoryComputeOptimizer(self.config)

    def test_optimizer_initialization(self):
        """Test main optimizer initialization"""
        self.assertIsNotNone(self.optimizer.memory_monitor)
        self.assertIsNotNone(self.optimizer.gradient_checkpointer)
        self.assertIsNotNone(self.optimizer.activation_checkpointer)
        self.assertIsNotNone(self.optimizer.compute_optimizer)
        self.assertEqual(self.optimizer.optimization_step, 0)

    def test_model_optimization(self):
        """Test model optimization"""
        # Create test model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )

        # Optimize model
        optimized_model = self.optimizer.optimize_model(model)

        # Should return a model
        self.assertIsInstance(optimized_model, nn.Module)

    def test_training_step_optimization(self):
        """Test training step optimization"""
        # Create simple model and data
        model = nn.Linear(10, 1)
        inputs = torch.randn(4, 10)
        targets = torch.randn(4, 1)
        loss_fn = nn.MSELoss()

        # Run optimized training step
        loss = self.optimizer.optimize_training_step(model, loss_fn, inputs, targets)

        # Should return a loss tensor
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())  # Scalar loss
        self.assertGreater(self.optimizer.optimization_step, 0)

    def test_optimization_report(self):
        """Test optimization report generation"""
        report = self.optimizer.get_optimization_report()

        # Check report structure
        self.assertIn("memory", report)
        self.assertIn("compute", report)
        self.assertIn("checkpointing", report)
        self.assertIn("optimization_step", report)

        # Check memory section
        memory_section = report["memory"]
        self.assertIn("current_stats", memory_section)
        self.assertIn("activation_stats", memory_section)
        self.assertIn("recommendations", memory_section)

        # Check compute section
        compute_section = report["compute"]
        self.assertIn("optimization_mode", compute_section)
        self.assertIn("kernel_fusion_enabled", compute_section)
        self.assertIn("mixed_precision_enabled", compute_section)

        # Check checkpointing section
        checkpointing_section = report["checkpointing"]
        self.assertIn("strategy", checkpointing_section)
        self.assertIn("current_ratio", checkpointing_section)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete optimization system"""

    def test_end_to_end_optimization(self):
        """Test complete end-to-end optimization workflow"""
        # Create configuration
        config = OptimizationConfig(
            checkpointing_strategy=CheckpointingStrategy.UNIFORM,
            memory_optimization_level=MemoryOptimizationLevel.MODERATE,
            compute_optimization_mode=ComputeOptimizationMode.BALANCED,
            enable_activation_checkpointing=True,
            enable_kernel_fusion=True,
            target_memory_usage_gb=1.0,
            memory_monitoring_enabled=False
        )

        # Create optimizer
        optimizer = MemoryComputeOptimizer(config)

        # Create test transformer model
        class SimpleTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Linear(50, 128)
                self.attention = nn.MultiheadAttention(128, 4, batch_first=True)
                self.norm = nn.LayerNorm(128)
                self.ff = nn.Sequential(
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128)
                )
                self.output = nn.Linear(128, 10)

            def forward(self, x):
                x = self.embedding(x)
                attn_out, _ = self.attention(x, x, x)
                x = self.norm(x + attn_out)
                ff_out = self.ff(x)
                x = self.norm(x + ff_out)
                return self.output(x)

        model = SimpleTransformer()

        # Optimize model
        optimized_model = optimizer.optimize_model(model)

        # Create training data
        batch_size, seq_len, input_dim = 2, 16, 50
        inputs = torch.randn(batch_size, seq_len, input_dim)
        targets = torch.randint(0, 10, (batch_size, seq_len))
        loss_fn = nn.CrossEntropyLoss()

        # Run multiple training steps
        losses = []
        for step in range(3):
            # Reshape targets for CrossEntropyLoss
            targets_flat = targets.view(-1)

            def modified_loss_fn(outputs, targets):
                outputs_flat = outputs.view(-1, outputs.size(-1))
                return loss_fn(outputs_flat, targets_flat)

            loss = optimizer.optimize_training_step(
                optimized_model, modified_loss_fn, inputs, targets
            )
            losses.append(loss.item())

        # Verify training ran successfully
        self.assertEqual(len(losses), 3)
        self.assertTrue(all(isinstance(loss, float) for loss in losses))
        self.assertEqual(optimizer.optimization_step, 3)

        # Get final report
        report = optimizer.get_optimization_report()

        # Verify report completeness
        self.assertGreater(report["optimization_step"], 0)
        self.assertIsInstance(report["memory"]["current_stats"]["allocated_mb"], float)
        self.assertIsInstance(report["checkpointing"]["current_ratio"], float)

    def test_memory_pressure_handling(self):
        """Test system behavior under memory pressure"""
        # Create configuration with low memory limits
        config = OptimizationConfig(
            target_memory_usage_gb=0.1,  # Very low limit
            memory_pressure_threshold=0.5,  # Low threshold
            checkpointing_strategy=CheckpointingStrategy.MEMORY_AWARE,
            memory_monitoring_enabled=False
        )

        optimizer = MemoryComputeOptimizer(config)

        # Create model that uses significant memory
        model = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100)
        )

        # Optimize model
        optimized_model = optimizer.optimize_model(model)

        # Create large batch to trigger memory pressure
        inputs = torch.randn(16, 1000)
        targets = torch.randn(16, 100)
        loss_fn = nn.MSELoss()

        # This should handle memory pressure gracefully
        loss = optimizer.optimize_training_step(optimized_model, loss_fn, inputs, targets)

        # Should complete without errors
        self.assertIsInstance(loss, torch.Tensor)

        # Check that memory recommendations are generated
        report = optimizer.get_optimization_report()
        recommendations = report["memory"]["recommendations"]

        # May have recommendations due to memory pressure
        self.assertIsInstance(recommendations, list)


def run_comprehensive_tests():
    """Run all tests with detailed output"""

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestOptimizationConfig,
        TestMemoryStats,
        TestMemoryMonitor,
        TestGradientCheckpointer,
        TestActivationCheckpointer,
        TestComputeOptimizer,
        TestFusedOperations,
        TestMemoryComputeOptimizer,
        TestIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n{'='*70}")
    print("MEMORY & COMPUTE OPTIMIZATION TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")

    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)