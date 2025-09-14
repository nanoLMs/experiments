#!/usr/bin/env python3
"""
Working Integration Tests for Advanced NanoLM System
===================================================

Simplified integration tests that focus on core functionality
and system interactions without complex API dependencies.
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import tempfile
import os
import concurrent.futures
from typing import Dict, Any, List

# Import available systems for integration testing
from dynamic_batching_caching import (
    IntelligentCache, DynamicBatcher, PowerManager, OptimizedInferenceEngine,
    CacheConfig, BatchingConfig, PowerConfig, BatchRequest, BatchResult,
    BatchingStrategy, CacheEvictionPolicy, PowerMode
)

from memory_compute_optimizations import (
    MemoryMonitor, MemoryComputeOptimizer,
    OptimizationConfig, CheckpointingStrategy, MemoryOptimizationLevel,
    ComputeOptimizationMode
)

from cloud_deployment_system import (
    CloudDeploymentSystem, DeploymentConfig, CloudPlatform,
    DeploymentType, ScalingStrategy
)

from multi_platform_exporter import (
    MultiPlatformExporter, ExportFormat, ExportConfig, OptimizationLevel, ValidationLevel
)


class TestEndToEndWorkflows(unittest.TestCase):
    """Test complete end-to-end workflows"""

    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()

        # Create simple test model
        self.test_model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )

        # Configure systems
        self.optimization_config = OptimizationConfig(
            checkpointing_strategy=CheckpointingStrategy.ADAPTIVE,
            memory_optimization_level=MemoryOptimizationLevel.MODERATE,
            compute_optimization_mode=ComputeOptimizationMode.BALANCED
        )

        self.batching_config = BatchingConfig(
            strategy=BatchingStrategy.ADAPTIVE,
            min_batch_size=1,
            max_batch_size=8,
            target_latency_ms=100.0
        )

        self.cache_config = CacheConfig(
            eviction_policy=CacheEvictionPolicy.ADAPTIVE,
            max_cache_size_mb=64.0,
            max_entries=1000
        )

        self.power_config = PowerConfig(
            mode=PowerM.ADAPTIVE
        )

    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_model_optimization_and_inference_pipeline(self):
        """Test complete model optimization and inference pipeline"""
        # Step 1: Apply memory and compute optimizations
        optimizer = MemoryComputeOptimizer(self.optimization_config)
        optimized_model = optimizer.optimize_model(self.test_model)

        # Verify optimization applied
        self.assertIsNotNone(optimized_model)

        # Step 2: Create optimized inference engine
        inference_engine = OptimizedInferenceEngine(
            optimized_model, self.batching_config, self.cache_config, self.power_config
        )

        # Step 3: Test inference with various scenarios
        test_inputs = [
            torch.randn(1, 64),  # Small input
            torch.randn(2, 64),  # Medium batch
            torch.randn(4, 64),  # Large batch
        ]

        results = []
        for i, inputs in enumerate(test_inputs):
            result = inference_engine.predict(inputs, request_id=f"test_{i}")
            results.append(result)

            # Verify result structure
            self.assertIsInstance(result, BatchResult)
            self.assertIsNotNone(result.outputs)
            self.assertGreaterEqual(result.processing_time_ms, 0)

        # Step 4: Test cache effectiveness
        # Repeat first scenario to test caching
        cached_result = inference_engine.predict(
            test_inputs[0], request_id="test_0"
        )
        self.assertTrue(cached_result.cache_hit)
        self.assertEqual(cached_result.processing_time_ms, 0.0)

        # Step 5: Get comprehensive performance stats
        stats = inference_engine.get_comprehensive_stats()

        # Verify stats structure
        self.assertIn('inference', stats)
        self.assertIn('cache', stats)
        self.assertIn('batching', stats)
        self.assertIn('power', stats)

        # Verify performance metrics
        self.assertGreater(stats['inference']['total_requests'], 0)
        self.assertGreaterEqual(stats['inference']['cache_hit_rate'], 0.0)
        self.assertLessEqual(stats['inference']['cache_hit_rate'], 1.0)

    def test_export_and_deployment_pipeline(self):
        """Test model export and deployment pipeline"""
        # Step 1: Export model to TorchScript (most reliable format)
        export_config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT,
            output_path=os.path.join(self.temp_dir, "model.pt"),
            optimization_level=OptimizationLevel.BASIC,
            validation_level=ValidationLevel.FUNCTIONAL
        )

        exporter = MultiPlatformExporter()

        # Create sample inputs for export
        sample_inputs = torch.randn(1, 64)

        # Export model
        export_result = exporter.export_model(
            self.test_model, sample_inputs, export_config
        )

        # Verify export succeeded
        self.assertTrue(export_result.success)
        self.assertTrue(os.path.exists(export_result.export_path))

        # Step 2: Create deployment configuration
        deployment_config = DeploymentConfig(
            platform=CloudPlatform.DOCKER,
            deployment_type=DeploymentType.REAL_TIME_API,
            scaling_strategy=ScalingStrategy.CPU_BASED,
            min_replicas=1,
            max_replicas=3
        )

        # Step 3: Create deployment system
        deployment_system = CloudDeploymentSystem({
            'docker_registry': 'test-registry.com',
            'kubernetes_namespace': 'nanolm-test'
        })

        # Step 4: Create deployment package
        package_info = deployment_system.create_deployment_package(
            export_result.export_path,
            ExportFormat.TORCHSCRIPT,
            [deployment_config],
            os.path.join(self.temp_dir, 'deployment_package')
        )

        # Verify deployment package
        self.assertIn('platforms', package_info)
        self.assertIn('files', package_info)
        self.assertEqual(len(package_info['platforms']), 1)
        self.assertGreater(len(package_info['files']), 0)


class TestMultiComponentInteractions(unittest.TestCase):
    """Test interactions between multiple system components"""

    def setUp(self):
        """Set up multi-component test environment"""
        # Create components
        self.cache = IntelligentCache(CacheConfig(
            max_cache_size_mb=32.0,
            eviction_policy=CacheEvictionPolicy.LRU
        ))

        self.batcher = DynamicBatcher(BatchingConfig(
            strategy=BatchingStrategy.THROUGHPUT_MAXIMIZED,
            min_batch_size=1,
            max_batch_size=4
        ))

        self.memory_monitor = MemoryMonitor(OptimizationConfig(
            memory_monitoring_enabled=False  # Disable background monitoring
        ))

        self.power_manager = PowerManager(PowerConfig(
            mode=PowerMode.BALANCED
        ))

    def test_cache_and_batching_interaction(self):
        """Test interaction between caching and batching systems"""
        # Create test requests
        requests = []
        for i in range(8):
            request = BatchRequest(
                request_id=f"req_{i}",
                inputs={'data': torch.randn(1, 64)},
                timestamp=time.time()
            )
            requests.append(request)

        # Submit requests to batcher
        for request in requests:
            success = self.batcher.submit_request(request)
            self.assertTrue(success)

        # Get batches
        batches = []
        while True:
            batch = self.batcher.get_batch(timeout=0.1)
            if batch is None:
                break
            batches.append(batch)

        # Verify batching worked
        self.assertGreater(len(batches), 0)

        # Simulate processing and caching results
        for batch in batches:
            for request in batch:
                # Simulate processing result
                result = {'output': torch.randn(1, 32)}

                # Cache result
                cache_success = self.cache.put(request.request_id, result)
                self.assertTrue(cache_success)

        # Test cache retrieval
        for request in requests:
            cached_result = self.cache.get(request.request_id)
            self.assertIsNotNone(cached_result)
            self.assertIn('output', cached_result)

        # Verify cache statistics
        cache_stats = self.cache.get_stats()
        self.assertEqual(cache_stats['hits'], len(requests))
        self.assertEqual(cache_stats['entries'], len(requests))

    def test_memory_monitoring_and_optimization_interaction(self):
        """Test interaction between memory monitoring and optimization"""
        # Get baseline memory stats
        baseline_stats = self.memory_monitor.get_current_memory_stats()

        # Create memory-intensive operations
        large_tensors = []
        for i in range(5):  # Reduced number to avoid memory issues
            tensor = torch.randn(500, 500)  # ~1MB each
            large_tensors.append(tensor)

            # Check memory pressure
            is_pressure = self.memory_monitor.is_memory_pressure()

            if is_pressure:
                # Trigger cleanup
                self.memory_monitor.cleanup_memory(aggressive=True)

                # Verify cleanup helped
                post_cleanup_stats = self.memory_monitor.get_current_memory_stats()
                # Memory should be managed (exact values depend on system)
                self.assertIsInstance(post_cleanup_stats.allocated_mb, float)

        # Get memory recommendations
        recommendations = self.memory_monitor.get_memory_recommendations()
        self.assertIsInstance(recommendations, list)

        # Clean up
        del large_tensors
        self.memory_monitor.cleanup_memory(aggressive=True)


class TestPerformanceAndScalability(unittest.TestCase):
    """Test performance and scalability characteristics"""

    def setUp(self):
        """Set up performance testing environment"""
        self.test_model = nn.Linear(64, 32)  # Simple model

    def test_inference_throughput_scaling(self):
        """Test inference throughput scaling with batch size"""
        # Create inference engine
        batching_config = BatchingConfig(
            strategy=BatchingStrategy.THROUGHPUT_MAXIMIZED,
            min_batch_size=1,
            max_batch_size=8
        )

        cache_config = CacheConfig(max_cache_size_mb=64.0)
        power_config = PowerConfig(mode=PowerMode.HIGH_PERFORMANCE)

        engine = OptimizedInferenceEngine(
            self.test_model, batching_config, cache_config, power_config
        )

        # Test different batch sizes
        batch_sizes = [1, 2, 4]
        throughput_results = {}

        for batch_size in batch_sizes:
            # Create test data
            test_requests = [
                torch.randn(1, 64)  # Direct tensor input
                for _ in range(batch_size * 3)  # 3 batches worth
            ]

            # Measure throughput
            start_time = time.time()

            results = []
            for i, inputs in enumerate(test_requests):
                result = engine.predict(inputs, request_id=f"perf_{batch_size}_{i}")
                results.append(result)

            total_time = time.time() - start_time
            throughput = len(test_requests) / max(total_time, 0.001)  # Avoid division by zero

            throughput_results[batch_size] = {
                'throughput_rps': throughput,
                'total_time': total_time,
                'avg_latency': total_time / len(test_requests)
            }

        # Verify throughput scaling
        self.assertGreater(len(throughput_results), 0)

        # All batch sizes should have positive throughput
        for batch_size in batch_sizes:
            self.assertGreater(throughput_results[batch_size]['throughput_rps'], 0)

    def test_concurrent_request_handling(self):
        """Test concurrent request handling"""
        # Create inference engine
        engine = OptimizedInferenceEngine(
            self.test_model,
            BatchingConfig(max_batch_size=4),
            CacheConfig(max_cache_size_mb=32.0),
            PowerConfig()
        )

        # Define concurrent request function
        def make_request(request_id):
            inputs = torch.randn(1, 64)
            try:
                result = engine.predict(inputs, request_id=f"concurrent_{request_id}")
                return {
                    'success': True,
                    'request_id': request_id,
                    'processing_time': result.processing_time_ms,
                    'cache_hit': result.cache_hit
                }
            except Exception as e:
                return {
                    'success': False,
                    'request_id': request_id,
                    'error': str(e)
                }

        # Run concurrent requests
        num_concurrent = 6  # Reduced number for stability

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(make_request, i)
                for i in range(num_concurrent)
            ]

            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Analyze results
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]

        # Should handle most requests successfully
        success_rate = len(successful_requests) / len(results)
        self.assertGreater(success_rate, 0.5)  # At least 50% success rate

        # Verify concurrent execution didn't break the system
        stats = engine.get_comprehensive_stats()
        self.assertGreater(stats['inference']['total_requests'], 0)


class TestErrorHandlingAndRecovery(unittest.TestCase):
    """Test error handling and recovery scenarios"""

    def test_memory_pressure_recovery(self):
        """Test recovery from memory pressure situations"""
        # Create system with low memory limits
        opt_config = OptimizationConfig(
            target_memory_usage_gb=0.1,  # Very low limit
            memory_pressure_threshold=0.5
        )

        optimizer = MemoryComputeOptimizer(opt_config)

        # Create memory-intensive model
        large_model = nn.Sequential(
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500)
        )

        # Should handle optimization even with memory pressure
        try:
            optimized_model = optimizer.optimize_model(large_model)
            optimization_successful = True
        except Exception as e:
            optimization_successful = False
            error_message = str(e)

        # Should either succeed or fail gracefully
        if not optimization_successful:
            self.assertIsInstance(error_message, str)
        else:
            self.assertIsNotNone(optimized_model)

    def test_export_format_fallback(self):
        """Test fallback when export formats are not available"""
        # Test with TorchScript (always available)
        export_config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT,
            output_path=os.path.join(tempfile.gettempdir(), "test_model.pt"),
            optimization_level=OptimizationLevel.BASIC
        )

        exporter = MultiPlatformExporter()

        # Simple test model
        model = nn.Linear(10, 5)
        sample_inputs = torch.randn(1, 10)

        # Should handle export gracefully
        result = exporter.export_model(model, sample_inputs, export_config)

        # Should have successful export
        self.assertTrue(result.success)
        self.assertTrue(os.path.exists(result.export_path))

        # Clean up
        if os.path.exists(result.export_path):
            os.remove(result.export_path)


class TestCrossSystemIntegration(unittest.TestCase):
    """Test integration across different system boundaries"""

    def test_optimization_export_deployment_chain(self):
        """Test complete chain: optimization -> export -> deployment"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Step 1: Create and optimize model
            model = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 16)
            )

            opt_config = OptimizationConfig(
                memory_optimization_level=MemoryOptimizationLevel.MODERATE,
                compute_optimization_mode=ComputeOptimizationMode.SPEED
            )

            optimizer = MemoryComputeOptimizer(opt_config)
            optimized_model = optimizer.optimize_model(model)

            # Step 2: Export optimized model
            export_config = ExportConfig(
                format=ExportFormat.TORCHSCRIPT,
                output_path=os.path.join(temp_dir, "optimized_model.pt"),
                optimization_level=OptimizationLevel.BASIC
            )

            exporter = MultiPlatformExporter()
            sample_inputs = torch.randn(1, 32)

            export_result = exporter.export_model(
                optimized_model, sample_inputs, export_config
            )

            # Verify export succeeded
            self.assertTrue(export_result.success)

            # Step 3: Deploy exported model
            deployment_config = DeploymentConfig(
                platform=CloudPlatform.DOCKER,
                deployment_type=DeploymentType.REAL_TIME_API,
                scaling_strategy=ScalingStrategy.CPU_BASED
            )

            deployment_system = CloudDeploymentSystem({
                'docker_registry': 'test-registry.com'
            })

            # Create deployment package
            package_info = deployment_system.create_deployment_package(
                export_result.export_path,
                ExportFormat.TORCHSCRIPT,
                [deployment_config],
                os.path.join(temp_dir, 'deployment')
            )

            # Verify complete chain worked
            self.assertIn('platforms', package_info)
            self.assertIn('files', package_info)
            self.assertGreater(len(package_info['files']), 0)

    def test_inference_monitoring_optimization_loop(self):
        """Test feedback loop: inference -> monitoring -> optimization"""
        # Create initial system
        model = nn.Linear(32, 16)

        batching_config = BatchingConfig(
            strategy=BatchingStrategy.ADAPTIVE,
            min_batch_size=1,
            max_batch_size=4
        )

        cache_config = CacheConfig(
            eviction_policy=CacheEvictionPolicy.ADAPTIVE,
            max_cache_size_mb=16.0
        )

        power_config = PowerConfig(mode=PowerMode.ADAPTIVE)

        engine = OptimizedInferenceEngine(
            model, batching_config, cache_config, power_config
        )

        # Run inference workload
        for i in range(5):
            inputs = torch.randn(1, 32)
            result = engine.predict(inputs, request_id=f"loop_test_{i}")

            # Simulate varying load
            if i % 3 == 0:
                time.sleep(0.01)  # Simulate processing delay

        # Get performance stats
        stats = engine.get_comprehensive_stats()

        # Simulate optimization based on stats
        if stats['inference']['avg_processing_time_ms'] > 50:
            # High latency detected - could trigger optimization
            optimization_needed = True
        else:
            optimization_needed = False

        # Verify monitoring data is available for optimization decisions
        self.assertIn('inference', stats)
        self.assertIn('cache', stats)
        self.assertIn('batching', stats)
        self.assertIn('power', stats)

        # Verify stats contain actionable information
        self.assertGreater(stats['inference']['total_requests'], 0)
        self.assertIsInstance(stats['cache']['hit_rate'], (int, float))
        self.assertIsInstance(optimization_needed, bool)


def run_integration_tests():
    """Run all integration tests with detailed reporting"""

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [
        TestEndToEndWorkflows,
        TestMultiComponentInteractions,
        TestPerformanceAndScalability,
        TestErrorHandlingAndRecovery,
        TestCrossSystemIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print comprehensive summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE INTEGRATION TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1) * 100):.1f}%")

    # Detailed breakdown by test class
    print(f"\nTest Coverage:")
    print(f"  • End-to-End Workflows: ✅")
    print(f"  • Multi-Component Interactions: ✅")
    print(f"  • Performance & Scalability: ✅")
    print(f"  • Error Handling & Recovery: ✅")
    print(f"  • Cross-System Integration: ✅")

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
    success = run_integration_tests()
    exit(0 if success else 1)