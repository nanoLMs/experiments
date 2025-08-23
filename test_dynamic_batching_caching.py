#!/usr/bin/env python3
"""
Test Suite for Dynamic Batching and Caching System
=================================================

Comprehensive tests for the dynamic batching and caching optimization system.
"""

import unittest
import time
import threading
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import the system under test
from dynamic_batching_caching import (
    IntelligentCache,
    DynamicBatcher,
    PowerManager,
    OptimizedInferenceEngine,
    CacheConfig,
    BatchingConfig,
    PowerConfig,
    BatchingStrategy,
    CacheEvictionPolicy,
    PowerMode,
    BatchRequest,
    BatchResult,
    CacheEntry
)


class TestCacheEntry(unittest.TestCase):
    """Test cache entry functionality"""

    def test_cache_entry_creation(self):
        """Test cache entry creation"""
        entry = CacheEntry("test_key", "test_value", 100)

        self.assertEqual(entry.key, "test_key")
        self.assertEqual(entry.value, "test_value")
        self.assertEqual(entry.size_bytes, 100)
        self.assertEqual(entry.access_count, 1)
        self.assertFalse(entry.compressed)

    def test_cache_entry_access(self):
        """Test cache entry access tracking"""
        entry = CacheEntry("test_key", "test_value")
        initial_access_time = entry.last_accessed
        initial_count = entry.access_count

        time.sleep(0.01)  # Small delay
        entry.access()

        self.assertGreater(entry.last_accessed, initial_access_time)
        self.assertEqual(entry.access_count, initial_count + 1)

    def test_cache_entry_age_and_idle(self):
        """Test age and idle time calculations"""
        entry = CacheEntry("test_key", "test_value")

        time.sleep(0.01)

        self.assertGreater(entry.age(), 0)
        self.assertGreater(entry.idle_time(), 0)


class TestIntelligentCache(unittest.TestCase):
    """Test intelligent caching system"""

    def setUp(self):
        self.config = CacheConfig(
            max_cache_size_mb=1.0,  # 1MB for testing
            max_entries=10,
            ttl_seconds=1.0,  # 1 second for testing
            eviction_policy=CacheEvictionPolicy.LRU
        )
        self.cache = IntelligentCache(self.config)

    def test_cache_put_and_get(self):
        """Test basic cache put and get operations"""
        self.assertTrue(self.cache.put("key1", "value1"))
        self.assertEqual(self.cache.get("key1"), "value1")
        self.assertIsNone(self.cache.get("nonexistent"))

    def test_cache_hit_miss_stats(self):
        """Test cache hit/miss statistics"""
        # Initial state
        stats = self.cache.get_stats()
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)

        # Miss
        self.assertIsNone(self.cache.get("key1"))
        stats = self.cache.get_stats()
        self.assertEqual(stats["misses"], 1)

        # Put and hit
        self.cache.put("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")
        stats = self.cache.get_stats()
        self.assertEqual(stats["hits"], 1)

    def test_lru_eviction(self):
        """Test LRU eviction policy"""
        # Fill cache to capacity
        for i in range(self.config.max_entries):
            self.cache.put(f"key{i}", f"value{i}")

        # Access first key to make it recently used
        self.cache.get("key0")

        # Add one more item to trigger eviction
        self.cache.put("new_key", "new_value")

        # key0 should still be there (recently accessed)
        self.assertEqual(self.cache.get("key0"), "value0")

        # key1 should be evicted (least recently used)
        self.assertIsNone(self.cache.get("key1"))

    def test_ttl_eviction(self):
        """Test TTL-based eviction"""
        ttl_config = CacheConfig(
            eviction_policy=CacheEvictionPolicy.TTL,
            ttl_seconds=0.1  # 100ms
        )
        ttl_cache = IntelligentCache(ttl_config)

        # Put item
        ttl_cache.put("key1", "value1")
        self.assertEqual(ttl_cache.get("key1"), "value1")

        # Wait for TTL expiration
        time.sleep(0.15)

        # Item should be expired
        self.assertIsNone(ttl_cache.get("key1"))

    def test_size_based_eviction(self):
        """Test size-based eviction"""
        size_config = CacheConfig(
            eviction_policy=CacheEvictionPolicy.SIZE_BASED,
            max_cache_size_mb=0.001  # Very small cache
        )
        size_cache = IntelligentCache(size_config)

        # Put small item
        size_cache.put("small", "x", size_hint=100)

        # Put large item that should trigger eviction
        size_cache.put("large", "x" * 1000, size_hint=1000)

        # Small item should be evicted
        self.assertIsNone(size_cache.get("small"))
        self.assertEqual(size_cache.get("large"), "x" * 1000)

    def test_cache_invalidation(self):
        """Test cache invalidation"""
        self.cache.put("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")

        # Invalidate
        self.assertTrue(self.cache.invalidate("key1"))
        self.assertIsNone(self.cache.get("key1"))

        # Invalidate non-existent key
        self.assertFalse(self.cache.invalidate("nonexistent"))

    def test_cache_clear(self):
        """Test cache clearing"""
        # Add items
        for i in range(5):
            self.cache.put(f"key{i}", f"value{i}")

        self.assertEqual(len(self.cache._cache), 5)

        # Clear cache
        self.cache.clear()

        self.assertEqual(len(self.cache._cache), 0)
        self.assertEqual(self.cache.total_size_bytes, 0)


class TestDynamicBatcher(unittest.TestCase):
    """Test dynamic batching system"""

    def setUp(self):
        self.config = BatchingConfig(
            strategy=BatchingStrategy.ADAPTIVE,
            min_batch_size=1,
            max_batch_size=4,
            target_latency_ms=50.0,
            batch_formation_timeout_ms=10.0
        )
        self.batcher = DynamicBatcher(self.config)

    def test_request_submission(self):
        """Test request submission"""
        request = BatchRequest(
            request_id="test_1",
            inputs={"x": torch.randn(1, 10)},
            timestamp=time.time()
        )

        self.assertTrue(self.batcher.submit_request(request))

    def test_batch_formation(self):
        """Test batch formation"""
        # Submit multiple requests
        requests = []
        for i in range(3):
            request = BatchRequest(
                request_id=f"test_{i}",
                inputs={"x": torch.randn(1, 10)},
                timestamp=time.time()
            )
            requests.append(request)
            self.batcher.submit_request(request)

        # Get batch
        batch = self.batcher.get_batch(timeout=1.0)

        self.assertIsNotNone(batch)
        self.assertGreater(len(batch), 0)
        self.assertLessEqual(len(batch), self.config.max_batch_size)

    def test_performance_metrics_update(self):
        """Test performance metrics updating"""
        initial_stats = self.batcher.get_stats()

        # Update metrics
        self.batcher.update_performance_metrics(
            batch_size=2,
            latency_ms=30.0,
            memory_mb=100.0
        )

        updated_stats = self.batcher.get_stats()

        self.assertGreater(updated_stats["processed_requests"], initial_stats["processed_requests"])
        self.assertGreater(updated_stats["total_batches"], initial_stats["total_batches"])

    def test_adaptive_batch_sizing(self):
        """Test adaptive batch size adjustment"""
        initial_batch_size = self.batcher._current_batch_size

        # Simulate high latency to trigger batch size reduction
        for _ in range(15):  # Need enough samples for adaptation
            self.batcher.update_performance_metrics(
                batch_size=self.batcher._current_batch_size,
                latency_ms=self.config.target_latency_ms * 2,  # High latency
                memory_mb=50.0
            )

        # Batch size should decrease due to high latency
        self.assertLessEqual(self.batcher._current_batch_size, initial_batch_size)

    def test_memory_constrained_batching(self):
        """Test memory-constrained batching strategy"""
        memory_config = BatchingConfig(
            strategy=BatchingStrategy.MEMORY_CONSTRAINED,
            memory_limit_mb=100.0
        )
        memory_batcher = DynamicBatcher(memory_config)

        initial_batch_size = memory_batcher._current_batch_size

        # Simulate high memory usage
        for _ in range(15):
            memory_batcher.update_performance_metrics(
                batch_size=memory_batcher._current_batch_size,
                latency_ms=30.0,
                memory_mb=memory_config.memory_limit_mb * 0.95  # High memory
            )

        # Batch size should decrease due to memory pressure
        self.assertLessEqual(memory_batcher._current_batch_size, initial_batch_size)


class TestPowerManager(unittest.TestCase):
    """Test power management system"""

    def setUp(self):
        self.config = PowerConfig(
            mode=PowerMode.ADAPTIVE,
            cpu_frequency_scaling=True,
            battery_threshold_low=0.3,
            thermal_threshold=70.0
        )
        self.power_manager = PowerManager(self.config)

    def test_power_mode_switching(self):
        """Test power mode switching"""
        initial_mode = self.power_manager.current_mode

        # Switch to power saver mode
        self.power_manager.set_power_mode(PowerMode.POWER_SAVER)

        self.assertEqual(self.power_manager.current_mode, PowerMode.POWER_SAVER)
        self.assertNotEqual(self.power_manager.current_mode, initial_mode)

    def test_power_scaling_factor(self):
        """Test power scaling factor calculation"""
        # Test different power modes
        test_cases = [
            (PowerMode.HIGH_PERFORMANCE, 1.0),
            (PowerMode.BALANCED, 0.8),
            (PowerMode.POWER_SAVER, 0.6),
            (PowerMode.ULTRA_LOW_POWER, 0.4)
        ]

        for mode, expected_factor in test_cases:
            self.power_manager.set_power_mode(mode)
            factor = self.power_manager.get_power_scaling_factor()
            self.assertEqual(factor, expected_factor)

    def test_power_recommendations(self):
        """Test power optimization recommendations"""
        # Update with high temperature
        self.power_manager.update_power_metrics(
            temperature_celsius=self.config.thermal_threshold + 10
        )

        recommendations = self.power_manager.get_power_recommendations()

        self.assertIn("recommendations", recommendations)
        self.assertIn("current_mode", recommendations)

        # Should have thermal throttling recommendation
        thermal_recs = [r for r in recommendations["recommendations"]
                       if r["type"] == "thermal_throttling"]
        self.assertGreater(len(thermal_recs), 0)

    @patch('dynamic_batching_caching.PSUTIL_AVAILABLE', True)
    @patch('psutil.sensors_battery')
    def test_battery_recommendations(self, mock_battery):
        """Test battery-based recommendations"""
        # Mock low battery
        mock_battery.return_value = Mock(percent=15.0)  # 15% battery

        recommendations = self.power_manager.get_power_recommendations()

        # Should have low battery recommendation
        battery_recs = [r for r in recommendations["recommendations"]
                       if r["type"] == "low_battery"]
        self.assertGreater(len(battery_recs), 0)


class TestOptimizedInferenceEngine(unittest.TestCase):
    """Test optimized inference engine"""

    def setUp(self):
        # Create simple test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        self.model = TestModel()
        self.model.eval()

        # Configure components
        self.batching_config = BatchingConfig(
            min_batch_size=1,
            max_batch_size=4,
            target_latency_ms=50.0
        )

        self.cache_config = CacheConfig(
            max_cache_size_mb=10.0,
            max_entries=100
        )

        self.power_config = PowerConfig(
            mode=PowerMode.BALANCED
        )

        self.engine = OptimizedInferenceEngine(
            self.model, self.batching_config, self.cache_config, self.power_config
        )

    def test_single_prediction(self):
        """Test single prediction"""
        inputs = {"x": torch.randn(1, 10)}

        result = self.engine.predict(inputs)

        self.assertIsInstance(result, BatchResult)
        self.assertIn("output", result.outputs)
        self.assertIsInstance(result.outputs["output"], torch.Tensor)
        self.assertGreater(result.processing_time_ms, 0)
        self.assertFalse(result.cache_hit)  # First time should not be cache hit

    def test_cache_hit(self):
        """Test cache hit functionality"""
        inputs = {"x": torch.randn(1, 10)}

        # First prediction
        result1 = self.engine.predict(inputs, request_id="test_cache")
        self.assertFalse(result1.cache_hit)

        # Second prediction with same request_id should hit cache
        result2 = self.engine.predict(inputs, request_id="test_cache")
        self.assertTrue(result2.cache_hit)
        self.assertEqual(result2.processing_time_ms, 0.0)

    def test_request_id_generation(self):
        """Test automatic request ID generation"""
        inputs = {"x": torch.randn(1, 10)}

        # Generate request ID
        request_id1 = self.engine._generate_request_id(inputs)
        request_id2 = self.engine._generate_request_id(inputs)

        # Same inputs should generate same ID
        self.assertEqual(request_id1, request_id2)

        # Different inputs should generate different ID
        different_inputs = {"x": torch.randn(1, 10)}
        request_id3 = self.engine._generate_request_id(different_inputs)
        self.assertNotEqual(request_id1, request_id3)

    def test_batch_input_combination(self):
        """Test batch input combination"""
        batch = [
            BatchRequest("req1", {"x": torch.randn(1, 10)}, time.time()),
            BatchRequest("req2", {"x": torch.randn(1, 10)}, time.time()),
            BatchRequest("req3", {"x": torch.randn(1, 10)}, time.time())
        ]

        combined = self.engine._combine_batch_inputs(batch)

        self.assertIn("x", combined)
        self.assertEqual(combined["x"].shape[0], 3)  # Batch size 3
        self.assertEqual(combined["x"].shape[1], 10)  # Feature size 10

    def test_batch_output_splitting(self):
        """Test batch output splitting"""
        batch_size = 3
        batch_output = torch.randn(batch_size, 5)

        individual_outputs = self.engine._split_batch_outputs(batch_output, batch_size)

        self.assertEqual(len(individual_outputs), batch_size)
        for output in individual_outputs:
            self.assertIn("output", output)
            self.assertEqual(output["output"].shape, (5,))

    def test_comprehensive_stats(self):
        """Test comprehensive statistics"""
        # Make some predictions to generate stats
        for i in range(5):
            inputs = {"x": torch.randn(1, 10)}
            self.engine.predict(inputs, request_id=f"test_{i}")

        stats = self.engine.get_comprehensive_stats()

        # Check structure
        self.assertIn("inference", stats)
        self.assertIn("cache", stats)
        self.assertIn("batching", stats)
        self.assertIn("power", stats)

        # Check inference stats
        inference_stats = stats["inference"]
        self.assertGreater(inference_stats["total_requests"], 0)
        self.assertGreaterEqual(inference_stats["cache_hit_rate"], 0.0)
        self.assertLessEqual(inference_stats["cache_hit_rate"], 1.0)

    @patch('dynamic_batching_caching.PSUTIL_AVAILABLE', True)
    @patch('psutil.Process')
    def test_memory_estimation(self, mock_process):
        """Test memory usage estimation"""
        # Mock memory info
        mock_process.return_value.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB

        memory_usage = self.engine._estimate_memory_usage()

        self.assertGreater(memory_usage, 0)
        self.assertEqual(memory_usage, 100.0)  # Should be 100MB


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Create model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(5, 3)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        model.eval()

        # Configure system
        batching_config = BatchingConfig(
            strategy=BatchingStrategy.ADAPTIVE,
            min_batch_size=1,
            max_batch_size=3,
            target_latency_ms=100.0
        )

        cache_config = CacheConfig(
            eviction_policy=CacheEvictionPolicy.LRU,
            max_cache_size_mb=5.0,
            max_entries=50
        )

        power_config = PowerConfig(
            mode=PowerMode.BALANCED
        )

        # Create engine
        engine = OptimizedInferenceEngine(model, batching_config, cache_config, power_config)

        # Test multiple predictions
        results = []
        for i in range(10):
            inputs = {"x": torch.randn(1, 5)}
            result = engine.predict(inputs, request_id=f"test_{i % 3}")  # Some duplicates for cache testing
            results.append(result)

        # Verify results
        self.assertEqual(len(results), 10)

        # Check that some requests hit cache (due to duplicate request_ids)
        cache_hits = sum(1 for r in results if r.cache_hit)
        self.assertGreater(cache_hits, 0)

        # Get final statistics
        stats = engine.get_comprehensive_stats()

        self.assertGreater(stats["inference"]["total_requests"], 0)
        self.assertGreater(stats["cache"]["entries"], 0)
        self.assertGreaterEqual(stats["inference"]["cache_hit_rate"], 0.0)

    def test_performance_under_load(self):
        """Test system performance under load"""
        # Create model
        class LoadTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(20, 50),
                    nn.ReLU(),
                    nn.Linear(50, 10)
                )

            def forward(self, x):
                return self.layers(x)

        model = LoadTestModel()
        model.eval()

        # Configure for performance
        batching_config = BatchingConfig(
            strategy=BatchingStrategy.THROUGHPUT_MAXIMIZED,
            min_batch_size=1,
            max_batch_size=8,
            target_latency_ms=200.0
        )

        cache_config = CacheConfig(
            eviction_policy=CacheEvictionPolicy.ADAPTIVE,
            max_cache_size_mb=20.0,
            max_entries=200
        )

        power_config = PowerConfig(mode=PowerMode.HIGH_PERFORMANCE)

        engine = OptimizedInferenceEngine(model, batching_config, cache_config, power_config)

        # Generate load
        num_requests = 50
        start_time = time.time()

        results = []
        for i in range(num_requests):
            inputs = {"x": torch.randn(1, 20)}
            result = engine.predict(inputs, request_id=f"load_test_{i % 10}")  # Some cache hits
            results.append(result)

        total_time = time.time() - start_time

        # Verify performance
        self.assertEqual(len(results), num_requests)

        # Calculate throughput
        throughput = num_requests / total_time
        self.assertGreater(throughput, 10)  # At least 10 requests per second

        # Check cache effectiveness
        cache_hits = sum(1 for r in results if r.cache_hit)
        cache_hit_rate = cache_hits / num_requests
        self.assertGreater(cache_hit_rate, 0.1)  # At least 10% cache hit rate

        # Get final stats
        stats = engine.get_comprehensive_stats()
        print(f"Load test results:")
        print(f"  Throughput: {throughput:.2f} req/s")
        print(f"  Cache hit rate: {cache_hit_rate:.2%}")
        print(f"  Avg processing time: {stats['inference']['avg_processing_time_ms']:.2f}ms")


def run_comprehensive_tests():
    """Run all tests with detailed output"""

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestCacheEntry,
        TestIntelligentCache,
        TestDynamicBatcher,
        TestPowerManager,
        TestOptimizedInferenceEngine,
        TestIntegration
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary
    print(f"\n{'='*60}")
    print("DYNAMIC BATCHING & CACHING TEST SUMMARY")
    print(f"{'='*60}")
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