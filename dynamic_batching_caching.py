#!/usr/bin/env python3
"""
Dynamic Batching and Caching Strategies
=======================================

Advanced performance optimization system including:
- Dynamic batch size adjustment based on memory and latency constraints
- Intelligent caching for edge devices with memory-aware eviction
- Low-power inference modes with adaptive computation
- Request queuing and batching optimization
- Cache warming and prefetching strategies
- Memory-efficient batch processing
"""

import torch
import torch.nn as nn
import time
import logging
import threading
import queue
import hashlib
import pickle
import os
import json
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import asyncio
from collections import OrderedDict, deque
import weakref
import gc

# Optional imports with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class BatchingStrategy(Enum):
    """Dynamic batching strategies"""
    MEMORY_CONSTRAINED = "memory_constrained"
    LATENCY_OPTIMIZED = "latency_optimized"
    THROUGHPUT_MAXIMIZED = "throughput_maximized"
    ADAPTIVE = "adaptive"
    POWER_EFFICIENT = "power_efficient"


class CacheEvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    SIZE_BASED = "size_based"
    MEMORY_PRESSURE = "memory_pressure"
    ADAPTIVE = "adaptive"


class PowerMode(Enum):
    """Power consumption modes"""
    HIGH_PERFORMANCE = "high_performance"
    BALANCED = "balanced"
    POWER_SAVER = "power_saver"
    ULTRA_LOW_POWER = "ultra_low_power"
    ADAPTIVE = "adaptive"


@dataclass
class BatchingConfig:
    """Configuration for dynamic batching"""
    strategy: BatchingStrategy = BatchingStrategy.ADAPTIVE
    min_batch_size: int = 1
    max_batch_size: int = 32
    target_latency_ms: float = 100.0
    memory_limit_mb: float = 512.0
    queue_timeout_ms: float = 50.0

    # Adaptive parameters
    latency_weight: float = 0.5
    throughput_weight: float = 0.3
    memory_weight: float = 0.2

    # Performance thresholds
    max_queue_size: int = 1000
    batch_formation_timeout_ms: float = 10.0
    enable_prefetching: bool = True


@dataclass
class CacheConfig:
    """Configuration for intelligent caching"""
    eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.ADAPTIVE
    max_cache_size_mb: float = 256.0
    max_entries: int = 10000
    ttl_seconds: float = 3600.0  # 1 hour

    # Memory management
    memory_pressure_threshold: float = 0.8
    cleanup_batch_size: int = 100
    enable_compression: bool = True

    # Cache warming
    enable_warming: bool = True
    warming_batch_size: int = 10
    prefetch_probability: float = 0.1


@dataclass
class PowerConfig:
    """Configuration for power management"""
    mode: PowerMode = PowerMode.ADAPTIVE
    cpu_frequency_scaling: bool = True
    gpu_power_limit: Optional[float] = None

    # Adaptive thresholds
    battery_threshold_low: float = 0.2
    battery_threshold_critical: float = 0.1
    thermal_threshold: float = 80.0  # Celsius

    # Power-saving features
    enable_model_pruning: bool = True
    enable_quantization_scaling: bool = True
    enable_compute_skipping: bool = True


@dataclass
class BatchRequest:
    """Individual request in a batch"""
    request_id: str
    inputs: Dict[str, torch.Tensor]
    timestamp: float
    priority: int = 0
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result of batch processing"""
    request_id: str
    outputs: Dict[str, torch.Tensor]
    processing_time_ms: float
    cache_hit: bool = False
    batch_size: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class CacheEntry:
    """Cache entry with metadata"""

    def __init__(self, key: str, value: Any, size_bytes: int = 0):
        self.key = key
        self.value = value
        self.size_bytes = size_bytes
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 1
        self.compressed = False

    def access(self):
        """Mark entry as accessed"""
        self.last_accessed = time.time()
        self.access_count += 1

    def age(self) -> float:
        """Get age in seconds"""
        return time.time() - self.created_at

    def idle_time(self) -> float:
        """Get idle time in seconds"""
        return time.time() - self.last_accessed


class IntelligentCache:
    """Intelligent caching system with adaptive eviction"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Cache storage
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_size_bytes = 0

        # Background cleanup
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()

        if self.config.eviction_policy in [CacheEvictionPolicy.MEMORY_PRESSURE, CacheEvictionPolicy.ADAPTIVE]:
            self._start_cleanup_thread()

        self.logger.info(f"✅ Intelligent cache initialized")
        self.logger.info(f"  • Policy: {config.eviction_policy.value}")
        self.logger.info(f"  • Max size: {config.max_cache_size_mb}MB")
        self.logger.info(f"  • Max entries: {config.max_entries}")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]

                # Check TTL
                if self.config.eviction_policy == CacheEvictionPolicy.TTL:
                    if entry.age() > self.config.ttl_seconds:
                        self._remove_entry(key)
                        self.misses += 1
                        return None

                # Update access info
                entry.access()

                # Move to end for LRU
                if self.config.eviction_policy == CacheEvictionPolicy.LRU:
                    self._cache.move_to_end(key)

                self.hits += 1
                return entry.value
            else:
                self.misses += 1
                return None

    def put(self, key: str, value: Any, size_hint: Optional[int] = None) -> bool:
        """Put value in cache"""
        with self._lock:
            # Calculate size
            if size_hint is not None:
                size_bytes = size_hint
            else:
                size_bytes = self._estimate_size(value)

            # Check if we need to make space
            if not self._make_space(size_bytes):
                return False

            # Compress if enabled and beneficial
            if self.config.enable_compression and size_bytes > 1024:
                try:
                    compressed_value = self._compress(value)
                    if len(compressed_value) < size_bytes * 0.8:  # 20% compression threshold
                        value = compressed_value
                        size_bytes = len(compressed_value)
                except Exception as e:
                    self.logger.debug(f"Compression failed for {key}: {e}")

            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)

            # Add new entry
            entry = CacheEntry(key, value, size_bytes)
            self._cache[key] = entry
            self.total_size_bytes += size_bytes

            return True

    def invalidate(self, key: str) -> bool:
        """Remove entry from cache"""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
            self.total_size_bytes = 0
            self.logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "entries": len(self._cache),
                "size_mb": self.total_size_bytes / (1024 * 1024),
                "avg_entry_size_kb": (self.total_size_bytes / len(self._cache) / 1024) if self._cache else 0
            }

    def _make_space(self, required_bytes: int) -> bool:
        """Make space for new entry"""
        max_size_bytes = self.config.max_cache_size_mb * 1024 * 1024

        # Check if we have space
        if (self.total_size_bytes + required_bytes <= max_size_bytes and
            len(self._cache) < self.config.max_entries):
            return True

        # Need to evict entries
        evicted_bytes = 0
        evicted_count = 0

        while ((self.total_size_bytes + required_bytes - evicted_bytes > max_size_bytes or
                len(self._cache) - evicted_count >= self.config.max_entries) and
               self._cache):

            victim_key = self._select_eviction_victim()
            if victim_key is None:
                break

            victim_entry = self._cache[victim_key]
            evicted_bytes += victim_entry.size_bytes
            evicted_count += 1

            self._remove_entry(victim_key)
            self.evictions += 1

        return (self.total_size_bytes + required_bytes <= max_size_bytes and
                len(self._cache) < self.config.max_entries)

    def _select_eviction_victim(self) -> Optional[str]:
        """Select entry to evict based on policy"""
        if not self._cache:
            return None

        if self.config.eviction_policy == CacheEvictionPolicy.LRU:
            # First item is least recently used
            return next(iter(self._cache))

        elif self.config.eviction_policy == CacheEvictionPolicy.LFU:
            # Find least frequently used
            min_access_count = float('inf')
            victim_key = None
            for key, entry in self._cache.items():
                if entry.access_count < min_access_count:
                    min_access_count = entry.access_count
                    victim_key = key
            return victim_key

        elif self.config.eviction_policy == CacheEvictionPolicy.TTL:
            # Find oldest entry
            oldest_time = float('inf')
            victim_key = None
            for key, entry in self._cache.items():
                if entry.created_at < oldest_time:
                    oldest_time = entry.created_at
                    victim_key = key
            return victim_key

        elif self.config.eviction_policy == CacheEvictionPolicy.SIZE_BASED:
            # Find largest entry
            max_size = 0
            victim_key = None
            for key, entry in self._cache.items():
                if entry.size_bytes > max_size:
                    max_size = entry.size_bytes
                    victim_key = key
            return victim_key

        elif self.config.eviction_policy in [CacheEvictionPolicy.MEMORY_PRESSURE, CacheEvictionPolicy.ADAPTIVE]:
            # Adaptive scoring
            return self._adaptive_eviction_victim()

        else:
            # Default to LRU
            return next(iter(self._cache))

    def _adaptive_eviction_victim(self) -> Optional[str]:
        """Select victim using adaptive scoring"""
        if not self._cache:
            return None

        best_score = float('-inf')
        victim_key = None

        current_time = time.time()

        for key, entry in self._cache.items():
            # Calculate composite score (higher = more likely to evict)
            age_score = entry.age() / 3600.0  # Normalize to hours
            idle_score = entry.idle_time() / 3600.0
            frequency_score = 1.0 / (entry.access_count + 1)
            size_score = entry.size_bytes / (1024 * 1024)  # Normalize to MB

            # Weighted combination
            composite_score = (
                0.3 * age_score +
                0.4 * idle_score +
                0.2 * frequency_score +
                0.1 * size_score
            )

            if composite_score > best_score:
                best_score = composite_score
                victim_key = key

        return victim_key

    def _remove_entry(self, key: str):
        """Remove entry and update statistics"""
        if key in self._cache:
            entry = self._cache.pop(key)
            self.total_size_bytes -= entry.size_bytes

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes"""
        try:
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            elif isinstance(value, (list, tuple)):
                return sum(self._estimate_size(item) for item in value)
            elif isinstance(value, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in value.items())
            else:
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default estimate

    def _compress(self, value: Any) -> bytes:
        """Compress value"""
        import gzip
        return gzip.compress(pickle.dumps(value))

    def _decompress(self, compressed_data: bytes) -> Any:
        """Decompress value"""
        import gzip
        return pickle.loads(gzip.decompress(compressed_data))

    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
            self._cleanup_thread.start()

    def _cleanup_worker(self):
        """Background cleanup worker"""
        while not self._stop_cleanup.wait(30):  # Check every 30 seconds
            try:
                self._periodic_cleanup()
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")

    def _periodic_cleanup(self):
        """Perform periodic cleanup"""
        with self._lock:
            if not self._cache:
                return

            # Check memory pressure
            if PSUTIL_AVAILABLE:
                memory_percent = psutil.virtual_memory().percent / 100.0
                if memory_percent > self.config.memory_pressure_threshold:
                    # Aggressive cleanup
                    cleanup_count = min(len(self._cache) // 4, self.config.cleanup_batch_size)
                    for _ in range(cleanup_count):
                        victim_key = self._select_eviction_victim()
                        if victim_key:
                            self._remove_entry(victim_key)
                            self.evictions += 1

            # TTL cleanup
            if self.config.eviction_policy in [CacheEvictionPolicy.TTL, CacheEvictionPolicy.ADAPTIVE]:
                expired_keys = []
                current_time = time.time()

                for key, entry in self._cache.items():
                    if entry.age() > self.config.ttl_seconds:
                        expired_keys.append(key)

                for key in expired_keys:
                    self._remove_entry(key)
                    self.evictions += 1

    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, '_stop_cleanup'):
            self._stop_cleanup.set()


class DynamicBatcher:
    """Dynamic batching system with adaptive sizing"""

    def __init__(self, config: BatchingConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Request queue
        self._request_queue = queue.Queue(maxsize=config.max_queue_size)
        self._batch_queue = queue.Queue()

        # Statistics
        self.processed_requests = 0
        self.total_batches = 0
        self.avg_batch_size = 0.0
        self.avg_latency_ms = 0.0

        # Adaptive parameters
        self._current_batch_size = config.min_batch_size
        self._latency_history = deque(maxlen=100)
        self._memory_history = deque(maxlen=50)
        self._throughput_history = deque(maxlen=50)

        # Worker threads
        self._batch_worker = None
        self._stop_workers = threading.Event()

        self._start_batch_worker()

        self.logger.info(f"✅ Dynamic batcher initialized")
        self.logger.info(f"  • Strategy: {config.strategy.value}")
        self.logger.info(f"  • Batch size range: {config.min_batch_size}-{config.max_batch_size}")
        self.logger.info(f"  • Target latency: {config.target_latency_ms}ms")

    def submit_request(self, request: BatchRequest) -> bool:
        """Submit request for batching"""
        try:
            self._request_queue.put(request, timeout=0.1)
            return True
        except queue.Full:
            self.logger.warning("Request queue full, dropping request")
            return False

    def get_batch(self, timeout: Optional[float] = None) -> Optional[List[BatchRequest]]:
        """Get next batch of requests"""
        try:
            return self._batch_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def update_performance_metrics(self, batch_size: int, latency_ms: float, memory_mb: float):
        """Update performance metrics for adaptive batching"""
        self._latency_history.append(latency_ms)
        self._memory_history.append(memory_mb)

        # Calculate throughput (requests per second)
        throughput = batch_size / (latency_ms / 1000.0)
        self._throughput_history.append(throughput)

        # Update statistics
        self.processed_requests += batch_size
        self.total_batches += 1
        self.avg_batch_size = self.processed_requests / self.total_batches
        self.avg_latency_ms = sum(self._latency_history) / len(self._latency_history)

        # Adapt batch size
        self._adapt_batch_size()

    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics"""
        return {
            "processed_requests": self.processed_requests,
            "total_batches": self.total_batches,
            "avg_batch_size": self.avg_batch_size,
            "avg_latency_ms": self.avg_latency_ms,
            "current_batch_size": self._current_batch_size,
            "queue_size": self._request_queue.qsize(),
            "strategy": self.config.strategy.value
        }

    def _start_batch_worker(self):
        """Start batch formation worker"""
        if self._batch_worker is None or not self._batch_worker.is_alive():
            self._batch_worker = threading.Thread(target=self._batch_worker_loop, daemon=True)
            self._batch_worker.start()

    def _batch_worker_loop(self):
        """Main batch formation loop"""
        while not self._stop_workers.is_set():
            try:
                batch = self._form_batch()
                if batch:
                    self._batch_queue.put(batch)
            except Exception as e:
                self.logger.error(f"Batch worker error: {e}")

    def _form_batch(self) -> Optional[List[BatchRequest]]:
        """Form a batch of requests"""
        batch = []
        batch_start_time = time.time()

        # Get first request (blocking)
        try:
            first_request = self._request_queue.get(timeout=1.0)
            batch.append(first_request)
        except queue.Empty:
            return None

        # Collect additional requests up to batch size or timeout
        while (len(batch) < self._current_batch_size and
               (time.time() - batch_start_time) * 1000 < self.config.batch_formation_timeout_ms):

            try:
                request = self._request_queue.get(timeout=0.001)
                batch.append(request)
            except queue.Empty:
                break

        return batch if batch else None

    def _adapt_batch_size(self):
        """Adapt batch size based on performance metrics"""
        if len(self._latency_history) < 10:
            return  # Need more data

        if self.config.strategy == BatchingStrategy.MEMORY_CONSTRAINED:
            self._adapt_for_memory()
        elif self.config.strategy == BatchingStrategy.LATENCY_OPTIMIZED:
            self._adapt_for_latency()
        elif self.config.strategy == BatchingStrategy.THROUGHPUT_MAXIMIZED:
            self._adapt_for_throughput()
        elif self.config.strategy == BatchingStrategy.ADAPTIVE:
            self._adapt_adaptive()
        elif self.config.strategy == BatchingStrategy.POWER_EFFICIENT:
            self._adapt_for_power()

    def _adapt_for_memory(self):
        """Adapt batch size for memory constraints"""
        if self._memory_history:
            avg_memory = sum(self._memory_history) / len(self._memory_history)
            if avg_memory > self.config.memory_limit_mb * 0.9:
                self._current_batch_size = max(self.config.min_batch_size, self._current_batch_size - 1)
            elif avg_memory < self.config.memory_limit_mb * 0.7:
                self._current_batch_size = min(self.config.max_batch_size, self._current_batch_size + 1)

    def _adapt_for_latency(self):
        """Adapt batch size for latency optimization"""
        avg_latency = sum(self._latency_history) / len(self._latency_history)
        if avg_latency > self.config.target_latency_ms * 1.1:
            self._current_batch_size = max(self.config.min_batch_size, self._current_batch_size - 1)
        elif avg_latency < self.config.target_latency_ms * 0.8:
            self._current_batch_size = min(self.config.max_batch_size, self._current_batch_size + 1)

    def _adapt_for_throughput(self):
        """Adapt batch size for throughput maximization"""
        if len(self._throughput_history) >= 2:
            recent_throughput = sum(list(self._throughput_history)[-5:]) / min(5, len(self._throughput_history))
            older_throughput = sum(list(self._throughput_history)[-10:-5]) / min(5, len(self._throughput_history) - 5)

            if recent_throughput > older_throughput:
                # Throughput improving, continue in same direction
                if self._current_batch_size < self.config.max_batch_size:
                    self._current_batch_size += 1
            else:
                # Throughput declining, reverse direction
                if self._current_batch_size > self.config.min_batch_size:
                    self._current_batch_size -= 1

    def _adapt_adaptive(self):
        """Adaptive batch size adjustment using weighted scoring"""
        if not (self._latency_history and self._memory_history and self._throughput_history):
            return

        # Calculate normalized scores (0-1, where 1 is better)
        avg_latency = sum(self._latency_history) / len(self._latency_history)
        latency_score = max(0, 1 - (avg_latency / self.config.target_latency_ms))

        avg_memory = sum(self._memory_history) / len(self._memory_history)
        memory_score = max(0, 1 - (avg_memory / self.config.memory_limit_mb))

        avg_throughput = sum(self._throughput_history) / len(self._throughput_history)
        max_theoretical_throughput = self.config.max_batch_size / (self.config.target_latency_ms / 1000.0)
        throughput_score = min(1, avg_throughput / max_theoretical_throughput)

        # Weighted composite score
        composite_score = (
            self.config.latency_weight * latency_score +
            self.config.memory_weight * memory_score +
            self.config.throughput_weight * throughput_score
        )

        # Adjust batch size based on composite score
        if composite_score > 0.8:
            # Performance is good, try to increase batch size
            self._current_batch_size = min(self.config.max_batch_size, self._current_batch_size + 1)
        elif composite_score < 0.6:
            # Performance is poor, decrease batch size
            self._current_batch_size = max(self.config.min_batch_size, self._current_batch_size - 1)

    def _adapt_for_power(self):
        """Adapt batch size for power efficiency"""
        # Power-efficient batching prefers larger batches to amortize overhead
        # but respects latency constraints
        avg_latency = sum(self._latency_history) / len(self._latency_history)
        if avg_latency < self.config.target_latency_ms * 0.7:
            # We have latency headroom, increase batch size for efficiency
            self._current_batch_size = min(self.config.max_batch_size, self._current_batch_size + 1)
        elif avg_latency > self.config.target_latency_ms:
            # Latency too high, reduce batch size
            self._current_batch_size = max(self.config.min_batch_size, self._current_batch_size - 1)

    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, '_stop_workers'):
            self._stop_workers.set()


class PowerManager:
    """Power management for low-power inference modes"""

    def __init__(self, config: PowerConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Power state
        self.current_mode = config.mode
        self.cpu_frequency_original = None
        self.gpu_power_limit_original = None

        # Monitoring
        self.power_history = deque(maxlen=100)
        self.thermal_history = deque(maxlen=50)

        # Initialize power management
        self._initialize_power_management()

        self.logger.info(f"✅ Power manager initialized")
        self.logger.info(f"  • Mode: {config.mode.value}")
        self.logger.info(f"  • CPU scaling: {config.cpu_frequency_scaling}")

    def set_power_mode(self, mode: PowerMode):
        """Set power mode"""
        if mode != self.current_mode:
            self.logger.info(f"Switching power mode: {self.current_mode.value} -> {mode.value}")
            self.current_mode = mode
            self._apply_power_mode()

    def get_power_recommendations(self) -> Dict[str, Any]:
        """Get power optimization recommendations"""
        recommendations = []

        # Check battery level
        if PSUTIL_AVAILABLE:
            try:
                battery = psutil.sensors_battery()
                if battery:
                    if battery.percent < self.config.battery_threshold_critical * 100:
                        recommendations.append({
                            "type": "critical_battery",
                            "message": "Critical battery level - switch to ultra low power mode",
                            "suggested_mode": PowerMode.ULTRA_LOW_POWER
                        })
                    elif battery.percent < self.config.battery_threshold_low * 100:
                        recommendations.append({
                            "type": "low_battery",
                            "message": "Low battery level - consider power saver mode",
                            "suggested_mode": PowerMode.POWER_SAVER
                        })
            except Exception:
                pass

        # Check thermal state
        if self.thermal_history:
            avg_temp = sum(self.thermal_history) / len(self.thermal_history)
            if avg_temp > self.config.thermal_threshold:
                recommendations.append({
                    "type": "thermal_throttling",
                    "message": f"High temperature ({avg_temp:.1f}°C) - reduce performance",
                    "suggested_mode": PowerMode.POWER_SAVER
                })

        return {
            "current_mode": self.current_mode.value,
            "recommendations": recommendations,
            "power_savings_enabled": len(recommendations) > 0
        }

    def update_power_metrics(self, power_watts: Optional[float] = None,
                           temperature_celsius: Optional[float] = None):
        """Update power and thermal metrics"""
        if power_watts is not None:
            self.power_history.append(power_watts)

        if temperature_celsius is not None:
            self.thermal_history.append(temperature_celsius)

        # Adaptive mode switching
        if self.config.mode == PowerMode.ADAPTIVE:
            self._adaptive_power_management()

    def get_power_scaling_factor(self) -> float:
        """Get current power scaling factor for model operations"""
        if self.current_mode == PowerMode.HIGH_PERFORMANCE:
            return 1.0
        elif self.current_mode == PowerMode.BALANCED:
            return 0.8
        elif self.current_mode == PowerMode.POWER_SAVER:
            return 0.6
        elif self.current_mode == PowerMode.ULTRA_LOW_POWER:
            return 0.4
        else:
            return 0.8  # Default for adaptive

    def _initialize_power_management(self):
        """Initialize power management features"""
        try:
            # Store original CPU frequency if scaling is enabled
            if self.config.cpu_frequency_scaling and PSUTIL_AVAILABLE:
                try:
                    cpu_freq = psutil.cpu_freq()
                    if cpu_freq:
                        self.cpu_frequency_original = cpu_freq.current
                except Exception as e:
                    self.logger.debug(f"Could not get CPU frequency: {e}")

            # Apply initial power mode
            self._apply_power_mode()

        except Exception as e:
            self.logger.warning(f"Power management initialization failed: {e}")

    def _apply_power_mode(self):
        """Apply current power mode settings"""
        try:
            if self.config.cpu_frequency_scaling:
                self._set_cpu_frequency()

            if self.config.gpu_power_limit is not None:
                self._set_gpu_power_limit()

        except Exception as e:
            self.logger.warning(f"Failed to apply power mode: {e}")

    def _set_cpu_frequency(self):
        """Set CPU frequency based on power mode"""
        if not PSUTIL_AVAILABLE or not self.cpu_frequency_original:
            return

        try:
            scaling_factor = self.get_power_scaling_factor()
            target_freq = self.cpu_frequency_original * scaling_factor

            # Note: Actual CPU frequency scaling would require system-level permissions
            # This is a placeholder for the interface
            self.logger.debug(f"CPU frequency scaling: {scaling_factor:.2f}x ({target_freq:.0f} MHz)")

        except Exception as e:
            self.logger.debug(f"CPU frequency scaling failed: {e}")

    def _set_gpu_power_limit(self):
        """Set GPU power limit based on power mode"""
        try:
            if self.config.gpu_power_limit:
                scaling_factor = self.get_power_scaling_factor()
                target_power = self.config.gpu_power_limit * scaling_factor

                # Note: Actual GPU power limiting would require nvidia-ml-py or similar
                # This is a placeholder for the interface
                self.logger.debug(f"GPU power limit: {target_power:.0f}W")

        except Exception as e:
            self.logger.debug(f"GPU power limiting failed: {e}")

    def _adaptive_power_management(self):
        """Adaptive power mode switching"""
        recommendations = self.get_power_recommendations()

        if recommendations["recommendations"]:
            # Take the most critical recommendation
            critical_rec = recommendations["recommendations"][0]
            suggested_mode = critical_rec.get("suggested_mode")

            if suggested_mode and suggested_mode != self.current_mode:
                self.set_power_mode(suggested_mode)


class OptimizedInferenceEngine:
    """Main inference engine with dynamic batching and caching"""

    def __init__(self, model: nn.Module,
                 batching_config: BatchingConfig,
                 cache_config: CacheConfig,
                 power_config: PowerConfig):
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.batcher = DynamicBatcher(batching_config)
        self.cache = IntelligentCache(cache_config)
        self.power_manager = PowerManager(power_config)

        # Processing thread
        self._processor_thread = None
        self._stop_processing = threading.Event()

        # Statistics
        self.total_requests = 0
        self.cache_hits = 0
        self.processing_times = deque(maxlen=1000)

        self._start_processor()

        self.logger.info("✅ Optimized inference engine initialized")

    async def predict_async(self, inputs: Dict[str, torch.Tensor],
                          request_id: Optional[str] = None) -> BatchResult:
        """Asynchronous prediction with batching and caching"""
        if request_id is None:
            request_id = self._generate_request_id(inputs)

        # Check cache first
        cached_result = self.cache.get(request_id)
        if cached_result is not None:
            self.cache_hits += 1
            return BatchResult(
                request_id=request_id,
                outputs=cached_result,
                processing_time_ms=0.0,
                cache_hit=True
            )

        # Create request
        request = BatchRequest(
            request_id=request_id,
            inputs=inputs,
            timestamp=time.time()
        )

        # Submit for batching
        if not self.batcher.submit_request(request):
            raise RuntimeError("Request queue full")

        # Wait for result (in real implementation, would use proper async/await)
        # This is a simplified synchronous version
        return self._process_single_request(request)

    def predict(self, inputs: Dict[str, torch.Tensor],
               request_id: Optional[str] = None) -> BatchResult:
        """Synchronous prediction"""
        # For simplicity, implement as synchronous version
        if request_id is None:
            request_id = self._generate_request_id(inputs)

        # Check cache
        cached_result = self.cache.get(request_id)
        if cached_result is not None:
            self.cache_hits += 1
            return BatchResult(
                request_id=request_id,
                outputs=cached_result,
                processing_time_ms=0.0,
                cache_hit=True
            )

        # Process directly
        start_time = time.time()

        # Apply power scaling
        power_factor = self.power_manager.get_power_scaling_factor()

        with torch.no_grad():
            # Simple single-item processing (in real implementation, would batch)
            if isinstance(inputs, dict) and len(inputs) == 1:
                input_tensor = next(iter(inputs.values()))
                outputs = self.model(input_tensor)
            else:
                outputs = self.model(**inputs)

        processing_time = (time.time() - start_time) * 1000

        # Cache result
        if isinstance(outputs, torch.Tensor):
            cache_outputs = {"output": outputs}
        else:
            cache_outputs = outputs

        self.cache.put(request_id, cache_outputs)

        # Update statistics
        self.total_requests += 1
        self.processing_times.append(processing_time)

        return BatchResult(
            request_id=request_id,
            outputs=cache_outputs,
            processing_time_ms=processing_time,
            cache_hit=False,
            batch_size=1
        )

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        cache_stats = self.cache.get_stats()
        batcher_stats = self.batcher.get_stats()
        power_recommendations = self.power_manager.get_power_recommendations()

        avg_processing_time = (sum(self.processing_times) / len(self.processing_times)
                             if self.processing_times else 0.0)

        return {
            "inference": {
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits,
                "cache_hit_rate": self.cache_hits / max(1, self.total_requests),
                "avg_processing_time_ms": avg_processing_time
            },
            "cache": cache_stats,
            "batching": batcher_stats,
            "power": power_recommendations
        }

    def _generate_request_id(self, inputs: Dict[str, torch.Tensor]) -> str:
        """Generate unique request ID based on inputs"""
        # Create hash of input tensors
        hasher = hashlib.md5()
        for key, tensor in inputs.items():
            hasher.update(key.encode())
            hasher.update(tensor.cpu().numpy().tobytes())
        return hasher.hexdigest()

    def _process_single_request(self, request: BatchRequest) -> BatchResult:
        """Process single request (simplified version)"""
        start_time = time.time()

        with torch.no_grad():
            if len(request.inputs) == 1:
                input_tensor = next(iter(request.inputs.values()))
                outputs = self.model(input_tensor)
            else:
                outputs = self.model(**request.inputs)

        processing_time = (time.time() - start_time) * 1000

        if isinstance(outputs, torch.Tensor):
            output_dict = {"output": outputs}
        else:
            output_dict = outputs

        # Cache result
        self.cache.put(request.request_id, output_dict)

        return BatchResult(
            request_id=request.request_id,
            outputs=output_dict,
            processing_time_ms=processing_time,
            cache_hit=False,
            batch_size=1
        )

    def _start_processor(self):
        """Start batch processor thread"""
        if self._processor_thread is None or not self._processor_thread.is_alive():
            self._processor_thread = threading.Thread(target=self._processor_loop, daemon=True)
            self._processor_thread.start()

    def _processor_loop(self):
        """Main processor loop for handling batches"""
        while not self._stop_processing.is_set():
            try:
                batch = self.batcher.get_batch(timeout=1.0)
                if batch:
                    self._process_batch(batch)
            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")

    def _process_batch(self, batch: List[BatchRequest]):
        """Process a batch of requests"""
        start_time = time.time()

        try:
            # Combine inputs into batch tensors
            batch_inputs = self._combine_batch_inputs(batch)

            # Run inference
            with torch.no_grad():
                batch_outputs = self.model(**batch_inputs)

            # Split outputs back to individual results
            individual_outputs = self._split_batch_outputs(batch_outputs, len(batch))

            processing_time = (time.time() - start_time) * 1000

            # Cache and return results
            for i, request in enumerate(batch):
                outputs = individual_outputs[i]
                self.cache.put(request.request_id, outputs)

                # In real implementation, would notify waiting coroutines
                if request.callback:
                    result = BatchResult(
                        request_id=request.request_id,
                        outputs=outputs,
                        processing_time_ms=processing_time / len(batch),
                        cache_hit=False,
                        batch_size=len(batch)
                    )
                    request.callback(result)

            # Update performance metrics
            memory_usage = self._estimate_memory_usage()
            self.batcher.update_performance_metrics(len(batch), processing_time, memory_usage)

        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")

    def _combine_batch_inputs(self, batch: List[BatchRequest]) -> Dict[str, torch.Tensor]:
        """Combine individual requests into batch tensors"""
        if not batch:
            return {}

        # Get keys from first request
        keys = list(batch[0].inputs.keys())
        batch_inputs = {}

        for key in keys:
            tensors = [request.inputs[key] for request in batch]
            # If tensors have batch dimension of 1, squeeze it before stacking
            if tensors[0].dim() > 1 and tensors[0].shape[0] == 1:
                tensors = [t.squeeze(0) for t in tensors]
            batch_inputs[key] = torch.stack(tensors, dim=0)

        return batch_inputs

    def _split_batch_outputs(self, batch_outputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
                           batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Split batch outputs back to individual results"""
        if isinstance(batch_outputs, torch.Tensor):
            # Single tensor output
            individual_outputs = []
            for i in range(batch_size):
                individual_outputs.append({"output": batch_outputs[i]})
            return individual_outputs

        elif isinstance(batch_outputs, dict):
            # Dictionary of tensors
            individual_outputs = []
            for i in range(batch_size):
                outputs = {}
                for key, tensor in batch_outputs.items():
                    outputs[key] = tensor[i]
                individual_outputs.append(outputs)
            return individual_outputs

        else:
            # Fallback
            return [{"output": batch_outputs}] * batch_size

    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB"""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                return process.memory_info().rss / (1024 * 1024)
            except Exception:
                pass

        # Fallback estimation
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            return 100.0  # Default estimate

    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, '_stop_processing'):
            self._stop_processing.set()


def create_optimized_inference_example():
    """Create example of optimized inference system"""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 5)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()
    model.eval()

    # Configure components
    batching_config = BatchingConfig(
        strategy=BatchingStrategy.ADAPTIVE,
        min_batch_size=1,
        max_batch_size=8,
        target_latency_ms=50.0,
        memory_limit_mb=256.0
    )

    cache_config = CacheConfig(
        eviction_policy=CacheEvictionPolicy.ADAPTIVE,
        max_cache_size_mb=128.0,
        max_entries=1000,
        enable_compression=True
    )

    power_config = PowerConfig(
        mode=PowerMode.ADAPTIVE,
        cpu_frequency_scaling=True,
        enable_model_pruning=True
    )

    # Create optimized inference engine
    engine = OptimizedInferenceEngine(model, batching_config, cache_config, power_config)

    # Test inference
    print("Testing optimized inference engine...")

    # Generate test inputs
    test_inputs = [
        {"x": torch.randn(1, 10)} for _ in range(20)
    ]

    # Run predictions
    results = []
    for i, inputs in enumerate(test_inputs):
        result = engine.predict(inputs, request_id=f"test_{i}")
        results.append(result)
        print(f"Request {i}: {result.processing_time_ms:.2f}ms, cache_hit={result.cache_hit}")

    # Test cache hits with repeated inputs
    print("\nTesting cache hits...")
    for i in range(5):
        result = engine.predict(test_inputs[0], request_id="test_0")  # Same as first request
        print(f"Repeat {i}: {result.processing_time_ms:.2f}ms, cache_hit={result.cache_hit}")

    # Print comprehensive statistics
    stats = engine.get_comprehensive_stats()
    print(f"\n{'='*50}")
    print("PERFORMANCE STATISTICS")
    print(f"{'='*50}")
    print(f"Total requests: {stats['inference']['total_requests']}")
    print(f"Cache hit rate: {stats['inference']['cache_hit_rate']:.2%}")
    print(f"Avg processing time: {stats['inference']['avg_processing_time_ms']:.2f}ms")
    print(f"Cache entries: {stats['cache']['entries']}")
    print(f"Cache size: {stats['cache']['size_mb']:.2f}MB")
    print(f"Avg batch size: {stats['batching']['avg_batch_size']:.2f}")
    print(f"Current power mode: {stats['power']['current_mode']}")

    return engine, stats


if __name__ == "__main__":
    engine, stats = create_optimized_inference_example()
    print("\n✅ Dynamic batching and caching system demonstration completed!")