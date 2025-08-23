#!/usr/bin/env python3
"""
Memory and Compute Optimizations
================================

Advanced memory and compute optimization system including:
- Gradient checkpointing optimization with selective activation saving
- Activation checkpointing for memory efficiency
- Compute graph optimizations and fusion
- Memory-efficient attention mechanisms
- Dynamic memory management and garbage collection
- Compute kernel optimization and scheduling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import gc
import time
import logging
import threading
import weakref
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import math
from collections import defaultdict, deque
import contextlib

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


class CheckpointingStrategy(Enum):
    """Gradient checkpointing strategies"""
    NONE = "none"
    UNIFORM = "uniform"  # Checkpoint every N layers
    ADAPTIVE = "adaptive"  # Based on memory usage
    SELECTIVE = "selective"  # Based on computation cost
    MEMORY_AWARE = "memory_aware"  # Based on available memory
    HYBRID = "hybrid"  # Combination of strategies


class MemoryOptimizationLevel(Enum):
    """Memory optimization levels"""
    MINIMAL = "minimal"  # Basic optimizations
    MODERATE = "moderate"  # Balanced memory/speed
    AGGRESSIVE = "aggressive"  # Maximum memory savings
    EXTREME = "extreme"  # Extreme memory savings, slower


class ComputeOptimizationMode(Enum):
    """Compute optimization modes"""
    SPEED = "speed"  # Optimize for speed
    MEMORY = "memory"  # Optimize for memory
    BALANCED = "balanced"  # Balance speed and memory
    ADAPTIVE = "adaptive"  # Adapt based on conditions


@dataclass
class OptimizationConfig:
    """Configuration for memory and compute optimizations"""
    # Gradient checkpointing
    checkpointing_strategy: CheckpointingStrategy = CheckpointingStrategy.ADAPTIVE
    checkpoint_ratio: float = 0.5  # Fraction of layers to checkpoint
    min_checkpoint_layers: int = 2
    max_checkpoint_layers: int = 8

    # Memory optimization
    memory_optimization_level: MemoryOptimizationLevel = MemoryOptimizationLevel.MODERATE
    target_memory_usage_gb: float = 4.0
    memory_pressure_threshold: float = 0.8
    enable_activation_checkpointing: bool = True

    # Compute optimization
    compute_optimization_mode: ComputeOptimizationMode = ComputeOptimizationMode.BALANCED
    enable_kernel_fusion: bool = True
    enable_mixed_precision: bool = True
    enable_graph_optimization: bool = True

    # Dynamic management
    enable_dynamic_optimization: bool = True
    optimization_interval_steps: int = 100
    memory_monitoring_enabled: bool = True

    # Advanced features
    enable_offloading: bool = False
    offload_device: str = "cpu"
    enable_compression: bool = False
    compression_ratio: float = 0.5


@dataclass
class MemoryStats:
    """Memory usage statistics"""
    allocated_mb: float = 0.0
    cached_mb: float = 0.0
    reserved_mb: float = 0.0
    peak_mb: float = 0.0
    system_available_mb: float = 0.0
    utilization_ratio: float = 0.0
    timestamp: float = 0.0


@dataclass
class ComputeStats:
    """Compute performance statistics"""
    forward_time_ms: float = 0.0
    backward_time_ms: float = 0.0
    total_time_ms: float = 0.0
    flops_per_second: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    kernel_efficiency: float = 0.0
    timestamp: float = 0.0


class MemoryMonitor:
    """Real-time memory monitoring and management"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Memory tracking
        self.memory_history = deque(maxlen=1000)
        self.peak_memory = 0.0
        self.baseline_memory = 0.0

        # Monitoring thread
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()

        if config.memory_monitoring_enabled:
            self._start_monitoring()

        self.logger.info("✅ Memory monitor initialized")

    def get_current_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        stats = MemoryStats(timestamp=time.time())

        if torch.cuda.is_available():
            # GPU memory
            stats.allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            stats.cached_mb = torch.cuda.memory_cached() / (1024 ** 2)
            stats.reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
            stats.peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # System memory
        if PSUTIL_AVAILABLE:
            memory_info = psutil.virtual_memory()
            stats.system_available_mb = memory_info.available / (1024 ** 2)
            stats.utilization_ratio = memory_info.percent / 100.0

        # Update peak tracking
        current_total = stats.allocated_mb + stats.cached_mb
        if current_total > self.peak_memory:
            self.peak_memory = current_total

        return stats

    def is_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        stats = self.get_current_memory_stats()

        # Check GPU memory pressure
        if torch.cuda.is_available():
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            gpu_utilization = (stats.allocated_mb + stats.cached_mb) / total_gpu_memory
            if gpu_utilization > self.config.memory_pressure_threshold:
                return True

        # Check system memory pressure
        if stats.utilization_ratio > self.config.memory_pressure_threshold:
            return True

        return False

    def cleanup_memory(self, aggressive: bool = False):
        """Perform memory cleanup"""
        if torch.cuda.is_available():
            if aggressive:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            else:
                torch.cuda.empty_cache()

        # Python garbage collection
        if aggressive:
            for _ in range(3):
                gc.collect()
        else:
            gc.collect()

        self.logger.debug("Memory cleanup performed")

    def get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations"""
        recommendations = []
        stats = self.get_current_memory_stats()

        if self.is_memory_pressure():
            recommendations.append("System under memory pressure - consider reducing batch size")
            recommendations.append("Enable more aggressive checkpointing")
            recommendations.append("Consider enabling activation offloading")

        if torch.cuda.is_available():
            total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            if stats.cached_mb > stats.allocated_mb * 2:
                recommendations.append("High cached memory - consider calling torch.cuda.empty_cache()")

            if stats.allocated_mb > total_gpu_memory * 0.9:
                recommendations.append("GPU memory nearly full - enable gradient checkpointing")

        return recommendations

    def _start_monitoring(self):
        """Start background memory monitoring"""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitoring_thread.start()

    def _monitoring_loop(self):
        """Background monitoring loop"""
        while not self._stop_monitoring.wait(5.0):  # Check every 5 seconds
            try:
                stats = self.get_current_memory_stats()
                self.memory_history.append(stats)

                # Automatic cleanup if needed
                if self.is_memory_pressure():
                    self.cleanup_memory(aggressive=False)

            except Exception as e:
                self.logger.error(f"Memory monitoring error: {e}")

    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, '_stop_monitoring'):
            self._stop_monitoring.set()


class GradientCheckpointer:
    """Advanced gradient checkpointing with adaptive strategies"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Checkpointing state
        self.checkpoint_layers = set()
        self.layer_costs = {}
        self.memory_savings = {}

        # Adaptive parameters
        self.current_checkpoint_ratio = config.checkpoint_ratio
        self.performance_history = deque(maxlen=100)

        self.logger.info("✅ Gradient checkpointer initialized")
        self.logger.info(f"  • Strategy: {config.checkpointing_strategy.value}")
        self.logger.info(f"  • Checkpoint ratio: {config.checkpoint_ratio}")

    def should_checkpoint_layer(self, layer_idx: int, total_layers: int,
                               memory_stats: Optional[MemoryStats] = None) -> bool:
        """Determine if a layer should be checkpointed"""

        if self.config.checkpointing_strategy == CheckpointingStrategy.NONE:
            return False

        elif self.config.checkpointing_strategy == CheckpointingStrategy.UNIFORM:
            # Checkpoint every N layers
            checkpoint_interval = max(1, int(1.0 / self.current_checkpoint_ratio))
            return layer_idx % checkpoint_interval == 0

        elif self.config.checkpointing_strategy == CheckpointingStrategy.ADAPTIVE:
            return self._adaptive_checkpoint_decision(layer_idx, total_layers, memory_stats)

        elif self.config.checkpointing_strategy == CheckpointingStrategy.SELECTIVE:
            return self._selective_checkpoint_decision(layer_idx)

        elif self.config.checkpointing_strategy == CheckpointingStrategy.MEMORY_AWARE:
            return self._memory_aware_checkpoint_decision(layer_idx, memory_stats)

        elif self.config.checkpointing_strategy == CheckpointingStrategy.HYBRID:
            return self._hybrid_checkpoint_decision(layer_idx, total_layers, memory_stats)

        return False

    def checkpoint_function(self, function: Callable, *args, **kwargs):
        """Apply checkpointing to a function"""
        if self.config.checkpointing_strategy == CheckpointingStrategy.NONE:
            return function(*args, **kwargs)
        else:
            return checkpoint(function, *args, **kwargs)

    def update_layer_cost(self, layer_idx: int, compute_time: float, memory_usage: float):
        """Update layer computational cost information"""
        self.layer_costs[layer_idx] = {
            'compute_time': compute_time,
            'memory_usage': memory_usage,
            'cost_ratio': compute_time / max(memory_usage, 1e-6)
        }

    def adapt_checkpointing_strategy(self, performance_metrics: Dict[str, float]):
        """Adapt checkpointing strategy based on performance"""
        self.performance_history.append(performance_metrics)

        if len(self.performance_history) < 10:
            return  # Need more data

        # Analyze recent performance
        recent_metrics = list(self.performance_history)[-10:]
        avg_memory_usage = sum(m.get('memory_usage', 0) for m in recent_metrics) / len(recent_metrics)
        avg_compute_time = sum(m.get('compute_time', 0) for m in recent_metrics) / len(recent_metrics)

        # Adjust checkpoint ratio based on memory pressure
        target_memory = self.config.target_memory_usage_gb * 1024  # Convert to MB

        if avg_memory_usage > target_memory * 1.2:
            # Increase checkpointing
            self.current_checkpoint_ratio = min(0.8, self.current_checkpoint_ratio + 0.1)
        elif avg_memory_usage < target_memory * 0.8:
            # Decrease checkpointing for better speed
            self.current_checkpoint_ratio = max(0.1, self.current_checkpoint_ratio - 0.1)

        self.logger.debug(f"Adapted checkpoint ratio to {self.current_checkpoint_ratio:.2f}")

    def _adaptive_checkpoint_decision(self, layer_idx: int, total_layers: int,
                                    memory_stats: Optional[MemoryStats]) -> bool:
        """Adaptive checkpointing decision"""
        # Base decision on uniform distribution
        uniform_decision = layer_idx % max(1, int(1.0 / self.current_checkpoint_ratio)) == 0

        # Adjust based on memory pressure
        if memory_stats and memory_stats.utilization_ratio > 0.8:
            # High memory pressure - checkpoint more aggressively
            return uniform_decision or (layer_idx % 2 == 0)
        elif memory_stats and memory_stats.utilization_ratio < 0.5:
            # Low memory pressure - checkpoint less
            return uniform_decision and (layer_idx % 3 == 0)

        return uniform_decision

    def _selective_checkpoint_decision(self, layer_idx: int) -> bool:
        """Selective checkpointing based on layer cost"""
        if layer_idx not in self.layer_costs:
            # Default to checkpointing if no cost info
            return layer_idx % 3 == 0

        cost_info = self.layer_costs[layer_idx]
        # Checkpoint layers with high memory usage but low compute cost
        return cost_info['cost_ratio'] < 0.5

    def _memory_aware_checkpoint_decision(self, layer_idx: int,
                                        memory_stats: Optional[MemoryStats]) -> bool:
        """Memory-aware checkpointing decision"""
        if not memory_stats:
            return layer_idx % 3 == 0

        # More aggressive checkpointing under memory pressure
        if memory_stats.utilization_ratio > 0.9:
            return True  # Checkpoint everything
        elif memory_stats.utilization_ratio > 0.7:
            return layer_idx % 2 == 0  # Checkpoint every other layer
        elif memory_stats.utilization_ratio > 0.5:
            return layer_idx % 3 == 0  # Checkpoint every third layer
        else:
            return layer_idx % 4 == 0  # Minimal checkpointing

    def _hybrid_checkpoint_decision(self, layer_idx: int, total_layers: int,
                                  memory_stats: Optional[MemoryStats]) -> bool:
        """Hybrid checkpointing combining multiple strategies"""
        # Combine adaptive and selective strategies
        adaptive_decision = self._adaptive_checkpoint_decision(layer_idx, total_layers, memory_stats)
        selective_decision = self._selective_checkpoint_decision(layer_idx)

        # Use OR logic - checkpoint if either strategy suggests it
        return adaptive_decision or selective_decision


class ActivationCheckpointer:
    """Activation checkpointing for memory efficiency"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Activation management
        self.saved_activations = {}
        self.activation_sizes = {}
        self.offloaded_activations = {}

        # Memory management
        self.max_activation_memory_mb = config.target_memory_usage_gb * 1024 * 0.3  # 30% for activations
        self.current_activation_memory_mb = 0.0

        self.logger.info("✅ Activation checkpointer initialized")
        self.logger.info(f"  • Max activation memory: {self.max_activation_memory_mb:.1f}MB")

    def save_activation(self, key: str, activation: torch.Tensor,
                       priority: int = 0) -> bool:
        """Save activation with optional offloading"""
        activation_size_mb = activation.numel() * activation.element_size() / (1024 ** 2)

        # Check if we have space
        if self.current_activation_memory_mb + activation_size_mb > self.max_activation_memory_mb:
            if not self._make_space(activation_size_mb):
                return False  # Could not make space

        # Save activation
        if self.config.enable_offloading and activation_size_mb > 10.0:  # Offload large activations
            self._offload_activation(key, activation)
        else:
            self.saved_activations[key] = activation.detach()
            self.current_activation_memory_mb += activation_size_mb

        self.activation_sizes[key] = activation_size_mb
        return True

    def get_activation(self, key: str) -> Optional[torch.Tensor]:
        """Retrieve saved activation"""
        if key in self.saved_activations:
            return self.saved_activations[key]
        elif key in self.offloaded_activations:
            return self._load_offloaded_activation(key)
        else:
            return None

    def remove_activation(self, key: str):
        """Remove saved activation"""
        if key in self.saved_activations:
            activation_size = self.activation_sizes.get(key, 0)
            del self.saved_activations[key]
            self.current_activation_memory_mb -= activation_size

        if key in self.offloaded_activations:
            del self.offloaded_activations[key]

        if key in self.activation_sizes:
            del self.activation_sizes[key]

    def clear_all_activations(self):
        """Clear all saved activations"""
        self.saved_activations.clear()
        self.offloaded_activations.clear()
        self.activation_sizes.clear()
        self.current_activation_memory_mb = 0.0

    def get_memory_usage(self) -> Dict[str, float]:
        """Get activation memory usage statistics"""
        return {
            'current_mb': self.current_activation_memory_mb,
            'max_mb': self.max_activation_memory_mb,
            'utilization': self.current_activation_memory_mb / self.max_activation_memory_mb,
            'num_activations': len(self.saved_activations),
            'num_offloaded': len(self.offloaded_activations)
        }

    def _make_space(self, required_mb: float) -> bool:
        """Make space for new activation"""
        # Sort activations by size (largest first) for eviction
        sorted_activations = sorted(
            self.activation_sizes.items(),
            key=lambda x: x[1],
            reverse=True
        )

        freed_mb = 0.0
        for key, size_mb in sorted_activations:
            if freed_mb >= required_mb:
                break

            if key in self.saved_activations:
                # Try to offload first
                if self.config.enable_offloading:
                    activation = self.saved_activations[key]
                    self._offload_activation(key, activation)
                    del self.saved_activations[key]
                    self.current_activation_memory_mb -= size_mb
                    freed_mb += size_mb
                else:
                    # Remove activation
                    self.remove_activation(key)
                    freed_mb += size_mb

        return freed_mb >= required_mb

    def _offload_activation(self, key: str, activation: torch.Tensor):
        """Offload activation to CPU or disk"""
        if self.config.offload_device == "cpu":
            # Move to CPU
            cpu_activation = activation.cpu()
            if self.config.enable_compression:
                # Simple compression by converting to half precision
                cpu_activation = cpu_activation.half()
            self.offloaded_activations[key] = cpu_activation
        else:
            # Could implement disk offloading here
            self.offloaded_activations[key] = activation.cpu()

    def _load_offloaded_activation(self, key: str) -> Optional[torch.Tensor]:
        """Load offloaded activation back to GPU"""
        if key not in self.offloaded_activations:
            return None

        activation = self.offloaded_activations[key]

        # Move back to original device
        if torch.cuda.is_available():
            activation = activation.cuda()

        # Restore precision if compressed
        if self.config.enable_compression and activation.dtype == torch.float16:
            activation = activation.float()

        return activation


class ComputeOptimizer:
    """Compute graph and kernel optimizations"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Optimization state
        self.fused_operations = {}
        self.kernel_cache = {}
        self.compute_stats = deque(maxlen=1000)

        # Graph optimization
        self.optimized_modules = weakref.WeakSet()

        self.logger.info("✅ Compute optimizer initialized")
        self.logger.info(f"  • Mode: {config.compute_optimization_mode.value}")
        self.logger.info(f"  • Kernel fusion: {config.enable_kernel_fusion}")
        self.logger.info(f"  • Mixed precision: {config.enable_mixed_precision}")

    def optimize_module(self, module: nn.Module) -> nn.Module:
        """Apply compute optimizations to a module"""
        if module in self.optimized_modules:
            return module  # Already optimized

        # Apply optimizations based on configuration
        if self.config.enable_graph_optimization:
            module = self._optimize_graph(module)

        if self.config.enable_kernel_fusion:
            module = self._apply_kernel_fusion(module)

        if self.config.enable_mixed_precision:
            module = self._apply_mixed_precision(module)

        self.optimized_modules.add(module)
        return module

    def create_fused_attention(self, embed_dim: int, num_heads: int) -> nn.Module:
        """Create memory-efficient fused attention"""
        return FusedMultiHeadAttention(embed_dim, num_heads, self.config)

    def create_fused_feedforward(self, input_dim: int, hidden_dim: int) -> nn.Module:
        """Create fused feedforward network"""
        return FusedFeedForward(input_dim, hidden_dim, self.config)

    def record_compute_stats(self, forward_time: float, backward_time: float,
                           flops: float, memory_bandwidth: float):
        """Record compute performance statistics"""
        stats = ComputeStats(
            forward_time_ms=forward_time * 1000,
            backward_time_ms=backward_time * 1000,
            total_time_ms=(forward_time + backward_time) * 1000,
            flops_per_second=flops / max(forward_time + backward_time, 1e-6),
            memory_bandwidth_gbps=memory_bandwidth,
            timestamp=time.time()
        )

        self.compute_stats.append(stats)

    def get_optimization_recommendations(self) -> List[str]:
        """Get compute optimization recommendations"""
        recommendations = []

        if len(self.compute_stats) < 10:
            return recommendations

        # Analyze recent performance
        recent_stats = list(self.compute_stats)[-10:]
        avg_forward_time = sum(s.forward_time_ms for s in recent_stats) / len(recent_stats)
        avg_backward_time = sum(s.backward_time_ms for s in recent_stats) / len(recent_stats)

        if avg_backward_time > avg_forward_time * 2:
            recommendations.append("Backward pass is slow - consider gradient checkpointing")

        if avg_forward_time > 100:  # > 100ms
            recommendations.append("Forward pass is slow - consider kernel fusion optimizations")

        avg_flops = sum(s.flops_per_second for s in recent_stats) / len(recent_stats)
        if avg_flops < 1e12:  # < 1 TFLOPS
            recommendations.append("Low compute utilization - consider mixed precision training")

        return recommendations

    def _optimize_graph(self, module: nn.Module) -> nn.Module:
        """Apply graph-level optimizations"""
        # In a real implementation, this would use torch.jit.script or similar
        # For now, we'll apply some basic optimizations

        # Fuse consecutive operations where possible
        optimized_module = self._fuse_consecutive_ops(module)

        return optimized_module

    def _apply_kernel_fusion(self, module: nn.Module) -> nn.Module:
        """Apply kernel fusion optimizations"""
        # Replace standard operations with fused versions
        for name, child in module.named_children():
            if isinstance(child, nn.MultiheadAttention):
                # Replace with fused attention
                fused_attention = self.create_fused_attention(
                    child.embed_dim, child.num_heads
                )
                setattr(module, name, fused_attention)
            elif isinstance(child, nn.Sequential):
                # Check for fusable patterns in sequential modules
                fused_seq = self._fuse_sequential_ops(child)
                setattr(module, name, fused_seq)

        return module

    def _apply_mixed_precision(self, module: nn.Module) -> nn.Module:
        """Apply mixed precision optimizations"""
        # Convert appropriate layers to half precision
        for name, child in module.named_children():
            if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # Keep in float32 for stability, but could use autocast
                pass
            elif isinstance(child, (nn.LayerNorm, nn.BatchNorm1d)):
                # Keep normalization layers in float32
                pass

        return module

    def _fuse_consecutive_ops(self, module: nn.Module) -> nn.Module:
        """Fuse consecutive operations"""
        # This is a simplified version - real implementation would be more complex
        return module

    def _fuse_sequential_ops(self, sequential: nn.Sequential) -> nn.Module:
        """Fuse operations in a sequential module"""
        # Look for patterns like Linear + Activation
        layers = list(sequential.children())
        fused_layers = []

        i = 0
        while i < len(layers):
            current_layer = layers[i]

            # Check for Linear + ReLU pattern
            if (isinstance(current_layer, nn.Linear) and
                i + 1 < len(layers) and
                isinstance(layers[i + 1], nn.ReLU)):

                # Create fused Linear + ReLU
                fused_layer = FusedLinearReLU(current_layer.in_features,
                                            current_layer.out_features)
                fused_layers.append(fused_layer)
                i += 2  # Skip both layers
            else:
                fused_layers.append(current_layer)
                i += 1

        return nn.Sequential(*fused_layers)


class FusedMultiHeadAttention(nn.Module):
    """Memory-efficient fused multi-head attention"""

    def __init__(self, embed_dim: int, num_heads: int, config: OptimizationConfig):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Scaling factor
        self.scale = self.head_dim ** -0.5

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # For self-attention, query, key, value are the same
        x = query  # Assume self-attention for simplicity
        batch_size, seq_len, embed_dim = x.shape

        # Fused QKV computation
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Memory-efficient attention computation
        if self.config.memory_optimization_level == MemoryOptimizationLevel.AGGRESSIVE:
            # Use memory-efficient attention
            attn_output = self._memory_efficient_attention(q, k, v, attn_mask)
        else:
            # Standard attention
            attn_output = self._standard_attention(q, k, v, attn_mask)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, embed_dim)
        output = self.out_proj(attn_output)

        return output, None  # Return (output, attention_weights) like PyTorch's MultiheadAttention

    def _standard_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard attention computation"""
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn_scores += attn_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)

        return attn_output

    def _memory_efficient_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                  attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Memory-efficient attention using chunking"""
        batch_size, num_heads, seq_len, head_dim = q.shape
        chunk_size = min(512, seq_len)  # Process in chunks

        output = torch.zeros_like(q)

        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = q[:, :, i:end_i, :]

            # Compute attention for this chunk
            attn_scores = torch.matmul(q_chunk, k.transpose(-2, -1)) * self.scale

            if attn_mask is not None:
                attn_scores += attn_mask[:, :, i:end_i, :]

            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_probs, v)

            output[:, :, i:end_i, :] = attn_output

        return output


class FusedFeedForward(nn.Module):
    """Fused feedforward network with optimizations"""

    def __init__(self, input_dim: int, hidden_dim: int, config: OptimizationConfig):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.config = config

        # Fused linear layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.GELU()

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused forward pass
        hidden = self.fc1(x)
        hidden = self.activation(hidden)

        if self.config.memory_optimization_level == MemoryOptimizationLevel.AGGRESSIVE:
            # Apply checkpointing to second linear layer
            output = checkpoint(self.fc2, hidden)
        else:
            output = self.fc2(hidden)

        output = self.dropout(output)
        return output


class FusedLinearReLU(nn.Module):
    """Fused Linear + ReLU operation"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused linear + ReLU computation
        return F.relu(self.linear(x), inplace=True)


class MemoryComputeOptimizer:
    """Main memory and compute optimization system"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.memory_monitor = MemoryMonitor(config)
        self.gradient_checkpointer = GradientCheckpointer(config)
        self.activation_checkpointer = ActivationCheckpointer(config)
        self.compute_optimizer = ComputeOptimizer(config)

        # Optimization state
        self.optimization_step = 0
        self.last_optimization_time = time.time()

        self.logger.info("✅ Memory and compute optimizer initialized")
        self.logger.info(f"  • Memory optimization: {config.memory_optimization_level.value}")
        self.logger.info(f"  • Compute optimization: {config.compute_optimization_mode.value}")
        self.logger.info(f"  • Dynamic optimization: {config.enable_dynamic_optimization}")

    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply comprehensive optimizations to model"""
        self.logger.info("🔧 Applying model optimizations...")

        # Apply compute optimizations
        optimized_model = self.compute_optimizer.optimize_module(model)

        # Wrap with checkpointing if enabled
        if self.config.checkpointing_strategy != CheckpointingStrategy.NONE:
            optimized_model = self._wrap_with_checkpointing(optimized_model)

        self.logger.info("✅ Model optimization completed")
        return optimized_model

    def optimize_training_step(self, model: nn.Module, loss_fn: Callable,
                             inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Optimize a single training step"""
        self.optimization_step += 1

        # Pre-step optimizations
        if self.config.enable_dynamic_optimization:
            self._dynamic_optimization_check()

        # Memory cleanup if needed
        if self.memory_monitor.is_memory_pressure():
            self.memory_monitor.cleanup_memory(aggressive=True)

        # Execute training step with optimizations
        start_time = time.time()

        # Forward pass with activation checkpointing
        with self._activation_checkpointing_context():
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

        forward_time = time.time() - start_time

        # Backward pass with gradient checkpointing
        backward_start = time.time()
        loss.backward()
        backward_time = time.time() - backward_start

        # Record performance metrics
        self._record_step_metrics(forward_time, backward_time, loss.item())

        return loss

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        memory_stats = self.memory_monitor.get_current_memory_stats()
        activation_stats = self.activation_checkpointer.get_memory_usage()
        memory_recommendations = self.memory_monitor.get_memory_recommendations()
        compute_recommendations = self.compute_optimizer.get_optimization_recommendations()

        return {
            "memory": {
                "current_stats": {
                    "allocated_mb": memory_stats.allocated_mb,
                    "cached_mb": memory_stats.cached_mb,
                    "peak_mb": memory_stats.peak_mb,
                    "utilization": memory_stats.utilization_ratio
                },
                "activation_stats": activation_stats,
                "recommendations": memory_recommendations
            },
            "compute": {
                "optimization_mode": self.config.compute_optimization_mode.value,
                "kernel_fusion_enabled": self.config.enable_kernel_fusion,
                "mixed_precision_enabled": self.config.enable_mixed_precision,
                "recommendations": compute_recommendations
            },
            "checkpointing": {
                "strategy": self.config.checkpointing_strategy.value,
                "current_ratio": self.gradient_checkpointer.current_checkpoint_ratio,
                "activation_checkpointing": self.config.enable_activation_checkpointing
            },
            "optimization_step": self.optimization_step
        }

    @contextlib.contextmanager
    def _activation_checkpointing_context(self):
        """Context manager for activation checkpointing"""
        if not self.config.enable_activation_checkpointing:
            yield
            return

        # Clear old activations
        self.activation_checkpointer.clear_all_activations()

        try:
            yield
        finally:
            # Cleanup after forward pass
            if self.memory_monitor.is_memory_pressure():
                self.activation_checkpointer.clear_all_activations()

    def _wrap_with_checkpointing(self, model: nn.Module) -> nn.Module:
        """Wrap model layers with gradient checkpointing"""
        # This is a simplified version - real implementation would be more sophisticated

        class CheckpointedModel(nn.Module):
            def __init__(self, original_model, checkpointer):
                super().__init__()
                self.model = original_model
                self.checkpointer = checkpointer

            def forward(self, x):
                # Apply checkpointing to appropriate layers
                return self.model(x)

        return CheckpointedModel(model, self.gradient_checkpointer)

    def _dynamic_optimization_check(self):
        """Perform dynamic optimization adjustments"""
        current_time = time.time()

        # Check if it's time for optimization adjustment
        if (current_time - self.last_optimization_time >
            self.config.optimization_interval_steps * 0.1):  # Approximate step time

            # Get current performance metrics
            memory_stats = self.memory_monitor.get_current_memory_stats()

            # Adapt checkpointing strategy
            performance_metrics = {
                'memory_usage': memory_stats.allocated_mb,
                'compute_time': 0.1,  # Placeholder
                'step': self.optimization_step
            }

            self.gradient_checkpointer.adapt_checkpointing_strategy(performance_metrics)
            self.last_optimization_time = current_time

    def _record_step_metrics(self, forward_time: float, backward_time: float, loss_value: float):
        """Record metrics for this training step"""
        # Record compute stats
        estimated_flops = 1e12  # Placeholder
        estimated_bandwidth = 100.0  # Placeholder

        self.compute_optimizer.record_compute_stats(
            forward_time, backward_time, estimated_flops, estimated_bandwidth
        )

        # Update memory statistics
        memory_stats = self.memory_monitor.get_current_memory_stats()
        self.memory_monitor.memory_history.append(memory_stats)


def create_optimization_example():
    """Create example of memory and compute optimization system"""

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create optimization configuration
    config = OptimizationConfig(
        checkpointing_strategy=CheckpointingStrategy.ADAPTIVE,
        memory_optimization_level=MemoryOptimizationLevel.MODERATE,
        compute_optimization_mode=ComputeOptimizationMode.BALANCED,
        enable_activation_checkpointing=True,
        enable_kernel_fusion=True,
        enable_mixed_precision=True,
        target_memory_usage_gb=2.0
    )

    # Create optimizer
    optimizer = MemoryComputeOptimizer(config)

    # Create test model
    class TestTransformerBlock(nn.Module):
        def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
            super().__init__()
            self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(embed_dim)
            self.ff = nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.ReLU(),
                nn.Linear(ff_dim, embed_dim)
            )
            self.norm2 = nn.LayerNorm(embed_dim)

        def forward(self, x):
            # Self-attention
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)

            # Feedforward
            ff_out = self.ff(x)
            x = self.norm2(x + ff_out)

            return x

    model = TestTransformerBlock(embed_dim=512, num_heads=8, ff_dim=2048)

    # Optimize model
    optimized_model = optimizer.optimize_model(model)

    # Test optimization with dummy training step
    print("Testing memory and compute optimizations...")

    # Create dummy data
    batch_size, seq_len, embed_dim = 4, 128, 512
    inputs = torch.randn(batch_size, seq_len, embed_dim)
    targets = torch.randn(batch_size, seq_len, embed_dim)

    # Define loss function
    loss_fn = nn.MSELoss()

    # Run optimized training steps
    for step in range(5):
        loss = optimizer.optimize_training_step(optimized_model, loss_fn, inputs, targets)
        print(f"Step {step + 1}: Loss = {loss.item():.4f}")

    # Get optimization report
    report = optimizer.get_optimization_report()

    print(f"\n{'='*60}")
    print("OPTIMIZATION REPORT")
    print(f"{'='*60}")
    print(f"Memory allocated: {report['memory']['current_stats']['allocated_mb']:.2f}MB")
    print(f"Memory utilization: {report['memory']['current_stats']['utilization']:.2%}")
    print(f"Checkpointing strategy: {report['checkpointing']['strategy']}")
    print(f"Checkpoint ratio: {report['checkpointing']['current_ratio']:.2f}")
    print(f"Activation checkpointing: {report['checkpointing']['activation_checkpointing']}")
    print(f"Kernel fusion: {report['compute']['kernel_fusion_enabled']}")
    print(f"Mixed precision: {report['compute']['mixed_precision_enabled']}")

    if report['memory']['recommendations']:
        print(f"\nMemory recommendations:")
        for rec in report['memory']['recommendations']:
            print(f"  • {rec}")

    if report['compute']['recommendations']:
        print(f"\nCompute recommendations:")
        for rec in report['compute']['recommendations']:
            print(f"  • {rec}")

    return optimizer, report


if __name__ == "__main__":
    optimizer, report = create_optimization_example()
    print("\n✅ Memory and compute optimization system demonstration completed!")