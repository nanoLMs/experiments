#!/usr/bin/env python3
"""
Comprehensive Monitoring and Diagnostics System
==============================================

Real-time performance monitoring, gradient analysis, and training diagnostics
for the advanced NanoLM system with detailed reporting and recommendations.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
from datetime import datetime, timedelta
import warnings

# Optional imports with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - system monitoring will be limited")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    logging.warning("GPUtil not available - GPU monitoring will be limited")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    # Suppress matplotlib warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib not available - visualization will be disabled")


class MetricType(Enum):
    """Types of metrics to monitor"""
    PERFORMANCE = "performance"
    GRADIENT = "gradient"
    LOSS = "loss"
    MEMORY = "memory"
    HARDWARE = "hardware"
    MODEL = "model"


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics"""
    tokens_per_second: float = 0.0
    batches_per_second: float = 0.0
    samples_per_second: float = 0.0
    forward_pass_time: float = 0.0
    backward_pass_time: float = 0.0
    optimizer_step_time: float = 0.0
    data_loading_time: float = 0.0
    total_step_time: float = 0.0
    throughput_efficiency: float = 0.0  # Actual vs theoretical throughput


@dataclass
class GradientMetrics:
    """Gradient analysis metrics"""
    gradient_norm: float = 0.0
    gradient_mean: float = 0.0
    gradient_std: float = 0.0
    gradient_max: float = 0.0
    gradient_min: float = 0.0
    gradient_sparsity: float = 0.0  # Percentage of zero gradients
    gradient_clipping_ratio: float = 0.0
    gradient_explosion_risk: float = 0.0
    gradient_vanishing_risk: float = 0.0
    layer_gradient_ratios: Dict[str, float] = field(default_factory=dict)


@dataclass
class MemoryMetrics:
    """Memory usage metrics"""
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    gpu_memory_percent: float = 0.0
    cpu_memory_used: float = 0.0
    cpu_memory_total: float = 0.0
    cpu_memory_percent: float = 0.0
    model_memory: float = 0.0
    optimizer_memory: float = 0.0
    activation_memory: float = 0.0
    cache_memory: float = 0.0


@dataclass
class HardwareMetrics:
    """Hardware utilization metrics"""
    gpu_utilization: float = 0.0
    gpu_temperature: float = 0.0
    gpu_power_usage: float = 0.0
    cpu_utilization: float = 0.0
    cpu_temperature: float = 0.0
    disk_io_read: float = 0.0
    disk_io_write: float = 0.0
    network_io_sent: float = 0.0
    network_io_recv: float = 0.0


@dataclass
class ModelMetrics:
    """Model-specific metrics"""
    parameter_count: int = 0
    trainable_parameters: int = 0
    model_size_mb: float = 0.0
    activation_memory_mb: float = 0.0
    layer_wise_memory: Dict[str, float] = field(default_factory=dict)
    quantization_ratio: float = 0.0
    sparsity_ratio: float = 0.0


@dataclass
class Alert:
    """System alert"""
    timestamp: datetime
    level: AlertLevel
    metric_type: MetricType
    message: str
    value: float
    threshold: float
    recommendation: str


class MetricsCollector:
    """Collects various system and training metrics"""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.start_time = time.time()

        # Initialize GPU monitoring if available
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available and GPUTIL_AVAILABLE:
            try:
                self.gpus = GPUtil.getGPUs()
            except:
                self.gpus = []
                logging.warning("GPU monitoring not available")
        else:
            self.gpus = []

        logging.info("✅ Metrics Collector initialized")

    def collect_performance_metrics(self, step_times: Dict[str, float],
                                  batch_size: int, sequence_length: int) -> PerformanceMetrics:
        """Collect performance metrics"""
        total_time = step_times.get('total', 0.001)  # Avoid division by zero
        tokens_processed = batch_size * sequence_length

        metrics = PerformanceMetrics(
            tokens_per_second=tokens_processed / total_time,
            batches_per_second=1.0 / total_time,
            samples_per_second=batch_size / total_time,
            forward_pass_time=step_times.get('forward', 0.0),
            backward_pass_time=step_times.get('backward', 0.0),
            optimizer_step_time=step_times.get('optimizer', 0.0),
            data_loading_time=step_times.get('data_loading', 0.0),
            total_step_time=total_time
        )

        # Calculate throughput efficiency (actual vs theoretical)
        if self.gpu_available and self.gpus:
            # Rough estimate based on GPU memory bandwidth
            gpu = self.gpus[0]
            theoretical_throughput = gpu.memoryTotal * 1000  # Rough estimate
            metrics.throughput_efficiency = min(1.0, metrics.tokens_per_second / theoretical_throughput)

        return metrics

    def collect_gradient_metrics(self, model: nn.Module,
                               max_grad_norm: Optional[float] = None) -> GradientMetrics:
        """Collect gradient analysis metrics"""
        gradients = []
        layer_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                gradients.append(grad.flatten())

                # Layer-wise gradient norms
                layer_norm = torch.norm(grad).item()
                layer_norms[name] = layer_norm

        if not gradients:
            return GradientMetrics()

        # Concatenate all gradients
        all_grads = torch.cat(gradients)

        # Calculate metrics
        grad_norm = torch.norm(all_grads).item()
        grad_mean = torch.mean(all_grads).item()
        grad_std = torch.std(all_grads).item()
        grad_max = torch.max(all_grads).item()
        grad_min = torch.min(all_grads).item()

        # Sparsity (percentage of near-zero gradients)
        sparsity = (torch.abs(all_grads) < 1e-8).float().mean().item()

        # Gradient clipping ratio
        clipping_ratio = 0.0
        if max_grad_norm is not None and grad_norm > max_grad_norm:
            clipping_ratio = max_grad_norm / grad_norm

        # Risk assessments
        explosion_risk = min(1.0, grad_norm / 10.0)  # Risk if norm > 10
        vanishing_risk = max(0.0, 1.0 - grad_norm * 1000)  # Risk if norm < 0.001

        return GradientMetrics(
            gradient_norm=grad_norm,
            gradient_mean=grad_mean,
            gradient_std=grad_std,
            gradient_max=grad_max,
            gradient_min=grad_min,
            gradient_sparsity=sparsity,
            gradient_clipping_ratio=clipping_ratio,
            gradient_explosion_risk=explosion_risk,
            gradient_vanishing_risk=vanishing_risk,
            layer_gradient_ratios=layer_norms
        )

    def collect_memory_metrics(self) -> MemoryMetrics:
        """Collect memory usage metrics"""
        metrics = MemoryMetrics()

        # GPU memory
        if self.gpu_available:
            try:
                gpu_memory = torch.cuda.memory_stats()
                metrics.gpu_memory_used = gpu_memory.get('allocated_bytes.all.current', 0) / 1024**3  # GB
                metrics.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                metrics.gpu_memory_percent = (metrics.gpu_memory_used / metrics.gpu_memory_total) * 100

                # Detailed memory breakdown
                metrics.activation_memory = gpu_memory.get('active_bytes.all.current', 0) / 1024**3
                metrics.cache_memory = gpu_memory.get('reserved_bytes.all.current', 0) / 1024**3
            except Exception as e:
                logging.warning(f"GPU memory collection failed: {e}")

        # CPU memory
        if PSUTIL_AVAILABLE:
            try:
                cpu_memory = psutil.virtual_memory()
                metrics.cpu_memory_used = cpu_memory.used / 1024**3  # GB
                metrics.cpu_memory_total = cpu_memory.total / 1024**3  # GB
                metrics.cpu_memory_percent = cpu_memory.percent
            except Exception as e:
                logging.warning(f"CPU memory collection failed: {e}")

        # Model memory estimation
        try:
            model_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            metrics.model_memory = model_size / 1024**3  # GB
        except Exception as e:
            logging.warning(f"Model memory estimation failed: {e}")

        return metrics

    def collect_hardware_metrics(self) -> HardwareMetrics:
        """Collect hardware utilization metrics"""
        metrics = HardwareMetrics()

        # GPU metrics
        if self.gpus:
            try:
                gpu = self.gpus[0]
                metrics.gpu_utilization = gpu.load * 100
                metrics.gpu_temperature = gpu.temperature
                metrics.gpu_power_usage = getattr(gpu, 'powerDraw', 0)
            except Exception as e:
                logging.warning(f"GPU hardware metrics failed: {e}")

        # CPU metrics
        if PSUTIL_AVAILABLE:
            try:
                metrics.cpu_utilization = psutil.cpu_percent(interval=0.1)

                # CPU temperature (if available)
                try:
                    temps = psutil.sensors_temperatures()
                    if 'coretemp' in temps:
                        metrics.cpu_temperature = temps['coretemp'][0].current
                except:
                    pass  # Temperature monitoring not available on all systems
            except Exception as e:
                logging.warning(f"CPU metrics failed: {e}")

            # Disk I/O
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    metrics.disk_io_read = disk_io.read_bytes / 1024**2  # MB
                    metrics.disk_io_write = disk_io.write_bytes / 1024**2  # MB
            except Exception as e:
                logging.warning(f"Disk I/O metrics failed: {e}")

            # Network I/O
            try:
                net_io = psutil.net_io_counters()
                if net_io:
                    metrics.network_io_sent = net_io.bytes_sent / 1024**2  # MB
                    metrics.network_io_recv = net_io.bytes_recv / 1024**2  # MB
            except Exception as e:
                logging.warning(f"Network I/O metrics failed: {e}")

        return metrics

    def collect_model_metrics(self) -> ModelMetrics:
        """Collect model-specific metrics"""
        metrics = ModelMetrics()

        try:
            # Parameter counts
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            metrics.parameter_count = total_params
            metrics.trainable_parameters = trainable_params

            # Model size
            model_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            metrics.model_size_mb = model_size / 1024**2

            # Layer-wise memory usage
            layer_memory = {}
            for name, param in self.model.named_parameters():
                param_size = param.numel() * param.element_size() / 1024**2  # MB
                layer_memory[name] = param_size
            metrics.layer_wise_memory = layer_memory

            # Sparsity analysis
            zero_params = sum((p == 0).sum().item() for p in self.model.parameters())
            metrics.sparsity_ratio = zero_params / total_params if total_params > 0 else 0.0

            # Quantization analysis (rough estimate)
            quantized_params = 0
            for param in self.model.parameters():
                if param.dtype in [torch.int8, torch.uint8]:
                    quantized_params += param.numel()
            metrics.quantization_ratio = quantized_params / total_params if total_params > 0 else 0.0

        except Exception as e:
            logging.warning(f"Model metrics collection failed: {e}")

        return metrics


class AnomalyDetector:
    """Detects anomalies in training metrics"""

    def __init__(self, window_size: int = 100, sensitivity: float = 2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity  # Standard deviations for anomaly threshold

        # Metric history for anomaly detection
        self.metric_history = defaultdict(lambda: deque(maxlen=window_size))
        self.anomaly_thresholds = {}

        logging.info("✅ Anomaly Detector initialized")

    def update_metrics(self, metrics: Dict[str, float]):
        """Update metric history and detect anomalies"""
        anomalies = []

        for metric_name, value in metrics.items():
            if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                continue

            history = self.metric_history[metric_name]
            history.append(value)

            # Detect anomalies if we have enough history
            if len(history) >= min(10, self.window_size // 2):
                anomaly = self._detect_anomaly(metric_name, value, history)
                if anomaly:
                    anomalies.append(anomaly)

        return anomalies

    def _detect_anomaly(self, metric_name: str, current_value: float,
                       history: deque) -> Optional[Alert]:
        """Detect if current value is anomalous"""
        if len(history) < 5:
            return None

        # Calculate statistics
        values = np.array(list(history)[:-1])  # Exclude current value
        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return None

        # Z-score based anomaly detection
        z_score = abs(current_value - mean) / std

        if z_score > self.sensitivity:
            # Determine alert level based on severity
            if z_score > 4.0:
                level = AlertLevel.EMERGENCY
            elif z_score > 3.0:
                level = AlertLevel.CRITICAL
            else:
                level = AlertLevel.WARNING

            # Generate recommendation
            recommendation = self._generate_recommendation(metric_name, current_value, mean, std)

            return Alert(
                timestamp=datetime.now(),
                level=level,
                metric_type=self._get_metric_type(metric_name),
                message=f"Anomaly detected in {metric_name}: {current_value:.4f} (z-score: {z_score:.2f})",
                value=current_value,
                threshold=mean + self.sensitivity * std,
                recommendation=recommendation
            )

        return None

    def _get_metric_type(self, metric_name: str) -> MetricType:
        """Determine metric type from name"""
        if 'gradient' in metric_name.lower():
            return MetricType.GRADIENT
        elif 'memory' in metric_name.lower():
            return MetricType.MEMORY
        elif 'gpu' in metric_name.lower() or 'cpu' in metric_name.lower():
            return MetricType.HARDWARE
        elif 'loss' in metric_name.lower():
            return MetricType.LOSS
        elif any(perf in metric_name.lower() for perf in ['tokens', 'throughput', 'time']):
            return MetricType.PERFORMANCE
        else:
            return MetricType.MODEL

    def _generate_recommendation(self, metric_name: str, current_value: float,
                               mean: float, std: float) -> str:
        """Generate recommendation based on anomaly"""
        metric_lower = metric_name.lower()

        if 'gradient_norm' in metric_lower:
            if current_value > mean:
                return "Consider reducing learning rate or increasing gradient clipping"
            else:
                return "Check for vanishing gradients, consider increasing learning rate"

        elif 'memory' in metric_lower:
            if current_value > mean:
                return "Reduce batch size or enable gradient checkpointing"
            else:
                return "Memory usage unexpectedly low, check data loading"

        elif 'tokens_per_second' in metric_lower:
            if current_value < mean:
                return "Performance degraded, check GPU utilization and memory usage"
            else:
                return "Performance improved, current settings are working well"

        elif 'loss' in metric_lower:
            if current_value > mean:
                return "Loss increased unexpectedly, check learning rate and data quality"
            else:
                return "Loss decreased significantly, monitor for overfitting"

        else:
            return f"Monitor {metric_name} closely and investigate cause of anomaly"


class TrainingDiagnostics:
    """Comprehensive training diagnostics and reporting"""

    def __init__(self, model: nn.Module, device: torch.device,
                 monitoring_interval: int = 10):
        self.model = model
        self.device = device
        self.monitoring_interval = monitoring_interval

        # Initialize components
        self.metrics_collector = MetricsCollector(model, device)
        self.anomaly_detector = AnomalyDetector()

        # Metric storage
        self.metrics_history = defaultdict(list)
        self.alerts = []

        # Timing
        self.last_collection_time = time.time()
        self.step_times = {}

        # Threading for background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None

        logging.info("✅ Training Diagnostics initialized")

    def start_monitoring(self):
        """Start background monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._background_monitoring)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logging.info("🔍 Background monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logging.info("⏹️ Background monitoring stopped")

    def _background_monitoring(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect hardware metrics periodically
                hardware_metrics = self.metrics_collector.collect_hardware_metrics()
                memory_metrics = self.metrics_collector.collect_memory_metrics()

                # Store metrics
                current_time = time.time()
                self._store_metrics('hardware', hardware_metrics.__dict__, current_time)
                self._store_metrics('memory', memory_metrics.__dict__, current_time)

                # Check for anomalies
                all_metrics = {**hardware_metrics.__dict__, **memory_metrics.__dict__}
                anomalies = self.anomaly_detector.update_metrics(all_metrics)
                self.alerts.extend(anomalies)

                time.sleep(self.monitoring_interval)

            except Exception as e:
                logging.error(f"Background monitoring error: {e}")
                time.sleep(self.monitoring_interval)

    def collect_step_metrics(self, step_times: Dict[str, float], batch_size: int,
                           sequence_length: int, loss: float, learning_rate: float):
        """Collect metrics for a training step"""
        current_time = time.time()

        try:
            # Performance metrics
            perf_metrics = self.metrics_collector.collect_performance_metrics(
                step_times, batch_size, sequence_length
            )

            # Gradient metrics
            grad_metrics = self.metrics_collector.collect_gradient_metrics(self.model)

            # Memory metrics (if not collected recently)
            if current_time - self.last_collection_time > 5.0:  # Every 5 seconds
                memory_metrics = self.metrics_collector.collect_memory_metrics()
                self._store_metrics('memory', memory_metrics.__dict__, current_time)
                self.last_collection_time = current_time

            # Store metrics
            self._store_metrics('performance', perf_metrics.__dict__, current_time)
            self._store_metrics('gradient', grad_metrics.__dict__, current_time)

            # Add training-specific metrics
            training_metrics = {
                'loss': loss,
                'learning_rate': learning_rate,
                'step_time': step_times.get('total', 0.0)
            }
            self._store_metrics('training', training_metrics, current_time)

            # Anomaly detection
            all_metrics = {
                **perf_metrics.__dict__,
                **grad_metrics.__dict__,
                **training_metrics
            }
            anomalies = self.anomaly_detector.update_metrics(all_metrics)
            self.alerts.extend(anomalies)

            # Log critical alerts immediately
            for alert in anomalies:
                if alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                    logging.warning(f"🚨 {alert.level.value.upper()}: {alert.message}")
                    logging.warning(f"   Recommendation: {alert.recommendation}")

        except Exception as e:
            logging.error(f"Step metrics collection failed: {e}")

    def _store_metrics(self, category: str, metrics: Dict[str, Any], timestamp: float):
        """Store metrics with timestamp"""
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                metric_key = f"{category}_{key}"
                self.metrics_history[metric_key].append((timestamp, value))

                # Keep only recent history (last 1000 points)
                if len(self.metrics_history[metric_key]) > 1000:
                    self.metrics_history[metric_key] = self.metrics_history[metric_key][-1000:]

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'performance_analysis': {},
            'gradient_analysis': {},
            'memory_analysis': {},
            'hardware_analysis': {},
            'alerts_summary': {},
            'recommendations': []
        }

        try:
            # Summary statistics
            report['summary'] = self._generate_summary_stats()

            # Performance analysis
            report['performance_analysis'] = self._analyze_performance()

            # Gradient analysis
            report['gradient_analysis'] = self._analyze_gradients()

            # Memory analysis
            report['memory_analysis'] = self._analyze_memory()

            # Hardware analysis
            report['hardware_analysis'] = self._analyze_hardware()

            # Alerts summary
            report['alerts_summary'] = self._summarize_alerts()

            # Generate recommendations
            report['recommendations'] = self._generate_recommendations()

        except Exception as e:
            logging.error(f"Report generation failed: {e}")
            report['error'] = str(e)

        return report

    def _generate_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        summary = {
            'total_metrics_collected': len(self.metrics_history),
            'monitoring_duration_hours': (time.time() - self.metrics_collector.start_time) / 3600,
            'total_alerts': len(self.alerts),
            'critical_alerts': len([a for a in self.alerts if a.level == AlertLevel.CRITICAL]),
            'emergency_alerts': len([a for a in self.alerts if a.level == AlertLevel.EMERGENCY])
        }

        # Recent performance
        if 'performance_tokens_per_second' in self.metrics_history:
            recent_tps = [v for t, v in self.metrics_history['performance_tokens_per_second'][-10:]]
            if recent_tps:
                summary['avg_tokens_per_second'] = np.mean(recent_tps)
                summary['peak_tokens_per_second'] = np.max(recent_tps)

        return summary

    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics"""
        analysis = {}

        # Throughput analysis
        if 'performance_tokens_per_second' in self.metrics_history:
            tps_data = self.metrics_history['performance_tokens_per_second']
            values = [v for t, v in tps_data]

            analysis['throughput'] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'trend': self._calculate_trend(values)
            }

        # Timing analysis
        timing_metrics = ['forward_pass_time', 'backward_pass_time', 'optimizer_step_time']
        for metric in timing_metrics:
            key = f'performance_{metric}'
            if key in self.metrics_history:
                values = [v for t, v in self.metrics_history[key]]
                analysis[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'trend': self._calculate_trend(values)
                }

        return analysis

    def _analyze_gradients(self) -> Dict[str, Any]:
        """Analyze gradient metrics"""
        analysis = {}

        gradient_metrics = ['gradient_norm', 'gradient_explosion_risk', 'gradient_vanishing_risk']
        for metric in gradient_metrics:
            key = f'gradient_{metric}'
            if key in self.metrics_history:
                values = [v for t, v in self.metrics_history[key]]
                analysis[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'current': values[-1] if values else 0,
                    'trend': self._calculate_trend(values),
                    'stability': np.std(values) / (np.mean(values) + 1e-8)  # Coefficient of variation
                }

        return analysis

    def _analyze_memory(self) -> Dict[str, Any]:
        """Analyze memory usage"""
        analysis = {}

        memory_metrics = ['gpu_memory_percent', 'cpu_memory_percent']
        for metric in memory_metrics:
            key = f'memory_{metric}'
            if key in self.metrics_history:
                values = [v for t, v in self.metrics_history[key]]
                analysis[metric] = {
                    'mean': np.mean(values),
                    'max': np.max(values),
                    'current': values[-1] if values else 0,
                    'trend': self._calculate_trend(values)
                }

        return analysis

    def _analyze_hardware(self) -> Dict[str, Any]:
        """Analyze hardware utilization"""
        analysis = {}

        hardware_metrics = ['gpu_utilization', 'cpu_utilization', 'gpu_temperature']
        for metric in hardware_metrics:
            key = f'hardware_{metric}'
            if key in self.metrics_history:
                values = [v for t, v in self.metrics_history[key]]
                analysis[metric] = {
                    'mean': np.mean(values),
                    'max': np.max(values),
                    'current': values[-1] if values else 0,
                    'efficiency': np.mean(values) / 100.0 if 'utilization' in metric else None
                }

        return analysis

    def _summarize_alerts(self) -> Dict[str, Any]:
        """Summarize alerts"""
        if not self.alerts:
            return {'total': 0}

        # Count by level
        level_counts = defaultdict(int)
        for alert in self.alerts:
            level_counts[alert.level.value] += 1

        # Count by type
        type_counts = defaultdict(int)
        for alert in self.alerts:
            type_counts[alert.metric_type.value] += 1

        # Recent alerts (last hour)
        recent_time = datetime.now() - timedelta(hours=1)
        recent_alerts = [a for a in self.alerts if a.timestamp > recent_time]

        return {
            'total': len(self.alerts),
            'by_level': dict(level_counts),
            'by_type': dict(type_counts),
            'recent_count': len(recent_alerts),
            'most_recent': self.alerts[-1].message if self.alerts else None
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate training recommendations"""
        recommendations = []

        # Performance recommendations
        if 'performance_tokens_per_second' in self.metrics_history:
            tps_values = [v for t, v in self.metrics_history['performance_tokens_per_second']]
            if tps_values and np.mean(tps_values) < 1000:  # Low throughput
                recommendations.append("Consider increasing batch size or optimizing data loading")

        # Memory recommendations
        if 'memory_gpu_memory_percent' in self.metrics_history:
            gpu_mem_values = [v for t, v in self.metrics_history['memory_gpu_memory_percent']]
            if gpu_mem_values and np.mean(gpu_mem_values) > 90:
                recommendations.append("GPU memory usage is high - consider reducing batch size or enabling gradient checkpointing")

        # Gradient recommendations
        if 'gradient_gradient_explosion_risk' in self.metrics_history:
            explosion_risk = [v for t, v in self.metrics_history['gradient_gradient_explosion_risk']]
            if explosion_risk and np.mean(explosion_risk) > 0.5:
                recommendations.append("High gradient explosion risk - consider reducing learning rate or increasing gradient clipping")

        # Alert-based recommendations
        critical_alerts = [a for a in self.alerts if a.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]]
        if critical_alerts:
            recommendations.append("Address critical alerts immediately to prevent training instability")

        return recommendations

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return "insufficient_data"

        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if abs(slope) < 1e-6:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

    def create_visualization(self, output_dir: str = "./monitoring_plots"):
        """Create visualization plots"""
        if not MATPLOTLIB_AVAILABLE:
            logging.warning("Matplotlib not available - skipping visualization")
            return

        try:
            import os
            os.makedirs(output_dir, exist_ok=True)

            # Set style
            try:
                plt.style.use('seaborn-v0_8')
            except:
                # Fallback if seaborn style not available
                plt.style.use('default')

            # Performance plot
            self._plot_performance_metrics(output_dir)

            # Memory usage plot
            self._plot_memory_metrics(output_dir)

            # Gradient analysis plot
            self._plot_gradient_metrics(output_dir)

            # Alerts timeline
            self._plot_alerts_timeline(output_dir)

            logging.info(f"📊 Visualization plots saved to {output_dir}")

        except Exception as e:
            logging.error(f"Visualization creation failed: {e}")

    def _plot_performance_metrics(self, output_dir: str):
        """Plot performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Metrics', fontsize=16)

        # Tokens per second
        if 'performance_tokens_per_second' in self.metrics_history:
            data = self.metrics_history['performance_tokens_per_second']
            times, values = zip(*data) if data else ([], [])
            axes[0, 0].plot(times, values)
            axes[0, 0].set_title('Tokens per Second')
            axes[0, 0].set_ylabel('Tokens/sec')

        # Step times
        timing_metrics = ['performance_forward_pass_time', 'performance_backward_pass_time', 'performance_optimizer_step_time']
        for i, metric in enumerate(timing_metrics):
            if metric in self.metrics_history:
                data = self.metrics_history[metric]
                times, values = zip(*data) if data else ([], [])
                axes[0, 1].plot(times, values, label=metric.split('_')[-2])
        axes[0, 1].set_title('Step Timing')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].legend()

        # Memory usage
        if 'memory_gpu_memory_percent' in self.metrics_history:
            data = self.metrics_history['memory_gpu_memory_percent']
            times, values = zip(*data) if data else ([], [])
            axes[1, 0].plot(times, values, color='red')
            axes[1, 0].set_title('GPU Memory Usage')
            axes[1, 0].set_ylabel('Memory %')

        # Hardware utilization
        if 'hardware_gpu_utilization' in self.metrics_history:
            data = self.metrics_history['hardware_gpu_utilization']
            times, values = zip(*data) if data else ([], [])
            axes[1, 1].plot(times, values, color='green')
            axes[1, 1].set_title('GPU Utilization')
            axes[1, 1].set_ylabel('Utilization %')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_memory_metrics(self, output_dir: str):
        """Plot memory metrics"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('Memory Usage Analysis', fontsize=16)

        # GPU and CPU memory
        memory_metrics = [
            ('memory_gpu_memory_percent', 'GPU Memory %', 'red'),
            ('memory_cpu_memory_percent', 'CPU Memory %', 'blue')
        ]

        for metric, label, color in memory_metrics:
            if metric in self.metrics_history:
                data = self.metrics_history[metric]
                times, values = zip(*data) if data else ([], [])
                axes[0].plot(times, values, label=label, color=color)

        axes[0].set_title('Memory Usage Over Time')
        axes[0].set_ylabel('Memory %')
        axes[0].legend()
        axes[0].grid(True)

        # Memory breakdown (if available)
        memory_breakdown = ['memory_model_memory', 'memory_activation_memory', 'memory_cache_memory']
        for metric in memory_breakdown:
            if metric in self.metrics_history:
                data = self.metrics_history[metric]
                times, values = zip(*data) if data else ([], [])
                axes[1].plot(times, values, label=metric.split('_')[-1])

        axes[1].set_title('Memory Breakdown')
        axes[1].set_ylabel('Memory (GB)')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'memory_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_gradient_metrics(self, output_dir: str):
        """Plot gradient metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Gradient Analysis', fontsize=16)

        # Gradient norm
        if 'gradient_gradient_norm' in self.metrics_history:
            data = self.metrics_history['gradient_gradient_norm']
            times, values = zip(*data) if data else ([], [])
            axes[0, 0].plot(times, values)
            axes[0, 0].set_title('Gradient Norm')
            axes[0, 0].set_ylabel('Norm')
            axes[0, 0].grid(True)

        # Gradient statistics
        grad_stats = ['gradient_gradient_mean', 'gradient_gradient_std']
        for metric in grad_stats:
            if metric in self.metrics_history:
                data = self.metrics_history[metric]
                times, values = zip(*data) if data else ([], [])
                axes[0, 1].plot(times, values, label=metric.split('_')[-1])
        axes[0, 1].set_title('Gradient Statistics')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Risk analysis
        risk_metrics = ['gradient_gradient_explosion_risk', 'gradient_gradient_vanishing_risk']
        for metric in risk_metrics:
            if metric in self.metrics_history:
                data = self.metrics_history[metric]
                times, values = zip(*data) if data else ([], [])
                axes[1, 0].plot(times, values, label=metric.split('_')[-2] + '_risk')
        axes[1, 0].set_title('Gradient Risk Analysis')
        axes[1, 0].set_ylabel('Risk Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Sparsity
        if 'gradient_gradient_sparsity' in self.metrics_history:
            data = self.metrics_history['gradient_gradient_sparsity']
            times, values = zip(*data) if data else ([], [])
            axes[1, 1].plot(times, values, color='purple')
            axes[1, 1].set_title('Gradient Sparsity')
            axes[1, 1].set_ylabel('Sparsity Ratio')
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_alerts_timeline(self, output_dir: str):
        """Plot alerts timeline"""
        if not self.alerts:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        # Group alerts by level
        level_colors = {
            AlertLevel.INFO: 'blue',
            AlertLevel.WARNING: 'orange',
            AlertLevel.CRITICAL: 'red',
            AlertLevel.EMERGENCY: 'darkred'
        }

        for level in AlertLevel:
            level_alerts = [a for a in self.alerts if a.level == level]
            if level_alerts:
                times = [a.timestamp for a in level_alerts]
                values = [level.value for a in level_alerts]
                ax.scatter(times, values, c=level_colors[level], label=level.value, alpha=0.7)

        ax.set_title('Alerts Timeline')
        ax.set_ylabel('Alert Level')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'alerts_timeline.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_active': self.monitoring_active,
            'recent_alerts': len([a for a in self.alerts if a.timestamp > datetime.now() - timedelta(minutes=10)]),
            'system_health': 'unknown'
        }

        # Determine system health
        recent_critical = len([a for a in self.alerts
                             if a.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]
                             and a.timestamp > datetime.now() - timedelta(minutes=30)])

        if recent_critical > 0:
            status['system_health'] = 'critical'
        elif len([a for a in self.alerts if a.timestamp > datetime.now() - timedelta(minutes=10)]) > 5:
            status['system_health'] = 'warning'
        else:
            status['system_health'] = 'healthy'

        # Add current metrics if available
        current_metrics = {}
        for key, history in self.metrics_history.items():
            if history:
                current_metrics[key] = history[-1][1]  # Latest value

        status['current_metrics'] = current_metrics

        return status


def create_monitoring_system(config: Any) -> TrainingDiagnostics:
    """Factory function to create monitoring system"""
    # This would normally get model and device from config
    # For now, create a dummy model for testing
    model = nn.Linear(10, 10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return TrainingDiagnostics(model, device, getattr(config, 'monitoring_interval', 10))


def test_monitoring_system():
    """Test the monitoring and diagnostics system"""
    print("🧪 Testing Monitoring and Diagnostics System")

    # Create test model
    model = nn.Sequential(
        nn.Linear(100, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Create diagnostics system
    diagnostics = TrainingDiagnostics(model, device, monitoring_interval=1)

    print("✅ Diagnostics system created")

    # Test metrics collection
    step_times = {
        'forward': 0.1,
        'backward': 0.15,
        'optimizer': 0.05,
        'total': 0.3
    }

    # Simulate training steps
    for step in range(10):
        # Add some noise to simulate real training
        noisy_times = {k: v * (1 + 0.1 * np.random.randn()) for k, v in step_times.items()}
        loss = 2.5 * np.exp(-step * 0.1) + 0.1 * np.random.randn()
        lr = 1e-4 * (1 - step * 0.05)

        diagnostics.collect_step_metrics(
            noisy_times, batch_size=32, sequence_length=128,
            loss=loss, learning_rate=lr
        )

    print("✅ Collected metrics for 10 training steps")

    # Generate report
    report = diagnostics.generate_performance_report()
    print(f"✅ Generated performance report with {len(report)} sections")

    # Test current status
    status = diagnostics.get_current_status()
    print(f"✅ System health: {status['system_health']}")

    # Test visualization (if matplotlib available)
    try:
        diagnostics.create_visualization("./test_plots")
        print("✅ Created visualization plots")
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")

    print("🎉 Monitoring system tests completed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    test_monitoring_system()