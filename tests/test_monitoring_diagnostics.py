#!/usr/bin/env python3
"""
Test suite for Monitoring and Diagnostics System
===============================================

Comprehensive tests for the monitoring system including:
- Metrics collection and analysis
- Anomaly detection
- Alert generation
- Performance reporting
- Visualization (if available)
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
import tempfile
import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring_diagnostics import (
    TrainingDiagnostics, MetricsCollector, AnomalyDetector,
    PerformanceMetrics, GradientMetrics, MemoryMetrics, HardwareMetrics, ModelMetrics,
    Alert, AlertLevel, MetricType, create_monitoring_system
)


class SimpleTestModel(nn.Module):
    """Simple model for testing"""

    def __init__(self, input_size=100, hidden_size=64, output_size=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        return self.layers(x)


@pytest.fixture
def test_model():
    return SimpleTestModel()


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def metrics_collector(test_model, device):
    test_model.to(device)
    return MetricsCollector(test_model, device)


@pytest.fixture
def anomaly_detector():
    return AnomalyDetector(window_size=50, sensitivity=2.0)


@pytest.fixture
def training_diagnostics(test_model, device):
    test_model.to(device)
    return TrainingDiagnostics(test_model, device, monitoring_interval=1)


class TestPerformanceMetrics:
    """Test performance metrics data structure"""

    def test_default_initialization(self):
        """Test default performance metrics initialization"""
        metrics = PerformanceMetrics()

        assert metrics.tokens_per_second == 0.0
        assert metrics.batches_per_second == 0.0
        assert metrics.samples_per_second == 0.0
        assert metrics.forward_pass_time == 0.0
        assert metrics.backward_pass_time == 0.0
        assert metrics.optimizer_step_time == 0.0
        assert metrics.data_loading_time == 0.0
        assert metrics.total_step_time == 0.0
        assert metrics.throughput_efficiency == 0.0

    def test_custom_initialization(self):
        """Test custom performance metrics initialization"""
        metrics = PerformanceMetrics(
            tokens_per_second=1000.0,
            batches_per_second=10.0,
            forward_pass_time=0.1,
            backward_pass_time=0.15
        )

        assert metrics.tokens_per_second == 1000.0
        assert metrics.batches_per_second == 10.0
        assert metrics.forward_pass_time == 0.1
        assert metrics.backward_pass_time == 0.15


class TestGradientMetrics:
    """Test gradient metrics data structure"""

    def test_default_initialization(self):
        """Test default gradient metrics initialization"""
        metrics = GradientMetrics()

        assert metrics.gradient_norm == 0.0
        assert metrics.gradient_mean == 0.0
        assert metrics.gradient_std == 0.0
        assert metrics.gradient_max == 0.0
        assert metrics.gradient_min == 0.0
        assert metrics.gradient_sparsity == 0.0
        assert metrics.gradient_clipping_ratio == 0.0
        assert metrics.gradient_explosion_risk == 0.0
        assert metrics.gradient_vanishing_risk == 0.0
        assert len(metrics.layer_gr_ratios) == 0


class TestMemoryMetrics:
    """Test memory metrics data structure"""

    def test_default_initialization(self):
        """Test default memory metrics initialization"""
        metrics = MemoryMetrics()

        assert metrics.gpu_memory_used == 0.0
        assert metrics.gpu_memory_total == 0.0
        assert metrics.gpu_memory_percent == 0.0
        assert metrics.cpu_memory_used == 0.0
        assert metrics.cpu_memory_total == 0.0
        assert metrics.cpu_memory_percent == 0.0
        assert metrics.model_memory == 0.0
        assert metrics.optimizer_memory == 0.0
        assert metrics.activation_memory == 0.0
        assert metrics.cache_memory == 0.0


class TestAlert:
    """Test alert system"""

    def test_alert_creation(self):
        """Test alert creation"""
        alert = Alert(
            timestamp=datetime.now(),
            level=AlertLevel.WARNING,
            metric_type=MetricType.GRADIENT,
            message="Test alert",
            value=5.0,
            threshold=3.0,
            recommendation="Test recommendation"
        )

        assert alert.level == AlertLevel.WARNING
        assert alert.metric_type == MetricType.GRADIENT
        assert alert.message == "Test alert"
        assert alert.value == 5.0
        assert alert.threshold == 3.0
        assert alert.recommendation == "Test recommendation"


class TestMetricsCollector:
    """Test metrics collection functionality"""

    def test_initialization(self, metrics_collector, test_model, device):
        """Test metrics collector initialization"""
        assert metrics_collector.model == test_model
        assert metrics_collector.device == device
        assert isinstance(metrics_collector.start_time, float)
        assert isinstance(metrics_collector.gpu_available, bool)
        assert isinstance(metrics_collector.gpus, list)

    def test_performance_metrics_collection(self, metrics_collector):
        """Test performance metrics collection"""
        step_times = {
            'forward': 0.1,
            'backward': 0.15,
            'optimizer': 0.05,
            'data_loading': 0.02,
            'total': 0.32
        }

        metrics = metrics_collector.collect_performance_metrics(
            step_times, batch_size=32, sequence_length=128
        )

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.tokens_per_second > 0
        assert metrics.batches_per_second > 0
        assert metrics.samples_per_second > 0
        assert metrics.forward_pass_time == 0.1
        assert metrics.backward_pass_time == 0.15
        assert metrics.optimizer_step_time == 0.05
        assert metrics.data_loading_time == 0.02
        assert metrics.total_step_time == 0.32

    def test_gradient_metrics_collection(self, metrics_collector, test_model, device):
        """Test gradient metrics collection"""
        # Create dummy input and compute gradients
        x = torch.randn(4, 100).to(device)
        y = torch.randn(4, 10).to(device)

        output = test_model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        metrics = metrics_collector.collect_gradient_metrics(test_model)

        assert isinstance(metrics, GradientMetrics)
        assert metrics.gradient_norm >= 0
        assert isinstance(metrics.gradient_mean, float)
        assert isinstance(metrics.gradient_std, float)
        assert isinstance(metrics.gradient_sparsity, float)
        assert 0 <= metrics.gradient_sparsity <= 1
        assert len(metrics.layer_gradient_ratios) > 0

    def test_memory_metrics_collection(self, metrics_collector):
        """Test memory metrics collection"""
        metrics = metrics_collector.collect_memory_metrics()

        assert isinstance(metrics, MemoryMetrics)
        assert metrics.model_memory >= 0
        # GPU memory might be 0 if no GPU available
        assert metrics.gpu_memory_used >= 0
        assert metrics.gpu_memory_total >= 0

    def test_hardware_metrics_collection(self, metrics_collector):
        """Test hardware metrics collection"""
        metrics = metrics_collector.collect_hardware_metrics()

        assert isinstance(metrics, HardwareMetrics)
        # Values might be 0 if monitoring not available
        assert metrics.gpu_utilization >= 0
        assert metrics.cpu_utilization >= 0
        assert metrics.gpu_temperature >= 0

    def test_model_metrics_collection(self, metrics_collector):
        """Test model metrics collection"""
        metrics = metrics_collector.collect_model_metrics()

        assert isinstance(metrics, ModelMetrics)
        assert metrics.parameter_count > 0
        assert metrics.trainable_parameters > 0
        assert metrics.model_size_mb > 0
        assert len(metrics.layer_wise_memory) > 0
        assert 0 <= metrics.sparsity_ratio <= 1
        assert 0 <= metrics.quantization_ratio <= 1


class TestAnomalyDetector:
    """Test anomaly detection functionality"""

    def test_initialization(self, anomaly_detector):
        """Test anomaly detector initialization"""
        assert anomaly_detector.window_size == 50
        assert anomaly_detector.sensitivity == 2.0
        assert len(anomaly_detector.metric_history) == 0
        assert len(anomaly_detector.anomaly_thresholds) == 0

    def test_normal_metrics_update(self, anomaly_detector):
        """Test updating with normal metrics (no anomalies)"""
        # Add normal values
        for i in range(20):
            metrics = {'test_metric': 1.0 + 0.1 * np.random.randn()}
            anomalies = anomaly_detector.update_metrics(metrics)

            # Should not detect anomalies in normal data
            if i > 10:  # After enough history
                assert len(anomalies) == 0

    def test_anomaly_detection(self, anomaly_detector):
        """Test anomaly detection with outliers"""
        # Add normal values
        for i in range(15):
            metrics = {'test_metric': 1.0 + 0.1 * np.random.randn()}
            anomaly_detector.update_metrics(metrics)

        # Add anomalous value
        anomalous_metrics = {'test_metric': 10.0}  # Clear outlier
        anomalies = anomaly_detector.update_metrics(anomalous_metrics)

        assert len(anomalies) > 0
        assert anomalies[0].level in [AlertLevel.WARNING, AlertLevel.CRITICAL, AlertLevel.EMERGENCY]
        assert anomalies[0].value == 10.0
        assert "anomaly" in anomalies[0].message.lower()

    def test_metric_type_classification(self, anomaly_detector):
        """Test metric type classification"""
        # Test different metric types
        test_cases = [
            ('gradient_norm', MetricType.GRADIENT),
            ('memory_usage', MetricType.MEMORY),
            ('gpu_utilization', MetricType.HARDWARE),
            ('loss_value', MetricType.LOSS),
            ('tokens_per_second', MetricType.PERFORMANCE),
            ('unknown_metric', MetricType.MODEL)
        ]

        for metric_name, expected_type in test_cases:
            detected_type = anomaly_detector._get_metric_type(metric_name)
            assert detected_type == expected_type

    def test_recommendation_generation(self, anomaly_detector):
        """Test recommendation generation"""
        # Test gradient norm recommendation
        rec = anomaly_detector._generate_recommendation('gradient_norm', 10.0, 1.0, 0.5)
        assert 'learning rate' in rec.lower() or 'gradient' in rec.lower()

        # Test memory recommendation
        rec = anomaly_detector._generate_recommendation('memory_usage', 95.0, 50.0, 10.0)
        assert 'batch size' in rec.lower() or 'memory' in rec.lower()

        # Test performance recommendation
        rec = anomaly_detector._generate_recommendation('tokens_per_second', 100.0, 1000.0, 100.0)
        assert 'performance' in rec.lower() or 'gpu' in rec.lower()


class TestTrainingDiagnostics:
    """Test training diagnostics functionality"""

    def test_initialization(self, training_diagnostics, test_model, device):
        """Test training diagnostics initialization"""
        assert training_diagnostics.model == test_model
        assert training_diagnostics.device == device
        assert isinstance(training_diagnostics.metrics_collector, MetricsCollector)
        assert isinstance(training_diagnostics.anomaly_detector, AnomalyDetector)
        assert len(training_diagnostics.metrics_history) == 0
        assert len(training_diagnostics.alerts) == 0
        assert training_diagnostics.monitoring_active == False

    def test_step_metrics_collection(self, training_diagnostics):
        """Test step metrics collection"""
        step_times = {
            'forward': 0.1,
            'backward': 0.15,
            'optimizer': 0.05,
            'total': 0.3
        }

        training_diagnostics.collect_step_metrics(
            step_times, batch_size=32, sequence_length=128,
            loss=2.5, learning_rate=1e-4
        )

        # Check that metrics were stored
        assert len(training_diagnostics.metrics_history) > 0

        # Check specific metrics
        assert 'performance_tokens_per_second' in training_diagnostics.metrics_history
        assert 'training_loss' in training_diagnostics.metrics_history
        assert 'training_learning_rate' in training_diagnostics.metrics_history

    def test_multiple_step_collection(self, training_diagnostics):
        """Test collecting metrics for multiple steps"""
        for step in range(10):
            step_times = {
                'forward': 0.1 + 0.01 * np.random.randn(),
                'backward': 0.15 + 0.01 * np.random.randn(),
                'optimizer': 0.05 + 0.005 * np.random.randn(),
                'total': 0.3 + 0.02 * np.random.randn()
            }

            loss = 3.0 * np.exp(-step * 0.1) + 0.1 * np.random.randn()
            lr = 1e-4 * (1 - step * 0.01)

            training_diagnostics.collect_step_metrics(
                step_times, batch_size=32, sequence_length=128,
                loss=loss, learning_rate=lr
            )

        # Check that we have history for multiple steps
        for key in training_diagnostics.metrics_history:
            assert len(training_diagnostics.metrics_history[key]) <= 10

    def test_performance_report_generation(self, training_diagnostics):
        """Test performance report generation"""
        # Collect some metrics first
        for step in range(5):
            step_times = {
                'forward': 0.1,
                'backward': 0.15,
                'optimizer': 0.05,
                'total': 0.3
            }

            training_diagnostics.collect_step_metrics(
                step_times, batch_size=32, sequence_length=128,
                loss=2.0 - step * 0.1, learning_rate=1e-4
            )

        report = training_diagnostics.generate_performance_report()

        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'summary' in report
        assert 'performance_analysis' in report
        assert 'gradient_analysis' in report
        assert 'memory_analysis' in report
        assert 'hardware_analysis' in report
        assert 'alerts_summary' in report
        assert 'recommendations' in report

        # Check summary
        summary = report['summary']
        assert 'total_metrics_collected' in summary
        assert 'monitoring_duration_hours' in summary
        assert 'total_alerts' in summary

    def test_current_status(self, training_diagnostics):
        """Test current status reporting"""
        status = training_diagnostics.get_current_status()

        assert isinstance(status, dict)
        assert 'timestamp' in status
        assert 'monitoring_active' in status
        assert 'recent_alerts' in status
        assert 'system_health' in status
        assert 'current_metrics' in status

        assert status['system_health'] in ['healthy', 'warning', 'critical']

    def test_monitoring_start_stop(self, training_diagnostics):
        """Test background monitoring start/stop"""
        # Start monitoring
        training_diagnostics.start_monitoring()
        assert training_diagnostics.monitoring_active == True
        assert training_diagnostics.monitoring_thread is not None

        # Give it a moment to start
        time.sleep(0.1)

        # Stop monitoring
        training_diagnostics.stop_monitoring()
        assert training_diagnostics.monitoring_active == False

    def test_visualization_creation(self, training_diagnostics):
        """Test visualization creation"""
        # Collect some metrics first
        for step in range(10):
            step_times = {
                'forward': 0.1,
                'backward': 0.15,
                'optimizer': 0.05,
                'total': 0.3
            }

            training_diagnostics.collect_step_metrics(
                step_times, batch_size=32, sequence_length=128,
                loss=2.0 - step * 0.1, learning_rate=1e-4
            )

        # Create visualization in temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            training_diagnostics.create_visualization(temp_dir)

            # Check if plots would be created (might not work without matplotlib)
            # This test mainly ensures the method doesn't crash
            assert os.path.exists(temp_dir)

    def test_trend_calculation(self, training_diagnostics):
        """Test trend calculation"""
        # Test increasing trend
        increasing_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        trend = training_diagnostics._calculate_trend(increasing_values)
        assert trend == "increasing"

        # Test decreasing trend
        decreasing_values = [5.0, 4.0, 3.0, 2.0, 1.0]
        trend = training_diagnostics._calculate_trend(decreasing_values)
        assert trend == "decreasing"

        # Test stable trend
        stable_values = [2.0, 2.0, 2.0, 2.0, 2.0]
        trend = training_diagnostics._calculate_trend(stable_values)
        assert trend == "stable"

        # Test insufficient data
        insufficient_values = [1.0]
        trend = training_diagnostics._calculate_trend(insufficient_values)
        assert trend == "insufficient_data"


class TestIntegration:
    """Integration tests for the complete monitoring system"""

    def test_end_to_end_monitoring(self, test_model, device):
        """Test complete end-to-end monitoring workflow"""
        # Create diagnostics system
        diagnostics = TrainingDiagnostics(test_model.to(device), device, monitoring_interval=1)

        # Simulate training loop
        optimizer = torch.optim.Adam(test_model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        for epoch in range(3):
            for step in range(5):
                # Simulate training step with timing
                start_time = time.time()

                # Forward pass
                forward_start = time.time()
                x = torch.randn(8, 100).to(device)
                y = torch.randn(8, 10).to(device)
                output = test_model(x)
                forward_time = time.time() - forward_start

                # Loss calculation
                loss = criterion(output, y)

                # Backward pass
                backward_start = time.time()
                loss.backward()
                backward_time = time.time() - backward_start

                # Optimizer step
                optimizer_start = time.time()
                optimizer.step()
                optimizer.zero_grad()
                optimizer_time = time.time() - optimizer_start

                total_time = time.time() - start_time

                # Collect metrics
                step_times = {
                    'forward': forward_time,
                    'backward': backward_time,
                    'optimizer': optimizer_time,
                    'total': total_time
                }

                diagnostics.collect_step_metrics(
                    step_times, batch_size=8, sequence_length=100,
                    loss=loss.item(), learning_rate=1e-4
                )

        # Generate report
        report = diagnostics.generate_performance_report()

        # Verify report contents
        assert len(report) > 0
        assert 'summary' in report
        assert report['summary']['total_metrics_collected'] > 0

        # Check that we have performance data
        assert 'performance_analysis' in report
        if 'throughput' in report['performance_analysis']:
            assert report['performance_analysis']['throughput']['mean'] > 0

        # Check system status
        status = diagnostics.get_current_status()
        assert status['system_health'] in ['healthy', 'warning', 'critical']

    def test_anomaly_detection_integration(self, test_model, device):
        """Test anomaly detection in integrated system"""
        diagnostics = TrainingDiagnostics(test_model.to(device), device)

        # Collect normal metrics
        for step in range(20):
            step_times = {
                'forward': 0.1 + 0.01 * np.random.randn(),
                'backward': 0.15 + 0.01 * np.random.randn(),
                'optimizer': 0.05 + 0.005 * np.random.randn(),
                'total': 0.3 + 0.02 * np.random.randn()
            }

            loss = 2.0 + 0.1 * np.random.randn()

            diagnostics.collect_step_metrics(
                step_times, batch_size=32, sequence_length=128,
                loss=loss, learning_rate=1e-4
            )

        # Inject anomalous metrics
        anomalous_times = {
            'forward': 1.0,  # 10x slower
            'backward': 1.5,
            'optimizer': 0.5,
            'total': 3.0
        }

        diagnostics.collect_step_metrics(
            anomalous_times, batch_size=32, sequence_length=128,
            loss=10.0,  # Much higher loss
            learning_rate=1e-4
        )

        # Check that anomalies were detected
        assert len(diagnostics.alerts) > 0

        # Check alert levels
        alert_levels = [alert.level for alert in diagnostics.alerts]
        assert any(level in [AlertLevel.WARNING, AlertLevel.CRITICAL, AlertLevel.EMERGENCY]
                  for level in alert_levels)


def test_create_monitoring_system():
    """Test factory function"""
    class MockConfig:
        monitoring_interval = 5

    config = MockConfig()
    monitoring_system = create_monitoring_system(config)

    assert isinstance(monitoring_system, TrainingDiagnostics)
    assert monitoring_system.monitoring_interval == 5


if __name__ == "__main__":
    # Run basic tests
    print("🧪 Running Monitoring Diagnostics Tests")

    # Test basic functionality
    model = SimpleTestModel()
    device = torch.device('cpu')  # Use CPU for testing

    # Test metrics collector
    collector = MetricsCollector(model, device)
    print("✅ MetricsCollector created")

    # Test performance metrics
    step_times = {'forward': 0.1, 'backward': 0.15, 'optimizer': 0.05, 'total': 0.3}
    perf_metrics = collector.collect_performance_metrics(step_times, 32, 128)
    print(f"✅ Performance metrics: {perf_metrics.tokens_per_second:.2f} tokens/sec")

    # Test anomaly detector
    detector = AnomalyDetector()
    for i in range(10):
        metrics = {'test_metric': 1.0 + 0.1 * np.random.randn()}
        anomalies = detector.update_metrics(metrics)

    # Add anomaly
    anomalies = detector.update_metrics({'test_metric': 10.0})
    print(f"✅ Anomaly detection: {len(anomalies)} anomalies detected")

    # Test full diagnostics
    diagnostics = TrainingDiagnostics(model, device)
    diagnostics.collect_step_metrics(step_times, 32, 128, 2.5, 1e-4)
    report = diagnostics.generate_performance_report()
    print(f"✅ Diagnostics report generated with {len(report)} sections")

    print("🎉 All basic tests passed!")