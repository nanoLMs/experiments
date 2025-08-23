#!/usr/bin/env python3
"""
Advanced Monitoring System for NanoLM Training
==============================================

Comprehensive monitoring system that tracks:
- Training metrics and loss components
- GPU memory and utilization
- System performance
- Model quality indicators
- Error detection and recovery
"""

import time
import psutil
import logging
import json
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import deque, defaultdict
import numpy as np

try:
    import torch
    import pynvml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False
    torch = None
    pynvml = None

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float

    # GPU metrics (if available)
    gpu_utilization: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_temperature: Optional[float] = None
    gpu_power_draw: Optional[float] = None


@dataclass
class TrainingMetrics:
    """Training-specific metrics"""
    timestamp: float
    step: int
    epoch: int

    # Loss components
    total_loss: float
    main_loss: float
    mtp_loss: float
    aux_loss: float
    reasoning_loss: float
    anti_hallucination_loss: float

    # Training dynamics
    learning_rate: float
    grad_norm: float
    tokens_per_second: float

    # Model-specific metrics
    expert_usage: Optional[Dict[int, float]] = None
    attention_entropy: Optional[float] = None
    activation_sparsity: Optional[float] = None


@dataclass
class QualityMetrics:
    """Model quality indicators"""
    timestamp: float
    step: int

    # Loss-based quality
    perplexity: float
    loss_variance: float
    convergence_score: float

    # Generation quality (if available)
    generation_coherence: Optional[float] = None
    generation_diversity: Optional[float] = None

    # Quantization quality
    quantization_error: Optional[float] = None
    weight_distribution: Optional[Dict[str, float]] = None


class GPUMonitor:
    """GPU monitoring using NVIDIA ML"""

    def __init__(self):
        self.available = False
        self.device_count = 0

        if NVIDIA_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                self.available = True
                logging.info(f"✅ GPU monitoring initialized ({self.device_count} devices)")
            except Exception as e:
                logging.warning(f"⚠️ GPU monitoring unavailable: {e}")

    def get_gpu_metrics(self, device_id: int = 0) -> Dict[str, Optional[float]]:
        """Get GPU metrics for specified device"""
        if not self.available or device_id >= self.device_count:
            return {
                'utilization': None,
                'memory_used_gb': None,
                'memory_total_gb': None,
                'temperature': None,
                'power_draw': None
            }

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utilization = util.gpu

            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used_gb = mem_info.used / (1024**3)
            memory_total_gb = mem_info.total / (1024**3)

            # Temperature
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temperature = None

            # Power draw
            try:
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            except:
                power_draw = None

            return {
                'utilization': utilization,
                'memory_used_gb': memory_used_gb,
                'memory_total_gb': memory_total_gb,
                'temperature': temperature,
                'power_draw': power_draw
            }

        except Exception as e:
            logging.warning(f"⚠️ Failed to get GPU metrics: {e}")
            return {
                'utilization': None,
                'memory_used_gb': None,
                'memory_total_gb': None,
                'temperature': None,
                'power_draw': None
            }


class PerformanceMonitor:
    """System performance monitoring"""

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.gpu_monitor = GPUMonitor()

        # Metrics history
        self.system_metrics: deque = deque(maxlen=history_size)
        self.training_metrics: deque = deque(maxlen=history_size)
        self.quality_metrics: deque = deque(maxlen=history_size)

        # Performance tracking
        self.start_time = time.time()
        self.last_update = time.time()

        logging.info("✅ Performance monitor initialized")

    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # GPU metrics
        gpu_metrics = self.gpu_monitor.get_gpu_metrics()

        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            memory_total_gb=memory.total / (1024**3),
            disk_usage_percent=disk.percent,
            gpu_utilization=gpu_metrics['utilization'],
            gpu_memory_used_gb=gpu_metrics['memory_used_gb'],
            gpu_memory_total_gb=gpu_metrics['memory_total_gb'],
            gpu_temperature=gpu_metrics['temperature'],
            gpu_power_draw=gpu_metrics['power_draw']
        )

    def update_training_metrics(self, metrics: TrainingMetrics):
        """Update training metrics"""
        self.training_metrics.append(metrics)
        self.last_update = time.time()

    def update_quality_metrics(self, metrics: QualityMetrics):
        """Update quality metrics"""
        self.quality_metrics.append(metrics)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # System metrics
        sys_metrics = self.get_system_metrics()

        # Training metrics
        if self.training_metrics:
            latest_training = self.training_metrics[-1]
            avg_tokens_per_sec = np.mean([m.tokens_per_second for m in list(self.training_metrics)[-10:]])
        else:
            latest_training = None
            avg_tokens_per_sec = 0.0

        # Quality metrics
        if self.quality_metrics:
            latest_quality = self.quality_metrics[-1]
        else:
            latest_quality = None

        return {
            'elapsed_time': elapsed_time,
            'system': asdict(sys_metrics),
            'training': asdict(latest_training) if latest_training else None,
            'quality': asdict(latest_quality) if latest_quality else None,
            'performance': {
                'avg_tokens_per_second': avg_tokens_per_sec,
                'steps_per_hour': len(self.training_metrics) / (elapsed_time / 3600) if elapsed_time > 0 else 0
            }
        }

    def detect_anomalies(self) -> List[str]:
        """Detect performance anomalies"""
        anomalies = []

        # Check system resources
        sys_metrics = self.get_system_metrics()

        if sys_metrics.memory_percent > 90:
            anomalies.append("High system memory usage (>90%)")

        if sys_metrics.cpu_percent > 95:
            anomalies.append("High CPU usage (>95%)")

        if sys_metrics.gpu_memory_used_gb and sys_metrics.gpu_memory_total_gb:
            gpu_usage = (sys_metrics.gpu_memory_used_gb / sys_metrics.gpu_memory_total_gb) * 100
            if gpu_usage > 95:
                anomalies.append("High GPU memory usage (>95%)")

        if sys_metrics.gpu_temperature and sys_metrics.gpu_temperature > 85:
            anomalies.append(f"High GPU temperature ({sys_metrics.gpu_temperature}°C)")

        # Check training metrics
        if len(self.training_metrics) > 10:
            recent_losses = [m.total_loss for m in list(self.training_metrics)[-10:]]
            if np.std(recent_losses) > np.mean(recent_losses) * 0.5:
                anomalies.append("High loss variance detected")

            recent_grad_norms = [m.grad_norm for m in list(self.training_metrics)[-10:]]
            if any(gn > 10.0 for gn in recent_grad_norms):
                anomalies.append("Gradient explosion detected")

            if any(gn < 1e-6 for gn in recent_grad_norms):
                anomalies.append("Vanishing gradients detected")

        return anomalies


class RichMonitorDisplay:
    """Rich-based live monitoring display"""

    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.console = Console() if RICH_AVAILABLE else None
        self.live = None
        self.running = False

        if not RICH_AVAILABLE:
            logging.warning("⚠️ Rich not available - using basic monitoring")

    def create_layout(self) -> Layout:
        """Create monitoring layout"""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )

        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )

        layout["left"].split_column(
            Layout(name="system", ratio=1),
            Layout(name="training", ratio=1)
        )

        layout["right"].split_column(
            Layout(name="quality", ratio=1),
            Layout(name="alerts", ratio=1)
        )

        return layout

    def update_display(self, layout: Layout):
        """Update display with current metrics"""
        summary = self.monitor.get_performance_summary()
        anomalies = self.monitor.detect_anomalies()

        # Header
        elapsed = summary['elapsed_time']
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)

        layout["header"].update(
            Panel(
                f"🚀 NanoLM Training Monitor - Elapsed: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}",
                style="bold blue"
            )
        )

        # System metrics
        sys_data = summary['system']
        system_table = Table(title="System Metrics")
        system_table.add_column("Metric", style="cyan")
        system_table.add_column("Value", style="green")

        system_table.add_row("CPU Usage", f"{sys_data['cpu_percent']:.1f}%")
        system_table.add_row("Memory", f"{sys_data['memory_used_gb']:.1f}GB / {sys_data['memory_total_gb']:.1f}GB ({sys_data['memory_percent']:.1f}%)")

        if sys_data['gpu_memory_used_gb']:
            gpu_usage = (sys_data['gpu_memory_used_gb'] / sys_data['gpu_memory_total_gb']) * 100
            system_table.add_row("GPU Memory", f"{sys_data['gpu_memory_used_gb']:.1f}GB / {sys_data['gpu_memory_total_gb']:.1f}GB ({gpu_usage:.1f}%)")

        if sys_data['gpu_utilization']:
            system_table.add_row("GPU Util", f"{sys_data['gpu_utilization']:.1f}%")

        if sys_data['gpu_temperature']:
            system_table.add_row("GPU Temp", f"{sys_data['gpu_temperature']:.1f}°C")

        layout["system"].update(Panel(system_table))

        # Training metrics
        if summary['training']:
            train_data = summary['training']
            training_table = Table(title="Training Metrics")
            training_table.add_column("Metric", style="cyan")
            training_table.add_column("Value", style="green")

            training_table.add_row("Step", f"{train_data['step']:,}")
            training_table.add_row("Epoch", f"{train_data['epoch']}")
            training_table.add_row("Total Loss", f"{train_data['total_loss']:.6f}")
            training_table.add_row("Main Loss", f"{train_data['main_loss']:.6f}")
            training_table.add_row("MTP Loss", f"{train_data['mtp_loss']:.6f}")
            training_table.add_row("Learning Rate", f"{train_data['learning_rate']:.2e}")
            training_table.add_row("Grad Norm", f"{train_data['grad_norm']:.4f}")
            training_table.add_row("Tokens/sec", f"{train_data['tokens_per_second']:.0f}")

            layout["training"].update(Panel(training_table))
        else:
            layout["training"].update(Panel("No training data yet"))

        # Quality metrics
        if summary['quality']:
            quality_data = summary['quality']
            quality_table = Table(title="Quality Metrics")
            quality_table.add_column("Metric", style="cyan")
            quality_table.add_column("Value", style="green")

            quality_table.add_row("Perplexity", f"{quality_data['perplexity']:.2f}")
            quality_table.add_row("Loss Variance", f"{quality_data['loss_variance']:.6f}")
            quality_table.add_row("Convergence", f"{quality_data['convergence_score']:.3f}")

            layout["quality"].update(Panel(quality_table))
        else:
            layout["quality"].update(Panel("No quality data yet"))

        # Alerts
        if anomalies:
            alert_text = "\n".join([f"⚠️ {alert}" for alert in anomalies])
            layout["alerts"].update(Panel(alert_text, title="Alerts", style="red"))
        else:
            layout["alerts"].update(Panel("✅ All systems normal", title="Status", style="green"))

        # Footer
        perf_data = summary['performance']
        layout["footer"].update(
            Panel(
                f"Performance: {perf_data['avg_tokens_per_second']:.0f} tokens/sec | "
                f"{perf_data['steps_per_hour']:.1f} steps/hour",
                style="dim"
            )
        )

    def start_live_display(self):
        """Start live monitoring display"""
        if not RICH_AVAILABLE:
            return

        layout = self.create_layout()
        self.running = True

        with Live(layout, refresh_per_second=1, screen=True) as live:
            self.live = live
            while self.running:
                try:
                    self.update_display(layout)
                    time.sleep(1)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logging.error(f"Display error: {e}")
                    time.sleep(1)

    def stop_live_display(self):
        """Stop live monitoring display"""
        self.running = False


class MonitoringSystem:
    """Complete monitoring system"""

    def __init__(self, save_dir: str = "monitoring", enable_live_display: bool = False):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        self.monitor = PerformanceMonitor()
        self.display = RichMonitorDisplay(self.monitor) if enable_live_display else None

        # Background monitoring
        self.monitoring_thread = None
        self.monitoring_active = False

        # Callbacks
        self.anomaly_callbacks: List[Callable[[List[str]], None]] = []

        logging.info(f"✅ Monitoring system initialized (save_dir: {save_dir})")

    def add_anomaly_callback(self, callback: Callable[[List[str]], None]):
        """Add callback for anomaly detection"""
        self.anomaly_callbacks.append(callback)

    def start_monitoring(self, interval: int = 60):
        """Start background monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True

        def monitor_loop():
            while self.monitoring_active:
                try:
                    # Update system metrics
                    sys_metrics = self.monitor.get_system_metrics()
                    self.monitor.system_metrics.append(sys_metrics)

                    # Check for anomalies
                    anomalies = self.monitor.detect_anomalies()
                    if anomalies:
                        for callback in self.anomaly_callbacks:
                            callback(anomalies)

                    # Save metrics periodically
                    if len(self.monitor.system_metrics) % 10 == 0:
                        self.save_metrics()

                    time.sleep(interval)

                except Exception as e:
                    logging.error(f"Monitoring error: {e}")
                    time.sleep(interval)

        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()

        logging.info(f"✅ Background monitoring started (interval: {interval}s)")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        if self.display:
            self.display.stop_live_display()

        logging.info("✅ Monitoring stopped")

    def update_training_metrics(self, step: int, epoch: int, losses: Dict[str, float],
                              lr: float, grad_norm: float, tokens_per_second: float,
                              **kwargs):
        """Update training metrics"""
        metrics = TrainingMetrics(
            timestamp=time.time(),
            step=step,
            epoch=epoch,
            total_loss=losses.get('total', 0.0),
            main_loss=losses.get('main', 0.0),
            mtp_loss=losses.get('mtp', 0.0),
            aux_loss=losses.get('aux', 0.0),
            reasoning_loss=losses.get('reasoning', 0.0),
            anti_hallucination_loss=losses.get('anti_hallucination', 0.0),
            learning_rate=lr,
            grad_norm=grad_norm,
            tokens_per_second=tokens_per_second,
            expert_usage=kwargs.get('expert_usage'),
            attention_entropy=kwargs.get('attention_entropy'),
            activation_sparsity=kwargs.get('activation_sparsity')
        )

        self.monitor.update_training_metrics(metrics)

    def update_quality_metrics(self, step: int, perplexity: float, loss_variance: float,
                             convergence_score: float, **kwargs):
        """Update quality metrics"""
        metrics = QualityMetrics(
            timestamp=time.time(),
            step=step,
            perplexity=perplexity,
            loss_variance=loss_variance,
            convergence_score=convergence_score,
            generation_coherence=kwargs.get('generation_coherence'),
            generation_diversity=kwargs.get('generation_diversity'),
            quantization_error=kwargs.get('quantization_error'),
            weight_distribution=kwargs.get('weight_distribution')
        )

        self.monitor.update_quality_metrics(metrics)

    def save_metrics(self):
        """Save metrics to disk"""
        timestamp = int(time.time())

        # Save system metrics
        if self.monitor.system_metrics:
            sys_data = [asdict(m) for m in self.monitor.system_metrics]
            with open(self.save_dir / f"system_metrics_{timestamp}.json", 'w') as f:
                json.dump(sys_data, f, indent=2)

        # Save training metrics
        if self.monitor.training_metrics:
            train_data = [asdict(m) for m in self.monitor.training_metrics]
            with open(self.save_dir / f"training_metrics_{timestamp}.json", 'w') as f:
                json.dump(train_data, f, indent=2)

        # Save quality metrics
        if self.monitor.quality_metrics:
            quality_data = [asdict(m) for m in self.monitor.quality_metrics]
            with open(self.save_dir / f"quality_metrics_{timestamp}.json", 'w') as f:
                json.dump(quality_data, f, indent=2)

        # Save summary
        summary = self.monitor.get_performance_summary()
        with open(self.save_dir / "latest_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

    def generate_report(self) -> str:
        """Generate monitoring report"""
        summary = self.monitor.get_performance_summary()
        anomalies = self.monitor.detect_anomalies()

        report = f"""
🔍 MONITORING SYSTEM REPORT
{'='*50}

⏱️ Runtime Information:
  • Elapsed Time: {summary['elapsed_time']/3600:.2f} hours
  • System Uptime: {psutil.boot_time()}

💻 System Status:
  • CPU Usage: {summary['system']['cpu_percent']:.1f}%
  • Memory Usage: {summary['system']['memory_percent']:.1f}% ({summary['system']['memory_used_gb']:.1f}GB / {summary['system']['memory_total_gb']:.1f}GB)
"""

        if summary['system']['gpu_memory_used_gb']:
            gpu_usage = (summary['system']['gpu_memory_used_gb'] / summary['system']['gpu_memory_total_gb']) * 100
            report += f"  • GPU Memory: {gpu_usage:.1f}% ({summary['system']['gpu_memory_used_gb']:.1f}GB / {summary['system']['gpu_memory_total_gb']:.1f}GB)\n"

        if summary['system']['gpu_utilization']:
            report += f"  • GPU Utilization: {summary['system']['gpu_utilization']:.1f}%\n"

        if summary['training']:
            train_data = summary['training']
            report += f"""
🚀 Training Status:
  • Current Step: {train_data['step']:,}
  • Current Epoch: {train_data['epoch']}
  • Total Loss: {train_data['total_loss']:.6f}
  • Learning Rate: {train_data['learning_rate']:.2e}
  • Tokens/Second: {train_data['tokens_per_second']:.0f}
  • Steps/Hour: {summary['performance']['steps_per_hour']:.1f}
"""

        if summary['quality']:
            quality_data = summary['quality']
            report += f"""
📊 Quality Metrics:
  • Perplexity: {quality_data['perplexity']:.2f}
  • Loss Variance: {quality_data['loss_variance']:.6f}
  • Convergence Score: {quality_data['convergence_score']:.3f}
"""

        if anomalies:
            report += f"\n⚠️ Detected Anomalies:\n"
            for anomaly in anomalies:
                report += f"  • {anomaly}\n"
        else:
            report += f"\n✅ No anomalies detected\n"

        return report


def create_monitoring_system(config) -> MonitoringSystem:
    """Create monitoring system from configuration"""
    monitoring = MonitoringSystem(
        save_dir=config.infrastructure.loss_tracking_dir,
        enable_live_display=config.infrastructure.rich_logging
    )

    # Add anomaly callback for logging
    def log_anomalies(anomalies: List[str]):
        for anomaly in anomalies:
            logging.warning(f"🚨 Anomaly detected: {anomaly}")

    monitoring.add_anomaly_callback(log_anomalies)

    # Start background monitoring if enabled
    if config.infrastructure.enable_memory_monitoring:
        monitoring.start_monitoring(interval=config.infrastructure.monitoring_interval)

    return monitoring


if __name__ == "__main__":
    # Demo usage
    print("🧪 Monitoring System Demo")

    # Create monitoring system
    monitoring = MonitoringSystem(save_dir="demo_monitoring", enable_live_display=True)

    # Start monitoring
    monitoring.start_monitoring(interval=5)

    # Simulate training updates
    import random
    for step in range(100):
        losses = {
            'total': 5.0 * np.exp(-step / 50) + random.uniform(-0.1, 0.1),
            'main': 3.0 * np.exp(-step / 50) + random.uniform(-0.05, 0.05),
            'mtp': 1.0 * np.exp(-step / 50) + random.uniform(-0.02, 0.02),
            'aux': 0.5 * np.exp(-step / 50) + random.uniform(-0.01, 0.01),
            'reasoning': 0.3 * np.exp(-step / 50) + random.uniform(-0.01, 0.01),
            'anti_hallucination': 0.2 * np.exp(-step / 50) + random.uniform(-0.01, 0.01)
        }

        monitoring.update_training_metrics(
            step=step,
            epoch=step // 20,
            losses=losses,
            lr=0.001 * (0.99 ** (step // 10)),
            grad_norm=random.uniform(0.1, 2.0),
            tokens_per_second=random.uniform(8000, 12000)
        )

        # Update quality metrics occasionally
        if step % 10 == 0:
            monitoring.update_quality_metrics(
                step=step,
                perplexity=np.exp(losses['total']),
                loss_variance=random.uniform(0.001, 0.01),
                convergence_score=min(1.0, step / 80.0)
            )

        time.sleep(0.1)

    # Generate report
    print(monitoring.generate_report())

    # Stop monitoring
    monitoring.stop_monitoring()

    print("✅ Demo completed!")