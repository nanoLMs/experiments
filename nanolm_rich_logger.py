#!/usr/bin/env python3
"""
NanoLM Rich Logging System
==========================

Comprehensive logging system with Rich styling optimized for:
- RTX 3060 TI (8GB VRAM)
- 32GB DDR4 RAM
- Core i5 13600K
- 1-bit/1.58-bit LLM training (BitNet architecture)
- Multi-platform export monitoring
- Real-time performance tracking

Based on research papers:
- BitNet b1.58: Ternary quantization {-1, 0, 1}
- T-MAC: CPU optimization for edge deployment
- BitVLA: Vision-Language-Action integration
- HRM: Hierarchical Reasoning Module
"""

import time
import psutil
import GPUtil
import torch
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
import queue
import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn, 
    MofNCompleteColumn, SpinnerColumn, TimeElapsedColumn
)
from rich.layout import Layout
from rich.live import Live
from rich.tree import Tree
from rich.text import Text
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback
from rich.status import Status
from rich.columns import Columns
from rich.align import Align
from rich.box import ROUNDED, HEAVY, DOUBLE
from rich.style import Style
from rich.theme import Theme
import logging

# Install rich traceback handler
install_rich_traceback()

# Custom theme for NanoLM
NANOLM_THEME = Theme({
    "nanolm.header": "bold cyan",
    "nanolm.success": "bold green",
    "nanolm.warning": "bold yellow",
    "nanolm.error": "bold red",
    "nanolm.info": "bold blue",
    "nanolm.memory": "magenta",
    "nanolm.gpu": "bright_green",
    "nanolm.loss": "bright_yellow",
    "nanolm.tokenizer": "bright_cyan",
    "nanolm.export": "bright_magenta",
    "nanolm.quantization": "gold1",
    "nanolm.hrm": "steel_blue1",
    "nanolm.mtp": "deep_pink2",
    "nanolm.moe": "chartreuse2"
})

@dataclass
class SystemSpecs:
    """RTX 3060 TI system specifications"""
    gpu_name: str = "RTX 3060 TI"
    gpu_memory_gb: float = 8.0
    cpu_name: str = "Core i5 13600K"
    ram_gb: float = 32.0
    max_vram_usage: float = 0.85  # 85% of 8GB = 6.8GB
    max_ram_usage: float = 0.80   # 80% of 32GB = 25.6GB


@dataclass
class TrainingMetrics:
    """Real-time training metrics"""
    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 8.0
    ram_used: float = 0.0
    ram_total: float = 32.0
    throughput: float = 0.0  # tokens/sec
    eta: str = "Unknown"
    
    # BitNet specific metrics
    quantization_bits: str = "1.58-bit"  # {-1, 0, 1}
    quantization_accuracy: float = 0.0
    
    # Component metrics
    hrm_convergence: float = 0.0
    mtp_accuracy: float = 0.0
    moe_router_efficiency: float = 0.0
    
    # Export metrics
    export_formats_ready: int = 0
    total_export_formats: int = 6  # torchscript, onnx, coreml, tflite, etc.


class NanoLMRichLogger:
    """Enhanced Rich logging system for NanoLM training"""
    
    def __init__(self, log_file: Optional[str] = None, console_width: int = 120):
        self.console = Console(theme=NANOLM_THEME, width=console_width)
        self.specs = SystemSpecs()
        self.metrics = TrainingMetrics()
        self.start_time = time.time()
        
        # Setup file logging
        if log_file:
            self.setup_file_logging(log_file)
        
        # Performance monitoring
        self.performance_queue = queue.Queue()
        self.monitoring_thread = None
        self.monitoring_active = False
        
        # Layout for live display
        self.layout = Layout()
        self.setup_layout()
        
        # Progress tracking
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console
        )
        
    def setup_file_logging(self, log_file: str):
        """Setup file logging with Rich handler"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                RichHandler(console=self.console, markup=True),
                logging.FileHandler(log_file)
            ]
        )
        
    def setup_layout(self):
        """Setup Rich layout for live monitoring"""
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=5)
        )
        
        self.layout["main"].split_row(
            Layout(name="metrics", ratio=1),
            Layout(name="progress", ratio=2)
        )
        
    def show_startup_banner(self):
        """Display NanoLM startup banner"""
        banner = Panel.fit(
            f"""[nanolm.header]🚀 NanoLM Training System[/]
[nanolm.info]Based on BitNet b1.58 Architecture[/]

[nanolm.gpu]GPU:[/] {self.specs.gpu_name} ({self.specs.gpu_memory_gb}GB VRAM)
[nanolm.info]CPU:[/] {self.specs.cpu_name}
[nanolm.memory]RAM:[/] {self.specs.ram_gb}GB DDR4

[nanolm.quantization]Quantization:[/] 1.58-bit ternary {{-1, 0, 1}}
[nanolm.export]Export Targets:[/] All devices (Mobile, Edge, Cloud, Web)
[nanolm.tokenizer]Tokenizer:[/] GPU-accelerated BPE (32K vocab)""",
            title="[nanolm.header]NanoLM v2.0[/]",
            border_style="nanolm.header",
            box=DOUBLE
        )
        
        self.console.print(banner)
        self.console.print()
        
    def log_step(self, step: int, loss: float, lr: float, **kwargs):
        """Log training step with Rich formatting"""
        self.metrics.step = step
        self.metrics.loss = loss
        self.metrics.learning_rate = lr
        
        # Update additional metrics
        for key, value in kwargs.items():
            if hasattr(self.metrics, key):
                setattr(self.metrics, key, value)
        
        # Update system metrics
        self._update_system_metrics()
        
        # Create metrics table
        metrics_table = Table(show_header=False, box=None, padding=(0, 1))
        metrics_table.add_column("Metric", style="bold")
        metrics_table.add_column("Value")
        
        metrics_table.add_row("Step", f"[nanolm.info]{step:,}[/]")
        metrics_table.add_row("Loss", f"[nanolm.loss]{loss:.6f}[/]")
        metrics_table.add_row("LR", f"[nanolm.info]{lr:.2e}[/]")
        metrics_table.add_row("GPU", f"[nanolm.gpu]{self.metrics.gpu_memory_used:.1f}GB/{self.specs.gpu_memory_gb}GB[/]")
        metrics_table.add_row("RAM", f"[nanolm.memory]{self.metrics.ram_used:.1f}GB/{self.specs.ram_gb}GB[/]")
        
        if hasattr(self.metrics, 'throughput') and self.metrics.throughput > 0:
            metrics_table.add_row("Speed", f"[nanolm.info]{self.metrics.throughput:.1f} tok/s[/]")
            
        self.console.print(Panel(
            metrics_table,
            title=f"[nanolm.header]Step {step}[/]",
            border_style="nanolm.info"
        ))
        
    def log_quantization_metrics(self, accuracy: float, memory_saved: float):
        """Log BitNet quantization metrics"""
        self.metrics.quantization_accuracy = accuracy
        
        quant_panel = Panel(
            f"""[nanolm.quantization]BitNet 1.58-bit Quantization Status[/]

[nanolm.success]✓[/] Ternary weights: {{-1, 0, 1}}
[nanolm.info]Accuracy retention:[/] {accuracy:.2%}
[nanolm.memory]Memory saved:[/] {memory_saved:.1f}GB
[nanolm.gpu]VRAM efficiency:[/] {self.metrics.gpu_memory_used/self.specs.gpu_memory_gb:.1%}""",
            title="[nanolm.quantization]Quantization[/]",
            border_style="nanolm.quantization"
        )
        
        self.console.print(quant_panel)
        
    def log_component_status(self, hrm_conv: float, mtp_acc: float, moe_eff: float):
        """Log advanced component metrics"""
        self.metrics.hrm_convergence = hrm_conv
        self.metrics.mtp_accuracy = mtp_acc
        self.metrics.moe_router_efficiency = moe_eff
        
        components_table = Table(title="[nanolm.header]Advanced Components[/]", box=ROUNDED)
        components_table.add_column("Component", style="bold")
        components_table.add_column("Status", justify="center")
        components_table.add_column("Metric", justify="right")
        
        components_table.add_row(
            "[nanolm.hrm]HRM (Hierarchical Reasoning)[/]",
            "🧠" if hrm_conv > 0.8 else "⚡",
            f"[nanolm.hrm]{hrm_conv:.1%}[/]"
        )
        
        components_table.add_row(
            "[nanolm.mtp]MTP (Multi-Token Prediction)[/]",
            "🎯" if mtp_acc > 0.7 else "📈",
            f"[nanolm.mtp]{mtp_acc:.1%}[/]"
        )
        
        components_table.add_row(
            "[nanolm.moe]MoE (Mixture of Experts)[/]",
            "⚡" if moe_eff > 0.85 else "🔄",
            f"[nanolm.moe]{moe_eff:.1%}[/]"
        )
        
        self.console.print(components_table)
        
    def log_export_progress(self, completed_formats: List[str]):
        """Log multi-platform export progress"""
        self.metrics.export_formats_ready = len(completed_formats)
        
        export_tree = Tree("[nanolm.export]🚀 Multi-Platform Export[/]")
        
        all_formats = [
            ("TorchScript", "📱 Mobile/Edge"),
            ("ONNX", "☁️ Cloud/Server"),
            ("CoreML", "🍎 iOS/macOS"),
            ("TensorFlow Lite", "📱 Android/Mobile"),
            ("OpenVINO", "💻 Intel CPUs"),
            ("HuggingFace", "🤗 Ecosystem")
        ]
        
        for format_name, target in all_formats:
            status = "✅" if format_name.lower() in [f.lower() for f in completed_formats] else "⏳"
            export_tree.add(f"{status} {format_name} → {target}")
            
        self.console.print(Panel(
            export_tree,
            title="[nanolm.export]Export Status[/]",
            border_style="nanolm.export"
        ))
        
    def log_tokenizer_analysis(self, vocab_size: int, tokens_processed: int, efficiency: float):
        """Log tokenizer performance"""
        tokenizer_panel = Panel(
            f"""[nanolm.tokenizer]🔤 GPU-Accelerated Tokenizer[/]

[nanolm.info]Vocabulary size:[/] {vocab_size:,} tokens
[nanolm.info]Tokens processed:[/] {tokens_processed:,}
[nanolm.gpu]GPU efficiency:[/] {efficiency:.1%}
[nanolm.success]Special tokens:[/] Legal domain optimized
[nanolm.info]Algorithm:[/] BPE with multimodal support""",
            title="[nanolm.tokenizer]Tokenization[/]",
            border_style="nanolm.tokenizer"
        )
        
        self.console.print(tokenizer_panel)
        
    def log_memory_optimization(self, optimization_type: str, memory_saved: float):
        """Log memory optimization actions"""
        optimization_styles = {
            "gradient_accumulation": "nanolm.info",
            "checkpointing": "nanolm.memory",
            "mixed_precision": "nanolm.quantization",
            "cache_cleanup": "nanolm.warning"
        }
        
        style = optimization_styles.get(optimization_type, "nanolm.info")
        
        self.console.print(Panel(
            f"[{style}]💾 Memory Optimization: {optimization_type.replace('_', ' ').title()}[/]\n"
            f"[nanolm.success]Saved:[/] {memory_saved:.1f}GB VRAM\n"
            f"[nanolm.gpu]Current usage:[/] {self.metrics.gpu_memory_used:.1f}GB / {self.specs.gpu_memory_gb}GB",
            border_style=style
        ))
        
    @contextmanager
    def training_session(self, total_steps: int):
        """Context manager for training session with live monitoring"""
        with self.progress:
            task = self.progress.add_task(
                "[nanolm.info]Training NanoLM...", 
                total=total_steps
            )
            
            try:
                self.start_monitoring()
                yield task
            finally:
                self.stop_monitoring()
                
    def update_progress(self, task_id, advance: int = 1, **kwargs):
        """Update progress bar"""
        self.progress.update(task_id, advance=advance, **kwargs)
        
    def start_monitoring(self):
        """Start system monitoring thread"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_system)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
            
    def _monitor_system(self):
        """Background system monitoring"""
        while self.monitoring_active:
            try:
                self._update_system_metrics()
                
                # Check memory thresholds
                if self.metrics.gpu_memory_used / self.specs.gpu_memory_gb > self.specs.max_vram_usage:
                    self.console.print("[nanolm.warning]⚠️ High VRAM usage detected![/]")
                    
                if self.metrics.ram_used / self.specs.ram_gb > self.specs.max_ram_usage:
                    self.console.print("[nanolm.warning]⚠️ High RAM usage detected![/]")
                    
                time.sleep(5)  # Monitor every 5 seconds
            except Exception as e:
                pass  # Silently handle monitoring errors
                
    def _update_system_metrics(self):
        """Update system metrics"""
        try:
            # GPU metrics
            if torch.cuda.is_available():
                self.metrics.gpu_memory_used = torch.cuda.memory_allocated() / 1e9
                self.metrics.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # RAM metrics
            memory = psutil.virtual_memory()
            self.metrics.ram_used = memory.used / 1e9
            self.metrics.ram_total = memory.total / 1e9
            
        except Exception:
            pass  # Handle gracefully if monitoring fails
            
    def log_training_complete(self, final_loss: float, training_time: float):
        """Log training completion"""
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        
        completion_panel = Panel(
            f"""[nanolm.success]🎉 NanoLM Training Complete![/]

[nanolm.info]Final loss:[/] {final_loss:.6f}
[nanolm.info]Training time:[/] {hours}h {minutes}m
[nanolm.quantization]Model size:[/] ~30M parameters (1.58-bit)
[nanolm.export]Export formats:[/] {self.metrics.export_formats_ready}/{self.metrics.total_export_formats} ready
[nanolm.gpu]Peak VRAM:[/] {self.metrics.gpu_memory_used:.1f}GB / {self.specs.gpu_memory_gb}GB

[nanolm.success]✅ Ready for deployment on all target devices[/]""",
            title="[nanolm.header]Training Complete[/]",
            border_style="nanolm.success",
            box=DOUBLE
        )
        
        self.console.print(completion_panel)
        
    def log_error(self, error: Exception, context: str = ""):
        """Log errors with Rich formatting"""
        error_panel = Panel(
            f"[nanolm.error]❌ Error in {context}[/]\n\n"
            f"[nanolm.error]{type(error).__name__}: {str(error)}[/]",
            title="[nanolm.error]Error[/]",
            border_style="nanolm.error"
        )
        
        self.console.print(error_panel)
        
    def create_summary_table(self) -> Table:
        """Create comprehensive training summary"""
        table = Table(title="[nanolm.header]NanoLM Training Summary[/]", box=HEAVY)
        table.add_column("Component", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Performance", justify="right")
        
        table.add_row(
            "BitNet Quantization",
            "✅" if self.metrics.quantization_accuracy > 0.9 else "⚠️",
            f"{self.metrics.quantization_accuracy:.1%}"
        )
        
        table.add_row(
            "HRM Convergence",
            "✅" if self.metrics.hrm_convergence > 0.8 else "⚠️",
            f"{self.metrics.hrm_convergence:.1%}"
        )
        
        table.add_row(
            "MTP Accuracy",
            "✅" if self.metrics.mtp_accuracy > 0.7 else "⚠️",
            f"{self.metrics.mtp_accuracy:.1%}"
        )
        
        table.add_row(
            "MoE Efficiency",
            "✅" if self.metrics.moe_router_efficiency > 0.85 else "⚠️",
            f"{self.metrics.moe_router_efficiency:.1%}"
        )
        
        table.add_row(
            "Export Readiness",
            "✅" if self.metrics.export_formats_ready >= 4 else "⚠️",
            f"{self.metrics.export_formats_ready}/{self.metrics.total_export_formats}"
        )
        
        return table


# Global logger instance
logger = None

def get_nanolm_logger(log_file: Optional[str] = None) -> NanoLMRichLogger:
    """Get or create the global NanoLM logger"""
    global logger
    if logger is None:
        logger = NanoLMRichLogger(log_file)
    return logger

def log_step(step: int, loss: float, lr: float, **kwargs):
    """Convenience function for logging training steps"""
    get_nanolm_logger().log_step(step, loss, lr, **kwargs)

def log_quantization(accuracy: float, memory_saved: float):
    """Convenience function for logging quantization metrics"""
    get_nanolm_logger().log_quantization_metrics(accuracy, memory_saved)

def log_components(hrm_conv: float, mtp_acc: float, moe_eff: float):
    """Convenience function for logging component metrics"""
    get_nanolm_logger().log_component_status(hrm_conv, mtp_acc, moe_eff)

def log_export(completed_formats: List[str]):
    """Convenience function for logging export progress"""
    get_nanolm_logger().log_export_progress(completed_formats)

def log_tokenizer(vocab_size: int, tokens_processed: int, efficiency: float):
    """Convenience function for logging tokenizer metrics"""
    get_nanolm_logger().log_tokenizer_analysis(vocab_size, tokens_processed, efficiency)


if __name__ == "__main__":
    # Demo of the logging system
    logger = NanoLMRichLogger("nanolm_training.log")
    logger.show_startup_banner()
    
    # Simulate training progress
    with logger.training_session(1000) as task:
        for step in range(1, 11):
            time.sleep(0.1)  # Simulate training time
            
            # Simulate metrics
            loss = 4.0 - (step * 0.3)
            lr = 1e-4
            
            logger.log_step(step, loss, lr)
            logger.update_progress(task, 100)
            
            if step == 5:
                logger.log_quantization_metrics(0.95, 2.5)
                logger.log_component_status(0.85, 0.72, 0.88)
                
            if step == 8:
                logger.log_export_progress(["torchscript", "onnx", "coreml"])
                
    logger.log_training_complete(1.2, 3600)
    logger.console.print(logger.create_summary_table())