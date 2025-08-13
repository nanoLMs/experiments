#!/usr/bin/env python3
"""
Rich-based output system for beautiful console displays
"""
import time
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn,
        TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn,
        TaskProgressColumn, ProgressColumn
    )
    from rich.text import Text
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.tree import Tree
    from rich.rule import Rule
    from rich.status import Status
    from rich.prompt import Prompt, Confirm
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.align import Align
    from rich.box import ROUNDED, DOUBLE, SIMPLE, HEAVY
    from rich import print as rprint
    RICH_AVAILABLE = True

    class TokensPerSecColumn(ProgressColumn):
        """Custom column to safely display tokens per second"""

        def render(self, task):
            """Render tokens per second with safe fallback"""
            tokens_per_sec = task.fields.get('tokens_per_sec', 0)
            return Text(f"{tokens_per_sec:.0f} tok/s", style="green")

except ImportError:
    RICH_AVAILABLE = False

    class TokensPerSecColumn:
        """Fallback when Rich is not available"""
        pass
    # Fallback for when Rich is not available
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
        def log(self, *args, **kwargs):
            print(*args)

@dataclass
class TrainingMetrics:
    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    lr: float = 0.0
    tokens_per_sec: float = 0.0
    gpu_memory: float = 0.0
    cpu_usage: float = 0.0
    elapsed_time: float = 0.0
    eta: Optional[float] = None
    best_loss: float = float('inf')

class RichOutputManager:
    """Manages all Rich-based console output for the training system."""

    def __init__(self, log_file: Optional[str] = None):
        if not RICH_AVAILABLE:
            print("⚠️  Rich module not available. Install with: pip install rich")
            self.console = Console()
            self.rich_enabled = False
        else:
            self.console = Console(log_path=log_file, record=True)
            self.rich_enabled = True

        self.start_time = time.time()
        self.training_active = False

    def print_header(self, title: str, subtitle: str = ""):
        """Print a beautiful header with title and subtitle."""
        if not self.rich_enabled:
            print(f"\n{'='*60}")
            print(f"{title}")
            if subtitle:
                print(f"{subtitle}")
            print('='*60)
            return

        # Create header panel
        header_text = Text(title, style="bold cyan", justify="center")
        if subtitle:
            header_text.append(f"\n{subtitle}", style="dim white")

        panel = Panel(
            header_text,
            box=DOUBLE,
            style="cyan",
            padding=(1, 2)
        )

        self.console.print("\n")
        self.console.print(panel)
        self.console.print()

    def print_system_info(self, gpu_name: str, gpu_memory: float, model_params: int):
        """Display system information in a nice table."""
        if not self.rich_enabled:
            print(f"GPU: {gpu_name}")
            print(f"Memory: {gpu_memory:.1f}GB")
            print(f"Model: {model_params/1e6:.1f}M parameters")
            return

        table = Table(title="🖥️  System Information", box=ROUNDED)
        table.add_column("Component", style="cyan", width=15)
        table.add_column("Details", style="green")

        table.add_row("🔧 GPU", f"{gpu_name}")
        table.add_row("💾 GPU Memory", f"{gpu_memory:.1f} GB")
        table.add_row("🧠 Model Size", f"{model_params/1e6:.1f}M parameters")
        table.add_row("⚡ Precision", "FP16 + 4-bit quantization")
        table.add_row("🏗️  Architecture", "MoE + MTP + Reasoning + Anti-Hallucination")

        self.console.print(table)
        self.console.print()

    def print_config_summary(self, config_dict: Dict):
        """Display training configuration in organized panels."""
        if not self.rich_enabled:
            print("Configuration:")
            for k, v in config_dict.items():
                print(f"  {k}: {v}")
            return

        # Split config into categories
        model_config = {}
        training_config = {}
        optimization_config = {}

        for key, value in config_dict.items():
            if any(x in key.lower() for x in ['layer', 'embd', 'head', 'expert', 'vocab']):
                model_config[key] = value
            elif any(x in key.lower() for x in ['lr', 'batch', 'epoch', 'step']):
                training_config[key] = value
            else:
                optimization_config[key] = value

        # Create three panels
        panels = []

        if model_config:
            model_table = Table(box=None, show_header=False)
            model_table.add_column("Key", style="cyan")
            model_table.add_column("Value", style="white")
            for k, v in model_config.items():
                model_table.add_row(k, str(v))
            panels.append(Panel(model_table, title="🏗️  Model Architecture", border_style="blue"))

        if training_config:
            train_table = Table(box=None, show_header=False)
            train_table.add_column("Key", style="cyan")
            train_table.add_column("Value", style="white")
            for k, v in training_config.items():
                train_table.add_row(k, str(v))
            panels.append(Panel(train_table, title="🎯 Training Config", border_style="green"))

        if optimization_config:
            opt_table = Table(box=None, show_header=False)
            opt_table.add_column("Key", style="cyan")
            opt_table.add_column("Value", style="white")
            for k, v in optimization_config.items():
                opt_table.add_row(k, str(v))
            panels.append(Panel(opt_table, title="⚡ Optimization", border_style="yellow"))

        self.console.print(Columns(panels, equal=True))
        self.console.print()

    def create_training_progress(self, total_steps: int) -> Progress:
        """Create a Rich progress bar for training with combined step and epoch tracking."""
        if not self.rich_enabled:
            return None

        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=50),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("•"),
            TokensPerSecColumn(),
            console=self.console,
            refresh_per_second=4
        )

    def create_combined_progress(self, total_steps: int, total_epochs: int) -> Progress:
        """Create a combined progress display for both steps and epochs."""
        if not self.rich_enabled:
            return None

        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("•"),
            TokensPerSecColumn(),
            console=self.console,
            refresh_per_second=4
        )

    def print_training_step(self, metrics: TrainingMetrics, progress: Optional[Progress] = None, task_id = None):
        """Print training step information."""

        # ALWAYS print plain text for logs (every 10 steps to avoid spam)
        if metrics.step % 10 == 0:
            print(f"Step {metrics.step:,} | Loss: {metrics.loss:.4f} | LR: {metrics.lr:.2e} | "
                  f"Tokens/s: {metrics.tokens_per_sec:.0f} | GPU: {metrics.gpu_memory:.1f}GB")

        if not self.rich_enabled:
            return

        # Update progress bar if provided (Rich display)
        if progress and task_id is not None:
            # Update the progress bar description with current metrics
            description = f"Step {metrics.step:,} | Loss: {metrics.loss:.4f} | LR: {metrics.lr:.2e}"
            progress.update(
                task_id,
                completed=metrics.step,
                tokens_per_sec=metrics.tokens_per_sec,
                description=description
            )

        # Create detailed step info (less frequent)
        if metrics.step % 25 == 0:  # Every 25 steps (less frequent to avoid interference)
            step_table = Table(box=None, show_header=False, show_edge=False)
            step_table.add_column("Metric", style="cyan", width=12)
            step_table.add_column("Value", style="white", width=15)
            step_table.add_column("Metric", style="cyan", width=12)
            step_table.add_column("Value", style="white", width=15)

            step_table.add_row(
                "Step", f"{metrics.step:,}",
                "Loss", f"{metrics.loss:.4f}"
            )
            step_table.add_row(
                "Epoch", f"{metrics.epoch}",
                "Best Loss", f"{metrics.best_loss:.4f}"
            )
            step_table.add_row(
                "LR", f"{metrics.lr:.2e}",
                "Tokens/sec", f"{metrics.tokens_per_sec:.0f}"
            )
            step_table.add_row(
                "GPU Mem", f"{metrics.gpu_memory:.1f}GB",
                "CPU", f"{metrics.cpu_usage:.1f}%"
            )

            if metrics.eta:
                eta_str = str(timedelta(seconds=int(metrics.eta)))
                step_table.add_row("ETA", eta_str, "", "")

            panel = Panel(
                step_table,
                title=f"📊 Step {metrics.step:,}",
                border_style="green",
                padding=(0, 1)
            )

            # Only print detailed info if no progress bar is active
            if progress is None:
                self.console.print(panel)

    def print_validation_results(self, val_loss: float, val_metrics: Dict[str, float]):
        """Display validation results in a nice format."""
        if not self.rich_enabled:
            print(f"Validation - Loss: {val_loss:.4f}")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.4f}")
            return

        val_table = Table(title="🎯 Validation Results", box=ROUNDED)
        val_table.add_column("Metric", style="cyan")
        val_table.add_column("Value", style="green")
        val_table.add_column("Status", style="white")

        val_table.add_row("Loss", f"{val_loss:.4f}", "📉" if val_loss < 2.0 else "📈")

        for metric_name, value in val_metrics.items():
            status = "✅" if value > 0.8 else "⚠️" if value > 0.6 else "❌"
            val_table.add_row(metric_name, f"{value:.4f}", status)

        self.console.print(val_table)
        self.console.print()

    def print_checkpoint_info(self, checkpoint_path: str, step: int, loss: float):
        """Display checkpoint save information."""
        if not self.rich_enabled:
            print(f"💾 Checkpoint saved: {checkpoint_path} (Step {step}, Loss {loss:.4f})")
            return

        checkpoint_panel = Panel(
            f"💾 [bold green]Checkpoint Saved[/bold green]\n"
            f"📁 Path: {checkpoint_path}\n"
            f"📊 Step: {step:,}\n"
            f"📉 Loss: {loss:.4f}",
            title="Checkpoint",
            border_style="green",
            padding=(0, 1)
        )

        self.console.print(checkpoint_panel)

    def print_model_export_progress(self, formats: List[str]) -> Progress:
        """Create progress for model export."""
        if not self.rich_enabled:
            return None

        export_progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        )

        return export_progress

    def print_export_summary(self, export_sizes: Dict[str, int]):
        """Display beautiful export summary with file sizes."""
        if not self.rich_enabled:
            print("Export Summary:")
            for format_name, size_bytes in export_sizes.items():
                size_mb = size_bytes / (1024 * 1024)
                print(f"  {format_name}: {size_mb:.1f} MB")
            return

        # Create export summary table
        export_table = Table(title="📦 Model Export Summary", box=ROUNDED)
        export_table.add_column("Format", style="cyan", width=20)
        export_table.add_column("Size", style="green", width=10)
        export_table.add_column("Use Case", style="white", width=25)
        export_table.add_column("Device", style="yellow", width=15)

        format_info = {
            'pytorch_fp32': ('Full Precision', 'Research/Development', '🖥️ Workstation'),
            'pytorch_fp16': ('Production', 'Server Inference', '☁️ Cloud'),
            'pytorch_4bit': ('Compact', 'Edge Computing', '📟 Edge'),
            'torchscript_mobile': ('Mobile Ready', 'iOS/Android Apps', '📱 Mobile'),
            'quantized_mobile': ('Ultra Compact', 'Mobile Apps', '📱 Mobile'),
            'onnx_standard': ('Cross Platform', 'Any Runtime', '🌐 Universal'),
            'onnx_optimized': ('Optimized', 'Web Deployment', '🌐 Web'),
            'coreml_ios': ('iOS Native', 'iPhone/iPad', '🍎 iOS'),
            'huggingface': ('HF Compatible', 'Transformers', '🤗 HuggingFace')
        }

        # Sort by size (smallest first)
        sorted_formats = sorted(export_sizes.items(), key=lambda x: x[1])

        for format_name, size_bytes in sorted_formats:
            size_mb = size_bytes / (1024 * 1024)
            info = format_info.get(format_name, ('Unknown', 'General Use', '❓ Unknown'))

            # Color code by size
            if size_mb < 100:
                size_style = "green"
            elif size_mb < 300:
                size_style = "yellow"
            else:
                size_style = "red"

            export_table.add_row(
                format_name,
                f"[{size_style}]{size_mb:.1f} MB[/{size_style}]",
                info[1],
                info[2]
            )

        self.console.print(export_table)

        # Add deployment recommendations
        deploy_panel = Panel(
            "[bold cyan]🚀 Deployment Recommendations[/bold cyan]\n\n"
            "[green]📱 Mobile Apps[/green]: Use quantized_mobile.ptl (~40MB)\n"
            "[blue]☁️  Cloud APIs[/blue]: Use pytorch_fp16.pt (~200MB)\n"
            "[yellow]🌐 Web Browser[/yellow]: Use onnx_optimized (~175MB)\n"
            "[red]🖥️  Development[/red]: Use pytorch_fp32.pt (~400MB)",
            title="Quick Start Guide",
            border_style="cyan"
        )

        self.console.print("\n")
        self.console.print(deploy_panel)

    def print_error(self, error_msg: str, exception: Optional[Exception] = None):
        """Display errors in a prominent way."""
        if not self.rich_enabled:
            print(f"❌ ERROR: {error_msg}")
            if exception:
                print(f"   Details: {str(exception)}")
            return

        error_text = f"[bold red]❌ ERROR[/bold red]\n{error_msg}"
        if exception:
            error_text += f"\n[dim]Details: {str(exception)}[/dim]"

        error_panel = Panel(
            error_text,
            title="Error",
            border_style="red",
            padding=(1, 2)
        )

        self.console.print(error_panel)

    def print_warning(self, warning_msg: str):
        """Display warnings."""
        if not self.rich_enabled:
            print(f"⚠️  WARNING: {warning_msg}")
            return

        warning_panel = Panel(
            f"[bold yellow]⚠️  WARNING[/bold yellow]\n{warning_msg}",
            border_style="yellow",
            padding=(0, 1)
        )

        self.console.print(warning_panel)

    def print_success(self, success_msg: str, details: Optional[str] = None):
        """Display success messages."""
        if not self.rich_enabled:
            print(f"✅ {success_msg}")
            if details:
                print(f"   {details}")
            return

        success_text = f"[bold green]✅ {success_msg}[/bold green]"
        if details:
            success_text += f"\n{details}"

        success_panel = Panel(
            success_text,
            border_style="green",
            padding=(0, 1)
        )

        self.console.print(success_panel)

    def create_live_dashboard(self, metrics: TrainingMetrics) -> str:
        """Create a live updating dashboard layout."""
        if not self.rich_enabled:
            return f"Step {metrics.step} | Loss: {metrics.loss:.4f} | GPU: {metrics.gpu_memory:.1f}GB"

        # Create a layout for the dashboard
        layout = Layout()

        # Split into main sections
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )

        # Header with title and time
        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        header_text = Text("🚀 NanoLM Training Dashboard", style="bold cyan", justify="center")
        header_text.append(f"\nElapsed: {elapsed_str}", style="dim white")
        layout["header"].update(Panel(header_text, box=SIMPLE))

        # Main content - split into left and right
        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )

        # Left side - metrics
        metrics_table = Table(box=None, show_header=False)
        metrics_table.add_column("Metric", style="cyan", width=12)
        metrics_table.add_column("Value", style="white")

        metrics_table.add_row("Step", f"{metrics.step:,}")
        metrics_table.add_row("Epoch", f"{metrics.epoch}")
        metrics_table.add_row("Loss", f"{metrics.loss:.4f}")
        metrics_table.add_row("Best Loss", f"{metrics.best_loss:.4f}")
        metrics_table.add_row("Learning Rate", f"{metrics.lr:.2e}")
        metrics_table.add_row("Tokens/sec", f"{metrics.tokens_per_sec:.0f}")

        layout["left"].update(Panel(metrics_table, title="📊 Training Metrics"))

        # Right side - system info
        system_table = Table(box=None, show_header=False)
        system_table.add_column("Resource", style="cyan", width=12)
        system_table.add_column("Usage", style="white")

        system_table.add_row("GPU Memory", f"{metrics.gpu_memory:.1f} GB")
        system_table.add_row("CPU Usage", f"{metrics.cpu_usage:.1f}%")

        if metrics.eta:
            eta_str = str(timedelta(seconds=int(metrics.eta)))
            system_table.add_row("ETA", eta_str)

        layout["right"].update(Panel(system_table, title="💻 System Status"))

        # Footer with progress bar
        if metrics.step > 0:
            progress_text = f"Progress: {metrics.step:,} steps completed"
        else:
            progress_text = "Initializing training..."

        layout["footer"].update(Panel(progress_text, box=SIMPLE))

        return layout

    def ask_confirmation(self, question: str, default: bool = True) -> bool:
        """Ask for user confirmation with Rich styling."""
        if not self.rich_enabled:
            response = input(f"{question} ({'Y/n' if default else 'y/N'}): ").strip().lower()
            if not response:
                return default
            return response.startswith('y')

        return Confirm.ask(question, default=default, console=self.console)

    def save_console_output(self, filename: str):
        """Save all Rich console output to HTML file."""
        if not self.rich_enabled:
            print(f"Rich not available - cannot save output to {filename}")
            return

        try:
            self.console.save_html(filename)
            self.print_success(f"Console output saved to {filename}")
        except Exception as e:
            self.print_error(f"Failed to save console output", e)

# Global instance
rich_output = RichOutputManager()

# Convenience functions
def print_header(title: str, subtitle: str = ""):
    rich_output.print_header(title, subtitle)

def print_system_info(gpu_name: str, gpu_memory: float, model_params: int):
    rich_output.print_system_info(gpu_name, gpu_memory, model_params)

def print_config_summary(config_dict: Dict):
    rich_output.print_config_summary(config_dict)

def print_training_step(metrics: TrainingMetrics, progress=None, task_id=None):
    rich_output.print_training_step(metrics, progress, task_id)

def create_combined_progress(total_steps: int, total_epochs: int):
    return rich_output.create_combined_progress(total_steps, total_epochs)

def print_validation_results(val_loss: float, val_metrics: Dict[str, float]):
    rich_output.print_validation_results(val_loss, val_metrics)

def print_checkpoint_info(checkpoint_path: str, step: int, loss: float):
    rich_output.print_checkpoint_info(checkpoint_path, step, loss)

def print_export_summary(export_sizes: Dict[str, int]):
    rich_output.print_export_summary(export_sizes)

def print_error(error_msg: str, exception: Optional[Exception] = None):
    rich_output.print_error(error_msg, exception)

def print_warning(warning_msg: str):
    rich_output.print_warning(warning_msg)

def print_success(success_msg: str, details: Optional[str] = None):
    rich_output.print_success(success_msg, details)

# Install check
def check_rich_installation():
    """Check if Rich is installed and suggest installation if not."""
    if not RICH_AVAILABLE:
        print("\n" + "="*60)
        print("🎨 RICH MODULE NOT FOUND")
        print("="*60)
        print("For beautiful console output, install Rich:")
        print("  pip install rich")
        print("\nRich provides:")
        print("  • Beautiful progress bars")
        print("  • Colorful tables and panels")
        print("  • Live updating dashboards")
        print("  • Better error formatting")
        print("="*60)
        return False
    return True
