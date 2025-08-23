#!/usr/bin/env python3
"""
Monitoring Integration Example
=============================

Demonstrates how to integrate the comprehensive monitoring system
with the advanced trainer for real-time training diagnostics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import logging
from typing import Dict, Any

# Import our systems
from monitoring_diagnostics import TrainingDiagnostics, create_monitoring_system
from advanced_trainer import TrainingConfig


class ExampleModel(nn.Module):
    """Example model for demonstration"""

    def __init__(self, vocab_size=1000, hidden_size=256, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embedding(input_ids)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        logits = self.output(x)

        outputs = {'logits': logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            outputs['loss'] = loss

        return outputs


def create_sample_dataset(vocab_size=1000, seq_len=64, num_samples=1000):
    """Create sample dataset for demonstration"""
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = torch.randint(0, vocab_size, (num_samples, seq_len))

    dataset = TensorDataset(input_ids, labels)
    return dataset


def demonstrate_monitoring_integration():
    """Demonstrate monitoring integration with training"""
    print("🚀 Monitoring Integration Demonstration")

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    model = ExampleModel(vocab_size=1000, hidden_size=256, num_layers=2)
    model.to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create dataset
    dataset = create_sample_dataset(vocab_size=1000, seq_len=64, num_samples=500)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(f"Dataset created with {len(dataset)} samples")

    # Initialize monitoring system
    diagnostics = TrainingDiagnostics(model, device, monitoring_interval=2)
    diagnostics.start_monitoring()
    print("✅ Monitoring system started")

    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    print("\n📊 Starting monitored training...")

    try:
        for epoch in range(3):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (input_ids, labels) in enumerate(dataloader):
                # Move to device
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                # Timing
                step_start_time = time.time()

                # Forward pass
                forward_start = time.time()
                outputs = model(input_ids=input_ids, labels=labels)
                forward_time = time.time() - forward_start

                loss = outputs['loss']

                # Backward pass
                backward_start = time.time()
                loss.backward()
                backward_time = time.time() - backward_start

                # Optimizer step
                optimizer_start = time.time()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                optimizer_time = time.time() - optimizer_start

                total_time = time.time() - step_start_time

                # Collect metrics
                step_times = {
                    'forward': forward_time,
                    'backward': backward_time,
                    'optimizer': optimizer_time,
                    'total': total_time,
                    'data_loading': 0.001  # Minimal for this example
                }

                diagnostics.collect_step_metrics(
                    step_times=step_times,
                    batch_size=input_ids.shape[0],
                    sequence_length=input_ids.shape[1],
                    loss=loss.item(),
                    learning_rate=optimizer.param_groups[0]['lr']
                )

                epoch_loss += loss.item()
                num_batches += 1

                # Log progress
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.4f}, "
                          f"Time = {total_time:.3f}s")

                # Check for alerts
                recent_alerts = [a for a in diagnostics.alerts
                               if (time.time() - a.timestamp.timestamp()) < 10]
                if recent_alerts:
                    for alert in recent_alerts[-1:]:  # Show latest alert
                        print(f"🚨 {alert.level.value.upper()}: {alert.message}")
                        print(f"   💡 {alert.recommendation}")

            # Update scheduler
            scheduler.step()

            avg_loss = epoch_loss / max(1, num_batches)
            print(f"\n📈 Epoch {epoch} completed: Avg Loss = {avg_loss:.4f}")

            # Generate interim report
            if epoch == 1:  # Generate report after second epoch
                print("\n📊 Generating interim performance report...")
                report = diagnostics.generate_performance_report()

                # Display key metrics
                summary = report.get('summary', {})
                print(f"   • Total metrics collected: {summary.get('total_metrics_collected', 0)}")
                print(f"   • Monitoring duration: {summary.get('monitoring_duration_hours', 0):.2f} hours")
                print(f"   • Total alerts: {summary.get('total_alerts', 0)}")

                if 'avg_tokens_per_second' in summary:
                    print(f"   • Average throughput: {summary['avg_tokens_per_second']:.0f} tokens/sec")

                # Performance analysis
                perf_analysis = report.get('performance_analysis', {})
                if 'throughput' in perf_analysis:
                    throughput = perf_analysis['throughput']
                    print(f"   • Throughput trend: {throughput.get('trend', 'unknown')}")

                # Recommendations
                recommendations = report.get('recommendations', [])
                if recommendations:
                    print("   💡 Recommendations:")
                    for rec in recommendations[:3]:  # Show top 3
                        print(f"     - {rec}")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logging.error(f"Training error: {e}")

    finally:
        # Stop monitoring
        diagnostics.stop_monitoring()
        print("\n⏹️ Monitoring stopped")

        # Generate final report
        print("\n📋 Generating final performance report...")
        final_report = diagnostics.generate_performance_report()

        # Display final summary
        summary = final_report.get('summary', {})
        print(f"\n🎯 Final Training Summary:")
        print(f"   • Total metrics collected: {summary.get('total_metrics_collected', 0)}")
        print(f"   • Total alerts generated: {summary.get('total_alerts', 0)}")
        print(f"   • Critical alerts: {summary.get('critical_alerts', 0)}")
        print(f"   • Emergency alerts: {summary.get('emergency_alerts', 0)}")

        if 'avg_tokens_per_second' in summary:
            print(f"   • Final throughput: {summary['avg_tokens_per_second']:.0f} tokens/sec")

        # System health
        status = diagnostics.get_current_status()
        health_emoji = {"healthy": "✅", "warning": "⚠️", "critical": "🚨"}.get(
            status['system_health'], "❓"
        )
        print(f"   • System health: {health_emoji} {status['system_health']}")

        # Create visualization if possible
        try:
            diagnostics.create_visualization("./monitoring_demo_plots")
            print("   📊 Visualization plots saved to ./monitoring_demo_plots")
        except Exception as e:
            print(f"   ⚠️ Visualization failed: {e}")

        print("\n🎉 Monitoring demonstration completed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run demonstration
    demonstrate_monitoring_integration()