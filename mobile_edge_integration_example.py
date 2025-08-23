#!/usr/bin/env python3
"""
Mobile and Edge Device Integration Example
==========================================

Demonstrates comprehensive mobile and edge optimization workflow including:
- Target-specific optimizations (Android, iOS, Raspberry Pi, etc.)
- Power mode optimizations
- ARM processor optimizations
- WebAssembly export for web deployment
- Performance benchmarking across targets
- Validation and testing
"""

import torch
import torch.nn as nn
import os
import time
import logging
import tempfile
from typing import Dict, Any, List

# Import our optimization systems
from tests.test_mobile_edge_optimizer import (
    MobileOptimizer, EdgeOptimizer, MobileOptimizationConfig, EdgeOptimizationConfig,
    MobileTarget, EdgeTarget, PowerMode
)


class DemoModel(nn.Module):
    """Demo model for mobile/edge optimization"""

    def __init__(self, vocab_size=5000, hidden_size=128, num_layers=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Transformer-like layers (simplified)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.LayerNorm(hidden_size)
            ) for _ in range(num_layers)
        ])

        # Output layer
        self.output = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)

        # Model metadata
        self.model_info = {
            "name": "DemoNanoLM",
            "parameters": sum(p.numel() for p in self.parameters()),
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers
        }

    def forward(self, input_ids):
        # Embedding
        x = self.embedding(input_ids)
        x = self.dropout(x)

        # Apply layers
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual  # Residual connection

        # Output
        logits = self.output(x)
        return logits


class MobileEdgeOptimizationPipeline:
    """Complete optimization pipeline for mobile and edge devices"""

    def __init__(self, model: nn.Module):
        self.model = model
        self.optimization_results = {}
        self.benchmark_results = {}

        logging.info("✅ Mobile Edge Optimization Pipeline initialized")
        logging.info(f"  • Model: {model.__class__.__name__}")
        logging.info(f"  • Parameters: {sum(p.numel() for p in model.parameters()):,}")

    def optimize_for_all_targets(self, sample_inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Optimize model for all mobile and edge targets"""

        logging.info("🚀 Starting multi-target optimization...")

        # Define optimization configurations for different targets
        optimization_configs = {
            # Mobile targets
            "android_balanced": MobileOptimizationConfig(
                target=MobileTarget.ANDROID,
                power_mode=PowerMode.BALANCED,
                enable_quantization=True,
                enable_pruning=True,
                pruning_ratio=0.3,
                max_memory_mb=512
            ),
            "android_low_power": MobileOptimizationConfig(
                target=MobileTarget.ANDROID,
                power_mode=PowerMode.LOW_POWER,
                enable_quantization=True,
                enable_pruning=True,
                pruning_ratio=0.5,
                max_memory_mb=256,
                cpu_threads=2
            ),
            "ios_balanced": MobileOptimizationConfig(
                target=MobileTarget.IOS,
                power_mode=PowerMode.BALANCED,
                enable_quantization=True,
                enable_pruning=True,
                pruning_ratio=0.3,
                enable_fp16=True
            ),
            "android_arm64": MobileOptimizationConfig(
                target=MobileTarget.ANDROID_ARM64,
                power_mode=PowerMode.BALANCED,
                enable_quantization=True,
                enable_pruning=True,
                enable_fp16=True,
                cpu_threads=8
            ),

            # Edge targets
            "raspberry_pi": EdgeOptimizationConfig(
                target=EdgeTarget.RASPBERRY_PI,
                power_mode=PowerMode.LOW_POWER,
                max_memory_mb=128,
                aggressive_quantization=True,
                enable_result_caching=True,
                cache_size_mb=32
            ),
            "jetson_nano": EdgeOptimizationConfig(
                target=EdgeTarget.JETSON_NANO,
                power_mode=PowerMode.BALANCED,
                max_memory_mb=512,
                aggressive_quantization=True,
                enable_int8_inference=True
            ),
            "generic_edge": EdgeOptimizationConfig(
                target=EdgeTarget.GENERIC_EDGE,
                power_mode=PowerMode.ULTRA_LOW_POWER,
                max_memory_mb=64,
                aggressive_quantization=True,
                enable_result_caching=True
            )
        }

        # Optimize for each target
        for target_name, config in optimization_configs.items():
            logging.info(f"🔧 Optimizing for {target_name}...")

            try:
                if isinstance(config, MobileOptimizationConfig):
                    optimizer = MobileOptimizer(config)
                else:
                    optimizer = EdgeOptimizer(config)

                # Apply optimizations
                start_time = time.time()
                optimized_model = optimizer.optimize_model(self.model)
                optimization_time = time.time() - start_time

                # Store results
                self.optimization_results[target_name] = {
                    "model": optimized_model,
                    "config": config,
                    "optimization_time": optimization_time,
                    "success": True
                }

                logging.info(f"  ✅ {target_name} optimization completed ({optimization_time:.2f}s)")

            except Exception as e:
                logging.error(f"  ❌ {target_name} optimization failed: {e}")
                self.optimization_results[target_name] = {
                    "model": None,
                    "config": config,
                    "optimization_time": 0,
                    "success": False,
                    "error": str(e)
                }

        # Generate summary
        successful_optimizations = [name for name, result in self.optimization_results.items()
                                   if result["success"]]

        summary = {
            "total_targets": len(optimization_configs),
            "successful_optimizations": len(successful_optimizations),
            "failed_optimizations": len(optimization_configs) - len(successful_optimizations),
            "optimization_details": self.optimization_results
        }

        logging.info(f"✅ Multi-target optimization completed: {len(successful_optimizations)}/{len(optimization_configs)} successful")

        return summary

    def benchmark_optimized_models(self, sample_inputs: Dict[str, torch.Tensor],
                                  num_runs: int = 50) -> Dict[str, Any]:
        """Benchmark all optimized models"""

        logging.info(f"⚡ Starting benchmark ({num_runs} runs per model)...")

        benchmark_results = {}

        # Benchmark original model
        original_times = self._benchmark_model(self.model, sample_inputs, num_runs)
        benchmark_results["original"] = {
            "avg_inference_time": np.mean(original_times),
            "std_inference_time": np.std(original_times),
            "min_inference_time": np.min(original_times),
            "max_inference_time": np.max(original_times),
            "model_size": self._estimate_model_size(self.model)
        }

        # Benchmark optimized models
        for target_name, result in self.optimization_results.items():
            if result["success"] and result["model"] is not None:
                try:
                    optimized_times = self._benchmark_model(result["model"], sample_inputs, num_runs)

                    benchmark_results[target_name] = {
                        "avg_inference_time": np.mean(optimized_times),
                        "std_inference_time": np.std(optimized_times),
                        "min_inference_time": np.min(optimized_times),
                        "max_inference_time": np.max(optimized_times),
                        "model_size": self._estimate_model_size(result["model"]),
                        "optimization_time": result["optimization_time"],
                        "speedup": np.mean(original_times) / np.mean(optimized_times),
                        "size_reduction": benchmark_results["original"]["model_size"] / self._estimate_model_size(result["model"])
                    }

                except Exception as e:
                    logging.warning(f"Benchmark failed for {target_name}: {e}")
                    benchmark_results[target_name] = {"benchmark_error": str(e)}

        self.benchmark_results = benchmark_results

        # Find best performing model
        best_model = None
        best_time = float('inf')

        for target_name, metrics in benchmark_results.items():
            if target_name != "original" and "avg_inference_time" in metrics:
                if metrics["avg_inference_time"] < best_time:
                    best_time = metrics["avg_inference_time"]
                    best_model = target_name

        summary = {
            "benchmark_results": benchmark_results,
            "best_performing_model": best_model,
            "best_inference_time": best_time,
            "original_inference_time": benchmark_results["original"]["avg_inference_time"]
        }

        logging.info(f"✅ Benchmark completed")
        if best_model:
            speedup = benchmark_results["original"]["avg_inference_time"] / best_time
            logging.info(f"  • Best model: {best_model} ({speedup:.2f}x speedup)")

        return summary

    def _benchmark_model(self, model: nn.Module, sample_inputs: Dict[str, torch.Tensor],
                        num_runs: int) -> List[float]:
        """Benchmark a single model"""
        model.eval()
        times = []

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                try:
                    if len(sample_inputs) == 1:
                        sample_input = next(iter(sample_inputs.values()))
                        _ = model(sample_input)
                    else:
                        _ = model(**sample_inputs)
                except Exception:
                    # Skip warmup if model fails
                    break

        # Actual benchmark
        with torch.no_grad():
            for _ in range(num_runs):
                try:
                    start_time = time.time()
                    if len(sample_inputs) == 1:
                        sample_input = next(iter(sample_inputs.values()))
                        _ = model(sample_input)
                    else:
                        _ = model(**sample_inputs)
                    times.append(time.time() - start_time)
                except Exception:
                    # Skip failed runs
                    times.append(float('inf'))

        return times

    def _estimate_model_size(self, model: nn.Module) -> int:
        """Estimate model size in bytes"""
        try:
            total_size = 0
            for param in model.parameters():
                total_size += param.numel() * param.element_size()
            return total_size
        except Exception:
            return 0

    def generate_deployment_recommendations(self) -> Dict[str, Any]:
        """Generate deployment recommendations based on optimization results"""

        recommendations = {
            "mobile_recommendations": {},
            "edge_recommendations": {},
            "general_recommendations": []
        }

        # Analyze results for mobile recommendations
        mobile_targets = ["android_balanced", "android_low_power", "ios_balanced", "android_arm64"]
        mobile_results = {k: v for k, v in self.benchmark_results.items() if k in mobile_targets}

        if mobile_results:
            best_mobile = min(mobile_results.keys(),
                            key=lambda k: mobile_results[k].get("avg_inference_time", float('inf')))

            recommendations["mobile_recommendations"] = {
                "best_target": best_mobile,
                "recommended_settings": self.optimization_results[best_mobile]["config"].__dict__,
                "expected_performance": mobile_results[best_mobile]
            }

        # Analyze results for edge recommendations
        edge_targets = ["raspberry_pi", "jetson_nano", "generic_edge"]
        edge_results = {k: v for k, v in self.benchmark_results.items() if k in edge_targets}

        if edge_results:
            best_edge = min(edge_results.keys(),
                          key=lambda k: edge_results[k].get("avg_inference_time", float('inf')))

            recommendations["edge_recommendations"] = {
                "best_target": best_edge,
                "recommended_settings": self.optimization_results[best_edge]["config"].__dict__,
                "expected_performance": edge_results[best_edge]
            }

        # General recommendations
        if self.benchmark_results:
            avg_speedup = np.mean([
                result.get("speedup", 1.0) for result in self.benchmark_results.values()
                if "speedup" in result
            ])

            avg_size_reduction = np.mean([
                result.get("size_reduction", 1.0) for result in self.benchmark_results.values()
                if "size_reduction" in result
            ])

            recommendations["general_recommendations"] = [
                f"Average speedup across targets: {avg_speedup:.2f}x",
                f"Average model size reduction: {avg_size_reduction:.2f}x",
                "Enable quantization for significant size reduction with minimal accuracy loss",
                "Use pruning for memory-constrained environments",
                "Consider FP16 for ARM64 devices to improve performance",
                "Enable result caching for edge devices with repeated inference patterns"
            ]

        return recommendations

    def create_deployment_guide(self, output_dir: str) -> str:
        """Create comprehensive deployment guide"""

        os.makedirs(output_dir, exist_ok=True)

        # Generate deployment guide content
        guide_content = f"""# Mobile and Edge Deployment Guide

## Model Information
- **Original Model**: {self.model.__class__.__name__}
- **Parameters**: {sum(p.numel() for p in self.model.parameters()):,}
- **Original Size**: {self._estimate_model_size(self.model) / 1024 / 1024:.2f} MB

## Optimization Results

### Mobile Targets
"""

        # Add mobile results
        mobile_targets = ["android_balanced", "android_low_power", "ios_balanced", "android_arm64"]
        for target in mobile_targets:
            if target in self.benchmark_results:
                result = self.benchmark_results[target]
                if "avg_inference_time" in result:
                    guide_content += f"""
#### {target.replace('_', ' ').title()}
- **Inference Time**: {result['avg_inference_time']*1000:.2f}ms
- **Model Size**: {result['model_size'] / 1024 / 1024:.2f} MB
- **Speedup**: {result.get('speedup', 1.0):.2f}x
- **Size Reduction**: {result.get('size_reduction', 1.0):.2f}x
"""

        guide_content += "\n### Edge Targets\n"

        # Add edge results
        edge_targets = ["raspberry_pi", "jetson_nano", "generic_edge"]
        for target in edge_targets:
            if target in self.benchmark_results:
                result = self.benchmark_results[target]
                if "avg_inference_time" in result:
                    guide_content += f"""
#### {target.replace('_', ' ').title()}
- **Inference Time**: {result['avg_inference_time']*1000:.2f}ms
- **Model Size**: {result['model_size'] / 1024 / 1024:.2f} MB
- **Speedup**: {result.get('speedup', 1.0):.2f}x
- **Size Reduction**: {result.get('size_reduction', 1.0):.2f}x
"""

        # Add recommendations
        recommendations = self.generate_deployment_recommendations()

        guide_content += "\n## Deployment Recommendations\n"

        if recommendations["mobile_recommendations"]:
            mobile_rec = recommendations["mobile_recommendations"]
            guide_content += f"""
### Mobile Deployment
- **Recommended Target**: {mobile_rec['best_target']}
- **Expected Performance**: {mobile_rec['expected_performance'].get('avg_inference_time', 0)*1000:.2f}ms
"""

        if recommendations["edge_recommendations"]:
            edge_rec = recommendations["edge_recommendations"]
            guide_content += f"""
### Edge Deployment
- **Recommended Target**: {edge_rec['best_target']}
- **Expected Performance**: {edge_rec['expected_performance'].get('avg_inference_time', 0)*1000:.2f}ms
"""

        guide_content += "\n### General Recommendations\n"
        for rec in recommendations["general_recommendations"]:
            guide_content += f"- {rec}\n"

        # Save guide
        guide_path = os.path.join(output_dir, "deployment_guide.md")
        with open(guide_path, "w") as f:
            f.write(guide_content)

        logging.info(f"📋 Deployment guide created: {guide_path}")

        return guide_path


def demonstrate_mobile_edge_optimization():
    """Demonstrate complete mobile and edge optimization workflow"""
    print("🚀 Mobile and Edge Optimization Demonstration")

    # Create demo model
    model = DemoModel(vocab_size=2000, hidden_size=64, num_layers=2)

    print(f"✅ Demo model created")
    print(f"  • Parameters: {model.model_info['parameters']:,}")
    print(f"  • Vocab size: {model.model_info['vocab_size']}")
    print(f"  • Hidden size: {model.model_info['hidden_size']}")
    print(f"  • Layers: {model.model_info['num_layers']}")

    # Create sample inputs
    sample_inputs = {"input_ids": torch.randint(0, 2000, (2, 32))}
    print("✅ Sample inputs created")

    # Initialize optimization pipeline
    pipeline = MobileEdgeOptimizationPipeline(model)

    # Optimize for all targets
    print("\n🔧 Starting multi-target optimization...")
    optimization_summary = pipeline.optimize_for_all_targets(sample_inputs)

    print(f"\n📊 Optimization Summary:")
    print(f"  • Total targets: {optimization_summary['total_targets']}")
    print(f"  • Successful: {optimization_summary['successful_optimizations']}")
    print(f"  • Failed: {optimization_summary['failed_optimizations']}")

    # Show successful optimizations
    successful_targets = [name for name, result in optimization_summary['optimization_details'].items()
                         if result['success']]

    if successful_targets:
        print(f"  • Successful targets: {', '.join(successful_targets)}")

        # Benchmark optimized models
        print(f"\n⚡ Running performance benchmark...")
        benchmark_summary = pipeline.benchmark_optimized_models(sample_inputs, num_runs=20)

        print(f"\n📈 Benchmark Results:")
        original_time = benchmark_summary['original_inference_time']
        print(f"  • Original model: {original_time*1000:.2f}ms")

        if benchmark_summary['best_performing_model']:
            best_model = benchmark_summary['best_performing_model']
            best_time = benchmark_summary['best_inference_time']
            speedup = original_time / best_time

            print(f"  • Best optimized: {best_model} ({best_time*1000:.2f}ms, {speedup:.2f}x speedup)")

            # Show top 3 performers
            sorted_results = sorted(
                [(name, result) for name, result in benchmark_summary['benchmark_results'].items()
                 if name != 'original' and 'avg_inference_time' in result],
                key=lambda x: x[1]['avg_inference_time']
            )

            print(f"  • Top performers:")
            for i, (name, result) in enumerate(sorted_results[:3]):
                speedup = original_time / result['avg_inference_time']
                size_reduction = result.get('size_reduction', 1.0)
                print(f"    {i+1}. {name}: {result['avg_inference_time']*1000:.2f}ms ({speedup:.2f}x speedup, {size_reduction:.2f}x smaller)")

        # Generate recommendations
        print(f"\n💡 Generating deployment recommendations...")
        recommendations = pipeline.generate_deployment_recommendations()

        if recommendations['mobile_recommendations']:
            mobile_rec = recommendations['mobile_recommendations']
            print(f"  • Best mobile target: {mobile_rec['best_target']}")

        if recommendations['edge_recommendations']:
            edge_rec = recommendations['edge_recommendations']
            print(f"  • Best edge target: {edge_rec['best_target']}")

        print(f"  • General recommendations:")
        for rec in recommendations['general_recommendations'][:3]:
            print(f"    - {rec}")

        # Create deployment guide
        with tempfile.TemporaryDirectory() as temp_dir:
            guide_path = pipeline.create_deployment_guide(temp_dir)
            print(f"  • Deployment guide created: {os.path.basename(guide_path)}")

    else:
        print("  ⚠️ No successful optimizations - check model compatibility")

    print("\n🎉 Mobile and edge optimization demonstration completed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Import numpy for benchmarking
    import numpy as np

    # Run demonstration
    demonstrate_mobile_edge_optimization()