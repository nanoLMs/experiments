#!/usr/bin/env python3
"""
Multi-Platform Export Integration Example
=========================================

Demonstrates comprehensive model export workflow including:
- Model preparation and optimization
- Multi-format export with validation
- Performance benchmarking
- Quality assurance and testing
- Deployment-ready package creation
"""

import torch
import torch.nn as nn
import os
import json
import time
import logging
from typing import Dict, Any, List
import tempfile
import shutil

# Import our export system
from multi_platform_exporter import (
    MultiPlatformExporter, ExportConfig, ExportFormat,
    OptimizationLevel, ValidationLevel, create_sample_inputs
)


class ProductionModel(nn.Module):
    """Production-ready model for export demonstration"""

    def __init__(self, vocab_size=10000, hidden_size=512, num_layers=6, num_heads=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Model architecture
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(1024, hidden_size)  # Max sequence length

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output layers
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)

        # Model metadata
        self.model_info = {
            "name": "ProductionNanoLM",
            "version": "1.0.0",
            "architecture": "transformer",
            "parameters": sum(p.numel() for p in self.parameters()),
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads
        }

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        # Create position ids
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        token_embeddings = self.embedding(input_ids)
        pos_embeddings = self.pos_embedding(position_ids)
        embeddings = token_embeddings + pos_embeddings
        embeddings = self.dropout(embeddings)

        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to transformer format (True = masked)
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        # Transformer
        hidden_states = self.transformer(
            embeddings,
            src_key_padding_mask=src_key_padding_mask
        )

        # Output
        hidden_states = self.layer_norm(hidden_states)
        logits = self.output(hidden_states)

        return logits

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for export metadata"""
        return self.model_info.copy()


class ExportPipeline:
    """Complete export pipeline with validation and optimization"""

    def __init__(self, model: nn.Module, model_name: str = "exported_model"):
        self.model = model
        self.model_name = model_name
        self.exporter = MultiPlatformExporter()

        # Export results storage
        self.export_results = {}
        self.benchmark_results = {}

        logging.info("✅ Export Pipeline initialized")
        logging.info(f"  • Model: {model.__class__.__name__}")
        logging.info(f"  • Parameters: {sum(p.numel() for p in model.parameters()):,}")

    def prepare_model(self, optimization_level: OptimizationLevel = OptimizationLevel.BASIC) -> nn.Module:
        """Prepare model for export with optimizations"""
        logging.info(f"🔧 Preparing model with {optimization_level.value} optimization")

        # Set to evaluation mode
        self.model.eval()

        # Apply optimizations based on level
        if optimization_level == OptimizationLevel.NONE:
            return self.model

        optimized_model = self.model

        # Basic optimizations
        if optimization_level in [OptimizationLevel.BASIC, OptimizationLevel.AGGRESSIVE]:
            # Fuse operations where possible
            try:
                # This is a placeholder - real optimization would be more sophisticated
                optimized_model = torch.jit.optimize_for_inference(torch.jit.script(self.model))
                logging.info("  ✅ Applied JIT optimizations")
            except Exception as e:
                logging.warning(f"  ⚠️ JIT optimization failed: {e}")
                optimized_model = self.model

        # Mobile/Edge optimizations
        if optimization_level in [OptimizationLevel.MOBILE, OptimizationLevel.EDGE]:
            try:
                # Apply quantization for mobile/edge
                optimized_model = torch.quantization.quantize_dynamic(
                    optimized_model, {nn.Linear}, dtype=torch.qint8
                )
                logging.info("  ✅ Applied dynamic quantization")
            except Exception as e:
                logging.warning(f"  ⚠️ Quantization failed: {e}")

        return optimized_model

    def create_export_configs(self, output_dir: str) -> List[ExportConfig]:
        """Create export configurations for different platforms"""
        configs = []

        # TorchScript - for PyTorch deployment
        configs.append(ExportConfig(
            format=ExportFormat.TORCHSCRIPT,
            output_path=os.path.join(output_dir, f"{self.model_name}.torchscript.pt"),
            model_name=self.model_name,
            optimization_level=OptimizationLevel.BASIC,
            validation_level=ValidationLevel.FUNCTIONAL,
            description="TorchScript export for PyTorch deployment",
            tags=["pytorch", "production"]
        ))

        # ONNX - for cross-platform deployment
        configs.append(ExportConfig(
            format=ExportFormat.ONNX,
            output_path=os.path.join(output_dir, f"{self.model_name}.onnx"),
            model_name=self.model_name,
            optimization_level=OptimizationLevel.BASIC,
            validation_level=ValidationLevel.FUNCTIONAL,
            onnx_opset_version=11,
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"}
            },
            description="ONNX export for cross-platform deployment",
            tags=["onnx", "cross-platform"]
        ))

        # CoreML - for Apple devices
        configs.append(ExportConfig(
            format=ExportFormat.COREML,
            output_path=os.path.join(output_dir, f"{self.model_name}.mlmodel"),
            model_name=self.model_name,
            optimization_level=OptimizationLevel.MOBILE,
            validation_level=ValidationLevel.BASIC,
            coreml_minimum_deployment_target="13.0",
            description="CoreML export for Apple devices",
            tags=["coreml", "apple", "mobile"]
        ))

        # HuggingFace - for Transformers ecosystem
        configs.append(ExportConfig(
            format=ExportFormat.HUGGINGFACE,
            output_path=os.path.join(output_dir, f"{self.model_name}_huggingface"),
            model_name=self.model_name,
            optimization_level=OptimizationLevel.NONE,
            validation_level=ValidationLevel.BASIC,
            description="HuggingFace export for Transformers ecosystem",
            tags=["huggingface", "transformers"]
        ))

        return configs

    def export_all_formats(self, output_dir: str, sample_inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Export model to all supported formats"""
        logging.info(f"🚀 Starting multi-format export to {output_dir}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Prepare model
        optimized_model = self.prepare_model(OptimizationLevel.BASIC)

        # Create export configurations
        configs = self.create_export_configs(output_dir)

        # Export to all formats
        results = self.exporter.export_multiple_formats(optimized_model, configs, sample_inputs)

        # Store results
        for result in results:
            self.export_results[result.format.value] = result

        # Generate summary
        successful_exports = [r for r in results if r.success]
        failed_exports = [r for r in results if not r.success]

        summary = {
            "total_formats": len(results),
            "successful_exports": len(successful_exports),
            "failed_exports": len(failed_exports),
            "total_size_mb": sum(r.file_size for r in successful_exports) / 1024 / 1024,
            "total_export_time": sum(r.export_time for r in results),
            "validation_passed": sum(1 for r in successful_exports if r.validation_passed),
            "export_details": {}
        }

        # Add detailed results
        for result in results:
            summary["export_details"][result.format.value] = {
                "success": result.success,
                "file_size_mb": result.file_size / 1024 / 1024 if result.success else 0,
                "export_time": result.export_time,
                "validation_passed": result.validation_passed,
                "error": result.error_message if not result.success else None
            }

        logging.info(f"✅ Export completed: {len(successful_exports)}/{len(results)} successful")

        return summary

    def benchmark_exports(self, sample_inputs: Dict[str, torch.Tensor], num_runs: int = 10) -> Dict[str, Any]:
        """Benchmark exported models"""
        logging.info(f"⚡ Starting export benchmark ({num_runs} runs)")

        # Get successful export configs
        successful_configs = []
        for format_name, result in self.export_results.items():
            if result.success:
                # Recreate config for benchmarking
                config = ExportConfig(
                    format=result.format,
                    output_path=result.output_path,
                    model_name=self.model_name
                )
                successful_configs.append(config)

        if not successful_configs:
            logging.warning("No successful exports to benchmark")
            return {}

        # Run benchmark
        benchmark_results = self.exporter.benchmark_exports(
            self.model, successful_configs, sample_inputs, num_runs
        )

        self.benchmark_results = benchmark_results

        # Generate benchmark summary
        summary = {
            "num_runs": num_runs,
            "formats_benchmarked": len(benchmark_results),
            "results": benchmark_results
        }

        # Find fastest format
        inference_times = {}
        for format_name, metrics in benchmark_results.items():
            if "avg_inference_time" in metrics:
                inference_times[format_name] = metrics["avg_inference_time"]

        if inference_times:
            fastest_format = min(inference_times.keys(), key=lambda k: inference_times[k])
            summary["fastest_format"] = fastest_format
            summary["fastest_time_ms"] = inference_times[fastest_format] * 1000

        logging.info("✅ Benchmark completed")

        return summary

    def validate_exports(self, sample_inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Validate all exported models"""
        logging.info("🔍 Validating exported models")

        validation_results = {}

        for format_name, result in self.export_results.items():
            if result.success:
                try:
                    validation_passed, metrics = self.exporter.validate_export(
                        self.model, result.output_path, result.format,
                        sample_inputs, tolerance=1e-4
                    )

                    validation_results[format_name] = {
                        "validation_passed": validation_passed,
                        "metrics": metrics
                    }

                except Exception as e:
                    validation_results[format_name] = {
                        "validation_passed": False,
                        "error": str(e)
                    }

        # Summary
        total_validated = len(validation_results)
        passed_validation = sum(1 for r in validation_results.values() if r["validation_passed"])

        summary = {
            "total_validated": total_validated,
            "passed_validation": passed_validation,
            "validation_rate": passed_validation / total_validated if total_validated > 0 else 0,
            "details": validation_results
        }

        logging.info(f"✅ Validation completed: {passed_validation}/{total_validated} passed")

        return summary

    def create_deployment_package(self, output_dir: str, sample_inputs: Dict[str, torch.Tensor]) -> str:
        """Create deployment package with all exports and metadata"""
        logging.info("📦 Creating deployment package")

        package_dir = os.path.join(output_dir, f"{self.model_name}_deployment_package")
        os.makedirs(package_dir, exist_ok=True)

        # Create subdirectories
        models_dir = os.path.join(package_dir, "models")
        docs_dir = os.path.join(package_dir, "docs")
        examples_dir = os.path.join(package_dir, "examples")

        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(docs_dir, exist_ok=True)
        os.makedirs(examples_dir, exist_ok=True)

        # Copy successful exports to package
        for format_name, result in self.export_results.items():
            if result.success:
                src_path = result.output_path
                if os.path.isfile(src_path):
                    dst_path = os.path.join(models_dir, os.path.basename(src_path))
                    shutil.copy2(src_path, dst_path)
                elif os.path.isdir(src_path):
                    dst_path = os.path.join(models_dir, os.path.basename(src_path))
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)

        # Create model metadata
        if hasattr(self.model, 'get_model_info'):
            model_info = self.model.get_model_info()
        else:
            model_info = {
                "name": self.model_name,
                "architecture": self.model.__class__.__name__,
                "parameters": sum(p.numel() for p in self.model.parameters())
            }

        metadata = {
            "model_info": model_info,
            "export_summary": {
                "formats": list(self.export_results.keys()),
                "successful_exports": [k for k, v in self.export_results.items() if v.success],
                "export_timestamp": time.time()
            },
            "benchmark_results": self.benchmark_results,
            "sample_inputs_info": {
                "input_names": list(sample_inputs.keys()),
                "input_shapes": {k: list(v.shape) for k, v in sample_inputs.items()},
                "input_dtypes": {k: str(v.dtype) for k, v in sample_inputs.items()}
            }
        }

        # Save metadata
        with open(os.path.join(package_dir, "model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Create README
        readme_content = f"""# {self.model_name} Deployment Package

## Model Information
- **Name**: {model_info.get('name', self.model_name)}
- **Architecture**: {model_info.get('architecture', 'Unknown')}
- **Parameters**: {model_info.get('parameters', 'Unknown'):,}

## Available Formats
"""

        for format_name, result in self.export_results.items():
            if result.success:
                readme_content += f"- **{format_name.upper()}**: `models/{os.path.basename(result.output_path)}`\n"

        readme_content += f"""
## Usage Examples
See the `examples/` directory for platform-specific usage examples.

## Benchmark Results
"""

        if self.benchmark_results:
            for format_name, metrics in self.benchmark_results.items():
                if "avg_inference_time" in metrics:
                    readme_content += f"- **{format_name}**: {metrics['avg_inference_time']*1000:.2f}ms average inference\n"

        with open(os.path.join(docs_dir, "README.md"), "w") as f:
            f.write(readme_content)

        # Create example usage scripts
        self._create_usage_examples(examples_dir, sample_inputs)

        logging.info(f"✅ Deployment package created: {package_dir}")

        return package_dir

    def _create_usage_examples(self, examples_dir: str, sample_inputs: Dict[str, torch.Tensor]):
        """Create usage examples for different formats"""

        # TorchScript example
        if "torchscript" in self.export_results and self.export_results["torchscript"].success:
            torchscript_example = f'''#!/usr/bin/env python3
"""
TorchScript Model Usage Example
"""

import torch

# Load the exported TorchScript model
model = torch.jit.load("../models/{os.path.basename(self.export_results["torchscript"].output_path)}")
model.eval()

# Create sample input
{self._generate_input_code(sample_inputs)}

# Run inference
with torch.no_grad():
    output = model(input_ids, attention_mask)
    print(f"Output shape: {{output.shape}}")
    print(f"Sample predictions: {{output[0, 0, :5]}}")
'''

            with open(os.path.join(examples_dir, "torchscript_example.py"), "w") as f:
                f.write(torchscript_example)

        # ONNX example
        if "onnx" in self.export_results and self.export_results["onnx"].success:
            onnx_example = f'''#!/usr/bin/env python3
"""
ONNX Model Usage Example
"""

import onnxruntime as ort
import numpy as np

# Load the exported ONNX model
session = ort.InferenceSession("../models/{os.path.basename(self.export_results["onnx"].output_path)}")

# Create sample input
{self._generate_numpy_input_code(sample_inputs)}

# Run inference
outputs = session.run(None, ort_inputs)
print(f"Output shape: {{outputs[0].shape}}")
print(f"Sample predictions: {{outputs[0][0, 0, :5]}}")
'''

            with open(os.path.join(examples_dir, "onnx_example.py"), "w") as f:
                f.write(onnx_example)

    def _generate_input_code(self, sample_inputs: Dict[str, torch.Tensor]) -> str:
        """Generate PyTorch input creation code"""
        lines = []
        for name, tensor in sample_inputs.items():
            if name == "input_ids":
                lines.append(f"{name} = torch.randint(0, 1000, {list(tensor.shape)})")
            elif name == "attention_mask":
                lines.append(f"{name} = torch.ones{list(tensor.shape)}")
            else:
                lines.append(f"{name} = torch.randn{list(tensor.shape)}")

        return "\n".join(lines)

    def _generate_numpy_input_code(self, sample_inputs: Dict[str, torch.Tensor]) -> str:
        """Generate NumPy input creation code"""
        lines = ["ort_inputs = {"]
        for name, tensor in sample_inputs.items():
            if name == "input_ids":
                lines.append(f'    "{name}": np.random.randint(0, 1000, {list(tensor.shape)}, dtype=np.int64),')
            elif name == "attention_mask":
                lines.append(f'    "{name}": np.ones({list(tensor.shape)}, dtype=np.int64),')
            else:
                lines.append(f'    "{name}": np.random.randn(*{list(tensor.shape)}).astype(np.float32),')
        lines.append("}")

        return "\n".join(lines)


def demonstrate_export_pipeline():
    """Demonstrate complete export pipeline"""
    print("🚀 Multi-Platform Export Pipeline Demonstration")

    # Create production model
    model = ProductionModel(
        vocab_size=5000,
        hidden_size=256,
        num_layers=4,
        num_heads=8
    )

    print(f"✅ Model created: {model.model_info['name']}")
    print(f"  • Parameters: {model.model_info['parameters']:,}")
    print(f"  • Architecture: {model.model_info['architecture']}")

    # Create sample inputs
    sample_inputs = create_sample_inputs(batch_size=2, seq_len=32, vocab_size=5000)
    print("✅ Sample inputs created")

    # Initialize export pipeline
    pipeline = ExportPipeline(model, "ProductionNanoLM_v1")

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = os.path.join(temp_dir, "exports")

        print(f"\n📤 Starting export pipeline to {output_dir}")

        # Export to all formats
        export_summary = pipeline.export_all_formats(output_dir, sample_inputs)

        print(f"\n📊 Export Summary:")
        print(f"  • Total formats: {export_summary['total_formats']}")
        print(f"  • Successful: {export_summary['successful_exports']}")
        print(f"  • Failed: {export_summary['failed_exports']}")
        print(f"  • Total size: {export_summary['total_size_mb']:.2f} MB")
        print(f"  • Total time: {export_summary['total_export_time']:.2f}s")
        print(f"  • Validation passed: {export_summary['validation_passed']}")

        # Show detailed results
        print(f"\n📋 Detailed Results:")
        for format_name, details in export_summary['export_details'].items():
            status = "✅" if details['success'] else "❌"
            validation = "✅" if details['validation_passed'] else "❌"

            print(f"  {status} {format_name.upper()}:")
            if details['success']:
                print(f"    • Size: {details['file_size_mb']:.2f} MB")
                print(f"    • Time: {details['export_time']:.2f}s")
                print(f"    • Validation: {validation}")
            else:
                print(f"    • Error: {details['error']}")

        # Benchmark successful exports
        if export_summary['successful_exports'] > 0:
            print(f"\n⚡ Running performance benchmark...")
            benchmark_summary = pipeline.benchmark_exports(sample_inputs, num_runs=5)

            if benchmark_summary:
                print(f"  • Formats benchmarked: {benchmark_summary['formats_benchmarked']}")
                if 'fastest_format' in benchmark_summary:
                    print(f"  • Fastest format: {benchmark_summary['fastest_format']} ({benchmark_summary['fastest_time_ms']:.2f}ms)")

                print(f"  • Benchmark details:")
                for format_name, metrics in benchmark_summary['results'].items():
                    if 'avg_inference_time' in metrics:
                        print(f"    - {format_name}: {metrics['avg_inference_time']*1000:.2f}ms avg")

        # Validate exports
        print(f"\n🔍 Validating exports...")
        validation_summary = pipeline.validate_exports(sample_inputs)

        print(f"  • Validated: {validation_summary['total_validated']}")
        print(f"  • Passed: {validation_summary['passed_validation']}")
        print(f"  • Success rate: {validation_summary['validation_rate']*100:.1f}%")

        # Create deployment package
        print(f"\n📦 Creating deployment package...")
        package_path = pipeline.create_deployment_package(output_dir, sample_inputs)

        # List package contents
        print(f"  • Package created: {os.path.basename(package_path)}")
        print(f"  • Contents:")
        for root, dirs, files in os.walk(package_path):
            level = root.replace(package_path, '').count(os.sep)
            indent = ' ' * 4 * (level + 1)
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 4 * (level + 2)
            for file in files:
                print(f"{subindent}{file}")

    print("\n🎉 Export pipeline demonstration completed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Run demonstration
    demonstrate_export_pipeline()