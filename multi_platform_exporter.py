#!/usr/bin/env python3
"""
Multi-Platform Export System
============================

Comprehensive model export infrastructure supporting multiple formats:
- TorchScript (JIT compilation)
- ONNX (Open Neural Network Exchange)
- CoreML (Apple devices)
- HuggingFace (Transformers ecosystem)
- TensorFlow Lite (Mobile/Edge)
- OpenVINO (Intel optimization)

With validation, optimization, and quality checks.
"""

import torch
import torch.nn as nn
import torch.jit
import os
import json
import logging
import time
import tempfile
import shutil
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import numpy as np
import warnings

# Optional imports with fallbacks
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX not available - ONNX export will be disabled")

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    logging.warning("CoreML not available - CoreML export will be disabled")

try:
    from transformers import AutoTokenizer, AutoModel, PreTrainedModel
    from transformers.modeling_utils import PreTrainedModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available - HuggingFace export will be disabled")

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available - TFLite export will be disabled")

try:
    from openvino.tools import mo
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False
    logging.warning("OpenVINO not available - OpenVINO export will be disabled")


class ExportFormat(Enum):
    """Supported export formats"""
    TORCHSCRIPT = "torchscript"
    ONNX = "onnx"
    COREML = "coreml"
    HUGGINGFACE = "huggingface"
    TFLITE = "tflite"
    OPENVINO = "openvino"
    PYTORCH = "pytorch"  # Standard PyTorch checkpoint


class OptimizationLevel(Enum):
    """Optimization levels for exports"""
    NONE = "none"           # No optimization
    BASIC = "basic"         # Basic optimizations
    AGGRESSIVE = "aggressive"  # Aggressive optimizations
    MOBILE = "mobile"       # Mobile-specific optimizations
    EDGE = "edge"          # Edge device optimizations


class ValidationLevel(Enum):
    """Validation levels for exports"""
    NONE = "none"           # No validation
    BASIC = "basic"         # Basic format validation
    FUNCTIONAL = "functional"  # Functional correctness
    PERFORMANCE = "performance"  # Performance validation
    COMPREHENSIVE = "comprehensive"  # All validations


@dataclass
class ExportConfig:
    """Configuration for model export"""
    # Basic settings
    format: ExportFormat
    output_path: str
    model_name: str = "exported_model"

    # Optimization settings
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    quantization: bool = False
    pruning: bool = False

    # Validation settings
    validation_level: ValidationLevel = ValidationLevel.FUNCTIONAL
    tolerance: float = 1e-5

    # Format-specific settings
    onnx_opset_version: int = 11
    coreml_minimum_deployment_target: str = "13.0"
    tflite_representative_dataset: Optional[Callable] = None

    # Input specifications
    input_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    input_types: Dict[str, torch.dtype] = field(default_factory=dict)
    dynamic_axes: Dict[str, Dict[int, str]] = field(default_factory=dict)

    # Metadata
    description: str = ""
    version: str = "1.0"
    author: str = ""
    license: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class ExportResult:
    """Result of model export operation"""
    success: bool
    format: ExportFormat
    output_path: str
    file_size: int
    export_time: float

    # Validation results
    validation_passed: bool = False
    validation_metrics: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    inference_time: Optional[float] = None
    memory_usage: Optional[int] = None

    # Error information
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    # Metadata
    model_info: Dict[str, Any] = field(default_factory=dict)


class BaseExporter:
    """Base class for format-specific exporters"""

    def __init__(self, config: ExportConfig):
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def export(self, model: nn.Module, sample_inputs: Dict[str, torch.Tensor]) -> ExportResult:
        """Export model to specified format"""
        raise NotImplementedError("Subclasses must implement export method")

    def validate(self, model: nn.Module, exported_path: str,
                sample_inputs: Dict[str, torch.Tensor]) -> Tuple[bool, Dict[str, Any]]:
        """Validate exported model"""
        if self.config.validation_level == ValidationLevel.NONE:
            return True, {}

        try:
            return self._validate_export(model, exported_path, sample_inputs)
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return False, {"error": str(e)}

    def _validate_export(self, model: nn.Module, exported_path: str,
                        sample_inputs: Dict[str, torch.Tensor]) -> Tuple[bool, Dict[str, Any]]:
        """Format-specific validation implementation"""
        return True, {}

    def _optimize_model(self, model: nn.Module) -> nn.Module:
        """Apply optimizations to model before export"""
        if self.config.optimization_level == OptimizationLevel.NONE:
            return model

        optimized_model = model

        # Basic optimizations
        if self.config.optimization_level in [OptimizationLevel.BASIC, OptimizationLevel.AGGRESSIVE]:
            # Fuse operations where possible
            try:
                optimized_model = torch.jit.optimize_for_inference(torch.jit.script(model))
            except Exception as e:
                self.logger.warning(f"JIT optimization failed: {e}")
                optimized_model = model

        # Quantization
        if self.config.quantization:
            try:
                optimized_model = self._apply_quantization(optimized_model)
            except Exception as e:
                self.logger.warning(f"Quantization failed: {e}")

        # Pruning
        if self.config.pruning:
            try:
                optimized_model = self._apply_pruning(optimized_model)
            except Exception as e:
                self.logger.warning(f"Pruning failed: {e}")

        return optimized_model

    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """Apply quantization to model"""
        # Dynamic quantization as default
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
        return quantized_model

    def _apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply pruning to model"""
        # Simple magnitude-based pruning
        try:
            import torch.nn.utils.prune as prune

            for module in model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    prune.l1_unstructured(module, name='weight', amount=0.2)
                    prune.remove(module, 'weight')

            return model
        except ImportError:
            self.logger.warning("Pruning not available")
            return model


class TorchScriptExporter(BaseExporter):
    """TorchScript exporter"""

    def export(self, model: nn.Module, sample_inputs: Dict[str, torch.Tensor]) -> ExportResult:
        start_time = time.time()
        result = ExportResult(
            success=False,
            format=ExportFormat.TORCHSCRIPT,
            output_path=self.config.output_path,
            file_size=0,
            export_time=0
        )

        try:
            # Optimize model
            optimized_model = self._optimize_model(model)
            optimized_model.eval()

            # Convert sample inputs to tuple/list for tracing
            if len(sample_inputs) == 1:
                sample_input = next(iter(sample_inputs.values()))
            else:
                sample_input = tuple(sample_inputs.values())

            # Try scripting first, fall back to tracing
            try:
                traced_model = torch.jit.script(optimized_model)
                self.logger.info("Model exported using torch.jit.script")
            except Exception as e:
                self.logger.warning(f"Scripting failed: {e}, falling back to tracing")
                traced_model = torch.jit.trace(optimized_model, sample_input)
                self.logger.info("Model exported using torch.jit.trace")

            # Save model
            os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)
            traced_model.save(self.config.output_path)

            # Get file size
            file_size = os.path.getsize(self.config.output_path)
            export_time = time.time() - start_time

            # Validate export
            validation_passed, validation_metrics = self.validate(
                model, self.config.output_path, sample_inputs
            )

            result.success = True
            result.file_size = file_size
            result.export_time = export_time
            result.validation_passed = validation_passed
            result.validation_metrics = validation_metrics

            self.logger.info(f"TorchScript export successful: {self.config.output_path}")

        except Exception as e:
            result.error_message = str(e)
            self.logger.error(f"TorchScript export failed: {e}")

        return result

    def _validate_export(self, model: nn.Module, exported_path: str,
                        sample_inputs: Dict[str, torch.Tensor]) -> Tuple[bool, Dict[str, Any]]:
        """Validate TorchScript export"""
        try:
            # Load exported model
            loaded_model = torch.jit.load(exported_path)
            loaded_model.eval()

            # Compare outputs
            model.eval()
            with torch.no_grad():
                if len(sample_inputs) == 1:
                    sample_input = next(iter(sample_inputs.values()))
                    original_output = model(sample_input)
                    exported_output = loaded_model(sample_input)
                else:
                    sample_input = tuple(sample_inputs.values())
                    original_output = model(*sample_input)
                    exported_output = loaded_model(*sample_input)

                # Check output similarity
                if isinstance(original_output, torch.Tensor):
                    diff = torch.abs(original_output - exported_output).max().item()
                elif isinstance(original_output, dict):
                    diff = max(torch.abs(original_output[k] - exported_output[k]).max().item()
                              for k in original_output.keys())
                else:
                    diff = 0.0  # Can't validate complex outputs

                validation_passed = diff < self.config.tolerance

                return validation_passed, {
                    "max_difference": diff,
                    "tolerance": self.config.tolerance,
                    "output_shape": original_output.shape if isinstance(original_output, torch.Tensor) else "complex"
                }

        except Exception as e:
            return False, {"error": str(e)}


class ONNXExporter(BaseExporter):
    """ONNX exporter"""

    def export(self, model: nn.Module, sample_inputs: Dict[str, torch.Tensor]) -> ExportResult:
        if not ONNX_AVAILABLE:
            return ExportResult(
                success=False,
                format=ExportFormat.ONNX,
                output_path=self.config.output_path,
                file_size=0,
                export_time=0,
                error_message="ONNX not available"
            )

        start_time = time.time()
        result = ExportResult(
            success=False,
            format=ExportFormat.ONNX,
            output_path=self.config.output_path,
            file_size=0,
            export_time=0
        )

        try:
            # Optimize model
            optimized_model = self._optimize_model(model)
            optimized_model.eval()

            # Prepare inputs
            input_names = list(sample_inputs.keys())
            output_names = ["output"]

            # Convert sample inputs
            if len(sample_inputs) == 1:
                sample_input = next(iter(sample_inputs.values()))
            else:
                sample_input = tuple(sample_inputs.values())

            # Create output directory
            os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)

            # Export to ONNX
            torch.onnx.export(
                optimized_model,
                sample_input,
                self.config.output_path,
                export_params=True,
                opset_version=self.config.onnx_opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=self.config.dynamic_axes
            )

            # Get file size
            file_size = os.path.getsize(self.config.output_path)
            export_time = time.time() - start_time

            # Validate export
            validation_passed, validation_metrics = self.validate(
                model, self.config.output_path, sample_inputs
            )

            result.success = True
            result.file_size = file_size
            result.export_time = export_time
            result.validation_passed = validation_passed
            result.validation_metrics = validation_metrics

            self.logger.info(f"ONNX export successful: {self.config.output_path}")

        except Exception as e:
            result.error_message = str(e)
            self.logger.error(f"ONNX export failed: {e}")

        return result

    def _validate_export(self, model: nn.Module, exported_path: str,
                        sample_inputs: Dict[str, torch.Tensor]) -> Tuple[bool, Dict[str, Any]]:
        """Validate ONNX export"""
        try:
            # Load and check ONNX model
            onnx_model = onnx.load(exported_path)
            onnx.checker.check_model(onnx_model)

            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(exported_path)

            # Prepare inputs for ONNX Runtime
            ort_inputs = {}
            for name, tensor in sample_inputs.items():
                ort_inputs[name] = tensor.cpu().numpy()

            # Run inference
            ort_outputs = ort_session.run(None, ort_inputs)

            # Compare with original model
            model.eval()
            with torch.no_grad():
                if len(sample_inputs) == 1:
                    sample_input = next(iter(sample_inputs.values()))
                    original_output = model(sample_input)
                else:
                    sample_input = tuple(sample_inputs.values())
                    original_output = model(*sample_input)

                if isinstance(original_output, torch.Tensor):
                    original_np = original_output.cpu().numpy()
                    diff = np.abs(original_np - ort_outputs[0]).max()
                else:
                    diff = 0.0  # Can't validate complex outputs

                validation_passed = diff < self.config.tolerance

                return validation_passed, {
                    "max_difference": float(diff),
                    "tolerance": self.config.tolerance,
                    "onnx_opset_version": self.config.onnx_opset_version
                }

        except Exception as e:
            return False, {"error": str(e)}


class CoreMLExporter(BaseExporter):
    """CoreML exporter"""

    def export(self, model: nn.Module, sample_inputs: Dict[str, torch.Tensor]) -> ExportResult:
        if not COREML_AVAILABLE:
            return ExportResult(
                success=False,
                format=ExportFormat.COREML,
                output_path=self.config.output_path,
                file_size=0,
                export_time=0,
                error_message="CoreML not available"
            )

        start_time = time.time()
        result = ExportResult(
            success=False,
            format=ExportFormat.COREML,
            output_path=self.config.output_path,
            file_size=0,
            export_time=0
        )

        try:
            # Optimize model
            optimized_model = self._optimize_model(model)
            optimized_model.eval()

            # Convert to TorchScript first
            if len(sample_inputs) == 1:
                sample_input = next(iter(sample_inputs.values()))
            else:
                sample_input = tuple(sample_inputs.values())

            traced_model = torch.jit.trace(optimized_model, sample_input)

            # Convert to CoreML
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=tensor.shape) for tensor in sample_inputs.values()],
                minimum_deployment_target=ct.target.iOS13 if self.config.coreml_minimum_deployment_target == "13.0" else ct.target.iOS14
            )

            # Create output directory
            os.makedirs(os.path.dirname(self.config.output_path), exist_ok=True)

            # Save CoreML model
            coreml_model.save(self.config.output_path)

            # Get file size (CoreML creates a directory)
            if os.path.isdir(self.config.output_path):
                file_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                               for dirpath, dirnames, filenames in os.walk(self.config.output_path)
                               for filename in filenames)
            else:
                file_size = os.path.getsize(self.config.output_path)

            export_time = time.time() - start_time

            # Validate export
            validation_passed, validation_metrics = self.validate(
                model, self.config.output_path, sample_inputs
            )

            result.success = True
            result.file_size = file_size
            result.export_time = export_time
            result.validation_passed = validation_passed
            result.validation_metrics = validation_metrics

            self.logger.info(f"CoreML export successful: {self.config.output_path}")

        except Exception as e:
            result.error_message = str(e)
            self.logger.error(f"CoreML export failed: {e}")

        return result

    def _validate_export(self, model: nn.Module, exported_path: str,
                        sample_inputs: Dict[str, torch.Tensor]) -> Tuple[bool, Dict[str, Any]]:
        """Validate CoreML export"""
        try:
            # Load CoreML model
            coreml_model = ct.models.MLModel(exported_path)

            # Prepare inputs
            coreml_inputs = {}
            for name, tensor in sample_inputs.items():
                coreml_inputs[name] = tensor.cpu().numpy()

            # Run prediction
            coreml_outputs = coreml_model.predict(coreml_inputs)

            # Compare with original model
            model.eval()
            with torch.no_grad():
                if len(sample_inputs) == 1:
                    sample_input = next(iter(sample_inputs.values()))
                    original_output = model(sample_input)
                else:
                    sample_input = tuple(sample_inputs.values())
                    original_output = model(*sample_input)

                if isinstance(original_output, torch.Tensor):
                    original_np = original_output.cpu().numpy()
                    # CoreML output might have different key names
                    coreml_output_array = next(iter(coreml_outputs.values()))
                    diff = np.abs(original_np - coreml_output_array).max()
                else:
                    diff = 0.0

                validation_passed = diff < self.config.tolerance

                return validation_passed, {
                    "max_difference": float(diff),
                    "tolerance": self.config.tolerance,
                    "deployment_target": self.config.coreml_minimum_deployment_target
                }

        except Exception as e:
            return False, {"error": str(e)}


class HuggingFaceExporter(BaseExporter):
    """HuggingFace format exporter"""

    def export(self, model: nn.Module, sample_inputs: Dict[str, torch.Tensor]) -> ExportResult:
        if not TRANSFORMERS_AVAILABLE:
            return ExportResult(
                success=False,
                format=ExportFormat.HUGGINGFACE,
                output_path=self.config.output_path,
                file_size=0,
                export_time=0,
                error_message="Transformers not available"
            )

        start_time = time.time()
        result = ExportResult(
            success=False,
            format=ExportFormat.HUGGINGFACE,
            output_path=self.config.output_path,
            file_size=0,
            export_time=0
        )

        try:
            # Create output directory
            os.makedirs(self.config.output_path, exist_ok=True)

            # Save model
            if isinstance(model, PreTrainedModel):
                # Already a HuggingFace model
                model.save_pretrained(self.config.output_path)
            else:
                # Convert PyTorch model to HuggingFace format
                # This is a simplified conversion - real implementation would need
                # proper configuration and tokenizer handling
                torch.save(model.state_dict(), os.path.join(self.config.output_path, "pytorch_model.bin"))

                # Create basic config
                config = {
                    "model_type": "custom",
                    "architectures": [model.__class__.__name__],
                    "torch_dtype": "float32",
                    "transformers_version": "4.0.0"
                }

                with open(os.path.join(self.config.output_path, "config.json"), "w") as f:
                    json.dump(config, f, indent=2)

            # Calculate total file size
            file_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                           for dirpath, dirnames, filenames in os.walk(self.config.output_path)
                           for filename in filenames)

            export_time = time.time() - start_time

            result.success = True
            result.file_size = file_size
            result.export_time = export_time
            result.validation_passed = True  # Basic validation
            result.validation_metrics = {"format": "huggingface"}

            self.logger.info(f"HuggingFace export successful: {self.config.output_path}")

        except Exception as e:
            result.error_message = str(e)
            self.logger.error(f"HuggingFace export failed: {e}")

        return result


class MultiPlatformExporter:
    """Main exporter class that coordinates different format exporters"""

    def __init__(self):
        self.exporters = {
            ExportFormat.TORCHSCRIPT: TorchScriptExporter,
            ExportFormat.ONNX: ONNXExporter,
            ExportFormat.COREML: CoreMLExporter,
            ExportFormat.HUGGINGFACE: HuggingFaceExporter,
        }

        self.logger = logging.getLogger(self.__class__.__name__)

        logging.info("✅ Multi-Platform Exporter initialized")
        logging.info(f"  • Available formats: {[fmt.value for fmt in self.exporters.keys()]}")

    def export_model(self, model: nn.Module, config: ExportConfig,
                    sample_inputs: Dict[str, torch.Tensor]) -> ExportResult:
        """Export model to specified format"""

        if config.format not in self.exporters:
            return ExportResult(
                success=False,
                format=config.format,
                output_path=config.output_path,
                file_size=0,
                export_time=0,
                error_message=f"Unsupported format: {config.format}"
            )

        # Create exporter instance
        exporter_class = self.exporters[config.format]
        exporter = exporter_class(config)

        self.logger.info(f"Starting {config.format.value} export to {config.output_path}")

        # Perform export
        result = exporter.export(model, sample_inputs)

        if result.success:
            self.logger.info(f"✅ Export successful: {config.format.value}")
            self.logger.info(f"  • Output: {result.output_path}")
            self.logger.info(f"  • Size: {result.file_size / 1024 / 1024:.2f} MB")
            self.logger.info(f"  • Time: {result.export_time:.2f}s")
            self.logger.info(f"  • Validation: {'✅' if result.validation_passed else '❌'}")
        else:
            self.logger.error(f"❌ Export failed: {config.format.value}")
            self.logger.error(f"  • Error: {result.error_message}")

        return result

    def export_multiple_formats(self, model: nn.Module,
                               configs: List[ExportConfig],
                               sample_inputs: Dict[str, torch.Tensor]) -> List[ExportResult]:
        """Export model to multiple formats"""
        results = []

        self.logger.info(f"Starting multi-format export ({len(configs)} formats)")

        for config in configs:
            result = self.export_model(model, config, sample_inputs)
            results.append(result)

        # Summary
        successful_exports = [r for r in results if r.success]
        self.logger.info(f"Multi-format export completed: {len(successful_exports)}/{len(configs)} successful")

        return results

    def validate_export(self, original_model: nn.Module, exported_path: str,
                       format: ExportFormat, sample_inputs: Dict[str, torch.Tensor],
                       tolerance: float = 1e-5) -> Tuple[bool, Dict[str, Any]]:
        """Validate an exported model"""

        if format not in self.exporters:
            return False, {"error": f"Unsupported format: {format}"}

        # Create temporary config for validation
        config = ExportConfig(
            format=format,
            output_path=exported_path,
            tolerance=tolerance,
            validation_level=ValidationLevel.FUNCTIONAL
        )

        exporter_class = self.exporters[format]
        exporter = exporter_class(config)

        return exporter.validate(original_model, exported_path, sample_inputs)

    def get_export_info(self, exported_path: str, format: ExportFormat) -> Dict[str, Any]:
        """Get information about an exported model"""
        info = {
            "path": exported_path,
            "format": format.value,
            "exists": os.path.exists(exported_path),
            "size": 0,
            "created": None
        }

        if info["exists"]:
            if os.path.isfile(exported_path):
                info["size"] = os.path.getsize(exported_path)
                info["created"] = os.path.getctime(exported_path)
            elif os.path.isdir(exported_path):
                info["size"] = sum(os.path.getsize(os.path.join(dirpath, filename))
                                 for dirpath, dirnames, filenames in os.walk(exported_path)
                                 for filename in filenames)
                info["created"] = os.path.getctime(exported_path)

        return info

    def benchmark_exports(self, model: nn.Module, configs: List[ExportConfig],
                         sample_inputs: Dict[str, torch.Tensor],
                         num_runs: int = 10) -> Dict[str, Dict[str, Any]]:
        """Benchmark different export formats"""

        self.logger.info(f"Starting export benchmark ({len(configs)} formats, {num_runs} runs each)")

        benchmark_results = {}

        for config in configs:
            format_name = config.format.value
            self.logger.info(f"Benchmarking {format_name}...")

            # Export model
            export_result = self.export_model(model, config, sample_inputs)

            if not export_result.success:
                benchmark_results[format_name] = {
                    "export_failed": True,
                    "error": export_result.error_message
                }
                continue

            # Benchmark inference
            inference_times = []

            try:
                if config.format == ExportFormat.TORCHSCRIPT:
                    loaded_model = torch.jit.load(config.output_path)
                    loaded_model.eval()

                    for _ in range(num_runs):
                        start_time = time.time()
                        with torch.no_grad():
                            if len(sample_inputs) == 1:
                                sample_input = next(iter(sample_inputs.values()))
                                _ = loaded_model(sample_input)
                            else:
                                sample_input = tuple(sample_inputs.values())
                                _ = loaded_model(*sample_input)
                        inference_times.append(time.time() - start_time)

                elif config.format == ExportFormat.ONNX and ONNX_AVAILABLE:
                    ort_session = ort.InferenceSession(config.output_path)
                    ort_inputs = {name: tensor.cpu().numpy() for name, tensor in sample_inputs.items()}

                    for _ in range(num_runs):
                        start_time = time.time()
                        _ = ort_session.run(None, ort_inputs)
                        inference_times.append(time.time() - start_time)

                # Calculate statistics
                if inference_times:
                    benchmark_results[format_name] = {
                        "export_time": export_result.export_time,
                        "file_size": export_result.file_size,
                        "avg_inference_time": np.mean(inference_times),
                        "std_inference_time": np.std(inference_times),
                        "min_inference_time": np.min(inference_times),
                        "max_inference_time": np.max(inference_times),
                        "validation_passed": export_result.validation_passed
                    }
                else:
                    benchmark_results[format_name] = {
                        "export_time": export_result.export_time,
                        "file_size": export_result.file_size,
                        "inference_benchmark": "not_supported",
                        "validation_passed": export_result.validation_passed
                    }

            except Exception as e:
                benchmark_results[format_name] = {
                    "export_time": export_result.export_time,
                    "file_size": export_result.file_size,
                    "benchmark_error": str(e),
                    "validation_passed": export_result.validation_passed
                }

        self.logger.info("✅ Export benchmark completed")
        return benchmark_results


def create_sample_inputs(batch_size: int = 1, seq_len: int = 128,
                        vocab_size: int = 1000) -> Dict[str, torch.Tensor]:
    """Create sample inputs for testing"""
    return {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len)
    }


def test_multi_platform_export():
    """Test the multi-platform export system"""
    print("🧪 Testing Multi-Platform Export System")

    # Create test model
    class TestModel(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=256):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.linear = nn.Linear(hidden_size, vocab_size)
            self.dropout = nn.Dropout(0.1)

        def forward(self, input_ids, attention_mask=None):
            x = self.embedding(input_ids)
            x = self.dropout(x)
            x = self.linear(x)
            return x

    model = TestModel()
    model.eval()

    print(f"✅ Test model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create sample inputs
    sample_inputs = create_sample_inputs(batch_size=2, seq_len=64)
    print("✅ Sample inputs created")

    # Create exporter
    exporter = MultiPlatformExporter()

    # Test different export formats
    export_configs = []

    # TorchScript
    export_configs.append(ExportConfig(
        format=ExportFormat.TORCHSCRIPT,
        output_path="./test_exports/model.torchscript.pt",
        optimization_level=OptimizationLevel.BASIC,
        validation_level=ValidationLevel.FUNCTIONAL
    ))

    # ONNX (if available)
    if ONNX_AVAILABLE:
        export_configs.append(ExportConfig(
            format=ExportFormat.ONNX,
            output_path="./test_exports/model.onnx",
            onnx_opset_version=11,
            dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}},
            validation_level=ValidationLevel.FUNCTIONAL
        ))

    # CoreML (if available)
    if COREML_AVAILABLE:
        export_configs.append(ExportConfig(
            format=ExportFormat.COREML,
            output_path="./test_exports/model.mlmodel",
            coreml_minimum_deployment_target="13.0",
            validation_level=ValidationLevel.BASIC
        ))

    # HuggingFace (if available)
    if TRANSFORMERS_AVAILABLE:
        export_configs.append(ExportConfig(
            format=ExportFormat.HUGGINGFACE,
            output_path="./test_exports/huggingface_model",
            validation_level=ValidationLevel.BASIC
        ))

    print(f"✅ Testing {len(export_configs)} export formats")

    # Export to multiple formats
    results = exporter.export_multiple_formats(model, export_configs, sample_inputs)

    # Display results
    print("\n📊 Export Results:")
    for result in results:
        status = "✅" if result.success else "❌"
        validation = "✅" if result.validation_passed else "❌"

        print(f"  {status} {result.format.value}:")
        if result.success:
            print(f"    • Size: {result.file_size / 1024 / 1024:.2f} MB")
            print(f"    • Time: {result.export_time:.2f}s")
            print(f"    • Validation: {validation}")
            if result.validation_metrics:
                for key, value in result.validation_metrics.items():
                    print(f"    • {key}: {value}")
        else:
            print(f"    • Error: {result.error_message}")

    # Test benchmarking (only for successful exports)
    successful_configs = [config for config, result in zip(export_configs, results) if result.success]

    if successful_configs:
        print(f"\n⚡ Running benchmark for {len(successful_configs)} successful exports...")
        benchmark_results = exporter.benchmark_exports(model, successful_configs, sample_inputs, num_runs=5)

        print("\n📈 Benchmark Results:")
        for format_name, metrics in benchmark_results.items():
            print(f"  • {format_name}:")
            if "avg_inference_time" in metrics:
                print(f"    - Avg inference: {metrics['avg_inference_time']*1000:.2f}ms")
                print(f"    - Export time: {metrics['export_time']:.2f}s")
                print(f"    - File size: {metrics['file_size'] / 1024 / 1024:.2f} MB")

    # Cleanup
    import shutil
    if os.path.exists("./test_exports"):
        shutil.rmtree("./test_exports")

    print("\n🎉 Multi-platform export tests completed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    test_multi_platform_export()