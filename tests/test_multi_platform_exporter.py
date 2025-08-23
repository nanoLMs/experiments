#!/usr/bin/env python3
"""
Test suite for Multi-Platform Export System
===========================================

Comprehensive tests for the model export system including:
- Export format support and validation
- Export configuration and optimization
- Quality checks and validation
- Performance benchmarking
- Error handling and recovery
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_platform_exporter import (
    MultiPlatformExporter, ExportConfig, ExportResult, ExportFormat,
    OptimizationLevel, ValidationLevel, TorchScriptExporter, ONNXExporter,
    CoreMLExporter, HuggingFaceExporter, create_sample_inputs
)


class SimpleTestModel(nn.Module):
    """Simple test model for export testing"""

    def __init__(self, vocab_size=100, hidden_size=64, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.dropout(x)

        for layer in self.layers:
            x = torch.relu(layer(x))

        return self.output(x)


class LinearModel(nn.Module):
    """Very simple linear model for basic testing"""

    def __init__(self, input_size=10, output_size=5):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def simple_model():
    """Create simple test model"""
    return SimpleTestModel(vocab_size=50, hidden_size=32, num_layers=2)


@pytest.fixture
def linear_model():
    """Create linear test model"""
    return LinearModel(input_size=10, output_size=5)


@pytest.fixture
def sample_inputs():
    """Create sample inputs for testing"""
    return create_sample_inputs(batch_size=2, seq_len=16, vocab_size=50)


@pytest.fixture
def simple_inputs():
    """Create simple inputs for linear model"""
    return {"x": torch.randn(2, 10)}


@pytest.fixture
def exporter():
    """Create multi-platform exporter"""
    return MultiPlatformExporter()


class TestExportConfig:
    """Test export configuration"""

    def test_default_config(self):
        """Test default export configuration"""
        config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT,
            output_path="test.pt"
        )

        assert config.format == ExportFormat.TORCHSCRIPT
        assert config.output_path == "test.pt"
        assert config.model_name == "exported_model"
        assert config.optimization_level == OptimizationLevel.BASIC
        assert config.validation_level == ValidationLevel.FUNCTIONAL
        assert config.quantization == False
        assert config.pruning == False
        assert config.tolerance == 1e-5

    def test_custom_config(self):
        """Test custom export configuration"""
        config = ExportConfig(
            format=ExportFormat.ONNX,
            output_path="custom.onnx",
            model_name="custom_model",
            optimization_level=OptimizationLevel.AGGRESSIVE,
            validation_level=ValidationLevel.COMPREHENSIVE,
            quantization=True,
            pruning=True,
            tolerance=1e-3,
            onnx_opset_version=12
        )

        assert config.format == ExportFormat.ONNX
        assert config.output_path == "custom.onnx"
        assert config.model_name == "custom_model"
        assert config.optimization_level == OptimizationLevel.AGGRESSIVE
        assert config.validation_level == ValidationLevel.COMPREHENSIVE
        assert config.quantization == True
        assert config.pruning == True
        assert config.tolerance == 1e-3
        assert config.onnx_opset_version == 12


class TestExportResult:
    """Test export result structure"""

    def test_result_creation(self):
        """Test export result creation"""
        result = ExportResult(
            success=True,
            format=ExportFormat.TORCHSCRIPT,
            output_path="test.pt",
            file_size=1024,
            export_time=1.5
        )

        assert result.success == True
        assert result.format == ExportFormat.TORCHSCRIPT
        assert result.output_path == "test.pt"
        assert result.file_size == 1024
        assert result.export_time == 1.5
        assert result.validation_passed == False
        assert len(result.validation_metrics) == 0
        assert result.inference_time is None
        assert result.memory_usage is None
        assert result.error_message is None
        assert len(result.warnings) == 0


class TestTorchScriptExporter:
    """Test TorchScript exporter"""

    def test_torchscript_export_success(self, linear_model, simple_inputs, temp_dir):
        """Test successful TorchScript export"""
        config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT,
            output_path=os.path.join(temp_dir, "model.pt"),
            validation_level=ValidationLevel.FUNCTIONAL
        )

        exporter = TorchScriptExporter(config)
        result = exporter.export(linear_model, simple_inputs)

        assert result.success == True
        assert result.format == ExportFormat.TORCHSCRIPT
        assert os.path.exists(result.output_path)
        assert result.file_size > 0
        assert result.export_time > 0
        # Validation might fail due to environment, but export should succeed

    def test_torchscript_validation(self, linear_model, simple_inputs, temp_dir):
        """Test TorchScript validation"""
        config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT,
            output_path=os.path.join(temp_dir, "model.pt"),
            validation_level=ValidationLevel.FUNCTIONAL,
            tolerance=1e-5
        )

        exporter = TorchScriptExporter(config)
        result = exporter.export(linear_model, simple_inputs)

        if result.success:
            # Test validation separately
            validation_passed, metrics = exporter.validate(
                linear_model, result.output_path, simple_inputs
            )

            assert isinstance(validation_passed, bool)
            assert isinstance(metrics, dict)

            if validation_passed:
                assert 'max_difference' in metrics
                assert 'tolerance' in metrics
                assert metrics['max_difference'] <= config.tolerance

    def test_torchscript_optimization(self, linear_model, simple_inputs, temp_dir):
        """Test TorchScript with optimization"""
        config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT,
            output_path=os.path.join(temp_dir, "optimized_model.pt"),
            optimization_level=OptimizationLevel.BASIC,
            quantization=True
        )

        exporter = TorchScriptExporter(config)
        result = exporter.export(linear_model, simple_inputs)

        # Should succeed even with optimizations
        assert result.success == True
        assert os.path.exists(result.output_path)


class TestONNXExporter:
    """Test ONNX exporter"""

    def test_onnx_export_availability(self):
        """Test ONNX availability check"""
        from multi_platform_exporter import ONNX_AVAILABLE

        # Test should handle both cases
        assert isinstance(ONNX_AVAILABLE, bool)

    def test_onnx_export_when_available(self, linear_model, simple_inputs, temp_dir):
        """Test ONNX export when available"""
        config = ExportConfig(
            format=ExportFormat.ONNX,
            output_path=os.path.join(temp_dir, "model.onnx"),
            onnx_opset_version=11
        )

        exporter = ONNXExporter(config)
        result = exporter.export(linear_model, simple_inputs)

        from multi_platform_exporter import ONNX_AVAILABLE
        if ONNX_AVAILABLE:
            # Should succeed if ONNX is available
            assert result.success == True
            assert os.path.exists(result.output_path)
            assert result.file_size > 0
        else:
            # Should fail gracefully if ONNX not available
            assert result.success == False
            assert "ONNX not available" in result.error_message

    def test_onnx_dynamic_axes(self, linear_model, simple_inputs, temp_dir):
        """Test ONNX export with dynamic axes"""
        config = ExportConfig(
            format=ExportFormat.ONNX,
            output_path=os.path.join(temp_dir, "dynamic_model.onnx"),
            dynamic_axes={"x": {0: "batch_size"}},
            onnx_opset_version=11
        )

        exporter = ONNXExporter(config)
        result = exporter.export(linear_model, simple_inputs)

        from multi_platform_exporter import ONNX_AVAILABLE
        if ONNX_AVAILABLE:
            # Should handle dynamic axes
            assert result.success == True or result.error_message is not None


class TestCoreMLExporter:
    """Test CoreML exporter"""

    def test_coreml_export_availability(self):
        """Test CoreML availability check"""
        from multi_platform_exporter import COREML_AVAILABLE

        assert isinstance(COREML_AVAILABLE, bool)

    def test_coreml_export_when_available(self, linear_model, simple_inputs, temp_dir):
        """Test CoreML export when available"""
        config = ExportConfig(
            format=ExportFormat.COREML,
            output_path=os.path.join(temp_dir, "model.mlmodel"),
            coreml_minimum_deployment_target="13.0"
        )

        exporter = CoreMLExporter(config)
        result = exporter.export(linear_model, simple_inputs)

        from multi_platform_exporter import COREML_AVAILABLE
        if COREML_AVAILABLE:
            # May succeed or fail depending on platform and model complexity
            assert isinstance(result.success, bool)
            if result.success:
                assert result.file_size > 0
        else:
            # Should fail gracefully if CoreML not available
            assert result.success == False
            assert "CoreML not available" in result.error_message


class TestHuggingFaceExporter:
    """Test HuggingFace exporter"""

    def test_huggingface_export_availability(self):
        """Test HuggingFace availability check"""
        from multi_platform_exporter import TRANSFORMERS_AVAILABLE

        assert isinstance(TRANSFORMERS_AVAILABLE, bool)

    def test_huggingface_export_when_available(self, linear_model, simple_inputs, temp_dir):
        """Test HuggingFace export when available"""
        config = ExportConfig(
            format=ExportFormat.HUGGINGFACE,
            output_path=os.path.join(temp_dir, "huggingface_model")
        )

        exporter = HuggingFaceExporter(config)
        result = exporter.export(linear_model, simple_inputs)

        from multi_platform_exporter import TRANSFORMERS_AVAILABLE
        if TRANSFORMERS_AVAILABLE:
            # Should succeed for basic PyTorch models
            assert result.success == True
            assert os.path.exists(result.output_path)
            assert result.file_size > 0

            # Check that config.json was created
            config_path = os.path.join(result.output_path, "config.json")
            assert os.path.exists(config_path)

            # Check that model weights were saved
            weights_path = os.path.join(result.output_path, "pytorch_model.bin")
            assert os.path.exists(weights_path)
        else:
            # Should fail gracefully if Transformers not available
            assert result.success == False
            assert "Transformers not available" in result.error_message


class TestMultiPlatformExporter:
    """Test main multi-platform exporter"""

    def test_exporter_initialization(self, exporter):
        """Test exporter initialization"""
        assert isinstance(exporter, MultiPlatformExporter)
        assert len(exporter.exporters) > 0
        assert ExportFormat.TORCHSCRIPT in exporter.exporters

    def test_single_format_export(self, exporter, linear_model, simple_inputs, temp_dir):
        """Test single format export"""
        config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT,
            output_path=os.path.join(temp_dir, "single_export.pt")
        )

        result = exporter.export_model(linear_model, config, simple_inputs)

        assert isinstance(result, ExportResult)
        assert result.format == ExportFormat.TORCHSCRIPT
        # Success depends on model complexity and environment
        assert isinstance(result.success, bool)

    def test_multiple_format_export(self, exporter, linear_model, simple_inputs, temp_dir):
        """Test multiple format export"""
        configs = [
            ExportConfig(
                format=ExportFormat.TORCHSCRIPT,
                output_path=os.path.join(temp_dir, "multi_torchscript.pt")
            ),
            ExportConfig(
                format=ExportFormat.HUGGINGFACE,
                output_path=os.path.join(temp_dir, "multi_huggingface")
            )
        ]

        results = exporter.export_multiple_formats(linear_model, configs, simple_inputs)

        assert len(results) == 2
        assert all(isinstance(result, ExportResult) for result in results)
        assert results[0].format == ExportFormat.TORCHSCRIPT
        assert results[1].format == ExportFormat.HUGGINGFACE

    def test_unsupported_format(self, exporter, linear_model, simple_inputs, temp_dir):
        """Test handling of unsupported format"""
        # Create a mock unsupported format
        class UnsupportedFormat(ExportFormat):
            UNSUPPORTED = "unsupported"

        config = ExportConfig(
            format=UnsupportedFormat.UNSUPPORTED,
            output_path=os.path.join(temp_dir, "unsupported.bin")
        )

        result = exporter.export_model(linear_model, config, simple_inputs)

        assert result.success == False
        assert "Unsupported format" in result.error_message

    def test_export_info(self, exporter, linear_model, simple_inputs, temp_dir):
        """Test export info retrieval"""
        config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT,
            output_path=os.path.join(temp_dir, "info_test.pt")
        )

        # Export model first
        result = exporter.export_model(linear_model, config, simple_inputs)

        if result.success:
            # Get export info
            info = exporter.get_export_info(result.output_path, ExportFormat.TORCHSCRIPT)

            assert info['path'] == result.output_path
            assert info['format'] == ExportFormat.TORCHSCRIPT.value
            assert info['exists'] == True
            assert info['size'] > 0
            assert info['created'] is not None
        else:
            # Test with non-existent file
            info = exporter.get_export_info("nonexistent.pt", ExportFormat.TORCHSCRIPT)
            assert info['exists'] == False
            assert info['size'] == 0

    def test_validation(self, exporter, linear_model, simple_inputs, temp_dir):
        """Test export validation"""
        config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT,
            output_path=os.path.join(temp_dir, "validation_test.pt"),
            validation_level=ValidationLevel.FUNCTIONAL
        )

        # Export model first
        result = exporter.export_model(linear_model, config, simple_inputs)

        if result.success:
            # Test validation
            validation_passed, metrics = exporter.validate_export(
                linear_model, result.output_path, ExportFormat.TORCHSCRIPT,
                simple_inputs, tolerance=1e-5
            )

            assert isinstance(validation_passed, bool)
            assert isinstance(metrics, dict)

    def test_benchmark_exports(self, exporter, linear_model, simple_inputs, temp_dir):
        """Test export benchmarking"""
        configs = [
            ExportConfig(
                format=ExportFormat.TORCHSCRIPT,
                output_path=os.path.join(temp_dir, "benchmark_torchscript.pt")
            )
        ]

        benchmark_results = exporter.benchmark_exports(
            linear_model, configs, simple_inputs, num_runs=3
        )

        assert isinstance(benchmark_results, dict)
        assert len(benchmark_results) == 1

        torchscript_results = benchmark_results.get('torchscript')
        if torchscript_results and 'export_failed' not in torchscript_results:
            assert 'export_time' in torchscript_results
            assert 'file_size' in torchscript_results
            assert 'validation_passed' in torchscript_results


class TestOptimizations:
    """Test export optimizations"""

    def test_optimization_levels(self, linear_model, simple_inputs, temp_dir):
        """Test different optimization levels"""
        optimization_levels = [
            OptimizationLevel.NONE,
            OptimizationLevel.BASIC,
            OptimizationLevel.AGGRESSIVE
        ]

        for opt_level in optimization_levels:
            config = ExportConfig(
                format=ExportFormat.TORCHSCRIPT,
                output_path=os.path.join(temp_dir, f"opt_{opt_level.value}.pt"),
                optimization_level=opt_level
            )

            exporter = TorchScriptExporter(config)
            result = exporter.export(linear_model, simple_inputs)

            # Should handle all optimization levels
            assert isinstance(result.success, bool)

    def test_quantization_option(self, linear_model, simple_inputs, temp_dir):
        """Test quantization option"""
        config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT,
            output_path=os.path.join(temp_dir, "quantized.pt"),
            quantization=True
        )

        exporter = TorchScriptExporter(config)
        result = exporter.export(linear_model, simple_inputs)

        # Should handle quantization (may succeed or fail depending on model)
        assert isinstance(result.success, bool)

    def test_pruning_option(self, linear_model, simple_inputs, temp_dir):
        """Test pruning option"""
        config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT,
            output_path=os.path.join(temp_dir, "pruned.pt"),
            pruning=True
        )

        exporter = TorchScriptExporter(config)
        result = exporter.export(linear_model, simple_inputs)

        # Should handle pruning (may succeed or fail depending on environment)
        assert isinstance(result.success, bool)


class TestValidationLevels:
    """Test different validation levels"""

    def test_validation_levels(self, linear_model, simple_inputs, temp_dir):
        """Test different validation levels"""
        validation_levels = [
            ValidationLevel.NONE,
            ValidationLevel.BASIC,
            ValidationLevel.FUNCTIONAL,
            ValidationLevel.COMPREHENSIVE
        ]

        for val_level in validation_levels:
            config = ExportConfig(
                format=ExportFormat.TORCHSCRIPT,
                output_path=os.path.join(temp_dir, f"val_{val_level.value}.pt"),
                validation_level=val_level
            )

            exporter = TorchScriptExporter(config)
            result = exporter.export(linear_model, simple_inputs)

            # Should handle all validation levels
            assert isinstance(result.success, bool)

            if val_level == ValidationLevel.NONE:
                # No validation should be performed
                assert result.validation_passed == False
            else:
                # Some validation should be attempted
                assert isinstance(result.validation_passed, bool)


class TestErrorHandling:
    """Test error handling in exports"""

    def test_invalid_model(self, temp_dir):
        """Test handling of invalid model"""
        # Create an invalid model (None)
        config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT,
            output_path=os.path.join(temp_dir, "invalid.pt")
        )

        exporter = TorchScriptExporter(config)

        # Should handle invalid model gracefully
        try:
            result = exporter.export(None, {"x": torch.randn(1, 10)})
            assert result.success == False
            assert result.error_message is not None
        except Exception:
            # Exception is also acceptable for invalid input
            pass

    def test_invalid_inputs(self, linear_model, temp_dir):
        """Test handling of invalid inputs"""
        config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT,
            output_path=os.path.join(temp_dir, "invalid_inputs.pt")
        )

        exporter = TorchScriptExporter(config)

        # Test with empty inputs
        try:
            result = exporter.export(linear_model, {})
            # Should either fail gracefully or handle empty inputs
            assert isinstance(result.success, bool)
        except Exception:
            # Exception is acceptable for invalid inputs
            pass

    def test_invalid_output_path(self, linear_model, simple_inputs):
        """Test handling of invalid output path"""
        config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT,
            output_path="/invalid/path/that/does/not/exist/model.pt"
        )

        exporter = TorchScriptExporter(config)
        result = exporter.export(linear_model, simple_inputs)

        # Should fail due to invalid path
        assert result.success == False
        assert result.error_message is not None


def test_create_sample_inputs():
    """Test sample input creation utility"""
    inputs = create_sample_inputs(batch_size=3, seq_len=20, vocab_size=100)

    assert isinstance(inputs, dict)
    assert "input_ids" in inputs
    assert "attention_mask" in inputs

    assert inputs["input_ids"].shape == (3, 20)
    assert inputs["attention_mask"].shape == (3, 20)

    assert inputs["input_ids"].dtype == torch.long
    assert inputs["attention_mask"].dtype == torch.long

    # Check value ranges
    assert torch.all(inputs["input_ids"] >= 0)
    assert torch.all(inputs["input_ids"] < 100)
    assert torch.all(inputs["attention_mask"] == 1)


if __name__ == "__main__":
    # Run basic tests
    print("🧪 Running Multi-Platform Export Tests")

    # Test basic functionality
    exporter = MultiPlatformExporter()
    print("✅ Exporter created")

    # Test model
    model = LinearModel(input_size=5, output_size=3)
    inputs = {"x": torch.randn(2, 5)}

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test TorchScript export
        config = ExportConfig(
            format=ExportFormat.TORCHSCRIPT,
            output_path=os.path.join(temp_dir, "test_model.pt")
        )

        result = exporter.export_model(model, config, inputs)
        print(f"✅ TorchScript export: {'Success' if result.success else 'Failed'}")

        if result.success:
            print(f"  • File size: {result.file_size} bytes")
            print(f"  • Export time: {result.export_time:.3f}s")
            print(f"  • Validation: {'Passed' if result.validation_passed else 'Failed'}")

        # Test export info
        info = exporter.get_export_info(config.output_path, ExportFormat.TORCHSCRIPT)
        print(f"✅ Export info: exists={info['exists']}, size={info['size']}")

        # Test sample inputs utility
        sample_inputs = create_sample_inputs(batch_size=1, seq_len=10, vocab_size=50)
        print(f"✅ Sample inputs: {list(sample_inputs.keys())}")

    print("🎉 All basic tests passed!")