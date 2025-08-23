#!/usr/bin/env python3
"""
Edge Device Validation System
=============================

Comprehensive validation system for edge device deployments including:
- Hardware compatibility testing
- Performance validation
- Memory usage validation
- Power consumption estimation
- Accuracy preservation testing
"""

import torch
import torch.nn as nn
import time
import logging
import psutil
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class ValidationResult(Enum):
    """Validation result status"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"


@dataclass
class EdgeValidationReport:
    """Comprehensive validation report for edge deployment"""
    device_target: str
    model_name: str
    validation_timestamp: float

    # Hardware compatibility
    hardware_compatible: ValidationResult = ValidationResult.SKIP
    memory_compatible: ValidationResult = ValidationResult.SKIP
    compute_compatible: ValidationResult = ValidationResult.SKIP

    # Performance metrics
    inference_time_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_utilization_percent: float = 0.0

    # Accuracy metrics
    accuracy_preserved: ValidationResult = ValidationResult.SKIP
    output_similarity: float = 0.0

    # Power estimation
    estimated_power_mw: float = 0.0
    battery_life_hours: float = 0.0

    # Overall assessment
    deployment_ready: bool = False
    confidence_score: float = 0.0
    recommendations: List[str] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class EdgeDeviceValidator:
    """Validates models for edge device deployment"""

    def __init__(self, device_specs: Dict[str, Any]):
        self.device_specs = device_specs
        self.logger = logging.getLogger(self.__class__.__name__)

        # Default device specifications if not provided
        self.default_specs = {
            "max_memory_mb": 256,
            "max_cpu_cores": 4,
            "has_gpu": False,
            "has_npu": False,
            "max_power_mw": 5000,
            "battery_capacity_mah": 3000,
            "target_inference_time_ms": 100.0
        }

        # Merge with provided specs
        for key, default_value in self.default_specs.items():
            if key not in self.device_specs:
                self.device_specs[key] = default_value

        logging.info("✅ Edge Device Validator initialized")
        logging.info(f"  • Max memory: {self.device_specs['max_memory_mb']}MB")
        logging.info(f"  • CPU cores: {self.device_specs['max_cpu_cores']}")
        logging.info(f"  • Target latency: {self.device_specs['target_inference_time_ms']}ms")

    def validate_model(self, model: nn.Module, sample_inputs: Dict[str, torch.Tensor],
                      original_model: Optional[nn.Module] = None,
                      device_target: str = "generic_edge") -> EdgeValidationReport:
        """Comprehensive model validation for edge deployment"""

        self.logger.info(f"🔍 Validating model for {device_target}")

        report = EdgeValidationReport(
            device_target=device_target,
            model_name=model.__class__.__name__,
            validation_timestamp=time.time()
        )

        # Hardware compatibility validation
        report.hardware_compatible = self._validate_hardware_compatibility(model)
        report.memory_compatible = self._validate_memory_compatibility(model)
        report.compute_compatible = self._validate_compute_compatibility(model)

        # Performance validation
        performance_metrics = self._validate_performance(model, sample_inputs)
        report.inference_time_ms = performance_metrics["inference_time_ms"]
        report.throughput_samples_per_sec = performance_metrics["throughput"]
        report.memory_usage_mb = performance_metrics["memory_usage_mb"]
        report.cpu_utilization_percent = performance_metrics["cpu_utilization"]

        # Accuracy validation (if original model provided)
        if original_model:
            accuracy_metrics = self._validate_accuracy(model, original_model, sample_inputs)
            report.accuracy_preserved = accuracy_metrics["accuracy_preserved"]
            report.output_similarity = accuracy_metrics["similarity"]

        # Power estimation
        power_metrics = self._estimate_power_consumption(model, sample_inputs)
        report.estimated_power_mw = power_metrics["power_mw"]
        report.battery_life_hours = power_metrics["battery_life_hours"]

        # Overall assessment
        report.deployment_ready, report.confidence_score, report.recommendations = \
            self._assess_deployment_readiness(report)

        self.logger.info(f"✅ Validation completed for {device_target}")
        self.logger.info(f"  • Deployment ready: {'✅' if report.deployment_ready else '❌'}")
        self.logger.info(f"  • Confidence: {report.confidence_score:.2f}")

        return report