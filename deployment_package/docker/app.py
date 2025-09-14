#!/usr/bin/env python3
"""
NanoLM Model Serving Application
===============================

High-performance model serving with health checks, metrics, and monitoring.
"""

import os
import time
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path

# Web framework imports
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    from flask import Flask, request, jsonify
    FASTAPI_AVAILABLE = False

# Model loading imports
import torch
import numpy as np

# Optional imports based on export format
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False


class ModelServer:
    """High-performance model serving class"""

    def __init__(self):
        self.model = None
        self.model_path = os.getenv("MODEL_PATH", "./model")
        self.port = int(os.getenv("PORT", "8080"))
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Performance metrics
        self.request_count = 0
        self.total_inference_time = 0.0
        self.startup_time = time.time()

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the exported model"""
        try:
            model_files = list(Path(self.model_path).glob("*"))

            if not model_files:
                raise FileNotFoundError(f"No model files found in {self.model_path}")

            # Determine model format and load accordingly
            if any(f.suffix == '.pt' for f in model_files):
                # TorchScript model
                model_file = next(f for f in model_files if f.suffix == '.pt')
                self.model = torch.jit.load(str(model_file))
                self.model.eval()
                self.model_type = "torchscript"

            elif any(f.suffix == '.onnx' for f in model_files) and ONNX_AVAILABLE:
                # ONNX model
                model_file = next(f for f in model_files if f.suffix == '.onnx')
                self.model = ort.InferenceSession(str(model_file))
                self.model_type = "onnx"

            elif any(f.suffix == '.mlmodel' for f in model_files) and COREML_AVAILABLE:
                # CoreML model
                model_file = next(f for f in model_files if f.suffix == '.mlmodel')
                self.model = ct.models.MLModel(str(model_file))
                self.model_type = "coreml"

            else:
                raise ValueError("Unsupported model format or missing dependencies")

            self.logger.info(f"✅ Model loaded successfully ({self.model_type})")

        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise

    def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run model inference"""
        start_time = time.time()

        try:
            # Convert inputs to appropriate format
            if self.model_type == "torchscript":
                # Convert to tensors
                tensor_inputs = {}
                for key, value in inputs.items():
                    if isinstance(value, list):
                        tensor_inputs[key] = torch.tensor(value, dtype=torch.float32)
                    else:
                        tensor_inputs[key] = torch.tensor([value], dtype=torch.float32)

                # Run inference
                with torch.no_grad():
                    if len(tensor_inputs) == 1:
                        input_tensor = next(iter(tensor_inputs.values()))
                        output = self.model(input_tensor)
                    else:
                        output = self.model(**tensor_inputs)

                # Convert output to list
                if isinstance(output, torch.Tensor):
                    result = output.cpu().numpy().tolist()
                else:
                    result = [o.cpu().numpy().tolist() for o in output]

            elif self.model_type == "onnx":
                # Convert to numpy arrays
                numpy_inputs = {}
                for key, value in inputs.items():
                    if isinstance(value, list):
                        numpy_inputs[key] = np.array(value, dtype=np.float32)
                    else:
                        numpy_inputs[key] = np.array([value], dtype=np.float32)

                # Run inference
                outputs = self.model.run(None, numpy_inputs)
                result = [output.tolist() for output in outputs]

            elif self.model_type == "coreml":
                # CoreML prediction
                outputs = self.model.predict(inputs)
                result = list(outputs.values())

            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            # Update metrics
            inference_time = time.time() - start_time
            self.request_count += 1
            self.total_inference_time += inference_time

            return {
                "predictions": result,
                "inference_time_ms": inference_time * 1000,
                "model_type": self.model_type
            }

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

    def get_health(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "status": "healthy",
            "model_loaded": self.model is not None,
            "model_type": getattr(self, 'model_type', 'unknown'),
            "uptime_seconds": time.time() - self.startup_time,
            "request_count": self.request_count
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Metrics endpoint"""
        avg_inference_time = (
            self.total_inference_time / self.request_count
            if self.request_count > 0 else 0
        )

        return {
            "request_count": self.request_count,
            "total_inference_time_seconds": self.total_inference_time,
            "average_inference_time_ms": avg_inference_time * 1000,
            "uptime_seconds": time.time() - self.startup_time,
            "model_type": getattr(self, 'model_type', 'unknown')
        }


# Initialize model server
model_server = ModelServer()

if FASTAPI_AVAILABLE:
    # FastAPI application
    app = FastAPI(
        title="NanoLM Model Server",
        description="High-performance model serving API",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/predict")
    async def predict(request: Dict[str, Any]):
        try:
            result = model_server.predict(request)
            return JSONResponse(content=result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        return model_server.get_health()

    @app.get("/metrics")
    async def metrics():
        return model_server.get_metrics()

    if __name__ == "__main__":
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=model_server.port,
            log_level=model_server.log_level.lower()
        )

else:
    # Flask fallback
    app = Flask(__name__)

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            data = request.get_json()
            result = model_server.predict(data)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/health")
    def health():
        return jsonify(model_server.get_health())

    @app.route("/metrics")
    def metrics():
        return jsonify(model_server.get_metrics())

    if __name__ == "__main__":
        app.run(
            host="0.0.0.0",
            port=model_server.port,
            debug=(model_server.log_level == "DEBUG")
        )
