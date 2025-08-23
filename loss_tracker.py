#!/usr/bin/env python3
"""
Advanced Loss Tracking and Prediction System for NanoLM
======================================================

Implements comprehensive loss tracking and prediction including:
- 4 prediction methods: exponential decay, power law, trend, learning curve
- Ensemble prediction with confidence scores
- Convergence detection and early stopping recommendations
- Loss visualization and reporting
"""

import torch
import numpy as np
import math
import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime


class PredictionMethod(Enum):
    """Loss prediction methods"""
    EXPONENTIAL_DECAY = "exponential_decay"
    POWER_LAW = "power_law"
    TREND_ANALYSIS = "trend_analysis"
    LEARNING_CURVE = "learning_curve"


class ConvergenceStatus(Enum):
    """Convergence status indicators"""
    CONVERGING = "converging"
    CONVERGED = "converged"
    DIVERGING = "diverging"
    STAGNANT = "stagnant"
    OSCILLATING = "oscillating"
    UNKNOWN = "unknown"


@dataclass
class PredictionResult:
    """Result from a single prediction method"""
    method: PredictionMethod
    predicted_loss: float
    confidence: float
    steps_to_target: Optional[int]
    convergence_estimate: Optional[float]
    method_params: Dict[str, Any]
    r_squared: Optional[float] = None
    mse: Optional[float] = None


@dataclass
class EnsemblePrediction:
    """Ensemble prediction combining multiple methods"""
    predicted_loss: float
    confidence: float
    individual_predictions: List[PredictionResult]
    consensus_score: float
    steps_to_target: Optional[int]
    convergence_probability: float
    recommended_action: str


@dataclass
class ConvergenceAnalysis:
    """Convergence analysis results"""
    status: ConvergenceStatus
    convergence_rate: float
    time_to_convergence: Optional[int]
    plateau_detection: bool
    oscillation_amplitude: float
    trend_strength: float
    stability_score: float
    early_stopping_recommendation: bool


@dataclass
class LossReport:
    """Comprehensive loss tracking report"""
    timestamp: str
    current_step: int
    current_loss: float
    loss_history: List[float]
    prediction: EnsemblePrediction
    convergence: ConvergenceAnalysis
    statistics: Dict[str, Any]
    recommendations: List[str]


class ExponentialDecayPredictor:
    """Exponential decay loss prediction"""

    def __init__(self, min_samples: int = 10):
        self.min_samples = min_samples
        self.name = "Exponential Decay"

    def predict(self, loss_history: List[float], target_loss: Optional[float] = None) -> PredictionResult:
        """Predict using exponential decay model: L(t) = L0 * exp(-λt) + L∞"""
        if len(loss_history) < self.min_samples:
            return PredictionResult(
                method=PredictionMethod.EXPONENTIAL_DECAY,
                predicted_loss=loss_history[-1] if loss_history else 0.0,
                confidence=0.0,
                steps_to_target=None,
                convergence_estimate=None,
                method_params={}
            )

        try:
            # Prepare data
            x = np.arange(len(loss_history))
            y = np.array(loss_history)

            # Remove invalid values
            valid_mask = np.isfinite(y) & (y > 0)
            x = x[valid_mask]
            y = y[valid_mask]

            if len(y) < self.min_samples:
                return PredictionResult(
                    method=PredictionMethod.EXPONENTIAL_DECAY,
                    predicted_loss=loss_history[-1],
                    confidence=0.0,
                    steps_to_target=None,
                    convergence_estimate=None,
                    method_params={}
                )

            # Estimate asymptote
            c_estimate = np.min(y[-min(10, len(y)):])
            y_shifted = y - c_estimate + 1e-8

            # Linear regression on log-transformed data
            log_y = np.log(y_shifted)
            A = np.vstack([x, np.ones(len(x))]).T
            coeffs, residuals, rank, s = np.linalg.lstsq(A, log_y, rcond=None)

            b = -coeffs[0]  # decay rate
            a = np.exp(coeffs[1])
            c = c_estimate

            # Calculate R-squared
            y_pred = a * np.exp(-b * x) + c
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Predict next loss
            next_step = len(loss_history)
            predicted_loss = a * np.exp(-b * next_step) + c

            # Calculate confidence
            confidence = max(0.0, min(1.0, r_squared * 0.8 + 0.2))

            # Steps to target
            steps_to_target = None
            if target_loss is not None and b > 0 and predicted_loss > target_loss:
                steps_remaining = -np.log((target_loss - c) / a) / b - next_step
                steps_to_target = max(0, int(steps_remaining)) if steps_remaining > 0 else None

            return PredictionResult(
                method=PredictionMethod.EXPONENTIAL_DECAY,
                predicted_loss=float(predicted_loss),
                confidence=float(confidence),
                steps_to_target=steps_to_target,
                convergence_estimate=float(c),
                method_params={'a': float(a), 'b': float(b), 'c': float(c)},
                r_squared=float(r_squared),
                mse=float(np.mean((y - y_pred) ** 2))
            )

        except Exception as e:
            logging.warning(f"Exponential decay prediction failed: {e}")
            return PredictionResult(
                method=PredictionMethod.EXPONENTIAL_DECAY,
                predicted_loss=loss_history[-1],
                confidence=0.0,
                steps_to_target=None,
                convergence_estimate=None,
                method_params={}
            )


class PowerLawPredictor:
    """Power law loss prediction"""

    def __init__(self, min_samples: int = 15):
        self.min_samples = min_samples
        self.name = "Power Law"

    def predict(self, loss_history: List[float], target_loss: Optional[float] = None) -> PredictionResult:
        """Predict using power law model: L(t) = a * t^(-b) + c"""
        if len(loss_history) < self.min_samples:
            return PredictionResult(
                method=PredictionMethod.POWER_LAW,
                predicted_loss=loss_history[-1] if loss_history else 0.0,
                confidence=0.0,
                steps_to_target=None,
                convergence_estimate=None,
                method_params={}
            )

        try:
            # Prepare data (start from step 1 to avoid log(0))
            x = np.arange(1, len(loss_history) + 1)
            y = np.array(loss_history)

            # Remove invalid values
            valid_mask = np.isfinite(y) & (y > 0)
            x = x[valid_mask]
            y = y[valid_mask]

            if len(y) < self.min_samples:
                return PredictionResult(
                    method=PredictionMethod.POWER_LAW,
                    predicted_loss=loss_history[-1],
                    confidence=0.0,
                    steps_to_target=None,
                    convergence_estimate=None,
                    method_params={}
                )

            # Estimate asymptote
            c_estimate = np.min(y[-min(10, len(y)):])
            y_shifted = y - c_estimate + 1e-8

            # Transform to linear: ln(y - c) = ln(a) - b * ln(x)
            log_x = np.log(x)
            log_y = np.log(y_shifted)

            # Linear regression
            A = np.vstack([log_x, np.ones(len(log_x))]).T
            coeffs, residuals, rank, s = np.linalg.lstsq(A, log_y, rcond=None)

            b = -coeffs[0]  # power law exponent
            a = np.exp(coeffs[1])
            c = c_estimate

            # Calculate R-squared
            y_pred = a * (x ** (-b)) + c
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Predict next loss
            next_step = len(loss_history) + 1
            predicted_loss = a * (next_step ** (-b)) + c

            # Confidence
            confidence = max(0.0, min(1.0, r_squared * 0.7 + 0.3)) if b > 0 else 0.0

            return PredictionResult(
                method=PredictionMethod.POWER_LAW,
                predicted_loss=float(predicted_loss),
                confidence=float(confidence),
                steps_to_target=None,
                convergence_estimate=float(c),
                method_params={'a': float(a), 'b': float(b), 'c': float(c)},
                r_squared=float(r_squared),
                mse=float(np.mean((y - y_pred) ** 2))
            )

        except Exception as e:
            logging.warning(f"Power law prediction failed: {e}")
            return PredictionResult(
                method=PredictionMethod.POWER_LAW,
                predicted_loss=loss_history[-1],
                confidence=0.0,
                steps_to_target=None,
                convergence_estimate=None,
                method_params={}
            )


class TrendAnalysisPredictor:
    """Trend analysis loss prediction"""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.name = "Trend Analysis"

    def predict(self, loss_history: List[float], target_loss: Optional[float] = None) -> PredictionResult:
        """Predict using trend analysis with multiple time scales"""
        if len(loss_history) < 5:
            return PredictionResult(
                method=PredictionMethod.TREND_ANALYSIS,
                predicted_loss=loss_history[-1] if loss_history else 0.0,
                confidence=0.0,
                steps_to_target=None,
                convergence_estimate=None,
                method_params={}
            )

        try:
            y = np.array(loss_history)
            valid_mask = np.isfinite(y)
            y = y[valid_mask]

            if len(y) < 5:
                return PredictionResult(
                    method=PredictionMethod.TREND_ANALYSIS,
                    predicted_loss=loss_history[-1],
                    confidence=0.0,
                    steps_to_target=None,
                    convergence_estimate=None,
                    method_params={}
                )

            # Multi-scale trend analysis
            trends = {}
            confidences = {}

            # Short-term trend
            short_window = max(5, min(20, len(y) // 5))
            short_data = y[-short_window:]
            short_x = np.arange(len(short_data))
            short_trend = np.polyfit(short_x, short_data, 1)[0]
            short_r2 = self._calculate_r_squared(short_x, short_data, 1)
            trends['short'] = short_trend
            confidences['short'] = max(0.0, short_r2)

            # Medium-term trend
            if len(y) > 10:
                med_window = max(10, min(self.window_size, len(y) // 2))
                med_data = y[-med_window:]
                med_x = np.arange(len(med_data))
                med_trend = np.polyfit(med_x, med_data, 1)[0]
                med_r2 = self._calculate_r_squared(med_x, med_data, 1)
                trends['medium'] = med_trend
                confidences['medium'] = max(0.0, med_r2)

            # Weighted ensemble of trends
            total_weight = sum(confidences.values())
            if total_weight > 0:
                weighted_trend = sum(trends[k] * confidences[k] for k in trends) / total_weight
                overall_confidence = np.mean(list(confidences.values()))
            else:
                weighted_trend = 0.0
                overall_confidence = 0.0

            # Predict next loss
            predicted_loss = max(0.0, y[-1] + weighted_trend)

            return PredictionResult(
                method=PredictionMethod.TREND_ANALYSIS,
                predicted_loss=float(predicted_loss),
                confidence=float(overall_confidence),
                steps_to_target=None,
                convergence_estimate=None,
                method_params={
                    'weighted_trend': float(weighted_trend),
                    'trends': {k: float(v) for k, v in trends.items()},
                    'confidences': {k: float(v) for k, v in confidences.items()}
                }
            )

        except Exception as e:
            logging.warning(f"Trend analysis prediction failed: {e}")
            return PredictionResult(
                method=PredictionMethod.TREND_ANALYSIS,
                predicted_loss=loss_history[-1],
                confidence=0.0,
                steps_to_target=None,
                convergence_estimate=None,
                method_params={}
            )

    def _calculate_r_squared(self, x: np.ndarray, y: np.ndarray, degree: int) -> float:
        """Calculate R-squared for polynomial fit"""
        try:
            coeffs = np.polyfit(x, y, degree)
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        except:
            return 0.0


class LearningCurvePredictor:
    """Learning curve-based loss prediction"""

    def __init__(self, min_samples: int = 20):
        self.min_samples = min_samples
        self.name = "Learning Curve"

    def predict(self, loss_history: List[float], target_loss: Optional[float] = None) -> PredictionResult:
        """Predict using learning curve analysis with plateau detection"""
        if len(loss_history) < self.min_samples:
            return PredictionResult(
                method=PredictionMethod.LEARNING_CURVE,
                predicted_loss=loss_history[-1] if loss_history else 0.0,
                confidence=0.0,
                steps_to_target=None,
                convergence_estimate=None,
                method_params={}
            )

        try:
            y = np.array(loss_history)
            valid_mask = np.isfinite(y)
            y = y[valid_mask]

            if len(y) < self.min_samples:
                return PredictionResult(
                    method=PredictionMethod.LEARNING_CURVE,
                    predicted_loss=loss_history[-1],
                    confidence=0.0,
                    steps_to_target=None,
                    convergence_estimate=None,
                    method_params={}
                )

            # Detect learning phases
            phases = self._detect_learning_phases(y)

            # Fit model based on detected phases
            if phases['plateau_detected']:
                predicted_loss, confidence, convergence_est = self._plateau_model(y)
            elif phases['exponential_phase']:
                predicted_loss, confidence, convergence_est = self._exponential_phase_model(y)
            else:
                predicted_loss, confidence, convergence_est = self._polynomial_model(y)

            return PredictionResult(
                method=PredictionMethod.LEARNING_CURVE,
                predicted_loss=float(predicted_loss),
                confidence=float(confidence),
                steps_to_target=None,
                convergence_estimate=convergence_est,
                method_params={
                    'phases': phases,
                    'learning_rate': float(np.mean(np.diff(y[-10:]))) if len(y) > 10 else 0.0
                }
            )

        except Exception as e:
            logging.warning(f"Learning curve prediction failed: {e}")
            return PredictionResult(
                method=PredictionMethod.LEARNING_CURVE,
                predicted_loss=loss_history[-1],
                confidence=0.0,
                steps_to_target=None,
                convergence_estimate=None,
                method_params={}
            )

    def _detect_learning_phases(self, y: np.ndarray) -> Dict[str, Any]:
        """Detect different phases in the learning curve"""
        phases = {
            'plateau_detected': False,
            'exponential_phase': False,
            'linear_phase': False,
            'oscillating': False
        }

        if len(y) < 10:
            return phases

        # Check for plateau
        recent_window = min(20, len(y) // 3)
        recent_data = y[-recent_window:]
        recent_std = np.std(recent_data)
        recent_mean = np.mean(recent_data)

        if recent_std / (recent_mean + 1e-8) < 0.05:
            phases['plateau_detected'] = True

        # Check for exponential phase
        if len(y) > 15:
            mid_point = len(y) // 2
            first_half_mean = np.mean(y[:mid_point])
            second_half_mean = np.mean(y[mid_point:])

            if first_half_mean > second_half_mean * 1.5:
                phases['exponential_phase'] = True

        return phases

    def _plateau_model(self, y: np.ndarray) -> Tuple[float, float, Optional[float]]:
        """Model for plateau phase"""
        recent_window = min(20, len(y) // 3)
        recent_data = y[-recent_window:]

        predicted_loss = np.mean(recent_data)
        confidence = 0.8
        convergence_estimate = predicted_loss

        return predicted_loss, confidence, convergence_estimate

    def _exponential_phase_model(self, y: np.ndarray) -> Tuple[float, float, Optional[float]]:
        """Model for exponential decay phase"""
        window_size = max(10, len(y) // 3)
        recent_data = y[-window_size:]

        return np.mean(recent_data[-3:]), 0.6, None

    def _polynomial_model(self, y: np.ndarray) -> Tuple[float, float, Optional[float]]:
        """Model using polynomial fit"""
        x = np.arange(len(y))

        try:
            coeffs = np.polyfit(x, y, 2)
            next_x = len(y)
            predicted_loss = max(0.0, np.polyval(coeffs, next_x))

            # Calculate R-squared
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            return predicted_loss, max(0.0, r_squared), None
        except:
            return y[-1], 0.3, None


class LossTracker:
    """Advanced loss tracking and prediction system"""

    def __init__(self, config):
        self.config = config
        self.loss_history = []
        self.step_history = []
        self.timestamp_history = []

        # Prediction methods
        self.predictors = {
            PredictionMethod.EXPONENTIAL_DECAY: ExponentialDecayPredictor(),
            PredictionMethod.POWER_LAW: PowerLawPredictor(),
            PredictionMethod.TREND_ANALYSIS: TrendAnalysisPredictor(),
            PredictionMethod.LEARNING_CURVE: LearningCurvePredictor()
        }

        # Configuration parameters
        self.convergence_window = getattr(config, 'convergence_window', 50)
        self.convergence_threshold = getattr(config, 'convergence_threshold', 1e-4)
        self.plateau_patience = getattr(config, 'plateau_patience', 100)
        self.target_loss = getattr(config, 'target_loss', 2.0)
        self.early_stopping_patience = getattr(config, 'early_stopping_patience', 200)
        self.min_improvement = getattr(config, 'min_improvement', 1e-3)

        logging.info(f"✅ Loss Tracker initialized with {len(self.predictors)} prediction methods")

    def update(self, loss: float, step: int, timestamp: Optional[str] = None):
        """Update loss history with new data point"""
        self.loss_history.append(loss)
        self.step_history.append(step)
        self.timestamp_history.append(timestamp or datetime.now().isoformat())

        # Keep history manageable (last 2000 points)
        if len(self.loss_history) > 2000:
            self.loss_history = self.loss_history[-2000:]
            self.step_history = self.step_history[-2000:]
            self.timestamp_history = self.timestamp_history[-2000:]

    def predict(self, target_loss: Optional[float] = None) -> EnsemblePrediction:
        """Generate ensemble prediction from all methods"""
        if not self.loss_history:
            return EnsemblePrediction(
                predicted_loss=0.0,
                confidence=0.0,
                individual_predictions=[],
                consensus_score=0.0,
                steps_to_target=None,
                convergence_probability=0.0,
                recommended_action="Insufficient data"
            )

        target = target_loss or self.target_loss
        individual_predictions = []

        # Get predictions from all methods
        for method, predictor in self.predictors.items():
            try:
                prediction = predictor.predict(self.loss_history, target)
                individual_predictions.append(prediction)
            except Exception as e:
                logging.warning(f"Prediction method {method} failed: {e}")

        if not individual_predictions:
            return EnsemblePrediction(
                predicted_loss=self.loss_history[-1],
                confidence=0.0,
                individual_predictions=[],
                consensus_score=0.0,
                steps_to_target=None,
                convergence_probability=0.0,
                recommended_action="All prediction methods failed"
            )

        # Ensemble prediction using weighted average
        total_weight = sum(pred.confidence for pred in individual_predictions)

        if total_weight > 0:
            weighted_prediction = sum(
                pred.predicted_loss * pred.confidence
                for pred in individual_predictions
            ) / total_weight
            ensemble_confidence = np.mean([pred.confidence for pred in individual_predictions])
        else:
            weighted_prediction = np.mean([pred.predicted_loss for pred in individual_predictions])
            ensemble_confidence = 0.0

        # Calculate consensus score
        predictions = [pred.predicted_loss for pred in individual_predictions if not np.isnan(pred.predicted_loss)]
        if len(predictions) > 1:
            pred_std = np.std(predictions)
            pred_mean = np.mean(predictions)
            if pred_mean > 0 and not np.isnan(pred_std) and not np.isnan(pred_mean):
                consensus_score = 1.0 / (1.0 + pred_std / (pred_mean + 1e-8))
            else:
                consensus_score = 0.0
        else:
            consensus_score = 1.0 if len(predictions) == 1 else 0.0

        # Convergence probability
        convergence_probability = self._calculate_convergence_probability(individual_predictions)

        # Recommended action
        recommended_action = self._generate_recommendation(
            weighted_prediction, ensemble_confidence, consensus_score, convergence_probability
        )

        return EnsemblePrediction(
            predicted_loss=float(weighted_prediction),
            confidence=float(ensemble_confidence),
            individual_predictions=individual_predictions,
            consensus_score=float(consensus_score),
            steps_to_target=None,
            convergence_probability=float(convergence_probability),
            recommended_action=recommended_action
        )

    def analyze_convergence(self) -> ConvergenceAnalysis:
        """Analyze convergence status and characteristics"""
        if len(self.loss_history) < 10:
            return ConvergenceAnalysis(
                status=ConvergenceStatus.UNKNOWN,
                convergence_rate=0.0,
                time_to_convergence=None,
                plateau_detection=False,
                oscillation_amplitude=0.0,
                trend_strength=0.0,
                stability_score=0.0,
                early_stopping_recommendation=False
            )

        y = np.array(self.loss_history)

        # Determine convergence status
        status = self._determine_convergence_status(y)

        # Calculate metrics
        convergence_rate = self._calculate_convergence_rate(y)
        time_to_convergence = self._estimate_time_to_convergence(y)
        plateau_detection = self._detect_plateau(y)
        oscillation_amplitude = self._calculate_oscillation_amplitude(y)
        trend_strength = self._calculate_trend_strength(y)
        stability_score = self._calculate_stability_score(y)
        early_stopping_recommendation = self._should_early_stop(y)

        return ConvergenceAnalysis(
            status=status,
            convergence_rate=convergence_rate,
            time_to_convergence=time_to_convergence,
            plateau_detection=plateau_detection,
            oscillation_amplitude=oscillation_amplitude,
            trend_strength=trend_strength,
            stability_score=stability_score,
            early_stopping_recommendation=early_stopping_recommendation
        )

    def generate_report(self) -> LossReport:
        """Generate comprehensive loss tracking report"""
        if not self.loss_history:
            return LossReport(
                timestamp=datetime.now().isoformat(),
                current_step=0,
                current_loss=0.0,
                loss_history=[],
                prediction=EnsemblePrediction(
                    predicted_loss=0.0,
                    confidence=0.0,
                    individual_predictions=[],
                    consensus_score=0.0,
                    steps_to_target=None,
                    convergence_probability=0.0,
                    recommended_action="No data"
                ),
                convergence=ConvergenceAnalysis(
                    status=ConvergenceStatus.UNKNOWN,
                    convergence_rate=0.0,
                    time_to_convergence=None,
                    plateau_detection=False,
                    oscillation_amplitude=0.0,
                    trend_strength=0.0,
                    stability_score=0.0,
                    early_stopping_recommendation=False
                ),
                statistics={},
                recommendations=[]
            )

        # Generate prediction and convergence analysis
        prediction = self.predict()
        convergence = self.analyze_convergence()

        # Calculate statistics
        statistics = self._calculate_statistics()

        # Generate recommendations
        recommendations = self._generate_recommendations(prediction, convergence)

        return LossReport(
            timestamp=datetime.now().isoformat(),
            current_step=self.step_history[-1],
            current_loss=self.loss_history[-1],
            loss_history=self.loss_history.copy(),
            prediction=prediction,
            convergence=convergence,
            statistics=statistics,
            recommendations=recommendations
        )

    def export_data(self, filepath: str) -> bool:
        """Export loss tracking data to JSON"""
        try:
            data = {
                'loss_history': self.loss_history,
                'step_history': self.step_history,
                'timestamp_history': self.timestamp_history,
                'config': {
                    'convergence_window': self.convergence_window,
                    'convergence_threshold': self.convergence_threshold,
                    'target_loss': self.target_loss
                },
                'export_timestamp': datetime.now().isoformat()
            }

            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            logging.info(f"Loss tracking data exported to {filepath}")
            return True

        except Exception as e:
            logging.error(f"Export failed: {e}")
            return False

    def import_data(self, filepath: str) -> bool:
        """Import loss tracking data from JSON"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.loss_history = data.get('loss_history', [])
            self.step_history = data.get('step_history', [])
            self.timestamp_history = data.get('timestamp_history', [])

            logging.info(f"Loss tracking data imported from {filepath}")
            return True

        except Exception as e:
            logging.error(f"Import failed: {e}")
            return False

# Helper methods for analysis
    def _calculate_convergence_probability(self, predictions: List[PredictionResult]) -> float:
        """Calculate probability of convergence based on predictions"""
        if not predictions:
            return 0.0

        factors = []

        # Consistent predictions across methods
        pred_values = [p.predicted_loss for p in predictions if p.confidence > 0.1]
        if len(pred_values) > 1:
            consistency = 1.0 / (1.0 + np.std(pred_values) / (np.mean(pred_values) + 1e-8))
            factors.append(consistency)

        # High confidence predictions
        avg_confidence = np.mean([p.confidence for p in predictions])
        factors.append(avg_confidence)

        # Decreasing trend in recent history
        if len(self.loss_history) > 10:
            recent_trend = np.polyfit(range(10), self.loss_history[-10:], 1)[0]
            trend_factor = max(0.0, min(1.0, -recent_trend * 10))
            factors.append(trend_factor)

        return np.mean(factors) if factors else 0.0

    def _generate_recommendation(self, predicted_loss: float, confidence: float,
                                consensus: float, convergence_prob: float) -> str:
        """Generate recommendation based on prediction analysis"""
        if confidence < 0.3:
            return "Insufficient data for reliable prediction. Continue training."

        if convergence_prob > 0.8 and consensus > 0.8:
            return "Strong convergence indicators. Training is progressing well."

        if convergence_prob < 0.3:
            return "Weak convergence signals. Consider adjusting learning rate or model architecture."

        if consensus < 0.5:
            return "Prediction methods disagree. Training may be unstable."

        if predicted_loss > self.target_loss * 2:
            return "Predicted loss is high. Consider increasing training duration or adjusting hyperparameters."

        return "Training is progressing normally. Continue monitoring."

    def _determine_convergence_status(self, y: np.ndarray) -> ConvergenceStatus:
        """Determine current convergence status"""
        if len(y) < 20:
            return ConvergenceStatus.UNKNOWN

        recent_window = min(50, len(y) // 4)
        recent_data = y[-recent_window:]

        # Check for plateau
        if np.std(recent_data) / (np.mean(recent_data) + 1e-8) < 0.02:
            return ConvergenceStatus.CONVERGED

        # Check trend
        trend = np.polyfit(range(len(recent_data)), recent_data, 1)[0]

        if trend < -self.convergence_threshold:
            return ConvergenceStatus.CONVERGING
        elif trend > self.convergence_threshold:
            return ConvergenceStatus.DIVERGING
        else:
            # Check for oscillations
            diff = np.diff(recent_data)
            sign_changes = np.sum(np.diff(np.sign(diff)) != 0)

            if sign_changes > len(recent_data) * 0.4:
                return ConvergenceStatus.OSCILLATING
            else:
                return ConvergenceStatus.STAGNANT

    def _calculate_convergence_rate(self, y: np.ndarray) -> float:
        """Calculate convergence rate (loss decrease per step)"""
        if len(y) < 10:
            return 0.0

        recent_window = min(100, len(y) // 2)
        recent_data = y[-recent_window:]
        x = np.arange(len(recent_data))

        try:
            slope = np.polyfit(x, recent_data, 1)[0]
            return -slope  # Negative slope means convergence
        except:
            return 0.0

    def _estimate_time_to_convergence(self, y: np.ndarray) -> Optional[int]:
        """Estimate steps to reach target loss"""
        if len(y) < 10:
            return None

        current_loss = y[-1]
        if current_loss <= self.target_loss:
            return 0

        convergence_rate = self._calculate_convergence_rate(y)
        if convergence_rate <= 0:
            return None

        steps_needed = (current_loss - self.target_loss) / convergence_rate
        return max(0, int(steps_needed))

    def _detect_plateau(self, y: np.ndarray) -> bool:
        """Detect if loss has plateaued"""
        if len(y) < self.plateau_patience:
            return False

        recent_data = y[-self.plateau_patience:]
        coefficient_of_variation = np.std(recent_data) / (np.mean(recent_data) + 1e-8)

        return coefficient_of_variation < 0.01

    def _calculate_oscillation_amplitude(self, y: np.ndarray) -> float:
        """Calculate amplitude of oscillations"""
        if len(y) < 10:
            return 0.0

        # Detrend the data
        x = np.arange(len(y))
        trend = np.polyfit(x, y, 1)
        detrended = y - np.polyval(trend, x)

        return float(np.std(detrended))

    def _calculate_trend_strength(self, y: np.ndarray) -> float:
        """Calculate strength of the overall trend"""
        if len(y) < 10:
            return 0.0

        x = np.arange(len(y))
        try:
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept

            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            return max(0.0, r_squared)
        except:
            return 0.0

    def _calculate_stability_score(self, y: np.ndarray) -> float:
        """Calculate overall stability score"""
        if len(y) < 10:
            return 0.0

        factors = []

        # Low oscillation amplitude
        osc_amp = self._calculate_oscillation_amplitude(y)
        mean_loss = np.mean(y)
        osc_factor = 1.0 / (1.0 + osc_amp / (mean_loss + 1e-8))
        factors.append(osc_factor)

        # Consistent trend
        trend_strength = self._calculate_trend_strength(y)
        factors.append(trend_strength)

        # Low variance in recent samples
        recent_window = min(50, len(y) // 4)
        recent_data = y[-recent_window:]
        variance_factor = 1.0 / (1.0 + np.var(recent_data) / (np.mean(recent_data) + 1e-8))
        factors.append(variance_factor)

        return np.mean(factors)

    def _should_early_stop(self, y: np.ndarray) -> bool:
        """Determine if early stopping should be recommended"""
        if len(y) < self.early_stopping_patience:
            return False

        recent_best = np.min(y[-self.early_stopping_patience:])
        overall_best = np.min(y)

        improvement = overall_best - recent_best

        return improvement < self.min_improvement

    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        if not self.loss_history:
            return {}

        y = np.array(self.loss_history)

        return {
            'mean': float(np.mean(y)),
            'median': float(np.median(y)),
            'std': float(np.std(y)),
            'min': float(np.min(y)),
            'max': float(np.max(y)),
            'current': float(y[-1]),
            'best': float(np.min(y)),
            'improvement_from_start': float(y[0] - y[-1]) if len(y) > 1 else 0.0,
            'improvement_from_best': float(np.min(y) - y[-1]),
            'total_steps': len(y),
            'variance': float(np.var(y))
        }

    def _generate_recommendations(self, prediction: EnsemblePrediction,
                                 convergence: ConvergenceAnalysis) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Based on convergence status
        if convergence.status == ConvergenceStatus.CONVERGED:
            recommendations.append("✅ Training has converged. Consider stopping or fine-tuning.")
        elif convergence.status == ConvergenceStatus.DIVERGING:
            recommendations.append("⚠️ Loss is diverging. Reduce learning rate immediately.")
        elif convergence.status == ConvergenceStatus.STAGNANT:
            recommendations.append("📈 Loss has stagnated. Consider adjusting learning rate or architecture.")
        elif convergence.status == ConvergenceStatus.OSCILLATING:
            recommendations.append("🌊 Loss is oscillating. Reduce learning rate or increase batch size.")

        # Based on prediction confidence
        if prediction.confidence < 0.3:
            recommendations.append("🔍 Low prediction confidence. Monitor training closely.")
        elif prediction.confidence > 0.8:
            recommendations.append("🎯 High prediction confidence. Training is stable.")

        # Based on early stopping
        if convergence.early_stopping_recommendation:
            recommendations.append("🛑 Consider early stopping to prevent overfitting.")

        # Based on stability
        if convergence.stability_score < 0.3:
            recommendations.append("📊 Low stability score. Consider regularization or different optimizer.")

        return recommendations


def create_loss_tracker(config) -> LossTracker:
    """Factory function to create loss tracker"""
    return LossTracker(config)


def test_loss_tracker():
    """Test the loss tracking system"""
    print("🧪 Testing Advanced Loss Tracking System")

    # Mock config
    class MockConfig:
        convergence_window = 50
        convergence_threshold = 1e-4
        plateau_patience = 100
        target_loss = 2.0
        early_stopping_patience = 200
        min_improvement = 1e-3

    config = MockConfig()

    # Create loss tracker
    print("✅ Testing Loss Tracker Initialization...")
    tracker = create_loss_tracker(config)
    print(f"  • Tracker initialized with {len(tracker.predictors)} prediction methods")

    # Test with synthetic data
    print("✅ Testing with Synthetic Loss Data...")

    # Generate synthetic loss curve (exponential decay with noise)
    steps = 200
    base_loss = 10.0
    decay_rate = 0.02
    noise_level = 0.1

    synthetic_losses = []
    for i in range(steps):
        loss = base_loss * np.exp(-decay_rate * i) + 2.0 + np.random.normal(0, noise_level)
        synthetic_losses.append(max(0.1, loss))
        tracker.update(loss, i)

    print(f"  • Added {len(synthetic_losses)} synthetic data points")
    print(f"  • Loss range: {min(synthetic_losses):.4f} - {max(synthetic_losses):.4f}")

    # Test individual predictors
    print("✅ Testing Individual Prediction Methods...")
    for method, predictor in tracker.predictors.items():
        try:
            prediction = predictor.predict(synthetic_losses, target_loss=2.0)
            print(f"  • {method.value}: pred={prediction.predicted_loss:.4f}, conf={prediction.confidence:.3f}")
        except Exception as e:
            print(f"  • {method.value}: Failed - {e}")

    # Test ensemble prediction
    print("✅ Testing Ensemble Prediction...")
    ensemble_pred = tracker.predict(target_loss=2.0)
    print(f"  • Ensemble prediction: {ensemble_pred.predicted_loss:.4f}")
    print(f"  • Confidence: {ensemble_pred.confidence:.3f}")
    print(f"  • Consensus score: {ensemble_pred.consensus_score:.3f}")
    print(f"  • Convergence probability: {ensemble_pred.convergence_probability:.3f}")
    print(f"  • Recommendation: {ensemble_pred.recommended_action}")

    # Test convergence analysis
    print("✅ Testing Convergence Analysis...")
    convergence = tracker.analyze_convergence()
    print(f"  • Status: {convergence.status.value}")
    print(f"  • Convergence rate: {convergence.convergence_rate:.6f}")
    print(f"  • Time to convergence: {convergence.time_to_convergence}")
    print(f"  • Plateau detected: {convergence.plateau_detection}")
    print(f"  • Oscillation amplitude: {convergence.oscillation_amplitude:.4f}")
    print(f"  • Trend strength: {convergence.trend_strength:.3f}")
    print(f"  • Stability score: {convergence.stability_score:.3f}")
    print(f"  • Early stopping recommended: {convergence.early_stopping_recommendation}")

    # Test comprehensive report
    print("✅ Testing Comprehensive Report...")
    report = tracker.generate_report()
    print(f"  • Report generated for step {report.current_step}")
    print(f"  • Current loss: {report.current_loss:.4f}")
    print(f"  • Statistics keys: {list(report.statistics.keys())}")
    print(f"  • Number of recommendations: {len(report.recommendations)}")

    # Test data export/import
    print("✅ Testing Data Export/Import...")
    export_success = tracker.export_data("test_loss_data.json")
    print(f"  • Export success: {export_success}")

    if export_success:
        new_tracker = create_loss_tracker(config)
        import_success = new_tracker.import_data("test_loss_data.json")
        print(f"  • Import success: {import_success}")
        print(f"  • Imported {len(new_tracker.loss_history)} data points")

    print("🎉 All loss tracking tests completed!")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    test_loss_tracker()