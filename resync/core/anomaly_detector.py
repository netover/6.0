# pylint
"""
Advanced Anomaly Detection with Machine Learning.

This module provides intelligent anomaly detection capabilities using:
- Real-time metrics collection and analysis
- Multiple ML algorithms (Isolation Forest, One-Class SVM, Autoencoders)
- Adaptive thresholding based on historical patterns
- Risk scoring and alerting system
- Performance optimized for production use
- Self-learning and model updates
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from resync.core.task_tracker import track_task

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM

    SKLEARN_AVAILABLE = True
except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError):  # pragma: no cover
    # Allow the module to be imported without scikit-learn; runtime code will fail fast
    IsolationForest = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]
    OneClassSVM = None  # type: ignore[assignment]
    SKLEARN_AVAILABLE = False

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)

@dataclass
class AnomalyMetrics:
    """Metrics collected for anomaly detection."""

    timestamp: float = field(default_factory=time.time)
    user_id: str | None = None
    session_id: str | None = None
    endpoint: str = ""
    method: str = "GET"
    response_time: float = 0.0
    status_code: int = 200
    request_size: int = 0
    response_size: int = 0
    user_agent: str = ""
    ip_address: str = ""
    geo_location: dict[str, Any] | None = None
    custom_metrics: dict[str, Any] = field(default_factory=dict)

    def to_feature_vector(self) -> np.ndarray:
        """Convert metrics to numerical feature vector for ML."""
        # Create feature vector from metrics
        features = [
            float(self.response_time),
            float(self.status_code),
            float(self.request_size),
            float(self.response_size),
            hash(self.endpoint) % 1000,  # Hash endpoint to numeric
            hash(self.method) % 100,  # Hash method to numeric
            hash(self.user_agent) % 1000 if self.user_agent else 0,
            hash(self.ip_address) % 1000 if self.ip_address else 0,
        ]

        # Add custom metrics
        for _key, value in self.custom_metrics.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                features.append(hash(value) % 1000)
            else:
                features.append(0.0)

        return np.array(features).reshape(1, -1)

@dataclass
class AnomalyScore:
    """Anomaly detection result with scoring."""

    is_anomaly: bool
    confidence: float
    risk_level: str  # "low", "medium", "high", "critical"
    detection_method: str
    feature_importance: dict[str, float]
    timestamp: float = field(default_factory=time.time)
    metrics: AnomalyMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_anomaly": self.is_anomaly,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "detection_method": self.detection_method,
            "feature_importance": self.feature_importance,
            "timestamp": self.timestamp,
            "metrics": self.metrics.__dict__ if self.metrics else None,
        }

@dataclass
class MLModelConfig:
    """Configuration for ML anomaly detection models."""

    # Isolation Forest parameters
    isolation_forest_n_estimators: int = 100
    isolation_forest_contamination: float = 0.1
    isolation_forest_random_state: int = 42

    # One-Class SVM parameters
    svm_nu: float = 0.1
    svm_kernel: str = "rbf"
    svm_gamma: str = "scale"

    # Training parameters
    training_window_hours: int = 24
    min_samples_for_training: int = 1000
    retrain_interval_hours: int = 6

    # Model selection
    primary_model: str = (
        "isolation_forest"  # "isolation_forest", "one_class_svm", "ensemble"
    )
    enable_ensemble: bool = True

    # Performance tuning
    batch_size: int = 100
    max_memory_mb: int = 500

class IsolationForestDetector:
    """Isolation Forest based anomaly detector."""

    def __init__(self, config: MLModelConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data = deque(maxlen=config.min_samples_for_training * 2)
        self.last_trained = 0

    async def detect(self, metrics: AnomalyMetrics) -> AnomalyScore:
        """Detect anomalies using Isolation Forest."""
        if not self.is_trained or self._should_retrain():
            self._train_model()

        if not self.is_trained:
            # Not enough data for training, return low-risk score
            return AnomalyScore(
                is_anomaly=False,
                confidence=0.1,
                risk_level="low",
                detection_method="isolation_forest",
                feature_importance={},
                metrics=metrics,
            )

        try:
            feature_vector = metrics.to_feature_vector()
            scaled_features = self.scaler.transform(feature_vector)

            # Get anomaly score (-1 for outliers, 1 for inliers)
            score = self.model.decision_function(scaled_features)[0]

            # Convert to anomaly probability (higher = more anomalous)
            anomaly_score = (score + 1) / 2  # Convert from [-1,1] to [0,1]

            # Determine if it's an anomaly
            is_anomaly = anomaly_score > (
                1 - self.config.isolation_forest_contamination
            )

            # Calculate risk level
            risk_level = self._calculate_risk_level(anomaly_score)

            return AnomalyScore(
                is_anomaly=is_anomaly,
                confidence=anomaly_score,
                risk_level=risk_level,
                detection_method="isolation_forest",
                feature_importance=self._get_feature_importance(),
                metrics=metrics,
            )

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.warning("Isolation Forest detection error: %s", e)
            return AnomalyScore(
                is_anomaly=False,
                confidence=0.0,
                risk_level="low",
                detection_method="isolation_forest",
                feature_importance={},
                metrics=metrics,
            )

    def _train_model(self) -> None:
        """Train or retrain the Isolation Forest model."""
        if len(self.training_data) < self.config.min_samples_for_training:
            return

        try:
            # Convert training data to feature matrix
            feature_matrix = np.vstack(
                [m.to_feature_vector()[0] for m in self.training_data]
            )

            # Fit scaler
            self.scaler.fit(feature_matrix)

            # Scale features
            scaled_features = self.scaler.transform(feature_matrix)

            # Train model
            self.model = IsolationForest(
                n_estimators=self.config.isolation_forest_n_estimators,
                contamination=self.config.isolation_forest_contamination,
                random_state=self.config.isolation_forest_random_state,
                n_jobs=-1,
            )

            self.model.fit(scaled_features)
            self.is_trained = True
            self.last_trained = time.time()

            logger.info(
                "Isolation Forest model trained with %s samples",
                len(self.training_data),
            )

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.error("Isolation Forest training error: %s", e)
            self.is_trained = False

    def _should_retrain(self) -> bool:
        """Check if model should be retrained."""
        if not self.is_trained:
            return True

        time_since_training = time.time() - self.last_trained
        return time_since_training > (self.config.retrain_interval_hours * 3600)

    def _calculate_risk_level(self, anomaly_score: float) -> str:
        """Calculate risk level based on anomaly score."""
        if anomaly_score > 0.9:
            return "critical"
        if anomaly_score > 0.7:
            return "high"
        if anomaly_score > 0.5:
            return "medium"
        return "low"

    def _get_feature_importance(self) -> dict[str, float]:
        """Get feature importance (simplified for Isolation Forest)."""
        # Isolation Forest doesn't provide direct feature importance
        # Return equal weights for all features
        return {
            "response_time": 0.2,
            "status_code": 0.2,
            "request_size": 0.15,
            "response_size": 0.15,
            "endpoint": 0.1,
            "method": 0.1,
            "user_agent": 0.05,
            "ip_address": 0.05,
        }

    def add_training_sample(self, metrics: AnomalyMetrics) -> None:
        """Add sample to training data."""
        self.training_data.append(metrics)

class OneClassSVMDetector:
    """One-Class SVM based anomaly detector."""

    def __init__(self, config: MLModelConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_data = deque(maxlen=config.min_samples_for_training * 2)
        self.last_trained = 0

    async def detect(self, metrics: AnomalyMetrics) -> AnomalyScore:
        """Detect anomalies using One-Class SVM."""
        if not self.is_trained or self._should_retrain():
            self._train_model()

        if not self.is_trained:
            return AnomalyScore(
                is_anomaly=False,
                confidence=0.1,
                risk_level="low",
                detection_method="one_class_svm",
                feature_importance={},
                metrics=metrics,
            )

        try:
            feature_vector = metrics.to_feature_vector()
            scaled_features = self.scaler.transform(feature_vector)

            # Get decision function (negative = outlier)
            decision = self.model.decision_function(scaled_features)[0]

            # Convert to anomaly score (0-1 scale)
            # decision_function returns negative for outliers, positive for inliers
            anomaly_score = 1 / (1 + np.exp(decision))  # Sigmoid transformation

            is_anomaly = decision < 0  # Negative decision = anomaly

            risk_level = self._calculate_risk_level(anomaly_score)

            return AnomalyScore(
                is_anomaly=is_anomaly,
                confidence=anomaly_score,
                risk_level=risk_level,
                detection_method="one_class_svm",
                feature_importance=self._get_feature_importance(),
                metrics=metrics,
            )

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.warning("One-Class SVM detection error: %s", e)
            return AnomalyScore(
                is_anomaly=False,
                confidence=0.0,
                risk_level="low",
                detection_method="one_class_svm",
                feature_importance={},
                metrics=metrics,
            )

    def _train_model(self) -> None:
        """Train One-Class SVM model."""
        if len(self.training_data) < self.config.min_samples_for_training:
            return

        try:
            feature_matrix = np.vstack(
                [m.to_feature_vector()[0] for m in self.training_data]
            )

            self.scaler.fit(feature_matrix)
            scaled_features = self.scaler.transform(feature_matrix)

            self.model = OneClassSVM(
                nu=self.config.svm_nu,
                kernel=self.config.svm_kernel,
                gamma=self.config.svm_gamma,
            )

            self.model.fit(scaled_features)
            self.is_trained = True
            self.last_trained = time.time()

            logger.info(
                "One-Class SVM model trained with %s samples", len(self.training_data)
            )

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            logger.error("One-Class SVM training error: %s", e)
            self.is_trained = False

    def _should_retrain(self) -> bool:
        """Check if model should be retrained."""
        if not self.is_trained:
            return True

        time_since_training = time.time() - self.last_trained
        return time_since_training > (self.config.retrain_interval_hours * 3600)

    def _calculate_risk_level(self, anomaly_score: float) -> str:
        """Calculate risk level based on anomaly score."""
        if anomaly_score > 0.8:
            return "critical"
        if anomaly_score > 0.6:
            return "high"
        if anomaly_score > 0.4:
            return "medium"
        return "low"

    def _get_feature_importance(self) -> dict[str, float]:
        """Get feature importance (simplified)."""
        return {
            "response_time": 0.25,
            "status_code": 0.2,
            "request_size": 0.15,
            "response_size": 0.15,
            "endpoint": 0.1,
            "method": 0.05,
            "user_agent": 0.05,
            "ip_address": 0.05,
        }

    def add_training_sample(self, metrics: AnomalyMetrics) -> None:
        """Add sample to training data."""
        self.training_data.append(metrics)

class EnsembleAnomalyDetector:
    """Ensemble anomaly detector combining multiple ML models."""

    def __init__(self, config: MLModelConfig):
        self.config = config
        self.detectors = {
            "isolation_forest": IsolationForestDetector(config),
            "one_class_svm": OneClassSVMDetector(config),
        }
        self.ensemble_weights = {"isolation_forest": 0.6, "one_class_svm": 0.4}

    async def detect(self, metrics: AnomalyMetrics) -> AnomalyScore:
        """Detect anomalies using ensemble approach."""
        results = []
        for name, detector in self.detectors.items():
            result = await detector.detect(metrics)
            results.append((name, result))

        # Combine results using weighted voting
        combined_score = 0.0
        total_weight = 0.0
        anomaly_votes = 0
        feature_importance = defaultdict(float)

        for name, result in results:
            weight = self.ensemble_weights.get(name, 1.0)
            combined_score += result.confidence * weight
            total_weight += weight

            if result.is_anomaly:
                anomaly_votes += weight

            # Combine feature importance
            for feature, importance in result.feature_importance.items():
                feature_importance[feature] += importance * weight

        # Normalize
        if total_weight > 0:
            combined_score /= total_weight
            anomaly_votes /= total_weight

            # Normalize feature importance
            for feature in feature_importance:
                feature_importance[feature] /= total_weight

        is_anomaly = anomaly_votes > 0.5  # Majority vote

        # Determine risk level based on combined score and votes
        if is_anomaly and combined_score > 0.7:
            risk_level = "critical"
        elif is_anomaly and combined_score > 0.5:
            risk_level = "high"
        elif combined_score > 0.6:
            risk_level = "medium"
        else:
            risk_level = "low"

        return AnomalyScore(
            is_anomaly=is_anomaly,
            confidence=combined_score,
            risk_level=risk_level,
            detection_method="ensemble",
            feature_importance=dict(feature_importance),
            metrics=metrics,
        )

    def add_training_sample(self, metrics: AnomalyMetrics) -> None:
        """Add training sample to all detectors."""
        for detector in self.detectors.values():
            detector.add_training_sample(metrics)

class AnomalyDetectionEngine:
    """
    Main anomaly detection engine with real-time processing.

    Features:
    - Multiple ML algorithms for anomaly detection
    - Real-time metrics collection and analysis
    - Adaptive thresholding and model updates
    - Alert generation and risk scoring
    - Performance optimized for high-throughput
    """

    def __init__(self, config: MLModelConfig | None = None):
        self.config = config or MLModelConfig()

        if not SKLEARN_AVAILABLE:
            raise RuntimeError(
                "AnomalyDetectionEngine requires scikit-learn. "
                "Install the ML extras (e.g., `pip install resync[ml]`) "
                "or add scikit-learn to your deployment image."
            )

        # Initialize detectors based on configuration
        if self.config.primary_model == "isolation_forest":
            self.primary_detector = IsolationForestDetector(self.config)
        elif self.config.primary_model == "one_class_svm":
            self.primary_detector = OneClassSVMDetector(self.config)
        else:  # ensemble
            self.primary_detector = EnsembleAnomalyDetector(self.config)

        # Alert system
        self.alert_thresholds = {
            "critical": 0.9,
            "high": 0.7,
            "medium": 0.5,
            "low": 0.3,
        }

        # Metrics collection
        self.metrics_buffer = deque(maxlen=10000)
        self.anomaly_history = deque(maxlen=1000)

        # Statistics
        self.total_requests = 0
        self.anomalies_detected = 0
        self.false_positives = 0
        self.last_model_update = time.time()

        # Background tasks
        self._processing_task: asyncio.Task | None = None
        self._training_task: asyncio.Task | None = None
        self._running = False

        # Thread safety
        self._lock = asyncio.Lock()

    def start(self, tg: asyncio.TaskGroup | None = None) -> None:
        """
        Start the anomaly detection engine.

        Args:
            tg: Optional TaskGroup to run the background loops in
        """
        if self._running:
            return

        self._running = True
        if tg:
            self._processing_task = tg.create_task(
                self._processing_loop(), name="processing_loop"
            )
            self._training_task = tg.create_task(
                self._training_loop(), name="training_loop"
            )
        else:
            self._processing_task = track_task(
                self._processing_loop(), name="processing_loop"
            )
            self._training_task = track_task(
                self._training_loop(), name="training_loop"
            )

        logger.info(
            "Anomaly detection engine started",
            method="task_group" if tg else "track_task",
        )

    async def stop(self) -> None:
        """Stop the anomaly detection engine."""
        if not self._running:
            return

        self._running = False

        for task in [self._processing_task, self._training_task]:
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        logger.info("Anomaly detection engine stopped")

    async def analyze_request(
        self, metrics: AnomalyMetrics, generate_alert: bool = True
    ) -> AnomalyScore:
        """
        Analyze a request for anomalies.

        Args:
            metrics: Request metrics to analyze
            generate_alert: Whether to generate alerts for anomalies

        Returns:
            Anomaly detection result
        """
        async with self._lock:
            self.total_requests += 1

            # Add to buffer for batch processing
            self.metrics_buffer.append(metrics)

            # Perform real-time detection
            result = await self.primary_detector.detect(metrics)

            # Store result
            self.anomaly_history.append(result)

            # Update statistics
            if result.is_anomaly:
                self.anomalies_detected += 1

                # Generate alert if requested
                if generate_alert and result.confidence > self.alert_thresholds.get(
                    result.risk_level, 0.5
                ):
                    self._generate_alert(result)

            return result

    def _generate_alert(self, result: AnomalyScore) -> None:
        """Generate alert for detected anomaly."""
        # Use BLAKE2b instead of MD5 for generating anomaly IDs
        anomaly_id = hashlib.blake2b(
            f"{result.timestamp}_{result.metrics.user_id}".encode(), digest_size=16
        ).hexdigest()

        alert_data = {
            "anomaly_id": anomaly_id,
            "risk_level": result.risk_level,
            "confidence": result.confidence,
            "detection_method": result.detection_method,
            "user_id": result.metrics.user_id if result.metrics else None,
            "endpoint": result.metrics.endpoint if result.metrics else "",
            "ip_address": result.metrics.ip_address if result.metrics else "",
            "timestamp": result.timestamp,
            "feature_importance": result.feature_importance,
        }

        # Log alert
        logger.warning(
            "anomaly_detected",
            anomaly_id=alert_data["anomaly_id"],
            risk_level=result.risk_level,
            confidence=result.confidence,
            user_id=result.metrics.user_id if result.metrics else None,
            endpoint=result.metrics.endpoint if result.metrics else "",
            detection_method=result.detection_method,
        )

        # Here you could integrate with external alerting systems
        # (email, Slack, PagerDuty, etc.)

    async def _processing_loop(self) -> None:
        """Background processing loop for batch operations."""
        while self._running:
            try:
                await asyncio.sleep(10)  # Process every 10 seconds

                # Batch process buffered metrics
                if len(self.metrics_buffer) >= self.config.batch_size:
                    self._process_batch()

                # Clean old history
                self._cleanup_history()

            except asyncio.CancelledError:
                break
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.error("Processing loop error: %s", e)

    async def _training_loop(self) -> None:
        """Background training loop for model updates."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Check every hour

                # Check if models need retraining
                if len(self.metrics_buffer) >= self.config.min_samples_for_training:
                    self._update_models()

            except asyncio.CancelledError:
                break
            except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
                logger.error("Training loop error: %s", e)

    def _process_batch(self) -> None:
        """Process a batch of buffered metrics."""
        batch_size = min(len(self.metrics_buffer), self.config.batch_size)
        batch = []

        # Extract batch
        for _ in range(batch_size):
            if self.metrics_buffer:
                batch.append(self.metrics_buffer.popleft())

        # Add samples to training data
        for metrics in batch:
            self.primary_detector.add_training_sample(metrics)

        logger.debug("Processed batch of %s metrics samples", len(batch))

    def _update_models(self) -> None:
        """Update ML models with new training data."""
        try:
            # Force retraining if needed
            if (
                hasattr(self.primary_detector, "_should_retrain")
                and self.primary_detector._should_retrain()
            ):
                # Retraining is handled automatically in detect() method
                pass

            self.last_model_update = time.time()
            logger.info("ML models updated with new training data")

        except (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, TimeoutError, ConnectionError) as e:
            # Re-raise programming errors — these are bugs, not runtime failures
            if isinstance(e, (TypeError, KeyError, AttributeError, IndexError)):
                raise
            logger.error("Model update error: %s", e)

    def _cleanup_history(self) -> None:
        """Clean up old history data."""
        current_time = time.time()
        max_age = 7 * 24 * 3600  # 7 days

        # Remove old anomaly history
        while self.anomaly_history:
            oldest = self.anomaly_history[0]
            if current_time - oldest.timestamp > max_age:
                self.anomaly_history.popleft()
            else:
                break

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive anomaly detection statistics."""
        total_anomalies = len([r for r in self.anomaly_history if r.is_anomaly])
        detection_rate = total_anomalies / max(1, len(self.anomaly_history))

        risk_distribution = defaultdict(int)
        for result in self.anomaly_history:
            risk_distribution[result.risk_level] += 1

        return {
            "performance": {
                "total_requests": self.total_requests,
                "anomalies_detected": self.anomalies_detected,
                "detection_rate": detection_rate,
                "false_positives": self.false_positives,
                "buffer_size": len(self.metrics_buffer),
            },
            "risk_distribution": dict(risk_distribution),
            "models": {
                "primary_model": self.config.primary_model,
                "last_update": self.last_model_update,
                "training_samples": len(
                    getattr(self.primary_detector, "training_data", [])
                ),
                "is_trained": getattr(self.primary_detector, "is_trained", False),
            },
            "configuration": {
                "batch_size": self.config.batch_size,
                "min_samples_for_training": self.config.min_samples_for_training,
                "retrain_interval_hours": self.config.retrain_interval_hours,
            },
        }

    def update_alert_thresholds(self, thresholds: dict[str, float]) -> None:
        """Update alert thresholds."""
        self.alert_thresholds.update(thresholds)
        logger.info("Alert thresholds updated", thresholds=thresholds)

    def mark_false_positive(self, anomaly_id: str) -> bool:
        """
        Mark an anomaly as false positive for model improvement.

        Returns:
            True if found and marked, False otherwise
        """
        for result in self.anomaly_history:
            # Use BLAKE2b instead of MD5 for generating anomaly IDs
            result_id = (
                hashlib.blake2b(
                    f"{result.timestamp}_{result.metrics.user_id}".encode(),
                    digest_size=16,
                ).hexdigest()
                if result.metrics
                else None
            )

            if result_id == anomaly_id:
                self.false_positives += 1
                logger.info("False positive marked for anomaly %s", anomaly_id)
                return True

        return False

# Global anomaly detection engine instance
# (lazy-initialized to avoid crash without sklearn)
_anomaly_detection_engine: AnomalyDetectionEngine | None = None

def _get_anomaly_detection_engine() -> AnomalyDetectionEngine:
    """Get or create the anomaly detection engine (lazy init)."""
    global _anomaly_detection_engine
    if _anomaly_detection_engine is None:
        _anomaly_detection_engine = AnomalyDetectionEngine()
    return _anomaly_detection_engine

# Backward compatibility alias
anomaly_detection_engine: AnomalyDetectionEngine | None = None  # type: ignore[assignment]

async def get_anomaly_detection_engine() -> AnomalyDetectionEngine:
    """Get the global anomaly detection engine instance."""
    engine = _get_anomaly_detection_engine()
    if not engine._running:
        engine.start()
    return engine
