"""
Advanced ML Models: LSTM and Transformer models for time series prediction.
Includes runtime flag to enable/disable based on performance.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT

# Try to import TensorFlow
try:
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras  # noqa: F401
    from tensorflow.keras import layers, callbacks  # noqa: F401

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Advanced ML models disabled.")

# Try to import PyTorch for Transformer (reserved for future use)
try:
    pass  # PyTorch reserved for future Transformer implementation
    PYTORCH_AVAILABLE = False  # Currently using TensorFlow only
except ImportError:
    PYTORCH_AVAILABLE = False


class ModelType(Enum):
    """Types of advanced ML models."""

    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"


class ModelStatus(Enum):
    """Model training/performance status."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    TRAINING = "training"
    EVALUATING = "evaluating"
    FAILED = "failed"


@dataclass
class ModelPerformance:
    """Performance metrics for a model."""

    model_type: ModelType
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    sample_count: int
    last_updated: str
    enabled: bool = True

    @property
    def overall_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        score = 0.0

        # Accuracy score (0-30)
        score += self.accuracy * 30

        # Sharpe ratio (0-25, assuming -2 to +4 range)
        sharpe_normalized = max(-2, min(4, self.sharpe_ratio))
        score += ((sharpe_normalized + 2) / 6) * 25

        # Win rate (0-25)
        score += self.win_rate * 25

        # Max drawdown penalty (0-20, lower is better)
        drawdown_penalty = max(0, 1 - abs(self.max_drawdown) / 0.5)
        score += drawdown_penalty * 20

        return round(score, 2)

    def should_enable(self, threshold: float = 50.0) -> bool:
        """Check if model should be enabled based on performance."""
        return self.enabled and self.overall_score >= threshold


class SequenceBuilder:
    """
    Build sequences for time series models.

    Features:
    - Sliding window sequences
    - Multi-feature support
    - Train/test splitting
    """

    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        feature_columns: List[str] = None,
    ):
        """
        Args:
            sequence_length: Number of time steps in input sequence
            prediction_horizon: Number of steps ahead to predict
            feature_columns: List of feature column names
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.feature_columns = feature_columns

    def create_sequences(
        self,
        data: pd.DataFrame,
        target_column: str = "target",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/Transformer training.

        Args:
            data: DataFrame with features and target
            target_column: Name of target column

        Returns:
            Tuple of (X, y) arrays
        """
        # Use all columns except target as features if not specified
        if self.feature_columns is None:
            feature_cols = [col for col in data.columns if col != target_column]
        else:
            feature_cols = [col for col in self.feature_columns if col in data.columns]

        # Drop NaN values
        data_clean = data[feature_cols + [target_column]].dropna()

        X_list = []
        y_list = []

        for i in range(
            len(data_clean) - self.sequence_length - self.prediction_horizon + 1
        ):
            # Input sequence
            X = data_clean.iloc[i : i + self.sequence_length][feature_cols].values
            X_list.append(X)

            # Target (future return)
            target_idx = i + self.sequence_length + self.prediction_horizon - 1
            y = data_clean.iloc[target_idx][target_column]
            y_list.append(y)

        X = np.array(X_list)
        y = np.array(y_list)

        return X, y

    def create_lstm_sequences(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences specifically for LSTM.

        Args:
            features: Feature array (samples, features)
            targets: Target array (samples,)

        Returns:
            Tuple of (X, y) reshaped for LSTM
        """
        X_list = []
        y_list = []

        for i in range(len(features) - self.sequence_length):
            X_list.append(features[i : i + self.sequence_length])
            y_list.append(targets[i + self.sequence_length])

        X = np.array(X_list)
        y = np.array(y_list)

        return X, y


class LSTMPredictor:
    """
    LSTM model for time series prediction.

    Features:
    - Multi-layer LSTM architecture
    - Dropout for regularization
    - Early stopping
    - Learning rate scheduling
    """

    def __init__(
        self,
        sequence_length: int = 60,
        lstm_units: List[int] = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ):
        """
        Args:
            sequence_length: Input sequence length
            lstm_units: List of LSTM units per layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is required for LSTM models")

        self.sequence_length = sequence_length
        self.lstm_units = lstm_units or [128, 64, 32]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.model: Optional["keras.Model"] = None
        self.history: Dict = None
        self.is_trained = False

    def build_model(
        self,
        n_features: int,
    ) -> "keras.Model":
        """
        Build LSTM model architecture.

        Args:
            n_features: Number of input features

        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=(self.sequence_length, n_features))

        # LSTM layers
        x = inputs
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
            )(x)
            x = layers.Dropout(self.dropout_rate)(x)

        # Dense layers for prediction
        x = layers.Dense(16, activation="relu")(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="lstm_predictor")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ) -> Dict:
        """
        Train LSTM model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            early_stopping_patience: Patience for early stopping

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model(X_train.shape[2])

        # Callbacks
        callback_list = []
        if X_val is not None and y_val is not None:
            early_stop = callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
            )
            callback_list.append(early_stop)

        # Train
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=0,
        )

        self.history = history.history
        self.is_trained = True

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions (0-1 range)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        return self.model.predict(X, verbose=0).flatten()

    def save(self, filepath: str) -> None:
        """Save model to file."""
        if self.model is not None:
            self.model.save(filepath)

    def load(self, filepath: str) -> None:
        """Load model from file."""
        self.model = keras.models.load_model(filepath)
        self.is_trained = True


class TransformerPredictor:
    """
    Transformer model for time series prediction.

    Features:
    - Multi-head self-attention
    - Positional encoding
    - Feed-forward networks
    """

    def __init__(
        self,
        sequence_length: int = 60,
        n_features: int = 10,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
    ):
        """
        Args:
            sequence_length: Input sequence length
            n_features: Number of input features
            d_model: Dimension of transformer
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout_rate: Dropout rate
        """
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is required for Transformer models")

        self.sequence_length = sequence_length
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate

        self.model: Optional["keras.Model"] = None
        self.history: Dict = None
        self.is_trained = False

    def build_model(self) -> "keras.Model":
        """Build Transformer model architecture."""
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))

        # Project inputs to d_model dimension
        x = layers.Dense(self.d_model)(inputs)

        # Positional encoding (learned)
        positions = layers.Embedding(
            input_dim=self.sequence_length,
            output_dim=self.d_model,
        )(tf.range(start=0, limit=self.sequence_length, delta=1))
        x = x + positions

        # Transformer layers
        for _ in range(self.n_layers):
            # Multi-head self-attention
            attn_output = layers.MultiHeadAttention(
                num_heads=self.n_heads,
                key_dim=self.d_model // self.n_heads,
                dropout=self.dropout_rate,
            )(x, x)
            x = layers.LayerNormalization()(x + attn_output)
            x = layers.Dropout(self.dropout_rate)(x)

            # Feed-forward network
            ffn = layers.Dense(self.d_model * 2, activation="relu")(x)
            ffn = layers.Dense(self.d_model)(ffn)
            x = layers.LayerNormalization()(x + ffn)
            x = layers.Dropout(self.dropout_rate)(x)

        # Global pooling and prediction
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(16, activation="relu")(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(
            inputs=inputs, outputs=outputs, name="transformer_predictor"
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> Dict:
        """Train Transformer model."""
        self.n_features = X_train.shape[2]

        if self.model is None:
            self.build_model()

        # Callbacks
        callback_list = []
        if X_val is not None and y_val is not None:
            early_stop = callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            )
            callback_list.append(early_stop)

        # Train
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=0,
        )

        self.history = history.history
        self.is_trained = True

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        return self.model.predict(X, verbose=0).flatten()

    def save(self, filepath: str) -> None:
        """Save model to file."""
        if self.model is not None:
            self.model.save(filepath)

    def load(self, filepath: str) -> None:
        """Load model from file."""
        if not TENSORFLOW_AVAILABLE:
            raise RuntimeError("TensorFlow is required to load models")
        from tensorflow import keras

        self.model = keras.models.load_model(filepath)
        self.is_trained = True


class ModelEnsemble:
    """
    Ensemble of multiple models for robust predictions.

    Features:
    - Combine LSTM, Transformer predictions
    - Weighted voting
    - Performance-based weighting
    """

    def __init__(
        self,
        models: Dict[ModelType, Any] = None,
        weights: Dict[ModelType, float] = None,
    ):
        """
        Args:
            models: Dictionary of trained models
            weights: Model weights for voting
        """
        self.models = models or {}
        self.weights = weights or {}

    def add_model(
        self,
        model_type: ModelType,
        model: Any,
        weight: float = 1.0,
    ) -> None:
        """Add a model to the ensemble."""
        self.models[model_type] = model
        self.weights[model_type] = weight

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.

        Args:
            X: Input features

        Returns:
            Weighted ensemble predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        predictions = []
        weights = []

        for model_type, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
            weights.append(self.weights.get(model_type, 1.0))

        # Weighted average
        weights = np.array(weights) / np.sum(weights)
        ensemble_pred = np.average(predictions, axis=0, weights=weights)

        return ensemble_pred

    def update_weights(
        self,
        performance_scores: Dict[ModelType, float],
    ) -> None:
        """
        Update model weights based on performance.

        Args:
            performance_scores: Performance score for each model
        """
        total_score = sum(performance_scores.values())

        if total_score > 0:
            for model_type, score in performance_scores.items():
                self.weights[model_type] = score / total_score


class AdvancedMLManager:
    """
    Manage advanced ML models with runtime enable/disable.

    Features:
    - Train and evaluate advanced models
    - Runtime performance tracking
    - Automatic enable/disable based on performance
    - Model persistence
    """

    def __init__(
        self,
        models_dir: str = None,
        performance_threshold: float = 50.0,
        auto_disable: bool = True,
    ):
        """
        Args:
            models_dir: Directory for saving models
            performance_threshold: Minimum score to enable model
            auto_disable: Auto-disable models below threshold
        """
        self.models_dir = models_dir or os.path.join(PROJECT_ROOT, "models", "advanced")
        os.makedirs(self.models_dir, exist_ok=True)

        self.performance_threshold = performance_threshold
        self.auto_disable = auto_disable

        self.models: Dict[ModelType, Any] = {}
        self.performances: Dict[ModelType, ModelPerformance] = {}
        self.sequence_builder = SequenceBuilder(sequence_length=60)

        # Load saved performances
        self._load_performances()

    def _load_performances(self) -> None:
        """Load saved model performances."""
        perf_file = os.path.join(self.models_dir, "performances.json")

        if os.path.exists(perf_file):
            try:
                with open(perf_file) as f:
                    data = json.load(f)

                for model_data in data.get("performances", []):
                    # Convert model_type string back to ModelType enum
                    model_data["model_type"] = ModelType(model_data["model_type"])
                    perf = ModelPerformance(**model_data)
                    self.performances[perf.model_type] = perf
            except Exception as e:
                print(f"Error loading performances: {e}")

    def _save_performances(self) -> None:
        """Save model performances."""
        from dataclasses import asdict

        # Convert ModelType enum to string for JSON serialization
        performances = [
            {
                **asdict(p),
                "model_type": p.model_type.value,  # Convert enum to string
            }
            for p in self.performances.values()
        ]

        data = {
            "last_updated": datetime.now().isoformat(),
            "performances": performances,
        }

        perf_file = os.path.join(self.models_dir, "performances.json")
        with open(perf_file, "w") as f:
            json.dump(data, f, indent=2)

    def is_model_enabled(self, model_type: ModelType) -> bool:
        """Check if a model is enabled."""
        if model_type not in self.performances:
            return True  # New models start enabled

        return self.performances[model_type].should_enable(self.performance_threshold)

    def train_lstm(
        self,
        data: pd.DataFrame,
        target_column: str = "target",
    ) -> ModelPerformance:
        """Train LSTM model and evaluate performance."""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Skipping LSTM training.")
            return None

        # Create sequences
        X, y = self.sequence_builder.create_sequences(data, target_column)

        if len(X) < 100:
            print(f"Insufficient data for LSTM: {len(X)} samples")
            return None

        # Split train/val
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Build and train
        predictor = LSTMPredictor(sequence_length=60)
        predictor.build_model(n_features=X.shape[2])
        predictor.train(X_train, y_train, X_val, y_val)

        # Evaluate
        val_preds = predictor.predict(X_val)
        val_preds_binary = (val_preds > 0.5).astype(int)

        performance = self._calculate_performance(
            ModelType.LSTM,
            y_val,
            val_preds_binary,
        )

        self.models[ModelType.LSTM] = predictor
        self.performances[ModelType.LSTM] = performance
        self._save_performances()

        # Save model if enabled
        if performance.should_enable(self.performance_threshold):
            model_path = os.path.join(self.models_dir, "lstm_model.keras")
            predictor.save(model_path)

        return performance

    def train_transformer(
        self,
        data: pd.DataFrame,
        target_column: str = "target",
    ) -> ModelPerformance:
        """Train Transformer model and evaluate performance."""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Skipping Transformer training.")
            return None

        # Create sequences
        X, y = self.sequence_builder.create_sequences(data, target_column)

        if len(X) < 100:
            print(f"Insufficient data for Transformer: {len(X)} samples")
            return None

        # Split train/val
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Build and train
        predictor = TransformerPredictor(
            sequence_length=60,
            n_features=X.shape[2],
        )
        predictor.build_model()
        predictor.train(X_train, y_train, X_val, y_val)

        # Evaluate
        val_preds = predictor.predict(X_val)
        val_preds_binary = (val_preds > 0.5).astype(int)

        performance = self._calculate_performance(
            ModelType.TRANSFORMER,
            y_val,
            val_preds_binary,
        )

        self.models[ModelType.TRANSFORMER] = predictor
        self.performances[ModelType.TRANSFORMER] = performance
        self._save_performances()

        # Save model if enabled
        if performance.should_enable(self.performance_threshold):
            model_path = os.path.join(self.models_dir, "transformer_model.keras")
            predictor.save(model_path)

        return performance

    def _calculate_performance(
        self,
        model_type: ModelType,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> ModelPerformance:
        """Calculate model performance metrics."""
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Simulate trading metrics
        win_rate = accuracy  # Simplified
        sharpe_ratio = (accuracy - 0.5) * 4  # Simplified
        max_drawdown = -0.1 + (accuracy - 0.5) * 0.2  # Simplified
        total_return = (accuracy - 0.5) * 0.5  # Simplified

        return ModelPerformance(
            model_type=model_type,
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            sharpe_ratio=round(sharpe_ratio, 2),
            max_drawdown=round(max_drawdown, 2),
            total_return=round(total_return, 2),
            win_rate=round(win_rate, 2),
            sample_count=len(y_true),
            last_updated=datetime.now().isoformat(),
            enabled=True,
        )

    def get_best_prediction(self, X: np.ndarray) -> Tuple[np.ndarray, ModelType]:
        """
        Get prediction from best performing enabled model.

        Args:
            X: Input features

        Returns:
            Tuple of (predictions, model_type)
        """
        # Filter enabled models
        enabled_models = {
            mt: m for mt, m in self.models.items() if self.is_model_enabled(mt)
        }

        if not enabled_models:
            # No enabled models, return neutral prediction
            return np.full(len(X), 0.5), None

        # Find best performing model
        best_model_type = max(
            enabled_models.keys(),
            key=lambda mt: self.performances.get(
                mt,
                ModelPerformance(
                    model_type=mt,
                    accuracy=0,
                    precision=0,
                    recall=0,
                    f1_score=0,
                    sharpe_ratio=0,
                    max_drawdown=0,
                    total_return=0,
                    win_rate=0,
                    sample_count=0,
                    last_updated=datetime.now().isoformat(),
                ),
            ).overall_score,
        )

        best_model = enabled_models[best_model_type]
        predictions = best_model.predict(X)

        return predictions, best_model_type

    def get_status_report(self) -> str:
        """Generate status report for all models."""
        report = "Advanced ML Models Status Report\n"
        report += f"{'=' * 50}\n\n"

        if not TENSORFLOW_AVAILABLE:
            report += "❌ TensorFlow is not available\n"
            report += "   Install with: pip install tensorflow\n"
            report += "\nNo models can be trained or used without TensorFlow.\n"
            return report

        report += f"Performance Threshold: {self.performance_threshold}\n"
        report += f"Auto-disable: {self.auto_disable}\n\n"

        if not self.performances:
            report += "No models trained yet.\n"
            report += "Run with --train-advanced to train models.\n"
            return report

        for model_type, performance in self.performances.items():
            enabled = (
                "✅ ENABLED" if self.is_model_enabled(model_type) else "❌ DISABLED"
            )
            report += f"{model_type.value.upper()}: {enabled}\n"
            report += f"  Overall Score: {performance.overall_score:.1f}/100\n"
            report += f"  Accuracy: {performance.accuracy:.2%}\n"
            report += f"  Sharpe Ratio: {performance.sharpe_ratio:.2f}\n"
            report += f"  Win Rate: {performance.win_rate:.2%}\n"
            report += f"  Last Updated: {performance.last_updated}\n\n"

        return report


# Global instance
_advanced_ml_manager: AdvancedMLManager = None


def get_advanced_ml_manager() -> AdvancedMLManager:
    """Get global AdvancedMLManager instance."""
    global _advanced_ml_manager

    if _advanced_ml_manager is None:
        _advanced_ml_manager = AdvancedMLManager()

    return _advanced_ml_manager


def is_advanced_ml_enabled(model_type: ModelType = None) -> bool:
    """
    Check if advanced ML model(s) are enabled.

    Args:
        model_type: Specific model type to check, or None for any

    Returns:
        True if model(s) are enabled
    """
    manager = get_advanced_ml_manager()

    if model_type is not None:
        return manager.is_model_enabled(model_type)

    return any(
        manager.is_model_enabled(mt) for mt in [ModelType.LSTM, ModelType.TRANSFORMER]
    )


def get_advanced_ml_prediction(X: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
    """
    Get prediction from best enabled advanced model.

    Args:
        X: Input features

    Returns:
        Tuple of (predictions, model_name)
    """
    manager = get_advanced_ml_manager()

    if not is_advanced_ml_enabled():
        return np.full(len(X), 0.5), None

    predictions, model_type = manager.get_best_prediction(X)

    if model_type is None:
        return predictions, None

    return predictions, model_type.value


def train_advanced_models(data: pd.DataFrame, target_column: str = "target") -> Dict:
    """
    Train all advanced models and evaluate.

    Args:
        data: Training data with features and target
        target_column: Name of target column

    Returns:
        Dict with training results
    """
    manager = get_advanced_ml_manager()

    results = {}

    # Train LSTM
    if TENSORFLOW_AVAILABLE:
        lstm_perf = manager.train_lstm(data, target_column)
        results["lstm"] = {
            "trained": lstm_perf is not None,
            "performance": lstm_perf.__dict__ if lstm_perf else None,
        }

        # Train Transformer
        transformer_perf = manager.train_transformer(data, target_column)
        results["transformer"] = {
            "trained": transformer_perf is not None,
            "performance": transformer_perf.__dict__ if transformer_perf else None,
        }

    return results
