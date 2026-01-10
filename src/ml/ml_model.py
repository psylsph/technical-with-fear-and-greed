"""
Machine learning model for FGI prediction.
"""

import importlib.util
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TENSORFLOW_AVAILABLE = importlib.util.find_spec("tensorflow") is not None

if TENSORFLOW_AVAILABLE:
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        TENSORFLOW_AVAILABLE = False
        print("TensorFlow not available. LSTM ensemble disabled.")
else:
    print("TensorFlow not available. LSTM ensemble disabled.")

from ..indicators import calculate_rsi  # noqa: E402

# Global variables for ML model and predictions
ml_model = None
pred_series = None


def prepare_ml_data(
    ohlcv: pd.DataFrame, fgi_df: pd.DataFrame, rsi: pd.Series
) -> pd.DataFrame:
    """Prepare dataset for ML training with enhanced features.

    Args:
        ohlcv: OHLCV price data (open, high, low, close, volume)
        fgi_df: Fear & Greed Index data
        rsi: RSI indicator series

    Returns:
        DataFrame with features and target variable
    """
    close = ohlcv["close"]
    volume = ohlcv["volume"]

    # Ensure all series use the same index
    common_index = close.index

    # Reindex fgi_df to match ohlcv index
    fgi_aligned = fgi_df.reindex(common_index, method="ffill")
    fgi_value = fgi_aligned["fgi_value"]

    # Reindex rsi to match
    rsi = rsi.reindex(common_index)

    # Price momentum features
    returns_3d = close.pct_change(3)
    returns_7d = close.pct_change(7)
    returns_30d = close.pct_change(30)

    # Volatility features
    volatility_7d = close.rolling(window=7).std()
    volatility_30d = close.rolling(window=30).std()

    # True Range (for ATR-like measure)
    true_range = pd.concat(
        [
            ohlcv["high"] - ohlcv["low"],
            (ohlcv["high"] - close.shift(1)).abs(),
            (ohlcv["low"] - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_14d = true_range.rolling(window=14).mean()

    # Volume features
    volume_ma_7d = volume.rolling(window=7).mean()
    volume_ratio = volume / volume_ma_7d

    # FGI features
    fgi_lag1 = fgi_value.shift(1)
    fgi_ma_7d = fgi_value.rolling(window=7).mean()

    # Price-FGI correlation (rolling)
    price_fgi_corr = close.rolling(window=30).corr(fgi_value).fillna(0)

    # Build DataFrame with explicit index to avoid duplicates
    data_dict = {
        # Price features
        "close": close.values,
        "returns_3d": returns_3d.values,
        "returns_7d": returns_7d.values,
        "returns_30d": returns_30d.values,
        # Volatility features
        "volatility_7d": volatility_7d.values,
        "volatility_30d": volatility_30d.values,
        "atr_14d": atr_14d.values,
        # RSI feature
        "rsi": rsi.values,
        # FGI features
        "fgi": fgi_value.values,
        "fgi_lag1": fgi_lag1.values,
        "fgi_ma_7d": fgi_ma_7d.values,
        # Volume features
        "volume": volume.values,
        "volume_ratio": volume_ratio.values,
        # Correlation feature
        "price_fgi_corr": price_fgi_corr.values,
    }

    df = pd.DataFrame(data_dict, index=common_index).dropna()

    # Target: predict PRICE direction (not FGI direction)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # Drop rows where target is NaN (last row)
    df = df.dropna(subset=["target"])

    return df


def train_ml_model(
    daily_ohlcv: pd.DataFrame,
    fgi_df: pd.DataFrame,
    lookback_days: int = 180,
    use_ensemble: bool = True,
) -> tuple[RandomForestClassifier, pd.Series, dict]:
    """Train ML model on historical data (with optional ensemble).

    Args:
        daily_ohlcv: Daily OHLCV price data
        fgi_df: Fear & Greed Index data
        lookback_days: Number of days of recent data to use for training
        use_ensemble: Whether to use RF+LSTM ensemble (default True)
    """
    global ml_model, pred_series

    # Use ensemble if requested and TensorFlow available
    if use_ensemble and TENSORFLOW_AVAILABLE:
        rf_model, lstm_model, ensemble_preds, ensemble_metrics = train_ml_ensemble(
            daily_ohlcv, fgi_df, lookback_days
        )
        return rf_model, ensemble_preds, ensemble_metrics

    # Fallback to Random Forest only
    print(
        "  Training Random Forest Model (Ensemble disabled or TensorFlow unavailable)..."
    )

    daily_rsi = calculate_rsi(daily_ohlcv["close"])
    ml_df = prepare_ml_data(daily_ohlcv, fgi_df, daily_rsi)

    # Use only recent data for training (rolling window)
    cutoff_date = ml_df.index.max() - pd.Timedelta(days=lookback_days)
    ml_df_recent = ml_df[ml_df.index >= cutoff_date]

    print(f"  Using data from {lookback_days} days ({len(ml_df_recent)} samples)")

    # All features
    feature_columns = [
        "close",
        "returns_3d",
        "returns_7d",
        "returns_30d",
        "volatility_7d",
        "volatility_30d",
        "atr_14d",
        "rsi",
        "fgi",
        "fgi_lag1",
        "fgi_ma_7d",
        "volume",
        "volume_ratio",
        "price_fgi_corr",
    ]

    features = ml_df_recent[feature_columns]
    target = ml_df_recent["target"]

    # Split into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Create recency weights - recent data gets higher weight
    sample_weights = np.linspace(0.5, 2.0, len(X_train))

    ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
    ml_model.fit(X_train, y_train, sample_weight=sample_weights)

    # Predictions for evaluation
    y_pred = ml_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "lookback_days": lookback_days,
        "features": len(feature_columns),
        "model_type": "random_forest",
        "ensemble_used": False,
    }

    print("  Random Forest Model Performance:")
    print(f"    Accuracy: {accuracy:.3f}")
    print(f"    Precision: {precision:.3f}")
    print(f"    Recall: {recall:.3f}")
    print(f"    Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"    Features: {len(feature_columns)}")

    # Predict on full dataset for backtesting
    pred_proba = ml_model.predict_proba(ml_df[feature_columns])[:, 1]
    pred_series = pd.Series(pred_proba, index=ml_df.index)

    return ml_model, pred_series, metrics


def predict_live_fgi(
    ohlcv: pd.DataFrame, fgi_df: pd.DataFrame, date: pd.Timestamp
) -> float:
    """Make live prediction for price direction using trained ML model."""
    global ml_model

    if ml_model is None:
        return 0.5  # Default if no model trained

    date_only = date.normalize()

    try:
        # Handle both OHLCV DataFrame and single-value Series
        if len(ohlcv) == 1 and isinstance(ohlcv, pd.Series):
            close_val = ohlcv.iloc[-1]
            close = pd.Series([close_val], index=ohlcv.index)
            volume = pd.Series([1], index=ohlcv.index)  # Default volume

            # Create minimal OHLCV for single value
            ohlcv_expanded = pd.DataFrame(
                {
                    "open": close_val,
                    "high": close_val,
                    "low": close_val,
                    "close": close_val,
                    "volume": 1,
                },
                index=close.index,
            )
        else:
            ohlcv_expanded = (
                ohlcv
                if isinstance(ohlcv, pd.DataFrame)
                else ohlcv.to_frame(name="close")
            )
            if "high" not in ohlcv_expanded.columns:
                ohlcv_expanded["high"] = ohlcv_expanded["close"]
            if "low" not in ohlcv_expanded.columns:
                ohlcv_expanded["low"] = ohlcv_expanded["close"]
            if "volume" not in ohlcv_expanded.columns:
                ohlcv_expanded["volume"] = 1

            close = ohlcv_expanded["close"]
            volume = ohlcv_expanded["volume"]
            close_val = close.iloc[-1]

        # Calculate same features as training
        # Price momentum (use default 0 if not enough history)
        returns_3d = close.pct_change(3).iloc[-1] if len(close) >= 3 else 0
        returns_7d = close.pct_change(7).iloc[-1] if len(close) >= 7 else 0
        returns_30d = close.pct_change(30).iloc[-1] if len(close) >= 30 else 0

        # Volatility
        volatility_7d = (
            close.rolling(window=7).std().iloc[-1]
            if len(close) >= 7
            else close_val * 0.02
        )
        volatility_30d = (
            close.rolling(window=30).std().iloc[-1]
            if len(close) >= 30
            else close_val * 0.05
        )

        # ATR (default to price * 0.02 if insufficient data)
        true_range = pd.concat(
            [
                ohlcv_expanded["high"] - ohlcv_expanded["low"],
                (ohlcv_expanded["high"] - close.shift(1)).abs(),
                (ohlcv_expanded["low"] - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr_14d = (
            true_range.rolling(window=14).mean().iloc[-1]
            if len(close) >= 14
            else close_val * 0.02
        )

        # RSI
        rsi_val = calculate_rsi(close, window=14).iloc[-1] if len(close) >= 14 else 50.0

        # FGI features
        fgi_val = fgi_df.loc[date_only, "fgi_value"]
        lagged_date = date_only - pd.Timedelta(days=1)
        fgi_lag1 = (
            fgi_df.loc[lagged_date, "fgi_value"]
            if lagged_date in fgi_df.index
            else fgi_val
        )
        fgi_ma_7d = (
            fgi_df["fgi_value"].rolling(window=7).mean().loc[date_only]
            if date_only in fgi_df.index
            else fgi_val
        )

        # Volume
        volume_ma_7d = (
            volume.rolling(window=7).mean().iloc[-1]
            if len(volume) >= 7
            else volume.iloc[-1]
        )
        volume_ratio = volume.iloc[-1] / volume_ma_7d if volume_ma_7d > 0 else 1

        # Price-FGI correlation (default to 0 for single-value input)
        price_fgi_corr = 0
        if len(close) >= 10 and len(fgi_df) >= 10:
            recent_window = min(30, len(fgi_df))
            fgi_recent = fgi_df["fgi_value"].iloc[-recent_window:]
            price_recent = (
                close.iloc[-recent_window:] if len(close) >= recent_window else close
            )
            price_fgi_corr = (
                price_recent.rolling(window=min(10, len(fgi_recent)))
                .corr(fgi_recent)
                .iloc[-1]
                if len(price_recent) >= 10
                else 0
            )

        # Create feature array (must match training features)
        features = [
            [
                close_val,  # close
                returns_3d,  # returns_3d
                returns_7d,  # returns_7d
                returns_30d,  # returns_30d
                volatility_7d,  # volatility_7d
                volatility_30d,  # volatility_30d
                atr_14d,  # atr_14d
                rsi_val,  # rsi
                fgi_val,  # fgi
                fgi_lag1,  # fgi_lag1
                fgi_ma_7d,  # fgi_ma_7d
                volume.iloc[-1],  # volume
                volume_ratio,  # volume_ratio
                price_fgi_corr,  # price_fgi_corr
            ]
        ]

        # Make prediction
        pred_proba = ml_model.predict_proba(features)[0, 1]
        return pred_proba

    except (KeyError, IndexError) as e:
        print(f"Error making ML prediction: {e}")
        return 0.5


# Global variables for ensemble models
lstm_model = None
ensemble_weights = {"rf": 0.6, "lstm": 0.4}  # Default weights


def create_lstm_sequences(data: pd.DataFrame, sequence_length: int = 10) -> tuple:
    """Create sequences for LSTM training.

    Args:
        data: DataFrame with features
        sequence_length: Length of sequences for LSTM

    Returns:
        Tuple of (X_sequences, y_targets)
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow not available for LSTM")

    feature_cols = [col for col in data.columns if col != "target"]
    sequences = []
    targets = []

    for i in range(len(data) - sequence_length):
        seq = data[feature_cols].iloc[i : i + sequence_length].values
        target = data["target"].iloc[i + sequence_length]
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)


def train_lstm_model(ml_df: pd.DataFrame, sequence_length: int = 10) -> tuple:
    """Train LSTM model for time series prediction.

    Args:
        ml_df: Prepared ML dataframe
        sequence_length: Sequence length for LSTM

    Returns:
        Tuple of (lstm_model, training_history)
    """
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Skipping LSTM training.")
        return None, None

    try:
        # Prepare sequences
        X_sequences, y_targets = create_lstm_sequences(ml_df, sequence_length)

        if len(X_sequences) == 0:
            print("Not enough data for LSTM sequences")
            return None, None

        # Split data
        split_idx = int(len(X_sequences) * 0.8)
        X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
        y_train, y_test = y_targets[:split_idx], y_targets[split_idx:]

        # Scale features
        scaler = StandardScaler()
        n_samples, seq_len, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_train = X_train_scaled.reshape(n_samples, seq_len, n_features)

        X_test_reshaped = X_test.reshape(-1, n_features)
        X_test_scaled = scaler.transform(X_test_reshaped)
        X_test = X_test_scaled.reshape(X_test.shape[0], seq_len, n_features)

        # Build LSTM model
        model = keras.Sequential(
            [
                layers.LSTM(
                    50,
                    activation="relu",
                    input_shape=(seq_len, n_features),
                    return_sequences=True,
                ),
                layers.Dropout(0.2),
                layers.LSTM(30, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid"),
            ]
        )

        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        # Train model
        history = model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0,
        )

        return model, history

    except Exception as e:
        print(f"Error training LSTM: {e}")
        return None, None


def train_ml_ensemble(
    daily_ohlcv: pd.DataFrame,
    fgi_df: pd.DataFrame,
    lookback_days: int = 180,
    lstm_sequence_length: int = 10,
) -> tuple:
    """Train ensemble ML model combining Random Forest and LSTM.

    Args:
        daily_ohlcv: Daily OHLCV price data
        fgi_df: Fear & Greed Index data
        lookback_days: Number of days of recent data to use for training
        lstm_sequence_length: Sequence length for LSTM

    Returns:
        Tuple of (rf_model, lstm_model, ensemble_predictions, metrics)
    """
    global ml_model, lstm_model, pred_series

    print("  Training ML Ensemble (Random Forest + LSTM)...")

    # Prepare data
    daily_rsi = calculate_rsi(daily_ohlcv["close"])
    ml_df = prepare_ml_data(daily_ohlcv, fgi_df, daily_rsi)

    # Use only recent data for training
    cutoff_date = ml_df.index.max() - pd.Timedelta(days=lookback_days)
    ml_df_recent = ml_df[ml_df.index >= cutoff_date]

    print(f"  Using data from {lookback_days} days ({len(ml_df_recent)} samples)")

    # Feature columns
    feature_columns = [
        "close",
        "returns_3d",
        "returns_7d",
        "returns_30d",
        "volatility_7d",
        "volatility_30d",
        "atr_14d",
        "rsi",
        "fgi",
        "fgi_lag1",
        "fgi_ma_7d",
        "volume",
        "volume_ratio",
        "price_fgi_corr",
    ]

    # Train Random Forest
    features = ml_df_recent[feature_columns]
    target = ml_df_recent["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    sample_weights = np.linspace(0.5, 2.0, len(X_train))
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train, sample_weight=sample_weights)

    # Train LSTM
    lstm_model, lstm_history = train_lstm_model(ml_df_recent, lstm_sequence_length)

    # Generate predictions
    rf_predictions = rf_model.predict_proba(ml_df[feature_columns])[:, 1]

    ensemble_predictions = rf_predictions.copy()  # Default to RF if LSTM fails

    if lstm_model is not None and TENSORFLOW_AVAILABLE:
        try:
            # Generate LSTM predictions (simplified - using last sequence)
            lstm_preds = []
            feature_cols = [col for col in feature_columns]

            for i in range(len(ml_df)):
                if i >= lstm_sequence_length:
                    seq = ml_df[feature_cols].iloc[i - lstm_sequence_length : i].values
                    # Scale sequence (simplified)
                    seq_scaled = (seq - seq.mean(axis=0)) / (seq.std(axis=0) + 1e-8)
                    pred = lstm_model.predict(
                        seq_scaled.reshape(1, lstm_sequence_length, -1), verbose=0
                    )[0][0]
                    lstm_preds.append(pred)
                else:
                    lstm_preds.append(0.5)  # Default for early data

            lstm_predictions = np.array(lstm_preds)

            # Ensemble combination
            rf_weight = ensemble_weights["rf"]
            lstm_weight = ensemble_weights["lstm"]
            ensemble_predictions = (
                rf_weight * rf_predictions + lstm_weight * lstm_predictions
            )

        except Exception as e:
            print(f"  LSTM prediction failed, using RF only: {e}")
    else:
        print("  LSTM not available, using Random Forest only")

    # Evaluate ensemble
    rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
    ensemble_accuracy = accuracy_score(
        target[len(target) - len(X_test) :],
        (ensemble_predictions[len(ensemble_predictions) - len(X_test) :] > 0.5).astype(
            int
        ),
    )

    metrics = {
        "accuracy": ensemble_accuracy,  # Primary metric for backward compatibility
        "precision": 0.0,  # Not calculated for ensemble
        "recall": 0.0,  # Not calculated for ensemble
        "rf_accuracy": rf_accuracy,
        "ensemble_accuracy": ensemble_accuracy,
        "improvement": ensemble_accuracy - rf_accuracy,
        "lstm_available": lstm_model is not None,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "lookback_days": lookback_days,
        "features": len(feature_columns),
    }

    print("  Ensemble Model Performance:")
    print(f"    RF Accuracy: {rf_accuracy:.3f}")
    print(f"    Ensemble Accuracy: {ensemble_accuracy:.3f}")
    print(f"    Improvement: {metrics['improvement']:+.3f}")
    print(f"    LSTM Available: {metrics['lstm_available']}")

    # Store predictions
    pred_series = pd.Series(ensemble_predictions, index=ml_df.index)
    ml_model = rf_model  # Keep RF as primary model for compatibility

    return rf_model, lstm_model, pred_series, metrics
