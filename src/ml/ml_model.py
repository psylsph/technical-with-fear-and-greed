"""
Machine learning model for FGI prediction.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from ..indicators import calculate_rsi

# Global variables for ML model and predictions
ml_model = None
pred_series = None


def prepare_ml_data(
    close: pd.Series, fgi_df: pd.DataFrame, rsi: pd.Series, volume: pd.Series = None
) -> pd.DataFrame:
    """Prepare dataset for ML training."""
    if volume is None:
        volume = close * 0.01  # dummy
    df = pd.DataFrame(
        {
            "fgi": fgi_df["fgi_value"],
            "close": close,
            "rsi": rsi,
            "volume": volume,
            "fgi_lag1": fgi_df["fgi_value"].shift(1),
        }
    ).dropna()
    df["target"] = (df["fgi"].shift(-1) > df["fgi"]).astype(int)  # 1 if next FGI up
    return df


def train_ml_model(
    daily_close: pd.Series, fgi_df: pd.DataFrame, lookback_days: int = 180
) -> tuple[RandomForestClassifier, pd.Series, dict]:
    """Train ML model on historical data.

    Args:
        daily_close: Daily price data
        fgi_df: Fear & Greed Index data
        lookback_days: Number of days of recent data to use for training
    """
    global ml_model, pred_series

    daily_rsi = calculate_rsi(daily_close)
    volume = daily_close * 0.01  # dummy volume
    ml_df = prepare_ml_data(daily_close, fgi_df, daily_rsi, volume)

    # Use only recent data for training (rolling window)
    cutoff_date = ml_df.index.max() - pd.Timedelta(days=lookback_days)
    ml_df_recent = ml_df[ml_df.index >= cutoff_date]

    print(f"  Using data from {lookback_days} days ({len(ml_df_recent)} samples)")

    features = ml_df_recent[["fgi", "close", "rsi", "volume", "fgi_lag1"]]
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
    }

    print("  ML Model Performance:")
    print(f"    Accuracy: {accuracy:.3f}")
    print(f"    Precision: {precision:.3f}")
    print(f"    Recall: {recall:.3f}")
    print(f"    Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Predict on full dataset for backtesting
    pred_proba = ml_model.predict_proba(
        ml_df[["fgi", "close", "rsi", "volume", "fgi_lag1"]]
    )[:, 1]
    pred_series = pd.Series(pred_proba, index=ml_df.index)

    return ml_model, pred_series, metrics


def predict_live_fgi(
    close: pd.Series, fgi_df: pd.DataFrame, date: pd.Timestamp
) -> float:
    """Make live prediction for FGI direction using trained ML model."""
    global ml_model

    if ml_model is None:
        return 0.5  # Default if no model trained

    date_only = date.normalize()

    # Get features for prediction
    try:
        fgi_val = fgi_df.loc[date_only, "fgi_value"]
        close_val = close.iloc[-1]
        rsi_val = calculate_rsi(close, window=14).iloc[-1] if len(close) >= 14 else 50.0
        volume_val = close_val * 0.01  # dummy volume

        # Get lagged FGI (previous day)
        lagged_date = date_only - pd.Timedelta(days=1)
        fgi_lag1 = (
            fgi_df.loc[lagged_date, "fgi_value"]
            if lagged_date in fgi_df.index
            else fgi_val
        )

        # Create feature array
        features = [[fgi_val, close_val, rsi_val, volume_val, fgi_lag1]]

        # Make prediction
        pred_proba = ml_model.predict_proba(features)[0, 1]
        return pred_proba

    except (KeyError, IndexError) as e:
        print(f"Error making ML prediction: {e}")
        return 0.5
