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
    fgi_value = fgi_df["fgi_value"]
    fgi_lag1 = fgi_value.shift(1)
    fgi_ma_7d = fgi_value.rolling(window=7).mean()

    # Price-FGI correlation (rolling)
    price_fgi_corr = close.rolling(window=30).corr(fgi_value).fillna(0)

    df = pd.DataFrame(
        {
            # Price features
            "close": close,
            "returns_3d": returns_3d,
            "returns_7d": returns_7d,
            "returns_30d": returns_30d,
            # Volatility features
            "volatility_7d": volatility_7d,
            "volatility_30d": volatility_30d,
            "atr_14d": atr_14d,
            # RSI feature
            "rsi": rsi,
            # FGI features
            "fgi": fgi_value,
            "fgi_lag1": fgi_lag1,
            "fgi_ma_7d": fgi_ma_7d,
            # Volume features
            "volume": volume,
            "volume_ratio": volume_ratio,
            # Correlation feature
            "price_fgi_corr": price_fgi_corr,
        }
    ).dropna()

    # Target: predict PRICE direction (not FGI direction)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    return df


def train_ml_model(
    daily_ohlcv: pd.DataFrame, fgi_df: pd.DataFrame, lookback_days: int = 180
) -> tuple[RandomForestClassifier, pd.Series, dict]:
    """Train ML model on historical data.

    Args:
        daily_ohlcv: Daily OHLCV price data
        fgi_df: Fear & Greed Index data
        lookback_days: Number of days of recent data to use for training
    """
    global ml_model, pred_series

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
    }

    print("  ML Model Performance:")
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
