"""
Machine learning model for FGI prediction.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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
    daily_close: pd.Series, fgi_df: pd.DataFrame
) -> tuple[RandomForestClassifier, pd.Series]:
    """Train ML model on historical data."""
    global ml_model, pred_series

    daily_rsi = calculate_rsi(daily_close)
    volume = daily_close * 0.01  # dummy volume
    ml_df = prepare_ml_data(daily_close, fgi_df, daily_rsi, volume)
    features = ml_df[["fgi", "close", "rsi", "volume", "fgi_lag1"]]
    target = ml_df["target"]

    ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
    ml_model.fit(features, target)
    pred_proba = ml_model.predict_proba(features)[:, 1]
    pred_series = pd.Series(pred_proba, index=ml_df.index)

    return ml_model, pred_series


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
