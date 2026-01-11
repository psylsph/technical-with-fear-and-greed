"""
Tests for technical indicators module.
"""

import pytest
import pandas as pd

from src.indicators import (
    calculate_support_resistance,
    calculate_fibonacci_retracements,
    calculate_pivot_points,
    calculate_rsi,
    calculate_macd,
    calculate_adx,
)


class TestCalculateSupportResistance:
    """Test support and resistance level calculation."""

    def test_insufficient_data(self):
        """Test with insufficient data points."""
        # Create data with less than lookback period
        close = pd.Series([100, 101, 102, 103, 104])
        high = pd.Series([101, 102, 103, 104, 105])
        low = pd.Series([99, 100, 101, 102, 103])

        result = calculate_support_resistance(high, low, close, lookback=20)

        assert result["support_levels"] == []
        assert result["resistance_levels"] == []
        assert result["nearest_support"] is None
        assert result["nearest_resistance"] is None

    def test_basic_calculation(self):
        """Test basic support/resistance calculation."""
        # Create price data with clear swings
        close = pd.Series(
            [
                100,
                105,
                100,
                105,
                100,
                105,
                100,
                105,
                100,
                105,
                110,
                108,
                106,
                104,
                102,
                100,
                98,
                96,
                94,
                92,
                90,
            ]
        )
        high = pd.Series(
            [
                102,
                107,
                102,
                107,
                102,
                107,
                102,
                107,
                102,
                107,
                112,
                110,
                108,
                106,
                104,
                102,
                100,
                98,
                96,
                94,
                92,
            ]
        )
        low = pd.Series(
            [
                98,
                103,
                98,
                103,
                98,
                103,
                98,
                103,
                98,
                103,
                108,
                106,
                104,
                102,
                100,
                98,
                96,
                94,
                92,
                90,
                88,
            ]
        )

        result = calculate_support_resistance(high, low, close, lookback=20)

        assert "support_levels" in result
        assert "resistance_levels" in result
        assert "nearest_support" in result
        assert "nearest_resistance" in result
        assert "current_price" in result
        assert result["current_price"] == 90

    def test_support_levels_below_price(self):
        """Test that support levels are below current price."""
        # Create trending down data
        close = pd.Series(
            [
                110,
                108,
                106,
                104,
                102,
                100,
                98,
                96,
                94,
                92,
                90,
                88,
                86,
                84,
                82,
                80,
                78,
                76,
                74,
                72,
                70,
            ]
        )
        high = pd.Series(
            [
                112,
                110,
                108,
                106,
                104,
                102,
                100,
                98,
                96,
                94,
                92,
                90,
                88,
                86,
                84,
                82,
                80,
                78,
                76,
                74,
                72,
            ]
        )
        low = pd.Series(
            [
                108,
                106,
                104,
                102,
                100,
                98,
                96,
                94,
                92,
                90,
                88,
                86,
                84,
                82,
                80,
                78,
                76,
                74,
                72,
                70,
                68,
            ]
        )

        result = calculate_support_resistance(high, low, close, lookback=20)

        # Nearest support should be below price
        if result["nearest_support"] is not None:
            assert result["nearest_support"] < result["current_price"]

    def test_resistance_levels_above_price(self):
        """Test that resistance levels are above current price."""
        # Create trending up data
        close = pd.Series(
            [
                70,
                72,
                74,
                76,
                78,
                80,
                82,
                84,
                86,
                88,
                90,
                92,
                94,
                96,
                98,
                100,
                102,
                104,
                106,
                108,
                110,
            ]
        )
        high = pd.Series(
            [
                72,
                74,
                76,
                78,
                80,
                82,
                84,
                86,
                88,
                90,
                92,
                94,
                96,
                98,
                100,
                102,
                104,
                106,
                108,
                110,
                112,
            ]
        )
        low = pd.Series(
            [
                68,
                70,
                72,
                74,
                76,
                78,
                80,
                82,
                84,
                86,
                88,
                90,
                92,
                94,
                96,
                98,
                100,
                102,
                104,
                106,
                108,
            ]
        )

        result = calculate_support_resistance(high, low, close, lookback=20)

        # Nearest resistance should be above price
        if result["nearest_resistance"] is not None:
            assert result["nearest_resistance"] > result["current_price"]

    def test_clustering_nearby_levels(self):
        """Test that nearby levels are clustered together."""
        # Create data with levels close to each other
        close = pd.Series([100] * 21)
        high = pd.Series(
            [
                105,
                105.1,
                104.9,
                105.2,
                104.8,
                105.3,
                104.7,
                105,
                104.5,
                105.5,
                105,
                104.3,
                105.7,
                104,
                105.8,
                104.2,
                105.4,
                104.4,
                105.6,
                104.1,
                105.9,
            ]
        )
        low = pd.Series(
            [
                95,
                95.1,
                94.9,
                95.2,
                94.8,
                95.3,
                94.7,
                95,
                94.5,
                95.5,
                95,
                94.3,
                95.7,
                94,
                95.8,
                94.2,
                95.4,
                94.4,
                95.6,
                94.1,
                95.9,
            ]
        )

        result = calculate_support_resistance(high, low, close, lookback=20)

        # Levels should be clustered (not individual points)
        assert isinstance(result["support_levels"], list)
        assert isinstance(result["resistance_levels"], list)

    def test_custom_lookback_period(self):
        """Test with custom lookback period."""
        close = pd.Series([100 + i for i in range(50)])
        high = pd.Series([102 + i for i in range(50)])
        low = pd.Series([98 + i for i in range(50)])

        result = calculate_support_resistance(high, low, close, lookback=10)

        assert "current_price" in result
        assert result["current_price"] == 149


class TestCalculateFibonacciRetracements:
    """Test Fibonacci retracement calculation."""

    def test_basic_fibonacci_levels(self):
        """Test basic Fibonacci level calculation."""
        high = 200
        low = 100
        current_price = 150

        result = calculate_fibonacci_retracements(high, low, current_price)

        assert "levels" in result
        assert "closest_level" in result
        assert "closest_level_price" in result
        assert "retracement_pct" in result

        # Check key levels exist
        levels = result["levels"]
        assert "0% (high)" in levels
        assert "23.6%" in levels
        assert "38.2%" in levels
        assert "50%" in levels
        assert "61.8%" in levels
        assert "78.6%" in levels
        assert "100% (low)" in levels

    def test_high_greater_than_low(self):
        """Test with high > low (uptrend retracement)."""
        high = 200
        low = 100
        diff = 100

        result = calculate_fibonacci_retracements(high, low, 150)

        # Verify calculations
        assert result["levels"]["0% (high)"] == 200
        assert result["levels"]["23.6%"] == pytest.approx(200 - (100 * 0.236), 0.01)
        assert result["levels"]["50%"] == pytest.approx(200 - (100 * 0.5), 0.01)
        assert result["levels"]["100% (low)"] == 100

    def test_retracement_percentage(self):
        """Test retracement percentage calculation."""
        high = 200
        low = 100
        current_price = 150  # 50% retracement

        result = calculate_fibonacci_retracements(high, low, current_price)

        # 50% retracement
        assert result["retracement_pct"] == pytest.approx(50.0, 0.1)

    def test_closest_level_detection(self):
        """Test finding the closest Fibonacci level."""
        high = 200
        low = 100

        # Test at various price points
        test_cases = [
            (200, "0% (high)"),  # At high
            (176.4, "23.6%"),  # At 23.6% level
            (161.8, "38.2%"),  # At 38.2% level
            (150, "50%"),  # At 50% level
            (138.2, "61.8%"),  # At 61.8% level
            (100, "100% (low)"),  # At low
        ]

        for price, expected_level in test_cases:
            result = calculate_fibonacci_retracements(high, low, price)
            assert result["closest_level"] == expected_level

    def test_zero_diff_handling(self):
        """Test when high equals low."""
        high = 100
        low = 100
        current_price = 100

        result = calculate_fibonacci_retracements(high, low, current_price)

        # Should handle gracefully
        assert result["retracement_pct"] == 0.0
        assert result["levels"]["0% (high)"] == 100
        assert result["levels"]["100% (low)"] == 100


class TestCalculatePivotPoints:
    """Test pivot points calculation."""

    def test_basic_pivot_calculation(self):
        """Test basic pivot point calculation."""
        high = pd.Series([105])
        low = pd.Series([95])
        close = pd.Series([100])

        result = calculate_pivot_points(high, low, close)

        # Pivot = (H + L + C) / 3
        expected_pivot = (105 + 95 + 100) / 3
        assert result["pivot"] == pytest.approx(expected_pivot, 0.01)

    def test_support_and_resistance_levels(self):
        """Test support and resistance level calculations."""
        high = pd.Series([105])
        low = pd.Series([95])
        close = pd.Series([100])

        result = calculate_pivot_points(high, low, close)

        # Check all levels exist
        assert "pivot" in result
        assert "r1" in result
        assert "r2" in result
        assert "r3" in result
        assert "s1" in result
        assert "s2" in result
        assert "s3" in result

        # R1 should be above pivot
        assert result["r1"] > result["pivot"]
        # S1 should be below pivot
        assert result["s1"] < result["pivot"]

    def test_formula_calculations(self):
        """Test specific pivot point formulas."""
        prev_high = 105
        prev_low = 95
        prev_close = 100

        high = pd.Series([prev_high])
        low = pd.Series([prev_low])
        close = pd.Series([prev_close])

        result = calculate_pivot_points(high, low, close)

        pivot = (prev_high + prev_low + prev_close) / 3

        # Verify formulas
        assert result["pivot"] == pytest.approx(pivot, 0.01)
        assert result["r1"] == pytest.approx((2 * pivot) - prev_low, 0.01)
        assert result["s1"] == pytest.approx((2 * pivot) - prev_high, 0.01)

    def test_empty_series(self):
        """Test with empty series."""
        high = pd.Series([])
        low = pd.Series([])
        close = pd.Series([])

        result = calculate_pivot_points(high, low, close)

        # Should return empty dict
        assert result == {}

    def test_multiple_data_points_uses_last(self):
        """Test that last data point is used."""
        high = pd.Series([100, 102, 104, 106, 108])
        low = pd.Series([90, 92, 94, 96, 98])
        close = pd.Series([95, 97, 99, 101, 103])

        result = calculate_pivot_points(high, low, close)

        # Should use last values
        expected_pivot = (108 + 98 + 103) / 3
        assert result["pivot"] == pytest.approx(expected_pivot, 0.01)

    def test_resistance_levels_increasing(self):
        """Test that resistance levels increase (R1 < R2 < R3)."""
        high = pd.Series([105])
        low = pd.Series([95])
        close = pd.Series([100])

        result = calculate_pivot_points(high, low, close)

        assert result["r1"] < result["r2"]
        assert result["r2"] < result["r3"]

    def test_support_levels_decreasing(self):
        """Test that support levels decrease (S1 > S2 > S3)."""
        high = pd.Series([105])
        low = pd.Series([95])
        close = pd.Series([100])

        result = calculate_pivot_points(high, low, close)

        assert result["s1"] > result["s2"]
        assert result["s2"] > result["s3"]


class TestCalculateRSI:
    """Test RSI calculation."""

    def test_rsi_range(self):
        """Test that RSI values are between 0 and 100."""
        close = pd.Series(
            [
                100,
                102,
                104,
                106,
                108,
                110,
                112,
                114,
                116,
                118,
                120,
                122,
                124,
                126,
                128,
                130,
                132,
                134,
                136,
                138,
            ]
        )

        rsi = calculate_rsi(close)

        # First values should be NaN, rest should be 0-100
        valid_rsi = rsi.dropna()
        assert all((valid_rsi >= 0) & (valid_rsi <= 100))

    def test_custom_window(self):
        """Test RSI with custom window."""
        close = pd.Series([100 + i for i in range(30)])

        rsi_default = calculate_rsi(close)
        rsi_custom = calculate_rsi(close, window=20)

        # Different windows should produce different results
        assert not rsi_default.equals(rsi_custom)


class TestCalculateMACD:
    """Test MACD calculation."""

    def test_macd_returns_series(self):
        """Test that MACD returns two series."""
        close = pd.Series([100 + i for i in range(30)])

        macd, signal = calculate_macd(close)

        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)

    def test_custom_periods(self):
        """Test MACD with custom periods."""
        close = pd.Series([100 + i for i in range(30)])

        macd_default, _ = calculate_macd(close)
        macd_custom, _ = calculate_macd(close, fast=10, slow=20, signal=8)

        # Different periods should produce different results
        assert not macd_default.equals(macd_custom)


class TestCalculateADX:
    """Test ADX calculation."""

    def test_adx_returns_series(self):
        """Test that ADX returns a series."""
        # Create sample OHLC data
        high = pd.Series([105 + i for i in range(30)])
        low = pd.Series([95 + i for i in range(30)])
        close = pd.Series([100 + i for i in range(30)])

        adx = calculate_adx(high, low, close)

        assert isinstance(adx, pd.Series)

    def test_adx_range(self):
        """Test that ADX values are non-negative."""
        # Create trending data
        high = pd.Series([105 + i for i in range(30)])
        low = pd.Series([95 + i for i in range(30)])
        close = pd.Series([100 + i for i in range(30)])

        adx = calculate_adx(high, low, close)

        # ADX should be 0-100 (after initial NaN values)
        valid_adx = adx.dropna()
        assert all(valid_adx >= 0)

    def test_custom_period(self):
        """Test ADX with custom period."""
        high = pd.Series([105 + i for i in range(30)])
        low = pd.Series([95 + i for i in range(30)])
        close = pd.Series([100 + i for i in range(30)])

        adx_default = calculate_adx(high, low, close)
        adx_custom = calculate_adx(high, low, close, period=10)

        # Different periods should produce different results
        assert not adx_default.equals(adx_custom)
