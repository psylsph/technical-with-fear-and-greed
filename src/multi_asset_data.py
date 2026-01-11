"""
Multi-asset data fetching and management.

This module extends the existing data fetching system to support
multiple assets with parallel processing and caching.
"""

import asyncio
import concurrent.futures
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config import START_DATE, END_DATE
from src.data.data_fetchers import (
    fetch_fear_greed_index,
    fetch_unified_price_data,
    get_current_price,
)
from src.multi_asset_config import get_asset_config


class MultiAssetDataManager:
    """Manager for fetching and processing data for multiple assets."""
    
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self._fgi_cache: Optional[pd.DataFrame] = None
        self._price_cache: Dict[str, pd.DataFrame] = {}
    
    def fetch_fgi_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch Fear & Greed Index data (shared across all assets)."""
        if self._fgi_cache is None or force_refresh:
            self._fgi_cache = fetch_fear_greed_index()
        return self._fgi_cache
    
    def fetch_asset_data(
        self,
        symbol: str,
        start_date: str = START_DATE,
        end_date: str = END_DATE,
        granularity: str = "ONE_DAY",
        force_refresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Fetch price data for a single asset."""
        cache_key = f"{symbol}_{start_date}_{end_date}_{granularity}"
        
        if cache_key not in self._price_cache or force_refresh:
            # Map granularity to freq parameter
            granularity_to_freq = {
                "ONE_DAY": "1d",
                "ONE_HOUR": "1h",
                "FOUR_HOUR": "4h",
                "FIFTEEN_MINUTE": "15m",
                "FIVE_MINUTE": "5m",
                "ONE_MINUTE": "1m",
            }
            freq = granularity_to_freq.get(granularity, "1d")
            
            data = fetch_unified_price_data(
                symbol=symbol,
                start=start_date,
                end=end_date,
                freq=freq,
            )
            if data is not None and not data.empty:
                self._price_cache[cache_key] = data
        
        return self._price_cache.get(cache_key)
    
    def fetch_multiple_assets(
        self,
        symbols: List[str],
        start_date: str = START_DATE,
        end_date: str = END_DATE,
        granularity: str = "ONE_DAY",
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """Fetch price data for multiple assets in parallel."""
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create futures for each asset
            future_to_symbol = {
                executor.submit(
                    self.fetch_asset_data,
                    symbol,
                    start_date,
                    end_date,
                    granularity,
                ): symbol
                for symbol in symbols
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    results[symbol] = data
                except Exception as e:
                    print(f"Error fetching data for {symbol}: {e}")
                    results[symbol] = None
        
        return results
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """Get current prices for multiple assets."""
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(get_current_price, symbol): symbol
                for symbol in symbols
            }
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    price = future.result()
                    results[symbol] = price
                except Exception as e:
                    print(f"Error getting current price for {symbol}: {e}")
                    results[symbol] = None
        
        return results
    
    def validate_asset_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        min_data_points: int = 100,
    ) -> Tuple[bool, str]:
        """Validate data quality for an asset."""
        if data is None or data.empty:
            return False, "No data available"
        
        if len(data) < min_data_points:
            return False, f"Insufficient data points: {len(data)} < {min_data_points}"
        
        # Check for missing values
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_pct > 0.05:  # More than 5% missing
            return False, f"High missing data: {missing_pct:.1%}"
        
        # Check for zero or negative prices
        if (data['close'] <= 0).any():
            return False, "Invalid price values (zero or negative)"
        
        # Check for extreme outliers (price changes > 50% in one period)
        price_changes = data['close'].pct_change().abs()
        if (price_changes > 0.5).any():
            return False, "Extreme price changes detected"
        
        return True, "Data validation passed"
    
    def prepare_training_data(
        self,
        symbol: str,
        lookback_days: Optional[int] = None,
    ) -> Optional[pd.DataFrame]:
        """Prepare training data for an asset with appropriate lookback."""
        asset_config = get_asset_config(symbol)
        
        if lookback_days is None:
            lookback_days = asset_config.training.lookback_days
        
        # Calculate start date based on lookback
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        # Fetch data
        data = self.fetch_asset_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            granularity="ONE_DAY",
        )
        
        if data is None:
            return None
        
        # Validate data
        is_valid, message = self.validate_asset_data(symbol, data)
        if not is_valid:
            print(f"Warning: Data validation failed for {symbol}: {message}")
            # Still return data but log warning
        
        return data
    
    def get_asset_statistics(self, symbol: str) -> Dict:
        """Calculate statistics for an asset."""
        data = self.fetch_asset_data(symbol)
        if data is None or data.empty:
            return {}
        
        returns = data['close'].pct_change().dropna()
        
        stats = {
            "symbol": symbol,
            "data_points": len(data),
            "start_date": data.index[0].strftime("%Y-%m-%d"),
            "end_date": data.index[-1].strftime("%Y-%m-%d"),
            "current_price": data['close'].iloc[-1],
            "avg_daily_return": returns.mean(),
            "daily_volatility": returns.std(),
            "sharpe_ratio": returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0,
            "max_drawdown": self._calculate_max_drawdown(data['close']),
            "avg_volume": data['volume'].mean(),
        }
        
        return stats
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown from price series."""
        cumulative_returns = (1 + prices.pct_change()).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cached data."""
        if symbol is None:
            self._price_cache.clear()
            self._fgi_cache = None
        else:
            # Remove all cache entries for this symbol
            keys_to_remove = [k for k in self._price_cache.keys() if k.startswith(symbol)]
            for key in keys_to_remove:
                del self._price_cache[key]


# Global data manager instance
data_manager = MultiAssetDataManager()


async def fetch_assets_async(
    symbols: List[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    granularity: str = "ONE_DAY",
) -> Dict[str, Optional[pd.DataFrame]]:
    """Async version of fetch_multiple_assets."""
    loop = asyncio.get_event_loop()
    
    # Run synchronous function in thread pool
    return await loop.run_in_executor(
        None,
        data_manager.fetch_multiple_assets,
        symbols,
        start_date,
        end_date,
        granularity,
    )


def get_portfolio_data(
    symbols: List[str],
    validation_threshold: float = 0.8,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Get validated data for a portfolio of assets.
    
    Returns:
        Tuple of (valid_data_dict, invalid_symbols)
    """
    all_data = data_manager.fetch_multiple_assets(symbols)
    
    valid_data = {}
    invalid_symbols = []
    
    for symbol, data in all_data.items():
        if data is None:
            invalid_symbols.append(symbol)
            continue
        
        is_valid, message = data_manager.validate_asset_data(symbol, data)
        if is_valid:
            valid_data[symbol] = data
        else:
            print(f"Asset {symbol} failed validation: {message}")
            invalid_symbols.append(symbol)
    
    # Check if we have enough valid assets
    valid_ratio = len(valid_data) / len(symbols)
    if valid_ratio < validation_threshold:
        print(f"Warning: Only {valid_ratio:.1%} of assets passed validation")
    
    return valid_data, invalid_symbols


def analyze_correlations(
    symbols: List[str],
    lookback_days: int = 90,
) -> pd.DataFrame:
    """Analyze correlations between assets."""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    
    all_data = data_manager.fetch_multiple_assets(
        symbols,
        start_date=start_date,
        end_date=end_date,
        granularity="ONE_DAY",
    )
    
    # Extract returns for each asset
    returns_data = {}
    for symbol, data in all_data.items():
        if data is not None and not data.empty:
            returns_data[symbol] = data['close'].pct_change().dropna()
    
    # Create correlation matrix
    if not returns_data:
        return pd.DataFrame()
    
    returns_df = pd.DataFrame(returns_data)
    correlation_matrix = returns_df.corr()
    
    return correlation_matrix


def optimize_data_fetching(
    symbols: List[str],
    batch_size: int = 3,
    retry_failed: bool = True,
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Optimized data fetching with batching and retry logic.
    
    Args:
        symbols: List of asset symbols
        batch_size: Number of assets to fetch in parallel
        retry_failed: Whether to retry failed fetches
    
    Returns:
        Dictionary of symbol -> data
    """
    all_results = {}
    failed_symbols = []
    
    # Process in batches
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        print(f"Fetching batch {i//batch_size + 1}/{(len(symbols) + batch_size - 1)//batch_size}: {batch}")
        
        batch_results = data_manager.fetch_multiple_assets(batch)
        
        for symbol, data in batch_results.items():
            if data is not None:
                all_results[symbol] = data
            else:
                failed_symbols.append(symbol)
    
    # Retry failed symbols if requested
    if retry_failed and failed_symbols:
        print(f"Retrying {len(failed_symbols)} failed symbols...")
        retry_results = data_manager.fetch_multiple_assets(failed_symbols)
        
        for symbol, data in retry_results.items():
            if data is not None:
                all_results[symbol] = data
                failed_symbols.remove(symbol)
    
    if failed_symbols:
        print(f"Failed to fetch data for: {failed_symbols}")
    
    return all_results
