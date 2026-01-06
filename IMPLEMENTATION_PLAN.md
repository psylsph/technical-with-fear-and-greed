# Implementation Plan for Trading Algorithm Improvements

## Current Status (2026-01-06)

**Completed:**
- ✅ Technical Indicators (RSI, MACD) with 100% test coverage
- ✅ Dynamic FGI Thresholds (rolling quantiles)
- ✅ Enhanced Risk Management (trailing stops, position sizing, take profit)
- ✅ ML Integration (Random Forest for FGI prediction)
- ✅ Live/Test Trading Engine with state persistence
- ✅ SQLite Data Caching with multi-source merging
- ✅ Modular Architecture (8 modules)
- ✅ Comprehensive Testing (52 tests, 48% coverage)

**In Progress:**
- ⚠️ Multiple Timeframe Analysis (signal filtering missing)

**Pending:**
- Multi-asset diversification
- Automated parameter optimization
- Kelly criterion / advanced risk
- Additional sentiment & on-chain metrics

**Next Priority:** Cross-timeframe signal filtering (see Stage 4 below)

---

## Objective
Enhance the existing fear & greed-based BTC strategy by integrating technical indicators, dynamic thresholds, improved risk management, and multi-timeframe analysis. The goal is to create a more robust, adaptable system that outperforms the current fixed-threshold approach while maintaining code quality per AGENTS.md guidelines (PEP 8, type hints, docstrings, etc.).

## Scope
Implement ideas 1-4 from initial suggestions. Prioritize in this order: 1. Technical Indicators, 2. Dynamic Thresholds, 3. Enhanced Risk Management, 4. Multiple Timeframe Analysis.

## Assumptions
- Target improvements for BTC-USD on 1D/1H timeframes.
- Use existing libraries (pandas, vectorbt) and follow code conventions.
- No live trading yet; focus on backtesting enhancements.
- User will provide feedback on tradeoffs (e.g., complexity vs. performance).

## TODO List

### 1. ✅ Add Technical Indicators (Foundation Layer) - **COMPLETED**
- [x] Research and select 2-3 indicators via codesearch for vectorbt examples.
- [x] Add indicator calculation functions in src/indicators.py (calculate_rsi, calculate_macd).
- [x] Modify run_strategy() to incorporate indicators in entry/exit logic (RSI < 30 for buy, RSI > 70 for sell).
- [x] Test with sample data; add unit tests for indicator accuracy.
- [x] Run linting (ruff check/format) and backtest to verify improved win rate.

**Status:**
- Created `src/indicators.py` with RSI and MACD calculation functions
- Integrated indicators into `src/strategy.py` buy/sell logic
- Added comprehensive test coverage in `tests/test_trading_strategy.py`
- All ruff checks passing (100% indicator test coverage)

**Post-Stage Check:** ✅ Pass - Indicators implemented, tests passing, code quality maintained

### 2. ✅ Dynamic Thresholds (Adaptability Layer) - **COMPLETED**
- [x] Implement threshold calculation (dynamic FGI thresholds using rolling quantiles).
- [x] Update run_strategy() to use dynamic thresholds instead of constants.
- [ ] Add backtesting across different market conditions (bull/bear periods).
- [x] Validate with historical simulations; compare to fixed thresholds.
- [x] Document changes in code with docstrings.

**Status:**
- Implemented dynamic FGI thresholds in `src/strategy.py` (lines 103-104, 113-114)
- Buy threshold: `fgi_df["fgi_value"].rolling(30, min_periods=1).quantile(buy_quantile)`
- Sell threshold: `fgi_df["fgi_value"].rolling(30, min_periods=1).quantile(sell_quantile)`
- Configurable via `buy_quantile` and `sell_quantile` parameters
- Default values: 0.2 (buy) and 0.8 (sell)

**Post-Stage Check:** ✅ Partial Pass - Dynamic thresholds implemented, needs market condition backtesting

### 3. ✅ Enhanced Risk Management (Protection Layer) - **COMPLETED**
- [x] Refactor run_strategy() for trailing stop logic (adjusts on price moves).
- [x] Implement position sizing function (10% of equity/cash, 95% allocation).
- [x] Add portfolio-level stops (take profit at 25%, trailing stop at 10%).
- [x] Backtest with vectorbt's portfolio analytics.
- [x] Ensure type hints and error handling per AGENTS.md.

**Status:**
- Trailing stop implemented: `trailing_stop = max(trailing_stop, price * (1 - trail_pct))`
- Take profit: 25% gain triggers exit
- Position sizing: 10% of equity (live) or cash (test)
- All exits include checks: extreme greed, overbought RSI, take profit, trailing stop
- Full type hints and error handling in place

**Post-Stage Check:** ✅ Pass - Risk management fully implemented with proper sizing and stops

### 4. ✅ Multiple Timeframe Analysis (Integration Layer) - **PARTIALLY COMPLETED**
- [x] Modify data fetching to support multiple granularities simultaneously.
- [x] Add timeframe alignment logic (resample daily data for ML features).
- [x] Update run_strategy() to support multiple granularities (runs independently).
- [ ] Test with cross-TF backtests (cross-timeframe signal filtering).
- [ ] Add logging for TF conflicts.

**Status:**
- `main.py` supports multiple granularities (ONE_FIFTEEN_MINUTE, ONE_HOUR, FOUR_HOUR, ONE_DAY)
- Each timeframe runs independently with optimized parameters
- ML model trained on daily data used across all timeframes
- **Missing**: Cross-timeframe signal filtering (e.g., daily FGI affects hourly trades)
- **Missing**: Cross-TF backtesting to verify signal consistency

**Post-Stage Check:** ⚠️ Partial Pass - Multi-TF data fetching works, signal filtering not implemented

## Additional Ideas for Further Improvement

## Additional Ideas for Further Improvement

### 5. ✅ Machine Learning Integration (Prediction Layer) - **COMPLETED**
- [x] Collect data, train model (Random Forest), integrate predictions, backtest with walk-forward validation.

**Status:**
- `src/ml/ml_model.py` implements Random Forest classifier
- Features: FGI, close price, RSI, volume, FGI lag-1
- Target: 1 if next FGI increases, 0 otherwise
- Live prediction function: `predict_live_fgi()`
- Integrated into `run_strategy()` as optional ML filter
- ML predictions buy signals when `pred_val > ml_thresh`

**Note:** Walk-forward validation not yet implemented; uses historical training only.

### 6. Portfolio Diversification (Expansion Layer)
- [ ] Fetch multi-asset data, adapt strategy, implement correlation checks, backtest portfolio.

**Status:**
- Currently single-asset (BTC-USD) only
- Architecture supports multi-asset expansion
- Database schema can handle multiple symbols

### 7. Parameter Optimization (Tuning Layer)
- [ ] Define ranges, use grid search, validate out-of-sample, update code.

**Status:**
- BEST_PARAMS defined in `src/config.py` with optimized values
- Manual optimization completed
- **Missing**: Automated grid search / parameter sweep

### 8. Advanced Risk Management (Protection Layer)
- [ ] Implement Kelly formula, calculate sizes, add portfolio stops, backtest.

**Status:**
- Basic risk management complete (trailing stops, position sizing)
- **Missing**: Kelly criterion implementation
- **Missing**: Portfolio-level circuit breakers (drawdown stops)

### 9. Sentiment and On-Chain Integration (Data Layer)
- [ ] Source APIs, add features, test correlation, backtest.

**Status:**
- Fear & Greed Index integrated
- **Missing**: Additional sentiment sources
- **Missing**: On-chain metrics

### 10. ✅ Live Trading and Monitoring Setup (Execution Layer) - **COMPLETED**
- [x] Integrate broker API, add execution logic, run paper/live trading.

**Status:**
- `src/trading/trading_engine.py` implements live trading
- Alpaca Trading API integration (optional, graceful fallback)
- Test mode: `--test` flag for simulated trading
- Portfolio state persistence: `test_portfolio_state.json`
- Trade logging: `trade_log.json`
- **Note**: Paper trading implemented, live trading requires Alpaca credentials

## Overall Status Summary

**Completed:** Stages 1-3, Items 5, 10
**In Progress:** Stage 4 (Multi-Timeframe signal filtering)
**Pending:** Items 6, 7, 8, 9

**Key Achievements:**
- ✅ Modular architecture (8 modules)
- ✅ SQLite data caching system
- ✅ Technical indicators (RSI, MACD)
- ✅ Dynamic FGI thresholds
- ✅ Enhanced risk management (trailing stops, position sizing)
- ✅ ML model integration (Random Forest)
- ✅ Live/test trading modes
- ✅ 48% test coverage (52 passing tests)
- ✅ All code quality checks passing (ruff)

**Code Quality:**
- PEP 8 compliant
- Type hints on all public functions
- Comprehensive docstrings
- Error handling throughout
- Automated testing (pytest)
- Linting/formatting (ruff)

## Overall Timeline
**Estimated:** 7-10 days (original)
**Actual:** Ongoing - Core implementation complete, optimization/monitoring phase

**Remaining Work:**
1. Multi-timeframe signal filtering (2-3 hours)
2. Cross-TF backtesting and validation (2-3 hours)
3. Optional: Kelly criterion / advanced risk (3-4 hours)
4. Optional: Parameter optimization framework (4-6 hours)

**Total Effort:** ~30-40 hours (core), +10-15 hours for optional enhancements

## Testing Protocol
- Baseline: Run current strategy first to record initial metrics (win rate, Sharpe, outperformance).
- After each stage: Run `python trading_strategy.py`, compare metrics to previous stage. Log in performance_log.md.
- If no improvement (e.g., win rate drops >5%), revert and iterate.
- Always verify fees (MAKER_FEE=0.0015, TAKER_FEE=0.0025) in Portfolio output.

## Resources Needed
- Access to codesearch/websearch for indicator implementations.
- Sample data for validation.
- User input on parameter values.

## Risks/Tradeoffs
- Increased complexity may slow backtests—tradeoff: performance gains vs. simplicity.
- Potential for data look-ahead bias—mitigate with walk-forward testing.
- No new dependencies to avoid bloat.
