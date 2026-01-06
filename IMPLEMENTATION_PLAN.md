# Implementation Plan for Trading Algorithm Improvements

## Objective
Enhance the existing fear & greed-based BTC strategy by integrating technical indicators, dynamic thresholds, improved risk management, and multi-timeframe analysis. The goal is to create a more robust, adaptable system that outperforms the current fixed-threshold approach while maintaining code quality per AGENTS.md guidelines (PEP 8, type hints, docstrings, etc.).

## Scope
Implement ideas 1-4 from initial suggestions. Prioritize in this order: 1. Technical Indicators, 2. Dynamic Thresholds, 3. Enhanced Risk Management, 4. Multiple Timeframe Analysis.

## Assumptions
- Target improvements for BTC-USD on 1D/1H timeframes.
- Use existing libraries (pandas, vectorbt) and follow code conventions.
- No live trading yet; focus on backtesting enhancements.
- User will provide feedback on tradeoffs (e.g., complexity vs. performance).

## Updated Steps by Idea

### 1. Add Technical Indicators (Foundation Layer)
**Description:** Combine FGI with complementary indicators (e.g., RSI for momentum, Bollinger Bands for volatility) to filter signals—e.g., buy only if FGI <=35 AND RSI <30 (oversold).

**Steps:**
1. Research and select 2-3 indicators via codesearch for vectorbt examples.
2. Add indicator calculation functions in trading_strategy.py (e.g., def calculate_rsi(close, window=14)).
3. Modify run_strategy() to incorporate indicators in entry/exit logic (e.g., conditional checks).
4. Test with sample data; add unit tests for indicator accuracy.
5. Run linting (ruff check/format) and backtest to verify improved win rate.

**Post-Stage Check:** Run strategy; verify win rate/outperformance improves (e.g., +5% win rate). Ensure fees included (check Portfolio output).

### 2. Dynamic Thresholds (Adaptability Layer)
**Description:** Replace fixed FGI 35/80 with adaptive values based on recent volatility (e.g., ATR-adjusted) or rolling averages for better market fit.

**Steps:**
1. Implement threshold calculation (e.g., def dynamic_fgi_thresholds(close, fgi_df, lookback=30)).
2. Update run_strategy() to use dynamic thresholds instead of constants.
3. Add backtesting across different market conditions (bull/bear periods).
4. Validate with historical simulations; compare to fixed thresholds.
5. Document changes in code with docstrings.

**Post-Stage Check:** Run; compare to stage 1—expect better adaptability in volatile periods. Fees verified.

### 3. Enhanced Risk Management (Protection Layer)
**Description:** Upgrade stop-loss/take-profit to trailing stops, add position sizing (e.g., Kelly criterion), and max drawdown limits to reduce losses.

**Steps:**
1. Refactor run_strategy() for trailing stop logic (e.g., adjust stop-loss on price moves).
2. Implement position sizing function (e.g., def position_size(capital, risk_pct=0.02)).
3. Add portfolio-level checks (e.g., exit if drawdown >20%).
4. Backtest with vectorbt's portfolio analytics.
5. Ensure type hints and error handling per AGENTS.md.

**Post-Stage Check:** Run; measure reduced drawdown (+10% improvement). Fees confirmed.

### 4. Multiple Timeframe Analysis (Integration Layer)
**Description:** Filter signals by higher timeframes (e.g., use daily FGI for hourly trades) to align with broader trends.

**Steps:**
1. Modify data fetching to support multiple granularities simultaneously.
2. Add timeframe alignment logic (e.g., resample data to common index).
3. Update run_strategy() to check higher TF conditions (e.g., if daily FGI >50, skip hourly buy).
4. Test with cross-TF backtests.
5. Add logging for TF conflicts.

**Post-Stage Check:** Run; verify signal consistency (e.g., fewer false positives). Fees intact.

## Additional Ideas for Further Improvement

### 5. Machine Learning Integration (Prediction Layer)
**Description:** Use ML to predict FGI trends or entry/exit probabilities.
**Steps:** Collect data, train model (e.g., Random Forest), integrate predictions, backtest with walk-forward validation.
**Potential Impact:** Higher accuracy in volatile markets.
**Challenges:** Overfitting; requires scikit-learn.
**Timeline:** 1-2 weeks.

### 6. Portfolio Diversification (Expansion Layer)
**Description:** Expand to correlated assets (e.g., ETH).
**Steps:** Fetch multi-asset data, adapt strategy, implement correlation checks, backtest portfolio.
**Potential Impact:** Reduced volatility.
**Challenges:** Data handling.
**Timeline:** 1 week.

### 7. Parameter Optimization (Tuning Layer)
**Description:** Optimize hyperparameters (e.g., RSI window, trail %).
**Steps:** Define ranges, use grid search, validate out-of-sample, update code.
**Potential Impact:** Fine-tuned performance.
**Challenges:** Computational cost.
**Timeline:** 3-5 days.

### 8. Advanced Risk Management (Protection Layer)
**Description:** Add Kelly position sizing and drawdown limits.
**Steps:** Implement Kelly formula, calculate sizes, add portfolio stops, backtest.
**Potential Impact:** Better capital preservation.
**Challenges:** Dynamic sizing.
**Timeline:** 1 week.

### 9. Sentiment and On-Chain Integration (Data Layer)
**Description:** Incorporate sentiment or on-chain metrics.
**Steps:** Source APIs, add features, test correlation, backtest.
**Potential Impact:** Earlier signals.
**Challenges:** API costs.
**Timeline:** 1-2 weeks.

### 10. Live Trading and Monitoring Setup (Execution Layer)
**Description:** Set up live execution.
**Steps:** Integrate broker API, add execution logic, run paper/live trading.
**Potential Impact:** Real-world profits.
**Challenges:** Slippage.
**Timeline:** 2 weeks.

## Overall Timeline
7-10 days, iterative (implement/test one idea before next). Total Effort: ~20-30 hours.

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
- No new dependencies to avoid bloat.</content>
<parameter name="filePath">/home/stuart/repos/technical-with-fear-and-greed/IMPLEMENTATION_PLAN.md