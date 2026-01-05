# Performance Log

## Baseline (Current Strategy)
Run: `python trading_strategy.py`  
Date: [Current Date]  

### Metrics:
- **ONE_DAY**: Return 1.95%, Benchmark -4.04%, Outper 5.99%, Win Rate 50%, Trades 3  
- **FOUR_HOUR**: -14.20%, -3.22%, -10.98%, Win Rate 33.3%, Trades 4  
- **ONE_HOUR**: -13.42%, -3.88%, -9.54%, Win Rate 33.3%, Trades 4  
- **Best Return**: ONE_DAY with 1.95%  
- **Best Sharpe**: ONE_DAY with 0.23  

Fees verified in Portfolio.from_signals: MAKER_FEE=0.0015, TAKER_FEE=0.0025

## Stage 1: Added RSI Filter
Changes: Added RSI calculation, buy only if FGI <=35 AND RSI <30, sell if FGI >80 OR RSI >70 OR pnl conditions.  
Run: `python trading_strategy.py`  
Date: [Current Date]  

### Metrics:
- **ONE_DAY**: Return 7.62%, Benchmark -4.04%, Outper 11.67%, Win Rate 66.7%, Trades 4  
- **FOUR_HOUR**: -9.50%, -3.22%, -6.28%, Win Rate 53.3%, Trades 15  
- **ONE_HOUR**: -5.26%, -3.88%, -1.38%, Win Rate 58.8%, Trades 40  
- **Best Return**: ONE_DAY with 7.62%  
- **Best Sharpe**: ONE_DAY with 0.39  

### Comparison to Baseline:
- ONE_DAY: Improved return (+5.67%), win rate (+16.7%), outperformance (+5.68%)  
- FOUR_HOUR: Improved return (+4.7%), win rate (+20%), but more trades  
- ONE_HOUR: Improved return (+8.16%), win rate (+25.5%), outperformance (+8.16%)  
- Overall: Significant improvement in win rates and returns, especially on higher timeframes. Fees included.

## Stage 2: Dynamic Thresholds (Attempted, Reverted)
Changes: Attempted dynamic FGI thresholds based on rolling mean/std (30 days).  
Run: `python trading_strategy.py`  
Date: [Current Date]  

### Metrics:
- **ONE_DAY**: Return -6.66%, Benchmark -4.04%, Outper -2.62%, Win Rate 72.2%, Trades 9  
- **FOUR_HOUR**: -19.66%, -3.22%, -16.44%, Win Rate 54.8%, Trades 21  
- **ONE_HOUR**: -15.49%, -3.88%, -11.61%, Win Rate 57.0%, Trades 64  

### Comparison to Stage 1:
- Worse returns across all timeframes; more trades but lower returns.  
- Reverted to fixed thresholds (35/80) for better performance. Dynamic thresholds may need tuning or different approach.

## Stage 3: Enhanced Risk Management (Attempted, Reverted)
Changes: Added trailing stop loss (15% from entry, trailing up).  
Run: `python trading_strategy.py`  
Date: [Current Date]  

### Metrics:
- **ONE_DAY**: Return 3.10%, Benchmark -4.04%, Outper 7.14%, Win Rate 50.0%, Trades 5  
- **FOUR_HOUR**: -9.50%, -3.22%, -6.28%, Win Rate 53.3%, Trades 15  
- **ONE_HOUR**: -5.26%, -3.88%, -1.38%, Win Rate 58.8%, Trades 40  

### Comparison to Stage 1:
- ONE_DAY: Worse return (7.62% to 3.10%), win rate (66.7% to 50.0%). Trailing stop caused premature exits.  
- Reverted to fixed stop loss. Trailing stops may need wider bands or different implementation.

## Final Strategy (Stage 1)
The strategy with RSI filter shows the best improvement over baseline. Further enhancements (dynamic thresholds, trailing stops) worsened performance in initial tests. Recommend sticking with RSI filter and exploring stage 4 (multi-timeframe) or parameter optimization next.</content>
<parameter name="filePath">/home/stuart/repos/technical-with-fear-and-greed/performance_log.md