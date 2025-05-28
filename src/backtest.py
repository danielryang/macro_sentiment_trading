"""
Expanding-window out-of-sample backtest and P&L simulation.

- 5-fold expanding window (train starts Jan 2015)
- Realistic transaction costs (0.02% FX, 0.05% bonds)
- Outputs Sharpe, CAGR, max drawdown
- Input: Model predictions, market data
- Output: Backtest results, performance metrics
- Reference: arXiv:2505.16136v1
"""
