#!/usr/bin/env python3

"""
Comprehensive Performance Metrics Module

Provides institutional-grade performance metrics for quantitative trading strategies.
Includes traditional metrics plus advanced risk-adjusted and behavioral finance measures.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for trading strategies.
    
    Provides traditional metrics, risk-adjusted returns, drawdown analysis,
    and behavioral finance measures commonly used in institutional settings.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
    
    def compute_comprehensive_metrics(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Compute comprehensive performance metrics.
        
        Args:
            returns: Series of strategy returns
            benchmark_returns: Optional benchmark returns for comparison
            
        Returns:
            Dictionary of comprehensive performance metrics
        """
        try:
            metrics = {}
            
            # Basic return metrics
            metrics.update(self._compute_return_metrics(returns))
            
            # Risk metrics
            metrics.update(self._compute_risk_metrics(returns))
            
            # Drawdown metrics
            metrics.update(self._compute_drawdown_metrics(returns))
            
            # Trade-based metrics
            metrics.update(self._compute_trade_metrics(returns))
            
            # Advanced risk-adjusted metrics
            metrics.update(self._compute_advanced_risk_metrics(returns))
            
            # Time-based metrics
            metrics.update(self._compute_time_metrics(returns))
            
            # Benchmark comparison (if provided)
            if benchmark_returns is not None:
                metrics.update(self._compute_benchmark_metrics(returns, benchmark_returns))
            
            # Statistical metrics
            metrics.update(self._compute_statistical_metrics(returns))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error computing comprehensive metrics: {e}")
            return {}
    
    def _compute_return_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Compute basic return metrics."""
        if len(returns) == 0:
            return {}
        
        cumulative_return = (1 + returns).prod() - 1
        total_days = len(returns)
        
        # Annualized return (assuming daily data)
        if total_days > 0:
            annualized_return = (1 + cumulative_return) ** (252 / total_days) - 1
        else:
            annualized_return = 0
        
        # Geometric mean return
        geometric_mean = (1 + returns).prod() ** (1 / len(returns)) - 1 if len(returns) > 0 else 0
        
        return {
            'total_return': cumulative_return,
            'annualized_return': annualized_return,
            'geometric_mean_return': geometric_mean,
            'arithmetic_mean_return': returns.mean(),
            'median_return': returns.median(),
            'return_skewness': returns.skew(),
            'return_kurtosis': returns.kurtosis()
        }
    
    def _compute_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Compute risk metrics."""
        if len(returns) == 0:
            return {}
        
        # Volatility metrics
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252)
        
        # Risk-adjusted returns
        excess_returns = returns.mean() - (self.risk_free_rate / 252)
        sharpe_ratio = (excess_returns / daily_vol) * np.sqrt(252) if daily_vol > 0 else 0
        
        # Downside deviation (using 0 as minimum acceptable return)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Sortino ratio
        sortino_ratio = (excess_returns / (downside_deviation / np.sqrt(252))) if downside_deviation > 0 else 0
        
        # Value at Risk (VaR) - 5% and 1% levels
        var_5 = returns.quantile(0.05)
        var_1 = returns.quantile(0.01)
        
        # Conditional VaR (Expected Shortfall)
        cvar_5 = returns[returns <= var_5].mean() if len(returns[returns <= var_5]) > 0 else var_5
        cvar_1 = returns[returns <= var_1].mean() if len(returns[returns <= var_1]) > 0 else var_1
        
        return {
            'volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'downside_deviation': downside_deviation,
            'var_5_percent': var_5,
            'var_1_percent': var_1,
            'cvar_5_percent': cvar_5,
            'cvar_1_percent': cvar_1,
            'daily_volatility': daily_vol
        }
    
    def _compute_drawdown_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Compute comprehensive drawdown metrics."""
        if len(returns) == 0:
            return {}
        
        # Cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Running maximum
        running_max = cumulative_returns.expanding().max()
        
        # Drawdowns
        drawdowns = (cumulative_returns / running_max) - 1
        
        # Maximum drawdown
        max_drawdown = drawdowns.min()
        
        # Average drawdown
        avg_drawdown = drawdowns[drawdowns < 0].mean() if len(drawdowns[drawdowns < 0]) > 0 else 0
        
        # Drawdown duration analysis
        in_drawdown = drawdowns < 0
        drawdown_periods = self._get_drawdown_periods(in_drawdown)
        
        max_drawdown_duration = max([len(period) for period in drawdown_periods]) if drawdown_periods else 0
        avg_drawdown_duration = np.mean([len(period) for period in drawdown_periods]) if drawdown_periods else 0
        
        # Recovery analysis
        recovery_times = self._calculate_recovery_times(drawdowns)
        avg_recovery_time = np.mean(recovery_times) if recovery_times else 0
        max_recovery_time = max(recovery_times) if recovery_times else 0
        
        # Calmar ratio
        annualized_return = self._compute_return_metrics(returns).get('annualized_return', 0)
        calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else float('inf')
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_drawdown_duration': avg_drawdown_duration,
            'avg_recovery_time': avg_recovery_time,
            'max_recovery_time': max_recovery_time,
            'calmar_ratio': calmar_ratio,
            'number_of_drawdown_periods': len(drawdown_periods)
        }
    
    def _compute_trade_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Compute trade-based metrics."""
        if len(returns) == 0:
            return {}
        
        # Trade statistics
        total_trades = len(returns)
        winning_trades = (returns > 0).sum()
        losing_trades = (returns < 0).sum()
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        loss_rate = losing_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average win/loss
        avg_win = returns[returns > 0].mean() if winning_trades > 0 else 0
        avg_loss = returns[returns < 0].mean() if losing_trades > 0 else 0
        
        # Win/Loss ratio
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Largest win/loss
        largest_win = returns.max()
        largest_loss = returns.min()
        
        # Consecutive wins/losses
        consecutive_wins = self._max_consecutive(returns > 0)
        consecutive_losses = self._max_consecutive(returns < 0)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses
        }
    
    def _compute_advanced_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Compute advanced risk-adjusted metrics."""
        if len(returns) == 0:
            return {}
        
        # Information Ratio (excess return per unit of tracking error)
        # Using 0 as benchmark for simplicity
        excess_returns = returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        # Treynor Ratio (assuming beta = 1 for simplicity)
        treynor_ratio = (returns.mean() * 252 - self.risk_free_rate) / 1.0
        
        # Maximum Adverse Excursion and Maximum Favorable Excursion
        mae = returns.min()  # Maximum loss in a single period
        mfe = returns.max()  # Maximum gain in a single period
        
        # Gain-to-Pain ratio
        gains = returns[returns > 0].sum()
        pains = abs(returns[returns < 0].sum())
        gain_to_pain = gains / pains if pains > 0 else float('inf')
        
        # Sterling Ratio (annualized return / average drawdown)
        avg_drawdown = self._compute_drawdown_metrics(returns).get('avg_drawdown', 0)
        sterling_ratio = (returns.mean() * 252) / abs(avg_drawdown) if avg_drawdown != 0 else float('inf')
        
        return {
            'information_ratio': information_ratio,
            'treynor_ratio': treynor_ratio,
            'mae': mae,
            'mfe': mfe,
            'gain_to_pain_ratio': gain_to_pain,
            'sterling_ratio': sterling_ratio
        }
    
    def _compute_time_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Compute time-based performance metrics."""
        if len(returns) == 0:
            return {}
        
        # Time in market (assuming non-zero returns indicate active positions)
        time_in_market = (returns != 0).mean()
        
        # Active trading days
        active_days = (returns != 0).sum()
        
        # Average time between trades (in days)
        non_zero_indices = returns[returns != 0].index
        if len(non_zero_indices) > 1:
            time_diffs = np.diff(non_zero_indices)
            avg_time_between_trades = np.mean(time_diffs) if len(time_diffs) > 0 else 0
        else:
            avg_time_between_trades = 0
        
        return {
            'time_in_market': time_in_market,
            'active_trading_days': active_days,
            'avg_days_between_trades': avg_time_between_trades
        }
    
    def _compute_benchmark_metrics(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        """Compute benchmark comparison metrics."""
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return {}
        
        # Align returns
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) == 0:
            return {}
        
        # Excess returns
        excess_returns = aligned_returns - aligned_benchmark
        
        # Tracking error
        tracking_error = excess_returns.std() * np.sqrt(252)
        
        # Information ratio
        information_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        # Beta
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = aligned_benchmark.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # Alpha
        benchmark_return = aligned_benchmark.mean() * 252
        strategy_return = aligned_returns.mean() * 252
        alpha = strategy_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
        
        # Correlation
        correlation = aligned_returns.corr(aligned_benchmark)
        
        return {
            'tracking_error': tracking_error,
            'information_ratio_vs_benchmark': information_ratio,
            'beta': beta,
            'alpha': alpha,
            'correlation_with_benchmark': correlation,
            'excess_return': excess_returns.mean() * 252
        }
    
    def _compute_statistical_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Compute statistical distribution metrics."""
        if len(returns) == 0:
            return {}
        
        # Higher moments
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Tail ratios
        left_tail = returns.quantile(0.05)
        right_tail = returns.quantile(0.95)
        tail_ratio = abs(right_tail / left_tail) if left_tail != 0 else float('inf')
        
        # Jarque-Bera test for normality (p-value)
        try:
            from scipy import stats
            _, jb_pvalue = stats.jarque_bera(returns.dropna())
        except ImportError:
            jb_pvalue = None
        
        return {
            'skewness': skewness,
            'excess_kurtosis': kurtosis,
            'tail_ratio': tail_ratio,
            'jarque_bera_pvalue': jb_pvalue
        }
    
    def _get_drawdown_periods(self, in_drawdown: pd.Series) -> List[List[int]]:
        """Get list of drawdown periods."""
        periods = []
        current_period = []
        
        for i, is_dd in enumerate(in_drawdown):
            if is_dd:
                current_period.append(i)
            else:
                if current_period:
                    periods.append(current_period)
                    current_period = []
        
        # Add final period if it ends in drawdown
        if current_period:
            periods.append(current_period)
        
        return periods
    
    def _calculate_recovery_times(self, drawdowns: pd.Series) -> List[int]:
        """Calculate recovery times from drawdowns."""
        recovery_times = []
        in_drawdown = False
        drawdown_start = None
        
        for i, dd in enumerate(drawdowns):
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                drawdown_start = i
            elif dd >= 0 and in_drawdown:
                # Recovery
                recovery_time = i - drawdown_start
                recovery_times.append(recovery_time)
                in_drawdown = False
                drawdown_start = None
        
        return recovery_times
    
    def _max_consecutive(self, condition: pd.Series) -> int:
        """Calculate maximum consecutive True values."""
        if len(condition) == 0:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for value in condition:
            if value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def generate_performance_report(self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None) -> str:
        """Generate a comprehensive performance report."""
        metrics = self.compute_comprehensive_metrics(returns, benchmark_returns)
        
        report = ["=" * 80]
        report.append("COMPREHENSIVE PERFORMANCE ANALYSIS")
        report.append("=" * 80)
        
        # Return Metrics
        report.append("\nRETURN METRICS:")
        report.append("-" * 40)
        report.append(f"Total Return:           {metrics.get('total_return', 0):8.2%}")
        report.append(f"Annualized Return:      {metrics.get('annualized_return', 0):8.2%}")
        report.append(f"Geometric Mean Return:  {metrics.get('geometric_mean_return', 0):8.2%}")
        report.append(f"Arithmetic Mean Return: {metrics.get('arithmetic_mean_return', 0):8.2%}")
        
        # Risk Metrics
        report.append("\nRISK METRICS:")
        report.append("-" * 40)
        report.append(f"Volatility:             {metrics.get('volatility', 0):8.2%}")
        report.append(f"Sharpe Ratio:           {metrics.get('sharpe_ratio', 0):8.2f}")
        report.append(f"Sortino Ratio:          {metrics.get('sortino_ratio', 0):8.2f}")
        report.append(f"Calmar Ratio:           {metrics.get('calmar_ratio', 0):8.2f}")
        
        # Drawdown Metrics
        report.append("\nDRAWDOWN METRICS:")
        report.append("-" * 40)
        report.append(f"Maximum Drawdown:       {metrics.get('max_drawdown', 0):8.2%}")
        report.append(f"Average Drawdown:       {metrics.get('avg_drawdown', 0):8.2%}")
        report.append(f"Max DD Duration:        {metrics.get('max_drawdown_duration', 0):8.0f} days")
        report.append(f"Avg Recovery Time:      {metrics.get('avg_recovery_time', 0):8.1f} days")
        
        # Trade Metrics
        report.append("\nTRADE METRICS:")
        report.append("-" * 40)
        report.append(f"Total Trades:           {metrics.get('total_trades', 0):8.0f}")
        report.append(f"Win Rate:               {metrics.get('win_rate', 0):8.2%}")
        report.append(f"Profit Factor:          {metrics.get('profit_factor', 0):8.2f}")
        report.append(f"Win/Loss Ratio:         {metrics.get('win_loss_ratio', 0):8.2f}")
        
        # Risk-Adjusted Metrics
        report.append("\nADVANCED RISK METRICS:")
        report.append("-" * 40)
        report.append(f"VaR (5%):               {metrics.get('var_5_percent', 0):8.2%}")
        report.append(f"CVaR (5%):              {metrics.get('cvar_5_percent', 0):8.2%}")
        report.append(f"Gain-to-Pain Ratio:     {metrics.get('gain_to_pain_ratio', 0):8.2f}")
        report.append(f"Sterling Ratio:         {metrics.get('sterling_ratio', 0):8.2f}")
        
        if benchmark_returns is not None:
            report.append("\nBENCHMARK COMPARISON:")
            report.append("-" * 40)
            report.append(f"Alpha:                  {metrics.get('alpha', 0):8.2%}")
            report.append(f"Beta:                   {metrics.get('beta', 0):8.2f}")
            report.append(f"Information Ratio:      {metrics.get('information_ratio_vs_benchmark', 0):8.2f}")
            report.append(f"Tracking Error:         {metrics.get('tracking_error', 0):8.2%}")
        
        report.append("=" * 80)
        
        return "\n".join(report)

# Alias for backward compatibility
PerformanceMetrics = PerformanceAnalyzer