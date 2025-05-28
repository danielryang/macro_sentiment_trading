use pyo3::prelude::*;

/// Efficient trade simulation: computes strategy returns and cumulative returns
/// given signals, returns, and transaction cost.
#[pyfunction]
pub fn simulate_trades(signals: Vec<f64>, returns: Vec<f64>, transaction_cost: f64) -> (Vec<f64>, Vec<f64>) {
    let mut strategy_returns = Vec::with_capacity(signals.len());
    let mut cumulative_returns = Vec::with_capacity(signals.len());
    let mut prev_signal = 0.0;
    let mut cumprod = 1.0;
    for i in 0..signals.len() {
        let position_change = (signals[i] - prev_signal).abs();
        let strat_return = if i == 0 {
            0.0
        } else {
            signals[i-1] * returns[i] - position_change * transaction_cost
        };
        cumprod *= 1.0 + strat_return;
        strategy_returns.push(strat_return);
        cumulative_returns.push(cumprod);
        prev_signal = signals[i];
    }
    (strategy_returns, cumulative_returns)
}

/// Python module definition
#[pymodule]
fn rust_trade_sim(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_trades, m)?)?;
    Ok(())
}

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
