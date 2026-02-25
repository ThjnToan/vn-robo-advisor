import pytest
import pandas as pd
import numpy as np
from backend.core.analytics import compute_equity_curve, run_monte_carlo_var
from backend.core.portfolio import optimize_portfolio, generate_trades, execute_trades, init_portfolio, load_portfolio

def test_compute_equity_curve_simple():
    dates = pd.date_range('2023-01-01', periods=3)
    prices = pd.DataFrame({
        'AAA': [10, 11, 12],
        'BBB': [20, 19, 18]
    }, index=dates)
    ledger = pd.DataFrame([
        {'Date': '2023-01-01', 'Ticker': 'AAA', 'Action': 'BUY', 'Shares': 10, 'Value': 100, 'Fee': 0},
        {'Date': '2023-01-02', 'Ticker': 'BBB', 'Action': 'BUY', 'Shares': 5, 'Value': 95, 'Fee': 0},
        {'Date': '2023-01-03', 'Ticker': 'AAA', 'Action': 'SELL', 'Shares': 5, 'Value': 60, 'Fee': 0},
    ])
    result = compute_equity_curve(prices, ledger, initial_capital=1000)
    assert result is not None
    assert len(result['portfolio_nav']) == 3

def test_run_monte_carlo_var():
    dates = pd.date_range('2023-01-01', periods=5)
    prices = pd.DataFrame({
        'AAA': np.linspace(10, 15, 5),
        'BBB': np.linspace(20, 25, 5)
    }, index=dates)
    holdings = {'AAA': 10, 'BBB': 5}
    cash = 500
    var = run_monte_carlo_var(prices, holdings, cash, num_simulations=100, sim_days=2)
    assert var is not None
    assert 'var_95_loss' in var

def test_optimize_and_trades():
    dates = pd.date_range('2023-01-01', periods=3)
    prices = pd.DataFrame({
        'AAA': [10, 11, 12],
        'BBB': [20, 19, 18]
    }, index=dates)
    target = ['AAA', 'BBB']
    weights = optimize_portfolio(prices, target, max_weight=0.6)
    assert isinstance(weights, dict)
    state = {'cash_balance': 1000}
    ledger = pd.DataFrame(columns=['Date','Ticker','Action','Shares','Price','Value','Fee'])
    trades = generate_trades(prices, state, ledger, weights)
    assert isinstance(trades, list)
    new_state, new_ledger = execute_trades(trades, state, ledger, 'testsession')
    assert isinstance(new_state, dict)
    assert isinstance(new_ledger, pd.DataFrame)
