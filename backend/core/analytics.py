import pandas as pd
import numpy as np
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

RISK_FREE_ANNUAL = 0.045  # 4.5% Vietnamese T-bill equivalent

STRESS_SCENARIOS = {
    "COVID Crash (Feb–Mar 2020)": ("2020-02-01", "2020-03-31"),
    "Global Rate Hike Shock (2022)": ("2022-01-01", "2022-12-31"),
    "Evergrande / VN Correction (Q4 2021)": ("2021-07-01", "2021-12-31"),
}

def compute_equity_curve(prices_df: pd.DataFrame, ledger: pd.DataFrame, initial_capital: float):
    """
    Replays the transaction ledger chronologically against market data.
    Returns:
      - portfolio_nav: list of {date, nav} dicts 
      - benchmark_nav: list of {date, nav} dicts (VN30 ETF, equal-start)
      - rolling_sharpe_portfolio: list of {date, sharpe} dicts
      - rolling_sharpe_benchmark: list of {date, sharpe} dicts
      - stress_results: list of stress test scenario dicts
      - summary_stats: dict of key performance metrics
    """
    if ledger.empty:
        return None

    ledger = ledger.copy()
    ledger["Date"] = pd.to_datetime(ledger["Date"])
    ledger = ledger.sort_values("Date")

    first_trade_date = ledger["Date"].min()
    latest_date = prices_df.index.max()
    timeline = prices_df.loc[first_trade_date:latest_date].index

    if len(timeline) < 2:
        return None

    # ── Step 1: Replay ledger day-by-day ───────────────────────────────────
    current_shares: dict = {}
    current_cash = initial_capital
    portfolio_history = []

    for date in timeline:
        day_trades = ledger[ledger["Date"] == date]
        for _, trade in day_trades.iterrows():
            ticker = trade["Ticker"]
            shares = trade["Shares"]
            total = trade["Value"] + trade["Fee"] if trade["Action"] == "BUY" else trade["Value"] - trade["Fee"]
            if trade["Action"] == "BUY":
                current_shares[ticker] = current_shares.get(ticker, 0) + shares
                current_cash -= total
            else:
                current_shares[ticker] = current_shares.get(ticker, 0) - shares
                current_cash += total

        eod_nav = current_cash
        for t, s in current_shares.items():
            if s > 0 and t in prices_df.columns:
                eod_nav += s * prices_df.loc[date, t]

        portfolio_history.append({"date": date.strftime("%Y-%m-%d"), "nav": round(eod_nav, 0)})

    history_df = pd.DataFrame(portfolio_history).set_index("date")

    # ── Step 2: Benchmark (VN30 ETF ─ E1VFVN30) ───────────────────────────
    benchmark_history = []
    if "E1VFVN30" in prices_df.columns:
        bench_prices = prices_df.loc[timeline, "E1VFVN30"].dropna()
        if not bench_prices.empty:
            bench_shares = initial_capital / bench_prices.iloc[0]
            for date in timeline:
                if date.strftime("%Y-%m-%d") in [b["date"] for b in benchmark_history]:
                    continue
                nav = bench_shares * prices_df.loc[date, "E1VFVN30"] if date in prices_df.index else None
                if nav is not None:
                    benchmark_history.append({"date": date.strftime("%Y-%m-%d"), "nav": round(float(nav), 0)})

    # ── Step 3: Rolling 60-Day Sharpe ──────────────────────────────────────
    nav_series = pd.Series(
        [r["nav"] for r in portfolio_history],
        index=pd.to_datetime([r["date"] for r in portfolio_history])
    )
    port_daily_ret = nav_series.pct_change().dropna()

    rolling_port_sharpe = []
    if len(port_daily_ret) >= 60:
        roll_ret = port_daily_ret.rolling(60).mean() * 252
        roll_std = port_daily_ret.rolling(60).std() * (252 ** 0.5)
        roll_sharpe = (roll_ret - RISK_FREE_ANNUAL) / roll_std.replace(0, float("nan"))
        for dt, v in roll_sharpe.dropna().items():
            rolling_port_sharpe.append({"date": dt.strftime("%Y-%m-%d"), "sharpe": round(float(v), 3)})

    rolling_bench_sharpe = []
    if benchmark_history and len(benchmark_history) >= 60:
        bench_series = pd.Series(
            [r["nav"] for r in benchmark_history],
            index=pd.to_datetime([r["date"] for r in benchmark_history])
        )
        bench_daily_ret = bench_series.pct_change().dropna()
        b_roll_ret = bench_daily_ret.rolling(60).mean() * 252
        b_roll_std = bench_daily_ret.rolling(60).std() * (252 ** 0.5)
        b_roll_sharpe = (b_roll_ret - RISK_FREE_ANNUAL) / b_roll_std.replace(0, float("nan"))
        for dt, v in b_roll_sharpe.dropna().items():
            rolling_bench_sharpe.append({"date": dt.strftime("%Y-%m-%d"), "sharpe": round(float(v), 3)})

    # ── Step 4: Historical Stress Test ─────────────────────────────────────
    held_tickers = [t for t in current_shares if current_shares[t] > 0 and t in prices_df.columns]
    latest_nav = history_df["nav"].iloc[-1] if not history_df.empty else initial_capital
    stress_results = []

    if held_tickers:
        total_stock_val = sum(current_shares[t] * prices_df.loc[latest_date, t] for t in held_tickers)
        if total_stock_val > 0:
            weights = {t: current_shares[t] * prices_df.loc[latest_date, t] / total_stock_val for t in held_tickers}
            for name, (s_str, e_str) in STRESS_SCENARIOS.items():
                try:
                    s_dt, e_dt = pd.to_datetime(s_str), pd.to_datetime(e_str)
                    period = prices_df[held_tickers].loc[s_dt:e_dt]
                    if len(period) < 2:
                        continue
                    period_rets = (period.iloc[-1] / period.iloc[0]) - 1
                    port_ret = sum(weights[t] * period_rets.get(t, 0) for t in held_tickers)
                    pnl = port_ret * latest_nav
                    stress_results.append({
                        "scenario": name,
                        "period": f"{s_str} → {e_str}",
                        "return_pct": round(port_ret * 100, 1),
                        "pnl": round(pnl, 0),
                        "severity": "Severe" if port_ret < -0.20 else ("Moderate" if port_ret < -0.10 else "Mild"),
                    })
                except Exception:
                    continue

    # ── Step 5: Summary Stats ──────────────────────────────────────────────
    final_nav = float(history_df["nav"].iloc[-1]) if not history_df.empty else initial_capital
    total_return = (final_nav / initial_capital - 1) * 100
    ann_return = float(port_daily_ret.mean() * 252 * 100) if not port_daily_ret.empty else 0
    ann_vol = float(port_daily_ret.std() * (252 ** 0.5) * 100) if not port_daily_ret.empty else 0
    sharpe = (ann_return / 100 - RISK_FREE_ANNUAL) / (ann_vol / 100) if ann_vol > 0 else 0
    max_dd = 0.0
    if not nav_series.empty:
        roll_max = nav_series.cummax()
        dd = (nav_series - roll_max) / roll_max
        max_dd = float(dd.min() * 100)

    return {
        "portfolio_nav": portfolio_history,
        "benchmark_nav": benchmark_history,
        "rolling_sharpe_portfolio": rolling_port_sharpe,
        "rolling_sharpe_benchmark": rolling_bench_sharpe,
        "stress_results": stress_results,
        "summary_stats": {
            "total_return_pct": round(total_return, 2),
            "annualized_return_pct": round(ann_return, 2),
            "annualized_vol_pct": round(ann_vol, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "data_points": len(portfolio_history),
        }
    }

def run_monte_carlo_var(prices_df, holdings_dict, current_cash, num_simulations=10000, sim_days=30):
    """ Runs correlated random-walk price paths using Cholesky Decomposition of historical covariance. """
    tickers = [t for t in holdings_dict.keys() if t in prices_df.columns]
    if not tickers:
        return None
        
    latest_date = prices_df.index.max()
    initial_values = np.array([holdings_dict[t] * prices_df.loc[latest_date, t] for t in tickers])
    current_nav = np.sum(initial_values) + current_cash
    
    historical_returns = prices_df[tickers].tail(252).pct_change().dropna()
    mu = historical_returns.mean().values
    cov = historical_returns.cov().values
    
    np.random.seed(42) 
    try:
        daily_sim_returns = np.random.multivariate_normal(mu, cov, (num_simulations, sim_days))
    except Exception:
        return None 
        
    cumulative_returns = np.cumprod(1 + daily_sim_returns, axis=1)
    final_returns = cumulative_returns[:, -1, :] 
    
    final_portfolio_values = np.sum(initial_values * final_returns, axis=1) + current_cash
    
    var_95_value = np.percentile(final_portfolio_values, 5)
    cvar_95_value = final_portfolio_values[final_portfolio_values <= var_95_value].mean()
    
    var_95_loss = current_nav - var_95_value
    cvar_95_loss = current_nav - cvar_95_value
    
    return {
        "final_values": final_portfolio_values.tolist(),
        "var_95_loss": float(var_95_loss),
        "cvar_95_loss": float(cvar_95_loss),
        "var_95_threshold": float(var_95_value),
        "current_nav": float(current_nav)
    }

def compute_efficient_frontier(prices_df, tickers_tuple, latest_date_str, n_portfolios=3000):
    tickers = list(tickers_tuple)
    latest_date = pd.to_datetime(latest_date_str)
    
    prices = prices_df[tickers].loc[:latest_date].tail(252)
    returns = prices.pct_change().dropna()
    
    if returns.shape[0] < 30 or returns.shape[1] < 2:
        return None
    
    mu_daily = returns.mean()
    cov_daily = returns.cov()
    ann_factor = 252
    
    np.random.seed(99)
    rand_vols, rand_rets, rand_sharpes = [], [], []
    
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(len(tickers)))
        p_ret = float(np.dot(w, mu_daily) * ann_factor)
        p_vol = float(np.sqrt(np.dot(w.T, np.dot(cov_daily.values * ann_factor, w))))
        p_sharpe = p_ret / p_vol if p_vol > 0 else 0
        rand_rets.append(p_ret)
        rand_vols.append(p_vol)
        rand_sharpes.append(p_sharpe)
    
    frontier_vols, frontier_rets = [], []
    try:
        mu_pf = expected_returns.mean_historical_return(prices)
        S_pf = risk_models.sample_cov(prices)
        target_rets = np.linspace(min(rand_rets) * 1.05, max(rand_rets) * 0.95, 40)
        for r in target_rets:
            try:
                ef = EfficientFrontier(mu_pf, S_pf, weight_bounds=(0, 0.40))
                ef.efficient_return(r)
                pw = ef.clean_weights()
                w_arr = np.array([pw.get(t, 0) for t in tickers])
                p_vol = float(np.sqrt(np.dot(w_arr.T, np.dot(cov_daily.values * ann_factor, w_arr))))
                frontier_vols.append(p_vol)
                frontier_rets.append(float(r))
            except Exception:
                continue
    except Exception:
        pass
    
    return {
        'rand_vols': rand_vols, 'rand_rets': rand_rets, 'rand_sharpes': rand_sharpes,
        'frontier_vols': frontier_vols, 'frontier_rets': frontier_rets
    }
