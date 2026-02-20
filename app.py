import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from vnstock import Vnstock

# --- Page Configuration ---
st.set_page_config(page_title="VN Robo-Advisor", page_icon="📈", layout="wide")

# --- Default Constants ---
DEFAULT_TICKERS = [
    "FPT", "HPG", "VCB", "VHM", "MWG", "REE", "MBB", "VNM", "TCB", "SSI",
    "VIC", "VRE", "MSN", "GAS", "SAB", "CTG", "BID", "VJC", "PNJ", "VPB"
]
LOOKBACK_DAYS = 252 # 1 year trading days
REBALANCE_FREQ = 21 # 1 month trading days

import os

@st.cache_data(ttl=timedelta(hours=6))
def fetch_data(tickers, start_date, end_date):
    combined_data = pd.DataFrame()
    for ticker in tickers:
        df = None
        # Streamlit cloud IPs often get blocked, try multiple sources
        for source in ['VCI', 'TCBS', 'SSI', 'VND']:
            try:
                stock = Vnstock().stock(symbol=ticker, source=source)
                df = stock.quote.history(start=start_date, end=end_date)
                if df is not None and not df.empty:
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                    df.sort_index(inplace=True)
                    combined_data[ticker] = df['close']
                    break # Success!
            except Exception as e:
                continue # Try next source
                
        # API Failed, load from our bundled CSV
        if df is None or df.empty:
            local_path = os.path.join(os.path.dirname(__file__), "market_data.csv")
            if os.path.exists(local_path):
                csv_data = pd.read_csv(local_path, parse_dates=['time'], index_col='time')
                if ticker in csv_data.columns:
                    mask = (csv_data.index >= pd.to_datetime(start_date)) & (csv_data.index <= pd.to_datetime(end_date))
                    combined_data[ticker] = csv_data.loc[mask, ticker]
            else:
                st.sidebar.error(f"Failed to fetch: {ticker} and no local fallback found.")
            
    if not combined_data.empty:
        combined_data.ffill(inplace=True)
        combined_data.dropna(inplace=True) 
    return combined_data

@st.cache_data(ttl=timedelta(hours=6))
def fetch_benchmark(start_date, end_date):
    # Try API for the E1VFVN30 ETF as our VN-Index proxy
    for source in ['VCI', 'TCBS', 'SSI', 'VND']:
        try:
            stock = Vnstock().stock(symbol='E1VFVN30', source=source)
            df = stock.quote.history(start=start_date, end=end_date)
            if df is not None and not df.empty:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                df.sort_index(inplace=True)
                return df['close']
        except Exception as e:
            continue
            
    # Fallback to local CSV
    local_path = os.path.join(os.path.dirname(__file__), "market_data.csv")
    if os.path.exists(local_path):
        csv_data = pd.read_csv(local_path, parse_dates=['time'], index_col='time')
        if 'E1VFVN30' in csv_data.columns:
            mask = (csv_data.index >= pd.to_datetime(start_date)) & (csv_data.index <= pd.to_datetime(end_date))
            return csv_data.loc[mask, 'E1VFVN30']
            
    st.error("Failed to fetch E1VFVN30 benchmark from API and local storage.")
    return None

def optimize_portfolio(prices_df, max_weight):
    mu = expected_returns.mean_historical_return(prices_df, frequency=252)
    S = risk_models.sample_cov(prices_df, frequency=252)
    
    ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
    try:
        raw_weights = ef.max_sharpe()
    except Exception as e:
        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
        raw_weights = ef.min_volatility()
        
    return ef.clean_weights()

def calculate_drawdown(returns_series):
    cumulative = (1 + returns_series).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative / peak) - 1
    return drawdown.min()

def calculate_sharpe(returns_series, risk_free_rate=0.04):
    # Assume 4% risk free rate for VN DKK
    annualized_return = returns_series.mean() * 252
    annualized_volatility = returns_series.std() * np.sqrt(252)
    if annualized_volatility == 0:
        return 0
    return (annualized_return - risk_free_rate) / annualized_volatility

def simulate_performance(prices_df, benchmark_series, initial_capital, max_weight, tx_fee_rate=0.0):
    daily_returns = prices_df.pct_change().dropna()
    benchmark_returns = benchmark_series.pct_change().dropna()
    
    common_dates = daily_returns.index.intersection(benchmark_returns.index)
    daily_returns = daily_returns.loc[common_dates]
    benchmark_returns = benchmark_returns.loc[common_dates]
    
    if len(common_dates) <= LOOKBACK_DAYS:
        return None, None, None, None, None
        
    portfolio_daily_returns = pd.Series(index=common_dates, dtype=float)
    num_assets = len(prices_df.columns)
    current_weights = np.repeat(1.0 / num_assets, num_assets)
    final_optimal_weights = {}

    # Streamlit Progress Bar
    progress_bar = st.progress(0, text="Simulating Portfolio Adjustments...")
    total_steps = len(range(LOOKBACK_DAYS, len(common_dates)))

    for step, i in enumerate(range(LOOKBACK_DAYS, len(common_dates))):
        fee_penalty = 0.0
        
        # Monthly Rebalance
        if (i - LOOKBACK_DAYS) % REBALANCE_FREQ == 0:
            start_idx = i - LOOKBACK_DAYS
            train_prices = prices_df.loc[common_dates[start_idx:i]]
            try:
                opt_weights_dict = optimize_portfolio(train_prices, max_weight)
                new_weights = np.array([opt_weights_dict.get(ticker, 0) for ticker in daily_returns.columns])
                
                # Calculate turnover to apply transaction fees (only applies on rebalance days)
                if tx_fee_rate > 0:
                    turnover = np.sum(np.abs(new_weights - current_weights))
                    # Assuming fee is charged on both buying and selling the difference
                    fee_penalty = turnover * tx_fee_rate
                    
                current_weights = new_weights
                final_optimal_weights = opt_weights_dict
            except Exception as e:
                pass 
                
        # Gross return today
        gross_return = np.dot(current_weights, daily_returns.iloc[i])
        
        # Net return (deducting fees on rebalance day)
        net_return = gross_return - fee_penalty
        portfolio_daily_returns.iloc[i] = net_return
        
        # Update progress bar every 5%
        if step % max(1, int(total_steps / 20)) == 0:
             progress_bar.progress(step / total_steps, text=f"Simulating: {common_dates[i].strftime('%Y-%m')}")
             
    progress_bar.empty() # Clear bar when done

    portfolio_daily_returns = portfolio_daily_returns.dropna()
    opt_cumulative_returns = (1 + portfolio_daily_returns).cumprod()
    opt_portfolio_value = initial_capital * opt_cumulative_returns
    
    bench_returns_matched = benchmark_returns.loc[portfolio_daily_returns.index]
    bench_cumulative_returns = (1 + bench_returns_matched).cumprod()
    bench_portfolio_value = initial_capital * bench_cumulative_returns

    # Calculate advanced metrics
    metrics = {
        'port_mdd': calculate_drawdown(portfolio_daily_returns),
        'bench_mdd': calculate_drawdown(bench_returns_matched),
        'port_sharpe': calculate_sharpe(portfolio_daily_returns),
        'bench_sharpe': calculate_sharpe(bench_returns_matched)
    }

    return opt_portfolio_value, bench_portfolio_value, final_optimal_weights, portfolio_daily_returns, metrics

# --- UI Layout ---
st.title("🇻🇳 VN Robo-Advisor Dashboard")
st.markdown("A quantitative portfolio optimizer for the Vietnamese Stock Market using Markowitz Mean-Variance.")

# Sidebar Controls
with st.sidebar:
    st.header("Simulation Parameters")
    
    st.subheader("Capital & Limits")
    initial_cap = st.number_input("Initial Capital (VND)", min_value=10_000_000, value=1_000_000_000, step=100_000_000, format="%d")
    max_weight_pct = st.slider("Max Allocation per Stock", min_value=10, max_value=100, value=30, step=5, format="%d%%", help="To ensure diversification, force the algorithm to limit exposure to any single stock.")
    max_weight_dec = max_weight_pct / 100.0
    
    st.subheader("Realism")
    tx_fee_pct = st.number_input("Trading Fee (%)", min_value=0.0, max_value=2.0, value=0.15, step=0.05, help="Brokerage and tax fees applied during monthly rebalancing turnover. Typical VN broker fee is ~0.15%")
    tx_fee_dec = tx_fee_pct / 100.0

    st.subheader("Timeframe")
    default_start = datetime.today() - timedelta(days=4*365) # 4 years default
    start_date = st.date_input("Start Date", value=default_start)
    end_date = st.date_input("End Date", value=datetime.today())
    
    st.subheader("Universe")
    selected_tickers = st.multiselect("Select VN30 Stocks", options=DEFAULT_TICKERS, default=["FPT", "HPG", "VCB", "VHM", "MWG", "REE", "MBB", "VNM", "TCB"])
    
    run_sim = st.button("Run Simulation", type="primary", use_container_width=True)

# --- Main App Logic ---
if run_sim or not st.session_state.get('sim_run'):
    st.session_state['sim_run'] = True
    
    if len(selected_tickers) < 2:
        st.error("Please select at least two stocks for the portfolio.")
        st.stop()
        
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    with st.spinner("Fetching Market Data..."):
        prices_df = fetch_data(selected_tickers, start_str, end_str)
        benchmark_series = fetch_benchmark(start_str, end_str)

    if prices_df.empty or benchmark_series is None:
        st.error("Failed to fetch sufficient data from vnstock. Please try adjusting the dates or tickers.")
        st.stop()

    # Run the heavy math
    opt_val, bench_val, final_weights, ret_series, metrics = simulate_performance(prices_df, benchmark_series, initial_cap, max_weight_dec, tx_fee_dec)

    if opt_val is None:
        st.error(f"Not enough data to run simulation. Ensure your date range is greater than {LOOKBACK_DAYS} trading days (approx 1 year).")
        st.stop()
        
    # --- Rendering Metrics ---
    final_nav = opt_val.iloc[-1]
    total_return = (final_nav - initial_cap) / initial_cap
    
    final_bench_nav = bench_val.iloc[-1]
    bench_return = (final_bench_nav - initial_cap) / initial_cap
    
    st.subheader("Performance Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Portfolio Value", f"{final_nav:,.0f} ₫", f"{final_nav - initial_cap:,.0f} ₫")
    col2.metric("Strategy Return", f"{total_return * 100:.2f}%", f"{(total_return - bench_return) * 100:.2f}% vs ETF Benchmark", delta_color="normal")
    col3.metric("Max Drawdown (Risk)", f"{metrics['port_mdd'] * 100:.2f}%", f"{(metrics['port_mdd'] - metrics['bench_mdd']) * 100:.2f}% vs ETF Benchmark", delta_color="inverse")
    col4.metric("Sharpe Ratio", f"{metrics['port_sharpe']:.2f}", f"{metrics['port_sharpe'] - metrics['bench_sharpe']:.2f} vs ETF Benchmark", delta_color="normal")

    st.divider()

    # --- Rendering Charts ---
    visual_col, pie_col = st.columns([2, 1])

    with visual_col:
        st.subheader("Equity Curve vs Benchmark (VN30 ETF)")
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=opt_val.index, y=opt_val, mode='lines', name='Dynamic Portfolio', line=dict(color='#00CC96', width=2)))
        fig_line.add_trace(go.Scatter(x=bench_val.index, y=bench_val, mode='lines', name='VN30 ETF (E1VFVN30)', line=dict(color='#636EFA', width=2)))
        
        fig_line.update_layout(
            yaxis_title="Portfolio Value (VND)",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with pie_col:
        st.subheader("Current Allocation")
        st.caption("Target weights from the latest monthly rebalance.")
        
        filtered_weights = {k: v for k, v in final_weights.items() if v > 0.01}
        labels = list(filtered_weights.keys())
        values = list(filtered_weights.values())
        
        if len(values) > 0:
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, hoverinfo="label+percent")])
            fig_pie.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.1))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No significant allocations available.")
