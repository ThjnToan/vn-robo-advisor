import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
import os
import json

# --- Page Configuration ---
st.set_page_config(page_title="VN Robo-Advisor Live", page_icon="📈", layout="wide")

# --- Constants & Paths ---
DEFAULT_TICKERS = [
    "ACB", "BCG", "BCM", "BID", "BMP", "BVH", "CII", "CMG", "CRE", "CTD", 
    "CTG", "DBC", "DCM", "DGC", "DGW", "DIG", "DPM", "DXG", "EIB", "FCN", 
    "FPT", "FRT", "GAS", "GEX", "GMD", "GVR", "HAG", "HAH", "HDB", "HDC", 
    "HDG", "HHV", "HPG", "HSG", "HT1", "ITA", "KBC", "KDC", "KDH", "LPB", 
    "MBB", "MSB", "MSN", "MWG", "NKG", "NLG", "NT2", "NVL", "OCB", "PAN", 
    "PC1", "PDR", "PHR", "PLX", "PNJ", "POW", "PTB", "PVD", "PVT", "REE", 
    "SAB", "SAM", "SBT", "SCS", "SHB", "SJS", "SSB", "SSI", "STB", "SZC", 
    "TCB", "TCH", "TCM", "TPB", "VCB", "VCG", "VCI", "VHC", "VHM", "VIB", 
    "VIC", "VIX", "VJC", "VND", "VNM", "VPB", "VPI", "VRE", "VSH"
]
LOOKBACK_DAYS = 252 # 1 year trading days
TX_FEE_RATE = 0.0015 # 0.15%

BASE_DIR = os.path.dirname(__file__)
DATA_FILE = os.path.join(BASE_DIR, "market_data.csv")
STATE_FILE = os.path.join(BASE_DIR, "portfolio_state.json")
LEDGER_FILE = os.path.join(BASE_DIR, "transactions.csv")

# --- Core Data Functions ---
@st.cache_data(ttl=timedelta(hours=6))
def load_market_data():
    if not os.path.exists(DATA_FILE):
        st.error("market_data.csv not found! Please run fetch_historical_data.py")
        return pd.DataFrame()
        
    df = pd.read_csv(DATA_FILE, parse_dates=['time'], index_col='time')
    
    # Auto-Update Logic: Fetch missing days up to today
    last_date = df.index.max()
    today_date = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))
    
    if last_date < today_date:
        start_fetch = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        end_fetch = today_date.strftime('%Y-%m-%d')
        
        new_data = pd.DataFrame()
        tickers_to_fetch = list(df.columns)
        
        # Don't show spinner if difference is just the weekend and it fails, but good to try.
        for ticker in tickers_to_fetch:
            for source in ['VCI', 'TCBS', 'SSI', 'VND']:
                try:
                    stock = Vnstock().stock(symbol=ticker, source=source)
                    temp_df = stock.quote.history(start=start_fetch, end=end_fetch)
                    if temp_df is not None and not temp_df.empty:
                        temp_df['time'] = pd.to_datetime(temp_df['time'])
                        temp_df.set_index('time', inplace=True)
                        new_data[ticker] = temp_df['close']
                        break
                except Exception:
                    continue
                    
        if not new_data.empty:
            new_data.ffill(inplace=True)
            combined = pd.concat([df, new_data])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined.sort_index(inplace=True)
            # Save raw vnstock values (shorthand thousands) back to CSV
            combined.to_csv(DATA_FILE)
            df = combined
            st.toast(f"Market data auto-updated to {end_fetch}!")
            
    # vnstock returns prices in shorthand thousands (e.g. 60). Convert to true VND (60,000)
    df = df * 1000
    return df

# --- State Management ---
def init_portfolio(initial_capital):
    state = {
        "start_date": datetime.today().strftime('%Y-%m-%d'),
        "initial_capital": initial_capital,
        "cash_balance": initial_capital,
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)
        
    df = pd.DataFrame(columns=['Date', 'Ticker', 'Action', 'Shares', 'Price', 'Value', 'Fee'])
    df.to_csv(LEDGER_FILE, index=False)
    st.success("Portfolio Initialized! Ready for the Robo-Advisor.")
    return state

def load_portfolio():
    if os.path.exists(STATE_FILE) and os.path.exists(LEDGER_FILE):
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
        ledger = pd.read_csv(LEDGER_FILE)
        return state, ledger
    return None, None

def calculate_current_holdings(ledger):
    if ledger.empty:
        return {}
    
    holdings = {}
    for _, row in ledger.iterrows():
        ticker = row['Ticker']
        shares = row['Shares']
        if row['Action'] == 'BUY':
            holdings[ticker] = holdings.get(ticker, 0) + shares
        elif row['Action'] == 'SELL':
            holdings[ticker] = holdings.get(ticker, 0) - shares
            
    # Filter out empty positions
    return {k: v for k, v in holdings.items() if v > 0}

# --- Markowitz Optimization Engine ---
def optimize_portfolio(prices_df, target_tickers, max_weight):
    mu = expected_returns.mean_historical_return(prices_df[target_tickers], frequency=252)
    S = risk_models.sample_cov(prices_df[target_tickers], frequency=252)
    
    ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
    try:
        raw_weights = ef.max_sharpe()
    except Exception:
        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
        raw_weights = ef.min_volatility()
        
    # Apply a strict 5% cutoff to zero-out tiny insignificant allocations 
    # and force the optimizer to concentrate on the absolute best stocks.
    return ef.clean_weights(cutoff=0.05)

def execute_rebalance(market_data, state, ledger, target_weights_dict):
    today = market_data.index.max() # Simulate "today" as the last available day in db
    today_str = today.strftime('%Y-%m-%d')
    
    current_holdings = calculate_current_holdings(ledger)
    cash = state['cash_balance']
    
    # Calculate current portfolio value
    portfolio_value = cash
    current_prices = {}
    for ticker, shares in current_holdings.items():
        if ticker in market_data.columns:
            price = market_data.loc[today, ticker]
            current_prices[ticker] = price
            portfolio_value += shares * price
            
    # Calculate target values and necessary trades
    new_ledger_rows = []
    
    # First: Execute all Sells to free up cash
    for ticker, current_shares in current_holdings.items():
        target_weight = target_weights_dict.get(ticker, 0)
        target_value = portfolio_value * target_weight
        price = market_data.loc[today, ticker] if ticker in market_data.columns else current_prices.get(ticker, 0)
        
        if price > 0:
            target_shares = int(target_value / price)
            # Apply Vietnam HOSE Board Lot constraint (multiples of 100)
            target_shares = (target_shares // 100) * 100
            
            if target_shares < current_shares:
                # Need to sell
                shares_to_sell = current_shares - target_shares
                trade_value = shares_to_sell * price
                fee = trade_value * TX_FEE_RATE
                
                cash += (trade_value - fee)
                new_ledger_rows.append({
                    'Date': today_str, 'Ticker': ticker, 'Action': 'SELL', 
                    'Shares': shares_to_sell, 'Price': price, 'Value': trade_value, 'Fee': fee
                })
                
    # Second: Execute all Buys with available cash
    for ticker, target_weight in target_weights_dict.items():
        if target_weight > 0:
            target_value = portfolio_value * target_weight
            price = market_data.loc[today, ticker]
            current_shares = current_holdings.get(ticker, 0)
            
            target_shares = int(target_value / price)
            # Apply Vietnam HOSE Board Lot constraint (multiples of 100)
            target_shares = (target_shares // 100) * 100
            
            if target_shares > current_shares:
                # Need to buy
                shares_to_buy = target_shares - current_shares
                trade_value = shares_to_buy * price
                fee = trade_value * TX_FEE_RATE
                
                total_cost = trade_value + fee
                if cash >= total_cost:
                    cash -= total_cost
                    new_ledger_rows.append({
                        'Date': today_str, 'Ticker': ticker, 'Action': 'BUY', 
                        'Shares': shares_to_buy, 'Price': price, 'Value': trade_value, 'Fee': fee
                    })
                else:
                    st.toast(f"Not enough cash to buy {ticker}. Skipping.")
                    
    # Save State
    if new_ledger_rows:
        new_df = pd.DataFrame(new_ledger_rows)
        updated_ledger = pd.concat([ledger, new_df], ignore_index=True)
        updated_ledger.to_csv(LEDGER_FILE, index=False)
        
        state['cash_balance'] = cash
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f)
            
        st.success(f"Robo-Advisor executed {len(new_ledger_rows)} trades successfully!")
    else:
        st.info("Portfolio already matches optimal weights. No trades needed.")

# --- App Layout ---
st.title("🇻🇳 Live Robo-Advisor Tracker")
st.markdown("A persistent portfolio tracker powered by the Markowitz Mean-Variance optimization engine.")

# Database Initialization
market_db = load_market_data()
if market_db.empty:
    st.stop()

state, ledger = load_portfolio()

with st.sidebar:
    st.header("Wallet Settings")
    if state is None:
        initial_cap = st.number_input("Initial Deposit (VND)", min_value=1_000_000, value=1_000_000_000, step=100_000_000)
        if st.button("Initialize Portfolio Wallet", type="primary"):
            state = init_portfolio(initial_cap)
            st.rerun()
    else:
        st.success("Wallet Active")
        if st.button("🔴 Reset/Delete Entire Portfolio"):
            if os.path.exists(STATE_FILE): os.remove(STATE_FILE)
            if os.path.exists(LEDGER_FILE): os.remove(LEDGER_FILE)
            st.rerun()
            
    st.divider()
    st.subheader("Robo-Advisor Constraints")
    max_weight_pct = st.slider("Max Allocation per Stock", min_value=10, max_value=100, value=30, step=5, format="%d%%")
    max_weight_dec = max_weight_pct / 100.0
    
    if "ticker_selector" not in st.session_state:
        st.session_state.ticker_selector = ["HPG", "VCB", "VHM", "REE", "MBB", "TCB"]

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Select All", use_container_width=True):
            st.session_state.ticker_selector = [t for t in DEFAULT_TICKERS if t in market_db.columns]
            st.rerun()
    with colB:
        if st.button("Clear", use_container_width=True):
            st.session_state.ticker_selector = []
            st.rerun()

    selected_tickers = st.multiselect(
        "Active ETF Universe", 
        options=DEFAULT_TICKERS, 
        key="ticker_selector"
    )

if state is None:
    st.info("👋 Welcome! Please initialize your wallet in the sidebar to start tracking your portfolio.")
    st.stop()

# --- Main Dashboard Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Current Holdings", "📈 Performance Tracker", "🤖 Robo-Advisor Terminal"])

# Calculate real-time stats
holdings = calculate_current_holdings(ledger)
latest_date = market_db.index.max()
cash = state['cash_balance']

nav = cash
holding_values = {}
for ticker, shares in holdings.items():
    if ticker in market_db.columns:
        price = market_db.loc[latest_date, ticker]
        val = shares * price
        holding_values[ticker] = val
        nav += val

with tab1:
    st.subheader("Live Portfolio Snapshot")
    st.caption(f"Based on last market close: {latest_date.strftime('%Y-%m-%d')}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Net Asset Value (NAV)", f"{nav:,.0f} ₫")
    col2.metric("Available Cash", f"{cash:,.0f} ₫")
    
    total_roi = ((nav / state['initial_capital']) - 1) * 100
    col3.metric("All-Time Return", f"{total_roi:.2f} %", f"{nav - state['initial_capital']:,.0f} ₫")
    
    st.divider()
    
    if holdings:
        colA, colB = st.columns([1, 2])
        with colA:
            st.write("### Current Positions")
            pos_df = pd.DataFrame([
                {"Ticker": k, "Shares": v, "Current Value (VND)": holding_values[k]}
                for k, v in holdings.items()
            ])
            
            st.dataframe(
                pos_df.style.format({
                    "Shares": "{:,.0f}", 
                    "Current Value (VND)": "{:,.0f}"
                }), 
                use_container_width=True, 
                hide_index=True
            )
            
        with colB:
            # Add cash to pie chart
            labels = list(holding_values.keys()) + ["Cash"]
            values = list(holding_values.values()) + [cash]
            fig = go.Figure(data=[go.Pie(
                labels=labels, 
                values=values, 
                hole=.4, 
                textinfo='label+percent',
                hovertemplate="<b>%{label}</b><br>Value: %{value:,.0f} ₫<br>Allocation: %{percent}<extra></extra>"
            )])
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Your portfolio is empty. Go to the Robo-Advisor tab to make your first investment!")

with tab2:
    st.subheader("Historical Tracking")
    if ledger.empty:
        st.info("No trading history yet. Run the Robo-Advisor to start tracking.")
    else:
        st.write("Replaying your actual trade ledger against historical market data...")
        
        # 1. Prepare timeline
        first_trade_date = pd.to_datetime(ledger['Date'].min())
        timeline = market_db.loc[first_trade_date:latest_date].index
        
        # 2. Replay ledger day by day to build equity curve
        portfolio_history = []
        current_shares = {}
        current_cash = state['initial_capital']
        
        # Sort ledger by date for chronological replay
        sorted_ledger = ledger.copy()
        sorted_ledger['Date'] = pd.to_datetime(sorted_ledger['Date'])
        sorted_ledger = sorted_ledger.sort_values('Date')
        
        for date in timeline:
            # Apply any trades that happened on or before this date
            day_trades = sorted_ledger[sorted_ledger['Date'] == date]
            for _, trade in day_trades.iterrows():
                ticker = trade['Ticker']
                shares = trade['Shares']
                total_cost_or_proceeds = trade['Value'] + trade['Fee'] if trade['Action'] == 'BUY' else trade['Value'] - trade['Fee']
                
                if trade['Action'] == 'BUY':
                    current_shares[ticker] = current_shares.get(ticker, 0) + shares
                    current_cash -= total_cost_or_proceeds
                else:
                    current_shares[ticker] = current_shares.get(ticker, 0) - shares
                    current_cash += total_cost_or_proceeds
                    
            # Calculate EOD NAV
            eod_nav = current_cash
            for ticker, shares in current_shares.items():
                if shares > 0 and ticker in market_db.columns:
                    eod_nav += shares * market_db.loc[date, ticker]
            
            portfolio_history.append({'time': date, 'NAV': eod_nav})
            
        history_df = pd.DataFrame(portfolio_history).set_index('time')
        
        # 3. Calculate Benchmark (VN30 ETF) Normalized
        if 'E1VFVN30' in market_db.columns:
            bench_prices = market_db.loc[timeline, 'E1VFVN30']
            if not bench_prices.empty:
                bench_shares_bought = state['initial_capital'] / bench_prices.iloc[0]
                bench_values = bench_prices * bench_shares_bought
            else:
                bench_values = pd.Series(state['initial_capital'], index=timeline)
        else:
            bench_values = pd.Series(state['initial_capital'], index=timeline)
            
        # 4. Plot Equity Curve
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=history_df.index, y=history_df['NAV'], 
            mode='lines', name='Actual Portfolio', 
            line=dict(color='#00CC96', width=2),
            hovertemplate="<b>Actual Portfolio</b><br>NAV: %{y:,.0f} ₫<extra></extra>"
        ))
        fig_line.add_trace(go.Scatter(
            x=bench_values.index, y=bench_values, 
            mode='lines', name='VN30 ETF (E1VFVN30)', 
            line=dict(color='#636EFA', width=2),
            hovertemplate="<b>VN30 ETF</b><br>NAV: %{y:,.0f} ₫<extra></extra>"
        ))
        
        fig_line.update_layout(
            yaxis_title="Portfolio Value (VND)",
            yaxis_tickformat=",.0f",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_line, use_container_width=True)

with tab3:
    st.subheader("Robo-Advisor Terminal")
    st.write("Clicking the optimization button will calculate the mathematically ideal portfolio based on the last 252 trading days, and automatically buy/sell shares using your available cash.")
    
    if st.button("🚀 Run Monthly Rebalance Engine", use_container_width=True):
        if len(selected_tickers) < 2:
            st.error("Select at least 2 stocks in the universe sidebar.")
        else:
            with st.spinner("Analyzing market data and calculating Maximum Sharpe Ratio..."):
                # Filter out any selected tickers that failed to download
                valid_tickers = [t for t in selected_tickers if t in market_db.columns]
                
                if len(valid_tickers) < 2:
                    st.error("Not enough valid historical data for selected tickers. The API may have failed to download them.")
                else:
                    # Filter last 1 year of data
                    train_data = market_db[valid_tickers].loc[:latest_date].tail(LOOKBACK_DAYS)
                    
                    # Get Target Weights
                    target_w = optimize_portfolio(train_data, valid_tickers, max_weight_dec)
                
                # Execute Virtual Trades
                execute_rebalance(market_db, state, ledger, target_w)
                
                # Reload UI
                st.rerun()
                
    st.divider()
    st.subheader("Transaction Ledger (Editable)")
    if not ledger.empty:
        st.write("Edit rows below to manually adjust trades or fix slippage prices. Click 'Save Edits' to recalculate your wallet.")
        
        # Display editable dataframe
        editable_ledger = ledger.copy()
        
        # Apply pandas styling to force commas in the UI while keeping the underlying data numeric
        styled_ledger = editable_ledger.style.format({
            "Shares": "{:,.0f}",
            "Price": "{:,.0f}",
            "Value": "{:,.0f}",
            "Fee": "{:,.0f}"
        })
        
        edited_ledger = st.data_editor(
            styled_ledger, 
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "Date": st.column_config.TextColumn("Date", required=True),
                "Ticker": st.column_config.TextColumn("Ticker", required=True),
                "Action": st.column_config.SelectboxColumn("Action", options=["BUY", "SELL"], required=True),
                "Shares": st.column_config.NumberColumn("Shares", min_value=1, step=100, required=True),
                "Price": st.column_config.NumberColumn("Price (VND)", min_value=0, required=True),
                "Value": st.column_config.NumberColumn("Value (VND)", disabled=True),
                "Fee": st.column_config.NumberColumn("Fee (VND)", disabled=True)
            },
            hide_index=True
        )
        
        if st.button("💾 Save Ledger Edits", type="primary"):
            # Reconciliation
            # Recalculate Value and Fee
            edited_ledger['Value'] = edited_ledger['Shares'] * edited_ledger['Price']
            edited_ledger['Fee'] = edited_ledger['Value'] * TX_FEE_RATE
            
            # Recalculate Cash Balance chronologically
            new_cash = state['initial_capital']
            for _, row in edited_ledger.iterrows():
                total_cost = row['Value'] + row['Fee'] if row['Action'] == 'BUY' else row['Value'] - row['Fee']
                if row['Action'] == 'BUY':
                    new_cash -= total_cost
                else:
                    new_cash += total_cost
                    
            state['cash_balance'] = new_cash
            
            # Save state
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f)
            
            edited_ledger.to_csv(LEDGER_FILE, index=False)
            st.success("Ledger and Wallet Cash Balance successfully updated!")
            st.rerun()
    else:
        st.write("No trades recorded.")
