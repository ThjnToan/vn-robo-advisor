import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import black_litterman
import os
import json
import io
from fpdf import FPDF
import base64

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

SECTOR_MAPPER = {
    # Financials (Banks & Brokerages)
    "ACB": "Financials", "BID": "Financials", "BVH": "Financials", "CTG": "Financials",
    "EIB": "Financials", "HDB": "Financials", "LPB": "Financials", "MBB": "Financials",
    "MSB": "Financials", "OCB": "Financials", "SHB": "Financials", "SSB": "Financials",
    "SSI": "Financials", "STB": "Financials", "TCB": "Financials", "TPB": "Financials",
    "VCB": "Financials", "VCI": "Financials", "VIB": "Financials", "VIX": "Financials",
    "VND": "Financials", "VPB": "Financials",
    
    # Real Estate & Construction
    "BCG": "Real Estate", "BCM": "Real Estate", "CII": "Real Estate", "CRE": "Real Estate",
    "CTD": "Real Estate", "DIG": "Real Estate", "DXG": "Real Estate", "FCN": "Real Estate",
    "HDC": "Real Estate", "HDG": "Real Estate", "HHV": "Real Estate", "ITA": "Real Estate",
    "KBC": "Real Estate", "KDH": "Real Estate", "NLG": "Real Estate", "NVL": "Real Estate",
    "PDR": "Real Estate", "SJS": "Real Estate", "SZC": "Real Estate", "TCH": "Real Estate",
    "VCG": "Real Estate", "VHM": "Real Estate", "VIC": "Real Estate", "VPI": "Real Estate",
    "VRE": "Real Estate",
    
    # Materials (Steel, Chemicals, Rubber)
    "BMP": "Materials", "DCM": "Materials", "DGC": "Materials", "DPM": "Materials",
    "GVR": "Materials", "HPG": "Materials", "HSG": "Materials", "HT1": "Materials",
    "NKG": "Materials", "PHR": "Materials",
    
    # Consumer Discretionary & Staples
    "DBC": "Consumer", "DGW": "Consumer", "FRT": "Consumer", "KDC": "Consumer",
    "MSN": "Consumer", "MWG": "Consumer", "PAN": "Consumer", "PNJ": "Consumer",
    "SAB": "Consumer", "SBT": "Consumer", "TCM": "Consumer", "VHC": "Consumer",
    "VNM": "Consumer",
    
    # Industrials & Logistics
    "GMD": "Industrials", "HAH": "Industrials", "PTB": "Industrials", "PVT": "Industrials",
    "REE": "Industrials", "SAM": "Industrials", "SCS": "Industrials", "VJC": "Industrials",
    
    # Energy & Utilities
    "GAS": "Utilities", "GEX": "Utilities", "NT2": "Utilities", "PC1": "Utilities",
    "PLX": "Utilities", "POW": "Utilities", "PVD": "Utilities", "VSH": "Utilities",
    
    # Technology
    "CMG": "Technology", "FPT": "Technology"
}

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

def fetch_live_prices(tickers):
    """
    Pings the VCI server for the current day's live trading bar.
    Returns a dictionary of ticker -> live_price.
    """
    live_prices = {}
    today_str = datetime.today().strftime('%Y-%m-%d')
    for ticker in tickers:
        try:
            # We fetch exactly today's bar. The 'close' value of an active daily bar is the live price.
            stock = Vnstock().stock(symbol=ticker, source='VCI')
            live_df = stock.quote.history(start=today_str, end=today_str)
            if not live_df.empty:
                # vnstock returns prices in shorthand thousands. Convert to true VND.
                live_prices[ticker] = live_df['close'].iloc[-1] * 1000
        except Exception as e:
            print(f"Failed to fetch live price for {ticker}: {e}")
            pass
    return live_prices

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

    return {k: v for k, v in holdings.items() if v > 0}

# --- Markowitz Optimization Engine ---
def optimize_portfolio(prices_df, target_tickers, max_weight, max_sector_weight=1.0):
    mu = expected_returns.mean_historical_return(prices_df[target_tickers], frequency=252)
    S = risk_models.sample_cov(prices_df[target_tickers], frequency=252)
    
    ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
    
    # Enforce Sector Constraints
    if max_sector_weight < 1.0:
        sector_lower = {sector: 0.0 for sector in set(SECTOR_MAPPER.values())}
        sector_upper = {sector: max_sector_weight for sector in set(SECTOR_MAPPER.values())}
        ef.add_sector_constraints(SECTOR_MAPPER, sector_lower, sector_upper)
        
    try:
        raw_weights = ef.max_sharpe()
    except Exception:
        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
        if max_sector_weight < 1.0:
            ef.add_sector_constraints(SECTOR_MAPPER, sector_lower, sector_upper)
        raw_weights = ef.min_volatility()
        
    # Apply a strict 5% cutoff to zero-out tiny insignificant allocations 
    # and force the optimizer to concentrate on the absolute best stocks.
    return ef.clean_weights(cutoff=0.05)

def optimize_black_litterman(prices_df, target_tickers, max_weight, tactical_views, max_sector_weight=1.0):
    """
    Runs the Black-Litterman robust optimization model blending historical 
    covariance with the user's subjective tactical views.
    """
    S = risk_models.sample_cov(prices_df[target_tickers], frequency=252)
    
    # 1. Calculate Market-Implied Prior Returns
    # Assuming equal market caps for the baseline VN100 universe since we don't fetch live caps
    mcaps = {ticker: 1.0 for ticker in target_tickers}
    delta = black_litterman.market_implied_risk_aversion(prices_df[target_tickers].mean(axis=1)) # Simple proxy for market
    market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)
    
    # 2. Integrate User Views
    # Map confidence (0.0 to 1.0) to omega (uncertainty matrix). Default PyPortfolioOpt handles this via Idzorek's method if we pass confidences,
    # but for simplicity, we pass absolute views and let BL generate default diagonal omega.
    view_dict = {}
    for row in tactical_views:
        if row['Ticker'] in target_tickers:
            view_dict[row['Ticker']] = row['Expected Return (%)'] / 100.0
            
    if not view_dict:
        # Fallback to standard markowitz if no valid views
        return optimize_portfolio(prices_df, target_tickers, max_weight, max_sector_weight)
        
    bl = black_litterman.BlackLittermanModel(S, pi=market_prior, absolute_views=view_dict)
    posterior_rets = bl.bl_returns()
    posterior_cov = bl.bl_cov()
    
    # 3. Maximize Sharpe on the Posterior Distribution
    ef = EfficientFrontier(posterior_rets, posterior_cov, weight_bounds=(0, max_weight))
    
    # Enforce Sector Constraints
    if max_sector_weight < 1.0:
        sector_lower = {sector: 0.0 for sector in set(SECTOR_MAPPER.values())}
        sector_upper = {sector: max_sector_weight for sector in set(SECTOR_MAPPER.values())}
        ef.add_sector_constraints(SECTOR_MAPPER, sector_lower, sector_upper)
        
    try:
        raw_weights = ef.max_sharpe()
    except Exception:
        ef = EfficientFrontier(posterior_rets, posterior_cov, weight_bounds=(0, max_weight))
        if max_sector_weight < 1.0:
            ef.add_sector_constraints(SECTOR_MAPPER, sector_lower, sector_upper)
        raw_weights = ef.min_volatility()
        
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

def run_monte_carlo_var(prices_df, holdings_dict, current_cash, num_simulations=10000, sim_days=30):
    """ Runs 10,000 correlated random-walk price paths using Cholesky Decomposition of historical covariance. """
    tickers = [t for t in holdings_dict.keys() if t in prices_df.columns]
    if not tickers:
        return None
        
    latest_date = prices_df.index.max()
    initial_values = np.array([holdings_dict[t] * prices_df.loc[latest_date, t] for t in tickers])
    current_nav = np.sum(initial_values) + current_cash
    
    # Calculate daily historical returns over the last year
    historical_returns = prices_df[tickers].tail(252).pct_change().dropna()
    mu = historical_returns.mean().values
    cov = historical_returns.cov().values
    
    # Generate random daily returns for all assets across all simulations and days
    # Shape: (10000, 30, num_assets)
    np.random.seed(42) # For reproducible deterministic demos
    try:
        daily_sim_returns = np.random.multivariate_normal(mu, cov, (num_simulations, sim_days))
    except Exception:
        return None # Fallback if cov matrix is not positive semi-definite
        
    # Calculate cumulative returns over the period
    cumulative_returns = np.cumprod(1 + daily_sim_returns, axis=1)
    
    # Get the final cumulative return at the end of sim_days
    final_returns = cumulative_returns[:, -1, :] # Shape: (sims, assets)
    
    # Calculate final portfolio values for all simulations
    final_portfolio_values = np.sum(initial_values * final_returns, axis=1) + current_cash
    
    # Calculate Risk Metrics
    var_95_value = np.percentile(final_portfolio_values, 5)
    cvar_95_value = final_portfolio_values[final_portfolio_values <= var_95_value].mean()
    
    var_95_loss = current_nav - var_95_value
    cvar_95_loss = current_nav - cvar_95_value
    
    
    return final_portfolio_values, var_95_loss, cvar_95_loss, var_95_value, current_nav

@st.cache_data(show_spinner=False)
def export_tearsheet(nav_value, roi_pct, holdings_json, cash_val):
    """ Builds a pure-Python PDF Tearsheet in a RAM buffer using fpdf2 """
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("helvetica", "B", 24)
    pdf.cell(0, 15, "Robo-Advisor: Quantitative Tearsheet", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "I", 12)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    
    # Portfolio Snapshot
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "1. Live Portfolio Snapshot", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 12)
    pdf.cell(0, 8, f"Net Asset Value (NAV): {nav_value:,.0f} VND", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Available Cash: {cash_val:,.0f} VND", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"All-Time Return (ROI): {roi_pct:.2f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    
    # Holdings Table
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "2. Current Allocations", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "B", 12)
    
    # Table Header
    pdf.cell(60, 10, "Ticker", border=1, align="C")
    pdf.cell(60, 10, "Shares", border=1, align="C")
    pdf.cell(60, 10, "Value (VND)", border=1, align="C", new_x="LMARGIN", new_y="NEXT")
    
    # Table Body
    pdf.set_font("helvetica", "", 12)
    holdings_dict = json.loads(holdings_json)
    for ticker, val in holdings_dict.items():
        pdf.cell(60, 10, str(ticker), border=1, align="C")
        pdf.cell(60, 10, f"{val['shares']:,.0f}", border=1, align="C")
        pdf.cell(60, 10, f"{val['value']:,.0f}", border=1, align="R", new_x="LMARGIN", new_y="NEXT")
        
    pdf.ln(10)
    
    # Risk Disclaimer
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "3. Risk Disclosure", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 10)
    pdf.multi_cell(0, 6, "This quantitative tearsheet is generated autonomously by the Capstone Markowitz Mean-Variance engine. Past performance does not guarantee future returns. The underlying VN100 Covariance matrix dynamically shifts based on real-time market volatility.")
    
    # Generate raw byte array directly to bypass Streamlit UUID bugs
    pdf_bytes = pdf.output()
    return bytes(pdf_bytes)

@st.cache_data(show_spinner=False)
def compute_efficient_frontier(tickers_tuple, latest_date_str, n_portfolios=3000):
    """Simulate n_portfolios random weight combinations and compute the efficient frontier."""
    market_data = load_market_data()
    tickers = list(tickers_tuple)
    latest_date = pd.to_datetime(latest_date_str)
    
    prices = market_data[tickers].loc[:latest_date].tail(252)
    returns = prices.pct_change().dropna()
    
    if returns.shape[0] < 30 or returns.shape[1] < 2:
        return None
    
    mu_daily = returns.mean()
    cov_daily = returns.cov()
    ann_factor = 252
    
    np.random.seed(99)
    rand_vols, rand_rets, rand_sharpes = [], [], []
    rand_weights_list = []
    
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(len(tickers)))
        p_ret = float(np.dot(w, mu_daily) * ann_factor)
        p_vol = float(np.sqrt(np.dot(w.T, np.dot(cov_daily.values * ann_factor, w))))
        p_sharpe = p_ret / p_vol if p_vol > 0 else 0
        rand_rets.append(p_ret)
        rand_vols.append(p_vol)
        rand_sharpes.append(p_sharpe)
        rand_weights_list.append(w)
    
    # Trace the actual efficient frontier line via PyPortfolioOpt
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
        if "init_cap_raw" not in st.session_state:
            st.session_state.init_cap_raw = "1,000,000,000"
            
        def format_cap():
            val = st.session_state.init_cap_raw.replace(',', '').replace('.', '').strip()
            if val.isdigit():
                st.session_state.init_cap_raw = f"{int(val):,}"

        initial_cap_str = st.text_input("Initial Deposit (VND)", key="init_cap_raw", on_change=format_cap)
        try:
            initial_cap = int(initial_cap_str.replace(',', '').replace('.', ''))
        except ValueError:
            st.error("Invalid number format. Defaulting to 1 Billion.")
            initial_cap = 1000000000
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
    max_weight = max_weight_pct / 100.0
    
    max_sector_pct = st.slider("Max Allocation per Sector", min_value=max_weight_pct, max_value=100, value=100, step=5, format="%d%%")
    max_sector_weight = max_sector_pct / 100.0
    
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
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        st.subheader("Live Portfolio Snapshot")
        st.caption(f"Historical EOD Baseline: {latest_date.strftime('%Y-%m-%d')}")
    with col_hdr2:
        if st.button("⚡ Ping Live NAV (VND)", use_container_width=True):
            with st.spinner("Pinging VCI servers for real-time orderbook..."):
                active_tickers = list(holdings.keys())
                live_prices = fetch_live_prices(active_tickers)
                st.session_state.live_prices_cache = live_prices
                st.session_state.live_timestamp = datetime.now().strftime("%H:%M:%S")
                st.rerun()

    # Determine if we are showing Live or EOD NAV
    showing_live = "live_prices_cache" in st.session_state
    
    display_nav = cash
    display_holding_values = {}
    
    if showing_live:
        for ticker, shares in holdings.items():
            price = st.session_state.live_prices_cache.get(ticker, market_db.loc[latest_date, ticker])
            val = shares * price
            display_holding_values[ticker] = val
            display_nav += val
    else:
        display_nav = nav
        display_holding_values = holding_values
        
    st.metric(
        label=f"Net Asset Value (NAV) {'[LIVE - ' + st.session_state.live_timestamp + ']' if showing_live else '[End of Day]'}", 
        value=f"{display_nav:,.0f} ₫",
        delta=f"{display_nav - nav:,.0f} ₫ (Intraday Change)" if showing_live else None
    )
    
    col1, col2 = st.columns(2)
    col1.metric("Available Cash", f"{cash:,.0f} ₫")
    
    total_roi = ((display_nav / state['initial_capital']) - 1) * 100
    col2.metric("All-Time Return", f"{total_roi:.2f} %", f"{display_nav - state['initial_capital']:,.0f} ₫")
    
    st.divider()
    
    if holdings:
        colA, colB = st.columns([1, 2])
        with colA:
            st.write("### Current Positions")
            pos_df = pd.DataFrame([
                {"Ticker": k, "Shares": v, "Current Value (VND)": display_holding_values[k]}
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
            labels = list(display_holding_values.keys()) + ["Cash"]
            values = list(display_holding_values.values()) + [cash]
            fig = go.Figure(data=[go.Pie(
                labels=labels, 
                values=values, 
                hole=.4, 
                textinfo='label+percent',
                hovertemplate="<b>%{label}</b><br>Value: %{value:,.0f} ₫<br>Allocation: %{percent}<extra></extra>"
            )])
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
        with st.expander("🔬 Institutional Risk Analytics (Monte Carlo VaR)", expanded=False):
            st.write("Simulate 10,000 future portfolio paths over the next 30 days based on historical volatility and correlation to calculate your Maximum Expected Downside.")
            if st.button("Run 10,000 Risk Simulations", use_container_width=True):
                with st.spinner("Crunching 10,000 multivariate normal random walks using Cholesky Decomposition..."):
                    results = run_monte_carlo_var(market_db, holdings, cash)
                    if results:
                        final_vals, var_loss, cvar_loss, var_threshold, port_nav = results
                        
                        rc1, rc2 = st.columns(2)
                        with rc1:
                            st.error(f"**95% Value-at-Risk (VaR)**: -{var_loss:,.0f} ₫")
                            st.caption("In 95 out of 100 simulated alternate realities, your portfolio will NOT lose more than this amount over the next 30 days.")
                        with rc2:
                            st.error(f"**Expected Shortfall (CVaR)**: -{cvar_loss:,.0f} ₫")
                            st.caption("If a catastrophic 5% tail-event DOES occur, this is your average expected mathematical loss.")
                        
                        # Plot Histogram
                        fig_hist = go.Figure(data=[go.Histogram(x=final_vals, nbinsx=50, marker_color='#636EFA')])
                        fig_hist.add_vline(x=var_threshold, line_width=3, line_dash="dash", line_color="red", 
                                          annotation_text="95% VaR Cutoff", annotation_position="top left")
                        fig_hist.add_vline(x=port_nav, line_width=2, line_dash="solid", line_color="green",
                                          annotation_text="Current NAV", annotation_position="top right")
                        
                        fig_hist.update_layout(
                            title="Distribution of 10,000 Simulated 30-Day Valuations",
                            xaxis_title="Final Portfolio Value (VND)",
                            yaxis_title="Frequency (Probability)",
                            xaxis_tickformat=",.0f",
                            margin=dict(l=0, r=0, t=40, b=0)
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                    else:
                        st.error("Not enough valid historical data to run simulations.")
    else:
        st.info("Your portfolio is empty. Go to the Robo-Advisor tab to make your first investment!")

with tab2:
    col_hdr_2a, col_hdr_2b = st.columns([3, 1])
    with col_hdr_2a:
        st.subheader("Historical Tracking")
    
    with col_hdr_2b:
        pdf_holdings = {k: {"shares": v, "value": display_holding_values.get(k, 0)} for k, v in holdings.items() if v > 0}
        pdf_stream = export_tearsheet(display_nav, total_roi, json.dumps(pdf_holdings), cash)
        # Use a JS Blob in a sandboxed iframe to bypass Streamlit's HTML sanitizer 
        # which strips the `download` attribute from anchor tags in st.markdown.
        b64_pdf = base64.b64encode(pdf_stream).decode('utf-8')
        pdf_filename = f"RoboAdvisor_Tearsheet_{datetime.now().strftime('%Y%m%d')}.pdf"
        dl_html = f"""
        <style>
          #dl-btn {{
            display: inline-block; padding: 0.4rem 1rem;
            background-color: #FF4B4B; color: white;
            text-decoration: none; border-radius: 0.4rem;
            font-weight: 600; font-size: 14px; cursor: pointer;
            border: none; width: 100%; text-align: center;
          }}
          #dl-btn:hover {{ background-color: #cc2222; }}
        </style>
        <button id="dl-btn" onclick="downloadPDF()">📄 Download PDF Tearsheet</button>
        <script>
        function downloadPDF() {{
          var b64 = "{b64_pdf}";
          var byteChars = atob(b64);
          var byteNums = new Uint8Array(byteChars.length);
          for (var i = 0; i < byteChars.length; i++) {{
            byteNums[i] = byteChars.charCodeAt(i);
          }}
          var blob = new Blob([byteNums], {{type: 'application/pdf'}});
          var url = URL.createObjectURL(blob);
          var a = document.createElement('a');
          a.href = url;
          a.download = "{pdf_filename}";
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
        }}
        </script>
        """
        import streamlit.components.v1 as components
        components.html(dl_html, height=50)
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
    
    # Tactical Views UI
    st.write("### 🧠 Black-Litterman Tactical Views")
    st.caption("Optional: Input your personal expected returns for specific stocks to override the purely historical Markowitz baseline.")
    
    if "tactical_views" not in st.session_state:
        st.session_state.tactical_views = pd.DataFrame(columns=["Ticker", "Expected Return (%)"])
        
    edited_views = st.data_editor(
        st.session_state.tactical_views,
        num_rows="dynamic",
        column_config={
            "Ticker": st.column_config.SelectboxColumn("Ticker", options=market_db.columns.tolist(), required=True),
            "Expected Return (%)": st.column_config.NumberColumn("Expected Return (%)", min_value=-100.0, max_value=500.0, required=True)
        },
        use_container_width=True,
        hide_index=True
    )
    st.session_state.tactical_views = edited_views
    
    use_bl = st.toggle("Enable Black-Litterman Optimization", value=False)
    
    if st.button("🚀 Run Monthly Rebalance Engine", use_container_width=True):
        if not selected_tickers:
            st.toast("Please select at least one ETF/Stock in the sidebar.")
        else:
            with st.spinner("Crunching historical covariance & running optimization..."):
                # Filter out any selected tickers that failed to download
                valid_tickers = [t for t in selected_tickers if t in market_db.columns]
                
                if len(valid_tickers) < 2:
                    st.error("Not enough valid historical data for selected tickers. The API may have failed to download them.")
                else:
                    # Filter last 1 year of data
                    train_data = market_db[valid_tickers].loc[:latest_date].tail(LOOKBACK_DAYS)
                    
                    # Get Target Weights
                    if use_bl and not st.session_state.tactical_views.empty:
                        views_list = st.session_state.tactical_views.to_dict('records')
                        target_w = optimize_black_litterman(train_data, valid_tickers, max_weight, views_list, max_sector_weight)
                    else:
                        target_w = optimize_portfolio(train_data, valid_tickers, max_weight, max_sector_weight)
                
                # Execute Virtual Trades
                execute_rebalance(market_db, state, ledger, target_w)
                
                # Reload UI
                st.rerun()
                
    st.divider()
    
    # --- Efficient Frontier Visualization ---
    with st.expander("📈 Efficient Frontier — Feasible Portfolio Universe", expanded=False):
        if not selected_tickers or len(selected_tickers) < 2:
            st.info("Select at least 2 tickers in the sidebar to plot the Efficient Frontier.")
        else:
            valid_ef_tickers = [t for t in selected_tickers if t in market_db.columns]
            if len(valid_ef_tickers) < 2:
                st.info("Not enough valid market data for the selected tickers.")
            else:
                with st.spinner("Simulating 3,000 random portfolios across the risk-return plane..."):
                    ef_data = compute_efficient_frontier(tuple(sorted(valid_ef_tickers)), str(latest_date))
                
                if ef_data:
                    # Calculate current portfolio's position on the frontier
                    curr_vol, curr_ret = None, None
                    if holdings:
                        prices_ef = market_db[valid_ef_tickers].tail(252)
                        rets_ef = prices_ef.pct_change().dropna()
                        total_val = sum(holdings.get(t, 0) * market_db[t].iloc[-1] for t in valid_ef_tickers if t in holdings)
                        if total_val > 0:
                            w_curr = np.array([holdings.get(t, 0) * market_db[t].iloc[-1] / total_val for t in valid_ef_tickers])
                            curr_ret = float(np.dot(w_curr, rets_ef.mean()) * 252)
                            curr_vol = float(np.sqrt(np.dot(w_curr.T, np.dot(rets_ef.cov().values * 252, w_curr))))
                    
                    fig_ef = go.Figure()
                    
                    # 1. Random portfolios scatter (feasible set cloud)
                    fig_ef.add_trace(go.Scatter(
                        x=[v * 100 for v in ef_data['rand_vols']],
                        y=[r * 100 for r in ef_data['rand_rets']],
                        mode='markers',
                        marker=dict(
                            color=ef_data['rand_sharpes'],
                            colorscale='Viridis',
                            size=4, opacity=0.5,
                            colorbar=dict(title='Sharpe Ratio', thickness=12, x=1.01)
                        ),
                        name='Random Portfolios',
                        hovertemplate='Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
                    ))
                    
                    # 2. Efficient frontier line
                    if ef_data['frontier_vols']:
                        fig_ef.add_trace(go.Scatter(
                            x=[v * 100 for v in ef_data['frontier_vols']],
                            y=[r * 100 for r in ef_data['frontier_rets']],
                            mode='lines',
                            line=dict(color='white', width=3),
                            name='Efficient Frontier',
                            hovertemplate='Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
                        ))
                    
                    # 3. Current portfolio star marker
                    if curr_vol is not None and curr_ret is not None:
                        fig_ef.add_trace(go.Scatter(
                            x=[curr_vol * 100], y=[curr_ret * 100],
                            mode='markers+text',
                            marker=dict(symbol='star', size=20, color='gold', line=dict(color='black', width=1)),
                            text=['Your Portfolio'], textposition='top center',
                            textfont=dict(color='gold', size=12),
                            name='Your Portfolio',
                            hovertemplate='<b>Your Portfolio</b><br>Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>'
                        ))
                    
                    fig_ef.update_layout(
                        title=dict(text='Mean-Variance Efficient Frontier', font=dict(size=16)),
                        xaxis_title='Annualised Volatility (Risk) %',
                        yaxis_title='Annualised Return %',
                        plot_bgcolor='#0e1117',
                        paper_bgcolor='#0e1117',
                        font=dict(color='white'),
                        margin=dict(l=0, r=0, t=50, b=0),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                    )
                    fig_ef.update_xaxes(gridcolor='#2d2d2d')
                    fig_ef.update_yaxes(gridcolor='#2d2d2d')
                    st.plotly_chart(fig_ef, use_container_width=True)
                    st.caption("Each dot is one randomly weighted portfolio. Color = Sharpe Ratio (yellow = higher). The white line traces the mathematically optimal Efficient Frontier. The ★ gold star is your current Robo-Advisor allocation.")
                else:
                    st.error("Could not compute the Efficient Frontier with the available data.")
    
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
