import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import black_litterman
import json
import os

# --- Constants & Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) # Root
DATA_FILE = os.path.join(BASE_DIR, "market_data.csv")
STATE_FILE = os.path.join(BASE_DIR, "portfolio_state.json")
LEDGER_FILE = os.path.join(BASE_DIR, "transactions.csv")

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

SECTOR_MAPPER = {
    "ACB": "Financials", "BID": "Financials", "BVH": "Financials", "CTG": "Financials",
    "EIB": "Financials", "HDB": "Financials", "LPB": "Financials", "MBB": "Financials",
    "MSB": "Financials", "OCB": "Financials", "SHB": "Financials", "SSB": "Financials",
    "SSI": "Financials", "STB": "Financials", "TCB": "Financials", "TPB": "Financials",
    "VCB": "Financials", "VCI": "Financials", "VIB": "Financials", "VIX": "Financials",
    "VND": "Financials", "VPB": "Financials",
    "BCG": "Real Estate", "BCM": "Real Estate", "CII": "Real Estate", "CRE": "Real Estate",
    "CTD": "Real Estate", "DIG": "Real Estate", "DXG": "Real Estate", "FCN": "Real Estate",
    "HDC": "Real Estate", "HDG": "Real Estate", "HHV": "Real Estate", "ITA": "Real Estate",
    "KBC": "Real Estate", "KDH": "Real Estate", "NLG": "Real Estate", "NVL": "Real Estate",
    "PDR": "Real Estate", "SJS": "Real Estate", "SZC": "Real Estate", "TCH": "Real Estate",
    "VCG": "Real Estate", "VHM": "Real Estate", "VIC": "Real Estate", "VPI": "Real Estate",
    "VRE": "Real Estate",
    "BMP": "Materials", "DCM": "Materials", "DGC": "Materials", "DPM": "Materials",
    "GVR": "Materials", "HPG": "Materials", "HSG": "Materials", "HT1": "Materials",
    "NKG": "Materials", "PHR": "Materials",
    "DBC": "Consumer", "DGW": "Consumer", "FRT": "Consumer", "KDC": "Consumer",
    "MSN": "Consumer", "MWG": "Consumer", "PAN": "Consumer", "PNJ": "Consumer",
    "SAB": "Consumer", "SBT": "Consumer", "TCM": "Consumer", "VHC": "Consumer",
    "VNM": "Consumer",
    "GMD": "Industrials", "HAH": "Industrials", "PTB": "Industrials", "PVT": "Industrials",
    "REE": "Industrials", "SAM": "Industrials", "SCS": "Industrials", "VJC": "Industrials",
    "GAS": "Utilities", "GEX": "Utilities", "NT2": "Utilities", "PC1": "Utilities",
    "PLX": "Utilities", "POW": "Utilities", "PVD": "Utilities", "VSH": "Utilities",
    "CMG": "Technology", "FPT": "Technology"
}

LOOKBACK_DAYS = 252 
TX_FEE_RATE = 0.0015

# --- Core Data Functions ---
def load_market_data():
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()
    df = pd.read_csv(DATA_FILE, parse_dates=['time'], index_col='time')
    # Assuming market data is already updated by a background process/script
    # or we handle updates gracefully elsewhere.
    return df

# --- State Management ---
def init_portfolio(initial_capital: float):
    state = {
        "start_date": datetime.today().strftime('%Y-%m-%d'),
        "initial_capital": initial_capital,
        "cash_balance": initial_capital,
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)
        
    df = pd.DataFrame(columns=['Date', 'Ticker', 'Action', 'Shares', 'Price', 'Value', 'Fee'])
    df.to_csv(LEDGER_FILE, index=False)
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
            
    return {k: v for k, v in holdings.items() if v > 0}

# --- Markowitz Optimization Engine ---
def optimize_portfolio(prices_df, target_tickers, max_weight, max_sector_weight=1.0):
    if prices_df.empty or len(target_tickers) < 2:
        return {}
    mu = expected_returns.mean_historical_return(prices_df[target_tickers], frequency=252)
    S = risk_models.sample_cov(prices_df[target_tickers], frequency=252)
    
    ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
    
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
        
    return ef.clean_weights(cutoff=0.05)

def optimize_black_litterman(prices_df, target_tickers, max_weight, tactical_views, max_sector_weight=1.0):
    if prices_df.empty or len(target_tickers) < 2:
        return {}
    S = risk_models.sample_cov(prices_df[target_tickers], frequency=252)
    
    mcaps = {ticker: 1.0 for ticker in target_tickers}
    delta = black_litterman.market_implied_risk_aversion(prices_df[target_tickers].mean(axis=1))
    market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)
    
    view_dict = {}
    for row in tactical_views:
        if row['Ticker'] in target_tickers:
            view_dict[row['Ticker']] = row['Expected Return (%)'] / 100.0
            
    if not view_dict:
        return optimize_portfolio(prices_df, target_tickers, max_weight, max_sector_weight)
        
    bl = black_litterman.BlackLittermanModel(S, pi=market_prior, absolute_views=view_dict)
    posterior_rets = bl.bl_returns()
    posterior_cov = bl.bl_cov()
    
    ef = EfficientFrontier(posterior_rets, posterior_cov, weight_bounds=(0, max_weight))
    
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

def generate_trades(market_data, state, ledger, target_weights_dict):
    today = market_data.index.max()
    today_str = today.strftime('%Y-%m-%d')
    
    current_holdings = calculate_current_holdings(ledger)
    cash = state['cash_balance']
    
    portfolio_value = cash
    current_prices = {}
    for ticker, shares in current_holdings.items():
        if ticker in market_data.columns:
            price = market_data.loc[today, ticker]
            current_prices[ticker] = price
            portfolio_value += shares * price
            
    proposed_trades = []
    
    for ticker, current_shares in current_holdings.items():
        target_weight = target_weights_dict.get(ticker, 0)
        target_value = portfolio_value * target_weight
        price = market_data.loc[today, ticker] if ticker in market_data.columns else current_prices.get(ticker, 0)
        
        if price > 0:
            target_shares = int(target_value / price)
            target_shares = (target_shares // 100) * 100
            
            if target_shares < current_shares:
                shares_to_sell = current_shares - target_shares
                trade_value = shares_to_sell * price
                fee = trade_value * TX_FEE_RATE
                
                proposed_trades.append({
                    'Date': today_str, 'Ticker': ticker, 'Action': 'SELL', 
                    'Shares': shares_to_sell, 'Price': price, 'Value': trade_value, 'Fee': fee
                })
                cash += (trade_value - fee)
                
    for ticker, target_weight in target_weights_dict.items():
        if target_weight > 0:
            target_value = portfolio_value * target_weight
            price = market_data.loc[today, ticker]
            current_shares = current_holdings.get(ticker, 0)
            
            target_shares = int(target_value / price)
            target_shares = (target_shares // 100) * 100
            
            if target_shares > current_shares:
                shares_to_buy = target_shares - current_shares
                trade_value = shares_to_buy * price
                fee = trade_value * TX_FEE_RATE
                
                total_cost = trade_value + fee
                # We do not strictly subtract cash here yet, just propose.
                proposed_trades.append({
                    'Date': today_str, 'Ticker': ticker, 'Action': 'BUY', 
                    'Shares': shares_to_buy, 'Price': price, 'Value': trade_value, 'Fee': fee
                })

    return proposed_trades

def execute_trades(proposed_trades, state, ledger):
    if not proposed_trades:
        return state, ledger

    cash = state['cash_balance']
    new_ledger_rows = []
    
    # Sell first
    for trade in proposed_trades:
        if trade['Action'] == 'SELL':
            cash += (trade['Value'] - trade['Fee'])
            new_ledger_rows.append(trade)

    # Buy second, checking cash
    for trade in proposed_trades:
        if trade['Action'] == 'BUY':
            total_cost = trade['Value'] + trade['Fee']
            if cash >= total_cost:
                cash -= total_cost
                new_ledger_rows.append(trade)
    
    new_df = pd.DataFrame(new_ledger_rows)
    updated_ledger = pd.concat([ledger, new_df], ignore_index=True) if not ledger.empty else new_df
    updated_ledger.to_csv(LEDGER_FILE, index=False)
    
    state['cash_balance'] = cash
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)
        
    return state, updated_ledger
