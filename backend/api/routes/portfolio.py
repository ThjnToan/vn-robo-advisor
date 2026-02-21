from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from backend.core.portfolio import (
    load_market_data, load_portfolio, init_portfolio, 
    calculate_current_holdings, generate_trades, execute_trades,
    optimize_portfolio, optimize_black_litterman, LOOKBACK_DAYS
)

router = APIRouter()

class InitWalletRequest(BaseModel):
    initial_capital: float

class RebalanceRequest(BaseModel):
    target_tickers: List[str]
    max_weight: float
    max_sector_weight: float
    use_black_litterman: bool = False
    tactical_views: List[Dict[str, Any]] = []

class ExecuteTradesRequest(BaseModel):
    trades: List[Dict[str, Any]]

@router.get("/state")
async def get_portfolio_state():
    state, ledger = load_portfolio()
    if not state:
        return {"status": "uninitialized"}
        
    holdings = calculate_current_holdings(ledger)
    market_db = load_market_data()
    latest_date = market_db.index.max()
    
    current_nav = state['cash_balance']
    holdings_values = {}
    
    for ticker, shares in holdings.items():
        if ticker in market_db.columns:
            val = shares * market_db.loc[latest_date, ticker]
            holdings_values[ticker] = {
                "shares": shares,
                "current_value": val
            }
            current_nav += val
            
    return {
        "status": "active",
        "cash_balance": state['cash_balance'],
        "net_asset_value": current_nav,
        "initial_capital": state['initial_capital'],
        "roi_pct": ((current_nav / state['initial_capital']) - 1) * 100,
        "holdings": holdings_values,
        "ledger": ledger.to_dict(orient="records") if not ledger.empty else []
    }

@router.post("/init")
async def initialize_wallet(request: InitWalletRequest):
    state = init_portfolio(request.initial_capital)
    return {"status": "success", "state": state}

@router.post("/reset")
async def reset_portfolio():
    from backend.core.portfolio import STATE_FILE, LEDGER_FILE
    import os
    
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
    if os.path.exists(LEDGER_FILE):
        os.remove(LEDGER_FILE)
        
    return {"status": "success", "message": "Portfolio completely reset."}

@router.post("/optimize")
async def get_optimal_weights(request: RebalanceRequest):
    market_db = load_market_data()
    if market_db.empty:
        raise HTTPException(status_code=500, detail="Market data not found")
        
    valid_tickers = [t for t in request.target_tickers if t in market_db.columns]
    if len(valid_tickers) < 2:
        raise HTTPException(status_code=400, detail="Not enough valid historical data for selected tickers.")
        
    latest_date = market_db.index.max()
    train_data = market_db[valid_tickers].loc[:latest_date].tail(LOOKBACK_DAYS)
    
    if request.use_black_litterman and request.tactical_views:
        target_w = optimize_black_litterman(
            train_data, valid_tickers, request.max_weight, 
            request.tactical_views, request.max_sector_weight
        )
    else:
        target_w = optimize_portfolio(
            train_data, valid_tickers, request.max_weight, request.max_sector_weight
        )
        
    state, ledger = load_portfolio()
    if state is None:
        raise HTTPException(status_code=400, detail="Wallet not initialized")
        
    proposed_trades = generate_trades(market_db, state, ledger, target_w)
    
    return {
        "target_weights": target_w,
        "proposed_trades": proposed_trades
    }

@router.post("/execute")
async def execute_target_trades(request: ExecuteTradesRequest):
    state, ledger = load_portfolio()
    if state is None:
        raise HTTPException(status_code=400, detail="Wallet not initialized")
        
    state, updated_ledger = execute_trades(request.trades, state, ledger)
    
    return {
        "status": "success",
        "cash_balance": state['cash_balance'],
        "trades_executed": len(request.trades)
    }
