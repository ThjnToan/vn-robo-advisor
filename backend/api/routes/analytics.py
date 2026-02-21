from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple
import datetime

from backend.core.portfolio import load_market_data, load_portfolio, calculate_current_holdings
from backend.core.analytics import run_monte_carlo_var, compute_efficient_frontier, compute_equity_curve

router = APIRouter()

@router.get("/equity-curve")
async def get_equity_curve():
    market_db = load_market_data()
    state, ledger = load_portfolio()

    if state is None or market_db.empty:
        raise HTTPException(status_code=400, detail="Portfolio not initialized or market data unavailable")

    # Multiply by 1000 to convert vnstock shorthand thousands to true VND
    market_db_vnd = market_db * 1000

    result = compute_equity_curve(market_db_vnd, ledger, state["initial_capital"])
    if not result:
        return {"status": "no_data", "message": "No trade history to replay yet."}

    return result



class EFRequest(BaseModel):
    target_tickers: List[str]

@router.get("/monte-carlo")
async def get_monte_carlo_var():
    market_db = load_market_data()
    state, ledger = load_portfolio()
    
    if state is None or market_db.empty:
        raise HTTPException(status_code=400, detail="Data unavailable")
        
    holdings = calculate_current_holdings(ledger)
    cash = state['cash_balance']
    
    if not holdings:
        return {"status": "no_holdings"}
        
    results = run_monte_carlo_var(market_db, holdings, cash)
    
    if not results:
        raise HTTPException(status_code=500, detail="Simulation failed")
        
    return results

@router.post("/efficient-frontier")
async def get_efficient_frontier(request: EFRequest):
    market_db = load_market_data()
    if market_db.empty or len(request.target_tickers) < 2:
        raise HTTPException(status_code=400, detail="Not enough data")
        
    latest_date = market_db.index.max()
    latest_date_str = latest_date.strftime("%Y-%m-%d")
    
    valid_ef_tickers = [t for t in request.target_tickers if t in market_db.columns]
    
    if len(valid_ef_tickers) < 2:
        raise HTTPException(status_code=400, detail="Not enough valid tickers")
        
    ef_data = compute_efficient_frontier(market_db, tuple(sorted(valid_ef_tickers)), latest_date_str)
    
    if not ef_data:
        raise HTTPException(status_code=500, detail="Efficient frontier calculation failed")
        
    return ef_data
