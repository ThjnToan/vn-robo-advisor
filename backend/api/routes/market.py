from fastapi import APIRouter, HTTPException
import pandas as pd
from typing import List

from backend.core.portfolio import load_market_data

router = APIRouter()

@router.get("/historical/{ticker}")
async def get_historical_data(ticker: str, days: int = 252):
    market_db = load_market_data()
    
    if market_db.empty or ticker not in market_db.columns:
        raise HTTPException(status_code=404, detail="Ticker not found in local database")
        
    prices = market_db[ticker].dropna().tail(days)
    
    return {
        "ticker": ticker,
        "dates": prices.index.strftime("%Y-%m-%d").tolist(),
        "prices": prices.tolist()
    }

@router.get("/screener")
async def get_screener_data(tickers: str = None):
    market_db = load_market_data()
    if market_db.empty:
        raise HTTPException(status_code=500, detail="Data unavailable")
        
    if tickers:
        ticker_list = [t.strip() for t in tickers.split(',')]
        screen_tickers = [t for t in ticker_list if t in market_db.columns]
    else:
        screen_tickers = market_db.columns.tolist()
        
    if len(screen_tickers) < 2:
        return []
        
    prices_screen = market_db[screen_tickers].tail(252)
    returns_screen = prices_screen.pct_change().dropna()
    
    from backend.core.portfolio import SECTOR_MAPPER
    
    screen_rows = []
    for t in screen_tickers:
        if prices_screen[t].empty: continue
        
        try:
            ret_1y = (prices_screen[t].iloc[-1] / prices_screen[t].iloc[0] - 1) * 100
            ret_3m = (prices_screen[t].iloc[-1] / prices_screen[t].iloc[-63] - 1) * 100 if len(prices_screen) >= 63 else 0
            ann_ret = returns_screen[t].mean() * 252 * 100
            ann_vol = returns_screen[t].std() * (252 ** 0.5) * 100
            sharpe = (ann_ret - 4.5) / ann_vol if ann_vol > 0 else 0
            sector = SECTOR_MAPPER.get(t, "Other")
            
            screen_rows.append({
                "ticker": t, 
                "sector": sector,
                "sharpe_1y": round(sharpe, 2),
                "return_3m_pct": round(ret_3m, 1),
                "return_1y_pct": round(ret_1y, 1),
                "volatility_pct": round(ann_vol, 1),
                "grade": "Top Pick" if sharpe > 1.5 and ret_3m > 5 else ("Good" if sharpe > 0.8 else "Weak")
            })
        except Exception:
            continue
            
    return screen_rows
