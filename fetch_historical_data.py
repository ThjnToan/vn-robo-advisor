import pandas as pd
from datetime import datetime, timedelta
from vnstock import Vnstock
import os

DEFAULT_TICKERS = [
    "FPT", "HPG", "VCB", "VHM", "MWG", "REE", "MBB", "VNM", "TCB", "SSI",
    "VIC", "VRE", "MSN", "GAS", "SAB", "CTG", "BID", "VJC", "PNJ", "VPB", "E1VFVN30"
]

def fetch_and_save():
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365) # 5 years just to be safe
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    combined_data = pd.DataFrame()
    print(f"Fetching data from {start_str} to {end_str}...")
    
    for ticker in DEFAULT_TICKERS:
        print(f"Fetching: {ticker}")
        df = None
        for source in ['VCI', 'TCBS', 'SSI', 'VND']:
            try:
                stock = Vnstock().stock(symbol=ticker, source=source)
                df = stock.quote.history(start=start_str, end=end_str)
                if df is not None and not df.empty:
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                    df.sort_index(inplace=True)
                    combined_data[ticker] = df['close']
                    print(f"  -> Success from {source}")
                    break
            except Exception as e:
                continue
        
        if df is None or df.empty:
            print(f"Error fetching {ticker} from all sources.")
            
    if not combined_data.empty:
        combined_data.ffill(inplace=True)
        combined_data.dropna(inplace=True)
        
        # Save to CSV
        output_file = os.path.join(os.path.dirname(__file__), "market_data.csv")
        combined_data.to_csv(output_file)
        print(f"Successfully saved market data to {output_file}")
    else:
        print("Failed to fetch any data.")

if __name__ == "__main__":
    fetch_and_save()
