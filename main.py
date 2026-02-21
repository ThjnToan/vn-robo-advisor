import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from vnstock import Vnstock

# --- Configuration ---
TICKERS = ["FPT", "HPG", "VCB", "VHM", "MWG", "REE", "MBB", "VNM"]
END_DATE = datetime.today().strftime('%Y-%m-%d')
START_DATE = (datetime.today() - timedelta(days=4*365)).strftime('%Y-%m-%d') # Fetch 4 years so we have 1 year of training data and 3 years of simulation
INITIAL_CAPITAL = 1_000_000_000  # 1 Billion VND
MAX_WEIGHT = 0.3  # Max 30% per stock
LOOKBACK_DAYS = 252 # 1 year trading days
REBALANCE_FREQ = 21 # 1 month trading days

def fetch_data(tickers, start_date, end_date):
    print(f"Fetching stock data from {start_date} to {end_date}...")
    combined_data = pd.DataFrame()
    for ticker in tickers:
        try:
            stock = Vnstock().stock(symbol=ticker, source='VCI')
            df = stock.quote.history(start=start_date, end=end_date)
            if df is not None and not df.empty:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                df.sort_index(inplace=True)
                combined_data[ticker] = df['close']
            else:
                print(f"Warning: No data found for {ticker}")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            
    # Forward fill missing data points
    combined_data.ffill(inplace=True)
    combined_data.dropna(inplace=True) 
    return combined_data

def fetch_benchmark(start_date, end_date):
    print(f"Fetching VN-INDEX benchmark from {start_date} to {end_date}...")
    try:
        stock = Vnstock().stock(symbol='VNINDEX', source='VCI')
        df = stock.quote.history(start=start_date, end=end_date)
        if df is not None and not df.empty:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)
            return df['close']
    except Exception as e:
        print(f"Error fetching benchmark: {e}")
    return None

def optimize_portfolio(prices_df):
    mu = expected_returns.mean_historical_return(prices_df, frequency=252)
    S = risk_models.sample_cov(prices_df, frequency=252)
    
    # constraint: maximum weight of any single asset is 30%
    ef = EfficientFrontier(mu, S, weight_bounds=(0, MAX_WEIGHT))
    
    try:
        raw_weights = ef.max_sharpe()
    except Exception as e:
        ef = EfficientFrontier(mu, S, weight_bounds=(0, MAX_WEIGHT))
        raw_weights = ef.min_volatility()
        
    return ef.clean_weights()

def simulate_performance(prices_df, benchmark_series, initial_capital):
    print("\nSimulating historical dynamic performance (Monthly Rebalancing)...")
    
    daily_returns = prices_df.pct_change().dropna()
    benchmark_returns = benchmark_series.pct_change().dropna()
    
    common_dates = daily_returns.index.intersection(benchmark_returns.index)
    daily_returns = daily_returns.loc[common_dates]
    benchmark_returns = benchmark_returns.loc[common_dates]
    
    if len(common_dates) <= LOOKBACK_DAYS:
        print("Not enough data to simulate based on lookback days.")
        return None, None, None
        
    portfolio_daily_returns = pd.Series(index=common_dates, dtype=float)
    current_weights = np.repeat(1.0 / len(prices_df.columns), len(prices_df.columns))
    final_optimal_weights = {}

    print(f"Total trading days: {len(common_dates)}. Initial lookback: {LOOKBACK_DAYS} days.")
    
    for i in range(LOOKBACK_DAYS, len(common_dates)):
        # Rebalance at the start of every iteration block
        if (i - LOOKBACK_DAYS) % REBALANCE_FREQ == 0:
            start_idx = i - LOOKBACK_DAYS
            train_prices = prices_df.loc[common_dates[start_idx:i]]
            try:
                opt_weights_dict = optimize_portfolio(train_prices)
                current_weights = np.array([opt_weights_dict.get(ticker, 0) for ticker in daily_returns.columns])
                final_optimal_weights = opt_weights_dict
            except Exception as e:
                pass # reuse previous weights if optimization fails
        
        # Apply weights to today's return
        today_return = np.dot(current_weights, daily_returns.iloc[i])
        portfolio_daily_returns.iloc[i] = today_return
        
    # Drop the unused lookback period
    portfolio_daily_returns = portfolio_daily_returns.dropna()
    
    opt_cumulative_returns = (1 + portfolio_daily_returns).cumprod()
    opt_portfolio_value = initial_capital * opt_cumulative_returns
    
    # Calculate benchmark performance over the EXACT same time frame
    bench_returns_matched = benchmark_returns.loc[portfolio_daily_returns.index]
    bench_cumulative_returns = (1 + bench_returns_matched).cumprod()
    bench_portfolio_value = initial_capital * bench_cumulative_returns

    return opt_portfolio_value, bench_portfolio_value, final_optimal_weights

def generate_visualizations(opt_value, bench_value, final_weights):
    print("Generating visualizations...")
    
    plt.figure(figsize=(12, 6))
    plt.plot(opt_value.index, opt_value, label='Dynamic Portfolio (Max 30% per stock)', color='green')
    plt.plot(bench_value.index, bench_value, label='VN-Index Benchmark', color='blue')
    
    plt.title('Robo-Advisor Simulation: Monthly Rebalancing Performance', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value (VND)')
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x/1e9), ',')))
    plt.ylabel('Portfolio Value (Billion VND)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('portfolio_performance.png')
    plt.close()
    
    filtered_weights = {k: v for k, v in final_weights.items() if v > 0.01}
    labels = list(filtered_weights.keys())
    sizes = list(filtered_weights.values())
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title('Latest Portfolio Allocation (Max 30%)', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('portfolio_allocation.png')
    plt.close()
    print("Saved charts.")

def main():
    print(f"Starting Dynamic Robo-Advisor Simulation")
    print(f"Rules: Rebalance every {REBALANCE_FREQ} days, lookback window {LOOKBACK_DAYS} days, max weight {MAX_WEIGHT:.0%} per stock.\n")
    
    prices_df = fetch_data(TICKERS, START_DATE, END_DATE)
    benchmark_series = fetch_benchmark(START_DATE, END_DATE)
    
    if prices_df.empty or benchmark_series is None:
        print("Failed to fetch data. Exiting.")
        return

    opt_val, bench_val, final_weights = simulate_performance(prices_df, benchmark_series, INITIAL_CAPITAL)
    
    if opt_val is None:
        return
        
    print("\n--- Latest Optimal Allocations (Last Rebalance) ---")
    for ticker, weight in final_weights.items():
        if weight > 0.001:
             print(f"{ticker}: {weight:.2%}")

    final_nav = opt_val.iloc[-1]
    total_return = (final_nav - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    final_bench_nav = bench_val.iloc[-1]
    bench_return = (final_bench_nav - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    print(f"\n--- Final Results ---")
    print(f"Ending Portfolio Value: {final_nav:,.0f} VND (Return: {total_return:.2%})")
    print(f"VN-Index Benchmark Value: {final_bench_nav:,.0f} VND (Return: {bench_return:.2%})")
    
    generate_visualizations(opt_val, bench_val, final_weights)
    print("\nSimulation complete!")

if __name__ == "__main__":
    main()
