const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export function getSessionId(): string {
  if (typeof window === "undefined") return "server-session";
  const SESSION_KEY = "guest_session_id";
  let sessionId = localStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, sessionId);
  }
  return sessionId;
}

function getHeaders(): HeadersInit {
  return {
    "x-session-id": getSessionId(),
  };
}

export interface Holding {
  shares: number;
  current_value: number;
}

export interface PortfolioState {
  status: string;
  cash_balance: number;
  net_asset_value: number;
  initial_capital: number;
  roi_pct: number;
  holdings: Record<string, Holding>;
  ledger: LedgerRow[];
}

export interface LedgerRow {
  Date: string;
  Ticker: string;
  Action: "BUY" | "SELL";
  Shares: number;
  Price: number;
  Value: number;
  Fee: number;
}

export interface ProposedTrade {
  Date: string;
  Ticker: string;
  Action: "BUY" | "SELL";
  Shares: number;
  Price: number;
  Value: number;
  Fee: number;
}

export interface OptimizeResponse {
  target_weights: Record<string, number>;
  proposed_trades: ProposedTrade[];
}

export interface ScreenerRow {
  ticker: string;
  sector: string;
  sharpe_1y: number;
  return_3m_pct: number;
  return_1y_pct: number;
  volatility_pct: number;
  grade: string;
}

export interface MonteCarloResult {
  final_values: number[];
  var_95_loss: number;
  cvar_95_loss: number;
  var_95_threshold: number;
  current_nav: number;
}

export interface EfficientFrontierData {
  rand_vols: number[];
  rand_rets: number[];
  rand_sharpes: number[];
  frontier_vols: number[];
  frontier_rets: number[];
}

export interface NavPoint { date: string; nav: number; }
export interface SharpePoint { date: string; sharpe: number; }
export interface StressScenario {
  scenario: string;
  period: string;
  return_pct: number;
  pnl: number;
  severity: "Mild" | "Moderate" | "Severe";
}
export interface EquityCurveSummary {
  total_return_pct: number;
  annualized_return_pct: number;
  annualized_vol_pct: number;
  sharpe_ratio: number;
  max_drawdown_pct: number;
  data_points: number;
}
export interface EquityCurveData {
  status?: string;
  portfolio_nav: NavPoint[];
  benchmark_nav: NavPoint[];
  rolling_sharpe_portfolio: SharpePoint[];
  rolling_sharpe_benchmark: SharpePoint[];
  stress_results: StressScenario[];
  summary_stats: EquityCurveSummary;
}


export async function fetchPortfolioState(): Promise<PortfolioState> {
  const res = await fetch(`${API_BASE}/api/portfolio/state`, {
    headers: getHeaders(),
  });
  if (!res.ok) throw new Error("Failed to fetch state");
  return res.json();
}

export async function resetPortfolio(): Promise<void> {
  const res = await fetch(`${API_BASE}/api/portfolio/reset`, {
    method: "POST",
    headers: getHeaders(),
  });
  if (!res.ok) throw new Error("Failed to reset portfolio");
}

export async function initWallet(initial_capital: number): Promise<void> {
  const res = await fetch(`${API_BASE}/api/portfolio/init`, {
    method: "POST",
    headers: { ...getHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ initial_capital }),
  });
  if (!res.ok) throw new Error("Failed to initialize wallet");
}

export async function optimizePortfolio(params: {
  target_tickers: string[];
  max_weight: number;
  max_sector_weight: number;
  use_black_litterman: boolean;
  tactical_views: { Ticker: string; "Expected Return (%)": number }[];
}): Promise<OptimizeResponse> {
  const res = await fetch(`${API_BASE}/api/portfolio/optimize`, {
    method: "POST",
    headers: { ...getHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail || "Optimization failed");
  }
  return res.json();
}

export async function executeTrades(trades: ProposedTrade[]): Promise<void> {
  const res = await fetch(`${API_BASE}/api/portfolio/execute`, {
    method: "POST",
    headers: { ...getHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ trades }),
  });
  if (!res.ok) throw new Error("Failed to execute trades");
}

export async function fetchScreenerData(tickers: string[]): Promise<ScreenerRow[]> {
  const res = await fetch(`${API_BASE}/api/market/screener?tickers=${tickers.join(",")}`, {
    headers: getHeaders(),
  });
  if (!res.ok) throw new Error("Failed to fetch screener data");
  return res.json();
}

export async function fetchMonteCarlo(): Promise<MonteCarloResult> {
  const res = await fetch(`${API_BASE}/api/analytics/monte-carlo`, {
    headers: getHeaders(),
  });
  if (!res.ok) throw new Error("Failed to fetch Monte Carlo data");
  return res.json();
}

export async function fetchEfficientFrontier(tickers: string[]): Promise<EfficientFrontierData> {
  const res = await fetch(`${API_BASE}/api/analytics/efficient-frontier`, {
    method: "POST",
    headers: { ...getHeaders(), "Content-Type": "application/json" },
    body: JSON.stringify({ target_tickers: tickers }),
  });
  if (!res.ok) throw new Error("Failed to compute efficient frontier");
  return res.json();
}

export async function fetchHistoricalPrices(ticker: string, days = 252): Promise<{ dates: string[]; prices: number[] }> {
  const res = await fetch(`${API_BASE}/api/market/historical/${ticker}?days=${days}`, {
    headers: getHeaders(),
  });
  if (!res.ok) throw new Error("Failed to fetch historical prices");
  return res.json();
}

export async function fetchEquityCurve(): Promise<EquityCurveData> {
  const res = await fetch(`${API_BASE}/api/analytics/equity-curve`, {
    headers: getHeaders(),
  });
  if (!res.ok) throw new Error("Failed to fetch equity curve data");
  return res.json();
}

