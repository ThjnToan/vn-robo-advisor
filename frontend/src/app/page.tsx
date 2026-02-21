"use client";

import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  BarChart2, TrendingUp, Bot, RefreshCw, Wallet,
  AlertTriangle, CheckCircle, ChevronDown, ChevronUp, X
} from "lucide-react";
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
  Legend, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  LineChart, Line, ScatterChart, Scatter, ZAxis
} from "recharts";
import {
  fetchPortfolioState, initWallet, optimizePortfolio,
  executeTrades, fetchMonteCarlo, fetchScreenerData, fetchEquityCurve, fetchEfficientFrontier, resetPortfolio, getSessionId,
  type PortfolioState, type ProposedTrade, type ScreenerRow, type MonteCarloResult, type EquityCurveData, type EfficientFrontierData
} from "@/lib/api";
import { DEFAULT_TICKERS, PRESET_UNIVERSES, formatVnd, formatPct } from "@/lib/constants";

type Tab = "holdings" | "performance" | "terminal";

const PIE_COLORS = ["#38bdf8", "#34d399", "#818cf8", "#fbbf24", "#f87171", "#06b6d4", "#f97316", "#a78bfa", "#2dd4bf", "#fb923c"];

// ─── Subcomponents ────────────────────────────────────────────────────────────

function MetricCard({ label, value, sub, color = "default", delayClass = "" }: {
  label: string; value: string; sub?: string; color?: "green" | "red" | "default"; delayClass?: string;
}) {
  const valueClass = color === "green" ? "text-emerald-400" : color === "red" ? "text-red-400" : "text-white";
  return (
    <div className={`glass-card p-6 flex flex-col gap-1 animate-fade-in-up ${delayClass}`}>
      <p className="text-xs font-bold uppercase tracking-widest text-slate-400">{label}</p>
      <p className={`metric-value mt-1 ${valueClass}`}>{value}</p>
      {sub && <p className="text-xs text-slate-500 mt-2 font-medium">{sub}</p>}
    </div>
  );
}

function SectionHeader({ title, icon }: { title: string; icon: React.ReactNode }) {
  return (
    <div className="flex items-center gap-3 mb-5">
      <div className="p-2.5 rounded-xl bg-sky-500/10 text-sky-400 shadow-[inset_0_1px_0_rgba(255,255,255,0.1)]">{icon}</div>
      <h2 className="text-lg font-bold text-white tracking-tight">{title}</h2>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function Home() {
  const [tab, setTab] = useState<Tab>("holdings");
  const [initCapital, setInitCapital] = useState("1000000000");
  const [showResetModal, setShowResetModal] = useState(false);

  const { data: portfolio = null, isLoading: loading, refetch: refresh } = useQuery({
    queryKey: ["portfolio"],
    queryFn: fetchPortfolioState,
    refetchInterval: 300000,
  });

  // Sidebar state
  const [selectedTickers, setSelectedTickers] = useState<string[]>(["HPG", "VCB", "VHM", "REE", "MBB", "TCB"]);
  const [maxWeight, setMaxWeight] = useState(0.30);
  const [maxSectorWeight, setMaxSectorWeight] = useState(1.0);

  // Terminal state
  const [proposedTrades, setProposedTrades] = useState<ProposedTrade[] | null>(null);
  const [targetWeights, setTargetWeights] = useState<Record<string, number>>({});
  const [efData, setEfData] = useState<EfficientFrontierData | null>(null);
  const [optimizing, setOptimizing] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [useBL, setUseBL] = useState(false);
  const [tacticalViews, setTacticalViews] = useState<Record<string, string>>({});

  // Analytics state
  const [mcResult, setMcResult] = useState<MonteCarloResult | null>(null);
  const [mcLoading, setMcLoading] = useState(false);
  const [screenerData, setScreenerData] = useState<ScreenerRow[]>([]);
  const [screenerLoading, setScreenerLoading] = useState(false);
  const [equityCurve, setEquityCurve] = useState<EquityCurveData | null>(null);
  const [ecLoading, setEcLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);

  useEffect(() => {
    setSessionId(getSessionId());
  }, []);



  async function handleInit() {
    const cap = parseFloat(initCapital.replace(/,/g, ""));
    if (isNaN(cap) || cap <= 0) return;
    await initWallet(cap);
    await refresh();
  }

  async function handleOptimize() {
    if (!selectedTickers.length) return;
    setOptimizing(true);
    setProposedTrades(null);
    setEfData(null);
    try {
      const formattedViews = Object.entries(tacticalViews)
        .filter(([_, val]) => val !== "")
        .map(([t, val]) => ({ "Ticker": t, "Expected Return (%)": parseFloat(val) }));

      const [optResult, efResult] = await Promise.all([
        optimizePortfolio({
          target_tickers: selectedTickers,
          max_weight: maxWeight,
          max_sector_weight: maxSectorWeight,
          use_black_litterman: useBL,
          tactical_views: formattedViews,
        }),
        fetchEfficientFrontier(selectedTickers).catch(() => null)
      ]);
      setProposedTrades(optResult.proposed_trades);
      setTargetWeights(optResult.target_weights);
      if (efResult) setEfData(efResult);
    } catch (e: unknown) {
      alert(`Optimization failed: ${e instanceof Error ? e.message : e}`);
    }
    setOptimizing(false);
  }

  async function handleExecute() {
    if (!proposedTrades) return;
    setExecuting(true);
    try {
      await executeTrades(proposedTrades);
      setProposedTrades(null);
      await refresh();
    } catch { alert("Trade execution failed."); }
    setExecuting(false);
  }

  async function handleReset() {
    try {
      await resetPortfolio();
      await refresh();
      setShowResetModal(false);
    } catch (e: unknown) {
      alert(`Reset failed: ${e instanceof Error ? e.message : e}`);
    }
  }

  async function handleRunMonteCarlo() {
    setMcLoading(true);
    try { setMcResult(await fetchMonteCarlo()); }
    catch { alert("Monte Carlo simulation failed."); }
    setMcLoading(false);
  }

  async function handleRunScreener() {
    setScreenerLoading(true);
    try { setScreenerData(await fetchScreenerData(selectedTickers)); }
    catch { alert("Screener failed."); }
    setScreenerLoading(false);
  }

  async function handleLoadPerformance() {
    setEcLoading(true);
    try { setEquityCurve(await fetchEquityCurve()); }
    catch { alert("Could not load performance data. Is the backend running?"); }
    setEcLoading(false);
  }

  // ─── Render ───────────────────────────────────────────────────────────────

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="text-4xl mb-4 animate-pulse">📈</div>
          <p className="text-slate-400">Loading Robo-Advisor...</p>
        </div>
      </div>
    );
  }

  const isUninitialized = !portfolio || portfolio.status === "uninitialized";
  const holdings = portfolio?.holdings ?? {};
  const holdingEntries = Object.entries(holdings);
  const nav = portfolio?.net_asset_value ?? 0;
  const cash = portfolio?.cash_balance ?? 0;
  const roi = portfolio?.roi_pct ?? 0;
  const initialCapital = portfolio?.initial_capital ?? 0;

  // Pie chart data
  const pieData = [
    ...holdingEntries.map(([ticker, h], i) => ({
      name: ticker, value: h.current_value, color: PIE_COLORS[i % PIE_COLORS.length]
    })),
    { name: "Cash", value: cash, color: "#475569" }
  ];

  // Drift data (if we have target weights from the last optimization)
  const driftData = Object.keys(targetWeights)
    .filter(t => t in holdings)
    .map(t => {
      const currentPct = nav > 0 ? (holdings[t]?.current_value ?? 0) / nav * 100 : 0;
      const targetPct = targetWeights[t] * 100;
      return { ticker: t, current: parseFloat(currentPct.toFixed(1)), target: parseFloat(targetPct.toFixed(1)) };
    });

  return (
    <div className="flex h-screen overflow-hidden">

      {/* ── Sidebar ── */}
      <aside className="w-72 flex-shrink-0 border-r border-slate-800 bg-[#0a0f1e] flex flex-col overflow-y-auto">
        <div className="p-5 border-b border-slate-800">
          <h1 className="text-lg font-extrabold gradient-text">🇻🇳 Robo-Advisor</h1>
          <p className="text-xs text-slate-500 mt-1">Markowitz MVO Engine</p>
          {sessionId && (
            <div className="mt-3 flex items-center gap-2 bg-slate-800/50 p-2 text-xs rounded border border-slate-700">
              <div className="h-1.5 w-1.5 rounded-full bg-emerald-500"></div>
              <p className="font-mono text-slate-300">Guest: {sessionId.split('-')[0]}</p>
            </div>
          )}
        </div>

        {/* Wallet Section */}
        <div className="p-4 border-b border-slate-800">
          {isUninitialized ? (
            <div>
              <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-3">Initialize Wallet</p>
              <input
                type="text"
                className="input-field mb-3"
                placeholder="Initial Capital (VND)"
                value={initCapital}
                onChange={e => setInitCapital(e.target.value)}
              />
              <button className="btn btn-primary w-full justify-center" onClick={handleInit}>
                <Wallet size={16} /> Initialize Wallet
              </button>
            </div>
          ) : (
            <div>
              <div className="flex items-center gap-2 mb-2">
                <span className="relative flex h-2 w-2"><span className="pulse-green relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span></span>
                <p className="text-xs font-semibold text-emerald-400">Wallet Active</p>
              </div>
              <p className="text-xl font-bold text-white">{formatVnd(nav)}</p>
              <p className={`text-xs mt-1 ${roi >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                {formatPct(roi)} all-time return
              </p>
              <div className="mt-4 pt-4 border-t border-slate-700/50">
                <button className="btn btn-ghost text-red-400 hover:text-red-300 hover:bg-red-400/10 w-full justify-center text-xs py-1.5" onClick={() => setShowResetModal(true)}>
                  <AlertTriangle size={14} className="mr-1" /> Reset Portfolio
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Constraints */}
        <div className="p-4 border-b border-slate-800">
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-4">Constraints</p>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-400">Max per stock</span>
                <span className="text-sky-400 font-bold">{Math.round(maxWeight * 100)}%</span>
              </div>
              <input type="range" min={10} max={100} step={5} value={maxWeight * 100}
                onChange={e => setMaxWeight(parseInt(e.target.value) / 100)}
                className="w-full accent-sky-400 h-1 rounded cursor-pointer" />
            </div>
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-400">Max per sector</span>
                <span className="text-sky-400 font-bold">{Math.round(maxSectorWeight * 100)}%</span>
              </div>
              <input type="range" min={Math.round(maxWeight * 100)} max={100} step={5}
                value={maxSectorWeight * 100}
                onChange={e => setMaxSectorWeight(parseInt(e.target.value) / 100)}
                className="w-full accent-sky-400 h-1 rounded cursor-pointer" />
            </div>
          </div>
        </div>

        {/* Universe Presets */}
        <div className="p-4 border-b border-slate-800">
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-3">Universe Presets</p>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(PRESET_UNIVERSES).map(([name, tickers]) => (
              <button key={name} className="btn btn-ghost text-xs py-2 px-3 justify-center"
                onClick={() => setSelectedTickers(tickers)}>
                {name}
              </button>
            ))}
            <button className="btn btn-ghost text-xs py-2 px-3 justify-center"
              onClick={() => setSelectedTickers([...DEFAULT_TICKERS])}>
              All VN100
            </button>
            <button className="btn btn-ghost text-xs py-2 px-3 justify-center text-red-400"
              onClick={() => setSelectedTickers([])}>
              Clear
            </button>
          </div>
        </div>

        {/* Ticker Selection */}
        <div className="p-4 flex-1">
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-3">
            Active Universe ({selectedTickers.length})
          </p>
          <div className="flex flex-wrap gap-1.5">
            {selectedTickers.map(t => (
              <span key={t} className="flex items-center gap-1 px-2 py-1 rounded-md bg-sky-500/10 text-sky-400 text-xs font-semibold cursor-pointer hover:bg-red-500/10 hover:text-red-400 transition-colors"
                onClick={() => setSelectedTickers(prev => prev.filter(x => x !== t))}>
                {t} <X size={10} />
              </span>
            ))}
            <select className="input-field text-xs mt-1" defaultValue=""
              onChange={e => { if (e.target.value && !selectedTickers.includes(e.target.value)) { setSelectedTickers(prev => [...prev, e.target.value!]); e.target.value = ""; } }}>
              <option value="" disabled>+ Add ticker</option>
              {DEFAULT_TICKERS.filter(t => !selectedTickers.includes(t)).map(t => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>
        </div>
      </aside>

      {/* ── Main Content ── */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top nav tabs */}
        <header className="flex items-center gap-2 px-6 py-3 border-b border-slate-800 bg-[#0a0f1e] flex-shrink-0">
          {([
            { id: "holdings", label: "Holdings", icon: <BarChart2 size={16} /> },
            { id: "performance", label: "Performance", icon: <TrendingUp size={16} /> },
            { id: "terminal", label: "Robo-Advisor", icon: <Bot size={16} /> },
          ] as { id: Tab; label: string; icon: React.ReactNode }[]).map(t => (
            <button key={t.id} className={`tab-btn ${tab === t.id ? "active" : ""}`}
              onClick={() => setTab(t.id as Tab)}>
              {t.icon} {t.label}
            </button>
          ))}
          <div className="flex-1" />
          <button className="btn btn-ghost text-xs py-2 px-3" onClick={() => refresh()}>
            <RefreshCw size={14} /> Refresh
          </button>
        </header>

        {/* Tab content */}
        <div className="flex-1 overflow-y-auto p-6">
          {isUninitialized && (
            <div className="flex flex-col items-center justify-center h-full text-center gap-4">
              <div className="text-6xl">👋</div>
              <h2 className="text-2xl font-bold text-white">Welcome to VN Robo-Advisor</h2>
              <p className="text-slate-400 max-w-sm">Initialize your wallet in the sidebar to start tracking your quantitative portfolio.</p>
            </div>
          )}

          {/* ── HOLDINGS TAB ── */}
          {!isUninitialized && tab === "holdings" && (
            <div className="space-y-6">
              {/* Metrics Row */}
              <div className="grid grid-cols-3 gap-5">
                <MetricCard label="Net Asset Value" value={formatVnd(nav)} sub={`Initialized at ${formatVnd(initialCapital)}`} />
                <MetricCard label="Available Cash" value={formatVnd(cash)} delayClass="delay-100" />
                <MetricCard label="All-Time Return" value={formatPct(roi)} sub={`${roi >= 0 ? "+" : ""}${formatVnd(nav - initialCapital)}`} color={roi >= 0 ? "green" : "red"} delayClass="delay-200" />
              </div>

              {holdingEntries.length === 0 ? (
                <div className="glass-card p-12 text-center">
                  <p className="text-slate-400">Your portfolio is empty. Run the Robo-Advisor to make your first investment!</p>
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-6">
                  {/* Positions Table */}
                  <div className="glass-card p-6 animate-fade-in-up delay-300">
                    <SectionHeader title="Current Positions" icon={<BarChart2 size={18} />} />
                    <table className="data-table">
                      <thead><tr><th>Ticker</th><th>Shares</th><th>Value</th><th>Weight</th></tr></thead>
                      <tbody>
                        {holdingEntries.map(([ticker, h]) => (
                          <tr key={ticker}>
                            <td><span className="font-bold text-sky-400">{ticker}</span></td>
                            <td className="text-slate-300">{h.shares.toLocaleString()}</td>
                            <td className="text-slate-200">{formatVnd(h.current_value)}</td>
                            <td><span className="badge badge-blue">{(h.current_value / nav * 100).toFixed(1)}%</span></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Pie Chart */}
                  <div className="glass-card p-6 flex flex-col animate-fade-in-up delay-400">
                    <SectionHeader title="Portfolio Allocation" icon={<BarChart2 size={18} />} />
                    <ResponsiveContainer width="100%" height={260}>
                      <PieChart>
                        <Pie data={pieData} cx="50%" cy="50%" innerRadius={60} outerRadius={100}
                          dataKey="value" nameKey="name" paddingAngle={2}
                          label={(props: any) => `${props.name} ${(props.percent * 100).toFixed(0)}%`}
                          labelLine={false}>
                          {pieData.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                        </Pie>
                        <Tooltip formatter={(v: any) => typeof v === 'number' ? formatVnd(v) : v} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* Drift Monitor */}
              {driftData.length > 0 && (
                <div className="glass-card p-6 animate-fade-in-up delay-400">
                  <SectionHeader title="Allocation Drift Monitor" icon={<AlertTriangle size={18} />} />
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={driftData} layout="vertical">
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis type="number" unit="%" tick={{ fill: "#64748b", fontSize: 11 }} />
                      <YAxis type="category" dataKey="ticker" tick={{ fill: "#94a3b8", fontSize: 11 }} width={40} />
                      <Tooltip formatter={(v: any) => `${v?.toFixed(1)}%`} contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }} />
                      <Bar dataKey="target" fill="#38bdf8" opacity={0.5} name="Target %" radius={[0, 4, 4, 0]} />
                      <Bar dataKey="current" fill="#34d399" name="Current %" radius={[0, 4, 4, 0]} />
                      <Legend />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Monte Carlo Risk */}
              <div className="glass-card p-6 animate-fade-in-up delay-200">
                <SectionHeader title="Monte Carlo Risk Analytics (VaR)" icon={<AlertTriangle size={18} />} />
                <p className="text-sm text-slate-400 mb-4">Simulate 10,000 correlated price paths over 30 days based on historical volatility.</p>
                <button className="btn btn-primary mb-4" onClick={handleRunMonteCarlo} disabled={mcLoading || holdingEntries.length === 0}>
                  {mcLoading ? "Simulating..." : "Run 10,000 Simulations"}
                </button>
                {mcResult && (
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4">
                        <p className="text-xs text-slate-400 mb-1">95% Value-at-Risk (30d)</p>
                        <p className="text-xl font-bold text-red-400">-{formatVnd(mcResult.var_95_loss)}</p>
                      </div>
                      <div className="bg-red-500/10 border border-red-500/20 rounded-xl p-4">
                        <p className="text-xs text-slate-400 mb-1">Expected Shortfall (CVaR)</p>
                        <p className="text-xl font-bold text-red-400">-{formatVnd(mcResult.cvar_95_loss)}</p>
                      </div>
                    </div>
                    <p className="text-xs text-slate-500">In 95 out of 100 scenarios, you will NOT lose more than your VaR in 30 days.</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── PERFORMANCE TAB ── */}
          {!isUninitialized && tab === "performance" && (
            <div className="space-y-6">

              {/* Load Button */}
              {!equityCurve && (
                <div className="glass-card p-8 text-center">
                  <div className="text-4xl mb-3">📈</div>
                  <h3 className="text-white font-bold mb-2">Portfolio Performance Analytics</h3>
                  <p className="text-slate-400 text-sm mb-5">Replay your full trading history against market data to compute your equity curve, Sharpe trajectory, and stress-test resilience.</p>
                  <button className="btn btn-primary" onClick={handleLoadPerformance} disabled={ecLoading}>
                    {ecLoading ? <><RefreshCw size={16} className="animate-spin" /> Computing...</> : "⚡ Load Performance Analytics"}
                  </button>
                </div>
              )}

              {/* Summary Stats */}
              {equityCurve && equityCurve.summary_stats && (
                <>
                  <div className="flex items-center justify-between">
                    <h2 className="text-lg font-bold text-white">Performance Summary</h2>
                    <button className="btn btn-ghost text-xs py-2 px-3" onClick={handleLoadPerformance} disabled={ecLoading}>
                      <RefreshCw size={14} className={ecLoading ? "animate-spin" : ""} /> Recalculate
                    </button>
                  </div>
                  <div className="grid grid-cols-5 gap-4">
                    <MetricCard label="Total Return" value={formatPct(equityCurve.summary_stats.total_return_pct)} color={equityCurve.summary_stats.total_return_pct >= 0 ? "green" : "red"} />
                    <MetricCard label="Ann. Return" value={formatPct(equityCurve.summary_stats.annualized_return_pct)} color={equityCurve.summary_stats.annualized_return_pct >= 0 ? "green" : "red"} delayClass="delay-100" />
                    <MetricCard label="Ann. Volatility" value={`${equityCurve.summary_stats.annualized_vol_pct.toFixed(1)}%`} delayClass="delay-200" />
                    <MetricCard label="Sharpe Ratio" value={equityCurve.summary_stats.sharpe_ratio.toFixed(2)} color={equityCurve.summary_stats.sharpe_ratio >= 1 ? "green" : equityCurve.summary_stats.sharpe_ratio >= 0 ? "default" : "red"} delayClass="delay-300" />
                    <MetricCard label="Max Drawdown" value={`${equityCurve.summary_stats.max_drawdown_pct.toFixed(1)}%`} color="red" delayClass="delay-400" />
                  </div>

                  {/* Equity Curve Chart */}
                  {equityCurve.portfolio_nav.length > 1 && (
                    <div className="glass-card p-6 animate-fade-in-up delay-200">
                      <SectionHeader title="Portfolio NAV vs. VN30 ETF Benchmark" icon={<TrendingUp size={18} />} />
                      <ResponsiveContainer width="100%" height={280}>
                        <LineChart data={equityCurve.portfolio_nav.map(p => {
                          const bench = equityCurve.benchmark_nav.find(b => b.date === p.date);
                          return { date: p.date, portfolio: p.nav, benchmark: bench?.nav ?? null };
                        })}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                          <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 10 }}
                            tickFormatter={d => d.slice(5)} interval={Math.floor(equityCurve.portfolio_nav.length / 6)} />
                          <YAxis tick={{ fill: "#64748b", fontSize: 11 }}
                            tickFormatter={v => `${(v / 1e9).toFixed(1)}B`} />
                          <Tooltip
                            formatter={(v: any, name: any) => [formatVnd(v), name === "portfolio" ? "Your Portfolio" : "VN30 ETF"]}
                            contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 12 }} />
                          <Line type="monotone" dataKey="portfolio" stroke="#38bdf8" strokeWidth={2} dot={false} name="portfolio" />
                          <Line type="monotone" dataKey="benchmark" stroke="#f97316" strokeWidth={1.5} dot={false} strokeDasharray="5 3" name="benchmark" />
                          <Legend formatter={(v) => v === "portfolio" ? "Your Portfolio" : "VN30 ETF Benchmark"} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {/* Rolling 60-Day Sharpe */}
                  {equityCurve.rolling_sharpe_portfolio.length > 1 && (
                    <div className="glass-card p-6 animate-fade-in-up delay-300">
                      <SectionHeader title="Rolling 60-Day Sharpe Ratio" icon={<TrendingUp size={18} />} />
                      <p className="text-xs text-slate-500 mb-3">A Sharpe above 1.0 indicates risk-adjusted outperformance. Dashed orange = VN30 ETF benchmark.</p>
                      <ResponsiveContainer width="100%" height={200}>
                        <LineChart data={equityCurve.rolling_sharpe_portfolio.map(p => {
                          const bench = equityCurve.rolling_sharpe_benchmark.find(b => b.date === p.date);
                          return { date: p.date, portfolio: p.sharpe, benchmark: bench?.sharpe ?? null };
                        })}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                          <XAxis dataKey="date" tick={{ fill: "#64748b", fontSize: 10 }}
                            tickFormatter={d => d.slice(5)} interval={Math.floor(equityCurve.rolling_sharpe_portfolio.length / 6)} />
                          <YAxis tick={{ fill: "#64748b", fontSize: 11 }} />
                          <Tooltip formatter={(v: any, name: any) => [v?.toFixed(2), name === "portfolio" ? "Portfolio Sharpe" : "Benchmark Sharpe"]}
                            contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 12 }} />
                          {/* Reference line at y=1 */}
                          <Line type="monotone" dataKey="portfolio" stroke="#34d399" strokeWidth={2} dot={false} name="portfolio" />
                          <Line type="monotone" dataKey="benchmark" stroke="#f97316" strokeWidth={1.5} dot={false} strokeDasharray="5 3" name="benchmark" />
                          <Legend formatter={(v) => v === "portfolio" ? "Portfolio (60d Sharpe)" : "VN30 ETF (60d Sharpe)"} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {/* Stress Tests */}
                  {equityCurve.stress_results.length > 0 && (
                    <div className="glass-card p-6 animate-fade-in-up delay-400">
                      <SectionHeader title="Historical Stress Test" icon={<AlertTriangle size={18} />} />
                      <p className="text-xs text-slate-500 mb-4">Simulates the impact of past market crises on your <em>current</em> portfolio weights.</p>
                      <table className="data-table">
                        <thead><tr><th>Scenario</th><th>Period</th><th>Return</th><th>Est. P&amp;L</th><th>Severity</th></tr></thead>
                        <tbody>
                          {equityCurve.stress_results.map((s, i) => (
                            <tr key={i}>
                              <td className="font-semibold text-slate-200">{s.scenario}</td>
                              <td className="text-slate-400 text-xs">{s.period}</td>
                              <td className={s.return_pct >= 0 ? "text-emerald-400 font-bold" : "text-red-400 font-bold"}>{formatPct(s.return_pct)}</td>
                              <td className={s.pnl >= 0 ? "text-emerald-400" : "text-red-400"}>{formatVnd(s.pnl)}</td>
                              <td><span className={`badge ${s.severity === "Severe" ? "badge-red" : s.severity === "Moderate" ? "badge-yellow" : "badge-green"}`}>{s.severity}</span></td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </>
              )}

              {/* Trade Ledger */}
              <div className="glass-card p-6 animate-fade-in-up delay-200">
                <SectionHeader title="Trade History" icon={<TrendingUp size={18} />} />
                {portfolio?.ledger && portfolio.ledger.length > 0 ? (
                  <table className="data-table">
                    <thead>
                      <tr><th>Date</th><th>Ticker</th><th>Action</th><th>Shares</th><th>Price</th><th>Value</th><th>Fee</th></tr>
                    </thead>
                    <tbody>
                      {[...portfolio.ledger].reverse().map((r, i) => (
                        <tr key={i}>
                          <td className="text-slate-400">{r.Date}</td>
                          <td><span className="font-bold text-sky-400">{r.Ticker}</span></td>
                          <td><span className={`badge ${r.Action === "BUY" ? "badge-green" : "badge-red"}`}>{r.Action}</span></td>
                          <td>{r.Shares.toLocaleString()}</td>
                          <td>{Number(r.Price).toLocaleString()}</td>
                          <td>{formatVnd(Number(r.Value))}</td>
                          <td className="text-slate-500">{formatVnd(Number(r.Fee))}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : (
                  <p className="text-slate-400">No trades yet. Run the Robo-Advisor to start trading.</p>
                )}
              </div>
            </div>
          )}

          {/* ── ROBO-ADVISOR TERMINAL TAB ── */}
          {!isUninitialized && tab === "terminal" && (
            <div className="space-y-6">
              {/* Screener */}
              <div className="glass-card p-6 animate-fade-in-up">
                <SectionHeader title="VN100 Stock Screener" icon={<Bot size={18} />} />
                <p className="text-sm text-slate-400 mb-4">Pre-filter your selected universe by Sharpe ratio and momentum before optimization.</p>
                <button className="btn btn-primary mb-4" onClick={handleRunScreener} disabled={screenerLoading || !selectedTickers.length}>
                  {screenerLoading ? "Screening..." : `Screen ${selectedTickers.length} Stocks`}
                </button>
                {screenerData.length > 0 && (
                  <table className="data-table">
                    <thead>
                      <tr><th>Ticker</th><th>Sector</th><th>Sharpe 1Y</th><th>Return 3M</th><th>Return 1Y</th><th>Volatility</th><th>Grade</th></tr>
                    </thead>
                    <tbody>
                      {screenerData.sort((a, b) => b.sharpe_1y - a.sharpe_1y).map(r => (
                        <tr key={r.ticker}>
                          <td><span className="font-bold text-sky-400">{r.ticker}</span></td>
                          <td className="text-slate-400">{r.sector}</td>
                          <td className={r.sharpe_1y > 1 ? "text-emerald-400 font-bold" : "text-slate-300"}>{r.sharpe_1y.toFixed(2)}</td>
                          <td className={r.return_3m_pct >= 0 ? "text-emerald-400" : "text-red-400"}>{formatPct(r.return_3m_pct)}</td>
                          <td className={r.return_1y_pct >= 0 ? "text-emerald-400" : "text-red-400"}>{formatPct(r.return_1y_pct)}</td>
                          <td className="text-slate-400">{r.volatility_pct.toFixed(1)}%</td>
                          <td>
                            <span className={`badge ${r.grade === "Top Pick" ? "badge-green" : r.grade === "Good" ? "badge-blue" : "badge-yellow"}`}>
                              {r.grade === "Top Pick" ? "⭐ Top Pick" : r.grade === "Good" ? "✅ Good" : "⚠️ Weak"}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>

              {/* Optimization Engine */}
              <div className="glass-card p-6 animate-fade-in-up delay-100">
                <SectionHeader title="Rebalance Engine" icon={<Bot size={18} />} />
                <p className="text-sm text-slate-400 mb-4">
                  Click optimize to calculate the mathematically optimal portfolio, then review and confirm the proposed trades.
                </p>
                <div className="flex items-center gap-4 mb-4">
                  <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                    <input type="checkbox" checked={useBL} onChange={e => setUseBL(e.target.checked)}
                      className="accent-sky-400 w-4 h-4 cursor-pointer" />
                    Enable Black-Litterman Model
                  </label>
                </div>

                {useBL && (
                  <div className="mb-6 p-4 border border-sky-500/30 bg-sky-500/5 rounded-lg">
                    <h4 className="font-bold text-sky-400 mb-2">Tactical Views (Expected Returns)</h4>
                    <p className="text-xs text-slate-400 mb-4">Enter your absolute expected annualized returns for specific stocks to override the market-implied prior.</p>
                    <div className="grid grid-cols-3 gap-3">
                      {selectedTickers.map(t => (
                        <div key={t} className="flex items-center justify-between bg-[#0f172a] p-2 rounded border border-slate-700">
                          <span className="font-bold text-slate-300">{t}</span>
                          <div className="flex items-center text-sm">
                            <input
                              type="number"
                              className="bg-transparent text-right w-16 outline-none text-sky-400"
                              placeholder="e.g. 15"
                              value={tacticalViews[t] || ""}
                              onChange={e => setTacticalViews({ ...tacticalViews, [t]: e.target.value })}
                            />
                            <span className="text-slate-500 ml-1">%</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                <button className="btn btn-primary" onClick={handleOptimize}
                  disabled={optimizing || !selectedTickers.length}>
                  {optimizing ? <><RefreshCw size={16} className="animate-spin" /> Crunching covariance...</> : "🚀 Run Rebalance Engine"}
                </button>

                {/* Proposed Trades */}
                {proposedTrades !== null && (
                  <div className="mt-6">
                    <div className="flex items-center justify-between mb-3">
                      <h3 className="font-bold text-white">
                        {proposedTrades.length > 0 ? `${proposedTrades.length} Proposed Trades — Review & Confirm` : "No trades needed. Portfolio is already optimal."}
                      </h3>
                      {proposedTrades.length > 0 && (
                        <div className="flex gap-2">
                          <button className="btn btn-success" onClick={handleExecute} disabled={executing}>
                            {executing ? "Executing..." : <><CheckCircle size={16} /> Confirm & Execute</>}
                          </button>
                          <button className="btn btn-danger" onClick={() => setProposedTrades(null)}>
                            <X size={16} /> Discard
                          </button>
                        </div>
                      )}
                    </div>
                    {proposedTrades.length > 0 && (
                      <table className="data-table">
                        <thead>
                          <tr><th>Action</th><th>Ticker</th><th>Shares</th><th>Price</th><th>Trade Value</th><th>Estimated Fee</th></tr>
                        </thead>
                        <tbody>
                          {proposedTrades.map((t, i) => (
                            <tr key={i}>
                              <td><span className={`badge ${t.Action === "BUY" ? "badge-green" : "badge-red"}`}>{t.Action}</span></td>
                              <td><span className="font-bold text-sky-400">{t.Ticker}</span></td>
                              <td>{t.Shares.toLocaleString()}</td>
                              <td>{Number(t.Price).toLocaleString()}</td>
                              <td>{formatVnd(Number(t.Value))}</td>
                              <td className="text-slate-500">{formatVnd(Number(t.Fee))}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    )}
                  </div>
                )}

                {/* Target Weights Chart */}
                {Object.keys(targetWeights).length > 0 && (
                  <div className="mt-6">
                    <h3 className="font-bold text-white mb-3">Optimal Weights (by Sharpe Maximization)</h3>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={Object.entries(targetWeights).filter(([, w]) => w > 0).map(([t, w]) => ({ ticker: t, weight: parseFloat((w * 100).toFixed(1)) }))}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="ticker" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                        <YAxis unit="%" tick={{ fill: "#64748b", fontSize: 11 }} />
                        <Tooltip formatter={(v: any) => `${v?.toFixed(1)}%`} contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }} />
                        <Bar dataKey="weight" fill="#38bdf8" radius={[4, 4, 0, 0]} name="Optimal Weight %" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Efficient Frontier Scatter Plot */}
                {efData && (
                  <div className="mt-6 border-t border-slate-800 pt-6">
                    <h3 className="font-bold text-white mb-2">Efficient Frontier vs. Random Portfolios</h3>
                    <p className="text-xs text-slate-500 mb-4">Blue dots represent 3,000 randomized portfolios. The green curve is the mathematically optimal Efficient Frontier.</p>
                    <ResponsiveContainer width="100%" height={350}>
                      <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis type="number" dataKey="vol" name="Volatility" unit="%" tick={{ fill: "#64748b", fontSize: 11 }} domain={['auto', 'auto']} />
                        <YAxis type="number" dataKey="ret" name="Return" unit="%" tick={{ fill: "#64748b", fontSize: 11 }} domain={['auto', 'auto']} />
                        <ZAxis type="number" dataKey="sharpe" range={[20, 20]} />
                        <Tooltip
                          cursor={{ strokeDasharray: '3 3' }}
                          contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 12 }}
                          formatter={(v: number | string | Array<number | string> | undefined) => {
                            if (typeof v === 'number') return [`${v.toFixed(2)}%`, "Return"];
                            return [`${v}`, "Return"];
                          }}
                        />
                        <Scatter name="Random Portfolios"
                          data={efData.rand_vols.map((v, i) => ({ vol: v * 100, ret: efData.rand_rets[i] * 100, sharpe: efData.rand_sharpes[i] }))}
                          fill="#38bdf8" opacity={0.3} />
                        <Scatter name="Efficient Frontier"
                          data={efData.frontier_vols.map((v, i) => ({ vol: v * 100, ret: efData.frontier_rets[i] * 100 }))}
                          line={{ stroke: '#34d399', strokeWidth: 3 }} shape="circle" fill="#10b981" />
                        <Legend />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Reset Confirmation Modal */}
      {showResetModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-[#0f172a] border border-slate-700 rounded-xl p-6 max-w-sm w-full shadow-2xl animate-fade-in-up">
            <h3 className="text-lg font-bold text-white flex items-center gap-2 mb-3">
              <AlertTriangle className="text-red-500" size={20} /> Reset Portfolio
            </h3>
            <p className="text-sm text-slate-400 mb-6 font-medium leading-relaxed">
              Are you sure you want to completely reset your portfolio? All active holdings and transaction history will be permanently deleted, and the wallet will be reset to its initial capital.
            </p>
            <div className="flex gap-3 justify-end">
              <button className="btn btn-ghost" onClick={() => setShowResetModal(false)}>Cancel</button>
              <button className="btn bg-red-500/10 text-red-500 border border-red-500/20 hover:bg-red-500 hover:text-white" onClick={handleReset}>
                Yes, Reset Portfolio
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
