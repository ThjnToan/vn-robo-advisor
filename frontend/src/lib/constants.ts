export const DEFAULT_TICKERS = [
    "ACB", "BCG", "BCM", "BID", "BMP", "BVH", "CII", "CMG", "CRE", "CTD",
    "CTG", "DBC", "DCM", "DGC", "DGW", "DIG", "DPM", "DXG", "EIB", "FCN",
    "FPT", "FRT", "GAS", "GEX", "GMD", "GVR", "HAG", "HAH", "HDB", "HDC",
    "HDG", "HHV", "HPG", "HSG", "HT1", "ITA", "KBC", "KDC", "KDH", "LPB",
    "MBB", "MSB", "MSN", "MWG", "NKG", "NLG", "NT2", "NVL", "OCB", "PAN",
    "PC1", "PDR", "PHR", "PLX", "PNJ", "POW", "PTB", "PVD", "PVT", "REE",
    "SAB", "SAM", "SBT", "SCS", "SHB", "SJS", "SSB", "SSI", "STB", "SZC",
    "TCB", "TCH", "TCM", "TPB", "VCB", "VCG", "VCI", "VHC", "VHM", "VIB",
    "VIC", "VIX", "VJC", "VND", "VNM", "VPB", "VPI", "VRE", "VSH",
];

export const SECTOR_COLORS: Record<string, string> = {
    "Financials": "#3b82f6",
    "Real Estate": "#f59e0b",
    "Materials": "#10b981",
    "Consumer": "#8b5cf6",
    "Industrials": "#06b6d4",
    "Utilities": "#f97316",
    "Technology": "#ec4899",
    "Other": "#6b7280",
};

export const PRESET_UNIVERSES: Record<string, string[]> = {
    "VN30 Blue Chips": ["VCB", "VHM", "HPG", "TCB", "MBB", "CTG", "BID", "FPT", "VIC", "VNM", "SAB", "MSN"],
    "Tech & Innovation": ["FPT", "CMG", "VND", "SSI", "VCI"],
    "Real Estate Recovery": ["VHM", "VIC", "NVL", "KDH", "NLG", "PDR", "DXG"],
    "Industrials & Logistics": ["GMD", "HAH", "REE", "VJC", "PVT"],
};

export function formatVnd(value: number): string {
    if (Math.abs(value) >= 1_000_000_000) {
        return `${(value / 1_000_000_000).toFixed(2)}B ₫`;
    }
    if (Math.abs(value) >= 1_000_000) {
        return `${(value / 1_000_000).toFixed(1)}M ₫`;
    }
    return `${value.toLocaleString("vi-VN")} ₫`;
}

export function formatPct(value: number, digits = 2): string {
    return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}%`;
}
