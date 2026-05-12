#!/usr/bin/env python3
"""
Portfolio Forecast Optimizer

Companion tool for advisor_scorecard monthly review.

Reads either:
  - reports/monthly_review/public_normalized_holdings_weights.csv
  - data/schwab_holdings.csv

Outputs scenario ranges, indicator summaries, constrained optimization candidates,
and charts using normalized weights only. No dollar values are written.
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception as exc:  # pragma: no cover
    yf = None

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None

import matplotlib.pyplot as plt

CASH_NAMES = {"CASH", "SWVXX", "SNVXX", "SNSXX", "SNOXX", "SGOV", "BIL", "SHV", "TFLO"}
TRADING_DAYS = 252


@dataclass
class Candidate:
    name: str
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float


def clean_symbol(symbol: object) -> str:
    if symbol is None or (isinstance(symbol, float) and math.isnan(symbol)):
        return ""
    return str(symbol).strip().upper()


def yahoo_symbol(symbol: str) -> str:
    sym = clean_symbol(symbol)
    # Yahoo uses dash for Berkshire-style symbols. Schwab may use BRK.B or BRK/B.
    return sym.replace("/", "-").replace(".", "-")


def is_cash_symbol(symbol: str) -> bool:
    return clean_symbol(symbol) in CASH_NAMES


def pct(x: float) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x * 100:.2f}%"


def safe_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def read_portfolio(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}

    if "portfolio_weight_pct" in cols_lower:
        sym_col = cols_lower.get("symbol")
        w_col = cols_lower["portfolio_weight_pct"]
        out = pd.DataFrame({
            "symbol": df[sym_col].map(clean_symbol),
            "display_name": df.get(cols_lower.get("display_name", ""), df[sym_col]).astype(str) if sym_col else "",
            "asset_class_name": df.get(cols_lower.get("asset_class_name", ""), "Unclassified"),
            "sector_clean": df.get(cols_lower.get("sector_clean", ""), "Unclassified"),
            "industry": df.get(cols_lower.get("industry", ""), ""),
            "weight": pd.to_numeric(df[w_col], errors="coerce").fillna(0) / 100.0,
        })
    elif "market_value" in cols_lower:
        sym_col = cols_lower.get("symbol")
        mv_col = cols_lower["market_value"]
        mv = pd.to_numeric(df[mv_col], errors="coerce").fillna(0)
        total = mv.sum()
        if total == 0:
            raise ValueError("market_value column sums to zero; cannot compute weights")
        out = pd.DataFrame({
            "symbol": df[sym_col].map(clean_symbol),
            "display_name": df.get(cols_lower.get("description", ""), df[sym_col]).astype(str),
            "asset_class_name": df.get(cols_lower.get("asset_class_name", ""), "Unclassified"),
            "sector_clean": df.get(cols_lower.get("sector_clean", ""), "Unclassified"),
            "industry": df.get(cols_lower.get("industry", ""), ""),
            "weight": mv / total,
        })
    else:
        raise ValueError(
            "Input must contain either portfolio_weight_pct or market_value. "
            "Use the monthly review public weights CSV or Schwab holdings CSV."
        )

    out = out[out["symbol"].astype(str).str.len() > 0].copy()
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce").fillna(0)
    out = out[out["weight"] > 0].copy()

    # Aggregate duplicate tickers across accounts.
    meta_cols = ["display_name", "asset_class_name", "sector_clean", "industry"]
    agg = out.groupby("symbol", as_index=False).agg({
        "weight": "sum",
        **{c: "first" for c in meta_cols},
    })
    agg["weight"] = agg["weight"] / agg["weight"].sum()
    return agg.sort_values("weight", ascending=False).reset_index(drop=True)


def download_prices(symbols: List[str], years: int, risk_free_rate: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if yf is None:
        raise RuntimeError("yfinance is not installed. Run: uv add yfinance")

    symbols = [clean_symbol(s) for s in symbols if clean_symbol(s)]
    price_status = []
    cash_symbols = [s for s in symbols if is_cash_symbol(s)]
    market_symbols = [s for s in symbols if not is_cash_symbol(s)]
    yahoo_map = {yahoo_symbol(s): s for s in market_symbols}

    prices = pd.DataFrame()
    if yahoo_map:
        tickers = list(yahoo_map.keys())
        raw = yf.download(
            tickers,
            period=f"{max(1, years + 1)}y",
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
        if raw is None or len(raw) == 0:
            raw_close = pd.DataFrame()
        elif isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                raw_close = raw["Close"].copy()
            else:
                raw_close = pd.DataFrame()
        else:
            raw_close = raw[["Close"]].copy() if "Close" in raw.columns else raw.copy()
            if len(tickers) == 1:
                raw_close.columns = [tickers[0]]

        for ysym, original in yahoo_map.items():
            if ysym in raw_close.columns:
                s = raw_close[ysym].dropna()
                if len(s) > 60:
                    prices[original] = s
                    price_status.append({"symbol": original, "yahoo_symbol": ysym, "status": "ok", "rows": len(s)})
                else:
                    price_status.append({"symbol": original, "yahoo_symbol": ysym, "status": "too_few_prices", "rows": len(s)})
            else:
                price_status.append({"symbol": original, "yahoo_symbol": ysym, "status": "missing", "rows": 0})

    # Model cash as a smooth daily risk-free return series aligned to the price index.
    if prices.empty:
        raise RuntimeError("No market prices downloaded. Check network/yfinance availability.")
    idx = prices.index
    daily_cash_return = (1.0 + risk_free_rate) ** (1.0 / TRADING_DAYS) - 1.0
    for sym in cash_symbols:
        prices[sym] = 100.0 * (1.0 + daily_cash_return) ** np.arange(len(idx))
        price_status.append({"symbol": sym, "yahoo_symbol": "cash_model", "status": "cash_model", "rows": len(idx)})

    prices = prices.sort_index().ffill().dropna(axis=1, how="all")
    return prices, pd.DataFrame(price_status)


def max_drawdown(series: pd.Series) -> float:
    if len(series) == 0:
        return np.nan
    running_max = series.cummax()
    dd = series / running_max - 1.0
    return float(dd.min())


def compute_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna(how="all")
    rows = []
    for sym in prices.columns:
        p = prices[sym].dropna()
        r = p.pct_change().dropna()
        if len(r) < 60:
            continue
        trailing_1y = p.iloc[-1] / p.iloc[max(0, len(p) - TRADING_DAYS)] - 1 if len(p) > 30 else np.nan
        trailing_6m = p.iloc[-1] / p.iloc[max(0, len(p) - TRADING_DAYS // 2)] - 1 if len(p) > 30 else np.nan
        vol = r.std() * np.sqrt(TRADING_DAYS)
        ann = (1 + r.mean()) ** TRADING_DAYS - 1
        sma200 = p.rolling(200).mean().iloc[-1] if len(p) >= 200 else np.nan
        dist_sma200 = p.iloc[-1] / sma200 - 1 if pd.notna(sma200) and sma200 != 0 else np.nan
        rows.append({
            "symbol": sym,
            "historical_return": ann,
            "volatility": vol,
            "trailing_1y_return": trailing_1y,
            "trailing_6m_return": trailing_6m,
            "max_drawdown": max_drawdown(p / p.iloc[0]),
            "distance_from_200d_sma": dist_sma200,
        })
    return pd.DataFrame(rows)


def estimate_expected_returns(indicators: pd.DataFrame, risk_free_rate: float) -> pd.Series:
    df = indicators.set_index("symbol").copy()
    # Model blend: historical return + momentum + trend, then shrink heavily toward median.
    raw = (
        0.35 * df["historical_return"].fillna(0)
        + 0.35 * df["trailing_1y_return"].fillna(0)
        + 0.15 * df["trailing_6m_return"].fillna(0)
        + 0.15 * df["distance_from_200d_sma"].fillna(0)
    )
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(risk_free_rate)
    median = raw.median()
    mu = 0.40 * raw + 0.60 * median
    mu = mu.clip(lower=-0.25, upper=0.45)
    for sym in mu.index:
        if is_cash_symbol(sym):
            mu.loc[sym] = risk_free_rate
    return mu


def shrink_cov(returns: pd.DataFrame, shrink: float = 0.30) -> pd.DataFrame:
    cov = returns.cov() * TRADING_DAYS
    diag = pd.DataFrame(np.diag(np.diag(cov)), index=cov.index, columns=cov.columns)
    return (1 - shrink) * cov + shrink * diag


def portfolio_stats(w: pd.Series, mu: pd.Series, cov: pd.DataFrame, rf: float) -> Tuple[float, float, float]:
    idx = w.index.intersection(mu.index).intersection(cov.index)
    wv = w.reindex(idx).fillna(0).values
    muv = mu.reindex(idx).fillna(rf).values
    cv = cov.reindex(index=idx, columns=idx).fillna(0).values
    ret = float(wv @ muv)
    vol = float(np.sqrt(max(wv @ cv @ wv, 0)))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


def optimize_candidates(current: pd.Series, mu: pd.Series, cov: pd.DataFrame, rf: float, max_weight: float, max_turnover: Optional[float]) -> List[Candidate]:
    if minimize is None:
        raise RuntimeError("scipy is not installed. Run: uv add scipy")

    idx = current.index.intersection(mu.index).intersection(cov.index)
    current = current.reindex(idx).fillna(0)
    current = current / current.sum()
    mu = mu.reindex(idx).fillna(rf)
    cov = cov.reindex(index=idx, columns=idx).fillna(0)
    n = len(idx)
    bounds = [(0.0, max_weight) for _ in range(n)]
    x0 = current.values.copy()
    if x0.sum() == 0:
        x0 = np.ones(n) / n
    x0 = np.minimum(x0, max_weight)
    x0 = x0 / x0.sum()
    muv = mu.values
    cv = cov.values

    def ret(x):
        return float(x @ muv)

    def vol(x):
        return float(np.sqrt(max(x @ cv @ x, 0)))

    def sharpe(x):
        v = vol(x)
        return (ret(x) - rf) / v if v > 1e-12 else -999

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    if max_turnover is not None and max_turnover > 0:
        constraints.append({"type": "ineq", "fun": lambda x: max_turnover - np.sum(np.abs(x - current.values)) / 2.0})

    out: List[Candidate] = []

    def add_candidate(name: str, x: np.ndarray):
        w = pd.Series(x, index=idx)
        r, v, s = portfolio_stats(w, mu, cov, rf)
        out.append(Candidate(name, w, r, v, s))

    r, v, s = portfolio_stats(current, mu, cov, rf)
    out.append(Candidate("Current Portfolio", current, r, v, s))
    current_vol = v

    res = minimize(lambda x: -sharpe(x), x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000})
    if res.success:
        add_candidate("Optimized: Max Sharpe", res.x)

    # Max expected return subject to not exceeding current volatility.
    constraints_return = list(constraints) + [{"type": "ineq", "fun": lambda x: current_vol - vol(x)}]
    res2 = minimize(lambda x: -ret(x), x0, method="SLSQP", bounds=bounds, constraints=constraints_return, options={"maxiter": 1000})
    if res2.success:
        add_candidate("Optimized: Max Return at Current Risk", res2.x)

    res3 = minimize(lambda x: vol(x), x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 1000})
    if res3.success:
        add_candidate("Optimized: Minimum Volatility", res3.x)

    return out


def simulate_ranges(candidates: List[Candidate], mu: pd.Series, cov: pd.DataFrame, rf: float, sims: int, horizon_years: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cand in candidates:
        idx = cand.weights.index.intersection(mu.index).intersection(cov.index)
        w = cand.weights.reindex(idx).fillna(0).values
        monthly_mu = mu.reindex(idx).fillna(rf).values / 12.0
        monthly_cov = cov.reindex(index=idx, columns=idx).fillna(0).values / 12.0
        months = horizon_years * 12
        try:
            draws = rng.multivariate_normal(monthly_mu, monthly_cov, size=(sims, months), method="svd")
        except TypeError:
            draws = rng.multivariate_normal(monthly_mu, monthly_cov, size=(sims, months))
        port_monthly = draws @ w
        cumulative = np.prod(1.0 + port_monthly, axis=1) - 1.0
        annualized = (1.0 + cumulative) ** (1.0 / horizon_years) - 1.0
        rows.append({
            "portfolio": cand.name,
            "horizon_years": horizon_years,
            "p05_cumulative_return": np.percentile(cumulative, 5),
            "p25_cumulative_return": np.percentile(cumulative, 25),
            "p50_cumulative_return": np.percentile(cumulative, 50),
            "p75_cumulative_return": np.percentile(cumulative, 75),
            "p95_cumulative_return": np.percentile(cumulative, 95),
            "p05_annualized_return": np.percentile(annualized, 5),
            "p50_annualized_return": np.percentile(annualized, 50),
            "p95_annualized_return": np.percentile(annualized, 95),
        })
    return pd.DataFrame(rows)


def random_frontier(mu: pd.Series, cov: pd.DataFrame, rf: float, max_weight: float, samples: int = 2500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = mu.index.intersection(cov.index)
    n = len(idx)
    rows = []
    if n == 0:
        return pd.DataFrame()
    for _ in range(samples):
        # Random capped weights via repeated clipping/renormalization.
        w = rng.dirichlet(np.ones(n))
        for _ in range(8):
            excess = np.maximum(w - max_weight, 0).sum()
            w = np.minimum(w, max_weight)
            if excess <= 1e-10:
                break
            under = w < max_weight
            if under.sum() == 0:
                break
            w[under] += excess * w[under] / w[under].sum()
        w = w / w.sum()
        ws = pd.Series(w, index=idx)
        r, v, s = portfolio_stats(ws, mu, cov, rf)
        rows.append({"expected_return": r, "volatility": v, "sharpe": s})
    return pd.DataFrame(rows)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_charts(outdir: Path, candidates: List[Candidate], frontier: pd.DataFrame, ranges: pd.DataFrame, portfolio: pd.DataFrame, mu: pd.Series, cov: pd.DataFrame) -> Dict[str, Path]:
    charts = outdir / "charts"
    ensure_dir(charts)
    paths = {}

    # Risk/return scatter with random frontier.
    fig, ax = plt.subplots(figsize=(9, 6))
    if not frontier.empty:
        ax.scatter(frontier["volatility"] * 100, frontier["expected_return"] * 100, s=8, alpha=0.25)
    for cand in candidates:
        ax.scatter(cand.volatility * 100, cand.expected_return * 100, s=70, marker="D")
        ax.annotate(cand.name.replace("Optimized: ", ""), (cand.volatility * 100, cand.expected_return * 100), fontsize=8)
    ax.set_title("Scenario Risk vs Expected Return")
    ax.set_xlabel("Expected Annual Volatility (%)")
    ax.set_ylabel("Model Expected Annual Return (%)")
    ax.grid(True, alpha=0.25)
    p = charts / "forecast_risk_return_frontier.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["risk_return_frontier"] = p

    # Distribution ranges.
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df = ranges.copy()
    labels = plot_df["portfolio"].tolist()
    med = plot_df["p50_annualized_return"].values * 100
    low = plot_df["p05_annualized_return"].values * 100
    high = plot_df["p95_annualized_return"].values * 100
    y = np.arange(len(labels))
    ax.errorbar(med, y, xerr=[med - low, high - med], fmt="o", capsize=5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Simulated Annualized Return Range: 5th to 95th Percentile (%)")
    ax.set_title("Probabilistic Return Range by Candidate Mix")
    ax.grid(True, axis="x", alpha=0.25)
    p = charts / "forecast_return_ranges.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["return_ranges"] = p

    # Weight changes vs Max Sharpe if available.
    current = candidates[0].weights
    target = next((c.weights for c in candidates if c.name == "Optimized: Max Sharpe"), candidates[-1].weights)
    delta = (target - current.reindex(target.index).fillna(0)).sort_values(key=lambda s: s.abs(), ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(delta.index[::-1], (delta.values[::-1] * 100))
    ax.set_xlabel("Change in Portfolio Weight (percentage points)")
    ax.set_title("Largest Suggested Weight Changes: Current vs Max-Sharpe Candidate")
    ax.grid(True, axis="x", alpha=0.25)
    p = charts / "forecast_weight_changes.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["weight_changes"] = p

    return paths


def df_to_html_table(df: pd.DataFrame, max_rows: int = 30, pct_cols: Optional[Iterable[str]] = None) -> str:
    if pct_cols is None:
        pct_cols = []
    show = df.head(max_rows).copy()
    for col in pct_cols:
        if col in show.columns:
            show[col] = show[col].map(lambda x: pct(x) if pd.notna(x) else "n/a")
    return show.to_html(index=False, escape=False, classes="report-table")


def write_html_report(outdir: Path, portfolio: pd.DataFrame, candidates: List[Candidate], ranges: pd.DataFrame, indicators: pd.DataFrame, changes: pd.DataFrame, charts: Dict[str, Path], args) -> Path:
    html = []
    html.append("""<!doctype html><html><head><meta charset='utf-8'><title>Portfolio Forecast Optimizer</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;margin:36px;color:#222;line-height:1.35}
h1{font-size:28px;margin-bottom:4px} h2{margin-top:34px;border-bottom:1px solid #ddd;padding-bottom:6px}
.note{background:#f6f8fa;border-left:4px solid #8a8f98;padding:10px 14px;margin:16px 0}
.report-table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 24px}.report-table th{background:#eef2f7;text-align:left}.report-table th,.report-table td{border:1px solid #ccd3dd;padding:6px 8px}
img{max-width:100%;height:auto;margin:8px 0 24px}.small{font-size:12px;color:#555}
</style></head><body>""")
    html.append("<h1>Portfolio Forecast Optimizer</h1>")
    html.append(f"<p class='small'>Generated with {args.years} years of trailing prices, {args.simulations:,} Monte Carlo simulations, max holding weight {args.max_weight:.0%}, and risk-free rate {args.risk_free_rate:.2%}.</p>")
    html.append("<div class='note'>This is a model-based scenario analysis. It estimates ranges from historical prices, momentum/trend indicators, covariance, and constrained optimization. It is designed to compare risk/opportunity tradeoffs, not to provide a guaranteed forecast.</div>")

    metrics_rows = []
    for c in candidates:
        metrics_rows.append({"candidate": c.name, "expected_return": c.expected_return, "volatility": c.volatility, "sharpe": c.sharpe})
    metrics = pd.DataFrame(metrics_rows)
    html.append("<h2>Candidate Mix Summary</h2>")
    html.append(df_to_html_table(metrics, pct_cols=["expected_return", "volatility"]))

    html.append("<h2>Probabilistic Return Ranges</h2>")
    if "return_ranges" in charts:
        html.append(f"<img src='{charts['return_ranges'].relative_to(outdir).as_posix()}'>")
    html.append(df_to_html_table(ranges, pct_cols=[c for c in ranges.columns if "return" in c]))

    html.append("<h2>Risk / Opportunity Frontier</h2>")
    if "risk_return_frontier" in charts:
        html.append(f"<img src='{charts['risk_return_frontier'].relative_to(outdir).as_posix()}'>")

    html.append("<h2>How the Mix Would Change</h2>")
    if "weight_changes" in charts:
        html.append(f"<img src='{charts['weight_changes'].relative_to(outdir).as_posix()}'>")
    html.append(df_to_html_table(changes, max_rows=40, pct_cols=["current_weight", "optimized_weight", "change"]))

    html.append("<h2>Holding Indicators</h2>")
    html.append(df_to_html_table(indicators.sort_values("weight", ascending=False), max_rows=60, pct_cols=["weight", "historical_return", "volatility", "trailing_1y_return", "trailing_6m_return", "max_drawdown", "distance_from_200d_sma", "model_expected_return"]))

    html.append("<h2>Method Notes</h2>")
    html.append("<ul><li>Expected returns are a conservative blend of historical annual return, trailing momentum, and distance from the 200-day average, then shrunk toward the cross-sectional median.</li><li>Risk is modeled from a shrinkage covariance matrix using trailing daily returns.</li><li>Optimization is long-only and capped by max holding weight. It can optionally cap turnover.</li><li>The return range uses Monte Carlo simulation under the model assumptions.</li></ul>")
    html.append("</body></html>")
    path = outdir / "forecast_optimizer_report.html"
    path.write_text("\n".join(html))
    return path


def maybe_write_pdf(outdir: Path, candidates: List[Candidate], ranges: pd.DataFrame, changes: pd.DataFrame, indicators: pd.DataFrame, charts: Dict[str, Path], args) -> Optional[Path]:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    except Exception:
        return None

    path = outdir / "forecast_optimizer_report.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Portfolio Forecast Optimizer", styles["Title"]))
    story.append(Paragraph(f"Trailing data: {args.years} years &nbsp;&nbsp; Simulations: {args.simulations:,} &nbsp;&nbsp; Max holding: {args.max_weight:.0%}", styles["Normal"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Model-based scenario analysis for comparing risk/opportunity tradeoffs. Outputs use normalized weights, not dollars.", styles["Italic"]))

    def add_table(title: str, df: pd.DataFrame, pct_cols: Iterable[str], max_rows: int = 20):
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(title, styles["Heading2"]))
        show = df.head(max_rows).copy()
        for col in pct_cols:
            if col in show.columns:
                show[col] = show[col].map(lambda x: pct(x) if pd.notna(x) else "n/a")
        data = [list(show.columns)] + show.astype(str).values.tolist()
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eaf0f7")),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 7),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
        ]))
        story.append(t)

    metrics = pd.DataFrame([{"candidate": c.name, "expected_return": c.expected_return, "volatility": c.volatility, "sharpe": c.sharpe} for c in candidates])
    add_table("Candidate Mix Summary", metrics, ["expected_return", "volatility"])
    story.append(PageBreak())
    for key, title in [("return_ranges", "Probabilistic Return Ranges"), ("risk_return_frontier", "Risk / Opportunity Frontier"), ("weight_changes", "Largest Weight Changes")]:
        if key in charts and charts[key].exists():
            story.append(Paragraph(title, styles["Heading2"]))
            story.append(Image(str(charts[key]), width=7.0*inch, height=4.5*inch))
            story.append(Spacer(1, 0.15*inch))
    story.append(PageBreak())
    add_table("Return Ranges", ranges, [c for c in ranges.columns if "return" in c], max_rows=10)
    add_table("Current vs Max-Sharpe Candidate", changes, ["current_weight", "optimized_weight", "change"], max_rows=30)
    story.append(PageBreak())
    add_table("Holding Indicators", indicators.sort_values("weight", ascending=False), ["weight", "historical_return", "volatility", "trailing_1y_return", "trailing_6m_return", "max_drawdown", "distance_from_200d_sma", "model_expected_return"], max_rows=50)
    doc.build(story)
    return path


def build(args) -> Dict[str, Path]:
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    portfolio = read_portfolio(Path(args.input))
    symbols = portfolio["symbol"].tolist()

    prices, price_status = download_prices(symbols, args.years, args.risk_free_rate)
    valid_symbols = [s for s in symbols if s in prices.columns]
    portfolio = portfolio[portfolio["symbol"].isin(valid_symbols)].copy()
    portfolio["weight"] = portfolio["weight"] / portfolio["weight"].sum()
    prices = prices[portfolio["symbol"].tolist()]
    returns = prices.pct_change().dropna(how="all")

    indicators = compute_indicators(prices)
    mu = estimate_expected_returns(indicators, args.risk_free_rate)
    cov = shrink_cov(returns, shrink=args.cov_shrink)

    current = portfolio.set_index("symbol")["weight"]
    candidates = optimize_candidates(current, mu, cov, args.risk_free_rate, args.max_weight, args.max_turnover)
    ranges = simulate_ranges(candidates, mu, cov, args.risk_free_rate, args.simulations, args.horizon_years, args.seed)
    frontier = random_frontier(mu, cov, args.risk_free_rate, args.max_weight, samples=args.frontier_samples, seed=args.seed)

    # Enriched indicators.
    ind = portfolio.merge(indicators, on="symbol", how="left")
    ind["model_expected_return"] = ind["symbol"].map(mu)
    ind = ind.sort_values("weight", ascending=False)

    target = next((c.weights for c in candidates if c.name == "Optimized: Max Sharpe"), candidates[-1].weights)
    changes = pd.DataFrame({
        "symbol": current.index.union(target.index),
    })
    changes["current_weight"] = changes["symbol"].map(current).fillna(0)
    changes["optimized_weight"] = changes["symbol"].map(target).fillna(0)
    changes["change"] = changes["optimized_weight"] - changes["current_weight"]
    changes = changes.merge(portfolio[["symbol", "display_name", "asset_class_name", "sector_clean"]], on="symbol", how="left")
    changes = changes.sort_values("change", key=lambda s: s.abs(), ascending=False)

    charts = save_charts(outdir, candidates, frontier, ranges, portfolio, mu, cov)

    outputs: Dict[str, Path] = {}
    portfolio.to_csv(outdir / "forecast_input_weights.csv", index=False)
    outputs["input_weights"] = outdir / "forecast_input_weights.csv"
    price_status.to_csv(outdir / "forecast_price_status.csv", index=False)
    outputs["price_status"] = outdir / "forecast_price_status.csv"
    ind.to_csv(outdir / "forecast_holding_indicators.csv", index=False)
    outputs["holding_indicators"] = outdir / "forecast_holding_indicators.csv"
    cand_df = pd.DataFrame([{"candidate": c.name, "expected_return": c.expected_return, "volatility": c.volatility, "sharpe": c.sharpe} for c in candidates])
    cand_df.to_csv(outdir / "forecast_candidate_metrics.csv", index=False)
    outputs["candidate_metrics"] = outdir / "forecast_candidate_metrics.csv"
    ranges.to_csv(outdir / "forecast_return_ranges.csv", index=False)
    outputs["return_ranges"] = outdir / "forecast_return_ranges.csv"
    changes.to_csv(outdir / "forecast_rebalance_changes.csv", index=False)
    outputs["rebalance_changes"] = outdir / "forecast_rebalance_changes.csv"
    frontier.to_csv(outdir / "forecast_frontier_points.csv", index=False)
    outputs["frontier_points"] = outdir / "forecast_frontier_points.csv"
    html = write_html_report(outdir, portfolio, candidates, ranges, ind, changes, charts, args)
    outputs["html_report"] = html
    pdf = maybe_write_pdf(outdir, candidates, ranges, changes, ind, charts, args)
    if pdf is not None:
        outputs["pdf_report"] = pdf
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Portfolio probabilistic forecast and constrained optimization companion tool.")
    parser.add_argument("--input", default="reports/monthly_review/public_normalized_holdings_weights.csv", help="Monthly public weights CSV or Schwab holdings CSV.")
    parser.add_argument("--outdir", default="reports/forecast_optimizer")
    parser.add_argument("--years", type=int, default=5, help="Trailing years of price history to use.")
    parser.add_argument("--horizon-years", type=int, default=1, help="Forecast horizon for simulated return ranges.")
    parser.add_argument("--risk-free-rate", type=float, default=0.04)
    parser.add_argument("--simulations", type=int, default=10000)
    parser.add_argument("--frontier-samples", type=int, default=2500)
    parser.add_argument("--max-weight", type=float, default=0.08, help="Maximum weight allowed for any one holding in optimized portfolios.")
    parser.add_argument("--max-turnover", type=float, default=None, help="Optional max turnover from current weights, e.g. 0.25 means at most 25% of portfolio changed.")
    parser.add_argument("--cov-shrink", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    outputs = build(args)
    print("Forecast optimizer complete. Generated files:")
    for name, path in outputs.items():
        print(f"  {name:24s} {path}")


if __name__ == "__main__":
    main()
