from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from .analysis import enrich_holdings, load_config, load_holdings

TRADING_DAYS = 252
ASSET_PROXY = {
    "us_stock": "VTI",
    "intl_stock": "VXUS",
    "bond": "BND",
    "cash": "SGOV",
    "other": "VNQ",
}

STANDARD_BENCHMARKS = {
    "S&P 500": {"SPY": 1.00},
    "Global Equity": {"VT": 1.00},
    "US 60/40": {"VTI": 0.60, "BND": 0.40},
    "Global 60/40": {"VTI": 0.36, "VXUS": 0.24, "BND": 0.40},
    "Conservative 40/60": {"VTI": 0.24, "VXUS": 0.16, "BND": 0.55, "SGOV": 0.05},
}


def _safe_pct(x: float) -> str:
    if pd.isna(x) or not np.isfinite(x):
        return "n/a"
    return f"{x * 100:.2f}%"


def _safe_num(x: float, digits: int = 2) -> str:
    if pd.isna(x) or not np.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


def _month_label(value: str | None = None) -> str:
    if value:
        return value
    return date.today().strftime("%Y-%m")


def _clean_symbol_for_yahoo(symbol: str) -> str:
    s = str(symbol or "").strip().upper()
    if s in {"", "NAN"}:
        return ""
    # Schwab and Yahoo differ for share classes.
    s = s.replace("/", "-")
    if s == "BRK.B":
        s = "BRK-B"
    elif s == "BF.B":
        s = "BF-B"
    else:
        s = s.replace(".", "-")
    return s


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    clean = {k: float(v) for k, v in weights.items() if k and pd.notna(v) and float(v) > 0}
    total = sum(clean.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in clean.items()}


def _benchmark_from_config(config: dict) -> Dict[str, float]:
    targets = config.get("target_allocation", {}) or {}
    weights: Dict[str, float] = {}
    for asset_class, weight in targets.items():
        proxy = ASSET_PROXY.get(str(asset_class))
        try:
            w = float(weight)
        except Exception:
            w = 0.0
        if proxy and w > 0:
            weights[proxy] = weights.get(proxy, 0.0) + w
    return _normalize_weights(weights)


def _portfolio_weights(enriched: pd.DataFrame) -> pd.DataFrame:
    h = enriched.copy()
    h["yahoo_symbol"] = h["symbol"].map(_clean_symbol_for_yahoo)
    # Use SGOV as a cash proxy for normalized return/risk series.
    cash_mask = h["asset_class"].eq("cash") | h["symbol"].astype(str).str.upper().eq("CASH")
    h.loc[cash_mask, "yahoo_symbol"] = "SGOV"
    grp = h.groupby("yahoo_symbol", dropna=False)["market_value"].sum().reset_index()
    grp = grp[grp["yahoo_symbol"].astype(str).str.len() > 0]
    total = grp["market_value"].sum()
    grp["weight"] = grp["market_value"] / total if total else 0.0
    return grp[["yahoo_symbol", "weight"]].rename(columns={"yahoo_symbol": "symbol"})


def _download_prices(symbols: Iterable[str], years: int, outdir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    symbols = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
    status_rows = []
    if not symbols:
        return pd.DataFrame(), pd.DataFrame()

    # yfinance period avoids date math and generally handles non-trading days better.
    period = f"{int(years)}y"
    try:
        raw = yf.download(
            tickers=symbols,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=False,
        )
    except Exception as e:
        # Try one-at-a-time fallback.
        frames = []
        for sym in symbols:
            try:
                one = yf.download(sym, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
                if not one.empty:
                    close = one[["Close"]].rename(columns={"Close": sym})
                    frames.append(close)
                    status_rows.append({"symbol": sym, "status": "ok", "rows": len(close), "message": "fallback"})
                else:
                    status_rows.append({"symbol": sym, "status": "empty", "rows": 0, "message": "fallback empty"})
            except Exception as e2:
                status_rows.append({"symbol": sym, "status": "error", "rows": 0, "message": str(e2)})
        prices = pd.concat(frames, axis=1) if frames else pd.DataFrame()
        return prices, pd.DataFrame(status_rows)

    if raw.empty:
        for sym in symbols:
            status_rows.append({"symbol": sym, "status": "empty", "rows": 0, "message": "download returned no rows"})
        return pd.DataFrame(), pd.DataFrame(status_rows)

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            # yfinance sometimes flips levels for single/multiple tickers.
            try:
                prices = raw.xs("Close", axis=1, level=-1).copy()
            except Exception:
                prices = pd.DataFrame()
    else:
        prices = raw[["Close"]].copy() if "Close" in raw.columns else pd.DataFrame()
        if len(symbols) == 1 and not prices.empty:
            prices.columns = symbols

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(symbols[0])

    prices = prices.dropna(how="all")
    for sym in symbols:
        if sym in prices.columns:
            rows = int(prices[sym].dropna().shape[0])
            status_rows.append({"symbol": sym, "status": "ok" if rows else "empty", "rows": rows, "message": ""})
        else:
            status_rows.append({"symbol": sym, "status": "missing", "rows": 0, "message": "not in downloaded close data"})
    return prices, pd.DataFrame(status_rows)


def _series_from_weights(returns: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    usable = [s for s in weights if s in returns.columns]
    if not usable:
        return pd.Series(dtype=float)
    w = pd.Series({s: weights[s] for s in usable}, dtype=float)
    w = w / w.sum()
    return returns[usable].fillna(0.0).dot(w).rename("return")


def _max_drawdown(growth: pd.Series) -> float:
    if growth.empty:
        return np.nan
    peak = growth.cummax()
    dd = growth / peak - 1.0
    return float(dd.min())


def _bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands on normalized portfolio/benchmark value series."""
    s = series.dropna().astype(float)
    if s.empty:
        return pd.DataFrame()
    mid = s.rolling(window).mean()
    std = s.rolling(window).std(ddof=0)
    out = pd.DataFrame({
        "value": s,
        "middle_band": mid,
        "upper_band": mid + num_std * std,
        "lower_band": mid - num_std * std,
    })
    out["percent_b"] = (out["value"] - out["lower_band"]) / (out["upper_band"] - out["lower_band"])
    out["band_width"] = (out["upper_band"] - out["lower_band"]) / out["middle_band"]
    return out


def _rolling_indicators(daily: pd.DataFrame, risk_free_rate: float) -> pd.DataFrame:
    """Rolling indicators for the shareable monthly report."""
    if daily.empty or "Your Portfolio" not in daily.columns:
        return pd.DataFrame()
    r = daily["Your Portfolio"].dropna()
    if r.empty:
        return pd.DataFrame()
    rf_daily = risk_free_rate / TRADING_DAYS
    rolling_63d_vol = r.rolling(63).std(ddof=0) * math.sqrt(TRADING_DAYS)
    rolling_252d_return = (1 + r).rolling(252).apply(lambda x: float(np.prod(x) - 1), raw=True)
    rolling_252d_sharpe = ((r - rf_daily).rolling(252).mean() / r.rolling(252).std(ddof=0)) * math.sqrt(TRADING_DAYS)
    return pd.DataFrame({
        "portfolio_daily_return": r,
        "rolling_3m_volatility": rolling_63d_vol,
        "rolling_12m_return": rolling_252d_return,
        "rolling_12m_sharpe": rolling_252d_sharpe,
    })

def _build_efficient_frontier_points(
    returns: pd.DataFrame,
    portfolio_weights: Dict[str, float],
    benchmarks: Dict[str, Dict[str, float]],
    risk_free_rate: float,
    simulations: int = 4000,
) -> pd.DataFrame:
    """Build a simple long-only efficient-frontier style cloud.

    This is intentionally based on broad asset-class proxies rather than every
    individual holding. It answers: how do standard mixes of stocks/bonds/cash/
    alternatives compare with the current portfolio and benchmark mixes?
    """
    proxy_symbols = []
    for sym in dict.fromkeys(ASSET_PROXY.values()):
        if sym in returns.columns:
            proxy_symbols.append(sym)
    if len(proxy_symbols) < 2:
        return pd.DataFrame()

    r = returns[proxy_symbols].dropna(how="any")
    if len(r) < 30:
        return pd.DataFrame()

    mean = r.mean() * TRADING_DAYS
    cov = r.cov() * TRADING_DAYS
    rng = np.random.default_rng(42)
    rows = []

    for i in range(simulations):
        w = rng.dirichlet(np.ones(len(proxy_symbols)))
        ann_return = float(np.dot(w, mean))
        ann_vol = float(np.sqrt(np.dot(w, np.dot(cov, w))))
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan
        rows.append({
            "name": f"Simulated mix {i+1}",
            "type": "simulated_mix",
            "annual_return": ann_return,
            "annual_volatility": ann_vol,
            "sharpe": sharpe,
        })

    # Overlay portfolio and benchmark points calculated from their actual daily return series.
    overlays = {"Your Portfolio": portfolio_weights}
    overlays.update({name: _normalize_weights(weights) for name, weights in benchmarks.items()})
    for name, weights in overlays.items():
        s = _series_from_weights(returns, weights).dropna()
        if s.empty:
            continue
        ann_return = float((1 + s).prod() ** (TRADING_DAYS / max(len(s), 1)) - 1)
        ann_vol = float(s.std(ddof=0) * math.sqrt(TRADING_DAYS))
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan
        rows.append({
            "name": name,
            "type": "current_or_benchmark",
            "annual_return": ann_return,
            "annual_volatility": ann_vol,
            "sharpe": sharpe,
        })

    return pd.DataFrame(rows)


def _metrics(series: pd.Series, benchmark: pd.Series | None, risk_free_rate: float) -> dict:
    r = series.dropna()
    if r.empty:
        return {}
    rf_daily = risk_free_rate / TRADING_DAYS
    excess = r - rf_daily
    ann_return = float((1 + r).prod() ** (TRADING_DAYS / max(len(r), 1)) - 1)
    ann_vol = float(r.std(ddof=0) * math.sqrt(TRADING_DAYS))
    downside = r[r < rf_daily] - rf_daily
    downside_dev = float(downside.std(ddof=0) * math.sqrt(TRADING_DAYS)) if len(downside) else np.nan
    sharpe = float(excess.mean() / r.std(ddof=0) * math.sqrt(TRADING_DAYS)) if r.std(ddof=0) > 0 else np.nan
    sortino = float(excess.mean() * TRADING_DAYS / downside_dev) if downside_dev and np.isfinite(downside_dev) and downside_dev > 0 else np.nan
    growth = (1 + r).cumprod()
    out = {
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": _max_drawdown(growth),
        "best_day": float(r.max()),
        "worst_day": float(r.min()),
        "positive_day_pct": float((r > 0).mean()),
    }
    if benchmark is not None and not benchmark.dropna().empty:
        aligned = pd.concat([r.rename("portfolio"), benchmark.rename("benchmark")], axis=1).dropna()
        if not aligned.empty:
            b = aligned["benchmark"]
            p = aligned["portfolio"]
            active = p - b
            bvar = b.var(ddof=0)
            beta = float(p.cov(b) / bvar) if bvar > 0 else np.nan
            corr = float(p.corr(b)) if len(aligned) > 2 else np.nan
            tracking = float(active.std(ddof=0) * math.sqrt(TRADING_DAYS))
            info = float(active.mean() / active.std(ddof=0) * math.sqrt(TRADING_DAYS)) if active.std(ddof=0) > 0 else np.nan
            alpha = float((p.mean() * TRADING_DAYS) - beta * (b.mean() * TRADING_DAYS)) if np.isfinite(beta) else np.nan
            up = aligned[b > 0]
            down = aligned[b < 0]
            upside_capture = float(up["portfolio"].mean() / up["benchmark"].mean()) if not up.empty and up["benchmark"].mean() != 0 else np.nan
            downside_capture = float(down["portfolio"].mean() / down["benchmark"].mean()) if not down.empty and down["benchmark"].mean() != 0 else np.nan
            out.update({
                "beta_vs_standard": beta,
                "correlation_vs_standard": corr,
                "alpha_vs_standard": alpha,
                "tracking_error": tracking,
                "information_ratio": info,
                "upside_capture": upside_capture,
                "downside_capture": downside_capture,
            })
    return out


def _make_charts(
    growth: pd.DataFrame,
    drawdowns: pd.DataFrame,
    metrics: pd.DataFrame,
    weights: pd.DataFrame,
    charts: Path,
    bollinger: pd.DataFrame | None = None,
    rolling: pd.DataFrame | None = None,
    asset_weights: pd.DataFrame | None = None,
    sector_weights: pd.DataFrame | None = None,
    frontier_points: pd.DataFrame | None = None,
) -> dict:
    charts.mkdir(parents=True, exist_ok=True)
    paths = {}
    import matplotlib.pyplot as plt

    if not growth.empty:
        ax = (growth * 100).plot(figsize=(11, 6), linewidth=1.4)
        ax.set_title("Normalized Growth of $1 (Shown as 100 at Start)")
        ax.set_ylabel("Normalized Value")
        ax.grid(True, alpha=0.3)
        p = charts / "monthly_normalized_growth.png"
        plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
        paths["normalized_growth"] = p

    if not drawdowns.empty:
        ax = (drawdowns * 100).plot(figsize=(11, 5), linewidth=1.2)
        ax.set_title("Drawdown from Prior Peak")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)
        p = charts / "monthly_drawdown.png"
        plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
        paths["drawdown"] = p

    if bollinger is not None and not bollinger.empty:
        fig, ax = plt.subplots(figsize=(11, 6))
        b = bollinger.dropna(subset=["middle_band"]).copy()
        if not b.empty:
            ax.plot(b.index, b["value"] * 100, label="Your Portfolio")
            ax.plot(b.index, b["middle_band"] * 100, label="20-day moving average", linewidth=1.0)
            ax.plot(b.index, b["upper_band"] * 100, label="Upper band", linewidth=0.9)
            ax.plot(b.index, b["lower_band"] * 100, label="Lower band", linewidth=0.9)
            ax.set_title("Portfolio Bollinger Bands on Normalized Value")
            ax.set_ylabel("Normalized Value")
            ax.grid(True, alpha=0.3)
            ax.legend()
            p = charts / "monthly_portfolio_bollinger_bands.png"
            plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
            paths["bollinger"] = p
        else:
            plt.close()

    if rolling is not None and not rolling.empty and "rolling_12m_sharpe" in rolling.columns:
        fig, ax = plt.subplots(figsize=(11, 5))
        rolling["rolling_12m_sharpe"].dropna().plot(ax=ax, linewidth=1.3)
        ax.set_title("Rolling 12-Month Sharpe Ratio")
        ax.set_ylabel("Sharpe")
        ax.grid(True, alpha=0.3)
        p = charts / "monthly_rolling_12m_sharpe.png"
        plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
        paths["rolling_sharpe"] = p

    if rolling is not None and not rolling.empty and "rolling_3m_volatility" in rolling.columns:
        fig, ax = plt.subplots(figsize=(11, 5))
        (rolling["rolling_3m_volatility"].dropna() * 100).plot(ax=ax, linewidth=1.3)
        ax.set_title("Rolling 3-Month Annualized Volatility")
        ax.set_ylabel("Volatility (%)")
        ax.grid(True, alpha=0.3)
        p = charts / "monthly_rolling_3m_volatility.png"
        plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
        paths["rolling_volatility"] = p

    if not metrics.empty and {"annual_volatility", "annual_return"}.issubset(metrics.columns):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(metrics["annual_volatility"] * 100, metrics["annual_return"] * 100)
        for _, row in metrics.iterrows():
            ax.annotate(row["name"], (row["annual_volatility"] * 100, row["annual_return"] * 100), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_title("Risk vs Return, Trailing Period")
        ax.set_xlabel("Annualized Volatility (%)")
        ax.set_ylabel("Annualized Return (%)")
        ax.grid(True, alpha=0.3)
        p = charts / "monthly_risk_return_scatter.png"
        plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
        paths["risk_return"] = p

    if asset_weights is not None and not asset_weights.empty:
        aw = asset_weights.dropna(subset=["asset_class_name"]).copy()
        aw = aw[aw["weight"] > 0].sort_values("weight", ascending=False)
        if not aw.empty:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(aw["weight"], labels=aw["asset_class_name"], autopct="%1.1f%%", startangle=90)
            ax.set_title("Portfolio by Asset Class")
            ax.axis("equal")
            p = charts / "monthly_asset_class_pie.png"
            plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
            paths["asset_class_pie"] = p

    if sector_weights is not None and not sector_weights.empty:
        sw = sector_weights.dropna(subset=["sector_clean"]).copy()
        sw = sw[sw["weight"] > 0].sort_values("weight", ascending=False)
        if len(sw) > 10:
            top = sw.head(9).copy()
            other_weight = sw.iloc[9:]["weight"].sum()
            sw = pd.concat([top, pd.DataFrame([{"sector_clean": "Other sectors", "weight": other_weight}])], ignore_index=True)
        if not sw.empty:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(sw["weight"], labels=sw["sector_clean"], autopct="%1.1f%%", startangle=90)
            ax.set_title("Portfolio by Sector")
            ax.axis("equal")
            p = charts / "monthly_sector_pie.png"
            plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
            paths["sector_pie"] = p

    if frontier_points is not None and not frontier_points.empty:
        fig, ax = plt.subplots(figsize=(9, 7))
        sim = frontier_points[frontier_points["type"].eq("simulated_mix")]
        pts = frontier_points[frontier_points["type"].eq("current_or_benchmark")]
        if not sim.empty:
            ax.scatter(sim["annual_volatility"] * 100, sim["annual_return"] * 100, s=8, alpha=0.25, label="Simulated asset-class mixes")
        if not pts.empty:
            ax.scatter(pts["annual_volatility"] * 100, pts["annual_return"] * 100, s=70, marker="D", label="Portfolio / benchmarks")
            for _, row in pts.iterrows():
                ax.annotate(row["name"], (row["annual_volatility"] * 100, row["annual_return"] * 100), fontsize=8, xytext=(5, 5), textcoords="offset points")
        ax.set_title("Efficient Frontier Style View — Broad Asset-Class Mixes")
        ax.set_xlabel("Annualized Volatility (%)")
        ax.set_ylabel("Annualized Return (%)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        p = charts / "monthly_efficient_frontier.png"
        plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
        paths["efficient_frontier"] = p

    if not weights.empty:
        top = weights.sort_values("weight", ascending=False).head(20).sort_values("weight")
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(top["label"], top["weight"] * 100)
        ax.set_title("Top Holdings by Normalized Weight")
        ax.set_xlabel("Portfolio Weight (%)")
        p = charts / "monthly_top_weights.png"
        plt.tight_layout(); plt.savefig(p, dpi=150); plt.close()
        paths["top_weights"] = p

    return paths


def _html_table(df: pd.DataFrame, pct_cols: list[str] | None = None, num_cols: list[str] | None = None) -> str:
    d = df.copy()
    for c in pct_cols or []:
        if c in d.columns:
            d[c] = d[c].map(_safe_pct)
    for c in num_cols or []:
        if c in d.columns:
            d[c] = d[c].map(lambda x: _safe_num(x, 2))
    return d.to_html(index=False, escape=False)



def _pdf_format_pct(x: float) -> str:
    if pd.isna(x) or not np.isfinite(x):
        return "n/a"
    return f"{x * 100:.2f}%"


def _pdf_format_num(x: float, digits: int = 2) -> str:
    if pd.isna(x) or not np.isfinite(x):
        return "n/a"
    return f"{x:.{digits}f}"


def _write_monthly_pdf_report(
    pdf_path: Path,
    label: str,
    standard_name: str,
    chart_paths: dict,
    interp: pd.DataFrame,
    metrics: pd.DataFrame,
    asset_weights: pd.DataFrame,
    sector_weights: pd.DataFrame,
    public_weights: pd.DataFrame,
    benchmark_weights_rows: list[dict],
) -> Path:
    """Create a paginated, shareable PDF monthly review.

    The PDF intentionally uses normalized percentages and omits dollar amounts.
    """
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "MonthlyTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        leading=27,
        spaceAfter=14,
    )
    h1 = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading1"],
        fontSize=15,
        leading=18,
        spaceBefore=12,
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "ChartTitle",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        spaceBefore=8,
        spaceAfter=5,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=12,
        spaceAfter=6,
    )
    small = ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["BodyText"],
        fontSize=7.5,
        leading=9,
        alignment=TA_LEFT,
    )
    table_head = ParagraphStyle(
        "TableHead",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9,
        alignment=TA_LEFT,
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )

    story = []

    def page_number(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawRightString(7.95 * inch, 0.32 * inch, f"Page {doc_.page}")
        canvas.restoreState()

    def ptxt(value):
        if value is None:
            value = ""
        return Paragraph(str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), table_cell)

    def add_df_table(df: pd.DataFrame, columns: list[str], widths: list[float] | None = None, max_rows: int | None = None):
        d = df.copy()
        if max_rows is not None:
            d = d.head(max_rows)
        data = [[Paragraph(c.replace("_", " ").title(), table_head) for c in columns]]
        for _, row in d.iterrows():
            data.append([ptxt(row.get(c, "")) for c in columns])
        if widths is None:
            widths = [doc.width / len(columns)] * len(columns)
        tbl = Table(data, colWidths=widths, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1F2A44")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C9D2E3")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFD")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.12 * inch))

    def add_chart(title: str, key: str, width: float = 7.2 * inch, max_height: float = 4.6 * inch):
        path = chart_paths.get(key)
        if not path or not Path(path).exists():
            return
        img = Image(str(path))
        scale = min(width / img.imageWidth, max_height / img.imageHeight)
        img.drawWidth = img.imageWidth * scale
        img.drawHeight = img.imageHeight * scale
        img.hAlign = "CENTER"

        # Keep the chart title and chart together. This prevents orphaned titles
        # at the bottom of one page with the image starting on the next page.
        story.append(KeepTogether([
            Paragraph(title, h2),
            img,
            Spacer(1, 0.08 * inch),
        ]))

    story.append(Paragraph(f"Monthly Portfolio Review - {label}", title_style))
    story.append(Paragraph(
        "Shareable version: dollar amounts are intentionally omitted. Holdings are shown as normalized percentages only. "
        f"The primary comparison benchmark is <b>{standard_name}</b>.",
        body,
    ))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Headline Interpretation", h1))
    if not interp.empty:
        add_df_table(interp.rename(columns={"item": "Topic", "reading": "Reading"}), ["Topic", "Reading"], widths=[1.45 * inch, 5.75 * inch])

    metric_cols = [
        "name", "annual_return", "annual_volatility", "sharpe", "sortino", "max_drawdown",
        "beta_vs_standard", "tracking_error", "information_ratio", "upside_capture", "downside_capture",
    ]
    metric_cols = [c for c in metric_cols if c in metrics.columns]
    metric_pdf = metrics.copy()
    for c in ["annual_return", "annual_volatility", "max_drawdown", "tracking_error", "upside_capture", "downside_capture"]:
        if c in metric_pdf.columns:
            metric_pdf[c] = metric_pdf[c].map(_pdf_format_pct)
    for c in ["sharpe", "sortino", "beta_vs_standard", "information_ratio"]:
        if c in metric_pdf.columns:
            metric_pdf[c] = metric_pdf[c].map(lambda x: _pdf_format_num(x, 2))
    story.append(Paragraph("Key Risk / Opportunity Metrics", h1))
    add_df_table(metric_pdf, metric_cols, max_rows=12)

    story.append(PageBreak())
    story.append(Paragraph("Portfolio Allocation", h1))
    add_chart("Portfolio by Asset Class", "asset_class_pie", width=3.5 * inch, max_height=3.5 * inch)
    add_chart("Portfolio by Sector", "sector_pie", width=4.4 * inch, max_height=4.4 * inch)

    # Start allocation tables on a fresh page so table headings stay with their tables.
    story.append(PageBreak())

    asset_pdf = asset_weights.copy()
    if "weight_pct" in asset_pdf.columns:
        asset_pdf["weight"] = asset_pdf["weight_pct"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "n/a")
    elif "weight" in asset_pdf.columns:
        asset_pdf["weight"] = asset_pdf["weight"].map(_pdf_format_pct)
    story.append(Paragraph("Asset-Class Weights", h1))
    add_df_table(asset_pdf.rename(columns={"asset_class_name": "asset_class"}), ["asset_class", "weight"], widths=[4.8 * inch, 1.4 * inch])

    sector_pdf = sector_weights.copy()
    if "weight_pct" in sector_pdf.columns:
        sector_pdf["weight"] = sector_pdf["weight_pct"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "n/a")
    elif "weight" in sector_pdf.columns:
        sector_pdf["weight"] = sector_pdf["weight"].map(_pdf_format_pct)
    story.append(Paragraph("Top Sector Weights", h1))
    add_df_table(sector_pdf.rename(columns={"sector_clean": "sector"}).head(18), ["sector", "weight"], widths=[4.8 * inch, 1.4 * inch])

    story.append(PageBreak())
    story.append(Paragraph("Portfolio vs Benchmarks", h1))
    add_chart("Normalized Growth", "normalized_growth")
    add_chart("Drawdown from Prior Peak", "drawdown")

    story.append(PageBreak())
    story.append(Paragraph("Risk and Opportunity", h1))
    add_chart("Risk vs Return", "risk_return", width=6.5 * inch, max_height=4.5 * inch)
    add_chart("Efficient Frontier Style View", "efficient_frontier", width=6.8 * inch, max_height=4.8 * inch)

    story.append(PageBreak())
    story.append(Paragraph("Technical / Rolling Indicators", h1))
    add_chart("Portfolio Bollinger Bands", "bollinger")
    add_chart("Rolling 12-Month Sharpe", "rolling_sharpe")
    add_chart("Rolling 3-Month Volatility", "rolling_volatility")

    story.append(PageBreak())
    story.append(Paragraph("Top Holdings", h1))
    add_chart("Top Normalized Holdings", "top_weights", width=7.1 * inch, max_height=4.7 * inch)
    holdings_pdf = public_weights.copy().head(30)
    if "portfolio_weight_pct" in holdings_pdf.columns:
        holdings_pdf["weight"] = holdings_pdf["portfolio_weight_pct"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "n/a")
    elif "weight" in holdings_pdf.columns:
        holdings_pdf["weight"] = holdings_pdf["weight"].map(_pdf_format_pct)
    keep_cols = [c for c in ["symbol", "display_name", "asset_class_name", "sector_clean", "weight"] if c in holdings_pdf.columns]
    add_df_table(holdings_pdf, keep_cols, widths=[0.7 * inch, 2.35 * inch, 1.3 * inch, 1.55 * inch, 0.75 * inch], max_rows=30)

    story.append(PageBreak())
    story.append(Paragraph("Benchmark Definitions", h1))
    story.append(Paragraph("Each benchmark portfolio below sums to 100%. The table lists several separate benchmark portfolios, so the whole table is not meant to sum to 100%.", body))
    bm_pdf = pd.DataFrame(benchmark_weights_rows)
    if not bm_pdf.empty:
        bm_pdf["weight"] = bm_pdf["weight_pct"].map(lambda x: f"{x:.2f}%" if pd.notna(x) else "n/a")
        add_df_table(bm_pdf, ["benchmark", "symbol", "weight"], widths=[3.2 * inch, 1.0 * inch, 1.0 * inch])

    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Notes", h1))
    story.append(Paragraph(
        "This report uses trailing market data, normalized portfolio weights, and standard benchmark portfolios. "
        "The efficient-frontier view is based on broad asset-class proxies and is best read as a risk/opportunity comparison, not a forecast.",
        small,
    ))

    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    return pdf_path

def build_monthly_review(
    holdings_path: str = "data/schwab_holdings.csv",
    config_path: str = "config.yaml",
    outdir: str = "reports",
    years: int = 3,
    risk_free_rate: float = 0.04,
    month: str | None = None,
    refresh_metadata: bool = False,
) -> dict:
    label = _month_label(month)
    out = Path(outdir)
    latest = out / "monthly_review"
    archive_dir = out / "monthly_reviews" / label
    latest.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    charts = latest / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    holdings = load_holdings(holdings_path)
    enriched = enrich_holdings(holdings, refresh_metadata=refresh_metadata)

    # Sanitized weights: no dollar values.
    # Keep account-level rows internally for sector/asset-class aggregation, but publish
    # one row per ticker so the shareable report does not show duplicate symbols when
    # the same security is held in multiple Schwab accounts.
    weights_table = enriched.copy()
    weights_table["symbol"] = weights_table["symbol"].astype(str).str.strip().str.upper()
    weights_table["label"] = np.where(
        weights_table["display_name"].astype(str).str.len() > 0,
        weights_table["symbol"] + " — " + weights_table["display_name"].astype(str),
        weights_table["symbol"],
    )

    def _first_nonempty(series):
        for value in series:
            if pd.notna(value) and str(value).strip() and str(value).strip().lower() != "nan":
                return value
        return ""

    public_weights = (
        weights_table.groupby("symbol", as_index=False, dropna=False)
        .agg(
            weight=("weight", "sum"),
            display_name=("display_name", _first_nonempty),
            asset_class_name=("asset_class_name", _first_nonempty),
            sector_clean=("sector_clean", _first_nonempty),
            industry=("industry", _first_nonempty),
        )
    )
    public_weights["portfolio_weight_pct"] = public_weights["weight"] * 100
    public_weights["label"] = np.where(
        public_weights["display_name"].astype(str).str.len() > 0,
        public_weights["symbol"] + " — " + public_weights["display_name"].astype(str),
        public_weights["symbol"],
    )
    public_weights = public_weights[
        ["symbol", "display_name", "asset_class_name", "sector_clean", "industry", "portfolio_weight_pct", "label", "weight"]
    ].sort_values("portfolio_weight_pct", ascending=False)
    public_weights[["symbol", "display_name", "asset_class_name", "sector_clean", "industry", "portfolio_weight_pct"]].to_csv(
        latest / "public_normalized_holdings_weights.csv", index=False
    )

    sector_weights = weights_table.groupby("sector_clean", dropna=False)["weight"].sum().reset_index().sort_values("weight", ascending=False)
    sector_weights["weight_pct"] = sector_weights["weight"] * 100
    sector_weights.drop(columns=["weight"]).to_csv(latest / "public_sector_weights.csv", index=False)

    asset_weights = weights_table.groupby("asset_class_name", dropna=False)["weight"].sum().reset_index().sort_values("weight", ascending=False)
    asset_weights["weight_pct"] = asset_weights["weight"] * 100
    asset_weights.drop(columns=["weight"]).to_csv(latest / "public_asset_class_weights.csv", index=False)

    portfolio_w_df = _portfolio_weights(enriched)
    portfolio_weights = dict(zip(portfolio_w_df["symbol"], portfolio_w_df["weight"]))

    benchmarks = dict(STANDARD_BENCHMARKS)
    target = _benchmark_from_config(config)
    if target:
        benchmarks = {"Target Policy Benchmark": target, **benchmarks}
    standard_name = "Target Policy Benchmark" if target else "Global 60/40"

    all_symbols = set(portfolio_weights)
    for bm in benchmarks.values():
        all_symbols.update(bm)

    prices, price_status = _download_prices(all_symbols, years, latest)
    price_status.to_csv(latest / "monthly_price_status.csv", index=False)
    if prices.empty:
        raise RuntimeError("No price data downloaded. Check internet connection and reports/monthly_review/monthly_price_status.csv")
    returns = prices.pct_change(fill_method=None).dropna(how="all")

    series = {"Your Portfolio": _series_from_weights(returns, portfolio_weights)}
    for name, bm_weights in benchmarks.items():
        series[name] = _series_from_weights(returns, _normalize_weights(bm_weights))

    daily = pd.DataFrame(series).dropna(how="all")
    daily.to_csv(latest / "monthly_daily_returns.csv")
    growth = (1 + daily.fillna(0)).cumprod()
    drawdowns = growth / growth.cummax() - 1.0
    growth.to_csv(latest / "monthly_normalized_growth.csv")
    drawdowns.to_csv(latest / "monthly_drawdowns.csv")

    bollinger = _bollinger_bands(growth["Your Portfolio"]) if "Your Portfolio" in growth.columns else pd.DataFrame()
    bollinger.to_csv(latest / "monthly_bollinger_bands.csv")
    rolling = _rolling_indicators(daily, risk_free_rate)
    rolling.to_csv(latest / "monthly_rolling_indicators.csv")

    benchmark_series = daily.get(standard_name)
    rows = []
    for name in daily.columns:
        m = _metrics(daily[name], benchmark_series if name != standard_name else None, risk_free_rate)
        m["name"] = name
        m["comparison_benchmark"] = standard_name if name != standard_name else "self"
        rows.append(m)
    metrics = pd.DataFrame(rows)
    cols = ["name", "comparison_benchmark"] + [c for c in metrics.columns if c not in {"name", "comparison_benchmark"}]
    metrics = metrics[cols]
    metrics.to_csv(latest / "monthly_risk_opportunity_metrics.csv", index=False)

    benchmark_weights_rows = []
    for name, bm in benchmarks.items():
        for sym, w in _normalize_weights(bm).items():
            benchmark_weights_rows.append({"benchmark": name, "symbol": sym, "weight_pct": w * 100})
    pd.DataFrame(benchmark_weights_rows).to_csv(latest / "public_standard_benchmark_weights.csv", index=False)

    frontier_points = _build_efficient_frontier_points(returns, portfolio_weights, benchmarks, risk_free_rate)
    frontier_points.to_csv(latest / "monthly_efficient_frontier_points.csv", index=False)

    chart_paths = _make_charts(
        growth,
        drawdowns,
        metrics,
        public_weights[["label", "weight"]].copy(),
        charts,
        bollinger=bollinger,
        rolling=rolling,
        asset_weights=asset_weights.copy(),
        sector_weights=sector_weights.copy(),
        frontier_points=frontier_points,
    )

    # Interpretation table keeps it simple and monthly-repeatable.
    p_row = metrics[metrics["name"].eq("Your Portfolio")].iloc[0].to_dict() if not metrics[metrics["name"].eq("Your Portfolio")].empty else {}
    b_row = metrics[metrics["name"].eq(standard_name)].iloc[0].to_dict() if not metrics[metrics["name"].eq(standard_name)].empty else {}
    interpretation = []
    if p_row:
        interpretation.append({"item": "Risk level", "reading": f"Trailing volatility {_safe_pct(p_row.get('annual_volatility', np.nan))}; beta vs {standard_name} {_safe_num(p_row.get('beta_vs_standard', np.nan), 2)}."})
        interpretation.append({"item": "Risk-adjusted return", "reading": f"Sharpe {_safe_num(p_row.get('sharpe', np.nan), 2)}; Sortino {_safe_num(p_row.get('sortino', np.nan), 2)}."})
        interpretation.append({"item": "Downside behavior", "reading": f"Max drawdown {_safe_pct(p_row.get('max_drawdown', np.nan))}; downside capture {_safe_pct(p_row.get('downside_capture', np.nan))}."})
        interpretation.append({"item": "Benchmark fit", "reading": f"Correlation vs {standard_name} {_safe_num(p_row.get('correlation_vs_standard', np.nan), 2)}; tracking error {_safe_pct(p_row.get('tracking_error', np.nan))}."})
        if not bollinger.empty and "percent_b" in bollinger.columns:
            latest_b = bollinger["percent_b"].dropna().iloc[-1] if not bollinger["percent_b"].dropna().empty else np.nan
            latest_width = bollinger["band_width"].dropna().iloc[-1] if not bollinger["band_width"].dropna().empty else np.nan
            interpretation.append({"item": "Bollinger position", "reading": f"Portfolio %B {_safe_num(latest_b, 2)}; band width {_safe_pct(latest_width)}. Near 1.0 means near the upper band; near 0.0 means near the lower band."})
    interpretation.append({"item": "How to read this", "reading": "This is a monthly comparison of the current portfolio mix against standard benchmark portfolios using trailing market data and normalized weights."})
    interp = pd.DataFrame(interpretation)
    interp.to_csv(latest / "monthly_interpretation.csv", index=False)

    key_metric_cols = [
        "name", "annual_return", "annual_volatility", "sharpe", "sortino",
        "max_drawdown", "beta_vs_standard", "tracking_error", "information_ratio",
        "upside_capture", "downside_capture", "correlation_vs_standard",
    ]
    key_metrics = metrics[[c for c in key_metric_cols if c in metrics.columns]].copy()
    key_metrics.to_csv(latest / "monthly_key_risk_metrics.csv", index=False)

    # Append archive row with no dollar values.
    archive_csv = out / "monthly_review_archive.csv"
    archive_row = {"month": label, "created_at": datetime.now().isoformat(timespec="seconds")}
    for k, v in p_row.items():
        if k not in {"name", "comparison_benchmark"}:
            archive_row[f"portfolio_{k}"] = v
    if b_row:
        for k, v in b_row.items():
            if k not in {"name", "comparison_benchmark"}:
                archive_row[f"benchmark_{k}"] = v
    arch = pd.DataFrame([archive_row])
    if archive_csv.exists():
        old = pd.read_csv(archive_csv)
        old = old[old.get("month", pd.Series(dtype=str)).astype(str) != label]
        arch = pd.concat([old, arch], ignore_index=True)
    arch.to_csv(archive_csv, index=False)

    # Write HTML.
    img_html = ""
    for title, path in [
        ("Portfolio by Asset Class", chart_paths.get("asset_class_pie")),
        ("Portfolio by Sector", chart_paths.get("sector_pie")),
        ("Normalized Growth", chart_paths.get("normalized_growth")),
        ("Drawdown", chart_paths.get("drawdown")),
        ("Risk vs Return", chart_paths.get("risk_return")),
        ("Efficient Frontier Style View", chart_paths.get("efficient_frontier")),
        ("Portfolio Bollinger Bands", chart_paths.get("bollinger")),
        ("Rolling 12-Month Sharpe", chart_paths.get("rolling_sharpe")),
        ("Rolling 3-Month Volatility", chart_paths.get("rolling_volatility")),
        ("Top Normalized Holdings", chart_paths.get("top_weights")),
    ]:
        if path and Path(path).exists():
            rel = Path("charts") / Path(path).name
            img_html += f"<h2>{title}</h2><img src='{rel.as_posix()}' alt='{title}'>\n"

    html = latest / "monthly_review_report.html"
    metric_cols_pct = ["annual_return", "annual_volatility", "max_drawdown", "best_day", "worst_day", "positive_day_pct", "tracking_error", "upside_capture", "downside_capture", "alpha_vs_standard"]
    metric_cols_num = ["sharpe", "sortino", "beta_vs_standard", "correlation_vs_standard", "information_ratio"]
    html.write_text(f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Monthly Portfolio Review {label}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:40px;max-width:1250px;line-height:1.45}}
table{{border-collapse:collapse;width:100%;margin:16px 0;font-size:14px}} th,td{{border:1px solid #ddd;padding:6px;text-align:left}} th{{background:#f4f4f4}}
img{{max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px;margin:8px 0 24px}}
.note{{background:#eef6ff;padding:12px;border-radius:8px}} .warn{{background:#fff8d6;padding:12px;border-radius:8px}}
</style></head><body>
<h1>Monthly Portfolio Review — {label}</h1>
<p class='note'>Shareable version: dollar amounts are intentionally omitted. Holdings are shown as normalized percentages only.</p>
<h2>Headline Interpretation</h2>{_html_table(interp)}
{img_html}
<h2>Risk / Opportunity Metrics</h2>{_html_table(metrics, pct_cols=metric_cols_pct, num_cols=metric_cols_num)}
<h2>Asset-Class Weights</h2>{_html_table(asset_weights.drop(columns=['weight']).rename(columns={'weight_pct':'weight'}), pct_cols=['weight'])}
<h2>Sector Weights</h2>{_html_table(sector_weights.drop(columns=['weight']).rename(columns={'weight_pct':'weight'}).head(25), pct_cols=['weight'])}
<h2>Top Holdings by Normalized Weight</h2>{_html_table(public_weights.head(30)[['symbol','display_name','asset_class_name','sector_clean','industry','weight']], pct_cols=['weight'])}
<h2>Standard Benchmark Weights</h2><p class='note'>Each benchmark portfolio below sums to 100%. The table lists several separate benchmark portfolios, so the entire table is not meant to sum to 100%.</p>{_html_table(pd.DataFrame(benchmark_weights_rows).assign(weight=lambda d: d['weight_pct']/100).drop(columns=['weight_pct']), pct_cols=['weight'])}
</body></html>""", encoding="utf-8")

    pdf = _write_monthly_pdf_report(
        latest / "monthly_review_report.pdf",
        label=label,
        standard_name=standard_name,
        chart_paths=chart_paths,
        interp=interp,
        metrics=metrics,
        asset_weights=asset_weights.copy(),
        sector_weights=sector_weights.copy(),
        public_weights=public_weights.copy(),
        benchmark_weights_rows=benchmark_weights_rows,
    )

    # Copy latest report outputs into archive folder for month-specific storage.
    for path in latest.iterdir():
        if path.is_file():
            target_path = archive_dir / path.name
            target_path.write_bytes(path.read_bytes())
    if charts.exists():
        archive_charts = archive_dir / "charts"
        archive_charts.mkdir(exist_ok=True)
        for path in charts.iterdir():
            if path.is_file():
                (archive_charts / path.name).write_bytes(path.read_bytes())

    return {
        "html": html,
        "pdf": pdf,
        "metrics": latest / "monthly_risk_opportunity_metrics.csv",
        "key_metrics": latest / "monthly_key_risk_metrics.csv",
        "public_weights": latest / "public_normalized_holdings_weights.csv",
        "sector_weights": latest / "public_sector_weights.csv",
        "asset_class_weights": latest / "public_asset_class_weights.csv",
        "benchmark_weights": latest / "public_standard_benchmark_weights.csv",
        "price_status": latest / "monthly_price_status.csv",
        "bollinger_bands": latest / "monthly_bollinger_bands.csv",
        "rolling_indicators": latest / "monthly_rolling_indicators.csv",
        "efficient_frontier_points": latest / "monthly_efficient_frontier_points.csv",
        "archive": archive_csv,
        "month_archive_dir": archive_dir,
    }
