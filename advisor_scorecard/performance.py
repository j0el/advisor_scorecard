from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .analysis import enrich_holdings, load_holdings


def _download_prices(symbols: list[str], years: int) -> pd.DataFrame:
    import yfinance as yf
    end = datetime.today()
    start = end - timedelta(days=int(years * 365.25) + 10)
    data = yf.download(symbols, start=start.date().isoformat(), end=end.date().isoformat(), auto_adjust=True, progress=False, group_by="column")
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        close = data.get("Close")
    else:
        close = data[["Close"]].rename(columns={"Close": symbols[0]})
    close = close.dropna(how="all")
    return close


def _metrics(returns: pd.Series, risk_free_rate: float = 0.0) -> dict:
    returns = returns.dropna()
    if returns.empty:
        return {}
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    total = (1 + returns).prod() - 1
    years = len(returns) / 252
    ann = (1 + total) ** (1 / years) - 1 if years > 0 else np.nan
    vol = returns.std() * np.sqrt(252)
    sharpe = excess.mean() / returns.std() * np.sqrt(252) if returns.std() else np.nan
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (ann - risk_free_rate) / downside if downside else np.nan
    wealth = (1 + returns).cumprod()
    dd = wealth / wealth.cummax() - 1
    return {"total_return": total, "annualized_return": ann, "annualized_volatility": vol, "sharpe": sharpe, "sortino": sortino, "max_drawdown": dd.min()}


def build_performance_report(holdings_path: str, config_path: str = "config.yaml", outdir: str = "reports", years: int = 3, risk_free_rate: float = 0.04) -> dict:
    out = Path(outdir)
    charts = out / "charts"
    out.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(Path(config_path).read_text()) if Path(config_path).exists() else {}
    h = enrich_holdings(load_holdings(holdings_path))
    h = h[h["market_value"] > 0].copy()
    total = h["market_value"].sum()
    h["weight"] = h["market_value"] / total

    symbols = sorted(h["symbol"].dropna().unique().tolist())
    prices = _download_prices(symbols, years)
    valid = [s for s in symbols if s in prices.columns and prices[s].dropna().shape[0] > 60]
    prices = prices[valid].dropna(how="all").ffill().dropna()
    weights = h.set_index("symbol")["weight"].reindex(valid).fillna(0)
    weights = weights / weights.sum() if weights.sum() else weights
    rets = prices.pct_change().dropna()
    port_ret = (rets * weights).sum(axis=1)

    # Benchmark from target allocation.
    target = config.get("target_allocation", {}) or {}
    bench_map = config.get("benchmark", {}) or {"us_stock": "VTI", "intl_stock": "VXUS", "bond": "BND", "cash": "SGOV", "other": "VNQ"}
    bench_symbols = [bench_map[k] for k, v in target.items() if v and k in bench_map]
    bench_ret = None
    if bench_symbols:
        bp = _download_prices(sorted(set(bench_symbols)), years).ffill().dropna()
        if not bp.empty:
            bw = pd.Series({bench_map[k]: float(v) for k, v in target.items() if v and k in bench_map})
            bw = bw.reindex(bp.columns).fillna(0)
            bw = bw / bw.sum() if bw.sum() else bw
            bench_ret = (bp.pct_change().dropna() * bw).sum(axis=1)

    metrics = []
    metrics.append({"series": "Current holdings structure", **_metrics(port_ret, risk_free_rate)})
    if bench_ret is not None and not bench_ret.empty:
        common = port_ret.index.intersection(bench_ret.index)
        metrics.append({"series": "Target benchmark", **_metrics(bench_ret.loc[common], risk_free_rate)})
        active = port_ret.loc[common] - bench_ret.loc[common]
        metrics.append({"series": "Active vs benchmark", "annualized_return": active.mean() * 252, "annualized_volatility": active.std() * np.sqrt(252), "sharpe": np.nan, "sortino": np.nan, "max_drawdown": np.nan, "total_return": np.nan})

    mdf = pd.DataFrame(metrics)
    mdf.to_csv(out / "performance_metrics.csv", index=False)

    import matplotlib.pyplot as plt
    wealth = (1 + port_ret).cumprod()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(wealth.index, wealth.values, label="Current holdings structure")
    if bench_ret is not None and not bench_ret.empty:
        common = wealth.index.intersection(bench_ret.index)
        ax.plot(common, (1 + bench_ret.loc[common]).cumprod(), label="Target benchmark")
    ax.set_title("Growth of $1")
    ax.legend()
    fig.tight_layout(); fig.savefig(charts / "portfolio_vs_benchmark.png", dpi=160); plt.close(fig)

    dd = wealth / wealth.cummax() - 1
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dd.index, dd.values)
    ax.set_title("Drawdown")
    fig.tight_layout(); fig.savefig(charts / "drawdown.png", dpi=160); plt.close(fig)

    # Simple random efficient frontier approximation over valid holdings.
    if len(valid) >= 2:
        mean = rets[valid].mean() * 252
        cov = rets[valid].cov() * 252
        n = min(len(valid), 35)  # keep readable and fast
        use = valid[:n]
        rng = np.random.default_rng(42)
        pts = []
        for _ in range(1500):
            w = rng.random(n); w = w / w.sum()
            ret = float(np.dot(w, mean.reindex(use).fillna(0)))
            vol = float(np.sqrt(np.dot(w, np.dot(cov.reindex(index=use, columns=use).fillna(0), w))))
            pts.append((vol, ret))
        pts = pd.DataFrame(pts, columns=["volatility", "return"])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(pts["volatility"], pts["return"], s=8, alpha=0.35)
        ax.scatter([port_ret.std()*np.sqrt(252)], [port_ret.mean()*252], marker="x", s=100, label="Current")
        ax.set_title("Exploratory Efficient Frontier")
        ax.set_xlabel("Annualized Volatility")
        ax.set_ylabel("Annualized Return")
        ax.legend()
        fig.tight_layout(); fig.savefig(charts / "efficient_frontier.png", dpi=160); plt.close(fig)

    rows = "".join(f"<tr>{''.join(f'<td>{v:.4f}</td>' if isinstance(v, (float, np.floating)) and pd.notna(v) else f'<td>{v}</td>' for v in row)}</tr>" for row in mdf.itertuples(index=False, name=None))
    heads = "".join(f"<th>{c}</th>" for c in mdf.columns)
    html = out / "performance_report.html"
    html.write_text(f"""<!doctype html><html><head><meta charset='utf-8'><title>Performance Report</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:40px;max-width:1100px}} table{{border-collapse:collapse}} th,td{{border:1px solid #ddd;padding:6px}} img{{max-width:100%;border:1px solid #ddd;border-radius:8px}}</style></head><body>
<h1>Performance Exploration</h1>
<p>This uses today's holdings weights and historical prices. It is not yet your true personal return.</p>
<h2>Metrics</h2><table><tr>{heads}</tr>{rows}</table>
<h2>Growth of $1</h2><img src='charts/portfolio_vs_benchmark.png'>
<h2>Drawdown</h2><img src='charts/drawdown.png'>
<h2>Efficient Frontier</h2><img src='charts/efficient_frontier.png'>
</body></html>""", encoding="utf-8")
    return {"performance_html": html, "metrics": out / "performance_metrics.csv"}
