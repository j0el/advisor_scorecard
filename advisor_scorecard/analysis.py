from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .classify import classify_from_metadata, friendly_asset_name, sector_from_metadata, clean_symbol, clean_text
from .ticker_metadata import get_metadata


def load_config(path: str = "config.yaml") -> dict:
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def normalize_holdings_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mapping = {}
    lower = {c.lower().strip(): c for c in out.columns}

    def pick(target, candidates):
        for c in candidates:
            if c.lower() in lower:
                mapping[lower[c.lower()]] = target
                return

    pick("symbol", ["symbol", "ticker", "instrument_symbol", "cusip"])
    pick("description", ["description", "company_name", "long_name", "name", "instrument_description"])
    pick("asset_type", ["asset_type", "asset type", "security_type", "type", "instrument_asset_type"])
    pick("sector", ["sector", "sector_clean"])
    pick("market_value", ["market_value", "market value", "value", "current_value", "long_market_value"])
    pick("quantity", ["quantity", "qty", "long_quantity"])
    pick("cost_basis", ["cost_basis", "cost basis", "average_long_price", "average_price", "average cost", "average_cost"])
    if mapping:
        out = out.rename(columns=mapping)
    for col in ["symbol", "description", "asset_type", "sector", "market_value", "quantity", "cost_basis"]:
        if col not in out.columns:
            out[col] = np.nan
    out["symbol"] = out["symbol"].map(clean_symbol)
    out["market_value"] = pd.to_numeric(out["market_value"], errors="coerce").fillna(0.0)
    out["quantity"] = pd.to_numeric(out["quantity"], errors="coerce")
    out["cost_basis"] = pd.to_numeric(out["cost_basis"], errors="coerce")
    return out


def load_holdings(path: str) -> pd.DataFrame:
    return normalize_holdings_columns(pd.read_csv(path))


def demo_holdings() -> pd.DataFrame:
    return normalize_holdings_columns(pd.DataFrame([
        {"symbol": "VTI", "description": "Vanguard Total Stock Market ETF", "asset_type": "ETF", "market_value": 350000},
        {"symbol": "VXUS", "description": "Vanguard Total International Stock ETF", "asset_type": "ETF", "market_value": 120000},
        {"symbol": "BND", "description": "Vanguard Total Bond Market ETF", "asset_type": "ETF", "market_value": 180000},
        {"symbol": "SGOV", "description": "iShares 0-3 Month Treasury Bond ETF", "asset_type": "ETF", "market_value": 50000},
        {"symbol": "AAPL", "description": "EQUITY", "asset_type": "EQUITY", "market_value": 50000},
        {"symbol": "MSFT", "description": "EQUITY", "asset_type": "EQUITY", "market_value": 45000},
    ]))


def enrich_holdings(df: pd.DataFrame, refresh_metadata: bool = False) -> pd.DataFrame:
    out = normalize_holdings_columns(df)
    meta = get_metadata(out["symbol"].dropna().unique(), refresh=refresh_metadata)
    out = out.merge(meta, on="symbol", how="left")

    out["company_name"] = out["company_name"].fillna("")
    out["display_name"] = np.where(
        out["company_name"].astype(str).str.len() > 0,
        out["company_name"],
        np.where(out["description"].astype(str).str.upper().eq("EQUITY"), out["symbol"], out["description"].astype(str))
    )

    out["asset_class"] = [
        classify_from_metadata(
            row.symbol,
            row.asset_type,
            row.sector_y if hasattr(row, "sector_y") else getattr(row, "sector", ""),
            row.industry,
            row.quote_type,
            row.fund_category,
            row.country,
            row.fund_family,
        )
        for row in out.itertuples()
    ]
    out["asset_class_name"] = out["asset_class"].map(friendly_asset_name)

    # If both original and metadata sector exist, prefer metadata sector.
    meta_sector = out["sector_y"] if "sector_y" in out.columns else out.get("sector", pd.Series("", index=out.index))
    orig_sector = out["sector_x"] if "sector_x" in out.columns else out.get("sector", pd.Series("", index=out.index))
    out["sector_clean"] = [
        sector_from_metadata(ms, row.industry, row.fund_category) if clean_text(ms) else sector_from_metadata(os, row.industry, row.fund_category)
        for ms, os, row in zip(meta_sector, orig_sector, out.itertuples())
    ]

    total = out["market_value"].sum()
    out["weight"] = out["market_value"] / total if total else 0
    out["unrealized_gain"] = np.where(out["cost_basis"].notna(), out["market_value"] - out["cost_basis"], np.nan)
    out["unrealized_gain_pct"] = np.where(out["cost_basis"] > 0, out["unrealized_gain"] / out["cost_basis"], np.nan)
    return out


def allocation_summary(enriched: pd.DataFrame, config: dict) -> pd.DataFrame:
    alloc = enriched.groupby(["asset_class", "asset_class_name"], dropna=False)["market_value"].sum().reset_index()
    total = alloc["market_value"].sum()
    alloc["actual_pct"] = alloc["market_value"] / total if total else 0
    targets = config.get("target_allocation", {}) or {}
    alloc["target_pct"] = alloc["asset_class"].map(lambda x: float(targets.get(x, 0.0)))
    alloc["difference_pct"] = alloc["actual_pct"] - alloc["target_pct"]
    return alloc.sort_values("market_value", ascending=False)


def concentration(enriched: pd.DataFrame) -> pd.DataFrame:
    cols = ["symbol", "display_name", "asset_class_name", "sector_clean", "market_value", "weight"]
    return enriched[cols].sort_values("market_value", ascending=False)


def sector_summary(enriched: pd.DataFrame) -> pd.DataFrame:
    sec = enriched.groupby("sector_clean", dropna=False)["market_value"].sum().reset_index()
    total = sec["market_value"].sum()
    sec["weight"] = sec["market_value"] / total if total else 0
    return sec.sort_values("market_value", ascending=False)


def fee_summary(enriched: pd.DataFrame) -> pd.DataFrame:
    fee_bps = float(os.getenv("ADVISOR_FEE_BPS", "100"))
    total = enriched["market_value"].sum()
    annual_fee = total * fee_bps / 10000.0
    return pd.DataFrame([{
        "portfolio_value": total,
        "advisor_fee_bps": fee_bps,
        "estimated_annual_advisor_fee": annual_fee,
        "estimated_monthly_advisor_fee": annual_fee / 12.0,
    }])


def write_advisor_questions(path: Path) -> None:
    path.write_text("""# Questions for Advisor Review

1. What is my written target allocation?
2. What benchmark should I use to evaluate this portfolio?
3. What is my all-in annual cost, including advisory fee and fund expenses?
4. Which holdings are intended to reduce risk, and which are intended to outperform?
5. How often do you rebalance, and what triggers a rebalance?
6. Are there embedded tax gains that make the current holdings hard to change?
7. Am I taking concentration risk that I may not realize?
8. What would you change if I needed lower risk or more income?
""", encoding="utf-8")


def write_html_report(enriched: pd.DataFrame, alloc: pd.DataFrame, sectors: pd.DataFrame, fees: pd.DataFrame, outdir: Path) -> Path:
    html = outdir / "portfolio_report.html"
    charts = outdir / "charts"
    def fmt_pct(s):
        return (s * 100).map(lambda x: f"{x:.2f}%")
    alloc_display = alloc.copy()
    for c in ["actual_pct", "target_pct", "difference_pct"]:
        alloc_display[c] = fmt_pct(alloc_display[c])
    conc_display = concentration(enriched).head(25).copy()
    conc_display["weight"] = fmt_pct(conc_display["weight"])
    sector_display = sectors.head(25).copy()
    sector_display["weight"] = fmt_pct(sector_display["weight"])

    img_tags = ""
    for img in ["asset_allocation_pie.png", "sector_pie.png", "top_holdings_bar.png", "allocation_vs_target.png"]:
        if (charts / img).exists():
            img_tags += f'<h2>{img.replace("_", " ").replace(".png", "").title()}</h2><img src="charts/{img}" style="max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px;">'

    html.write_text(f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Advisor Scorecard</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:40px;max-width:1200px}} table{{border-collapse:collapse;width:100%;margin:16px 0}} th,td{{border:1px solid #ddd;padding:6px;text-align:left}} th{{background:#f4f4f4}} .note{{background:#fff8d6;padding:12px;border-radius:8px}}</style>
</head><body>
<h1>Advisor Scorecard</h1>
<p class='note'>Read-only analysis. Not investment advice. Verify classifications and benchmark assumptions.</p>
<h2>Portfolio Value</h2>
<p><strong>${enriched['market_value'].sum():,.2f}</strong></p>
{img_tags}
<h2>Asset Allocation</h2>{alloc_display.to_html(index=False)}
<h2>Top Holdings</h2>{conc_display.to_html(index=False)}
<h2>Sector Summary</h2>{sector_display.to_html(index=False)}
<h2>Estimated Advisor Fee</h2>{fees.to_html(index=False)}
</body></html>""", encoding="utf-8")
    return html


def write_outputs(holdings: pd.DataFrame, config: dict, outdir: str = "reports", refresh_metadata: bool = False) -> dict:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "charts").mkdir(exist_ok=True)
    enriched = enrich_holdings(holdings, refresh_metadata=refresh_metadata)
    alloc = allocation_summary(enriched, config)
    sectors = sector_summary(enriched)
    conc = concentration(enriched)
    fees = fee_summary(enriched)

    enriched.to_csv(out / "holdings_enriched.csv", index=False)
    alloc.to_csv(out / "allocation_summary.csv", index=False)
    sectors.to_csv(out / "sector_summary.csv", index=False)
    conc.to_csv(out / "concentration.csv", index=False)
    fees.to_csv(out / "fee_summary.csv", index=False)
    write_advisor_questions(out / "advisor_questions.md")

    try:
        from .charts import make_portfolio_charts
        make_portfolio_charts(enriched, alloc, sectors, out / "charts")
    except Exception as e:
        (out / "chart_error.txt").write_text(str(e), encoding="utf-8")

    html = write_html_report(enriched, alloc, sectors, fees, out)
    return {
        "html": html,
        "holdings_enriched": out / "holdings_enriched.csv",
        "allocation": out / "allocation_summary.csv",
        "sectors": out / "sector_summary.csv",
        "concentration": out / "concentration.csv",
        "fees": out / "fee_summary.csv",
        "questions": out / "advisor_questions.md",
    }
