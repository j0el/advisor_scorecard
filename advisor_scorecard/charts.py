from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_portfolio_charts(enriched: pd.DataFrame, alloc: pd.DataFrame, sectors: pd.DataFrame, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # Asset allocation pie
    a = alloc[alloc["market_value"] > 0].copy()
    if not a.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.pie(a["market_value"], labels=a["asset_class_name"], autopct="%1.1f%%", startangle=90)
        ax.set_title("Asset Allocation")
        _save(fig, outdir / "asset_allocation_pie.png")

    # Sector pie: keep top 10, group rest.
    s = sectors[sectors["market_value"] > 0].copy()
    if not s.empty:
        top = s.head(10).copy()
        rest_value = s.iloc[10:]["market_value"].sum()
        if rest_value > 0:
            top = pd.concat([top, pd.DataFrame([{"sector_clean": "Other sectors", "market_value": rest_value}])], ignore_index=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(top["market_value"], labels=top["sector_clean"], autopct="%1.1f%%", startangle=90)
        ax.set_title("Sector Allocation")
        _save(fig, outdir / "sector_pie.png")

    # Top holdings bar
    h = enriched.sort_values("market_value", ascending=False).head(15).copy()
    if not h.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        labels = h["symbol"].astype(str) + " - " + h["display_name"].astype(str).str.slice(0, 28)
        ax.barh(labels[::-1], h["market_value"][::-1])
        ax.set_title("Top Holdings")
        ax.set_xlabel("Market Value")
        _save(fig, outdir / "top_holdings_bar.png")

    # Actual vs target allocation
    if not a.empty and "target_pct" in a.columns:
        x = range(len(a))
        fig, ax = plt.subplots(figsize=(8, 5))
        width = 0.35
        ax.bar([i - width / 2 for i in x], a["actual_pct"], width, label="Actual")
        ax.bar([i + width / 2 for i in x], a["target_pct"], width, label="Target")
        ax.set_xticks(list(x))
        ax.set_xticklabels(a["asset_class_name"], rotation=25, ha="right")
        ax.set_title("Actual vs Target Allocation")
        ax.set_ylabel("Portfolio Weight")
        ax.legend()
        _save(fig, outdir / "allocation_vs_target.png")
