#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _clean_symbol(s: object) -> str:
    if s is None or pd.isna(s):
        return ""
    return str(s).strip().upper().replace("/", "-").replace(".", "-")


def _display_symbol(s: object) -> str:
    if s is None or pd.isna(s):
        return ""
    return str(s).strip().upper()


def _today_str() -> str:
    return date.today().isoformat()


def _pct_to_fraction(series: pd.Series) -> pd.Series:
    v = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if v.abs().max() > 1.5:
        v = v / 100.0
    return v


def _normalize_weights(w: pd.Series) -> pd.Series:
    w = pd.to_numeric(w, errors="coerce").fillna(0.0)
    w = w[w > 0]
    total = w.sum()
    if total <= 0:
        return w
    return w / total


def load_current_weights(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "symbol" not in df.columns:
        raise ValueError(f"{path} must contain a 'symbol' column.")
    weight_col = None
    for c in df.columns:
        lc = c.lower()
        if c == "portfolio_weight_pct" or ("weight" in lc and ("pct" in lc or "percent" in lc)):
            weight_col = c
            break
    if weight_col is None:
        raise ValueError(f"Could not find a portfolio weight column in {path}. Columns: {list(df.columns)}")
    out = df.copy()
    out["symbol_original"] = out["symbol"].map(_display_symbol)
    out["symbol_yahoo"] = out["symbol"].map(_clean_symbol)
    out["weight"] = _pct_to_fraction(out[weight_col])
    agg = {"weight": "sum", "symbol_original": "first"}
    for optional in ["display_name", "asset_class_name", "sector_clean", "industry"]:
        if optional in out.columns:
            agg[optional] = "first"
    out = out.groupby("symbol_yahoo", as_index=False).agg(agg)
    weights = _normalize_weights(out.set_index("symbol_yahoo")["weight"])
    out["weight"] = out["symbol_yahoo"].map(weights).fillna(0.0)
    return out[out["weight"] > 0].sort_values("weight", ascending=False).reset_index(drop=True)


def _find_col(columns: Iterable[str], must: Iterable[str] = (), any_of: Iterable[str] = (), avoid: Iterable[str] = ()) -> Optional[str]:
    for c in list(columns):
        lc = c.lower()
        if all(m.lower() in lc for m in must) and (not any_of or any(a.lower() in lc for a in any_of)) and not any(a.lower() in lc for a in avoid):
            return c
    return None


def load_model_weights_from_rebalance(path: Path, current: pd.DataFrame, model_name: str) -> pd.DataFrame:
    rb = pd.read_csv(path)
    if "symbol" not in rb.columns:
        raise ValueError(f"{path} must contain a 'symbol' column. Columns: {list(rb.columns)}")
    rb = rb.copy()
    rb["symbol_yahoo"] = rb["symbol"].map(_clean_symbol)
    model_tokens = [t for t in re.split(r"[^A-Za-z0-9]+", model_name.lower()) if t]
    model_tokens = [t for t in model_tokens if t not in {"optimized", "portfolio"}] or ["max", "sharpe"]
    cols = list(rb.columns)
    current_col = (_find_col(cols, must=["current"], any_of=["weight", "pct"]) or
                   _find_col(cols, must=["your"], any_of=["weight", "pct"]))
    target_col = None
    for c in cols:
        lc = c.lower()
        if all(t in lc for t in model_tokens) and "change" not in lc and "delta" not in lc and "pp" not in lc:
            target_col = c
            break
    if target_col is None:
        target_col = (_find_col(cols, must=["target"], any_of=["weight", "pct"], avoid=["change"]) or
                      _find_col(cols, must=["optimized"], any_of=["weight", "pct"], avoid=["change"]) or
                      _find_col(cols, must=["candidate"], any_of=["weight", "pct"], avoid=["change"]))
    change_col = None
    for c in cols:
        lc = c.lower()
        if ("change" in lc or "delta" in lc or "pp" in lc) and all(t in lc for t in model_tokens):
            change_col = c
            break
    if change_col is None:
        change_col = (_find_col(cols, must=["change"], any_of=["weight", "pct", "pp"]) or
                      _find_col(cols, must=["delta"], any_of=["weight", "pct", "pp"]))
    rb_small = rb[["symbol_yahoo"]].copy()
    if target_col is not None:
        rb_small["model_weight"] = _pct_to_fraction(rb[target_col])
    elif current_col is not None and change_col is not None:
        cur = _pct_to_fraction(rb[current_col])
        change = pd.to_numeric(rb[change_col], errors="coerce").fillna(0.0)
        if change.abs().max() > 1.5:
            change = change / 100.0
        rb_small["model_weight"] = cur + change
    else:
        raise ValueError("Could not infer Max Sharpe target weights from rebalance file.\n" +
                         f"File: {path}\nColumns: {list(rb.columns)}\n" +
                         "Expected either a Max Sharpe weight column or current weight + change column.")
    rb_small = rb_small.groupby("symbol_yahoo", as_index=False)["model_weight"].sum()
    rb_small["model_weight"] = rb_small["model_weight"].clip(lower=0.0)
    weights = _normalize_weights(rb_small.set_index("symbol_yahoo")["model_weight"])
    rb_small["model_weight"] = rb_small["symbol_yahoo"].map(weights).fillna(0.0)
    meta = current.drop_duplicates("symbol_yahoo").set_index("symbol_yahoo")
    rb_small["symbol_original"] = rb_small["symbol_yahoo"].map(lambda x: meta["symbol_original"].get(x, x) if "symbol_original" in meta else x)
    for optional in ["display_name", "asset_class_name", "sector_clean", "industry"]:
        if optional in current.columns:
            rb_small[optional] = rb_small["symbol_yahoo"].map(lambda x: meta[optional].get(x, "") if optional in meta else "")
    return rb_small[rb_small["model_weight"] > 0].sort_values("model_weight", ascending=False).reset_index(drop=True)


def save_weights_csv(df: pd.DataFrame, path: Path, weight_col: str) -> None:
    cols = ["symbol_original", "symbol_yahoo", weight_col]
    for c in ["display_name", "asset_class_name", "sector_clean", "industry"]:
        if c in df.columns:
            cols.append(c)
    out = df[cols].copy()
    out[weight_col.replace("weight", "weight_pct")] = out[weight_col] * 100
    out = out.drop(columns=[weight_col])
    out.to_csv(path, index=False)


def cmd_start(args: argparse.Namespace) -> None:
    current_path = Path(args.current_weights)
    rebalance_path = Path(args.rebalance_changes)
    current = load_current_weights(current_path)
    model = load_model_weights_from_rebalance(rebalance_path, current, args.model_name)
    start_date = args.start_date or _today_str()
    outdir = Path(args.output_root) / start_date
    outdir.mkdir(parents=True, exist_ok=True)
    save_weights_csv(current.rename(columns={"weight": "current_weight"}), outdir / "current_portfolio_weights.csv", "current_weight")
    save_weights_csv(model.rename(columns={"model_weight": "model_weight"}), outdir / "max_sharpe_weights.csv", "model_weight")
    manifest = {
        "start_date": start_date,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "current_weights_source": str(current_path),
        "model_weights_source": str(rebalance_path),
        "model_name": args.model_name,
        "current_weight_sum": float(current["weight"].sum()),
        "model_weight_sum": float(model["model_weight"].sum()),
        "notes": "Forward test stores normalized weights only. No dollar amounts are used.",
    }
    (outdir / "forward_test_start.json").write_text(json.dumps(manifest, indent=2))
    print(f"Forward test started: {outdir}")
    print(f"  current weights: {outdir / 'current_portfolio_weights.csv'}")
    print(f"  model weights:   {outdir / 'max_sharpe_weights.csv'}")
    print("\nOne month later, run:")
    print(f"  uv run python3 portfolio_forward_test.py compare --test-date {start_date}")


def _latest_test_dir(root: Path) -> Path:
    dirs = sorted([p for p in root.iterdir() if p.is_dir() and (p / "forward_test_start.json").exists()])
    if not dirs:
        raise FileNotFoundError(f"No forward tests found under {root}")
    return dirs[-1]


def _load_saved_weights(path: Path, weight_hint: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "symbol_yahoo" not in df.columns:
        if "symbol" in df.columns:
            df["symbol_yahoo"] = df["symbol"].map(_clean_symbol)
        elif "symbol_original" in df.columns:
            df["symbol_yahoo"] = df["symbol_original"].map(_clean_symbol)
        else:
            raise ValueError(f"{path} lacks symbol_yahoo/symbol columns.")
    weight_col = None
    for c in df.columns:
        lc = c.lower()
        if weight_hint in lc and ("pct" in lc or "weight" in lc):
            weight_col = c
            break
    if weight_col is None:
        for c in df.columns:
            lc = c.lower()
            if "weight" in lc or "pct" in lc:
                weight_col = c
                break
    if weight_col is None:
        raise ValueError(f"No weight column found in {path}. Columns: {list(df.columns)}")
    df["weight"] = _pct_to_fraction(df[weight_col])
    df = df.groupby("symbol_yahoo", as_index=False)["weight"].sum()
    weights = _normalize_weights(df.set_index("symbol_yahoo")["weight"])
    df["weight"] = df["symbol_yahoo"].map(weights).fillna(0.0)
    return df[df["weight"] > 0]


def download_prices(symbols: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    import yfinance as yf
    status_rows = []
    series = []
    for sym in symbols:
        if sym == "CASH":
            idx = pd.date_range(start=start, end=end, freq="B")
            series.append(pd.Series(1.0, index=idx, name=sym))
            status_rows.append({"symbol": sym, "status": "flat_cash_proxy", "rows": len(idx)})
            continue
        try:
            data = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False, threads=False)
            if data is None or data.empty:
                status_rows.append({"symbol": sym, "status": "no_price_data", "rows": 0})
                continue
            close = data["Close"].iloc[:, 0] if isinstance(data.columns, pd.MultiIndex) else data["Close"]
            close = close.dropna()
            if close.empty:
                status_rows.append({"symbol": sym, "status": "empty_close", "rows": 0})
                continue
            close.name = sym
            series.append(close)
            status_rows.append({"symbol": sym, "status": "ok", "rows": len(close)})
        except Exception as e:
            status_rows.append({"symbol": sym, "status": f"error: {type(e).__name__}: {e}", "rows": 0})
    if not series:
        return pd.DataFrame(), pd.DataFrame(status_rows)
    return pd.concat(series, axis=1).sort_index().ffill().dropna(how="all"), pd.DataFrame(status_rows)


def portfolio_index(prices: pd.DataFrame, weights: pd.Series) -> pd.Series:
    valid = [s for s in weights.index if s in prices.columns]
    if not valid:
        return pd.Series(dtype=float)
    px = prices[valid].copy().ffill().dropna(how="all").dropna(axis=1, how="all")
    valid = list(px.columns)
    w = _normalize_weights(weights.reindex(valid).fillna(0.0))
    norm = px / px.iloc[0]
    idx = (norm * w).sum(axis=1)
    return idx / idx.iloc[0] * 100.0


def perf_stats(idx: pd.Series) -> Dict[str, float]:
    idx = idx.dropna()
    if len(idx) < 2:
        return {"return_pct": np.nan, "volatility_ann_pct": np.nan, "max_drawdown_pct": np.nan, "days": len(idx)}
    ret = idx.pct_change().dropna()
    total_return = idx.iloc[-1] / idx.iloc[0] - 1.0
    vol = ret.std() * math.sqrt(252)
    dd = idx / idx.cummax() - 1.0
    return {"return_pct": total_return * 100.0, "volatility_ann_pct": vol * 100.0, "max_drawdown_pct": dd.min() * 100.0, "days": int(len(idx))}


def contribution_table(prices: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    rows = []
    for sym, w in weights.items():
        if sym not in prices.columns:
            continue
        s = prices[sym].dropna()
        if len(s) < 2:
            continue
        r = s.iloc[-1] / s.iloc[0] - 1.0
        rows.append({"symbol": sym, "weight_pct": w * 100.0, "period_return_pct": r * 100.0, "contribution_pct": w * r * 100.0})
    return pd.DataFrame(rows).sort_values("contribution_pct", ascending=False)


def make_charts(current_idx: pd.Series, model_idx: pd.Series, outdir: Path) -> Dict[str, Path]:
    import matplotlib.pyplot as plt
    charts = {}
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    current_idx.plot(label="Your Current Portfolio")
    model_idx.plot(label="Max Sharpe Model")
    plt.title("Forward Test: Normalized Growth of 100")
    plt.ylabel("Normalized value")
    plt.legend()
    plt.tight_layout()
    p = outdir / "forward_growth.png"
    plt.savefig(p, dpi=160)
    plt.close()
    charts["growth"] = p
    plt.figure(figsize=(10, 5))
    (current_idx / current_idx.cummax() - 1.0).mul(100).plot(label="Your Current Portfolio")
    (model_idx / model_idx.cummax() - 1.0).mul(100).plot(label="Max Sharpe Model")
    plt.title("Forward Test: Drawdown")
    plt.ylabel("Drawdown (%)")
    plt.legend()
    plt.tight_layout()
    p = outdir / "forward_drawdown.png"
    plt.savefig(p, dpi=160)
    plt.close()
    charts["drawdown"] = p
    return charts


def make_html_report(outdir: Path, start_date: str, end_date: str, metrics: pd.DataFrame, contrib_diff: pd.DataFrame, charts: Dict[str, Path]) -> Path:
    html = ["<html><head><meta charset='utf-8'>",
            "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:32px;line-height:1.4;color:#222}h1,h2{margin-bottom:.3rem}table{border-collapse:collapse;width:100%;font-size:14px;margin:14px 0 28px}th,td{border:1px solid #ddd;padding:6px 8px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#f3f5f7}.note{background:#f7f7f7;border-left:4px solid #999;padding:10px 14px;margin:12px 0}img{max-width:100%;border:1px solid #ddd;margin:10px 0 28px}</style></head><body>",
            "<h1>Portfolio Forward Test</h1>", f"<p><b>Period:</b> {start_date} to {end_date}</p>",
            "<div class='note'>This is an out-of-sample comparison using saved normalized weights. It uses no dollar amounts.</div>",
            "<h2>Results</h2>", metrics.to_html(index=False, float_format=lambda x: f"{x:,.2f}")]
    for name, path in charts.items():
        html.append(f"<h2>{name.title()}</h2>")
        html.append(f"<img src='{path.relative_to(outdir).as_posix()}'>")
    html += ["<h2>Contribution Difference: Model minus Current</h2>",
             "<p>Positive values helped the Max Sharpe model relative to your current portfolio. Negative values hurt it.</p>",
             contrib_diff.head(30).to_html(index=False, float_format=lambda x: f"{x:,.2f}"), "</body></html>"]
    out = outdir / "forward_test_comparison.html"
    out.write_text("\n".join(html))
    return out


def make_pdf_report(html_path: Path, pdf_path: Path) -> bool:
    try:
        from weasyprint import HTML
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        return True
    except Exception:
        return False


def cmd_compare(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    test_dir = _latest_test_dir(root) if args.latest else root / args.test_date
    manifest_path = test_dir / "forward_test_start.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    start_date = manifest["start_date"]
    end_date = args.end_date or _today_str()
    current = _load_saved_weights(test_dir / "current_portfolio_weights.csv", "current")
    model = _load_saved_weights(test_dir / "max_sharpe_weights.csv", "model")
    all_symbols = sorted(set(current["symbol_yahoo"]).union(set(model["symbol_yahoo"])))
    end_plus = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).date().isoformat()
    prices, status = download_prices(all_symbols, start_date, end_plus)
    status.to_csv(test_dir / "forward_price_status.csv", index=False)
    cur_w = current.set_index("symbol_yahoo")["weight"]
    mod_w = model.set_index("symbol_yahoo")["weight"]
    cur_idx = portfolio_index(prices, cur_w)
    mod_idx = portfolio_index(prices, mod_w)
    if cur_idx.empty or mod_idx.empty:
        raise RuntimeError("Could not build one or both portfolio indexes. Check forward_price_status.csv.")
    combined = pd.concat({"Current Portfolio": cur_idx, "Max Sharpe Model": mod_idx}, axis=1).dropna()
    cur_idx = combined["Current Portfolio"]
    mod_idx = combined["Max Sharpe Model"]
    metrics = pd.DataFrame([{"portfolio": "Your Current Portfolio", **perf_stats(cur_idx)}, {"portfolio": "Max Sharpe Model", **perf_stats(mod_idx)}])
    diff_row = {"portfolio": "Difference: Model - Current", "return_pct": metrics.loc[1, "return_pct"] - metrics.loc[0, "return_pct"], "volatility_ann_pct": metrics.loc[1, "volatility_ann_pct"] - metrics.loc[0, "volatility_ann_pct"], "max_drawdown_pct": metrics.loc[1, "max_drawdown_pct"] - metrics.loc[0, "max_drawdown_pct"], "days": int(metrics.loc[0, "days"])}
    metrics = pd.concat([metrics, pd.DataFrame([diff_row])], ignore_index=True)
    metrics.to_csv(test_dir / "forward_test_metrics.csv", index=False)
    cur_contrib = contribution_table(prices, cur_w).rename(columns={"weight_pct": "current_weight_pct", "contribution_pct": "current_contribution_pct"})
    mod_contrib = contribution_table(prices, mod_w).rename(columns={"weight_pct": "model_weight_pct", "contribution_pct": "model_contribution_pct"})[["symbol", "model_weight_pct", "model_contribution_pct"]]
    contrib = pd.merge(cur_contrib, mod_contrib, on="symbol", how="outer").fillna(0.0)
    contrib["contribution_difference_pct"] = contrib["model_contribution_pct"] - contrib["current_contribution_pct"]
    contrib = contrib.sort_values("contribution_difference_pct", ascending=False)
    contrib.to_csv(test_dir / "forward_test_contributions.csv", index=False)
    pd.concat({"current_index": cur_idx, "model_index": mod_idx}, axis=1).to_csv(test_dir / "forward_test_daily_indexes.csv")
    charts = make_charts(cur_idx, mod_idx, test_dir / "charts")
    html = make_html_report(test_dir, start_date, end_date, metrics, contrib, charts)
    pdf = test_dir / "forward_test_comparison.pdf"
    made_pdf = make_pdf_report(html, pdf)
    print(f"Forward test comparison complete: {test_dir}")
    print(f"  metrics:       {test_dir / 'forward_test_metrics.csv'}")
    print(f"  contributions: {test_dir / 'forward_test_contributions.csv'}")
    print(f"  html:          {html}")
    if made_pdf:
        print(f"  pdf:           {pdf}")
    else:
        print("  pdf:           not created; install/use WeasyPrint dependencies if needed")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Forward-test current portfolio weights against saved Max Sharpe model weights.")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("start", help="Save current and Max Sharpe weights for a future comparison.")
    s.add_argument("--current-weights", default="reports/monthly_review/public_normalized_holdings_weights.csv")
    s.add_argument("--rebalance-changes", default="reports/forecast_optimizer/forecast_rebalance_changes.csv")
    s.add_argument("--model-name", default="Optimized: Max Sharpe")
    s.add_argument("--start-date", default=None, help="YYYY-MM-DD; defaults to today.")
    s.add_argument("--output-root", default="reports/forward_tests")
    s.set_defaults(func=cmd_start)
    c = sub.add_parser("compare", help="Compare saved weights over the elapsed forward-test period.")
    c.add_argument("--test-date", default=None, help="YYYY-MM-DD folder under reports/forward_tests.")
    c.add_argument("--latest", action="store_true", help="Use the latest saved forward test.")
    c.add_argument("--end-date", default=None, help="YYYY-MM-DD; defaults to today.")
    c.add_argument("--output-root", default="reports/forward_tests")
    c.set_defaults(func=cmd_compare)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "compare" and not args.latest and not args.test_date:
        parser.error("compare requires --test-date YYYY-MM-DD or --latest")
    args.func(args)


if __name__ == "__main__":
    main()
