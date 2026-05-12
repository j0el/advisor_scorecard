from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from .classify import clean_symbol, clean_text

CACHE_PATH = Path("data/ticker_metadata_cache.csv")

META_COLUMNS = [
    "symbol", "company_name", "security_type", "quote_type", "sector", "industry",
    "country", "exchange", "fund_category", "fund_family", "currency", "metadata_source", "lookup_status"
]


def _empty_cache() -> pd.DataFrame:
    return pd.DataFrame(columns=META_COLUMNS)


def load_cache(path: Path = CACHE_PATH) -> pd.DataFrame:
    if not path.exists():
        return _empty_cache()
    df = pd.read_csv(path)
    for col in META_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df["symbol"] = df["symbol"].map(clean_symbol)
    return df[META_COLUMNS].drop_duplicates("symbol", keep="last")


def save_cache(df: pd.DataFrame, path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in META_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out[META_COLUMNS].drop_duplicates("symbol", keep="last").sort_values("symbol").to_csv(path, index=False)


def _lookup_yfinance(symbol: str) -> dict:
    try:
        import yfinance as yf
        t = yf.Ticker(symbol)
        info = t.get_info() or {}
        name = info.get("longName") or info.get("shortName") or info.get("displayName") or symbol
        return {
            "symbol": symbol,
            "company_name": clean_text(name),
            "security_type": clean_text(info.get("quoteType") or info.get("typeDisp") or ""),
            "quote_type": clean_text(info.get("quoteType") or ""),
            "sector": clean_text(info.get("sector") or ""),
            "industry": clean_text(info.get("industry") or ""),
            "country": clean_text(info.get("country") or ""),
            "exchange": clean_text(info.get("exchange") or info.get("fullExchangeName") or ""),
            "fund_category": clean_text(info.get("category") or ""),
            "fund_family": clean_text(info.get("fundFamily") or ""),
            "currency": clean_text(info.get("currency") or ""),
            "metadata_source": "yfinance",
            "lookup_status": "ok" if name else "partial",
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "company_name": symbol,
            "security_type": "",
            "quote_type": "",
            "sector": "",
            "industry": "",
            "country": "",
            "exchange": "",
            "fund_category": "",
            "fund_family": "",
            "currency": "",
            "metadata_source": "yfinance",
            "lookup_status": f"error: {type(e).__name__}: {e}",
        }


def get_metadata(symbols: Iterable[str], refresh: bool = False, sleep_seconds: float = 0.05) -> pd.DataFrame:
    symbols = sorted({clean_symbol(s) for s in symbols if clean_symbol(s)})
    cache = load_cache()
    cached_symbols = set(cache["symbol"].tolist()) if not cache.empty else set()

    rows = [] if refresh else cache.to_dict("records")
    need = symbols if refresh else [s for s in symbols if s not in cached_symbols]

    for sym in need:
        rows.append(_lookup_yfinance(sym))
        if sleep_seconds:
            time.sleep(sleep_seconds)

    out = pd.DataFrame(rows) if rows else _empty_cache()
    for col in META_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out = out[META_COLUMNS].drop_duplicates("symbol", keep="last")
    save_cache(out)
    return out[out["symbol"].isin(symbols)].copy()
