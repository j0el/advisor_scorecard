from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv


def get_schwab_client():
    load_dotenv()
    api_key = os.getenv("SCHWAB_API_KEY")
    app_secret = os.getenv("SCHWAB_APP_SECRET")
    callback_url = os.getenv("SCHWAB_CALLBACK_URL", "https://127.0.0.1:8182")
    token_path = os.getenv("SCHWAB_TOKEN_PATH", "./data/schwab_token.json")
    if not api_key or not app_secret:
        raise RuntimeError("Missing SCHWAB_API_KEY or SCHWAB_APP_SECRET in .env")
    Path(token_path).parent.mkdir(parents=True, exist_ok=True)
    from schwab import auth
    return auth.easy_client(api_key, app_secret, callback_url, token_path)


def _json(resp):
    resp.raise_for_status()
    return resp.json()


def _find_positions(account: dict[str, Any]) -> list[dict[str, Any]]:
    sec = account.get("securitiesAccount", account)
    return sec.get("positions", []) or []


def _flatten_position(pos: dict[str, Any], account_hash: str = "") -> dict[str, Any]:
    inst = pos.get("instrument", {}) or {}
    sym = inst.get("symbol") or pos.get("symbol") or ""
    mv = pos.get("marketValue") or pos.get("longMarketValue") or pos.get("currentValue") or 0
    qty = pos.get("longQuantity") or pos.get("quantity") or pos.get("shortQuantity") or 0
    avg = pos.get("averageLongPrice") or pos.get("averagePrice") or pos.get("averagePriceLong")
    cost = pos.get("costBasis") or pos.get("longOpenProfitLoss")
    # If Schwab only gives avg price, estimate position cost basis.
    if cost is None and avg is not None:
        try:
            cost = float(avg) * float(qty)
        except Exception:
            cost = None
    return {
        "account_hash": account_hash,
        "symbol": sym,
        "description": inst.get("description") or inst.get("cusip") or "",
        "asset_type": inst.get("assetType") or inst.get("type") or "",
        "quantity": qty,
        "market_value": mv,
        "average_price": avg,
        "cost_basis": cost,
        "raw_instrument_type": inst.get("type") or "",
        "cusip": inst.get("cusip") or "",
    }


def pull_schwab_holdings(output: str = "data/schwab_holdings.csv") -> pd.DataFrame:
    client = get_schwab_client()
    accounts_resp = client.get_accounts(fields=client.Account.Fields.POSITIONS)
    accounts = _json(accounts_resp)
    rows = []
    for acct in accounts:
        account_hash = acct.get("hashValue") or acct.get("accountNumber") or ""
        for pos in _find_positions(acct):
            rows.append(_flatten_position(pos, account_hash))
    df = pd.DataFrame(rows)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df
