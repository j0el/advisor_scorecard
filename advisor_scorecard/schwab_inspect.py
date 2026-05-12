from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .schwab_client import get_schwab_client, _json, _find_positions

KEYWORDS = ["cost", "basis", "price", "gain", "loss", "market", "value", "quantity", "tax", "average", "profit"]


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten(v, key))
    elif isinstance(obj, list):
        out[prefix] = f"list[{len(obj)}]"
    else:
        out[prefix] = obj
    return out


def inspect_schwab(outdir: str = "reports") -> dict[str, Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    client = get_schwab_client()
    accounts = _json(client.get_accounts(fields=client.Account.Fields.POSITIONS))
    (out / "schwab_accounts_raw_sample.json").write_text(json.dumps(accounts[:1], indent=2), encoding="utf-8")
    rows = []
    for acct_idx, acct in enumerate(accounts):
        for pos_idx, pos in enumerate(_find_positions(acct)):
            flat = flatten(pos)
            symbol = flat.get("instrument.symbol", "")
            for key, value in flat.items():
                rows.append({"account_index": acct_idx, "position_index": pos_idx, "symbol": symbol, "field": key, "value": value})
    df = pd.DataFrame(rows)
    all_path = out / "schwab_all_position_fields.csv"
    df.to_csv(all_path, index=False)
    mask = df["field"].str.lower().apply(lambda f: any(k in f for k in KEYWORDS))
    interesting = df[mask].copy()
    interesting_path = out / "schwab_interesting_position_fields.csv"
    interesting.to_csv(interesting_path, index=False)
    summary = out / "schwab_position_field_summary.txt"
    fields = sorted(df["field"].unique().tolist()) if not df.empty else []
    summary.write_text("Schwab position field summary\n\n" + "\n".join(fields), encoding="utf-8")
    return {"all_fields": all_path, "interesting_fields": interesting_path, "summary": summary, "raw_sample": out / "schwab_accounts_raw_sample.json"}
