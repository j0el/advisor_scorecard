from __future__ import annotations

CASH_SYMBOLS = {
    "CASH", "SWVXX", "SNVXX", "SNSXX", "SNOXX", "VMFXX", "SPAXX", "FDRXX",
    "BIL", "SGOV", "SHV", "TFLO", "USFR"
}
BOND_SYMBOLS = {
    "BND", "AGG", "SCHZ", "IUSB", "GOVT", "IEF", "TLT", "SHY", "VGIT", "VGLT",
    "BIV", "BSV", "MUB", "VTEB", "LQD", "VCIT", "VCSH", "HYG", "JNK", "TIP", "SCHP"
}
INTL_STOCK_SYMBOLS = {
    "VXUS", "VEA", "VWO", "IXUS", "SCHF", "SCHE", "IEFA", "EFA", "EEM", "ACWX", "VEU", "IDEV", "IEMG"
}
US_STOCK_SYMBOLS = {
    "VTI", "VOO", "SPY", "IVV", "SCHB", "SCHX", "SCHA", "SCHG", "SCHV", "IWB", "IWM",
    "QQQ", "DIA", "VUG", "VTV", "VB", "VO", "VIG", "VYM", "SCHD"
}
OTHER_SYMBOLS = {"GLD", "IAU", "SLV", "VNQ", "SCHH", "VNQI", "DBC", "PDBC", "USO"}


def clean_text(value) -> str:
    if value is None:
        return ""
    try:
        import pandas as pd
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def clean_symbol(value) -> str:
    return clean_text(value).upper()


def friendly_asset_name(asset_class: str) -> str:
    names = {
        "us_stock": "U.S. Stock",
        "intl_stock": "International Stock",
        "bond": "Bond / Fixed Income",
        "cash": "Cash / Money Market",
        "other": "Other / Alternatives",
    }
    key = clean_text(asset_class)
    return names.get(key, key.replace("_", " ").title())


def classify_from_metadata(symbol: str, asset_type: str = "", sector: str = "", industry: str = "", quote_type: str = "", category: str = "", country: str = "", fund_family: str = "") -> str:
    sym = clean_symbol(symbol)
    typ = clean_text(asset_type).lower()
    sec = clean_text(sector).lower()
    ind = clean_text(industry).lower()
    qt = clean_text(quote_type).lower()
    cat = clean_text(category).lower()
    ctry = clean_text(country).lower()
    fam = clean_text(fund_family).lower()
    blob = " ".join([typ, sec, ind, qt, cat, ctry, fam])

    if sym in CASH_SYMBOLS or "money market" in blob or "cash" in blob or "treasury bill" in blob or "ultra short" in blob:
        return "cash"
    if sym in BOND_SYMBOLS or "bond" in blob or "fixed income" in blob or "municipal" in blob or "treasury" in blob:
        return "bond"
    if sym in OTHER_SYMBOLS or "real estate" in blob or "reit" in blob or "gold" in blob or "commodity" in blob:
        return "other"
    if sym in INTL_STOCK_SYMBOLS or "international" in blob or "foreign" in blob or "emerging" in blob or "developed markets" in blob:
        return "intl_stock"
    if ctry and ctry not in {"united states", "usa", "us"} and qt in {"equity", "stock"}:
        return "intl_stock"
    if sym in US_STOCK_SYMBOLS:
        return "us_stock"
    if qt in {"etf", "mutualfund", "mutual fund"}:
        return "us_stock"
    return "us_stock"


def sector_from_metadata(sector: str = "", industry: str = "", category: str = "") -> str:
    sec = clean_text(sector)
    if sec:
        return sec
    cat = clean_text(category)
    if cat:
        return cat
    ind = clean_text(industry)
    if ind:
        return ind
    return "Unclassified"
