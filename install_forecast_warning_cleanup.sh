#!/usr/bin/env bash
set -euo pipefail

TARGET="portfolio_forecast_optimizer.py"

if [ ! -f "$TARGET" ]; then
  echo "ERROR: $TARGET not found."
  echo "Run this from inside your advisor_scorecard folder."
  exit 1
fi

cp "$TARGET" "$TARGET.bak.$(date +%Y%m%d_%H%M%S)"

python3 - <<'PY'
from pathlib import Path
import re
import py_compile

p = Path("portfolio_forecast_optimizer.py")
s = p.read_text()

block = '''# Suppress a harmless pandas warning caused by iterative price-table assembly.
# The report output is unaffected; this only keeps the terminal clean.
import warnings
try:
    from pandas.errors import PerformanceWarning
    warnings.simplefilter("ignore", PerformanceWarning)
except Exception:
    pass

'''

if 'warnings.simplefilter("ignore", PerformanceWarning)' in s:
    print("PerformanceWarning suppression is already installed.")
else:
    if s.startswith("from __future__ import annotations\n"):
        s = s.replace("from __future__ import annotations\n", "from __future__ import annotations\n\n" + block, 1)
    else:
        m = re.search(r"^(import |from )", s, flags=re.M)
        if m:
            s = s[:m.start()] + block + s[m.start():]
        else:
            s = block + s

    p.write_text(s)
    print("Installed PerformanceWarning suppression in portfolio_forecast_optimizer.py.")

py_compile.compile(str(p), doraise=True)
print("Syntax check passed.")
PY

echo
echo "Now rerun:"
echo "  uv run python3 portfolio_forecast_optimizer.py"
