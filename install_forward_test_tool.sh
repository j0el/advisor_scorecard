#!/usr/bin/env bash
set -euo pipefail

if [ ! -f "pyproject.toml" ]; then
  echo "ERROR: pyproject.toml not found."
  echo "Run this from inside your advisor_scorecard folder."
  exit 1
fi

cp portfolio_forward_test.py ./portfolio_forward_test.py

echo "Installed portfolio_forward_test.py"
echo
echo "Start a forward test after you have run monthly-review and portfolio_forecast_optimizer.py:"
echo "  uv run python3 portfolio_forward_test.py start"
echo
echo "One month later:"
echo "  uv run python3 portfolio_forward_test.py compare --latest"
