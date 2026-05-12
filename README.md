# Portfolio Forward Test Tool

This companion tool saves your current portfolio weights and the optimizer's Max Sharpe weights, then compares their actual forward return later.

It uses normalized weights only and does not use dollar amounts.

## Install

From inside your `advisor_scorecard` folder:

```bash
unzip -o ~/Downloads/portfolio_forward_test_tool.zip
chmod +x install_forward_test_tool.sh
./install_forward_test_tool.sh
```

## Start a forward test

Run your usual monthly workflow first:

```bash
uv run advisor-scorecard snapshot

uv run advisor-scorecard monthly-review \
  --holdings data/schwab_holdings.csv \
  --years 3 \
  --risk-free-rate 0.04

uv run python3 portfolio_forecast_optimizer.py
```

Then save the forward-test weights:

```bash
uv run python3 portfolio_forward_test.py start
```

This creates:

```text
reports/forward_tests/YYYY-MM-DD/
  forward_test_start.json
  current_portfolio_weights.csv
  max_sharpe_weights.csv
```

## Compare later

One month later:

```bash
uv run python3 portfolio_forward_test.py compare --latest
```

Or specify a test date:

```bash
uv run python3 portfolio_forward_test.py compare --test-date YYYY-MM-DD
```

Outputs:

```text
reports/forward_tests/YYYY-MM-DD/
  forward_test_metrics.csv
  forward_test_contributions.csv
  forward_test_daily_indexes.csv
  forward_test_comparison.html
  forward_test_comparison.pdf
  charts/
```

## Notes

This is a forward test, not a prediction. It answers: "From the day we saved the weights, which portfolio performed better over the elapsed period?"
