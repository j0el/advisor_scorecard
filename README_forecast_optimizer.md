# Portfolio Forecast Optimizer Companion

This companion program reads your current portfolio weights and produces a model-based risk/opportunity comparison. It does **not** use dollar amounts in the report.

## Install

From inside your `advisor_scorecard` folder:

```bash
unzip -o ~/Downloads/portfolio_forecast_optimizer_companion.zip
chmod +x install_forecast_optimizer.sh
./install_forecast_optimizer.sh
```

## Monthly workflow

First generate your normal monthly review:

```bash
uv run advisor-scorecard snapshot
uv run advisor-scorecard monthly-review --holdings data/schwab_holdings.csv --years 3 --risk-free-rate 0.04
```

Then run the forecast optimizer:

```bash
uv run python3 portfolio_forecast_optimizer.py \
  --input reports/monthly_review/public_normalized_holdings_weights.csv \
  --years 5 \
  --horizon-years 1 \
  --risk-free-rate 0.04 \
  --max-weight 0.08 \
  --simulations 10000
```

Open:

```bash
open reports/forecast_optimizer/forecast_optimizer_report.html
open reports/forecast_optimizer/forecast_optimizer_report.pdf
```

## What it does

- Downloads trailing prices for your holdings.
- Computes holding indicators: historical return, volatility, trailing momentum, drawdown, distance from 200-day moving average.
- Builds a conservative expected-return model from the indicators.
- Builds a shrinkage covariance matrix for risk.
- Simulates probabilistic return ranges for the current portfolio and optimized candidates.
- Computes constrained candidates:
  - Current Portfolio
  - Optimized: Max Sharpe
  - Optimized: Max Return at Current Risk
  - Optimized: Minimum Volatility
- Shows how weights would change, in percentages only.

## Important limits

The optimizer is intentionally constrained. Without constraints, a pure “maximize expected return” optimizer would usually put everything into the single asset with the highest model expected return. That would be mathematically valid but not a sensible portfolio-review tool.

Use the output to frame questions and compare risk/opportunity tradeoffs, not as an automatic trade list.
