# Advisor Scorecard Monthly Review

Read-only monthly portfolio review tool for Schwab holdings. The main goal is to compare the current portfolio mix against standard benchmark portfolios using normalized percentages, risk metrics, and charts that can be shared without exposing dollar amounts.

## What it produces

The monthly review creates both HTML and PDF reports:

- `reports/monthly_review/monthly_review_report.pdf`
- `reports/monthly_review/monthly_review_report.html`

The PDF includes:

- headline interpretation
- key risk/opportunity metrics
- Sharpe and Sortino ratios
- max drawdown
- beta, tracking error, information ratio
- upside/downside capture
- portfolio asset-class pie chart
- portfolio sector pie chart
- normalized growth vs benchmarks
- drawdown chart
- risk vs return chart
- efficient-frontier style view
- Bollinger Bands on normalized portfolio value
- rolling 12-month Sharpe
- rolling 3-month volatility
- top holdings by normalized percentage
- benchmark definitions

Dollar amounts are intentionally omitted from the monthly report outputs.

## Setup on a new computer

Install `uv` first:

```bash
brew install uv
```

Create a project folder and unzip this distribution inside it:

```bash
mkdir -p ~/Documents/advisor_scorecard
cd ~/Documents/advisor_scorecard
unzip ~/Downloads/advisor_scorecard_monthly_report_clean.zip
```

Create your private environment file:

```bash
cp .env.example .env
open -a TextEdit .env
```

Fill in your Schwab credentials:

```text
SCHWAB_API_KEY=your_app_key
SCHWAB_APP_SECRET=your_app_secret
SCHWAB_CALLBACK_URL=https://127.0.0.1:8182
SCHWAB_TOKEN_PATH=./data/schwab_token.json
ADVISOR_FEE_BPS=100
```

Install dependencies:

```bash
uv sync
```

## Monthly workflow

Pull fresh Schwab holdings:

```bash
uv run advisor-scorecard snapshot
```

Generate the monthly report:

```bash
uv run advisor-scorecard monthly-review \
  --holdings data/schwab_holdings.csv \
  --years 3 \
  --risk-free-rate 0.04
```

Open the PDF:

```bash
open reports/monthly_review/monthly_review_report.pdf
```

## Private files

Do not share or commit these files:

- `.env`
- `data/schwab_token.json`
- `data/schwab_holdings.csv`
- `data/transaction_exports/`
- `reports/` if you consider ticker weights private

The program is read-only. It does not place trades.
