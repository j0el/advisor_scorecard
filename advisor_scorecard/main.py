from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .analysis import demo_holdings, load_config, load_holdings, write_outputs
from .performance import build_performance_report
from .schwab_client import pull_schwab_holdings
from .schwab_inspect import inspect_schwab
from .monthly_review import build_monthly_review

console = Console()


def _print_paths(title: str, paths: dict):
    table = Table(title=title)
    table.add_column("Output")
    table.add_column("Path")
    for name, path in paths.items():
        table.add_row(name, str(path))
    console.print(table)


def cmd_demo(args):
    config = load_config(args.config)
    paths = write_outputs(demo_holdings(), config, args.outdir, refresh_metadata=args.refresh_metadata)
    console.print("[green]Demo report complete.[/green]")
    _print_paths("Generated files", paths)
    console.print(f"\nOpen this file in your browser:\n[bold]{paths['html']}[/bold]")


def cmd_snapshot(args):
    df = pull_schwab_holdings(args.output)
    console.print(f"[green]Wrote {len(df)} Schwab holding rows to {args.output}[/green]")


def cmd_analyze(args):
    holdings = load_holdings(args.holdings)
    config = load_config(args.config)
    paths = write_outputs(holdings, config, args.outdir, refresh_metadata=args.refresh_metadata)
    console.print("[green]Analysis report complete.[/green]")
    _print_paths("Generated files", paths)
    console.print(f"\nOpen this file in your browser:\n[bold]{paths['html']}[/bold]")


def cmd_inspect_schwab(args):
    paths = inspect_schwab(args.outdir)
    console.print("[green]Schwab inspection complete.[/green]")
    _print_paths("Generated Schwab inspection files", paths)


def cmd_performance(args):
    paths = build_performance_report(args.holdings, args.config, args.outdir, args.years, args.risk_free_rate)
    console.print("[green]Performance report complete.[/green]")
    _print_paths("Generated performance files", paths)
    console.print(f"\nOpen this file in your browser:\n[bold]{paths['performance_html']}[/bold]")



def cmd_monthly_review(args):
    paths = build_monthly_review(
        holdings_path=args.holdings,
        config_path=args.config,
        outdir=args.outdir,
        years=args.years,
        risk_free_rate=args.risk_free_rate,
        month=args.month,
        refresh_metadata=args.refresh_metadata,
    )
    console.print("[green]Monthly review complete.[/green]")
    _print_paths("Generated monthly review files", paths)
    console.print(f"\nOpen this file in your browser:\n[bold]{paths['html']}[/bold]")

def main():
    parser = argparse.ArgumentParser(prog="advisor-scorecard", description="Read-only portfolio scorecard for advisor evaluation.")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Run the scorecard on sample holdings.")
    demo.add_argument("--config", default="config.yaml")
    demo.add_argument("--outdir", default="reports")
    demo.add_argument("--refresh-metadata", action="store_true")
    demo.set_defaults(func=cmd_demo)

    snap = sub.add_parser("snapshot", help="Pull current holdings from Schwab into a CSV.")
    snap.add_argument("--output", default="data/schwab_holdings.csv")
    snap.set_defaults(func=cmd_snapshot)

    analyze = sub.add_parser("analyze", help="Analyze a holdings CSV.")
    analyze.add_argument("--holdings", default="data/schwab_holdings.csv")
    analyze.add_argument("--config", default="config.yaml")
    analyze.add_argument("--outdir", default="reports")
    analyze.add_argument("--refresh-metadata", action="store_true", help="Refresh ticker metadata cache from yfinance.")
    analyze.set_defaults(func=cmd_analyze)

    insp = sub.add_parser("inspect-schwab", help="Inspect raw Schwab position fields for cost basis and related data.")
    insp.add_argument("--outdir", default="reports")
    insp.set_defaults(func=cmd_inspect_schwab)

    perf = sub.add_parser("performance", help="Historical price performance exploration using current holdings weights.")
    perf.add_argument("--holdings", default="data/schwab_holdings.csv")
    perf.add_argument("--config", default="config.yaml")
    perf.add_argument("--outdir", default="reports")
    perf.add_argument("--years", type=int, default=3)
    perf.add_argument("--risk-free-rate", type=float, default=0.04)
    perf.set_defaults(func=cmd_performance)


    monthly = sub.add_parser("monthly-review", help="Create a shareable monthly normalized portfolio-vs-benchmark report.")
    monthly.add_argument("--holdings", default="data/schwab_holdings.csv")
    monthly.add_argument("--config", default="config.yaml")
    monthly.add_argument("--outdir", default="reports")
    monthly.add_argument("--years", type=int, default=3)
    monthly.add_argument("--risk-free-rate", type=float, default=0.04)
    monthly.add_argument("--month", default=None, help="Optional archive label, e.g. 2026-05")
    monthly.add_argument("--refresh-metadata", action="store_true")
    monthly.set_defaults(func=cmd_monthly_review)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
